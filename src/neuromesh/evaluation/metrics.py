"""Per-run metrics computed from the two log tables.

Log format (what the runner writes and this module reads)
---------------------------------------------------------
``runs`` -- one row per network run: the ``RunResult`` fields except ``calls``
(``run_id, topology, n, task, seed, removal, wall_clock_s, final_answers``)
plus ``instance``, the ``TaskInstance`` as JSON. The instance carries the
answer key, so every score can be recomputed from the logs alone.

``calls`` -- one row per LLM call: the ``CallRecord`` fields
(``run_id, round, node_id, role, prompt_tokens, completion_tokens, latency_s,
parse_ok, retries, state``).

Nested fields (``state``, ``final_answers``, ``removal``, ``instance``) may be
dicts or JSON strings, so the same code reads in-memory frames and Parquet.
"""

from __future__ import annotations

import json
from collections.abc import Iterable

import pandas as pd

from neuromesh.contracts import CallRecord, RunResult, TaskInstance
from neuromesh.tasks import get_task

RUN_KEYS = ["run_id", "topology", "n", "task", "seed"]


def _load(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def to_frames(
    results: Iterable[tuple[RunResult, TaskInstance]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Flatten run results into the ``runs`` / ``calls`` tables described above."""
    run_rows, call_rows = [], []
    for result, instance in results:
        row = result.model_dump(exclude={"calls"})
        row["removal"] = json.dumps(row["removal"]) if row["removal"] else None
        row["final_answers"] = json.dumps({str(k): v for k, v in row["final_answers"].items()})
        row["instance"] = instance.model_dump_json()
        run_rows.append(row)
        for call in result.calls:
            c = call.model_dump()
            c["state"] = json.dumps(c["state"])
            call_rows.append(c)
    calls = pd.DataFrame(call_rows, columns=list(CallRecord.model_fields))
    return pd.DataFrame(run_rows), calls


def _rebuild(run: pd.Series, calls: pd.DataFrame) -> tuple[RunResult, TaskInstance]:
    records = [
        CallRecord(**{**rec, "state": _load(rec["state"]) or {}})
        for rec in calls.to_dict("records")
    ]
    final = _load(run.get("final_answers")) or {}
    removal = _load(run.get("removal"))
    result = RunResult(
        run_id=str(run["run_id"]),
        topology=run["topology"],
        n=int(run["n"]),
        task=run["task"],
        seed=int(run["seed"]),
        removal=removal if isinstance(removal, dict) else None,
        wall_clock_s=float(run.get("wall_clock_s") or 0.0),
        calls=records,
        final_answers={int(k): v for k, v in final.items()},
    )
    instance = TaskInstance.model_validate(_load(run["instance"]))
    return result, instance


def cost_metrics(calls: pd.DataFrame) -> dict[str, float]:
    """Tokens, call counts, parse failures and context size for one run's calls."""
    if calls.empty:
        return {"n_calls": 0.0, "prompt_tokens": 0.0, "completion_tokens": 0.0,
                "total_tokens": 0.0, "max_context_tokens": 0.0, "parse_error_rate": 0.0,
                "retries": 0.0, "mean_latency_s": 0.0}
    prompt = calls["prompt_tokens"].astype(float)
    completion = calls["completion_tokens"].astype(float)
    return {
        "n_calls": float(len(calls)),
        "prompt_tokens": float(prompt.sum()),
        "completion_tokens": float(completion.sum()),
        "total_tokens": float(prompt.sum() + completion.sum()),
        # context length of a call = its prompt; the max over calls flags hub overload
        "max_context_tokens": float(prompt.max()),
        "parse_error_rate": float(1.0 - calls["parse_ok"].astype(bool).mean()),
        "retries": float(calls["retries"].astype(float).sum()),
        "mean_latency_s": float(calls["latency_s"].astype(float).mean()),
    }


def compute_metrics(runs: pd.DataFrame, calls: pd.DataFrame) -> pd.DataFrame:
    """One row per run: propagation delay, fidelity, agreement, accuracy, tokens, time.

    Task scores come from ``Task.score`` (so the network and the scorer tests
    use one definition); cost metrics come from the call log.
    """
    by_run = {rid: grp for rid, grp in calls.groupby("run_id", sort=False)} if len(calls) else {}
    rows = []
    for _, run in runs.iterrows():
        run_calls = by_run.get(run["run_id"], calls.iloc[0:0])
        result, instance = _rebuild(run, run_calls)
        removal = result.removal
        row = {
            "run_id": result.run_id,
            "topology": result.topology,
            "n": result.n,
            "task": result.task,
            "seed": result.seed,
            "removal_mode": removal.mode if removal else "none",
            "removal_fraction": removal.fraction if removal else 0.0,
            "wall_clock_s": result.wall_clock_s,
        }
        row.update(cost_metrics(run_calls))
        row["tokens_per_agent"] = row["total_tokens"] / max(result.n, 1)
        row.update(get_task(result.task).score(instance, result))
        rows.append(row)
    return pd.DataFrame(rows)


#: The headline metric of each task (used by summaries, figures and regression).
PRIMARY_METRIC = {
    "fact_propagation": "readout_fidelity",
    "consensus": "agreement_ratio",
    "aggregation": "exact_accuracy",
}
