"""Distributed Aggregation (hidden-profile design).

Each of the seven Yeo networks holds one secret integer; every agent of that
network sees it. No network knows the others' values, so the Frontoparietal
readout agents can only answer by collecting all seven through the graph.
Readout agents report ``state["values"]`` (``{network: int}``) and
``state["sum"]``.

Difficulty is set by ``low``/``high`` (the value range). After node removal a
network may have no surviving agent; its value is then unrecoverable, which is
exactly what the robustness experiment measures.
"""

from __future__ import annotations

import networkx as nx

from neuromesh.contracts import YEO7, RunResult, TaskInstance
from neuromesh.tasks import _common as c

NAME = "aggregation"


class AggregationTask:
    name = NAME

    def __init__(self, low: int = 10, high: int = 99) -> None:
        if high - low + 1 < len(YEO7):
            raise ValueError("value range must hold at least 7 distinct values")
        self.low, self.high = low, high

    def make_instance(self, graph: nx.Graph, seed: int) -> TaskInstance:
        c.require_roles(graph)
        rng = c.rng_for(NAME, seed, graph)
        values = dict(zip(YEO7, rng.sample(range(self.low, self.high + 1), len(YEO7)), strict=True))
        roles = dict(graph.nodes(data="role"))
        private_info = {
            int(v): f"Your network ({r}) holds the secret value {r.upper()} = {values[r]}."
            for v, r in roles.items()
        }
        present = sorted({r for r in roles.values()}, key=YEO7.index)
        return TaskInstance(
            task=NAME,
            seed=seed,
            private_info=private_info,
            initial_state={int(v): {"values": {r: values[r]}} for v, r in roles.items()},
            readout_nodes=c.readout_nodes(graph),
            ground_truth={
                "values": values,
                "sum": sum(values.values()),
                "networks_present": present,
                "value_range": [self.low, self.high],
            },
        )

    def node_scores(self, instance: TaskInstance, state: dict) -> tuple[float, float, float]:
        """(fraction of the 7 values correct, sum correct, everything correct).

        ``values_complete`` (all 7 values right, sum ignored) is derived from the
        first element; it separates information integration from arithmetic, which
        the Week 5 baseline showed to be a failure mode of its own.
        """
        truth: dict[str, int] = instance.ground_truth["values"]
        reported = state.get("values") if isinstance(state, dict) else None
        reported = reported if isinstance(reported, dict) else {}
        reported = {str(k).strip().lower(): v for k, v in reported.items()}
        hits = sum(1 for k, v in truth.items() if c.as_int(reported.get(k)) == v)
        partial = hits / len(truth)
        sum_ok = 1.0 if c.as_int(state.get("sum") if isinstance(state, dict) else None) == (
            instance.ground_truth["sum"]
        ) else 0.0
        exact = 1.0 if (hits == len(truth) and sum_ok) else 0.0
        return partial, sum_ok, exact

    def score(self, instance: TaskInstance, result: RunResult) -> dict[str, float]:
        final = c.final_states(result)
        readout = [v for v in instance.readout_nodes if v in final] or list(instance.readout_nodes)
        rows = [self.node_scores(instance, final.get(v, {})) for v in readout]
        n = len(rows)
        present = instance.ground_truth.get("networks_present", list(YEO7))
        return {
            "exact_accuracy": sum(r[2] for r in rows) / n if n else 0.0,
            "partial_accuracy": sum(r[0] for r in rows) / n if n else 0.0,
            "sum_accuracy": sum(r[1] for r in rows) / n if n else 0.0,
            "values_complete": sum(1.0 for r in rows if r[0] == 1.0) / n if n else 0.0,
            "max_recoverable": len(present) / len(YEO7),
            "n_readout": float(n),
        }

    def baseline_prompt(self, instance: TaskInstance) -> str:
        values = instance.ground_truth["values"]
        listing = "\n".join(f"{r} network: {r.upper()} = {v}" for r, v in values.items())
        keys = ", ".join(f'"{r}": <int>' for r in values)
        return (
            "You are the only agent in this system. These are the secret values held by the "
            f"seven brain networks:\n{listing}\n\n"
            "Report all seven values and their sum. Answer with JSON only, in this form: "
            f'{{"values": {{{keys}}}, "sum": <int>}}'
        )

    def score_baseline(self, instance: TaskInstance, answer_text: str) -> dict[str, float]:
        parsed = c.parse_json_object(answer_text)
        if parsed is None:
            return {"parse_ok": 0.0, "exact_accuracy": 0.0, "partial_accuracy": 0.0,
                    "sum_accuracy": 0.0, "values_complete": 0.0}
        partial, sum_ok, exact = self.node_scores(instance, parsed)
        return {"parse_ok": 1.0, "exact_accuracy": exact, "partial_accuracy": partial,
                "sum_accuracy": sum_ok, "values_complete": 1.0 if partial == 1.0 else 0.0}
