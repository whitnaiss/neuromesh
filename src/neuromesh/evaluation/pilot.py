"""Preliminary evaluation with rule-based reference agents (no LLM).

Runs the full Track C pipeline -- task instances on Track A's graphs, logs in
the real ``runs``/``calls`` format, ``compute_metrics``, CI summaries, the
covariate regression and the figures -- on every topology and size::

    python -m neuromesh.evaluation.pilot --out results/pilot_reference

Two agent settings are run: ``ideal`` (perfect relays) and ``noisy`` (each
newly relayed fact code is corrupted with probability 0.05). Outputs:
``metrics.csv``, ``summary.csv``, ``regression_*.txt`` and ``fig_*.png``.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from neuromesh.evaluation.metrics import compute_metrics, to_frames
from neuromesh.evaluation.plots import plot_fidelity_vs_hop, plot_metric_vs_n
from neuromesh.evaluation.reference import run_reference
from neuromesh.evaluation.stats import covariate_regression, summarize
from neuromesh.graphs import assign_roles, generate_graph, structural_metrics
from neuromesh.tasks import TASK_NAMES, get_task

TOPOLOGIES = ["er", "ws", "ba", "modular", "yeo_sbm"]
SIZES = [7, 14, 28, 56, 112]
COVARIATES = ["average_shortest_path_length", "average_clustering", "degree_max",
              "input_readout_mean_distance"]
SUMMARY_METRICS = ["readout_fidelity", "network_fidelity", "distortion_rate",
                   "propagation_delay_t50", "fidelity_hop_slope", "agreement_ratio",
                   "consensus_reached", "exact_accuracy", "partial_accuracy",
                   "max_context_tokens", "total_tokens"]


def run(out: Path, seeds: list[int], relay_error: float, rounds: int) -> pd.DataFrame:
    out.mkdir(parents=True, exist_ok=True)
    frames = []
    for setting, err in (("ideal", 0.0), ("noisy", relay_error)):
        results, structure = [], []
        for topo in TOPOLOGIES:
            for n in SIZES:
                for seed in seeds:
                    graph = assign_roles(generate_graph(topo, n, seed=seed), seed=seed)
                    sm = structural_metrics(graph)
                    for task_name in TASK_NAMES:
                        inst = get_task(task_name).make_instance(graph, seed)
                        rid = f"{setting}-{topo}-n{n:03d}-{task_name}-s{seed}"
                        results.append((run_reference(graph, inst, rounds, err, seed, rid), inst))
                        structure.append({"run_id": rid, **{k: sm.get(k) for k in COVARIATES}})
        runs, calls = to_frames(results)
        m = compute_metrics(runs, calls).merge(pd.DataFrame(structure), on="run_id")
        m.insert(1, "setting", setting)
        frames.append(m)
    metrics = pd.concat(frames, ignore_index=True)
    metrics.to_csv(out / "metrics.csv", index=False)
    return metrics


def analyse(metrics: pd.DataFrame, out: Path) -> None:
    summary = summarize(metrics, SUMMARY_METRICS, by=("setting", "task", "topology", "n"))
    summary.to_csv(out / "summary.csv", index=False)

    ideal = summary[summary["setting"] == "ideal"]
    noisy = summary[summary["setting"] == "noisy"]
    plot_metric_vs_n(
        ideal,
        [("fact_propagation", "propagation_delay_t50", "Propagation delay t50 (rounds)"),
         ("consensus", "agreement_ratio", "Final agreement ratio"),
         ("fact_propagation", "max_context_tokens", "Max per-agent context (proxy tokens)")],
        out / "fig_reference_scaling.png",
    )
    plot_metric_vs_n(
        noisy,
        [("fact_propagation", "readout_fidelity", "Readout fidelity (5% relay noise)"),
         ("fact_propagation", "distortion_rate", "Distortion rate (5% relay noise)")],
        out / "fig_noisy_fidelity.png",
    )
    fp = metrics[(metrics["setting"] == "noisy") & (metrics["task"] == "fact_propagation")
                 & (metrics["n"] == 112)]
    hop_cols = [c for c in fp.columns if c.startswith("fidelity_hop_") and c[13:].isdigit()]
    hop_long = fp.melt(id_vars=["topology"], value_vars=hop_cols, var_name="hop",
                       value_name="fidelity").dropna()
    hop_long["hop"] = hop_long["hop"].str.replace("fidelity_hop_", "").astype(int)
    plot_fidelity_vs_hop(hop_long, out / "fig_fidelity_vs_hop.png",
                         "N = 112, reference agents, 5% relay noise")

    reg_specs = [("noisy", "fact_propagation", "readout_fidelity"),
                 ("ideal", "consensus", "agreement_ratio")]
    for setting, task, outcome in reg_specs:
        data = metrics[(metrics["setting"] == setting) & (metrics["task"] == task)
                       & (metrics["n"] >= 14)]
        for name, covs in (("topology_only", []), ("with_covariates", COVARIATES)):
            res = covariate_regression(data, outcome, covariates=covs)
            text = res.to_text() + (f"\ndropped: {res.dropped}" if res.dropped else "")
            (out / f"regression_{task}_{outcome}_{name}.txt").write_text(text + "\n")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m neuromesh.evaluation.pilot")
    p.add_argument("--out", default="results/pilot_reference")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--relay-error", type=float, default=0.05)
    p.add_argument("--rounds", type=int, default=10)
    args = p.parse_args(argv)
    out = Path(args.out)
    t0 = time.perf_counter()
    metrics = run(out, args.seeds, args.relay_error, args.rounds)
    analyse(metrics, out)
    print(f"{len(metrics)} runs analysed in {time.perf_counter() - t0:.1f}s -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
