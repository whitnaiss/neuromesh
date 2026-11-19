"""Figure functions (matplotlib, headless). Each returns the saved path."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

TOPOLOGY_ORDER = ["er", "ws", "ba", "modular", "yeo_sbm"]
TOPOLOGY_LABEL = {"er": "ER", "ws": "WS", "ba": "BA", "modular": "Modular", "yeo_sbm": "Yeo-SBM"}
TOPOLOGY_COLOR = {"er": "#1f4e9c", "ws": "#b8860b", "ba": "#1b7a3d", "modular": "#a33b8f",
                  "yeo_sbm": "#c0392b"}  # same palette as Track A's figures
#: Metrics that are proportions; their CIs are clipped to [0, 1] when drawn.
BOUNDED_METRICS = {"readout_fidelity", "network_fidelity", "readout_complete", "distortion_rate",
                   "agreement_ratio", "consensus_reached", "valid_vote_rate", "exact_accuracy",
                   "partial_accuracy", "sum_accuracy"}
TASK_LABEL = {"fact_propagation": "Fact Propagation", "consensus": "Consensus",
              "aggregation": "Distributed Aggregation"}


def _finish(fig, path: str | Path, layout: bool = True) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if layout:
        fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_metric_vs_n(
    summary: pd.DataFrame,
    panels: list[tuple[str, str, str]],
    path: str | Path,
    baseline: dict[str, float] | None = None,
) -> Path:
    """One panel per (task, metric, y-label): mean +/- 95% CI against N, one line per topology.

    ``summary`` is the long table from :func:`neuromesh.evaluation.stats.summarize`.
    ``baseline`` maps task -> single-agent score, drawn as a dashed line.
    """
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.4), squeeze=False)
    for ax, (task, metric, ylabel) in zip(axes[0], panels, strict=True):
        sub = summary[(summary["task"] == task) & (summary["metric"] == metric)]
        for topo in TOPOLOGY_ORDER:
            s = sub[sub["topology"] == topo].sort_values("n")
            if s.empty:
                continue
            lo, hi = s["ci_low"], s["ci_high"]
            if metric in BOUNDED_METRICS:
                lo, hi = lo.clip(0, 1), hi.clip(0, 1)
            ax.errorbar(s["n"], s["mean"], yerr=[s["mean"] - lo, hi - s["mean"]],
                        label=TOPOLOGY_LABEL[topo], color=TOPOLOGY_COLOR[topo], marker="o",
                        ms=4, lw=1.5, capsize=2)
        if baseline and task in baseline:
            ax.axhline(baseline[task], color="black", ls="--", lw=1, label="Single agent")
        if metric in BOUNDED_METRICS:
            ax.set_ylim(-0.02, 1.05)
        ax.set_xscale("log", base=2)
        ns = sorted(sub["n"].unique())
        ax.set_xticks(ns, [str(int(n)) for n in ns])
        ax.set_xlabel("Number of agents N")
        ax.set_ylabel(ylabel)
        ax.set_title(TASK_LABEL.get(task, task), fontsize=10)
        ax.grid(alpha=0.3)
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:  # one shared legend to the right of the panels, never over the data
        fig.legend(handles, labels, fontsize=7, frameon=False, loc="center right")
        fig.tight_layout(rect=(0, 0, 0.92, 1))
        return _finish(fig, path, layout=False)
    return _finish(fig, path)


def plot_fidelity_vs_hop(
    hop_long: pd.DataFrame, path: str | Path, title: str = "", min_runs: int = 3
) -> Path:
    """Mean fidelity against hop distance from the input agents, one line per topology.

    ``hop_long`` has columns ``topology, hop, fidelity`` (one row per run x hop).
    Hops observed in fewer than ``min_runs`` runs are not drawn (too few agents).
    """
    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    for topo in TOPOLOGY_ORDER:
        s = hop_long[hop_long["topology"] == topo]
        if s.empty:
            continue
        g = s.groupby("hop")["fidelity"].agg(["mean", "sem", "count"]).reset_index()
        g = g[g["count"] >= min_runs]
        ax.errorbar(g["hop"], g["mean"], yerr=1.96 * g["sem"].fillna(0), marker="o", ms=4,
                    capsize=2, color=TOPOLOGY_COLOR[topo], label=TOPOLOGY_LABEL[topo])
    ax.set_xlabel("Hop distance from nearest Visual (input) agent")
    ax.set_ylabel("Fact fidelity")
    ax.set_ylim(-0.02, 1.02)
    if title:
        ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=7, frameon=False)
    return _finish(fig, path)


def plot_baseline_vs_network(rows: pd.DataFrame, path: str | Path) -> Path:
    """Grouped bars: ``rows`` has ``label, system, score`` (system = baseline or a topology)."""
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    labels = list(dict.fromkeys(rows["label"]))
    systems = list(dict.fromkeys(rows["system"]))
    width = 0.8 / max(len(systems), 1)
    for i, sysname in enumerate(systems):
        s = rows[rows["system"] == sysname].set_index("label").reindex(labels)
        ax.bar([j + i * width for j in range(len(labels))], s["score"], width,
               label=TOPOLOGY_LABEL.get(sysname, sysname),
               color=TOPOLOGY_COLOR.get(sysname, "#555555"))
    ax.set_xticks([j + width * (len(systems) - 1) / 2 for j in range(len(labels))], labels,
                  fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.legend(fontsize=7, frameon=False, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    return _finish(fig, path)
