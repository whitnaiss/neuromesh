"""Week 4 analysis datasets: assembly, aggregation and deterministic export.

This module turns the per-graph and per-node measurements into the four tables
the Week 4 deliverables call for, and writes them as plain CSV and JSON.

    structural   125 rows   5 topologies x 5 sizes x 5 seeds, one per graph
    conditions    25 rows   the 125 aggregated over seeds, one per condition
    positional  5425 rows   one per node: 5 x 5 x (7+14+28+56+112)
    robustness  ~3400 rows  complete decay curves, 2 strategies per graph
    auc          250 rows   one descriptive AUC per trajectory

Everything is regenerated from seeds; nothing is cached or hand-edited.

Determinism
-----------
The tables are byte-reproducible. Three things secure that:

1. **Fixed iteration order.** Topologies in ``TOPOLOGIES`` order, sizes in
   ``POPULATION_SIZES`` order, seeds ascending, nodes ascending. No dict or set
   iteration order reaches the output.
2. **Fixed column order.** Column tuples live in the modules that define the
   measurements (``metrics.STRUCTURAL_FIELDS`` and so on), so a column cannot
   appear or move without an explicit edit.
3. **Round-trip float formatting.** Floats are written with ``repr``, which in
   Python 3 is the shortest string that reads back as the identical double.
   ``f"{x:.6f}"`` was rejected: it silently discards precision, so a table
   written and re-read would no longer reproduce the metric it came from.

Two standard-deviation conventions, both stated
-----------------------------------------------
``metrics.degree_statistics`` reports the **population** standard deviation
(``ddof=0``) of a graph's degree sequence, because those degrees are the whole
population of that graph's nodes.

:func:`aggregate_conditions` reports the **sample** standard deviation
(``ddof=1``, Bessel-corrected) across the five seeds, because five seeds are a
sample from the generator's ensemble and the claim being made is about the
topology rather than about those five graphs.

They are different questions and the two conventions are not interchangeable.
Every exported column carries its convention in this docstring and in the
written JSON metadata.

Undefined values are never averaged away
----------------------------------------
Some metrics are legitimately ``None`` for some graphs -- a regular graph has
no defined assortativity. :func:`aggregate_conditions` therefore reports an
``*_n`` count of valid observations beside every statistic, aggregates only
over the defined values, and emits ``None`` for a statistic with no valid
observations. A mean over 3 of 5 seeds is never presented as a mean over 5.
"""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from .curves import (
    AUC_FIELDS,
    CHECKPOINT_FRACTIONS,
    MAX_REMOVAL_FRACTION,
    ROBUSTNESS_FIELDS,
    robustness_auc,
    robustness_trajectory,
)
from .metrics import (
    POSITIONAL_FIELDS,
    STRUCTURAL_FIELDS,
    positional_covariates,
    structural_metrics,
)
from .robustness import DEFAULT_ROUNDING_RULE, DEFAULT_TIE_BREAK, REMOVAL_STRATEGIES
from .spec import EXPERIMENT_SEEDS, POPULATION_SIZES, TOPOLOGIES, GraphParameterError

__all__ = [
    "AGGREGATE_METRICS",
    "CONDITION_FIELDS",
    "DEFAULT_OUTPUT_DIR",
    "aggregate_conditions",
    "build_positional_table",
    "build_robustness_tables",
    "build_structural_table",
    "expected_node_row_count",
    "format_value",
    "read_csv",
    "run_week4_analysis",
    "write_csv",
    "write_json",
]

#: Default output directory, under the Track A-owned ``data/`` tree. Relative
#: to the working directory, matching ``fixtures.DEFAULT_FIXTURE_DIR``'s
#: existing convention (and inheriting its documented limitation: it resolves
#: from the repository root).
DEFAULT_OUTPUT_DIR = Path("data") / "analysis" / "week4"

#: Metrics aggregated across seeds into the 25 condition rows. The first eight
#: are the minimum the Week 4 brief requires; the rest are carried because they
#: cost nothing and answer obvious follow-up questions.
AGGREGATE_METRICS: tuple[str, ...] = (
    "average_clustering",
    "transitivity",
    "average_shortest_path_length",
    "diameter",
    "global_efficiency",
    "degree_variance",
    "degree_assortativity",
    "modularity_role_partition",
    "degree_heterogeneity",
    "degree_max",
    "edge_density",
    "input_readout_mean_distance",
)

#: Statistics computed per metric per condition.
_STATS: tuple[str, ...] = ("mean", "std", "min", "max", "n")

CONDITION_FIELDS: tuple[str, ...] = (
    "topology",
    "n",
    "condition_id",
    "n_seeds",
    "seeds",
    *[f"{metric}_{stat}" for metric in AGGREGATE_METRICS for stat in _STATS],
)


# --------------------------------------------------------------------------
# serialisation
# --------------------------------------------------------------------------

def format_value(value: Any) -> str:
    """Render one cell for CSV, deterministically and round-trippably.

    ``None`` becomes the empty field. Booleans become ``True``/``False``.
    Floats use ``repr``, the shortest representation that reads back as the
    identical double -- so a written table reproduces its source numbers
    exactly rather than to six decimal places.
    """
    if value is None:
        return ""
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if math.isnan(number):
            return ""
        return repr(number)
    return str(value)


def _escape(cell: str) -> str:
    if any(ch in cell for ch in (",", '"', "\n", "\r")):
        return '"' + cell.replace('"', '""') + '"'
    return cell


def write_csv(path: Path, rows: Sequence[dict], fields: Sequence[str]) -> Path:
    """Write ``rows`` as CSV with exactly ``fields`` as the header.

    Written with ``\\n`` line endings explicitly (``newline=""`` is not enough
    on Windows, where the default text mode would translate to CRLF and the
    file would differ byte-for-byte between platforms).

    A row carrying a key absent from ``fields`` raises, rather than silently
    dropping a measured column.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    field_set = set(fields)
    lines = [",".join(_escape(str(f)) for f in fields)]
    for index, row in enumerate(rows):
        extra = set(row) - field_set
        if extra:
            raise GraphParameterError(
                f"row {index} carries column(s) {sorted(extra)} absent from the "
                "declared header; refusing to drop measured data"
            )
        lines.append(",".join(_escape(format_value(row.get(f))) for f in fields))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return path


def read_csv(path: Path) -> list[dict]:
    """Read a CSV written by :func:`write_csv` back into raw string dicts.

    Values stay strings; this exists for the round-trip tests rather than as a
    general loader. Empty fields read back as ``None``.
    """
    import csv

    with Path(path).open(encoding="utf-8", newline="") as handle:
        return [
            {k: (v if v != "" else None) for k, v in record.items()}
            for record in csv.DictReader(handle)
        ]


def write_json(path: Path, payload: Any) -> Path:
    """Write indented, key-sorted JSON with ``\\n`` endings."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    path.write_text(text + "\n", encoding="utf-8", newline="\n")
    return path


# --------------------------------------------------------------------------
# table builders
# --------------------------------------------------------------------------

def expected_node_row_count(
    topologies: Sequence[str] = TOPOLOGIES,
    sizes: Sequence[int] = POPULATION_SIZES,
    seeds: Sequence[int] = EXPERIMENT_SEEDS,
) -> int:
    """Node rows the positional table must contain: |T| x |S| x sum(sizes).

    For the study's grid that is 5 x 5 x (7+14+28+56+112) = 5425.
    """
    return len(topologies) * len(seeds) * sum(int(s) for s in sizes)


def _graph(topology: str, n: int, seed: int) -> nx.Graph:
    """One study graph with roles assigned. Roles are required here.

    The positional covariates and the predefined-partition modularity are both
    defined relative to the seven functional networks, so a graph without roles
    could not be measured. ``assign_roles`` is imported lazily because
    ``neuromesh.graphs.__init__`` imports this module.
    """
    from . import assign_roles, generate_graph

    graph = generate_graph(topology, int(n), seed=int(seed))
    assign_roles(graph, seed=int(seed))
    return graph


def build_structural_table(
    topologies: Sequence[str] = TOPOLOGIES,
    sizes: Sequence[int] = POPULATION_SIZES,
    seeds: Sequence[int] = EXPERIMENT_SEEDS,
) -> list[dict]:
    """One structural row per graph instance. 125 rows for the study grid."""
    rows: list[dict] = []
    for topology in topologies:
        for n in sizes:
            for seed in seeds:
                graph = _graph(topology, n, seed)
                rows.append(
                    structural_metrics(
                        graph,
                        topology=topology,
                        n=int(n),
                        seed=int(seed),
                        source="generated",
                    )
                )
    return rows


def build_positional_table(
    topologies: Sequence[str] = TOPOLOGIES,
    sizes: Sequence[int] = POPULATION_SIZES,
    seeds: Sequence[int] = EXPERIMENT_SEEDS,
) -> list[dict]:
    """One row per node per graph instance. 5425 rows for the study grid."""
    rows: list[dict] = []
    for topology in topologies:
        for n in sizes:
            for seed in seeds:
                graph = _graph(topology, n, seed)
                rows.extend(
                    positional_covariates(
                        graph, topology=topology, n=int(n), seed=int(seed)
                    )
                )
    return rows


def aggregate_conditions(
    structural_rows: Sequence[dict],
    metrics: Sequence[str] = AGGREGATE_METRICS,
) -> list[dict]:
    """Aggregate the per-instance rows into one row per (topology, N) condition.

    For every metric: ``mean``, ``std``, ``min``, ``max`` and ``n`` -- the
    count of seeds for which the metric was DEFINED.

    * ``std`` is the SAMPLE standard deviation (``ddof=1``). It is ``None``
      when fewer than two valid observations exist, because a single
      observation has no sample standard deviation -- reporting 0.0 there would
      claim zero variability from one measurement.
    * ``mean``, ``min`` and ``max`` are ``None`` when no observation is
      defined, never 0.0.
    * ``n`` lets a reader see immediately that a statistic covers 3 of 5 seeds
      rather than 5 of 5.
    """
    groups: dict[tuple, list[dict]] = {}
    for row in structural_rows:
        groups.setdefault((row["topology"], row["n"]), []).append(row)

    def sort_key(key: tuple) -> tuple:
        topology, n = key
        topology_rank = (
            list(TOPOLOGIES).index(topology)
            if topology in TOPOLOGIES
            else len(TOPOLOGIES)
        )
        return (topology_rank, n if n is not None else -1)

    out: list[dict] = []
    for key in sorted(groups, key=sort_key):
        topology, n = key
        group = sorted(groups[key], key=lambda r: (r["seed"] is None, r["seed"]))
        entry: dict[str, Any] = {
            "topology": topology,
            "n": n,
            "condition_id": f"{topology}_n{int(n):03d}" if n is not None else None,
            "n_seeds": len(group),
            "seeds": " ".join(str(r["seed"]) for r in group),
        }
        for metric in metrics:
            values = [
                float(r[metric])
                for r in group
                if r.get(metric) is not None and math.isfinite(float(r[metric]))
            ]
            entry[f"{metric}_n"] = len(values)
            entry[f"{metric}_mean"] = statistics.fmean(values) if values else None
            entry[f"{metric}_std"] = (
                statistics.stdev(values) if len(values) > 1 else None
            )
            entry[f"{metric}_min"] = min(values) if values else None
            entry[f"{metric}_max"] = max(values) if values else None
        out.append(entry)
    return out


def build_robustness_tables(
    topologies: Sequence[str] = TOPOLOGIES,
    sizes: Sequence[int] = POPULATION_SIZES,
    seeds: Sequence[int] = EXPERIMENT_SEEDS,
    strategies: Sequence[str] = REMOVAL_STRATEGIES,
    *,
    max_fraction: float = MAX_REMOVAL_FRACTION,
    checkpoints: Sequence[float] = CHECKPOINT_FRACTIONS,
    tie_break: str = DEFAULT_TIE_BREAK,
    rule: str = DEFAULT_ROUNDING_RULE,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Build every robustness trajectory and its AUC.

    Returns ``(trajectory_point_rows, auc_rows, trajectories)``.

    The removal seed is the GRAPH's seed. That keeps the whole analysis
    addressable by one ``(topology, n, seed)`` key, and ``robustness.py`` salts
    the removal RNG away from generation anyway, so the removal draw is not
    correlated with the graph it attacks despite sharing the integer.
    """
    rows: list[dict] = []
    auc_rows: list[dict] = []
    trajectories: list[dict] = []
    for topology in topologies:
        for n in sizes:
            for seed in seeds:
                graph = _graph(topology, n, seed)
                for strategy in strategies:
                    trajectory = robustness_trajectory(
                        graph,
                        strategy,
                        seed=int(seed),
                        tie_break=tie_break,
                        max_fraction=max_fraction,
                        checkpoints=checkpoints,
                        rule=rule,
                        topology=topology,
                        n=int(n),
                        graph_seed=int(seed),
                    )
                    trajectories.append(trajectory)
                    rows.extend(trajectory["points"])
                    auc_rows.append(robustness_auc(trajectory))
    return rows, auc_rows, trajectories


# --------------------------------------------------------------------------
# the driver
# --------------------------------------------------------------------------

def run_week4_analysis(
    out_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    topologies: Sequence[str] = TOPOLOGIES,
    sizes: Sequence[int] = POPULATION_SIZES,
    seeds: Sequence[int] = EXPERIMENT_SEEDS,
    max_fraction: float = MAX_REMOVAL_FRACTION,
    write: bool = True,
    with_figures: bool = True,
) -> dict:
    """Run the whole Week 4 analysis and (optionally) write every output.

    Deterministic and offline: every graph is regenerated from its seed, the
    committed empirical Yeo matrix is read from the package data, and nothing
    is downloaded.

    Returns a dict holding every table plus a ``manifest`` describing what was
    written. Set ``write=False`` to compute without touching the filesystem,
    which is what the fast unit tests do.
    """
    out_dir = Path(out_dir)

    structural = build_structural_table(topologies, sizes, seeds)
    conditions = aggregate_conditions(structural)
    positional = build_positional_table(topologies, sizes, seeds)
    robustness, auc, trajectories = build_robustness_tables(
        topologies, sizes, seeds, max_fraction=max_fraction
    )
    checkpoints = [p for p in robustness if p["is_checkpoint"]]

    manifest: dict[str, Any] = {
        "generated_by": "neuromesh.graphs.datasets.run_week4_analysis",
        "track": "A",
        "week": 4,
        "grid": {
            "topologies": list(topologies),
            "sizes": [int(s) for s in sizes],
            "seeds": [int(s) for s in seeds],
        },
        "row_counts": {
            "structural": len(structural),
            "conditions": len(conditions),
            "positional": len(positional),
            "robustness_points": len(robustness),
            "robustness_checkpoints": len(checkpoints),
            "auc": len(auc),
            "trajectories": len(trajectories),
        },
        "conventions": {
            "per_graph_degree_std": "population (ddof=0)",
            "across_seed_std": "sample (ddof=1, Bessel-corrected)",
            "robustness_denominator": "original N (never the surviving count)",
            "removal_rounding_rule": DEFAULT_ROUNDING_RULE,
            "removal_tie_break": DEFAULT_TIE_BREAK,
            "max_removal_fraction": float(max_fraction),
            "checkpoint_fractions": list(CHECKPOINT_FRACTIONS),
            "modularity": "predefined seven-network role partition, resolution 1.0",
            "undefined_values": "null / empty CSV field, never 0.0",
            "auc": "trapezoidal over measured points; no extrapolation",
        },
        "disclaimer": (
            "Graph-only structural measurements. These are NOT agent "
            "communication success, LLM response fidelity, or task accuracy. "
            "No LLM was involved in producing any number here."
        ),
        "networkx_version": nx.__version__,
        "numpy_version": np.__version__,
    }

    result: dict[str, Any] = {
        "structural": structural,
        "conditions": conditions,
        "positional": positional,
        "robustness": robustness,
        "checkpoints": checkpoints,
        "auc": auc,
        "trajectories": trajectories,
        "manifest": manifest,
    }

    if not write:
        return result

    written: list[str] = []
    written.append(
        str(write_csv(out_dir / "structural_metrics.csv", structural, STRUCTURAL_FIELDS))
    )
    written.append(
        str(write_csv(out_dir / "condition_summaries.csv", conditions, CONDITION_FIELDS))
    )
    written.append(
        str(
            write_csv(
                out_dir / "positional_covariates.csv", positional, POSITIONAL_FIELDS
            )
        )
    )
    written.append(
        str(write_csv(out_dir / "robustness_curves.csv", robustness, ROBUSTNESS_FIELDS))
    )
    written.append(
        str(
            write_csv(
                out_dir / "robustness_checkpoints.csv", checkpoints, ROBUSTNESS_FIELDS
            )
        )
    )
    written.append(str(write_csv(out_dir / "robustness_auc.csv", auc, AUC_FIELDS)))

    if with_figures:
        from .figures import write_figures

        written.extend(str(p) for p in write_figures(out_dir, result))

    manifest["files"] = sorted(Path(p).name for p in written)
    written.append(str(write_json(out_dir / "manifest.json", manifest)))
    result["written"] = written
    return result
