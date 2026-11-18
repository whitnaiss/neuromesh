"""Shortest-path structure of the generated graphs, and the round budget.

Why this module exists
----------------------
The experiment runs a fixed number of synchronous communication rounds
(``contracts.DEFAULT_ROUNDS``, currently 10). Under a protocol where a message
crosses at most one edge per round, a graph of diameter ``d`` needs **at least
``d`` rounds** before the most distant pair of agents can have exchanged
anything at all. If ``d`` exceeds the round budget, the far end of the network
is unreachable *by construction* and no amount of model quality fixes it.

So the round budget is a graph-theoretic question before it is an LLM
question, and this module answers the graph-theoretic part exactly: it measures
diameter and average shortest-path length for every (topology, seed) condition
at a given N and reports which conditions fit inside which budget.

What "sufficient" does and does not mean
----------------------------------------
``d <= rounds`` is **necessary, not sufficient.** It is the weakest possible
requirement -- that a shortest path has time to be traversed once. It says
nothing about whether the agents in between forward the information faithfully,
whether a consensus task actually converges, or whether an LLM reasons
correctly over what arrives. Treating ``diameter <= rounds`` as a guarantee of
task success would be a serious overclaim, and :func:`round_budget_assessment`
labels its own output accordingly.

Three further caveats, all of them real:

1. **Off-by-one depends on the protocol.** Whether ``d`` rounds suffice, or
   ``d + 1`` are needed, depends on conventions Track B owns and has not
   implemented yet: whether round 1 is the first *send* or an initialisation
   turn, whether a message sent in round ``r`` is readable in round ``r`` or
   ``r + 1``, and whether the final readout consumes a round. The numbers here
   are pure graph distances; the mapping to rounds is stated as conditional on
   an ideal synchronous one-hop-per-round protocol, and the table reports
   ``d <= budget`` so a team that later finds an extra round is consumed can
   re-read it against ``budget - 1`` without remeasuring.
2. **Multi-hop aggregation needs more than one traversal.** A task that must
   combine information *from* the far end and return a verdict *to* a readout
   agent needs something closer to ``d`` plus the input-to-readout distance,
   not ``d``. That is Week 4 work (positional covariates) and is not computed
   here.
3. **Diameter is a worst case over pairs.** Average shortest-path length is
   reported alongside it because most pairs are far closer than ``d``, so a
   diameter violation degrades the tail of the pair distribution rather than
   the whole run.

Disconnected graphs
-------------------
Diameter is infinite for a disconnected graph, and this module never invents a
finite stand-in. ``diameter`` is ``None``, and
``max_finite_distance`` -- the largest shortest-path distance over pairs that
*are* connected, i.e. the diameter of the widest component -- is reported
separately and clearly labelled. Every graph the generators ship is connected
by contract, so this path exists for attacked graphs and for reporting a
generator contract breach honestly rather than concealing it.
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable, Sequence
from typing import Any

import networkx as nx

from .spec import EXPERIMENT_SEEDS, TOPOLOGIES, GraphParameterError

__all__ = [
    "DEFAULT_ROUND_BUDGETS",
    "aggregate_path_metrics",
    "diameter_table",
    "measure_graph",
    "path_metrics",
    "render_aggregate_table",
    "render_measurement_table",
    "round_budget_assessment",
]

#: The two budgets the Week 3 allocation asks about.
DEFAULT_ROUND_BUDGETS: tuple[int, ...] = (10, 12)

#: Standard deviation convention used throughout this module: the *sample*
#: standard deviation (``ddof = 1``, Bessel-corrected). The five seeds are
#: treated as a sample from the generator's ensemble rather than as the whole
#: population, because the claim being made is about the topology, not about
#: these five graphs. The population value (``ddof = 0``) is also returned,
#: under ``std_population``, so a reader who prefers it never has to recompute.
STD_CONVENTION = "sample (ddof=1, Bessel-corrected)"


def path_metrics(graph: nx.Graph) -> dict:
    """Shortest-path structure of one graph.

    Returns
    -------
    dict
        ``n_nodes``, ``n_edges``, ``mean_degree``, ``connected``,
        ``n_components``, ``largest_component_size``, ``diameter``,
        ``average_shortest_path_length``, ``max_finite_distance``, ``radius``.

        For a **connected** graph, ``diameter`` is the graph diameter and
        ``max_finite_distance`` equals it. For a **disconnected** graph,
        ``diameter`` and ``radius`` are ``None`` (infinite, not invented) and
        ``average_shortest_path_length`` is ``None``; ``max_finite_distance``
        holds the diameter of the widest component, and
        ``average_shortest_path_length_largest_component`` is also provided.
    """
    if not isinstance(graph, nx.Graph):
        raise GraphParameterError(
            f"expected a networkx.Graph, got {type(graph).__name__}"
        )
    if graph.is_directed():
        raise GraphParameterError("path metrics here are defined for undirected graphs")

    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()
    metrics: dict[str, Any] = {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "mean_degree": (2.0 * n_edges / n_nodes) if n_nodes else None,
        "connected": False,
        "n_components": 0,
        "largest_component_size": 0,
        "diameter": None,
        "radius": None,
        "average_shortest_path_length": None,
        "max_finite_distance": None,
        "average_shortest_path_length_largest_component": None,
    }
    if n_nodes == 0:
        return metrics

    components = list(nx.connected_components(graph))
    metrics["n_components"] = len(components)
    largest = max(components, key=len)
    metrics["largest_component_size"] = len(largest)
    metrics["connected"] = len(components) == 1

    if metrics["connected"]:
        metrics["diameter"] = int(nx.diameter(graph))
        metrics["radius"] = int(nx.radius(graph))
        average = (
            float(nx.average_shortest_path_length(graph)) if n_nodes > 1 else 0.0
        )
        metrics["average_shortest_path_length"] = average
        metrics["max_finite_distance"] = metrics["diameter"]
        metrics["average_shortest_path_length_largest_component"] = average
        return metrics

    # Disconnected: no finite diameter exists. Report the widest component's
    # diameter under a different name so the two can never be confused.
    widest = 0
    for component in components:
        subgraph = graph.subgraph(component)
        if subgraph.number_of_nodes() > 1:
            widest = max(widest, int(nx.diameter(subgraph)))
    metrics["max_finite_distance"] = widest
    largest_subgraph = graph.subgraph(largest)
    metrics["average_shortest_path_length_largest_component"] = (
        float(nx.average_shortest_path_length(largest_subgraph))
        if largest_subgraph.number_of_nodes() > 1
        else 0.0
    )
    return metrics


def measure_graph(
    graph: nx.Graph,
    *,
    topology: str,
    n: int,
    seed: int,
    source: str = "generated",
    round_budgets: Sequence[int] = DEFAULT_ROUND_BUDGETS,
) -> dict:
    """One measurement row: identity, structure and the round-budget checks.

    ``round_check`` maps each budget to ``True``/``False``/``None``: whether a
    shortest path between the most distant pair fits inside that many
    one-hop-per-round steps. ``None`` means the graph is disconnected, so the
    question has no finite answer -- never ``False`` by default, because
    "unreachable at any budget" and "needs more rounds" are different findings.
    """
    metrics = path_metrics(graph)
    diameter = metrics["diameter"]
    row = {
        "topology": topology,
        "n": int(n),
        "seed": int(seed),
        "source": source,
        **metrics,
        "round_check": {
            int(budget): (None if diameter is None else diameter <= int(budget))
            for budget in round_budgets
        },
        "caveat": _caveat(metrics),
    }
    return row


def _caveat(metrics: dict) -> str:
    if metrics["n_nodes"] == 0:
        return "empty graph"
    if not metrics["connected"]:
        return (
            f"disconnected ({metrics['n_components']} components): diameter is "
            f"infinite; widest component diameter {metrics['max_finite_distance']}"
        )
    return ""


def diameter_table(
    n: int,
    *,
    topologies: Sequence[str] = TOPOLOGIES,
    seeds: Sequence[int] = EXPERIMENT_SEEDS,
    round_budgets: Sequence[int] = DEFAULT_ROUND_BUDGETS,
    mean_degree: float | None = None,
) -> list[dict]:
    """Measure every ``(topology, seed)`` graph at ``n``, regenerating each one.

    Graphs come from the public :func:`neuromesh.graphs.generate_graph`, at its
    default settings, so the rows describe exactly the graphs the experiment
    will run -- no analysis-only parameters. Deterministic: identical arguments
    give identical rows.
    """
    # Imported here rather than at module scope: ``neuromesh.graphs.__init__``
    # imports this module, so a top-level import would be circular.
    from . import generate_graph

    rows: list[dict] = []
    for topology in topologies:
        for seed in seeds:
            kwargs = {} if mean_degree is None else {"mean_degree": mean_degree}
            graph = generate_graph(topology, int(n), seed=int(seed), **kwargs)
            rows.append(
                measure_graph(
                    graph,
                    topology=topology,
                    n=int(n),
                    seed=int(seed),
                    source="generated",
                    round_budgets=round_budgets,
                )
            )
    return rows


def aggregate_path_metrics(
    rows: Iterable[dict], round_budgets: Sequence[int] = DEFAULT_ROUND_BUDGETS
) -> list[dict]:
    """Per-topology aggregate of a :func:`diameter_table`.

    Disconnected conditions are counted in ``n_disconnected`` and excluded from
    the min/max/mean/median/std of diameter, because an infinite value has no
    mean. They are *not* silently dropped: a topology with any disconnected
    condition carries ``n_disconnected > 0`` and its statistics are explicitly
    over the connected subset.
    """
    rows = list(rows)
    budgets = [int(b) for b in round_budgets]
    by_topology: dict[str, list[dict]] = {}
    for row in rows:
        by_topology.setdefault(row["topology"], []).append(row)

    aggregates: list[dict] = []
    for topology, group in by_topology.items():
        diameters = [r["diameter"] for r in group if r["diameter"] is not None]
        averages = [
            r["average_shortest_path_length"]
            for r in group
            if r["average_shortest_path_length"] is not None
        ]
        entry: dict[str, Any] = {
            "topology": topology,
            "n_conditions": len(group),
            "n_connected": len(diameters),
            "n_disconnected": len(group) - len(diameters),
            "std_convention": STD_CONVENTION,
            "min": min(diameters) if diameters else None,
            "max": max(diameters) if diameters else None,
            "mean": (sum(diameters) / len(diameters)) if diameters else None,
            "median": statistics.median(diameters) if diameters else None,
            "std": (
                statistics.stdev(diameters) if len(diameters) > 1 else 0.0
            )
            if diameters
            else None,
            "std_population": (
                statistics.pstdev(diameters) if diameters else None
            ),
            "mean_average_shortest_path": (
                sum(averages) / len(averages) if averages else None
            ),
        }
        for budget in budgets:
            within = [d for d in diameters if d <= budget]
            entry[f"n_within_{budget}"] = len(within)
            entry[f"n_above_{budget}"] = len(diameters) - len(within)
        aggregates.append(entry)

    aggregates.sort(key=lambda e: list(TOPOLOGIES).index(e["topology"])
                    if e["topology"] in TOPOLOGIES else len(TOPOLOGIES))
    return aggregates


def round_budget_assessment(
    rows: Iterable[dict], round_budgets: Sequence[int] = DEFAULT_ROUND_BUDGETS
) -> dict:
    """Which conditions fit which budget, and the smallest budget covering all.

    The returned dict is deliberately blunt about what it is claiming. Read
    ``interpretation``: this is a *necessary reachability* result under an
    ideal synchronous one-hop-per-round protocol, not a statement about
    consensus, reasoning or task success.
    """
    rows = list(rows)
    budgets = [int(b) for b in round_budgets]
    connected = [r for r in rows if r["diameter"] is not None]
    disconnected = [r for r in rows if r["diameter"] is None]

    per_budget = {}
    for budget in budgets:
        violations = [
            {
                "topology": r["topology"],
                "n": r["n"],
                "seed": r["seed"],
                "diameter": r["diameter"],
            }
            for r in connected
            if r["diameter"] > budget
        ]
        per_budget[budget] = {
            "n_conditions": len(rows),
            "n_within": len(connected) - len(violations),
            "n_above": len(violations),
            "covers_all_measured": not violations and not disconnected,
            "violations": violations,
        }

    max_diameter = max((r["diameter"] for r in connected), default=None)
    return {
        "n_conditions": len(rows),
        "n_connected": len(connected),
        "n_disconnected": len(disconnected),
        "max_diameter": max_diameter,
        "argmax": [
            {"topology": r["topology"], "n": r["n"], "seed": r["seed"]}
            for r in connected
            if max_diameter is not None and r["diameter"] == max_diameter
        ],
        "minimum_budget_covering_all_measured": max_diameter,
        "budgets": per_budget,
        "interpretation": (
            "Necessary-reachability only, under an ideal synchronous protocol "
            "in which information crosses at most one edge per round. A graph "
            "with diameter d needs at least d rounds before the most distant "
            "pair can have exchanged anything. This is NOT a guarantee of "
            "consensus, faithful forwarding, correct reasoning or task "
            "success, and it does not account for Track B's unimplemented "
            "initialisation, delivery and readout conventions, which may "
            "consume an extra round."
        ),
    }


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------

def _render(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    widths = [len(h) for h in header]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    lines = [
        "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(header)) + " |",
        "|" + "|".join("-" * (w + 2) for w in widths) + "|",
    ]
    for row in rows:
        lines.append(
            "| " + " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) + " |"
        )
    return "\n".join(lines)


def _flag(value: bool | None) -> str:
    if value is None:
        return "n/a"
    return "yes" if value else "NO"


def render_measurement_table(
    rows: Sequence[dict], round_budgets: Sequence[int] = DEFAULT_ROUND_BUDGETS
) -> str:
    """The full per-condition table, one row per measured graph."""
    budgets = [int(b) for b in round_budgets]
    header = [
        "Topology",
        "N",
        "Seed",
        "Nodes",
        "Edges",
        "MeanDeg",
        "Conn",
        "Comp",
        "Diameter",
        "AvgPath",
        *[f"<={b}r" for b in budgets],
        "Caveat",
    ]
    table = []
    for row in rows:
        table.append(
            [
                row["topology"],
                str(row["n"]),
                str(row["seed"]),
                str(row["n_nodes"]),
                str(row["n_edges"]),
                f"{row['mean_degree']:.4f}" if row["mean_degree"] is not None else "n/a",
                "yes" if row["connected"] else "no",
                str(row["n_components"]),
                str(row["diameter"]) if row["diameter"] is not None else "inf",
                (
                    f"{row['average_shortest_path_length']:.4f}"
                    if row["average_shortest_path_length"] is not None
                    else "n/a"
                ),
                *[_flag(row["round_check"].get(b)) for b in budgets],
                row["caveat"] or "-",
            ]
        )
    return _render(header, table)


def render_aggregate_table(
    aggregates: Sequence[dict], round_budgets: Sequence[int] = DEFAULT_ROUND_BUDGETS
) -> str:
    """The per-topology aggregate table."""
    budgets = [int(b) for b in round_budgets]
    header = [
        "Topology",
        "Min",
        "Max",
        "Mean",
        "Median",
        "StdDev",
        *[f"<={b}" for b in budgets],
        *[f">{b}" for b in budgets],
        "Disconn",
    ]
    table = []
    for entry in aggregates:
        table.append(
            [
                entry["topology"],
                str(entry["min"]) if entry["min"] is not None else "n/a",
                str(entry["max"]) if entry["max"] is not None else "n/a",
                f"{entry['mean']:.2f}" if entry["mean"] is not None else "n/a",
                f"{entry['median']:.1f}" if entry["median"] is not None else "n/a",
                f"{entry['std']:.4f}" if entry["std"] is not None else "n/a",
                *[str(entry[f"n_within_{b}"]) for b in budgets],
                *[str(entry[f"n_above_{b}"]) for b in budgets],
                str(entry["n_disconnected"]),
            ]
        )
    return _render(header, table)
