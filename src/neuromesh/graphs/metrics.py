"""Structural graph metrics and node-level positional covariates (Week 4).

What this module is for
-----------------------
The study compares five topologies at five scales, matched on mean degree. That
match is deliberate: edge count is a *controlled* variable, so any performance
difference has to come from how the edges are *arranged*. This module measures
that arrangement, at two levels:

* :func:`structural_metrics` -- one row per graph. The whole-graph summary:
  clustering, path length, efficiency, degree dispersion, assortativity,
  and the modularity of the predefined seven-network partition.
* :func:`positional_covariates` -- one row per node. Where each agent sits:
  centralities, core number, and its position relative to the functional
  networks (within- vs between-network degree, participation coefficient,
  within-module degree z-score).

The node-level table exists because the study's outcome is measured at
specific agents -- information enters at ``INPUT_ROLE`` and is read out at
``READOUT_ROLE``. Whether a readout agent happened to land on a hub or on a
periphery node is a positional covariate that a per-graph average cannot
express, and without it a topology effect and a placement effect are not
separable.

These are STRUCTURAL quantities only
------------------------------------
Everything here is computed from the adjacency and the role labels. Nothing
here measures agent behaviour, message fidelity, consensus, or task accuracy.
A node with high betweenness is structurally well placed to relay information;
whether an LLM at that node actually relays it faithfully is a different
question that no metric in this file answers.

Undefined values are ``None``, never 0.0
----------------------------------------
Several of these quantities genuinely do not exist for some graphs: a
disconnected graph has no finite diameter, a regular graph has no defined
degree assortativity (its correlation is 0/0). Returning 0.0 in those cases
would be a false measurement that silently pollutes any average computed over
it. Every such field is ``None``, and the aggregation code in
``datasets.py`` counts valid observations per field rather than assuming five.

Full-graph vs largest-component metrics are named apart
-------------------------------------------------------
``average_shortest_path_length`` and ``diameter`` are whole-graph quantities
and are ``None`` when the graph is disconnected. The corresponding
largest-component readings are separate fields carrying
``_largest_component`` in their names. They are never substituted for the
whole-graph value, because quoting a component's diameter as the graph's
diameter understates exactly the damage a robustness experiment is trying to
measure.
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import networkx as nx
import numpy as np

from neuromesh.contracts import INPUT_ROLE, READOUT_ROLE, YEO7

from .analysis import path_metrics
from .spec import GraphParameterError

__all__ = [
    "EIGENVECTOR_MAX_ITER",
    "EIGENVECTOR_TOL",
    "PAGERANK_ALPHA",
    "PAGERANK_MAX_ITER",
    "PAGERANK_TOL",
    "POSITIONAL_FIELDS",
    "STRUCTURAL_FIELDS",
    "degree_assortativity",
    "degree_heterogeneity",
    "degree_statistics",
    "input_readout_distance",
    "partition_modularity",
    "positional_covariates",
    "role_balance",
    "role_partition",
    "structural_metrics",
]

#: Fixed PageRank parameters. Recorded as constants rather than left to the
#: NetworkX defaults so the numbers in the Week 4 tables stay reproducible if a
#: future NetworkX release changes its own defaults. ``tol`` is tightened well
#: below the default 1e-6 and ``max_iter`` raised, so the result is converged
#: rather than merely iteration-limited.
PAGERANK_ALPHA: float = 0.85
PAGERANK_TOL: float = 1e-12
PAGERANK_MAX_ITER: int = 1000

#: Eigenvector-centrality parameters. The pure-Python power iteration is used
#: deliberately: ``nx.eigenvector_centrality_numpy`` needs SciPy, which is NOT
#: in this project's declared dependencies, so it would work locally and fail
#: in CI. Non-convergence is reported as ``None``, never as a silent number --
#: see :func:`positional_covariates`.
EIGENVECTOR_TOL: float = 1e-10
EIGENVECTOR_MAX_ITER: int = 2000

#: Modularity resolution. 1.0 is the standard Newman-Girvan value; stated
#: explicitly because modularity is not comparable across resolutions.
MODULARITY_RESOLUTION: float = 1.0

#: Column order for the per-graph structural table. Fixed so the CSV is stable.
STRUCTURAL_FIELDS: tuple[str, ...] = (
    "topology",
    "n",
    "seed",
    "graph_id",
    "source",
    "n_nodes",
    "n_edges",
    "mean_degree",
    "edge_density",
    "n_components",
    "largest_component_size",
    "largest_component_fraction",
    "average_clustering",
    "transitivity",
    "average_shortest_path_length",
    "diameter",
    "radius",
    "average_shortest_path_length_largest_component",
    "diameter_largest_component",
    "degree_min",
    "degree_max",
    "degree_mean",
    "degree_variance",
    "degree_std",
    "degree_heterogeneity",
    "global_efficiency",
    "degree_assortativity",
    "modularity_role_partition",
    "roles_assigned",
    "roles_balanced",
    "roles_per_network",
    "n_roles_present",
    "input_readout_mean_distance",
    "input_readout_min_distance",
    "input_readout_max_distance",
    "input_readout_unreachable_pairs",
    "input_readout_pairs",
    "hub_role",
    "hub_degree",
    "target_mean_degree",
    "degree_matched_exactly",
    "generator_version",
    "networkx_version",
)

#: Column order for the node-level covariate table.
POSITIONAL_FIELDS: tuple[str, ...] = (
    "topology",
    "n",
    "seed",
    "graph_id",
    "node_id",
    "role",
    "community",
    "degree",
    "degree_centrality",
    "betweenness_centrality",
    "closeness_centrality",
    "clustering",
    "core_number",
    "eigenvector_centrality",
    "pagerank",
    "within_network_degree",
    "between_network_degree",
    "within_network_degree_fraction",
    "between_network_degree_fraction",
    "n_neighbor_networks",
    "participation_coefficient",
    "within_module_z",
)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _require_graph(graph: Any) -> nx.Graph:
    if not isinstance(graph, nx.Graph):
        raise GraphParameterError(
            f"expected a networkx.Graph, got {type(graph).__name__}"
        )
    if graph.is_directed():
        raise GraphParameterError(
            "the Week 4 metrics are defined for undirected graphs; got a "
            "directed graph"
        )
    return graph


def _finite_or_none(value: Any) -> float | None:
    """Return ``float(value)`` unless it is NaN or infinite, else ``None``.

    The gate that keeps a NaN from a degenerate correlation out of the tables.
    """
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def graph_id(topology: str, n: int, seed: int) -> str:
    """Stable instance identifier, e.g. ``er_n014_seed0``.

    Deliberately identical to :func:`neuromesh.graphs.fixtures.fixture_filename`
    minus the extension, so a row in the structural table, a row in the node
    table, a robustness trajectory and a committed fixture file all join on the
    same key.
    """
    return f"{topology}_n{int(n):03d}_seed{int(seed)}"


def degree_statistics(graph: nx.Graph) -> dict:
    """Degree sequence summary: min, max, mean, variance, std, heterogeneity.

    ``degree_variance`` is the POPULATION variance (``ddof=0``). The degree
    sequence is the complete population of that graph's nodes, not a sample
    drawn from it, so Bessel's correction does not apply here. This is a
    different convention from the across-seed aggregation in ``datasets.py``,
    which uses the SAMPLE standard deviation (``ddof=1``) because five seeds
    *are* a sample from the generator's ensemble. Both are stated wherever
    they are reported.
    """
    graph = _require_graph(graph)
    if graph.number_of_nodes() == 0:
        return {
            "degree_min": None,
            "degree_max": None,
            "degree_mean": None,
            "degree_variance": None,
            "degree_std": None,
            "degree_heterogeneity": None,
        }
    degrees = np.asarray([d for _, d in graph.degree()], dtype=float)
    mean = float(degrees.mean())
    variance = float(degrees.var(ddof=0))
    return {
        "degree_min": int(degrees.min()),
        "degree_max": int(degrees.max()),
        "degree_mean": mean,
        "degree_variance": variance,
        "degree_std": float(math.sqrt(variance)),
        "degree_heterogeneity": degree_heterogeneity(graph),
    }


def degree_heterogeneity(graph: nx.Graph) -> float | None:
    """``<k^2> / <k>^2``, the standard degree-heterogeneity ratio.

    Equals exactly 1.0 for any regular graph and grows as the degree
    distribution broadens, so it separates BA's hub structure from WS's ring
    lattice independently of mean degree. ``None`` for an empty graph or one
    with no edges at all (``<k> = 0`` makes the ratio 0/0).

    Chosen over the alternative ``variance / mean`` because the ratio is
    dimensionless and scale-free: two graphs matched on mean degree, as all of
    this study's are, can be compared directly.
    """
    graph = _require_graph(graph)
    if graph.number_of_nodes() == 0:
        return None
    degrees = np.asarray([d for _, d in graph.degree()], dtype=float)
    mean = degrees.mean()
    if mean <= 0:
        return None
    return float((degrees**2).mean() / (mean**2))


def degree_assortativity(graph: nx.Graph) -> float | None:
    """Pearson degree-degree correlation over edges, or ``None`` if undefined.

    NetworkX returns NaN when the coefficient is 0/0, which happens for every
    regular graph -- the WS ring lattice before rewiring is exactly that case.
    NaN is converted to ``None`` here, and the RuntimeWarning NumPy emits is
    suppressed, because a degenerate correlation is a *missing* measurement
    rather than a numerical accident worth a warning in every test run.

    Reported as ``None`` and NOT as 0.0: zero assortativity means "measured,
    and neutral", which is a different and much stronger claim.
    """
    graph = _require_graph(graph)
    if graph.number_of_edges() == 0:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with np.errstate(invalid="ignore", divide="ignore"):
            try:
                value = nx.degree_assortativity_coefficient(graph)
            except (ZeroDivisionError, ValueError):
                return None
    return _finite_or_none(value)


def role_partition(graph: nx.Graph) -> list[set] | None:
    """The predefined seven-network partition, as a list of node sets.

    Built from the ``role`` node attribute in :data:`YEO7` order, so the
    partition is the SAME OBJECT for every topology family: the seven
    functional networks. Returns ``None`` when the graph carries no roles.

    Empty networks are dropped from the returned list, because
    ``nx.community.modularity`` rejects an empty community. The count of
    surviving networks is reported separately as ``n_roles_present`` so a
    partition that lost a network to an attack is still visible.
    """
    graph = _require_graph(graph)
    roles = nx.get_node_attributes(graph, "role")
    if not roles:
        return None
    buckets: dict[str, set] = {role: set() for role in YEO7}
    for node, role in roles.items():
        if role not in buckets:
            raise GraphParameterError(
                f"node {node!r} carries unknown role {role!r}; expected one of "
                f"{list(YEO7)}"
            )
        buckets[role].add(node)
    unlabelled = set(graph.nodes()) - set(roles)
    if unlabelled:
        raise GraphParameterError(
            f"{len(unlabelled)} of {graph.number_of_nodes()} nodes carry no "
            "'role' attribute; a partial partition cannot be scored"
        )
    return [nodes for nodes in buckets.values() if nodes]


def partition_modularity(graph: nx.Graph) -> float | None:
    """Newman-Girvan modularity of the PREDEFINED seven-network partition.

    ``nx.community.modularity(graph, role_partition(graph),
    resolution=MODULARITY_RESOLUTION)``.

    This is deliberately NOT community detection. The question Week 4 asks is
    "how modular is the arrangement of links *with respect to the seven
    functional networks the agents actually play*", and that has one answer per
    graph. Running a detection algorithm would answer a different question --
    "what is the most modular partition available" -- which is topology-
    dependent in a way that makes the five families incomparable, and which
    would score ER against a partition ER does not have.

    The same fixed partition is therefore used for all five topologies. For
    ``modular`` and ``yeo_sbm`` it coincides with the planted blocks by
    construction, so their values are expected to be high; for ER, WS and BA
    the roles are a random labelling of structural positions, so values near
    the configuration-model baseline are expected. That contrast is the
    measurement, not an artefact -- but see ``docs/track_a.md`` on the
    role-placement confound before comparing across the two groups.

    ``None`` when the graph carries no roles or has no edges (modularity is
    undefined with zero edges).
    """
    graph = _require_graph(graph)
    if graph.number_of_edges() == 0:
        return None
    partition = role_partition(graph)
    if partition is None:
        return None
    value = nx.community.modularity(
        graph, partition, resolution=MODULARITY_RESOLUTION
    )
    return _finite_or_none(value)


def role_balance(graph: nx.Graph) -> dict:
    """Role histogram plus whether the seven networks are exactly balanced."""
    graph = _require_graph(graph)
    roles = nx.get_node_attributes(graph, "role")
    if not roles:
        return {
            "roles_assigned": False,
            "roles_balanced": None,
            "roles_per_network": None,
            "n_roles_present": 0,
            "role_counts": None,
        }
    counts = dict.fromkeys(YEO7, 0)
    for role in roles.values():
        if role not in counts:
            raise GraphParameterError(f"unknown role {role!r}")
        counts[role] += 1
    n = graph.number_of_nodes()
    expected = n // len(YEO7)
    balanced = (
        len(roles) == n
        and n % len(YEO7) == 0
        and set(counts.values()) == {expected}
    )
    return {
        "roles_assigned": True,
        "roles_balanced": bool(balanced),
        "roles_per_network": expected if balanced else None,
        "n_roles_present": sum(1 for v in counts.values() if v > 0),
        "role_counts": counts,
    }


def input_readout_distance(graph: nx.Graph) -> dict:
    """Shortest-path distances from ``INPUT_ROLE`` to ``READOUT_ROLE`` agents.

    Named in the ``contracts.py`` description of ``structural_metrics``, and
    directly relevant to the round budget: information must cross this distance
    before a readout agent can report anything at all.

    Distances are taken over ordered (input, readout) pairs. Pairs in different
    components are UNREACHABLE and excluded from the mean, with the excluded
    count reported, rather than being scored as zero or as infinity. ``None``
    everywhere when either role group is absent or no pair is reachable.
    """
    graph = _require_graph(graph)
    roles = nx.get_node_attributes(graph, "role")
    if not roles:
        return {
            "input_readout_mean_distance": None,
            "input_readout_min_distance": None,
            "input_readout_max_distance": None,
            "input_readout_unreachable_pairs": None,
            "input_readout_pairs": None,
        }
    sources = sorted(n for n, r in roles.items() if r == INPUT_ROLE)
    targets = sorted(n for n, r in roles.items() if r == READOUT_ROLE)
    if not sources or not targets:
        return {
            "input_readout_mean_distance": None,
            "input_readout_min_distance": None,
            "input_readout_max_distance": None,
            "input_readout_unreachable_pairs": None,
            "input_readout_pairs": 0,
        }

    distances: list[int] = []
    unreachable = 0
    for source in sources:
        lengths = nx.single_source_shortest_path_length(graph, source)
        for target in targets:
            if target == source:
                continue
            if target in lengths:
                distances.append(int(lengths[target]))
            else:
                unreachable += 1
    total_pairs = len(distances) + unreachable
    if not distances:
        return {
            "input_readout_mean_distance": None,
            "input_readout_min_distance": None,
            "input_readout_max_distance": None,
            "input_readout_unreachable_pairs": unreachable,
            "input_readout_pairs": total_pairs,
        }
    return {
        "input_readout_mean_distance": float(sum(distances) / len(distances)),
        "input_readout_min_distance": int(min(distances)),
        "input_readout_max_distance": int(max(distances)),
        "input_readout_unreachable_pairs": unreachable,
        "input_readout_pairs": total_pairs,
    }


def _hub(graph: nx.Graph) -> dict:
    """The highest-degree node's role and degree; lowest ID breaks ties.

    "Hub roles" is named in the ``contracts.py`` description of
    ``structural_metrics``. The tie-break matches
    ``robustness.DEFAULT_TIE_BREAK`` so the node reported here is the node a
    targeted attack would strike first.
    """
    if graph.number_of_nodes() == 0:
        return {"hub_role": None, "hub_degree": None}
    degrees = dict(graph.degree())
    top = max(degrees.values())
    hub = min(node for node, degree in degrees.items() if degree == top)
    roles = nx.get_node_attributes(graph, "role")
    return {"hub_role": roles.get(hub), "hub_degree": int(top)}


# --------------------------------------------------------------------------
# Deliverable A: per-graph structural metrics
# --------------------------------------------------------------------------

def structural_metrics(
    graph: nx.Graph,
    *,
    topology: str | None = None,
    n: int | None = None,
    seed: int | None = None,
    source: str = "generated",
) -> dict:
    """Whole-graph structural summary. One row of the Week 4 structural table.

    This is the Track A entry point named in ``contracts.py``. The identity
    keyword arguments are optional: when omitted they are read from
    ``graph.graph`` (``topology``, ``n``, ``seed``), which every graph from
    :func:`neuromesh.graphs.generate_graph` carries. So the contract's
    one-argument call ``structural_metrics(graph)`` works, and the table
    builder can pass identities explicitly for a graph that lost them.

    Returns
    -------
    dict
        Keys are exactly :data:`STRUCTURAL_FIELDS` plus the extra
        input-readout bookkeeping fields. Undefined values are ``None``.

    Metric definitions, in one place
    --------------------------------
    ``mean_degree``                 2E/N.
    ``edge_density``                2E / (N(N-1)); ``None`` for N < 2.
    ``average_clustering``          Mean over nodes of the local clustering
                                    coefficient (``nx.average_clustering``);
                                    a degree-<2 node contributes 0 by the
                                    NetworkX convention.
    ``transitivity``                3 x triangles / triads
                                    (``nx.transitivity``) -- a GLOBAL ratio,
                                    not the mean of the local values, so it
                                    differs from ``average_clustering`` and
                                    both are reported.
    ``average_shortest_path_length`` Whole graph, ``None`` if disconnected.
    ``diameter`` / ``radius``       Whole graph, ``None`` if disconnected.
    ``*_largest_component``         The same two quantities on the largest
                                    component; never substituted above.
    ``degree_variance``             Population variance, ddof=0.
    ``degree_heterogeneity``        <k^2>/<k>^2; 1.0 for a regular graph.
    ``global_efficiency``           Mean over ordered node pairs of 1/d, with
                                    1/d = 0 for an unreachable pair
                                    (``nx.global_efficiency``). Defined for
                                    disconnected graphs, which is why it is the
                                    robustness curves' efficiency measure.
                                    In [0, 1]; 1.0 only for a complete graph.
    ``degree_assortativity``        Pearson degree-degree correlation;
                                    ``None`` when undefined (regular graphs).
    ``modularity_role_partition``   Newman-Girvan modularity of the predefined
                                    seven-network partition at resolution 1.0.
                                    See :func:`partition_modularity`.
    """
    graph = _require_graph(graph)
    attrs = graph.graph
    topology = topology if topology is not None else attrs.get("topology")
    n = n if n is not None else attrs.get("n", graph.number_of_nodes())
    seed = seed if seed is not None else attrs.get("seed")

    paths = path_metrics(graph)
    n_nodes = paths["n_nodes"]
    n_edges = paths["n_edges"]

    density = (
        (2.0 * n_edges) / (n_nodes * (n_nodes - 1)) if n_nodes >= 2 else None
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        clustering = (
            float(nx.average_clustering(graph)) if n_nodes else None
        )
        transitivity = float(nx.transitivity(graph)) if n_nodes else None
        efficiency = float(nx.global_efficiency(graph)) if n_nodes > 1 else (
            0.0 if n_nodes == 1 else None
        )

    balance = role_balance(graph)
    distances = input_readout_distance(graph)
    generator_metadata = attrs.get("generator_metadata") or {}

    row: dict[str, Any] = {
        "topology": topology,
        "n": int(n) if n is not None else None,
        "seed": int(seed) if seed is not None else None,
        "graph_id": (
            graph_id(topology, n, seed)
            if None not in (topology, n, seed)
            else None
        ),
        "source": source,
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "mean_degree": _finite_or_none(paths["mean_degree"]),
        "edge_density": _finite_or_none(density),
        "n_components": paths["n_components"],
        "largest_component_size": paths["largest_component_size"],
        "largest_component_fraction": (
            paths["largest_component_size"] / n_nodes if n_nodes else None
        ),
        "average_clustering": _finite_or_none(clustering),
        "transitivity": _finite_or_none(transitivity),
        "average_shortest_path_length": _finite_or_none(
            paths["average_shortest_path_length"]
        ),
        "diameter": paths["diameter"],
        "radius": paths["radius"],
        "average_shortest_path_length_largest_component": _finite_or_none(
            paths["average_shortest_path_length_largest_component"]
        ),
        "diameter_largest_component": paths["max_finite_distance"],
        **degree_statistics(graph),
        "global_efficiency": _finite_or_none(efficiency),
        "degree_assortativity": degree_assortativity(graph),
        "modularity_role_partition": partition_modularity(graph),
        "roles_assigned": balance["roles_assigned"],
        "roles_balanced": balance["roles_balanced"],
        "roles_per_network": balance["roles_per_network"],
        "n_roles_present": balance["n_roles_present"],
        **distances,
        **_hub(graph),
        "target_mean_degree": _finite_or_none(attrs.get("target_mean_degree")),
        "degree_matched_exactly": generator_metadata.get(
            "degree_matched_exactly"
        ),
        "generator_version": generator_metadata.get("generator_version"),
        "networkx_version": nx.__version__,
    }
    return row


# --------------------------------------------------------------------------
# Deliverable B: node-level positional covariates
# --------------------------------------------------------------------------

def _eigenvector_centrality(graph: nx.Graph) -> tuple[dict, str]:
    """Power-iteration eigenvector centrality, honestly reporting failure.

    Returns ``(values, status)``. On non-convergence the values are empty and
    the status says so; the caller then writes ``None`` into every node's
    field rather than a number that did not converge.

    The pure-Python iteration is used on purpose: the SciPy-backed variant is
    not available under this project's declared dependencies.
    """
    if graph.number_of_nodes() == 0:
        return {}, "empty_graph"
    if graph.number_of_edges() == 0:
        # Every node is isolated; the adjacency matrix is zero and the
        # principal eigenvector is not defined. Reported, not invented.
        return {}, "no_edges"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            values = nx.eigenvector_centrality(
                graph, max_iter=EIGENVECTOR_MAX_ITER, tol=EIGENVECTOR_TOL
            )
        return {k: float(v) for k, v in values.items()}, "converged"
    except (nx.PowerIterationFailedConvergence, nx.NetworkXError) as exc:
        return {}, f"failed: {type(exc).__name__}"


def positional_covariates(
    graph: nx.Graph,
    *,
    topology: str | None = None,
    n: int | None = None,
    seed: int | None = None,
) -> list[dict]:
    """One row per node: where that agent sits in the graph and in its network.

    Rows are returned in ascending node-ID order, so the table is stable.

    Returns
    -------
    list[dict]
        Each dict's keys are exactly :data:`POSITIONAL_FIELDS`.

    Covariate definitions
    ---------------------
    ``degree``                     Number of neighbours.
    ``degree_centrality``          degree / (N-1); ``None`` for N < 2.
    ``betweenness_centrality``     ``nx.betweenness_centrality``,
                                   normalised (divided by (N-1)(N-2)/2 for an
                                   undirected graph), so values are in [0, 1]
                                   and comparable across the five scales.
    ``closeness_centrality``       ``nx.closeness_centrality`` with
                                   ``wf_improved=True`` (the NetworkX
                                   default): the Wasserman-Faust correction,
                                   which scales each node's value by the
                                   fraction of the graph it can reach. This
                                   matters only for disconnected graphs; all
                                   125 study graphs are connected, where it
                                   reduces to standard closeness.
    ``clustering``                 Local clustering coefficient; 0 for a node
                                   of degree < 2 (NetworkX convention).
    ``core_number``                Largest k for which the node survives in
                                   the k-core.
    ``eigenvector_centrality``     Power iteration, tol 1e-10, max_iter 2000.
                                   ``None`` for EVERY node if the iteration
                                   did not converge -- never a partial or
                                   unconverged value.
    ``pagerank``                   alpha 0.85, tol 1e-12, max_iter 1000.
    ``role``                       The node's Yeo functional network.
    ``community``                  The planted block index, for ``modular``
                                   and ``yeo_sbm``; ``None`` otherwise.
    ``within_network_degree``      Neighbours sharing the node's role.
    ``between_network_degree``     Neighbours with a different role.
    ``*_fraction``                 The two above divided by degree. ``None``
                                   for a degree-0 node, because 0/0 is not a
                                   fraction -- NOT 0.0, which would read as
                                   "this node has no within-network links"
                                   when in truth it has no links at all.
    ``n_neighbor_networks``        Count of distinct roles among neighbours
                                   (0..7).
    ``participation_coefficient``  ``P_i = 1 - sum_s (k_is / k_i)^2`` over the
                                   seven networks. 0.0 BY DEFINITION for a
                                   degree-0 node: an isolated agent
                                   participates in nothing. In [0, 6/7] for
                                   seven networks, so never 1.0 exactly.
    ``within_module_z``            ``z_i = (k_in,i - mean) / std`` over the
                                   nodes of the SAME role, using the
                                   population std (ddof=0) of that module.
                                   0.0 when the module's within-degrees have
                                   ZERO VARIANCE (including a single-node
                                   module, as at N = 7): every member is
                                   exactly at its module's mean, so a
                                   deviation of 0 is the correct reading and
                                   the alternative is a division by zero.
    """
    graph = _require_graph(graph)
    attrs = graph.graph
    topology = topology if topology is not None else attrs.get("topology")
    n = n if n is not None else attrs.get("n", graph.number_of_nodes())
    seed = seed if seed is not None else attrs.get("seed")
    gid = (
        graph_id(topology, n, seed) if None not in (topology, n, seed) else None
    )

    nodes = sorted(graph.nodes())
    n_nodes = len(nodes)
    if n_nodes == 0:
        return []

    roles = nx.get_node_attributes(graph, "role")
    communities = nx.get_node_attributes(graph, "community")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        betweenness = nx.betweenness_centrality(graph, normalized=True)
        closeness = nx.closeness_centrality(graph, wf_improved=True)
        clustering = nx.clustering(graph)
        core = nx.core_number(graph)
        pagerank = nx.pagerank(
            graph,
            alpha=PAGERANK_ALPHA,
            tol=PAGERANK_TOL,
            max_iter=PAGERANK_MAX_ITER,
        )
    eigenvector, _eig_status = _eigenvector_centrality(graph)

    # Within-module degree per node, needed before the z-scores can be taken.
    within_degree: dict[Any, int] = {}
    between_degree: dict[Any, int] = {}
    neighbour_networks: dict[Any, set] = {}
    for node in nodes:
        own = roles.get(node)
        within = 0
        between = 0
        seen: set = set()
        for neighbour in graph.neighbors(node):
            other = roles.get(neighbour)
            seen.add(other)
            if own is not None and other == own:
                within += 1
            else:
                between += 1
        within_degree[node] = within
        between_degree[node] = between
        neighbour_networks[node] = seen

    # Module statistics for the z-score, keyed by role.
    module_stats: dict[Any, tuple[float, float]] = {}
    if roles:
        by_role: dict[Any, list[int]] = {}
        for node in nodes:
            by_role.setdefault(roles.get(node), []).append(within_degree[node])
        for role, values in by_role.items():
            array = np.asarray(values, dtype=float)
            module_stats[role] = (float(array.mean()), float(array.std(ddof=0)))

    rows: list[dict] = []
    for node in nodes:
        degree = int(graph.degree(node))
        own_role = roles.get(node)

        if degree == 0:
            within_fraction = None
            between_fraction = None
            participation = 0.0
        else:
            within_fraction = within_degree[node] / degree
            between_fraction = between_degree[node] / degree
            counts: dict[Any, int] = {}
            for neighbour in graph.neighbors(node):
                key = roles.get(neighbour)
                counts[key] = counts.get(key, 0) + 1
            participation = 1.0 - sum(
                (count / degree) ** 2 for count in counts.values()
            )

        if own_role is not None and own_role in module_stats:
            mean, std = module_stats[own_role]
            z = 0.0 if std == 0.0 else (within_degree[node] - mean) / std
        else:
            z = None

        rows.append(
            {
                "topology": topology,
                "n": int(n) if n is not None else None,
                "seed": int(seed) if seed is not None else None,
                "graph_id": gid,
                "node_id": int(node) if isinstance(node, (int, np.integer)) else node,
                "role": own_role,
                "community": (
                    int(communities[node]) if node in communities else None
                ),
                "degree": degree,
                "degree_centrality": (
                    degree / (n_nodes - 1) if n_nodes >= 2 else None
                ),
                "betweenness_centrality": _finite_or_none(betweenness.get(node)),
                "closeness_centrality": _finite_or_none(closeness.get(node)),
                "clustering": _finite_or_none(clustering.get(node)),
                "core_number": int(core[node]),
                "eigenvector_centrality": (
                    _finite_or_none(eigenvector.get(node))
                    if eigenvector
                    else None
                ),
                "pagerank": _finite_or_none(pagerank.get(node)),
                "within_network_degree": within_degree[node],
                "between_network_degree": between_degree[node],
                "within_network_degree_fraction": _finite_or_none(within_fraction),
                "between_network_degree_fraction": _finite_or_none(
                    between_fraction
                ),
                "n_neighbor_networks": len(
                    {r for r in neighbour_networks[node] if r is not None}
                ),
                "participation_coefficient": float(participation),
                "within_module_z": _finite_or_none(z),
            }
        )
    return rows


def eigenvector_status(graph: nx.Graph) -> str:
    """Convergence status of the eigenvector-centrality iteration.

    Exposed so the analysis driver can record non-convergence in its report
    rather than leaving an unexplained column of ``None``.
    """
    return _eigenvector_centrality(_require_graph(graph))[1]
