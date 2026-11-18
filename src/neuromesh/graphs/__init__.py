"""Track A (Mihir): graph generation, roles, node removal, structural metrics.

Public entry points only. Implement them in this package (e.g. generators.py,
yeo_sbm.py, roles.py, removal.py, metrics.py) and keep these signatures.

Status
------
``generate_graph`` is implemented for all five topologies: ``er``, ``ws``,
``ba``, ``modular`` and ``yeo_sbm``. ``assign_roles`` is implemented and
assigns the seven Yeo functional roles in balance. ``remove_nodes`` is
implemented as of Week 3 -- random failure and adaptive highest-degree attack,
in ``robustness.py``. ``structural_metrics`` is implemented as of Week 4, in
``metrics.py``, alongside node-level positional covariates
(``positional_covariates``), graph-only robustness curves (``curves.py``), the
Week 4 analysis datasets (``datasets.py``) and dependency-free SVG figures
(``figures.py``). No Track A entry point in ``contracts.py`` is a placeholder
any more.

``yeo_sbm`` is defined by empirical data: it needs the derived 7x7 inter-network
structural matrix (``data/yeo/yeo7x7_sc.json``) and raises
:class:`YeoMatrixUnavailableError` when that file is absent, rather than
falling back to invented block weights.

Two layers live here, deliberately
----------------------------------
1. **The shared interface** the runner calls, defined by the team in
   ``contracts.py``: ``generate_graph(topology, n, mean_degree, seed)`` returns
   a bare :class:`networkx.Graph` and attaches ``topology``, ``n``, ``seed``
   and ``mean_degree`` to ``G.graph``.
2. **The Track A generator layer** (``spec.py``, ``generators.py``,
   ``validation.py``, ``fixtures.py``, ``cli.py``), which returns
   ``(graph, GraphMetadata)`` so that provenance -- the rejection-sampling draw
   count, the exact generator parameters, the NetworkX/NumPy versions -- is
   never lost. Its spec-based dispatcher is exported here as
   :func:`generate_from_spec`; it is unchanged and still available as
   ``neuromesh.graphs.generators.generate_graph``.

Layer 1 is a thin adapter over layer 2. No mathematics happens in the adapter.

Quick start
-----------

The shared interface:

>>> from neuromesh.graphs import generate_graph
>>> graph = generate_graph("ws", 14)
>>> graph.number_of_nodes(), graph.graph["mean_degree"]
(14, 4.0)

Loading a committed fixture instead of generating:

>>> from neuromesh.graphs import load_graph
>>> graph, meta = load_graph("ws", 14)     # doctest: +SKIP
>>> graph.number_of_nodes(), meta["mean_degree"]   # doctest: +SKIP
(14, 4.0)

The generator layer, with full metadata:

>>> from neuromesh.graphs import GraphSpec, generate_from_spec
>>> graph, meta = generate_from_spec(GraphSpec("er", n=28, seed=3))
>>> graph.number_of_nodes(), meta.attempts >= 1
(28, True)
"""

from __future__ import annotations

import networkx as nx

from neuromesh.contracts import DEFAULT_MEAN_DEGREE, RemovalSpec, Topology

from .analysis import (
    DEFAULT_ROUND_BUDGETS,
    STD_CONVENTION,
    aggregate_path_metrics,
    diameter_table,
    measure_graph,
    path_metrics,
    render_aggregate_table,
    render_measurement_table,
    round_budget_assessment,
)
from .blocks import (
    N_BLOCKS,
    allocate_block_edges,
    balanced_block_sizes,
    block_assignment,
    community_profile,
    pair_capacity,
    sample_block_graph,
    validate_block_weights,
)
from .curves import (
    AUC_FIELDS,
    CHECKPOINT_FRACTIONS,
    MAX_REMOVAL_FRACTION,
    ROBUSTNESS_FIELDS,
    checkpoint_rows,
    robustness_auc,
    robustness_trajectory,
    trajectory_rows,
)
from .datasets import (
    AGGREGATE_METRICS,
    CONDITION_FIELDS,
    DEFAULT_OUTPUT_DIR,
    aggregate_conditions,
    build_positional_table,
    build_robustness_tables,
    build_structural_table,
    expected_node_row_count,
    run_week4_analysis,
)
from .fixtures import (
    DEFAULT_FIXTURE_DIR,
    SCHEMA_VERSION,
    fixture_filename,
    generate_fixture_set,
    graph_to_record,
    load_fixture,
    load_fixture_set,
    load_graph,
    record_to_graph,
    save_fixture,
)
from .generators import (
    GraphMetadata,
    barabasi_albert,
    erdos_renyi,
    mean_degree,
    modular,
    target_edge_count,
    watts_strogatz,
    yeo_sbm,
)
from .generators import (
    generate_graph as generate_from_spec,
)
from .metrics import (
    POSITIONAL_FIELDS,
    STRUCTURAL_FIELDS,
    degree_assortativity,
    degree_heterogeneity,
    degree_statistics,
    graph_id,
    input_readout_distance,
    partition_modularity,
    positional_covariates,
    role_balance,
    role_partition,
)
from .metrics import (
    structural_metrics as _structural_metrics,
)
from .robustness import (
    DEFAULT_ROUNDING_RULE,
    DEFAULT_TIE_BREAK,
    REMOVAL_SEED_SALT,
    REMOVAL_STRATEGIES,
    ROUNDING_RULES,
    TIE_BREAKS,
    RemovalResult,
    canonical_nodes,
    random_removal_order,
    removal_count,
    remove_node_set,
    remove_nodes_by_count,
    remove_nodes_by_fraction,
    static_removal_order,
    targeted_removal_order,
    validate_attacked_graph,
)
from .roles import assign_roles, role_counts, roles_are_balanced
from .spec import (
    BLOCK_TOPOLOGIES,
    BOUNDARY_N,
    DEFAULT_FIXTURE_SEED,
    DEFAULT_WITHIN_BETWEEN_RATIO,
    DEGREE_TOLERANCE,
    EXPERIMENT_SEEDS,
    GENERATOR_VERSION,
    GENERATOR_VERSIONS,
    POPULATION_SIZES,
    TARGET_MEAN_DEGREE,
    TOPOLOGIES,
    GraphParameterError,
    GraphSpec,
    generator_version,
)
from .validation import (
    Check,
    ValidationResult,
    check_reproducible,
    degree_deviation,
    validate_graph,
)
from .yeo import (
    DEFAULT_YEO_MATRIX_PATH,
    SCHAEFER_TO_YEO7,
    YeoMatrixUnavailableError,
    derive_yeo7_matrix,
    load_yeo_matrix,
    parse_schaefer_label,
    validate_yeo7_record,
)

#: Topologies named in ``contracts.Topology`` that Track A has not built yet.
#: Empty since Week 2: all five declared topologies are implemented.
PLANNED_TOPOLOGIES: tuple[str, ...] = ()


def generate_graph(
    topology: Topology, n: int, mean_degree: float = DEFAULT_MEAN_DEGREE, seed: int = 0
) -> nx.Graph:
    """Connected graph with nodes 0..n-1, mean degree within +/-10% (N >= 14).

    This is the shared entry point from ``contracts.py``. It is an adapter over
    :func:`generate_from_spec`: it does no generation of its own, so the
    mathematics stays in ``generators.py`` where it is documented and tested.

    Parameters
    ----------
    topology:
        Any of ``"er"``, ``"ws"``, ``"ba"``, ``"modular"`` or ``"yeo_sbm"`` --
        the full set declared in ``contracts.Topology``. The two block models
        additionally give every node a ``community`` attribute in ``0..6``.
    n:
        Number of agents. Node IDs are always the integers ``0 .. n-1``.
    mean_degree:
        Target mean degree ``2E/N``, default 4.0. All three Week 1 generators
        hit it *exactly* at every one of the five population sizes (ER uses
        ``G(N, M)`` with ``M = 2N``, WS rewiring preserves the ring lattice's
        edge count, and BA is seeded so ``E = m*N``).
    seed:
        Integer seed. Identical arguments always yield an identical edge set.

    Returns
    -------
    networkx.Graph
        A bare graph -- *not* a ``(graph, metadata)`` tuple. ``G.graph`` carries
        the four keys the contract requires (``topology``, ``n``, ``seed``,
        ``mean_degree``) plus ``target_mean_degree`` and
        ``generator_metadata``, the full :class:`GraphMetadata` provenance dict
        (draw count, generator parameters, library versions).

        ``G.graph["mean_degree"]`` is the *realised* ``2E/N`` of the graph
        returned, and ``G.graph["target_mean_degree"]`` is what was requested.
        For every shipped configuration the two are equal; they are kept
        separate so a future topology that can only match in expectation cannot
        misreport itself.

    Raises
    ------
    GraphParameterError
        For an unknown topology or structurally invalid ``n``, ``seed`` or
        ``mean_degree``. Subclasses ``ValueError``.
    YeoMatrixUnavailableError
        For ``"yeo_sbm"`` when the derived empirical 7x7 matrix is absent.

    Notes
    -----
    ER draws from the *connected* ER ensemble by rejection sampling, because
    mean degree 4 is below the ``ln N`` connectivity threshold at N = 112. That
    conditions the ensemble; the number of draws is recorded in
    ``G.graph["generator_metadata"]["attempts"]``. See ``docs/track_a.md``.
    """
    if topology not in TOPOLOGIES:
        raise GraphParameterError(
            f"unknown topology {topology!r}; implemented {TOPOLOGIES}"
        )

    graph, metadata = generate_from_spec(
        GraphSpec(topology=topology, n=n, seed=seed, target_mean_degree=mean_degree)
    )
    graph.graph.update(
        {
            "topology": metadata.topology,
            "n": metadata.n,
            "seed": metadata.seed,
            "mean_degree": metadata.mean_degree,
            "target_mean_degree": metadata.target_mean_degree,
            "generator_metadata": metadata.to_dict(),
        }
    )
    return graph


def remove_nodes(graph: nx.Graph, spec: RemovalSpec, seed: int) -> nx.Graph:
    """Random or adaptive highest-degree removal, by fraction. Shared interface.

    This is the entry point named in ``contracts.py`` and called by the runner.
    Like :func:`generate_graph` it is a thin adapter: the algorithms live in
    ``robustness.py`` and nothing here decides anything scientific.

    Parameters
    ----------
    graph:
        The graph to attack. **Not modified.** Usually a generated graph with
        roles already assigned, so the surviving graph keeps its ``role`` and
        ``community`` attributes.
    spec:
        ``RemovalSpec(mode=..., fraction=...)``. ``mode="random"`` is
        indiscriminate failure; ``mode="targeted"`` is the adaptive
        highest-degree attack (degrees recomputed after every removal). The
        fraction converts to a node count by
        :func:`~neuromesh.graphs.robustness.removal_count` with the **floor**
        rule, so at N = 112 the study's 10/20/30% levels remove 11, 22 and 33
        nodes (9.82%, 19.64%, 29.46%).
    seed:
        Non-negative integer. Identical ``(graph, spec, seed)`` always yields
        an identical surviving graph.

    Returns
    -------
    networkx.Graph
        The surviving induced subgraph: original node IDs, original node and
        edge attributes, and exactly those original edges whose endpoints both
        survived. Node IDs are **not** renumbered, roles are **not**
        rebalanced, and connectivity is **not** restored.

        ``G.graph["removal"]`` carries the full attack record (removal order,
        the degree each node had when it was removed, component structure,
        surviving role histogram) -- the same dict as
        ``RemovalResult.to_dict()``. ``G.graph["mean_degree"]`` is updated to
        the realised post-attack ``2E/N'``; ``topology``, ``n`` and ``seed``
        are left untouched because they identify the experimental condition,
        not the live node count. Use ``G.graph["removal"]["n_surviving"]`` for
        that.

        Callers needing the removal order itself should prefer
        :func:`~neuromesh.graphs.robustness.remove_nodes_by_count`, which
        returns a :class:`~neuromesh.graphs.robustness.RemovalResult` instead
        of discarding everything but the graph.

    Raises
    ------
    GraphParameterError
        Invalid seed, or a non-graph/directed input. Subclasses ``ValueError``.

    Notes
    -----
    **An attacked graph is expected to be disconnected and off mean degree
    4.** It must not be passed to :func:`validate_graph`, whose connectivity
    and mean-degree checks encode *generator* contracts;
    :func:`~neuromesh.graphs.robustness.validate_attacked_graph` is the
    checker for this output.

    **Readout agents are not protected.** The Week 1 placeholder docstring for
    this function promised it would "never remove every readout node". That
    promise is deliberately not implemented, because silently sparing nodes
    would make the attack something other than the attack that was requested
    and would hide exactly the failure mode the robustness experiment exists
    to find. Instead the outcome is measured and reported:
    ``G.graph["removal"]["readout_surviving"]`` and ``["roles_intact"]``. A
    caller who genuinely wants a protected set can pass ``protected=`` to
    ``remove_nodes_by_count``. Raised for the team as a design question --
    see ``docs/track_a.md`` §16.6.
    """
    result = remove_nodes_by_fraction(
        graph,
        spec.fraction,
        spec.mode,
        seed=seed,
        rule=DEFAULT_ROUNDING_RULE,
    )
    return result.graph


def structural_metrics(graph: nx.Graph) -> dict:
    """Path length, clustering, modularity, max degree, input-readout distance, hub roles.

    The shared ``contracts.py`` entry point, implemented in Week 4. Like
    :func:`generate_graph` and :func:`remove_nodes` it is a thin adapter: the
    measurement lives in ``metrics.py``.

    Everything the contract names is present, under these keys:

    ======================== ==================================================
    Contract wording         Key(s)
    ======================== ==================================================
    path length              ``average_shortest_path_length``, ``diameter``,
                             ``radius`` (plus ``*_largest_component``)
    clustering               ``average_clustering``, ``transitivity``
    modularity               ``modularity_role_partition`` -- the PREDEFINED
                             seven-network partition at resolution 1.0, not
                             community detection
    max degree               ``degree_max`` (with min/mean/variance/std and
                             ``degree_heterogeneity``)
    input-readout distance   ``input_readout_mean_distance``, ``_min_``,
                             ``_max_``, plus the unreachable-pair count
    hub roles                ``hub_role``, ``hub_degree``
    ======================== ==================================================

    It also returns edge density, component structure, global efficiency,
    degree assortativity and role-balance fields -- the full Week 4 structural
    row. See :func:`neuromesh.graphs.metrics.structural_metrics` for every
    definition and the undefined-value policy.

    Parameters
    ----------
    graph:
        Any undirected graph. ``topology``, ``n`` and ``seed`` are read from
        ``graph.graph`` when present, so a graph from :func:`generate_graph`
        identifies itself; they are ``None`` otherwise. Roles are needed for
        the modularity, input-readout and hub-role fields, which are ``None``
        on a graph that has none.

    Returns
    -------
    dict
        JSON-friendly scalars. **Undefined values are ``None``, never 0.0** --
        a disconnected graph has no diameter and a regular graph has no
        defined degree assortativity, and reporting 0.0 for either would be a
        false measurement.

    Notes
    -----
    Safe to call on an attacked graph: unlike :func:`validate_graph`, nothing
    here requires connectivity or mean degree 4. The whole-graph path metrics
    simply become ``None`` and the ``_largest_component`` fields carry the
    readable values.
    """
    return _structural_metrics(graph)


__all__ = [
    # shared interface (contracts.py)
    "assign_roles",
    "generate_graph",
    "remove_nodes",
    "structural_metrics",
    # spec / constants
    "BLOCK_TOPOLOGIES",
    "BOUNDARY_N",
    "DEFAULT_FIXTURE_SEED",
    "DEFAULT_WITHIN_BETWEEN_RATIO",
    "DEGREE_TOLERANCE",
    "EXPERIMENT_SEEDS",
    "GENERATOR_VERSION",
    "GENERATOR_VERSIONS",
    "POPULATION_SIZES",
    "PLANNED_TOPOLOGIES",
    "TARGET_MEAN_DEGREE",
    "TOPOLOGIES",
    "GraphParameterError",
    "GraphSpec",
    "generator_version",
    # generators
    "GraphMetadata",
    "barabasi_albert",
    "erdos_renyi",
    "generate_from_spec",
    "mean_degree",
    "modular",
    "target_edge_count",
    "watts_strogatz",
    "yeo_sbm",
    # block models (modular, yeo_sbm)
    "N_BLOCKS",
    "allocate_block_edges",
    "balanced_block_sizes",
    "block_assignment",
    "community_profile",
    "pair_capacity",
    "sample_block_graph",
    "validate_block_weights",
    # roles
    "role_counts",
    "roles_are_balanced",
    # node removal (Week 3)
    "DEFAULT_ROUNDING_RULE",
    "DEFAULT_TIE_BREAK",
    "REMOVAL_SEED_SALT",
    "REMOVAL_STRATEGIES",
    "ROUNDING_RULES",
    "TIE_BREAKS",
    "RemovalResult",
    "canonical_nodes",
    "random_removal_order",
    "remove_node_set",
    "remove_nodes_by_count",
    "remove_nodes_by_fraction",
    "removal_count",
    "static_removal_order",
    "targeted_removal_order",
    "validate_attacked_graph",
    # structural metrics and positional covariates (Week 4)
    "POSITIONAL_FIELDS",
    "STRUCTURAL_FIELDS",
    "degree_assortativity",
    "degree_heterogeneity",
    "degree_statistics",
    "graph_id",
    "input_readout_distance",
    "partition_modularity",
    "positional_covariates",
    "role_balance",
    "role_partition",
    # graph-only robustness curves (Week 4)
    "AUC_FIELDS",
    "CHECKPOINT_FRACTIONS",
    "MAX_REMOVAL_FRACTION",
    "ROBUSTNESS_FIELDS",
    "checkpoint_rows",
    "robustness_auc",
    "robustness_trajectory",
    "trajectory_rows",
    # Week 4 analysis datasets
    "AGGREGATE_METRICS",
    "CONDITION_FIELDS",
    "DEFAULT_OUTPUT_DIR",
    "aggregate_conditions",
    "build_positional_table",
    "build_robustness_tables",
    "build_structural_table",
    "expected_node_row_count",
    "run_week4_analysis",
    # shortest-path / round-budget analysis (Week 3)
    "DEFAULT_ROUND_BUDGETS",
    "STD_CONVENTION",
    "aggregate_path_metrics",
    "diameter_table",
    "measure_graph",
    "path_metrics",
    "render_aggregate_table",
    "render_measurement_table",
    "round_budget_assessment",
    # Yeo connectivity matrix
    "DEFAULT_YEO_MATRIX_PATH",
    "SCHAEFER_TO_YEO7",
    "YeoMatrixUnavailableError",
    "derive_yeo7_matrix",
    "load_yeo_matrix",
    "parse_schaefer_label",
    "validate_yeo7_record",
    # validation
    "Check",
    "ValidationResult",
    "check_reproducible",
    "degree_deviation",
    "validate_graph",
    # fixtures
    "DEFAULT_FIXTURE_DIR",
    "SCHEMA_VERSION",
    "fixture_filename",
    "generate_fixture_set",
    "graph_to_record",
    "load_fixture",
    "load_fixture_set",
    "load_graph",
    "record_to_graph",
    "save_fixture",
]
