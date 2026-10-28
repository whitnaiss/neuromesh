"""Track A (Mihir): graph generation, roles, node removal, structural metrics.

Public entry points only. Implement them in this package (e.g. generators.py,
yeo_sbm.py, roles.py, removal.py, metrics.py) and keep these signatures.

Week 1 status
-------------
``generate_graph`` is implemented for the ``er``, ``ws`` and ``ba``
topologies. ``modular`` and ``yeo_sbm`` are Week 2 work and raise
``NotImplementedError``; ``assign_roles``, ``remove_nodes`` and
``structural_metrics`` are still placeholders owned by later weeks.

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
    target_edge_count,
    watts_strogatz,
)
from .generators import (
    generate_graph as generate_from_spec,
)
from .spec import (
    BOUNDARY_N,
    DEFAULT_FIXTURE_SEED,
    DEGREE_TOLERANCE,
    EXPERIMENT_SEEDS,
    GENERATOR_VERSION,
    POPULATION_SIZES,
    TARGET_MEAN_DEGREE,
    TOPOLOGIES,
    GraphParameterError,
    GraphSpec,
)
from .validation import (
    Check,
    ValidationResult,
    check_reproducible,
    degree_deviation,
    validate_graph,
)

#: Topologies named in ``contracts.Topology`` that Track A has not built yet.
PLANNED_TOPOLOGIES: tuple[str, ...] = ("modular", "yeo_sbm")


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
        ``"er"``, ``"ws"`` or ``"ba"``. ``"modular"`` and ``"yeo_sbm"`` are
        declared in ``contracts.Topology`` but are Week 2 work and raise
        ``NotImplementedError``.
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
    NotImplementedError
        For ``"modular"`` and ``"yeo_sbm"`` (Week 2).
    GraphParameterError
        For an unknown topology or structurally invalid ``n``, ``seed`` or
        ``mean_degree``. Subclasses ``ValueError``.

    Notes
    -----
    ER draws from the *connected* ER ensemble by rejection sampling, because
    mean degree 4 is below the ``ln N`` connectivity threshold at N = 112. That
    conditions the ensemble; the number of draws is recorded in
    ``G.graph["generator_metadata"]["attempts"]``. See ``docs/track_a.md``.
    """
    if topology in PLANNED_TOPOLOGIES:
        raise NotImplementedError(
            f"Track A: topology {topology!r} is Week 2 work (generate_graph "
            f"implements {TOPOLOGIES})"
        )
    if topology not in TOPOLOGIES:
        raise GraphParameterError(
            f"unknown topology {topology!r}; implemented {TOPOLOGIES}, "
            f"planned {PLANNED_TOPOLOGIES}"
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


def assign_roles(graph: nx.Graph, seed: int) -> nx.Graph:
    """Give every node a YEO7 role, exactly n/7 per role; SBM keeps its blocks."""
    raise NotImplementedError("Track A: assign_roles")


def remove_nodes(graph: nx.Graph, spec: RemovalSpec, seed: int) -> nx.Graph:
    """Random or highest-degree-first removal; never removes every readout node."""
    raise NotImplementedError("Track A: remove_nodes")


def structural_metrics(graph: nx.Graph) -> dict:
    """Path length, clustering, modularity, max degree, input-readout distance, hub roles."""
    raise NotImplementedError("Track A: structural_metrics")


__all__ = [
    # shared interface (contracts.py)
    "assign_roles",
    "generate_graph",
    "remove_nodes",
    "structural_metrics",
    # spec / constants
    "BOUNDARY_N",
    "DEFAULT_FIXTURE_SEED",
    "DEGREE_TOLERANCE",
    "EXPERIMENT_SEEDS",
    "GENERATOR_VERSION",
    "POPULATION_SIZES",
    "PLANNED_TOPOLOGIES",
    "TARGET_MEAN_DEGREE",
    "TOPOLOGIES",
    "GraphParameterError",
    "GraphSpec",
    # generators
    "GraphMetadata",
    "barabasi_albert",
    "erdos_renyi",
    "generate_from_spec",
    "mean_degree",
    "target_edge_count",
    "watts_strogatz",
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
