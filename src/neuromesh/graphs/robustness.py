"""Node removal: random failure and adaptive targeted attack (Week 3).

Scope and intent
----------------
This module *damages* graphs on purpose. It is the structural half of the
robustness experiment: take a graph the generators produced -- connected, mean
degree 4, balanced roles -- delete ``k`` of its agents, and hand back exactly
what survives. Nothing is repaired, rebalanced or renumbered.

That has to be said explicitly, because every other module in this package
enforces the opposite invariants. ``validate_graph`` treats a disconnected
graph as a ``FAIL``, the generators reject a mean degree outside +/-10%, and
``roles.assign_roles`` insists on exactly N/7 agents per functional network.
**None of those contracts apply to an attacked graph**, and applying them
would be a category error: a targeted attack that does *not* fragment the
network has failed to measure anything. :func:`validate_attacked_graph` is the
appropriate checker here -- it asserts the properties an induced subgraph must
have, and reports connectivity instead of requiring it.

Two strategies
--------------
``"random"`` -- **random failure.**
    ``k`` distinct nodes drawn uniformly without replacement from an isolated
    seeded generator. Models indiscriminate loss (an agent crashes, a request
    times out). See :func:`random_removal_order`.

``"targeted"`` -- **adaptive highest-degree attack.**
    Remove the current highest-degree node, *recompute degrees on what is
    left*, repeat. Models an adversary who can see the network and strikes its
    busiest agent. See :func:`targeted_removal_order`.

The word *adaptive* is the whole point and is not decoration. Sorting the
original degree sequence once and deleting its top ``k`` is a different,
weaker attack: it ignores that removing a hub demotes the hubs that depended
on it. :func:`static_removal_order` implements that weaker version -- not for
use in the experiment, but so the tests can prove the two differ and the
report can show by how much.

What a result carries
---------------------
:class:`RemovalResult` is a frozen record of one attack: the surviving graph,
the removal order, the degree each node had *at the moment it was removed*,
and the realised post-attack structure (edges, mean degree, components). It is
deliberately more than ``nx.Graph``, because the removal *order* and the
per-step degrees are the data a robustness curve is built from, and recovering
them afterwards from the surviving graph alone is impossible.

Determinism
-----------
Identical ``(graph, k, strategy, seed, tie_break)`` always yields an identical
removal order, surviving graph and metadata. Three things secure that:

1. **Canonical node ordering.** Every candidate list is sorted before any
   sampling or tie-break (:func:`canonical_nodes`), so nothing depends on
   dict or set iteration order, on insertion history, or on PYTHONHASHSEED.
2. **An isolated RNG.** Randomness comes from
   ``np.random.default_rng(SeedSequence(entropy=[seed, REMOVAL_SEED_SALT]))``
   -- never the global ``random`` or ``numpy.random`` state. The salt mirrors
   ``roles.ROLE_SEED_SALT``: without it, ``remove_nodes(G, ..., seed=s)`` would
   consume the same bits that generated ``G`` at seed ``s``.
3. **A deterministic default tie-break.** Ties for highest degree are
   extremely common in these graphs (mean degree 4 at N = 7 leaves almost
   nothing to distinguish nodes), so the policy is fixed and documented rather
   than left to whichever node a heap happened to surface.

Nesting, which matters for Week 6
---------------------------------
Random order is drawn as a full permutation of the eligible nodes, truncated
to ``k``. So the 10% removal set is a subset of the 20% set, which is a subset
of the 30% set, at a fixed seed. The robustness curve is then a curve over one
nested sequence of failures rather than three unrelated draws, which removes a
source of variance the study never intended to measure. Adaptive targeted
order is nested for the same reason: the first ``k`` steps of a longer attack
are the shorter attack.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import numpy as np

from neuromesh.contracts import READOUT_ROLE, YEO7

from .generators import mean_degree
from .spec import GraphParameterError

__all__ = [
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
]

#: Mixed into the seed so removal draws from a different random stream than
#: graph generation and role assignment. Compare ``roles.ROLE_SEED_SALT``.
REMOVAL_SEED_SALT: int = 0x52454D56  # "REMV"

#: Strategy names. These are exactly ``contracts.RemovalMode``'s two values, so
#: a ``RemovalSpec`` can be passed straight through without translation.
REMOVAL_STRATEGIES: tuple[str, ...] = ("random", "targeted")

#: Tie-break policies for :func:`targeted_removal_order`.
TIE_BREAKS: tuple[str, ...] = ("smallest_id", "largest_id", "random")

#: Fraction-to-count conversion rules for :func:`removal_count`.
ROUNDING_RULES: tuple[str, ...] = ("floor", "nearest")

#: Default tie-break: lowest canonical node ID among the current maxima.
DEFAULT_TIE_BREAK = "smallest_id"

#: Default fraction rounding. ``floor`` guarantees ``k <= fraction * N``, hence
#: ``k < N`` for every ``fraction < 1`` that ``RemovalSpec`` permits, so a
#: fractional request can never silently delete the entire network.
DEFAULT_ROUNDING_RULE = "floor"


# --------------------------------------------------------------------------
# validation helpers
# --------------------------------------------------------------------------

def _require_graph(graph: Any) -> nx.Graph:
    if not isinstance(graph, nx.Graph):
        raise GraphParameterError(
            f"expected a networkx.Graph, got {type(graph).__name__}"
        )
    if graph.is_directed():
        raise GraphParameterError(
            "node removal is defined for undirected graphs; got a directed graph"
        )
    return graph


def _require_int(value: Any, name: str) -> int:
    """Strictly an ``int`` (or NumPy integer). ``bool`` and ``float`` are not.

    A ``TypeError`` for a non-integer type and a ``ValueError`` for an
    out-of-range integer is the standard Python split, and this module keeps
    it. ``3.0`` raises rather than being accepted as ``3``: silently rounding
    a float here is how ``k=2.5`` turns into an attack nobody requested.
    """
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(
            f"{name} must be an int, got {type(value).__name__} ({value!r}); "
            "a float is rejected even when integral, so the removal count is "
            "never inferred by rounding"
        )
    return int(value)


def _validate_seed(seed: Any, *, required: bool, name: str = "seed") -> int | None:
    if seed is None:
        if required:
            raise GraphParameterError(f"{name} is required for this operation")
        return None
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise GraphParameterError(
            f"{name} must be an integer or None, got {type(seed).__name__}"
        )
    if seed < 0:
        raise GraphParameterError(f"{name} must be non-negative, got {seed}")
    return int(seed)


def _rng(seed: int) -> np.random.Generator:
    """Isolated generator, salted away from generation and role assignment."""
    sequence = np.random.SeedSequence(entropy=[seed, REMOVAL_SEED_SALT])
    return np.random.default_rng(sequence)


def canonical_nodes(nodes: Iterable) -> list:
    """Node IDs in a stable, reproducible order.

    The project's convention is integer IDs ``0 .. n-1``, which sort directly.
    A graph with non-consecutive or mixed-type IDs still gets a total order
    here (by type name, then ``repr``) instead of silently inheriting set
    iteration order, because an unordered candidate list would make seeded
    sampling non-reproducible across runs.
    """
    node_list = list(nodes)
    try:
        return sorted(node_list)
    except TypeError:
        return sorted(node_list, key=lambda v: (type(v).__name__, repr(v)))


def removal_count(n: int, fraction: float, rule: str = DEFAULT_ROUNDING_RULE) -> int:
    """Convert a removal *fraction* into a node count, by one fixed rule.

    Parameters
    ----------
    n:
        Number of nodes available for removal.
    fraction:
        In ``[0, 1]``. ``contracts.RemovalSpec`` narrows this further to the
        open interval ``(0, 1)``.
    rule:
        ``"floor"`` (default) or ``"nearest"``. The *same* rule is applied at
        every population size -- the study must never remove 29.5% at one
        scale and 30.4% at another because of a size-dependent rounding
        choice.

    Returns
    -------
    int
        ``floor(fraction * n)`` or ``round(fraction * n)``.

    Notes
    -----
    ``floor`` is the default for two reasons. It never removes more than the
    requested share, and it guarantees ``k < n`` for every ``fraction < 1``,
    so a fractional request cannot wipe out the network through rounding
    (``nearest`` at ``n=7, fraction=0.95`` would give ``k=7``).

    The study's three levels at N = 112 therefore remove 11, 22 and 33 nodes,
    i.e. 9.82%, 19.64% and 29.46% rather than exactly 10/20/30%. The realised
    fraction is recorded in :attr:`RemovalResult.realised_fraction` and must be
    quoted in the write-up rather than the nominal one.

    ``floor`` also rounds small-N requests down to zero: 10% of N = 7 is
    ``floor(0.7) = 0``, a no-op. The robustness experiment is specified at
    N = 112 only, so this does not affect it, but a caller sweeping fractions
    at small N must check :attr:`RemovalResult.n_removed` rather than assume.
    """
    n = _require_int(n, "n")
    if n < 0:
        raise GraphParameterError(f"n must be non-negative, got {n}")
    if isinstance(fraction, bool) or not isinstance(
        fraction, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"fraction must be a number, got {type(fraction).__name__}")
    fraction = float(fraction)
    if not 0.0 <= fraction <= 1.0:
        raise GraphParameterError(f"fraction must lie in [0, 1], got {fraction}")
    if rule not in ROUNDING_RULES:
        raise GraphParameterError(
            f"unknown rounding rule {rule!r}; expected one of {ROUNDING_RULES}"
        )

    exact = fraction * n
    count = int(np.floor(exact)) if rule == "floor" else int(np.rint(exact))
    return min(count, n)


def _eligible_nodes(graph: nx.Graph, protected: Collection | None) -> list:
    """Canonically ordered nodes that may be removed."""
    nodes = canonical_nodes(graph.nodes())
    if not protected:
        return nodes
    protected_set = set(protected)
    unknown = protected_set - set(nodes)
    if unknown:
        raise GraphParameterError(
            f"protected nodes absent from the graph: {canonical_nodes(unknown)[:5]}"
        )
    return [node for node in nodes if node not in protected_set]


def _validate_count_against(k: Any, eligible: Sequence, n_nodes: int) -> int:
    k = _require_int(k, "k")
    if k < 0:
        raise GraphParameterError(f"k must be non-negative, got {k}")
    if k > len(eligible):
        if len(eligible) == n_nodes:
            raise GraphParameterError(
                f"cannot remove k={k} nodes from a graph with {n_nodes}"
            )
        raise GraphParameterError(
            f"cannot remove k={k} nodes: only {len(eligible)} of {n_nodes} are "
            "eligible (the rest are protected)"
        )
    return k


# --------------------------------------------------------------------------
# removal orders
# --------------------------------------------------------------------------

def random_removal_order(
    graph: nx.Graph,
    k: int,
    seed: int,
    *,
    protected: Collection | None = None,
) -> tuple:
    """``k`` distinct nodes drawn uniformly without replacement, in draw order.

    The draw is a full permutation of the eligible nodes truncated to ``k``, so
    orders *nest*: at a fixed seed, the k=11 removal set is a prefix of the
    k=22 one. See the module docstring.

    Raises
    ------
    TypeError
        ``k`` is not an ``int``.
    GraphParameterError
        ``k`` is negative or exceeds the eligible node count; ``seed`` is
        absent, negative or not an integer.
    """
    graph = _require_graph(graph)
    seed_value = _validate_seed(seed, required=True)
    eligible = _eligible_nodes(graph, protected)
    k = _validate_count_against(k, eligible, graph.number_of_nodes())
    if k == 0:
        return ()

    order = _rng(seed_value).permutation(len(eligible))
    return tuple(eligible[int(i)] for i in order[:k])


def targeted_removal_order(
    graph: nx.Graph,
    k: int,
    *,
    seed: int | None = None,
    tie_break: str = DEFAULT_TIE_BREAK,
    protected: Collection | None = None,
) -> tuple:
    """Adaptive highest-degree-first order: recompute degrees after every step.

    At each of the ``k`` steps the node of maximum degree *in the graph that
    currently survives* is removed. Degrees are therefore recomputed ``k``
    times, which is what distinguishes this from :func:`static_removal_order`.

    Parameters
    ----------
    tie_break:
        How to choose among several nodes sharing the current maximum degree.
        ``"smallest_id"`` (default) and ``"largest_id"`` are deterministic and
        need no seed; ``"random"`` draws from the seeded RNG and *requires*
        ``seed``. Ties are the normal case, not the exception: at mean degree
        4 most nodes in these graphs share a degree with several others, so
        this choice visibly changes the attack sequence and is never left
        implicit.
    protected:
        Nodes that may not be removed. They keep contributing degree to their
        neighbours -- they are protected from deletion, not hidden from the
        graph.

    Complexity
    ----------
    ``O(k * N + E)``. Removal happens on a working copy, so each deletion costs
    ``O(deg(v))`` and each maximum scan ``O(N)``. The working copy never leaks
    into the returned subgraph: the result is built from the original graph.
    """
    graph = _require_graph(graph)
    if tie_break not in TIE_BREAKS:
        raise GraphParameterError(
            f"unknown tie_break {tie_break!r}; expected one of {TIE_BREAKS}"
        )
    seed_value = _validate_seed(seed, required=tie_break == "random")
    eligible = _eligible_nodes(graph, protected)
    k = _validate_count_against(k, eligible, graph.number_of_nodes())
    if k == 0:
        return ()

    rng = _rng(seed_value) if tie_break == "random" else None
    working = graph.copy()
    # Canonical order fixed once; filtered each step. Both the maximum scan and
    # the tie-break therefore see a list that cannot depend on iteration order.
    remaining = [node for node in eligible]
    removed: list = []

    for _ in range(k):
        degrees = [working.degree(node) for node in remaining]
        top = max(degrees)
        tied = [
            node
            for node, degree in zip(remaining, degrees, strict=True)
            if degree == top
        ]

        if tie_break == "smallest_id":
            chosen = tied[0]
        elif tie_break == "largest_id":
            chosen = tied[-1]
        else:
            chosen = tied[int(rng.integers(len(tied)))]

        removed.append(chosen)
        remaining.remove(chosen)
        working.remove_node(chosen)

    return tuple(removed)


def static_removal_order(
    graph: nx.Graph,
    k: int,
    *,
    tie_break: str = DEFAULT_TIE_BREAK,
    protected: Collection | None = None,
) -> tuple:
    """Non-adaptive baseline: the top ``k`` of the *original* degree sequence.

    Provided for comparison and testing only. The experiment uses
    :func:`targeted_removal_order`; this function exists so the difference
    between "sort once" and "recompute every step" is demonstrable rather than
    asserted, and so a reader of the report can see a worked example where the
    two disagree.
    """
    graph = _require_graph(graph)
    if tie_break not in ("smallest_id", "largest_id"):
        raise GraphParameterError(
            f"static removal supports deterministic tie-breaks only, got {tie_break!r}"
        )
    eligible = _eligible_nodes(graph, protected)
    k = _validate_count_against(k, eligible, graph.number_of_nodes())
    if k == 0:
        return ()

    reverse_id = tie_break == "largest_id"
    ranked = sorted(
        eligible,
        key=lambda node: (-graph.degree(node), _sort_key(node, reverse_id)),
    )
    return tuple(ranked[:k])


def _sort_key(node: Any, reverse: bool) -> Any:
    """Secondary sort key honouring the requested ID direction."""
    if isinstance(node, (int, np.integer)) and not isinstance(node, bool):
        return (0, -int(node)) if reverse else (0, int(node))
    text = f"{type(node).__name__}:{node!r}"
    chars = tuple(-ord(c) for c in text) if reverse else tuple(ord(c) for c in text)
    return (1, chars)


# --------------------------------------------------------------------------
# applying a removal
# --------------------------------------------------------------------------

def remove_node_set(graph: nx.Graph, nodes: Iterable) -> nx.Graph:
    """The induced subgraph on the survivors, as an independent copy.

    Everything the study needs preserved is preserved, because this is exactly
    what "induced subgraph" means:

    * **Node IDs are untouched.** No renumbering, no compaction. Node 57 is
      still node 57 in a graph that no longer has a node 56. A log joined on
      node ID across removal levels therefore joins correctly.
    * **Edges are exactly the original edges with both endpoints surviving.**
      None are added, none are rewired.
    * **Node and edge attributes survive**, so ``role`` and ``community``
      stay attached to the agents that had them.
    * **The input graph is not modified.** ``nx.Graph.subgraph`` returns a
      *view* over the original; ``.copy()`` detaches it, so later mutation of
      the result cannot reach back into the caller's graph.
    """
    graph = _require_graph(graph)
    doomed = set(nodes)
    survivors = [node for node in canonical_nodes(graph.nodes()) if node not in doomed]
    return graph.subgraph(survivors).copy()


@dataclass(frozen=True)
class RemovalResult:
    """One attack, recorded in full.

    Attributes
    ----------
    graph:
        The surviving induced subgraph, an independent copy.
    removed_nodes:
        Removal order, first removed first. For ``"targeted"`` this is the
        adaptive attack sequence; for ``"random"`` the draw order.
    removed_degrees:
        ``removed_degrees[i]`` is the degree node ``removed_nodes[i]`` had in
        the graph that survived steps ``0 .. i-1`` -- its degree *at the moment
        it was removed*. For an adaptive attack this is strictly the quantity
        the strategy maximised, and it cannot be recovered from the surviving
        graph afterwards. Non-increasing for ``"targeted"``.
    original_degrees:
        Degree of each removed node in the *original* graph, same order. The
        gap between this and ``removed_degrees`` is the adaptivity, measured.
    realised_fraction:
        ``n_removed / n_original``. Quote this, not the requested fraction:
        they differ whenever the fraction did not land on a whole node.
    connected / n_components:
        Measured on the surviving graph. A disconnected result is an expected
        outcome of an attack, never an error.
    role_counts / readout_surviving:
        Surviving role histogram and the number of surviving
        ``READOUT_ROLE`` agents, or ``None`` when the graph carries no roles.
        Recorded, not enforced: see :attr:`roles_intact`.
    """

    graph: nx.Graph
    removed_nodes: tuple = ()
    surviving_nodes: tuple = ()
    strategy: str = "random"
    seed: int | None = None
    tie_break: str = DEFAULT_TIE_BREAK
    requested_k: int = 0
    requested_fraction: float | None = None
    rounding_rule: str | None = None
    n_original: int = 0
    n_surviving: int = 0
    n_removed: int = 0
    n_edges_original: int = 0
    n_edges_surviving: int = 0
    mean_degree_original: float | None = None
    mean_degree_surviving: float | None = None
    realised_fraction: float = 0.0
    connected: bool = False
    n_components: int = 0
    largest_component_size: int = 0
    removed_degrees: tuple = ()
    original_degrees: tuple = ()
    protected_nodes: tuple = ()
    role_counts: dict | None = None
    readout_surviving: int | None = None
    provenance: dict = field(default_factory=dict)

    @property
    def roles_intact(self) -> bool | None:
        """True when every YEO7 role still has at least one surviving agent.

        ``None`` when the graph carries no roles. This is *reported*, not
        guaranteed: a targeted attack that happens to empty a functional
        network has told us something, and silently protecting that network
        would have thrown the finding away. See ``docs/track_a.md`` §16.6.
        """
        if self.role_counts is None:
            return None
        return all(self.role_counts.get(role, 0) > 0 for role in YEO7)

    def to_dict(self) -> dict:
        """JSON-serialisable summary. Excludes ``graph`` by design."""
        return {
            "strategy": self.strategy,
            "seed": self.seed,
            "tie_break": self.tie_break,
            "requested_k": self.requested_k,
            "requested_fraction": self.requested_fraction,
            "rounding_rule": self.rounding_rule,
            "n_original": self.n_original,
            "n_surviving": self.n_surviving,
            "n_removed": self.n_removed,
            "n_edges_original": self.n_edges_original,
            "n_edges_surviving": self.n_edges_surviving,
            "mean_degree_original": self.mean_degree_original,
            "mean_degree_surviving": self.mean_degree_surviving,
            "realised_fraction": self.realised_fraction,
            "connected": self.connected,
            "n_components": self.n_components,
            "largest_component_size": self.largest_component_size,
            "removed_nodes": list(self.removed_nodes),
            "removed_degrees": list(self.removed_degrees),
            "original_degrees": list(self.original_degrees),
            "protected_nodes": list(self.protected_nodes),
            "role_counts": self.role_counts,
            "readout_surviving": self.readout_surviving,
            "roles_intact": self.roles_intact,
            "provenance": dict(self.provenance),
        }

    def summary(self) -> str:
        roles = "" if self.role_counts is None else f" readout={self.readout_surviving}"
        return (
            f"{self.strategy}: removed {self.n_removed}/{self.n_original} "
            f"({self.realised_fraction:.2%}) -> {self.n_surviving} nodes, "
            f"{self.n_edges_surviving} edges, k={self.mean_degree_surviving}, "
            f"components={self.n_components}{roles}"
        )


def _role_summary(graph: nx.Graph) -> tuple[dict | None, int | None]:
    roles = nx.get_node_attributes(graph, "role")
    if not roles:
        return None, None
    counts = dict.fromkeys(YEO7, 0)
    for role in roles.values():
        counts[role] = counts.get(role, 0) + 1
    return counts, counts.get(READOUT_ROLE, 0)


def remove_nodes_by_count(
    graph: nx.Graph,
    k: int,
    strategy: str = "random",
    *,
    seed: int | None = None,
    tie_break: str = DEFAULT_TIE_BREAK,
    protected: Collection | None = None,
    _requested_fraction: float | None = None,
    _rounding_rule: str | None = None,
) -> RemovalResult:
    """Remove ``k`` nodes by ``strategy``; return the full :class:`RemovalResult`.

    This is the primary Track A removal entry point. Integer ``k`` is the
    primary semantics; :func:`remove_nodes_by_fraction` layers a documented
    fraction conversion on top.

    Parameters
    ----------
    graph:
        Undirected graph. **Not modified.**
    k:
        Number of nodes to remove, ``0 <= k <= N``. Strictly an ``int``.
    strategy:
        ``"random"`` or ``"targeted"`` -- the two values of
        ``contracts.RemovalMode``.
    seed:
        Required for ``"random"`` and for ``tie_break="random"``; ignored by
        the deterministic tie-breaks.
    tie_break:
        Only meaningful for ``"targeted"``. See :func:`targeted_removal_order`.
    protected:
        Optional nodes that may not be removed. Empty by default: the attack
        is unconstrained unless a caller explicitly asks otherwise.

    Returns
    -------
    RemovalResult
        Frozen record. ``result.graph`` is the surviving induced subgraph.

    Raises
    ------
    TypeError
        ``k`` is not an ``int`` (including ``3.0`` and ``True``).
    GraphParameterError
        A ``ValueError`` subclass: unknown strategy or tie-break, ``k``
        negative or above the eligible count, bad seed, directed or non-graph
        input, protected nodes absent from the graph.

    Examples
    --------
    >>> from neuromesh.graphs import generate_graph, remove_nodes_by_count
    >>> graph = generate_graph("ba", 28, seed=0)
    >>> result = remove_nodes_by_count(graph, 5, "targeted")
    >>> result.n_surviving, graph.number_of_nodes()
    (23, 28)
    >>> list(result.removed_degrees) == sorted(result.removed_degrees, reverse=True)
    True
    """
    graph = _require_graph(graph)
    if strategy not in REMOVAL_STRATEGIES:
        raise GraphParameterError(
            f"unknown removal strategy {strategy!r}; "
            f"expected one of {REMOVAL_STRATEGIES}"
        )

    n_original = graph.number_of_nodes()
    e_original = graph.number_of_edges()

    if strategy == "random":
        order = random_removal_order(graph, k, seed, protected=protected)
        seed_value = _validate_seed(seed, required=True)
        effective_tie_break = "n/a"
    else:
        order = targeted_removal_order(
            graph, k, seed=seed, tie_break=tie_break, protected=protected
        )
        seed_value = _validate_seed(seed, required=tie_break == "random")
        effective_tie_break = tie_break

    # Degree at the moment of removal: replay the sequence on a working copy.
    # Recomputed here rather than returned by the order functions so that both
    # strategies share one measurement path and cannot disagree.
    working = graph.copy()
    removed_degrees: list[int] = []
    for node in order:
        removed_degrees.append(int(working.degree(node)))
        working.remove_node(node)

    survivor_graph = remove_node_set(graph, order)
    has_nodes = survivor_graph.number_of_nodes() > 0
    components = nx.number_connected_components(survivor_graph) if has_nodes else 0
    component_sizes = (
        sorted((len(c) for c in nx.connected_components(survivor_graph)), reverse=True)
        if has_nodes
        else []
    )
    role_counts, readout = _role_summary(survivor_graph)
    n_removed = len(order)

    result = RemovalResult(
        graph=survivor_graph,
        removed_nodes=tuple(order),
        surviving_nodes=tuple(canonical_nodes(survivor_graph.nodes())),
        strategy=strategy,
        seed=seed_value,
        tie_break=effective_tie_break,
        requested_k=int(k),
        requested_fraction=_requested_fraction,
        rounding_rule=_rounding_rule,
        n_original=n_original,
        n_surviving=survivor_graph.number_of_nodes(),
        n_removed=n_removed,
        n_edges_original=e_original,
        n_edges_surviving=survivor_graph.number_of_edges(),
        mean_degree_original=mean_degree(graph) if n_original else None,
        mean_degree_surviving=mean_degree(survivor_graph) if has_nodes else None,
        realised_fraction=(n_removed / n_original) if n_original else 0.0,
        connected=components == 1,
        n_components=components,
        largest_component_size=component_sizes[0] if component_sizes else 0,
        removed_degrees=tuple(removed_degrees),
        original_degrees=tuple(int(graph.degree(node)) for node in order),
        protected_nodes=tuple(canonical_nodes(protected)) if protected else (),
        role_counts=role_counts,
        readout_surviving=readout,
        provenance={
            "topology": graph.graph.get("topology"),
            "n": graph.graph.get("n"),
            "graph_seed": graph.graph.get("seed"),
            "removal_seed_salt": REMOVAL_SEED_SALT,
            "networkx_version": nx.__version__,
            "numpy_version": np.__version__,
        },
    )

    # Attack provenance on the surviving graph itself, so a graph that reaches
    # Track B or Track D without its RemovalResult still says what happened.
    # ``topology``, ``n`` and ``seed`` are left untouched: they identify the
    # experimental condition, not the live node count. ``mean_degree`` *is*
    # updated, because docs/track_a.md defines it as the realised 2E/N and a
    # stale 4.0 on an attacked graph would be a false claim.
    survivor_graph.graph["removal"] = result.to_dict()
    if has_nodes:
        survivor_graph.graph["mean_degree"] = mean_degree(survivor_graph)
    return result


def remove_nodes_by_fraction(
    graph: nx.Graph,
    fraction: float,
    strategy: str = "random",
    *,
    seed: int | None = None,
    tie_break: str = DEFAULT_TIE_BREAK,
    protected: Collection | None = None,
    rule: str = DEFAULT_ROUNDING_RULE,
) -> RemovalResult:
    """Remove ``removal_count(N, fraction, rule)`` nodes. See :func:`removal_count`.

    The count is taken over the *whole* node set ``N``, not over the eligible
    subset, so "remove 20%" means 20% of the network regardless of how many
    nodes a caller protected.
    """
    graph = _require_graph(graph)
    k = removal_count(graph.number_of_nodes(), fraction, rule)
    return remove_nodes_by_count(
        graph,
        k,
        strategy,
        seed=seed,
        tie_break=tie_break,
        protected=protected,
        _requested_fraction=float(fraction),
        _rounding_rule=rule,
    )


# --------------------------------------------------------------------------
# validation for attacked graphs
# --------------------------------------------------------------------------

def validate_attacked_graph(original: nx.Graph, result: RemovalResult) -> list:
    """Check the invariants an attacked graph *must* satisfy, independently.

    Deliberately separate from :func:`neuromesh.graphs.validate_graph`, which
    checks generator contracts -- connectivity and mean degree 4 -- that an
    attacked graph is *expected* to violate. Running that validator on an
    attacked graph and reporting the ``FAIL`` would be a methodological error.

    Returns a list of :class:`neuromesh.graphs.validation.Check`: the node
    partition, induced-edge equality, attribute preservation, no duplicate
    removals, and no renumbering. Connectivity is measured and reported as a
    declared ``EXCEPTION`` when absent, never as a ``FAIL``.
    """
    from .validation import EXCEPTION, FAIL, PASS, Check

    checks: list = []
    survivor = result.graph
    removed = set(result.removed_nodes)
    surviving = set(survivor.nodes())
    original_nodes = set(original.nodes())

    checks.append(
        Check("no_duplicate_removals", PASS, f"{len(removed)} distinct")
        if len(removed) == len(result.removed_nodes)
        else Check("no_duplicate_removals", FAIL, "a node was removed twice")
    )
    partition_ok = not (removed & surviving) and (removed | surviving) == original_nodes
    checks.append(
        Check(
            "partition",
            PASS,
            f"{len(surviving)} + {len(removed)} = {len(original_nodes)}",
        )
        if partition_ok
        else Check(
            "partition", FAIL, "removed and surviving do not partition the original"
        )
    )

    expected_edges = {
        tuple(sorted((u, v)))
        for u, v in original.edges()
        if u in surviving and v in surviving
    }
    actual_edges = {tuple(sorted((u, v))) for u, v in survivor.edges()}
    checks.append(
        Check("induced_subgraph", PASS, f"{len(actual_edges)} edges")
        if expected_edges == actual_edges
        else Check(
            "induced_subgraph",
            FAIL,
            f"edge set differs by {len(expected_edges ^ actual_edges)} edge(s)",
        )
    )

    bad_attrs = [
        node
        for node in surviving
        if dict(original.nodes[node]) != dict(survivor.nodes[node])
    ]
    checks.append(
        Check("node_attributes_preserved", PASS, "all surviving nodes match")
        if not bad_attrs
        else Check(
            "node_attributes_preserved",
            FAIL,
            f"changed at {canonical_nodes(bad_attrs)[:5]}",
        )
    )
    bad_edges = [
        (u, v)
        for u, v in survivor.edges()
        if dict(original.edges[u, v]) != dict(survivor.edges[u, v])
    ]
    checks.append(
        Check("edge_attributes_preserved", PASS, "all surviving edges match")
        if not bad_edges
        else Check("edge_attributes_preserved", FAIL, f"changed at {bad_edges[:5]}")
    )
    checks.append(
        Check("no_renumbering", PASS, "surviving IDs unchanged")
        if surviving <= original_nodes
        else Check("no_renumbering", FAIL, "surviving graph invented node IDs")
    )

    detail = (
        f"{result.n_components} component(s), "
        f"largest {result.largest_component_size}"
    )
    checks.append(
        Check("connectivity_reported", PASS, detail)
        if result.connected
        else Check(
            "connectivity_reported", EXCEPTION, detail + " -- expected after an attack"
        )
    )
    return checks
