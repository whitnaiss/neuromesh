"""Tests for node removal (owner: Mihir, Track A).

These tests verify the removal invariants *independently* rather than by
comparing the implementation against itself. Specifically:

* the surviving edge set is recomputed from the original graph by a filter
  written here, not read back from ``RemovalResult``;
* adaptive targeted removal is checked step by step against a degree maximum
  recomputed here on a separate working copy;
* determinism is checked by calling twice and comparing, and non-determinism
  of the *wrong* kind (global RNG leakage) is checked by perturbing
  ``random`` and ``numpy.random`` between calls;
* a hand-built graph pins the exact adaptive sequence, and proves it differs
  from the non-adaptive one.

The attacked graph is deliberately *not* run through ``validate_graph``:
connectivity and mean degree 4 are generator contracts and an attack is meant
to break them. ``validate_attacked_graph`` is the right checker and is
exercised below.
"""

from __future__ import annotations

import dataclasses
import random

import networkx as nx
import numpy as np
import pytest

from neuromesh.contracts import READOUT_ROLE, YEO7, RemovalSpec
from neuromesh.graphs import (
    POPULATION_SIZES,
    REMOVAL_STRATEGIES,
    TIE_BREAKS,
    TOPOLOGIES,
    GraphParameterError,
    RemovalResult,
    assign_roles,
    canonical_nodes,
    generate_graph,
    random_removal_order,
    removal_count,
    remove_node_set,
    remove_nodes,
    remove_nodes_by_count,
    remove_nodes_by_fraction,
    static_removal_order,
    targeted_removal_order,
    validate_attacked_graph,
)

SEEDS = (0, 1, 2, 3, 4)

# Generating an N=112 ER graph costs a rejection-sampling loop, and the matrix
# below reuses the same graphs many times. One cache keeps the module fast
# without weakening any assertion: every test still gets the real graph the
# generators produce, and nothing mutates a cached graph (immutability of the
# input is itself asserted).
_GRAPH_CACHE: dict[tuple, nx.Graph] = {}


def graph_for(topology: str, n: int, seed: int = 0, roles: bool = False) -> nx.Graph:
    """Cached generated graph, optionally with roles assigned."""
    key = (topology, n, seed, roles)
    if key not in _GRAPH_CACHE:
        graph = generate_graph(topology, n, seed=seed)
        if roles:
            assign_roles(graph, seed=seed)
        _GRAPH_CACHE[key] = graph
    return _GRAPH_CACHE[key]


# --------------------------------------------------------------------------
# independent reference implementations used by the assertions below
# --------------------------------------------------------------------------

def induced_edges(graph: nx.Graph, survivors: set) -> set:
    """Edges of ``graph`` with both endpoints in ``survivors``, as sorted pairs.

    Written out by hand rather than delegating to ``nx.subgraph`` so that the
    assertion does not share an implementation with the code under test.
    """
    return {
        tuple(sorted((u, v)))
        for u, v in graph.edges()
        if u in survivors and v in survivors
    }


def replay_adaptive_maxima(graph: nx.Graph, order) -> list[tuple[int, int]]:
    """For each step, ``(degree of the chosen node, maximum degree available)``.

    Recomputed on an independent copy. For a correct adaptive attack the two
    are equal at every step.
    """
    working = graph.copy()
    pairs = []
    for node in order:
        degrees = dict(working.degree())
        pairs.append((degrees[node], max(degrees.values())))
        working.remove_node(node)
    return pairs


# ==========================================================================
# 1. the hand-built graph: adaptive and static orders genuinely differ
# ==========================================================================

def hub_chain_graph() -> nx.Graph:
    """Twelve nodes built so that adapting changes the answer.

    ::

        3  4  5  6          7   8            9  10  11
         \\ |  | /            \\ /              \\  |  /
          \\|  |/              1  (B, deg 3)      2  (C, deg 3)
           0 (A, deg 5) ------/
           |
           (0-1 edge)

    Degrees in the original graph: A=0 has 5, B=1 has 3, C=2 has 3, every
    other node has 1.

    * **Static** (sort the original degrees once, lowest ID breaking ties):
      ``0, 1, 2``.
    * **Adaptive**: removing A costs B one edge, so B drops to degree 2 while
      C is untouched at 3. The second strike therefore lands on C, not B:
      ``0, 2, 1``.

    At k=2 the two strategies remove *different sets* (``{0,1}`` vs
    ``{0,2}``); at k=3 they remove the same set in a different *order*. Both
    are asserted.

    The graph is also disconnected by construction (C's star is a separate
    component), which doubles as a disconnected-input case.
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(12))
    graph.add_edges_from([(0, 1), (0, 3), (0, 4), (0, 5), (0, 6)])
    graph.add_edges_from([(1, 7), (1, 8)])
    graph.add_edges_from([(2, 9), (2, 10), (2, 11)])
    return graph


def test_hand_built_graph_has_the_degrees_the_example_claims():
    graph = hub_chain_graph()
    assert graph.degree(0) == 5
    assert graph.degree(1) == 3
    assert graph.degree(2) == 3
    assert all(graph.degree(v) == 1 for v in range(3, 12))
    assert nx.number_connected_components(graph) == 2


def test_adaptive_order_is_exactly_the_expected_sequence():
    assert targeted_removal_order(hub_chain_graph(), 3) == (0, 2, 1)


def test_static_order_is_exactly_the_expected_sequence():
    assert static_removal_order(hub_chain_graph(), 3) == (0, 1, 2)


def test_adaptive_and_static_remove_different_sets_at_k2():
    graph = hub_chain_graph()
    adaptive = targeted_removal_order(graph, 2)
    static = static_removal_order(graph, 2)
    assert adaptive == (0, 2)
    assert static == (0, 1)
    assert set(adaptive) != set(static)


def test_adaptive_and_static_differ_in_order_at_k3():
    graph = hub_chain_graph()
    adaptive = targeted_removal_order(graph, 3)
    static = static_removal_order(graph, 3)
    assert adaptive != static
    assert set(adaptive) == set(static)


def test_adaptive_degrees_are_the_current_maximum_at_every_step():
    graph = hub_chain_graph()
    order = targeted_removal_order(graph, 4)
    for chosen, available in replay_adaptive_maxima(graph, order):
        assert chosen == available


def test_adaptive_removed_degrees_differ_from_original_degrees():
    """The recorded per-step degrees must be the adaptive ones, not the static.

    B is removed last with degree 2, although its degree in the original graph
    was 3. A result that reported 3 would have measured the wrong quantity.
    """
    result = remove_nodes_by_count(hub_chain_graph(), 3, "targeted")
    assert result.removed_nodes == (0, 2, 1)
    assert result.removed_degrees == (5, 3, 2)
    assert result.original_degrees == (5, 3, 3)
    assert result.removed_degrees != result.original_degrees


# ==========================================================================
# 2. core invariants, over the real generated graphs
# ==========================================================================

@pytest.mark.parametrize("topology", TOPOLOGIES)
@pytest.mark.parametrize("strategy", REMOVAL_STRATEGIES)
def test_partition_and_counts(topology, strategy):
    graph = graph_for(topology, 28)
    result = remove_nodes_by_count(graph, 7, strategy, seed=0)

    removed = set(result.removed_nodes)
    surviving = set(result.graph.nodes())
    original = set(graph.nodes())

    # exact counts
    assert result.n_removed == 7
    assert result.n_original == 28
    assert result.n_surviving == 21
    assert result.graph.number_of_nodes() == 21
    # no duplicate removals
    assert len(removed) == len(result.removed_nodes) == 7
    # disjoint, and their union is the original node set
    assert removed.isdisjoint(surviving)
    assert removed | surviving == original


@pytest.mark.parametrize("topology", TOPOLOGIES)
@pytest.mark.parametrize("strategy", REMOVAL_STRATEGIES)
def test_surviving_edges_are_exactly_the_induced_subgraph(topology, strategy):
    graph = graph_for(topology, 56)
    result = remove_nodes_by_count(graph, 13, strategy, seed=1)

    survivors = set(result.graph.nodes())
    expected = induced_edges(graph, survivors)
    actual = {tuple(sorted((u, v))) for u, v in result.graph.edges()}
    assert actual == expected
    # no edge was invented, none dropped
    assert result.n_edges_surviving == len(expected)


@pytest.mark.parametrize("topology", TOPOLOGIES)
@pytest.mark.parametrize("strategy", REMOVAL_STRATEGIES)
def test_input_graph_is_not_modified(topology, strategy):
    graph = graph_for(topology, 28)
    before_nodes = sorted(graph.nodes())
    before_edges = sorted(tuple(sorted(e)) for e in graph.edges())
    before_attrs = {v: dict(d) for v, d in graph.nodes(data=True)}
    before_graph_attrs = dict(graph.graph)

    result = remove_nodes_by_count(graph, 9, strategy, seed=2)
    # mutate the survivor hard; the original must not notice
    result.graph.add_edge(*sorted(graph.nodes())[:2])
    result.graph.graph["poisoned"] = True

    assert sorted(graph.nodes()) == before_nodes
    assert sorted(tuple(sorted(e)) for e in graph.edges()) == before_edges
    assert {v: dict(d) for v, d in graph.nodes(data=True)} == before_attrs
    assert dict(graph.graph) == before_graph_attrs
    assert "removal" not in graph.graph
    assert "poisoned" not in graph.graph


@pytest.mark.parametrize("topology", TOPOLOGIES)
def test_node_and_edge_attributes_are_preserved(topology):
    graph = graph_for(topology, 28, roles=True).copy()
    for index, (u, v) in enumerate(sorted(tuple(sorted(e)) for e in graph.edges())):
        graph.edges[u, v]["weight"] = 0.5 + index
        graph.edges[u, v]["tag"] = f"e{index}"
    for node in graph.nodes():
        graph.nodes[node]["marker"] = f"n{node}"

    result = remove_nodes_by_count(graph, 8, "targeted")
    for node in result.graph.nodes():
        assert result.graph.nodes[node] == graph.nodes[node]
        assert result.graph.nodes[node]["marker"] == f"n{node}"
    for u, v in result.graph.edges():
        assert result.graph.edges[u, v] == graph.edges[u, v]


@pytest.mark.parametrize("topology", TOPOLOGIES)
def test_roles_stay_with_their_original_nodes(topology):
    graph = graph_for(topology, 56, roles=True)
    result = remove_nodes_by_count(graph, 14, "random", seed=3)
    for node in result.graph.nodes():
        assert result.graph.nodes[node]["role"] == graph.nodes[node]["role"]


def test_no_renumbering_of_surviving_nodes():
    graph = graph_for("er", 28)
    result = remove_nodes_by_count(graph, 10, "random", seed=0)
    assert set(result.graph.nodes()) <= set(graph.nodes())
    # the surviving IDs are a strict subset with gaps, not a 0..k-1 relabelling
    survivors = sorted(result.graph.nodes())
    assert survivors != list(range(len(survivors)))


def test_roles_are_not_rebalanced_after_removal():
    """The surviving role histogram is whatever the attack left, unmodified.

    Checked two ways: against a histogram recomputed here from the surviving
    nodes, and by asserting the result is genuinely *un*balanced for at least
    one of the five topologies -- so the test cannot pass by the attack
    happening to leave a balanced network.
    """
    unbalanced_seen = False
    for topology in TOPOLOGIES:
        graph = graph_for(topology, 28, roles=True)
        result = remove_nodes_by_count(graph, 14, "targeted")

        expected = dict.fromkeys(YEO7, 0)
        for node in result.graph.nodes():
            expected[graph.nodes[node]["role"]] += 1
        assert result.role_counts == expected
        assert sum(result.role_counts.values()) == 14

        if set(result.role_counts.values()) != {2}:
            unbalanced_seen = True

    assert unbalanced_seen, "no topology produced an unbalanced survivor set"


def test_disconnection_is_reported_not_raised():
    """A targeted attack on BA fragments the graph; that is the finding."""
    graph = graph_for("ba", 112)
    assert nx.is_connected(graph)
    result = remove_nodes_by_count(graph, 33, "targeted")
    assert result.n_components > 1
    assert result.connected is False
    assert result.largest_component_size < result.n_surviving
    # and the attacked-graph validator treats it as a declared exception
    statuses = {c.name: c.status for c in validate_attacked_graph(graph, result)}
    assert statuses["connectivity_reported"] == "EXCEPTION"
    assert all(status != "FAIL" for status in statuses.values())


@pytest.mark.parametrize("topology", TOPOLOGIES)
@pytest.mark.parametrize("strategy", REMOVAL_STRATEGIES)
def test_attacked_graph_validator_passes_on_every_topology(topology, strategy):
    graph = graph_for(topology, 56, roles=True)
    result = remove_nodes_by_count(graph, 17, strategy, seed=4)
    checks = validate_attacked_graph(graph, result)
    failures = [c for c in checks if c.status == "FAIL"]
    assert not failures, [c.detail for c in failures]
    assert {c.name for c in checks} >= {
        "partition",
        "induced_subgraph",
        "node_attributes_preserved",
        "edge_attributes_preserved",
        "no_duplicate_removals",
        "no_renumbering",
    }


def test_isolated_nodes_and_nonconsecutive_ids():
    graph = nx.Graph()
    graph.add_nodes_from([5, 9, 11, 40])  # non-consecutive, 40 isolated
    graph.add_edges_from([(5, 9), (9, 11)])
    result = remove_nodes_by_count(graph, 2, "targeted")
    # degrees are 5->1, 9->2, 11->1, 40->0, so node 9 goes first; afterwards
    # every survivor is isolated and the smallest-ID tie-break picks 5.
    assert result.removed_nodes == (9, 5)
    assert set(result.graph.nodes()) == {11, 40}
    assert result.graph.number_of_edges() == 0
    assert result.n_components == 2


def test_graph_with_only_isolated_nodes():
    graph = nx.Graph()
    graph.add_nodes_from([0, 1, 2])
    result = remove_nodes_by_count(graph, 2, "targeted")
    assert result.removed_degrees == (0, 0)
    assert result.n_surviving == 1
    assert result.n_edges_surviving == 0


# ==========================================================================
# 3. determinism
# ==========================================================================

@pytest.mark.parametrize("topology", TOPOLOGIES)
@pytest.mark.parametrize("strategy", REMOVAL_STRATEGIES)
def test_identical_inputs_give_identical_results(topology, strategy):
    graph = graph_for(topology, 56)
    first = remove_nodes_by_count(graph, 17, strategy, seed=2)
    second = remove_nodes_by_count(graph, 17, strategy, seed=2)

    assert first.removed_nodes == second.removed_nodes
    assert first.removed_degrees == second.removed_degrees
    assert sorted(first.graph.nodes()) == sorted(second.graph.nodes())
    assert sorted(tuple(sorted(e)) for e in first.graph.edges()) == sorted(
        tuple(sorted(e)) for e in second.graph.edges()
    )
    assert first.to_dict() == second.to_dict()


@pytest.mark.parametrize("topology", TOPOLOGIES)
@pytest.mark.parametrize("seed", SEEDS)
def test_random_removal_replays_for_every_tested_configuration(topology, seed):
    graph = graph_for(topology, 28)
    first = random_removal_order(graph, 9, seed)
    second = random_removal_order(graph, 9, seed)
    assert first == second
    assert len(set(first)) == 9


def test_removal_does_not_touch_or_read_global_random_state():
    """Perturbing the global RNGs between calls must change nothing."""
    graph = graph_for("ws", 56)
    random.seed(1234)
    np.random.seed(1234)
    first = remove_nodes_by_count(graph, 13, "random", seed=7)
    state_random = random.getstate()
    state_numpy = np.random.get_state()

    random.seed(999)
    np.random.seed(999)
    [random.random() for _ in range(50)]
    np.random.random(50)
    second = remove_nodes_by_count(graph, 13, "random", seed=7)

    assert first.removed_nodes == second.removed_nodes
    # and the call itself consumed nothing from the global streams
    random.setstate(state_random)
    np.random.set_state(state_numpy)
    before = random.random(), float(np.random.random())
    random.setstate(state_random)
    np.random.set_state(state_numpy)
    remove_nodes_by_count(graph, 13, "random", seed=7)
    after = random.random(), float(np.random.random())
    assert before == after


@pytest.mark.parametrize("topology", TOPOLOGIES)
def test_different_seeds_give_different_random_removals(topology):
    graph = graph_for(topology, 112)
    orders = {random_removal_order(graph, 33, seed) for seed in SEEDS}
    assert len(orders) == len(SEEDS)


def test_random_order_nests_across_removal_levels():
    """10% must be a prefix of 20%, which must be a prefix of 30%."""
    graph = graph_for("er", 112)
    k10, k20, k30 = (removal_count(112, f) for f in (0.1, 0.2, 0.3))
    assert (k10, k20, k30) == (11, 22, 33)
    o10 = random_removal_order(graph, k10, 0)
    o20 = random_removal_order(graph, k20, 0)
    o30 = random_removal_order(graph, k30, 0)
    assert o20[:k10] == o10
    assert o30[:k20] == o20


def test_targeted_order_nests_across_removal_levels():
    graph = graph_for("ba", 112)
    o11 = targeted_removal_order(graph, 11)
    o33 = targeted_removal_order(graph, 33)
    assert o33[:11] == o11


# ==========================================================================
# 4. tie-breaking
# ==========================================================================

def regular_ring(n: int, k: int = 4) -> nx.Graph:
    """A k-regular ring lattice: every node ties with every other on degree."""
    graph = nx.circulant_graph(n, list(range(1, k // 2 + 1)))
    assert len({d for _, d in graph.degree()}) == 1
    return graph


def test_smallest_id_tie_break_is_the_documented_default():
    graph = regular_ring(14)
    assert targeted_removal_order(graph, 1) == (0,)
    assert targeted_removal_order(graph, 1, tie_break="smallest_id") == (0,)


def test_largest_id_tie_break_picks_the_other_end():
    graph = regular_ring(14)
    assert targeted_removal_order(graph, 1, tie_break="largest_id") == (13,)


def test_tie_break_choice_changes_the_sequence():
    """On a 4-regular ring the whole sequence is decided by the tie-break.

    The two sequences are pinned exactly, and they are *not* ``0,1,2,3,4`` and
    ``13,12,11,10,9``. That is the adaptivity showing: removing node 0 drops
    the degree of its neighbours 1, 2, 12 and 13, so the next node still at
    the maximum degree is 3, not 1. A non-adaptive implementation would walk
    the ring one step at a time, which is exactly what this pins against.
    """
    graph = regular_ring(14)
    small = targeted_removal_order(graph, 5, tie_break="smallest_id")
    large = targeted_removal_order(graph, 5, tie_break="largest_id")

    assert small != large
    assert small == (0, 3, 6, 9, 10)
    assert large == (13, 10, 7, 4, 3)
    # the non-adaptive order on a regular graph is just the ID order
    assert static_removal_order(graph, 5) == (0, 1, 2, 3, 4)
    assert small != static_removal_order(graph, 5)


def test_random_tie_break_requires_a_seed():
    graph = regular_ring(14)
    with pytest.raises(GraphParameterError, match="seed is required"):
        targeted_removal_order(graph, 3, tie_break="random")


def test_random_tie_break_is_deterministic_given_a_seed():
    graph = regular_ring(14)
    first = targeted_removal_order(graph, 5, seed=3, tie_break="random")
    second = targeted_removal_order(graph, 5, seed=3, tie_break="random")
    assert first == second


def test_random_tie_break_is_not_biased_towards_low_ids():
    """Catch a tie-break that silently falls back to the lowest node ID.

    On a 4-regular ring of 28 nodes every node ties at the first step, so a
    correct random tie-break puts first-strike mass across the whole ID range.
    A ``smallest_id`` fallback would return node 0 for every seed, and an
    implementation that merely sorted descending would return 27 for every
    seed. Both are caught: 40 seeds must produce several distinct first
    strikes, with at least one above the midpoint and one below it.
    """
    graph = regular_ring(28)
    firsts = [
        targeted_removal_order(graph, 1, seed=s, tie_break="random")[0]
        for s in range(40)
    ]
    assert len(set(firsts)) > 5, firsts
    assert any(node < 14 for node in firsts)
    assert any(node >= 14 for node in firsts)
    assert set(firsts) != {0}


def test_random_tie_break_still_only_picks_maximum_degree_nodes():
    graph = graph_for("ba", 56)
    order = targeted_removal_order(graph, 10, seed=1, tie_break="random")
    for chosen, available in replay_adaptive_maxima(graph, order):
        assert chosen == available


def test_unknown_tie_break_is_rejected():
    with pytest.raises(GraphParameterError, match="unknown tie_break"):
        targeted_removal_order(regular_ring(14), 1, tie_break="coin_flip")


def test_static_removal_rejects_random_tie_break():
    with pytest.raises(GraphParameterError, match="deterministic tie-breaks only"):
        static_removal_order(regular_ring(14), 1, tie_break="random")


@pytest.mark.parametrize("tie_break", TIE_BREAKS)
def test_every_declared_tie_break_works(tie_break):
    graph = regular_ring(14)
    seed = 0 if tie_break == "random" else None
    order = targeted_removal_order(graph, 4, seed=seed, tie_break=tie_break)
    assert len(set(order)) == 4


# ==========================================================================
# 5. boundary cases
# ==========================================================================

@pytest.mark.parametrize("strategy", REMOVAL_STRATEGIES)
def test_k_zero_is_an_exact_no_op(strategy):
    graph = graph_for("er", 28, roles=True)
    result = remove_nodes_by_count(graph, 0, strategy, seed=0)
    assert result.removed_nodes == ()
    assert result.n_removed == 0
    assert result.realised_fraction == 0.0
    assert sorted(result.graph.nodes()) == sorted(graph.nodes())
    assert sorted(tuple(sorted(e)) for e in result.graph.edges()) == sorted(
        tuple(sorted(e)) for e in graph.edges()
    )
    assert result.connected is nx.is_connected(graph)
    assert result.graph is not graph


@pytest.mark.parametrize("strategy", REMOVAL_STRATEGIES)
def test_k_one(strategy):
    graph = graph_for("ws", 14)
    result = remove_nodes_by_count(graph, 1, strategy, seed=0)
    assert result.n_removed == 1
    assert result.n_surviving == 13


@pytest.mark.parametrize("strategy", REMOVAL_STRATEGIES)
def test_k_equals_n_minus_one_leaves_one_isolated_node(strategy):
    graph = graph_for("er", 14)
    result = remove_nodes_by_count(graph, 13, strategy, seed=0)
    assert result.n_surviving == 1
    assert result.graph.number_of_edges() == 0
    assert result.n_components == 1
    assert result.connected is True
    assert result.mean_degree_surviving == 0.0


@pytest.mark.parametrize("strategy", REMOVAL_STRATEGIES)
def test_k_equals_n_empties_the_graph(strategy):
    graph = graph_for("er", 14)
    result = remove_nodes_by_count(graph, 14, strategy, seed=0)
    assert result.n_surviving == 0
    assert result.graph.number_of_nodes() == 0
    assert result.n_components == 0
    assert result.connected is False
    assert result.mean_degree_surviving is None
    assert result.realised_fraction == 1.0
    assert set(result.removed_nodes) == set(graph.nodes())


@pytest.mark.parametrize("strategy", REMOVAL_STRATEGIES)
def test_k_greater_than_n_raises_value_error(strategy):
    graph = graph_for("er", 14)
    with pytest.raises(GraphParameterError, match="cannot remove k=15"):
        remove_nodes_by_count(graph, 15, strategy, seed=0)
    assert issubclass(GraphParameterError, ValueError)


@pytest.mark.parametrize("strategy", REMOVAL_STRATEGIES)
def test_negative_k_raises_value_error(strategy):
    graph = graph_for("er", 14)
    with pytest.raises(GraphParameterError, match="k must be non-negative"):
        remove_nodes_by_count(graph, -1, strategy, seed=0)


@pytest.mark.parametrize("bad", [1.5, 3.0, "3", None, True, [3]])
@pytest.mark.parametrize("strategy", REMOVAL_STRATEGIES)
def test_non_integer_k_raises_type_error(bad, strategy):
    """Documented: a non-int ``k`` is a ``TypeError``, even when integral."""
    graph = graph_for("er", 14)
    with pytest.raises(TypeError, match="k must be an int"):
        remove_nodes_by_count(graph, bad, strategy, seed=0)


def test_empty_graph_accepts_k_zero_and_rejects_more():
    graph = nx.Graph()
    result = remove_nodes_by_count(graph, 0, "targeted")
    assert result.n_original == 0
    assert result.n_surviving == 0
    assert result.mean_degree_original is None
    assert result.realised_fraction == 0.0
    with pytest.raises(GraphParameterError, match="cannot remove k=1"):
        remove_nodes_by_count(graph, 1, "targeted")


def test_single_node_graph():
    graph = nx.Graph()
    graph.add_node(0)
    assert remove_nodes_by_count(graph, 0, "targeted").n_surviving == 1
    result = remove_nodes_by_count(graph, 1, "targeted")
    assert result.removed_nodes == (0,)
    assert result.n_surviving == 0


def test_disconnected_input_graph_is_accepted():
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (1, 2), (5, 6)])
    graph.add_node(9)
    result = remove_nodes_by_count(graph, 1, "targeted")
    assert result.removed_nodes == (1,)
    assert result.n_components == 4  # {0}, {2}, {5,6}, {9}


def test_unknown_strategy_is_rejected():
    with pytest.raises(GraphParameterError, match="unknown removal strategy"):
        remove_nodes_by_count(graph_for("er", 14), 1, "sabotage", seed=0)


def test_random_strategy_requires_a_seed():
    with pytest.raises(GraphParameterError, match="seed is required"):
        remove_nodes_by_count(graph_for("er", 14), 1, "random")


@pytest.mark.parametrize("bad_seed", [-1, 1.5, "0", True])
def test_invalid_seed_is_rejected(bad_seed):
    with pytest.raises(GraphParameterError):
        remove_nodes_by_count(graph_for("er", 14), 1, "random", seed=bad_seed)


def test_non_graph_input_is_rejected():
    with pytest.raises(GraphParameterError, match="expected a networkx.Graph"):
        remove_nodes_by_count([(0, 1)], 1, "targeted")


def test_directed_input_is_rejected():
    with pytest.raises(GraphParameterError, match="undirected"):
        remove_nodes_by_count(nx.DiGraph([(0, 1)]), 1, "targeted")


# ==========================================================================
# 6. fraction semantics
# ==========================================================================

@pytest.mark.parametrize(
    ("n", "fraction", "expected"),
    [
        (112, 0.1, 11),
        (112, 0.2, 22),
        (112, 0.3, 33),
        (56, 0.1, 5),
        (56, 0.3, 16),
        (28, 0.1, 2),
        (14, 0.1, 1),
        (7, 0.1, 0),
        (7, 0.5, 3),
        (112, 0.0, 0),
        (112, 1.0, 112),
    ],
)
def test_floor_rule_counts(n, fraction, expected):
    assert removal_count(n, fraction) == expected
    assert removal_count(n, fraction, "floor") == expected


@pytest.mark.parametrize(
    ("n", "fraction", "expected"),
    [(112, 0.1, 11), (112, 0.3, 34), (7, 0.1, 1), (56, 0.3, 17)],
)
def test_nearest_rule_counts(n, fraction, expected):
    assert removal_count(n, fraction, "nearest") == expected


def test_the_two_rules_differ_and_the_default_is_floor():
    from neuromesh.graphs import DEFAULT_ROUNDING_RULE

    assert DEFAULT_ROUNDING_RULE == "floor"
    assert removal_count(112, 0.3, "floor") == 33
    assert removal_count(112, 0.3, "nearest") == 34


def test_floor_rule_is_applied_identically_at_every_scale():
    """One rule for all N -- never floor at one scale and round at another."""
    for n in POPULATION_SIZES:
        assert removal_count(n, 0.3) == int(0.3 * n)


def test_floor_never_empties_the_graph_for_a_valid_removal_spec():
    for n in POPULATION_SIZES:
        for fraction in (0.01, 0.5, 0.9, 0.999):
            assert removal_count(n, fraction) < n


def test_unknown_rounding_rule_is_rejected():
    with pytest.raises(GraphParameterError, match="unknown rounding rule"):
        removal_count(112, 0.3, "ceiling")


@pytest.mark.parametrize("bad", ["0.3", None, [0.3]])
def test_non_numeric_fraction_raises_type_error(bad):
    with pytest.raises(TypeError, match="fraction must be a number"):
        removal_count(112, bad)


@pytest.mark.parametrize("bad", [-0.1, 1.1])
def test_out_of_range_fraction_is_rejected(bad):
    with pytest.raises(GraphParameterError, match=r"fraction must lie in \[0, 1\]"):
        removal_count(112, bad)


def test_remove_nodes_by_fraction_records_what_was_requested():
    graph = graph_for("er", 112)
    result = remove_nodes_by_fraction(graph, 0.3, "random", seed=0)
    assert result.requested_fraction == 0.3
    assert result.rounding_rule == "floor"
    assert result.requested_k == 33
    assert result.n_removed == 33
    assert result.realised_fraction == pytest.approx(33 / 112)
    assert result.realised_fraction < 0.3


# ==========================================================================
# 7. the shared contract adapter
# ==========================================================================

@pytest.mark.parametrize("mode", ["random", "targeted"])
def test_contract_adapter_returns_a_bare_graph(mode):
    graph = graph_for("er", 112, roles=True)
    out = remove_nodes(graph, RemovalSpec(mode=mode, fraction=0.2), seed=0)
    assert isinstance(out, nx.Graph)
    assert not isinstance(out, tuple)
    assert out.number_of_nodes() == 112 - 22


@pytest.mark.parametrize("mode", ["random", "targeted"])
def test_contract_adapter_attaches_the_attack_record(mode):
    graph = graph_for("ws", 112, roles=True)
    out = remove_nodes(graph, RemovalSpec(mode=mode, fraction=0.1), seed=1)
    record = out.graph["removal"]

    assert record["strategy"] == mode
    assert record["n_removed"] == 11
    assert record["n_original"] == 112
    assert record["n_surviving"] == 101
    assert record["requested_fraction"] == 0.1
    assert record["rounding_rule"] == "floor"
    assert len(record["removed_nodes"]) == 11
    assert record["readout_surviving"] is not None
    assert record["roles_intact"] in (True, False)


def test_contract_adapter_updates_realised_mean_degree_but_not_the_condition():
    graph = graph_for("er", 112)
    out = remove_nodes(graph, RemovalSpec(mode="targeted", fraction=0.3), seed=0)
    # the condition labels stay put: they identify the experimental cell
    assert out.graph["topology"] == "er"
    assert out.graph["n"] == 112
    assert out.graph["seed"] == 0
    # ...but mean_degree is the realised 2E/N of what survived, not a stale 4.0
    assert out.graph["mean_degree"] != 4.0
    assert out.graph["mean_degree"] == pytest.approx(
        2 * out.number_of_edges() / out.number_of_nodes()
    )


def test_contract_adapter_is_deterministic():
    graph = graph_for("modular", 112, roles=True)
    spec = RemovalSpec(mode="random", fraction=0.2)
    first = remove_nodes(graph, spec, seed=4)
    second = remove_nodes(graph, spec, seed=4)
    assert first.graph["removal"]["removed_nodes"] == second.graph["removal"][
        "removed_nodes"
    ]


def test_contract_adapter_does_not_protect_readout_nodes():
    """Documented deviation from the Week 1 placeholder docstring.

    Readout agents are deliberately *not* spared. The outcome is measured
    instead, so the team can see when an attack has removed the readout group.
    """
    graph = assign_roles(generate_graph("er", 14, seed=0), seed=0)
    readout = [
        node
        for node, data in graph.nodes(data=True)
        if data["role"] == READOUT_ROLE
    ]
    assert readout
    # k = 13 of 14 leaves one node; across every single-survivor outcome the
    # readout group is wiped out unless the survivor happens to be a readout.
    result = remove_nodes_by_count(graph, 13, "targeted")
    assert result.readout_surviving in (0, 1)
    assert result.n_surviving == 1
    # ...and nothing raised or silently spared a node
    assert len(result.removed_nodes) == 13


def test_protected_nodes_are_opt_in_and_respected():
    graph = graph_for("ba", 28, roles=True)
    readout = tuple(
        node
        for node, data in graph.nodes(data=True)
        if data["role"] == READOUT_ROLE
    )
    result = remove_nodes_by_count(graph, 20, "targeted", protected=readout)
    assert set(result.removed_nodes).isdisjoint(readout)
    assert set(readout) <= set(result.graph.nodes())
    assert result.protected_nodes == tuple(sorted(readout))


def test_protecting_too_many_nodes_is_rejected():
    graph = graph_for("ba", 28)
    with pytest.raises(GraphParameterError, match="only 8 of 28 are eligible"):
        remove_nodes_by_count(graph, 20, "random", seed=0, protected=range(20))


def test_protected_node_absent_from_the_graph_is_rejected():
    with pytest.raises(GraphParameterError, match="protected nodes absent"):
        remove_nodes_by_count(graph_for("er", 14), 1, "random", seed=0, protected=[99])


# ==========================================================================
# 8. the five-topology compatibility matrix
# ==========================================================================

def removal_counts_for(n: int) -> tuple[int, ...]:
    """k = 0, 1, a moderate fraction, N-1 and N."""
    return (0, 1, max(1, n // 4), n - 1, n)


@pytest.mark.parametrize("topology", TOPOLOGIES)
@pytest.mark.parametrize("n", POPULATION_SIZES)
@pytest.mark.parametrize("strategy", REMOVAL_STRATEGIES)
def test_full_topology_scale_matrix(topology, n, strategy):
    """Every topology, every population size, five removal counts each.

    5 topologies x 5 sizes x 2 strategies x 5 counts = 250 executed cases.
    """
    graph = graph_for(topology, n, roles=True)
    for k in removal_counts_for(n):
        result = remove_nodes_by_count(graph, k, strategy, seed=0)

        assert result.n_removed == k
        assert result.n_surviving == n - k
        assert len(set(result.removed_nodes)) == k
        survivors = set(result.graph.nodes())
        assert survivors == set(graph.nodes()) - set(result.removed_nodes)
        assert {tuple(sorted(e)) for e in result.graph.edges()} == induced_edges(
            graph, survivors
        )
        if strategy == "targeted":
            for chosen, available in replay_adaptive_maxima(
                graph, result.removed_nodes
            ):
                assert chosen == available
        else:
            assert (
                random_removal_order(graph, k, 0) == result.removed_nodes
            )  # deterministic replay


@pytest.mark.parametrize("topology", TOPOLOGIES)
@pytest.mark.parametrize("seed", SEEDS)
def test_all_five_seeds_at_the_experiment_scale(topology, seed):
    """N = 112 at every study seed, at the three planned removal levels."""
    graph = graph_for(topology, 112, seed=seed, roles=True)
    for fraction, expected_k in ((0.1, 11), (0.2, 22), (0.3, 33)):
        for strategy in REMOVAL_STRATEGIES:
            result = remove_nodes_by_fraction(
                graph, fraction, strategy, seed=seed
            )
            assert result.n_removed == expected_k
            assert result.n_surviving == 112 - expected_k
            checks = validate_attacked_graph(graph, result)
            assert not [c for c in checks if c.status == "FAIL"]


@pytest.mark.parametrize("topology", ["modular", "yeo_sbm"])
def test_community_attribute_survives_removal(topology):
    graph = graph_for(topology, 112, roles=True)
    result = remove_nodes_by_count(graph, 33, "targeted")
    for node in result.graph.nodes():
        assert result.graph.nodes[node]["community"] == graph.nodes[node]["community"]
        assert result.graph.nodes[node]["role"] == graph.nodes[node]["role"]


@pytest.mark.parametrize("topology", TOPOLOGIES)
def test_targeted_is_at_least_as_damaging_as_random_at_112(topology):
    """Sanity check on the science, not just the plumbing.

    A degree-targeted attack should fragment the network at least as much as
    indiscriminate failure of the same size. Asserted on the largest connected
    component, averaged over the five seeds, so one unlucky seed cannot flip
    it.
    """
    targeted, randomised = [], []
    for seed in SEEDS:
        graph = graph_for(topology, 112, seed=seed)
        targeted.append(
            remove_nodes_by_count(graph, 33, "targeted").largest_component_size
        )
        randomised.append(
            remove_nodes_by_count(
                graph, 33, "random", seed=seed
            ).largest_component_size
        )
    assert sum(targeted) / 5 <= sum(randomised) / 5


# ==========================================================================
# 9. the result object
# ==========================================================================

def test_result_is_frozen():
    result = remove_nodes_by_count(graph_for("er", 14), 2, "random", seed=0)
    assert isinstance(result, RemovalResult)
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.n_removed = 99


def test_result_to_dict_is_json_serialisable_and_excludes_the_graph():
    import json

    result = remove_nodes_by_count(
        graph_for("yeo_sbm", 28, roles=True), 7, "targeted"
    )
    payload = result.to_dict()
    assert "graph" not in payload
    text = json.dumps(payload)
    assert json.loads(text) == payload
    assert payload["n_removed"] == 7
    assert payload["provenance"]["topology"] == "yeo_sbm"


def test_result_reports_roles_intact_only_when_roles_exist():
    without = remove_nodes_by_count(graph_for("er", 14), 2, "random", seed=0)
    assert without.role_counts is None
    assert without.roles_intact is None

    with_roles = remove_nodes_by_count(
        graph_for("er", 112, roles=True), 11, "random", seed=0
    )
    assert set(with_roles.role_counts) == set(YEO7)
    assert with_roles.roles_intact is True


def test_result_summary_mentions_the_essentials():
    result = remove_nodes_by_count(
        graph_for("ba", 112, roles=True), 33, "targeted"
    )
    text = result.summary()
    assert "targeted" in text
    assert "33/112" in text
    assert "components=" in text


def test_remove_node_set_matches_the_strategies():
    graph = graph_for("ws", 28)
    order = targeted_removal_order(graph, 7)
    direct = remove_node_set(graph, order)
    viaresult = remove_nodes_by_count(graph, 7, "targeted").graph
    assert sorted(direct.nodes()) == sorted(viaresult.nodes())
    assert sorted(tuple(sorted(e)) for e in direct.edges()) == sorted(
        tuple(sorted(e)) for e in viaresult.edges()
    )


def test_canonical_nodes_orders_mixed_types_without_raising():
    assert canonical_nodes([3, 1, 2]) == [1, 2, 3]
    mixed = canonical_nodes(["b", 2, "a", 1])
    assert len(mixed) == 4
    assert canonical_nodes(["b", 2, "a", 1]) == mixed  # stable across calls


def test_negative_n_in_removal_count_is_rejected():
    with pytest.raises(GraphParameterError, match="n must be non-negative"):
        removal_count(-1, 0.3)


def test_static_removal_of_zero_nodes_is_a_no_op():
    assert static_removal_order(graph_for("er", 14), 0) == ()


def test_static_removal_handles_string_node_ids():
    """Non-integer IDs still get a total, direction-aware order."""
    graph = nx.Graph()
    graph.add_edges_from([("a", "b"), ("b", "c"), ("c", "a"), ("c", "d")])
    assert static_removal_order(graph, 1) == ("c",)  # degree 3
    # among the degree-2 ties a/b, the ID direction decides
    assert static_removal_order(graph, 2) == ("c", "a")
    assert static_removal_order(graph, 2, tie_break="largest_id") == ("c", "b")
