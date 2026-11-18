"""Tests for the block-model core and the modular topology.

The block machinery is shared by ``modular`` and ``yeo_sbm``, so it is tested
once here against the uniform planted partition, where the expected structure
is analytically known. Where finite size makes a structural property
impossible, the test asserts the *impossibility* rather than relaxing the
assertion.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from neuromesh.graphs import (
    N_BLOCKS,
    POPULATION_SIZES,
    TARGET_MEAN_DEGREE,
    GraphParameterError,
    GraphSpec,
    allocate_block_edges,
    balanced_block_sizes,
    block_assignment,
    community_profile,
    generate_from_spec,
    generate_graph,
    mean_degree,
    modular,
    pair_capacity,
    sample_block_graph,
    validate_block_weights,
)

SEEDS = (0, 1, 2, 3, 4)


def _uniform_weights(ratio: float = 10.0) -> np.ndarray:
    weights = np.ones((N_BLOCKS, N_BLOCKS))
    np.fill_diagonal(weights, ratio)
    return weights


# --------------------------------------------------------------------------
# block layout
# --------------------------------------------------------------------------

@pytest.mark.parametrize("n", POPULATION_SIZES)
def test_blocks_are_exactly_balanced(n):
    sizes = balanced_block_sizes(n)
    assert len(sizes) == N_BLOCKS
    assert set(sizes) == {n // N_BLOCKS}
    assert sum(sizes) == n


@pytest.mark.parametrize("n", POPULATION_SIZES)
def test_block_assignment_covers_every_node_contiguously(n):
    blocks = block_assignment(n)
    assert len(blocks) == n
    assert sorted(set(blocks)) == list(range(N_BLOCKS))
    assert all(blocks.count(b) == n // N_BLOCKS for b in range(N_BLOCKS))
    # contiguous runs: block index is non-decreasing along node IDs
    assert list(blocks) == sorted(blocks)


@pytest.mark.parametrize("n", [8, 10, 13, 100])
def test_sizes_not_divisible_by_seven_are_rejected(n):
    with pytest.raises(GraphParameterError, match="not divisible by 7"):
        balanced_block_sizes(n)


def test_balanced_block_sizes_rejects_non_integer():
    with pytest.raises(GraphParameterError, match="n must be an integer"):
        balanced_block_sizes(14.0)


@pytest.mark.parametrize("n", POPULATION_SIZES)
def test_pair_capacity_matches_the_combinatorics(n):
    sizes = balanced_block_sizes(n)
    capacity = pair_capacity(sizes)
    m = n // N_BLOCKS

    for a in range(N_BLOCKS):
        assert capacity[a, a] == m * (m - 1) // 2
        for b in range(a + 1, N_BLOCKS):
            assert capacity[a, b] == capacity[b, a] == m * m
    # the upper triangle must account for every node pair exactly once
    assert capacity[np.triu_indices(N_BLOCKS)].sum() == n * (n - 1) // 2


# --------------------------------------------------------------------------
# weight validation
# --------------------------------------------------------------------------

def test_valid_weights_are_returned_as_a_float_array():
    weights = validate_block_weights(_uniform_weights())
    assert weights.shape == (N_BLOCKS, N_BLOCKS)
    assert weights.dtype == float


@pytest.mark.parametrize(
    "bad, match",
    [
        (np.ones((3, 3)), "must be a 7x7 matrix"),
        (np.ones(7), "must be a 7x7 matrix"),
        (np.full((7, 7), np.nan), "must all be finite"),
        (np.full((7, 7), -1.0), "must be non-negative"),
        (np.zeros((7, 7)), "must not be identically zero"),
    ],
)
def test_invalid_weight_matrices_are_rejected(bad, match):
    with pytest.raises(GraphParameterError, match=match):
        validate_block_weights(bad)


def test_asymmetric_weights_are_rejected():
    weights = _uniform_weights()
    weights[0, 1] = 5.0  # leaves weights[1, 0] at 1.0
    with pytest.raises(GraphParameterError, match="must be symmetric"):
        validate_block_weights(weights)


# --------------------------------------------------------------------------
# edge budget allocation
# --------------------------------------------------------------------------

@pytest.mark.parametrize("n", POPULATION_SIZES)
def test_allocation_spends_the_whole_budget_and_respects_capacity(n):
    sizes = balanced_block_sizes(n)
    capacity = pair_capacity(sizes)
    budget = 2 * n

    allocation, info = allocate_block_edges(sizes, _uniform_weights(), budget)

    upper = np.triu_indices(N_BLOCKS)
    assert allocation[upper].sum() == budget
    assert (allocation <= capacity).all()
    assert (allocation >= 0).all()
    assert np.array_equal(allocation, allocation.T)
    assert info["n_within_edges"] + info["n_between_edges"] == budget


def test_allocation_prefers_within_blocks_when_capacity_allows():
    """At N >= 28 the requested preference is realisable and must be realised."""
    sizes = balanced_block_sizes(28)
    _, info = allocate_block_edges(sizes, _uniform_weights(ratio=10.0), 56)

    assert info["within_fraction"] > 0.5
    assert info["capacity_clamped"] is False


def test_ratio_one_gives_a_capacity_proportional_split():
    """With no preference the split must follow capacity, not block identity.

    The match is exact only up to integer rounding. Proportional demand leaves
    7 edges of the 224 to be handed out by largest remainder, and the diagonal
    cells happen to carry the larger remainders, so the realised within share
    can sit up to ``7/224`` above the capacity share. The tolerance below is
    that granularity, not a fitted value.
    """
    sizes = balanced_block_sizes(112)
    capacity = pair_capacity(sizes)
    expected = capacity[np.diag_indices(N_BLOCKS)].sum() / capacity[
        np.triu_indices(N_BLOCKS)
    ].sum()

    allocation, info = allocate_block_edges(sizes, _uniform_weights(ratio=1.0), 224)

    deficit = 224 - int(np.floor(224 * capacity[np.triu_indices(N_BLOCKS)] / capacity[
        np.triu_indices(N_BLOCKS)
    ].sum()).sum())
    assert info["within_fraction"] == pytest.approx(expected, abs=deficit / 224)
    # and it must be far below what an actual preference produces
    _, preferred = allocate_block_edges(sizes, _uniform_weights(ratio=10.0), 224)
    assert info["within_fraction"] < preferred["within_fraction"] / 2


def test_n_seven_has_no_within_block_capacity_at_all():
    """One node per block: a within-block edge is impossible, not merely rare."""
    sizes = balanced_block_sizes(7)
    allocation, info = allocate_block_edges(sizes, _uniform_weights(), 14)

    assert info["within_capacity"] == 0
    assert info["n_within_edges"] == 0
    assert info["within_structure_possible"] is False
    assert all(allocation[a, a] == 0 for a in range(N_BLOCKS))


def test_n_fourteen_is_capped_by_within_block_capacity():
    """Two nodes per block cap the within share at 7 of 28 edges (25%)."""
    sizes = balanced_block_sizes(14)
    _, info = allocate_block_edges(sizes, _uniform_weights(ratio=10.0), 28)

    assert info["within_capacity"] == 7
    assert info["n_within_edges"] == 7
    assert info["within_fraction"] == pytest.approx(0.25)
    assert info["capacity_clamped"] is True, (
        "a preference that capacity cannot satisfy must be reported, not hidden"
    )


def test_budget_beyond_total_capacity_is_rejected():
    sizes = balanced_block_sizes(7)
    with pytest.raises(GraphParameterError, match="admits at most"):
        allocate_block_edges(sizes, _uniform_weights(), 22)  # C(7,2) = 21


def test_negative_budget_is_rejected():
    with pytest.raises(GraphParameterError, match="non-negative"):
        allocate_block_edges(balanced_block_sizes(14), _uniform_weights(), -1)


def test_zero_weight_blocks_fall_back_to_capacity_and_say_so():
    """A matrix with no usable weight must not silently produce nothing."""
    sizes = balanced_block_sizes(14)
    weights = np.zeros((N_BLOCKS, N_BLOCKS))
    weights[0, 0] = 1.0  # only a zero-capacity diagonal cell at N = 14? no: cap 1

    allocation, info = allocate_block_edges(sizes, weights, 28)
    assert allocation[np.triu_indices(N_BLOCKS)].sum() == 28
    assert info["degenerate_weights"] is False


def test_weights_with_zero_demand_everywhere_are_reported_as_degenerate():
    sizes = balanced_block_sizes(7)  # every within-block capacity is 0
    weights = np.zeros((N_BLOCKS, N_BLOCKS))
    np.fill_diagonal(weights, 1.0)  # all weight on zero-capacity blocks

    allocation, info = allocate_block_edges(sizes, weights, 14)
    assert info["degenerate_weights"] is True
    assert allocation[np.triu_indices(N_BLOCKS)].sum() == 14


# --------------------------------------------------------------------------
# sampling core
# --------------------------------------------------------------------------

@pytest.mark.parametrize("n", POPULATION_SIZES)
@pytest.mark.parametrize("seed", SEEDS)
def test_sampled_block_graph_is_simple_connected_and_degree_matched(n, seed):
    graph, info = sample_block_graph(n, _uniform_weights(), seed)

    assert set(graph.nodes()) == set(range(n))
    assert graph.number_of_edges() == 2 * n
    assert mean_degree(graph) == pytest.approx(TARGET_MEAN_DEGREE)
    assert not graph.is_directed()
    assert not graph.is_multigraph()
    assert nx.number_of_selfloops(graph) == 0
    assert nx.is_connected(graph)
    assert info["attempts"] >= 1


@pytest.mark.parametrize("n", POPULATION_SIZES)
def test_sampled_graph_realises_the_planned_allocation_exactly(n):
    graph, info = sample_block_graph(n, _uniform_weights(), 0)
    communities = nx.get_node_attributes(graph, "community")

    realised = np.zeros((N_BLOCKS, N_BLOCKS), dtype=int)
    for u, v in graph.edges():
        a, b = sorted((communities[u], communities[v]))
        realised[a, b] += 1

    planned = np.array(info["allocation"])
    for a in range(N_BLOCKS):
        for b in range(a, N_BLOCKS):
            assert realised[a, b] == planned[a, b]


@pytest.mark.parametrize("n", POPULATION_SIZES)
def test_every_node_carries_a_community_attribute(n):
    graph, _ = sample_block_graph(n, _uniform_weights(), 0)
    communities = nx.get_node_attributes(graph, "community")

    assert len(communities) == n
    assert sorted(set(communities.values())) == list(range(N_BLOCKS))


def test_sampling_rejects_bad_max_attempts():
    with pytest.raises(GraphParameterError, match="max_attempts"):
        sample_block_graph(14, _uniform_weights(), 0, max_attempts=0)


def test_sampling_rejects_an_invalid_n():
    with pytest.raises(GraphParameterError, match="n must be >= 3"):
        sample_block_graph(2, _uniform_weights(), 0)


# --------------------------------------------------------------------------
# the modular generator
# --------------------------------------------------------------------------

@pytest.mark.parametrize("n", POPULATION_SIZES)
@pytest.mark.parametrize("seed", SEEDS)
def test_modular_matches_mean_degree_exactly(n, seed):
    graph, meta = modular(n, seed=seed)

    assert graph.number_of_edges() == 2 * n
    assert mean_degree(graph) == pytest.approx(TARGET_MEAN_DEGREE)
    assert meta.degree_matched_exactly is True
    assert meta.topology == "modular"


@pytest.mark.parametrize("n", POPULATION_SIZES)
@pytest.mark.parametrize("seed", SEEDS)
def test_modular_is_connected_and_simple(n, seed):
    graph, meta = modular(n, seed=seed)

    assert nx.is_connected(graph)
    assert meta.connected is True
    assert meta.n_components == 1
    assert nx.number_of_selfloops(graph) == 0
    assert not graph.is_multigraph()


@pytest.mark.parametrize("n", POPULATION_SIZES)
def test_modular_is_reproducible(n):
    first, _ = modular(n, seed=2)
    second, _ = modular(n, seed=2)

    assert {tuple(sorted(e)) for e in first.edges()} == {
        tuple(sorted(e)) for e in second.edges()
    }


@pytest.mark.parametrize("n", [14, 28, 56, 112])
def test_modular_different_seeds_give_different_graphs(n):
    edge_sets = {
        frozenset(tuple(sorted(e)) for e in modular(n, seed=s)[0].edges())
        for s in SEEDS
    }
    assert len(edge_sets) > 1


@pytest.mark.parametrize("n", [28, 56, 112])
def test_modular_has_real_community_structure_where_capacity_allows(n):
    """At N >= 28 the graph must be measurably more modular than chance."""
    graph, _ = modular(n, seed=0)
    profile = community_profile(graph)

    assert profile["within_fraction"] > 0.5
    assert profile["modularity"] > 0.3
    assert profile["n_communities"] == N_BLOCKS
    assert set(profile["community_sizes"]) == {n // N_BLOCKS}


def test_modular_at_n_seven_has_no_community_structure():
    """A declared boundary case: one node per block, so no within-block edge.

    N = 7 modular is a labelled random graph, not a modular graph, and its
    negative modularity says so. It must not be presented as a modular topology.
    """
    graph, meta = modular(7, seed=0)
    profile = community_profile(graph)

    assert profile["n_within_edges"] == 0
    assert profile["modularity"] < 0
    assert meta.params["within_structure_possible"] is False


def test_modular_at_n_fourteen_is_capacity_limited():
    """Two nodes per block cap the within share at 25% whatever the preference."""
    graph, meta = modular(14, seed=0)
    profile = community_profile(graph)

    assert profile["n_within_edges"] == 7
    assert profile["within_fraction"] == pytest.approx(0.25)
    assert meta.params["capacity_clamped"] is True


def test_higher_ratio_yields_more_within_block_edges():
    """The preference parameter must actually drive the structure."""
    low = community_profile(modular(112, seed=0, within_between_ratio=1.0)[0])
    high = community_profile(modular(112, seed=0, within_between_ratio=20.0)[0])

    assert high["n_within_edges"] > low["n_within_edges"]
    assert high["modularity"] > low["modularity"]


def test_modular_records_its_parameters():
    _, meta = modular(56, seed=0)

    assert meta.params["model"] == "planted_partition"
    assert meta.params["within_between_ratio"] == 10.0
    assert meta.params["n_blocks"] == N_BLOCKS
    assert meta.params["block_sizes"] == [8] * N_BLOCKS
    assert (
        meta.params["n_within_edges"] + meta.params["n_between_edges"]
        == meta.n_edges
    )


def test_modular_version_is_week_two_and_week_one_is_untouched():
    """New topologies carry their own version; existing fixtures must not move."""
    _, modular_meta = modular(28, seed=0)
    _, er_meta = generate_from_spec(GraphSpec("er", n=28, seed=0))

    assert modular_meta.generator_version == "trackA-week2-1.0.0"
    assert er_meta.generator_version == "trackA-week1-1.0.0"


@pytest.mark.parametrize("bad_ratio", [0, -1, float("nan"), float("inf"), "10", None])
def test_modular_rejects_an_invalid_ratio(bad_ratio):
    with pytest.raises(GraphParameterError, match="within_between_ratio"):
        modular(28, seed=0, within_between_ratio=bad_ratio)


def test_modular_rejects_a_size_not_divisible_by_seven():
    with pytest.raises(GraphParameterError, match="not divisible by 7"):
        modular(10, seed=0)


# --------------------------------------------------------------------------
# community_profile itself
# --------------------------------------------------------------------------

def test_community_profile_rejects_an_unlabelled_graph():
    with pytest.raises(GraphParameterError, match="'community' attribute"):
        community_profile(generate_graph("er", 14))


def test_community_profile_rejects_a_non_graph():
    with pytest.raises(GraphParameterError, match="networkx.Graph"):
        community_profile("not a graph")


def test_community_profile_counts_edges_correctly():
    graph = nx.Graph()
    graph.add_nodes_from(range(4))
    nx.set_node_attributes(graph, {0: 0, 1: 0, 2: 1, 3: 1}, "community")
    graph.add_edges_from([(0, 1), (2, 3), (1, 2)])

    profile = community_profile(graph)
    assert profile["n_within_edges"] == 2
    assert profile["n_between_edges"] == 1
    assert profile["within_fraction"] == pytest.approx(2 / 3)
