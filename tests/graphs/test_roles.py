"""Tests for balanced Yeo role assignment.

Role placement decides where information enters (``INPUT_ROLE``) and where it
is read out (``READOUT_ROLE``), so an assignment that is unbalanced, partial or
irreproducible would silently change what every task measures. These tests
assert the exact counts rather than a tolerance.
"""

from __future__ import annotations

import networkx as nx
import pytest

from neuromesh.contracts import INPUT_ROLE, READOUT_ROLE, YEO7
from neuromesh.graphs import (
    BLOCK_TOPOLOGIES,
    POPULATION_SIZES,
    TOPOLOGIES,
    GraphParameterError,
    assign_roles,
    generate_graph,
    role_counts,
    roles_are_balanced,
)

SEEDS = (0, 1, 2, 3, 4)
PLAIN_TOPOLOGIES = tuple(t for t in TOPOLOGIES if t not in BLOCK_TOPOLOGIES)


# --------------------------------------------------------------------------
# balance: exactly n/7 agents per functional network
# --------------------------------------------------------------------------

@pytest.mark.parametrize("topology", TOPOLOGIES)
@pytest.mark.parametrize("n", POPULATION_SIZES)
@pytest.mark.parametrize("seed", SEEDS)
def test_every_node_gets_exactly_one_valid_role(topology, n, seed):
    graph = assign_roles(generate_graph(topology, n, seed=seed), seed)

    roles = nx.get_node_attributes(graph, "role")
    assert len(roles) == n, "every node must carry a role"
    assert set(roles) == set(range(n))
    assert set(roles.values()) <= set(YEO7)


@pytest.mark.parametrize("topology", TOPOLOGIES)
@pytest.mark.parametrize("n", POPULATION_SIZES)
@pytest.mark.parametrize("seed", SEEDS)
def test_roles_are_balanced_at_exactly_n_over_seven(topology, n, seed):
    """The design premise: N/7 agents per Yeo network, at every scale."""
    graph = assign_roles(generate_graph(topology, n, seed=seed), seed)

    assert roles_are_balanced(graph)
    assert role_counts(graph) == dict.fromkeys(YEO7, n // len(YEO7))


@pytest.mark.parametrize("topology", TOPOLOGIES)
@pytest.mark.parametrize("n", POPULATION_SIZES)
def test_input_and_readout_roles_are_always_present(topology, n):
    """Every task needs at least one Visual source and one Frontoparietal sink."""
    graph = assign_roles(generate_graph(topology, n), 0)
    counts = role_counts(graph)

    assert counts[INPUT_ROLE] == n // len(YEO7) >= 1
    assert counts[READOUT_ROLE] == n // len(YEO7) >= 1


# --------------------------------------------------------------------------
# the graph itself must not be touched
# --------------------------------------------------------------------------

@pytest.mark.parametrize("topology", TOPOLOGIES)
def test_assignment_does_not_modify_edges_or_nodes(topology):
    graph = generate_graph(topology, 28, seed=2)
    edges_before = {tuple(sorted(e)) for e in graph.edges()}
    nodes_before = set(graph.nodes())

    assign_roles(graph, 2)

    assert {tuple(sorted(e)) for e in graph.edges()} == edges_before
    assert set(graph.nodes()) == nodes_before


@pytest.mark.parametrize("topology", TOPOLOGIES)
def test_assignment_is_in_place_and_returns_the_same_object(topology):
    """Both ``assign_roles(g, s)`` and ``g = assign_roles(g, s)`` must work."""
    graph = generate_graph(topology, 14)
    returned = assign_roles(graph, 0)

    assert returned is graph
    assert len(nx.get_node_attributes(graph, "role")) == 14


def test_generated_graphs_carry_no_role_until_assigned():
    """Roles are opt-in: generation must not invent them."""
    graph = generate_graph("er", 28)
    assert nx.get_node_attributes(graph, "role") == {}


# --------------------------------------------------------------------------
# reproducibility
# --------------------------------------------------------------------------

@pytest.mark.parametrize("topology", TOPOLOGIES)
@pytest.mark.parametrize("n", POPULATION_SIZES)
def test_same_graph_and_seed_give_the_same_assignment(topology, n):
    first = assign_roles(generate_graph(topology, n, seed=1), 3)
    second = assign_roles(generate_graph(topology, n, seed=1), 3)

    assert nx.get_node_attributes(first, "role") == nx.get_node_attributes(second, "role")


@pytest.mark.parametrize("topology", PLAIN_TOPOLOGIES)
@pytest.mark.parametrize("n", [14, 28, 56, 112])
def test_different_seeds_give_different_assignments(topology, n):
    """Randomised placement must actually vary, or redrawing per seed is vacuous.

    N = 7 is excluded: with one agent per network every balanced assignment is
    a permutation of seven distinct labels over seven nodes, and while those do
    differ, the property is asserted from N = 14 for consistency with the rest
    of the suite, where N = 7 is a declared boundary case.
    """
    assignments = {
        tuple(
            nx.get_node_attributes(
                assign_roles(generate_graph(topology, n, seed=0), seed), "role"
            )[node]
            for node in range(n)
        )
        for seed in SEEDS
    }
    assert len(assignments) > 1


@pytest.mark.parametrize("topology", PLAIN_TOPOLOGIES)
def test_role_stream_is_independent_of_the_graph_seed(topology):
    """Role placement must not be correlated with the draw that made the graph.

    The same graph seed with different role seeds must give different
    placements, which is only true if the two consume separate random streams.
    """
    graph_a = assign_roles(generate_graph(topology, 56, seed=4), 0)
    graph_b = assign_roles(generate_graph(topology, 56, seed=4), 1)

    assert nx.get_node_attributes(graph_a, "role") != nx.get_node_attributes(
        graph_b, "role"
    )


# --------------------------------------------------------------------------
# community-structured topologies keep their blocks
# --------------------------------------------------------------------------

@pytest.mark.parametrize("topology", BLOCK_TOPOLOGIES)
@pytest.mark.parametrize("n", POPULATION_SIZES)
def test_block_topologies_map_each_community_to_one_role(topology, n):
    """Block b is functional network b; roles must not be reshuffled across it."""
    graph = assign_roles(generate_graph(topology, n, seed=0), 0)

    for node, data in graph.nodes(data=True):
        assert data["role"] == YEO7[data["community"]], (
            f"node {node} is in community {data['community']} but carries role "
            f"{data['role']!r}"
        )


@pytest.mark.parametrize("topology", BLOCK_TOPOLOGIES)
def test_block_topology_roles_are_one_to_one_with_communities(topology):
    graph = assign_roles(generate_graph(topology, 56, seed=0), 0)

    pairs = {(d["community"], d["role"]) for _, d in graph.nodes(data=True)}
    assert len(pairs) == len(YEO7)
    assert len({c for c, _ in pairs}) == len({r for _, r in pairs}) == len(YEO7)


@pytest.mark.parametrize("topology", BLOCK_TOPOLOGIES)
def test_block_topology_assignment_is_seed_independent_by_design(topology):
    """Community membership fixes the role, so no freedom is left for the seed.

    This is a property of the design, not an oversight: reshuffling roles across
    blocks would break the correspondence between the generator's 7x7 weight
    matrix and the roles the agents play.
    """
    first = assign_roles(generate_graph(topology, 28, seed=0), 0)
    second = assign_roles(generate_graph(topology, 28, seed=0), 4)

    assert nx.get_node_attributes(first, "role") == nx.get_node_attributes(
        second, "role"
    )
    assert first.graph["role_assignment"]["seed_dependent"] is False


@pytest.mark.parametrize("topology", PLAIN_TOPOLOGIES)
def test_plain_topologies_report_the_randomised_regime(topology):
    graph = assign_roles(generate_graph(topology, 28), 0)
    record = graph.graph["role_assignment"]

    assert record["regime"] == "randomised_balanced"
    assert record["seed_dependent"] is True
    assert record["per_role"] == 4
    assert record["roles"] == list(YEO7)


@pytest.mark.parametrize("topology", BLOCK_TOPOLOGIES)
def test_block_topologies_report_the_community_regime(topology):
    graph = assign_roles(generate_graph(topology, 28), 0)
    assert graph.graph["role_assignment"]["regime"] == "community"


# --------------------------------------------------------------------------
# boundary case: N = 7
# --------------------------------------------------------------------------

@pytest.mark.parametrize("topology", TOPOLOGIES)
def test_n_seven_gives_exactly_one_agent_per_network(topology):
    """At the boundary size each functional network has a single agent."""
    graph = assign_roles(generate_graph(topology, 7), 0)

    assert role_counts(graph) == dict.fromkeys(YEO7, 1)
    assert len(set(nx.get_node_attributes(graph, "role").values())) == len(YEO7)


# --------------------------------------------------------------------------
# error handling
# --------------------------------------------------------------------------

def test_graph_whose_size_is_not_divisible_by_seven_is_rejected():
    graph = nx.path_graph(10)
    with pytest.raises(GraphParameterError, match="not divisible by 7"):
        assign_roles(graph, 0)


def test_non_contiguous_node_ids_are_rejected():
    graph = nx.relabel_nodes(nx.path_graph(14), {0: "agent_0"})
    with pytest.raises(GraphParameterError, match="exactly the integers"):
        assign_roles(graph, 0)


def test_empty_graph_is_rejected():
    with pytest.raises(GraphParameterError, match="empty graph"):
        assign_roles(nx.Graph(), 0)


def test_non_graph_input_is_rejected():
    with pytest.raises(GraphParameterError, match="networkx.Graph"):
        assign_roles("not a graph", 0)


@pytest.mark.parametrize("bad_seed", [-1, 1.5, "0", None, True])
def test_invalid_seed_is_rejected(bad_seed):
    graph = generate_graph("er", 14)
    with pytest.raises(GraphParameterError, match="seed must be"):
        assign_roles(graph, bad_seed)


def test_partially_community_labelled_graph_is_rejected():
    """An ambiguous half-labelled graph must not be silently randomised."""
    graph = generate_graph("er", 14)
    graph.nodes[0]["community"] = 0

    with pytest.raises(GraphParameterError, match="partially community-labelled"):
        assign_roles(graph, 0)


def test_community_indices_outside_zero_to_six_are_rejected():
    graph = generate_graph("er", 14)
    nx.set_node_attributes(graph, {i: i % 3 for i in range(14)}, "community")

    with pytest.raises(GraphParameterError, match="community indices must be"):
        assign_roles(graph, 0)


def test_role_counts_rejects_an_unknown_role():
    graph = generate_graph("er", 14)
    nx.set_node_attributes(graph, {0: "cerebellum"}, "role")

    with pytest.raises(GraphParameterError, match="unknown role"):
        role_counts(graph)


def test_role_counts_rejects_a_non_graph():
    with pytest.raises(GraphParameterError, match="networkx.Graph"):
        role_counts(object())


def test_unassigned_graph_is_not_balanced():
    assert roles_are_balanced(generate_graph("er", 14)) is False


def test_graph_with_unassignable_size_is_not_balanced():
    assert roles_are_balanced(nx.path_graph(10)) is False
