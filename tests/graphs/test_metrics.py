"""Tests for structural metrics and positional covariates (owner: Mihir, Track A).

The metrics are verified against INDEPENDENTLY KNOWN mathematics wherever one
exists -- a path graph's average shortest path, a star's degree assortativity
of exactly -1, a complete graph's clustering of exactly 1 -- rather than
against the implementation's own output. Where no closed form is handy, the
assertion recomputes the quantity here by a different route (a hand-written
loop, or NetworkX called directly) so the test and the code cannot share a
mistake.
"""

from __future__ import annotations

import math

import networkx as nx
import pytest

from neuromesh.contracts import INPUT_ROLE, READOUT_ROLE, YEO7
from neuromesh.graphs import (
    POPULATION_SIZES,
    POSITIONAL_FIELDS,
    STRUCTURAL_FIELDS,
    TOPOLOGIES,
    GraphParameterError,
    assign_roles,
    degree_assortativity,
    degree_heterogeneity,
    degree_statistics,
    generate_graph,
    graph_id,
    input_readout_distance,
    partition_modularity,
    positional_covariates,
    remove_nodes_by_count,
    role_balance,
    role_partition,
    structural_metrics,
)
from neuromesh.graphs.metrics import (
    MODULARITY_RESOLUTION,
    PAGERANK_ALPHA,
    eigenvector_status,
)

SEEDS = (0, 1, 2, 3, 4)

_CACHE: dict[tuple, nx.Graph] = {}


def study_graph(topology: str, n: int, seed: int = 0) -> nx.Graph:
    """Cached study graph with roles assigned."""
    key = (topology, n, seed)
    if key not in _CACHE:
        graph = generate_graph(topology, n, seed=seed)
        assign_roles(graph, seed=seed)
        _CACHE[key] = graph
    return _CACHE[key]


def roled(graph: nx.Graph) -> nx.Graph:
    """Attach balanced roles to a hand-built graph whose N is divisible by 7."""
    return assign_roles(graph, seed=0)


# ==========================================================================
# 1. hand-built graphs with known closed-form values
# ==========================================================================

def test_path_graph_known_values():
    """P5: avg path exactly 2.0, diameter 4, no triangles."""
    metrics = structural_metrics(nx.path_graph(5))
    assert metrics["n_nodes"] == 5
    assert metrics["n_edges"] == 4
    assert metrics["mean_degree"] == pytest.approx(1.6)
    assert metrics["average_shortest_path_length"] == pytest.approx(2.0)
    assert metrics["diameter"] == 4
    assert metrics["radius"] == 2
    assert metrics["average_clustering"] == 0.0
    assert metrics["transitivity"] == 0.0
    assert metrics["n_components"] == 1
    assert metrics["largest_component_fraction"] == pytest.approx(1.0)
    # density 2E/(N(N-1)) = 8/20
    assert metrics["edge_density"] == pytest.approx(0.4)
    # degrees [1,2,2,2,1]: <k>=1.6, <k^2>=2.8  ->  2.8/2.56
    assert metrics["degree_heterogeneity"] == pytest.approx(2.8 / 1.6**2)


def test_cycle_graph_known_values():
    """C10: diameter exactly N//2, perfectly regular so no assortativity."""
    metrics = structural_metrics(nx.cycle_graph(10))
    assert metrics["diameter"] == 5
    assert metrics["mean_degree"] == pytest.approx(2.0)
    assert metrics["degree_heterogeneity"] == pytest.approx(1.0)
    assert metrics["degree_min"] == metrics["degree_max"] == 2
    assert metrics["degree_variance"] == 0.0
    assert metrics["average_clustering"] == 0.0
    # a regular graph has an undefined degree correlation
    assert metrics["degree_assortativity"] is None


def test_star_graph_known_values():
    """A star is perfectly disassortative: r = -1 exactly."""
    metrics = structural_metrics(nx.star_graph(6))  # 7 nodes
    assert metrics["n_nodes"] == 7
    assert metrics["degree_max"] == 6
    assert metrics["degree_min"] == 1
    assert metrics["diameter"] == 2
    assert metrics["average_clustering"] == 0.0
    assert metrics["transitivity"] == 0.0
    assert metrics["degree_assortativity"] == pytest.approx(-1.0)
    assert metrics["hub_degree"] == 6


def test_complete_graph_known_values():
    """K7: every normalised quantity is exactly 1, heterogeneity exactly 1."""
    metrics = structural_metrics(nx.complete_graph(7))
    assert metrics["average_clustering"] == pytest.approx(1.0)
    assert metrics["transitivity"] == pytest.approx(1.0)
    assert metrics["diameter"] == 1
    assert metrics["radius"] == 1
    assert metrics["edge_density"] == pytest.approx(1.0)
    assert metrics["global_efficiency"] == pytest.approx(1.0)
    assert metrics["degree_heterogeneity"] == pytest.approx(1.0)
    assert metrics["average_shortest_path_length"] == pytest.approx(1.0)


def test_two_disconnected_components():
    """Whole-graph path metrics are None; component readings are separate."""
    graph = nx.disjoint_union(nx.path_graph(3), nx.path_graph(3))
    metrics = structural_metrics(graph)

    assert metrics["n_components"] == 2
    assert metrics["average_shortest_path_length"] is None
    assert metrics["diameter"] is None
    assert metrics["radius"] is None
    # ...but the largest-component values are available under their own names
    assert metrics["diameter_largest_component"] == 2
    assert metrics["average_shortest_path_length_largest_component"] == pytest.approx(
        4 / 3
    )
    assert metrics["largest_component_size"] == 3
    assert metrics["largest_component_fraction"] == pytest.approx(0.5)
    # global efficiency IS defined for a disconnected graph:
    # each P3 contributes 2*(1+1+0.5) = 5, two of them = 10, over 6*5=30 pairs
    assert metrics["global_efficiency"] == pytest.approx(10 / 30)


def test_graph_with_an_isolated_node():
    graph = nx.path_graph(4)
    graph.add_node(99)
    metrics = structural_metrics(graph)
    assert metrics["n_components"] == 2
    assert metrics["degree_min"] == 0
    assert metrics["diameter"] is None
    assert metrics["diameter_largest_component"] == 3
    assert metrics["largest_component_size"] == 4


def test_empty_and_single_node_graphs():
    empty = structural_metrics(nx.Graph())
    assert empty["n_nodes"] == 0
    assert empty["mean_degree"] is None
    assert empty["edge_density"] is None
    assert empty["degree_variance"] is None
    assert empty["global_efficiency"] is None

    single = structural_metrics(nx.empty_graph(1))
    assert single["n_nodes"] == 1
    assert single["edge_density"] is None
    assert single["diameter"] == 0
    assert single["global_efficiency"] == 0.0


def test_directed_and_non_graph_inputs_are_rejected():
    with pytest.raises(GraphParameterError, match="expected a networkx.Graph"):
        structural_metrics([(0, 1)])
    with pytest.raises(GraphParameterError, match="undirected"):
        structural_metrics(nx.DiGraph([(0, 1)]))
    with pytest.raises(GraphParameterError, match="undirected"):
        positional_covariates(nx.DiGraph([(0, 1)]))


# ==========================================================================
# 2. undefined values are None, never a misleading zero
# ==========================================================================

def test_assortativity_is_none_not_zero_for_regular_graphs():
    """The distinction matters: 0.0 would claim "measured, and neutral"."""
    for graph in (nx.cycle_graph(12), nx.circulant_graph(14, [1, 2])):
        assert degree_assortativity(graph) is None
    # and a graph that genuinely has one returns a number
    assert degree_assortativity(nx.star_graph(5)) == pytest.approx(-1.0)


def test_assortativity_emits_no_runtime_warning():
    """A degenerate correlation is a missing measurement, not a warning."""
    with warnings_as_errors():
        assert degree_assortativity(nx.cycle_graph(10)) is None


class warnings_as_errors:  # noqa: N801 - a tiny test helper, not a public class
    def __enter__(self):
        import warnings

        self._ctx = warnings.catch_warnings()
        self._ctx.__enter__()
        warnings.simplefilter("error")
        return self

    def __exit__(self, *exc):
        return self._ctx.__exit__(*exc)


def test_assortativity_none_for_an_edgeless_graph():
    assert degree_assortativity(nx.empty_graph(5)) is None


def test_heterogeneity_none_when_there_are_no_edges():
    assert degree_heterogeneity(nx.empty_graph(4)) is None
    assert degree_heterogeneity(nx.Graph()) is None


def test_modularity_none_without_roles_or_edges():
    assert partition_modularity(nx.path_graph(5)) is None  # no roles
    assert partition_modularity(roled(nx.empty_graph(7))) is None  # no edges


def test_structural_metrics_never_report_nan():
    """Any NaN that reaches a table would poison every average over it."""
    for topology in TOPOLOGIES:
        row = structural_metrics(study_graph(topology, 28))
        for key, value in row.items():
            if isinstance(value, float):
                assert math.isfinite(value), f"{topology}: {key} is {value}"


# ==========================================================================
# 3. modularity uses the PREDEFINED partition, not community detection
# ==========================================================================

def test_role_partition_is_the_seven_networks():
    graph = study_graph("er", 28)
    partition = role_partition(graph)
    assert len(partition) == len(YEO7)
    assert sum(len(block) for block in partition) == 28
    assert set().union(*partition) == set(graph.nodes())
    # the blocks are disjoint
    for i, left in enumerate(partition):
        for right in partition[i + 1 :]:
            assert not (left & right)


def test_role_partition_is_none_without_roles():
    assert role_partition(nx.path_graph(7)) is None


def test_partial_role_labelling_is_rejected():
    graph = nx.path_graph(7)
    nx.set_node_attributes(graph, {0: "visual"}, "role")
    with pytest.raises(GraphParameterError, match="carry no 'role'"):
        role_partition(graph)


def test_unknown_role_is_rejected():
    graph = nx.path_graph(7)
    nx.set_node_attributes(graph, dict.fromkeys(range(7), "not_a_network"), "role")
    with pytest.raises(GraphParameterError, match="unknown role"):
        role_partition(graph)


def test_modularity_matches_networkx_called_directly():
    """Independent recomputation, including the stated resolution."""
    for topology in TOPOLOGIES:
        graph = study_graph(topology, 56)
        expected = nx.community.modularity(
            graph, role_partition(graph), resolution=MODULARITY_RESOLUTION
        )
        assert partition_modularity(graph) == pytest.approx(expected)


def test_block_topologies_score_far_higher_than_non_block_ones():
    """The predefined partition IS the planted blocks for the block models.

    This is the measurement that makes the role-placement confound visible,
    and it is asserted rather than merely described. N = 28 and above, because
    N = 7 has one agent per network and N = 14 is capacity-capped (Week 2).
    """
    for n in (28, 56, 112):
        block = [
            partition_modularity(study_graph(t, n)) for t in ("modular", "yeo_sbm")
        ]
        non_block = [
            partition_modularity(study_graph(t, n)) for t in ("er", "ws", "ba")
        ]
        assert min(block) > max(non_block), f"N={n}: {block} vs {non_block}"
        # non-block roles are a random labelling, so Q sits at or below 0
        assert max(non_block) < 0.05


def test_modularity_is_in_the_theoretical_range():
    """Q lies in [-1/2, 1) for any partition of any graph."""
    for topology in TOPOLOGIES:
        for n in POPULATION_SIZES:
            value = partition_modularity(study_graph(topology, n))
            assert value is not None
            assert -0.5 <= value < 1.0


# ==========================================================================
# 4. role balance and input-readout distance
# ==========================================================================

@pytest.mark.parametrize("topology", TOPOLOGIES)
@pytest.mark.parametrize("n", POPULATION_SIZES)
def test_role_balance_is_exactly_n_over_seven(topology, n):
    balance = role_balance(study_graph(topology, n))
    assert balance["roles_assigned"] is True
    assert balance["roles_balanced"] is True
    assert balance["roles_per_network"] == n // 7
    assert balance["n_roles_present"] == 7
    assert set(balance["role_counts"].values()) == {n // 7}


def test_role_balance_without_roles():
    balance = role_balance(nx.path_graph(5))
    assert balance["roles_assigned"] is False
    assert balance["roles_balanced"] is None
    assert balance["n_roles_present"] == 0


def test_role_balance_detects_imbalance_after_an_attack():
    graph = study_graph("ba", 28)
    attacked = remove_nodes_by_count(graph, 14, "targeted").graph
    balance = role_balance(attacked)
    assert balance["roles_assigned"] is True
    assert balance["roles_balanced"] is False
    assert balance["roles_per_network"] is None


def test_input_readout_distance_on_a_known_graph():
    """A 7-node path: roles are dealt to positions, distances are checkable."""
    graph = roled(nx.path_graph(7))
    roles = nx.get_node_attributes(graph, "role")
    source = next(n for n, r in roles.items() if r == INPUT_ROLE)
    target = next(n for n, r in roles.items() if r == READOUT_ROLE)
    result = input_readout_distance(graph)
    # exactly one input and one readout node at N = 7, so all three agree
    assert result["input_readout_pairs"] == 1
    assert result["input_readout_mean_distance"] == pytest.approx(abs(source - target))
    assert result["input_readout_min_distance"] == abs(source - target)
    assert result["input_readout_max_distance"] == abs(source - target)
    assert result["input_readout_unreachable_pairs"] == 0


def test_input_readout_all_pairs_unreachable_gives_none_not_zero():
    """Two P7 components, seed 0: every input is in the component with no
    readout, so all four pairs are unreachable.

    The distances must be ``None`` -- a mean of zero, or of infinity, would
    both be wrong. The pair count still reports 4 so the reader can see that
    the measurement was attempted and nothing was reachable.
    """
    graph = roled(nx.disjoint_union(nx.path_graph(7), nx.path_graph(7)))
    roles = nx.get_node_attributes(graph, "role")
    inputs = {n for n, r in roles.items() if r == INPUT_ROLE}
    readouts = {n for n, r in roles.items() if r == READOUT_ROLE}
    # the premise of this test, asserted rather than assumed
    assert inputs <= set(range(7, 14))
    assert readouts <= set(range(7))

    result = input_readout_distance(graph)
    assert result["input_readout_pairs"] == 4
    assert result["input_readout_unreachable_pairs"] == 4
    assert result["input_readout_mean_distance"] is None
    assert result["input_readout_min_distance"] is None
    assert result["input_readout_max_distance"] is None


def test_input_readout_mixes_reachable_and_unreachable_pairs():
    """Roles placed by hand so exactly one of two pairs spans a component."""
    graph = nx.disjoint_union(nx.path_graph(7), nx.path_graph(7))
    # component 0 = nodes 0..6, component 1 = nodes 7..13.
    # one input and one readout inside component 0 (distance 3), plus a second
    # input stranded in component 1 (unreachable from that readout).
    roles = {0: INPUT_ROLE, 3: READOUT_ROLE, 7: INPUT_ROLE}
    for node in graph.nodes():
        roles.setdefault(node, "limbic")
    nx.set_node_attributes(graph, roles, "role")

    result = input_readout_distance(graph)
    assert result["input_readout_pairs"] == 2
    assert result["input_readout_unreachable_pairs"] == 1
    # only the reachable pair enters the statistics
    assert result["input_readout_mean_distance"] == pytest.approx(3.0)
    assert result["input_readout_min_distance"] == 3
    assert result["input_readout_max_distance"] == 3


def test_input_readout_distance_without_roles():
    result = input_readout_distance(nx.path_graph(5))
    assert result["input_readout_mean_distance"] is None
    assert result["input_readout_pairs"] is None


@pytest.mark.parametrize("topology", TOPOLOGIES)
def test_input_readout_distance_matches_an_independent_bfs(topology):
    graph = study_graph(topology, 56)
    roles = nx.get_node_attributes(graph, "role")
    sources = [n for n, r in roles.items() if r == INPUT_ROLE]
    targets = [n for n, r in roles.items() if r == READOUT_ROLE]
    expected = [
        nx.shortest_path_length(graph, s, t)
        for s in sources
        for t in targets
        if s != t
    ]
    result = input_readout_distance(graph)
    assert result["input_readout_mean_distance"] == pytest.approx(
        sum(expected) / len(expected)
    )
    assert result["input_readout_min_distance"] == min(expected)
    assert result["input_readout_max_distance"] == max(expected)


# ==========================================================================
# 5. structural metrics over the study grid
# ==========================================================================

@pytest.mark.parametrize("topology", TOPOLOGIES)
@pytest.mark.parametrize("n", POPULATION_SIZES)
def test_study_graph_invariants_hold(topology, n):
    row = structural_metrics(study_graph(topology, n))
    assert row["n_nodes"] == n
    assert row["n_edges"] == 2 * n            # exactly 2N edges
    assert row["mean_degree"] == pytest.approx(4.0)
    assert row["n_components"] == 1           # connected by generator contract
    assert row["largest_component_fraction"] == pytest.approx(1.0)
    assert row["roles_balanced"] is True
    assert 0.0 <= row["average_clustering"] <= 1.0
    assert 0.0 <= row["transitivity"] <= 1.0
    assert 0.0 <= row["global_efficiency"] <= 1.0
    assert 0.0 < row["edge_density"] <= 1.0
    assert row["degree_heterogeneity"] >= 1.0  # equality only for regular graphs
    assert row["diameter"] >= 1
    assert row["average_shortest_path_length"] >= 1.0


def test_structural_metrics_keys_are_exactly_the_declared_fields():
    row = structural_metrics(study_graph("er", 28))
    assert set(row) == set(STRUCTURAL_FIELDS)


def test_identity_is_read_from_the_graph_when_not_passed():
    """The contract's one-argument call must still identify the graph."""
    row = structural_metrics(study_graph("ws", 14, seed=3))
    assert row["topology"] == "ws"
    assert row["n"] == 14
    assert row["seed"] == 3
    assert row["graph_id"] == "ws_n014_seed3"
    assert row["graph_id"] == graph_id("ws", 14, 3)


def test_identity_is_none_for_a_bare_graph():
    row = structural_metrics(nx.path_graph(5))
    assert row["topology"] is None
    assert row["seed"] is None
    assert row["graph_id"] is None


def test_structural_metrics_are_deterministic():
    first = structural_metrics(study_graph("yeo_sbm", 56))
    second = structural_metrics(study_graph("yeo_sbm", 56))
    assert first == second


def test_structural_metrics_do_not_modify_the_graph():
    graph = study_graph("ba", 28)
    before = (
        sorted(graph.nodes()),
        sorted(tuple(sorted(e)) for e in graph.edges()),
        {v: dict(d) for v, d in graph.nodes(data=True)},
        dict(graph.graph),
    )
    structural_metrics(graph)
    positional_covariates(graph)
    assert (
        sorted(graph.nodes()),
        sorted(tuple(sorted(e)) for e in graph.edges()),
        {v: dict(d) for v, d in graph.nodes(data=True)},
        dict(graph.graph),
    ) == before


def test_degree_statistics_match_a_hand_computation():
    graph = study_graph("ba", 56)
    degrees = [d for _, d in graph.degree()]
    mean = sum(degrees) / len(degrees)
    variance = sum((d - mean) ** 2 for d in degrees) / len(degrees)  # ddof=0
    stats = degree_statistics(graph)
    assert stats["degree_mean"] == pytest.approx(mean)
    assert stats["degree_variance"] == pytest.approx(variance)
    assert stats["degree_std"] == pytest.approx(math.sqrt(variance))
    assert stats["degree_min"] == min(degrees)
    assert stats["degree_max"] == max(degrees)


def test_ba_has_the_broadest_degree_distribution():
    """Hub structure should dominate degree variance at the largest scale."""
    variances = {
        t: degree_statistics(study_graph(t, 112))["degree_variance"]
        for t in TOPOLOGIES
    }
    assert variances["ba"] > 2 * max(
        v for t, v in variances.items() if t != "ba"
    ), variances


def test_watts_strogatz_keeps_high_clustering_at_scale():
    """The small-world signature: clustering stays high while others decay."""
    clustering = {
        t: structural_metrics(study_graph(t, 112))["average_clustering"]
        for t in TOPOLOGIES
    }
    assert clustering["ws"] > 0.3
    for other in ("er", "modular", "yeo_sbm"):
        assert clustering[other] < 0.1, clustering


# ==========================================================================
# 6. positional covariates
# ==========================================================================

def test_positional_rows_are_one_per_node_in_id_order():
    graph = study_graph("er", 28)
    rows = positional_covariates(graph)
    assert len(rows) == 28
    assert [r["node_id"] for r in rows] == sorted(graph.nodes())
    assert set(rows[0]) == set(POSITIONAL_FIELDS)


def test_positional_covariates_on_an_empty_graph():
    assert positional_covariates(nx.Graph()) == []


def test_degree_and_centrality_match_independent_computation():
    graph = study_graph("ba", 56)
    expected_betweenness = nx.betweenness_centrality(graph, normalized=True)
    expected_closeness = nx.closeness_centrality(graph, wf_improved=True)
    expected_core = nx.core_number(graph)
    expected_clustering = nx.clustering(graph)
    for row in positional_covariates(graph):
        node = row["node_id"]
        assert row["degree"] == graph.degree(node)
        assert row["degree_centrality"] == pytest.approx(graph.degree(node) / 55)
        assert row["betweenness_centrality"] == pytest.approx(
            expected_betweenness[node]
        )
        assert row["closeness_centrality"] == pytest.approx(expected_closeness[node])
        assert row["core_number"] == expected_core[node]
        assert row["clustering"] == pytest.approx(expected_clustering[node])


def test_centralities_are_in_their_mathematical_ranges():
    for topology in TOPOLOGIES:
        for row in positional_covariates(study_graph(topology, 56)):
            assert 0.0 <= row["betweenness_centrality"] <= 1.0
            assert 0.0 <= row["closeness_centrality"] <= 1.0
            assert 0.0 <= row["clustering"] <= 1.0
            assert 0.0 <= row["degree_centrality"] <= 1.0
            assert 0.0 < row["pagerank"] < 1.0
            assert row["core_number"] >= 0


def test_pagerank_sums_to_one_and_uses_the_documented_alpha():
    graph = study_graph("modular", 56)
    rows = positional_covariates(graph)
    assert sum(r["pagerank"] for r in rows) == pytest.approx(1.0, abs=1e-9)
    expected = nx.pagerank(graph, alpha=PAGERANK_ALPHA, tol=1e-12, max_iter=1000)
    for row in rows:
        assert row["pagerank"] == pytest.approx(expected[row["node_id"]])


def test_betweenness_of_a_path_graph_centre_is_the_known_value():
    """P5 centre lies on 4 of the 6 unordered pairs -> normalised 4/6."""
    rows = positional_covariates(nx.path_graph(5))
    centre = next(r for r in rows if r["node_id"] == 2)
    assert centre["betweenness_centrality"] == pytest.approx(4 / 6)
    ends = [r for r in rows if r["node_id"] in (0, 4)]
    assert all(r["betweenness_centrality"] == 0.0 for r in ends)


def test_star_centre_has_betweenness_one():
    rows = positional_covariates(nx.star_graph(6))
    centre = next(r for r in rows if r["node_id"] == 0)
    assert centre["betweenness_centrality"] == pytest.approx(1.0)


def test_core_number_of_a_complete_graph():
    rows = positional_covariates(nx.complete_graph(7))
    assert {r["core_number"] for r in rows} == {6}


def test_eigenvector_centrality_converges_on_every_study_graph():
    """Recorded honestly: if it ever fails, the status says so."""
    for topology in TOPOLOGIES:
        for n in POPULATION_SIZES:
            graph = study_graph(topology, n)
            assert eigenvector_status(graph) == "converged"
            rows = positional_covariates(graph)
            assert all(r["eigenvector_centrality"] is not None for r in rows)
            assert all(r["eigenvector_centrality"] >= 0.0 for r in rows)


def test_eigenvector_centrality_is_none_when_undefined():
    """An edgeless graph has no principal eigenvector; None, not a number."""
    graph = roled(nx.empty_graph(7))
    assert eigenvector_status(graph) == "no_edges"
    rows = positional_covariates(graph)
    assert all(r["eigenvector_centrality"] is None for r in rows)
    assert eigenvector_status(nx.Graph()) == "empty_graph"


# ==========================================================================
# 7. network-relative covariates
# ==========================================================================

@pytest.mark.parametrize("topology", TOPOLOGIES)
def test_within_plus_between_degree_equals_degree(topology):
    for row in positional_covariates(study_graph(topology, 56)):
        assert row["within_network_degree"] + row["between_network_degree"] == (
            row["degree"]
        )


@pytest.mark.parametrize("topology", TOPOLOGIES)
def test_degree_fractions_sum_to_one(topology):
    for row in positional_covariates(study_graph(topology, 56)):
        if row["degree"] == 0:
            assert row["within_network_degree_fraction"] is None
            assert row["between_network_degree_fraction"] is None
        else:
            total = (
                row["within_network_degree_fraction"]
                + row["between_network_degree_fraction"]
            )
            assert total == pytest.approx(1.0)


def test_degree_zero_fractions_are_none_not_zero():
    """0/0 is not a fraction. 0.0 would read as 'no within-network links'."""
    graph = nx.path_graph(7)
    graph.add_node(7)  # isolated; N=8 so roles are not balanced -> assign manually
    nx.set_node_attributes(
        graph, {i: YEO7[i % 7] for i in graph.nodes()}, "role"
    )
    isolated = next(r for r in positional_covariates(graph) if r["node_id"] == 7)
    assert isolated["degree"] == 0
    assert isolated["within_network_degree_fraction"] is None
    assert isolated["between_network_degree_fraction"] is None
    assert isolated["participation_coefficient"] == 0.0  # documented convention


def test_participation_coefficient_matches_the_formula():
    """P_i = 1 - sum_s (k_is/k_i)^2, recomputed here from the neighbours."""
    for topology in TOPOLOGIES:
        graph = study_graph(topology, 56)
        roles = nx.get_node_attributes(graph, "role")
        for row in positional_covariates(graph):
            node = row["node_id"]
            degree = graph.degree(node)
            if degree == 0:
                assert row["participation_coefficient"] == 0.0
                continue
            counts: dict[str, int] = {}
            for neighbour in graph.neighbors(node):
                key = roles[neighbour]
                counts[key] = counts.get(key, 0) + 1
            expected = 1.0 - sum((c / degree) ** 2 for c in counts.values())
            assert row["participation_coefficient"] == pytest.approx(expected)


@pytest.mark.parametrize("topology", TOPOLOGIES)
def test_participation_coefficient_is_in_its_valid_range(topology):
    """With 7 networks, P lies in [0, 1 - 1/7] = [0, 6/7]."""
    limit = 1.0 - 1.0 / len(YEO7)
    for row in positional_covariates(study_graph(topology, 112)):
        assert 0.0 <= row["participation_coefficient"] <= limit + 1e-12


def test_participation_is_zero_when_all_neighbours_share_the_role():
    """A node whose links all stay inside its own network participates in none."""
    graph = nx.star_graph(6)
    nx.set_node_attributes(graph, dict.fromkeys(graph.nodes(), "visual"), "role")
    rows = positional_covariates(graph)
    assert all(r["participation_coefficient"] == pytest.approx(0.0) for r in rows)


def test_participation_is_maximal_for_evenly_spread_neighbours():
    """A degree-7 node with one neighbour in each network reaches 6/7 exactly."""
    graph = nx.star_graph(7)  # node 0 plus 7 leaves
    roles = {0: "visual"}
    roles.update({i + 1: YEO7[i] for i in range(7)})
    nx.set_node_attributes(graph, roles, "role")
    centre = next(r for r in positional_covariates(graph) if r["node_id"] == 0)
    assert centre["degree"] == 7
    assert centre["n_neighbor_networks"] == 7
    assert centre["participation_coefficient"] == pytest.approx(1.0 - 1.0 / 7)


def test_neighbor_network_count_is_bounded_by_seven():
    for topology in TOPOLOGIES:
        for row in positional_covariates(study_graph(topology, 112)):
            assert 0 <= row["n_neighbor_networks"] <= len(YEO7)
            assert row["n_neighbor_networks"] <= row["degree"]


def test_within_module_z_matches_the_formula():
    """z = (k_in - mean_module) / std_module, population std, recomputed here."""
    graph = study_graph("modular", 56)
    roles = nx.get_node_attributes(graph, "role")
    rows = positional_covariates(graph)
    within = {r["node_id"]: r["within_network_degree"] for r in rows}

    by_role: dict[str, list[int]] = {}
    for node, role in roles.items():
        by_role.setdefault(role, []).append(within[node])

    for row in rows:
        values = by_role[roles[row["node_id"]]]
        mean = sum(values) / len(values)
        var = sum((v - mean) ** 2 for v in values) / len(values)
        std = math.sqrt(var)
        expected = 0.0 if std == 0 else (within[row["node_id"]] - mean) / std
        assert row["within_module_z"] == pytest.approx(expected)


def test_within_module_z_is_zero_for_a_zero_variance_module():
    """At N = 7 every network holds one agent, so every z is exactly 0."""
    for topology in TOPOLOGIES:
        rows = positional_covariates(study_graph(topology, 7))
        assert all(r["within_module_z"] == 0.0 for r in rows), topology


def test_within_module_z_sums_to_zero_within_each_module():
    """A property of any z-score: deviations from the mean cancel."""
    graph = study_graph("yeo_sbm", 112)
    roles = nx.get_node_attributes(graph, "role")
    totals: dict[str, float] = {}
    for row in positional_covariates(graph):
        role = roles[row["node_id"]]
        totals[role] = totals.get(role, 0.0) + row["within_module_z"]
    for role, total in totals.items():
        assert total == pytest.approx(0.0, abs=1e-9), role


def test_within_module_z_is_none_without_roles():
    rows = positional_covariates(nx.path_graph(5))
    assert all(r["within_module_z"] is None for r in rows)
    assert all(r["role"] is None for r in rows)


def test_block_topologies_expose_their_community_attribute():
    for topology in ("modular", "yeo_sbm"):
        rows = positional_covariates(study_graph(topology, 56))
        assert all(r["community"] is not None for r in rows)
        # role is exactly YEO7[community] for the block models
        assert all(r["role"] == YEO7[r["community"]] for r in rows)
    for topology in ("er", "ws", "ba"):
        rows = positional_covariates(study_graph(topology, 56))
        assert all(r["community"] is None for r in rows)


def test_block_topologies_have_higher_within_network_degree():
    """The planted partition really does concentrate links inside networks."""
    means = {}
    for topology in TOPOLOGIES:
        rows = positional_covariates(study_graph(topology, 112))
        means[topology] = sum(r["within_network_degree"] for r in rows) / len(rows)
    assert min(means["modular"], means["yeo_sbm"]) > max(
        means["er"], means["ws"], means["ba"]
    ), means


def test_positional_covariates_are_deterministic():
    first = positional_covariates(study_graph("ws", 56))
    second = positional_covariates(study_graph("ws", 56))
    assert first == second


def test_positional_covariates_work_on_an_attacked_graph():
    """Disconnection must not break the node-level table."""
    graph = study_graph("ba", 112)
    attacked = remove_nodes_by_count(graph, 33, "targeted").graph
    rows = positional_covariates(attacked)
    assert len(rows) == attacked.number_of_nodes()
    assert all(r["node_id"] in attacked for r in rows)
    assert all(0.0 <= r["participation_coefficient"] <= 1.0 for r in rows)
