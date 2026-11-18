"""Tests for the shortest-path / round-budget analysis (owner: Mihir, Track A).

The diameter numbers themselves are checked against NetworkX computed
independently in the test, and the three Week 1 values already recorded in
``docs/track_a.md`` §10 (ER 7, BA 5, WS 10 at N = 112, seed 0) are pinned as
regression anchors -- if a generator change moves them, that is a finding and
this file says so out loud.
"""

from __future__ import annotations

import math
import statistics

import networkx as nx
import pytest

from neuromesh.graphs import (
    DEFAULT_ROUND_BUDGETS,
    EXPERIMENT_SEEDS,
    STD_CONVENTION,
    TOPOLOGIES,
    GraphParameterError,
    aggregate_path_metrics,
    diameter_table,
    generate_graph,
    measure_graph,
    path_metrics,
    remove_nodes_by_count,
    render_aggregate_table,
    render_measurement_table,
    round_budget_assessment,
)

#: Measured once and shared: 25 N=112 graphs is the costly part of this module.
_TABLE_CACHE: list[dict] = []


def table() -> list[dict]:
    if not _TABLE_CACHE:
        _TABLE_CACHE.extend(diameter_table(112))
    return _TABLE_CACHE


# ==========================================================================
# path_metrics on known graphs
# ==========================================================================

def test_path_length_on_a_path_graph():
    """P5: diameter 4, radius 2, average path length 2.0 exactly."""
    metrics = path_metrics(nx.path_graph(5))
    assert metrics["connected"] is True
    assert metrics["n_components"] == 1
    assert metrics["diameter"] == 4
    assert metrics["radius"] == 2
    assert metrics["average_shortest_path_length"] == pytest.approx(2.0)
    assert metrics["max_finite_distance"] == 4


def test_path_length_on_a_complete_graph():
    metrics = path_metrics(nx.complete_graph(7))
    assert metrics["diameter"] == 1
    assert metrics["average_shortest_path_length"] == pytest.approx(1.0)
    assert metrics["mean_degree"] == pytest.approx(6.0)


def test_cycle_graph_diameter():
    metrics = path_metrics(nx.cycle_graph(10))
    assert metrics["diameter"] == 5


def test_single_node_graph():
    metrics = path_metrics(nx.empty_graph(1))
    assert metrics["connected"] is True
    assert metrics["diameter"] == 0
    assert metrics["average_shortest_path_length"] == 0.0


def test_empty_graph_reports_nothing_rather_than_guessing():
    metrics = path_metrics(nx.Graph())
    assert metrics["n_nodes"] == 0
    assert metrics["mean_degree"] is None
    assert metrics["diameter"] is None
    assert metrics["connected"] is False


def test_disconnected_graph_has_no_finite_diameter():
    """Two P3 components: diameter is infinite, the widest component's is 2."""
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (1, 2), (10, 11), (11, 12)])
    metrics = path_metrics(graph)

    assert metrics["connected"] is False
    assert metrics["n_components"] == 2
    assert metrics["diameter"] is None, "a finite diameter must never be invented"
    assert metrics["radius"] is None
    assert metrics["average_shortest_path_length"] is None
    assert metrics["max_finite_distance"] == 2
    assert metrics["average_shortest_path_length_largest_component"] == pytest.approx(
        4 / 3
    )


def test_widest_component_is_not_necessarily_the_largest():
    """``max_finite_distance`` is over *all* components, not just the biggest."""
    graph = nx.Graph()
    graph.add_edges_from(nx.path_graph(5).edges())  # 5 nodes, diameter 4
    graph.add_edges_from([(u + 100, v + 100) for u, v in nx.star_graph(7).edges()])
    metrics = path_metrics(graph)
    assert metrics["largest_component_size"] == 8  # the star
    assert metrics["max_finite_distance"] == 4  # the path


def test_isolated_nodes_make_the_graph_disconnected():
    graph = nx.path_graph(4)
    graph.add_node(99)
    metrics = path_metrics(graph)
    assert metrics["n_components"] == 2
    assert metrics["diameter"] is None
    assert metrics["max_finite_distance"] == 3


def test_attacked_graph_metrics_do_not_raise():
    graph = generate_graph("ba", 112, seed=0)
    result = remove_nodes_by_count(graph, 33, "targeted")
    metrics = path_metrics(result.graph)
    assert metrics["n_components"] == result.n_components
    assert metrics["largest_component_size"] == result.largest_component_size
    if not metrics["connected"]:
        assert metrics["diameter"] is None
        assert metrics["max_finite_distance"] is not None


def test_non_graph_and_directed_inputs_are_rejected():
    with pytest.raises(GraphParameterError, match="expected a networkx.Graph"):
        path_metrics([(0, 1)])
    with pytest.raises(GraphParameterError, match="undirected"):
        path_metrics(nx.DiGraph([(0, 1)]))


# ==========================================================================
# the N = 112 measurement table
# ==========================================================================

def test_table_has_exactly_twenty_five_rows():
    rows = table()
    assert len(rows) == 25
    assert len({(r["topology"], r["seed"]) for r in rows}) == 25
    assert {r["topology"] for r in rows} == set(TOPOLOGIES)
    assert {r["seed"] for r in rows} == set(EXPERIMENT_SEEDS)
    assert all(r["n"] == 112 for r in rows)


def test_every_generated_graph_satisfies_the_connectivity_contract():
    """All 25 originals must be connected; a breach is a finding, not a skip."""
    disconnected = [
        (r["topology"], r["seed"]) for r in table() if not r["connected"]
    ]
    assert not disconnected, f"generator contract breached at {disconnected}"


def test_every_generated_graph_is_matched_on_mean_degree():
    for row in table():
        assert row["n_nodes"] == 112
        assert row["n_edges"] == 224
        assert row["mean_degree"] == pytest.approx(4.0)


def test_diameters_match_networkx_computed_independently():
    for row in table():
        graph = generate_graph(row["topology"], 112, seed=row["seed"])
        assert row["diameter"] == nx.diameter(graph)
        assert row["average_shortest_path_length"] == pytest.approx(
            nx.average_shortest_path_length(graph)
        )


def test_week_one_documented_diameters_are_unchanged():
    """``docs/track_a.md`` §10 records ER 7, BA 5, WS 10 at N=112 seed 0.

    Pinned so a generator change that moves them cannot slip through silently.
    """
    by_key = {(r["topology"], r["seed"]): r for r in table()}
    assert by_key[("er", 0)]["diameter"] == 7
    assert by_key[("ba", 0)]["diameter"] == 5
    assert by_key[("ws", 0)]["diameter"] == 10
    assert by_key[("er", 0)]["average_shortest_path_length"] == pytest.approx(
        3.5072, abs=5e-4
    )
    assert by_key[("ba", 0)]["average_shortest_path_length"] == pytest.approx(
        2.9791, abs=5e-4
    )
    assert by_key[("ws", 0)]["average_shortest_path_length"] == pytest.approx(
        5.1860, abs=5e-4
    )


def test_committed_seed_zero_fixtures_give_the_same_diameters(fixture_dir):
    """The shipped graphs and the regenerated graphs must agree."""
    from neuromesh.graphs import load_graph

    by_key = {(r["topology"], r["seed"]): r for r in table()}
    for topology in TOPOLOGIES:
        graph, _ = load_graph(topology, 112, 0, fixture_dir=fixture_dir)
        measured = measure_graph(graph, topology=topology, n=112, seed=0, source="fixture")
        assert measured["diameter"] == by_key[(topology, 0)]["diameter"]
        assert measured["n_edges"] == by_key[(topology, 0)]["n_edges"]
        assert measured["average_shortest_path_length"] == pytest.approx(
            by_key[(topology, 0)]["average_shortest_path_length"]
        )


def test_table_is_deterministic():
    first = diameter_table(14)
    second = diameter_table(14)
    assert first == second


def test_round_check_flags_are_consistent_with_the_diameter():
    for row in table():
        for budget, flag in row["round_check"].items():
            assert flag is (row["diameter"] <= budget)


def test_round_check_is_none_for_a_disconnected_graph():
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (5, 6)])
    row = measure_graph(graph, topology="test", n=4, seed=0)
    assert row["diameter"] is None
    assert all(flag is None for flag in row["round_check"].values())
    assert "disconnected" in row["caveat"]


def test_connected_rows_carry_no_caveat():
    assert all(row["caveat"] == "" for row in table())


# ==========================================================================
# aggregates
# ==========================================================================

def test_aggregate_has_one_row_per_topology_in_canonical_order():
    aggregates = aggregate_path_metrics(table())
    assert len(aggregates) == 5
    assert [a["topology"] for a in aggregates] == list(TOPOLOGIES)
    assert all(a["n_conditions"] == 5 for a in aggregates)
    assert all(a["n_disconnected"] == 0 for a in aggregates)


def test_aggregate_statistics_match_statistics_module():
    rows = table()
    for entry in aggregate_path_metrics(rows):
        diameters = [
            r["diameter"] for r in rows if r["topology"] == entry["topology"]
        ]
        assert entry["min"] == min(diameters)
        assert entry["max"] == max(diameters)
        assert entry["mean"] == pytest.approx(statistics.fmean(diameters))
        assert entry["median"] == pytest.approx(statistics.median(diameters))
        assert entry["std"] == pytest.approx(statistics.stdev(diameters))
        assert entry["std_population"] == pytest.approx(statistics.pstdev(diameters))
        assert entry["std_convention"] == STD_CONVENTION


def test_std_convention_is_the_sample_standard_deviation():
    """Stated, and distinguishable from the population value."""
    assert "ddof=1" in STD_CONVENTION
    entry = next(
        a for a in aggregate_path_metrics(table()) if a["topology"] == "ws"
    )
    assert entry["std"] > entry["std_population"]
    assert entry["std"] == pytest.approx(entry["std_population"] * math.sqrt(5 / 4))


def test_aggregate_budget_counts_add_up():
    for entry in aggregate_path_metrics(table()):
        for budget in DEFAULT_ROUND_BUDGETS:
            assert (
                entry[f"n_within_{budget}"] + entry[f"n_above_{budget}"]
                == entry["n_connected"]
            )


def test_watts_strogatz_is_the_only_topology_exceeding_ten_rounds():
    """The headline Week 3 diameter finding, asserted."""
    aggregates = {a["topology"]: a for a in aggregate_path_metrics(table())}
    for topology in ("er", "ba", "modular", "yeo_sbm"):
        assert aggregates[topology]["n_above_10"] == 0
        assert aggregates[topology]["max"] <= 8
    assert aggregates["ws"]["n_above_10"] == 2
    assert aggregates["ws"]["n_above_12"] == 1
    assert aggregates["ws"]["max"] == 13


def test_aggregate_excludes_disconnected_rows_but_counts_them():
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (5, 6)])
    rows = [
        measure_graph(nx.path_graph(5), topology="t", n=5, seed=0),
        measure_graph(graph, topology="t", n=4, seed=1),
    ]
    entry = aggregate_path_metrics(rows)[0]
    assert entry["n_conditions"] == 2
    assert entry["n_connected"] == 1
    assert entry["n_disconnected"] == 1
    assert entry["min"] == entry["max"] == 4
    assert entry["std"] == 0.0


def test_aggregate_of_all_disconnected_rows_reports_none_not_zero():
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (5, 6)])
    entry = aggregate_path_metrics(
        [measure_graph(graph, topology="t", n=4, seed=0)]
    )[0]
    assert entry["n_connected"] == 0
    assert entry["min"] is None
    assert entry["mean"] is None
    assert entry["std"] is None


# ==========================================================================
# the round-budget assessment
# ==========================================================================

def test_assessment_counts_and_violations():
    assessment = round_budget_assessment(table())
    assert assessment["n_conditions"] == 25
    assert assessment["n_connected"] == 25
    assert assessment["n_disconnected"] == 0
    assert assessment["max_diameter"] == 13
    assert assessment["argmax"] == [{"topology": "ws", "n": 112, "seed": 4}]
    assert assessment["minimum_budget_covering_all_measured"] == 13


def test_ten_rounds_do_not_cover_every_measured_graph():
    entry = round_budget_assessment(table())["budgets"][10]
    assert entry["n_within"] == 23
    assert entry["n_above"] == 2
    assert entry["covers_all_measured"] is False
    assert {(v["topology"], v["seed"]) for v in entry["violations"]} == {
        ("ws", 1),
        ("ws", 4),
    }


def test_twelve_rounds_also_do_not_cover_every_measured_graph():
    """The decision the team has to make: 12 still leaves WS seed 4 short."""
    entry = round_budget_assessment(table())["budgets"][12]
    assert entry["n_within"] == 24
    assert entry["n_above"] == 1
    assert entry["covers_all_measured"] is False
    assert entry["violations"] == [
        {"topology": "ws", "n": 112, "seed": 4, "diameter": 13}
    ]


def test_thirteen_rounds_cover_every_measured_graph():
    entry = round_budget_assessment(table(), (13,))["budgets"][13]
    assert entry["n_within"] == 25
    assert entry["n_above"] == 0
    assert entry["covers_all_measured"] is True


def test_assessment_states_that_reachability_is_necessary_not_sufficient():
    text = round_budget_assessment(table())["interpretation"]
    assert "NOT a guarantee" in text
    assert "consensus" in text
    assert "necessary" in text.lower()


def test_assessment_does_not_treat_disconnection_as_a_budget_violation():
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (5, 6)])
    rows = [measure_graph(graph, topology="t", n=4, seed=0)]
    assessment = round_budget_assessment(rows)
    assert assessment["n_disconnected"] == 1
    assert assessment["budgets"][10]["violations"] == []
    assert assessment["budgets"][10]["covers_all_measured"] is False


# ==========================================================================
# rendering
# ==========================================================================

def test_measurement_table_renders_every_row():
    text = render_measurement_table(table())
    lines = text.splitlines()
    assert len(lines) == 25 + 2  # header + separator
    assert "Diameter" in lines[0]
    assert "<=10r" in lines[0] and "<=12r" in lines[0]
    for topology in TOPOLOGIES:
        assert text.count(f"| {topology}") >= 5


def test_aggregate_table_renders_five_rows_with_the_required_columns():
    text = render_aggregate_table(aggregate_path_metrics(table()))
    lines = text.splitlines()
    assert len(lines) == 5 + 2
    for column in ("Min", "Max", "Mean", "Median", "StdDev", "<=10", "<=12", ">10", ">12"):
        assert column in lines[0]


def test_disconnected_row_renders_as_infinite_not_as_a_number():
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (5, 6)])
    text = render_measurement_table([measure_graph(graph, topology="t", n=4, seed=0)])
    assert "inf" in text
    assert "n/a" in text
    assert "disconnected" in text


def test_smaller_scales_can_also_be_measured():
    rows = diameter_table(14)
    assert len(rows) == 25
    assert all(r["n"] == 14 for r in rows)
    assert all(r["connected"] for r in rows)
    assert all(r["diameter"] <= 10 for r in rows)


def test_empty_graph_row_carries_an_empty_graph_caveat():
    row = measure_graph(nx.Graph(), topology="t", n=0, seed=0)
    assert row["caveat"] == "empty graph"
    assert row["diameter"] is None
    assert all(flag is None for flag in row["round_check"].values())
