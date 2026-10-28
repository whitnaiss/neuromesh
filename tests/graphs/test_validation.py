"""Tests for the validation layer itself.

A validator that always returns PASS would make every other test meaningless,
so these tests feed it deliberately broken graphs and assert that it catches
them.
"""

from __future__ import annotations

import networkx as nx
import pytest

from neuromesh.graphs import (
    TARGET_MEAN_DEGREE,
    GraphParameterError,
    check_reproducible,
    degree_deviation,
    erdos_renyi,
    validate_graph,
)


def _good_graph(n: int = 14):
    graph, _ = erdos_renyi(n, seed=0)
    return graph


def test_clean_graph_passes_every_check():
    result = validate_graph(_good_graph(), n=14, topology="er", seed=0)
    assert result.ok
    assert not result.exceptions
    assert {c.status for c in result.checks} == {"PASS"}
    assert len(result.checks) == 6


def test_wrong_node_count_is_caught():
    graph = _good_graph(14)
    graph.add_node(99)
    result = validate_graph(graph, n=14)
    assert not result.ok
    assert result.status_of("node_count") == "FAIL"


def test_non_contiguous_node_ids_are_caught():
    """Relabelled nodes must fail even when the count is right."""
    graph = nx.relabel_nodes(_good_graph(14), {0: "agent_0"})
    result = validate_graph(graph, n=14)
    assert result.status_of("node_count") == "FAIL"
    assert graph.number_of_nodes() == 14


def test_self_loop_is_caught():
    graph = _good_graph(14)
    graph.add_edge(3, 3)
    result = validate_graph(graph, n=14)
    assert result.status_of("no_self_loops") == "FAIL"
    assert not result.ok


def test_directed_graph_is_caught():
    graph = _good_graph(14).to_directed()
    result = validate_graph(graph, n=14)
    assert result.status_of("undirected") == "FAIL"


def test_multigraph_is_caught():
    graph = nx.MultiGraph()
    graph.add_nodes_from(range(14))
    graph.add_edges_from([(i, (i + 1) % 14) for i in range(14)] * 2)
    result = validate_graph(graph, n=14)
    assert result.status_of("no_parallel_edges") == "FAIL"


def test_disconnected_graph_is_caught():
    graph = nx.Graph()
    graph.add_nodes_from(range(14))
    nx.add_cycle(graph, range(7))
    nx.add_cycle(graph, range(7, 14))
    result = validate_graph(graph, n=14)
    assert result.status_of("connected") == "FAIL"
    assert result.connected is False


def test_disconnected_is_an_exception_when_connectivity_not_required():
    """Two disjoint 4-regular components: mean degree is exactly 4, so
    connectivity is the only property in question."""
    component = nx.circulant_graph(7, [1, 2])  # 4-regular on 7 nodes, 14 edges
    graph = nx.disjoint_union(component, component)
    assert graph.number_of_nodes() == 14 and graph.number_of_edges() == 28

    result = validate_graph(graph, n=14, require_connected=False)
    assert result.status_of("mean_degree_tolerance") == "PASS"
    assert result.status_of("connected") == "EXCEPTION"
    assert result.ok, "a declared exception is not a defect"
    assert result.exceptions


@pytest.mark.parametrize("n_edges, expected", [(14, "FAIL"), (28, "PASS"), (40, "FAIL")])
def test_mean_degree_tolerance_boundary(n_edges, expected):
    """14 edges at N=14 is k=2 (-50%); 28 is k=4; 40 is k=5.71 (+42.9%)."""
    graph = nx.gnm_random_graph(14, n_edges, seed=0)
    graph.add_nodes_from(range(14))
    result = validate_graph(graph, n=14, require_connected=False)
    assert result.status_of("mean_degree_tolerance") == expected


def test_mean_degree_exception_requires_an_explicit_reason():
    """A tolerance miss only becomes an EXCEPTION when a reason is supplied."""
    graph = nx.gnm_random_graph(14, 14, seed=0)
    graph.add_nodes_from(range(14))

    without = validate_graph(graph, n=14, require_connected=False)
    assert without.status_of("mean_degree_tolerance") == "FAIL"

    with_reason = validate_graph(
        graph,
        n=14,
        require_connected=False,
        degree_exception_reason="model constraint under test",
    )
    check = next(
        c for c in with_reason.checks if c.name == "mean_degree_tolerance"
    )
    assert check.status == "EXCEPTION"
    assert "model constraint under test" in check.detail


def test_tolerance_detail_shows_the_2e_over_n_arithmetic():
    graph, _ = erdos_renyi(28, seed=0)
    result = validate_graph(graph, n=28)
    detail = next(
        c.detail for c in result.checks if c.name == "mean_degree_tolerance"
    )
    assert "2E/N" in detail and "2*56/28" in detail


def test_degree_deviation_sign_and_magnitude():
    graph = nx.gnm_random_graph(14, 14, seed=0)
    assert degree_deviation(graph, TARGET_MEAN_DEGREE) == pytest.approx(-0.5)
    graph = nx.gnm_random_graph(14, 42, seed=0)
    assert degree_deviation(graph, TARGET_MEAN_DEGREE) == pytest.approx(0.5)


def test_validator_rejects_bad_input():
    with pytest.raises(GraphParameterError, match="networkx.Graph"):
        validate_graph("not a graph", n=14)
    with pytest.raises(GraphParameterError, match="degree_tolerance"):
        validate_graph(_good_graph(), n=14, degree_tolerance=-0.1)


def test_result_serialises_to_json_friendly_dict():
    result = validate_graph(_good_graph(), n=14, topology="er", seed=0)
    payload = result.to_dict()
    assert payload["ok"] is True
    assert payload["n_nodes"] == 14
    assert {c["name"] for c in payload["checks"]} == {
        "node_count",
        "undirected",
        "no_self_loops",
        "no_parallel_edges",
        "mean_degree_tolerance",
        "connected",
    }


def test_status_of_unknown_check_raises():
    result = validate_graph(_good_graph(), n=14)
    with pytest.raises(KeyError):
        result.status_of("not_a_check")


# --------------------------------------------------------------------------
# the reproducibility checker must be able to fail
# --------------------------------------------------------------------------

def test_check_reproducible_detects_a_nondeterministic_generator():
    counter = {"calls": 0}

    def unseeded(n):
        counter["calls"] += 1
        return nx.gnm_random_graph(n, 2 * n, seed=counter["calls"])

    check = check_reproducible(unseeded, 14, repeats=3)
    assert check.status == "FAIL"


def test_check_reproducible_detects_relabelling():
    """Same edges, different node labels, must still be flagged."""
    counter = {"calls": 0}

    def relabeller(n):
        counter["calls"] += 1
        graph = nx.cycle_graph(n)
        if counter["calls"] > 1:
            graph = nx.relabel_nodes(graph, {0: n, n - 1: 0, n: n - 1})
        return graph

    assert check_reproducible(relabeller, 14, repeats=2).status == "FAIL"


def test_check_reproducible_passes_for_a_seeded_generator():
    check = check_reproducible(erdos_renyi, 28, 0, repeats=4)
    assert check.status == "PASS"


def test_check_reproducible_requires_at_least_two_repeats():
    with pytest.raises(GraphParameterError, match="repeats must be >= 2"):
        check_reproducible(erdos_renyi, 14, 0, repeats=1)


# --------------------------------------------------------------------------
# a declared exception is not a defect, but it is not sweep-eligible either
# --------------------------------------------------------------------------

def test_clean_graph_is_both_ok_and_sweep_eligible():
    result = validate_graph(_good_graph(), n=14, topology="er", seed=0)
    assert result.ok
    assert result.sweep_eligible
    assert "[not sweep-eligible]" not in result.summary()


def test_declared_exception_is_ok_but_not_sweep_eligible():
    """The distinction that keeps the matched-degree design honest.

    Standard BA at N = 7 misses mean degree 4 by -28.6%. That is a documented
    property of the model, so it is not a defect -- but the graph is *not*
    matched on the study's controlled variable, so it must not silently enter
    the main sweep on the strength of ``ok`` alone.
    """
    graph = nx.gnm_random_graph(14, 14, seed=0)
    graph.add_nodes_from(range(14))
    result = validate_graph(
        graph,
        n=14,
        require_connected=False,
        degree_exception_reason="standard BA cannot reach k=4 at small N",
    )
    assert result.status_of("mean_degree_tolerance") == "EXCEPTION"
    assert result.ok, "a declared exception is not a defect"
    assert not result.sweep_eligible, "but it is not matched on mean degree"
    assert "[not sweep-eligible]" in result.summary()


def test_failure_is_neither_ok_nor_sweep_eligible():
    graph = _good_graph(14)
    graph.add_edge(3, 3)
    result = validate_graph(graph, n=14)
    assert not result.ok
    assert not result.sweep_eligible


def test_sweep_eligibility_is_serialised():
    """Track D reads the report as JSON; the flag must travel with it."""
    clean = validate_graph(_good_graph(), n=14, topology="er", seed=0).to_dict()
    assert clean["ok"] is True
    assert clean["sweep_eligible"] is True

    component = nx.circulant_graph(7, [1, 2])
    excepted = validate_graph(
        nx.disjoint_union(component, component), n=14, require_connected=False
    ).to_dict()
    assert excepted["ok"] is True
    assert excepted["sweep_eligible"] is False
