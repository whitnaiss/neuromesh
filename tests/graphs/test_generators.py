"""Structural tests for the graph generators.

Covers every topology at every population size, node and edge validity,
mean-degree tolerance, connectivity, seed reproducibility and invalid inputs.

Assertions are deliberately exact where the mathematics is exact. Where a model
genuinely cannot reach the target (standard BA at small N), the test asserts
the *analytically predicted* value rather than relaxing the bound -- so the
limitation is pinned down, not papered over.
"""

from __future__ import annotations

import math

import networkx as nx
import pytest

from neuromesh.graphs import (
    DEGREE_TOLERANCE,
    POPULATION_SIZES,
    TARGET_MEAN_DEGREE,
    TOPOLOGIES,
    GraphParameterError,
    GraphSpec,
    barabasi_albert,
    check_reproducible,
    degree_deviation,
    erdos_renyi,
    generate_from_spec,
    mean_degree,
    target_edge_count,
    validate_graph,
    watts_strogatz,
)

SEEDS = (0, 1, 2, 3, 4)

#: The default settings that produce the shipped fixtures.
CANONICAL_GENERATORS = {
    "er": erdos_renyi,
    "ws": watts_strogatz,
    "ba": barabasi_albert,
}


# --------------------------------------------------------------------------
# 1-4, 6: node/edge validity and connectivity, every topology x every size
# --------------------------------------------------------------------------

@pytest.mark.parametrize("topology", TOPOLOGIES)
@pytest.mark.parametrize("n", POPULATION_SIZES)
def test_graph_is_simple_undirected_with_contiguous_node_ids(topology, n):
    graph, meta = generate_from_spec(GraphSpec(topology=topology, n=n, seed=0))

    assert graph.number_of_nodes() == n
    assert set(graph.nodes()) == set(range(n)), "node IDs must be exactly 0..n-1"
    assert not graph.is_directed()
    assert not graph.is_multigraph()
    assert nx.number_of_selfloops(graph) == 0
    assert meta.n_nodes == n
    assert meta.n_edges == graph.number_of_edges()


@pytest.mark.parametrize("topology", TOPOLOGIES)
@pytest.mark.parametrize("n", POPULATION_SIZES)
def test_edge_list_has_no_duplicates(topology, n):
    graph, _ = generate_from_spec(GraphSpec(topology=topology, n=n, seed=0))
    edges = [tuple(sorted(e)) for e in graph.edges()]
    assert len(edges) == len(set(edges))


@pytest.mark.parametrize("topology", TOPOLOGIES)
@pytest.mark.parametrize("n", POPULATION_SIZES)
@pytest.mark.parametrize("seed", SEEDS)
def test_connected_under_default_settings(topology, n, seed):
    """Every shipped configuration must be connected at every seed.

    ER reaches this by rejection sampling from the connected ensemble (see the
    generators module docstring); WS and BA are connected without it.
    """
    graph, meta = generate_from_spec(GraphSpec(topology=topology, n=n, seed=seed))
    assert nx.is_connected(graph)
    assert meta.connected is True
    assert meta.n_components == 1


# --------------------------------------------------------------------------
# 5: mean degree 2E/N within +/-10%
# --------------------------------------------------------------------------

@pytest.mark.parametrize("topology", TOPOLOGIES)
@pytest.mark.parametrize("n", POPULATION_SIZES)
@pytest.mark.parametrize("seed", SEEDS)
def test_mean_degree_matches_target_exactly(topology, n, seed):
    """All three default generators hit mean degree 4 exactly, not just within 10%.

    ER uses G(n,m) with m = 2N, WS rewiring preserves the ring lattice's edge
    count, and degree-matched BA is seeded so that E = m*N. Asserting exact
    equality is therefore the correct (stronger) assertion here.
    """
    graph, meta = generate_from_spec(GraphSpec(topology=topology, n=n, seed=seed))

    expected_edges = target_edge_count(n, TARGET_MEAN_DEGREE)
    assert graph.number_of_edges() == expected_edges
    assert mean_degree(graph) == pytest.approx(TARGET_MEAN_DEGREE)
    assert abs(degree_deviation(graph, TARGET_MEAN_DEGREE)) <= DEGREE_TOLERANCE
    assert meta.degree_matched_exactly is True


@pytest.mark.parametrize("n", POPULATION_SIZES)
def test_mean_degree_formula_is_two_e_over_n(n):
    """Guard the definition itself against a refactor."""
    graph, _ = generate_from_spec(GraphSpec(topology="ws", n=n, seed=0))
    assert mean_degree(graph) == 2 * graph.number_of_edges() / graph.number_of_nodes()
    assert mean_degree(graph) == pytest.approx(
        sum(d for _, d in graph.degree()) / n
    )


# --------------------------------------------------------------------------
# 7: seed reproducibility
# --------------------------------------------------------------------------

@pytest.mark.parametrize("topology", TOPOLOGIES)
@pytest.mark.parametrize("n", POPULATION_SIZES)
def test_identical_seeds_give_identical_graphs(topology, n):
    check = check_reproducible(
        CANONICAL_GENERATORS[topology], n, 0, repeats=3
    )
    assert check.status == "PASS", check.detail


@pytest.mark.parametrize("topology", TOPOLOGIES)
@pytest.mark.parametrize("n", POPULATION_SIZES)
def test_different_seeds_give_different_graphs(topology, n):
    """Seeding must actually vary the draw, otherwise reproducibility is vacuous.

    At N = 7 the degree-matched BA graph is a clique-plus-two, which has very
    few distinct realisations, so this is only asserted from N = 14 -- the
    scale at which topology comparisons begin.
    """
    if n == 7:
        pytest.skip("N=7 is a boundary case with too few distinct realisations")

    edge_sets = set()
    for seed in SEEDS:
        graph, _ = generate_from_spec(GraphSpec(topology=topology, n=n, seed=seed))
        edge_sets.add(frozenset(tuple(sorted(e)) for e in graph.edges()))
    assert len(edge_sets) > 1, f"{topology} N={n} produced one graph for 5 seeds"


# --------------------------------------------------------------------------
# ER-specific behaviour
# --------------------------------------------------------------------------

@pytest.mark.parametrize("n", POPULATION_SIZES)
def test_er_gnm_edge_count_is_exactly_two_n(n):
    graph, meta = erdos_renyi(n, seed=0)
    assert meta.params["model"] == "gnm"
    assert meta.params["m"] == 2 * n
    assert graph.number_of_edges() == 2 * n


@pytest.mark.parametrize("n", POPULATION_SIZES)
def test_er_gnp_uses_probability_k_over_n_minus_one(n):
    _, meta = erdos_renyi(n, seed=0, model="gnp", ensure_connected=False)
    assert meta.params["model"] == "gnp"
    assert meta.params["p"] == pytest.approx(TARGET_MEAN_DEGREE / (n - 1))


def test_er_gnp_misses_the_tolerance_at_small_n():
    """Documents the reason G(n,p) is not the default.

    At N = 7 with p = 2/3 the edge count is Binomial(21, 2/3), and the
    acceptance window is only E in {13,14,15}. This asserts that at least one
    of the five study seeds lands outside the +/-10% window, which is exactly
    why the default is G(n,m).
    """
    deviations = []
    for seed in SEEDS:
        graph, _ = erdos_renyi(7, seed=seed, model="gnp", ensure_connected=False)
        deviations.append(abs(degree_deviation(graph, TARGET_MEAN_DEGREE)))
    assert max(deviations) > DEGREE_TOLERANCE


def test_er_rejection_sampling_is_recorded_and_sometimes_needed():
    """At N=112 and mean degree 4 the unconditioned graph is usually disconnected.

    ln(112) ~ 4.72 > 4, so connectivity requires more than one draw for most
    seeds. The attempt count must be recorded rather than hidden.
    """
    attempts = []
    for seed in SEEDS:
        _, meta = erdos_renyi(112, seed=seed)
        assert meta.connected is True
        assert meta.attempts >= 1
        attempts.append(meta.attempts)
    assert max(attempts) > 1, (
        "expected at least one seed at N=112 to need a redraw for connectivity"
    )


def test_er_without_ensure_connected_can_be_disconnected():
    """The unconditioned ensemble must remain reachable for honest comparison."""
    disconnected = 0
    for seed in range(20):
        graph, meta = erdos_renyi(112, seed=seed, ensure_connected=False)
        assert meta.attempts == 1
        if not nx.is_connected(graph):
            disconnected += 1
    assert disconnected > 0


# --------------------------------------------------------------------------
# WS-specific behaviour
# --------------------------------------------------------------------------

@pytest.mark.parametrize("n", POPULATION_SIZES)
@pytest.mark.parametrize("rewiring_prob", [0.0, 0.1, 0.5, 1.0])
def test_ws_rewiring_never_changes_the_edge_count(n, rewiring_prob):
    graph, meta = watts_strogatz(n, seed=0, rewiring_prob=rewiring_prob)
    assert graph.number_of_edges() == 2 * n
    assert mean_degree(graph) == pytest.approx(TARGET_MEAN_DEGREE)
    assert meta.params["rewiring_prob"] == rewiring_prob


@pytest.mark.parametrize("n", POPULATION_SIZES)
def test_ws_p_zero_is_the_ring_lattice(n):
    """With p=0 every node has degree exactly k -- a sanity anchor for the model."""
    graph, _ = watts_strogatz(n, seed=0, rewiring_prob=0.0)
    assert {d for _, d in graph.degree()} == {TARGET_MEAN_DEGREE}


def test_ws_records_the_rewiring_probability_used():
    """Generation parameters must be recorded, never changed silently."""
    _, meta = watts_strogatz(28, seed=0)
    assert meta.params["rewiring_prob"] == 0.1
    assert meta.params["k"] == TARGET_MEAN_DEGREE


# --------------------------------------------------------------------------
# BA-specific behaviour, including the documented small-N limitation
# --------------------------------------------------------------------------

@pytest.mark.parametrize("n", POPULATION_SIZES)
def test_ba_standard_matches_the_analytic_edge_count(n):
    """Standard BA yields exactly E = m(N-m); assert it, do not relax it."""
    graph, meta = barabasi_albert(n, seed=0, variant="standard")
    m = meta.params["m"]
    assert m == 2
    assert graph.number_of_edges() == m * (n - m)
    assert mean_degree(graph) == pytest.approx(TARGET_MEAN_DEGREE - 8.0 / n)


@pytest.mark.parametrize(
    "n, expected_deviation",
    [(7, -8 / 7 / 4), (14, -8 / 14 / 4), (28, -8 / 28 / 4), (56, -8 / 56 / 4), (112, -8 / 112 / 4)],
)
def test_ba_standard_small_n_deviation_is_the_predicted_one(n, expected_deviation):
    """Pin the known limitation exactly: -28.57% at N=7, -14.29% at N=14."""
    graph, _ = barabasi_albert(n, seed=0, variant="standard")
    assert degree_deviation(graph, TARGET_MEAN_DEGREE) == pytest.approx(
        expected_deviation
    )
    outside = abs(expected_deviation) > DEGREE_TOLERANCE
    assert outside == (n in (7, 14)), (
        "standard BA is expected to miss +/-10% at exactly N=7 and N=14"
    )


@pytest.mark.parametrize("n", POPULATION_SIZES)
def test_ba_degree_matched_hits_the_target_exactly(n):
    """The K_(2m+1) seed core makes E = m*N for every N."""
    graph, meta = barabasi_albert(n, seed=0, variant="degree_matched")
    m = meta.params["m"]
    assert meta.params["core_size"] == 2 * m + 1
    assert graph.number_of_edges() == m * n
    assert mean_degree(graph) == pytest.approx(2 * m)
    assert mean_degree(graph) == pytest.approx(TARGET_MEAN_DEGREE)


@pytest.mark.parametrize("n", POPULATION_SIZES)
def test_ba_is_connected_by_construction(n):
    for variant in ("standard", "degree_matched"):
        graph, _ = barabasi_albert(n, seed=0, variant=variant)
        assert nx.is_connected(graph)


@pytest.mark.parametrize("n", [28, 56, 112])
def test_ba_degree_matched_retains_hub_structure(n):
    """The seed clique must not destroy preferential attachment at useful N.

    A BA hub is far above the mean; a clique-only graph is not. At N >= 28 the
    degree-matched variant must still produce a max degree well above the mean
    and comparable to standard BA.
    """
    matched, _ = barabasi_albert(n, seed=0, variant="degree_matched")
    standard, _ = barabasi_albert(n, seed=0, variant="standard")
    max_matched = max(d for _, d in matched.degree())
    max_standard = max(d for _, d in standard.degree())
    assert max_matched >= 2 * TARGET_MEAN_DEGREE
    assert max_matched >= 0.5 * max_standard


def test_ba_degree_matched_rejects_n_below_the_seed_core():
    """n=5 with m=3 needs a K_7 core, which does not fit.

    (n=4 with the default m=2 is rejected earlier and separately, by the
    feasibility check: mean degree 4 is impossible on 4 nodes at all.)
    """
    with pytest.raises(GraphParameterError, match="seat its complete seed core"):
        barabasi_albert(5, seed=0, variant="degree_matched", m=3)


def test_ba_rejects_mean_degree_infeasible_for_n():
    with pytest.raises(GraphParameterError, match="infeasible for n=4"):
        barabasi_albert(4, seed=0, variant="degree_matched")


# --------------------------------------------------------------------------
# 8: invalid parameters
# --------------------------------------------------------------------------

@pytest.mark.parametrize("topology", TOPOLOGIES)
@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"n": 2}, "n must be >= 3"),
        ({"n": 0}, "n must be >= 3"),
        ({"n": -5}, "n must be >= 3"),
        ({"n": 14.5}, "n must be an integer"),
        ({"n": "14"}, "n must be an integer"),
        ({"n": None}, "n must be an integer"),
        ({"seed": -1}, "seed must be non-negative"),
        ({"seed": 1.5}, "seed must be an integer"),
        ({"seed": "abc"}, "seed must be an integer"),
        ({"target_mean_degree": 0}, "must be positive"),
        ({"target_mean_degree": -4}, "must be positive"),
        ({"target_mean_degree": float("nan")}, "must be positive and finite"),
        ({"target_mean_degree": 500}, "infeasible"),
    ],
)
def test_invalid_parameters_raise(topology, kwargs, match):
    generator = CANONICAL_GENERATORS[topology]
    call = {"n": 14, "seed": 0}
    call.update(kwargs)
    n = call.pop("n")
    seed = call.pop("seed")
    with pytest.raises(GraphParameterError, match=match):
        generator(n, seed, **call)


def test_unknown_topology_rejected():
    with pytest.raises(GraphParameterError, match="unknown topology"):
        GraphSpec(topology="modular", n=14)
    with pytest.raises(GraphParameterError, match="unknown topology"):
        GraphSpec(topology="yeo_sbm", n=14)


def test_unknown_generator_param_rejected():
    with pytest.raises(GraphParameterError, match="invalid params"):
        generate_from_spec(GraphSpec("er", n=14, params={"nonexistent_kwarg": 1}))


def test_generate_from_spec_rejects_non_spec():
    with pytest.raises(GraphParameterError, match="expected a GraphSpec"):
        generate_from_spec({"topology": "er", "n": 14})


def test_er_rejects_unknown_model():
    with pytest.raises(GraphParameterError, match="model must be"):
        erdos_renyi(14, seed=0, model="gnx")


def test_ba_rejects_unknown_variant():
    with pytest.raises(GraphParameterError, match="variant must be"):
        barabasi_albert(14, seed=0, variant="scalefree")


@pytest.mark.parametrize("bad_prob", [-0.1, 1.5, "0.1", None])
def test_ws_rejects_invalid_rewiring_probability(bad_prob):
    with pytest.raises(GraphParameterError, match="rewiring_prob"):
        watts_strogatz(14, seed=0, rewiring_prob=bad_prob)


def test_ws_rejects_odd_mean_degree():
    with pytest.raises(GraphParameterError, match="even mean degree"):
        watts_strogatz(14, seed=0, target_mean_degree=3)


def test_ws_rejects_k_not_less_than_n():
    with pytest.raises(GraphParameterError, match="infeasible|k < n"):
        watts_strogatz(4, seed=0, target_mean_degree=4)


def test_rejection_sampling_rejects_bad_max_attempts():
    with pytest.raises(GraphParameterError, match="max_attempts"):
        erdos_renyi(14, seed=0, max_attempts=0)


def test_mean_degree_undefined_for_empty_graph():
    with pytest.raises(GraphParameterError, match="empty graph"):
        mean_degree(nx.Graph())


# --------------------------------------------------------------------------
# 9: all five population sizes are supported end to end
# --------------------------------------------------------------------------

def test_all_fifteen_topology_size_combinations_validate():
    """The full matrix: 3 topologies x 5 sizes, all checks clean."""
    results = []
    for topology in TOPOLOGIES:
        for n in POPULATION_SIZES:
            graph, _ = generate_from_spec(GraphSpec(topology=topology, n=n, seed=0))
            result = validate_graph(graph, n=n, topology=topology, seed=0)
            results.append(result)
            assert result.ok, result.summary()
            assert not result.exceptions, result.summary()
    assert len(results) == 15


def test_population_sizes_are_the_studys_five():
    assert POPULATION_SIZES == (7, 14, 28, 56, 112)
    assert TARGET_MEAN_DEGREE == 4
    assert math.isclose(DEGREE_TOLERANCE, 0.10)


# --------------------------------------------------------------------------
# BA attachment parameter vs requested mean degree
# --------------------------------------------------------------------------

@pytest.mark.parametrize("variant", ["degree_matched", "standard"])
def test_ba_rejects_m_that_contradicts_the_requested_mean_degree(variant):
    """m governs BA's mean degree (2m), so an m that disagrees with k is a bug.

    Previously ``barabasi_albert(28, m=3, target_mean_degree=4)`` returned a
    mean-degree-6 graph, with the mismatch visible only afterwards through
    ``metadata.degree_matched_exactly``. A caller asking for two different mean
    degrees in one call now gets told.
    """
    with pytest.raises(GraphParameterError, match="inconsistent Barabasi-Albert"):
        barabasi_albert(28, seed=0, variant=variant, m=3, target_mean_degree=4)


def test_ba_accepts_m_consistent_with_a_non_default_mean_degree():
    """The check must not block a legitimate non-default request."""
    graph, meta = barabasi_albert(28, seed=0, m=3, target_mean_degree=6)
    assert meta.params["m"] == 3
    assert graph.number_of_edges() == 3 * 28
    assert mean_degree(graph) == pytest.approx(6.0)
    assert meta.degree_matched_exactly is True


def test_ba_rejects_a_mean_degree_no_integer_m_can_reach():
    """Each arrival contributes an integer m edges, so 2m is always even."""
    with pytest.raises(GraphParameterError, match="only even mean degrees"):
        barabasi_albert(28, seed=0, target_mean_degree=5)


def test_ba_seed_core_error_still_wins_over_the_consistency_error():
    """Ordering matters: the more specific diagnosis must be reported first."""
    with pytest.raises(GraphParameterError, match="seat its complete seed core"):
        barabasi_albert(5, seed=0, variant="degree_matched", m=3)


def test_ba_default_m_remains_two_for_the_studys_mean_degree():
    """The whole experiment runs at k=4; pin the m it implies."""
    for n in POPULATION_SIZES:
        _, meta = barabasi_albert(n, seed=0)
        assert meta.params["m"] == 2
        assert 2 * meta.params["m"] == TARGET_MEAN_DEGREE
