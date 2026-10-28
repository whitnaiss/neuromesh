"""Tests for the shared graph interface the rest of the project calls.

``tests/graphs/test_generators.py`` tests the Track A generator layer
(``(graph, GraphMetadata)``); this module tests the thin adapter published in
``contracts.py``:

    neuromesh.graphs.generate_graph(topology, n, mean_degree, seed) -> nx.Graph

so a change that breaks Armman's, Yutong's or Andrew's call site fails here,
in Mihir's own test directory, rather than in theirs.
"""

from __future__ import annotations

import networkx as nx
import pytest

from neuromesh.contracts import DEFAULT_MEAN_DEGREE, YEO7, Topology
from neuromesh.graphs import (
    DEGREE_TOLERANCE,
    PLANNED_TOPOLOGIES,
    POPULATION_SIZES,
    TARGET_MEAN_DEGREE,
    TOPOLOGIES,
    GraphParameterError,
    GraphSpec,
    assign_roles,
    generate_from_spec,
    generate_graph,
    mean_degree,
    remove_nodes,
    structural_metrics,
    target_edge_count,
    validate_graph,
)

#: The seeds the experiment configs use (configs/full.yaml and robustness.yaml).
EXPERIMENT_SEEDS = (0, 1, 2, 3, 4)


# --------------------------------------------------------------------------
# the return type is a bare graph, not a (graph, metadata) tuple
# --------------------------------------------------------------------------

@pytest.mark.parametrize("topology", TOPOLOGIES)
def test_returns_a_bare_graph_not_a_tuple(topology):
    """The single most likely integration break: the layers differ here.

    ``generate_from_spec`` returns ``(graph, metadata)``; the shared interface
    must return the graph alone.
    """
    result = generate_graph(topology, 14)
    assert isinstance(result, nx.Graph)
    assert not isinstance(result, tuple)
    # A tuple would also satisfy `len(result) == 2`; a graph must not.
    assert result.number_of_nodes() == 14


@pytest.mark.parametrize("topology", TOPOLOGIES)
def test_default_arguments_match_the_contract(topology):
    """mean_degree defaults to contracts.DEFAULT_MEAN_DEGREE and seed to 0."""
    assert DEFAULT_MEAN_DEGREE == TARGET_MEAN_DEGREE
    explicit = generate_graph(topology, 28, DEFAULT_MEAN_DEGREE, 0)
    implicit = generate_graph(topology, 28)
    assert set(implicit.edges()) == set(explicit.edges())


def test_keyword_call_matches_the_contract_signature():
    """Andrew's runner may call this by keyword; the names are part of the API."""
    graph = generate_graph(topology="ws", n=28, mean_degree=4.0, seed=2)
    assert graph.graph["topology"] == "ws"
    assert graph.graph["n"] == 28
    assert graph.graph["seed"] == 2


# --------------------------------------------------------------------------
# G.graph metadata required by contracts.py
# --------------------------------------------------------------------------

@pytest.mark.parametrize("topology", TOPOLOGIES)
@pytest.mark.parametrize("n", POPULATION_SIZES)
def test_graph_attributes_required_by_the_contract(topology, n):
    """contracts.py: "G.graph holds topology, n, seed and mean_degree"."""
    graph = generate_graph(topology, n, seed=3)

    for key in ("topology", "n", "seed", "mean_degree"):
        assert key in graph.graph, f"G.graph is missing {key!r}"
    assert graph.graph["topology"] == topology
    assert graph.graph["n"] == n
    assert graph.graph["seed"] == 3
    assert graph.graph["mean_degree"] == pytest.approx(mean_degree(graph))
    assert graph.graph["target_mean_degree"] == pytest.approx(TARGET_MEAN_DEGREE)


@pytest.mark.parametrize("topology", TOPOLOGIES)
def test_graph_attributes_are_json_friendly_scalars(topology):
    """Track D writes these into Parquet, so they must be plain scalars."""
    graph = generate_graph(topology, 14)
    assert isinstance(graph.graph["topology"], str)
    assert isinstance(graph.graph["n"], int) and not isinstance(graph.graph["n"], bool)
    assert isinstance(graph.graph["seed"], int)
    assert isinstance(graph.graph["mean_degree"], float)


@pytest.mark.parametrize("topology", TOPOLOGIES)
def test_full_generator_provenance_survives_the_adapter(topology):
    """The adapter must not drop the Track A metadata on the floor."""
    graph = generate_graph(topology, 112)
    provenance = graph.graph["generator_metadata"]

    assert provenance["topology"] == topology
    assert provenance["n"] == 112
    assert provenance["degree_matched_exactly"] is True
    assert provenance["attempts"] >= 1
    assert provenance["generator_version"]
    assert provenance["networkx_version"] == nx.__version__
    assert provenance["params"]


def test_er_rejection_sampling_draw_count_is_visible_through_the_adapter():
    """The ER connectivity conditioning must stay auditable from the shared API.

    Mean degree 4 is below the ln N connectivity threshold at N = 112, so the
    shipped ER graphs come from the *connected* ensemble by rejection sampling.
    A caller must be able to see how many draws that took.
    """
    attempts = [
        generate_graph("er", 112, seed=seed).graph["generator_metadata"]["attempts"]
        for seed in EXPERIMENT_SEEDS
    ]
    assert all(a >= 1 for a in attempts)
    assert max(attempts) > 1, "expected at least one N=112 seed to need a redraw"


# --------------------------------------------------------------------------
# every topology x every population size x every experiment seed
# --------------------------------------------------------------------------

@pytest.mark.parametrize("topology", TOPOLOGIES)
@pytest.mark.parametrize("n", POPULATION_SIZES)
@pytest.mark.parametrize("seed", EXPERIMENT_SEEDS)
def test_whole_sweep_is_connected_with_contiguous_int_node_ids(topology, n, seed):
    """The contract's node promise: "nodes are ints 0 .. n-1"."""
    graph = generate_graph(topology, n, seed=seed)

    assert set(graph.nodes()) == set(range(n))
    assert all(isinstance(v, int) and not isinstance(v, bool) for v in graph.nodes())
    assert not graph.is_directed()
    assert not graph.is_multigraph()
    assert nx.number_of_selfloops(graph) == 0
    assert nx.is_connected(graph)


@pytest.mark.parametrize("topology", TOPOLOGIES)
@pytest.mark.parametrize("n", POPULATION_SIZES)
@pytest.mark.parametrize("seed", EXPERIMENT_SEEDS)
def test_whole_sweep_is_matched_on_mean_degree(topology, n, seed):
    """Mean degree is the study's controlled variable, so assert it exactly.

    The contract only promises +/-10% for N >= 14, but all three Week 1
    generators hit 4 exactly at every size -- including the N = 7 boundary
    case. Asserting the stronger property is the right test; if a future change
    weakens it to "within tolerance", that must be a deliberate edit here.
    """
    graph = generate_graph(topology, n, seed=seed)

    assert graph.number_of_edges() == target_edge_count(n, TARGET_MEAN_DEGREE) == 2 * n
    assert mean_degree(graph) == pytest.approx(TARGET_MEAN_DEGREE)
    assert graph.graph["mean_degree"] == pytest.approx(TARGET_MEAN_DEGREE)
    deviation = abs(mean_degree(graph) - TARGET_MEAN_DEGREE) / TARGET_MEAN_DEGREE
    assert deviation <= DEGREE_TOLERANCE


@pytest.mark.parametrize("topology", TOPOLOGIES)
@pytest.mark.parametrize("n", POPULATION_SIZES)
def test_whole_sweep_passes_the_structural_validator(topology, n):
    graph = generate_graph(topology, n)
    result = validate_graph(graph, n=n, topology=topology, seed=0)
    assert result.ok, result.summary()
    assert result.sweep_eligible, result.summary()


@pytest.mark.parametrize("n", POPULATION_SIZES)
def test_n_over_seven_is_integral_at_every_scale(n):
    """The boundary-case premise: balanced roles need N divisible by 7.

    Week 2 owns assign_roles, but the sizes it has to serve are fixed now, so
    the premise is worth pinning here.
    """
    assert n % len(YEO7) == 0
    assert POPULATION_SIZES[0] == 7, "N=7 is the documented boundary case"


# --------------------------------------------------------------------------
# reproducibility through the shared interface
# --------------------------------------------------------------------------

@pytest.mark.parametrize("topology", TOPOLOGIES)
@pytest.mark.parametrize("n", POPULATION_SIZES)
def test_same_seed_gives_the_same_graph(topology, n):
    first = generate_graph(topology, n, seed=1)
    second = generate_graph(topology, n, seed=1)
    assert {tuple(sorted(e)) for e in first.edges()} == {
        tuple(sorted(e)) for e in second.edges()
    }
    assert set(first.nodes()) == set(second.nodes())


@pytest.mark.parametrize("topology", TOPOLOGIES)
@pytest.mark.parametrize("n", [14, 28, 56, 112])
def test_different_seeds_give_different_graphs(topology, n):
    """N = 7 is excluded for the documented boundary-case reason.

    At N = 7 the degree-matched BA graph is K_5 plus two preferentially
    attached nodes, which has too few distinct realisations for this assertion
    to mean anything. See test_generators.py, where the same exclusion is a
    pytest.skip over the full parametrisation.
    """
    edge_sets = {
        frozenset(tuple(sorted(e)) for e in generate_graph(topology, n, seed=s).edges())
        for s in EXPERIMENT_SEEDS
    }
    assert len(edge_sets) > 1


@pytest.mark.parametrize("topology", TOPOLOGIES)
def test_adapter_and_generator_layer_agree(topology):
    """The adapter must not perturb the draw: same seed, same edges.

    This is what makes the committed fixtures (produced through the generator
    layer) valid for consumers calling the shared interface.
    """
    through_adapter = generate_graph(topology, 56, seed=4)
    through_spec, _ = generate_from_spec(GraphSpec(topology, n=56, seed=4))
    assert {tuple(sorted(e)) for e in through_adapter.edges()} == {
        tuple(sorted(e)) for e in through_spec.edges()
    }


# --------------------------------------------------------------------------
# error handling
# --------------------------------------------------------------------------

@pytest.mark.parametrize("topology", PLANNED_TOPOLOGIES)
def test_week_two_topologies_raise_not_implemented(topology):
    """modular and yeo_sbm are declared in contracts.Topology but not built.

    They must fail as "not implemented yet", not as "unknown topology", so a
    teammate reading the error knows it is a roadmap item rather than a typo.
    """
    with pytest.raises(NotImplementedError, match="Week 2"):
        generate_graph(topology, 28)


def test_planned_topologies_are_exactly_the_contract_minus_week_one():
    """Keep this list honest against contracts.Topology, which is frozen Nov 1."""
    declared = set(Topology.__args__)
    assert declared == set(TOPOLOGIES) | set(PLANNED_TOPOLOGIES)
    assert set(TOPOLOGIES) == {"er", "ws", "ba"}


def test_unknown_topology_is_rejected():
    with pytest.raises(GraphParameterError, match="unknown topology"):
        generate_graph("small_world", 28)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"n": 2}, "n must be >= 3"),
        ({"n": 0}, "n must be >= 3"),
        ({"n": -14}, "n must be >= 3"),
        ({"n": 14.5}, "n must be an integer"),
        ({"n": "14"}, "n must be an integer"),
        ({"seed": -1}, "seed must be non-negative"),
        ({"seed": 1.5}, "seed must be an integer"),
        ({"mean_degree": 0}, "must be positive"),
        ({"mean_degree": -4}, "must be positive"),
        ({"mean_degree": float("nan")}, "must be positive and finite"),
        ({"mean_degree": 500}, "infeasible"),
    ],
)
def test_invalid_arguments_raise_through_the_adapter(kwargs, match):
    """The adapter must surface the generator's own errors, not swallow them."""
    call = {"topology": "er", "n": 14, "mean_degree": 4.0, "seed": 0}
    call.update(kwargs)
    with pytest.raises(GraphParameterError, match=match):
        generate_graph(**call)


def test_graph_parameter_error_is_a_value_error():
    """Callers that only catch ValueError must keep working."""
    assert issubclass(GraphParameterError, ValueError)
    with pytest.raises(ValueError):
        generate_graph("er", 2)


def test_non_default_mean_degree_is_honoured():
    """mean_degree is a real parameter, not decoration."""
    graph = generate_graph("ws", 28, mean_degree=6, seed=0)
    assert graph.number_of_edges() == target_edge_count(28, 6) == 84
    assert mean_degree(graph) == pytest.approx(6.0)
    assert graph.graph["target_mean_degree"] == pytest.approx(6.0)


# --------------------------------------------------------------------------
# later-week entry points keep their placeholder behaviour
# --------------------------------------------------------------------------

def test_later_week_entry_points_are_still_placeholders():
    """Week 1 is graphs only. These must not pretend to work."""
    graph = generate_graph("er", 14)
    with pytest.raises(NotImplementedError, match="assign_roles"):
        assign_roles(graph, seed=0)
    with pytest.raises(NotImplementedError, match="structural_metrics"):
        structural_metrics(graph)


def test_remove_nodes_is_still_a_placeholder():
    from neuromesh.contracts import RemovalSpec

    graph = generate_graph("er", 14)
    with pytest.raises(NotImplementedError, match="remove_nodes"):
        remove_nodes(graph, RemovalSpec(mode="random", fraction=0.1), seed=0)


def test_no_yeo_roles_are_assigned_yet():
    """Role assignment is Week 2: no node may carry a fabricated role."""
    graph = generate_graph("ba", 28)
    assert all("role" not in data for _, data in graph.nodes(data=True))
