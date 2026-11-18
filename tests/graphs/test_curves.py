"""Tests for graph-only robustness curves and the structural AUC.

Owner: Mihir, Track A.

Two things are checked especially carefully, because both are easy to get
subtly wrong and neither would announce itself:

* **The denominator.** ``largest_component_fraction`` must divide by the
  ORIGINAL N, not by the surviving count. A survivor-normalised curve makes
  total collapse look like perfect robustness, so the tests pin the exact
  expected value on a hand-built graph where the two differ.
* **Reuse of the Week 3 engine.** The trajectory's removal order must be
  character-for-character the order ``robustness.py`` produces, including the
  adaptive recomputation and the ``smallest_id`` tie-break. If this module
  ever grew its own removal logic, these tests would catch the divergence.

The AUC is checked against trapezoid areas worked out by hand, not against
``numpy.trapz`` and not against the implementation.
"""

from __future__ import annotations

import networkx as nx
import pytest

from neuromesh.graphs import (
    AUC_FIELDS,
    CHECKPOINT_FRACTIONS,
    MAX_REMOVAL_FRACTION,
    POPULATION_SIZES,
    ROBUSTNESS_FIELDS,
    TOPOLOGIES,
    GraphParameterError,
    assign_roles,
    checkpoint_rows,
    generate_graph,
    random_removal_order,
    removal_count,
    robustness_auc,
    robustness_trajectory,
    targeted_removal_order,
    trajectory_rows,
)

SEEDS = (0, 1, 2, 3, 4)

_CACHE: dict[tuple, nx.Graph] = {}


def study_graph(topology: str, n: int, seed: int = 0) -> nx.Graph:
    key = (topology, n, seed)
    if key not in _CACHE:
        graph = generate_graph(topology, n, seed=seed)
        assign_roles(graph, seed=seed)
        _CACHE[key] = graph
    return _CACHE[key]


# ==========================================================================
# 1. the curve is complete and starts at the untouched graph
# ==========================================================================

@pytest.mark.parametrize("topology", TOPOLOGIES)
@pytest.mark.parametrize("strategy", ["random", "targeted"])
def test_curve_covers_every_integer_removal_count(topology, strategy):
    graph = study_graph(topology, 112)
    trajectory = robustness_trajectory(graph, strategy, seed=0)
    expected_k_max = removal_count(112, MAX_REMOVAL_FRACTION)
    assert expected_k_max == 33
    assert trajectory["k_max"] == 33
    assert [p["k_removed"] for p in trajectory["points"]] == list(range(34))


@pytest.mark.parametrize("topology", TOPOLOGIES)
@pytest.mark.parametrize("strategy", ["random", "targeted"])
def test_zero_removal_baseline_is_the_original_graph(topology, strategy):
    """Requirement: LCC fraction is exactly 1 at k=0 for a connected input."""
    graph = study_graph(topology, 56)
    first = robustness_trajectory(graph, strategy, seed=0)["points"][0]
    assert first["k_removed"] == 0
    assert first["fraction_removed"] == 0.0
    assert first["n_remaining"] == 56
    assert first["n_edges_remaining"] == graph.number_of_edges()
    assert first["largest_component_size"] == 56
    assert first["largest_component_fraction"] == pytest.approx(1.0)
    assert first["n_components"] == 1
    assert first["mean_degree_remaining"] == pytest.approx(4.0)
    assert first["is_checkpoint"] is True
    assert first["requested_fraction"] == 0.0


def test_point_keys_are_exactly_the_declared_fields():
    trajectory = robustness_trajectory(study_graph("er", 28), "random", seed=0)
    assert set(trajectory["points"][0]) == set(ROBUSTNESS_FIELDS)


@pytest.mark.parametrize("n", POPULATION_SIZES)
def test_k_max_follows_the_week3_rounding_rule(n):
    """No new rounding rule: floor(0.30 N), inherited from Week 3."""
    trajectory = robustness_trajectory(study_graph("er", n), "random", seed=0)
    assert trajectory["k_max"] == removal_count(n, 0.30) == int(0.30 * n)
    assert trajectory["rounding_rule"] == "floor"


# ==========================================================================
# 2. the denominator is the ORIGINAL N
# ==========================================================================

def test_largest_component_fraction_divides_by_the_original_n():
    """A hand-built graph where survivor- and original-normalisation differ.

    A 10-node star plus an isolated pair. Removing the star's centre leaves
    9 nodes in 8 components, the biggest holding 2 of them. Over the ORIGINAL
    10 that is 0.2; over the 9 SURVIVORS it would be 0.222. The first is the
    required convention and is the number that must appear.
    """
    graph = nx.star_graph(7)          # nodes 0..7, centre 0
    graph.add_edge(100, 101)          # a separate pair
    assert graph.number_of_nodes() == 10

    trajectory = robustness_trajectory(graph, "targeted", max_fraction=0.3)
    after_one = trajectory["points"][1]
    assert after_one["k_removed"] == 1
    assert trajectory["order"][0] == 0            # the centre, degree 7
    assert after_one["n_remaining"] == 9
    assert after_one["largest_component_size"] == 2
    assert after_one["largest_component_fraction"] == pytest.approx(2 / 10)
    assert after_one["largest_component_fraction"] != pytest.approx(2 / 9)
    assert after_one["fraction_removed"] == pytest.approx(1 / 10)
    assert after_one["fraction_remaining"] == pytest.approx(9 / 10)


@pytest.mark.parametrize("topology", TOPOLOGIES)
@pytest.mark.parametrize("strategy", ["random", "targeted"])
def test_fractions_stay_within_zero_and_one(topology, strategy):
    for n in (28, 112):
        trajectory = robustness_trajectory(study_graph(topology, n), strategy, seed=0)
        for point in trajectory["points"]:
            assert 0.0 <= point["largest_component_fraction"] <= 1.0
            assert 0.0 <= point["fraction_removed"] <= 1.0
            assert 0.0 <= point["fraction_remaining"] <= 1.0
            assert point["fraction_removed"] + point["fraction_remaining"] == (
                pytest.approx(1.0)
            )
            assert point["largest_component_size"] <= point["n_remaining"]


# ==========================================================================
# 3. monotonicity -- the largest component can only shrink
# ==========================================================================

@pytest.mark.parametrize("topology", TOPOLOGIES)
@pytest.mark.parametrize("strategy", ["random", "targeted"])
@pytest.mark.parametrize("seed", SEEDS)
def test_largest_component_never_grows(topology, strategy, seed):
    """Removing a node cannot enlarge any component. Checked at every step."""
    trajectory = robustness_trajectory(study_graph(topology, 56, seed), strategy, seed=seed)
    sizes = [p["largest_component_size"] for p in trajectory["points"]]
    for earlier, later in zip(sizes, sizes[1:], strict=False):
        assert later <= earlier, sizes
    # node and edge counts are monotone too
    remaining = [p["n_remaining"] for p in trajectory["points"]]
    assert remaining == sorted(remaining, reverse=True)
    edges = [p["n_edges_remaining"] for p in trajectory["points"]]
    for earlier, later in zip(edges, edges[1:], strict=False):
        assert later <= earlier


def test_node_count_decreases_by_exactly_one_per_step():
    trajectory = robustness_trajectory(study_graph("ws", 112), "targeted")
    for index, point in enumerate(trajectory["points"]):
        assert point["n_remaining"] == 112 - index
        assert point["k_removed"] == index


# ==========================================================================
# 4. reuse of the Week 3 removal engine
# ==========================================================================

@pytest.mark.parametrize("topology", TOPOLOGIES)
def test_targeted_order_is_exactly_the_week3_adaptive_order(topology):
    graph = study_graph(topology, 112)
    trajectory = robustness_trajectory(graph, "targeted", seed=0)
    assert trajectory["order"] == targeted_removal_order(graph, 33)
    assert trajectory["tie_break"] == "smallest_id"


@pytest.mark.parametrize("topology", TOPOLOGIES)
def test_random_order_is_exactly_the_week3_random_order(topology):
    graph = study_graph(topology, 112)
    trajectory = robustness_trajectory(graph, "random", seed=3)
    assert trajectory["order"] == random_removal_order(graph, 33, 3)


@pytest.mark.parametrize("topology", TOPOLOGIES)
def test_targeted_removal_is_adaptive_highest_degree_at_every_step(topology):
    """Independently replayed: the struck node held the current maximum."""
    graph = study_graph(topology, 112)
    order = robustness_trajectory(graph, "targeted", seed=0)["order"]
    working = graph.copy()
    for node in order:
        degrees = dict(working.degree())
        assert degrees[node] == max(degrees.values())
        working.remove_node(node)


def test_random_strategy_requires_a_seed():
    with pytest.raises(GraphParameterError, match="seed is required"):
        robustness_trajectory(study_graph("er", 28), "random")


def test_unknown_strategy_is_rejected():
    with pytest.raises(GraphParameterError, match="unknown removal strategy"):
        robustness_trajectory(study_graph("er", 28), "sabotage", seed=0)


@pytest.mark.parametrize("bad", [-0.1, 1.5])
def test_max_fraction_out_of_range_is_rejected(bad):
    with pytest.raises(GraphParameterError, match=r"max_fraction must lie in \[0, 1\]"):
        robustness_trajectory(study_graph("er", 28), "targeted", max_fraction=bad)


def test_max_fraction_must_be_numeric():
    with pytest.raises(GraphParameterError, match="max_fraction must be a number"):
        robustness_trajectory(study_graph("er", 28), "targeted", max_fraction="0.3")


# ==========================================================================
# 5. the input graph is never modified
# ==========================================================================

@pytest.mark.parametrize("strategy", ["random", "targeted"])
def test_trajectory_never_modifies_the_input_graph(strategy):
    graph = study_graph("ba", 112)
    before = (
        sorted(graph.nodes()),
        sorted(tuple(sorted(e)) for e in graph.edges()),
        {v: dict(d) for v, d in graph.nodes(data=True)},
        dict(graph.graph),
    )
    robustness_trajectory(graph, strategy, seed=0)
    assert (
        sorted(graph.nodes()),
        sorted(tuple(sorted(e)) for e in graph.edges()),
        {v: dict(d) for v, d in graph.nodes(data=True)},
        dict(graph.graph),
    ) == before
    assert "removal" not in graph.graph


# ==========================================================================
# 6. determinism
# ==========================================================================

@pytest.mark.parametrize("topology", TOPOLOGIES)
@pytest.mark.parametrize("strategy", ["random", "targeted"])
def test_trajectories_reproduce_exactly(topology, strategy):
    graph = study_graph(topology, 56)
    first = robustness_trajectory(graph, strategy, seed=2)
    second = robustness_trajectory(graph, strategy, seed=2)
    assert first["order"] == second["order"]
    assert first["points"] == second["points"]


@pytest.mark.parametrize("seed", SEEDS)
def test_random_trajectories_differ_between_seeds_but_replay_per_seed(seed):
    graph = study_graph("er", 112)
    first = robustness_trajectory(graph, "random", seed=seed)
    second = robustness_trajectory(graph, "random", seed=seed)
    assert first["order"] == second["order"]
    other = robustness_trajectory(graph, "random", seed=(seed + 1) % 5)
    assert first["order"] != other["order"]


def test_global_rng_perturbation_does_not_change_a_trajectory():
    import random

    import numpy as np

    graph = study_graph("modular", 56)
    random.seed(1)
    np.random.seed(1)
    first = robustness_trajectory(graph, "random", seed=4)
    random.seed(99)
    np.random.seed(99)
    [random.random() for _ in range(40)]
    np.random.random(40)
    second = robustness_trajectory(graph, "random", seed=4)
    assert first["points"] == second["points"]


# ==========================================================================
# 7. checkpoints
# ==========================================================================

def test_checkpoints_are_the_study_levels_at_n112():
    trajectory = robustness_trajectory(study_graph("er", 112), "random", seed=0)
    marked = [p for p in trajectory["points"] if p["is_checkpoint"]]
    assert [p["k_removed"] for p in marked] == [0, 11, 22, 33]
    assert [p["requested_fraction"] for p in marked] == [0.0, 0.1, 0.2, 0.3]
    for point in marked:
        assert point["k_removed"] == removal_count(112, point["requested_fraction"])


def test_checkpoint_values_come_from_the_same_curve():
    """Checkpoints are flagged within the curve, so they cannot disagree."""
    trajectory = robustness_trajectory(study_graph("ba", 112), "targeted")
    by_k = {p["k_removed"]: p for p in trajectory["points"]}
    for point in trajectory["points"]:
        if point["is_checkpoint"]:
            assert point is by_k[point["k_removed"]]


def test_n7_collapses_the_zero_and_ten_percent_checkpoints():
    """A real boundary artefact, asserted so it is not discovered later.

    floor(0.1 * 7) = 0, so at N = 7 the 10% checkpoint IS the 0% checkpoint.
    Only three distinct checkpoints exist there, and the recorded
    ``requested_fraction`` is 0.0 (the first request to claim k = 0).
    """
    trajectory = robustness_trajectory(study_graph("er", 7), "random", seed=0)
    marked = [p for p in trajectory["points"] if p["is_checkpoint"]]
    assert trajectory["k_max"] == 2
    assert [p["k_removed"] for p in marked] == [0, 1, 2]
    assert [p["requested_fraction"] for p in marked] == [0.0, 0.2, 0.3]
    assert removal_count(7, 0.1) == 0


def test_checkpoint_and_trajectory_row_helpers():
    trajectories = [
        robustness_trajectory(study_graph("er", 28), s, seed=0)
        for s in ("random", "targeted")
    ]
    all_rows = trajectory_rows(trajectories)
    assert len(all_rows) == sum(len(t["points"]) for t in trajectories)
    marked = checkpoint_rows(trajectories)
    assert marked
    assert all(r["is_checkpoint"] for r in marked)
    assert len(marked) < len(all_rows)


def test_default_checkpoints_are_the_experiment_levels():
    assert CHECKPOINT_FRACTIONS == (0.0, 0.1, 0.2, 0.3)
    assert MAX_REMOVAL_FRACTION == 0.30


# ==========================================================================
# 8. efficiency, under both normalisations
# ==========================================================================

def test_both_efficiency_normalisations_agree_at_zero_removal():
    for topology in TOPOLOGIES:
        first = robustness_trajectory(
            study_graph(topology, 56), "random", seed=0
        )["points"][0]
        assert first["global_efficiency"] == pytest.approx(
            first["global_efficiency_original_n"]
        )


def test_original_n_efficiency_is_the_rescaled_survivor_efficiency():
    """The rescaling is exact, so it can be checked arithmetically."""
    trajectory = robustness_trajectory(study_graph("ba", 112), "targeted")
    for point in trajectory["points"]:
        survivors = point["n_remaining"]
        if survivors < 2:
            continue
        expected = (
            point["global_efficiency"] * survivors * (survivors - 1) / (112 * 111)
        )
        assert point["global_efficiency_original_n"] == pytest.approx(expected)


def test_efficiency_stays_in_zero_one():
    for topology in TOPOLOGIES:
        for strategy in ("random", "targeted"):
            for point in robustness_trajectory(
                study_graph(topology, 56), strategy, seed=0
            )["points"]:
                assert 0.0 <= point["global_efficiency"] <= 1.0
                assert 0.0 <= point["global_efficiency_original_n"] <= 1.0


def test_original_n_efficiency_decreases_under_attack():
    """The survivor-normalised figure can stay flat; the original-N one cannot.

    This is exactly why both are exported: under random failure BA's survivor
    efficiency barely moves while the share of the original network's capacity
    falls steadily.
    """
    points = robustness_trajectory(study_graph("ba", 112), "random", seed=0)["points"]
    first, last = points[0], points[-1]
    assert last["global_efficiency_original_n"] < 0.6 * first[
        "global_efficiency_original_n"
    ]
    # ...while the survivor-normalised value stays close to its starting value
    assert last["global_efficiency"] > 0.8 * first["global_efficiency"]


# ==========================================================================
# 9. the structural AUC
# ==========================================================================

def test_auc_keys_are_exactly_the_declared_fields():
    trajectory = robustness_trajectory(study_graph("er", 28), "random", seed=0)
    assert set(robustness_auc(trajectory)) == set(AUC_FIELDS)


def test_auc_matches_a_hand_computed_trapezoid():
    """A graph whose decay curve is short enough to integrate by hand.

    C7 (a 7-cycle). floor(0.30*7) = 2, so the curve has three points at
    x = 0, 1/7, 2/7. Removing one node of a cycle leaves a connected P6, so
    the LCC fraction is 6/7; removing a second (adaptive, smallest-id, so
    node 0 then node 2 after 1 and 6 lose degree) splits the remainder.

    The trapezoid area is then
        0.5*(x1-x0)*(y0+y1) + 0.5*(x2-x1)*(y1+y2)
    computed explicitly below from the measured y values, which are asserted
    separately. Both the raw and normalised values are checked.
    """
    trajectory = robustness_trajectory(nx.cycle_graph(7), "targeted")
    points = trajectory["points"]
    assert [p["k_removed"] for p in points] == [0, 1, 2]

    xs = [p["fraction_removed"] for p in points]
    ys = [p["largest_component_fraction"] for p in points]
    assert xs == pytest.approx([0.0, 1 / 7, 2 / 7])
    assert ys[0] == pytest.approx(1.0)
    assert ys[1] == pytest.approx(6 / 7)  # one removal leaves a connected path

    expected_raw = 0.5 * (xs[1] - xs[0]) * (ys[0] + ys[1]) + 0.5 * (
        xs[2] - xs[1]
    ) * (ys[1] + ys[2])
    result = robustness_auc(trajectory)
    assert result["auc_raw"] == pytest.approx(expected_raw)
    assert result["auc_normalized"] == pytest.approx(expected_raw / (2 / 7))
    assert result["x_min"] == 0.0
    assert result["x_max"] == pytest.approx(2 / 7)
    assert result["n_points"] == 3


def test_auc_of_a_non_fragmenting_graph_is_the_ceiling_not_one():
    """A complete graph never fragments, yet its normalised AUC is 0.85.

    K10: removing any 3 of 10 nodes leaves K7, still fully connected, so the
    largest component always holds all survivors -- but the fraction is over
    the ORIGINAL N, so y falls as 1.0, 0.9, 0.8, 0.7 and the AUC is NOT 1.
    The "perfectly robust" reference is instead the mean of those, computed
    here in closed form.
    """
    trajectory = robustness_trajectory(nx.complete_graph(10), "targeted")
    points = trajectory["points"]
    assert [p["n_components"] for p in points] == [1, 1, 1, 1]
    assert [p["largest_component_fraction"] for p in points] == pytest.approx(
        [1.0, 0.9, 0.8, 0.7]
    )
    # y falls along a straight line from 1.0 to 0.7, so the trapezoid sum
    # collapses to the single trapezoid over the whole span.
    expected = 0.5 * (0.3 - 0.0) * (1.0 + 0.7)
    result = robustness_auc(trajectory)
    assert result["auc_raw"] == pytest.approx(expected)
    assert result["auc_normalized"] == pytest.approx(expected / 0.3)
    assert result["auc_normalized"] == pytest.approx(0.85)


def test_auc_is_higher_for_random_than_targeted_removal():
    """The headline structural result, asserted on every topology at N=112."""
    for topology in TOPOLOGIES:
        graph = study_graph(topology, 112)
        random_auc = robustness_auc(
            robustness_trajectory(graph, "random", seed=0)
        )["auc_normalized"]
        targeted_auc = robustness_auc(
            robustness_trajectory(graph, "targeted", seed=0)
        )["auc_normalized"]
        assert targeted_auc <= random_auc, topology


def test_ba_is_the_most_fragile_under_targeted_attack():
    """Scale-free robust-yet-fragile, measured rather than asserted verbally."""
    aucs = {}
    for topology in TOPOLOGIES:
        aucs[topology] = robustness_auc(
            robustness_trajectory(study_graph(topology, 112), "targeted")
        )["auc_normalized"]
    assert aucs["ba"] == min(aucs.values())
    assert aucs["ba"] < 0.5
    assert all(v > 0.6 for t, v in aucs.items() if t != "ba"), aucs


def test_auc_reports_its_own_axes_and_method():
    result = robustness_auc(
        robustness_trajectory(study_graph("er", 56), "random", seed=0)
    )
    assert "trapezoidal" in result["integration"]
    assert "no extrapolation" in result["integration"]
    assert "original N" in result["x_axis"]
    assert "original N" in result["y_axis"]


def test_auc_can_integrate_another_field():
    trajectory = robustness_trajectory(study_graph("er", 56), "random", seed=0)
    result = robustness_auc(trajectory, y_field="global_efficiency_original_n")
    assert result["auc_raw"] is not None
    assert "global_efficiency_original_n" in result["y_axis"]


def test_auc_is_none_when_a_curve_has_fewer_than_two_points():
    trajectory = robustness_trajectory(
        study_graph("er", 7), "random", seed=0, max_fraction=0.0
    )
    assert len(trajectory["points"]) == 1
    result = robustness_auc(trajectory)
    assert result["auc_raw"] is None
    assert result["auc_normalized"] is None


def test_auc_is_none_when_a_y_value_is_undefined():
    trajectory = robustness_trajectory(study_graph("er", 28), "random", seed=0)
    result = robustness_auc(trajectory, y_field="mean_degree_remaining")
    # mean degree is defined here, so this must produce a number...
    assert result["auc_raw"] is not None
    # ...but a field that is None at some point yields None overall
    trajectory["points"][2]["mean_degree_remaining"] = None
    assert robustness_auc(trajectory, y_field="mean_degree_remaining")[
        "auc_raw"
    ] is None


def test_auc_normalised_is_in_zero_one_for_a_fraction_curve():
    for topology in TOPOLOGIES:
        for strategy in ("random", "targeted"):
            result = robustness_auc(
                robustness_trajectory(study_graph(topology, 112), strategy, seed=0)
            )
            assert 0.0 <= result["auc_normalized"] <= 1.0


# ==========================================================================
# 10. the AUC ceiling is 1 - x_max/2, NOT 1.0
# ==========================================================================
#
# Added after an independent review found the docstring and the Week 4 report
# both claiming 1.0 meant "no damage". Under ORIGINAL-N normalisation that is
# impossible: removing a node lowers y even when the graph never fragments.
# These tests pin the real ceiling so the claim cannot regress.

def auc_ceiling(x_max: float) -> float:
    """Highest attainable normalised AUC: the mean of y = 1 - x over [0, x_max]."""
    return 1.0 - x_max / 2.0


def test_unfragmented_curve_hits_the_ceiling_exactly_not_one():
    """K10 never fragments, yet its normalised AUC is 0.85, not 1.0."""
    trajectory = robustness_trajectory(nx.complete_graph(10), "targeted")
    result = robustness_auc(trajectory)
    assert [p["n_components"] for p in trajectory["points"]] == [1, 1, 1, 1]
    assert result["x_max"] == pytest.approx(0.3)
    assert result["auc_normalized"] == pytest.approx(auc_ceiling(0.3))
    assert result["auc_normalized"] == pytest.approx(0.85)
    assert result["auc_normalized"] < 1.0


@pytest.mark.parametrize("n", POPULATION_SIZES)
def test_ceiling_is_one_minus_half_x_max_at_every_scale(n):
    trajectory = robustness_trajectory(study_graph("er", n), "random", seed=0)
    result = robustness_auc(trajectory)
    expected = 0.857142857142857 if n != 112 else 0.852678571428571
    assert auc_ceiling(result["x_max"]) == pytest.approx(expected)
    assert result["auc_normalized"] <= auc_ceiling(result["x_max"]) + 1e-12


@pytest.mark.parametrize("topology", TOPOLOGIES)
@pytest.mark.parametrize("strategy", ["random", "targeted"])
@pytest.mark.parametrize("n", POPULATION_SIZES)
def test_no_measured_auc_can_exceed_its_ceiling(topology, strategy, n):
    """The invariant, over the whole grid: AUC <= 1 - x_max/2, always."""
    for seed in SEEDS:
        result = robustness_auc(
            robustness_trajectory(study_graph(topology, n, seed), strategy, seed=seed)
        )
        assert result["auc_normalized"] <= auc_ceiling(result["x_max"]) + 1e-12


def test_n7_random_removal_sits_on_the_ceiling():
    """Why N = 7 discriminates nothing: the curve cannot leave its ceiling.

    Two random removals from a connected 7-node graph of mean degree 4
    essentially never fragment it, so y falls along the unavoidable straight
    line and the AUC pins at 6/7.
    """
    for topology in TOPOLOGIES:
        result = robustness_auc(
            robustness_trajectory(study_graph(topology, 7), "random", seed=0)
        )
        assert result["auc_normalized"] == pytest.approx(6 / 7)
        assert result["auc_normalized"] == pytest.approx(auc_ceiling(2 / 7))
