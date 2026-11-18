"""Tests for the Week 4 datasets, figures and CLI (owner: Mihir, Track A).

The three full-grid builds -- 125 structural rows, 5425 node rows and 250
robustness trajectories -- are the actual Week 4 acceptance criteria, so they
are tested on the REAL grid rather than a reduced one. Each is built exactly
once per test session via a session-scoped fixture, which keeps the cost to
roughly 12 s for the whole module instead of repeating it per test.

Everything else (CSV round-trips, figure rendering, the CLI) runs on a reduced
grid so the suite stays quick.
"""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET

import pytest

from neuromesh.graphs import (
    AGGREGATE_METRICS,
    AUC_FIELDS,
    CONDITION_FIELDS,
    DEFAULT_OUTPUT_DIR,
    EXPERIMENT_SEEDS,
    POPULATION_SIZES,
    POSITIONAL_FIELDS,
    ROBUSTNESS_FIELDS,
    STRUCTURAL_FIELDS,
    TOPOLOGIES,
    GraphParameterError,
    aggregate_conditions,
    build_positional_table,
    build_robustness_tables,
    build_structural_table,
    expected_node_row_count,
)
from neuromesh.graphs.cli import main
from neuromesh.graphs.datasets import format_value, read_csv, write_csv, write_json
from neuromesh.graphs.figures import PALETTE, write_figures

#: A cheap grid for the tests that do not need the full sweep.
SMALL = {"topologies": ("er", "modular"), "sizes": (7, 14), "seeds": (0, 1)}


@pytest.fixture(scope="session")
def structural_full() -> list[dict]:
    """All 125 structural rows. Built once per session."""
    return build_structural_table()


@pytest.fixture(scope="session")
def positional_full() -> list[dict]:
    """All 5425 node rows. Built once per session."""
    return build_positional_table()


@pytest.fixture(scope="session")
def robustness_full() -> tuple[list[dict], list[dict], list[dict]]:
    """All 250 trajectories with their points and AUCs. Built once."""
    return build_robustness_tables()


# ==========================================================================
# 1. the required row counts
# ==========================================================================

def test_structural_table_has_exactly_125_rows(structural_full):
    assert len(structural_full) == 125
    assert len(TOPOLOGIES) * len(POPULATION_SIZES) * len(EXPERIMENT_SEEDS) == 125


def test_condition_summary_has_exactly_25_rows(structural_full):
    conditions = aggregate_conditions(structural_full)
    assert len(conditions) == 25
    assert len(TOPOLOGIES) * len(POPULATION_SIZES) == 25


def test_positional_table_has_exactly_5425_rows(positional_full):
    assert len(positional_full) == 5425
    # 5 topologies x 5 seeds x (7+14+28+56+112) = 5 x 5 x 217
    assert 5 * 5 * sum(POPULATION_SIZES) == 5425
    assert expected_node_row_count() == 5425


def test_robustness_has_exactly_250_trajectories(robustness_full):
    _, auc, trajectories = robustness_full
    assert len(trajectories) == 250   # 125 graphs x 2 strategies
    assert len(auc) == 250
    assert {t["strategy"] for t in trajectories} == {"random", "targeted"}


def test_robustness_point_count_matches_the_curve_lengths(robustness_full):
    rows, _, trajectories = robustness_full
    assert len(rows) == sum(len(t["points"]) for t in trajectories)
    # per (topology, seed): sum over N of floor(0.3N)+1 = 3+5+9+17+34 = 68
    assert len(rows) == 68 * 25 * 2 == 3400


# ==========================================================================
# 2. grid coverage and key uniqueness
# ==========================================================================

def test_all_topologies_sizes_and_seeds_are_present(structural_full):
    assert {r["topology"] for r in structural_full} == set(TOPOLOGIES)
    assert {r["n"] for r in structural_full} == set(POPULATION_SIZES)
    assert {r["seed"] for r in structural_full} == set(EXPERIMENT_SEEDS)


def test_structural_graph_ids_are_unique(structural_full):
    ids = [r["graph_id"] for r in structural_full]
    assert len(set(ids)) == len(ids) == 125
    assert all(ids)


def test_no_duplicate_condition_keys(structural_full):
    conditions = aggregate_conditions(structural_full)
    keys = [(r["topology"], r["n"]) for r in conditions]
    assert len(set(keys)) == len(keys) == 25
    assert len({r["condition_id"] for r in conditions}) == 25


def test_positional_keys_are_unique_within_their_scope(positional_full):
    keys = [(r["graph_id"], r["node_id"]) for r in positional_full]
    assert len(set(keys)) == len(keys) == 5425


def test_positional_node_ids_are_valid_for_their_graph(positional_full):
    by_graph: dict[str, set] = {}
    for row in positional_full:
        by_graph.setdefault(row["graph_id"], set()).add(row["node_id"])
    assert len(by_graph) == 125
    for graph_id_value, nodes in by_graph.items():
        n = int(graph_id_value.split("_n")[1].split("_")[0])
        assert nodes == set(range(n)), graph_id_value


def test_robustness_graph_ids_join_the_structural_table(
    structural_full, robustness_full
):
    """A consistent instance identifier across every Week 4 table."""
    rows, auc, _ = robustness_full
    structural_ids = {r["graph_id"] for r in structural_full}
    assert {r["graph_id"] for r in rows} == structural_ids
    assert {r["graph_id"] for r in auc} == structural_ids


def test_positional_graph_ids_join_the_structural_table(
    structural_full, positional_full
):
    assert {r["graph_id"] for r in positional_full} == {
        r["graph_id"] for r in structural_full
    }


# ==========================================================================
# 3. structural invariants over the whole grid
# ==========================================================================

def test_every_graph_has_n_nodes_and_2n_edges(structural_full):
    for row in structural_full:
        assert row["n_nodes"] == row["n"]
        assert row["n_edges"] == 2 * row["n"]


def test_every_original_graph_is_connected(structural_full):
    for row in structural_full:
        assert row["n_components"] == 1, row["graph_id"]
        assert row["largest_component_fraction"] == pytest.approx(1.0)
        assert row["diameter"] is not None
        assert row["average_shortest_path_length"] is not None


def test_mean_degree_is_exactly_four_everywhere(structural_full):
    for row in structural_full:
        assert row["mean_degree"] == pytest.approx(4.0)
        assert row["degree_mean"] == pytest.approx(4.0)
        assert row["target_mean_degree"] == pytest.approx(4.0)
        assert row["degree_matched_exactly"] is True


def test_role_balance_is_n_over_seven_everywhere(structural_full):
    for row in structural_full:
        assert row["roles_assigned"] is True
        assert row["roles_balanced"] is True
        assert row["roles_per_network"] == row["n"] // 7
        assert row["n_roles_present"] == 7


def test_bounded_metrics_stay_in_range(structural_full):
    for row in structural_full:
        assert 0.0 <= row["average_clustering"] <= 1.0
        assert 0.0 <= row["transitivity"] <= 1.0
        assert 0.0 <= row["global_efficiency"] <= 1.0
        assert 0.0 < row["edge_density"] <= 1.0
        assert -0.5 <= row["modularity_role_partition"] < 1.0
        if row["degree_assortativity"] is not None:
            assert -1.0 <= row["degree_assortativity"] <= 1.0


def test_all_structural_numbers_are_finite_or_explicitly_none(structural_full):
    for row in structural_full:
        for key, value in row.items():
            if isinstance(value, float):
                assert math.isfinite(value), f"{row['graph_id']}.{key} = {value}"


def test_all_positional_numbers_are_finite_or_explicitly_none(positional_full):
    for row in positional_full:
        for key, value in row.items():
            if isinstance(value, float):
                assert math.isfinite(value), f"{row['graph_id']}.{key} = {value}"


def test_positional_ranges_over_the_whole_grid(positional_full):
    limit = 1.0 - 1.0 / 7
    for row in positional_full:
        assert 0.0 <= row["betweenness_centrality"] <= 1.0
        assert 0.0 <= row["closeness_centrality"] <= 1.0
        assert 0.0 <= row["clustering"] <= 1.0
        assert 0.0 <= row["degree_centrality"] <= 1.0
        assert 0.0 < row["pagerank"] < 1.0
        assert 0.0 <= row["participation_coefficient"] <= limit + 1e-12
        assert row["degree"] >= 1          # all study graphs are connected
        assert row["core_number"] >= 1
        assert row["role"] is not None
        assert 0 <= row["n_neighbor_networks"] <= 7


def test_pagerank_sums_to_one_per_graph(positional_full):
    totals: dict[str, float] = {}
    for row in positional_full:
        totals[row["graph_id"]] = totals.get(row["graph_id"], 0.0) + row["pagerank"]
    assert len(totals) == 125
    for graph_id_value, total in totals.items():
        assert total == pytest.approx(1.0, abs=1e-8), graph_id_value


def test_node_rows_per_graph_match_that_graph_size(positional_full):
    counts: dict[str, int] = {}
    for row in positional_full:
        counts[row["graph_id"]] = counts.get(row["graph_id"], 0) + 1
    for graph_id_value, count in counts.items():
        n = int(graph_id_value.split("_n")[1].split("_")[0])
        assert count == n


# ==========================================================================
# 4. aggregation
# ==========================================================================

def test_aggregates_match_hand_computed_statistics(structural_full):
    """Recomputed here with plain Python, not with the statistics module."""
    conditions = aggregate_conditions(structural_full)
    for entry in conditions:
        group = [
            r
            for r in structural_full
            if r["topology"] == entry["topology"] and r["n"] == entry["n"]
        ]
        assert entry["n_seeds"] == 5
        for metric in AGGREGATE_METRICS:
            values = [r[metric] for r in group if r[metric] is not None]
            assert entry[f"{metric}_n"] == len(values)
            if not values:
                assert entry[f"{metric}_mean"] is None
                assert entry[f"{metric}_std"] is None
                continue
            mean = sum(values) / len(values)
            assert entry[f"{metric}_mean"] == pytest.approx(mean)
            assert entry[f"{metric}_min"] == pytest.approx(min(values))
            assert entry[f"{metric}_max"] == pytest.approx(max(values))
            if len(values) > 1:
                variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
                assert entry[f"{metric}_std"] == pytest.approx(math.sqrt(variance))
            else:
                assert entry[f"{metric}_std"] is None


def test_aggregate_std_is_the_sample_convention(structural_full):
    """ddof=1, so it differs from the population value. Pinned explicitly."""
    conditions = aggregate_conditions(structural_full)
    entry = next(
        e for e in conditions if e["topology"] == "ws" and e["n"] == 112
    )
    group = [
        r["average_shortest_path_length"]
        for r in structural_full
        if r["topology"] == "ws" and r["n"] == 112
    ]
    mean = sum(group) / 5
    sample = math.sqrt(sum((v - mean) ** 2 for v in group) / 4)
    population = math.sqrt(sum((v - mean) ** 2 for v in group) / 5)
    assert entry["average_shortest_path_length_std"] == pytest.approx(sample)
    assert entry["average_shortest_path_length_std"] != pytest.approx(population)


def test_aggregation_counts_valid_observations_separately(structural_full):
    """WS at N=7 is regular for at least one seed, so assortativity is partial.

    The point of the test is that the mean is then taken over the DEFINED
    observations and the count says so, rather than the undefined seeds being
    silently treated as zero.
    """
    conditions = aggregate_conditions(structural_full)
    partial = [
        e
        for e in conditions
        if e["degree_assortativity_n"] < e["n_seeds"]
    ]
    assert partial, "expected at least one condition with an undefined assortativity"
    for entry in partial:
        group = [
            r["degree_assortativity"]
            for r in structural_full
            if r["topology"] == entry["topology"] and r["n"] == entry["n"]
        ]
        defined = [v for v in group if v is not None]
        assert entry["degree_assortativity_n"] == len(defined)
        assert len(defined) < 5
        assert entry["degree_assortativity_mean"] == pytest.approx(
            sum(defined) / len(defined)
        )


def test_aggregate_rows_are_in_canonical_order(structural_full):
    conditions = aggregate_conditions(structural_full)
    expected = [(t, n) for t in TOPOLOGIES for n in POPULATION_SIZES]
    assert [(e["topology"], e["n"]) for e in conditions] == expected


def test_aggregate_fields_match_the_declared_header(structural_full):
    conditions = aggregate_conditions(structural_full)
    assert set(conditions[0]) == set(CONDITION_FIELDS)


def test_single_seed_condition_reports_no_standard_deviation():
    rows = build_structural_table(("er",), (14,), (0,))
    entry = aggregate_conditions(rows)[0]
    assert entry["n_seeds"] == 1
    assert entry["average_clustering_n"] == 1
    assert entry["average_clustering_mean"] is not None
    assert entry["average_clustering_std"] is None


# ==========================================================================
# 5. determinism of the tables
# ==========================================================================

def test_structural_table_is_reproducible():
    first = build_structural_table(**SMALL)
    second = build_structural_table(**SMALL)
    assert first == second


def test_positional_table_is_reproducible():
    first = build_positional_table(**SMALL)
    second = build_positional_table(**SMALL)
    assert first == second


def test_robustness_tables_are_reproducible():
    first_rows, first_auc, _ = build_robustness_tables(**SMALL)
    second_rows, second_auc, _ = build_robustness_tables(**SMALL)
    assert first_rows == second_rows
    assert first_auc == second_auc


# ==========================================================================
# 6. CSV / JSON serialisation
# ==========================================================================

def test_float_formatting_round_trips_exactly():
    """repr, not a fixed number of decimals, so precision is not lost."""
    for value in (0.1, 1 / 3, 3.4596203346203347, 1e-17, 2.0):
        assert float(format_value(value)) == value
    assert format_value(None) == ""
    assert format_value(True) == "True"
    assert format_value(False) == "False"
    assert format_value(7) == "7"
    assert format_value(float("nan")) == ""
    assert format_value("er") == "er"


def test_csv_round_trip_preserves_every_value(tmp_path):
    rows = build_structural_table(("er",), (14,), (0, 1))
    path = write_csv(tmp_path / "s.csv", rows, STRUCTURAL_FIELDS)
    back = read_csv(path)

    assert len(back) == len(rows)
    assert list(back[0]) == list(STRUCTURAL_FIELDS)
    for original, reloaded in zip(rows, back, strict=True):
        for field in STRUCTURAL_FIELDS:
            value = original[field]
            if value is None:
                assert reloaded[field] is None
            elif isinstance(value, bool):
                assert reloaded[field] == str(value)
            elif isinstance(value, float):
                assert float(reloaded[field]) == value
            elif isinstance(value, int):
                assert int(reloaded[field]) == value
            else:
                assert reloaded[field] == value


def test_csv_writing_is_byte_identical_across_calls(tmp_path):
    rows = build_structural_table(**SMALL)
    first = write_csv(tmp_path / "a.csv", rows, STRUCTURAL_FIELDS).read_bytes()
    second = write_csv(tmp_path / "b.csv", rows, STRUCTURAL_FIELDS).read_bytes()
    assert first == second


def test_csv_uses_lf_line_endings(tmp_path):
    """Explicit, so the file is identical on Windows and Linux."""
    rows = build_structural_table(("er",), (7,), (0,))
    raw = write_csv(tmp_path / "s.csv", rows, STRUCTURAL_FIELDS).read_bytes()
    assert b"\r\n" not in raw
    assert raw.endswith(b"\n")


def test_csv_refuses_to_silently_drop_a_measured_column(tmp_path):
    """The guard that caught a real bug during development."""
    rows = [{"a": 1, "b": 2}]
    with pytest.raises(GraphParameterError, match="absent from the declared header"):
        write_csv(tmp_path / "x.csv", rows, ("a",))


def test_csv_quotes_values_containing_commas(tmp_path):
    rows = [{"a": "x,y", "b": 'he said "hi"'}]
    path = write_csv(tmp_path / "q.csv", rows, ("a", "b"))
    assert read_csv(path)[0] == {"a": "x,y", "b": 'he said "hi"'}


def test_json_writer_rejects_nan(tmp_path):
    with pytest.raises(ValueError):
        write_json(tmp_path / "bad.json", {"x": float("nan")})


def test_json_writer_is_sorted_and_reproducible(tmp_path):
    payload = {"b": 1, "a": [3, 2, 1]}
    first = write_json(tmp_path / "a.json", payload).read_bytes()
    second = write_json(tmp_path / "b.json", payload).read_bytes()
    assert first == second
    assert json.loads(first.decode()) == payload
    assert first.decode().index('"a"') < first.decode().index('"b"')


def test_declared_output_directory_is_under_data():
    assert DEFAULT_OUTPUT_DIR.parts[0] == "data"


# ==========================================================================
# 7. figures
# ==========================================================================

@pytest.fixture(scope="module")
def small_analysis() -> dict:
    structural = build_structural_table(**SMALL)
    rows, _, _ = build_robustness_tables(**SMALL)
    return {"structural": structural, "robustness": rows}


def test_figures_are_written_and_are_valid_xml(tmp_path, small_analysis):
    written = write_figures(tmp_path, small_analysis)
    names = {p.name for p in written}
    assert names == {
        "figure1_structural_metrics.svg",
        "figure2_robustness_curves.svg",
        "figure3_robustness_by_topology.svg",
        "figure_data_1_structural_metrics.csv",
        "figure_data_2_robustness_curves.csv",
        "figure_data_3_robustness_by_topology.csv",
    }
    for path in written:
        assert path.is_file()
        assert path.stat().st_size > 0
        if path.suffix == ".svg":
            root = ET.parse(path).getroot()
            assert root.tag.endswith("svg")
            assert root.get("viewBox")


def test_figures_need_no_third_party_plotting_dependency():
    """matplotlib is NOT a declared project dependency; figures must not use it."""
    import neuromesh.graphs.figures as module

    source = module.__file__
    with open(source, encoding="utf-8") as handle:
        text = handle.read()
    for forbidden in ("import matplotlib", "import plotly", "import seaborn"):
        assert forbidden not in text


def test_figures_are_byte_reproducible(tmp_path, small_analysis):
    first_dir = tmp_path / "one"
    second_dir = tmp_path / "two"
    first = write_figures(first_dir, small_analysis)
    second = write_figures(second_dir, small_analysis)
    for left, right in zip(sorted(first), sorted(second), strict=True):
        assert left.read_bytes() == right.read_bytes(), left.name


def test_figures_carry_their_graph_only_disclaimer(tmp_path, small_analysis):
    """No figure may imply an LLM or task-performance claim."""
    for path in write_figures(tmp_path, small_analysis):
        if path.suffix != ".svg":
            continue
        text = path.read_text(encoding="utf-8")
        assert "Graph-only" in text or "graph-only" in text
        # figures 1 and 3 aggregate over seeds; figure 2 names the single
        # seed it plotted. Either way the seed basis must be on the face.
        assert "seed" in text


def test_figure_data_csvs_are_readable_and_nonempty(tmp_path, small_analysis):
    for path in write_figures(tmp_path, small_analysis):
        if path.suffix == ".csv":
            rows = read_csv(path)
            assert rows, path.name
            assert "figure" in rows[0]


def test_every_topology_has_a_distinct_colour():
    assert set(PALETTE) == set(TOPOLOGIES)
    assert len(set(PALETTE.values())) == len(TOPOLOGIES)


# ==========================================================================
# 8. the CLI
# ==========================================================================

def test_cli_week4_runs_on_a_small_grid_without_writing(capsys):
    code = main(
        [
            "week4-analysis",
            "--topologies", "er", "modular",
            "--sizes", "7", "14",
            "--seeds", "0", "1",
            "--no-write",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "Track A Week 4 analysis" in out
    assert "8 instance rows" in out        # 2 topologies x 2 sizes x 2 seeds
    assert "84 node rows" in out           # 2 topologies x 2 seeds x (7+14)
    assert "All structural invariants held." in out
    assert "--no-write: nothing written to disk" in out


def test_cli_node_row_count_matches_the_formula(capsys):
    main(
        [
            "week4-analysis",
            "--topologies", "er",
            "--sizes", "7", "14",
            "--seeds", "0",
            "--only", "positional",
            "--no-write",
        ]
    )
    out = capsys.readouterr().out
    assert "21 node rows (expected 21)" in out   # 1 x 1 x (7+14)


def test_cli_writes_every_expected_file(tmp_path, capsys):
    code = main(
        [
            "week4-analysis",
            "--out", str(tmp_path),
            "--topologies", "er", "modular",
            "--sizes", "7", "14",
            "--seeds", "0", "1",
        ]
    )
    capsys.readouterr()
    assert code == 0
    produced = {p.name for p in tmp_path.iterdir()}
    assert produced == {
        "structural_metrics.csv",
        "condition_summaries.csv",
        "positional_covariates.csv",
        "robustness_curves.csv",
        "robustness_checkpoints.csv",
        "robustness_auc.csv",
        "figure1_structural_metrics.svg",
        "figure2_robustness_curves.svg",
        "figure3_robustness_by_topology.svg",
        "figure_data_1_structural_metrics.csv",
        "figure_data_2_robustness_curves.csv",
        "figure_data_3_robustness_by_topology.csv",
        "manifest.json",
    }
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["week"] == 4
    assert manifest["track"] == "A"
    assert "NOT agent communication success" in manifest["disclaimer"]
    assert "timings_seconds" not in manifest


def test_cli_output_is_byte_reproducible(tmp_path, capsys):
    first, second = tmp_path / "a", tmp_path / "b"
    for target in (first, second):
        main(
            [
                "week4-analysis",
                "--out", str(target),
                "--topologies", "er",
                "--sizes", "14",
                "--seeds", "0", "1",
            ]
        )
    capsys.readouterr()
    for left in sorted(first.iterdir()):
        right = second / left.name
        assert left.read_bytes() == right.read_bytes(), left.name


@pytest.mark.parametrize(
    ("command", "expected", "absent"),
    [
        ("structural-analysis", "[structural]", "[positional]"),
        ("positional-covariates", "[positional]", "[structural]"),
        ("robustness-curves", "[robustness]", "[positional]"),
    ],
)
def test_focused_cli_aliases_run_only_their_stage(command, expected, absent, capsys):
    code = main(
        [command, "--topologies", "er", "--sizes", "7", "--seeds", "0", "--no-write"]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert expected in out
    assert absent not in out


def test_cli_reports_the_auc_table(capsys):
    main(
        [
            "robustness-curves",
            "--topologies", "er", "ba",
            "--sizes", "28",
            "--seeds", "0",
            "--no-write",
        ]
    )
    out = capsys.readouterr().out
    assert "Mean normalised AUC" in out
    assert "NOT a task-performance measure" in out
    assert "er / random" in out
    assert "ba / targeted" in out


def test_cli_max_fraction_is_configurable(capsys):
    main(
        [
            "robustness-curves",
            "--topologies", "er",
            "--sizes", "112",
            "--seeds", "0",
            "--max-fraction", "0.1",
            "--no-write",
        ]
    )
    out = capsys.readouterr().out
    # floor(0.1 * 112) = 11, so 12 points per trajectory, 2 trajectories
    assert "24 curve points" in out


def test_cli_figures_stage_implies_its_inputs(capsys, tmp_path):
    code = main(
        [
            "week4-analysis",
            "--out", str(tmp_path),
            "--topologies", "er",
            "--sizes", "7",
            "--seeds", "0",
            "--only", "figures",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    # asking for figures must have pulled in structural and robustness
    assert "[structural]" in out
    assert "[robustness]" in out
    assert (tmp_path / "figure1_structural_metrics.svg").is_file()


def test_cli_rejects_an_unknown_topology():
    with pytest.raises(SystemExit):
        main(["week4-analysis", "--topologies", "not_a_topology"])


def test_robustness_fields_and_auc_fields_are_distinct_tuples():
    assert set(ROBUSTNESS_FIELDS) != set(AUC_FIELDS)
    assert "auc_normalized" in AUC_FIELDS
    assert "largest_component_fraction" in ROBUSTNESS_FIELDS
    assert set(POSITIONAL_FIELDS) & set(STRUCTURAL_FIELDS) == {
        "topology",
        "n",
        "seed",
        "graph_id",
    }


# ==========================================================================
# 9. the programmatic driver (run_week4_analysis)
# ==========================================================================

def test_run_week4_analysis_without_writing():
    """The exported driver must work without touching the filesystem."""
    from neuromesh.graphs import run_week4_analysis

    result = run_week4_analysis(
        out_dir="unused", topologies=("er",), sizes=(7, 14), seeds=(0, 1), write=False
    )
    assert set(result) >= {
        "structural",
        "conditions",
        "positional",
        "robustness",
        "checkpoints",
        "auc",
        "trajectories",
        "manifest",
    }
    assert "written" not in result
    assert len(result["structural"]) == 4            # 1 x 2 sizes x 2 seeds
    assert len(result["conditions"]) == 2            # 1 x 2 sizes
    assert len(result["positional"]) == 1 * 2 * (7 + 14)
    assert len(result["trajectories"]) == 8          # 4 graphs x 2 strategies
    assert len(result["auc"]) == 8
    assert result["manifest"]["row_counts"]["structural"] == 4
    assert result["manifest"]["week"] == 4


def test_run_week4_analysis_writes_and_records_a_manifest(tmp_path):
    from neuromesh.graphs import run_week4_analysis

    result = run_week4_analysis(
        out_dir=tmp_path, topologies=("er", "ba"), sizes=(14,), seeds=(0, 1)
    )
    assert result["written"]
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["row_counts"]["structural"] == 4
    assert manifest["conventions"]["across_seed_std"].startswith("sample")
    assert manifest["conventions"]["robustness_denominator"].startswith("original N")
    assert manifest["conventions"]["undefined_values"].endswith("never 0.0")
    assert "No LLM was involved" in manifest["disclaimer"]
    for name in manifest["files"]:
        assert (tmp_path / name).is_file(), name
    assert (tmp_path / "figure1_structural_metrics.svg").is_file()


def test_run_week4_analysis_can_skip_figures(tmp_path):
    from neuromesh.graphs import run_week4_analysis

    run_week4_analysis(
        out_dir=tmp_path,
        topologies=("er",),
        sizes=(7,),
        seeds=(0,),
        with_figures=False,
    )
    assert (tmp_path / "structural_metrics.csv").is_file()
    assert not (tmp_path / "figure1_structural_metrics.svg").exists()


def test_run_week4_analysis_is_deterministic():
    from neuromesh.graphs import run_week4_analysis

    kwargs = {
        "out_dir": "unused",
        "topologies": ("modular",),
        "sizes": (14,),
        "seeds": (0,),
        "write": False,
    }
    first = run_week4_analysis(**kwargs)
    second = run_week4_analysis(**kwargs)
    for key in ("structural", "conditions", "positional", "robustness", "auc"):
        assert first[key] == second[key], key


# ==========================================================================
# 10. remaining edge cases
# ==========================================================================

def test_role_balance_rejects_an_unknown_role():
    import networkx as nx

    from neuromesh.graphs import role_balance

    graph = nx.path_graph(7)
    nx.set_node_attributes(graph, dict.fromkeys(range(7), "cerebellum"), "role")
    with pytest.raises(GraphParameterError, match="unknown role"):
        role_balance(graph)


def test_input_readout_distance_with_one_role_group_missing():
    import networkx as nx

    from neuromesh.contracts import INPUT_ROLE
    from neuromesh.graphs import input_readout_distance

    graph = nx.path_graph(7)
    # inputs but no readout agents at all
    nx.set_node_attributes(graph, dict.fromkeys(range(7), INPUT_ROLE), "role")
    result = input_readout_distance(graph)
    assert result["input_readout_pairs"] == 0
    assert result["input_readout_mean_distance"] is None
    assert result["input_readout_unreachable_pairs"] is None


def test_trajectory_on_a_single_node_graph():
    import networkx as nx

    from neuromesh.graphs import robustness_trajectory

    trajectory = robustness_trajectory(nx.empty_graph(1), "targeted", max_fraction=1.0)
    points = trajectory["points"]
    assert [p["k_removed"] for p in points] == [0, 1]
    assert points[0]["global_efficiency"] == 0.0
    assert points[0]["largest_component_fraction"] == pytest.approx(1.0)
    # after removing the only node: nothing left
    assert points[1]["n_remaining"] == 0
    assert points[1]["largest_component_size"] == 0
    assert points[1]["largest_component_fraction"] == 0.0
    assert points[1]["n_components"] == 0
    assert points[1]["global_efficiency"] is None
    assert points[1]["global_efficiency_original_n"] == 0.0
    assert points[1]["mean_degree_remaining"] is None


def test_trajectory_removing_every_node():
    import networkx as nx

    from neuromesh.graphs import robustness_trajectory

    trajectory = robustness_trajectory(nx.path_graph(4), "targeted", max_fraction=1.0)
    last = trajectory["points"][-1]
    assert last["k_removed"] == 4
    assert last["n_remaining"] == 0
    assert last["n_components"] == 0
    assert last["largest_component_fraction"] == 0.0
