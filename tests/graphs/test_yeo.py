"""Tests for the Yeo 7x7 matrix and the Yeo-SBM topology.

Two kinds of test live here and must not be confused:

* **Synthetic** tests exercise the sampling machinery with hand-written 7x7
  matrices. They verify mechanics only. A synthetic matrix is never brain data
  and the graphs it produces are never empirical results.
* **Empirical** tests run against the derived ``data/yeo/yeo7x7_sc.json``.
  They skip with an explicit reason when that file is absent, because the
  topology is defined by its data and cannot be faked.

The distinction is enforced in the generator too: a caller-supplied matrix sets
``metadata.params["matrix_source"] = "caller_supplied"``, so no downstream log
can mistake a mechanics test for a data-informed run.
"""

from __future__ import annotations

import json

import networkx as nx
import numpy as np
import pytest

from neuromesh.contracts import YEO7
from neuromesh.graphs import (
    N_BLOCKS,
    POPULATION_SIZES,
    SCHAEFER_TO_YEO7,
    TARGET_MEAN_DEGREE,
    GraphParameterError,
    YeoMatrixUnavailableError,
    assign_roles,
    community_profile,
    derive_yeo7_matrix,
    generate_graph,
    load_yeo_matrix,
    mean_degree,
    parse_schaefer_label,
    validate_yeo7_record,
    yeo_sbm,
)
from neuromesh.graphs.yeo import YEO_MATRIX_SCHEMA_VERSION, yeo_matrix_path

SEEDS = (0, 1, 2, 3, 4)


def _synthetic_matrix(within: float = 5.0, between: float = 1.0) -> np.ndarray:
    """A clearly synthetic 7x7 preference matrix. NOT brain data."""
    matrix = np.full((N_BLOCKS, N_BLOCKS), float(between))
    np.fill_diagonal(matrix, float(within))
    return matrix


def _empirical_available() -> bool:
    return yeo_matrix_path().is_file()


requires_empirical = pytest.mark.skipif(
    not _empirical_available(),
    reason=(
        "derived empirical Yeo matrix (data/yeo/yeo7x7_sc.json) is absent; "
        "run 'python -m neuromesh.graphs.cli yeo-derive' with a parcel-level "
        "connectome to produce it"
    ),
)


# --------------------------------------------------------------------------
# label parsing: parcel -> functional network
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "label, expected",
    [
        ("7Networks_LH_Vis_1", "visual"),
        ("7Networks_RH_SomMot_3", "somatomotor"),
        ("7Networks_LH_DorsAttn_Post_2", "dorsal_attention"),
        ("7Networks_RH_SalVentAttn_Med_1", "ventral_attention"),
        ("7Networks_LH_Limbic_OFC_1", "limbic"),
        ("7Networks_RH_Cont_Par_1", "frontoparietal"),
        ("7Networks_LH_Default_PFC_7", "default"),
    ],
)
def test_schaefer_labels_map_to_the_contract_roles(label, expected):
    assert parse_schaefer_label(label) == expected


def test_every_schaefer_abbreviation_maps_to_a_contract_role():
    """The mapping must be a bijection onto YEO7, or a block would be lost."""
    assert sorted(SCHAEFER_TO_YEO7.values()) == sorted(YEO7)
    assert len(SCHAEFER_TO_YEO7) == len(YEO7) == N_BLOCKS


@pytest.mark.parametrize(
    "bad", ["Vis_1", "17Networks_LH_Vis_1", "7Networks_LH", "", "7Networks_XX_Vis_1"]
)
def test_malformed_labels_are_rejected(bad):
    with pytest.raises(GraphParameterError, match="7Networks|unknown network"):
        parse_schaefer_label(bad)


def test_unknown_network_token_is_rejected():
    with pytest.raises(GraphParameterError, match="unknown network"):
        parse_schaefer_label("7Networks_LH_Cerebellum_1")


def test_non_string_label_is_rejected():
    with pytest.raises(GraphParameterError, match="must be a string"):
        parse_schaefer_label(42)


# --------------------------------------------------------------------------
# derivation, against a small synthetic connectome
# --------------------------------------------------------------------------

def _write_synthetic_connectome(tmp_path, n_per_network=2, value=3.0):
    """A tiny, obviously synthetic parcel-level connectome for mechanics tests."""
    labels = [
        f"7Networks_LH_{abbr}_{i + 1}"
        for abbr in SCHAEFER_TO_YEO7
        for i in range(n_per_network)
    ]
    size = len(labels)
    matrix = np.full((size, size), value)
    np.fill_diagonal(matrix, 0.0)
    matrix_path = tmp_path / "m.csv"
    labels_path = tmp_path / "l.csv"
    np.savetxt(matrix_path, matrix, delimiter=",")
    labels_path.write_text(",".join(labels), encoding="utf-8")
    return matrix_path, labels_path


def test_derivation_produces_a_valid_record(tmp_path):
    matrix_path, labels_path = _write_synthetic_connectome(tmp_path)
    record = derive_yeo7_matrix(matrix_path, labels_path)

    assert record["schema_version"] == YEO_MATRIX_SCHEMA_VERSION
    assert record["networks"] == list(YEO7)
    assert np.array(record["matrix"]).shape == (N_BLOCKS, N_BLOCKS)
    assert record["provenance"]["n_parcels"] == 14
    assert record["provenance"]["parcels_per_network"] == [2] * 7
    validate_yeo7_record(record)


def test_derivation_is_a_plain_mean_over_parcel_pairs(tmp_path):
    """A constant off-diagonal connectome must give a constant block matrix."""
    matrix_path, labels_path = _write_synthetic_connectome(tmp_path, value=3.0)
    matrix = np.array(derive_yeo7_matrix(matrix_path, labels_path)["matrix"])

    # Every between-network parcel pair is 3.0, so every off-diagonal block is 3.0.
    off = matrix[~np.eye(N_BLOCKS, dtype=bool)]
    assert np.allclose(off, 3.0)
    # Within a 2-parcel network the single off-diagonal pair is also 3.0.
    assert np.allclose(np.diag(matrix), 3.0)


def test_derivation_uses_the_mean_not_the_sum(tmp_path):
    """Unequal network sizes must not leak into the block values.

    A sum would make a large network's blocks larger purely because it has more
    parcels; the mean must be invariant to that.
    """
    labels = (
        [f"7Networks_LH_Vis_{i}" for i in range(1, 11)]
        + [f"7Networks_LH_{a}_{i}" for a in list(SCHAEFER_TO_YEO7)[1:] for i in (1, 2)]
    )
    size = len(labels)
    matrix = np.full((size, size), 2.0)
    np.fill_diagonal(matrix, 0.0)
    np.savetxt(tmp_path / "m.csv", matrix, delimiter=",")
    (tmp_path / "l.csv").write_text(",".join(labels), encoding="utf-8")

    record = derive_yeo7_matrix(tmp_path / "m.csv", tmp_path / "l.csv")
    block = np.array(record["matrix"])
    assert record["provenance"]["parcels_per_network"] == [10, 2, 2, 2, 2, 2, 2]
    assert np.allclose(block, 2.0), "block means must not scale with parcel count"


def test_single_parcel_network_is_refused_rather_than_guessed(tmp_path):
    """A one-parcel network has no within-network pair, so its mean is undefined.

    Substituting 0.0 would assert an absence of connectivity that the data does
    not report, so the derivation refuses instead.
    """
    labels = [f"7Networks_LH_{a}_1" for a in SCHAEFER_TO_YEO7]
    matrix = np.full((7, 7), 2.0)
    np.fill_diagonal(matrix, 0.0)
    np.savetxt(tmp_path / "m.csv", matrix, delimiter=",")
    (tmp_path / "l.csv").write_text(",".join(labels), encoding="utf-8")

    with pytest.raises(GraphParameterError, match="within-network mean is undefined"):
        derive_yeo7_matrix(tmp_path / "m.csv", tmp_path / "l.csv")


def test_derivation_rejects_an_asymmetric_source(tmp_path):
    matrix_path, labels_path = _write_synthetic_connectome(tmp_path)
    matrix = np.loadtxt(matrix_path, delimiter=",")
    matrix[0, 1] = 99.0
    np.savetxt(matrix_path, matrix, delimiter=",")

    with pytest.raises(GraphParameterError, match="not symmetric"):
        derive_yeo7_matrix(matrix_path, labels_path)


def test_derivation_rejects_a_label_count_mismatch(tmp_path):
    matrix_path, labels_path = _write_synthetic_connectome(tmp_path)
    labels_path.write_text("7Networks_LH_Vis_1,7Networks_LH_Vis_2", encoding="utf-8")

    with pytest.raises(GraphParameterError, match="labels were read"):
        derive_yeo7_matrix(matrix_path, labels_path)


def test_derivation_rejects_missing_networks(tmp_path):
    """A connectome that does not cover all seven networks must not aggregate."""
    labels = [f"7Networks_LH_Vis_{i}" for i in range(1, 5)]
    matrix = np.full((4, 4), 1.0)
    np.fill_diagonal(matrix, 0.0)
    np.savetxt(tmp_path / "m.csv", matrix, delimiter=",")
    (tmp_path / "l.csv").write_text(",".join(labels), encoding="utf-8")

    with pytest.raises(GraphParameterError, match="do not cover all seven networks"):
        derive_yeo7_matrix(tmp_path / "m.csv", tmp_path / "l.csv")


def test_derivation_rejects_a_non_square_source(tmp_path):
    _, labels_path = _write_synthetic_connectome(tmp_path)
    np.savetxt(tmp_path / "bad.csv", np.ones((4, 6)), delimiter=",")

    with pytest.raises(GraphParameterError, match="square matrix"):
        derive_yeo7_matrix(tmp_path / "bad.csv", labels_path)


def test_derivation_reports_a_missing_source_file(tmp_path):
    with pytest.raises(YeoMatrixUnavailableError, match="no such source file"):
        derive_yeo7_matrix(tmp_path / "absent.csv", tmp_path / "also_absent.csv")


# --------------------------------------------------------------------------
# record validation and loading
# --------------------------------------------------------------------------

def _valid_record() -> dict:
    return {
        "schema_version": YEO_MATRIX_SCHEMA_VERSION,
        "networks": list(YEO7),
        "matrix": _synthetic_matrix().tolist(),
        "provenance": {"source": "synthetic test fixture"},
    }


def test_valid_record_passes_validation():
    matrix = validate_yeo7_record(_valid_record())
    assert matrix.shape == (N_BLOCKS, N_BLOCKS)


@pytest.mark.parametrize("missing", ["networks", "matrix", "provenance"])
def test_record_missing_a_required_key_is_rejected(missing):
    record = _valid_record()
    record.pop(missing)
    with pytest.raises(GraphParameterError, match=missing):
        validate_yeo7_record(record)


def test_record_with_an_unsupported_schema_version_is_rejected():
    record = _valid_record()
    record["schema_version"] = 999
    with pytest.raises(GraphParameterError, match="schema_version"):
        validate_yeo7_record(record)


def test_record_with_reordered_networks_is_rejected():
    """Row order *is* the block-to-network mapping; a permutation is a defect."""
    record = _valid_record()
    record["networks"] = list(reversed(YEO7))
    with pytest.raises(GraphParameterError, match="YEO7 roles in order"):
        validate_yeo7_record(record)


def test_record_with_a_wrong_sized_matrix_is_rejected():
    record = _valid_record()
    record["matrix"] = np.ones((3, 3)).tolist()
    with pytest.raises(GraphParameterError, match="7x7"):
        validate_yeo7_record(record)


def test_record_with_negative_values_is_rejected():
    record = _valid_record()
    record["matrix"][0][0] = -1.0
    with pytest.raises(GraphParameterError, match="non-negative"):
        validate_yeo7_record(record)


def test_non_dict_record_is_rejected():
    with pytest.raises(GraphParameterError, match="must be a dict"):
        validate_yeo7_record([1, 2, 3])


def test_loading_an_absent_matrix_raises_a_specific_error(tmp_path):
    with pytest.raises(YeoMatrixUnavailableError, match="no derived Yeo matrix"):
        load_yeo_matrix(tmp_path / "nothing_here.json")


def test_unavailable_error_is_a_file_not_found_error():
    assert issubclass(YeoMatrixUnavailableError, FileNotFoundError)


def test_round_trip_through_a_written_file(tmp_path):
    path = tmp_path / "yeo.json"
    path.write_text(json.dumps(_valid_record()), encoding="utf-8")

    matrix, record = load_yeo_matrix(path)
    assert matrix.shape == (N_BLOCKS, N_BLOCKS)
    assert record["networks"] == list(YEO7)


# --------------------------------------------------------------------------
# Yeo-SBM sampling: synthetic matrices (mechanics only)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("n", POPULATION_SIZES)
@pytest.mark.parametrize("seed", SEEDS)
def test_synthetic_yeo_sbm_is_valid_and_degree_matched(n, seed):
    graph, meta = yeo_sbm(n, seed=seed, matrix=_synthetic_matrix())

    assert set(graph.nodes()) == set(range(n))
    assert graph.number_of_edges() == 2 * n
    assert mean_degree(graph) == pytest.approx(TARGET_MEAN_DEGREE)
    assert nx.is_connected(graph)
    assert nx.number_of_selfloops(graph) == 0
    assert not graph.is_multigraph()
    assert meta.topology == "yeo_sbm"


def test_caller_supplied_matrix_is_flagged_as_not_empirical():
    """A mechanics test must never be mistakable for a data-informed run."""
    _, meta = yeo_sbm(28, seed=0, matrix=_synthetic_matrix())

    assert meta.params["matrix_source"] == "caller_supplied"
    assert "NOT empirical" in meta.params["matrix_provenance"]["source"]


@pytest.mark.parametrize("n", POPULATION_SIZES)
def test_synthetic_yeo_sbm_is_reproducible(n):
    first, _ = yeo_sbm(n, seed=3, matrix=_synthetic_matrix())
    second, _ = yeo_sbm(n, seed=3, matrix=_synthetic_matrix())

    assert {tuple(sorted(e)) for e in first.edges()} == {
        tuple(sorted(e)) for e in second.edges()
    }


def test_block_preferences_drive_the_realised_structure():
    """A matrix favouring one block pair must put more edges there.

    Sampled without the connectivity condition: concentrating the budget on one
    block pair starves the rest of the graph, so the unconditioned draw is the
    one that isolates the effect of the weights.
    """
    weights = np.ones((N_BLOCKS, N_BLOCKS))
    weights[0, 1] = weights[1, 0] = 50.0
    graph, _ = yeo_sbm(112, seed=0, matrix=weights, ensure_connected=False)

    communities = nx.get_node_attributes(graph, "community")

    def edges_between(x, y):
        return sum(
            1 for u, v in graph.edges() if {communities[u], communities[v]} == {x, y}
        )

    assert edges_between(0, 1) > edges_between(2, 3) * 3


def test_an_extreme_matrix_that_cannot_connect_fails_loudly():
    """Starving most block pairs makes a connected graph unreachable.

    The generator must say so rather than return a disconnected graph while
    claiming ``ensure_connected=True``.
    """
    weights = np.ones((N_BLOCKS, N_BLOCKS))
    weights[0, 1] = weights[1, 0] = 50.0

    with pytest.raises(RuntimeError, match="no connected graph found"):
        yeo_sbm(112, seed=0, matrix=weights, max_attempts=25)


def test_yeo_sbm_rejects_an_invalid_matrix():
    with pytest.raises(GraphParameterError, match="7x7"):
        yeo_sbm(28, seed=0, matrix=np.ones((5, 5)))


def test_yeo_sbm_rejects_an_asymmetric_matrix():
    weights = _synthetic_matrix()
    weights[0, 1] = 99.0
    with pytest.raises(GraphParameterError, match="symmetric"):
        yeo_sbm(28, seed=0, matrix=weights)


def test_yeo_sbm_without_a_matrix_file_fails_loudly(tmp_path):
    """The production path must never silently substitute invented weights."""
    with pytest.raises(YeoMatrixUnavailableError):
        yeo_sbm(28, seed=0, matrix_path=tmp_path / "absent.json")


# --------------------------------------------------------------------------
# Yeo-SBM against the derived empirical matrix
# --------------------------------------------------------------------------

@requires_empirical
def test_empirical_matrix_loads_and_validates():
    matrix, record = load_yeo_matrix()

    assert matrix.shape == (N_BLOCKS, N_BLOCKS)
    assert np.allclose(matrix, matrix.T)
    assert (matrix >= 0).all()
    assert np.isfinite(matrix).all()
    assert record["networks"] == list(YEO7)


@requires_empirical
def test_empirical_matrix_records_its_provenance():
    """A derived file without provenance is unusable in a methods section."""
    _, record = load_yeo_matrix()
    provenance = record["provenance"]

    for key in (
        "source",
        "parcellation",
        "modality",
        "aggregation",
        "licence",
        "citation",
        "n_parcels",
        "parcels_per_network",
    ):
        assert provenance.get(key), f"provenance is missing {key!r}"
    assert provenance["modality"] == "structural", (
        "the design specifies structural connectivity; functional connectivity "
        "is partly circular with the Yeo parcellation"
    )


@requires_empirical
def test_empirical_matrix_retains_the_bsd_licence_notice():
    """Naming the licence is not enough to satisfy it.

    BSD-3-Clause requires redistributions to retain the copyright notice, the
    list of conditions and the disclaimer. The notice therefore travels inside
    the derived file rather than only being referenced from documentation.
    """
    _, record = load_yeo_matrix()
    provenance = record["provenance"]

    notice = provenance.get("licence_notice", "")
    assert "Copyright (c) 2020, saratheriver" in notice
    assert "BSD 3-Clause" in notice
    assert "DISCLAIMED" in notice
    assert provenance.get("licence_url", "").startswith("https://github.com/MICA-MNI/ENIGMA")


@requires_empirical
def test_empirical_matrix_has_within_network_preference():
    """The derived data must actually show the structure the design assumes."""
    matrix, _ = load_yeo_matrix()
    diagonal = np.diag(matrix).mean()
    off_diagonal = matrix[~np.eye(N_BLOCKS, dtype=bool)].mean()

    assert diagonal > off_diagonal, (
        "within-network structural connectivity should exceed between-network"
    )


@requires_empirical
def test_empirical_matrix_is_heterogeneous_between_networks():
    """A flat matrix would make Yeo-SBM indistinguishable from modular."""
    matrix, _ = load_yeo_matrix()
    off_diagonal = matrix[~np.eye(N_BLOCKS, dtype=bool)]

    assert off_diagonal.std() > 0.1 * off_diagonal.mean()


@requires_empirical
@pytest.mark.parametrize("n", POPULATION_SIZES)
@pytest.mark.parametrize("seed", SEEDS)
def test_empirical_yeo_sbm_is_valid_and_degree_matched(n, seed):
    graph, meta = yeo_sbm(n, seed=seed)

    assert set(graph.nodes()) == set(range(n))
    assert graph.number_of_edges() == 2 * n
    assert mean_degree(graph) == pytest.approx(TARGET_MEAN_DEGREE)
    assert meta.degree_matched_exactly is True
    assert nx.is_connected(graph)
    assert meta.params["matrix_source"] == "empirical"


@requires_empirical
@pytest.mark.parametrize("n", POPULATION_SIZES)
def test_empirical_yeo_sbm_blocks_map_to_the_seven_roles(n):
    graph = assign_roles(generate_graph("yeo_sbm", n, seed=0), 0)

    for _, data in graph.nodes(data=True):
        assert data["role"] == YEO7[data["community"]]


@requires_empirical
@pytest.mark.parametrize("n", [28, 56, 112])
def test_empirical_yeo_sbm_shows_community_structure(n):
    profile = community_profile(generate_graph("yeo_sbm", n, seed=0))

    assert profile["n_communities"] == N_BLOCKS
    assert profile["modularity"] > 0.0


@requires_empirical
def test_empirical_yeo_sbm_at_n_seven_has_no_within_block_edges():
    """The N = 7 boundary case applies to the empirical topology too."""
    _, meta = yeo_sbm(7, seed=0)

    assert meta.params["n_within_edges"] == 0
    assert meta.params["within_structure_possible"] is False


@requires_empirical
def test_empirical_matrix_is_reachable_regardless_of_working_directory(tmp_path, monkeypatch):
    """Loading must not depend on where the process happens to be running."""
    monkeypatch.chdir(tmp_path)
    matrix, _ = load_yeo_matrix()
    assert matrix.shape == (N_BLOCKS, N_BLOCKS)
