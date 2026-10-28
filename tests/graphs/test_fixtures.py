"""Tests for fixture writing, loading and round-tripping.

Includes tests against the *committed* fixture set, so a regression that
changes graphs downstream consumers already depend on fails loudly.
"""

from __future__ import annotations

import json

import networkx as nx
import pytest

from neuromesh.graphs import (
    DEFAULT_FIXTURE_SEED,
    POPULATION_SIZES,
    SCHEMA_VERSION,
    TARGET_MEAN_DEGREE,
    TOPOLOGIES,
    GraphParameterError,
    GraphSpec,
    fixture_filename,
    generate_fixture_set,
    generate_from_spec,
    graph_to_record,
    load_fixture,
    load_fixture_set,
    load_graph,
    record_to_graph,
    save_fixture,
    validate_graph,
)

# --------------------------------------------------------------------------
# round-tripping
# --------------------------------------------------------------------------

@pytest.mark.parametrize("topology", TOPOLOGIES)
@pytest.mark.parametrize("n", POPULATION_SIZES)
def test_round_trip_preserves_nodes_and_edges(tmp_path, topology, n):
    graph, meta = generate_from_spec(GraphSpec(topology=topology, n=n, seed=0))
    path = save_fixture(tmp_path / fixture_filename(topology, n, 0), graph, meta)

    reloaded, reloaded_meta = load_fixture(path)
    assert set(reloaded.nodes()) == set(graph.nodes())
    assert {tuple(sorted(e)) for e in reloaded.edges()} == {
        tuple(sorted(e)) for e in graph.edges()
    }
    assert nx.is_isomorphic(reloaded, graph)
    assert reloaded_meta["topology"] == topology
    assert reloaded_meta["n"] == n
    assert reloaded_meta["seed"] == 0
    assert reloaded_meta["generator_version"] == meta.generator_version


def test_record_format_promises(tmp_path):
    graph, meta = generate_from_spec(GraphSpec("ws", n=28, seed=0))
    record = graph_to_record(graph, meta)

    assert record["schema_version"] == SCHEMA_VERSION
    assert record["nodes"] == list(range(28))
    assert all(len(e) == 2 and e[0] < e[1] for e in record["edges"])
    assert record["edges"] == sorted(record["edges"])
    assert len(record["edges"]) == len({tuple(e) for e in record["edges"]})
    # Must survive a real JSON encode/decode cycle.
    assert json.loads(json.dumps(record)) == record


def test_fixture_filename_is_zero_padded():
    assert fixture_filename("er", 7, 0) == "er_n007_seed0.json"
    assert fixture_filename("ba", 112, 3) == "ba_n112_seed3.json"


def test_regenerating_a_fixture_is_byte_identical(tmp_path):
    """No timestamps in graph files, so regeneration must be reproducible."""
    first = tmp_path / "a"
    second = tmp_path / "b"
    generate_fixture_set(out_dir=first, sizes=(14,), write_manifest=False)
    generate_fixture_set(out_dir=second, sizes=(14,), write_manifest=False)

    files = sorted(p.name for p in first.glob("*.json"))
    assert files
    for name in files:
        assert (first / name).read_bytes() == (second / name).read_bytes()


# --------------------------------------------------------------------------
# corrupted fixtures must fail loudly
# --------------------------------------------------------------------------

def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_fixture(tmp_path / "absent.json")


def test_unsupported_schema_version_raises():
    with pytest.raises(GraphParameterError, match="schema_version"):
        record_to_graph({"schema_version": 999, "metadata": {}, "nodes": [], "edges": []})


@pytest.mark.parametrize("missing", ["metadata", "nodes", "edges"])
def test_missing_key_raises(missing):
    record = {
        "schema_version": SCHEMA_VERSION,
        "metadata": {},
        "nodes": [0, 1, 2],
        "edges": [[0, 1], [1, 2]],
    }
    record.pop(missing)
    with pytest.raises(GraphParameterError, match=missing):
        record_to_graph(record)


def test_non_contiguous_nodes_raise():
    with pytest.raises(GraphParameterError, match="consecutive integers"):
        record_to_graph(
            {
                "schema_version": SCHEMA_VERSION,
                "metadata": {},
                "nodes": [0, 1, 5],
                "edges": [[0, 1]],
            }
        )


def test_self_loop_in_fixture_raises():
    with pytest.raises(GraphParameterError, match="self-loop"):
        record_to_graph(
            {
                "schema_version": SCHEMA_VERSION,
                "metadata": {},
                "nodes": [0, 1, 2],
                "edges": [[0, 1], [2, 2]],
            }
        )


def test_duplicate_edge_in_fixture_raises():
    with pytest.raises(GraphParameterError, match="duplicate edge"):
        record_to_graph(
            {
                "schema_version": SCHEMA_VERSION,
                "metadata": {},
                "nodes": [0, 1, 2],
                "edges": [[0, 1], [1, 0]],
            }
        )


def test_malformed_edge_raises():
    with pytest.raises(GraphParameterError, match="malformed edge"):
        record_to_graph(
            {
                "schema_version": SCHEMA_VERSION,
                "metadata": {},
                "nodes": [0, 1, 2],
                "edges": [[0, 1, 2]],
            }
        )


def test_metadata_edge_count_mismatch_raises():
    with pytest.raises(GraphParameterError, match="claims 99 edges"):
        record_to_graph(
            {
                "schema_version": SCHEMA_VERSION,
                "metadata": {"n_edges": 99},
                "nodes": [0, 1, 2],
                "edges": [[0, 1], [1, 2]],
            }
        )


def test_record_to_graph_rejects_non_dict():
    with pytest.raises(GraphParameterError, match="must be a dict"):
        record_to_graph([1, 2, 3])


# --------------------------------------------------------------------------
# fixture set generation
# --------------------------------------------------------------------------

def test_generate_fixture_set_writes_manifest_and_all_combinations(tmp_path):
    manifest = generate_fixture_set(out_dir=tmp_path)

    assert manifest["count"] == len(TOPOLOGIES) * len(POPULATION_SIZES) == 15
    assert (tmp_path / "manifest.json").is_file()
    for entry in manifest["fixtures"]:
        assert (tmp_path / entry["file"]).is_file()

    loaded = load_fixture_set(tmp_path)
    assert len(loaded) == 15
    assert set(loaded) == {
        (t, n, DEFAULT_FIXTURE_SEED) for t in TOPOLOGIES for n in POPULATION_SIZES
    }


def test_load_fixture_set_without_manifest_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="manifest.json"):
        load_fixture_set(tmp_path)


def test_multi_seed_fixture_set(tmp_path):
    manifest = generate_fixture_set(out_dir=tmp_path, sizes=(14,), seeds=(0, 1, 2))
    assert manifest["count"] == 9
    assert len(load_fixture_set(tmp_path)) == 9


# --------------------------------------------------------------------------
# the committed fixture set
# --------------------------------------------------------------------------

def test_committed_fixtures_exist_and_cover_the_matrix(fixture_dir):
    if not (fixture_dir / "manifest.json").is_file():
        pytest.skip(
            f"no committed fixtures at {fixture_dir}; run "
            "'python -m neuromesh.graphs.cli fixtures'"
        )
    loaded = load_fixture_set(fixture_dir)
    assert set(loaded) == {
        (t, n, DEFAULT_FIXTURE_SEED) for t in TOPOLOGIES for n in POPULATION_SIZES
    }


@pytest.mark.parametrize("topology", TOPOLOGIES)
@pytest.mark.parametrize("n", POPULATION_SIZES)
def test_committed_fixture_revalidates(fixture_dir, topology, n):
    path = fixture_dir / fixture_filename(topology, n, DEFAULT_FIXTURE_SEED)
    if not path.is_file():
        pytest.skip(f"no committed fixture at {path}")

    graph, meta = load_fixture(path)
    result = validate_graph(graph, n=n, topology=topology, seed=DEFAULT_FIXTURE_SEED)
    assert result.ok, result.summary()
    assert result.n_edges == 2 * n
    assert result.mean_degree == pytest.approx(TARGET_MEAN_DEGREE)
    assert meta["connected"] is True


@pytest.mark.parametrize("topology", TOPOLOGIES)
@pytest.mark.parametrize("n", POPULATION_SIZES)
def test_committed_fixture_matches_a_fresh_generation(fixture_dir, topology, n):
    """The files on disk must be exactly what the generators produce today."""
    path = fixture_dir / fixture_filename(topology, n, DEFAULT_FIXTURE_SEED)
    if not path.is_file():
        pytest.skip(f"no committed fixture at {path}")

    from_disk, _ = load_fixture(path)
    fresh, _ = generate_from_spec(
        GraphSpec(topology=topology, n=n, seed=DEFAULT_FIXTURE_SEED)
    )
    assert {tuple(sorted(e)) for e in from_disk.edges()} == {
        tuple(sorted(e)) for e in fresh.edges()
    }


def test_load_graph_convenience_api(fixture_dir):
    """The documented convenience call for loading a committed fixture."""
    if not (fixture_dir / fixture_filename("ws", 14, 0)).is_file():
        pytest.skip("no committed fixtures")

    graph, meta = load_graph("ws", 14, fixture_dir=fixture_dir)
    assert graph.number_of_nodes() == 14
    assert meta["mean_degree"] == pytest.approx(TARGET_MEAN_DEGREE)
    assert sorted(graph.neighbors(0))


# --------------------------------------------------------------------------
# every structural metadata field is cross-checked, not trusted
# --------------------------------------------------------------------------

def _record(metadata: dict, edges=((0, 1), (1, 2))) -> dict:
    """A 3-node record whose metadata can be made to lie about it."""
    return {
        "schema_version": SCHEMA_VERSION,
        "metadata": metadata,
        "nodes": [0, 1, 2],
        "edges": [list(e) for e in edges],
    }


@pytest.mark.parametrize(
    "metadata, match",
    [
        ({"n_edges": 99}, "claims 99 edges"),
        ({"n_nodes": 99}, "claims 99 nodes"),
        ({"n": 99}, "claims 99 nodes"),
        ({"mean_degree": 4.0}, "claims mean degree 4.0"),
        ({"n_components": 3}, "claims 3 connected component"),
        ({"connected": False}, "claims connected=False"),
    ],
)
def test_metadata_that_contradicts_the_edge_list_raises(metadata, match):
    """A hand-edited fixture must not be able to advertise what it is not.

    The 3-node path graph here has 2 edges, mean degree 4/3 and one connected
    component. Each case claims otherwise and must be caught at load time
    rather than feeding a mis-described graph into an experiment.
    """
    with pytest.raises(GraphParameterError, match=match):
        record_to_graph(_record(metadata))


def test_truthful_metadata_is_accepted():
    """The checks must not be over-eager: correct metadata still loads."""
    graph, meta = record_to_graph(
        _record(
            {
                "n": 3,
                "n_nodes": 3,
                "n_edges": 2,
                "mean_degree": 4.0 / 3.0,
                "connected": True,
                "n_components": 1,
            }
        )
    )
    assert graph.number_of_nodes() == 3
    assert meta["connected"] is True


def test_partial_metadata_is_still_loadable():
    """Absent fields get fewer checks; they are never invented."""
    graph, meta = record_to_graph(_record({"topology": "er"}))
    assert graph.number_of_edges() == 2
    assert "mean_degree" not in meta


def test_disconnected_fixture_declaring_itself_disconnected_is_accepted():
    """Only *inconsistency* is an error. An honestly-labelled graph loads."""
    record = _record(
        {"connected": False, "n_components": 2}, edges=((0, 1),)
    )
    graph, _ = record_to_graph(record)
    assert nx.number_connected_components(graph) == 2


def test_non_dict_metadata_raises():
    with pytest.raises(GraphParameterError, match="metadata must be a dict"):
        record_to_graph(
            {
                "schema_version": SCHEMA_VERSION,
                "metadata": [1, 2, 3],
                "nodes": [0, 1],
                "edges": [[0, 1]],
            }
        )


@pytest.mark.parametrize("topology", TOPOLOGIES)
@pytest.mark.parametrize("n", POPULATION_SIZES)
def test_committed_fixture_metadata_is_internally_consistent(fixture_dir, topology, n):
    """The shipped files must satisfy the stricter cross-checks unmodified.

    ``load_fixture`` runs them, so reaching this assertion already proves it;
    the explicit comparison is here so a failure says *which* field drifted.
    """
    path = fixture_dir / fixture_filename(topology, n, DEFAULT_FIXTURE_SEED)
    if not path.is_file():
        pytest.skip(f"no committed fixture at {path}")

    graph, meta = load_fixture(path)
    assert meta["n"] == meta["n_nodes"] == graph.number_of_nodes() == n
    assert meta["n_edges"] == graph.number_of_edges()
    assert meta["mean_degree"] == pytest.approx(
        2 * graph.number_of_edges() / graph.number_of_nodes()
    )
    assert meta["n_components"] == nx.number_connected_components(graph) == 1
    assert meta["connected"] is True
