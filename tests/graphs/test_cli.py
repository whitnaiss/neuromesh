"""Tests for the Track A command-line entry points.

The CLI is how the fixture set and the derived Yeo matrix are produced, so an
unnoticed break here silently stops the study's data being reproducible. These
tests exercise each subcommand end to end in a temporary directory; none of
them writes to the committed data.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from neuromesh.contracts import YEO7
from neuromesh.graphs import POPULATION_SIZES, SCHAEFER_TO_YEO7, TOPOLOGIES
from neuromesh.graphs.cli import main


def _synthetic_connectome(tmp_path, n_per_network=3, value=2.0):
    """A small, obviously synthetic parcel-level connectome. NOT brain data."""
    labels = [
        f"7Networks_LH_{abbr}_{i + 1}"
        for abbr in SCHAEFER_TO_YEO7
        for i in range(n_per_network)
    ]
    size = len(labels)
    matrix = np.full((size, size), value)
    np.fill_diagonal(matrix, 0.0)
    matrix_path = tmp_path / "matrix.csv"
    labels_path = tmp_path / "labels.csv"
    np.savetxt(matrix_path, matrix, delimiter=",")
    labels_path.write_text(",".join(labels), encoding="utf-8")
    return matrix_path, labels_path


# --------------------------------------------------------------------------
# validate
# --------------------------------------------------------------------------

def test_validate_reports_success_for_every_canonical_row(capsys):
    assert main(["validate"]) == 0

    out = capsys.readouterr().out
    assert "Canonical generators" in out
    # 5 topologies x 5 sizes must all be eligible for the matched-degree sweep
    assert "Sweep-eligible rows: 25/25" in out
    assert "FAILURES" not in out


def test_validate_still_reports_the_known_model_exceptions(capsys):
    main(["validate"])
    out = capsys.readouterr().out

    assert "Diagnostics" in out
    assert "EXCEPTION" in out
    assert "Barabasi-Albert" in out


def test_validate_accepts_a_seed(capsys):
    assert main(["validate", "--seed", "3"]) == 0
    assert "(seed=3)" in capsys.readouterr().out


# --------------------------------------------------------------------------
# fixtures / verify-fixtures
# --------------------------------------------------------------------------

def test_fixtures_writes_the_whole_matrix(tmp_path, capsys):
    assert main(["fixtures", "--out", str(tmp_path)]) == 0

    written = sorted(p.name for p in tmp_path.glob("*.json"))
    assert "manifest.json" in written
    assert len(written) == len(TOPOLOGIES) * len(POPULATION_SIZES) + 1 == 26
    assert "Wrote 25 fixtures" in capsys.readouterr().out


def test_fixtures_accepts_multiple_seeds(tmp_path):
    assert main(["fixtures", "--out", str(tmp_path), "--seeds", "0", "1"]) == 0

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["count"] == len(TOPOLOGIES) * len(POPULATION_SIZES) * 2


def test_verify_fixtures_passes_on_a_freshly_written_set(tmp_path, capsys):
    main(["fixtures", "--out", str(tmp_path)])
    capsys.readouterr()

    assert main(["verify-fixtures", "--out", str(tmp_path)]) == 0
    assert "25/25 fixtures reloaded and revalidated" in capsys.readouterr().out


def test_verify_fixtures_fails_on_a_corrupted_fixture(tmp_path, capsys):
    """A tampered fixture must make the command exit non-zero, not pass quietly."""
    main(["fixtures", "--out", str(tmp_path)])
    capsys.readouterr()

    target = tmp_path / "er_n014_seed0.json"
    record = json.loads(target.read_text(encoding="utf-8"))
    record["edges"] = record["edges"][:-1]  # drop an edge, leaving metadata lying
    target.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(Exception, match="n_edges|claims"):
        main(["verify-fixtures", "--out", str(tmp_path)])


# --------------------------------------------------------------------------
# yeo-derive / yeo-show
# --------------------------------------------------------------------------

def test_yeo_derive_writes_a_loadable_record(tmp_path, capsys):
    matrix_path, labels_path = _synthetic_connectome(tmp_path)
    out = tmp_path / "yeo7x7.json"

    assert (
        main(
            [
                "yeo-derive",
                "--matrix",
                str(matrix_path),
                "--labels",
                str(labels_path),
                "--out",
                str(out),
                "--source-url",
                "https://example.invalid/synthetic",
                "--notes",
                "synthetic test fixture",
            ]
        )
        == 0
    )

    record = json.loads(out.read_text(encoding="utf-8"))
    assert record["networks"] == list(YEO7)
    assert np.array(record["matrix"]).shape == (7, 7)
    assert record["provenance"]["notes"] == "synthetic test fixture"
    assert record["provenance"]["source_urls"] == ["https://example.invalid/synthetic"]
    assert "Wrote" in capsys.readouterr().out


def test_yeo_derive_records_source_hashes(tmp_path):
    """Provenance must pin the exact inputs the matrix was derived from."""
    matrix_path, labels_path = _synthetic_connectome(tmp_path)
    out = tmp_path / "yeo7x7.json"
    main(
        ["yeo-derive", "--matrix", str(matrix_path), "--labels", str(labels_path),
         "--out", str(out)]
    )

    hashes = json.loads(out.read_text(encoding="utf-8"))["provenance"]["source_sha256"]
    assert set(hashes) == {"matrix.csv", "labels.csv"}
    assert all(len(h) == 64 for h in hashes.values())


def test_yeo_show_prints_and_revalidates(tmp_path, capsys):
    matrix_path, labels_path = _synthetic_connectome(tmp_path)
    out = tmp_path / "yeo7x7.json"
    main(["yeo-derive", "--matrix", str(matrix_path), "--labels", str(labels_path),
          "--out", str(out)])
    capsys.readouterr()

    assert main(["yeo-show", "--path", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "Loaded and validated" in printed
    for name in YEO7:
        assert name[:8] in printed


def test_yeo_show_reports_an_absent_matrix_without_crashing(tmp_path, capsys):
    assert main(["yeo-show", "--path", str(tmp_path / "absent.json")]) == 1
    assert "Yeo matrix unavailable" in capsys.readouterr().out


def test_yeo_derive_propagates_a_derivation_failure(tmp_path):
    """A connectome missing networks must fail the command, not write a file."""
    labels = ["7Networks_LH_Vis_1", "7Networks_LH_Vis_2"]
    matrix = np.array([[0.0, 1.0], [1.0, 0.0]])
    np.savetxt(tmp_path / "m.csv", matrix, delimiter=",")
    (tmp_path / "l.csv").write_text(",".join(labels), encoding="utf-8")
    out = tmp_path / "never_written.json"

    with pytest.raises(Exception, match="do not cover all seven networks"):
        main(["yeo-derive", "--matrix", str(tmp_path / "m.csv"),
              "--labels", str(tmp_path / "l.csv"), "--out", str(out)])
    assert not out.exists()


# --------------------------------------------------------------------------
# parser
# --------------------------------------------------------------------------

def test_a_subcommand_is_required():
    with pytest.raises(SystemExit):
        main([])


def test_unknown_subcommand_is_rejected():
    with pytest.raises(SystemExit):
        main(["not-a-command"])
