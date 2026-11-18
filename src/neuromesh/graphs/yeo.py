"""The 7x7 Yeo-network structural connectivity matrix: derivation and loading.

The Yeo-SBM topology needs one number per pair of functional networks: how
strongly the two are structurally connected. No published, citable 7x7
Yeo-network structural matrix exists, so the matrix is **derived** here from a
parcel-level group-average connectome plus the Yeo label of each parcel. The
derivation is a computation this module owns, documents and tests; it is not a
download, and the aggregation rule below is a methodological choice that must
be stated wherever the matrix is used.

Source
------
ENIGMA Toolbox group-average Human Connectome Project connectome, Schaefer-200
parcellation in ``7Networks`` order:

* ``strucMatrix_ctx_schaefer_200.csv`` -- 200x200 cortico-cortical structural
  connectivity, from diffusion-weighted tractography (MRtrix3, anatomically
  constrained, SIFT2), **distance-dependent thresholded and log-transformed**
  by the providers.
* ``strucLabels_ctx_schaefer_200.csv`` -- the 200 parcel names, each encoding
  its Yeo network, e.g. ``7Networks_LH_Vis_1``.

Both are redistributed by the ENIGMA Toolbox under the BSD 3-Clause licence.

Two consequences of the providers' preprocessing carry into any write-up:

1. Entries are **log-transformed** streamline counts, so block means are means
   of transformed values, not of raw counts. The transform is non-linear and
   not inverted here.
2. Thresholding leaves most parcel pairs at exactly zero (about 88% of them),
   so a block mean mixes connection *strength* with connection *density*.
   :func:`derive_yeo7_matrix` therefore records the per-block density
   alongside the mean, so the two can be told apart.

Structural, not functional
--------------------------
``strucMatrix`` is used, never ``funcMatrix``. Functional connectivity is
partly circular here: the Yeo networks were themselves defined by clustering
resting-state functional connectivity, so an FC-derived block matrix would
largely re-encode the parcellation rather than provide independent structural
evidence.
"""

from __future__ import annotations

import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from neuromesh.contracts import YEO7

from .spec import GraphParameterError

__all__ = [
    "DEFAULT_YEO_MATRIX_PATH",
    "SCHAEFER_TO_YEO7",
    "YEO_MATRIX_SCHEMA_VERSION",
    "YeoMatrixUnavailableError",
    "derive_yeo7_matrix",
    "load_yeo_matrix",
    "parse_schaefer_label",
    "validate_yeo7_record",
    "yeo_matrix_path",
]

YEO_MATRIX_SCHEMA_VERSION = 1

#: Schaefer ``7Networks`` abbreviations mapped to the project's role names.
#: ``Cont`` is the control network, which Yeo et al. (2011) and this project
#: both call frontoparietal; ``SalVentAttn`` is the salience / ventral
#: attention network.
SCHAEFER_TO_YEO7: dict[str, str] = {
    "Vis": "visual",
    "SomMot": "somatomotor",
    "DorsAttn": "dorsal_attention",
    "SalVentAttn": "ventral_attention",
    "Limbic": "limbic",
    "Cont": "frontoparietal",
    "Default": "default",
}

#: Location of the derived matrix, relative to the repository root.
DEFAULT_YEO_MATRIX_PATH = Path("data") / "yeo" / "yeo7x7_sc.json"

#: BSD-3-Clause requires that redistributions retain the copyright notice, the
#: list of conditions and the disclaimer. Naming the licence is not enough, so
#: the notice travels inside every derived matrix file.
ENIGMA_LICENCE_NOTICE = (
    "Derived from data redistributed by the ENIGMA Toolbox under the "
    "BSD 3-Clause License. Copyright (c) 2020, saratheriver. All rights "
    "reserved. Redistribution and use in source and binary forms, with or "
    "without modification, are permitted provided that the conditions of the "
    "BSD 3-Clause License are met, including retention of this copyright "
    "notice, the list of conditions and the following disclaimer. THIS "
    "SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS \"AS IS\" "
    "AND ANY EXPRESS OR IMPLIED WARRANTIES ARE DISCLAIMED. See the full "
    "licence text at the URL in 'licence_url'. Neither the name of the "
    "copyright holder nor the names of its contributors endorse this project."
)

#: Where the full, unabridged licence text lives.
ENIGMA_LICENCE_URL = "https://github.com/MICA-MNI/ENIGMA/blob/master/LICENSE"

_LABEL_PATTERN = re.compile(r"^7Networks_(LH|RH)_([A-Za-z]+)_")


class YeoMatrixUnavailableError(FileNotFoundError):
    """Raised when the empirical 7x7 matrix is not present.

    Subclasses :class:`FileNotFoundError` so ordinary file handling still
    catches it. It is deliberately distinct from
    :class:`~neuromesh.graphs.spec.GraphParameterError`: a missing matrix is a
    missing *dependency*, not a bad argument, and the Yeo-SBM generator must
    fail loudly rather than fall back to invented numbers.
    """


def _repository_root() -> Path:
    """Best-effort repository root, for locating committed data files.

    Resolved from this file's location (``<root>/src/neuromesh/graphs/yeo.py``)
    rather than from the process working directory, so that loading the matrix
    does not depend on where the caller happens to be running.
    """
    return Path(__file__).resolve().parents[3]


def yeo_matrix_path(path: str | Path | None = None) -> Path:
    """Resolve the matrix location, preferring an explicit path.

    With no argument, the committed location is tried relative to the
    repository root first and the working directory second.
    """
    if path is not None:
        return Path(path)
    candidate = _repository_root() / DEFAULT_YEO_MATRIX_PATH
    return candidate if candidate.is_file() else DEFAULT_YEO_MATRIX_PATH


# --------------------------------------------------------------------------
# label parsing
# --------------------------------------------------------------------------

def parse_schaefer_label(label: str) -> str:
    """Return the project role name for one Schaefer ``7Networks`` parcel label.

    >>> parse_schaefer_label("7Networks_LH_Cont_Par_1")
    'frontoparietal'

    A label that does not match the expected form, or that names a network
    outside :data:`SCHAEFER_TO_YEO7`, raises rather than being dropped: a
    silently unassigned parcel would corrupt every block it belongs to.
    """
    if not isinstance(label, str):
        raise GraphParameterError(
            f"parcel label must be a string, got {type(label).__name__}"
        )
    match = _LABEL_PATTERN.match(label.strip())
    if match is None:
        raise GraphParameterError(
            f"parcel label {label!r} is not in Schaefer 7Networks form "
            "'7Networks_<LH|RH>_<Network>_...'"
        )
    token = match.group(2)
    if token not in SCHAEFER_TO_YEO7:
        raise GraphParameterError(
            f"parcel label {label!r} names unknown network {token!r}; "
            f"expected one of {sorted(SCHAEFER_TO_YEO7)}"
        )
    return SCHAEFER_TO_YEO7[token]


# --------------------------------------------------------------------------
# derivation
# --------------------------------------------------------------------------

def derive_yeo7_matrix(
    matrix_csv: str | Path,
    labels_csv: str | Path,
    *,
    source: str = "ENIGMA Toolbox group-average HCP connectome",
    parcellation: str = "Schaefer2018_200Parcels_7Networks",
    modality: str = "structural",
    source_urls: list[str] | None = None,
    source_sha256: dict[str, str] | None = None,
    licence: str = "BSD-3-Clause (ENIGMA Toolbox)",
    licence_notice: str = ENIGMA_LICENCE_NOTICE,
    licence_url: str = ENIGMA_LICENCE_URL,
    notes: str | None = None,
) -> dict:
    """Aggregate a parcel-level connectome into the 7x7 block matrix record.

    Aggregation rule, stated explicitly because different defensible rules give
    different matrices: block ``(A, B)`` is the **arithmetic mean of the matrix
    entries over all parcel pairs** ``i in A``, ``j in B``, excluding the
    parcel self-pairs when ``A == B``. The mean is required rather than the sum
    because the seven networks contain unequal numbers of parcels (12 to 46 at
    Schaefer-200), so sums would mostly re-encode network size.

    Thresholded-away pairs contribute their zeros to the mean, so each block
    value blends strength and density. The per-block density is recorded
    separately in ``provenance["block_density"]`` so the two can be separated
    by anyone reading the file.

    Every structural claim is verified rather than assumed: squareness,
    symmetry, finiteness, label count, label parseability, presence of all
    seven networks, and non-negativity of the resulting block means.
    """
    matrix_csv, labels_csv = Path(matrix_csv), Path(labels_csv)
    for path in (matrix_csv, labels_csv):
        if not path.is_file():
            raise YeoMatrixUnavailableError(f"no such source file: {path}")

    with open(labels_csv, newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.reader(handle) if row]
    if not rows:
        raise GraphParameterError(f"{labels_csv} contains no labels")
    labels = [cell.strip() for cell in rows[0] if cell.strip()]

    matrix = np.loadtxt(matrix_csv, delimiter=",")
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise GraphParameterError(
            f"{matrix_csv} must hold a square matrix, got shape {matrix.shape}"
        )
    if matrix.shape[0] != len(labels):
        raise GraphParameterError(
            f"matrix is {matrix.shape[0]}x{matrix.shape[0]} but {len(labels)} "
            "labels were read; the two files must describe the same parcels"
        )
    if not np.isfinite(matrix).all():
        raise GraphParameterError(f"{matrix_csv} contains non-finite entries")
    symmetric = bool(np.allclose(matrix, matrix.T, rtol=1e-9, atol=1e-12))
    if not symmetric:
        raise GraphParameterError(
            f"{matrix_csv} is not symmetric; structural connectivity between a "
            "pair of parcels must not depend on their order"
        )

    networks = list(YEO7)
    index_of = {name: i for i, name in enumerate(networks)}
    membership = np.array([index_of[parse_schaefer_label(label)] for label in labels])

    present = sorted({int(x) for x in membership})
    if present != list(range(len(networks))):
        missing = [networks[i] for i in range(len(networks)) if i not in present]
        raise GraphParameterError(
            f"the parcel labels do not cover all seven networks; missing {missing}"
        )

    size = len(networks)
    block = np.zeros((size, size))
    density = np.zeros((size, size))
    for a in range(size):
        rows_a = np.where(membership == a)[0]
        for b in range(a, size):
            cols_b = np.where(membership == b)[0]
            sub = matrix[np.ix_(rows_a, cols_b)]
            values = sub[np.triu_indices(len(rows_a), k=1)] if a == b else sub.ravel()
            if values.size == 0:
                # A single-parcel network has no within-network parcel pair, so
                # its within-network connectivity is not merely zero -- it is
                # unmeasurable. Substituting 0.0 would assert "no within-network
                # connection", which the data does not say, so this is refused.
                raise GraphParameterError(
                    f"network {networks[a]!r} has "
                    f"{len(rows_a)} parcel(s), so its within-network mean is "
                    "undefined (no parcel pair). Use a parcellation fine enough "
                    "to give every network at least two parcels; Schaefer-200 "
                    "gives 12 to 46."
                )
            block[a, b] = block[b, a] = float(values.mean())
            density[a, b] = density[b, a] = float((values > 0).mean())

    if (block < 0).any():
        raise GraphParameterError(
            "aggregation produced a negative block mean, which cannot be used "
            f"as a connection preference (minimum {block.min():.6f}). The "
            "source matrix is log-transformed and a small number of entries "
            "are negative; a shift or re-transform must be chosen and "
            "documented before such a matrix is used."
        )

    parcel_counts = [int((membership == i).sum()) for i in range(size)]
    off_diagonal = block[~np.eye(size, dtype=bool)]
    return {
        "schema_version": YEO_MATRIX_SCHEMA_VERSION,
        "networks": networks,
        "source_networks": [
            next(k for k, v in SCHAEFER_TO_YEO7.items() if v == name)
            for name in networks
        ],
        "matrix": [[float(x) for x in row] for row in block],
        "diagonal_is": "within-network mean edge weight (parcel self-pairs excluded)",
        "provenance": {
            "source": source,
            "source_urls": source_urls or [],
            "source_sha256": source_sha256 or {},
            "parcellation": parcellation,
            "modality": modality,
            "n_parcels": int(matrix.shape[0]),
            "parcels_per_network": parcel_counts,
            "aggregation": (
                "arithmetic mean of source matrix entries over all parcel pairs "
                "in the block; parcel self-pairs excluded on the diagonal"
            ),
            "source_transform": (
                "provider-applied distance-dependent thresholding and log "
                "transform; not inverted here"
            ),
            "symmetrised": False,
            "source_is_symmetric": symmetric,
            "block_density": [[float(x) for x in row] for row in density],
            "within_between_ratio": float(np.mean(np.diag(block)) / np.mean(off_diagonal)),
            "licence": licence,
            "licence_notice": licence_notice,
            "licence_url": licence_url,
            "retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "citation": [
                "Yeo et al. 2011, J Neurophysiol 106(3), DOI 10.1152/jn.00338.2011",
                "Schaefer et al. 2018, Cerebral Cortex 28(9), 3095-3114",
                "Lariviere et al. 2021, Nature Methods 18(7), ENIGMA Toolbox, "
                "DOI 10.1101/2020.12.21.423838",
            ],
            "notes": notes or "",
        },
    }


# --------------------------------------------------------------------------
# validation and loading
# --------------------------------------------------------------------------

def validate_yeo7_record(record) -> np.ndarray:
    """Validate a matrix record and return its 7x7 array.

    Checks the record's own promises rather than trusting them, so a truncated
    or hand-edited file fails at load time instead of quietly reshaping the
    experiment.
    """
    if not isinstance(record, dict):
        raise GraphParameterError(
            f"Yeo matrix record must be a dict, got {type(record).__name__}"
        )
    version = record.get("schema_version")
    if version != YEO_MATRIX_SCHEMA_VERSION:
        raise GraphParameterError(
            f"unsupported Yeo matrix schema_version {version!r}; this build "
            f"reads version {YEO_MATRIX_SCHEMA_VERSION}"
        )
    for key in ("networks", "matrix", "provenance"):
        if key not in record:
            raise GraphParameterError(f"Yeo matrix record is missing {key!r}")

    networks = record["networks"]
    if list(networks) != list(YEO7):
        raise GraphParameterError(
            "Yeo matrix networks must be exactly the project's YEO7 roles in "
            f"order; got {list(networks)}"
        )

    from .blocks import validate_block_weights

    matrix = validate_block_weights(record["matrix"], len(YEO7))
    return matrix


def load_yeo_matrix(path: str | Path | None = None) -> tuple[np.ndarray, dict]:
    """Load and validate the derived 7x7 matrix; return ``(matrix, record)``.

    Raises :class:`YeoMatrixUnavailableError` when the file is absent. That is
    deliberate: the Yeo-SBM topology is defined by empirical data, so without
    the data there is no Yeo-SBM, and a placeholder matrix would silently turn
    a data-informed topology into an invented one.
    """
    resolved = yeo_matrix_path(path)
    if not resolved.is_file():
        raise YeoMatrixUnavailableError(
            f"no derived Yeo matrix at {resolved}. The Yeo-SBM topology "
            "requires the empirical 7x7 structural matrix; it is derived from "
            "a parcel-level connectome with "
            "'python -m neuromesh.graphs.cli yeo-derive --matrix <csv> "
            "--labels <csv>'. See data/yeo/README_YEO_SOURCES.md."
        )
    record = json.loads(resolved.read_text(encoding="utf-8"))
    matrix = validate_yeo7_record(record)
    return matrix, record
