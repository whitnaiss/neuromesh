"""Shared fixtures for the Track A graph tests (owner: Mihir, Track A).

The Week 1 standalone version of these tests carried a ``sys.path`` shim so
``import neuromesh`` worked from a plain clone. This repository installs the
package (``pip install -e ".[dev]"``) and uses a ``src/`` layout, so the shim
is deliberately gone: adding the repository root to ``sys.path`` here would
risk importing a shadow top-level ``neuromesh`` package instead of the
installed one.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from neuromesh.graphs import DEFAULT_FIXTURE_DIR

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def fixture_dir(repo_root: Path) -> Path:
    """Directory holding the committed fixture set (``data/fixtures/graphs``)."""
    return repo_root / DEFAULT_FIXTURE_DIR
