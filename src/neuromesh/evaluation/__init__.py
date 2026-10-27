"""Track C (Yutong): metrics, statistics and figures from the Parquet logs."""

from __future__ import annotations

import pandas as pd


def compute_metrics(runs: pd.DataFrame, calls: pd.DataFrame) -> pd.DataFrame:
    """One row per run: propagation delay, fidelity, agreement, accuracy, tokens, time."""
    raise NotImplementedError("Track C: compute_metrics")


__all__ = ["compute_metrics"]
