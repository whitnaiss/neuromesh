"""Track C (Yutong): metrics, statistics and figures from the Parquet logs.

Public entry point for the runner: :func:`compute_metrics`. Statistics
(:mod:`.stats`), figures (:mod:`.plots`) and the rule-based reference agents
(:mod:`.reference`) are used by the analysis scripts.
"""

from __future__ import annotations

from neuromesh.evaluation.metrics import PRIMARY_METRIC, compute_metrics, cost_metrics, to_frames
from neuromesh.evaluation.stats import covariate_regression, mean_ci, summarize

__all__ = [
    "PRIMARY_METRIC",
    "compute_metrics",
    "cost_metrics",
    "covariate_regression",
    "mean_ci",
    "summarize",
    "to_frames",
]
