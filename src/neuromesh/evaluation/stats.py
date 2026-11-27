"""Condition summaries (mean and 95% CI) and the covariate regression.

The regression is ordinary least squares with HC1 heteroskedasticity-robust
standard errors, written with NumPy/SciPy so it adds no dependency. It answers
"does topology matter once graph properties are controlled for?": the outcome
is regressed on log2(N), topology dummies (ER is the reference) and structural
covariates such as path length, clustering and input-readout distance.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


def mean_ci(values: Sequence[float], level: float = 0.95) -> tuple[float, float, float]:
    """(mean, lower, upper) with a t-interval; zero width for n < 2."""
    x = np.asarray([v for v in values if v is not None and not np.isnan(v)], dtype=float)
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    m = float(x.mean())
    if x.size < 2:
        return m, m, m
    half = float(stats.t.ppf(0.5 + level / 2, x.size - 1) * x.std(ddof=1) / np.sqrt(x.size))
    return m, m - half, m + half


def summarize(
    df: pd.DataFrame,
    metrics: Sequence[str],
    by: Sequence[str] = ("task", "topology", "n"),
    level: float = 0.95,
) -> pd.DataFrame:
    """Long table: one row per group x metric with mean, CI bounds, sd and count."""
    rows = []
    for keys, grp in df.groupby(list(by), sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        for metric in metrics:
            if metric not in grp:
                continue
            vals = grp[metric].dropna().astype(float)
            if vals.empty:
                continue
            m, lo, hi = mean_ci(vals.tolist(), level)
            rows.append({**dict(zip(by, keys, strict=True)), "metric": metric, "mean": m,
                         "ci_low": lo, "ci_high": hi,
                         "sd": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                         "n_runs": int(len(vals))})
    return pd.DataFrame(rows)


@dataclass
class RegressionResult:
    table: pd.DataFrame  # term, coef, se, t, p, ci_low, ci_high
    r2: float
    adj_r2: float
    n_obs: int
    dropped: list[str]  # constant or collinear columns removed before fitting

    def to_text(self) -> str:
        head = (f"OLS, HC1 robust SE  n = {self.n_obs}  R2 = {self.r2:.3f}  "
                f"adj R2 = {self.adj_r2:.3f}")
        return head + "\n" + self.table.to_string(index=False, float_format=lambda v: f"{v:.4f}")


def covariate_regression(
    df: pd.DataFrame,
    outcome: str,
    covariates: Sequence[str] = (),
    categorical: Sequence[str] = ("topology",),
    log2_n: bool = True,
    reference: dict[str, str] | None = None,
) -> RegressionResult:
    """OLS of ``outcome`` on log2(N), categorical dummies and numeric covariates.

    Rows with any missing value in the model columns are dropped. Constant or
    perfectly collinear columns are removed (and reported in ``dropped``) so the
    fit never fails silently on a small pilot design.
    """
    reference = {"topology": "er", **(reference or {})}
    cols = [outcome, *covariates, *categorical] + (["n"] if log2_n else [])
    data = df[cols].dropna().copy()
    design = pd.DataFrame(index=data.index)
    design["const"] = 1.0
    if log2_n:
        design["log2_n"] = np.log2(data["n"].astype(float))
    for cov in covariates:
        design[cov] = data[cov].astype(float)
    for cat in categorical:
        levels = sorted(data[cat].astype(str).unique())
        ref = reference.get(cat, levels[0] if levels else None)
        for lvl in levels:
            if lvl != ref:
                design[f"{cat}[{lvl}]"] = (data[cat].astype(str) == lvl).astype(float)

    dropped: list[str] = []
    for col in list(design.columns):  # remove constants other than the intercept
        if col != "const" and design[col].nunique() <= 1:
            dropped.append(col)
            design = design.drop(columns=col)
    keep: list[str] = []  # greedy rank check to drop collinear columns
    for col in design.columns:
        trial = design[keep + [col]].to_numpy()
        if np.linalg.matrix_rank(trial) == len(keep) + 1:
            keep.append(col)
        else:
            dropped.append(col)
    X = design[keep].to_numpy(dtype=float)
    y = data[outcome].to_numpy(dtype=float)
    n, k = X.shape
    if n <= k:
        raise ValueError(f"need more observations than parameters (n={n}, k={k})")

    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    resid = y - X @ beta
    meat = X.T @ (X * (resid**2)[:, None])
    cov = xtx_inv @ meat @ xtx_inv * (n / (n - k))  # HC1
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(se > 0, beta / se, np.nan)
    p = 2 * stats.t.sf(np.abs(t), n - k)
    crit = stats.t.ppf(0.975, n - k)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - float((resid**2).sum()) / ss_tot if ss_tot > 0 else float("nan")
    adj = 1 - (1 - r2) * (n - 1) / (n - k) if ss_tot > 0 else float("nan")
    table = pd.DataFrame({"term": keep, "coef": beta, "se": se, "t": t, "p": p,
                          "ci_low": beta - crit * se, "ci_high": beta + crit * se})
    return RegressionResult(table=table, r2=r2, adj_r2=adj, n_obs=n, dropped=dropped)
