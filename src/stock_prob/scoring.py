"""Proper scoring rules + cone diagnostics."""
from __future__ import annotations

import numpy as np
import pandas as pd


def brier_score(y_true: np.ndarray | pd.Series, p_up: np.ndarray | pd.Series) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(p_up, dtype=float)
    m = np.isfinite(y) & np.isfinite(p)
    if m.sum() == 0:
        return float("nan")
    return float(np.mean((p[m] - y[m]) ** 2))


def log_loss_safe(y_true, p_up, eps: float = 1e-6) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(p_up, dtype=float), eps, 1 - eps)
    m = np.isfinite(y) & np.isfinite(p)
    if m.sum() == 0:
        return float("nan")
    return float(-np.mean(y[m] * np.log(p[m]) + (1 - y[m]) * np.log(1 - p[m])))


def hit_rate(y_true, p_up, threshold: float = 0.5) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(p_up, dtype=float)
    m = np.isfinite(y) & np.isfinite(p)
    if m.sum() == 0:
        return float("nan")
    pred = (p[m] >= threshold).astype(float)
    return float(np.mean(pred == y[m]))


def cone_coverage(
    actual_prices: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    a = np.asarray(actual_prices, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    m = np.isfinite(a) & np.isfinite(lo) & np.isfinite(hi)
    if m.sum() == 0:
        return float("nan")
    return float(np.mean((a[m] >= lo[m]) & (a[m] <= hi[m])))


def cone_sharpness(lower: np.ndarray, upper: np.ndarray, mid: np.ndarray | None = None) -> float:
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    m = np.isfinite(lo) & np.isfinite(hi)
    if m.sum() == 0:
        return float("nan")
    width = hi[m] - lo[m]
    if mid is not None:
        midv = np.asarray(mid, dtype=float)[m]
        # relative width
        return float(np.mean(width / np.maximum(np.abs(midv), 1e-8)))
    return float(np.mean(width))


def reliability_bins(y_true, p_up, n_bins: int = 10) -> pd.DataFrame:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(p_up, dtype=float)
    m = np.isfinite(y) & np.isfinite(p)
    y, p = y[m], p[m]
    if len(y) == 0:
        return pd.DataFrame(columns=["bin_left", "bin_right", "mean_pred", "frac_up", "count"])
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            sel = (p >= lo) & (p <= hi)
        else:
            sel = (p >= lo) & (p < hi)
        if sel.sum() == 0:
            continue
        rows.append(
            {
                "bin_left": lo,
                "bin_right": hi,
                "mean_pred": float(p[sel].mean()),
                "frac_up": float(y[sel].mean()),
                "count": int(sel.sum()),
            }
        )
    return pd.DataFrame(rows)


def dm_test(
    y_true: np.ndarray | pd.Series,
    p_model: np.ndarray | pd.Series,
    p_base: np.ndarray | pd.Series,
    h: int = 1,
    min_obs: int = 10,
) -> dict[str, float | int]:
    """
    Diebold-Mariano test with Newey-West (HAC) standard error.
    d_t = (p_model - y)^2 - (p_base - y)^2
    H0: E[d_t] = 0 vs H1: E[d_t] < 0 (model has lower Brier score)
    """
    import statsmodels.api as sm

    y = np.asarray(y_true, dtype=float)
    p1 = np.asarray(p_model, dtype=float)
    p0 = np.asarray(p_base, dtype=float)
    m = np.isfinite(y) & np.isfinite(p1) & np.isfinite(p0)
    n_obs = int(m.sum())
    if n_obs < min_obs:
        return {
            "dm_stat": float("nan"),
            "p_value_two_sided": float("nan"),
            "p_value_one_sided": float("nan"),
            "mean_diff": float("nan"),
            "n_obs": n_obs,
        }

    d = (p1[m] - y[m]) ** 2 - (p0[m] - y[m]) ** 2
    X = np.ones((len(d), 1))

    max_lags = max(1, int(h) - 1)
    res = sm.OLS(d, X).fit(cov_type="HAC", cov_kwds={"maxlags": max_lags})

    t_stat = float(res.tvalues[0])
    p_two_sided = float(res.pvalues[0])
    p_one_sided = p_two_sided / 2.0 if t_stat < 0 else 1.0 - (p_two_sided / 2.0)

    return {
        "dm_stat": t_stat,
        "p_value_two_sided": p_two_sided,
        "p_value_one_sided": p_one_sided,
        "mean_diff": float(np.mean(d)),
        "n_obs": n_obs,
    }


def brier_by_regime(
    df: pd.DataFrame,
    regime_col: str = "regime",
    y_col: str = "y_true",
    p_col: str = "prob_up_model",
) -> pd.DataFrame:
    """Group Brier Score evaluation across CALM, ELEVATED, and PANIC regimes."""
    if df is None or len(df) == 0 or regime_col not in df.columns or y_col not in df.columns or p_col not in df.columns:
        return pd.DataFrame(columns=["regime", "n_obs", "brier"])

    rows = []
    for reg_val, sub in df.groupby(regime_col):
        b = brier_score(sub[y_col], sub[p_col])
        rows.append(
            {
                "regime": str(reg_val),
                "n_obs": len(sub),
                "brier": float(b),
            }
        )
    return pd.DataFrame(rows)
