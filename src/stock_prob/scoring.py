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
