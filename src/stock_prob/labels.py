"""Multi-horizon labels + embargo sampling (time-safe)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def forward_return(close: pd.Series, horizon: int) -> pd.Series:
    """Return from t to t+horizon using only future prices for the *label* (known at t+h)."""
    return close.shift(-horizon) / close - 1.0


def direction_label(close: pd.Series, horizon: int) -> pd.Series:
    """1 if forward return > 0 else 0; NaN where incomplete."""
    fr = forward_return(close, horizon)
    lab = (fr > 0).astype(float)
    lab[fr.isna()] = np.nan
    return lab


def apply_embargo(index: pd.DatetimeIndex | pd.Index, horizon: int) -> np.ndarray:
    """
    Return boolean mask selecting non-overlapping sample starts.

    With horizon H, labels for t and t+1 overlap. Keep every H-th date
    (embargo gap = horizon) to reduce dependence.
    """
    n = len(index)
    mask = np.zeros(n, dtype=bool)
    if n == 0:
        return mask
    step = max(int(horizon), 1)
    mask[::step] = True
    return mask


def make_supervised(
    features: pd.DataFrame,
    close: pd.Series,
    horizon: int,
    *,
    feature_cols: list[str] | None = None,
    use_embargo: bool = True,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Return X, y, forward_ret aligned and dropped NaNs.
    If use_embargo, thin samples by horizon.
    """
    close = close.reindex(features.index)
    y = direction_label(close, horizon)
    fr = forward_return(close, horizon)
    cols = feature_cols or [c for c in features.columns if c != "close"]
    X = features[cols].copy()
    df = X.join(y.rename("y")).join(fr.rename("fwd_ret")).dropna()
    if use_embargo and len(df):
        m = apply_embargo(df.index, horizon)
        df = df.loc[m]
    return df[cols], df["y"], df["fwd_ret"]
