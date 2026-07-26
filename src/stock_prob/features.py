"""Point-in-time feature engineering. Pure transforms — no ticker constants."""
from __future__ import annotations

import numpy as np
import pandas as pd


def log_returns(close: pd.Series) -> pd.Series:
    return np.log(close.astype(float)).diff()


def rolling_beta(y: pd.Series, x: pd.Series, window: int) -> pd.Series:
    """Rolling OLS beta of y on x, point-in-time (trailing window only)."""
    y, x = y.align(x, join="inner")
    cov = y.rolling(window, min_periods=max(10, window // 3)).cov(x)
    var = x.rolling(window, min_periods=max(10, window // 3)).var()
    return cov / var.replace(0, np.nan)


def rolling_corr(y: pd.Series, x: pd.Series, window: int) -> pd.Series:
    y, x = y.align(x, join="inner")
    return y.rolling(window, min_periods=max(10, window // 3)).corr(x)


def rolling_vol(r: pd.Series, window: int) -> pd.Series:
    return r.rolling(window, min_periods=max(10, window // 3)).std()


def build_feature_frame(
    equity_close: pd.Series,
    *,
    domestic_close: pd.Series | None = None,
    us_close: pd.Series | None = None,
    macro_close: pd.Series | None = None,
    window: int = 60,
) -> pd.DataFrame:
    """
    Build PIT features for one equity series.

    All inputs are close-price series indexed by date. No future data is used:
    rolling stats at t use only observations ≤ t.
    """
    eq = equity_close.astype(float).sort_index()
    r = log_returns(eq)

    feats = pd.DataFrame(index=eq.index)
    feats["ret_1d"] = r
    feats["mom_5"] = eq.pct_change(5)
    feats["mom_21"] = eq.pct_change(21)
    feats["vol_21"] = rolling_vol(r, 21)
    feats["vol_60"] = rolling_vol(r, window)

    if domestic_close is not None:
        d = domestic_close.astype(float).reindex(eq.index).ffill()
        rd = log_returns(d)
        feats["beta_dom"] = rolling_beta(r, rd, window)
        feats["corr_dom"] = rolling_corr(r, rd, window)
        feats["ret_dom_1d"] = rd

    if us_close is not None:
        u = us_close.astype(float).reindex(eq.index).ffill()
        ru = log_returns(u)
        feats["beta_us"] = rolling_beta(r, ru, window)
        feats["corr_us"] = rolling_corr(r, ru, window)
        feats["ret_us_1d"] = ru

    if macro_close is not None:
        m = macro_close.astype(float).reindex(eq.index).ffill()
        feats["macro_lvl"] = m
        feats["macro_chg"] = m.pct_change()

    feats["close"] = eq
    return feats


def feature_columns(df: pd.DataFrame) -> list[str]:
    skip = {"close"}
    return [c for c in df.columns if c not in skip and pd.api.types.is_numeric_dtype(df[c])]
