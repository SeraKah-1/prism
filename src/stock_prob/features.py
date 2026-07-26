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
    universe_median_close: pd.Series | None = None,
    sector_median_close: pd.Series | None = None,
    window: int = 60,
) -> pd.DataFrame:
    """
    Build PIT features for one equity series.

    All inputs are close-price series indexed by date. No future data is used:
    rolling stats at t use only observations ≤ t.
    """
    from stock_prob.ingest import align_us_to_idx

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
        # Raw overnight US lag feature via single source of truth alignment
        feats["us_ret_1d_lagged"] = align_us_to_idx(ru, eq.index)

    if macro_close is not None:
        m = macro_close.astype(float).reindex(eq.index).ffill()
        feats["macro_lvl"] = m
        feats["macro_chg"] = m.pct_change()

    if universe_median_close is not None:
        um = universe_median_close.astype(float).reindex(eq.index).ffill()
        feats["ret_vs_universe"] = eq.pct_change(21) - um.pct_change(21)

    if sector_median_close is not None:
        sm = sector_median_close.astype(float).reindex(eq.index).ffill()
        feats["ret_vs_sector"] = eq.pct_change(21) - sm.pct_change(21)

    feats["close"] = eq
    return feats


def validate_brier_delta(brier_old: float, brier_new: float, threshold: float = -0.002) -> bool:
    """Return True if new feature set improves Brier score beyond threshold."""
    if not (np.isfinite(brier_old) and np.isfinite(brier_new)):
        return False
    return float(brier_new - brier_old) <= threshold


def apply_adaptive_r2_selection(
    feats: pd.DataFrame,
    r2_market: float | None = None,
    threshold: float = 0.70,
) -> pd.DataFrame:
    """
    Adaptive R² Feature Selection:
    If stock movement is heavily market-driven (R² > 0.70), drop noisy short-term technicals (ret_1d, mom_5)
    and keep structural beta/volatility features.
    """
    if feats is None or len(feats) == 0 or r2_market is None or not np.isfinite(r2_market):
        return feats

    out = feats.copy()
    if float(r2_market) > threshold:
        drop_cols = [c for c in ("mom_5", "ret_1d") if c in out.columns]
        if drop_cols:
            out = out.drop(columns=drop_cols)
    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    skip = {"close"}
    return [c for c in df.columns if c not in skip and pd.api.types.is_numeric_dtype(df[c])]
