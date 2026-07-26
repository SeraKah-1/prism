"""Simple market regime detection from vol / correlation."""
from __future__ import annotations

import numpy as np
import pandas as pd


def classify_regime(
    equity_close: pd.Series,
    *,
    us_close: pd.Series | None = None,
    vol_window: int = 21,
    corr_window: int = 60,
    vol_panic_quantile: float = 0.80,
) -> pd.DataFrame:
    """
    Return daily regime labels: CALM / ELEVATED / PANIC.

    Rule-based (interpretable): high realized vol and/or high corr to US → risk-on panic mode.
    """
    eq = equity_close.astype(float).sort_index()
    r = np.log(eq).diff()
    vol = r.rolling(vol_window, min_periods=max(5, vol_window // 3)).std()
    thr = float(vol.quantile(vol_panic_quantile)) if vol.notna().any() else 0.02

    out = pd.DataFrame(index=eq.index)
    out["vol_21"] = vol
    out["vol_threshold"] = thr

    corr = pd.Series(np.nan, index=eq.index)
    if us_close is not None:
        u = us_close.astype(float).reindex(eq.index).ffill()
        ru = np.log(u).diff()
        corr = r.rolling(corr_window, min_periods=max(10, corr_window // 3)).corr(ru)
    out["corr_us"] = corr

    labels = []
    for i in range(len(out)):
        v = out["vol_21"].iloc[i]
        c = out["corr_us"].iloc[i]
        if not np.isfinite(v):
            labels.append("UNKNOWN")
        elif v >= thr * 1.15 or (np.isfinite(c) and c >= 0.75 and v >= thr * 0.9):
            labels.append("PANIC")
        elif v >= thr * 0.85:
            labels.append("ELEVATED")
        else:
            labels.append("CALM")
    out["regime"] = labels
    return out


def latest_regime(regime_df: pd.DataFrame) -> str:
    if regime_df is None or len(regime_df) == 0:
        return "UNKNOWN"
    return str(regime_df["regime"].iloc[-1])


def regime_vol_multiplier(regime: str) -> float:
    """Widen cones in panic, tighten in calm."""
    return {"PANIC": 1.45, "ELEVATED": 1.15, "CALM": 0.90, "UNKNOWN": 1.0}.get(regime, 1.0)
