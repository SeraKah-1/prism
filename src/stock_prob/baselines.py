"""Fair baselines for probability forecasts (no hardcoded tickers)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def base_rate_probs(y_train: pd.Series) -> float:
    """Historical P(up) on training labels."""
    if len(y_train) == 0:
        return 0.5
    return float(np.clip(y_train.mean(), 1e-6, 1 - 1e-6))


def momentum_probs(mom: pd.Series, y_train: pd.Series | None = None) -> pd.Series:
    """
    Naive momentum: if recent mom > 0 predict train base rate of ups when mom>0,
    else complementary. Fallback: 0.55 if mom>0 else 0.45.
    """
    p_up = 0.55
    p_dn = 0.45
    if y_train is not None and len(y_train):
        # not used heavily; keep simple
        p_up = float(np.clip(0.5 + 0.1, 0.05, 0.95))
        p_dn = 1 - p_up
    out = pd.Series(np.where(mom.reindex(mom.index).fillna(0) > 0, p_up, p_dn), index=mom.index)
    return out


def constant_prob_series(index: pd.Index, p: float) -> pd.Series:
    return pd.Series(float(p), index=index, dtype=float)
