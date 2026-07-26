"""GARCH(1,1) volatility for breathing prediction cones."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from stock_prob.models import cone_table, monte_carlo_cone


def fit_garch_vol(returns: pd.Series, *, last_n: int = 500) -> dict[str, float]:
    """
    Fit GARCH(1,1) on log returns; return conditional vol (daily) + mean.
    Falls back to sample std if arch fit fails.
    """
    r = returns.dropna().astype(float)
    if last_n:
        r = r.tail(last_n)
    if len(r) < 50:
        vol = float(r.std()) if len(r) else 0.02
        mu = float(r.mean()) if len(r) else 0.0
        return {"mu": mu, "vol": max(vol, 1e-6), "method": "sample", "ok": False}

    try:
        from arch import arch_model

        # arch prefers percent scale for numerical stability
        am = arch_model(r * 100.0, vol="Garch", p=1, q=1, mean="Constant", dist="normal")
        res = am.fit(disp="off", show_warning=False)
        # one-step ahead variance forecast
        fcast = res.forecast(horizon=1)
        var = float(fcast.variance.values[-1, 0])
        vol = np.sqrt(max(var, 1e-12)) / 100.0
        mu = float(res.params.get("mu", r.mean() * 100.0)) / 100.0
        return {"mu": mu, "vol": float(max(vol, 1e-6)), "method": "garch11", "ok": True}
    except Exception as e:
        vol = float(r.std())
        mu = float(r.mean())
        return {
            "mu": mu,
            "vol": max(vol, 1e-6),
            "method": "sample_fallback",
            "ok": False,
            "error": str(e)[:200],
        }


def garch_cone_table(
    close: pd.Series,
    horizon: int,
    *,
    n_paths: int = 1500,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build cone using GARCH conditional vol at last date."""
    close = close.dropna().astype(float).sort_index()
    r = np.log(close).diff()
    fit = fit_garch_vol(r)
    last_date = close.index.max()
    last_price = float(close.iloc[-1])
    cone = cone_table(
        last_date,
        last_price,
        fit["mu"],
        fit["vol"],
        horizon,
        n_paths=n_paths,
        random_state=random_state,
    )
    return cone, fit
