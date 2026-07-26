"""Live P(up) + cone construction — robust, used by pipeline and UI recovery."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from stock_prob.features import feature_columns, log_returns
from stock_prob.labels import make_supervised
from stock_prob.models import cone_table, fit_logistic


def compute_live_forecast(
    feats: pd.DataFrame,
    close: pd.Series,
    horizons: list[int],
    *,
    mc_paths: int = 500,
    random_state: int = 42,
    min_train: int = 30,
) -> dict[str, Any]:
    """
    Always attempt P(up) + cone per horizon.
    Returns:
      live_probs: dict[str, float]  (only finite entries + errors for failures)
      live_cones: dict[int, DataFrame]
      errors: dict[str, str]
      last_price, last_date, mu, vol
    """
    close = close.dropna().astype(float).sort_index()
    if len(close) < 40:
        return {
            "live_probs": {},
            "live_cones": {},
            "errors": {"_all": f"history too short ({len(close)} bars)"},
            "last_price": float("nan"),
            "last_date": None,
            "mu": 0.0,
            "vol": 0.02,
        }

    fcols = feature_columns(feats)
    if not fcols:
        # fallback minimal features from close only
        feats = feats.copy()
        r = log_returns(close)
        feats["ret_1d"] = r
        feats["mom_5"] = close.pct_change(5)
        feats["mom_21"] = close.pct_change(21)
        feats["vol_21"] = r.rolling(21, min_periods=5).std()
        feats["close"] = close
        fcols = [c for c in ("ret_1d", "mom_5", "mom_21", "vol_21") if c in feats.columns]

    last_date = close.index.max()
    last_price = float(close.iloc[-1])
    r = log_returns(close).dropna()
    mu = float(r.tail(60).mean()) if len(r) else 0.0
    vol = float(r.tail(60).std()) if len(r) else 0.02
    if not np.isfinite(mu):
        mu = 0.0
    if not np.isfinite(vol) or vol <= 0:
        vol = 0.02

    live_probs: dict[str, float] = {}
    live_cones: dict[int, pd.DataFrame] = {}
    errors: dict[str, str] = {}

    # latest feature row: prefer full dropna, else forward-fill then dropna
    feat_block = feats.reindex(close.index)[fcols].replace([np.inf, -np.inf], np.nan)
    latest = feat_block.dropna()
    if len(latest) == 0:
        latest = feat_block.ffill().dropna()
    if len(latest) == 0:
        # last resort: zeros
        latest = pd.DataFrame([{c: 0.0 for c in fcols}], index=[last_date])

    latest_row = latest.iloc[[-1]]

    for h in horizons:
        h = int(h)
        # always build cone (visual must not depend on logistic success)
        try:
            live_cones[h] = cone_table(
                pd.Timestamp(last_date),
                last_price,
                mu,
                vol,
                h,
                n_paths=max(200, int(mc_paths)),
                random_state=random_state + h,
            )
        except Exception as e:
            errors[f"cone_{h}"] = str(e)[:160]

        try:
            X, y, _ = make_supervised(
                feats, close, h, feature_cols=fcols, use_embargo=False
            )
            X = X.replace([np.inf, -np.inf], np.nan).dropna()
            y = y.reindex(X.index)
            if len(X) < min_train:
                errors[str(h)] = f"insufficient train rows ({len(X)}<{min_train})"
                continue
            if y.nunique() < 2:
                # single class — still give base rate style probability
                p = float(y.mean()) if len(y) else 0.5
                p = float(np.clip(p, 0.02, 0.98))
                live_probs[str(h)] = p
                errors[str(h)] = "single_class_fallback_base_rate"
                continue
            model = fit_logistic(X, y, list(X.columns), horizon=h, random_state=random_state)
            # align latest columns
            row = latest_row.reindex(columns=list(X.columns)).fillna(0.0)
            p = float(model.predict_proba_up(row)[0])
            if not np.isfinite(p):
                errors[str(h)] = "non_finite_probability"
                continue
            live_probs[str(h)] = float(np.clip(p, 1e-4, 1 - 1e-4))
        except Exception as e:
            errors[str(h)] = str(e)[:200]
            # last-ditch: momentum sign probability
            try:
                mom = float(close.pct_change(min(h, 21)).iloc[-1])
                live_probs[str(h)] = 0.55 if mom > 0 else 0.45
                errors[str(h)] = f"momentum_fallback: {errors[str(h)]}"
            except Exception:
                pass

    return {
        "live_probs": live_probs,
        "live_cones": live_cones,
        "errors": errors,
        "last_price": last_price,
        "last_date": last_date,
        "mu": mu,
        "vol": vol,
        "n_bars": int(len(close)),
        "feature_cols": fcols,
    }


def history_frame(close: pd.Series, n: int = 400) -> pd.DataFrame:
    s = close.dropna().astype(float).sort_index().tail(n)
    return pd.DataFrame({"date": pd.to_datetime(s.index), "close": s.values})
