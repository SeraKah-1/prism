"""IDX overnight spillover: US move → next-day open/close reaction probability."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_spillover_frame(
    equity_ohlc: pd.DataFrame,
    us_close: pd.Series,
) -> pd.DataFrame:
    """
    equity_ohlc: columns date, open, close (ticker local session).
    us_close: US index/equity close (prior session proxy for overnight news).
    """
    eq = equity_ohlc.copy()
    if "date" in eq.columns:
        eq = eq.set_index("date")
    eq.index = pd.to_datetime(eq.index).tz_localize(None)
    eq = eq.sort_index()

    us = us_close.astype(float).sort_index()
    us.index = pd.to_datetime(us.index).tz_localize(None)
    us_ret = us.pct_change().rename("us_ret_1d")

    # Align: local day t reacts to US return from previous available US bar
    us_ret_lag = us_ret.shift(1)
    df = eq[["open", "close"]].copy() if "open" in eq.columns else eq[["close"]].copy()
    if "open" not in df.columns:
        df["open"] = df["close"]
    df = df.join(us_ret_lag, how="left")
    df["us_ret_1d"] = df["us_ret_1d"].ffill()
    df["local_ret"] = df["close"].pct_change()
    df["gap"] = df["open"] / df["close"].shift(1) - 1.0
    df["down_day"] = (df["local_ret"] < 0).astype(float)
    df["gap_down"] = (df["gap"] < 0).astype(float)
    return df.dropna()


def fit_spillover_model(frame: pd.DataFrame, target: str = "gap_down") -> Pipeline | None:
    if target not in frame.columns or len(frame) < 80:
        return None
    y = frame[target].astype(int)
    if y.nunique() < 2:
        return None
    X = frame[["us_ret_1d"]].copy()
    # nonlinear-ish expansion
    X["us_ret_sq"] = X["us_ret_1d"] ** 2
    X["us_down"] = (X["us_ret_1d"] < 0).astype(float)
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=400, class_weight="balanced")),
        ]
    )
    pipe.fit(X, y)
    return pipe


def spillover_probability(
    model: Pipeline | None,
    us_ret_last: float,
) -> dict[str, float | str]:
    if model is None or not np.isfinite(us_ret_last):
        return {"p_gap_down": float("nan"), "us_ret": us_ret_last, "status": "unavailable"}
    X = pd.DataFrame(
        {
            "us_ret_1d": [us_ret_last],
            "us_ret_sq": [us_ret_last**2],
            "us_down": [1.0 if us_ret_last < 0 else 0.0],
        }
    )
    proba = model.predict_proba(X)[0]
    classes = list(model.named_steps["clf"].classes_)
    idx = classes.index(1) if 1 in classes else int(np.argmax(classes))
    return {
        "p_gap_down": float(proba[idx]),
        "us_ret": float(us_ret_last),
        "status": "ok",
    }
