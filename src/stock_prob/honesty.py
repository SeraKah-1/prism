"""Honesty score: does model uncertainty track realized difficulty?"""
from __future__ import annotations

import numpy as np
import pandas as pd


def prediction_confidence(p_up: float) -> float:
    """Distance from 0.5 — higher = more confident."""
    if p_up != p_up:
        return float("nan")
    return abs(float(p_up) - 0.5) * 2.0  # 0..1


def cone_uncertainty(width_rel: float) -> float:
    if width_rel != width_rel:
        return float("nan")
    return float(np.clip(width_rel, 0, 5))


def honesty_frame(preds: pd.DataFrame) -> pd.DataFrame:
    """
    From walk-forward preds with prob_up_model and y_true.
    Difficulty = absolute error |y - p| (or 1-hit).
    """
    if preds is None or len(preds) == 0:
        return pd.DataFrame()
    df = preds.copy()
    df["confidence"] = df["prob_up_model"].map(prediction_confidence)
    df["abs_err"] = (df["y_true"] - df["prob_up_model"]).abs()
    df["hit"] = ((df["prob_up_model"] >= 0.5).astype(float) == df["y_true"]).astype(float)
    return df


def honesty_score(frame: pd.DataFrame) -> dict[str, float]:
    """
    Correlation between confidence and (1 - abs_err):
    if model is honest, high confidence ↔ low error.
    """
    if frame is None or len(frame) < 10:
        return {"honesty_corr": float("nan"), "n": 0.0}
    sub = frame.dropna(subset=["confidence", "abs_err"])
    if len(sub) < 10:
        return {"honesty_corr": float("nan"), "n": float(len(sub))}
    # higher confidence should mean lower abs_err → negative corr desired, report flipped skill
    corr = float(sub["confidence"].corr(sub["abs_err"]))
    # honesty_skill: -corr so positive = honest
    return {
        "honesty_corr_conf_vs_err": corr,
        "honesty_skill": float(-corr) if corr == corr else float("nan"),
        "n": float(len(sub)),
        "mean_confidence": float(sub["confidence"].mean()),
        "mean_abs_err": float(sub["abs_err"].mean()),
    }
