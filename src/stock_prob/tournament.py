"""Multi-model tournament + inverse-Brier forecast combination."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from stock_prob.scoring import brier_score


def score_models_on_predictions(preds: pd.DataFrame) -> pd.DataFrame:
    """
    preds columns: horizon, y_true, prob_up_model, prob_up_base, prob_up_momentum
    Optional: prob_up_ensemble
    """
    rows = []
    if preds is None or len(preds) == 0:
        return pd.DataFrame()
    models = [c for c in preds.columns if c.startswith("prob_up_")]
    for h, sub in preds.groupby("horizon"):
        for col in models:
            name = col.replace("prob_up_", "")
            rows.append(
                {
                    "horizon": int(h),
                    "model": name,
                    "n": len(sub),
                    "brier": brier_score(sub["y_true"], sub[col]),
                }
            )
    return pd.DataFrame(rows)


def inverse_brier_weights(briers: dict[str, float], eps: float = 1e-6) -> dict[str, float]:
    inv = {k: 1.0 / max(float(v), eps) for k, v in briers.items() if v == v}
    s = sum(inv.values()) or 1.0
    return {k: v / s for k, v in inv.items()}


def blend_probs(prob_map: dict[str, float], weights: dict[str, float]) -> float:
    num = 0.0
    den = 0.0
    for k, w in weights.items():
        if k in prob_map and prob_map[k] == prob_map[k]:
            num += w * float(prob_map[k])
            den += w
    if den <= 0:
        return float("nan")
    return num / den


def tournament_table(metrics_by_model: pd.DataFrame) -> pd.DataFrame:
    """Rank models by mean Brier (lower better) across horizons."""
    if metrics_by_model is None or len(metrics_by_model) == 0:
        return pd.DataFrame()
    g = metrics_by_model.groupby("model", as_index=False).agg(
        mean_brier=("brier", "mean"),
        n=("n", "sum"),
    )
    g = g.sort_values("mean_brier")
    g["rank"] = range(1, len(g) + 1)
    return g


def promote_champion(leaderboard: pd.DataFrame) -> dict[str, Any]:
    if leaderboard is None or len(leaderboard) == 0:
        return {"champion": None, "mean_brier": None}
    row = leaderboard.iloc[0]
    return {"champion": row["model"], "mean_brier": float(row["mean_brier"]), "n": int(row["n"])}
