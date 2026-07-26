"""Probability surface: horizon × conviction grid."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def build_probability_surface(live_probs: dict[str, float]) -> pd.DataFrame:
    """
    live_probs: {"5": 0.55, "21": 0.48, ...}
    Returns tidy frame for heatmap.
    """
    rows = []
    for h, p in sorted(live_probs.items(), key=lambda x: int(x[0])):
        if p != p:
            continue
        p = float(p)
        rows.append(
            {
                "horizon": int(h),
                "prob_up": p,
                "prob_down": 1.0 - p,
                "conviction": abs(p - 0.5) * 2.0,
                "direction": "up" if p >= 0.5 else "down",
            }
        )
    return pd.DataFrame(rows)


def surface_matrix(panel_live: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Rows = tickers, cols = horizons, values = P(up).
    panel_live items: {ticker, live_probs}
    """
    records = []
    for item in panel_live:
        t = item.get("ticker")
        probs = item.get("live_probs") or {}
        row = {"ticker": t}
        for h, p in probs.items():
            row[f"h{h}"] = p
        records.append(row)
    return pd.DataFrame(records)
