"""Price structure: support/resistance zones + market character (trend × vol)."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _pivots(close: pd.Series, left: int = 5, right: int = 5) -> tuple[pd.Series, pd.Series]:
    """Swing highs / lows using a simple fractal window (no future look past `right`)."""
    c = close.astype(float).dropna().sort_index()
    n = len(c)
    hi = pd.Series(np.nan, index=c.index)
    lo = pd.Series(np.nan, index=c.index)
    vals = c.values
    for i in range(left, n - right):
        w = vals[i - left : i + right + 1]
        if vals[i] >= w.max() and vals[i] == w[left]:
            hi.iloc[i] = vals[i]
        if vals[i] <= w.min() and vals[i] == w[left]:
            lo.iloc[i] = vals[i]
    return hi, lo


def _cluster_levels(
    prices: list[float],
    *,
    last_price: float,
    tol_frac: float = 0.012,
    max_levels: int = 4,
) -> list[dict[str, Any]]:
    if not prices or not np.isfinite(last_price) or last_price <= 0:
        return []
    prices = sorted(float(p) for p in prices if np.isfinite(p) and p > 0)
    if not prices:
        return []
    clusters: list[list[float]] = []
    for p in prices:
        placed = False
        for cl in clusters:
            center = float(np.mean(cl))
            if abs(p - center) / center <= tol_frac:
                cl.append(p)
                placed = True
                break
        if not placed:
            clusters.append([p])
    levels = []
    for cl in clusters:
        price = float(np.median(cl))
        levels.append(
            {
                "price": price,
                "touches": len(cl),
                "strength": float(min(1.0, len(cl) / 4.0)),
                "distance_pct": float((price / last_price - 1.0) * 100.0),
            }
        )
    # prefer more touches, then closer to last
    levels.sort(key=lambda x: (-x["touches"], abs(x["distance_pct"])))
    return levels[:max_levels]


def support_resistance(
    close: pd.Series,
    *,
    left: int = 5,
    right: int = 5,
    max_levels: int = 4,
) -> dict[str, Any]:
    """
    Swing-based support (below last) and resistance (above last).
    Levels are zones approximated as single prices (median of a cluster).
    """
    c = close.astype(float).dropna().sort_index()
    if len(c) < 40:
        return {
            "supports": [],
            "resistances": [],
            "nearest_support": None,
            "nearest_resistance": None,
            "last_price": float(c.iloc[-1]) if len(c) else float("nan"),
        }
    last = float(c.iloc[-1])
    hi, lo = _pivots(c, left=left, right=right)
    highs = [float(x) for x in hi.dropna().tolist()]
    lows = [float(x) for x in lo.dropna().tolist()]
    # also seed with recent rolling min/max
    for w in (21, 63, 126):
        if len(c) > w:
            highs.append(float(c.tail(w).max()))
            lows.append(float(c.tail(w).min()))

    supports = [lv for lv in _cluster_levels(lows, last_price=last, max_levels=max_levels * 2) if lv["price"] < last * 0.998]
    resistances = [
        lv for lv in _cluster_levels(highs, last_price=last, max_levels=max_levels * 2) if lv["price"] > last * 1.002
    ]
    supports = sorted(supports, key=lambda x: -x["price"])[:max_levels]
    resistances = sorted(resistances, key=lambda x: x["price"])[:max_levels]
    return {
        "supports": supports,
        "resistances": resistances,
        "nearest_support": supports[0] if supports else None,
        "nearest_resistance": resistances[0] if resistances else None,
        "last_price": last,
    }


def market_character(
    close: pd.Series,
    *,
    us_close: pd.Series | None = None,
) -> dict[str, Any]:
    """
    Interpretable market character (not retail slang).

    Axes:
      - trend: up / down / sideways (from multi-horizon momentum)
      - vol: low / normal / high (realized vol percentile)
      - label: e.g. "quiet uptrend", "volatile range", "volatile downtrend"
    """
    c = close.astype(float).dropna().sort_index()
    if len(c) < 40:
        return {
            "trend": "unknown",
            "vol": "unknown",
            "label": "unknown",
            "mom_21": float("nan"),
            "mom_63": float("nan"),
            "vol_21": float("nan"),
            "vol_percentile": float("nan"),
        }
    r = np.log(c).diff()
    mom_21 = float(c.pct_change(21).iloc[-1]) if len(c) > 21 else float(c.pct_change().iloc[-1])
    mom_63 = float(c.pct_change(63).iloc[-1]) if len(c) > 63 else mom_21
    vol_21 = float(r.tail(21).std()) if len(r.dropna()) >= 10 else float("nan")
    vol_hist = r.rolling(21, min_periods=10).std().dropna()
    vol_pct = float((vol_hist < vol_21).mean()) if len(vol_hist) and np.isfinite(vol_21) else float("nan")

    # trend
    if np.isfinite(mom_21) and np.isfinite(mom_63):
        if mom_21 > 0.02 and mom_63 > 0:
            trend = "up"
        elif mom_21 < -0.02 and mom_63 < 0:
            trend = "down"
        elif abs(mom_21) <= 0.02 and abs(mom_63) <= 0.05:
            trend = "sideways"
        elif mom_21 > 0:
            trend = "up"
        elif mom_21 < 0:
            trend = "down"
        else:
            trend = "sideways"
    else:
        trend = "unknown"

    if np.isfinite(vol_pct):
        if vol_pct >= 0.75:
            vol = "high"
        elif vol_pct <= 0.30:
            vol = "low"
        else:
            vol = "normal"
    else:
        vol = "unknown"

    # compression: recent range vs longer range
    compression = False
    if len(c) > 63:
        range_21 = float(c.tail(21).max() / c.tail(21).min() - 1.0)
        range_63 = float(c.tail(63).max() / c.tail(63).min() - 1.0)
        if range_63 > 0 and range_21 / range_63 < 0.35:
            compression = True

    if trend == "sideways" and vol == "low":
        label = "quiet range"
    elif trend == "sideways" and vol == "high":
        label = "volatile range"
    elif trend == "sideways":
        label = "range-bound"
    elif trend == "up" and vol == "low":
        label = "quiet uptrend"
    elif trend == "up" and vol == "high":
        label = "volatile uptrend"
    elif trend == "up":
        label = "uptrend"
    elif trend == "down" and vol == "low":
        label = "quiet downtrend"
    elif trend == "down" and vol == "high":
        label = "volatile downtrend"
    elif trend == "down":
        label = "downtrend"
    else:
        label = f"{vol} {trend}".strip()

    if compression and trend == "sideways":
        label = "volatility compression (range)"

    corr_us = float("nan")
    if us_close is not None:
        u = us_close.astype(float).reindex(c.index).ffill()
        ru = np.log(u).diff()
        joined = pd.concat([r, ru], axis=1).dropna().tail(60)
        if len(joined) > 20:
            corr_us = float(joined.iloc[:, 0].corr(joined.iloc[:, 1]))

    return {
        "trend": trend,
        "vol": vol,
        "label": label,
        "mom_21": mom_21,
        "mom_63": mom_63,
        "vol_21": vol_21,
        "vol_percentile": vol_pct,
        "compression": compression,
        "corr_us": corr_us,
    }
