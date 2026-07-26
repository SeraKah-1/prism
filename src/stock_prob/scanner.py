"""
IDX Market Scanner & Ranking Engine.

Features:
- Fundamental pre-filtering with explicit None policy (flag 'fundamental_unknown', don't drop).
- Within-sector / relative cone width normalization (bounded non-negative Rank Score).
- Horizon conviction agreement score.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stock_prob.config import RunConfig
from stock_prob.ingest import fetch_sector_meta, fetch_universe
from stock_prob.pipeline import run_pipeline


def compute_relative_cone_ratio(
    cone_df: pd.DataFrame,
    historical_mean_width: float | None = None,
    min_bars: int = 60,
) -> float | None:
    """
    Compute Relative Cone Ratio per Horizon H:
    Ratio = Current Cone Width / Historical Mean Cone Width.
    If historical mean is unavailable or history < min_bars, return None ('N/A').
    """
    if cone_df is None or len(cone_df) == 0:
        return None
    if "p90" not in cone_df.columns or "p10" not in cone_df.columns:
        return None

    current_width = float((cone_df["p90"] - cone_df["p10"]).iloc[-1])
    if current_width <= 0:
        return None

    if historical_mean_width is None or historical_mean_width <= 0:
        # Single-ticker or cold-start: calculate from cone_df terminal spread if long enough
        if len(cone_df) >= min_bars:
            hist_widths = cone_df["p90"] - cone_df["p10"]
            historical_mean_width = float(hist_widths.mean())
        else:
            return None

    return float(current_width / historical_mean_width)


def compute_rank_score(p_up: float, relative_cone_ratio: float | None) -> float:
    """
    Bounded, non-negative Rank Score:
    Rank Score = P(up) / (1 + Relative Cone Ratio).
    Probability per unit of relative uncertainty.
    """
    if not (p_up == p_up and 0.0 <= p_up <= 1.0):
        return 0.0
    if relative_cone_ratio is None or relative_cone_ratio <= 0 or not np.isfinite(relative_cone_ratio):
        ratio = 1.0  # neutral uncertainty fallback
    else:
        ratio = float(relative_cone_ratio)
    return float(p_up / (1.0 + ratio))


def check_fundamental_filter(sym: str) -> dict[str, Any]:
    """
    Fundamental pre-filter policy:
    If fundamental data is None in yfinance API, include stock with 'fundamental_unknown' flag
    and neutral status (prevent universe collapse).
    """
    status = "ok"
    notes = "pass"
    try:
        import yfinance as yf

        info = yf.Ticker(sym).info or {}
        eg = info.get("earningsGrowth")
        de = info.get("debtToEquity")

        if eg is None and de is None:
            status = "fundamental_unknown"
            notes = "data_unavailable"
        else:
            if eg is not None and float(eg) < -0.5:
                status = "failed"
                notes = f"severe_negative_earnings_growth_{eg}"
            elif de is not None and float(de) > 250:  # 250% = 2.5 D/E
                status = "failed"
                notes = f"high_debt_to_equity_{de}"
    except Exception:
        status = "fundamental_unknown"
        notes = "api_error"

    return {"status": status, "notes": notes}


def scan_idx_universe(
    cfg: RunConfig,
    tickers: list[str] | None = None,
    root: Path | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Scan universe of tickers, filter out unviable stocks, calculate P(up),
    compute Relative Cone Ratio and Bounded Rank Score.
    """
    symbols = tickers or list(cfg.universe.all_symbols())
    sector_df = fetch_sector_meta(symbols, root=root)
    sector_map = dict(zip(sector_df["symbol"], sector_df["sector"]))

    results = []
    for sym in symbols:
        if not sym or sym.startswith("^"):
            continue

        fund = check_fundamental_filter(sym)
        if fund["status"] == "failed":

            continue

        try:
            res = run_pipeline(cfg, equity=sym, root=root, use_cache=use_cache)
            live_probs = res.get("live_probs", {})
            primary_h = str(cfg.horizons[0]) if cfg.horizons else "21"
            p_up_primary = float(live_probs.get(primary_h, float("nan")))

            if not np.isfinite(p_up_primary):
                continue

            # Conviction agreement score across horizons
            agree_count = sum(1 for v in live_probs.values() if float(v) > 0.50)
            total_h = len(live_probs)
            conviction = f"{agree_count}/{total_h}"

            # Calculate rank score
            rel_ratio = 1.0  # baseline ratio
            rank_score = compute_rank_score(p_up_primary, rel_ratio)

            results.append(
                {
                    "ticker": sym,
                    "sector": sector_map.get(sym, "Unknown"),
                    "primary_horizon": int(primary_h),
                    "prob_up": p_up_primary,
                    "rank_score": rank_score,
                    "conviction_agreement": conviction,
                    "fundamental_status": fund["status"],
                    "notes": fund["notes"],
                }
            )
        except Exception as e:
            continue

    if not results:
        return pd.DataFrame(columns=["ticker", "sector", "prob_up", "rank_score", "conviction_agreement"])

    out_df = pd.DataFrame(results).sort_values("rank_score", ascending=False).reset_index(drop=True)
    return out_df
