"""Multi-ticker panel runs + market (IDX vs US) comparison."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from stock_prob.config import RunConfig, UniverseConfig
from stock_prob.pipeline import run_pipeline


def market_of(ticker: str) -> str:
    t = ticker.upper()
    if t.endswith(".JK") or t.endswith(".JA"):
        return "IDX"
    return "US"


def run_panel(
    equities: list[str],
    *,
    domestic_index_idx: str = "^JKSE",
    domestic_index_us: str = "^GSPC",
    us_index: str = "^GSPC",
    macro: str = "^VIX",
    horizons: list[int] | None = None,
    root=None,
    use_cache: bool = True,
    run_name: str = "panel",
    mc_paths: int = 600,
    min_train_rows: int = 200,
) -> list[dict[str, Any]]:
    """Run pipeline for each equity with market-appropriate domestic index."""
    horizons = horizons or [5, 21, 252]
    results = []
    for eq in equities:
        mkt = market_of(eq)
        dom = domestic_index_idx if mkt == "IDX" else domestic_index_us
        uni = UniverseConfig(
            equities=[eq],
            domestic_index=dom,
            us_index=us_index,
            macro=macro,
        )
        cfg = RunConfig(
            universe=uni,
            horizons=list(horizons),
            mc_paths=mc_paths,
            min_train_rows=min_train_rows,
            run_name=f"{run_name}_{mkt.lower()}",
        )
        try:
            res = run_pipeline(cfg, equity=eq, root=root, use_cache=use_cache)
            res["market"] = mkt
            results.append(res)
        except Exception as e:
            results.append(
                {
                    "ticker": eq,
                    "market": mkt,
                    "error": str(e)[:300],
                    "live_probs": {},
                    "metrics": pd.DataFrame(),
                }
            )
    return results


def summarize_panel(results: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for r in results:
        if r.get("error"):
            rows.append({"ticker": r.get("ticker"), "market": r.get("market"), "error": r["error"]})
            continue
        m = r.get("metrics")
        if m is None or len(m) == 0:
            continue
        for _, row in m.iterrows():
            rows.append(
                {
                    "ticker": r["ticker"],
                    "market": r.get("market", market_of(r["ticker"])),
                    "horizon": int(row["horizon"]),
                    "n": int(row["n"]),
                    "brier_model": float(row["brier_model"]),
                    "brier_base": float(row["brier_base"]),
                    "brier_skill": float(row["brier_base"] - row["brier_model"]),
                    "cone_coverage_80": float(row.get("cone_coverage_80", np.nan)),
                    "prob_up_live": r.get("live_probs", {}).get(str(int(row["horizon"]))),
                    "run_id": r.get("run_id"),
                }
            )
    return pd.DataFrame(rows)


def country_skill_table(panel_df: pd.DataFrame) -> pd.DataFrame:
    if panel_df is None or len(panel_df) == 0 or "brier_skill" not in panel_df.columns:
        return pd.DataFrame()
    g = (
        panel_df.dropna(subset=["brier_skill"])
        .groupby(["market", "horizon"], as_index=False)
        .agg(
            mean_skill=("brier_skill", "mean"),
            mean_brier_model=("brier_model", "mean"),
            mean_brier_base=("brier_base", "mean"),
            n_tickers=("ticker", "nunique"),
            mean_coverage=("cone_coverage_80", "mean"),
        )
    )
    return g.sort_values(["horizon", "market"])
