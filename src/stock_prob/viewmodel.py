"""Presentation ViewModel — charts always from in-memory objects, not scavenged files."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np
import pandas as pd

from stock_prob.features import log_returns
from stock_prob.models import cone_table


@dataclass
class PrismViewModel:
    ticker: str
    name: str = ""
    asof: str = ""
    last_price: float = float("nan")
    regime: str = "n/a"
    probs: dict[str, float] = field(default_factory=dict)
    ensemble_probs: dict[str, float] = field(default_factory=dict)
    history: pd.DataFrame = field(default_factory=pd.DataFrame)  # date, close
    cones: dict[int, pd.DataFrame] = field(default_factory=dict)  # horizon -> cone df
    metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    beat_baseline: dict[str, bool] = field(default_factory=dict)
    brier_skill: dict[str, float] = field(default_factory=dict)
    run_id: str = ""
    art_dir: str = ""
    report_html: str = ""
    honesty_skill: float | None = None
    spillover: dict[str, Any] = field(default_factory=dict)
    champion: str | None = None
    status: str = "ready"
    error: str = ""

    def primary_horizon(self) -> int:
        if self.cones:
            return sorted(self.cones.keys())[0]
        if self.probs:
            return int(sorted(self.probs.keys(), key=lambda x: int(x))[0])
        return 21

    def summary_cards(self) -> list[dict[str, Any]]:
        cards = []
        for h, p in sorted(self.probs.items(), key=lambda x: int(x[0])):
            if p != p:
                continue
            skill = self.brier_skill.get(str(h), self.brier_skill.get(h))
            cards.append(
                {
                    "horizon": int(h),
                    "prob_up": float(p),
                    "direction": "up" if p >= 0.5 else "down",
                    "conviction": abs(float(p) - 0.5) * 2,
                    "beat_baseline": self.beat_baseline.get(str(h), self.beat_baseline.get(h)),
                    "brier_skill": skill,
                }
            )
        return cards


def _history_from_result(result: dict[str, Any], ticker: str) -> pd.DataFrame:
    from pathlib import Path

    art = Path(result.get("art_dir") or "")
    export_xlsx = art / "export.xlsx"
    if export_xlsx.exists():
        try:
            h = pd.read_excel(export_xlsx, sheet_name="history_tail")
            if {"date", "close"}.issubset(h.columns):
                return h[["date", "close"]].copy()
        except Exception:
            pass
    # fetch live/cache
    try:
        from stock_prob.ingest import fetch_symbol

        raw = fetch_symbol(ticker, period="2y", use_cache=True)
        return raw[["date", "close"]].tail(400).copy()
    except Exception:
        pass
    preds = result.get("predictions")
    if preds is not None and len(preds) and "close" in preds.columns:
        return preds[["date", "close"]].drop_duplicates("date").tail(200).copy()
    return pd.DataFrame(columns=["date", "close"])


def _cones_from_result(result: dict[str, Any], history: pd.DataFrame, horizons: list[int]) -> dict[int, pd.DataFrame]:
    from pathlib import Path

    cones: dict[int, pd.DataFrame] = {}
    art = Path(result.get("art_dir") or "")
    for h in horizons:
        p = art / f"garch_cone_{h}.csv"
        if p.exists():
            try:
                cones[h] = pd.read_csv(p)
                continue
            except Exception:
                pass
    # always ensure in-memory cone for each horizon with history
    if history is not None and len(history) > 30:
        s = history.set_index(pd.to_datetime(history["date"]))["close"].astype(float)
        r = log_returns(s).dropna()
        mu = float(r.tail(60).mean()) if len(r) else 0.0
        vol = float(r.tail(60).std()) if len(r) else 0.02
        vol = max(vol if vol == vol else 0.02, 1e-4)
        last_date = pd.Timestamp(history["date"].iloc[-1])
        last_price = float(s.iloc[-1])
        for h in horizons:
            if h not in cones:
                cones[h] = cone_table(
                    last_date, last_price, mu, vol, int(h), n_paths=800, random_state=42
                )
    return cones


def build_viewmodel(result: dict[str, Any], *, name: str = "") -> PrismViewModel:
    ticker = result.get("ticker") or "?"
    probs = {str(k): float(v) for k, v in (result.get("live_probs") or {}).items() if v == v}
    horizons = [int(h) for h in probs.keys()] or [5, 21, 252]
    history = _history_from_result(result, ticker)
    cones = _cones_from_result(result, history, horizons)

    metrics = result.get("metrics")
    if metrics is None:
        metrics = pd.DataFrame()
    beat = {}
    skill = {}
    if metrics is not None and len(metrics):
        for _, row in metrics.iterrows():
            h = str(int(row["horizon"]))
            bm = float(row.get("brier_model", np.nan))
            bb = float(row.get("brier_base", np.nan))
            if bm == bm and bb == bb:
                skill[h] = bb - bm
                beat[h] = bm < bb

    last_price = float("nan")
    asof = ""
    if history is not None and len(history):
        last_price = float(history["close"].iloc[-1])
        asof = str(pd.to_datetime(history["date"].iloc[-1]).date())

    champ = None
    t = result.get("tournament") or {}
    if isinstance(t, dict):
        c = t.get("champion") or {}
        if isinstance(c, dict):
            champ = c.get("champion")
        else:
            champ = c

    hon = result.get("honesty") or {}
    hskill = hon.get("honesty_skill") if isinstance(hon, dict) else None

    return PrismViewModel(
        ticker=ticker,
        name=name or ticker,
        asof=asof,
        last_price=last_price,
        regime=str(result.get("regime") or "n/a"),
        probs=probs,
        ensemble_probs={str(k): float(v) for k, v in (result.get("ensemble_live") or {}).items() if v == v},
        history=history,
        cones=cones,
        metrics=metrics if metrics is not None else pd.DataFrame(),
        beat_baseline=beat,
        brier_skill=skill,
        run_id=str(result.get("run_id") or ""),
        art_dir=str(result.get("art_dir") or ""),
        report_html=str(result.get("lab_report") or result.get("report_html") or ""),
        honesty_skill=float(hskill) if hskill is not None and hskill == hskill else None,
        spillover=result.get("spillover") or {},
        champion=str(champ) if champ else None,
        status="ready",
    )
