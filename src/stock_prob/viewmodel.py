"""Presentation ViewModel — charts always from in-memory objects."""
from __future__ import annotations

from dataclasses import dataclass, field
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
    history: pd.DataFrame = field(default_factory=pd.DataFrame)
    cones: dict[int, pd.DataFrame] = field(default_factory=dict)
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
    forecast_errors: dict[str, str] = field(default_factory=dict)

    def primary_horizon(self) -> int:
        if self.cones:
            return int(sorted(self.cones.keys())[0])
        if self.probs:
            return int(sorted(self.probs.keys(), key=lambda x: int(x))[0])
        return 21

    def summary_cards(self) -> list[dict[str, Any]]:
        cards = []
        for h, p in sorted(self.probs.items(), key=lambda x: int(x[0])):
            try:
                pf = float(p)
            except Exception:
                continue
            if not np.isfinite(pf):
                continue
            skill = self.brier_skill.get(str(h), self.brier_skill.get(h))
            cards.append(
                {
                    "horizon": int(h),
                    "prob_up": pf,
                    "direction": "up" if pf >= 0.5 else "down",
                    "conviction": abs(pf - 0.5) * 2,
                    "beat_baseline": self.beat_baseline.get(str(h), self.beat_baseline.get(h)),
                    "brier_skill": skill,
                }
            )
        return cards

    def has_probs(self) -> bool:
        return len(self.summary_cards()) > 0

    def has_charts(self) -> bool:
        return (
            self.history is not None
            and len(self.history) > 30
            and bool(self.cones)
        )


def _normalize_history(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["date", "close"])
    out = df.copy()
    # multiindex / odd columns
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [c[0] if isinstance(c, tuple) else c for c in out.columns]
    cols = {str(c).lower(): c for c in out.columns}
    date_c = cols.get("date") or cols.get("datetime")
    close_c = cols.get("close") or cols.get("adj close") or cols.get("adj_close")
    if date_c is None or close_c is None:
        return pd.DataFrame(columns=["date", "close"])
    out = out[[date_c, close_c]].copy()
    out.columns = ["date", "close"]
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out = out.dropna().sort_values("date").drop_duplicates("date")
    return out.reset_index(drop=True)


def _history_from_result(result: dict[str, Any], ticker: str) -> pd.DataFrame:
    from pathlib import Path

    # 1) in-memory history from pipeline
    if isinstance(result.get("history"), pd.DataFrame) and len(result["history"]):
        h = _normalize_history(result["history"])
        if len(h):
            return h.tail(400)

    art = Path(result.get("art_dir") or "")
    # 2) parquet/csv written by pipeline
    for name in ("history.parquet", "history.csv"):
        p = art / name
        if p.exists():
            try:
                h = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
                h = _normalize_history(h)
                if len(h):
                    return h.tail(400)
            except Exception:
                pass
    # 3) excel export
    export_xlsx = art / "export.xlsx"
    if export_xlsx.exists():
        try:
            h = _normalize_history(pd.read_excel(export_xlsx, sheet_name="history_tail"))
            if len(h):
                return h
        except Exception:
            pass
    # 4) fetch
    try:
        from stock_prob.ingest import fetch_symbol

        raw = fetch_symbol(ticker, period="5y", use_cache=True)
        return _normalize_history(raw[["date", "close"]]).tail(400)
    except Exception:
        pass
    preds = result.get("predictions")
    if preds is not None and len(preds) and "close" in getattr(preds, "columns", []):
        return _normalize_history(preds[["date", "close"]]).tail(200)
    return pd.DataFrame(columns=["date", "close"])


def _cones_from_result(
    result: dict[str, Any], history: pd.DataFrame, horizons: list[int]
) -> dict[int, pd.DataFrame]:
    from pathlib import Path

    cones: dict[int, pd.DataFrame] = {}

    # 1) in-memory cones from pipeline
    raw_cones = result.get("live_cones") or {}
    if isinstance(raw_cones, dict):
        for k, v in raw_cones.items():
            try:
                hi = int(k)
                if isinstance(v, pd.DataFrame) and len(v):
                    cones[hi] = v
            except Exception:
                pass

    art = Path(result.get("art_dir") or "")
    for h in horizons:
        if h in cones:
            continue
        p = art / f"garch_cone_{h}.csv"
        if p.exists():
            try:
                cones[h] = pd.read_csv(p)
            except Exception:
                pass

    # 2) always rebuild missing cones from history
    hist = _normalize_history(history)
    if len(hist) > 30:
        s = hist.set_index(pd.to_datetime(hist["date"]))["close"].astype(float)
        r = log_returns(s).dropna()
        mu = float(r.tail(60).mean()) if len(r) else 0.0
        vol = float(r.tail(60).std()) if len(r) else 0.02
        if not np.isfinite(mu):
            mu = 0.0
        if not np.isfinite(vol) or vol <= 0:
            vol = 0.02
        last_date = pd.Timestamp(hist["date"].iloc[-1])
        last_price = float(s.iloc[-1])
        for h in horizons:
            if h not in cones:
                try:
                    cones[h] = cone_table(
                        last_date,
                        last_price,
                        mu,
                        vol,
                        int(h),
                        n_paths=600,
                        random_state=42 + int(h),
                    )
                except Exception:
                    pass
    return cones


def build_viewmodel(result: dict[str, Any], *, name: str = "") -> PrismViewModel:
    ticker = str(result.get("ticker") or "?")
    raw_probs = result.get("live_probs") or {}
    probs: dict[str, float] = {}
    for k, v in raw_probs.items():
        try:
            pf = float(v)
            if np.isfinite(pf):
                probs[str(int(k) if str(k).isdigit() else k)] = pf
        except Exception:
            continue

    # horizons for cones: from probs, result, or default
    horizons: list[int] = []
    for k in probs:
        try:
            horizons.append(int(k))
        except Exception:
            pass
    if not horizons:
        for h in result.get("horizons") or []:
            try:
                horizons.append(int(h))
            except Exception:
                pass
    if not horizons:
        horizons = [5, 21, 252]

    history = _history_from_result(result, ticker)
    cones = _cones_from_result(result, history, horizons)

    metrics = result.get("metrics")
    if metrics is None:
        metrics = pd.DataFrame()
    beat: dict[str, bool] = {}
    skill: dict[str, float] = {}
    if metrics is not None and len(metrics):
        for _, row in metrics.iterrows():
            try:
                h = str(int(row["horizon"]))
                bm = float(row.get("brier_model", np.nan))
                bb = float(row.get("brier_base", np.nan))
                if np.isfinite(bm) and np.isfinite(bb):
                    skill[h] = bb - bm
                    beat[h] = bm < bb
            except Exception:
                pass

    last_price = float("nan")
    asof = ""
    if result.get("last_price") is not None:
        try:
            last_price = float(result["last_price"])
        except Exception:
            pass
    if history is not None and len(history):
        last_price = float(history["close"].iloc[-1])
        asof = str(pd.to_datetime(history["date"].iloc[-1]).date())
    elif result.get("last_date") is not None:
        asof = str(pd.Timestamp(result["last_date"]).date())

    champ = None
    t = result.get("tournament") or {}
    if isinstance(t, dict):
        c = t.get("champion") or {}
        champ = c.get("champion") if isinstance(c, dict) else c

    hon = result.get("honesty") or {}
    hskill = hon.get("honesty_skill") if isinstance(hon, dict) else None
    ferr = result.get("forecast_errors") or {}

    status = "ready"
    error = ""
    if not probs and not cones:
        status = "empty"
        error = "No probabilities or cones produced"
    elif not probs:
        status = "partial"
        error = "Cone only — probabilities missing"

    return PrismViewModel(
        ticker=ticker,
        name=name or ticker,
        asof=asof,
        last_price=last_price,
        regime=str(result.get("regime") or "n/a"),
        probs=probs,
        ensemble_probs={
            str(k): float(v)
            for k, v in (result.get("ensemble_live") or {}).items()
            if v == v
        },
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
        status=status,
        error=error,
        forecast_errors={str(k): str(v) for k, v in ferr.items()} if isinstance(ferr, dict) else {},
    )
