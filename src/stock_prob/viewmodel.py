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
    # new research layers
    supports: list[dict[str, Any]] = field(default_factory=list)
    resistances: list[dict[str, Any]] = field(default_factory=list)
    character: dict[str, Any] = field(default_factory=dict)
    decision: dict[str, Any] = field(default_factory=dict)
    mu: float = 0.0
    vol: float = 0.02
    model_meta: dict[str, Any] = field(default_factory=dict)
    horizons: list[int] = field(default_factory=list)

    def primary_horizon(self) -> int:
        from stock_prob.horizon_keys import parse_horizon_key

        cone_hs = []
        for k in self.cones.keys():
            h = parse_horizon_key(k)
            if h is not None:
                cone_hs.append(h)
        if cone_hs:
            return int(sorted(cone_hs)[0])
        prob_hs = []
        for k in self.probs.keys():
            h = parse_horizon_key(k)
            if h is not None:
                prob_hs.append(h)
        if prob_hs:
            return int(sorted(prob_hs)[0])
        if self.horizons:
            return int(sorted(self.horizons)[0])
        return 21

    def summary_cards(self) -> list[dict[str, Any]]:
        from stock_prob.horizon_keys import parse_horizon_key

        cards = []
        items = []
        for k, p in self.probs.items():
            h = parse_horizon_key(k)
            if h is None:
                continue
            try:
                pf = float(p)
            except Exception:
                continue
            if not np.isfinite(pf):
                continue
            items.append((h, pf))
        for h, pf in sorted(items, key=lambda x: x[0]):
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

    def cone_for(self, horizon: int) -> pd.DataFrame | None:
        h = int(horizon)
        c = self.cones.get(h)
        if c is None:
            c = self.cones.get(str(h))  # type: ignore[arg-type]
        return c

    def to_session(self) -> dict[str, Any]:
        """JSON-friendly payload for Gradio State (horizon switch + entry sim)."""
        cones = {}
        for k, df in self.cones.items():
            try:
                hi = int(k)
            except Exception:
                continue
            if isinstance(df, pd.DataFrame) and len(df):
                d = df.copy()
                if "date" in d.columns:
                    d["date"] = pd.to_datetime(d["date"]).astype(str)
                cones[str(hi)] = d.to_dict(orient="records")
        hist = pd.DataFrame()
        if self.history is not None and len(self.history):
            hist = self.history.copy()
            if "date" in hist.columns:
                hist["date"] = pd.to_datetime(hist["date"]).astype(str)
        return {
            "ticker": self.ticker,
            "name": self.name,
            "asof": self.asof,
            "last_price": self.last_price,
            "regime": self.regime,
            "probs": {str(k): float(v) for k, v in self.probs.items()},
            "history": hist.to_dict(orient="records") if len(hist) else [],
            "cones": cones,
            "supports": self.supports,
            "resistances": self.resistances,
            "character": self.character,
            "decision": self.decision,
            "mu": self.mu,
            "vol": self.vol,
            "model_meta": self.model_meta,
            "horizons": list(self.horizons) or sorted(int(k) for k in cones.keys()),
            "beat_baseline": {str(k): bool(v) for k, v in self.beat_baseline.items()},
            "brier_skill": {str(k): float(v) for k, v in self.brier_skill.items() if v == v},
            "run_id": self.run_id,
            "art_dir": self.art_dir,
        }


def session_to_frames(session: dict[str, Any]) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    hist = pd.DataFrame(session.get("history") or [])
    if len(hist) and "date" in hist.columns:
        hist["date"] = pd.to_datetime(hist["date"])
    cones: dict[int, pd.DataFrame] = {}
    for k, rows in (session.get("cones") or {}).items():
        try:
            hi = int(k)
        except Exception:
            continue
        df = pd.DataFrame(rows)
        if len(df) and "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        cones[hi] = df
    return hist, cones


def _normalize_history(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["date", "close"])
    out = df.copy()
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

    if isinstance(result.get("history"), pd.DataFrame) and len(result["history"]):
        h = _normalize_history(result["history"])
        if len(h):
            return h.tail(500)

    art = Path(result.get("art_dir") or "")
    for name in ("history.parquet", "history.csv"):
        p = art / name
        if p.exists():
            try:
                h = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
                h = _normalize_history(h)
                if len(h):
                    return h.tail(500)
            except Exception:
                pass
    export_xlsx = art / "export.xlsx"
    if export_xlsx.exists():
        try:
            h = _normalize_history(pd.read_excel(export_xlsx, sheet_name="history_tail"))
            if len(h):
                return h
        except Exception:
            pass
    try:
        from stock_prob.ingest import fetch_symbol

        raw = fetch_symbol(ticker, period="5y", use_cache=True)
        return _normalize_history(raw[["date", "close"]]).tail(500)
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

    hist = _normalize_history(history)
    if len(hist) > 30:
        s = hist.set_index(pd.to_datetime(hist["date"]))["close"].astype(float)
        r = log_returns(s).dropna()
        mu = float(result.get("mu")) if result.get("mu") is not None else (
            float(r.tail(60).mean()) if len(r) else 0.0
        )
        vol = float(result.get("vol")) if result.get("vol") is not None else (
            float(r.tail(60).std()) if len(r) else 0.02
        )
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
    from stock_prob.decision import build_decision
    from stock_prob.horizon_keys import normalize_prob_map, parse_horizon_key
    from stock_prob.structure import market_character, support_resistance

    ticker = str(result.get("ticker") or "?")
    probs = normalize_prob_map(result.get("live_probs") or {})
    if not probs and result.get("ensemble_live"):
        probs = normalize_prob_map(result.get("ensemble_live"))

    horizons: list[int] = []
    for h in result.get("horizons") or []:
        ph = parse_horizon_key(h)
        if ph is not None:
            horizons.append(ph)
    for k in probs.keys():
        ph = parse_horizon_key(k)
        if ph is not None and ph not in horizons:
            horizons.append(ph)
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

    # levels / character
    levels = result.get("levels") or {}
    supports = list(levels.get("supports") or [])
    resistances = list(levels.get("resistances") or [])
    character = dict(result.get("character") or {})
    if (not supports and not resistances) and history is not None and len(history) > 40:
        try:
            s = history.set_index(pd.to_datetime(history["date"]))["close"]
            levels = support_resistance(s)
            supports = list(levels.get("supports") or [])
            resistances = list(levels.get("resistances") or [])
            if not character:
                character = market_character(s)
        except Exception:
            pass

    mu = float(result.get("mu") or 0.0)
    vol = float(result.get("vol") or 0.02)
    if (not np.isfinite(vol) or vol <= 0) and history is not None and len(history) > 30:
        r = log_returns(history.set_index(pd.to_datetime(history["date"]))["close"]).dropna()
        mu = float(r.tail(60).mean()) if len(r) else 0.0
        vol = float(r.tail(60).std()) if len(r) else 0.02

    primary_h = None
    if horizons:
        primary_h = int(sorted(horizons)[0])
    decision = build_decision(
        probs=probs,
        beat_baseline=beat,
        brier_skill=skill,
        last_price=last_price,
        supports=supports,
        resistances=resistances,
        character=character,
        primary_horizon=primary_h,
    )

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

    model_meta = dict(result.get("model_meta") or {})
    if not model_meta:
        model_meta = {
            "mu": mu,
            "vol": vol,
            "n_bars": int(len(history)) if history is not None else 0,
            "cone_method": "gbm_monte_carlo",
            "direction_model": "logistic_regression_sklearn",
        }

    regime = str(result.get("regime") or "n/a")
    if character.get("label") and regime in ("n/a", "na", "", "UNKNOWN"):
        # still show vol regime from classify if missing
        pass

    return PrismViewModel(
        ticker=ticker,
        name=name or ticker,
        asof=asof,
        last_price=last_price,
        regime=regime,
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
        supports=supports,
        resistances=resistances,
        character=character,
        decision=decision,
        mu=mu,
        vol=vol,
        model_meta=model_meta,
        horizons=sorted(set(int(h) for h in horizons)),
    )
