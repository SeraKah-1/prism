"""End-to-end pipeline: dynamic universe → fetch → features → forecast → WF → artifacts."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stock_prob.backtest import walk_forward_equity
from stock_prob.config import RunConfig, UniverseConfig, save_run_config
from stock_prob.features import build_feature_frame, feature_columns, log_returns
from stock_prob.ingest import align_close_panel, fetch_universe
from stock_prob.labels import make_supervised
from stock_prob.ledger import append_ledger, new_run_id, write_run_artifacts
from stock_prob.models import cone_table, fit_logistic
from stock_prob.paths import ensure_layout, get_project_root
from stock_prob.viz import build_fan_figure, build_metrics_figure, export_metrics_excel, write_html_report


def resolve_context_closes(
    panel: pd.DataFrame, universe: UniverseConfig
) -> dict[str, pd.Series | None]:
    def get(sym: str | None) -> pd.Series | None:
        if sym and sym in panel.columns:
            return panel[sym].dropna()
        return None

    return {
        "domestic": get(universe.domestic_index),
        "us": get(universe.us_index),
        "macro": get(universe.macro),
    }


def run_pipeline(
    cfg: RunConfig,
    *,
    equity: str | None = None,
    root: Path | None = None,
    use_cache: bool = True,
    force_refresh: bool = False,
    write_ledger: bool = True,
) -> dict[str, Any]:
    """
    Run full lab core for one equity from cfg.universe.equities (or override `equity`).

    Universe symbols are never baked into algorithms — only taken from cfg.
    """
    root = root or get_project_root()
    paths = ensure_layout(root)

    equities = list(cfg.universe.equities)
    if equity is not None:
        if equity not in equities:
            equities = [equity] + equities
        target = equity
    else:
        if not equities:
            raise ValueError("universe.equities is empty — supply tickers via config/fetch")
        target = equities[0]

    symbols = cfg.universe.all_symbols()
    if target not in symbols:
        symbols = [target] + symbols

    frames = fetch_universe(
        symbols, period=cfg.history_period, use_cache=use_cache, force_refresh=force_refresh
    )
    panel = align_close_panel(frames)
    ctx = resolve_context_closes(panel, cfg.universe)

    eq_close = panel[target].dropna()
    feats = build_feature_frame(
        eq_close,
        domestic_close=ctx["domestic"],
        us_close=ctx["us"],
        macro_close=ctx["macro"],
        window=cfg.rolling_window,
    )

    # Live forecast at last available date with sufficient history
    fcols = feature_columns(feats)
    live_probs: dict[str, float] = {}
    live_cones: dict[int, pd.DataFrame] = {}
    last_date = eq_close.index.max()
    last_price = float(eq_close.iloc[-1])
    r = log_returns(eq_close).dropna()
    mu = float(r.tail(60).mean()) if len(r) else 0.0
    vol = float(r.tail(60).std()) if len(r) else 0.02
    vol = vol if vol > 0 else 0.02

    for h in cfg.horizons:
        # Live fit uses DENSE labels (no embargo). Embargo is only for WF OOS scoring.
        # Embargo-thinned 252d samples are too few (~4) to train; dense n is ~O(T-h).
        X, y, _ = make_supervised(
            feats, eq_close, h, feature_cols=fcols, use_embargo=False
        )
        if len(X) < 30 or y.nunique() < 2:
            live_probs[str(h)] = float("nan")
            continue
        model = fit_logistic(X, y, fcols, horizon=h, random_state=cfg.random_state)
        latest = feats[fcols].dropna().iloc[[-1]]
        p = float(model.predict_proba_up(latest)[0])
        if not (p == p):  # NaN guard
            live_probs[str(h)] = float("nan")
            continue
        live_probs[str(h)] = p
        live_cones[h] = cone_table(
            last_date,
            last_price,
            mu,
            vol,
            h,
            n_paths=cfg.mc_paths,
            random_state=cfg.random_state,
        )

    # Walk-forward OOS (fast GUI skips in-loop MC diagnostics)
    wf = walk_forward_equity(
        feats,
        eq_close,
        cfg.horizons,
        min_train_rows=cfg.min_train_rows,
        refit_every=cfg.walkforward_refit_every,
        rolling_window=cfg.rolling_window,
        mc_paths=min(cfg.mc_paths, 800),
        random_state=cfg.random_state,
        cone_diagnostics=bool(getattr(cfg, "cone_diagnostics", True)),
        max_oos_per_horizon=getattr(cfg, "max_oos_per_horizon", None),
    )

    run_id = new_run_id(prefix=cfg.run_name)
    config_dict = cfg.to_dict()
    config_dict["target_equity"] = target
    config_dict["symbols_fetched"] = list(symbols)

    art_dir = write_run_artifacts(
        run_id,
        config=config_dict,
        predictions=wf.predictions,
        metrics=wf.metrics,
        extra={"live_probs": live_probs, "last_price": last_price, "last_date": str(last_date)},
        root=root,
    )
    save_run_config(cfg, art_dir / "run_config.yaml")

    # primary cone for viz: shortest horizon with data, else first
    primary_h = None
    for h in cfg.horizons:
        if h in live_cones:
            primary_h = h
            break
    hist = frames[target][["date", "close"]].copy()
    # limit history display
    hist = hist.tail(400)

    fan = None
    metrics_fig = None
    report_path = None
    excel_path = None
    if primary_h is not None:
        fan = build_fan_figure(
            hist,
            live_cones[primary_h],
            title=f"{target} — {primary_h}d cone",
        )
        metrics_fig = build_metrics_figure(wf.metrics, title=f"{target} Brier vs baselines")
        report_path = write_html_report(
            art_dir / "report.html",
            fan_fig=fan,
            metrics_fig=metrics_fig,
            metrics_df=wf.metrics,
            probs=live_probs,
            meta={
                "run_id": run_id,
                "ticker": target,
                "universe_equities": ",".join(cfg.universe.equities),
                "domestic_index": cfg.universe.domestic_index,
                "us_index": cfg.universe.us_index,
                "macro": cfg.universe.macro,
                "last_date": str(last_date.date()) if hasattr(last_date, "date") else str(last_date),
                "last_price": f"{last_price:.4f}",
            },
        )
        # gallery copy
        gallery = paths["gallery"] / f"{run_id}_{_safe(target)}.html"
        gallery.write_text(report_path.read_text())
        report_path = report_path  # keep run-local as primary

        excel_path = export_metrics_excel(
            art_dir / "export.xlsx",
            {
                "metrics": wf.metrics,
                "predictions": wf.predictions.head(5000) if len(wf.predictions) else pd.DataFrame(),
                "live_cone": live_cones[primary_h],
                "history_tail": hist,
            },
        )

    if write_ledger and len(wf.predictions):
        led = wf.predictions.copy()
        led["run_id"] = run_id
        led["ticker"] = target
        led["model_version"] = "logistic_v1"
        append_ledger(led, root=root)

    # also dump live probs
    pd.DataFrame(
        [{"horizon": int(k), "prob_up": v} for k, v in live_probs.items()]
    ).to_csv(art_dir / "live_probs.csv", index=False)

    return {
        "run_id": run_id,
        "ticker": target,
        "art_dir": str(art_dir),
        "report_html": str(report_path) if report_path else None,
        "excel": str(excel_path) if excel_path else None,
        "live_probs": live_probs,
        "metrics": wf.metrics,
        "predictions": wf.predictions,
        "features_tail": feats.tail(5),
        "symbols": symbols,
    }


def run_universe(
    cfg: RunConfig,
    *,
    root: Path | None = None,
    use_cache: bool = True,
) -> list[dict[str, Any]]:
    """Run pipeline for every equity in the configured universe."""
    results = []
    for eq in cfg.universe.equities:
        results.append(
            run_pipeline(cfg, equity=eq, root=root, use_cache=use_cache, write_ledger=True)
        )
    return results


def _safe(symbol: str) -> str:
    return symbol.replace("^", "").replace("/", "_")
