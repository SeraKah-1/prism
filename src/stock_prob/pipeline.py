"""End-to-end pipeline: dynamic universe → fetch → features → forecast → WF → artifacts."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stock_prob.backtest import walk_forward_equity
from stock_prob.config import RunConfig, UniverseConfig, save_run_config
from stock_prob.features import build_feature_frame
from stock_prob.forecast import compute_live_forecast, history_frame
from stock_prob.ingest import align_close_panel, fetch_universe
from stock_prob.ledger import append_ledger, new_run_id, write_run_artifacts
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
    # Heal short equity series (corrupt stub cache was writing 6-bar files)
    if target in frames and len(frames[target]) < 40:
        from stock_prob.ingest import fetch_symbol

        frames[target] = fetch_symbol(
            target, period="max", use_cache=True, force_refresh=True, min_bars=1
        )
    panel = align_close_panel(frames)
    ctx = resolve_context_closes(panel, cfg.universe)

    if target not in panel.columns:
        raise ValueError(f"No price series for {target} after fetch.")
    eq_close = panel[target].dropna()
    # Prefer native frame length if panel alignment dropped rows incorrectly
    if len(eq_close) < 40 and target in frames and len(frames[target]) >= 40:
        eq_close = frames[target].set_index("date")["close"].astype(float).dropna()

    feats = build_feature_frame(
        eq_close,
        domestic_close=ctx["domestic"],
        us_close=ctx["us"],
        macro_close=ctx["macro"],
        window=min(cfg.rolling_window, max(20, len(eq_close) // 3)),
    )

    # Live forecast — robust helper (cone even if logistic fails)
    if len(eq_close) < 40:
        raise ValueError(
            f"History too short for {target}: {len(eq_close)} bars (need ~40+). "
            f"Cache may be corrupt — try again; fetch will force-refresh short stubs."
        )
    live = compute_live_forecast(
        feats,
        eq_close,
        list(cfg.horizons),
        mc_paths=cfg.mc_paths,
        random_state=cfg.random_state,
        min_train=30,
    )
    live_probs: dict[str, float] = dict(live["live_probs"])
    live_cones: dict[int, pd.DataFrame] = dict(live["live_cones"])
    last_date = live["last_date"]
    last_price = float(live["last_price"])
    forecast_errors = live.get("errors") or {}

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

    hist = history_frame(eq_close, n=400)
    # Always persist history (even if logistic probs empty) so UI can rebuild charts
    art_dir = write_run_artifacts(
        run_id,
        config=config_dict,
        predictions=wf.predictions,
        metrics=wf.metrics,
        extra={
            "live_probs": live_probs,
            "last_price": last_price,
            "last_date": str(last_date),
            "forecast_errors": forecast_errors,
            "n_bars": int(len(eq_close)),
        },
        root=root,
    )
    save_run_config(cfg, art_dir / "run_config.yaml")
    hist.to_parquet(art_dir / "history.parquet", index=False)
    hist.to_csv(art_dir / "history.csv", index=False)

    primary_h = None
    for h in cfg.horizons:
        if int(h) in live_cones:
            primary_h = int(h)
            break
    if primary_h is None and live_cones:
        primary_h = sorted(live_cones.keys())[0]

    fan = None
    metrics_fig = None
    report_path = None
    excel_path = None
    # Always write excel + report when we have history (not only when probs exist)
    sheets = {
        "metrics": wf.metrics if wf.metrics is not None else pd.DataFrame(),
        "predictions": wf.predictions.head(5000) if len(wf.predictions) else pd.DataFrame(),
        "history_tail": hist,
        "live_probs": pd.DataFrame(
            [{"horizon": int(k), "prob_up": v} for k, v in live_probs.items()]
        ),
    }
    if primary_h is not None:
        sheets["live_cone"] = live_cones[primary_h]
        try:
            fan = build_fan_figure(
                hist,
                live_cones[primary_h],
                title=f"{target} — {primary_h}d cone",
            )
        except Exception:
            fan = None
    if wf.metrics is not None and len(wf.metrics):
        try:
            metrics_fig = build_metrics_figure(wf.metrics, title=f"{target} Brier vs baselines")
        except Exception:
            metrics_fig = None
    excel_path = export_metrics_excel(art_dir / "export.xlsx", sheets)
    try:
        report_path = write_html_report(
            art_dir / "report.html",
            fan_fig=fan,
            metrics_fig=metrics_fig,
            metrics_df=wf.metrics if wf.metrics is not None else pd.DataFrame(),
            probs=live_probs,
            meta={
                "run_id": run_id,
                "ticker": target,
                "universe_equities": ",".join(cfg.universe.equities),
                "domestic_index": cfg.universe.domestic_index,
                "us_index": cfg.universe.us_index,
                "macro": cfg.universe.macro,
                "last_date": str(pd.Timestamp(last_date).date()),
                "last_price": f"{last_price:.4f}",
            },
        )
        gallery = paths["gallery"] / f"{run_id}_{_safe(target)}.html"
        gallery.write_text(report_path.read_text())
    except Exception:
        report_path = None

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
        "live_cones": live_cones,
        "history": hist,
        "last_price": last_price,
        "last_date": last_date,
        "forecast_errors": forecast_errors,
        "metrics": wf.metrics,
        "predictions": wf.predictions,
        "features_tail": feats.tail(5),
        "symbols": symbols,
        "ok": bool(live_probs) or bool(live_cones),
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
