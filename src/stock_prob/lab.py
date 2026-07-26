"""Full lab orchestrator: core pipeline + GARCH/regime/spillover/tournament/panel/twins."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from stock_prob.config import RunConfig, UniverseConfig, universe_from_symbols
from stock_prob.features import log_returns
from stock_prob.garch import garch_cone_table
from stock_prob.honesty import honesty_frame, honesty_score
from stock_prob.ingest import align_close_panel, fetch_universe
from stock_prob.ledger import new_run_id, run_dir
from stock_prob.panel import country_skill_table, market_of, run_panel, summarize_panel
from stock_prob.paths import ensure_layout, get_project_root
from stock_prob.pipeline import run_pipeline
from stock_prob.regime import classify_regime, latest_regime, regime_vol_multiplier
from stock_prob.spillover import build_spillover_frame, fit_spillover_model, spillover_probability
from stock_prob.surface import build_probability_surface, surface_matrix
from stock_prob.tournament import (
    blend_probs,
    inverse_brier_weights,
    promote_champion,
    score_models_on_predictions,
    tournament_table,
)
from stock_prob.twins import cluster_twins, form_feature_vector, twin_pairs
from stock_prob.viz import build_fan_figure, build_metrics_figure, write_html_report


def enrich_single(
    cfg: RunConfig,
    *,
    equity: str | None = None,
    root: Path | None = None,
    use_cache: bool = True,
) -> dict[str, Any]:
    """Core pipeline + GARCH cone, regime, spillover (IDX), tournament, honesty, surface."""
    root = root or get_project_root()
    base = run_pipeline(cfg, equity=equity, root=root, use_cache=use_cache)
    target = base["ticker"]
    art = Path(base["art_dir"])

    # Reload closes for enrichment
    symbols = list(cfg.universe.all_symbols())
    if target not in symbols:
        symbols = [target] + symbols
    frames = fetch_universe(symbols, period=cfg.history_period, use_cache=use_cache)
    panel = align_close_panel(frames)
    eq_close = panel[target].dropna()

    us_sym = cfg.universe.us_index
    us_close = panel[us_sym].dropna() if us_sym and us_sym in panel.columns else None
    dom_sym = cfg.universe.domestic_index
    dom_close = panel[dom_sym].dropna() if dom_sym and dom_sym in panel.columns else None

    # Regime
    reg = classify_regime(eq_close, us_close=us_close)
    regime = latest_regime(reg)
    mult = regime_vol_multiplier(regime)
    reg.tail(30).to_csv(art / "regime_tail.csv")

    # GARCH cones per horizon (+ regime width mult applied to vol via wider paths)
    garch_meta = {}
    garch_cones = {}
    for h in cfg.horizons:
        cone, fit = garch_cone_table(
            eq_close,
            h,
            n_paths=cfg.mc_paths,
            random_state=cfg.random_state + h,
        )
        # apply regime multiplier to band width around median
        if "p50" in cone.columns and mult != 1.0:
            for col in ("p10", "p25", "p75", "p90"):
                if col in cone.columns:
                    cone[col] = cone["p50"] + (cone[col] - cone["p50"]) * mult
        garch_cones[h] = cone
        garch_meta[str(h)] = fit
        cone.to_csv(art / f"garch_cone_{h}.csv", index=False)
    (art / "garch_meta.json").write_text(json.dumps(garch_meta, indent=2, default=str))

    # Spillover (meaningful for IDX)
    spill = {"status": "skipped", "market": market_of(target)}
    if market_of(target) == "IDX" and us_close is not None and target in frames:
        ohlc = frames[target]
        sp_frame = build_spillover_frame(ohlc, us_close)
        model = fit_spillover_model(sp_frame)
        us_last = float(us_close.pct_change().iloc[-1]) if len(us_close) > 1 else float("nan")
        spill = spillover_probability(model, us_last)
        spill["market"] = "IDX"
        sp_frame.tail(100).to_csv(art / "spillover_frame_tail.csv")
    (art / "spillover.json").write_text(json.dumps(spill, indent=2, default=str))

    # Tournament from WF preds
    preds = base.get("predictions")
    t_scores = score_models_on_predictions(preds if preds is not None else pd.DataFrame())
    t_board = tournament_table(t_scores)
    champion = promote_champion(t_board)
    # ensemble weights from mean brier per model
    weights = {}
    if len(t_board):
        weights = inverse_brier_weights(
            {r["model"]: r["mean_brier"] for _, r in t_board.iterrows()}
        )
    live = base.get("live_probs") or {}
    # map live model/base approx for blend (base from metrics mean)
    ens_probs = {}
    for h, p in live.items():
        pm = {"model": p}
        # use momentum 0.55 as weak third if needed
        sub = preds[preds["horizon"] == int(h)] if preds is not None and len(preds) else pd.DataFrame()
        if len(sub):
            pm["base"] = float(sub["prob_up_base"].iloc[-1])
            pm["momentum"] = float(sub["prob_up_momentum"].iloc[-1])
        ens_probs[h] = blend_probs(pm, weights) if weights else p
    if len(t_scores):
        t_scores.to_csv(art / "tournament_scores.csv", index=False)
    if len(t_board):
        t_board.to_csv(art / "tournament_board.csv", index=False)
    (art / "tournament_champion.json").write_text(json.dumps({"champion": champion, "weights": weights, "ensemble_live": ens_probs}, indent=2, default=str))

    # Honesty
    hf = honesty_frame(preds if preds is not None else pd.DataFrame())
    hon = honesty_score(hf)
    (art / "honesty.json").write_text(json.dumps(hon, indent=2))
    if len(hf):
        hf.tail(200).to_csv(art / "honesty_frame_tail.csv", index=False)

    # Probability surface
    surf = build_probability_surface(live)
    if len(surf):
        surf.to_csv(art / "probability_surface.csv", index=False)

    # Prefer GARCH cone for report if available
    primary_h = None
    for h in cfg.horizons:
        if h in garch_cones:
            primary_h = h
            break
    if primary_h is not None and target in frames:
        hist = frames[target][["date", "close"]].tail(500)
        levels = base.get("levels") or {}
        fan = build_fan_figure(
            hist,
            garch_cones[primary_h],
            title=f"{target} — GARCH cone {primary_h}d [{regime}]",
            supports=levels.get("supports"),
            resistances=levels.get("resistances"),
            horizon=int(primary_h),
            nested_cones=garch_cones,
            show_nested=len(garch_cones) > 1,
        )
        metrics_fig = build_metrics_figure(base["metrics"], title=f"{target} Brier tournament inputs")
        # Use bare horizon keys ("5") not "h5" — "h5" breaks primary_horizon/int parse
        clean_probs = {}
        for k, v in (live or {}).items():
            try:
                clean_probs[str(int(k))] = float(v)
            except Exception:
                try:
                    from stock_prob.horizon_keys import parse_horizon_key

                    hk = parse_horizon_key(k)
                    if hk is not None:
                        clean_probs[str(hk)] = float(v)
                except Exception:
                    pass
        try:
            # Attach history/cones so report can render charts even if probs keys were messy
            base_for_vm = dict(base)
            base_for_vm["live_probs"] = clean_probs or live
            if "history" not in base_for_vm and target in frames:
                base_for_vm["history"] = frames[target][["date", "close"]].tail(400)
            if garch_cones:
                base_for_vm["live_cones"] = garch_cones
            write_html_report(
                art / "report_lab.html",
                fan_fig=fan,
                metrics_fig=metrics_fig,
                metrics_df=base["metrics"],
                probs=clean_probs,
                meta={
                    "run_id": base["run_id"],
                    "ticker": target,
                    "regime": regime,
                    "regime_vol_mult": mult,
                    "garch_method": garch_meta.get(str(primary_h), {}).get("method"),
                    "spillover": spill,
                    "champion": champion,
                    "honesty_skill": hon.get("honesty_skill"),
                    "lab": "full",
                    "_result": base_for_vm,
                },
            )
        except Exception as e:
            # Never crash the whole analysis because of a report renderer bug
            (art / "report_lab_error.txt").write_text(str(e))

    base["regime"] = regime
    base["garch_meta"] = garch_meta
    base["spillover"] = spill
    base["tournament"] = {"board": t_board, "champion": champion, "weights": weights}
    base["ensemble_live"] = ens_probs
    base["honesty"] = hon
    base["surface"] = surf
    # Prefer GARCH cones in live payload when available
    if garch_cones:
        base["live_cones"] = garch_cones
    # update model meta for Advanced mode
    mm = dict(base.get("model_meta") or {})
    mm["cone_model"] = "garch11_or_sample_fallback"
    if garch_meta:
        sample_h = str(primary_h if primary_h is not None else next(iter(garch_meta), ""))
        if sample_h in garch_meta:
            mm["cone_method"] = garch_meta[sample_h].get("method", "garch")
            mm["mu"] = garch_meta[sample_h].get("mu", mm.get("mu"))
            mm["vol"] = garch_meta[sample_h].get("vol", mm.get("vol"))
            base["mu"] = mm.get("mu", base.get("mu"))
            base["vol"] = mm.get("vol", base.get("vol"))
    base["model_meta"] = mm
    base["lab_report"] = str(art / "report_lab.html") if (art / "report_lab.html").exists() else base.get("report_html")
    return base


def run_full_lab(
    equities: list[str],
    *,
    horizons: list[int] | None = None,
    root: Path | None = None,
    use_cache: bool = True,
    lab_name: str = "full_lab",
) -> dict[str, Any]:
    """Panel + enrich first N with lab modules + twins + country study."""
    root = root or get_project_root()
    paths = ensure_layout(root)
    horizons = horizons or [5, 21, 252]
    lab_id = new_run_id(prefix=lab_name)
    lab_dir = run_dir(lab_id, root)

    # Panel WF for all equities
    panel_results = run_panel(
        equities,
        horizons=horizons,
        root=root,
        use_cache=use_cache,
        run_name=lab_name,
        mc_paths=500,
        min_train_rows=200,
    )
    panel_df = summarize_panel(panel_results)
    country = country_skill_table(panel_df)
    if len(panel_df):
        panel_df.to_csv(lab_dir / "panel_summary.csv", index=False)
        panel_df.to_parquet(lab_dir / "panel_summary.parquet", index=False)
    if len(country):
        country.to_csv(lab_dir / "country_skill.csv", index=False)

    # Enrich up to 4 representatives (speed)
    enriched = []
    for r in panel_results:
        if r.get("error") or not r.get("ticker"):
            continue
        if len(enriched) >= 4:
            break
        mkt = r.get("market", market_of(r["ticker"]))
        dom = "^JKSE" if mkt == "IDX" else "^GSPC"
        uni = universe_from_symbols(
            [r["ticker"]], domestic_index=dom, us_index="^GSPC", macro="^VIX"
        )
        cfg = RunConfig(
            universe=uni,
            horizons=list(horizons),
            mc_paths=800,
            min_train_rows=200,
            run_name=f"{lab_name}_enrich",
        )
        try:
            en = enrich_single(cfg, equity=r["ticker"], root=root, use_cache=use_cache)
            enriched.append(en)
        except Exception as e:
            enriched.append({"ticker": r["ticker"], "error": str(e)[:200]})

    # Twins across panel symbols that fetched
    form_rows = []
    all_syms = set(equities) | {"^JKSE", "^GSPC", "^VIX"}
    frames = fetch_universe(list(all_syms), period="5y", use_cache=use_cache)
    panel_px = align_close_panel(frames)
    for eq in equities:
        if eq not in panel_px.columns:
            continue
        mkt = market_of(eq)
        dom = panel_px["^JKSE"] if mkt == "IDX" and "^JKSE" in panel_px.columns else panel_px.get("^GSPC")
        us = panel_px.get("^GSPC")
        vec = form_feature_vector(panel_px[eq].dropna(), dom, us)
        vec["ticker"] = eq
        form_rows.append(vec)
    twins_df = cluster_twins(form_rows)
    pairs = twin_pairs(twins_df)
    if len(twins_df):
        twins_df.to_csv(lab_dir / "twins_clusters.csv", index=False)
    if len(pairs):
        pairs.to_csv(lab_dir / "twin_pairs.csv", index=False)

    # Surface matrix
    live_items = [{"ticker": r.get("ticker"), "live_probs": r.get("live_probs") or {}} for r in panel_results if not r.get("error")]
    smat = surface_matrix(live_items)
    if len(smat):
        smat.to_csv(lab_dir / "probability_surface_matrix.csv", index=False)

    # Lab index HTML
    country_html = country.to_html(index=False) if len(country) else "<p>No country table</p>"
    panel_html = panel_df.head(50).to_html(index=False) if len(panel_df) else "<p>No panel</p>"
    twins_html = twins_df.to_html(index=False) if len(twins_df) else "<p>No twins</p>"
    links = []
    for en in enriched:
        if en.get("lab_report"):
            links.append(f'<li><a href="{en["lab_report"]}">{en.get("ticker")}</a> regime={en.get("regime")}</li>')
        elif en.get("report_html"):
            links.append(f'<li><a href="{en["report_html"]}">{en.get("ticker")}</a></li>')
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/><title>SPE Full Lab {lab_id}</title>
<style>
body{{font-family:system-ui,sans-serif;margin:24px;background:#0f172a;color:#e2e8f0}}
h1,h2{{color:#38bdf8}} a{{color:#7dd3fc}}
.card{{background:#1e293b;border-radius:12px;padding:16px;margin:12px 0}}
table{{border-collapse:collapse;width:100%;font-size:13px}}
th,td{{border-bottom:1px solid #334155;padding:6px 8px;text-align:left}}
</style></head><body>
<h1>Prism — Full Lab</h1>
<div class="card"><p><b>lab_id</b>: {lab_id}<br/><b>created</b>: {datetime.now(timezone.utc).isoformat()}<br/>
<b>equities</b>: {', '.join(equities)}</p></div>
<div class="card"><h2>Country skill (IDX vs US)</h2>{country_html}</div>
<div class="card"><h2>Panel summary (head)</h2>{panel_html}</div>
<div class="card"><h2>Twin clusters (form)</h2>{twins_html}</div>
<div class="card"><h2>Enriched reports</h2><ul>{''.join(links) or '<li>none</li>'}</ul></div>
</body></html>"""
    index_path = lab_dir / "index.html"
    index_path.write_text(html)
    # gallery copy
    gal = paths["gallery"] / f"{lab_id}_index.html"
    gal.write_text(html)

    meta = {
        "lab_id": lab_id,
        "equities": equities,
        "horizons": horizons,
        "n_panel_ok": sum(1 for r in panel_results if not r.get("error")),
        "n_enriched": len(enriched),
        "index_html": str(index_path),
        "country_skill": country.to_dict(orient="records") if len(country) else [],
    }
    (lab_dir / "lab_meta.json").write_text(json.dumps(meta, indent=2, default=str))

    return {
        "lab_id": lab_id,
        "lab_dir": str(lab_dir),
        "index_html": str(index_path),
        "panel_results": panel_results,
        "panel_df": panel_df,
        "country": country,
        "enriched": enriched,
        "twins": twins_df,
        "twin_pairs": pairs,
        "surface_matrix": smat,
        "meta": meta,
    }
