"""
Colab / Jupyter GUI (no web server).

Uses ipywidgets + Plotly inline so you can change tickers, run the pipeline,
and see fan charts + tables inside the notebook cell output.
"""
from __future__ import annotations

import json
import traceback
from typing import Any, Callable

import pandas as pd

from stock_prob.config import RunConfig, universe_from_symbols
from stock_prob.lab import enrich_single
from stock_prob.paths import get_project_root
from stock_prob.pipeline import run_pipeline
from stock_prob.viz import build_fan_figure, build_metrics_figure


def _try_display(obj: Any) -> None:
    try:
        from IPython.display import display

        display(obj)
    except Exception:
        print(obj)


def _try_html(html: str) -> None:
    try:
        from IPython.display import HTML, display

        display(HTML(html))
    except Exception:
        print(html)


def run_once(
    equity: str,
    *,
    domestic_index: str = "^JKSE",
    us_index: str = "^GSPC",
    macro: str = "^VIX",
    horizons: list[int] | None = None,
    use_lab: bool = True,
    mc_paths: int = 800,
) -> dict[str, Any]:
    """Programmatic one-shot run (also used by the widget GUI)."""
    horizons = horizons or [5, 21, 252]
    equity = equity.strip()
    uni = universe_from_symbols(
        [equity],
        domestic_index=domestic_index.strip() or None,
        us_index=us_index.strip() or None,
        macro=macro.strip() or None,
    )
    # Auto domestic index: IDX stocks use JKSE unless user overrode
    if equity.upper().endswith(".JK") and (not domestic_index or domestic_index == "^GSPC"):
        uni.domestic_index = domestic_index.strip() if domestic_index.strip() else "^JKSE"
    elif not equity.upper().endswith(".JK") and domestic_index.strip() in ("", "^JKSE"):
        uni.domestic_index = "^GSPC"

    cfg = RunConfig(
        universe=uni,
        horizons=list(horizons),
        mc_paths=mc_paths,
        min_train_rows=200,
        run_name="gui",
    )
    root = get_project_root()
    if use_lab:
        return enrich_single(cfg, equity=equity, root=root, use_cache=True)
    return run_pipeline(cfg, equity=equity, root=root, use_cache=True)


def render_result(result: dict[str, Any]) -> None:
    """Show summary, tables, and Plotly figures in the notebook."""
    ticker = result.get("ticker", "?")
    probs = result.get("live_probs") or {}
    ens = result.get("ensemble_live") or {}
    metrics = result.get("metrics")
    regime = result.get("regime", "n/a")
    spill = result.get("spillover") or {}
    honesty = result.get("honesty") or {}
    champ = (result.get("tournament") or {}).get("champion") or {}

    _try_html(
        f"""
        <div style="font-family:system-ui;padding:12px 16px;background:#0f172a;color:#e2e8f0;
                    border-radius:12px;margin-bottom:12px">
          <h2 style="margin:0 0 8px;color:#38bdf8">Prism — {ticker}</h2>
          <div><b>run_id</b>: {result.get('run_id')}</div>
          <div><b>regime</b>: {regime} &nbsp;|&nbsp; <b>champion</b>: {champ.get('champion')}</div>
          <div><b>honesty_skill</b>: {honesty.get('honesty_skill')}
               &nbsp;|&nbsp; <b>spillover</b>: {spill}</div>
          <div><b>report</b>: <code>{result.get('lab_report') or result.get('report_html')}</code></div>
        </div>
        """
    )

    # Prob table
    rows = []
    for h, p in sorted(probs.items(), key=lambda x: int(x[0])):
        rows.append(
            {
                "horizon": int(h),
                "P(up) model": p,
                "P(up) ensemble": ens.get(h, ens.get(str(h))),
            }
        )
    if rows:
        _try_html("<h3>Live probabilities</h3>")
        _try_display(pd.DataFrame(rows))

    if metrics is not None and len(metrics):
        _try_html("<h3>Walk-forward metrics vs baselines (lower Brier is better)</h3>")
        _try_display(metrics)

    # Fan chart: GARCH CSV → else rebuild short cone from last close in metrics path
    try:
        from pathlib import Path

        from stock_prob.ingest import fetch_symbol
        from stock_prob.models import cone_table
        from stock_prob.features import log_returns

        art = Path(result.get("art_dir") or "")
        cone = None
        primary_h = None
        for h in (5, 21, 63, 252):
            p = art / f"garch_cone_{h}.csv"
            if p.exists():
                cone = pd.read_csv(p)
                primary_h = h
                break
        export_xlsx = art / "export.xlsx"
        history = None
        if export_xlsx.exists():
            try:
                history = pd.read_excel(export_xlsx, sheet_name="history_tail")
            except Exception:
                history = None
        if history is None and ticker and ticker != "?":
            try:
                raw = fetch_symbol(ticker, period="2y", use_cache=True)
                history = raw[["date", "close"]].tail(400)
            except Exception:
                history = None
        if history is None and result.get("predictions") is not None and len(result["predictions"]):
            history = result["predictions"][["date", "close"]].drop_duplicates("date").tail(200)

        if cone is None and history is not None and len(history) > 30:
            primary_h = 21
            if probs:
                primary_h = int(sorted(probs.keys(), key=lambda x: int(x))[0])
            s = history.set_index("date")["close"].astype(float)
            r = log_returns(s).dropna()
            mu = float(r.tail(60).mean()) if len(r) else 0.0
            vol = float(r.tail(60).std()) if len(r) else 0.02
            cone = cone_table(
                pd.Timestamp(history["date"].iloc[-1]),
                float(s.iloc[-1]),
                mu,
                max(vol, 1e-4),
                primary_h,
                n_paths=600,
            )

        if history is not None and cone is not None:
            fig = build_fan_figure(
                history,
                cone,
                title=f"{ticker} — cone {primary_h}d [{regime}]",
            )
            _try_html("<h3>Fan chart (prediction cone)</h3>")
            fig.show()
        if metrics is not None and len(metrics):
            figm = build_metrics_figure(metrics, title=f"{ticker} Brier vs baselines")
            _try_html("<h3>Metrics chart</h3>")
            figm.show()
    except Exception as e:
        print("Plot note:", e)

    surface = result.get("surface")
    if surface is not None and len(surface):
        _try_html("<h3>Probability surface</h3>")
        _try_display(surface)


def launch_gui(
    default_equity: str = "BBCA.JK",
    default_domestic: str = "^JKSE",
    default_us: str = "^GSPC",
    default_macro: str = "^VIX",
) -> Any:
    """
    Launch interactive controls in Colab/Jupyter.

    Usage in a notebook cell:
        from stock_prob.ui_colab import launch_gui
        launch_gui()
    """
    try:
        import ipywidgets as W
        from IPython.display import clear_output, display
    except ImportError as e:
        raise ImportError("ipywidgets + IPython required for GUI") from e

    equity = W.Text(value=default_equity, description="Equity", style={"description_width": "90px"})
    domestic = W.Text(value=default_domestic, description="Dom index")
    us_idx = W.Text(value=default_us, description="US index")
    macro = W.Text(value=default_macro, description="Macro")
    horizons = W.SelectMultiple(
        options=[5, 21, 63, 126, 252],
        value=[5, 21, 252],
        description="Horizons",
        rows=5,
    )
    use_lab = W.Checkbox(value=True, description="Full lab (GARCH/regime/tournament)")
    run_btn = W.Button(description="▶ Run analysis", button_style="primary", icon="play")
    status = W.HTML(value="<i>Set tickers then click Run. Data is cached after first fetch.</i>")
    out = W.Output()

    def on_run(_):
        with out:
            clear_output(wait=True)
            status.value = "<b style='color:#38bdf8'>Running…</b> (first fetch may take ~10–30s)"
            try:
                res = run_once(
                    equity.value,
                    domestic_index=domestic.value,
                    us_index=us_idx.value,
                    macro=macro.value,
                    horizons=list(horizons.value) or [5, 21, 252],
                    use_lab=bool(use_lab.value),
                )
                render_result(res)
                status.value = (
                    f"<b style='color:#4ade80'>Done.</b> run_id=<code>{res.get('run_id')}</code>"
                )
            except Exception:
                status.value = "<b style='color:#f87171'>Error</b> — see traceback below"
                traceback.print_exc()

    run_btn.on_click(on_run)

    # Presets
    def preset_idx(_):
        equity.value = "BBCA.JK"
        domestic.value = "^JKSE"
        us_idx.value = "^GSPC"
        macro.value = "^VIX"

    def preset_us(_):
        equity.value = "AAPL"
        domestic.value = "^GSPC"
        us_idx.value = "^GSPC"
        macro.value = "^VIX"

    b_idx = W.Button(description="Preset IDX")
    b_us = W.Button(description="Preset US")
    b_idx.on_click(preset_idx)
    b_us.on_click(preset_us)

    header = W.HTML(
        """
        <div style="font-family:system-ui;margin-bottom:8px">
          <h2 style="margin:0;color:#0ea5e9">Prism — Lab GUI</h2>
          <p style="margin:4px 0 0;color:#64748b">
            Change any ticker anytime · no web server · results render below
          </p>
        </div>
        """
    )
    form = W.VBox(
        [
            header,
            W.HBox([b_idx, b_us, use_lab]),
            equity,
            domestic,
            us_idx,
            macro,
            horizons,
            run_btn,
            status,
            out,
        ]
    )
    display(form)
    return form


def launch_tui() -> None:
    """
    Simple terminal UI (no browser). Works in Colab terminal / local shell.
    """
    print("=" * 56)
    print("  Prism Terminal UI (TUI)")
    print("=" * 56)
    equity = input("Equity ticker [BBCA.JK]: ").strip() or "BBCA.JK"
    if equity.upper().endswith(".JK"):
        dom = input("Domestic index [^JKSE]: ").strip() or "^JKSE"
    else:
        dom = input("Domestic index [^GSPC]: ").strip() or "^GSPC"
    us = input("US index [^GSPC]: ").strip() or "^GSPC"
    macro = input("Macro [^VIX]: ").strip() or "^VIX"
    hraw = input("Horizons comma-sep [5,21,252]: ").strip() or "5,21,252"
    horizons = [int(x) for x in hraw.split(",") if x.strip()]
    lab = (input("Full lab enrich? [Y/n]: ").strip().lower() or "y").startswith("y")
    print("\nRunning…\n")
    res = run_once(
        equity,
        domestic_index=dom,
        us_index=us,
        macro=macro,
        horizons=horizons,
        use_lab=lab,
    )
    print(f"ticker     : {res.get('ticker')}")
    print(f"run_id     : {res.get('run_id')}")
    print(f"regime     : {res.get('regime')}")
    print(f"live P(up) : {json.dumps(res.get('live_probs'), indent=2)}")
    if res.get("metrics") is not None and len(res["metrics"]):
        print("\nmetrics:")
        print(res["metrics"].to_string(index=False))
    path = res.get("lab_report") or res.get("report_html")
    print(f"\nHTML report (open file / download from Drive): {path}")
    print("Done.")
