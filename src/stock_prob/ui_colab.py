"""
Prism Colab entry.

IMPORTANT: launch_gui() always uses Gradio (ui_gradio), never the old ipywidgets form.
If you still see "Equity / Dom index / Horizons" text fields only, you loaded a
stale module — this file purges stock_prob.* from sys.modules on launch.
"""
from __future__ import annotations

import importlib
import json
import sys
import traceback
from typing import Any

import pandas as pd

from stock_prob.config import RunConfig, universe_from_symbols
from stock_prob.lab import enrich_single
from stock_prob.paths import get_project_root
from stock_prob.pipeline import run_pipeline


def _purge_stock_prob_modules() -> None:
    """Drop cached imports so Colab cannot keep serving an old ui_colab."""
    for k in list(sys.modules):
        if k == "stock_prob" or k.startswith("stock_prob."):
            del sys.modules[k]


def run_once(
    equity: str,
    *,
    domestic_index: str = "^JKSE",
    us_index: str = "^GSPC",
    macro: str = "^VIX",
    horizons: list[int] | None = None,
    use_lab: bool = True,
    mc_paths: int = 800,
    speed: str = "fast",
) -> dict[str, Any]:
    """
    Programmatic one-shot run.

    speed:
      - "fast" (default for GUI): fewer MC paths, coarser walk-forward, no in-loop cone MC
      - "full": heavier research backtest (slower; Colab free is only ~2 CPUs)
    """
    horizons = horizons or [5, 21, 252]
    equity = equity.strip()
    uni = universe_from_symbols(
        [equity],
        domestic_index=domestic_index.strip() or None,
        us_index=us_index.strip() or None,
        macro=macro.strip() or None,
    )
    if equity.upper().endswith(".JK") and (not domestic_index or domestic_index == "^GSPC"):
        uni.domestic_index = domestic_index.strip() if domestic_index.strip() else "^JKSE"
    elif not equity.upper().endswith(".JK") and domestic_index.strip() in ("", "^JKSE"):
        uni.domestic_index = "^GSPC"

    fast = (speed or "fast").lower() != "full"
    # Colab free tier ≈ 2 CPUs: GPU does not accelerate logistic/GARCH/MC much
    cfg = RunConfig(
        universe=uni,
        horizons=list(horizons),
        mc_paths=min(mc_paths, 400) if fast else mc_paths,
        min_train_rows=160 if fast else 200,
        walkforward_refit_every=63 if fast else 21,
        cone_diagnostics=not fast,
        max_oos_per_horizon=40 if fast else None,
        speed="fast" if fast else "full",
        run_name="gui",
    )
    # Advanced lab on free Colab is slow; still allowed but user chooses Lanjutan
    root = get_project_root()
    if use_lab and not fast:
        return enrich_single(cfg, equity=equity, root=root, use_cache=True)
    if use_lab and fast:
        # light enrich still does GARCH once — skip full lab on fast for speed
        return run_pipeline(cfg, equity=equity, root=root, use_cache=True)
    return run_pipeline(cfg, equity=equity, root=root, use_cache=True)


def render_result(result: dict[str, Any]) -> None:
    """Notebook-friendly render via ViewModel + HTML report path."""
    from IPython.display import HTML, display

    from stock_prob.report_html import write_prism_report
    from stock_prob.viewmodel import build_viewmodel
    from stock_prob.viz import build_fan_figure, build_metrics_figure, build_prob_gauge_figure

    vm = build_viewmodel(result)
    if vm.art_dir:
        from pathlib import Path

        p = Path(vm.art_dir) / "prism_report.html"
        write_prism_report(vm, p)
        display(HTML(f"<p><b>Report:</b> <code>{p}</code></p>"))
        display(HTML(p.read_text()))
    else:
        h = vm.primary_horizon()
        if vm.history is not None and h in vm.cones:
            build_fan_figure(vm.history, vm.cones[h]).show()
        if vm.probs:
            build_prob_gauge_figure(vm.probs).show()
        if vm.metrics is not None and len(vm.metrics):
            build_metrics_figure(vm.metrics).show()


def launch_gui(**kwargs: Any) -> Any:
    """
    Launch Gradio Prism desk.

    Always reloads ui_gradio from disk to defeat Colab import cache.
    """
    print("=" * 64)
    print("Prism launch_gui → Gradio (NOT ipywidgets)")
    # Prove which file is executing
    print("ui_colab file:", __file__)

    # Hard fail if someone still has a ghost old function body somehow
    import stock_prob.design as design

    importlib.reload(design)
    print("UX_BUILD:", design.UX_BUILD)
    print("UX_LABEL:", design.UX_LABEL)
    if "PRISM_UX_" not in design.UX_BUILD:
        raise RuntimeError(
            "Stale Prism package without UX stamp. "
            "Runtime → Disconnect and delete runtime, then clone again."
        )

    # Reload gradio UI module from disk
    import stock_prob.ui_gradio as ui_gradio

    importlib.reload(ui_gradio)
    print("ui_gradio file:", ui_gradio.__file__)
    print("=" * 64)

    # Default Colab-friendly kwargs
    try:
        import google.colab  # noqa: F401

        in_colab = True
    except Exception:
        in_colab = False

    if in_colab:
        # share=True is the reliable way for Gradio to render in Colab
        # (inline-only often fails / shows nothing / confuses with old outputs)
        kwargs.setdefault("share", True)
        print(
            "Colab detected: launching Gradio with share=True "
            "(temporary *.gradio.live link + embed). "
            "This is NOT the old Equity/Dom-index text form."
        )

    return ui_gradio.launch_gui(**kwargs)


def launch_tui() -> None:
    """Terminal wizard (text). Prefer launch_gui() for visuals."""
    print("=" * 56)
    print("  Prism Terminal UI (text only — use launch_gui for charts)")
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
    print(f"\nHTML report: {path}")
    print("Done.")
