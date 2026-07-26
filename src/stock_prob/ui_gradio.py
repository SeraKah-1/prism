"""
Prism Gradio UI — Colab-friendly, no public web deploy required.

- Dynamic ticker search (yfinance), combobox-style dropdown
- In-memory ViewModel → fan chart + metric cards
- Writes shadcn-like HTML report
"""
from __future__ import annotations

import traceback
from typing import Any

import pandas as pd

from stock_prob.report_html import write_prism_report
from stock_prob.tickers import (
    default_context_for_symbol,
    parse_symbol_from_label,
    resolve_ticker,
    search_labels,
    search_tickers,
)
from stock_prob.ui_colab import run_once
from stock_prob.viewmodel import build_viewmodel
from stock_prob.viz import build_fan_figure, build_metrics_figure


def _choices_for_query(q: str) -> list[str]:
    labels = search_labels(q or "", max_results=15)
    return labels if labels else ([] if not (q or "").strip() else [q.strip()])


def on_search_change(query: str):
    import gradio as gr

    ch = _choices_for_query(query)
    # keep current typed value selectable
    val = ch[0] if ch else (query or "")
    return gr.Dropdown(choices=ch, value=val)


def on_ticker_select(label: str):
    sym = parse_symbol_from_label(label or "")
    if not sym:
        return "^JKSE", "^GSPC", "^VIX", "Type a company or symbol, then pick a match."
    ctx = default_context_for_symbol(sym)
    hits = search_tickers(sym, max_results=3)
    name = hits[0].name if hits else sym
    return (
        ctx["domestic_index"],
        ctx["us_index"],
        ctx["macro"],
        f"Selected **{sym}** — {name}. Context indexes auto-filled (editable).",
    )


def run_analysis(
    ticker_label: str,
    domestic: str,
    us_index: str,
    macro: str,
    horizons: list[str] | list[int],
    use_lab: bool,
    progress=None,
):
    import gradio as gr
    import plotly.io as pio

    try:
        pio.renderers.default = "plotly_mimetype+notebook_connected"
    except Exception:
        pass

    empty_fig = None
    try:
        import plotly.graph_objects as go

        empty_fig = go.Figure()
        empty_fig.update_layout(
            template="plotly_white",
            height=420,
            title="Run analysis to see cone",
            margin=dict(l=40, r=20, t=50, b=40),
        )
    except Exception:
        empty_fig = None

    try:
        if progress is not None:
            progress(0.05, desc="Resolving ticker…")
        resolved = resolve_ticker(ticker_label or "")
        if not resolved.get("ok"):
            msg = f"❌ Ticker not resolved: {resolved.get('error') or ticker_label}"
            return msg, empty_fig, empty_fig, pd.DataFrame(), pd.DataFrame(), ""

        symbol = resolved["symbol"]
        name = resolved.get("name") or symbol

        hz = []
        for x in horizons or ["5", "21", "252"]:
            try:
                hz.append(int(x))
            except Exception:
                pass
        if not hz:
            hz = [5, 21, 252]

        if progress is not None:
            progress(0.2, desc=f"Running Prism on {symbol}…")

        result = run_once(
            symbol,
            domestic_index=domestic or "^GSPC",
            us_index=us_index or "^GSPC",
            macro=macro or "^VIX",
            horizons=hz,
            use_lab=bool(use_lab),
            mc_paths=700,
        )
        if progress is not None:
            progress(0.75, desc="Building charts…")

        vm = build_viewmodel(result, name=name)
        # always write pretty report next to artifacts
        report_path = ""
        if vm.art_dir:
            from pathlib import Path

            report_path = str(Path(vm.art_dir) / "prism_report.html")
            write_prism_report(vm, report_path)
            vm.report_html = report_path

        h = vm.primary_horizon()
        cone = vm.cones.get(h)
        fan = empty_fig
        if vm.history is not None and len(vm.history) and cone is not None:
            fan = build_fan_figure(
                vm.history,
                cone,
                title=f"{vm.ticker} · {h}d cone · regime {vm.regime}",
            )
            fan.update_layout(template="plotly_white", height=480)

        met = empty_fig
        if vm.metrics is not None and len(vm.metrics):
            met = build_metrics_figure(vm.metrics, title="Brier vs baselines (lower is better)")
            met.update_layout(template="plotly_white", height=360)

        cards = vm.summary_cards()
        rows = []
        for c in cards:
            rows.append(
                {
                    "horizon_d": c["horizon"],
                    "P_up_%": round(c["prob_up"] * 100, 2),
                    "direction": c["direction"],
                    "beat_baseline": c["beat_baseline"],
                    "brier_skill": None
                    if c["brier_skill"] is None
                    else round(float(c["brier_skill"]), 4),
                }
            )
        prob_df = pd.DataFrame(rows)
        metrics_df = vm.metrics.copy() if vm.metrics is not None else pd.DataFrame()

        # HTML summary header
        beat_n = sum(1 for c in cards if c.get("beat_baseline"))
        summary = f"""
### {vm.ticker} <span style="color:#737373;font-weight:500">{name}</span>
**Price** {vm.last_price:,.2f} · **asof** {vm.asof} · **regime** `{vm.regime}`  
**Run** `{vm.run_id}`  
Skill: positive brier_skill = better than base rate · horizons beating base: **{beat_n}/{len(cards)}**  
📄 Report: `{report_path or vm.report_html or "—"}`
"""
        if progress is not None:
            progress(1.0, desc="Done")
        return summary, fan, met, prob_df, metrics_df, report_path or ""

    except Exception:
        err = traceback.format_exc()
        return f"❌ Error\n```\n{err[-2500:]}\n```", empty_fig, empty_fig, pd.DataFrame(), pd.DataFrame(), ""


def build_app():
    import gradio as gr

    # initial choices from empty search (recents) or popular-neutral empty
    initial = _choices_for_query("")
    if not initial:
        # one live search seed so dropdown isn't empty on first paint — still fetched, not hardcoded product list
        initial = _choices_for_query("AAPL")[:5]

    with gr.Blocks(title="Prism") as demo:
        gr.Markdown(
            """
# Prism
**Honest equity probabilities · fan-chart cones · not a trading bot**

Search a company or symbol → pick from suggestions → Run. Charts use live ViewModel data (not text dumps).
            """
        )
        with gr.Row():
            with gr.Column(scale=3):
                search = gr.Textbox(
                    label="Search company or symbol",
                    placeholder="e.g. BBCA, bank central asia, AAPL, telkom…",
                    lines=1,
                )
                ticker = gr.Dropdown(
                    label="Pick ticker (typeahead results)",
                    choices=initial,
                    value=initial[0] if initial else None,
                    allow_custom_value=True,
                    filterable=True,
                )
            with gr.Column(scale=2):
                status = gr.Markdown("Search, then select a match. Indexes auto-fill.")
        with gr.Row():
            domestic = gr.Textbox(value="^JKSE", label="Domestic index")
            us_index = gr.Textbox(value="^GSPC", label="US index")
            macro = gr.Textbox(value="^VIX", label="Macro")
        with gr.Row():
            horizons = gr.CheckboxGroup(
                choices=["5", "21", "63", "126", "252"],
                value=["5", "21", "252"],
                label="Horizons (days)",
            )
            use_lab = gr.Checkbox(value=True, label="Full lab (GARCH / regime / tournament)")
        run_btn = gr.Button("Run analysis", variant="primary")

        summary = gr.Markdown()
        with gr.Row():
            fan = gr.Plot(label="Prediction cone")
        with gr.Row():
            met = gr.Plot(label="Brier vs baselines")
        with gr.Row():
            prob_df = gr.Dataframe(label="P(up) by horizon")
            metrics_df = gr.Dataframe(label="Walk-forward metrics")
        report = gr.Textbox(label="HTML report path", interactive=False)

        search.change(on_search_change, inputs=[search], outputs=[ticker])
        ticker.change(on_ticker_select, inputs=[ticker], outputs=[domestic, us_index, macro, status])
        run_btn.click(
            run_analysis,
            inputs=[ticker, domestic, us_index, macro, horizons, use_lab],
            outputs=[summary, fan, met, prob_df, metrics_df, report],
        )

    return demo


def launch_gui(
    share: bool = False,
    server_name: str = "127.0.0.1",
    server_port: int | None = None,
    inline: bool = True,
) -> Any:
    """
    Launch Prism UI.

    In Colab: launch_gui() opens an inline/proxied app (not a public site by default).
    """
    import os

    demo = build_app()
    # Colab niceties
    in_colab = False
    try:
        import google.colab  # noqa: F401

        in_colab = True
    except Exception:
        pass

    import gradio as gr

    kwargs = {
        "share": share,
        "show_error": True,
        "theme": gr.themes.Soft(
            primary_hue="slate",
            secondary_hue="teal",
            neutral_hue="slate",
            radius_size="lg",
        ),
        "css": ".gradio-container { max-width: 1100px !important; }",
    }
    if server_port:
        kwargs["server_port"] = server_port
    if in_colab:
        kwargs.setdefault("share", False)
        return demo.launch(**kwargs)
    return demo.launch(server_name=server_name, **kwargs)


# Back-compat alias used by README / notebooks
def launch_prism(**kwargs):
    return launch_gui(**kwargs)
