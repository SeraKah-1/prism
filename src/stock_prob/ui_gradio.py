"""
Prism Gradio UI — light editorial chrome, staged motion, live ticker fetch.

Colab-first (no public site required). Charts from in-memory ViewModel.
"""
from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any

import pandas as pd

from stock_prob.design import (
    ACCENT,
    BG,
    BG_ELEV,
    BORDER,
    DOWN,
    FG,
    FG_MUTED,
    FONT,
    UP,
    UX_BUILD,
    UX_LABEL,
)
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
from stock_prob.viz import build_fan_figure, build_metrics_figure, build_prob_gauge_figure


PRISM_CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=JetBrains+Mono:wght@400;500&display=swap');

:root {{
  --prism-bg: {BG};
  --prism-elev: {BG_ELEV};
  --prism-fg: {FG};
  --prism-muted: {FG_MUTED};
  --prism-border: {BORDER};
  --prism-accent: {ACCENT};
  --prism-up: {UP};
  --prism-down: {DOWN};
  --prism-ease: cubic-bezier(0.16, 1, 0.3, 1);
}}

.gradio-container {{
  max-width: 1120px !important;
  margin: 0 auto !important;
  font-family: "DM Sans", {FONT} !important;
  background:
    radial-gradient(900px 420px at 0% 0%, #e4efe9 0%, transparent 55%),
    radial-gradient(800px 380px at 100% 0%, #f2e8dc 0%, transparent 50%),
    var(--prism-bg) !important;
  color: var(--prism-fg) !important;
  padding-bottom: 48px !important;
}}

/* hide default footer clutter */
footer {{ display: none !important; }}

/* hero */
.prism-hero {{
  padding: 8px 4px 4px;
  animation: prism-rise 700ms var(--prism-ease) both;
}}
.prism-hero h1 {{
  font-size: clamp(1.9rem, 3.2vw, 2.5rem) !important;
  letter-spacing: -0.045em !important;
  font-weight: 700 !important;
  margin: 0 0 6px !important;
  color: var(--prism-fg) !important;
}}
.prism-hero p {{
  color: var(--prism-muted) !important;
  font-size: 0.98rem !important;
  margin: 0 !important;
  max-width: 52ch;
}}
.prism-kicker {{
  font-size: 11px !important;
  letter-spacing: 0.16em !important;
  text-transform: uppercase !important;
  font-weight: 600 !important;
  color: var(--prism-accent) !important;
  margin-bottom: 8px !important;
}}

/* controls card */
.prism-panel {{
  background: color-mix(in srgb, var(--prism-elev) 94%, white) !important;
  border: 1px solid var(--prism-border) !important;
  border-radius: 18px !important;
  padding: 14px 14px 8px !important;
  box-shadow: 0 1px 0 rgba(28,25,23,0.03), 0 18px 40px -28px rgba(28,25,23,0.35) !important;
  animation: prism-rise 700ms var(--prism-ease) both;
  animation-delay: 60ms;
}}

.prism-panel label,
.prism-panel .label-wrap span {{
  font-size: 12px !important;
  color: var(--prism-muted) !important;
  letter-spacing: 0.02em !important;
}}

.prism-panel input,
.prism-panel textarea,
.prism-panel select {{
  border-radius: 12px !important;
  border-color: var(--prism-border) !important;
  background: #fffdf9 !important;
  transition: border-color 180ms var(--prism-ease), box-shadow 220ms var(--prism-ease) !important;
}}
.prism-panel input:focus,
.prism-panel textarea:focus {{
  border-color: color-mix(in srgb, var(--prism-accent) 45%, var(--prism-border)) !important;
  box-shadow: 0 0 0 4px color-mix(in srgb, var(--prism-accent) 12%, transparent) !important;
}}

button.primary,
.prism-run button {{
  border-radius: 12px !important;
  background: var(--prism-accent) !important;
  border: none !important;
  color: #f8faf9 !important;
  font-weight: 600 !important;
  letter-spacing: -0.01em !important;
  box-shadow: 0 10px 24px -12px color-mix(in srgb, var(--prism-accent) 70%, black) !important;
  transition: transform 180ms var(--prism-ease), box-shadow 220ms var(--prism-ease), filter 180ms !important;
}}
button.primary:hover,
.prism-run button:hover {{
  transform: translateY(-1px) !important;
  filter: brightness(1.05);
  box-shadow: 0 16px 28px -12px color-mix(in srgb, var(--prism-accent) 65%, black) !important;
}}

/* result surface */
.prism-results {{
  animation: prism-rise 720ms var(--prism-ease) both;
  animation-delay: 40ms;
}}
.prism-summary {{
  background: linear-gradient(125deg, #dcece7 0%, #fffcf7 48%, #f3ebe3 100%);
  border: 1px solid var(--prism-border);
  border-radius: 18px;
  padding: 16px 18px;
  margin-bottom: 12px;
  box-shadow: 0 12px 32px -28px rgba(28,25,23,0.4);
}}
.prism-summary h2 {{
  margin: 0 0 6px;
  letter-spacing: -0.035em;
  font-size: 1.45rem;
}}
.prism-summary .meta {{
  color: var(--prism-muted);
  font-size: 0.92rem;
}}
.prism-summary code {{
  font-family: "JetBrains Mono", ui-monospace, monospace;
  font-size: 11px;
  background: rgba(255,255,255,0.7);
  border: 1px solid var(--prism-border);
  padding: 2px 7px;
  border-radius: 7px;
}}
.prism-pills {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }}
.prism-pill {{
  font-size: 11px; font-weight: 600; border-radius: 999px;
  padding: 4px 10px; border: 1px solid var(--prism-border);
  background: rgba(255,255,255,0.75); color: var(--prism-muted);
  text-transform: uppercase; letter-spacing: 0.04em;
  animation: prism-pop 500ms var(--prism-ease) both;
}}
.prism-pill.up {{ color: var(--prism-up); background: #e6f5ec; border-color: #b7e0c7; }}
.prism-pill.down {{ color: var(--prism-down); background: #fdecea; border-color: #f2c1bc; }}
.prism-pill.ok {{ color: var(--prism-accent); background: #e7f1ef; }}

.plot-wrap, .gradio-plotly {{
  border-radius: 16px !important;
  overflow: hidden;
  border: 1px solid var(--prism-border);
  background: var(--prism-elev) !important;
  box-shadow: 0 10px 30px -26px rgba(28,25,23,0.45);
  animation: prism-rise 780ms var(--prism-ease) both;
}}

@keyframes prism-rise {{
  from {{ opacity: 0; transform: translateY(16px); }}
  to {{ opacity: 1; transform: translateY(0); }}
}}
@keyframes prism-pop {{
  from {{ opacity: 0; transform: scale(0.92); }}
  to {{ opacity: 1; transform: scale(1); }}
}}
"""


def _choices_for_query(q: str) -> list[str]:
    labels = search_labels(q or "", max_results=15)
    return labels if labels else ([] if not (q or "").strip() else [q.strip()])


def on_search_change(query: str):
    import gradio as gr

    ch = _choices_for_query(query)
    val = ch[0] if ch else (query or None)
    return gr.Dropdown(choices=ch or [], value=val)


def on_ticker_select(label: str):
    sym = parse_symbol_from_label(label or "")
    if not sym:
        return "^JKSE", "^GSPC", "^VIX", "_Type a company or symbol, then pick a live match._"
    ctx = default_context_for_symbol(sym)
    hits = search_tickers(sym, max_results=3)
    name = hits[0].name if hits else sym
    return (
        ctx["domestic_index"],
        ctx["us_index"],
        ctx["macro"],
        f"Selected **{sym}** — {name}. Context indexes auto-filled (you can still edit).",
    )


def _empty_fig(title: str = ""):
    import plotly.graph_objects as go
    from stock_prob.design import plotly_layout_base

    fig = go.Figure()
    fig.update_layout(**plotly_layout_base(360, title or "Waiting for a run"))
    fig.add_annotation(
        text="Run analysis to materialize the cone",
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(color=FG_MUTED, size=13),
    )
    return fig


def _summary_html(vm, name: str, report_path: str, cards: list[dict]) -> str:
    pills = []
    for i, c in enumerate(cards):
        cls = "up" if c["direction"] == "up" else "down"
        beat = c.get("beat_baseline")
        extra = " · beats base" if beat is True else (" · below base" if beat is False else "")
        delay = i * 60
        pills.append(
            f'<span class="prism-pill {cls}" style="animation-delay:{delay}ms">'
            f'{c["horizon"]}d  {c["prob_up"]*100:.1f}%{extra}</span>'
        )
    beat_n = sum(1 for c in cards if c.get("beat_baseline") is True)
    return f"""
<div class="prism-summary">
  <div class="prism-kicker" style="margin-bottom:6px">Live desk · {vm.regime}</div>
  <h2>{vm.ticker} <span style="color:{FG_MUTED};font-weight:500;font-size:0.7em">{name}</span></h2>
  <div class="meta">
    last <b>{vm.last_price:,.2f}</b> · asof <b>{vm.asof}</b> ·
    horizons beating base rate <b>{beat_n}/{len(cards)}</b>
  </div>
  <div class="prism-pills">{''.join(pills)}</div>
  <div class="meta" style="margin-top:12px">
    run <code>{vm.run_id}</code><br/>
    report <code>{report_path or "—"}</code>
  </div>
</div>
"""


def run_analysis(
    ticker_label: str,
    domestic: str,
    us_index: str,
    macro: str,
    horizons: list[str] | list[int],
    use_lab: bool,
    progress=None,
):
    empty = _empty_fig()
    try:
        if progress is not None:
            progress(0.06, desc="Resolving ticker via live fetch…")
        resolved = resolve_ticker(ticker_label or "")
        if not resolved.get("ok"):
            msg = (
                f'<div class="prism-summary"><h2>Ticker not resolved</h2>'
                f'<div class="meta">{resolved.get("error") or ticker_label}</div></div>'
            )
            return msg, empty, empty, empty, pd.DataFrame(), pd.DataFrame(), ""

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
            progress(0.22, desc=f"Computing Prism desk for {symbol}…")

        result = run_once(
            symbol,
            domestic_index=domestic or "^GSPC",
            us_index=us_index or "^GSPC",
            macro=macro or "^VIX",
            horizons=hz,
            use_lab=bool(use_lab),
            mc_paths=900,
        )

        if progress is not None:
            progress(0.72, desc="Composing charts & report…")

        vm = build_viewmodel(result, name=name)
        report_path = ""
        if vm.art_dir:
            report_path = str(Path(vm.art_dir) / "prism_report.html")
            write_prism_report(vm, report_path)
            vm.report_html = report_path

        h = vm.primary_horizon()
        cone = vm.cones.get(h)
        fan = empty
        if vm.history is not None and len(vm.history) and cone is not None:
            fan = build_fan_figure(
                vm.history,
                cone,
                title=f"{vm.ticker} · {h}d probability cone · {vm.regime}",
            )

        gauge = build_prob_gauge_figure(vm.probs)
        met = empty
        if vm.metrics is not None and len(vm.metrics):
            met = build_metrics_figure(vm.metrics, title="Brier vs baselines (lower is better)")

        cards = vm.summary_cards()
        summary = _summary_html(vm, name, report_path, cards)

        prob_df = pd.DataFrame(
            [
                {
                    "horizon_d": c["horizon"],
                    "P_up_%": round(c["prob_up"] * 100, 2),
                    "direction": c["direction"],
                    "beat_baseline": c["beat_baseline"],
                    "brier_skill": None
                    if c["brier_skill"] is None
                    else round(float(c["brier_skill"]), 4),
                }
                for c in cards
            ]
        )
        metrics_df = vm.metrics.copy() if vm.metrics is not None else pd.DataFrame()

        if progress is not None:
            progress(1.0, desc="Ready")
        return summary, fan, gauge, met, prob_df, metrics_df, report_path or ""

    except Exception:
        err = traceback.format_exc()
        msg = f'<div class="prism-summary"><h2>Something broke</h2><pre style="white-space:pre-wrap;font-size:11px">{err[-2200:]}</pre></div>'
        return msg, empty, empty, empty, pd.DataFrame(), pd.DataFrame(), ""


def build_app():
    import gradio as gr

    initial = _choices_for_query("")
    if not initial:
        initial = _choices_for_query("equity")[:8] or _choices_for_query("A")[:8]

    with gr.Blocks(title="Prism") as demo:
        gr.HTML(
            f"""
            <div class="prism-hero">
              <div class="prism-kicker">Prism · {UX_LABEL}</div>
              <h1>Probability desk</h1>
              <p>Search any name or symbol · live fetch · fan-chart cones · baseline honesty.
              Light warm paper UI — if you still see raw text fields only, you loaded an old copy.</p>
              <p style="margin-top:8px;font-size:11px;color:#78716c;font-family:ui-monospace,monospace">build <b>{UX_BUILD}</b></p>
            </div>
            """
        )

        with gr.Group(elem_classes=["prism-panel"]):
            with gr.Row():
                search = gr.Textbox(
                    label="Search",
                    placeholder="BBCA · bank central asia · AAPL · telkom · ^GSPC",
                    lines=1,
                    scale=3,
                )
                ticker = gr.Dropdown(
                    label="Match (typeahead from live search)",
                    choices=initial,
                    value=initial[0] if initial else None,
                    allow_custom_value=True,
                    filterable=True,
                    scale=4,
                )
            status = gr.Markdown("Start typing — results stream from market search, not a hardcoded list.")
            with gr.Row():
                domestic = gr.Textbox(value="^JKSE", label="Domestic index")
                us_index = gr.Textbox(value="^GSPC", label="US index")
                macro = gr.Textbox(value="^VIX", label="Macro")
            with gr.Row():
                horizons = gr.CheckboxGroup(
                    choices=["5", "21", "63", "126", "252"],
                    value=["5", "21", "252"],
                    label="Horizons",
                )
                use_lab = gr.Checkbox(value=True, label="Full lab (GARCH / regime / tournament)")
            with gr.Row(elem_classes=["prism-run"]):
                run_btn = gr.Button("Run analysis", variant="primary", scale=2)
                gr.Markdown(
                    "<span style='color:#78716c;font-size:12px'>First run may take ~20–40s · later runs use cache</span>",
                    scale=3,
                )

        with gr.Column(elem_classes=["prism-results"]):
            summary = gr.HTML()
            fan = gr.Plot(label="Prediction cone")
            with gr.Row():
                gauge = gr.Plot(label="P(up) bars")
                met = gr.Plot(label="Skill vs baselines")
            with gr.Row():
                prob_df = gr.Dataframe(label="Probabilities", wrap=True)
                metrics_df = gr.Dataframe(label="Walk-forward metrics", wrap=True)
            report = gr.Textbox(label="Open this HTML report for full motion layout", interactive=False)

        search.change(on_search_change, inputs=[search], outputs=[ticker])
        ticker.change(on_ticker_select, inputs=[ticker], outputs=[domestic, us_index, macro, status])
        run_btn.click(
            run_analysis,
            inputs=[ticker, domestic, us_index, macro, horizons, use_lab],
            outputs=[summary, fan, gauge, met, prob_df, metrics_df, report],
        )

        gr.Markdown(
            """
<div style="color:#78716c;font-size:12px;margin-top:8px;line-height:1.5">
Taste notes: warm paper surface · deep ink accent · staged rise motion · charts before tables.<br/>
Research software — not financial advice.
</div>
            """
        )
    return demo


def launch_gui(
    share: bool = False,
    server_name: str = "127.0.0.1",
    server_port: int | None = None,
    inline: bool = True,
) -> Any:
    import gradio as gr

    demo = build_app()
    in_colab = False
    try:
        import google.colab  # noqa: F401

        in_colab = True
    except Exception:
        pass

    kwargs: dict[str, Any] = {
        "share": share,
        "show_error": True,
        "css": PRISM_CSS,
        "theme": gr.themes.Soft(
            primary_hue="stone",
            secondary_hue="teal",
            neutral_hue="stone",
            radius_size="lg",
            font=gr.themes.GoogleFont("DM Sans"),
        ),
    }
    if server_port:
        kwargs["server_port"] = server_port
    if in_colab:
        kwargs.setdefault("share", False)
        return demo.launch(**kwargs)
    return demo.launch(server_name=server_name, **kwargs)


def launch_prism(**kwargs):
    return launch_gui(**kwargs)
