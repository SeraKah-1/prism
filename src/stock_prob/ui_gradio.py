"""
Prism Gradio UI — plain English, matplotlib SVG charts (no Plotly/JS).
"""
from __future__ import annotations

import shutil
import tempfile
import traceback
import uuid
from pathlib import Path
from typing import Any

import pandas as pd

from stock_prob.design import UX_BUILD
from stock_prob.report_html import write_prism_report
from stock_prob.tickers import (
    parse_symbol_from_label,
    resolve_ticker,
    search_labels,
    search_tickers,
)
from stock_prob.ui_colab import run_once
from stock_prob.viewmodel import build_viewmodel
from stock_prob.viz import (
    build_fan_figure,
    build_metrics_figure,
    build_prob_gauge_figure,
    fig_to_html,
)

INDEX_PRESETS = {
    "Indonesia · JCI (Jakarta)": {
        "domestic_index": "^JKSE",
        "us_index": "^GSPC",
        "macro": "^VIX",
        "hint": "IDX stocks. Context: JCI + S&P 500 + VIX.",
    },
    "United States · S&P 500 + VIX": {
        "domestic_index": "^GSPC",
        "us_index": "^GSPC",
        "macro": "^VIX",
        "hint": "US stocks. Context: S&P 500 + VIX.",
    },
    "Global · S&P 500 + USD (DXY)": {
        "domestic_index": "^GSPC",
        "us_index": "^GSPC",
        "macro": "DX-Y.NYB",
        "hint": "S&P 500 + US dollar strength (often useful for emerging markets).",
    },
}

HORIZON_CHOICES = [
    ("5 days (≈ 1 week)", 5),
    ("21 days (≈ 1 month)", 21),
    ("63 days (≈ 1 quarter)", 63),
    ("126 days (≈ 6 months)", 126),
    ("252 days (≈ 1 year)", 252),
]

GRADIO_EXPORT_DIR = Path(tempfile.gettempdir()) / "prism_gradio_exports"


def _gradio_safe_file(src: Path) -> str:
    GRADIO_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    dst = GRADIO_EXPORT_DIR / f"{src.stem}_{uuid.uuid4().hex[:8]}{src.suffix}"
    shutil.copy2(src, dst)
    return str(dst)


PRISM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,400;9..40,500;9..40,600;9..40,700&display=swap');

:root {
  --bg: #f6f3ee;
  --card: #fffcf7;
  --fg: #1c1917;
  --muted: #78716c;
  --border: #e7e0d5;
  --accent: #0f3d3e;
  --radius: 16px;
  --gap: 16px;
  --ease: cubic-bezier(0.16, 1, 0.3, 1);
}

.gradio-container {
  max-width: 960px !important;
  margin: 0 auto !important;
  font-family: "DM Sans", system-ui, sans-serif !important;
  background:
    radial-gradient(900px 380px at 0% 0%, #e4efe9 0%, transparent 55%),
    radial-gradient(720px 340px at 100% 0%, #f3e9df 0%, transparent 50%),
    var(--bg) !important;
  color: var(--fg) !important;
  padding: 12px 12px 48px !important;
}
footer { display: none !important; }

.hero { padding: 8px 4px 12px; animation: rise 0.55s var(--ease) both; }
.hero .kicker {
  font-size: 11px; letter-spacing: 0.14em; text-transform: uppercase;
  font-weight: 600; color: var(--accent); margin: 0 0 6px;
}
.hero h1 {
  margin: 0 0 8px !important; font-size: clamp(1.7rem, 3vw, 2.2rem) !important;
  letter-spacing: -0.04em !important; font-weight: 700 !important; color: var(--fg) !important;
}
.hero p { margin: 0 !important; color: var(--muted) !important; line-height: 1.5; max-width: 56ch; }

.card {
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  padding: 16px !important;
  margin: 0 0 var(--gap) !important;
  box-shadow: 0 1px 0 rgba(28,25,23,0.03), 0 14px 36px -28px rgba(28,25,23,0.35) !important;
  animation: rise 0.55s var(--ease) both;
}

.section-title {
  margin: 0 0 4px !important;
  font-size: 0.95rem !important;
  font-weight: 650 !important;
  letter-spacing: -0.02em !important;
  color: var(--fg) !important;
}
.hint {
  margin: 0 0 12px !important;
  font-size: 12.5px !important;
  color: var(--muted) !important;
  line-height: 1.45 !important;
}

.frame {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  overflow: hidden;
  margin: 0 0 var(--gap);
  box-shadow: 0 12px 32px -28px rgba(28,25,23,0.4);
  animation: rise 0.6s var(--ease) both;
}
.frame-head {
  padding: 12px 16px 8px;
  border-bottom: 1px solid var(--border);
  background: color-mix(in srgb, var(--card) 90%, #efeae2);
}
.frame-head h3 {
  margin: 0 !important; font-size: 0.92rem !important;
  letter-spacing: -0.02em !important; font-weight: 650 !important;
}
.frame-head p {
  margin: 4px 0 0 !important; font-size: 12px !important; color: var(--muted) !important;
}
.frame-body {
  padding: 10px 12px 14px;
  min-height: 80px;
  overflow: auto;
  max-width: 100%;
}
.frame-body .chart-svg svg { max-width: 100%; height: auto; display: block; }

.summary {
  background: linear-gradient(125deg, #dcece7 0%, #fffcf7 52%, #f3ebe3 100%);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 16px 18px;
  margin-bottom: var(--gap);
  animation: rise 0.5s var(--ease) both;
}
.summary h2 {
  margin: 0 0 6px !important; font-size: 1.35rem !important;
  letter-spacing: -0.03em !important;
}
.summary .lead { margin: 8px 0 0; color: #44403c; line-height: 1.55; font-size: 0.97rem; }
.pills { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
.pill {
  font-size: 12px; font-weight: 600; border-radius: 999px;
  padding: 5px 11px; border: 1px solid var(--border); background: #fff; color: #57534e;
}
.pill.up { background: #e6f5ec; color: #1b7a4e; border-color: #b7e0c7; }
.pill.down { background: #fdecea; color: #b42318; border-color: #f2c1bc; }
.pill.ok { background: #e7f1ef; color: #0f3d3e; }

button.primary {
  border-radius: 12px !important;
  background: var(--accent) !important;
  font-weight: 600 !important;
  min-height: 44px !important;
}

@keyframes rise {
  from { opacity: 0; transform: translateY(12px); }
  to { opacity: 1; transform: translateY(0); }
}

.progress-panel {
  background: #fffcf7;
  border: 1px solid #e7e0d5;
  border-radius: 16px;
  padding: 18px 18px 16px;
  margin: 0 0 16px;
  box-shadow: 0 12px 32px -28px rgba(28,25,23,0.4);
  animation: rise 0.35s var(--ease) both;
}
.progress-panel .title {
  margin: 0 0 6px;
  font-size: 1.05rem;
  font-weight: 700;
  letter-spacing: -0.02em;
  color: #1c1917;
}
.progress-panel .detail {
  margin: 0 0 12px;
  font-size: 13px;
  color: #78716c;
  line-height: 1.45;
}
.progress-track {
  height: 10px;
  border-radius: 999px;
  background: #efeae2;
  overflow: hidden;
  border: 1px solid #e7e0d5;
}
.progress-fill {
  height: 100%;
  border-radius: 999px;
  background: linear-gradient(90deg, #0f3d3e, #2d6a4f);
  width: 0%;
  transition: width 0.35s var(--ease);
}
.progress-fill.indeterminate {
  width: 40% !important;
  animation: slide 1.1s ease-in-out infinite;
}
@keyframes slide {
  0% { transform: translateX(-120%); }
  100% { transform: translateX(320%); }
}
.progress-meta {
  display: flex;
  justify-content: space-between;
  margin-top: 8px;
  font-size: 12px;
  color: #78716c;
  font-variant-numeric: tabular-nums;
}
.progress-steps {
  margin: 12px 0 0;
  padding: 0;
  list-style: none;
}
.progress-steps li {
  font-size: 12.5px;
  color: #a8a29e;
  padding: 4px 0 4px 18px;
  position: relative;
}
.progress-steps li::before {
  content: "";
  width: 8px; height: 8px; border-radius: 99px;
  background: #d6d3d1;
  position: absolute; left: 0; top: 9px;
}
.progress-steps li.done { color: #1b7a4e; }
.progress-steps li.done::before { background: #1b7a4e; }
.progress-steps li.active { color: #0f3d3e; font-weight: 600; }
.progress-steps li.active::before {
  background: #0f3d3e;
  box-shadow: 0 0 0 4px rgba(15,61,62,0.15);
}
.status-bar {
  border: 1px solid #e7e0d5;
  background: #fffcf7;
  border-radius: 12px;
  padding: 10px 14px;
  font-size: 13px;
  color: #44403c;
  margin: 8px 0 0;
}
.status-bar.running { border-color: #a7c4be; background: #eef6f3; color: #0f3d3e; }
.status-bar.ok { border-color: #b7e0c7; background: #e6f5ec; color: #1b7a4e; }
.status-bar.err { border-color: #f2c1bc; background: #fdecea; color: #b42318; }
"""


def _frame(title: str, hint: str, body_html: str, delay_ms: int = 0) -> str:
    return f"""
<section class="frame" style="animation-delay:{delay_ms}ms">
  <div class="frame-head">
    <h3>{title}</h3>
    <p>{hint}</p>
  </div>
  <div class="frame-body">{body_html}</div>
</section>
"""


def _choices(q: str) -> list[str]:
    labels = search_labels(q or "", max_results=12)
    if labels:
        return labels
    q = (q or "").strip()
    return [q] if q else []


def on_type_search(query: str):
    import gradio as gr

    ch = _choices(query)
    val = ch[0] if ch else (query or None)
    return gr.Dropdown(choices=ch, value=val)


def on_pick_ticker(label: str, market_preset: str):
    sym = parse_symbol_from_label(label or "")
    if not sym:
        return market_preset, '<p class="hint" style="margin:0">Type a company name or ticker, then pick a match.</p>'
    preset = (
        "Indonesia · JCI (Jakarta)"
        if sym.upper().endswith(".JK")
        else "United States · S&P 500 + VIX"
    )
    hits = search_tickers(sym, max_results=2)
    name = hits[0].name if hits else sym
    msg = (
        f'<p class="hint" style="margin:0">Selected <b>{sym}</b> — {name}. '
        f"Market context set automatically (you can change it).</p>"
    )
    return preset, msg


def on_market_change(preset: str):
    info = INDEX_PRESETS.get(preset) or next(iter(INDEX_PRESETS.values()))
    return f'<p class="hint" style="margin:0">{info["hint"]}</p>'


def _summary_html(vm, name: str) -> str:
    cards = vm.summary_cards()
    if not cards:
        lead = "No probabilities for the selected time horizons."
    else:
        c0 = cards[0]
        direction = "up" if c0["direction"] == "up" else "down"
        conf = "leaning" if c0["conviction"] >= 0.25 else "near 50%"
        beat = sum(1 for c in cards if c.get("beat_baseline") is True)
        lead = (
            f"For <b>{vm.ticker}</b>, estimated chance price goes "
            f"<b>{direction}</b> in ~<b>{c0['horizon']} days</b>: "
            f"<b>{c0['prob_up']*100:.0f}%</b> ({conf}). "
            f"Model beats base rate on <b>{beat} of {len(cards)}</b> time horizons."
        )
    pills = [f'<span class="pill ok">regime: {vm.regime}</span>']
    for c in cards:
        cls = "up" if c["direction"] == "up" else "down"
        beat = c.get("beat_baseline")
        if beat is True:
            tag = " · beats baseline"
        elif beat is False:
            tag = " · below baseline"
        else:
            tag = ""
        pills.append(
            f'<span class="pill {cls}">{c["horizon"]}d · '
            f'P(up) {c["prob_up"]*100:.1f}%{tag}</span>'
        )
    price = f"{vm.last_price:,.2f}" if vm.last_price == vm.last_price else "—"
    return f"""
<div class="summary">
  <div class="kicker" style="font-size:11px;letter-spacing:.14em;text-transform:uppercase;font-weight:600;color:#0f3d3e">Result</div>
  <h2>{vm.ticker} <span style="color:#78716c;font-weight:500;font-size:.72em">{name if name != vm.ticker else ""}</span></h2>
  <div class="hint" style="margin:0">Last price <b>{price}</b> · as of <b>{vm.asof or "—"}</b></div>
  <p class="lead">{lead}</p>
  <div class="pills">{''.join(pills)}</div>
</div>
"""


def _loading_panel(pct: int, step: int, total: int, title: str, detail: str, steps: list[str] | None = None) -> str:
    pct = max(0, min(100, int(pct)))
    steps = steps or [
        "Validate ticker",
        "Fetch prices",
        "Fit model & walk-forward",
        "Build charts",
        "Write report",
    ]
    items = []
    for i, label in enumerate(steps, start=1):
        cls = "done" if i < step else ("active" if i == step else "")
        items.append(f'<li class="{cls}">{i}. {label}</li>')
    return f"""
<div class="progress-panel">
  <p class="title">{title}</p>
  <p class="detail">{detail}</p>
  <div class="progress-track"><div class="progress-fill" style="width:{pct}%"></div></div>
  <div class="progress-meta"><span>Step {step}/{total}</span><span>{pct}%</span></div>
  <ul class="progress-steps">{''.join(items)}</ul>
</div>
"""


def _status(msg: str, kind: str = "running") -> str:
    return f'<div class="status-bar {kind}">{msg}</div>'


def _placeholder_frame(title: str, msg: str) -> str:
    body = (
        f'<div style="padding:28px 12px;text-align:center;color:#78716c;font-size:13px">'
        f'<div class="progress-track" style="max-width:220px;margin:0 auto 10px">'
        f'<div class="progress-fill indeterminate"></div></div>{msg}</div>'
    )
    return _frame(title, "Working…", body)


def _summary_html_error(title: str, body: str) -> str:
    return f'<div class="summary"><h2>{title}</h2><p class="lead">{body}</p></div>'


def _is_advanced(mode: str) -> bool:
    s = str(mode or "").lower()
    return s.startswith("advanced") or s.startswith("lanjutan")


def run_analysis(ticker_label, market_preset, horizon_labels, mode, progress=None):
    """Generator: yield progress so Colab never looks blank mid-run."""
    try:
        import gradio as gr

        if progress is None:

            class _P:
                def __call__(self, *a, **k):
                    return None

            progress = _P()
    except Exception:
        pass

    idle = _frame("Waiting", "Results show up here.", "<p class='hint'>No data yet.</p>")
    empty_table = pd.DataFrame()

    def pack(summary, fan, gauge, met, table, report, status_html):
        return summary, fan, gauge, met, table, report, status_html

    try:
        yield pack(
            _loading_panel(4, 1, 5, "Starting…", "Setting up the pipeline."),
            _placeholder_frame("Price range", "Waiting for prices…"),
            _placeholder_frame("P(up)", "Waiting for model…"),
            _placeholder_frame("Model vs baselines", "Waiting for evaluation…"),
            empty_table,
            None,
            _status("0–5% · Starting…", "running"),
        )
        if progress is not None:
            progress(0.05, desc="Starting…")

        yield pack(
            _loading_panel(12, 1, 5, "Validating ticker…", f"Looking up: <b>{ticker_label or '—'}</b>"),
            _placeholder_frame("Price range", "Validating…"),
            _placeholder_frame("P(up)", "Validating…"),
            _placeholder_frame("Model vs baselines", "Validating…"),
            empty_table,
            None,
            _status("12% · Validating ticker…", "running"),
        )
        if progress is not None:
            progress(0.12, desc="Validating…")

        resolved = resolve_ticker(ticker_label or "")
        if not resolved.get("ok"):
            err = _summary_html_error(
                "Ticker not found",
                "Try a company name (e.g. Apple) or a ticker (AAPL, BBCA.JK).",
            )
            yield pack(err, idle, idle, idle, empty_table, None, _status("Failed: ticker not found.", "err"))
            return

        symbol = resolved["symbol"]
        name = resolved.get("name") or symbol
        preset = INDEX_PRESETS.get(market_preset) or next(iter(INDEX_PRESETS.values()))
        domestic, us_index, macro = preset["domestic_index"], preset["us_index"], preset["macro"]
        if symbol.upper().endswith(".JK") and domestic == "^GSPC":
            domestic = "^JKSE"

        label_to_h = {lab: h for lab, h in HORIZON_CHOICES}
        hz: list[int] = []
        for lab in horizon_labels or []:
            if lab in label_to_h:
                hz.append(label_to_h[lab])
        if not hz:
            hz = [5, 21, 252]

        use_lab = _is_advanced(mode)
        mode_txt = "Advanced" if use_lab else "Standard"

        yield pack(
            _loading_panel(
                35,
                2,
                5,
                f"Fetching & modeling · {symbol}",
                f"{name} · {mode_txt} · horizons {hz}. "
                f"Standard is faster. Advanced adds more stats (slower). GPU does not help this stack.",
            ),
            _placeholder_frame("Price range", f"Loading prices for {symbol}…"),
            _placeholder_frame("P(up)", "Building features…"),
            _placeholder_frame("Model vs baselines", "Walk-forward pending…"),
            empty_table,
            None,
            _status(f"35% · Computing · {symbol}…", "running"),
        )
        if progress is not None:
            progress(0.35, desc=f"Computing {symbol}…")

        result = run_once(
            symbol,
            domestic_index=domestic,
            us_index=us_index,
            macro=macro,
            horizons=hz,
            use_lab=use_lab,
            mc_paths=500 if not use_lab else 900,
            speed="full" if use_lab else "fast",
        )
        result = dict(result)
        result["horizons"] = list(hz)

        yield pack(
            _loading_panel(78, 4, 5, "Building charts…", f"Model done for {symbol}. Rendering SVG charts."),
            _placeholder_frame("Price range", "Drawing fan chart…"),
            _placeholder_frame("P(up)", "Drawing bars…"),
            _placeholder_frame("Model vs baselines", "Drawing Brier bars…"),
            empty_table,
            None,
            _status("78% · Rendering charts…", "running"),
        )
        if progress is not None:
            progress(0.78, desc="Rendering…")

        vm = build_viewmodel(result, name=name)

        if not vm.has_probs():
            try:
                from stock_prob.ingest import fetch_universe, align_close_panel
                from stock_prob.features import build_feature_frame
                from stock_prob.forecast import compute_live_forecast, history_frame

                ctx_syms = [symbol, domestic, us_index, macro]
                frames = fetch_universe([s for s in ctx_syms if s], period="5y", use_cache=True)
                panel = align_close_panel(frames)
                if symbol in panel.columns:
                    eq = panel[symbol].dropna()
                    feats = build_feature_frame(
                        eq,
                        domestic_close=panel[domestic] if domestic in panel.columns else None,
                        us_close=panel[us_index] if us_index in panel.columns else None,
                        macro_close=panel[macro] if macro in panel.columns else None,
                    )
                    live = compute_live_forecast(feats, eq, hz, mc_paths=500)
                    result["live_probs"] = live["live_probs"]
                    result["live_cones"] = live["live_cones"]
                    result["history"] = history_frame(eq, 400)
                    result["last_price"] = live["last_price"]
                    result["last_date"] = live["last_date"]
                    result["forecast_errors"] = live.get("errors") or {}
                    vm = build_viewmodel(result, name=name)
            except Exception as e:
                vm.forecast_errors["_recovery"] = str(e)[:200]

        h = vm.primary_horizon()
        cone = vm.cones.get(h)
        if cone is None:
            cone = vm.cones.get(str(h))
        fan_fig = None
        if vm.history is not None and len(vm.history) > 5 and cone is not None:
            try:
                fan_fig = build_fan_figure(
                    vm.history,
                    cone,
                    title=f"Price range · {vm.ticker} · {h}d ahead",
                )
            except Exception as e:
                vm.forecast_errors["fan"] = str(e)[:160]
        gauge_fig = None
        if vm.has_probs():
            try:
                gauge_fig = build_prob_gauge_figure(vm.probs, title="P(up) by horizon")
            except Exception as e:
                vm.forecast_errors["gauge"] = str(e)[:160]
        met_fig = None
        if vm.metrics is not None and len(vm.metrics):
            try:
                met_fig = build_metrics_figure(
                    vm.metrics,
                    title="Brier vs simple baselines (lower is better)",
                )
            except Exception as e:
                vm.forecast_errors["metrics_fig"] = str(e)[:160]

        if vm.art_dir:
            raw_report = Path(vm.art_dir) / "prism_report.html"
        else:
            GRADIO_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
            raw_report = GRADIO_EXPORT_DIR / f"prism_{uuid.uuid4().hex[:8]}.html"
        write_prism_report(vm, raw_report)
        safe_report = _gradio_safe_file(raw_report)

        fan_block = _frame(
            "Price range (fan chart)",
            "Dashed line = median. Shaded bands = likely range. Wider = more uncertainty.",
            fig_to_html(fan_fig, height=480)
            if fan_fig is not None
            else "<p class='hint'>Could not build price range (not enough history).</p>",
            40,
        )
        gauge_block = _frame(
            "P(up) by horizon",
            "50% is a coin flip. Distance from 50% is conviction, not a guarantee.",
            fig_to_html(gauge_fig, height=280)
            if gauge_fig is not None
            else "<p class='hint'>No probabilities for these horizons.</p>",
            80,
        )
        met_block = _frame(
            "Model vs baselines",
            "Base rate = historical up frequency. If the model loses, treat the signal as weak.",
            fig_to_html(met_fig, height=320)
            if met_fig is not None
            else "<p class='hint'>Walk-forward metrics not available.</p>",
            120,
        )

        cards = vm.summary_cards()
        table = pd.DataFrame(
            [
                {
                    "Horizon": f"{c['horizon']} days",
                    "P(up) %": round(c["prob_up"] * 100, 1),
                    "Direction": "Up" if c["direction"] == "up" else "Down",
                    "Beats baseline?": (
                        "Yes" if c["beat_baseline"] is True else ("No" if c["beat_baseline"] is False else "—")
                    ),
                }
                for c in cards
            ]
        )

        if not vm.has_probs() and not vm.has_charts():
            err_detail = "; ".join(f"{k}: {v}" for k, v in list(vm.forecast_errors.items())[:6])
            summary = _summary_html_error(
                "Incomplete result",
                "Pipeline finished but produced no probabilities or chart. "
                f"Detail: {err_detail or 'unknown'}. Try Standard mode, shorter horizons, or another ticker.",
            )
            yield pack(
                summary,
                fan_block,
                gauge_block,
                met_block,
                table,
                safe_report,
                _status("Done with errors — no numbers/charts.", "err"),
            )
            return

        if not vm.has_probs():
            summary = _summary_html(vm, name)
            summary = summary.replace("Result", "Partial result (chart only)")
            ferr = ""
            if vm.forecast_errors:
                ferr = "<p class='hint'>Notes: " + "; ".join(
                    f"{k}={v}" for k, v in list(vm.forecast_errors.items())[:4]
                ) + "</p>"
            summary = summary + ferr + (
                "<p class='lead'>Probabilities failed; price range still shown from historical volatility.</p>"
            )
            yield pack(
                summary,
                fan_block,
                gauge_block,
                met_block,
                table,
                safe_report,
                _status("Partial — chart ok, probabilities empty.", "err"),
            )
            return

        if progress is not None:
            progress(1.0, desc="Done")

        summary = _summary_html(vm, name)
        if str(vm.regime) in ("n/a", "na", ""):
            summary += (
                "<p class='hint' style='margin-top:8px'>Note: Standard mode skips market regime "
                "(shows n/a). Use Advanced for the regime label.</p>"
            )

        yield pack(
            summary,
            fan_block,
            gauge_block,
            met_block,
            table,
            safe_report,
            _status("100% · Done. Scroll for charts; download the HTML report if needed.", "ok"),
        )
    except Exception:
        err = traceback.format_exc()
        msg = (
            f'<div class="summary"><h2>Run failed</h2>'
            f'<pre style="white-space:pre-wrap;font-size:11px;color:#78716c">{err[-1800:]}</pre></div>'
        )
        idle = _frame("Error", "This panel is empty because the run failed.", "<p class='hint'>See the message above.</p>")
        yield pack(
            msg,
            idle,
            idle,
            idle,
            pd.DataFrame(),
            None,
            _status("Error — details in the result panel.", "err"),
        )


def build_app():
    import gradio as gr

    seed = _choices("bank") or _choices("stock") or []

    with gr.Blocks(title="Prism") as demo:
        gr.HTML(
            """
            <div class="hero">
              <div class="kicker">Prism</div>
              <h1>Stock direction probabilities</h1>
              <p>Search a company, pick time horizons, see price range bands and P(up) —
              checked against simple baselines. Research tool, not a forecast guarantee.</p>
            </div>
            """
        )

        with gr.Group(elem_classes=["card"]):
            gr.HTML(
                """
                <div class="section-title">1 · Pick a stock</div>
                <p class="hint">Type a name or ticker (e.g. Apple, AAPL, BBCA). Choose a match from the list.</p>
                """
            )
            search = gr.Textbox(
                label="Search",
                placeholder="Apple · AAPL · bank central asia · BBCA",
                lines=1,
            )
            ticker = gr.Dropdown(
                label="Matches (pick one)",
                choices=seed,
                value=seed[0] if seed else None,
                allow_custom_value=True,
                filterable=True,
            )
            status = gr.HTML('<p class="hint" style="margin:0">Start typing in Search.</p>')

            gr.HTML(
                """
                <div class="section-title" style="margin-top:14px">2 · Market context</div>
                <p class="hint">Indexes used as context for the model.</p>
                """
            )
            market = gr.Radio(
                choices=list(INDEX_PRESETS.keys()),
                value="Indonesia · JCI (Jakarta)",
                label="Market context",
            )
            market_hint = gr.HTML(
                f'<p class="hint" style="margin:0">{INDEX_PRESETS["Indonesia · JCI (Jakarta)"]["hint"]}</p>'
            )

            gr.HTML(
                """
                <div class="section-title" style="margin-top:14px">3 · Time horizons</div>
                <p class="hint">Trading days. You can select more than one.</p>
                """
            )
            horizons = gr.CheckboxGroup(
                choices=[lab for lab, _ in HORIZON_CHOICES],
                value=["5 days (≈ 1 week)", "21 days (≈ 1 month)", "252 days (≈ 1 year)"],
                label="Time horizons",
            )

            gr.HTML(
                """
                <div class="section-title" style="margin-top:14px">4 · Mode</div>
                <p class="hint">Standard is enough for price range and P(up). Advanced adds more stats (slower).</p>
                """
            )
            mode = gr.Radio(
                choices=[
                    "Standard — faster",
                    "Advanced — more detail (slower)",
                ],
                value="Standard — faster",
                label="Analysis mode",
            )

            run_btn = gr.Button("Run analysis", variant="primary")
            run_status = gr.HTML("")

        gr.HTML('<div class="section-title" style="margin:8px 4px">Results</div>')
        summary = gr.HTML()
        fan = gr.HTML()
        gauge = gr.HTML()
        met = gr.HTML()

        with gr.Group(elem_classes=["card"]):
            gr.HTML(
                """
                <div class="section-title">Summary table</div>
                <p class="hint">Same numbers as the charts, in a table.</p>
                """
            )
            table = gr.Dataframe(label="Summary", wrap=True, interactive=False)

        with gr.Group(elem_classes=["card"]):
            gr.HTML(
                """
                <div class="section-title">Full report</div>
                <p class="hint">Download the HTML file and open it in a new tab.</p>
                """
            )
            report_file = gr.File(label="Download HTML report", file_count="single")

        gr.HTML(
            f'<p class="hint" style="text-align:center;margin-top:8px">'
            f"Prism · research · not investment advice"
            f'<span style="opacity:.45"> · {UX_BUILD}</span></p>'
        )

        search.change(on_type_search, inputs=[search], outputs=[ticker])
        ticker.change(on_pick_ticker, inputs=[ticker, market], outputs=[market, status])
        market.change(on_market_change, inputs=[market], outputs=[market_hint])
        run_btn.click(
            run_analysis,
            inputs=[ticker, market, horizons, mode],
            outputs=[summary, fan, gauge, met, table, report_file, run_status],
        )

    return demo


def launch_gui(
    share: bool | None = None,
    server_name: str = "0.0.0.0",
    server_port: int | None = None,
    **kwargs: Any,
) -> Any:
    import gradio as gr

    GRADIO_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    print("[Prism] UI ·", UX_BUILD)
    print("[Prism] Charts: matplotlib SVG (no Plotly JS)")
    print("[Prism] Gradio export dir:", GRADIO_EXPORT_DIR)

    demo = build_app()
    in_colab = False
    try:
        import google.colab  # noqa: F401

        in_colab = True
    except Exception:
        pass

    if share is None:
        share = bool(in_colab)

    launch_kwargs: dict[str, Any] = {
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
        "quiet": False,
        "allowed_paths": [str(GRADIO_EXPORT_DIR), tempfile_dir()],
    }
    if server_port:
        launch_kwargs["server_port"] = server_port
    launch_kwargs.update(kwargs)

    if in_colab:
        print("[Prism] Colab launch share=", share)
        return demo.launch(**launch_kwargs)
    return demo.launch(server_name=server_name, **launch_kwargs)


def tempfile_dir() -> str:
    return tempfile.gettempdir()


def launch_prism(**kwargs):
    return launch_gui(**kwargs)
