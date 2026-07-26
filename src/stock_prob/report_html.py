"""Self-contained Prism HTML report — matplotlib SVG charts, plain English."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from stock_prob.design import (
    ACCENT,
    ACCENT_SOFT,
    BG,
    BG_ELEV,
    BG_MUTED,
    BORDER,
    DOWN,
    DOWN_SOFT,
    DUR,
    DUR_FAST,
    DUR_SLOW,
    EASE,
    FG,
    FG_MUTED,
    FONT,
    FONT_MONO,
    UP,
    UP_SOFT,
)
from stock_prob.viewmodel import PrismViewModel
from stock_prob.viz import (
    build_fan_figure,
    build_metrics_figure,
    build_prob_gauge_figure,
    fig_to_html,
)


def _fig_div(fig: Any) -> str:
    if fig is None:
        return '<p class="muted">Chart unavailable</p>'
    return fig_to_html(fig)


def write_prism_report(vm: PrismViewModel, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    h = vm.primary_horizon()
    cone = vm.cones.get(h)
    if cone is None:
        cone = vm.cones.get(str(h))
    if cone is None and vm.cones:
        cone = next(iter(vm.cones.values()))
    fan = None
    try:
        hist_ok = vm.history is not None and len(vm.history) > 0
        cone_ok = cone is not None and len(cone) > 0
        if hist_ok and cone_ok:
            fan = build_fan_figure(
                vm.history,
                cone,
                title=f"{vm.ticker} · price range ({h}d) · regime {vm.regime}",
            )
    except Exception:
        fan = None
    met_fig = None
    try:
        if vm.metrics is not None and len(vm.metrics) > 0:
            met_fig = build_metrics_figure(vm.metrics, title="Brier score vs baselines")
    except Exception:
        met_fig = None
    gauge = None
    try:
        if vm.probs:
            gauge = build_prob_gauge_figure(vm.probs, title="P(up) by horizon")
    except Exception:
        gauge = None

    cards = vm.summary_cards()
    card_bits = []
    for i, c in enumerate(cards):
        up = c["direction"] == "up"
        color = UP if up else DOWN
        bg = UP_SOFT if up else DOWN_SOFT
        arrow = "▲" if up else "▼"
        beat = c.get("beat_baseline")
        if beat is True:
            skill = '<span class="pill ok">beats baseline</span>'
        elif beat is False:
            skill = '<span class="pill bad">below baseline</span>'
        else:
            skill = '<span class="pill">—</span>'
        card_bits.append(
            f"""
            <article class="card metric reveal" style="--d:{i * 70}ms;background:{bg};border-color:{color}33">
              <div class="metric-top">
                <span class="label">{c['horizon']} day</span>
                {skill}
              </div>
              <div class="metric-val" style="color:{color}">
                <span class="arrow">{arrow}</span>
                <span class="num">{c['prob_up']*100:.1f}%</span>
              </div>
              <div class="label">P(up) · conviction {c['conviction']*100:.0f}%</div>
              <div class="bar"><i style="width:{c['prob_up']*100:.1f}%;background:{color}"></i></div>
            </article>
            """
        )

    metrics_table = ""
    if vm.metrics is not None and len(vm.metrics):
        metrics_table = vm.metrics.to_html(
            index=False, float_format=lambda x: f"{x:.4f}", classes="tbl"
        )

    price_txt = f"{vm.last_price:,.2f}" if vm.last_price == vm.last_price else "—"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Prism · {vm.ticker}</title>
<style>
  :root {{
    --bg: {BG}; --elev: {BG_ELEV}; --muted-bg: {BG_MUTED};
    --fg: {FG}; --muted: {FG_MUTED}; --border: {BORDER};
    --accent: {ACCENT}; --accent-soft: {ACCENT_SOFT};
    --up: {UP}; --down: {DOWN};
    --ease: {EASE}; --dur: {DUR}; --dur-fast: {DUR_FAST}; --dur-slow: {DUR_SLOW};
    --font: {FONT}; --mono: {FONT_MONO};
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    font-family: var(--font);
    background:
      radial-gradient(1200px 600px at 10% -10%, #e7f0ea 0%, transparent 55%),
      radial-gradient(900px 500px at 100% 0%, #f3e9df 0%, transparent 50%),
      var(--bg);
    color: var(--fg);
    line-height: 1.55;
    min-height: 100vh;
  }}
  .wrap {{ max-width: 1120px; margin: 0 auto; padding: 32px 20px 64px; }}
  .reveal {{
    opacity: 0;
    transform: translateY(14px);
    animation: rise var(--dur-slow) var(--ease) forwards;
    animation-delay: var(--d, 0ms);
  }}
  @keyframes rise {{
    to {{ opacity: 1; transform: translateY(0); }}
  }}
  header.hero {{
    display: grid;
    grid-template-columns: 1fr auto;
    gap: 16px;
    align-items: end;
    margin-bottom: 22px;
  }}
  @media (max-width: 720px) {{
    header.hero {{ grid-template-columns: 1fr; }}
  }}
  .eyebrow {{
    font-size: 11px; letter-spacing: 0.14em; text-transform: uppercase;
    color: var(--muted); font-weight: 600; margin-bottom: 6px;
  }}
  h1 {{
    margin: 0; font-size: clamp(1.8rem, 3vw, 2.35rem);
    letter-spacing: -0.04em; font-weight: 700; line-height: 1.1;
  }}
  h1 .sub {{ color: var(--muted); font-weight: 500; font-size: 0.55em; }}
  .chips {{ display: flex; flex-wrap: wrap; gap: 8px; justify-content: flex-end; }}
  .chip {{
    border: 1px solid var(--border); background: var(--elev);
    border-radius: 999px; padding: 6px 12px; font-size: 12px; color: var(--muted);
  }}
  .chip b {{ color: var(--fg); font-weight: 600; font-family: var(--mono); font-size: 11px; }}
  .grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(168px, 1fr));
    gap: 12px; margin: 8px 0 18px;
  }}
  .card {{
    background: color-mix(in srgb, var(--elev) 92%, white);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 16px 18px;
    box-shadow: 0 1px 0 rgba(28,25,23,0.03), 0 12px 32px -24px rgba(28,25,23,0.25);
  }}
  .metric-top {{ display: flex; justify-content: space-between; align-items: center; gap: 8px; }}
  .label {{ font-size: 12px; color: var(--muted); }}
  .metric-val {{
    display: flex; align-items: baseline; gap: 8px;
    margin: 8px 0 4px; letter-spacing: -0.03em;
  }}
  .metric-val .num {{ font-size: 2rem; font-weight: 700; font-variant-numeric: tabular-nums; }}
  .metric-val .arrow {{ font-size: 1rem; opacity: 0.85; }}
  .pill {{
    font-size: 10px; font-weight: 600;
    border-radius: 999px; padding: 3px 8px; border: 1px solid var(--border);
    background: var(--elev); color: var(--muted); text-transform: uppercase;
  }}
  .pill.ok {{ background: {UP_SOFT}; color: var(--up); }}
  .pill.bad {{ background: {DOWN_SOFT}; color: var(--down); }}
  .bar {{
    margin-top: 10px; height: 6px; border-radius: 99px; background: rgba(28,25,23,0.06); overflow: hidden;
  }}
  .bar i {{ display: block; height: 100%; border-radius: 99px; }}
  .panel {{ margin-top: 14px; }}
  .panel h2 {{ margin: 0 0 4px; font-size: 0.95rem; letter-spacing: -0.02em; }}
  .panel .hint {{ margin: 0 0 12px; color: var(--muted); font-size: 12.5px; }}
  .chart-svg svg {{ max-width: 100%; height: auto; display: block; }}
  .tbl {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  .tbl th, .tbl td {{
    border-bottom: 1px solid var(--border); padding: 10px 8px; text-align: left;
    font-variant-numeric: tabular-nums;
  }}
  .tbl th {{ color: var(--muted); font-weight: 500; font-size: 11px; text-transform: uppercase; }}
  .banner {{
    display: flex; gap: 12px; align-items: center; flex-wrap: wrap;
    padding: 12px 16px; border-radius: 14px; border: 1px solid var(--border);
    background: linear-gradient(120deg, var(--accent-soft), var(--elev) 55%);
    margin-bottom: 16px;
  }}
  .banner .mark {{
    width: 10px; height: 10px; border-radius: 99px; background: var(--accent);
  }}
  footer {{
    margin-top: 28px; color: var(--muted); font-size: 12px;
    display: flex; justify-content: space-between; gap: 12px; flex-wrap: wrap;
  }}
  code {{
    font-family: var(--mono); font-size: 11px; background: var(--muted-bg);
    padding: 2px 7px; border-radius: 6px; color: var(--fg);
  }}
</style>
</head>
<body>
  <div class="wrap">
    <header class="hero reveal" style="--d:0ms">
      <div>
        <div class="eyebrow">Prism · probability lab</div>
        <h1>{vm.ticker} <span class="sub">{vm.name if vm.name and vm.name != vm.ticker else ""}</span></h1>
      </div>
      <div class="chips">
        <span class="chip">as of <b>{vm.asof or "—"}</b></span>
        <span class="chip">last <b>{price_txt}</b></span>
        <span class="chip">regime <b>{vm.regime}</b></span>
        <span class="chip">run <b>{(vm.run_id or "—")[:16]}</b></span>
      </div>
    </header>

    <div class="banner reveal" style="--d:60ms">
      <span class="mark"></span>
      <div>
        <strong>Research view, not a trade signal.</strong>
        <span class="label"> Bands widen with horizon. Compare Brier to the base rate.</span>
      </div>
    </div>

    <section class="grid">
      {''.join(card_bits) or '<div class="card"><span class="label">No probabilities</span></div>'}
    </section>

    <section class="card panel reveal" style="--d:160ms">
      <h2>Price range (fan chart)</h2>
      <p class="hint">Median path with 50% and 80% bands. Wider band = more uncertainty.</p>
      {_fig_div(fan)}
    </section>

    <section class="card panel reveal" style="--d:220ms">
      <h2>Direction</h2>
      <p class="hint">50% is a coin flip. Distance from 50% is model conviction, not a guarantee.</p>
      {_fig_div(gauge)}
    </section>

    <section class="card panel reveal" style="--d:280ms">
      <h2>Model vs baselines</h2>
      <p class="hint">Lower Brier is better. If the model loses to the base rate, treat the signal as weak.</p>
      {_fig_div(met_fig)}
      <div style="overflow:auto;margin-top:8px">{metrics_table}</div>
    </section>

    <section class="card panel reveal" style="--d:340ms">
      <h2>Lab notes</h2>
      <p class="hint" style="margin-bottom:0">
        champion <code>{vm.champion or "—"}</code>
        · honesty skill <code>{vm.honesty_skill if vm.honesty_skill is not None else "—"}</code>
        · artifacts <code>{vm.art_dir or "—"}</code>
      </p>
    </section>

    <footer class="reveal" style="--d:400ms">
      <span>Prism · research software · not financial advice</span>
      <span>Charts: matplotlib SVG</span>
    </footer>
  </div>
</body>
</html>
"""
    path.write_text(html)
    return path
