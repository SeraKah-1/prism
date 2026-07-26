"""Self-contained HTML report — shadcn-like visual language, embedded Plotly."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from stock_prob.viewmodel import PrismViewModel
from stock_prob.viz import build_fan_figure, build_metrics_figure


def _fig_html(fig: Any) -> str:
    if fig is None:
        return ""
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False})


def write_prism_report(vm: PrismViewModel, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    h = vm.primary_horizon()
    cone = vm.cones.get(h)
    fan = None
    if vm.history is not None and len(vm.history) and cone is not None and len(cone):
        fan = build_fan_figure(
            vm.history,
            cone,
            title=f"{vm.ticker} · {h}d cone · {vm.regime}",
        )
    met_fig = None
    if vm.metrics is not None and len(vm.metrics):
        met_fig = build_metrics_figure(vm.metrics, title="Brier vs baselines (lower better)")

    cards = vm.summary_cards()
    cards_html = []
    for c in cards:
        color = "#16a34a" if c["direction"] == "up" else "#dc2626"
        bg = "#f0fdf4" if c["direction"] == "up" else "#fef2f2"
        beat = c.get("beat_baseline")
        beat_txt = "beats base" if beat else ("≠ base" if beat is False else "—")
        cards_html.append(
            f"""
            <div class="card metric" style="border-color:{color}22;background:{bg}">
              <div class="muted">{c['horizon']}d horizon</div>
              <div class="big" style="color:{color}">{c['prob_up']*100:.1f}%</div>
              <div class="muted">P(up) · {beat_txt}</div>
            </div>
            """
        )

    metrics_table = ""
    if vm.metrics is not None and len(vm.metrics):
        show = vm.metrics.copy()
        metrics_table = show.to_html(index=False, float_format=lambda x: f"{x:.4f}", classes="tbl")

    skill_line = []
    for k, v in sorted(vm.brier_skill.items(), key=lambda x: int(x[0])):
        skill_line.append(f"h{k}: {v:+.4f}")
    skill_txt = " · ".join(skill_line) if skill_line else "n/a"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Prism · {vm.ticker}</title>
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <style>
    :root {{
      --bg: #fafafa;
      --card: #ffffff;
      --fg: #0a0a0a;
      --muted: #737373;
      --border: #e5e5e5;
      --primary: #0f172a;
      --ring: #94a3b8;
      --radius: 12px;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
      background: var(--bg); color: var(--fg); line-height: 1.5;
    }}
    .wrap {{ max-width: 1100px; margin: 0 auto; padding: 28px 20px 48px; }}
    header {{
      display: flex; flex-wrap: wrap; justify-content: space-between; gap: 12px;
      align-items: flex-end; margin-bottom: 20px;
    }}
    h1 {{ font-size: 1.75rem; letter-spacing: -0.03em; margin: 0; }}
    .badge {{
      display: inline-flex; align-items: center; gap: 6px;
      border: 1px solid var(--border); background: var(--card);
      border-radius: 999px; padding: 4px 12px; font-size: 12px; color: var(--muted);
    }}
    .badge strong {{ color: var(--fg); font-weight: 600; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin: 16px 0 20px; }}
    .card {{
      background: var(--card); border: 1px solid var(--border); border-radius: var(--radius);
      padding: 16px 18px; box-shadow: 0 1px 2px rgb(0 0 0 / 4%);
    }}
    .metric .big {{ font-size: 1.75rem; font-weight: 700; letter-spacing: -0.03em; margin: 4px 0; }}
    .muted {{ color: var(--muted); font-size: 12px; }}
    h2 {{ font-size: 1rem; margin: 0 0 12px; letter-spacing: -0.02em; }}
    .panel {{ margin-top: 16px; }}
    .tbl {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    .tbl th, .tbl td {{ border-bottom: 1px solid var(--border); padding: 8px 10px; text-align: left; }}
    .tbl th {{ color: var(--muted); font-weight: 500; }}
    footer {{ margin-top: 28px; color: var(--muted); font-size: 12px; }}
    code {{ background: #f4f4f5; padding: 2px 6px; border-radius: 6px; font-size: 12px; }}
  </style>
</head>
<body>
  <div class="wrap">
    <header>
      <div>
        <div class="muted" style="margin-bottom:4px">Prism · probability lab</div>
        <h1>{vm.ticker} <span class="muted" style="font-size:1rem;font-weight:500">{vm.name if vm.name != vm.ticker else ""}</span></h1>
      </div>
      <div style="display:flex;gap:8px;flex-wrap:wrap">
        <span class="badge">asof <strong>{vm.asof or "—"}</strong></span>
        <span class="badge">price <strong>{vm.last_price:,.2f}</strong></span>
        <span class="badge">regime <strong>{vm.regime}</strong></span>
        <span class="badge">run <strong>{vm.run_id[:18] if vm.run_id else "—"}…</strong></span>
      </div>
    </header>

    <div class="grid">
      {''.join(cards_html) or '<div class="card muted">No probabilities</div>'}
    </div>

    <div class="card panel">
      <h2>Prediction cone</h2>
      {_fig_html(fan) or '<p class="muted">Cone unavailable</p>'}
    </div>

    <div class="card panel">
      <h2>Skill vs baselines</h2>
      <p class="muted" style="margin-top:0">Brier skill (base − model); positive = better than base rate. {skill_txt}</p>
      {_fig_html(met_fig)}
      <div style="overflow:auto;margin-top:12px">{metrics_table}</div>
    </div>

    <div class="card panel">
      <h2>Details</h2>
      <p class="muted" style="margin:0">
        champion: <strong>{vm.champion or "—"}</strong> ·
        honesty: <strong>{vm.honesty_skill if vm.honesty_skill is not None else "—"}</strong><br/>
        artifacts: <code>{vm.art_dir or "—"}</code>
      </p>
    </div>

    <footer>
      Prism is research software — not financial advice. Charts show uncertainty, not promises.
    </footer>
  </div>
</body>
</html>
"""
    path.write_text(html)
    return path
