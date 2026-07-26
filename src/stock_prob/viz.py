"""Fan chart + metrics views. Works headless (write HTML/PNG)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:  # pragma: no cover
    go = None
    make_subplots = None


def build_fan_figure(
    history: pd.DataFrame,
    cone: pd.DataFrame,
    *,
    title: str = "Prediction cone",
    price_col: str = "close",
    date_col: str = "date",
) -> Any:
    if go is None:
        raise ImportError("plotly required for fan charts")

    fig = go.Figure()
    h = history.copy()
    h[date_col] = pd.to_datetime(h[date_col])
    fig.add_trace(
        go.Scatter(
            x=h[date_col],
            y=h[price_col],
            mode="lines",
            name="Close",
            line=dict(color="#1f77b4", width=2),
        )
    )

    c = cone.copy()
    c[date_col] = pd.to_datetime(c[date_col])
    # bands: p10-p90, p25-p75
    if {"p10", "p90"}.issubset(c.columns):
        fig.add_trace(
            go.Scatter(
                x=list(c[date_col]) + list(c[date_col][::-1]),
                y=list(c["p90"]) + list(c["p10"][::-1]),
                fill="toself",
                fillcolor="rgba(31,119,180,0.15)",
                line=dict(color="rgba(255,255,255,0)"),
                name="80% cone",
                hoverinfo="skip",
            )
        )
    if {"p25", "p75"}.issubset(c.columns):
        fig.add_trace(
            go.Scatter(
                x=list(c[date_col]) + list(c[date_col][::-1]),
                y=list(c["p75"]) + list(c["p25"][::-1]),
                fill="toself",
                fillcolor="rgba(31,119,180,0.28)",
                line=dict(color="rgba(255,255,255,0)"),
                name="50% cone",
                hoverinfo="skip",
            )
        )
    if "p50" in c.columns:
        fig.add_trace(
            go.Scatter(
                x=c[date_col],
                y=c["p50"],
                mode="lines",
                name="Median",
                line=dict(color="#ff7f0e", width=2, dash="dash"),
            )
        )

    fig.update_layout(
        title=title,
        template="plotly_white",
        xaxis_title="Date",
        yaxis_title="Price",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=40, r=20, t=60, b=40),
        height=480,
    )
    return fig


def build_metrics_figure(metrics: pd.DataFrame, title: str = "Brier vs baselines") -> Any:
    if go is None:
        raise ImportError("plotly required")
    fig = go.Figure()
    if len(metrics) == 0:
        return fig
    x = metrics["horizon"].astype(str)
    for col, name in [
        ("brier_model", "Model"),
        ("brier_base", "Base rate"),
        ("brier_momentum", "Momentum"),
    ]:
        if col in metrics.columns:
            fig.add_trace(go.Bar(name=name, x=x, y=metrics[col]))
    fig.update_layout(
        barmode="group",
        title=title,
        template="plotly_white",
        yaxis_title="Brier (lower better)",
        xaxis_title="Horizon (days)",
        height=360,
    )
    return fig


def write_html_report(
    path: str | Path,
    *,
    fan_fig: Any,
    metrics_fig: Any | None,
    metrics_df: pd.DataFrame,
    probs: dict[str, float],
    meta: dict[str, Any],
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    metrics_html = metrics_df.to_html(index=False, float_format=lambda x: f"{x:.4f}")
    probs_html = "<ul>" + "".join(f"<li>h={k}: P(up)={v:.3f}</li>" for k, v in probs.items()) + "</ul>"
    meta_html = "<ul>" + "".join(f"<li><b>{k}</b>: {v}</li>" for k, v in meta.items()) + "</ul>"

    fan_div = fan_fig.to_html(full_html=False, include_plotlyjs="cdn") if fan_fig is not None else ""
    met_div = (
        metrics_fig.to_html(full_html=False, include_plotlyjs=False)
        if metrics_fig is not None
        else ""
    )

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>SPE Report — {meta.get('ticker', '')}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; background: #f7f8fa; color: #1a1a1a; }}
    h1,h2 {{ color: #0f2744; }}
    .card {{ background: white; border-radius: 10px; padding: 16px 20px; margin-bottom: 16px;
             box-shadow: 0 1px 3px rgba(0,0,0,.08); }}
    table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
    th, td {{ border-bottom: 1px solid #e5e7eb; padding: 6px 8px; text-align: left; }}
  </style>
</head>
<body>
  <h1>Prism — Run Report</h1>
  <div class="card"><h2>Meta</h2>{meta_html}</div>
  <div class="card"><h2>P(up) by horizon</h2>{probs_html}</div>
  <div class="card"><h2>Fan chart</h2>{fan_div}</div>
  <div class="card"><h2>Metrics vs baselines</h2>{met_div}{metrics_html}</div>
</body>
</html>
"""
    path.write_text(html)
    return path


def export_metrics_excel(path: str | Path, sheets: dict[str, pd.DataFrame]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        for name, df in sheets.items():
            df.to_excel(w, sheet_name=name[:31], index=False)
    return path
