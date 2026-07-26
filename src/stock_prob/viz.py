"""Polished Plotly charts — light editorial theme, cone-first."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stock_prob.design import (
    ACCENT,
    BG_ELEV,
    BORDER,
    CHART_LINE,
    CHART_MEDIAN,
    CONE_INNER,
    CONE_OUTER,
    DOWN,
    FG,
    FG_MUTED,
    FONT,
    UP,
    plotly_layout_base,
)

try:
    import plotly.graph_objects as go
except ImportError:  # pragma: no cover
    go = None


def build_fan_figure(
    history: pd.DataFrame,
    cone: pd.DataFrame,
    *,
    title: str = "Prediction cone",
    price_col: str = "close",
    date_col: str = "date",
    animate: bool = True,
) -> Any:
    if go is None:
        raise ImportError("plotly required for fan charts")

    h = history.copy()
    h[date_col] = pd.to_datetime(h[date_col])
    c = cone.copy()
    c[date_col] = pd.to_datetime(c[date_col])

    fig = go.Figure()

    # History — ink line
    fig.add_trace(
        go.Scatter(
            x=h[date_col],
            y=h[price_col],
            mode="lines",
            name="Close",
            line=dict(color=CHART_LINE, width=2.2, shape="spline"),
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:,.2f}<extra>Close</extra>",
        )
    )

    # Outer 80% band
    if {"p10", "p90"}.issubset(c.columns):
        fig.add_trace(
            go.Scatter(
                x=list(c[date_col]) + list(c[date_col][::-1]),
                y=list(c["p90"]) + list(c["p10"][::-1]),
                fill="toself",
                fillcolor=CONE_OUTER,
                line=dict(color="rgba(0,0,0,0)"),
                name="80% band",
                hoverinfo="skip",
            )
        )
    # Inner 50% band
    if {"p25", "p75"}.issubset(c.columns):
        fig.add_trace(
            go.Scatter(
                x=list(c[date_col]) + list(c[date_col][::-1]),
                y=list(c["p75"]) + list(c["p25"][::-1]),
                fill="toself",
                fillcolor=CONE_INNER,
                line=dict(color="rgba(0,0,0,0)"),
                name="50% band",
                hoverinfo="skip",
            )
        )
    # Median
    if "p50" in c.columns:
        fig.add_trace(
            go.Scatter(
                x=c[date_col],
                y=c["p50"],
                mode="lines",
                name="Median path",
                line=dict(color=CHART_MEDIAN, width=2.4, dash="dot"),
                hovertemplate="%{x|%Y-%m-%d}<br>median %{y:,.2f}<extra></extra>",
            )
        )

    # Seam marker: last history point
    if len(h):
        fig.add_trace(
            go.Scatter(
                x=[h[date_col].iloc[-1]],
                y=[h[price_col].iloc[-1]],
                mode="markers",
                name="Now",
                marker=dict(size=10, color=ACCENT, line=dict(width=2, color=BG_ELEV)),
                hovertemplate="Now %{y:,.2f}<extra></extra>",
            )
        )

    layout = plotly_layout_base(height=500, title=title)
    layout["yaxis"]["title"] = "Price"
    layout["xaxis"]["title"] = ""
    fig.update_layout(**layout)

    if animate:
        # Subtle fade-in via frames (satisfying without gimmick)
        fig.update_layout(transition_duration=500)

    return fig


def build_metrics_figure(metrics: pd.DataFrame, title: str = "Brier vs baselines") -> Any:
    if go is None:
        raise ImportError("plotly required")
    fig = go.Figure()
    if metrics is None or len(metrics) == 0:
        fig.update_layout(**plotly_layout_base(320, title))
        return fig

    x = metrics["horizon"].astype(str) + "d"
    series = [
        ("brier_model", "Model", ACCENT),
        ("brier_base", "Base rate", FG_MUTED),
        ("brier_momentum", "Momentum", "#a8a29e"),
    ]
    for col, name, color in series:
        if col not in metrics.columns:
            continue
        fig.add_trace(
            go.Bar(
                name=name,
                x=x,
                y=metrics[col],
                marker=dict(color=color, cornerradius=4),
                hovertemplate="%{x}<br>%{y:.4f}<extra>" + name + "</extra>",
            )
        )
    layout = plotly_layout_base(height=340, title=title)
    layout["barmode"] = "group"
    layout["bargap"] = 0.28
    layout["yaxis"]["title"] = "Brier (lower better)"
    layout["yaxis"]["gridcolor"] = "#ebe4d8"
    fig.update_layout(**layout)
    return fig


def build_prob_gauge_figure(probs: dict[str, float], title: str = "P(up) by horizon") -> Any:
    """Horizontal conviction bars — readable at a glance."""
    if go is None:
        raise ImportError("plotly required")
    items = sorted(
        [(int(k), float(v)) for k, v in probs.items() if v == v],
        key=lambda x: x[0],
    )
    if not items:
        fig = go.Figure()
        fig.update_layout(**plotly_layout_base(220, title))
        return fig
    hs = [f"{h}d" for h, _ in items]
    vs = [p * 100 for _, p in items]
    colors = [UP if p >= 50 else DOWN for p in vs]
    fig = go.Figure(
        go.Bar(
            x=vs,
            y=hs,
            orientation="h",
            marker=dict(color=colors, cornerradius=6),
            text=[f"{v:.1f}%" for v in vs],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
        )
    )
    layout = plotly_layout_base(height=220 + 28 * len(items), title=title)
    layout["xaxis"] = dict(
        range=[0, 100],
        title="P(up) %",
        gridcolor="#ebe4d8",
        zeroline=False,
    )
    layout["yaxis"] = dict(autorange="reversed", title="")
    fig.update_layout(**layout)
    # neutral mid line
    fig.add_vline(x=50, line_width=1, line_dash="dot", line_color=BORDER)
    return fig


def export_metrics_excel(path: str | Path, sheets: dict[str, pd.DataFrame]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        for name, df in sheets.items():
            df.to_excel(w, sheet_name=str(name)[:31], index=False)
    return path


def write_html_report(
    path: str | Path,
    *,
    fan_fig: Any,
    metrics_fig: Any | None,
    metrics_df: pd.DataFrame,
    probs: dict[str, float],
    meta: dict[str, Any],
) -> Path:
    """Legacy thin wrapper — prefer report_html.write_prism_report."""
    from stock_prob.horizon_keys import normalize_prob_map
    from stock_prob.report_html import write_prism_report
    from stock_prob.viewmodel import PrismViewModel

    vm = PrismViewModel(
        ticker=str(meta.get("ticker", "")),
        probs=normalize_prob_map(probs or {}),
        metrics=metrics_df if metrics_df is not None else pd.DataFrame(),
        run_id=str(meta.get("run_id", "")),
        regime=str(meta.get("regime", "n/a")),
        report_html="",
    )
    # Prefer full result dict if caller attached history/cones via meta
    if meta.get("_result"):
        from stock_prob.viewmodel import build_viewmodel

        vm = build_viewmodel(meta["_result"], name=str(meta.get("ticker", "")))
        if probs:
            vm.probs = normalize_prob_map(probs)
    return write_prism_report(vm, Path(path))
