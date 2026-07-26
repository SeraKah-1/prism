"""Charts via matplotlib → inline SVG.

Why not Plotly in Gradio: gr.HTML often blocks or fails to run Plotly's
<script> + CDN. SVG needs no JS, no CDN, no JSON inject — it just shows.
"""
from __future__ import annotations

from io import StringIO
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
    DOWN,
    FG,
    FG_MUTED,
    GRID,
    UP,
)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
except ImportError:  # pragma: no cover
    plt = None
    Figure = Any  # type: ignore
    mdates = None


def _style_ax(ax) -> None:
    ax.set_facecolor(BG_ELEV)
    ax.figure.patch.set_facecolor(BG_ELEV)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(BORDER)
    ax.spines["bottom"].set_color(BORDER)
    ax.tick_params(colors=FG_MUTED, labelsize=9)
    ax.yaxis.label.set_color(FG_MUTED)
    ax.xaxis.label.set_color(FG_MUTED)
    ax.title.set_color(FG)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)


def fig_to_html(fig: Any, *, height: int | None = None) -> str:
    """Turn a matplotlib figure into inline SVG HTML (Gradio-safe)."""
    if fig is None:
        return '<p class="hint" style="padding:12px">Chart not available.</p>'
    if isinstance(fig, str):
        return fig
    if plt is None:
        return '<p class="hint" style="padding:12px">matplotlib is required for charts.</p>'
    try:
        if height and hasattr(fig, "set_size_inches"):
            w, _ = fig.get_size_inches()
            fig.set_size_inches(w, max(2.2, height / 96.0))
        buf = StringIO()
        fig.savefig(
            buf,
            format="svg",
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
            edgecolor="none",
        )
        svg = buf.getvalue()
        plt.close(fig)
        # Strip XML declaration for cleaner embed
        if svg.lstrip().startswith("<?xml"):
            svg = svg.split("?>", 1)[-1]
        return f'<div class="chart-svg" style="width:100%;overflow:auto">{svg}</div>'
    except Exception as e:
        try:
            plt.close(fig)
        except Exception:
            pass
        return f'<p class="hint" style="padding:12px">Chart render failed: {e}</p>'


def build_fan_figure(
    history: pd.DataFrame,
    cone: pd.DataFrame,
    *,
    title: str = "Price range",
    price_col: str = "close",
    date_col: str = "date",
    animate: bool = False,  # kept for API compat; unused
) -> Any:
    if plt is None:
        raise ImportError("matplotlib required for charts")

    h = history.copy()
    h[date_col] = pd.to_datetime(h[date_col])
    c = cone.copy()
    c[date_col] = pd.to_datetime(c[date_col])

    fig, ax = plt.subplots(figsize=(9.2, 4.2), dpi=120)
    _style_ax(ax)

    ax.plot(
        h[date_col],
        h[price_col].astype(float),
        color=CHART_LINE,
        lw=1.8,
        label="Close",
        zorder=3,
    )

    if {"p10", "p90"}.issubset(c.columns):
        ax.fill_between(
            c[date_col],
            c["p10"].astype(float),
            c["p90"].astype(float),
            color=ACCENT,
            alpha=0.12,
            label="80% band",
            linewidth=0,
            zorder=1,
        )
    if {"p25", "p75"}.issubset(c.columns):
        ax.fill_between(
            c[date_col],
            c["p25"].astype(float),
            c["p75"].astype(float),
            color=ACCENT,
            alpha=0.22,
            label="50% band",
            linewidth=0,
            zorder=2,
        )
    if "p50" in c.columns:
        ax.plot(
            c[date_col],
            c["p50"].astype(float),
            color=CHART_MEDIAN,
            lw=1.8,
            ls="--",
            label="Median",
            zorder=4,
        )

    if len(h):
        ax.scatter(
            [h[date_col].iloc[-1]],
            [float(h[price_col].iloc[-1])],
            s=36,
            color=ACCENT,
            zorder=5,
            edgecolors=BG_ELEV,
            linewidths=1.2,
            label="Now",
        )

    ax.set_title(title, fontsize=12, fontweight="600", loc="left", pad=10)
    ax.set_ylabel("Price")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=7))
    fig.autofmt_xdate(rotation=0, ha="center")
    leg = ax.legend(
        loc="upper left",
        frameon=False,
        fontsize=8,
        ncols=3,
        labelcolor=FG_MUTED,
    )
    for t in leg.get_texts():
        t.set_color(FG_MUTED)
    fig.tight_layout()
    return fig


def build_metrics_figure(metrics: pd.DataFrame, title: str = "Brier vs baselines") -> Any:
    if plt is None:
        raise ImportError("matplotlib required for charts")

    fig, ax = plt.subplots(figsize=(8.5, 3.4), dpi=120)
    _style_ax(ax)

    if metrics is None or len(metrics) == 0:
        ax.set_title(title, fontsize=12, fontweight="600", loc="left", pad=10)
        ax.text(0.5, 0.5, "No metrics", ha="center", va="center", color=FG_MUTED, transform=ax.transAxes)
        fig.tight_layout()
        return fig

    x_labels = [f"{int(h)}d" for h in metrics["horizon"]]
    x = np.arange(len(x_labels))
    series = [
        ("brier_model", "Model", ACCENT),
        ("brier_base", "Base rate", FG_MUTED),
        ("brier_momentum", "Momentum", "#a8a29e"),
    ]
    present = [(col, name, color) for col, name, color in series if col in metrics.columns]
    width = 0.8 / max(len(present), 1)
    for i, (col, name, color) in enumerate(present):
        vals = metrics[col].astype(float).values
        ax.bar(x + i * width - (len(present) - 1) * width / 2, vals, width * 0.9, label=name, color=color, zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("Brier (lower is better)")
    ax.set_title(title, fontsize=12, fontweight="600", loc="left", pad=10)
    ax.legend(loc="upper right", frameon=False, fontsize=8, labelcolor=FG_MUTED)
    fig.tight_layout()
    return fig


def build_prob_gauge_figure(probs: dict[str, float], title: str = "P(up) by horizon") -> Any:
    """Horizontal bars for P(up) per time horizon."""
    if plt is None:
        raise ImportError("matplotlib required for charts")

    from stock_prob.horizon_keys import parse_horizon_key

    items = []
    for k, v in (probs or {}).items():
        h = parse_horizon_key(k)
        if h is None:
            continue
        try:
            pf = float(v)
        except Exception:
            continue
        if pf == pf:
            items.append((h, pf * 100))
    items.sort(key=lambda x: x[0])

    height = max(2.4, 1.2 + 0.45 * max(len(items), 1))
    fig, ax = plt.subplots(figsize=(8.5, height), dpi=120)
    _style_ax(ax)

    if not items:
        ax.set_title(title, fontsize=12, fontweight="600", loc="left", pad=10)
        ax.text(0.5, 0.5, "No probabilities", ha="center", va="center", color=FG_MUTED, transform=ax.transAxes)
        fig.tight_layout()
        return fig

    labels = [f"{h}d" for h, _ in items]
    vals = [v for _, v in items]
    colors = [UP if v >= 50 else DOWN for v in vals]
    y = np.arange(len(labels))

    ax.barh(y, vals, color=colors, height=0.62, zorder=2)
    ax.axvline(50, color=BORDER, ls=":", lw=1.2, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 100)
    ax.invert_yaxis()
    ax.set_xlabel("P(up) %")
    ax.set_title(title, fontsize=12, fontweight="600", loc="left", pad=10)
    for yi, v in zip(y, vals):
        ax.text(min(v + 1.5, 96), yi, f"{v:.1f}%", va="center", ha="left", fontsize=9, color=FG_MUTED)
    fig.tight_layout()
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
    if meta.get("_result"):
        from stock_prob.viewmodel import build_viewmodel

        vm = build_viewmodel(meta["_result"], name=str(meta.get("ticker", "")))
        if probs:
            vm.probs = normalize_prob_map(probs)
    return write_prism_report(vm, Path(path))
