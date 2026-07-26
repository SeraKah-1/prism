"""Charts via matplotlib → inline SVG.

Why not Plotly in Gradio: gr.HTML often blocks Plotly's <script> + CDN.
SVG needs no JS, no CDN — it just shows.
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
        if svg.lstrip().startswith("<?xml"):
            svg = svg.split("?>", 1)[-1]
        return f'<div class="chart-svg" style="width:100%;overflow:auto">{svg}</div>'
    except Exception as e:
        try:
            plt.close(fig)
        except Exception:
            pass
        return f'<p class="hint" style="padding:12px">Chart render failed: {e}</p>'


def _zoom_history(history: pd.DataFrame, horizon: int | None, date_col: str = "date") -> pd.DataFrame:
    """Keep enough history to read the cone: ~3× horizon (min 60 bars), or full if short."""
    h = history.copy()
    if horizon is None or horizon <= 0 or len(h) <= 80:
        return h
    keep = int(max(60, min(len(h), horizon * 3 + 20)))
    return h.tail(keep).reset_index(drop=True)


def build_fan_figure(
    history: pd.DataFrame,
    cone: pd.DataFrame | None = None,
    *,
    title: str = "Price range",
    price_col: str = "close",
    date_col: str = "date",
    supports: list[dict[str, Any]] | None = None,
    resistances: list[dict[str, Any]] | None = None,
    horizon: int | None = None,
    nested_cones: dict[int, pd.DataFrame] | None = None,
    show_nested: bool = False,
    animate: bool = False,  # API compat
) -> Any:
    """
    Fan chart with:
      - clear as-of line + last price label
      - p10/p50/p90 end labels
      - optional support/resistance
      - zoom scaled to selected horizon
      - optional nested cones for other horizons (lighter)
    """
    if plt is None:
        raise ImportError("matplotlib required for charts")

    h = history.copy()
    h[date_col] = pd.to_datetime(h[date_col])
    h = h.sort_values(date_col).reset_index(drop=True)

    # horizon from cone length if not given
    if horizon is None and cone is not None and len(cone):
        horizon = len(cone)
    h_zoom = _zoom_history(h, horizon, date_col=date_col)

    fig, ax = plt.subplots(figsize=(9.4, 4.6), dpi=120)
    _style_ax(ax)

    closes = h_zoom[price_col].astype(float)
    dates = h_zoom[date_col]
    ax.plot(dates, closes, color=CHART_LINE, lw=1.8, label="Close", zorder=3)

    last_date = pd.Timestamp(dates.iloc[-1]) if len(dates) else None
    last_price = float(closes.iloc[-1]) if len(closes) else float("nan")

    # nested background cones (other horizons)
    if show_nested and nested_cones:
        for hh, cdf in sorted(nested_cones.items(), key=lambda x: -int(x[0])):
            if horizon is not None and int(hh) == int(horizon):
                continue
            if cdf is None or len(cdf) == 0:
                continue
            cc = cdf.copy()
            cc[date_col] = pd.to_datetime(cc[date_col])
            if {"p10", "p90"}.issubset(cc.columns):
                ax.fill_between(
                    cc[date_col],
                    cc["p10"].astype(float),
                    cc["p90"].astype(float),
                    color=ACCENT,
                    alpha=0.04,
                    linewidth=0,
                    zorder=1,
                )

    # primary cone
    if cone is not None and len(cone):
        c = cone.copy()
        c[date_col] = pd.to_datetime(c[date_col])

        # split band color above/below last for direction readability
        if {"p10", "p90"}.issubset(c.columns) and np.isfinite(last_price):
            p10 = c["p10"].astype(float).values
            p90 = c["p90"].astype(float).values
            x = c[date_col]
            # full 80% band (neutral)
            ax.fill_between(
                x, p10, p90, color=ACCENT, alpha=0.10, label="80% band", linewidth=0, zorder=1
            )
            # emphasize upside portion above last
            ax.fill_between(
                x,
                np.maximum(p10, last_price),
                p90,
                where=p90 > last_price,
                color=UP,
                alpha=0.10,
                linewidth=0,
                zorder=2,
                label="Above last",
            )
            ax.fill_between(
                x,
                p10,
                np.minimum(p90, last_price),
                where=p10 < last_price,
                color=DOWN,
                alpha=0.08,
                linewidth=0,
                zorder=2,
                label="Below last",
            )
        if {"p25", "p75"}.issubset(c.columns):
            ax.fill_between(
                c[date_col],
                c["p25"].astype(float),
                c["p75"].astype(float),
                color=ACCENT,
                alpha=0.18,
                label="50% band",
                linewidth=0,
                zorder=2,
            )
        if "p50" in c.columns:
            med = c["p50"].astype(float)
            ax.plot(
                c[date_col],
                med,
                color=CHART_MEDIAN,
                lw=2.0,
                ls="--",
                label="Median",
                zorder=4,
            )

        # end-of-horizon price markers
        end_x = c[date_col].iloc[-1]
        for col, color, name in (
            ("p10", DOWN, "p10"),
            ("p50", CHART_MEDIAN, "p50"),
            ("p90", UP, "p90"),
        ):
            if col not in c.columns:
                continue
            y = float(c[col].iloc[-1])
            ax.scatter([end_x], [y], s=28, color=color, zorder=6, edgecolors=BG_ELEV, linewidths=0.8)
            ax.annotate(
                f"{name} {y:,.2f}",
                xy=(end_x, y),
                xytext=(8, 0),
                textcoords="offset points",
                fontsize=8,
                color=color,
                va="center",
                fontweight="600",
            )

    # as-of line + last price
    if last_date is not None and np.isfinite(last_price):
        ax.axvline(last_date, color=ACCENT, ls="-", lw=1.2, alpha=0.85, zorder=5)
        ax.scatter(
            [last_date],
            [last_price],
            s=48,
            color=ACCENT,
            zorder=7,
            edgecolors=BG_ELEV,
            linewidths=1.4,
            label="Now",
        )
        ax.annotate(
            f"Last {last_price:,.2f}",
            xy=(last_date, last_price),
            xytext=(-8, 12),
            textcoords="offset points",
            fontsize=9,
            color=FG,
            fontweight="700",
            ha="right",
            bbox=dict(boxstyle="round,pad=0.25", fc=BG_ELEV, ec=BORDER, alpha=0.95),
        )
        # subtle last-price horizontal guide
        ax.axhline(last_price, color=BORDER, ls=":", lw=1.0, zorder=1)

    # support / resistance
    for lv in supports or []:
        try:
            y = float(lv["price"])
        except Exception:
            continue
        ax.axhline(y, color=UP, ls="--", lw=1.0, alpha=0.55, zorder=1)
        ax.text(
            0.01,
            y,
            f"  S {y:,.2f}",
            transform=ax.get_yaxis_transform(),
            fontsize=8,
            color=UP,
            va="bottom",
            fontweight="600",
        )
    for lv in resistances or []:
        try:
            y = float(lv["price"])
        except Exception:
            continue
        ax.axhline(y, color=DOWN, ls="--", lw=1.0, alpha=0.55, zorder=1)
        ax.text(
            0.01,
            y,
            f"  R {y:,.2f}",
            transform=ax.get_yaxis_transform(),
            fontsize=8,
            color=DOWN,
            va="top",
            fontweight="600",
        )

    ax.set_title(title, fontsize=12, fontweight="600", loc="left", pad=10)
    ax.set_ylabel("Price")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=7))
    fig.autofmt_xdate(rotation=0, ha="center")
    # de-dupe legend labels
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    H, L = [], []
    for hh, ll in zip(handles, labels):
        if ll in seen:
            continue
        seen.add(ll)
        H.append(hh)
        L.append(ll)
    leg = ax.legend(H, L, loc="upper left", frameon=False, fontsize=7.5, ncols=3, labelcolor=FG_MUTED)
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
        ax.bar(
            x + i * width - (len(present) - 1) * width / 2,
            vals,
            width * 0.9,
            label=name,
            color=color,
            zorder=2,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("Brier (lower is better)")
    ax.set_title(title, fontsize=12, fontweight="600", loc="left", pad=10)
    ax.legend(loc="upper right", frameon=False, fontsize=8, labelcolor=FG_MUTED)
    fig.tight_layout()
    return fig


def build_prob_gauge_figure(probs: dict[str, float], title: str = "P(up) by horizon") -> Any:
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
