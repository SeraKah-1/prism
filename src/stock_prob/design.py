"""
Prism visual language — light, editorial, anti-slop.

Influences (taste, not clone):
- shadcn/ui light surfaces + density
- Linear/Vercel product chrome (whitespace, restrained accent)
- motion-ui-design principles: short ease, staged reveal, purposeful motion
- Avoid: full black, neon cyan/purple “AI dashboard”, rainbow gradients
"""
from __future__ import annotations

# Visible stamp so Colab users can verify they loaded the polished build
UX_BUILD = "PRISM_UX_2026_07_26_HUMAN_V3"
UX_LABEL = "Prism UX · card frames · safe files · v3"

# —— Tokens (light) ——
BG = "#f6f3ee"          # warm paper
BG_ELEV = "#fffcf7"     # card
BG_MUTED = "#efeae2"
FG = "#1c1917"          # stone-900
FG_MUTED = "#78716c"    # stone-500
BORDER = "#e7e0d5"
ACCENT = "#0f3d3e"      # deep teal-ink (not neon)
ACCENT_SOFT = "#d8ebe6"
UP = "#1b7a4e"
UP_SOFT = "#e6f5ec"
DOWN = "#b42318"
DOWN_SOFT = "#fdecea"
WARN = "#b45309"
CHART_LINE = "#1c1917"
CHART_MEDIAN = "#c2410c"  # restrained orange
CONE_OUTER = "rgba(15, 61, 62, 0.10)"
CONE_INNER = "rgba(15, 61, 62, 0.20)"
GRID = "#ebe4d8"

FONT = "ui-sans-serif, system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif"
FONT_MONO = "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace"

# Motion (CSS) — ease out expo-ish, short
EASE = "cubic-bezier(0.16, 1, 0.3, 1)"
DUR_FAST = "180ms"
DUR = "420ms"
DUR_SLOW = "700ms"


def plotly_layout_base(height: int = 460, title: str = "") -> dict:
    return dict(
        template="plotly_white",
        height=height,
        title=dict(
            text=title,
            font=dict(family=FONT, size=15, color=FG),
            x=0.0,
            xanchor="left",
        ),
        font=dict(family=FONT, size=12, color=FG_MUTED),
        paper_bgcolor=BG_ELEV,
        plot_bgcolor=BG_ELEV,
        margin=dict(l=48, r=24, t=56, b=48),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            x=0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11, color=FG_MUTED),
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor=GRID,
            zeroline=False,
            linecolor=BORDER,
            tickfont=dict(color=FG_MUTED),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=GRID,
            zeroline=False,
            linecolor=BORDER,
            tickfont=dict(color=FG_MUTED),
        ),
        hoverlabel=dict(bgcolor=BG_ELEV, font_size=12, font_family=FONT, bordercolor=BORDER),
        transition=dict(duration=450, easing="cubic-in-out"),
    )
