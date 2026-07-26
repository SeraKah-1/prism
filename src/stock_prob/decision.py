"""Data-driven buy / wait / avoid conclusion + suggested zones (research only)."""
from __future__ import annotations

from typing import Any

import numpy as np

from stock_prob.horizon_keys import parse_horizon_key


def _num(x: Any, default: float = float("nan")) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def build_decision(
    *,
    probs: dict[str, float],
    beat_baseline: dict[str, bool] | None = None,
    brier_skill: dict[str, float] | None = None,
    last_price: float,
    supports: list[dict[str, Any]] | None = None,
    resistances: list[dict[str, Any]] | None = None,
    character: dict[str, Any] | None = None,
    primary_horizon: int | None = None,
) -> dict[str, Any]:
    """
    Honest research summary — not a trade order.

    Uses:
      - P(up) vs 50%
      - whether model beats base-rate baseline (walk-forward)
      - distance to nearest support / resistance
      - market character (trend × vol)
    """
    beat_baseline = beat_baseline or {}
    brier_skill = brier_skill or {}
    supports = supports or []
    resistances = resistances or []
    character = character or {}

    items: list[tuple[int, float]] = []
    for k, p in (probs or {}).items():
        h = parse_horizon_key(k)
        if h is None:
            continue
        pf = _num(p)
        if np.isfinite(pf):
            items.append((int(h), pf))
    items.sort(key=lambda x: x[0])
    if not items:
        return {
            "action": "NO_SIGNAL",
            "stance": "No probability signal",
            "confidence": 0.0,
            "buy_zone": None,
            "sell_zone": None,
            "summary": "No P(up) available. Do not act on empty model output.",
            "bullets": ["Model did not produce probabilities for the selected horizons."],
            "primary_horizon": primary_horizon,
            "primary_p_up": None,
        }

    if primary_horizon is None:
        primary_horizon = items[0][0]
    # prefer requested primary if present
    pmap = {h: p for h, p in items}
    p_up = pmap.get(int(primary_horizon), items[0][1])
    primary_horizon = int(primary_horizon if int(primary_horizon) in pmap else items[0][0])
    p_up = float(pmap[primary_horizon])

    # aggregate edge across horizons
    n_beat = 0
    n_known = 0
    for h, _ in items:
        b = beat_baseline.get(str(h), beat_baseline.get(h))
        if b is True:
            n_beat += 1
            n_known += 1
        elif b is False:
            n_known += 1

    conviction = abs(p_up - 0.5) * 2.0  # 0..1
    ns = supports[0] if supports else None
    nr = resistances[0] if resistances else None
    dist_sup = abs(_num(ns.get("distance_pct") if ns else None)) if ns else float("nan")
    dist_res = abs(_num(nr.get("distance_pct") if nr else None)) if nr else float("nan")

    near_support = np.isfinite(dist_sup) and dist_sup <= 3.0
    near_resistance = np.isfinite(dist_res) and dist_res <= 3.0
    model_edge = n_known == 0 or n_beat >= max(1, (n_known + 1) // 2)

    bullets: list[str] = []
    bullets.append(
        f"Primary horizon {primary_horizon}d: P(up) = {p_up * 100:.1f}% "
        f"({'above' if p_up >= 0.5 else 'below'} a coin flip)."
    )
    if n_known:
        bullets.append(
            f"Walk-forward: model beats base rate on {n_beat} of {n_known} scored horizons."
        )
    else:
        bullets.append("Walk-forward baseline comparison not available for these horizons.")

    char_label = str(character.get("label") or "n/a")
    if char_label and char_label != "n/a":
        bullets.append(f"Market character: {char_label}.")

    if ns:
        bullets.append(
            f"Nearest support ≈ {ns['price']:,.2f} ({ns['distance_pct']:+.1f}% from last)."
        )
    if nr:
        bullets.append(
            f"Nearest resistance ≈ {nr['price']:,.2f} ({nr['distance_pct']:+.1f}% from last)."
        )

    # Stance
    if not model_edge and conviction < 0.15:
        action = "WEAK_SIGNAL"
        stance = "Weak — no clear edge"
        summary = (
            "Model does not clearly beat the base rate and P(up) is near 50%. "
            "Treat any directional call as low quality."
        )
    elif not model_edge:
        action = "WEAK_SIGNAL"
        stance = "Weak model edge"
        summary = (
            "P(up) leans one way, but walk-forward scores do not clearly beat a simple base rate. "
            "Prefer waiting or smaller size in research terms."
        )
    elif p_up >= 0.58 and near_support:
        action = "BUY_ZONE"
        stance = "Lean long near support"
        summary = (
            f"P(up) is elevated ({p_up * 100:.0f}%) with model edge, and price is near support. "
            "Data-friendly long area is around support; resistance is a natural take-profit reference."
        )
    elif p_up >= 0.58:
        action = "LEAN_LONG"
        stance = "Lean long"
        summary = (
            f"P(up) is elevated ({p_up * 100:.0f}%) with model edge. "
            "Not necessarily 'buy now' — better entries are closer to support if available."
        )
    elif p_up <= 0.42 and near_resistance:
        action = "AVOID_LONG"
        stance = "Lean short / avoid new longs"
        summary = (
            f"P(up) is low ({p_up * 100:.0f}%) with model edge, and price is near resistance. "
            "Data-friendly reduce/sell reference is resistance; support is a downside risk marker."
        )
    elif p_up <= 0.42:
        action = "LEAN_SHORT"
        stance = "Lean short / defensive"
        summary = (
            f"P(up) is low ({p_up * 100:.0f}%) with model edge. "
            "Favor defense or waiting for a better level rather than chasing."
        )
    else:
        action = "WAIT"
        stance = "Wait"
        summary = (
            "P(up) is close to a coin flip. No strong buy/sell call from the probability model alone."
        )

    buy_zone = None
    sell_zone = None
    if ns and last_price == last_price:
        lo = float(ns["price"])
        hi = float(min(last_price, lo * 1.015)) if lo < last_price else float(last_price)
        buy_zone = {"low": min(lo, hi), "high": max(lo, hi), "label": "Near support"}
    elif p_up >= 0.55 and last_price == last_price:
        buy_zone = {
            "low": float(last_price * 0.98),
            "high": float(last_price),
            "label": "Pullback toward last (no clear support)",
        }

    if nr and last_price == last_price:
        sell_zone = {
            "low": float(last_price),
            "high": float(nr["price"]),
            "label": "Toward resistance",
        }
    elif p_up <= 0.45 and last_price == last_price:
        sell_zone = {
            "low": float(last_price),
            "high": float(last_price * 1.02),
            "label": "Near last (no clear resistance)",
        }

    # confidence 0..1
    conf = float(np.clip(0.35 * conviction + 0.45 * (n_beat / max(n_known, 1)) + 0.2 * (1 if model_edge else 0), 0, 1))

    return {
        "action": action,
        "stance": stance,
        "confidence": conf,
        "buy_zone": buy_zone,
        "sell_zone": sell_zone,
        "summary": summary,
        "bullets": bullets,
        "primary_horizon": primary_horizon,
        "primary_p_up": p_up,
        "beats_on": n_beat,
        "scored_horizons": n_known,
        "disclaimer": "Research signal only — not investment advice.",
    }


def decision_html(d: dict[str, Any], *, last_price: float | None = None) -> str:
    if not d:
        return ""
    action = str(d.get("action") or "NO_SIGNAL")
    color = {
        "BUY_ZONE": "#1b7a4e",
        "LEAN_LONG": "#1b7a4e",
        "AVOID_LONG": "#b42318",
        "LEAN_SHORT": "#b42318",
        "WAIT": "#78716c",
        "WEAK_SIGNAL": "#b45309",
        "NO_SIGNAL": "#78716c",
    }.get(action, "#78716c")
    bg = {
        "BUY_ZONE": "#e6f5ec",
        "LEAN_LONG": "#e6f5ec",
        "AVOID_LONG": "#fdecea",
        "LEAN_SHORT": "#fdecea",
        "WAIT": "#f5f5f4",
        "WEAK_SIGNAL": "#fff7ed",
        "NO_SIGNAL": "#f5f5f4",
    }.get(action, "#f5f5f4")

    buy = d.get("buy_zone") or {}
    sell = d.get("sell_zone") or {}
    buy_txt = "—"
    sell_txt = "—"
    if buy:
        buy_txt = f"{buy.get('low', 0):,.2f} – {buy.get('high', 0):,.2f} ({buy.get('label', '')})"
    if sell:
        sell_txt = f"{sell.get('low', 0):,.2f} – {sell.get('high', 0):,.2f} ({sell.get('label', '')})"

    bullets = "".join(f"<li>{b}</li>" for b in (d.get("bullets") or []))
    conf = float(d.get("confidence") or 0) * 100
    price_bit = f"{last_price:,.2f}" if last_price is not None and last_price == last_price else "—"

    return f"""
<div class="summary" style="background:{bg};border-color:{color}33">
  <div class="kicker" style="font-size:11px;letter-spacing:.14em;text-transform:uppercase;font-weight:600;color:{color}">
    Data conclusion
  </div>
  <h2 style="color:{color}">{d.get('stance', '—')}</h2>
  <p class="lead">{d.get('summary', '')}</p>
  <div class="pills">
    <span class="pill ok">Last {price_bit}</span>
    <span class="pill">Horizon {d.get('primary_horizon', '—')}d</span>
    <span class="pill">P(up) {float(d.get('primary_p_up') or 0)*100:.1f}%</span>
    <span class="pill">Confidence {conf:.0f}%</span>
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:14px">
    <div style="background:#fff;border:1px solid #e7e0d5;border-radius:12px;padding:12px">
      <div style="font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:#78716c;font-weight:600">Buy reference</div>
      <div style="font-weight:650;margin-top:4px;color:#1c1917">{buy_txt}</div>
    </div>
    <div style="background:#fff;border:1px solid #e7e0d5;border-radius:12px;padding:12px">
      <div style="font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:#78716c;font-weight:600">Sell / target reference</div>
      <div style="font-weight:650;margin-top:4px;color:#1c1917">{sell_txt}</div>
    </div>
  </div>
  <ul style="margin:12px 0 0;padding-left:18px;color:#44403c;font-size:13.5px;line-height:1.5">{bullets}</ul>
  <p class="hint" style="margin-top:10px">{d.get('disclaimer', '')}</p>
</div>
"""
