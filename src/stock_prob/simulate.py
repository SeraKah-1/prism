"""Entry simulation from Monte Carlo paths + directional probability."""
from __future__ import annotations

from typing import Any

import numpy as np

from stock_prob.models import monte_carlo_cone


def simulate_entry(
    entry_price: float,
    *,
    mu_daily: float,
    vol_daily: float,
    horizon: int,
    n_paths: int = 2000,
    random_state: int = 42,
    p_up: float | None = None,
    side: str = "long",
) -> dict[str, Any]:
    """
    Simulate holding from `entry_price` for `horizon` trading days under GBM.

    Returns terminal distribution stats and P&L for long (default) or short.
    Also reports model P(up) if provided for comparison with MC P(terminal > entry).
    """
    entry = float(entry_price)
    h = max(1, int(horizon))
    if not np.isfinite(entry) or entry <= 0:
        return {"ok": False, "error": "Entry price must be a positive number."}
    if not np.isfinite(mu_daily):
        mu_daily = 0.0
    if not np.isfinite(vol_daily) or vol_daily <= 0:
        vol_daily = 0.02

    cone = monte_carlo_cone(
        entry,
        float(mu_daily),
        float(vol_daily),
        h,
        n_paths=max(200, int(n_paths)),
        random_state=int(random_state),
    )
    terminal = np.asarray(cone["terminal"], dtype=float)
    terminal = terminal[np.isfinite(terminal)]
    if len(terminal) == 0:
        return {"ok": False, "error": "Simulation produced no paths."}

    side = (side or "long").lower()
    if side == "short":
        rets = entry / terminal - 1.0
        profit_mask = terminal < entry
    else:
        rets = terminal / entry - 1.0
        profit_mask = terminal > entry
        side = "long"

    def q(a: float) -> float:
        return float(np.quantile(terminal, a))

    def rq(a: float) -> float:
        return float(np.quantile(rets, a))

    p_profit = float(np.mean(profit_mask))
    p_mc_up = float(np.mean(terminal > entry))
    median_exit = float(np.median(terminal))
    mean_exit = float(np.mean(terminal))

    out: dict[str, Any] = {
        "ok": True,
        "side": side,
        "entry": entry,
        "horizon": h,
        "n_paths": int(len(terminal)),
        "mu_daily": float(mu_daily),
        "vol_daily": float(vol_daily),
        "median_exit": median_exit,
        "mean_exit": mean_exit,
        "p10_exit": q(0.10),
        "p25_exit": q(0.25),
        "p50_exit": q(0.50),
        "p75_exit": q(0.75),
        "p90_exit": q(0.90),
        "median_return_pct": rq(0.50) * 100.0,
        "p10_return_pct": rq(0.10) * 100.0,
        "p90_return_pct": rq(0.90) * 100.0,
        "mean_return_pct": float(np.mean(rets)) * 100.0,
        "p_profit": p_profit,
        "p_mc_up": p_mc_up,
        "model_p_up": float(p_up) if p_up is not None and np.isfinite(p_up) else None,
        "expected_pnl_per_unit": float(np.mean(rets)) * entry if side == "long" else float(np.mean(rets)) * entry,
        "method": "GBM Monte Carlo (daily mu/vol from recent returns or GARCH when available)",
    }
    return out


def simulate_html(sim: dict[str, Any], *, ticker: str = "") -> str:
    if not sim or not sim.get("ok"):
        err = (sim or {}).get("error") or "Simulation failed."
        return f'<div class="summary"><h2>Simulation</h2><p class="lead">{err}</p></div>'

    p_up_model = sim.get("model_p_up")
    model_bit = (
        f"{p_up_model * 100:.1f}%"
        if p_up_model is not None
        else "n/a"
    )
    title = f"Entry simulation · {ticker}" if ticker else "Entry simulation"
    return f"""
<div class="summary">
  <div class="kicker" style="font-size:11px;letter-spacing:.14em;text-transform:uppercase;font-weight:600;color:#0f3d3e">
    Monte Carlo entry sim
  </div>
  <h2>{title}</h2>
  <p class="lead">
    {sim['side'].title()} from <b>{sim['entry']:,.2f}</b> for <b>{sim['horizon']} trading days</b>
    · {sim['n_paths']:,} paths · {sim.get('method', 'MC')}
  </p>
  <div class="pills">
    <span class="pill ok">P(profit) {sim['p_profit']*100:.1f}%</span>
    <span class="pill">MC P(up) {sim['p_mc_up']*100:.1f}%</span>
    <span class="pill">Model P(up) {model_bit}</span>
    <span class="pill">Median exit {sim['median_exit']:,.2f}</span>
  </div>
  <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:14px">
    <div style="background:#fff;border:1px solid #e7e0d5;border-radius:12px;padding:12px">
      <div style="font-size:11px;color:#78716c;font-weight:600">Downside (p10)</div>
      <div style="font-weight:700;margin-top:4px">{sim['p10_exit']:,.2f}</div>
      <div style="font-size:12px;color:#b42318">{sim['p10_return_pct']:+.1f}%</div>
    </div>
    <div style="background:#fff;border:1px solid #e7e0d5;border-radius:12px;padding:12px">
      <div style="font-size:11px;color:#78716c;font-weight:600">Median (p50)</div>
      <div style="font-weight:700;margin-top:4px">{sim['p50_exit']:,.2f}</div>
      <div style="font-size:12px;color:#44403c">{sim['median_return_pct']:+.1f}%</div>
    </div>
    <div style="background:#fff;border:1px solid #e7e0d5;border-radius:12px;padding:12px">
      <div style="font-size:11px;color:#78716c;font-weight:600">Upside (p90)</div>
      <div style="font-weight:700;margin-top:4px">{sim['p90_exit']:,.2f}</div>
      <div style="font-size:12px;color:#1b7a4e">{sim['p90_return_pct']:+.1f}%</div>
    </div>
  </div>
  <p class="hint" style="margin-top:10px">
    Mean path return ≈ {sim['mean_return_pct']:+.2f}%. MC uses recent drift/vol;
    logistic P(up) is a separate directional model. Both can disagree.
  </p>
</div>
"""
