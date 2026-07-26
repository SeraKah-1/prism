"""Human-readable description of what Prism actually computes (not a print stub)."""
from __future__ import annotations

from typing import Any


MODEL_MATH = {
    "direction": (
        "Logistic regression on point-in-time features (momentum, realized vol, "
        "betas/corrs to index and macro when available). Label = 1 if forward return "
        "over H trading days is positive. Features at t use only data ≤ t."
    ),
    "cone": (
        "Monte Carlo geometric Brownian motion (GBM) paths: "
        "r_t = (μ − ½σ²) + σ·Z. Bands are path quantiles (p10…p90). "
        "Advanced mode can set σ from GARCH(1,1) conditional vol."
    ),
    "validation": (
        "Walk-forward: expanding train, embargo-spaced out-of-sample scores. "
        "Brier score of the model vs base-rate and momentum baselines."
    ),
    "structure": (
        "Support/resistance from swing highs/lows clustered into zones. "
        "Market character = trend (multi-horizon momentum) × volatility regime."
    ),
}


def model_info_html(meta: dict[str, Any] | None = None) -> str:
    meta = meta or {}
    bits = []
    if meta.get("n_bars"):
        bits.append(f"{meta['n_bars']} price bars")
    if meta.get("mu") is not None:
        bits.append(f"μ daily={float(meta['mu']):.5f}")
    if meta.get("vol") is not None:
        bits.append(f"σ daily={float(meta['vol']):.5f}")
    if meta.get("cone_method"):
        bits.append(f"cone={meta['cone_method']}")
    if meta.get("feature_cols"):
        n = len(meta["feature_cols"]) if isinstance(meta["feature_cols"], list) else meta["feature_cols"]
        bits.append(f"{n} features")
    chip = " · ".join(bits) if bits else "Run analysis to fill live parameters."

    return f"""
<div class="card" style="margin-top:0">
  <div class="section-title">How the model works</div>
  <p class="hint" style="margin-bottom:10px">
    This is a real fit + simulation stack, not random text. Live run: <b>{chip}</b>
  </p>
  <ul style="margin:0;padding-left:18px;color:#44403c;font-size:13.5px;line-height:1.55">
    <li><b>P(up)</b> — {MODEL_MATH['direction']}</li>
    <li><b>Price range (cone)</b> — {MODEL_MATH['cone']}</li>
    <li><b>Honesty check</b> — {MODEL_MATH['validation']}</li>
    <li><b>Levels & character</b> — {MODEL_MATH['structure']}</li>
  </ul>
  <p class="hint" style="margin-top:10px;margin-bottom:0">
    If walk-forward Brier is worse than the base rate, treat P(up) as weak even if it is not 50%.
  </p>
</div>
"""
