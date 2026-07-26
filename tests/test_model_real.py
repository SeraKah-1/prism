"""Prove the stack is real math, not a print generator."""
from __future__ import annotations

import numpy as np
import pandas as pd

from stock_prob.decision import build_decision
from stock_prob.labels import make_supervised
from stock_prob.models import fit_logistic, monte_carlo_cone
from stock_prob.simulate import simulate_entry
from stock_prob.structure import market_character, support_resistance


def _synthetic_close(n: int = 400, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0004, 0.015, size=n)
    # inject mild momentum structure
    for i in range(1, n):
        r[i] += 0.08 * r[i - 1]
    px = 100 * np.exp(np.cumsum(r))
    idx = pd.bdate_range("2022-01-03", periods=n)
    return pd.Series(px, index=idx, name="close")


def test_logistic_fit_is_real_not_constant():
    close = _synthetic_close()
    feats = pd.DataFrame(
        {
            "mom_5": close.pct_change(5),
            "mom_21": close.pct_change(21),
            "vol_21": np.log(close).diff().rolling(21).std(),
            "ret_1d": np.log(close).diff(),
        },
        index=close.index,
    )
    X, y, _ = make_supervised(feats, close, 5, feature_cols=list(feats.columns), use_embargo=False)
    assert len(X) > 50
    assert y.nunique() >= 2
    model = fit_logistic(X, y, list(X.columns), horizon=5, random_state=7)
    p = model.predict_proba_up(X.tail(30))
    assert len(p) == 30
    assert np.all((p > 0) & (p < 1))
    # not a single hard-coded constant for every row
    assert float(np.std(p)) > 1e-4


def test_monte_carlo_paths_spread():
    cone = monte_carlo_cone(100.0, 0.0002, 0.02, 21, n_paths=800, random_state=1)
    term = cone["terminal"]
    assert len(term) == 800
    assert float(np.std(term)) > 0.5
    assert cone["p10"][-1] < cone["p50"][-1] < cone["p90"][-1]


def test_simulate_entry_long_stats():
    sim = simulate_entry(100.0, mu_daily=0.0005, vol_daily=0.02, horizon=21, n_paths=1000, p_up=0.6)
    assert sim["ok"] is True
    assert sim["p10_exit"] < sim["p90_exit"]
    assert 0.0 <= sim["p_profit"] <= 1.0
    assert sim["n_paths"] == 1000


def test_support_resistance_and_character():
    close = _synthetic_close(500, seed=3)
    levels = support_resistance(close)
    assert "supports" in levels and "resistances" in levels
    ch = market_character(close)
    assert ch["label"]
    assert ch["trend"] in ("up", "down", "sideways", "unknown")


def test_decision_emits_action():
    d = build_decision(
        probs={"5": 0.62, "21": 0.58},
        beat_baseline={"5": True, "21": True},
        last_price=100.0,
        supports=[{"price": 97.0, "distance_pct": -3.0, "touches": 3, "strength": 0.7}],
        resistances=[{"price": 108.0, "distance_pct": 8.0, "touches": 2, "strength": 0.5}],
        character={"label": "quiet uptrend"},
        primary_horizon=5,
    )
    assert d["action"] in {
        "BUY_ZONE",
        "LEAN_LONG",
        "WAIT",
        "WEAK_SIGNAL",
        "AVOID_LONG",
        "LEAN_SHORT",
        "NO_SIGNAL",
    }
    assert d["primary_p_up"] == 0.62
    assert d["buy_zone"] is not None
