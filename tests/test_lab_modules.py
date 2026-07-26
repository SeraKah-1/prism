"""Tests for GARCH/regime/tournament/twins/honesty/surface (shipped modules)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stock_prob.garch import fit_garch_vol
from stock_prob.honesty import honesty_frame, honesty_score
from stock_prob.regime import classify_regime, latest_regime
from stock_prob.surface import build_probability_surface
from stock_prob.tournament import inverse_brier_weights, blend_probs, tournament_table, score_models_on_predictions
from stock_prob.twins import cluster_twins, form_feature_vector, twin_pairs


def test_fit_garch_or_fallback():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0, 0.01, 400))
    out = fit_garch_vol(r)
    assert out["vol"] > 0
    assert out["method"] in ("garch11", "sample", "sample_fallback")


def test_regime_labels():
    idx = pd.bdate_range("2020-01-01", periods=200)
    rng = np.random.default_rng(1)
    px = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx)))), index=idx)
    us = pd.Series(1000 * np.exp(np.cumsum(rng.normal(0, 0.008, len(idx)))), index=idx)
    reg = classify_regime(px, us_close=us)
    assert "regime" in reg.columns
    assert latest_regime(reg) in ("CALM", "ELEVATED", "PANIC", "UNKNOWN")


def test_tournament_weights():
    w = inverse_brier_weights({"model": 0.2, "base": 0.25, "momentum": 0.3})
    assert abs(sum(w.values()) - 1.0) < 1e-6
    p = blend_probs({"model": 0.6, "base": 0.55, "momentum": 0.5}, w)
    assert 0.5 <= p <= 0.6


def test_twins_cluster():
    rows = []
    rng = np.random.default_rng(2)
    for i, name in enumerate(["A", "B", "C", "D", "E", "F"]):
        rows.append(
            {
                "ticker": name,
                "vol_21": 0.01 + 0.01 * (i % 2),
                "vol_60": 0.015,
                "mom_21": rng.normal(),
                "beta_dom": 0.8 + 0.2 * (i % 2),
                "corr_dom": 0.5,
                "beta_us": 0.4,
                "corr_us": 0.3 + 0.2 * (i % 2),
            }
        )
    cl = cluster_twins(rows, n_clusters=2)
    assert "cluster" in cl.columns
    pairs = twin_pairs(cl)
    assert len(pairs) >= 1


def test_honesty_score_runs():
    rng = np.random.default_rng(3)
    n = 80
    p = rng.uniform(0.2, 0.8, n)
    y = (rng.random(n) < p).astype(float)
    df = pd.DataFrame({"prob_up_model": p, "y_true": y, "horizon": 5})
    hf = honesty_frame(df)
    sc = honesty_score(hf)
    assert "honesty_skill" in sc
    assert sc["n"] == n


def test_surface():
    s = build_probability_surface({"5": 0.6, "21": 0.48, "252": 0.55})
    assert len(s) == 3
    assert set(s["horizon"]) == {5, 21, 252}


def test_score_models_on_predictions():
    df = pd.DataFrame(
        {
            "horizon": [5, 5, 5, 5],
            "y_true": [1, 0, 1, 0],
            "prob_up_model": [0.7, 0.4, 0.6, 0.3],
            "prob_up_base": [0.55, 0.55, 0.55, 0.55],
            "prob_up_momentum": [0.55, 0.45, 0.55, 0.45],
        }
    )
    sc = score_models_on_predictions(df)
    assert set(sc["model"]) >= {"model", "base", "momentum"}
    board = tournament_table(sc)
    assert board.iloc[0]["rank"] == 1
