"""Unit tests for FASE 7: coverage for cost_bps, twin_drift, and fit_spillover_magnitude."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stock_prob.simulate import simulate_entry
from stock_prob.spillover import fit_spillover_magnitude
from stock_prob.twins import twin_drift


def test_transaction_cost_bps_reduces_profit():
    # Long simulation with 0 cost vs 100 bps (1%) cost
    sim_no_cost = simulate_entry(100.0, mu_daily=0.0005, vol_daily=0.02, horizon=21, n_paths=1000, cost_bps=0.0)
    sim_with_cost = simulate_entry(100.0, mu_daily=0.0005, vol_daily=0.02, horizon=21, n_paths=1000, cost_bps=100.0)

    assert sim_no_cost["ok"] is True
    assert sim_with_cost["ok"] is True
    # P(profit) with transaction friction must be <= P(profit) without cost
    assert sim_with_cost["p_profit"] <= sim_no_cost["p_profit"]


def test_twin_drift_tracking():
    old_df = pd.DataFrame({"ticker": ["BBCA.JK", "BMRI.JK", "TLKM.JK"], "cluster": [0, 0, 1]})
    new_df = pd.DataFrame({"ticker": ["BBCA.JK", "BMRI.JK", "TLKM.JK"], "cluster": [0, 1, 1]})

    drift = twin_drift(old_df, new_df)
    assert len(drift) == 3
    drift_map = dict(zip(drift["ticker"], drift["status"]))
    assert drift_map["BBCA.JK"] == "STABLE"
    assert drift_map["BMRI.JK"] == "DRIFTED"


def test_fit_spillover_magnitude():
    # Mock frame with us_ret_1d and gap
    df = pd.DataFrame(
        {
            "us_ret_1d": [0.01, -0.02, 0.005, -0.015] * 25,
            "gap": [0.005, -0.01, 0.002, -0.008] * 25,
        }
    )
    model = fit_spillover_magnitude(df)
    assert model is not None
