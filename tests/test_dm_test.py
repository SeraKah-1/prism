"""Unit tests for FASE 5: Diebold-Mariano test with Newey-West HAC & regime scoring."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stock_prob.scoring import brier_by_regime, dm_test


def test_dm_test_output_structure():
    rng = np.random.default_rng(42)
    y = rng.integers(0, 2, size=50)
    p_model = rng.uniform(0.4, 0.6, size=50)
    p_base = rng.uniform(0.3, 0.7, size=50)

    res = dm_test(y, p_model, p_base, h=1, min_obs=10)
    assert "dm_stat" in res
    assert "p_value_one_sided" in res
    assert "p_value_two_sided" in res
    assert "n_obs" in res
    assert res["n_obs"] == 50
    assert np.isfinite(res["dm_stat"])


def test_brier_by_regime_grouping():
    df = pd.DataFrame(
        {
            "regime": ["CALM", "CALM", "PANIC", "PANIC", "ELEVATED"],
            "y_true": [1, 0, 1, 0, 1],
            "prob_up_model": [0.6, 0.4, 0.5, 0.5, 0.7],
        }
    )
    res = brier_by_regime(df)
    assert len(res) == 3
    assert set(res["regime"]) == {"CALM", "PANIC", "ELEVATED"}
