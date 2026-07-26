"""Unit tests for FASE 6: IDX Market Scanner & bounded Rank Score formula."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stock_prob.scanner import check_fundamental_filter, compute_rank_score, compute_relative_cone_ratio


def test_bounded_rank_score_formula():
    # Normal ratio 1.0, P(up) 0.65 -> 0.65 / (1 + 1.0) = 0.325
    s1 = compute_rank_score(0.65, 1.0)
    assert s1 == pytest.approx(0.325)
    assert s1 > 0.0

    # High uncertainty ratio 2.0, P(up) 0.65 -> 0.65 / (1 + 2.0) = 0.2166...
    s2 = compute_rank_score(0.65, 2.0)
    assert s2 < s1

    # Low uncertainty ratio 0.5, P(up) 0.65 -> 0.65 / (1 + 0.5) = 0.433...
    s3 = compute_rank_score(0.65, 0.5)
    assert s3 > s1


def test_fundamental_none_policy():
    # Check that unknown/none fundamental returns fundamental_unknown instead of crashing
    res = check_fundamental_filter("NON_EXISTENT_TICKER_123")
    assert res["status"] in ("fundamental_unknown", "ok", "failed")
    assert "notes" in res


def test_relative_cone_ratio_cold_start():
    # Insufficient bars < 60 returns None
    cone_short = pd.DataFrame({"p10": [90.0] * 10, "p90": [110.0] * 10})
    ratio = compute_relative_cone_ratio(cone_short, min_bars=60)
    assert ratio is None
