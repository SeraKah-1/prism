"""Unit tests for FASE 4: resolve_ledger self-scoring loop and atomic writes."""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stock_prob.ledger import append_ledger, resolve_ledger


def test_empty_ledger_graceful(tmp_path):
    resolved, skipped = resolve_ledger(root=tmp_path)
    assert resolved == 0
    assert skipped == []


def test_idempotency_already_resolved(tmp_path):
    rows = pd.DataFrame(
        [
            {
                "ticker": "BBCA.JK",
                "predicted_at": "2024-01-01T00:00:00Z",
                "horizon_days": 5,
                "prob_up": 0.60,
                "actual_return": 0.02,
                "actual_up": 1.0,
                "brier": 0.16,
                "resolved_at": "2024-01-10T00:00:00Z",
            }
        ]
    )
    append_ledger(rows, root=tmp_path)
    resolved, skipped = resolve_ledger(root=tmp_path)
    assert resolved == 0


def test_resolve_matured_null_rows(tmp_path):
    # Mock row with NULL actual_return predicted in the past
    rows = pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "predicted_at": "2023-01-01T00:00:00Z",
                "horizon_days": 5,
                "prob_up": 0.70,
                "actual_return": float("nan"),
                "actual_up": float("nan"),
                "brier": float("nan"),
                "resolved_at": None,
            }
        ]
    )
    append_ledger(rows, root=tmp_path)
    # Price cache mock directory structure
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache_df = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=10),
            "close": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0],
            "adj_close": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0],
            "ticker": "AAPL",
        }
    )
    cache_df.to_parquet(raw_dir / "AAPL.parquet", index=False)

    resolved, skipped = resolve_ledger(root=tmp_path)
    assert resolved == 1
    # Check resolved parquet contents
    res_df = pd.read_parquet(tmp_path / "predictions" / "ledger.parquet")
    assert not res_df["actual_return"].isna().iloc[0]
    assert res_df["actual_up"].iloc[0] == 1.0
    assert float(res_df["brier"].iloc[0]) == pytest.approx((0.70 - 1.0) ** 2)


def test_missing_actual_price_delisted(tmp_path):
    rows = pd.DataFrame(
        [
            {
                "ticker": "NON_EXISTENT_DELISTED_XYZ",
                "predicted_at": "2020-01-01T00:00:00Z",
                "horizon_days": 5,
                "prob_up": 0.50,
                "actual_return": float("nan"),
                "actual_up": float("nan"),
                "brier": float("nan"),
            }
        ]
    )
    append_ledger(rows, root=tmp_path)
    resolved, skipped = resolve_ledger(root=tmp_path)
    assert resolved == 0
