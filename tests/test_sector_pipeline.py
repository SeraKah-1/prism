"""Unit tests for FASE 1: Sector metadata pipeline."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stock_prob.ingest import fetch_sector_meta


def test_fetch_sector_meta_fallback_and_cache(tmp_path):
    df = fetch_sector_meta(["BBCA.JK", "UNTR.JK", "UNKNOWN_XYZ"], root=ROOT)
    assert len(df) >= 3
    assert "symbol" in df.columns and "sector" in df.columns
    mapped = dict(zip(df["symbol"], df["sector"]))
    assert isinstance(mapped.get("BBCA.JK"), str) and len(mapped.get("BBCA.JK")) > 0
    assert isinstance(mapped.get("UNTR.JK"), str) and len(mapped.get("UNTR.JK")) > 0
    assert mapped.get("UNKNOWN_XYZ") == "Unknown"
