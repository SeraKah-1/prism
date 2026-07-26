"""Tests for dynamic ticker search / resolve / validate (not hardcoded universes)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stock_prob.tickers import (
    parse_symbol_from_label,
    resolve_ticker,
    search_labels,
    search_tickers,
    validate_symbol,
)


def test_search_tickers_live_aapl():
    hits = search_tickers("AAPL", max_results=8)
    assert hits, "search_tickers('AAPL') returned no hits — fetch/search broken"
    symbols = [h.symbol.upper() for h in hits]
    assert any(s == "AAPL" or s.startswith("AAPL") for s in symbols)


def test_search_tickers_live_bbca_jk():
    hits = search_tickers("BBCA.JK", max_results=5)
    assert hits, "search BBCA.JK failed"
    assert any(h.symbol.upper() == "BBCA.JK" for h in hits)


def test_search_by_company_name_dynamic():
    hits = search_tickers("bank central asia", max_results=8)
    assert hits, "name search returned empty"
    assert any("BBCA" in h.symbol.upper() for h in hits)


def test_search_labels_format():
    labels = search_labels("MSFT", max_results=5)
    assert labels
    assert "MSFT" in labels[0].upper() or "msft" in labels[0].lower()
    # label should be parseable
    sym = parse_symbol_from_label(labels[0])
    assert sym


def test_parse_symbol_from_label():
    assert parse_symbol_from_label("BBCA.JK — Bank Central Asia") == "BBCA.JK"
    assert parse_symbol_from_label("AAPL") == "AAPL"


def test_validate_symbol_fetch_works():
    v = validate_symbol("AAPL", period="5d")
    assert v["ok"] is True, v
    assert v["n_bars"] >= 1
    assert v["last_close"] == v["last_close"]


def test_validate_symbol_bad():
    v = validate_symbol("THISISNOTAREALTICKERXYZ123", period="5d")
    assert v["ok"] is False


def test_resolve_ticker_from_label_and_bare():
    r = resolve_ticker("AAPL — Apple Inc.")
    assert r["ok"] is True
    assert r["symbol"].upper().startswith("AAPL")

    r2 = resolve_ticker("BBCA.JK")
    assert r2["ok"] is True
    assert r2["symbol"].upper() == "BBCA.JK"


def test_search_not_using_hardcoded_only_list():
    """Core search module must not define a fixed pilot universe as the result source."""
    src = (ROOT / "src/stock_prob/tickers.py").read_text()
    assert "Search(" in src
    assert "equities = [" not in src
    assert "HARDCODED_TICKERS" not in src
    assert "PILOT_UNIVERSE" not in src
    # must not return a fixed list literal of equities as primary API
    assert "return [\"AAPL\"" not in src.replace(" ", "")
