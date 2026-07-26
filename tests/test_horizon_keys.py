"""Horizon key normalization (h5 vs 5 bug)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stock_prob.horizon_keys import normalize_prob_map, parse_horizon_key
from stock_prob.viewmodel import PrismViewModel, build_viewmodel
from stock_prob.report_html import write_prism_report


def test_parse_horizon_key_variants():
    assert parse_horizon_key(5) == 5
    assert parse_horizon_key("5") == 5
    assert parse_horizon_key("h5") == 5
    assert parse_horizon_key("ens_21") == 21
    assert parse_horizon_key("H252") == 252


def test_normalize_strips_h_prefix_and_ens():
    raw = {"h5": 0.6, "21": 0.55, "ens_21": 0.5, "bad": 0.1}
    out = normalize_prob_map(raw)
    assert out["5"] == 0.6
    assert out["21"] == 0.55
    assert "ens_21" not in out


def test_primary_horizon_with_h_keys():
    vm = PrismViewModel(ticker="X", probs={"h5": 0.6, "h21": 0.4})
    # primary_horizon uses parse — but probs may still have h keys until normalized
    # summary_cards must not crash
    cards = vm.summary_cards()
    assert len(cards) == 2
    assert vm.primary_horizon() in (5, 21)


def test_build_viewmodel_h_keys_and_report():
    hist = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=80, freq="B"),
            "close": range(100, 180),
        }
    )
    result = {
        "ticker": "AMMN.JK",
        "live_probs": {"h5": 0.55, "h21": 0.6, "h252": 0.4},
        "history": hist,
        "horizons": [5, 21, 252],
        "metrics": pd.DataFrame(),
    }
    vm = build_viewmodel(result, name="Amman")
    assert vm.has_probs()
    assert "5" in vm.probs or 5 in [c["horizon"] for c in vm.summary_cards()]
    assert vm.primary_horizon() >= 5
    out = ROOT / "exports" / "_test_hkey_report.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_prism_report(vm, out)
    assert out.exists() and out.stat().st_size > 5000
