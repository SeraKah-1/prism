"""Dynamic universe + science tests against shipped modules."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stock_prob.config import RunConfig, load_run_config, universe_from_symbols
from stock_prob.features import build_feature_frame, feature_columns
from stock_prob.ingest import fetch_universe, load_cached
from stock_prob.labels import apply_embargo, direction_label, forward_return, make_supervised
from stock_prob.pipeline import run_pipeline
from stock_prob.scoring import brier_score


def test_universe_not_in_feature_signature():
    """Core feature builder accepts series only — no ticker constants required."""
    idx = pd.bdate_range("2020-01-01", periods=200)
    rng = np.random.default_rng(0)
    eq = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx)))), index=idx)
    dom = pd.Series(1000 * np.exp(np.cumsum(rng.normal(0, 0.008, len(idx)))), index=idx)
    feats = build_feature_frame(eq, domestic_close=dom, window=30)
    assert "beta_dom" in feats.columns
    assert feats["mom_5"].isna().sum() >= 1  # first rows NaN = PIT lag, not future


def test_no_lookahead_labels():
    """Label at t uses close[t+h]/close[t]; feature row at t must not include that return."""
    idx = pd.bdate_range("2020-01-01", periods=100)
    close = pd.Series(np.linspace(100, 150, len(idx)), index=idx)
    h = 5
    fr = forward_return(close, h)
    # forward return at t equals realized path — known only after t+h
    t = idx[10]
    assert fr.loc[t] == pytest.approx(close.loc[idx[15]] / close.loc[t] - 1.0)
    # last h labels are NaN (cannot look past end)
    assert fr.tail(h).isna().all()


def test_embargo_spacing():
    idx = pd.bdate_range("2020-01-01", periods=100)
    m = apply_embargo(idx, horizon=5)
    selected = np.where(m)[0]
    if len(selected) > 1:
        gaps = np.diff(selected)
        assert gaps.min() >= 5


def test_make_supervised_embargo_thins():
    idx = pd.bdate_range("2020-01-01", periods=300)
    rng = np.random.default_rng(1)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx)))), index=idx)
    feats = build_feature_frame(close, window=40)
    fcols = feature_columns(feats)
    X1, y1, _ = make_supervised(feats, close, 10, feature_cols=fcols, use_embargo=False)
    X2, y2, _ = make_supervised(feats, close, 10, feature_cols=fcols, use_embargo=True)
    assert len(X2) < len(X1)
    assert len(X2) > 0


def test_two_universes_config_only(tmp_path):
    """Two different ticker sets from config objects — no code edit to symbol lists in core."""
    root = tmp_path / "lab"
    # Use cache if available from project; still fetch_universe by symbol list only
    uni_a = universe_from_symbols(
        ["BBCA.JK"], domestic_index="^JKSE", us_index="^GSPC", macro="^VIX"
    )
    uni_b = universe_from_symbols(
        ["AAPL"], domestic_index="^GSPC", us_index="^GSPC", macro="^VIX"
    )
    assert uni_a.all_symbols() != uni_b.all_symbols()
    assert "BBCA.JK" in uni_a.all_symbols() and "AAPL" not in uni_a.equities
    assert "AAPL" in uni_b.all_symbols()

    # fetch both universes (cache-friendly)
    fa = fetch_universe(uni_a.all_symbols(), period="5y", use_cache=True)
    fb = fetch_universe(uni_b.all_symbols(), period="5y", use_cache=True)
    assert "BBCA.JK" in fa and len(fa["BBCA.JK"]) > 100
    assert "AAPL" in fb and len(fb["AAPL"]) > 100

    # build features for each without shared hardcoded path
    from stock_prob.ingest import align_close_panel

    pa = align_close_panel(fa)
    pb = align_close_panel(fb)
    feata = build_feature_frame(
        pa["BBCA.JK"].dropna(),
        domestic_close=pa.get("^JKSE"),
        us_close=pa.get("^GSPC"),
        macro_close=pa.get("^VIX"),
    )
    featb = build_feature_frame(
        pb["AAPL"].dropna(),
        domestic_close=pb.get("^GSPC"),
        us_close=pb.get("^GSPC"),
        macro_close=pb.get("^VIX"),
    )
    assert len(feata.dropna()) > 50
    assert len(featb.dropna()) > 50


def test_load_example_configs():
    cfg_a = load_run_config(ROOT / "configs" / "universe_idx_example.yaml")
    cfg_b = load_run_config(ROOT / "configs" / "universe_us_example.yaml")
    assert cfg_a.universe.equities != cfg_b.universe.equities
    # ensure configs differ and both non-empty
    assert cfg_a.universe.equities and cfg_b.universe.equities


def test_brier_perfect_and_base():
    y = np.array([1.0, 0.0, 1.0, 0.0])
    assert brier_score(y, y) == pytest.approx(0.0)
    assert brier_score(y, np.full(4, 0.5)) == pytest.approx(0.25)


def test_no_hardcoded_pilot_only_in_core_modules():
    """Core algorithm modules must not assign fixed pilot tickers as defaults driving logic."""
    core = [
        ROOT / "src/stock_prob/features.py",
        ROOT / "src/stock_prob/models.py",
        ROOT / "src/stock_prob/backtest.py",
        ROOT / "src/stock_prob/labels.py",
        ROOT / "src/stock_prob/scoring.py",
    ]
    banned = ["BBCA.JK", "AAPL", "^JKSE", "^GSPC", "^VIX"]
    for path in core:
        text = path.read_text()
        for b in banned:
            assert b not in text, f"{path.name} contains hardcoded {b}"
