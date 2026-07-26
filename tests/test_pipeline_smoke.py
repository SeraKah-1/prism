"""Integration smoke: pipeline on config-driven universes (cache-friendly)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stock_prob.config import load_run_config
from stock_prob.pipeline import run_pipeline


def _assert_result(res: dict, horizons: list[int]) -> None:
    assert res["run_id"]
    assert res["live_probs"], "expected live P(up) per horizon"
    # Gating: finite P(up) for EVERY configured horizon (not just ≥1)
    for h in horizons:
        key = str(h)
        assert key in res["live_probs"], f"missing live prob for horizon {h}"
        v = res["live_probs"][key]
        assert v == v and np.isfinite(v), f"P(up) for horizon {h} is not finite: {v}"
        assert 0.0 <= float(v) <= 1.0, f"P(up) out of range for horizon {h}: {v}"
    assert res["metrics"] is not None and len(res["metrics"]) >= 1
    m = res["metrics"]
    assert "brier_model" in m.columns and "brier_base" in m.columns
    assert np.isfinite(m["brier_model"].iloc[0])
    assert res["report_html"] and Path(res["report_html"]).exists()
    art = Path(res["art_dir"])
    assert (art / "metrics.json").exists()
    assert (art / "predictions.parquet").exists() or (art / "predictions.csv").exists()


def test_pipeline_idx_config(tmp_path):
    cfg = load_run_config(ROOT / "configs" / "universe_idx_example.yaml")
    cfg.mc_paths = 400
    cfg.min_train_rows = 200
    # write under project root so cache is reused; artifacts still unique run_id
    res = run_pipeline(cfg, equity="BBCA.JK", root=ROOT, use_cache=True)
    _assert_result(res, list(cfg.horizons))


def test_pipeline_us_config():
    cfg = load_run_config(ROOT / "configs" / "universe_us_example.yaml")
    cfg.mc_paths = 400
    cfg.min_train_rows = 200
    res = run_pipeline(cfg, equity="AAPL", root=ROOT, use_cache=True)
    _assert_result(res, list(cfg.horizons))


def test_live_probs_finite_all_horizons_252():
    """Regression: dense live fit must produce finite P(up) including 252d."""
    cfg = load_run_config(ROOT / "configs" / "universe_idx_example.yaml")
    cfg.horizons = [5, 21, 252]
    cfg.mc_paths = 300
    cfg.min_train_rows = 200
    res = run_pipeline(cfg, equity="BBCA.JK", root=ROOT, use_cache=True)
    for h in (5, 21, 252):
        v = res["live_probs"][str(h)]
        assert np.isfinite(v) and 0.0 <= float(v) <= 1.0
