"""Fan chart + session horizon switch smoke tests."""
from __future__ import annotations

import pandas as pd

from stock_prob.viewmodel import PrismViewModel, session_to_frames
from stock_prob.viz import build_fan_figure, fig_to_html


def test_fan_has_svg_and_labels():
    hist = pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-01", periods=120),
            "close": range(100, 220),
        }
    )
    cone = pd.DataFrame(
        {
            "date": pd.bdate_range("2024-06-18", periods=21),
            "p10": [210 + i * 0.2 for i in range(21)],
            "p25": [215 + i * 0.3 for i in range(21)],
            "p50": [220 + i * 0.4 for i in range(21)],
            "p75": [225 + i * 0.5 for i in range(21)],
            "p90": [230 + i * 0.6 for i in range(21)],
        }
    )
    fig = build_fan_figure(
        hist,
        cone,
        title="test",
        supports=[{"price": 150.0}],
        resistances=[{"price": 250.0}],
        horizon=21,
    )
    html = fig_to_html(fig)
    assert "<svg" in html.lower()
    assert "chart-svg" in html


def test_session_roundtrip():
    hist = pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-01", periods=50),
            "close": range(50, 100),
        }
    )
    cone = pd.DataFrame(
        {
            "date": pd.bdate_range("2024-03-12", periods=5),
            "p10": [90, 91, 92, 93, 94],
            "p50": [95, 96, 97, 98, 99],
            "p90": [100, 101, 102, 103, 104],
        }
    )
    vm = PrismViewModel(
        ticker="TEST",
        history=hist,
        cones={5: cone},
        probs={"5": 0.55},
        last_price=99.0,
        horizons=[5],
        mu=0.0,
        vol=0.02,
    )
    sess = vm.to_session()
    h2, c2 = session_to_frames(sess)
    assert len(h2) == 50
    assert 5 in c2
    assert len(c2[5]) == 5
