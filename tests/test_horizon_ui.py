"""UI horizon parsing + All-time-horizons expansion (chart switch bug)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stock_prob.ui_gradio import (
    ALL_HORIZON_DAYS,
    ALL_HORIZONS_LABEL,
    _parse_horizon_label,
    _render_fan_from_session,
    _resolve_horizon_days,
    on_horizons_toggle,
)


def test_parse_chart_labels_not_all_digits():
    """Regression: '5d · ~1 week' must be 5, not 51 (concatenated digits)."""
    cases = {
        "5d · ~1 week": 5,
        "21d · ~1 month": 21,
        "63d · ~1 quarter": 63,
        "126d · ~6 months": 126,
        "252d · ~1 year": 252,
        "5 days (≈ 1 week)": 5,
        "21": 21,
        "h252": 252,
        "—": None,
        None: None,
    }
    for lab, expect in cases.items():
        assert _parse_horizon_label(lab) == expect, lab


def test_parse_with_default():
    assert _parse_horizon_label("—", default=21) == 21
    assert _parse_horizon_label(None, default=5) == 5


def test_resolve_all_time_horizons():
    assert _resolve_horizon_days([ALL_HORIZONS_LABEL]) == ALL_HORIZON_DAYS
    assert _resolve_horizon_days([ALL_HORIZONS_LABEL, "5 days (≈ 1 week)"]) == ALL_HORIZON_DAYS
    partial = _resolve_horizon_days(["5 days (≈ 1 week)", "21 days (≈ 1 month)"])
    assert partial == [5, 21]


def test_horizons_toggle_select_all():
    box, new_val = on_horizons_toggle(
        [ALL_HORIZONS_LABEL],
        prev=["5 days (≈ 1 week)"],
    )
    assert ALL_HORIZONS_LABEL in new_val
    assert "252 days (≈ 1 year)" in new_val
    assert len([x for x in new_val if x != ALL_HORIZONS_LABEL]) == len(ALL_HORIZON_LABELS)


def test_horizons_toggle_uncheck_one_drops_all():
    full = [ALL_HORIZONS_LABEL] + list(ALL_HORIZON_LABELS)
    without_5 = [ALL_HORIZONS_LABEL] + list(ALL_HORIZON_LABELS)[1:]
    _, new_val = on_horizons_toggle(without_5, prev=full)
    assert ALL_HORIZONS_LABEL not in new_val
    assert "5 days (≈ 1 week)" not in new_val
    assert "21 days (≈ 1 month)" in new_val


def test_fan_switch_changes_horizon_marker():
    """Switching radio labels must re-render with the correct data-horizon."""
    hist = pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-01", periods=100),
            "close": range(100, 200),
        }
    )
    cones = {}
    for h in (5, 21, 252):
        cones[h] = pd.DataFrame(
            {
                "date": pd.bdate_range("2024-05-20", periods=h),
                "p10": [190 + i * 0.1 for i in range(h)],
                "p25": [192 + i * 0.1 for i in range(h)],
                "p50": [195 + i * 0.15 for i in range(h)],
                "p75": [198 + i * 0.2 for i in range(h)],
                "p90": [200 + i * 0.25 for i in range(h)],
            }
        )
    session = {
        "ticker": "TEST",
        "history": hist.assign(date=hist["date"].astype(str)).to_dict(orient="records"),
        "cones": {
            str(k): v.assign(date=v["date"].astype(str)).to_dict(orient="records")
            for k, v in cones.items()
        },
        "horizons": [5, 21, 252],
        "supports": [],
        "resistances": [],
    }
    html5 = _render_fan_from_session(session, "5d · ~1 week")
    html21 = _render_fan_from_session(session, "21d · ~1 month")
    html252 = _render_fan_from_session(session, "252d · ~1 year")
    assert 'data-horizon="5"' in html5
    assert 'data-horizon="21"' in html21
    assert 'data-horizon="252"' in html252
    assert "5d ahead" in html5
    assert "21d ahead" in html21
    assert "252d ahead" in html252
    # Must not collapse distinct labels to the same cone (old digit-concat bug)
    assert html5 != html21
    assert html21 != html252
