"""Walk-forward evaluation with embargo-safe scoring. Universe-agnostic."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from stock_prob.baselines import base_rate_probs
from stock_prob.features import feature_columns
from stock_prob.labels import apply_embargo, make_supervised
from stock_prob.models import fit_logistic, monte_carlo_cone
from stock_prob.scoring import brier_score, hit_rate


@dataclass
class WalkForwardResult:
    predictions: pd.DataFrame
    metrics: pd.DataFrame
    cone_diag: dict[str, Any]


def walk_forward_equity(
    features: pd.DataFrame,
    close: pd.Series,
    horizons: list[int],
    *,
    min_train_rows: int = 252,
    refit_every: int = 21,
    rolling_window: int = 60,
    mc_paths: int = 1000,
    random_state: int = 42,
    cone_diagnostics: bool = True,
    max_oos_per_horizon: int | None = None,
) -> WalkForwardResult:
    """
    Expanding-window walk-forward for one equity feature frame.

    Training uses all past labeled rows (dense). Evaluation scores only
    embargo-spaced timestamps (gap = horizon) to reduce overlapping-label bias.

    Fast GUI mode: cone_diagnostics=False (skip MC inside loop), larger refit_every,
    max_oos_per_horizon to cap scored points.
    """
    close = close.reindex(features.index).astype(float)
    fcols = feature_columns(features)
    pred_rows: list[dict] = []
    cone_hits: dict[int, list[float]] = {h: [] for h in horizons}
    cone_widths: dict[int, list[float]] = {h: [] for h in horizons}

    empty_preds = pd.DataFrame(
        columns=[
            "date",
            "horizon",
            "prob_up_model",
            "prob_up_base",
            "prob_up_momentum",
            "y_true",
            "fwd_ret",
            "close",
        ]
    )

    for h in horizons:
        # Dense supervised set for training capacity
        X_all, y_all, fr_all = make_supervised(
            features, close, h, feature_cols=fcols, use_embargo=False
        )
        if len(X_all) < min(min_train_rows, 80) + 5:
            continue

        dates = list(X_all.index)
        n = len(dates)
        # Embargo mask on the dense labeled index
        embargo_mask = apply_embargo(X_all.index, h)
        train_floor = min(min_train_rows, max(60, n // 3))

        model = None
        last_fit_pos = -10**9
        p_base = 0.5
        oos_count = 0

        for i in range(train_floor, n):
            if not embargo_mask[i]:
                continue
            if max_oos_per_horizon is not None and oos_count >= max_oos_per_horizon:
                break

            # Refit on all prior dense samples
            if model is None or (i - last_fit_pos) >= max(1, refit_every):
                X_tr = X_all.iloc[:i]
                y_tr = y_all.iloc[:i]
                if y_tr.nunique() < 2 or len(X_tr) < 30:
                    continue
                model = fit_logistic(
                    X_tr, y_tr, fcols, horizon=h, random_state=random_state
                )
                p_base = base_rate_probs(y_tr)
                last_fit_pos = i

            if model is None:
                continue

            dt = dates[i]
            x_row = X_all.iloc[[i]]
            p_model = float(model.predict_proba_up(x_row)[0])
            y_true = float(y_all.iloc[i])
            fwd = float(fr_all.iloc[i])
            mom = float(x_row["mom_5"].iloc[0]) if "mom_5" in x_row.columns else 0.0
            p_mom = 0.55 if mom > 0 else 0.45

            pred_rows.append(
                {
                    "date": dt,
                    "horizon": h,
                    "prob_up_model": p_model,
                    "prob_up_base": p_base,
                    "prob_up_momentum": p_mom,
                    "y_true": y_true,
                    "fwd_ret": fwd,
                    "close": float(close.loc[dt]) if dt in close.index else np.nan,
                }
            )
            oos_count += 1

            # Cone diagnostic (expensive) — skip in fast GUI mode
            if not cone_diagnostics:
                continue
            hist = close.loc[:dt].dropna()
            if len(hist) > 30:
                r = np.log(hist).diff().dropna()
                mu = float(r.tail(60).mean()) if len(r) else 0.0
                vol = float(r.tail(60).std()) if len(r) else 0.0
                vol = vol if np.isfinite(vol) and vol > 0 else 0.01
                mu = mu if np.isfinite(mu) else 0.0
                cone = monte_carlo_cone(
                    float(hist.iloc[-1]),
                    mu,
                    vol,
                    h,
                    n_paths=mc_paths,
                    random_state=random_state + int(h),
                )
                try:
                    pos = close.index.get_loc(dt)
                    if isinstance(pos, slice):
                        pos = pos.start
                    if isinstance(pos, np.ndarray):
                        pos = int(pos.ravel()[0])
                    target_pos = int(pos) + h
                    if target_pos < len(close):
                        actual = float(close.iloc[target_pos])
                        lo, hi = float(cone["p10"][-1]), float(cone["p90"][-1])
                        mid = float(cone["p50"][-1])
                        cone_hits[h].append(1.0 if lo <= actual <= hi else 0.0)
                        cone_widths[h].append(abs(hi - lo) / max(abs(mid), 1e-8))
                except Exception:
                    pass

    preds = pd.DataFrame(pred_rows) if pred_rows else empty_preds.copy()
    metrics_rows = []
    for h in horizons:
        if len(preds) == 0 or "horizon" not in preds.columns:
            continue
        sub = preds[preds["horizon"] == h]
        if len(sub) == 0:
            continue
        metrics_rows.append(
            {
                "horizon": h,
                "n": len(sub),
                "brier_model": brier_score(sub["y_true"], sub["prob_up_model"]),
                "brier_base": brier_score(sub["y_true"], sub["prob_up_base"]),
                "brier_momentum": brier_score(sub["y_true"], sub["prob_up_momentum"]),
                "hit_model": hit_rate(sub["y_true"], sub["prob_up_model"]),
                "hit_base": hit_rate(sub["y_true"], sub["prob_up_base"]),
                "cone_coverage_80": float(np.mean(cone_hits[h])) if cone_hits[h] else float("nan"),
                "cone_sharpness_rel": float(np.mean(cone_widths[h]))
                if cone_widths[h]
                else float("nan"),
            }
        )
    metrics = pd.DataFrame(metrics_rows)
    cone_diag = {"hits": cone_hits, "widths": cone_widths}
    return WalkForwardResult(predictions=preds, metrics=metrics, cone_diag=cone_diag)
