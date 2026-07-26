"""Logistic P(up) + Monte Carlo prediction cones. Ticker-agnostic."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class ProbModel:
    pipeline: Pipeline
    feature_cols: list[str]
    horizon: int

    def predict_proba_up(self, X: pd.DataFrame) -> np.ndarray:
        Xs = X[self.feature_cols]
        proba = self.pipeline.predict_proba(Xs)
        # class 1 = up
        classes = list(self.pipeline.named_steps["clf"].classes_)
        if 1.0 in classes or 1 in classes:
            idx = classes.index(1.0) if 1.0 in classes else classes.index(1)
        else:
            idx = int(np.argmax(classes))
        return proba[:, idx]


def fit_logistic(
    X: pd.DataFrame,
    y: pd.Series,
    feature_cols: list[str],
    *,
    horizon: int,
    random_state: int = 42,
) -> ProbModel:
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=500,
                    random_state=random_state,
                    class_weight="balanced",
                ),
            ),
        ]
    )
    y_int = y.astype(int)
    pipe.fit(X[feature_cols], y_int)
    return ProbModel(pipeline=pipe, feature_cols=list(feature_cols), horizon=horizon)


def monte_carlo_cone(
    last_price: float,
    mu_daily: float,
    vol_daily: float,
    horizon: int,
    *,
    n_paths: int = 2000,
    random_state: int = 42,
    quantiles: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9),
) -> dict[str, np.ndarray]:
    """
    GBM-style path simulation. Returns arrays of shape (horizon,) for each quantile
    plus full terminal distribution stats.
    """
    rng = np.random.default_rng(random_state)
    # shocks: (n_paths, horizon)
    z = rng.standard_normal((n_paths, horizon))
    # daily log returns
    rets = (mu_daily - 0.5 * vol_daily**2) + vol_daily * z
    log_paths = np.cumsum(rets, axis=1)
    prices = last_price * np.exp(log_paths)

    q_levels = np.array(quantiles)
    band = np.quantile(prices, q_levels, axis=0)  # (Q, H)
    out: dict[str, np.ndarray] = {"steps": np.arange(1, horizon + 1)}
    for i, q in enumerate(quantiles):
        key = f"p{int(q * 100):02d}"
        out[key] = band[i]
    out["terminal"] = prices[:, -1]
    return out


def cone_table(
    last_date: pd.Timestamp,
    last_price: float,
    mu_daily: float,
    vol_daily: float,
    horizon: int,
    **kwargs,
) -> pd.DataFrame:
    cone = monte_carlo_cone(last_price, mu_daily, vol_daily, horizon, **kwargs)
    # business-day index approx
    dates = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=horizon)
    df = pd.DataFrame({"date": dates[:horizon]})
    for k, v in cone.items():
        if k in ("steps", "terminal"):
            continue
        df[k] = v[: len(df)]
    df["horizon"] = horizon
    return df
