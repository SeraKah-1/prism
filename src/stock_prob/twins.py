"""Twin stocks: cluster by statistical form (beta/vol/corr), not sector content."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def form_feature_vector(
    equity_close: pd.Series,
    domestic_close: pd.Series | None,
    us_close: pd.Series | None,
    *,
    window: int = 60,
) -> dict[str, float]:
    eq = equity_close.astype(float).dropna().sort_index()
    r = np.log(eq).diff().dropna()
    out = {
        "vol_21": float(r.tail(21).std()) if len(r) >= 21 else float(r.std()),
        "vol_60": float(r.tail(window).std()) if len(r) >= 10 else float(r.std()),
        "mom_21": float(eq.pct_change(21).iloc[-1]) if len(eq) > 21 else 0.0,
    }
    if domestic_close is not None:
        d = domestic_close.reindex(eq.index).ffill()
        rd = np.log(d.astype(float)).diff()
        aligned = pd.concat([r, rd], axis=1).dropna().tail(window)
        if len(aligned) > 10:
            out["beta_dom"] = float(aligned.iloc[:, 0].cov(aligned.iloc[:, 1]) / (aligned.iloc[:, 1].var() + 1e-12))
            out["corr_dom"] = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
        else:
            out["beta_dom"] = 0.0
            out["corr_dom"] = 0.0
    else:
        out["beta_dom"] = 0.0
        out["corr_dom"] = 0.0
    if us_close is not None:
        u = us_close.reindex(eq.index).ffill()
        ru = np.log(u.astype(float)).diff()
        aligned = pd.concat([r, ru], axis=1).dropna().tail(window)
        if len(aligned) > 10:
            out["beta_us"] = float(aligned.iloc[:, 0].cov(aligned.iloc[:, 1]) / (aligned.iloc[:, 1].var() + 1e-12))
            out["corr_us"] = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
        else:
            out["beta_us"] = 0.0
            out["corr_us"] = 0.0
    else:
        out["beta_us"] = 0.0
        out["corr_us"] = 0.0
    return out


def cluster_twins(
    form_rows: list[dict],
    *,
    n_clusters: int | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    form_rows: list of dicts with key 'ticker' + form features.
    Returns dataframe with cluster id (twins share cluster).
    """
    if not form_rows:
        return pd.DataFrame()
    df = pd.DataFrame(form_rows).set_index("ticker")
    feat_cols = [c for c in df.columns if c != "ticker"]
    X = df[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    n = len(X)
    k = n_clusters or max(2, min(5, n // 2 or 2))
    k = min(k, n)
    if k < 2:
        df["cluster"] = 0
        return df.reset_index()
    Xs = StandardScaler().fit_transform(X.values)
    km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    df["cluster"] = km.fit_predict(Xs)
    return df.reset_index()


def twin_pairs(clustered: pd.DataFrame) -> pd.DataFrame:
    """List pairs that share a cluster (excluding self)."""
    rows = []
    if clustered is None or len(clustered) == 0 or "cluster" not in clustered.columns:
        return pd.DataFrame(columns=["ticker_a", "ticker_b", "cluster"])
    for c, g in clustered.groupby("cluster"):
        tickers = list(g["ticker"])
        for i in range(len(tickers)):
            for j in range(i + 1, len(tickers)):
                rows.append({"ticker_a": tickers[i], "ticker_b": tickers[j], "cluster": int(c)})
    return pd.DataFrame(rows)
