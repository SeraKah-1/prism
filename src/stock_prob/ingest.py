"""Fetch & cache market bars for arbitrary symbols (no hardcoded tickers)."""
from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Iterable

import pandas as pd

from stock_prob.paths import ensure_layout, get_project_root


def _safe_name(symbol: str) -> str:
    """Filesystem-safe name for a ticker symbol."""
    s = symbol.strip()
    s = s.replace("^", "")
    s = re.sub(r"[^\w.\-]+", "_", s)
    return s or "UNKNOWN"


def cache_path(symbol: str, data_dir: Path | None = None) -> Path:
    root = get_project_root()
    paths = ensure_layout(root)
    base = data_dir or paths["data"]
    # prefer data/raw if present layout; also accept flat data/
    raw = paths["raw"]
    raw.mkdir(parents=True, exist_ok=True)
    return raw / f"{_safe_name(symbol)}.parquet"


def _normalize_ohlcv(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume", "ticker"])

    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    df = df.reset_index()
    # date column
    cols_lower = {c: str(c).lower() for c in df.columns}
    df = df.rename(columns=cols_lower)
    date_col = None
    for cand in ("date", "datetime", "index"):
        if cand in df.columns:
            date_col = cand
            break
    if date_col is None:
        # first column
        date_col = df.columns[0]
    df = df.rename(columns={date_col: "date"})
    df["date"] = pd.to_datetime(df["date"], utc=False).dt.tz_localize(None)

    rename_map = {}
    for c in df.columns:
        cl = c.lower()
        if cl in ("adj close", "adj_close", "adjclose"):
            rename_map[c] = "close"  # prefer adj as close if present later
        elif cl in ("open", "high", "low", "close", "volume"):
            rename_map[c] = cl
    df = df.rename(columns=rename_map)

    # If both close and adj, prefer close after auto_adjust
    keep = ["date"]
    for c in ("open", "high", "low", "close", "volume"):
        if c in df.columns:
            keep.append(c)
    out = df[keep].copy()
    if "close" not in out.columns:
        raise ValueError(f"No close column for {symbol}")
    for c in ("open", "high", "low"):
        if c not in out.columns:
            out[c] = out["close"]
    if "volume" not in out.columns:
        out["volume"] = 0
    out["ticker"] = symbol
    out = out.dropna(subset=["close"]).sort_values("date").drop_duplicates("date")
    out = out.reset_index(drop=True)
    return out


def load_cached(symbol: str, data_dir: Path | None = None) -> pd.DataFrame | None:
    path = cache_path(symbol, data_dir)
    name = f"{_safe_name(symbol)}.parquet"
    candidates = [
        path,
        get_project_root() / "data" / name,
        get_project_root() / "data" / "raw" / name,
        Path("/content/stock-prob/data") / name,
        Path("/content/stock-prob/data/raw") / name,
        Path("/content/drive/MyDrive/stock-prob/data") / name,
        Path("/content/drive/MyDrive/stock-prob/data/raw") / name,
    ]
    for p in candidates:
        if p is None:
            continue
        try:
            if p.exists():
                df = pd.read_parquet(p)
                if "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
                if "ticker" not in df.columns:
                    df["ticker"] = symbol
                return df.sort_values("date").reset_index(drop=True)
        except Exception:
            continue
    return None


def save_cached(df: pd.DataFrame, symbol: str, data_dir: Path | None = None) -> Path:
    path = cache_path(symbol, data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out.to_parquet(path, index=False)
    # also mirror flat data for convenience
    flat = get_project_root() / "data" / f"{_safe_name(symbol)}.parquet"
    flat.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(flat, index=False)
    return path


def fetch_symbol(
    symbol: str,
    *,
    period: str = "5y",
    use_cache: bool = True,
    force_refresh: bool = False,
    sleep_s: float = 0.35,
) -> pd.DataFrame:
    """Fetch one symbol via yfinance; cache-friendly incremental update."""
    cached = load_cached(symbol) if use_cache and not force_refresh else None

    import yfinance as yf

    if cached is not None and len(cached) > 0 and not force_refresh:
        # incremental: pull recent window and merge
        last = cached["date"].max()
        start = (last - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
        try:
            fresh = yf.download(
                symbol,
                start=start,
                progress=False,
                threads=False,
                auto_adjust=True,
            )
            fresh_n = _normalize_ohlcv(fresh, symbol)
            if len(fresh_n):
                merged = (
                    pd.concat([cached, fresh_n], ignore_index=True)
                    .drop_duplicates("date", keep="last")
                    .sort_values("date")
                    .reset_index(drop=True)
                )
                save_cached(merged, symbol)
                time.sleep(sleep_s)
                return merged
        except Exception:
            # network flake: return cache
            return cached
        return cached

    # full download
    raw = yf.download(symbol, period=period, progress=False, threads=False, auto_adjust=True)
    df = _normalize_ohlcv(raw, symbol)
    if len(df) == 0:
        raise RuntimeError(f"No data returned for symbol={symbol!r}")
    if use_cache:
        save_cached(df, symbol)
    time.sleep(sleep_s)
    return df


def fetch_universe(
    symbols: Iterable[str],
    *,
    period: str = "5y",
    use_cache: bool = True,
    force_refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """Fetch/load many symbols. Returns {symbol: OHLCV frame}."""
    out: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        if not sym:
            continue
        out[sym] = fetch_symbol(
            sym, period=period, use_cache=use_cache, force_refresh=force_refresh
        )
    return out


def align_close_panel(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Wide panel of close prices, columns = symbols, index = date."""
    series = []
    for sym, df in frames.items():
        s = df.set_index("date")["close"].rename(sym)
        series.append(s)
    if not series:
        return pd.DataFrame()
    panel = pd.concat(series, axis=1).sort_index()
    return panel
