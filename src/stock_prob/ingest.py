"""Fetch & cache market bars for arbitrary symbols (no hardcoded tickers)."""
from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Iterable

import pandas as pd

from stock_prob.paths import ensure_layout, get_project_root

# Prefer long history; never silently replace a good cache with a stub.
MIN_USEFUL_BARS = 40


def _safe_name(symbol: str) -> str:
    s = symbol.strip().replace("^", "")
    s = re.sub(r"[^\w.\-]+", "_", s)
    return s or "UNKNOWN"


def cache_path(symbol: str, data_dir: Path | None = None) -> Path:
    root = get_project_root()
    paths = ensure_layout(root)
    raw = paths["raw"]
    raw.mkdir(parents=True, exist_ok=True)
    return raw / f"{_safe_name(symbol)}.parquet"


def _normalize_ohlcv(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume", "ticker"])

    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        # yfinance multi-ticker style: ('Close', 'AMMN.JK') → take first level
        df.columns = [
            c[0] if isinstance(c, tuple) else c for c in df.columns
        ]

    df = df.reset_index()
    cols_lower = {c: str(c).lower() for c in df.columns}
    df = df.rename(columns=cols_lower)

    date_col = None
    for cand in ("date", "datetime", "index"):
        if cand in df.columns:
            date_col = cand
            break
    if date_col is None:
        date_col = df.columns[0]
    df = df.rename(columns={date_col: "date"})
    df["date"] = pd.to_datetime(df["date"], utc=False, errors="coerce")
    # strip tz if any
    if getattr(df["date"].dt, "tz", None) is not None:
        df["date"] = df["date"].dt.tz_localize(None)

    rename_map = {}
    for c in df.columns:
        cl = str(c).lower()
        if cl in ("adj close", "adj_close", "adjclose"):
            rename_map[c] = "close"
        elif cl in ("open", "high", "low", "close", "volume"):
            rename_map[c] = cl
    df = df.rename(columns=rename_map)

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

    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out["ticker"] = symbol
    out = out.dropna(subset=["date", "close"]).sort_values("date").drop_duplicates("date", keep="last")
    return out.reset_index(drop=True)


def _read_parquet(path: Path, symbol: str) -> pd.DataFrame | None:
    try:
        if not path.exists():
            return None
        df = pd.read_parquet(path)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            if getattr(df["date"].dt, "tz", None) is not None:
                df["date"] = df["date"].dt.tz_localize(None)
        if "ticker" not in df.columns:
            df["ticker"] = symbol
        if "close" not in df.columns:
            return None
        df = df.dropna(subset=["date", "close"]).sort_values("date").drop_duplicates("date", keep="last")
        return df.reset_index(drop=True)
    except Exception:
        return None


def _cache_candidates(symbol: str, data_dir: Path | None = None) -> list[Path]:
    name = f"{_safe_name(symbol)}.parquet"
    root = get_project_root()
    paths = ensure_layout(root)
    cands = [
        cache_path(symbol, data_dir),
        paths["data"] / name,
        paths["raw"] / name,
        Path("/content/prism/data") / name,
        Path("/content/prism/data/raw") / name,
        Path("/content/stock-prob/data") / name,
        Path("/content/stock-prob/data/raw") / name,
        Path("/content/drive/MyDrive/stock-prob/data") / name,
        Path("/content/drive/MyDrive/stock-prob/data/raw") / name,
        Path("/content/drive/MyDrive/prism/data") / name,
        Path("/content/drive/MyDrive/prism/data/raw") / name,
    ]
    # unique preserve order
    seen = set()
    out = []
    for p in cands:
        sp = str(p.resolve()) if p.exists() else str(p)
        if sp in seen:
            continue
        seen.add(sp)
        out.append(p)
    return out


def load_cached(symbol: str, data_dir: Path | None = None) -> pd.DataFrame | None:
    """Load the *longest* valid cached series among known locations."""
    best: pd.DataFrame | None = None
    for p in _cache_candidates(symbol, data_dir):
        df = _read_parquet(p, symbol)
        if df is None or len(df) == 0:
            continue
        if best is None or len(df) > len(best):
            best = df
    return best


def save_cached(
    df: pd.DataFrame,
    symbol: str,
    data_dir: Path | None = None,
    *,
    allow_shorter: bool = False,
) -> Path:
    """
    Write cache. By default refuse to overwrite a longer existing series
    with a shorter one (prevents 728 → 6 bar corruption).
    """
    existing = load_cached(symbol, data_dir)
    if (
        existing is not None
        and len(existing) > len(df)
        and not allow_shorter
    ):
        # keep longer history; still merge if overlapping
        merged = (
            pd.concat([existing, df], ignore_index=True)
            .drop_duplicates("date", keep="last")
            .sort_values("date")
            .reset_index(drop=True)
        )
        df = merged

    path = cache_path(symbol, data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    flat = get_project_root() / "data" / f"{_safe_name(symbol)}.parquet"
    flat.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(flat, index=False)
    return path


def _download_yf(symbol: str, *, period: str | None = None, start: str | None = None) -> pd.DataFrame:
    import yfinance as yf

    if start:
        raw = yf.download(
            symbol, start=start, progress=False, threads=False, auto_adjust=True
        )
    else:
        raw = yf.download(
            symbol, period=period or "max", progress=False, threads=False, auto_adjust=True
        )
    return _normalize_ohlcv(raw, symbol)


def fetch_symbol(
    symbol: str,
    *,
    period: str = "max",
    use_cache: bool = True,
    force_refresh: bool = False,
    sleep_s: float = 0.25,
    min_bars: int = MIN_USEFUL_BARS,
) -> pd.DataFrame:
    """
    Fetch OHLCV with safe cache.

    Critical fix: never trust a stub cache (e.g. 6 bars) when full history exists
    or can be re-downloaded; never overwrite long cache with short download.
    """
    symbol = symbol.strip()
    cached = load_cached(symbol) if use_cache and not force_refresh else None

    # Treat short cache as invalid — force full download
    if cached is not None and len(cached) < min_bars:
        cached = None

    if cached is not None and len(cached) > 0 and not force_refresh:
        last = cached["date"].max()
        start = (pd.Timestamp(last) - pd.Timedelta(days=14)).strftime("%Y-%m-%d")
        try:
            fresh_n = _download_yf(symbol, start=start)
            if len(fresh_n):
                merged = (
                    pd.concat([cached, fresh_n], ignore_index=True)
                    .drop_duplicates("date", keep="last")
                    .sort_values("date")
                    .reset_index(drop=True)
                )
                # safety: never shrink
                if len(merged) >= len(cached):
                    if use_cache:
                        save_cached(merged, symbol)
                    time.sleep(sleep_s)
                    return merged
            time.sleep(sleep_s)
            return cached
        except Exception:
            return cached

    # full download
    df = _download_yf(symbol, period=period)
    # if download short but we have longer cache somewhere, prefer cache merge
    longer = load_cached(symbol)
    if longer is not None and len(longer) > len(df):
        df = (
            pd.concat([longer, df], ignore_index=True)
            .drop_duplicates("date", keep="last")
            .sort_values("date")
            .reset_index(drop=True)
        )

    if len(df) == 0:
        # last resort: try max period
        df = _download_yf(symbol, period="max")
    if len(df) == 0:
        raise RuntimeError(f"No data returned for symbol={symbol!r}")

    if use_cache:
        save_cached(df, symbol, allow_shorter=False)
    time.sleep(sleep_s)
    return df


def fetch_universe(
    symbols: Iterable[str],
    *,
    period: str = "max",
    use_cache: bool = True,
    force_refresh: bool = False,
    min_bars: int = MIN_USEFUL_BARS,
) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        if not sym:
            continue
        df = fetch_symbol(
            sym,
            period=period,
            use_cache=use_cache,
            force_refresh=force_refresh,
            min_bars=min_bars,
        )
        # auto-heal short series once
        if len(df) < min_bars and not force_refresh:
            df = fetch_symbol(
                sym, period="max", use_cache=True, force_refresh=True, min_bars=1
            )
        out[sym] = df
    return out


def align_close_panel(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    series = []
    for sym, df in frames.items():
        if df is None or len(df) == 0 or "close" not in df.columns:
            continue
        s = df.set_index("date")["close"].astype(float).rename(sym)
        series.append(s)
    if not series:
        return pd.DataFrame()
    return pd.concat(series, axis=1).sort_index()


def align_us_to_idx(us_series: pd.Series, idx_dates: pd.DatetimeIndex | pd.Series) -> pd.Series:
    """Single Source of Truth: Align US close prices/returns to IDX dates (overnight lag)."""
    us = us_series.astype(float).copy()
    us.index = pd.to_datetime(us.index).tz_localize(None)
    us_shifted = us.shift(1)
    idx_idx = pd.to_datetime(idx_dates).tz_localize(None)
    return us_shifted.reindex(idx_idx).ffill()


def fetch_sector_meta(symbols: Iterable[str], root: Path | None = None) -> pd.DataFrame:
    """
    Fetch sector metadata for symbols using yfinance info + yaml fallback + parquet cache.
    Data Layer function: pure data retrieval with atomic write pattern.
    """
    import yaml

    root = root or get_project_root()
    paths = ensure_layout(root)
    cache_file = paths["data"] / "sectors.parquet"
    yaml_file = root / "configs" / "sectors.yaml"

    yaml_map: dict[str, str] = {}
    if yaml_file.exists():
        try:
            yaml_map = yaml.safe_load(yaml_file.read_text()) or {}
        except Exception:
            yaml_map = {}

    cached_df = pd.DataFrame(columns=["symbol", "sector"])
    if cache_file.exists():
        try:
            cached_df = pd.read_parquet(cache_file)
        except Exception:
            cached_df = pd.DataFrame(columns=["symbol", "sector"])

    cached_map = dict(zip(cached_df["symbol"], cached_df["sector"])) if len(cached_df) else {}
    updated = False
    result_rows = []

    for sym in symbols:
        if not sym:
            continue
        sym = str(sym).strip()
        sec = cached_map.get(sym)
        if not sec or sec == "Unknown":
            # try yfinance
            try:
                import yfinance as yf

                info = yf.Ticker(sym).info or {}
                sec = info.get("sector") or info.get("industry")
            except Exception:
                sec = None

            if not sec:
                sec = yaml_map.get(sym, "Unknown")
            cached_map[sym] = sec
            updated = True

        result_rows.append({"symbol": sym, "sector": sec or "Unknown"})

    out_df = pd.DataFrame(result_rows).drop_duplicates("symbol").reset_index(drop=True)
    if updated:
        try:
            tmp = cache_file.with_suffix(".tmp")
            out_df.to_parquet(tmp, index=False)
            tmp.replace(cache_file)
        except Exception:
            pass

    return out_df
