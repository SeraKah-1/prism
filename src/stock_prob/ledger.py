"""Append-only prediction ledger + run artifact writer."""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from stock_prob.paths import ensure_layout, get_project_root


def new_run_id(prefix: str = "run") -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{ts}_{uuid.uuid4().hex[:8]}"


def run_dir(run_id: str, root: Path | None = None) -> Path:
    paths = ensure_layout(root or get_project_root())
    d = paths["runs"] / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_run_artifacts(
    run_id: str,
    *,
    config: dict[str, Any],
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    extra: dict[str, Any] | None = None,
    root: Path | None = None,
) -> Path:
    d = run_dir(run_id, root)
    (d / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    if len(predictions):
        predictions.to_parquet(d / "predictions.parquet", index=False)
        predictions.to_csv(d / "predictions.csv", index=False)
    if len(metrics):
        metrics.to_parquet(d / "metrics.parquet", index=False)
        metrics.to_csv(d / "metrics.csv", index=False)
        metrics_json = metrics.to_dict(orient="records")
    else:
        metrics_json = []
    payload = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics_json,
        "n_predictions": int(len(predictions)),
        "extra": extra or {},
    }
    (d / "metrics.json").write_text(json.dumps(payload, indent=2, default=str))
    return d


def append_ledger(
    rows: pd.DataFrame,
    root: Path | None = None,
) -> Path:
    paths = ensure_layout(root or get_project_root())
    path = paths["predictions"] / "ledger.parquet"
    if path.exists():
        old = pd.read_parquet(path)
        out = pd.concat([old, rows], ignore_index=True)
    else:
        out = rows.copy()

    # Atomic write pattern
    tmp_path = path.with_suffix(".tmp")
    out.to_parquet(tmp_path, index=False)
    tmp_path.replace(path)

    # csv mirror for eyeballing
    out.tail(5000).to_csv(paths["predictions"] / "ledger_tail.csv", index=False)
    return path


def resolve_ledger(root: Path | None = None) -> tuple[int, list[tuple[str, str]]]:
    """
    Scan ledger.parquet at lab/run launch:
    1. Refresh price cache via fetch_symbol() to prevent stale data.
    2. Use adj_close (or close if auto_adjust=True) to avoid dividend/split distortion.
    3. Log skip reasons to (symbol, reason) list.
    4. Write atomically (.tmp -> replace) to prevent parquet corruption.
    """
    from stock_prob.ingest import fetch_symbol

    paths = ensure_layout(root or get_project_root())
    ledger_path = paths["predictions"] / "ledger.parquet"
    if not ledger_path.exists():
        return 0, []

    try:
        df = pd.read_parquet(ledger_path)
    except Exception as e:
        return 0, [("ALL", f"read_ledger_failed: {e}")]

    if len(df) == 0 or "actual_return" not in df.columns:
        return 0, []

    unresolved = df[df["actual_return"].isna() & df["predicted_at"].notna()]
    if len(unresolved) == 0:
        return 0, []

    resolved_count = 0
    skipped_log: list[tuple[str, str]] = []
    now = pd.Timestamp.now(tz=timezone.utc).tz_localize(None)

    for idx, row in unresolved.iterrows():
        symbol = str(row["ticker"])
        pred_date = pd.to_datetime(row["predicted_at"]).tz_localize(None)
        h = int(row.get("horizon_days", 21))

        if now < pred_date + pd.Timedelta(days=int(h * 1.45)):
            continue

        try:
            cached = fetch_symbol(symbol, use_cache=True)
        except Exception as e:
            skipped_log.append((symbol, f"fetch_failed: {e}"))
            continue

        if cached is None or len(cached) == 0:
            skipped_log.append((symbol, "cache_empty"))
            continue

        cached["date"] = pd.to_datetime(cached["date"]).dt.tz_localize(None)
        post_dates = cached[cached["date"] >= pred_date].sort_values("date")

        if len(post_dates) <= h:
            skipped_log.append((symbol, f"insufficient_history: {len(post_dates)}/{h+1} bars"))
            continue

        price_col = "adj_close" if "adj_close" in post_dates.columns else "close"
        entry_px = float(post_dates[price_col].iloc[0])
        exit_px = float(post_dates[price_col].iloc[h])

        if entry_px <= 0 or not (entry_px == entry_px and exit_px == exit_px):
            skipped_log.append((symbol, "invalid_price_zero_or_nan"))
            continue

        act_ret = (exit_px / entry_px) - 1.0
        act_up = 1.0 if act_ret > 0 else 0.0
        p_up = float(row.get("prob_up", float("nan")))

        df.at[idx, "actual_return"] = act_ret
        df.at[idx, "actual_up"] = act_up
        if p_up == p_up:
            df.at[idx, "brier"] = (p_up - act_up) ** 2
        df.at[idx, "resolved_at"] = now.isoformat()
        resolved_count += 1

    if resolved_count > 0:
        tmp_path = ledger_path.with_suffix(".tmp")
        df.to_parquet(tmp_path, index=False)
        tmp_path.replace(ledger_path)
        df.tail(5000).to_csv(paths["predictions"] / "ledger_tail.csv", index=False)

    return resolved_count, skipped_log
