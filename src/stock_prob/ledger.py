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
    out.to_parquet(path, index=False)
    # csv mirror for eyeballing
    out.tail(5000).to_csv(paths["predictions"] / "ledger_tail.csv", index=False)
    return path
