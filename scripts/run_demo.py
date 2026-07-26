#!/usr/bin/env python3
"""CLI demo entry — universe always from config or CLI flags (never hardcoded core)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# allow running without install
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stock_prob.config import load_run_config, universe_from_symbols  # noqa: E402
from stock_prob.paths import ensure_layout, get_project_root  # noqa: E402
from stock_prob.pipeline import run_pipeline  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Stock Probability Engine demo")
    p.add_argument("--config", type=str, default=None, help="Path to YAML/JSON run config")
    p.add_argument(
        "--equities",
        type=str,
        default=None,
        help="Comma-separated equity tickers (overrides config equities)",
    )
    p.add_argument("--domestic-index", type=str, default=None)
    p.add_argument("--us-index", type=str, default=None)
    p.add_argument("--macro", type=str, default=None)
    p.add_argument("--equity", type=str, default=None, help="Single equity to forecast")
    p.add_argument("--horizons", type=str, default=None, help="e.g. 5,21,252")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--root", type=str, default=None)
    args = p.parse_args(argv)

    root = Path(args.root) if args.root else get_project_root()
    ensure_layout(root)

    if args.config:
        cfg = load_run_config(args.config)
    else:
        # minimal empty; must get equities from CLI
        cfg = load_run_config(
            None,
        )
        cfg.universe = universe_from_symbols([])

    if args.equities:
        cfg.universe.equities = [x.strip() for x in args.equities.split(",") if x.strip()]
    if args.domestic_index:
        cfg.universe.domestic_index = args.domestic_index
    if args.us_index:
        cfg.universe.us_index = args.us_index
    if args.macro:
        cfg.universe.macro = args.macro
    if args.horizons:
        cfg.horizons = [int(x) for x in args.horizons.split(",") if x.strip()]

    if not cfg.universe.equities and not args.equity:
        print("ERROR: provide --config with universe.equities or --equities / --equity", file=sys.stderr)
        return 2

    target = args.equity or cfg.universe.equities[0]
    if target not in cfg.universe.equities:
        cfg.universe.equities = [target] + list(cfg.universe.equities)

    print(f"[demo] root={root}")
    print(f"[demo] target={target}")
    print(f"[demo] universe={cfg.universe.all_symbols()}")
    print(f"[demo] horizons={cfg.horizons}")

    result = run_pipeline(
        cfg,
        equity=target,
        root=root,
        use_cache=not args.no_cache,
    )
    summary = {
        "run_id": result["run_id"],
        "ticker": result["ticker"],
        "live_probs": result["live_probs"],
        "art_dir": result["art_dir"],
        "report_html": result["report_html"],
        "excel": result["excel"],
        "metrics": result["metrics"].to_dict(orient="records")
        if result["metrics"] is not None and len(result["metrics"])
        else [],
    }
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
