#!/usr/bin/env python3
"""Run full lab: panel + GARCH/regime/spillover/tournament/twins/surface."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml

from stock_prob.lab import run_full_lab
from stock_prob.paths import ensure_layout, get_project_root


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "configs" / "panel_core.yaml"))
    p.add_argument("--equities", default=None, help="Comma-separated override")
    p.add_argument("--horizons", default="5,21,252")
    p.add_argument("--root", default=None)
    args = p.parse_args(argv)

    root = Path(args.root) if args.root else get_project_root()
    ensure_layout(root)

    equities = []
    if args.equities:
        equities = [x.strip() for x in args.equities.split(",") if x.strip()]
    else:
        data = yaml.safe_load(Path(args.config).read_text()) or {}
        equities = list(data.get("equities") or [])
    if not equities:
        print("No equities provided", file=sys.stderr)
        return 2

    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    print(f"[full_lab] root={root}")
    print(f"[full_lab] equities={equities}")
    print(f"[full_lab] horizons={horizons}")

    out = run_full_lab(equities, horizons=horizons, root=root, use_cache=True)
    summary = {
        "lab_id": out["lab_id"],
        "lab_dir": out["lab_dir"],
        "index_html": out["index_html"],
        "n_panel_ok": out["meta"]["n_panel_ok"],
        "n_enriched": out["meta"]["n_enriched"],
        "country_skill": out["meta"]["country_skill"],
        "n_twins": int(len(out["twins"])) if out.get("twins") is not None else 0,
        "n_pairs": int(len(out["twin_pairs"])) if out.get("twin_pairs") is not None else 0,
    }
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
