#!/usr/bin/env python3
"""Terminal UI entry — no browser, no web server."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stock_prob.ui_colab import launch_tui

if __name__ == "__main__":
    launch_tui()
