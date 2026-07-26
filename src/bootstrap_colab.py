"""Session bootstrap for Stock Probability Engine (Colab).

Run once per runtime. Prefer Drive when available; fall back to /content/stock-prob.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _pip_install(pkgs: list[str]) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])


def detect_base() -> Path:
    drive_base = Path("/content/drive/MyDrive/stock-prob")
    local_base = Path("/content/stock-prob")
    if drive_base.exists() or Path("/content/drive/MyDrive").exists():
        # Prefer Drive if mounted
        if Path("/content/drive/MyDrive").exists():
            drive_base.mkdir(parents=True, exist_ok=True)
            return drive_base
    return local_base


def ensure_dirs(base: Path) -> dict[str, Path]:
    paths = {
        "base": base,
        "data": base / "data",
        "predictions": base / "predictions",
        "exports": base / "exports",
        "src": base / "src",
        "notebooks": base / "notebooks",
        "docs": base / "docs",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def mount_drive_if_possible(force: bool = False) -> bool:
    """Interactive Drive mount. Returns True if /content/drive is available."""
    if Path("/content/drive/MyDrive").exists() and not force:
        return True
    try:
        from google.colab import drive  # type: ignore

        drive.mount("/content/drive")
        return Path("/content/drive/MyDrive").exists()
    except Exception as e:
        print(f"[bootstrap] Drive mount skipped/failed: {e}")
        return False


def bootstrap(install_extras: bool = True, try_drive: bool = False) -> dict:
    if try_drive:
        mount_drive_if_possible()
    if install_extras:
        _pip_install(["arch", "kaleido==0.2.1", "xlsxwriter", "curl_cffi"])
    try:
        import plotly.io as pio

        pio.renderers.default = "colab"
    except Exception:
        pass
    base = detect_base()
    paths = ensure_dirs(base)
    # ensure src on path
    src = paths["src"]
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    if str(paths["base"]) not in sys.path:
        sys.path.insert(0, str(paths["base"]))
    info = {
        "base": str(base),
        "on_drive": str(base).startswith("/content/drive"),
        "python": sys.version.split()[0],
        "paths": {k: str(v) for k, v in paths.items()},
    }
    print("[bootstrap] base =", info["base"], "| on_drive =", info["on_drive"])
    return info


if __name__ == "__main__":
    bootstrap(install_extras=False, try_drive=False)
