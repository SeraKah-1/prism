"""Project root resolution: prefer Drive lake, fall back to local Colab path."""
from __future__ import annotations

from pathlib import Path

_DRIVE = Path("/content/drive/MyDrive/stock-prob")
_LOCAL = Path("/content/stock-prob")


def get_project_root(prefer_drive: bool = True) -> Path:
    if prefer_drive and _DRIVE.exists():
        return _DRIVE
    if _LOCAL.exists():
        return _LOCAL
    # last resort: create local
    _LOCAL.mkdir(parents=True, exist_ok=True)
    return _LOCAL


def ensure_layout(root: Path | None = None) -> dict[str, Path]:
    root = root or get_project_root()
    paths = {
        "root": root,
        "data": root / "data",
        "raw": root / "data" / "raw",
        "features": root / "data" / "features",
        "predictions": root / "predictions",
        "runs": root / "runs",
        "exports": root / "exports",
        "gallery": root / "exports" / "gallery",
        "configs": root / "configs",
        "src": root / "src",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths
