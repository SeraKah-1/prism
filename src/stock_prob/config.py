"""Universe & run configuration — symbols live here, never in algorithms."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class UniverseConfig:
    """Dynamic equity universe. All symbol strings are data, not code constants."""

    equities: list[str] = field(default_factory=list)
    domestic_index: str | None = None
    us_index: str | None = None
    macro: str | None = None
    # optional extras (e.g. second macro) — ignored by core if unused
    extra: list[str] = field(default_factory=list)

    def all_symbols(self) -> list[str]:
        out: list[str] = []
        for s in (
            list(self.equities)
            + ([self.domestic_index] if self.domestic_index else [])
            + ([self.us_index] if self.us_index else [])
            + ([self.macro] if self.macro else [])
            + list(self.extra)
        ):
            if s and s not in out:
                out.append(s)
        return out

    def context_symbols(self) -> dict[str, str | None]:
        return {
            "domestic_index": self.domestic_index,
            "us_index": self.us_index,
            "macro": self.macro,
        }


@dataclass
class RunConfig:
    universe: UniverseConfig
    horizons: list[int] = field(default_factory=lambda: [5, 21, 252])
    history_period: str = "5y"
    rolling_window: int = 60
    mc_paths: int = 2000
    walkforward_refit_every: int = 21  # trading days between refits
    min_train_rows: int = 252
    random_state: int = 42
    run_name: str = "default"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


def load_universe_from_mapping(data: dict[str, Any]) -> UniverseConfig:
    u = data.get("universe", data)
    return UniverseConfig(
        equities=list(u.get("equities") or u.get("tickers") or []),
        domestic_index=u.get("domestic_index") or u.get("index_domestic"),
        us_index=u.get("us_index") or u.get("index_us"),
        macro=u.get("macro"),
        extra=list(u.get("extra") or []),
    )


def load_run_config(path: str | Path | None = None, **overrides: Any) -> RunConfig:
    """Load YAML/JSON config file; optional kwargs override fields."""
    data: dict[str, Any] = {}
    if path is not None:
        p = Path(path)
        text = p.read_text()
        if p.suffix.lower() in {".yaml", ".yml"}:
            data = yaml.safe_load(text) or {}
        elif p.suffix.lower() == ".json":
            data = json.loads(text)
        else:
            # try yaml first
            data = yaml.safe_load(text) or {}

    universe = load_universe_from_mapping(data)
    horizons = data.get("horizons", [5, 21, 252])
    cfg = RunConfig(
        universe=universe,
        horizons=list(horizons),
        history_period=str(data.get("history_period", "5y")),
        rolling_window=int(data.get("rolling_window", 60)),
        mc_paths=int(data.get("mc_paths", 2000)),
        walkforward_refit_every=int(data.get("walkforward_refit_every", 21)),
        min_train_rows=int(data.get("min_train_rows", 252)),
        random_state=int(data.get("random_state", 42)),
        run_name=str(data.get("run_name", data.get("name", "default"))),
    )
    # apply simple overrides
    for k, v in overrides.items():
        if k == "universe" and isinstance(v, (dict, UniverseConfig)):
            if isinstance(v, dict):
                cfg.universe = load_universe_from_mapping(v)
            else:
                cfg.universe = v
        elif k == "equities":
            cfg.universe.equities = list(v)
        elif hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


def universe_from_symbols(
    equities: list[str],
    *,
    domestic_index: str | None = None,
    us_index: str | None = None,
    macro: str | None = None,
    extra: list[str] | None = None,
) -> UniverseConfig:
    """Programmatic constructor — preferred over hardcoding in demos."""
    return UniverseConfig(
        equities=list(equities),
        domestic_index=domestic_index,
        us_index=us_index,
        macro=macro,
        extra=list(extra or []),
    )


def save_run_config(cfg: RunConfig, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
    return p
