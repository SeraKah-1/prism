"""Normalize horizon keys like 5, '5', 'h5', 'ens_21' → int | None."""
from __future__ import annotations

import re
from typing import Any


_DIGITS = re.compile(r"(\d+)")


def parse_horizon_key(key: Any) -> int | None:
    if key is None:
        return None
    if isinstance(key, (int, float)) and key == key:
        try:
            return int(key)
        except Exception:
            return None
    s = str(key).strip().lower()
    if s.isdigit():
        return int(s)
    m = _DIGITS.search(s)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None


def normalize_prob_map(raw: dict[Any, Any] | None) -> dict[str, float]:
    """Keep only finite probs keyed by bare horizon digits as str."""
    out: dict[str, float] = {}
    if not raw:
        return out
    for k, v in raw.items():
        # skip ensemble-only keys for primary cards if prefixed ens_
        sk = str(k).lower()
        if sk.startswith("ens"):
            continue
        h = parse_horizon_key(k)
        if h is None:
            continue
        try:
            pf = float(v)
        except Exception:
            continue
        if pf == pf:  # finite
            out[str(h)] = pf
    return out
