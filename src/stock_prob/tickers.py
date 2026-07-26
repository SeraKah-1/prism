"""
Dynamic ticker search & resolve — no hardcoded equity universe in logic.

Uses yfinance Search / Ticker validation. Optional recent-cache on disk is
user history, not a fixed product universe.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from stock_prob.paths import ensure_layout, get_project_root


@dataclass
class TickerHit:
    symbol: str
    name: str
    exchange: str = ""
    quote_type: str = ""
    score: float = 0.0

    def label(self) -> str:
        bits = [self.symbol]
        if self.name:
            bits.append(self.name)
        if self.exchange:
            bits.append(f"[{self.exchange}]")
        return " — ".join(bits[:2]) + (f"  {bits[2]}" if len(bits) > 2 else "")


def _recents_path(root: Path | None = None) -> Path:
    paths = ensure_layout(root or get_project_root())
    return paths["data"] / "ticker_recents.json"


def load_recents(limit: int = 20, root: Path | None = None) -> list[dict[str, str]]:
    p = _recents_path(root)
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text())
        if isinstance(data, list):
            return data[:limit]
    except Exception:
        pass
    return []


def push_recent(symbol: str, name: str = "", root: Path | None = None) -> None:
    symbol = symbol.strip()
    if not symbol:
        return
    cur = load_recents(50, root)
    cur = [x for x in cur if x.get("symbol") != symbol]
    cur.insert(0, {"symbol": symbol, "name": name or symbol})
    p = _recents_path(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cur[:40], indent=2))


def _normalize_query(q: str) -> str:
    return re.sub(r"\s+", " ", (q or "").strip())


def search_tickers(
    query: str,
    *,
    max_results: int = 12,
    prefer_equity: bool = True,
) -> list[TickerHit]:
    """
    Live search via yfinance. Empty query → recent symbols (if any), else [].
    Never depends on a fixed in-code ticker list for results.
    """
    q = _normalize_query(query)
    if not q:
        rec = load_recents(max_results)
        return [
            TickerHit(symbol=r["symbol"], name=r.get("name") or r["symbol"], exchange="RECENT")
            for r in rec
        ]

    import yfinance as yf

    try:
        raw = yf.Search(q, max_results=max(max_results * 2, 8))
        quotes = list(getattr(raw, "quotes", None) or [])
    except Exception:
        quotes = []

    hits: list[TickerHit] = []
    seen: set[str] = set()
    for item in quotes:
        sym = (item.get("symbol") or "").strip()
        if not sym or sym in seen:
            continue
        seen.add(sym)
        qtype = (item.get("quoteType") or item.get("typeDisp") or "").upper()
        name = item.get("longname") or item.get("shortname") or sym
        exch = item.get("exchange") or item.get("exchDisp") or ""
        score = float(item.get("score") or 0.0)
        # soft prefer equities / indexes over random
        if prefer_equity and qtype in ("EQUITY", "INDEX", "ETF"):
            score += 1e6
        # boost exact / prefix matches
        qu = q.upper()
        su = sym.upper()
        if su == qu or su.startswith(qu):
            score += 5e6
        if qu in (name or "").upper():
            score += 1e5
        hits.append(
            TickerHit(
                symbol=sym,
                name=str(name),
                exchange=str(exch),
                quote_type=qtype,
                score=score,
            )
        )

    hits.sort(key=lambda h: (-h.score, h.symbol))
    return hits[:max_results]


def search_labels(query: str, max_results: int = 12) -> list[str]:
    """Labels for UI dropdowns: 'SYMBOL — Name'."""
    return [h.label() for h in search_tickers(query, max_results=max_results)]


def parse_symbol_from_label(label: str) -> str:
    """Extract symbol from 'SYM — Name' or bare symbol."""
    if not label:
        return ""
    lab = label.strip()
    if "—" in lab:
        return lab.split("—", 1)[0].strip()
    if " - " in lab:
        return lab.split(" - ", 1)[0].strip()
    return lab.split()[0].strip()


def resolve_ticker(symbol_or_label: str) -> dict[str, Any]:
    """
    Normalize user input to a tradable symbol.
    Tries: parse label → search fallback → validate with a tiny price pull.
    """
    raw = (symbol_or_label or "").strip()
    if not raw:
        return {"ok": False, "symbol": "", "error": "empty"}

    sym = parse_symbol_from_label(raw)

    # If user typed a free name without symbol, search first hit
    if " " in sym or (not re.search(r"[\.\^=]", sym) and len(sym) > 5 and not sym.isupper()):
        hits = search_tickers(raw, max_results=5)
        if hits:
            sym = hits[0].symbol

    # Bare alphabetic codes: prefer exchange-local matches from live search (.JK etc.)
    if re.fullmatch(r"[A-Za-z]{3,5}", sym):
        hits = search_tickers(sym, max_results=8)
        jk = next((h for h in hits if h.symbol.upper().endswith(".JK")), None)
        exact = next((h for h in hits if h.symbol.upper() == sym.upper()), None)
        if jk and (not exact or exact.exchange in ("JKT", "JK", "")):
            if any(h.exchange in ("JKT",) or h.symbol.upper().endswith(".JK") for h in hits[:5]):
                if jk.symbol.upper().startswith(sym.upper()):
                    sym = jk.symbol
        if exact:
            sym = exact.symbol

    valid = validate_symbol(sym)
    if not valid["ok"]:
        # last chance: search and validate first equity
        for h in search_tickers(raw, max_results=6):
            v = validate_symbol(h.symbol)
            if v["ok"]:
                push_recent(h.symbol, h.name)
                return v
        return {"ok": False, "symbol": sym, "error": valid.get("error", "not found")}

    push_recent(valid["symbol"], valid.get("name") or "")
    return valid


def validate_symbol(symbol: str, *, period: str = "5d") -> dict[str, Any]:
    """Confirm symbol has recent OHLCV (network/cache)."""
    symbol = (symbol or "").strip()
    if not symbol:
        return {"ok": False, "symbol": "", "error": "empty"}
    try:
        from stock_prob.ingest import fetch_symbol

        df = fetch_symbol(symbol, period=period, use_cache=True, force_refresh=False)
        if df is None or len(df) == 0:
            return {"ok": False, "symbol": symbol, "error": "no bars"}
        name = symbol
        try:
            import yfinance as yf

            info = yf.Ticker(symbol).fast_info
            # fast_info may not have longName
            name = getattr(info, "get", lambda k, d=None: d)("longName", name) if False else symbol
            # try short name via history metadata skip
        except Exception:
            pass
        # improve name via search
        for h in search_tickers(symbol, max_results=3):
            if h.symbol.upper() == symbol.upper():
                name = h.name
                break
        return {
            "ok": True,
            "symbol": symbol,
            "name": name,
            "n_bars": int(len(df)),
            "last_date": str(df["date"].iloc[-1].date()) if "date" in df.columns else None,
            "last_close": float(df["close"].iloc[-1]),
        }
    except Exception as e:
        return {"ok": False, "symbol": symbol, "error": str(e)[:200]}


def default_context_for_symbol(symbol: str) -> dict[str, str]:
    """Heuristic context indexes — still not a fixed equity universe."""
    s = symbol.upper()
    if s.endswith(".JK") or s.endswith(".JA"):
        return {"domestic_index": "^JKSE", "us_index": "^GSPC", "macro": "^VIX"}
    return {"domestic_index": "^GSPC", "us_index": "^GSPC", "macro": "^VIX"}
