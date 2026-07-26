# TradingView indicators (SPE)

Pine Script v5 indicators that **mirror SPE concepts** on the chart.

> **Important:** TradingView cannot run the Python logistic / walk-forward lab.  
> These scripts are **visual proxies** (triangulation, regime, vol cone).  
> For calibrated `P(up)`, Brier scores, and honest OOS metrics, use the Colab GUI / Python lab.

## Files

| File | What it shows |
|------|----------------|
| `SPE_Triangulation.pine` | Rolling β & ρ vs domestic + US index, R²-style market coupling |
| `SPE_Regime_ProbProxy.pine` | Regime shading + 0–100 probability **proxy** score |
| `SPE_Cone_Bands.pine` | Forward vol-based uncertainty bands (cone-like) on price |

## Install on TradingView

1. Open [TradingView](https://www.tradingview.com/) → chart for **any** ticker (e.g. `BBCA` on IDX, or `AAPL`).
2. Bottom panel → **Pine Editor**.
3. Open a `.pine` file from this folder → copy all → paste into editor.
4. **Save** → **Add to chart**.
5. Gear icon on the indicator → set **Domestic index** / **US index** symbols available on your TV plan  
   - Indonesia composite often `IDXCOMPOSITE` or exchange-specific  
   - US: `SPX` or `SP:SPX`

## Suggested layout

1. Price chart + `SPE_Cone_Bands`  
2. Subpane: `SPE_Triangulation`  
3. Subpane: `SPE_Regime_ProbProxy`  

## Syncing with the Python lab

| Question | Where |
|----------|--------|
| Calibrated P(up), Brier, walk-forward | Colab GUI / `run_demo.py` |
| Live chart overlay while trading | These Pine scripts |
| Same ticker universe | Use the **same symbols** in GUI and on TV (mapping may differ: `BBCA.JK` vs `BBCA`) |

No webhook bridge is required for v1. Optional future: export lab levels to TV alerts via webhook.
