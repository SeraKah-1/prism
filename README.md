# Stock Probability Engine (SPE)

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-research%20lab-lightgrey.svg)](#)
[![Colab](https://img.shields.io/badge/Colab-ready-orange.svg)](https://colab.research.google.com/)

> **Calibrated multi-horizon equity probabilities + honest prediction cones.**  
> Not a price oracle. Not a trading bot. A **probability laboratory** you can audit.

**Disclaimer:** Research only — see [CODE_OF_ETHICS.md](CODE_OF_ETHICS.md). Not financial advice.

---

## Why this exists

Retail “stock predictors” often overfit noise and hide uncertainty. SPE does the opposite:

1. **Dynamic universe** — tickers are **config/CLI data**, never hardcoded in core algorithms  
2. **Triangulation features** — rolling beta/corr to domestic + US index + one macro  
3. **`P(up)` + fan-chart cones** — multi-horizon (5d / 21d / 252d)  
4. **Walk-forward OOS** with embargo scoring vs **base rate** & **momentum**  
5. **Full lab modules** — GARCH cones, regime, IDX spillover, tournament/ensemble, twins, honesty score, probability surface  
6. **Everything recorded** — `runs/{run_id}/`, append-only ledger, HTML reports  

---

## Architecture

```mermaid
flowchart LR
  CFG[YAML / CLI universe] --> IN[ingest + parquet cache]
  IN --> FE[PIT features]
  FE --> MD[logistic P_up]
  FE --> GC[GARCH / MC cone]
  FE --> RG[regime]
  MD --> WF[walk-forward + embargo]
  WF --> TR[tournament + ensemble]
  WF --> LG[ledger + metrics]
  GC --> RP[HTML fan report]
  TR --> RP
  FE --> TW[twin clusters]
  WF --> HO[honesty score]
  MD --> SF[probability surface]
```

---

## Quickstart

### Colab / Linux

```bash
cd stock-prob   # or /content/stock-prob
export PYTHONPATH=$PWD/src
pip install -r requirements.txt

# Single ticker (config)
python scripts/run_demo.py --config configs/universe_idx_example.yaml

# Fully dynamic CLI — change tickers anytime
python scripts/run_demo.py \
  --equities TLKM.JK \
  --domestic-index '^JKSE' \
  --us-index '^GSPC' \
  --macro '^VIX' \
  --equity TLKM.JK \
  --horizons 5,21,252

# Full lab: multi-ticker panel + enrichment
python scripts/run_full_lab.py --config configs/panel_core.yaml
```

Artifacts:

| Path | Content |
|------|---------|
| `runs/{run_id}/report.html` | Fan chart + metrics |
| `runs/{run_id}/report_lab.html` | GARCH cone + regime + tournament |
| `runs/{lab_id}/index.html` | Panel + country skill + twins |
| `predictions/ledger.parquet` | Append-only prediction journal |
| `exports/gallery/` | HTML copies |

When Google Drive is mounted, the lake prefers  
`/content/drive/MyDrive/stock-prob`.

---

## Dynamic tickers (design rule)

Core modules (`features`, `models`, `backtest`, `labels`, `scoring`) **contain no pilot symbols**.

Universe is always supplied as data:

```yaml
# configs/universe_idx_example.yaml
universe:
  equities: [BBCA.JK]      # change freely
  domestic_index: "^JKSE"
  us_index: "^GSPC"
  macro: "^VIX"
```

---

## Method (short)

| Step | What |
|------|------|
| Features | Log returns, rolling β/ρ to indexes, vol, momentum, macro Δ |
| Direction | Logistic regression → calibrated-style `P(up)` per horizon |
| Cone | Monte Carlo paths; lab path uses **GARCH(1,1)** vol + regime width |
| Validation | Expanding walk-forward; score on embargo grid |
| Baselines | Base rate, momentum — always on the board |
| Lab extras | IDX overnight spillover, model tournament, form-based twins, honesty skill |

**Metrics:** Brier (primary), hit rate (secondary), cone coverage @80%, sharpness, honesty skill.

---

## Project layout

```
src/stock_prob/          # library
  ingest.py features.py models.py backtest.py scoring.py
  pipeline.py lab.py garch.py regime.py spillover.py
  tournament.py panel.py twins.py honesty.py surface.py viz.py
configs/                 # example universes (data, not code)
scripts/run_demo.py
scripts/run_full_lab.py
tests/
docs/
```

---

## Tests

```bash
PYTHONPATH=src pytest tests/ -v
```

Covers: dynamic two-universe path, no look-ahead labels, embargo, finite live `P(up)` for all horizons, GARCH/regime/tournament/twins/honesty.

---

## Roadmap status

| Wave | Scope | Status |
|------|--------|--------|
| 0–3 | Dynamic core, WF, reports, ledger | **Done** |
| 4 | GARCH, regime, spillover, tournament | **Done** |
| 5 | Multi-ticker panel, IDX vs US skill | **Done** |
| 6 | Twins, honesty, probability surface | **Done** |
| 7 | OSS packaging + GitHub | **Done** |

---

## Citation

If this helps a portfolio or paper:

```
Stock Probability Engine (SPE) — multi-horizon equity probability lab.
https://github.com/SeraKah-1/stock-probability-engine
```

---

## License

MIT — see [LICENSE](LICENSE).
