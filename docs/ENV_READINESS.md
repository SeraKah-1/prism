# Environment Readiness — Stock Probability Engine

Generated: 2026-07-26 | Runtime: Google Colab CPU

## 1. What this machine is

| Item | Value |
|---|---|
| Host | Google Colab (Docker), Linux x86_64 |
| Image | `COLAB_IMAGE_TYPE=cpu` (no GPU — correct for this project) |
| CPU | 2 vCPU |
| RAM | ~12 GB |
| Disk | ~88 GB free under `/` |
| Python | 3.12.13 |
| Workspace | `/content` |
| Project (local) | `/content/stock-prob` |
| Drive | **Not mounted** (`/content/drive` missing) |

**Mental model:** Colab is the kitchen (ephemeral compute). Without Drive/Git, everything dies when the runtime is recycled. Free tier: long jobs can be interrupted; idle disconnects; not a always-on server.

### Colab “effectiveness” rules (legitimate ops, not abuse)

1. **CPU only** — GPU wastes quota; logistic/GARCH/MC do not need it.
2. **Cache everything** — yfinance once → parquet; never re-download the full history each cell.
3. **Idempotent sessions** — every session: install extras → load parquet → compute → write artifacts.
4. **Persist off-runtime** — Drive and/or GitHub; local `/content` is temporary.
5. **Avoid heavy parallel Yahoo hits** — sequential + sleep + incremental fetch.
6. **Plotly renderer** — `plotly.io.renderers.default = 'colab'`.
7. **Do not treat Colab as a live monitor** — max ~12h, idle kills; journal is offline scoring.

---

## 2. What already works (autonomous — no user auth)

### Packages present

| Package | Status | Use |
|---|---|---|
| yfinance 0.2.66 | OK | prices / indices |
| pandas, numpy, scipy | OK | core |
| scikit-learn | OK | logistic + calibration |
| statsmodels | OK | stats / later DM tests |
| plotly | OK | fan charts |
| pyarrow | OK | parquet |
| gspread | OK install | needs **user OAuth** to write Sheets |
| ipywidgets | OK | UI |
| openpyxl | OK | Excel export |
| lightgbm | OK | season 2+ only if needed |
| torch / tensorflow | present, **unused** | do not use for MVP |
| arch | **installed** (session) | GARCH season 2 |
| kaleido 0.2.1 | **installed** | static chart export |
| xlsxwriter | **installed** | multi-tab Excel |
| curl_cffi | **installed** | yfinance resilience |

### Data sources tested live

| Source | Result | Notes |
|---|---|---|
| **yfinance** | **PASS** | AAPL, ^GSPC, BBCA.JK, ^JKSE, ^VIX, DX-Y.NYB — 5y daily OK |
| Yahoo chart HTTP + UA | PASS | 200 |
| Stooq CSV URL | FAIL (bot JS/PoW) | skip for now; not needed |
| FRED site | reachable | optional VIXCLS later; need free API key only if using `fredapi` |
| Direct bare Yahoo (no UA) | 429 | always send browser-like client via yfinance |

### Smoke data already cached locally

```
/content/stock-prob/data/
  BBCA.JK.parquet   ~1203 rows (2021-07-26 → 2026-07-24)
  JKSE.parquet
  AAPL.parquet      ~1255 rows
  GSPC.parquet
  VIX.parquet
  DXY.parquet
```

---

## 3. What I can do alone (no user action)

- Install/upgrade pip packages in this runtime
- Create project tree under `/content/stock-prob`
- Download & cache market data via yfinance
- Implement full MVP pipeline (features, logistic, MC cone, walk-forward, plots, export Excel)
- Write notebooks + modular `src/*.py`
- Export artifacts to `/content/stock-prob/exports`
- Use git locally inside the container (no remote until token/repo)

---

## 4. What needs YOU (blocking vs optional)

### Blocking for “survives runtime death”

| Need | Why | How you give it |
|---|---|---|
| **Google Drive mount** | Permanent storage for data/ledger | In Colab: run `drive.mount` and approve the OAuth popup, **or** tell me to print the mount cell for you to run |
| *(alt)* GitHub repo + token | Same persistence via git push | Create empty repo + PAT with `repo` scope; paste once via Colab Secrets / chat (prefer Secrets) |

Without one of these, work is fine **inside this session**, but **gone** on reconnect.

### Optional (nice, not MVP-blocking)

| Need | Why | How |
|---|---|---|
| Google Sheets OAuth | Mobile journal / self-score mirror | Colab `auth.authenticate_user()` popup, then share sheet to your account |
| FRED API key | Official VIX/macro series | Free at https://fred.stlouisfed.org/docs/api/api_key.html — only if we leave Yahoo VIX |
| Paid market API | Higher reliability / fundamentals | Alpha Vantage / Polygon / Tiingo keys — **not required** for MVP |
| Captcha-solving services | Stooq-style walls | **Out of scope** — we do not need Stooq; yfinance works |

**Not needed:** GPU, High-RAM runtime, service accounts, proxy farms, captcha solvers, scraping behind logins.

---

## 5. Recommended persistence strategy

```
PRIMARY (MVP):  Google Drive  →  /content/drive/MyDrive/stock-prob/
FALLBACK now:   /content/stock-prob/   (session-local — already used)
LATER:          GitHub for code only; parquet/ledger stay on Drive
Sheets:         mirror of ledger only (Fase 2)
```

Sync plan after you mount Drive:

1. `drive.mount('/content/drive')`
2. Copy `/content/stock-prob` → `MyDrive/stock-prob` (or bootstrap creates there)
3. All subsequent I/O uses Drive path

---

## 6. Data API policy for this project

| Priority | Provider | Auth | Role |
|---|---|---|---|
| 1 | **yfinance** (Yahoo) | none | prices, indices, VIX, DXY |
| 2 | Cached parquet | none | all re-runs |
| 3 | FRED (optional) | free key | macro backup |
| skip | Stooq from this IP | bot wall | avoid |
| skip | Unofficial captcha-break scrapers | risk/ToS | avoid |

Fundamentals (Piotroski-style): deferred; yfinance fundamentals for IDX are incomplete.

---

## 7. Project tree (prepared)

```
/content/stock-prob/
├── requirements.txt
├── data/*.parquet          # pilot cache
├── predictions/            # ledger (empty)
├── exports/
├── notebooks/
├── docs/ENV_READINESS.md   # this file
└── src/bootstrap_colab.py
```

---

## 8. Status checklist

- [x] Colab CPU environment mapped
- [x] Core packages verified / extras installed
- [x] yfinance live test IDX + US + VIX + DXY
- [x] Local project scaffold + 5y parquet cache
- [x] Bootstrap helper written
- [ ] Drive mounted (needs you)
- [ ] Sheets auth (optional, later)
- [ ] GitHub remote (optional)
- [ ] MVP code (waiting for your go)

---

## 9. Next actions

**You (1 click path):**

1. Mount Drive in Colab UI / cell when ready (I will give exact cell).
2. Say: “lanjut MVP” (or “copy ke Drive dulu lalu MVP”).

**Me after go:**

1. Sync project to Drive if mounted  
2. Implement Fase 0–1 per plan: ingest incremental → features → logistic + MC → walk-forward + embargo → fan chart → ledger → export
