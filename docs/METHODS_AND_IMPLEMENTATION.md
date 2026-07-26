# Prism — Methods and Implementation

Technical record of what the repository implements, which data-science and statistical methods are used, and how pieces connect. This is documentation of the current codebase, not a product brief.

Package path: `src/stock_prob/`.  
Primary entrypoints: `pipeline.run_pipeline`, `lab.enrich_single`, `ui_colab.run_once` / `ui_gradio.launch_gui`.

---

## 1. Scope of the system

Prism is a multi-horizon equity research lab. For a user-supplied ticker it:

1. Fetches OHLCV (via Yahoo Finance through `yfinance`) and optional context series (domestic index, US index, macro).
2. Builds point-in-time features.
3. Fits a directional probability model \(P(\text{up})\) at each chosen horizon \(H\).
4. Builds a forward price range (prediction cone / fan chart) via Monte Carlo (and optionally GARCH volatility).
5. Evaluates the probability model out-of-sample with walk-forward scoring against simple baselines.
6. Optionally classifies market regime, estimates support/resistance, summarizes a research stance, and simulates entry P&amp;L under the same path model.
7. Writes artifacts (CSV/Parquet/JSON/HTML) and serves a Gradio UI with matplotlib SVG charts.

Tickers are never hardcoded into model logic. Symbols come from runtime config, search, or CLI arguments.

What it is not:

- Not a live trading bot or order router.
- Not a guarantee of future prices.
- Not a pure template/JSON chart injector; charts are rendered from fitted model outputs (matplotlib → inline SVG).

---

## 2. Data pipeline

### 2.1 Ingest

| Step | Module | Behavior |
|------|--------|----------|
| Symbol search / resolve | `tickers.py` | Query Yahoo search API; map UI labels to Yahoo symbols (e.g. `AMMN.JK`). |
| Download | `ingest.fetch_symbol` / `fetch_universe` | Daily bars for equities and context (`period` typically multi-year / `max` on heal). |
| Cache | local parquet/CSV under project `data/` | Prefer longest history; short stub caches (&lt; ~40 bars) are force-refreshed so corrupt caches do not poison runs. |
| Align | `ingest.align_close_panel` | Close-price panel aligned on dates across symbols. |

### 2.2 Features (point-in-time)

Module: `features.py`.

All rolling statistics at date \(t\) use only observations \(\le t\) (no peeking into the label horizon).

| Feature family | Definition (conceptual) |
|----------------|-------------------------|
| Log return | \(r_t = \log P_t - \log P_{t-1}\) |
| Momentum | Simple returns over 5 and 21 sessions: \(P_t/P_{t-k}-1\) |
| Realized volatility | Rolling standard deviation of log returns (21d, 60d) |
| Rolling beta | \(\widehat\beta = \mathrm{Cov}(r, r_{\text{idx}}) / \mathrm{Var}(r_{\text{idx}})\) on a trailing window |
| Rolling correlation | Trailing Pearson corr of equity returns vs domestic / US index |
| Macro | Level and % change of macro series (e.g. VIX, DXY) when configured |

Feature matrix columns exclude raw `close` when training; `close` is kept on the frame for labeling and charts.

### 2.3 Labels

Module: `labels.py`.

For horizon \(H\) (trading days):

\[
R_{t,H} = \frac{P_{t+H}}{P_t} - 1, \qquad
y_{t,H} = \mathbf{1}\{R_{t,H} &gt; 0\}
\]

Incomplete end-of-sample rows (no \(P_{t+H}\)) are dropped.

**Embargo / non-overlapping evaluation samples:**  
When scoring OOS points, sample starts are thinned by step \(H\) (`apply_embargo`) so heavily overlapping forward windows do not dominate the walk-forward metrics. Training can still use denser rows for sample size.

---

## 3. Directional probability model

### 3.1 Model form

Module: `models.fit_logistic`, used by `forecast.compute_live_forecast` and `backtest.walk_forward_equity`.

- Classifier: **logistic regression** (`sklearn.linear_model.LogisticRegression`).
- Preprocessing: **standardization** of features (`StandardScaler`) inside a `Pipeline`.
- Class weights: `balanced` to reduce bias when up/down frequencies differ.
- Target class \(1\) = price up over horizon \(H\).
- Live output: \(P(y=1 \mid x_t)\) clipped to \((10^{-4}, 1-10^{-4})\).

This is a classical probabilistic classifier, not a neural sequence model and not a random number printer. Coefficients are estimated by maximum likelihood (sklearn default solver, `max_iter=500`).

### 3.2 Live forecast path

Module: `forecast.compute_live_forecast`.

For each configured \(H\):

1. Build supervised set from features + labels.
2. Fit logistic on all available labeled history (with minimum train size guard).
3. Score the latest feature row → `live_probs[H]`.
4. Independently build a Monte Carlo cone (below) so the chart does not depend on logistic success.

Failure modes and fallbacks (logged in `forecast_errors`):

| Condition | Behavior |
|-----------|----------|
| Too few train rows | Skip prob; error string recorded |
| Single class in \(y\) | Fall back to empirical base rate of \(y\), clipped |
| Fit / predict exception | Optional weak momentum fallback (0.55 / 0.45) |

### 3.3 Baselines for comparison

Module: `baselines.py`, scored in walk-forward:

| Baseline | Meaning |
|----------|---------|
| **Base rate** | \(\hat p = \overline{y}\) on the training labels up to the OOS date |
| **Momentum** | Fixed naive rule: ~0.55 if recent momentum &gt; 0 else ~0.45 |

The model is considered more useful when its Brier score is **lower** than the base rate on the same OOS set.

---

## 4. Walk-forward evaluation

Module: `backtest.walk_forward_equity`.

Design:

- **Expanding training window** (all past labeled rows up to index \(i\)).
- **Refit cadence** controlled by `walkforward_refit_every` (larger in GUI “fast” mode).
- **OOS points** restricted by embargo mask (spacing \(H\)).
- Optional cap `max_oos_per_horizon` for speed on Colab.
- Optional in-loop Monte Carlo cone diagnostics (`cone_diagnostics`).

Stored per OOS row: date, horizon, `prob_up_model`, `prob_up_base`, `prob_up_momentum`, `y_true`, `fwd_ret`, `close`.

### 4.1 Scoring rules

Module: `scoring.py`.

| Metric | Formula / use |
|--------|----------------|
| **Brier score** | \(\frac{1}{n}\sum_i (p_i - y_i)^2\) — lower is better |
| **Log loss** | Binary cross-entropy with probability clipping |
| **Hit rate** | Fraction of days where \(\mathbf{1}\{p \ge 0.5\} = y\) |
| **Brier skill** (UI) | \(\text{Brier}_{\text{base}} - \text{Brier}_{\text{model}}\) — positive means model beats base rate |
| **Cone coverage** | Share of realized prices inside [lower, upper] band |
| **Cone sharpness** | Mean band width (optionally relative to mid price) |
| **Reliability bins** | Calibration table: mean predicted \(p\) vs empirical frequency of up |

Metrics are exported to run artifacts (`metrics.csv` / parquet / excel).

---

## 5. Price range (cone / fan chart)

### 5.1 Monte Carlo GBM

Module: `models.monte_carlo_cone` / `cone_table`.

Daily log-return simulation under geometric Brownian motion style dynamics:

\[
r_{i,t} = \big(\mu - \tfrac12 \sigma^2\big) + \sigma\, Z_{i,t}, \quad
Z_{i,t} \sim \mathcal N(0,1) \ \text{i.i.d.}
\]

\[
P_{i,k} = P_0 \exp\!\Big(\sum_{t=1}^{k} r_{i,t}\Big), \quad k=1,\ldots,H
\]

Parameters \(\mu,\sigma\) (daily):

- Default live path: sample mean and sample std of the last ~60 log returns.
- Advanced / GARCH path: conditional vol and mean from GARCH (below).

From \(N\) paths, day-wise quantiles form bands: **p10, p25, p50, p75, p90**.  
Terminal distribution (prices at step \(H\)) is retained for entry simulation.

Calendar axis for the cone uses business-day ranges starting the day after the last bar (`pd.bdate_range`).

### 5.2 GARCH volatility (Advanced lab)

Module: `garch.py` (`arch` package when available).

- Fit **GARCH(1,1)** with constant mean on log returns (scaled to percent for numerical stability).
- One-step conditional variance forecast → daily \(\sigma\).
- Fallback: sample mean/std if fit fails.
- Regime multiplier can widen/narrow bands around the median after simulation (`regime_vol_multiplier`).

### 5.3 Chart construction (presentation)

Module: `viz.build_fan_figure` (matplotlib Agg → SVG).

Display rules (not additional statistical models):

- History zoom ≈ \(3 \times H\) bars so short horizons remain readable.
- Vertical as-of line and last price annotation.
- End-of-horizon labels for p10 / p50 / p90.
- Optional nested cones for other selected horizons.
- Optional support/resistance horizontal levels.
- Soft color split above/below last price for readability.

Gradio embeds **inline SVG** (no Plotly JS/CDN) because HTML script execution in Gradio is unreliable in Colab.

---

## 6. Market structure and regime

### 6.1 Volatility / correlation regime

Module: `regime.py`.

Rule-based labels on trailing realized vol (and optional correlation to US equities):

| Label | Intent |
|-------|--------|
| `CALM` | Vol below elevated threshold |
| `ELEVATED` | Vol near high historical quantile |
| `PANIC` | Vol very high and/or high US corr with elevated vol |
| `UNKNOWN` | Insufficient data |

Used for band width multipliers and UI context. Standard pipeline mode now attaches the latest regime as well.

### 6.2 Market character (trend × vol)

Module: `structure.market_character`.

| Axis | Inputs |
|------|--------|
| Trend | Sign/magnitude of ~21d and ~63d simple returns → up / down / sideways |
| Vol | Percentile of trailing 21d realized vol in its own history → low / normal / high |
| Compression | Short-window price range small relative to longer window |

Combined labels examples: `quiet range`, `volatile uptrend`, `volatility compression (range)`.

### 6.3 Support and resistance

Module: `structure.support_resistance`.

1. Detect swing highs/lows with a left/right fractal window.
2. Seed with recent rolling min/max (21 / 63 / 126).
3. Cluster nearby prices (relative tolerance).
4. Split into supports below last price and resistances above; keep strongest by touch count and proximity.

These levels are **technical structure**, not model validation. Model honesty remains walk-forward Brier vs baselines and (optionally) historical cone coverage.

---

## 7. Research decision summary

Module: `decision.build_decision`.

Combines:

- Primary-horizon \(P(\text{up})\) and distance from 0.5.
- Whether the model beats base rate on a majority of scored horizons.
- Proximity to nearest support / resistance.
- Market character string.

Outputs a coarse **action** (`LEAN_LONG`, `BUY_ZONE`, `WAIT`, `WEAK_SIGNAL`, `LEAN_SHORT`, `AVOID_LONG`, `NO_SIGNAL`), plain-language summary, and optional buy/sell **reference zones** (price bands for research display only).

This layer does not place trades and is not optimized as a trading strategy; it is a deterministic rule stack on top of model and structure outputs.

---

## 8. Entry simulation

Module: `simulate.simulate_entry`.

User supplies entry price, horizon, and side (long/short). The same GBM Monte Carlo engine simulates \(N\) paths from the entry:

| Output | Definition |
|--------|------------|
| Exit quantiles | p10…p90 of terminal price |
| Return quantiles | Path returns under long or short definition |
| \(P(\text{profit})\) | Fraction of paths with positive side-specific return |
| MC \(P(\text{up})\) | Fraction of terminals above entry (price up) |
| Model \(P(\text{up})\) | Logistic probability for that horizon, shown for comparison |

Note: logistic \(P(\text{up})\) and Monte Carlo \(P(P_H &gt; P_0)\) can disagree; they use different information (features vs drift/vol of returns).

---

## 9. Optional lab enrichments (Advanced)

Module: `lab.enrich_single` and related modules.

| Component | Module | Method summary |
|-----------|--------|----------------|
| GARCH cones | `garch.py` | Conditional vol for MC cones |
| Spillover (IDX) | `spillover.py` | Logistic / regression link of local moves to US returns (when applicable) |
| Tournament | `tournament.py` | Score model variants on WF preds; inverse-Brier blend weights; champion label |
| Honesty aggregate | `honesty.py` | Aggregate skill / honesty frame from predictions |
| Probability surface | `surface.py` | Tabular reshape of multi-horizon probs |
| Panel / twins | `panel.py`, `twins.py` | Multi-name runs; form-factor vectors and simple clustering of “twin” names |

Standard GUI mode prioritizes `run_pipeline` (logistic + sample-vol MC + walk-forward + structure). Advanced full speed uses `enrich_single`.

---

## 10. Horizon key handling

Module: `horizon_keys.py`.

Internal contract: probability maps use **bare digit keys** as strings (`"5"`, `"21"`).  
Prefixes like `h5` / `ens_21` are normalized when present so `int(key)` never receives a non-numeric string in report/UI paths.

---

## 11. Presentation architecture

| Layer | Role |
|-------|------|
| `pipeline` / `lab` | Compute results dict |
| `viewmodel.build_viewmodel` | Normalize history, cones, metrics, structure, decision; session export for UI state |
| `viz` | Matplotlib figures → SVG HTML fragments |
| `report_html.write_prism_report` | Self-contained HTML report |
| `ui_gradio` | Gradio Blocks: search, run (generator progress), horizon radio, entry sim, downloads |

Session state after a run stores serializable history/cones/probs/μ/σ so horizon switching and entry simulation do not re-fetch or re-fit.

---

## 12. Configuration and speed modes

Module: `config.RunConfig`, `ui_colab.run_once`.

| Mode | Typical settings |
|------|------------------|
| **fast** (GUI Standard) | Fewer MC paths, coarser refit, limited OOS points, cone diagnostics off in WF loop |
| **full** (GUI Advanced full) | More paths, denser WF, lab enrichments |

Colab free tier is CPU-bound; logistic / GARCH / pure-Python MC do not meaningfully use GPU.

---

## 13. Artifacts written per run

Under `runs/<run_id>/` (names may vary slightly by mode):

- `run_config.yaml` / `config.yaml`
- `predictions` (csv/parquet)
- `metrics` (csv/json/parquet)
- `live_probs.csv`
- `history` (csv/parquet)
- `export.xlsx`
- `report.html` / `prism_report.html` / `report_lab.html`
- Optional: `garch_cone_*.csv`, `structure.json`, `model_meta.json`, tournament/honesty/spillover JSON/CSV

Ledger append (`ledger.py`) records prediction rows across runs when enabled.

---

## 14. Verification that computation is real

Automated checks under `tests/` include:

| Test area | What is asserted |
|-----------|------------------|
| `test_model_real.py` | Logistic probabilities vary across rows (not a constant stub); MC terminal dispersion &gt; 0; p10 &lt; p50 &lt; p90; entry sim returns finite stats; structure and decision produce structured outputs |
| `test_horizon_keys.py` | Key normalization and report paths under mixed keys |
| `test_fan_chart_v8.py` | SVG embed; session round-trip for cones |
| `test_pipeline_smoke.py` | End-to-end pipeline smoke |
| `test_ticker_fetch.py` | Dynamic ticker fetch behavior |

Manual E2E (example): `run_once("AMMN.JK", …)` returns non-empty `live_probs`, `live_cones`, `mu`/`vol`, regime, structure levels, and a decision action.

---

## 15. Method inventory (quick reference)

| Domain | Methods used in code |
|--------|----------------------|
| Returns | Log returns, simple multi-day returns |
| Dependence | Rolling covariance, beta, Pearson correlation |
| Volatility | Sample realized vol; GARCH(1,1) (optional) |
| Supervised learning | Logistic regression + feature standardization |
| Class imbalance | `class_weight="balanced"` |
| Label leakage control | Point-in-time features; embargo spacing for OOS |
| Validation design | Expanding-window walk-forward |
| Probabilistic scoring | Brier, log loss, hit rate, reliability bins |
| Baselines | Historical base rate; momentum heuristic |
| Path simulation | Monte Carlo GBM-style multi-path quantiles |
| Regime | Rule-based vol/corr thresholds |
| Technical structure | Swing pivots, level clustering |
| Decision layer | Deterministic rules on probs + edge + levels |
| Visualization | Fan chart quantiles; Brier bar comparison; probability bars |

Not implemented (explicitly out of current scope unless added later): full Bayesian hierarchical models, transformers/RNNs, options-implied vol surfaces as primary signal, automated execution, portfolio optimization.

---

## 16. Limitations (statistical and practical)

1. **Stationarity:** Equity return distributions change; expanding-window logistic can lag regime shifts.
2. **Overlapping horizons:** Embargo reduces but does not eliminate dependence between multi-horizon evaluations.
3. **GBM cones:** Constant \(\mu,\sigma\) (or single GARCH \(\sigma\)) ignore jumps, gaps, and discrete corporate actions beyond adjusted prices from the data vendor.
4. **Calibration:** Brier skill can be negative; UI must treat those signals as weak.
5. **Data vendor:** Yahoo-sourced bars can have gaps, adjustments, or delays; cache heal only addresses trivial stub files.
6. **Decision / S/R:** Heuristic overlays; not backtested as a full trading system in this document.
7. **UI language drift:** Colab sessions may serve stale cloned trees; build stamp `UX_BUILD` in `design.py` identifies the loaded UI version.

---

## 17. Main source map

| Concern | Primary files |
|---------|----------------|
| Config | `config.py` |
| Fetch / cache | `ingest.py`, `tickers.py` |
| Features / labels | `features.py`, `labels.py` |
| Logistic + MC | `models.py`, `forecast.py` |
| Walk-forward | `backtest.py`, `baselines.py`, `scoring.py` |
| GARCH / regime | `garch.py`, `regime.py` |
| Structure / decision / sim | `structure.py`, `decision.py`, `simulate.py` |
| Orchestration | `pipeline.py`, `lab.py` |
| View / charts / report | `viewmodel.py`, `viz.py`, `report_html.py` |
| UI | `ui_gradio.py`, `ui_colab.py` |
| Model description helper | `model_info.py` |

---

## 18. Document control

| Field | Value |
|-------|--------|
| Subject | Methods and implementation status of Prism (`stock_prob`) |
| Language | English |
| Audience | Maintainers and technical reviewers |
| Related | `docs/USAGE_AND_VIZ.md`, `docs/COMPREHENSIVE_PLAN.md`, `CODE_OF_ETHICS.md` |

When behavior changes (new model class, new scoring rule, different cone dynamics), update this file in the same PR as the code change.
