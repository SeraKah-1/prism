# Cara pakai & visualisasi SPE

## 1. Tiga cara pakai (tanpa bikin website)

| Mode | Kapan | Command / entry |
|------|--------|------------------|
| **Colab GUI** | Lihat chart + ganti ticker klik-klik | `notebooks/01_lab_gui.ipynb` → `launch_gui()` |
| **TUI terminal** | Cepat di shell, teks + path HTML | `python scripts/tui_lab.py` |
| **CLI** | Otomasi / batch | `python scripts/run_demo.py ...` |

Tidak perlu Streamlit, Flask, atau hosting.

---

## 2. Colab GUI (disarankan)

1. Buka Colab, mount Drive (project di `MyDrive/stock-prob`) **atau** clone repo.
2. Buka `notebooks/01_lab_gui.ipynb`.
3. Jalankan cell setup, lalu:

```python
from stock_prob.ui_colab import launch_gui
launch_gui(default_equity="BBCA.JK")
```

4. Isi **Equity / Dom index / US index / Macro** → pilih horizons → **Run analysis**.
5. Di bawah form muncul:
   - kartu meta (regime, champion, spillover)
   - tabel **P(up)** live
   - tabel **Brier vs baseline**
   - **fan chart** Plotly (cone)
   - probability surface

Ganti ticker kapan saja (mis. `TLKM.JK`, `MSFT`) — tidak hardcode di library.

### API tanpa tombol

```python
from stock_prob.ui_colab import run_once, render_result
res = run_once("BBCA.JK", domestic_index="^JKSE", us_index="^GSPC", macro="^VIX")
render_result(res)
```

---

## 3. Visualisasi apa saja

| Visual | Isi | File / komponen |
|--------|-----|------------------|
| Fan chart | Harga + cone p10–p90 | Plotly di GUI / `report.html` |
| Metrics bars | Brier model vs base vs momentum | GUI + report |
| Regime badge | CALM / ELEVATED / PANIC | lab enrich |
| Lab HTML | GARCH cone + meta | `runs/*/report_lab.html` |
| Panel index | IDX vs US skill + twins | `runs/full_lab_*/index.html` |
| TradingView | Overlay/proxy di chart live | folder `tradingview/*.pine` |

Buka HTML dari file browser Colab / download dari Drive — **bukan** server web SPE.

---

## 4. TradingView

Lihat `tradingview/README.md`.

Ringkas: copy `SPE_Triangulation.pine` / `SPE_Regime_ProbProxy.pine` / `SPE_Cone_Bands.pine` ke Pine Editor → Add to chart.

Proxy TV **bukan** pengganti skor Brier lab Python.

---

## 5. CLI cepat

```bash
export PYTHONPATH=src   # atau /content/stock-prob/src
python scripts/run_demo.py --equities BBCA.JK --domestic-index '^JKSE' --us-index '^GSPC' --macro '^VIX'
python scripts/run_full_lab.py --config configs/panel_core.yaml
python scripts/tui_lab.py
```

---

## 6. Alur data (ingat)

```
ticker input → yfinance cache parquet → features → model/cone
    → walk-forward metrics → (opsional) GARCH/regime/tournament
    → tampil di GUI / HTML di folder runs/
```
