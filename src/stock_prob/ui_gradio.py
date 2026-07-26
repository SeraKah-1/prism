"""
Prism Gradio UI — human-first layout (shadcn-like cards), reliable charts & files.

Layout model (inspired by shadcn dashboard + card primitives):
  page > section.card > section.head + section.body
  fixed gap, 1px border, clear frames so plots never bleed into each other.

Files: always copy reports into /tmp/prism_gradio_exports for gr.File
  (Gradio blocks paths outside cwd/temp unless allowed_paths).

Copy: pure HTML for tips (no Markdown ** inside HTML that shows as asterisks).
"""
from __future__ import annotations

import shutil
import tempfile
import traceback
import uuid
from pathlib import Path
from typing import Any

import pandas as pd

from stock_prob.design import UX_BUILD
from stock_prob.report_html import write_prism_report
from stock_prob.tickers import (
    parse_symbol_from_label,
    resolve_ticker,
    search_labels,
    search_tickers,
)
from stock_prob.ui_colab import run_once
from stock_prob.viewmodel import build_viewmodel
from stock_prob.viz import build_fan_figure, build_metrics_figure, build_prob_gauge_figure

# —— Human labels (Yahoo symbols only behind the scenes) ——
INDEX_PRESETS = {
    "Indonesia · IHSG (Jakarta)": {
        "domestic_index": "^JKSE",
        "us_index": "^GSPC",
        "macro": "^VIX",
        "hint": "Untuk saham IDX. Acuan: IHSG + pengaruh S&P 500 + suasana pasar (VIX).",
    },
    "United States · S&P 500 + VIX": {
        "domestic_index": "^GSPC",
        "us_index": "^GSPC",
        "macro": "^VIX",
        "hint": "Untuk saham AS. Acuan: S&P 500 + volatilitas pasar (VIX).",
    },
    "Global · S&P 500 + USD (DXY)": {
        "domestic_index": "^GSPC",
        "us_index": "^GSPC",
        "macro": "DX-Y.NYB",
        "hint": "S&P 500 + kekuatan dollar (sering relevan emerging market).",
    },
}

HORIZON_CHOICES = [
    ("5 hari (≈ 1 minggu)", 5),
    ("21 hari (≈ 1 bulan)", 21),
    ("63 hari (≈ 1 kuartal)", 63),
    ("126 hari (≈ 6 bulan)", 126),
    ("252 hari (≈ 1 tahun)", 252),
]

GRADIO_EXPORT_DIR = Path(tempfile.gettempdir()) / "prism_gradio_exports"


def _gradio_safe_file(src: Path) -> str:
    """Copy into /tmp so gr.File accepts it (cwd may be /content/prism)."""
    GRADIO_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    dst = GRADIO_EXPORT_DIR / f"{src.stem}_{uuid.uuid4().hex[:8]}{src.suffix}"
    shutil.copy2(src, dst)
    return str(dst)


# —— CSS: card primitives, clear frames, no sticky overlap ——
PRISM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,400;9..40,500;9..40,600;9..40,700&display=swap');

:root {
  --bg: #f6f3ee;
  --card: #fffcf7;
  --fg: #1c1917;
  --muted: #78716c;
  --border: #e7e0d5;
  --accent: #0f3d3e;
  --radius: 16px;
  --gap: 16px;
  --ease: cubic-bezier(0.16, 1, 0.3, 1);
}

.gradio-container {
  max-width: 960px !important;
  margin: 0 auto !important;
  font-family: "DM Sans", system-ui, sans-serif !important;
  background:
    radial-gradient(900px 380px at 0% 0%, #e4efe9 0%, transparent 55%),
    radial-gradient(720px 340px at 100% 0%, #f3e9df 0%, transparent 50%),
    var(--bg) !important;
  color: var(--fg) !important;
  padding: 12px 12px 48px !important;
}
footer { display: none !important; }

/* kill markdown asterisk glitches: we mostly use HTML */
.prose, .markdown-body { max-width: 100% !important; }

.hero { padding: 8px 4px 12px; animation: rise 0.55s var(--ease) both; }
.hero .kicker {
  font-size: 11px; letter-spacing: 0.14em; text-transform: uppercase;
  font-weight: 600; color: var(--accent); margin: 0 0 6px;
}
.hero h1 {
  margin: 0 0 8px !important; font-size: clamp(1.7rem, 3vw, 2.2rem) !important;
  letter-spacing: -0.04em !important; font-weight: 700 !important; color: var(--fg) !important;
}
.hero p { margin: 0 !important; color: var(--muted) !important; line-height: 1.5; max-width: 56ch; }

/* shadcn-like card */
.card {
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  padding: 16px !important;
  margin: 0 0 var(--gap) !important;
  box-shadow: 0 1px 0 rgba(28,25,23,0.03), 0 14px 36px -28px rgba(28,25,23,0.35) !important;
  animation: rise 0.55s var(--ease) both;
}
.card + .card { margin-top: 0 !important; }

.section-title {
  margin: 0 0 4px !important;
  font-size: 0.95rem !important;
  font-weight: 650 !important;
  letter-spacing: -0.02em !important;
  color: var(--fg) !important;
}
.hint {
  margin: 0 0 12px !important;
  font-size: 12.5px !important;
  color: var(--muted) !important;
  line-height: 1.45 !important;
}

/* result frames — hard clip so plots stay inside */
.frame {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  overflow: hidden;
  margin: 0 0 var(--gap);
  box-shadow: 0 12px 32px -28px rgba(28,25,23,0.4);
  animation: rise 0.6s var(--ease) both;
}
.frame-head {
  padding: 12px 16px 8px;
  border-bottom: 1px solid var(--border);
  background: color-mix(in srgb, var(--card) 90%, #efeae2);
}
.frame-head h3 {
  margin: 0 !important; font-size: 0.92rem !important;
  letter-spacing: -0.02em !important; font-weight: 650 !important;
}
.frame-head p {
  margin: 4px 0 0 !important; font-size: 12px !important; color: var(--muted) !important;
}
.frame-body {
  padding: 10px 12px 14px;
  min-height: 80px;
  overflow: auto;
  max-width: 100%;
}
.frame-body .plotly-graph-div,
.frame-body .js-plotly-plot {
  max-width: 100% !important;
}

.summary {
  background: linear-gradient(125deg, #dcece7 0%, #fffcf7 52%, #f3ebe3 100%);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 16px 18px;
  margin-bottom: var(--gap);
  animation: rise 0.5s var(--ease) both;
}
.summary h2 {
  margin: 0 0 6px !important; font-size: 1.35rem !important;
  letter-spacing: -0.03em !important;
}
.summary .lead { margin: 8px 0 0; color: #44403c; line-height: 1.55; font-size: 0.97rem; }
.pills { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
.pill {
  font-size: 12px; font-weight: 600; border-radius: 999px;
  padding: 5px 11px; border: 1px solid var(--border); background: #fff; color: #57534e;
}
.pill.up { background: #e6f5ec; color: #1b7a4e; border-color: #b7e0c7; }
.pill.down { background: #fdecea; color: #b42318; border-color: #f2c1bc; }
.pill.ok { background: #e7f1ef; color: #0f3d3e; }

button.primary {
  border-radius: 12px !important;
  background: var(--accent) !important;
  font-weight: 600 !important;
  min-height: 44px !important;
}

/* spacing between Gradio rows inside a card */
.card .gr-row, .card .gap { gap: 10px !important; }

@keyframes rise {
  from { opacity: 0; transform: translateY(12px); }
  to { opacity: 1; transform: translateY(0); }
}

/* —— Live progress (always visible during run) —— */
.progress-panel {
  background: #fffcf7;
  border: 1px solid #e7e0d5;
  border-radius: 16px;
  padding: 18px 18px 16px;
  margin: 0 0 16px;
  box-shadow: 0 12px 32px -28px rgba(28,25,23,0.4);
  animation: rise 0.35s var(--ease) both;
}
.progress-panel .title {
  margin: 0 0 6px;
  font-size: 1.05rem;
  font-weight: 700;
  letter-spacing: -0.02em;
  color: #1c1917;
}
.progress-panel .detail {
  margin: 0 0 12px;
  font-size: 13px;
  color: #78716c;
  line-height: 1.45;
}
.progress-track {
  height: 10px;
  border-radius: 999px;
  background: #efeae2;
  overflow: hidden;
  border: 1px solid #e7e0d5;
}
.progress-fill {
  height: 100%;
  border-radius: 999px;
  background: linear-gradient(90deg, #0f3d3e, #2d6a4f);
  width: 0%;
  transition: width 0.35s var(--ease);
  position: relative;
}
.progress-fill.indeterminate {
  width: 40% !important;
  animation: slide 1.1s ease-in-out infinite;
}
@keyframes slide {
  0% { transform: translateX(-120%); }
  100% { transform: translateX(320%); }
}
.progress-meta {
  display: flex;
  justify-content: space-between;
  margin-top: 8px;
  font-size: 12px;
  color: #78716c;
  font-variant-numeric: tabular-nums;
}
.progress-steps {
  margin: 12px 0 0;
  padding: 0;
  list-style: none;
}
.progress-steps li {
  font-size: 12.5px;
  color: #a8a29e;
  padding: 4px 0 4px 18px;
  position: relative;
}
.progress-steps li::before {
  content: "";
  width: 8px; height: 8px; border-radius: 99px;
  background: #d6d3d1;
  position: absolute; left: 0; top: 9px;
}
.progress-steps li.done { color: #1b7a4e; }
.progress-steps li.done::before { background: #1b7a4e; }
.progress-steps li.active { color: #0f3d3e; font-weight: 600; }
.progress-steps li.active::before {
  background: #0f3d3e;
  box-shadow: 0 0 0 4px rgba(15,61,62,0.15);
  animation: pulse 1.2s ease infinite;
}
@keyframes pulse {
  0%, 100% { box-shadow: 0 0 0 3px rgba(15,61,62,0.12); }
  50% { box-shadow: 0 0 0 6px rgba(15,61,62,0.08); }
}
.status-bar {
  border: 1px solid #e7e0d5;
  background: #fffcf7;
  border-radius: 12px;
  padding: 10px 14px;
  font-size: 13px;
  color: #44403c;
  margin: 8px 0 0;
}
.status-bar.running { border-color: #a7c4be; background: #eef6f3; color: #0f3d3e; }
.status-bar.ok { border-color: #b7e0c7; background: #e6f5ec; color: #1b7a4e; }
.status-bar.err { border-color: #f2c1bc; background: #fdecea; color: #b42318; }
"""


def _fig_html(fig: Any, height: int = 440) -> str:
    if fig is None:
        return '<p class="hint" style="padding:12px">Grafik belum tersedia.</p>'
    try:
        fig.update_layout(height=height, margin=dict(l=44, r=18, t=44, b=40))
        return fig.to_html(
            full_html=False,
            include_plotlyjs="cdn",
            config={"displayModeBar": False, "responsive": True},
        )
    except Exception as e:
        return f'<p class="hint" style="padding:12px">Gagal merender grafik: {e}</p>'


def _frame(title: str, hint: str, body_html: str, delay_ms: int = 0) -> str:
    return f"""
<section class="frame" style="animation-delay:{delay_ms}ms">
  <div class="frame-head">
    <h3>{title}</h3>
    <p>{hint}</p>
  </div>
  <div class="frame-body">{body_html}</div>
</section>
"""


def _choices(q: str) -> list[str]:
    labels = search_labels(q or "", max_results=12)
    if labels:
        return labels
    q = (q or "").strip()
    return [q] if q else []


def on_type_search(query: str):
    import gradio as gr

    ch = _choices(query)
    val = ch[0] if ch else (query or None)
    return gr.Dropdown(choices=ch, value=val)


def on_pick_ticker(label: str, market_preset: str):
    sym = parse_symbol_from_label(label or "")
    if not sym:
        return market_preset, "Ketik nama perusahaan atau kode, lalu pilih saran."
    preset = (
        "Indonesia · IHSG (Jakarta)"
        if sym.upper().endswith(".JK")
        else "United States · S&P 500 + VIX"
    )
    hits = search_tickers(sym, max_results=2)
    name = hits[0].name if hits else sym
    # HTML status — no markdown asterisks
    msg = (
        f'<p class="hint" style="margin:0">Dipilih <b>{sym}</b> — {name}. '
        f"Pasar diset otomatis (boleh diganti).</p>"
    )
    return preset, msg


def on_market_change(preset: str):
    info = INDEX_PRESETS.get(preset) or next(iter(INDEX_PRESETS.values()))
    return f'<p class="hint" style="margin:0">{info["hint"]}</p>'


def _summary_html(vm, name: str) -> str:
    cards = vm.summary_cards()
    if not cards:
        lead = "Model belum menghasilkan probabilitas untuk horizon yang dipilih."
    else:
        c0 = cards[0]
        direction = "naik" if c0["direction"] == "up" else "turun"
        conf = "cukup condong" if c0["conviction"] >= 0.25 else "hampir netral"
        beat = sum(1 for c in cards if c.get("beat_baseline") is True)
        lead = (
            f"Untuk <b>{vm.ticker}</b>, model memperkirakan peluang harga "
            f"<b>{direction}</b> dalam ~<b>{c0['horizon']} hari</b> sekitar "
            f"<b>{c0['prob_up']*100:.0f}%</b> ({conf}). "
            f"Pada <b>{beat}/{len(cards)}</b> horizon, skor model lebih baik "
            f"daripada tebakan base-rate."
        )
    pills = [
        f'<span class="pill ok">suasana: {vm.regime}</span>'
    ]
    for c in cards:
        cls = "up" if c["direction"] == "up" else "down"
        beat = c.get("beat_baseline")
        tag = " · unggul base" if beat is True else (" · kalah base" if beat is False else "")
        pills.append(
            f'<span class="pill {cls}">{c["horizon"]} hari · '
            f'P(naik) {c["prob_up"]*100:.1f}%{tag}</span>'
        )
    price = f"{vm.last_price:,.2f}" if vm.last_price == vm.last_price else "—"
    return f"""
<div class="summary">
  <div class="kicker" style="font-size:11px;letter-spacing:.14em;text-transform:uppercase;font-weight:600;color:#0f3d3e">Hasil analisis</div>
  <h2>{vm.ticker} <span style="color:#78716c;font-weight:500;font-size:.72em">{name if name != vm.ticker else ""}</span></h2>
  <div class="hint" style="margin:0">Harga terakhir <b>{price}</b> · data s.d. <b>{vm.asof or "—"}</b></div>
  <p class="lead">{lead}</p>
  <div class="pills">{''.join(pills)}</div>
</div>
"""


def _loading_panel(pct: int, step: int, total: int, title: str, detail: str, steps: list[str] | None = None) -> str:
    """Big, always-visible progress UI (not a tiny Gradio spinner)."""
    pct = max(0, min(100, int(pct)))
    steps = steps or [
        "Validasi saham",
        "Ambil data harga",
        "Hitung model & walk-forward",
        "Susun grafik cone",
        "Siapkan laporan",
    ]
    items = []
    for i, label in enumerate(steps, start=1):
        cls = "done" if i < step else ("active" if i == step else "")
        items.append(f'<li class="{cls}">{i}. {label}</li>')
    return f"""
<div class="progress-panel">
  <p class="title">{title}</p>
  <p class="detail">{detail}</p>
  <div class="progress-track"><div class="progress-fill" style="width:{pct}%"></div></div>
  <div class="progress-meta"><span>Langkah {step}/{total}</span><span>{pct}%</span></div>
  <ul class="progress-steps">{''.join(items)}</ul>
</div>
"""


def _status(msg: str, kind: str = "running") -> str:
    return f'<div class="status-bar {kind}">{msg}</div>'


def _placeholder_frame(title: str, msg: str) -> str:
    body = (
        f'<div style="padding:28px 12px;text-align:center;color:#78716c;font-size:13px">'
        f'<div class="progress-track" style="max-width:220px;margin:0 auto 10px">'
        f'<div class="progress-fill indeterminate"></div></div>{msg}</div>'
    )
    return _frame(title, "Sedang diproses…", body)


def run_analysis(ticker_label, market_preset, horizon_labels, mode, progress=None):
    """
    Generator: yields intermediate UI so Colab never looks like a white screen.
    Gradio shows each yield immediately (progress panel + skeleton frames).
    """
    # Gradio injects Progress when the param is named progress; keep optional for tests
    try:
        import gradio as gr

        if progress is None:
            # no-op stub
            class _P:
                def __call__(self, *a, **k):
                    return None

            progress = _P()
    except Exception:
        pass
    idle = _frame("Menunggu", "Hasil akan muncul di bingkai ini.", "<p class='hint'>Belum ada data.</p>")
    empty_table = pd.DataFrame()

    def pack(summary, fan, gauge, met, table, report, status_html):
        return summary, fan, gauge, met, table, report, status_html

    try:
        # —— 0% immediate feedback ——
        yield pack(
            _loading_panel(4, 1, 5, "Memulai analisis…", "Menyiapkan pipeline. Jangan tutup tab ini."),
            _placeholder_frame("Kisaran harga (cone)", "Menunggu data harga…"),
            _placeholder_frame("Peluang naik", "Menunggu model…"),
            _placeholder_frame("Kejujuran model", "Menunggu evaluasi…"),
            empty_table,
            None,
            _status("0–5% · Memulai…", "running"),
        )
        if progress is not None:
            progress(0.05, desc="Memulai…")

        # —— validate ticker ——
        yield pack(
            _loading_panel(12, 1, 5, "Validasi saham…", f"Mencari: <b>{ticker_label or '—'}</b>"),
            _placeholder_frame("Kisaran harga (cone)", "Validasi ticker…"),
            _placeholder_frame("Peluang naik", "Validasi ticker…"),
            _placeholder_frame("Kejujuran model", "Validasi ticker…"),
            empty_table,
            None,
            _status("12% · Validasi saham (fetch live)…", "running"),
        )
        if progress is not None:
            progress(0.12, desc="Validasi saham…")

        resolved = resolve_ticker(ticker_label or "")
        if not resolved.get("ok"):
            err = _summary_html_error(
                "Saham tidak ditemukan",
                "Coba nama perusahaan (contoh: bank central asia) atau kode (BBCA, AAPL).",
            )
            yield pack(err, idle, idle, idle, empty_table, None, _status("Gagal: saham tidak ditemukan.", "err"))
            return

        symbol = resolved["symbol"]
        name = resolved.get("name") or symbol
        preset = INDEX_PRESETS.get(market_preset) or next(iter(INDEX_PRESETS.values()))
        domestic, us_index, macro = preset["domestic_index"], preset["us_index"], preset["macro"]
        if symbol.upper().endswith(".JK") and domestic == "^GSPC":
            domestic = "^JKSE"

        label_to_h = {lab: h for lab, h in HORIZON_CHOICES}
        hz: list[int] = []
        for lab in horizon_labels or []:
            if lab in label_to_h:
                hz.append(label_to_h[lab])
        if not hz:
            hz = [5, 21, 252]

        use_lab = str(mode).lower().startswith("lanjutan")
        mode_txt = "Lanjutan (lebih lama)" if use_lab else "Standar"

        # —— fetch + model (slow) ——
        yield pack(
            _loading_panel(
                35,
                2,
                5,
                f"Mengambil data & menghitung · {symbol}",
                f"{name} · mode {mode_txt} · horizon {hz}. "
                f"Langkah ini biasanya 15–45 detik (cache membuat run berikutnya lebih cepat).",
            ),
            _placeholder_frame("Kisaran harga (cone)", f"Mengunduh / cache harga {symbol}…"),
            _placeholder_frame("Peluang naik", "Menyiapkan fitur multi-faktor…"),
            _placeholder_frame("Kejujuran model", "Walk-forward belum jalan…"),
            empty_table,
            None,
            _status(f"35% · Hitung model {symbol} (bisa 15–45 dtk)…", "running"),
        )
        if progress is not None:
            progress(0.35, desc=f"Hitung model {symbol}…")

        result = run_once(
            symbol,
            domestic_index=domestic,
            us_index=us_index,
            macro=macro,
            horizons=hz,
            use_lab=use_lab,
            mc_paths=900,
        )

        # —— build viewmodel / charts ——
        yield pack(
            _loading_panel(
                78,
                4,
                5,
                "Menyusun grafik…",
                f"Model {symbol} selesai. Membuat cone, bar peluang, dan skor kejujuran.",
            ),
            _placeholder_frame("Kisaran harga (cone)", "Merender Plotly cone…"),
            _placeholder_frame("Peluang naik", "Merender bar…"),
            _placeholder_frame("Kejujuran model", "Merender Brier…"),
            empty_table,
            None,
            _status("78% · Merender grafik…", "running"),
        )
        if progress is not None:
            progress(0.78, desc="Merender grafik…")

        vm = build_viewmodel(result, name=name)
        if not vm.probs and (result.get("live_probs") is None):
            # hard empty result
            err = _summary_html_error(
                "Hasil kosong",
                "Pipeline selesai tapi tidak ada probabilitas. Coba horizon lebih pendek atau mode Standar.",
            )
            yield pack(err, idle, idle, idle, empty_table, None, _status("Selesai tanpa angka — coba lagi.", "err"))
            return

        if vm.art_dir:
            raw_report = Path(vm.art_dir) / "prism_report.html"
        else:
            GRADIO_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
            raw_report = GRADIO_EXPORT_DIR / f"prism_{uuid.uuid4().hex[:8]}.html"
        write_prism_report(vm, raw_report)
        safe_report = _gradio_safe_file(raw_report)

        h = vm.primary_horizon()
        cone = vm.cones.get(h)
        fan = None
        if vm.history is not None and len(vm.history) and cone is not None:
            fan = build_fan_figure(
                vm.history,
                cone,
                title=f"Kisaran harga · {vm.ticker} · {h} hari ke depan",
            )
        gauge = build_prob_gauge_figure(vm.probs, title="Peluang naik per horizon")
        met = None
        if vm.metrics is not None and len(vm.metrics):
            met = build_metrics_figure(
                vm.metrics,
                title="Brier vs tebakan sederhana (lebih rendah = lebih baik)",
            )

        summary = _summary_html(vm, name)
        fan_block = _frame(
            "Kisaran harga ke depan (cone)",
            "Garis putus-putus = median. Pita = rentang kemungkinan. Makin jauh, biasanya makin lebar.",
            _fig_html(fan, 480),
            40,
        )
        gauge_block = _frame(
            "Peluang naik (P up)",
            "50% = koin. Jarak dari 50% = seberapa condong model — bukan jaminan.",
            _fig_html(gauge, 280),
            80,
        )
        met_block = _frame(
            "Kejujuran vs tebakan sederhana",
            "Base-rate = frekuensi historis naik. Jika model tidak mengalahkannya, anggap sinyal lemah.",
            _fig_html(met, 320),
            120,
        )

        table = pd.DataFrame(
            [
                {
                    "Horizon": f"{c['horizon']} hari",
                    "Peluang naik (%)": round(c["prob_up"] * 100, 1),
                    "Arah": "Naik" if c["direction"] == "up" else "Turun",
                    "Lebih baik dari base-rate?": (
                        "Ya" if c["beat_baseline"] is True else ("Tidak" if c["beat_baseline"] is False else "—")
                    ),
                }
                for c in vm.summary_cards()
            ]
        )

        if progress is not None:
            progress(1.0, desc="Selesai")

        yield pack(
            summary,
            fan_block,
            gauge_block,
            met_block,
            table,
            safe_report,
            _status("100% · Selesai. Scroll untuk cone; unduh laporan HTML bila perlu.", "ok"),
        )
    except Exception:
        err = traceback.format_exc()
        msg = (
            f'<div class="summary"><h2>Gagal menghitung</h2>'
            f'<pre style="white-space:pre-wrap;font-size:11px;color:#78716c">{err[-1800:]}</pre></div>'
        )
        idle = _frame("Error", "Panel ini kosong karena proses gagal.", "<p class='hint'>Lihat pesan di atas.</p>")
        yield pack(
            msg,
            idle,
            idle,
            idle,
            pd.DataFrame(),
            None,
            _status("Error — detail di panel hasil.", "err"),
        )


def _summary_html_error(title: str, body: str) -> str:
    return f'<div class="summary"><h2>{title}</h2><p class="lead">{body}</p></div>'


def build_app():
    import gradio as gr

    seed = _choices("bank") or _choices("stock") or []

    with gr.Blocks(title="Prism") as demo:
        # —— HERO ——
        gr.HTML(
            """
            <div class="hero">
              <div class="kicker">Prism</div>
              <h1>Analisis peluang pergerakan saham</h1>
              <p>Cari perusahaan, pilih jangka waktu, lihat kisaran harga (cone) dan peluang naik —
              dibandingkan tebakan sederhana, bukan ramalan absolut.</p>
            </div>
            """
        )

        # —— INPUT CARD ——
        with gr.Group(elem_classes=["card"]):
            gr.HTML(
                """
                <div class="section-title">1 · Pilih saham</div>
                <p class="hint">Ketik nama atau kode (contoh: bank central asia, BBCA, Apple).
                Pilih salah satu saran di daftar — tidak perlu menghafal format Yahoo.</p>
                """
            )
            search = gr.Textbox(
                label="Cari",
                placeholder="bank central asia · BBCA · Apple · telkom",
                lines=1,
            )
            ticker = gr.Dropdown(
                label="Hasil pencarian (pilih satu)",
                choices=seed,
                value=seed[0] if seed else None,
                allow_custom_value=True,
                filterable=True,
            )
            status = gr.HTML('<p class="hint" style="margin:0">Mulai ketik di kotak Cari.</p>')

            gr.HTML(
                """
                <div class="section-title" style="margin-top:14px">2 · Konteks pasar</div>
                <p class="hint">Dipakai model untuk membandingkan saham ke indeks pasar (otomatis, berlabel jelas).</p>
                """
            )
            market = gr.Radio(
                choices=list(INDEX_PRESETS.keys()),
                value="Indonesia · IHSG (Jakarta)",
                label="Pasar acuan",
            )
            market_hint = gr.HTML(
                f'<p class="hint" style="margin:0">{INDEX_PRESETS["Indonesia · IHSG (Jakarta)"]["hint"]}</p>'
            )

            gr.HTML(
                """
                <div class="section-title" style="margin-top:14px">3 · Jangka waktu</div>
                <p class="hint">Dalam hari bursa. Boleh centang lebih dari satu.</p>
                """
            )
            horizons = gr.CheckboxGroup(
                choices=[lab for lab, _ in HORIZON_CHOICES],
                value=["5 hari (≈ 1 minggu)", "21 hari (≈ 1 bulan)", "252 hari (≈ 1 tahun)"],
                label="Horizon",
            )

            gr.HTML(
                """
                <div class="section-title" style="margin-top:14px">4 · Mode</div>
                <p class="hint">Standar cukup untuk cone & peluang naik. Lanjutan menambah detail statistik (lebih lambat).</p>
                """
            )
            mode = gr.Radio(
                choices=[
                    "Standar — cepat",
                    "Lanjutan — detail ekstra (lebih lambat)",
                ],
                value="Standar — cepat",
                label="Mode analisis",
            )

            run_btn = gr.Button("Jalankan analisis", variant="primary")
            run_status = gr.HTML("")

        # —— RESULTS (each chart in its own framed panel) ——
        gr.HTML('<div class="section-title" style="margin:8px 4px">Hasil</div>')
        summary = gr.HTML()
        fan = gr.HTML()
        gauge = gr.HTML()
        met = gr.HTML()

        with gr.Group(elem_classes=["card"]):
            gr.HTML(
                """
                <div class="section-title">Tabel ringkas</div>
                <p class="hint">Angka yang sama dengan grafik, dalam bentuk tabel.</p>
                """
            )
            table = gr.Dataframe(label="Ringkasan", wrap=True, interactive=False)

        with gr.Group(elem_classes=["card"]):
            gr.HTML(
                """
                <div class="section-title">Laporan lengkap</div>
                <p class="hint">Unduh file HTML, lalu buka di tab baru (bukan path server).</p>
                """
            )
            report_file = gr.File(label="Unduh laporan HTML", file_count="single")

        gr.HTML(
            f'<p class="hint" style="text-align:center;margin-top:8px">'
            f"Prism · riset · bukan saran investasi"
            f'<span style="opacity:.45"> · {UX_BUILD}</span></p>'
        )

        search.change(on_type_search, inputs=[search], outputs=[ticker])
        ticker.change(on_pick_ticker, inputs=[ticker, market], outputs=[market, status])
        market.change(on_market_change, inputs=[market], outputs=[market_hint])
        run_btn.click(
            run_analysis,
            inputs=[ticker, market, horizons, mode],
            outputs=[summary, fan, gauge, met, table, report_file, run_status],
        )

    return demo


def launch_gui(
    share: bool | None = None,
    server_name: str = "0.0.0.0",
    server_port: int | None = None,
    **kwargs: Any,
) -> Any:
    import gradio as gr

    GRADIO_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    print("[Prism] UI human v3 ·", UX_BUILD)
    print("[Prism] Gradio export dir:", GRADIO_EXPORT_DIR)

    demo = build_app()
    in_colab = False
    try:
        import google.colab  # noqa: F401

        in_colab = True
    except Exception:
        pass

    if share is None:
        share = bool(in_colab)

    launch_kwargs: dict[str, Any] = {
        "share": share,
        "show_error": True,
        "css": PRISM_CSS,
        "theme": gr.themes.Soft(
            primary_hue="stone",
            secondary_hue="teal",
            neutral_hue="stone",
            radius_size="lg",
            font=gr.themes.GoogleFont("DM Sans"),
        ),
        "quiet": False,
        # allow /tmp exports (and keep cwd)
        "allowed_paths": [str(GRADIO_EXPORT_DIR), tempfile_dir()],
    }
    if server_port:
        launch_kwargs["server_port"] = server_port
    launch_kwargs.update(kwargs)

    if in_colab:
        print("[Prism] Colab launch share=", share)
        return demo.launch(**launch_kwargs)
    return demo.launch(server_name=server_name, **launch_kwargs)


def tempfile_dir() -> str:
    return tempfile.gettempdir()


def launch_prism(**kwargs):
    return launch_gui(**kwargs)
