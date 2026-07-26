"""
Prism Gradio UI — human-first, charts as HTML (reliable in Colab/share).

Design principles:
- One smart search (not dual search/match boxes)
- Human labels for indexes (not only ^JKSE)
- Horizons with day units; Advanced mode hides GARCH jargon
- Charts via embedded Plotly HTML (avoids Gradio Plot broken-image on share)
- Report inline + download file (never raw /content paths as the only output)
"""
from __future__ import annotations

import tempfile
import traceback
from pathlib import Path
from typing import Any

import pandas as pd

from stock_prob.design import UX_BUILD
from stock_prob.report_html import write_prism_report
from stock_prob.tickers import (
    default_context_for_symbol,
    parse_symbol_from_label,
    resolve_ticker,
    search_labels,
    search_tickers,
)
from stock_prob.ui_colab import run_once
from stock_prob.viewmodel import build_viewmodel
from stock_prob.viz import build_fan_figure, build_metrics_figure, build_prob_gauge_figure

# Friendly index presets (values still Yahoo symbols under the hood — user sees labels)
INDEX_PRESETS = {
    "Indonesia · IHSG (Jakarta Composite)": {
        "domestic_index": "^JKSE",
        "us_index": "^GSPC",
        "macro": "^VIX",
        "hint": "Saham IDX (.JK) · domestik IHSG · spillover S&P 500 · regime VIX",
    },
    "United States · S&P 500 + VIX": {
        "domestic_index": "^GSPC",
        "us_index": "^GSPC",
        "macro": "^VIX",
        "hint": "Saham US · beta ke S&P 500 · volatilitas VIX",
    },
    "Global · S&P 500 + Dollar (DXY)": {
        "domestic_index": "^GSPC",
        "us_index": "^GSPC",
        "macro": "DX-Y.NYB",
        "hint": "S&P 500 + kekuatan USD (relevan emerging markets)",
    },
}

HORIZON_CHOICES = [
    ("5 days (1 week)", 5),
    ("21 days (~1 month)", 21),
    ("63 days (~1 quarter)", 63),
    ("126 days (~6 months)", 126),
    ("252 days (~1 year)", 252),
]


PRISM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,400;9..40,500;9..40,600;9..40,700&display=swap');

.gradio-container {
  max-width: 980px !important;
  margin: 0 auto !important;
  font-family: "DM Sans", system-ui, sans-serif !important;
  background:
    radial-gradient(900px 400px at 0% 0%, #e4efe9 0%, transparent 55%),
    radial-gradient(700px 360px at 100% 0%, #f3e9df 0%, transparent 50%),
    #f6f3ee !important;
  padding-bottom: 40px !important;
}
footer { display: none !important; }

.prism-hero h1 {
  font-size: clamp(1.75rem, 3vw, 2.25rem) !important;
  letter-spacing: -0.04em !important;
  font-weight: 700 !important;
  margin: 0 0 8px !important;
  color: #1c1917 !important;
}
.prism-hero p { color: #78716c !important; margin: 0 !important; max-width: 58ch; line-height: 1.5; }
.prism-kicker {
  font-size: 11px !important; letter-spacing: 0.14em !important;
  text-transform: uppercase !important; font-weight: 600 !important;
  color: #0f3d3e !important; margin-bottom: 6px !important;
}

.prism-card {
  background: #fffcf7 !important;
  border: 1px solid #e7e0d5 !important;
  border-radius: 16px !important;
  padding: 14px !important;
  box-shadow: 0 12px 32px -28px rgba(28,25,23,0.4) !important;
}

button.primary {
  border-radius: 12px !important;
  background: #0f3d3e !important;
  font-weight: 600 !important;
}

.prism-result-card {
  background: linear-gradient(125deg, #dcece7 0%, #fffcf7 50%, #f3ebe3 100%);
  border: 1px solid #e7e0d5;
  border-radius: 16px;
  padding: 16px 18px;
  margin-bottom: 12px;
  animation: rise 0.55s cubic-bezier(0.16,1,0.3,1) both;
}
.prism-result-card h2 { margin: 0 0 6px; letter-spacing: -0.03em; font-size: 1.4rem; color: #1c1917; }
.prism-result-card .lead { color: #44403c; font-size: 0.98rem; line-height: 1.5; margin: 8px 0 0; }
.prism-pills { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
.prism-pill {
  font-size: 12px; font-weight: 600; border-radius: 999px; padding: 5px 11px;
  border: 1px solid #e7e0d5; background: #fff; color: #57534e;
}
.prism-pill.up { background: #e6f5ec; color: #1b7a4e; border-color: #b7e0c7; }
.prism-pill.down { background: #fdecea; color: #b42318; border-color: #f2c1bc; }
.prism-pill.ok { background: #e7f1ef; color: #0f3d3e; }
.chart-frame {
  background: #fffcf7; border: 1px solid #e7e0d5; border-radius: 16px;
  padding: 8px; margin: 10px 0; overflow: hidden;
  animation: rise 0.65s cubic-bezier(0.16,1,0.3,1) both;
}
.micro { font-size: 12px; color: #78716c; margin: 2px 0 8px; }
@keyframes rise {
  from { opacity: 0; transform: translateY(12px); }
  to { opacity: 1; transform: translateY(0); }
}
"""


def _fig_to_html(fig: Any, height: int = 460) -> str:
    """Embed Plotly as self-contained HTML — works on Gradio share (no broken image)."""
    if fig is None:
        return '<div class="chart-frame"><p class="micro">Chart unavailable.</p></div>'
    try:
        fig.update_layout(height=height, margin=dict(l=40, r=20, t=48, b=40))
        inner = fig.to_html(
            full_html=False,
            include_plotlyjs="cdn",
            config={"displayModeBar": False, "responsive": True},
        )
        return f'<div class="chart-frame">{inner}</div>'
    except Exception as e:
        return f'<div class="chart-frame"><p class="micro">Chart error: {e}</p></div>'


def _choices(q: str) -> list[str]:
    labels = search_labels(q or "", max_results=12)
    if labels:
        return labels
    q = (q or "").strip()
    return [q] if q else []


def on_type_search(query: str):
    """Single-box typeahead: typing updates the same dropdown choices + value."""
    import gradio as gr

    ch = _choices(query)
    val = ch[0] if ch else (query or None)
    return gr.Dropdown(choices=ch, value=val)


def on_pick_ticker(label: str, market_preset: str):
    """When user picks a match, auto-set market preset if still default-ish."""
    sym = parse_symbol_from_label(label or "")
    if not sym:
        return market_preset, "_Ketik nama perusahaan atau kode saham, lalu pilih saran._"
    # auto market from symbol
    if sym.upper().endswith(".JK"):
        preset = "Indonesia · IHSG (Jakarta Composite)"
    else:
        preset = "United States · S&P 500 + VIX"
    hits = search_tickers(sym, max_results=2)
    name = hits[0].name if hits else sym
    return preset, f"Dipilih **{sym}** — {name}. Pasar diset otomatis (bisa diganti di bawah)."


def on_market_change(preset: str):
    info = INDEX_PRESETS.get(preset) or INDEX_PRESETS["United States · S&P 500 + VIX"]
    return info["hint"]


def _human_summary(vm, name: str) -> str:
    cards = vm.summary_cards()
    if not cards:
        lead = "Model belum menghasilkan probabilitas untuk horizon yang dipilih."
    else:
        # pick shortest horizon for headline
        c0 = cards[0]
        direction = "naik" if c0["direction"] == "up" else "turun"
        conf = "cukup yakin" if c0["conviction"] >= 0.25 else "hampir netral / ragu"
        beat = sum(1 for c in cards if c.get("beat_baseline") is True)
        lead = (
            f"Untuk **{vm.ticker}**, model memperkirakan peluang harga **{direction}** "
            f"dalam ~**{c0['horizon']} hari** sekitar **{c0['prob_up']*100:.0f}%** ({conf}). "
            f"Pada {beat}/{len(cards)} horizon, skor model **lebih baik** dari tebakan base-rate."
        )

    pills = []
    for c in cards:
        cls = "up" if c["direction"] == "up" else "down"
        beat = c.get("beat_baseline")
        tag = " · unggul base" if beat is True else (" · kalah base" if beat is False else "")
        pills.append(
            f'<span class="prism-pill {cls}">{c["horizon"]} hari · '
            f'P(naik) {c["prob_up"]*100:.1f}%{tag}</span>'
        )
    regime_pill = f'<span class="prism-pill ok">suasana pasar: {vm.regime}</span>'

    return f"""
<div class="prism-result-card">
  <div class="prism-kicker">Hasil analisis</div>
  <h2>{vm.ticker} <span style="color:#78716c;font-weight:500;font-size:.72em">{name}</span></h2>
  <div class="micro">Harga terakhir <b>{vm.last_price:,.2f}</b> · data s.d. <b>{vm.asof}</b></div>
  <p class="lead">{lead}</p>
  <div class="prism-pills">{regime_pill}{''.join(pills)}</div>
  <p class="micro" style="margin-top:12px">
    Cone di bawah = rentang kemungkinan harga (bukan ramalan satu garis).
    Band lebih lebar = model lebih tidak yakin.
  </p>
</div>
"""


def run_analysis(
    ticker_label: str,
    market_preset: str,
    horizon_labels: list[str],
    mode: str,
    progress=None,
):
    empty_html = '<div class="chart-frame"><p class="micro">Jalankan analisis untuk melihat grafik.</p></div>'
    try:
        if progress:
            progress(0.08, desc="Mencari & memvalidasi saham…")

        resolved = resolve_ticker(ticker_label or "")
        if not resolved.get("ok"):
            err = (
                f'<div class="prism-result-card"><h2>Saham tidak ditemukan</h2>'
                f'<p class="lead">Coba nama perusahaan (mis. <i>bank central asia</i>) '
                f'atau kode (mis. <i>BBCA</i> / <i>AAPL</i>). '
                f'Detail: {resolved.get("error")}</p></div>'
            )
            return err, empty_html, empty_html, empty_html, pd.DataFrame(), None, ""

        symbol = resolved["symbol"]
        name = resolved.get("name") or symbol
        preset = INDEX_PRESETS.get(market_preset) or INDEX_PRESETS["United States · S&P 500 + VIX"]
        # still allow symbol to override market if IDX vs US mismatch mild
        ctx = default_context_for_symbol(symbol)
        # Prefer explicit user market preset
        domestic = preset["domestic_index"]
        us_index = preset["us_index"]
        macro = preset["macro"]
        # If user picked Indonesia preset but US stock, keep preset (user intent)
        # If IDX stock and US preset, nudge domestic to JKSE for better model
        if symbol.upper().endswith(".JK") and domestic == "^GSPC":
            domestic = "^JKSE"

        # parse horizons from friendly labels
        label_to_h = {lab: h for lab, h in HORIZON_CHOICES}
        hz = []
        for lab in horizon_labels or []:
            if lab in label_to_h:
                hz.append(label_to_h[lab])
            else:
                # fallback parse leading int
                try:
                    hz.append(int(str(lab).split()[0]))
                except Exception:
                    pass
        if not hz:
            hz = [5, 21, 252]

        use_lab = (mode or "").lower().startswith("advanced") or (mode or "").lower().startswith("lanjutan")

        if progress:
            progress(0.25, desc=f"Menghitung model untuk {symbol}…")

        result = run_once(
            symbol,
            domestic_index=domestic,
            us_index=us_index,
            macro=macro,
            horizons=hz,
            use_lab=use_lab,
            mc_paths=900,
        )

        if progress:
            progress(0.7, desc="Menyusun grafik & laporan…")

        vm = build_viewmodel(result, name=name)

        # Always write report to temp + art_dir; expose as downloadable file
        report_path = None
        if vm.art_dir:
            report_path = Path(vm.art_dir) / "prism_report.html"
            write_prism_report(vm, report_path)
        else:
            report_path = Path(tempfile.gettempdir()) / f"prism_{symbol.replace('.','_')}.html"
            write_prism_report(vm, report_path)

        h = vm.primary_horizon()
        cone = vm.cones.get(h)
        fan = None
        if vm.history is not None and len(vm.history) and cone is not None:
            fan = build_fan_figure(
                vm.history,
                cone,
                title=f"Kisaran kemungkinan harga · {vm.ticker} · {h} hari ke depan",
            )
        gauge = build_prob_gauge_figure(vm.probs, title="Peluang naik (P up) per horizon")
        met = None
        if vm.metrics is not None and len(vm.metrics):
            met = build_metrics_figure(
                vm.metrics,
                title="Seberapa jujur model vs tebakan sederhana (Brier ↓ lebih baik)",
            )

        summary = _human_summary(vm, name)
        fan_html = _fig_to_html(fan, 500)
        gauge_html = _fig_to_html(gauge, 280)
        met_html = _fig_to_html(met, 340)

        # Friendly table
        rows = []
        for c in vm.summary_cards():
            rows.append(
                {
                    "Horizon": f"{c['horizon']} hari",
                    "Peluang naik %": round(c["prob_up"] * 100, 1),
                    "Arah": "Naik" if c["direction"] == "up" else "Turun",
                    "Lebih baik dari base-rate?": (
                        "Ya" if c["beat_baseline"] is True else ("Tidak" if c["beat_baseline"] is False else "—")
                    ),
                }
            )
        table = pd.DataFrame(rows)

        # Inline mini-report excerpt note (download has full)
        note = (
            f"<p class='micro'>Laporan lengkap (grafik + tabel) bisa diunduh di bawah. "
            f"Mode: <b>{'Lanjutan' if use_lab else 'Standar'}</b> · "
            f"Konteks: {market_preset}</p>"
        )

        if progress:
            progress(1.0, desc="Selesai")

        return (
            summary + note,
            fan_html,
            gauge_html,
            met_html,
            table,
            str(report_path) if report_path else None,
            "Selesai. Scroll untuk cone & unduh laporan HTML bila perlu.",
        )

    except Exception:
        err = traceback.format_exc()
        msg = (
            f'<div class="prism-result-card"><h2>Gagal menghitung</h2>'
            f'<pre style="white-space:pre-wrap;font-size:11px;color:#78716c">{err[-2000:]}</pre></div>'
        )
        return msg, empty_html, empty_html, empty_html, pd.DataFrame(), None, "Error — lihat pesan di atas."


def build_app():
    import gradio as gr

    # Seed dropdown with a live search (not a hardcoded product universe)
    seed = _choices("stock") or _choices("bank") or []

    with gr.Blocks(title="Prism") as demo:
        gr.HTML(
            """
            <div class="prism-hero">
              <div class="prism-kicker">Prism</div>
              <h1>Analisis peluang pergerakan saham</h1>
              <p>Cari perusahaan, pilih horizon, lihat <b>kisaran harga (cone)</b> dan
              peluang naik — dibandingkan tebakan sederhana (base rate), bukan ramalan absolut.</p>
            </div>
            """
        )

        with gr.Group(elem_classes=["prism-card"]):
            gr.Markdown("### 1. Pilih saham")
            gr.Markdown(
                "<p class='micro'>Ketik **nama** atau **kode** (contoh: <code>bank central asia</code>, "
                "<code>BBCA</code>, <code>Apple</code>). Saran muncul di daftar di bawah — "
                "pilih salah satu.</p>"
            )
            search = gr.Textbox(
                label="Cari saham",
                placeholder="Contoh: BBCA, telkom, apple, bank central asia…",
                lines=1,
            )
            ticker = gr.Dropdown(
                label="Pilih dari hasil pencarian",
                choices=seed,
                value=seed[0] if seed else None,
                allow_custom_value=True,
                filterable=True,
                info="Diperbarui otomatis saat Anda mengetik di kotak pencarian.",
            )
            status = gr.Markdown("Mulai ketik di kotak pencarian.")

            gr.Markdown("### 2. Konteks pasar")
            gr.Markdown(
                "<p class='micro'>Dipakai model untuk membandingkan saham ke **indeks pasar** "
                "(bukan input rahasia). Pilih bahasa manusia — kode Yahoo diurus di belakang.</p>"
            )
            market = gr.Radio(
                choices=list(INDEX_PRESETS.keys()),
                value="Indonesia · IHSG (Jakarta Composite)",
                label="Pasar & indeks acuan",
            )
            market_hint = gr.Markdown(INDEX_PRESETS["Indonesia · IHSG (Jakarta Composite)"]["hint"])

            gr.Markdown("### 3. Horizon waktu")
            horizons = gr.CheckboxGroup(
                choices=[lab for lab, _ in HORIZON_CHOICES],
                value=["5 days (1 week)", "21 days (~1 month)", "252 days (~1 year)"],
                label="Jangka waktu prediksi (hari bursa)",
                info="Centang satu atau lebih. 5 hari = sekitar 1 minggu perdagangan.",
            )

            gr.Markdown("### 4. Mode analisis")
            mode = gr.Radio(
                choices=[
                    "Standar — cepat, cukup untuk lihat cone & P(naik)",
                    "Lanjutan — lebih detail (volatilitas dinamis, rezim pasar, turnamen model)",
                ],
                value="Standar — cepat, cukup untuk lihat cone & P(naik)",
                label="Mode",
                info="Pilih Lanjutan hanya jika Anda ingin detail statistik ekstra.",
            )

            run_btn = gr.Button("Jalankan analisis", variant="primary")
            run_status = gr.Markdown("")

        with gr.Column():
            summary = gr.HTML()
            gr.Markdown("### Kisaran harga ke depan (probability cone)")
            gr.Markdown(
                "<p class='micro'>Garis putus-putus = median. Pita = rentang kemungkinan. "
                "Makin jauh ke depan, biasanya makin lebar.</p>"
            )
            fan = gr.HTML()
            gr.Markdown("### Peluang naik per horizon")
            gauge = gr.HTML()
            gr.Markdown("### Kejujuran model vs tebakan sederhana")
            gr.Markdown(
                "<p class='micro'><b>Brier score</b> lebih rendah = lebih baik. "
                "Jika model tidak mengalahkan base-rate, anggap sinyal lemah.</p>"
            )
            met = gr.HTML()
            table = gr.Dataframe(label="Ringkasan angka", wrap=True)
            report_file = gr.File(
                label="Unduh laporan HTML lengkap (buka di tab baru setelah unduh)",
                file_count="single",
            )

        # wiring: single search drives dropdown
        search.change(on_type_search, inputs=[search], outputs=[ticker])
        ticker.change(on_pick_ticker, inputs=[ticker, market], outputs=[market, status])
        market.change(on_market_change, inputs=[market], outputs=[market_hint])
        run_btn.click(
            run_analysis,
            inputs=[ticker, market, horizons, mode],
            outputs=[summary, fan, gauge, met, table, report_file, run_status],
        )

        gr.Markdown(
            f"""
<p class='micro' style="margin-top:16px">
Prism · riset probabilitas saham · bukan saran investasi.
<span style="opacity:0.5"> · {UX_BUILD}</span>
</p>
"""
        )
    return demo


def launch_gui(
    share: bool | None = None,
    server_name: str = "0.0.0.0",
    server_port: int | None = None,
    inline: bool = True,
) -> Any:
    import gradio as gr

    print("[Prism] Gradio human UI ·", UX_BUILD)
    demo = build_app()
    in_colab = False
    try:
        import google.colab  # noqa: F401

        in_colab = True
    except Exception:
        pass

    if share is None:
        share = bool(in_colab)

    kwargs: dict[str, Any] = {
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
    }
    if server_port:
        kwargs["server_port"] = server_port
    if in_colab:
        print("[Prism] Colab: share=", share, "— charts are HTML embeds (not gr.Plot images)")
        return demo.launch(**kwargs)
    return demo.launch(server_name=server_name, **kwargs)


def launch_prism(**kwargs):
    return launch_gui(**kwargs)
