# =============================================================================
# PASTE THIS ENTIRE CELL INTO COLAB AND RUN
# (Runtime → Disconnect and delete runtime FIRST if UI still looks old)
# =============================================================================
import os, sys, shutil, subprocess
from pathlib import Path

# 1) Kill nested / stale trees that caused /content/prism/prism/prism
for p in [Path("/content/prism"), Path("/content/prism/prism")]:
    if p.exists():
        shutil.rmtree(p, ignore_errors=True)

# 2) Drop cached Python modules (THIS is why git pull looked "unchanged")
for k in list(sys.modules):
    if k == "stock_prob" or k.startswith("stock_prob."):
        del sys.modules[k]

# 3) Always clone fresh to a fixed path (never %cd prism repeatedly)
TARGET = Path("/content/prism")
subprocess.check_call(
    ["git", "clone", "--depth", "1", "https://github.com/SeraKah-1/prism.git", str(TARGET)]
)
os.chdir(TARGET)
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt", "gradio>=4.0"])

# 4) ONLY this src on path (prepend + remove drive stock-prob if present)
sys.path = [p for p in sys.path if "stock-prob" not in p and "stock_prob" not in p]
sys.path.insert(0, str(TARGET / "src"))

import stock_prob
from stock_prob.design import UX_BUILD, UX_LABEL, BG, ACCENT

print("=" * 64)
print("PKG FILE :", stock_prob.__file__)
print("UX_BUILD :", UX_BUILD)
print("UX_LABEL :", UX_LABEL)
print("THEME    :", BG, ACCENT)
print("CWD      :", os.getcwd())
print("=" * 64)
assert stock_prob.__file__.startswith("/content/prism/"), stock_prob.__file__
assert "HUMAN_V3" in UX_BUILD, f"Wrong build {UX_BUILD} — delete runtime and re-run"

import plotly.io as pio
pio.renderers.default = "colab"

from stock_prob.ui_colab import launch_gui
# Gradio on Colab uses share=True so you get a real UI (not stale widget output)
launch_gui()
