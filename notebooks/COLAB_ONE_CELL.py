# Paste THIS ENTIRE CELL into any Colab notebook and run once.
# It ignores old Drive copies and always uses latest GitHub Prism.

import os, sys, shutil
from pathlib import Path

for k in list(sys.modules):
    if k == "stock_prob" or k.startswith("stock_prob."):
        del sys.modules[k]

ROOT = Path("/content/prism")
if ROOT.exists():
    shutil.rmtree(ROOT)

import subprocess
subprocess.check_call(["git", "clone", "--depth", "1", "https://github.com/SeraKah-1/prism.git", "/content/prism"])
os.chdir("/content/prism")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"])

sys.path.insert(0, "/content/prism/src")

from stock_prob.design import UX_BUILD, UX_LABEL, BG, ACCENT
print("=" * 60)
print("LOADED:", UX_LABEL)
print("BUILD :", UX_BUILD)
print("THEME :", BG, ACCENT)
import stock_prob
print("PKG   :", stock_prob.__file__)
print("=" * 60)
assert "PRISM_UX_2026" in UX_BUILD, "Old code — Runtime > Disconnect and delete runtime"

import plotly.io as pio
pio.renderers.default = "colab"
from stock_prob.ui_colab import launch_gui
launch_gui()
