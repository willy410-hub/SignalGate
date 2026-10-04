"""Shared look for all figures: dark card, Inter type, text converted to paths so the font
renders identically on GitHub (which cannot load custom fonts)."""
import matplotlib as mpl
import matplotlib.pyplot as plt

BG, CARD, GRID = "#0B1220", "#111B2E", "#22304A"
TEXT, MUTED = "#E6EDF7", "#8FA3C0"
TEAL, AMBER, ROSE, VIOLET, SKY, LIME = "#2DD4BF", "#F5A524", "#FB7185", "#A78BFA", "#38BDF8", "#A3E635"

mpl.rcParams.update({
    "svg.fonttype": "path",
    "font.family": ["Inter", "DejaVu Sans"],
    "figure.facecolor": BG, "savefig.facecolor": BG, "axes.facecolor": CARD,
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": TEXT, "axes.titlecolor": TEXT, "axes.titleweight": "bold", "axes.titlesize": 13,
    "axes.titlelocation": "left", "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "legend.labelcolor": TEXT, "font.size": 10.5,
})


def heading(fig, title, sub=None):
    fig.text(0.02, 0.965, title, fontsize=17, fontweight="bold", family=["Inter Display", "Inter"], va="top")
    if sub:
        fig.text(0.02, 0.905, sub, fontsize=10.5, color=MUTED, va="top")


def footer(fig, text):
    fig.text(0.02, 0.012, text, fontsize=8.5, color=MUTED, va="bottom", family=["DejaVu Sans Mono"])

from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
IMG = ROOT / "docs" / "img"
IMG.mkdir(parents=True, exist_ok=True)
