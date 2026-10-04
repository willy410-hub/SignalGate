import numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from figstyle import *

fig = plt.figure(figsize=(12.8, 3.6), dpi=100)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 128); ax.set_ylim(0, 36); ax.axis("off")
ax.imshow(np.linspace(0, 1, 256).reshape(1, -1), extent=[0, 128, 0, 36], aspect="auto",
          cmap=mpl.colors.LinearSegmentedColormap.from_list("g", ["#0B1220", "#10233F", "#0B3B47"]), zorder=0)
# noisy signal -> gate -> clean signal
rng = np.random.default_rng(4)
x = np.linspace(66, 126, 600)
gate_x = 92
noisy = 18 + 3.2 * np.sin((x - 66) * 0.55) + rng.normal(0, 1.5, x.size) * np.exp(-(x - 66) / 55)
clean = 18 + 3.2 * np.sin((x - 66) * 0.55)
ax.plot(x[x < gate_x], noisy[x < gate_x] * 1 + 0, color=ROSE, lw=1.8, alpha=0.9, zorder=2)
ax.plot(x[x >= gate_x], clean[x >= gate_x], color=TEAL, lw=2.6, zorder=2)
for i, gx in enumerate([gate_x - 1.2, gate_x + 1.2]):
    ax.add_patch(Rectangle((gx - 0.45, 4), 0.9, 28, color=AMBER, zorder=3, alpha=0.95))
ax.text(gate_x, 2.2, "GATE", ha="center", fontsize=9, color=AMBER, fontweight="bold", family=["DejaVu Sans Mono"])
ax.text(77, 31.5, "noisy signal", ha="center", fontsize=9, color=ROSE, family=["DejaVu Sans Mono"])
ax.text(112, 31.5, "trusted signal", ha="center", fontsize=9, color=TEAL, family=["DejaVu Sans Mono"])
ax.text(5, 22.5, "SignalGate", fontsize=54, fontweight="heavy", family=["Inter Display", "Inter"], color=TEXT, va="center")
ax.text(5.4, 11.8, "A reference implementation of a research-signal quality gate", fontsize=14.5, color="#C7D4E8", va="center")
ax.text(5.4, 7.9, "evaluation  ·  annotation calibration  ·  leakage checks  ·  constrained-RL data allocation",
        fontsize=10.5, color=MUTED, va="center")
for i, (t, c) in enumerate([("block", ROSE), ("scoped release", AMBER), ("canary", TEAL)]):
    x0 = 5.4 + i * 17.5
    ax.add_patch(FancyBboxPatch((x0, 2.2), 15.5, 3.2, boxstyle="round,pad=0.02,rounding_size=1.4", fc="none", ec=c, lw=1.6))
    ax.text(x0 + 7.75, 3.8, t, ha="center", va="center", fontsize=9.5, color=c, fontweight="bold", family=["DejaVu Sans Mono"])
fig.savefig(IMG / "banner.svg"); fig.savefig(ROOT.parent / "preview" / "banner.png", dpi=100)
