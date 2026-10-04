"""RL figures, computed from results/rl_results.json (produced by sgate_eval.rl.experiments)."""
import json
import numpy as np
import matplotlib.pyplot as plt
from figstyle import *
from sgate_eval.stats import t_interval

R = json.loads((ROOT / "results" / "rl_results.json").read_text())
runs, base = R["runs"], R["baselines"]


def agg(exp, name, fam, key):
    v = [r["eval"][fam][key] for r in runs if r["exp"] == exp and r["name"] == name]
    m, lo, hi = t_interval(v)
    return m, lo, hi, v


SUMMARY = {}
for exp, names in (("E1", ["unconstrained", "shield", "lagrangian"]), ("E2", ["proxy_agreement", "true_gain"])):
    for n in names:
        for fam in ("train", "heldout"):
            SUMMARY[f"{exp}/{n}/{fam}"] = {k: agg(exp, n, fam, k)[:3] for k in ("ep_gain", "ep_violations", "ep_vendor_steps", "ep_spend_frac", "ep_proxy")}
for n in ("random", "heuristic"):
    for fam in ("train", "heldout"):
        SUMMARY[f"baseline/{n}/{fam}"] = {k: (v, None, None) for k, v in base[n][fam].items()}
(ROOT / "results" / "rl_summary.json").write_text(json.dumps(SUMMARY, indent=1, default=float))

COL = {"unconstrained": ROSE, "shield": AMBER, "lagrangian": TEAL, "random": MUTED, "heuristic": VIOLET}
LAB = {"unconstrained": "PPO, no constraint", "shield": "PPO + shield", "lagrangian": "PPO + Lagrangian (CMDP)",
       "random": "random", "heuristic": "rule of thumb"}

# ------------------------------------------------------------ E1: gain vs violations
fig = plt.figure(figsize=(10.5, 4.9), dpi=100)
heading(fig, "Shield vs Lagrangian: what each buys, and what it costs",
        f"{len(R['seeds'])} training seeds per method, mean with 95% t-interval; evaluated on training-like and held-out (shifted) annotator pools")
for k, (key, title, lo_good) in enumerate((("ep_gain", "Simulated downstream gain (higher is better)", False),
                                           ("ep_violations", "Quality-gate violations per episode (lower is better)", True))):
    ax = fig.add_axes([0.07 + k * 0.5, 0.15, 0.40, 0.64]); ax.set_title(title, fontsize=11)
    names = ["unconstrained", "shield", "lagrangian"]
    w = 0.36
    for j, fam in enumerate(("train", "heldout")):
        for i, n in enumerate(names):
            m, lo, hi, _ = agg("E1", n, fam, key)
            x = i + (j - 0.5) * w
            ax.bar(x, m, w * 0.92, color=COL[n], alpha=1.0 if fam == "train" else 0.55, hatch=None if fam == "train" else "///",
                   edgecolor=BG, zorder=3)
            ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], color=TEXT, capsize=3, elinewidth=1.2, zorder=4)
    for n, ls in (("heuristic", "--"),):
        ax.axhline(base[n]["heldout"][key], color=COL[n], ls=ls, lw=1.2)
        ax.text(2.45, base[n]["heldout"][key], "rule of thumb\n(held-out)", color=COL[n], fontsize=8, va="bottom", ha="right")
    ax.set_xticks(range(3)); ax.set_xticklabels(["no\nconstraint", "shield", "Lagrangian"]); ax.grid(axis="x", visible=False)
    if key == "ep_violations":
        ax.axhline(0.5, color=AMBER, lw=1, ls=":"); ax.text(2.45, 0.62, "CMDP budget 0.5", color=AMBER, fontsize=8, ha="right")
from matplotlib.patches import Patch
fig.legend(handles=[Patch(fc=MUTED, label="training-like pool"), Patch(fc=MUTED, alpha=0.55, hatch="///", label="held-out pool (shifted)")],
           loc="upper right", bbox_to_anchor=(0.985, 0.80), ncol=1, fontsize=8.5)
footer(fig, "simulated annotators and labels  ·  scripts/make_figs_rl.py  ·  results/rl_results.json")
fig.savefig(IMG / "rl_constraints.svg"); fig.savefig(ROOT.parent / "preview" / "rl_constraints.png")

# ------------------------------------------------------------ E2: reward hacking
fig = plt.figure(figsize=(10.5, 4.6), dpi=100)
heading(fig, "Reward hacking, reproduced: optimise agreement and the agent buys the herding vendor",
        "Same learner, two rewards, hidden-gold audit OFF during training and ON at evaluation; 95% t-intervals over seeds")
panels = (("ep_proxy", "Agreement-volume reward\n(what the proxy sees)"), ("ep_gain", "True simulated gain\n(what we want)"),
          ("ep_vendor_steps", "Batches bought from the\nherding vendor (of ~24)"))
for k, (key, title) in enumerate(panels):
    ax = fig.add_axes([0.06 + k * 0.32, 0.15, 0.26, 0.6]); ax.set_title(title, fontsize=10.5)
    for i, (n, col, lab) in enumerate((("proxy_agreement", ROSE, "agreement reward"), ("true_gain", TEAL, "true-gain reward"))):
        m, lo, hi, _ = agg("E2", n, "train", key)
        ax.bar(i, m, 0.7, color=col, zorder=3); ax.errorbar(i, m, yerr=[[m - lo], [hi - m]], color=TEXT, capsize=3, zorder=4)
        ax.text(i, hi + 0.03 * max(1e-9, ax.get_ylim()[1]), f"{m:.2f}", ha="center", fontsize=9.5, family=["DejaVu Sans Mono"])
    ax.set_xticks([0, 1]); ax.set_xticklabels(["agreement\nreward", "true-gain\nreward"]); ax.grid(axis="x", visible=False)
    ax.margins(y=0.2)
footer(fig, "simulated annotators and labels  ·  scripts/make_figs_rl.py  ·  results/rl_results.json")
fig.savefig(IMG / "reward_hacking.svg"); fig.savefig(ROOT.parent / "preview" / "reward_hacking.png")

# ------------------------------------------------------------ learning curves
fig = plt.figure(figsize=(10.5, 4.2), dpi=100)
heading(fig, "Training dynamics", "Mean over seeds; the Lagrange multiplier rises until violations fall to the budget, then relaxes")
ax1 = fig.add_axes([0.07, 0.16, 0.40, 0.62]); ax2 = fig.add_axes([0.56, 0.16, 0.40, 0.62])
ax1.set_title("Violations per episode while training", fontsize=11); ax2.set_title("Lagrange multiplier", fontsize=11)
for n in ("unconstrained", "shield", "lagrangian"):
    H = [r["history"] for r in runs if r["exp"] == "E1" and r["name"] == n]
    steps = [h["steps"] for h in H[0]]
    v = np.nanmean([[h["ep_violations"] for h in hh] for hh in H], axis=0)
    ax1.plot(steps, v, color=COL[n], lw=2.2, label=LAB[n])
    if n == "lagrangian":
        ax2.plot(steps, np.mean([[h["lagrange"] for h in hh] for hh in H], axis=0), color=TEAL, lw=2.4)
ax1.set_yscale("symlog", linthresh=1); ax1.legend(fontsize=8.5, loc="center right")
for a in (ax1, ax2): a.set_xlabel("environment steps")
footer(fig, "scripts/make_figs_rl.py  ·  results/rl_results.json")
fig.savefig(IMG / "rl_training.svg"); fig.savefig(ROOT.parent / "preview" / "rl_training.png")
print(json.dumps({k: {kk: [round(x, 3) if x is not None else None for x in vv] for kk, vv in v.items() if kk in ("ep_gain", "ep_violations", "ep_vendor_steps")} for k, v in SUMMARY.items()}, indent=1))
