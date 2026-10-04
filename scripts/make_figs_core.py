"""Figures for the gate demo and the retrieval suite. Every plotted number is computed here."""
import json
import numpy as np
import matplotlib.pyplot as plt
from figstyle import *
from sgate_eval.gate import PreRegistration, SliceSpec, run_gate
from sgate_eval.retrieval import BM25, TfIdf, RRFHybrid, AuthorityRerank, evaluate, generate_corpus
from sgate_eval.stats import wilson_interval

OUT = {}

# ---------------------------------------------------------------- gate demo
def gate_demo():
    rng = np.random.default_rng(0)
    shifts = {"general": 0.06, "medical": -0.04, "chitchat": 0.06, "legal": 0.05}
    per, cs, bs = {}, [], []
    for k, s in shifts.items():
        base = rng.normal(0.6, 0.1, 150)
        cand = base + s + rng.normal(0, 0.03, 150)
        per[k] = (cand, base); cs.append(cand); bs.append(base)
    reg = PreRegistration(
        primary_metric="accuracy",
        slices=[SliceSpec(name="general"), SliceSpec(name="medical", critical=True, tolerance=0.01),
                SliceSpec(name="chitchat"), SliceSpec(name="legal")],
        n_boot=5000, min_effect=0.005)
    rep = run_gate(reg, (np.concatenate(cs), np.concatenate(bs)), per)
    OUT["gate"] = {"decision": rep.decision.value, "overall": rep.overall,
                   "slices": [{k: getattr(s, k) for k in ("name", "mean_diff", "ci_low", "ci_high", "adj_p", "breach", "regression", "critical")} for s in rep.slices]}

    fig = plt.figure(figsize=(10, 4.6), dpi=100)
    heading(fig, "The average improved. The gate still said no.",
            "Paired bootstrap 95% CI per slice (candidate - baseline accuracy), BH-adjusted at q = 0.05")
    ax = fig.add_axes([0.16, 0.16, 0.8, 0.62])
    rows = [("overall", rep.overall["mean_diff"], rep.overall["ci_low"], rep.overall["ci_high"], "ov")] + \
           [(s.name + (" (critical)" if s.critical else ""), s.mean_diff, s.ci_low, s.ci_high,
             "bad" if (s.breach or s.regression) else "ok") for s in rep.slices]
    for i, (n, m, lo, hi, kind) in enumerate(rows[::-1]):
        col = {"ov": SKY, "ok": TEAL, "bad": ROSE}[kind]
        ax.plot([lo, hi], [i, i], color=col, lw=3.2, solid_capstyle="round")
        ax.scatter([m], [i], color=col, s=70, zorder=3, edgecolor=BG, linewidth=1.5)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows[::-1]])
    ax.axvline(0, color=MUTED, lw=1, ls="--"); ax.axvline(-0.01, color=AMBER, lw=1.2, ls=":")
    ax.set_ylim(-0.6, len(rows) - 0.2); ax.text(-0.0105, len(rows) - 0.55, "critical-slice tolerance", color=AMBER, ha="right", va="center", fontsize=9)
    ax.set_xlabel("accuracy difference"); ax.grid(axis="y", visible=False)
    fig.text(0.97, 0.915, f"decision: {rep.decision.value.upper()}", color=ROSE, fontsize=15, fontweight="bold",
             ha="right", va="top", family=["Inter Display", "Inter"])
    footer(fig, "synthetic scores, seed 0  ·  scripts/make_figs_core.py")
    fig.savefig(IMG / "gate_forest.svg"); fig.savefig(ROOT.parent / "preview" / "gate_forest.png")

# ---------------------------------------------------------------- retrieval
def retrieval_fig():
    c = generate_corpus(50, seed=0)
    bm = BM25(c.docs)
    runs = [evaluate(r, c) for r in (bm, TfIdf(c.docs), RRFHybrid(c.docs))]
    # choose the authority weight on a SEPARATE dev corpus (seed 100), then report on the test corpus (seed 0)
    dev = generate_corpus(50, seed=100); dev_bm = BM25(dev.docs)
    grid = (0.0, 0.1, 0.2, 0.3, 0.5, 1.0)
    dev_scores = {w: evaluate(AuthorityRerank(dev_bm, weight=w), dev).metric("success@1").mean() for w in grid}
    best_w = max(grid, key=lambda w: (round(dev_scores[w], 6), -w))
    OUT["authority_weight_selected_on_dev"] = best_w
    runs.append(evaluate(AuthorityRerank(bm, weight=best_w), c))
    names = ["BM25", "TF-IDF", "Hybrid\n(RRF)", "BM25 +\nauthority"]
    n = len(c.queries)
    res = {}
    for nm, r in zip(names, runs):
        s1 = int(r.metric("success@1").sum()); rc = int(r.metric("recall@10").sum())
        res[nm] = {"success@1": (s1 / n, *wilson_interval(s1, n)), "recall@10": (rc / n, *wilson_interval(rc, n)),
                   "ndcg@10": float(np.mean(r.metric("ndcg@10"))), "buried": float(np.mean(r.metric("buried")))}
    OUT["retrieval"] = {k.replace("\n", " "): v for k, v in res.items()}
    # weight sweep
    sweep = []
    for w in grid:
        r = evaluate(AuthorityRerank(bm, weight=w), c)
        sweep.append({"weight": w, "success@1": float(r.metric("success@1").mean()), "recall@10": float(r.metric("recall@10").mean())})
    OUT["authority_weight_sweep"] = sweep

    fig = plt.figure(figsize=(10, 4.8), dpi=100)
    heading(fig, "Retrieved is not the same as ranked first",
            f"Synthetic authority corpus, {n} queries, 95% Wilson intervals; recall@10 vs Success@1 on the authoritative document")
    ax = fig.add_axes([0.07, 0.14, 0.9, 0.64])
    x = np.arange(len(names)); w = 0.36
    for j, (m, col) in enumerate((("recall@10", SKY), ("success@1", TEAL))):
        vals = np.array([res[nm][m][0] for nm in names]); lo = np.array([res[nm][m][1] for nm in names]); hi = np.array([res[nm][m][2] for nm in names])
        bars = ax.bar(x + (j - 0.5) * w, vals, w * 0.92, color=col, label=m, zorder=3)
        ax.errorbar(x + (j - 0.5) * w, vals, yerr=[vals - lo, hi - vals], fmt="none", ecolor=TEXT, elinewidth=1.2, capsize=3, zorder=4)
        for xi, v, h in zip(x + (j - 0.5) * w, vals, hi):
            ax.text(xi, h + 0.03, f"{v:.2f}", ha="center", fontsize=9.5, color=TEXT, family=["DejaVu Sans Mono"])
    ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylim(0, 1.3); ax.legend(loc="upper right", ncol=2); ax.grid(axis="x", visible=False)
    footer(fig, f"authority prior from noisy metadata (90% reliable), weight {best_w} picked on a separate dev corpus  ·  scripts/make_figs_core.py")
    fig.savefig(IMG / "retrieval.svg"); fig.savefig(ROOT.parent / "preview" / "retrieval.png")

gate_demo(); retrieval_fig()
(ROOT / "results").mkdir(exist_ok=True)
(ROOT / "results" / "core_results.json").write_text(json.dumps(OUT, indent=1, default=float))
print(json.dumps(OUT["retrieval"], indent=1, default=float)); print(OUT["authority_weight_sweep"]); print(OUT["gate"]["decision"])
