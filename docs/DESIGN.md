# SignalGate: design notes

SignalGate is a **reference implementation**: small, tested, readable code for the decisions an
evaluation owner has to make repeatedly. It is not a product and not a benchmark of any real model.
Everything that looks like data (annotators, labels, documents, model scores) is **generated** and
labelled as such in the code.

## The job it is built around

Each module exists because one of nine responsibilities needs a concrete, checkable artifact.

| # | Responsibility | What the repo does about it | Where |
|---|---|---|---|
| 1 | Own research and evaluation end to end: framing, data design, quality calibration, signal validation | A frozen **pre-registration** (metric, slices, tolerances, FDR level) is hashed before any result is seen; the gate refuses to run if the rules changed afterwards | `gate/config.py`, `gate/decision.py` |
| 2 | Design ML-oriented data systems: task definitions, schemas, rubrics, incentives | Graded-relevance task definition (authoritative = 2, superseded = 1, other = 0); annotation schema with calibration and hidden gold; an RL environment that tests **incentives** directly | `retrieval/corpus.py`, `annotation/`, `rl/env.py` |
| 3 | Analyse failures for root causes and edge cases | Failure-slice discovery (Fisher exact test per attribute value, Benjamini-Hochberg corrected, ranked by risk ratio); a `buried` metric for "retrieved but not ranked first" | `analysis/failure_slices.py`, `retrieval/metrics.py` |
| 4 | Turn ambiguous real behaviour into evaluation frameworks and new data categories | The authority-aware retrieval suite: the vague complaint "it finds the right page but answers from the old one" becomes a measurable metric pair (recall@10 vs Success@1) | `retrieval/` |
| 5 | Calibrate quality early with domain experts and keep raising the bar | Krippendorff's alpha (nominal / ordinal / interval, missing data) and Cohen's kappa; a calibration gate that says "scale" or "rewrite the guideline"; hidden-gold audit that flags only when confident | `annotation/agreement.py`, `annotation/quality.py` |
| 6 | Iterate rapidly | Everything runs on a laptop CPU; seeds everywhere; resumable experiment runner; a CLI (`sgate`); a SQLite registry of runs | `registry.py`, `cli.py`, `rl/experiments.py` |
| 7 | Act as a quality gate: block claims, pause work, force scope changes | Three outcomes: **block**, **scoped release** (regressed non-critical slices excluded), **canary** (2% then 10%) | `gate/decision.py` |
| 8 | Partner with other teams and give them a credible narrative | Every gate run renders a short report: decision, reasons with numbers, one next step | `GateReport.to_markdown()` |
| 9 | Recommend where to invest, iterate or stop | Learning-curve fit with a bootstrap lower bound on the gain from buying more data; plus an RL environment that learns *how* to spend a labelling budget | `decision/investment.py`, `rl/` |

## Statistical choices, and why

* **Paired resampling, not unpaired.** Candidate and baseline are scored on the same items, so the
  paired difference removes item difficulty. Bootstrap gives the interval; a sign-flip permutation
  test gives the p-value. Both are seeded.
* **Benjamini-Hochberg across slices.** With 40 slices at alpha 0.05 you expect about two false
  alarms by chance. BH *controls* the expected share of false discoveries among the slices it flags;
  it does not remove them. For a single critical slice, use the interval rule below instead.
* **Decide on the interval, not the point estimate.** A critical slice **breaches** when even the
  optimistic end of its interval is worse than its pre-registered tolerance. A non-critical slice
  **regresses** when it survives BH and the drop exceeds the minimum effect.
* **Overall gain must be confirmed too.** The lower bound of the overall interval must clear the
  minimum effect, otherwise the gate blocks: "it went up" is not evidence.
* **Under-powered slices cannot be cleared.** They are reported, and they keep the release at canary.
* **Exact bounds for rare failures.** Clopper-Pearson upper bound and the rule of three, so "0 failures
  in 100" is reported as "below about 3%", never as "safe".
* **Power.** Minimum detectable effect and required sample size for paired comparisons.
* **Leakage.** 13-gram overlap, MinHash Jaccard, and canary strings. Canaries only *detect*; a clean
  canary proves nothing. Paraphrases evade all three, and the tests assert that limitation.
* **Drift.** Jensen-Shannon divergence for categorical mixes (bounded, finite when a category is
  missing); Wasserstein distance for continuous values, in the metric's own units.

## The RL environment

`rl/env.py` is a gymnasium environment. One episode is one labelling budget.

* **Action:** domain (3) x annotator tier (4) x batch size (3), plus STOP. 37 actions.
* **State:** remaining budget, step, simulated capability per domain, and a running alpha estimate per
  (domain, tier), initialised from a calibration batch.
* **Dynamics:** buying labels improves capability with diminishing returns, scaled by *true* label
  accuracy. Domain weights make the hard, high-weight domain matter most.
* **Quality gate (the constraint):** a batch is a *violation* if its measured Krippendorff alpha is
  below 0.7, or, when the audit is on, its accuracy on hidden gold items is below 0.8. The constraint is a
  governance rule, not a physical effect: low-alpha labels still carry some signal in the simulator, which
  is what makes ignoring the rule tempting.
* **The trap:** tier 3 is a cheap bulk vendor whose annotators copy each other. It shows very high
  agreement and low true accuracy. Agreement alone cannot see it; the gold audit can.
* **Held-out family:** the same tiers with worse accuracy in two domains. The calibration table does
  not know about the shift.

Three ways of handling the constraint are compared:

| Method | How the constraint is enforced | Expected strength | Expected weakness |
|---|---|---|---|
| PPO, no constraint | not at all | highest raw gain | breaks the gate constantly |
| PPO + shield | rule-based action masking from the calibration table, plus "do not repeat a tier that just failed" | fast, no violations *when the table is right* | the table can be wrong under shift, and a measured alpha is noisy even when the table is right |
| PPO + Lagrangian | learned multiplier on a violation cost (CMDP), budget 0.5 per episode | adapts to shift from observed alphas | violations during training, more conservative |

The experiment script reports whatever happens; see the README for the measured numbers.

## What is simulated, and what is not

* Annotators, labels, documents and model scores are generated. **No real human agreement data is
  used or implied.** `annotation/simulate.py` exists to exercise the alpha and gold-audit code and to
  drive the RL environment.
* The "judge" checks are tested with simulated judges that have known biases. No language model is called.
* The retrieval suite uses lexical rankers (BM25, TF-IDF, rank fusion) plus a metadata re-ranker. It does
  not include a neural dense retriever, so it says nothing about how such models behave.
* RL results are from a simulator I wrote. They show how the *methods* behave under stated assumptions,
  not how a labelling programme would perform in practice.

## Limits worth stating

* BH assumes independent or positively dependent tests. Slices that overlap heavily violate that.
* Thresholds (alpha 0.7, q 0.05, tolerance, canary steps) are defaults for the demonstration. They are
  meant to be set per task with domain experts, not copied.
* The shield uses a calibration table measured once on nominal conditions. That is a deliberate,
  realistic weakness, and the held-out family is there to expose it.
