"""Why a 'hard' shield still sees violations: the gate statistic is measured on a finite batch.
For every (domain, tier) the shield allows on nominal conditions, simulate many batches and
report how often the measured gate fails anyway."""
import json
import numpy as np
from sgate_eval.rl.env import (BATCH_SIZES, DOMAINS, NOMINAL_ACC, TIERS, VENDOR, SimConfig, calibration_table, measure_batch)
from figstyle import ROOT

rng = np.random.default_rng(0)
calib = calibration_table(np.random.default_rng(12345), SimConfig())
out = []
for d in range(len(DOMAINS)):
    for t in range(len(TIERS)):
        if t == VENDOR or calib[d, t] < 0.75:
            continue  # the shield masks these
        for n in BATCH_SIZES:
            fails = np.mean([
                (lambda m: m[0] < 0.7 or m[1] < 0.8)(measure_batch(rng, NOMINAL_ACC[d, t], n, False)) for _ in range(3000)])
            out.append({"domain": DOMAINS[d], "tier": TIERS[t], "batch": n, "calibration_alpha": round(float(calib[d, t]), 3),
                        "expected_alpha": round(float((2 * NOMINAL_ACC[d, t] - 1) ** 2), 3), "fail_rate": round(float(fails), 3)})
(ROOT / "results" / "shield_diagnostic.json").write_text(json.dumps(out, indent=1))
for r in out: print(r)
