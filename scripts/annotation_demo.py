"""Agreement can go UP while the quality of the consensus label goes DOWN (simulated annotators)."""
import json
import numpy as np
from sgate_eval.annotation import krippendorff_alpha, simulate_annotators
from figstyle import ROOT


def majority(m):
    out = []
    for row in m:
        vals, counts = np.unique(row, return_counts=True)
        out.append(vals[np.argmax(counts)] if counts.max() > 1 else row[0])
    return np.array(out)


res = {"honest": [], "herding": []}
for seed in range(20):
    rng = np.random.default_rng(seed)
    truth = rng.integers(0, 3, size=400)
    honest = simulate_annotators(truth, [0.7, 0.7, 0.7], 3, seed=seed + 100)
    herd = simulate_annotators(truth, [0.7, 0.7, 0.7], 3, seed=seed + 100, herding=0.9)
    for name, m in (("honest", honest), ("herding", herd)):
        res[name].append((krippendorff_alpha(m.tolist()), float(np.mean(majority(m) == truth))))
out = {k: {"alpha_mean": float(np.mean([a for a, _ in v])), "consensus_accuracy_mean": float(np.mean([c for _, c in v])),
           "alpha_range": [float(min(a for a, _ in v)), float(max(a for a, _ in v))], "runs": len(v)} for k, v in res.items()}
(ROOT / "results" / "annotation_demo.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
