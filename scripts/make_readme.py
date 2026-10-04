"""Fill docs/README.template.md from results/*.json so every number in README.md is traceable."""
import json
import re
import subprocess
import sys
from figstyle import ROOT

R = json.loads((ROOT / "results" / "rl_results.json").read_text())
RS = json.loads((ROOT / "results" / "rl_summary.json").read_text())
CORE = json.loads((ROOT / "results" / "core_results.json").read_text())
ANN = json.loads((ROOT / "results" / "annotation_demo.json").read_text())
DIAG = json.loads((ROOT / "results" / "shield_diagnostic.json").read_text())

n_tests = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q"], cwd=ROOT, capture_output=True, text=True).stdout
n_tests = int(re.search(r"(\d+) tests? collected", n_tests).group(1))


def walk(obj, parts):
    for p in parts:
        obj = obj[int(p)] if isinstance(obj, list) else obj[p]
    return obj


def resolve(ns, path):
    if ns == "meta":
        return {"tests": n_tests}[path]
    if ns == "core":
        return walk(CORE, path.split("."))
    if ns == "ann":
        return walk(ANN, path.split("."))
    if ns == "rl":
        if path == "steps":
            return R["steps"]
        parts = path.split("/")
        return walk(RS["/".join(parts[:3])], parts[3:])
    if ns == "gap":
        e, m = path.split("/")
        tr, ho = RS[f"{e}/{m}/train"]["ep_gain"][0], RS[f"{e}/{m}/heldout"]["ep_gain"][0]
        return (tr - ho) / tr
    raise KeyError(ns)


def diag_table():
    rows = {}
    for r in DIAG:
        rows.setdefault((r["domain"], r["tier"]), {})[r["batch"]] = r["fail_rate"]
    lines = ["| domain | tier | batch 50 | batch 100 | batch 200 |", "|---|---|---|---|---|"]
    for (d, t), v in rows.items():
        lines.append(f"| {d} | {t} | {v[50]:.1%} | {v[100]:.1%} | {v[200]:.1%} |")
    return "\n".join(lines)


def sub(m):
    if m.group(1) == "diag":
        return diag_table()
    ns, path, *fmt = m.group(1).split("|")
    val = resolve(ns, path)
    return format(val, fmt[0]) if fmt else str(val)


text = (ROOT / "docs" / "README.template.md").read_text()
out = re.sub(r"@@(.+?)@@", sub, text)
assert "@@" not in out
(ROOT / "README.md").write_text(out)
print("README.md written;", n_tests, "tests")
