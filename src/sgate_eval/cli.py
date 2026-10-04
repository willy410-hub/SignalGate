"""Command line: `sgate demo` prints a gate report on synthetic data; `sgate gate` runs it on your scores."""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from .gate import PreRegistration, run_gate


def _demo() -> None:
    rng = np.random.default_rng(0)
    shifts = {"general": 0.06, "medical": -0.04, "chitchat": 0.06}
    per, cs, bs = {}, [], []
    for k, s in shifts.items():
        b = rng.normal(0.6, 0.1, 150)
        c = b + s + rng.normal(0, 0.03, 150)
        per[k] = (c, b); cs.append(c); bs.append(b)
    reg = PreRegistration.model_validate({
        "primary_metric": "accuracy", "n_boot": 4000,
        "slices": [{"name": "general"}, {"name": "medical", "critical": True, "tolerance": 0.01}, {"name": "chitchat"}],
    })
    print("SYNTHETIC DEMO DATA\n")
    print(run_gate(reg, (np.concatenate(cs), np.concatenate(bs)), per).to_markdown())


def _gate(reg_path: str, scores_path: str, recorded_hash: str | None) -> None:
    reg = PreRegistration.model_validate_json(open(reg_path).read())
    raw = json.load(open(scores_path))  # {"overall": [cand, base], "slices": {name: [cand, base]}}
    report = run_gate(reg, tuple(raw["overall"]), {k: tuple(v) for k, v in raw["slices"].items()}, recorded_hash)
    print(report.to_markdown())
    sys.exit({"block": 2, "scoped_release": 1, "canary": 0}[report.decision.value])


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="sgate")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("demo")
    g = sub.add_parser("gate")
    g.add_argument("--registration", required=True)
    g.add_argument("--scores", required=True)
    g.add_argument("--recorded-hash")
    a = p.parse_args(argv)
    _demo() if a.cmd == "demo" else _gate(a.registration, a.scores, a.recorded_hash)


if __name__ == "__main__":
    main()
