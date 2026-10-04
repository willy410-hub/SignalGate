import json

import numpy as np
import pytest

from sgate_eval.cli import main
from sgate_eval.gate import PreRegistration, SliceSpec, run_gate


def test_demo_prints_block_report(capsys):
    main(["demo"])
    out = capsys.readouterr().out
    assert "SYNTHETIC DEMO DATA" in out and "block" in out


def test_gate_command_exit_code_and_hash_guard(tmp_path, capsys):
    rng = np.random.default_rng(0)
    reg = {"primary_metric": "acc", "n_boot": 1000,
           "slices": [{"name": "a"}, {"name": "b", "critical": True, "tolerance": 0.01}]}
    scores = {"slices": {}}
    cs, bs = [], []
    for name, shift in (("a", 0.05), ("b", 0.04)):
        base = rng.normal(0.6, 0.1, 120)
        cand = base + shift + rng.normal(0, 0.03, 120)
        scores["slices"][name] = [cand.tolist(), base.tolist()]
        cs.append(cand); bs.append(base)
    scores["overall"] = [np.concatenate(cs).tolist(), np.concatenate(bs).tolist()]
    (tmp_path / "reg.json").write_text(json.dumps(reg))
    (tmp_path / "scores.json").write_text(json.dumps(scores))
    with pytest.raises(SystemExit) as e:
        main(["gate", "--registration", str(tmp_path / "reg.json"), "--scores", str(tmp_path / "scores.json")])
    assert e.value.code == 0                       # canary
    with pytest.raises(ValueError):                # wrong recorded hash is refused
        main(["gate", "--registration", str(tmp_path / "reg.json"), "--scores", str(tmp_path / "scores.json"),
              "--recorded-hash", "deadbeef"])


def test_readme_snippet_runs():
    rng = np.random.default_rng(1)
    pairs = {}
    for k in ("general", "medical", "chitchat"):
        b = rng.normal(0.6, 0.1, 100)
        pairs[k] = (b + 0.04 + rng.normal(0, 0.03, 100), b)
    cand_all = np.concatenate([v[0] for v in pairs.values()]); base_all = np.concatenate([v[1] for v in pairs.values()])
    reg = PreRegistration(
        primary_metric="accuracy",
        slices=[SliceSpec(name="general"), SliceSpec(name="medical", critical=True, tolerance=0.01), SliceSpec(name="chitchat")],
        fdr_q=0.05, min_effect=0.005,
    )
    recorded = reg.fingerprint()
    report = run_gate(reg, overall=(cand_all, base_all), per_slice=pairs, registered_hash=recorded)
    assert report.to_markdown().startswith("**Decision:")
