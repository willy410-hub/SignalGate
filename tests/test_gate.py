import numpy as np
import pytest

from sgate_eval.gate import Decision, PreRegistration, SliceSpec, run_gate


def make_reg(**kw):
    return PreRegistration(
        primary_metric="accuracy",
        slices=[
            SliceSpec(name="general", min_n=30),
            SliceSpec(name="medical", critical=True, tolerance=0.01, min_n=30),
            SliceSpec(name="chitchat", min_n=30),
        ],
        n_boot=2000,
        **kw,
    )


def synth(rng, n, shift):
    base = rng.normal(0.6, 0.1, n)
    return base + shift + rng.normal(0, 0.03, n), base


def build(shifts, n=150, seed=0):
    rng = np.random.default_rng(seed)
    per = {k: synth(rng, n, v) for k, v in shifts.items()}
    cand = np.concatenate([c for c, _ in per.values()])
    base = np.concatenate([b for _, b in per.values()])
    return (cand, base), per


def test_overall_up_but_critical_slice_down_is_blocked():
    # the classic "ship or not" scenario
    overall, per = build({"general": 0.06, "medical": -0.04, "chitchat": 0.06})
    rep = run_gate(make_reg(), overall, per)
    assert rep.overall["mean_diff"] > 0
    assert rep.decision == Decision.BLOCK
    assert any("medical" in r for r in rep.reasons)


def test_non_critical_regression_gives_scoped_release():
    overall, per = build({"general": 0.08, "medical": 0.05, "chitchat": -0.05})
    rep = run_gate(make_reg(), overall, per)
    assert rep.decision == Decision.SCOPED_RELEASE
    assert rep.excluded_slices == ["chitchat"]


def test_clean_win_goes_to_canary_not_full_release():
    overall, per = build({"general": 0.05, "medical": 0.04, "chitchat": 0.05})
    rep = run_gate(make_reg(), overall, per)
    assert rep.decision == Decision.CANARY
    assert "2%" in rep.next_step and "10%" in rep.next_step


def test_no_confirmed_gain_is_blocked():
    overall, per = build({"general": 0.0, "medical": 0.0, "chitchat": 0.0})
    rep = run_gate(make_reg(), overall, per)
    assert rep.decision == Decision.BLOCK


def test_changed_registration_is_refused():
    reg = make_reg()
    recorded = reg.fingerprint()
    changed = make_reg(min_effect=0.0)  # loosened after the fact
    overall, per = build({"general": 0.05, "medical": 0.04, "chitchat": 0.05})
    with pytest.raises(ValueError):
        run_gate(changed, overall, per, registered_hash=recorded)
    run_gate(reg, overall, per, registered_hash=recorded)  # same rules pass


def test_fingerprint_is_stable_and_sensitive():
    assert make_reg().fingerprint() == make_reg().fingerprint()
    assert make_reg().fingerprint() != make_reg(fdr_q=0.1).fingerprint()


def test_missing_slice_scores_raise():
    overall, per = build({"general": 0.05, "medical": 0.04, "chitchat": 0.05})
    per.pop("chitchat")
    with pytest.raises(ValueError):
        run_gate(make_reg(), overall, per)


def test_duplicate_slice_names_rejected():
    with pytest.raises(ValueError):
        PreRegistration(primary_metric="m", slices=[SliceSpec(name="a"), SliceSpec(name="a")])


def test_report_renders_markdown_and_json():
    overall, per = build({"general": 0.05, "medical": 0.04, "chitchat": 0.05})
    rep = run_gate(make_reg(), overall, per)
    assert "Decision:" in rep.to_markdown()
    assert '"decision": "canary"' in rep.to_json()
