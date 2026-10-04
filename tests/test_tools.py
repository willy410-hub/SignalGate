import numpy as np
import pytest

from sgate_eval.analysis import discover_failure_slices
from sgate_eval.decision import Action, recommend
from sgate_eval.judge import agreement_with_labels, position_bias, verbosity_bias
from sgate_eval.registry import Registry


# ---------------------------------------------------------------- failure slices
def make_records(seed=0, n=600):
    rng = np.random.default_rng(seed)
    recs = []
    for _ in range(n):
        lang = rng.choice(["en", "ar", "fr"])
        length = rng.choice(["short", "long"])
        p_fail = 0.10 + (0.35 if (lang == "ar" and length == "long") else 0.0) + (0.02 if lang == "fr" else 0.0)
        recs.append({"lang": lang, "length": length, "failed": rng.random() < p_fail})
    return recs


def test_failure_slice_finds_planted_cause_and_not_noise():
    findings = discover_failure_slices(make_records(), "failed", ["lang", "length"])
    sig = [(f.attribute, f.value) for f in findings if f.significant]
    assert ("lang", "ar") in sig
    assert ("lang", "fr") not in sig          # tiny planted bump is not claimed
    assert findings[0].risk_ratio > 1.5


def test_failure_slices_on_pure_noise_find_nothing():
    rng = np.random.default_rng(3)
    recs = [{"a": rng.choice(list("xyzw")), "b": rng.choice(list("pq")), "failed": rng.random() < 0.2} for _ in range(800)]
    assert not any(f.significant for f in discover_failure_slices(recs, "failed", ["a", "b"]))


# ---------------------------------------------------------------- judge validation
import zlib


def fair_judge(prompt, a, b):
    """SIMULATED unbiased judge: decides from a content hash of (prompt, answer), so it is
    consistent under position swaps and unrelated to answer length."""
    sa, sb = zlib.crc32((prompt + a).encode()), zlib.crc32((prompt + b).encode())
    return "A" if sa >= sb else "B"


def first_position_judge(prompt, a, b):
    return "A"


def long_lover(prompt, a, b):
    return "A" if len(a) >= len(b) else "B"


ITEMS = [(f"p{i}", f"answer{i}x", f"answer{i}y") for i in range(60)]


def test_position_bias_detected_for_biased_judge_only():
    assert position_bias(first_position_judge, ITEMS).biased
    rep = position_bias(fair_judge, ITEMS)
    assert not rep.biased and rep.inconsistent_rate == 0.0


def test_verbosity_bias_detected():
    pairs = [(f"p{i}", "same quality short", "same quality but much longer, padded with filler words") for i in range(40)]
    assert verbosity_bias(long_lover, pairs).biased
    assert not verbosity_bias(fair_judge, pairs).biased


def test_judge_agreement_threshold():
    trusted = ["pass", "fail"] * 30
    good = list(trusted)
    for i in range(0, 60, 15):
        good[i] = "fail" if good[i] == "pass" else "pass"
    assert agreement_with_labels(good, trusted).passed
    assert not agreement_with_labels(["pass"] * 60, trusted).passed


# ---------------------------------------------------------------- investment decision
def curve(n, a=0.85, b=2.0, c=0.6):
    return a - b * np.power(n, -c)


def test_invest_when_curve_still_rising():
    n = np.array([50, 100, 200, 400, 800])
    rng = np.random.default_rng(0)
    y = 0.95 - 3.0 * n ** -0.35 + rng.normal(0, 0.002, n.size)
    rep = recommend(n, y, extra=800, min_gain=0.01, noise_sd=0.002)
    assert rep.action == Action.INVEST and rep.gain_ci[0] >= 0.01


def test_stop_when_plateaued():
    n = np.array([200, 400, 800, 1600, 3200])
    y = curve(n, a=0.80, b=40.0, c=1.2) + np.random.default_rng(1).normal(0, 0.001, n.size)
    rep = recommend(n, y, extra=1000, min_gain=0.01, noise_sd=0.001)
    assert rep.action == Action.STOP


def test_noisy_curve_is_not_enough_to_spend():
    n = np.array([100, 200, 400, 800, 1600])
    y = 0.7 - 1.0 * n ** -0.5 + np.random.default_rng(2).normal(0, 0.03, n.size)
    rep = recommend(n, y, extra=1000, min_gain=0.02, noise_sd=0.03)
    assert rep.action != Action.INVEST


# ---------------------------------------------------------------- registry + regression suite
def test_registry_logs_runs_with_registration_hash():
    reg = Registry()
    rid = reg.log_run("exp1", "abc123", 7, "canary", {"success@1": 0.88})
    row = reg.runs("exp1")[0]
    assert rid == row["id"] and row["registration_hash"] == "abc123" and row["metrics"]["success@1"] == 0.88


def test_regression_suite_detects_reintroduced_failures():
    reg = Registry()
    reg.add_failures(["a", "b", "c"], note="v1 failures")
    reg.mark_fixed(["a", "b"])
    out = reg.check_regressions(failed_now=["b", "c", "z"])
    assert out["regressed"] == ["b"]        # was fixed, failing again
    assert out["still_open"] == ["c"]
    assert out["newly_fixed"] == []
