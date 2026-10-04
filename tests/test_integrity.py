import numpy as np
import pytest

from sgate_eval.integrity import (
    MinHasher, categorical_drift, continuous_drift, contamination_report, js_divergence, wasserstein,
)

TRAIN = [
    "the quick brown fox jumps over the lazy dog while the committee reviews the quarterly budget "
    "and the auditors check every line of the ledger before the board meeting on friday afternoon",
    "gradient descent updates parameters in the direction opposite to the gradient of the loss "
    "function scaled by a learning rate that is usually decayed over the course of training",
]


def test_verbatim_leak_is_flagged_and_clean_item_is_not():
    leaked = TRAIN[0]
    clean = (
        "a completely different question about how annotators should resolve disagreement when two "
        "senior reviewers give conflicting safety ratings for the same borderline response today"
    )
    res = contamination_report([leaked, clean], TRAIN)
    assert res[0].flagged and res[0].ngram_overlap == pytest.approx(1.0)
    assert not res[1].flagged


def test_light_edit_caught_by_minhash_but_paraphrase_is_not():
    light = TRAIN[1].replace("usually", "typically").replace("decayed", "reduced")
    para = (
        "optimisers move weights against the slope of the objective, multiplied by a step size "
        "that practitioners commonly shrink as training progresses over many epochs"
    )
    res = contamination_report([light, para], TRAIN, n=13, overlap_threshold=0.9, minhash_threshold=0.5)
    assert res[0].max_minhash > 0.5  # near-duplicate detected
    assert not res[1].flagged  # honest limit: paraphrase evades lexical detectors


def test_canary_detection():
    canary = "CANARY-7f3a9c1e"
    train = TRAIN + [f"some scraped page containing {canary} by accident"]
    item = f"benchmark question text {canary}"
    res = contamination_report([item], train, canaries=[canary])
    assert res[0].canary_hit and res[0].flagged


def test_minhash_estimates_jaccard_reasonably():
    h = MinHasher(num_perm=256)
    a = "one two three four five six seven eight nine ten eleven twelve"
    b = "one two three four five six seven eight nine ten eleven thirteen"
    # true shingle-3 Jaccard = 9/11
    est = h.jaccard(h.signature(a), h.signature(b))
    assert est == pytest.approx(9 / 11, abs=0.12)
    assert h.jaccard(h.signature(a), h.signature(a)) == 1.0


def test_js_properties():
    p = {"a": 0.5, "b": 0.5}
    assert js_divergence(p, p) == pytest.approx(0.0, abs=1e-12)
    assert js_divergence({"a": 1.0}, {"b": 1.0}) == pytest.approx(1.0)  # bounded, finite
    assert js_divergence(p, {"a": 0.9, "b": 0.1}) == pytest.approx(js_divergence({"a": 0.9, "b": 0.1}, p))


def test_wasserstein_matches_pure_shift_and_flags_drift():
    rng = np.random.default_rng(0)
    ref = rng.normal(10, 2, 5000)
    assert wasserstein(ref, ref + 3.0) == pytest.approx(3.0, abs=1e-9)
    assert continuous_drift(ref, ref + 3.0, threshold=0.2 * ref.std()).drifted
    assert not continuous_drift(ref, rng.normal(10, 2, 5000), threshold=0.2 * ref.std()).drifted


def test_categorical_drift_threshold():
    ref = {"easy": 0.5, "medium": 0.3, "hard": 0.2}
    assert not categorical_drift(ref, {"easy": 0.48, "medium": 0.31, "hard": 0.21}).drifted
    assert categorical_drift(ref, {"easy": 0.1, "medium": 0.2, "hard": 0.7}).drifted
