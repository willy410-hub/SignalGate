import numpy as np
import pytest

from sgate_eval.annotation import (
    calibration_gate, cohens_kappa, gold_audit, krippendorff_alpha, pairwise_kappa, simulate_annotators,
)

nan = float("nan")
# Krippendorff's published example: 4 coders, 12 units, missing data
KRIPP = [
    [1, 1, nan, 1],
    [2, 2, 3, 2],
    [3, 3, 3, 3],
    [3, 3, 3, 3],
    [2, 2, 2, 2],
    [1, 2, 3, 4],
    [4, 4, 4, 4],
    [1, 1, 2, 1],
    [2, 2, 2, 2],
    [nan, 5, 5, 5],
    [nan, nan, 1, 1],
    [nan, 3, nan, nan],
]


def test_alpha_matches_published_example():
    assert krippendorff_alpha(KRIPP, "nominal") == pytest.approx(0.743, abs=0.001)
    assert krippendorff_alpha(KRIPP, "interval") == pytest.approx(0.849, abs=0.001)


def test_alpha_perfect_and_random():
    perfect = [[1, 1, 1], [2, 2, 2], [3, 3, 3], [1, 1, 1]]
    assert krippendorff_alpha(perfect) == pytest.approx(1.0)
    rng = np.random.default_rng(0)
    rand = rng.integers(0, 3, size=(500, 3)).tolist()
    assert abs(krippendorff_alpha(rand)) < 0.1


def test_alpha_requires_data():
    with pytest.raises(ValueError):
        krippendorff_alpha([[1, nan], [nan, 2]])


def test_alpha_ordinal_penalizes_far_disagreement_more():
    near = [[1, 2], [2, 3], [3, 4], [4, 5], [1, 1], [5, 5]]
    far = [[1, 5], [2, 4], [3, 5], [4, 1], [1, 1], [5, 5]]
    assert krippendorff_alpha(near, "ordinal") > krippendorff_alpha(far, "ordinal")


def test_cohens_kappa_textbook():
    a = ["y"] * 25 + ["n"] * 25
    b = ["y"] * 20 + ["n"] * 5 + ["y"] * 10 + ["n"] * 15
    assert cohens_kappa(a, b) == pytest.approx(0.4)
    assert cohens_kappa(a, a) == pytest.approx(1.0)


def test_pairwise_kappa_returns_all_pairs():
    labels = [[0, 1, 1, 0], [0, 1, 0, 0], [0, 1, 1, 0]]
    assert set(pairwise_kappa(labels)) == {(0, 1), (0, 2), (1, 2)}


def test_calibration_gate_passes_good_and_blocks_noisy_annotators():
    rng = np.random.default_rng(0)
    truth = rng.integers(1, 6, size=60)
    good = simulate_annotators(truth, [0.95, 0.95, 0.95], n_classes=6, seed=1)
    noisy = simulate_annotators(truth, [0.55, 0.5, 0.5], n_classes=6, seed=2)
    assert calibration_gate(good.tolist(), "nominal").passed
    res = calibration_gate(noisy.tolist(), "nominal")
    assert not res.passed and "rewrite" in res.action


def test_herding_inflates_agreement_while_hurting_accuracy():
    rng = np.random.default_rng(3)
    truth = rng.integers(0, 3, size=400)
    honest = simulate_annotators(truth, [0.7, 0.7, 0.7], 3, seed=4, herding=0.0)
    herd = simulate_annotators(truth, [0.7, 0.4, 0.4], 3, seed=4, herding=0.9)
    assert krippendorff_alpha(herd.tolist()) > krippendorff_alpha(honest.tolist())
    acc = lambda m: float(np.mean(m[:, 1:] == truth[:, None]))
    assert acc(herd) < acc(honest)


def test_gold_audit_flags_only_confidently_bad_annotators():
    gold = {f"g{i}": i % 2 for i in range(30)}
    answers = {
        "good": {k: v for k, v in gold.items()},
        "bad": {k: 1 - v for k, v in gold.items()},
        "tiny_bad": {k: 1 - v for k, v in list(gold.items())[:3]},
    }
    res = {r.annotator: r for r in gold_audit(answers, gold)}
    assert not res["good"].flagged
    assert res["bad"].flagged
    assert not res["tiny_bad"].flagged  # too few items to be sure
