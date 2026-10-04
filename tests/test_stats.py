import math

import numpy as np
import pytest

from sgate_eval.stats import (
    benjamini_hochberg, bonferroni, clopper_pearson_upper, min_detectable_effect,
    paired_bootstrap, paired_permutation_test, required_sample_size, rule_of_three,
    t_interval, wilson_interval,
)


def test_bh_textbook_example():
    # classic example: m=5, q=0.05 -> only the two smallest are rejected
    p = [0.001, 0.008, 0.039, 0.041, 0.6]
    reject, adj = benjamini_hochberg(p, q=0.05)
    # hand-computed: p*m/rank = [.005, .02, .065, .05125, .6]; step-up min from the
    # top gives adjusted [.005, .02, .05125, .05125, .6] -> only the two smallest pass
    assert reject.tolist() == [True, True, False, False, False]
    assert adj[0] == pytest.approx(0.005)
    assert adj[1] == pytest.approx(0.02)
    assert adj[2] == pytest.approx(0.05125, abs=1e-4)
    assert adj[3] == pytest.approx(0.05125, abs=1e-4)
    assert adj[4] == pytest.approx(0.6)


def test_bh_preserves_input_order():
    p = [0.6, 0.001, 0.04]
    reject, _ = benjamini_hochberg(p, q=0.05)
    assert reject[1] and not reject[0]


def test_bh_is_less_strict_than_bonferroni():
    p = np.array([0.001, 0.012, 0.02, 0.03, 0.9])
    bh, _ = benjamini_hochberg(p, 0.05)
    bf, _ = bonferroni(p, 0.05)
    assert bh.sum() >= bf.sum()


def test_bh_rejects_bad_input():
    with pytest.raises(ValueError):
        benjamini_hochberg([1.2], 0.05)
    with pytest.raises(ValueError):
        benjamini_hochberg([], 0.05)


def test_bh_null_slices_few_false_discoveries():
    # 40 slices, no real effect: raw alpha=0.05 gives ~2 hits, BH should give ~0
    rng = np.random.default_rng(1)
    raw_hits, bh_hits = 0, 0
    for _ in range(200):
        p = rng.uniform(size=40)
        raw_hits += (p < 0.05).sum()
        bh_hits += benjamini_hochberg(p, 0.05)[0].sum()
    assert raw_hits / 200 > 1.5
    assert bh_hits / 200 < 0.2


def test_clopper_pearson_zero_failures_matches_closed_form():
    ub = clopper_pearson_upper(0, 100, 0.95)
    assert ub == pytest.approx(1 - 0.05 ** (1 / 100), rel=1e-6)
    assert rule_of_three(100) == pytest.approx(0.03)
    assert ub < rule_of_three(100) + 1e-3


def test_wilson_known_value():
    lo, hi = wilson_interval(8, 10, 0.95)
    assert lo == pytest.approx(0.490, abs=0.005)
    assert hi == pytest.approx(0.943, abs=0.005)
    assert 0 <= wilson_interval(0, 10)[0] <= wilson_interval(0, 10)[1] <= 1


def test_t_interval_matches_manual():
    mean, lo, hi = t_interval([0.70, 0.72, 0.74, 0.71, 0.73])
    assert mean == pytest.approx(0.72)
    assert lo < mean < hi
    with pytest.raises(ValueError):
        t_interval([1.0])


def test_bootstrap_detects_real_shift_and_is_deterministic():
    rng = np.random.default_rng(0)
    base = rng.normal(0.6, 0.1, 200)
    cand = base + 0.05 + rng.normal(0, 0.02, 200)
    m1 = paired_bootstrap(cand, base, 2000, seed=3)
    m2 = paired_bootstrap(cand, base, 2000, seed=3)
    assert m1 == m2
    assert m1[1] > 0  # CI excludes zero
    assert paired_permutation_test(cand, base, 2000, seed=3) < 0.01


def test_permutation_null_is_not_significant():
    rng = np.random.default_rng(5)
    base = rng.normal(0.6, 0.1, 100)
    cand = base + rng.normal(0, 0.05, 100)
    assert paired_permutation_test(cand, base, 2000, seed=1) > 0.05


def test_power_roundtrip():
    n = required_sample_size(0.02, 0.1)
    assert min_detectable_effect(n, 0.1) <= 0.02 + 1e-9
    assert min_detectable_effect(n - 1, 0.1) > 0.02 - 1e-3
    assert required_sample_size(0.01, 0.1) > required_sample_size(0.02, 0.1)
    assert math.isfinite(min_detectable_effect(50, 0.1))
