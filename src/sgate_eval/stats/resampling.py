"""Paired resampling tests for model-vs-baseline comparisons.

Both tests are two-sided on the paired difference (candidate - baseline) and
deterministic given a seed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class PairedResult:
    mean_diff: float
    ci_low: float
    ci_high: float
    p_value: float
    n: int


def _paired(a: Sequence[float], b: Sequence[float]) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim != 1 or a.size < 2:
        raise ValueError("need two equal-length 1-D arrays with n >= 2")
    return a - b


def paired_bootstrap(
    candidate: Sequence[float],
    baseline: Sequence[float],
    n_boot: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI for mean paired difference. Returns (mean, lo, hi)."""
    d = _paired(candidate, baseline)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    means = d[idx].mean(axis=1)
    alpha = 1 - confidence
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(d.mean()), float(lo), float(hi)


def paired_permutation_test(
    candidate: Sequence[float],
    baseline: Sequence[float],
    n_perm: int = 10_000,
    seed: int = 0,
) -> float:
    """Two-sided sign-flip permutation p-value for mean paired difference."""
    d = _paired(candidate, baseline)
    rng = np.random.default_rng(seed)
    obs = abs(d.mean())
    signs = rng.choice([-1.0, 1.0], size=(n_perm, d.size))
    perm = np.abs((signs * d).mean(axis=1))
    # +1 smoothing so p is never exactly 0
    return float((np.sum(perm >= obs - 1e-15) + 1) / (n_perm + 1))


def compare_paired(
    candidate: Sequence[float],
    baseline: Sequence[float],
    n_boot: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> PairedResult:
    mean, lo, hi = paired_bootstrap(candidate, baseline, n_boot, confidence, seed)
    p = paired_permutation_test(candidate, baseline, n_boot, seed)
    return PairedResult(mean, lo, hi, p, len(candidate))
