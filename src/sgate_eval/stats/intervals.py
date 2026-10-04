"""Confidence intervals used by the gate.

- t_interval: Student's t for replicated continuous scores (small n).
- wilson_interval: binomial proportion, well-behaved near 0/1.
- clopper_pearson_upper: exact upper bound for rare critical failures.
- rule_of_three: quick 95% upper bound when zero failures were observed.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy import stats


def t_interval(values: Sequence[float], confidence: float = 0.95) -> tuple[float, float, float]:
    """Return (mean, lo, hi). Requires at least 2 values."""
    x = np.asarray(values, dtype=float)
    n = x.size
    if n < 2:
        raise ValueError("t_interval needs at least 2 replicates")
    mean = float(x.mean())
    sem = float(x.std(ddof=1) / math.sqrt(n))
    h = float(stats.t.ppf(0.5 + confidence / 2, df=n - 1)) * sem
    return mean, mean - h, mean + h


def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    if n <= 0 or not 0 <= successes <= n:
        raise ValueError("need 0 <= successes <= n and n > 0")
    z = float(stats.norm.ppf(0.5 + confidence / 2))
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def clopper_pearson_upper(failures: int, n: int, confidence: float = 0.95) -> float:
    """One-sided exact upper bound on the failure rate."""
    if n <= 0 or not 0 <= failures <= n:
        raise ValueError("need 0 <= failures <= n and n > 0")
    if failures == n:
        return 1.0
    return float(stats.beta.ppf(confidence, failures + 1, n - failures))


def rule_of_three(n: int) -> float:
    """Approximate 95% upper bound on the rate after 0 failures in n trials (3/n)."""
    if n <= 0:
        raise ValueError("n must be positive")
    return min(1.0, 3.0 / n)
