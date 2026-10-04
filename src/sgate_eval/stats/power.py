"""Power analysis for paired comparisons (normal approximation)."""
from __future__ import annotations

import math

from scipy import stats


def min_detectable_effect(n: int, sd_diff: float, alpha: float = 0.05, power: float = 0.8) -> float:
    """Smallest true mean paired difference detectable with the given power."""
    if n < 2 or sd_diff <= 0:
        raise ValueError("need n >= 2 and sd_diff > 0")
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    return float((z_a + z_b) * sd_diff / math.sqrt(n))


def required_sample_size(effect: float, sd_diff: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Paired items needed to detect `effect` with the given power."""
    if effect <= 0 or sd_diff <= 0:
        raise ValueError("effect and sd_diff must be positive")
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    return int(math.ceil(((z_a + z_b) * sd_diff / effect) ** 2))
