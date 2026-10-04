"""Distribution drift: Jensen-Shannon for categorical mixes, Wasserstein for continuous values."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy import stats
from scipy.spatial.distance import jensenshannon


def js_divergence(p: Mapping[str, float] | Sequence[float], q: Mapping[str, float] | Sequence[float]) -> float:
    """Jensen-Shannon divergence in bits, bounded in [0, 1]. Symmetric and finite
    even when a category is missing from one side (unlike KL)."""
    if isinstance(p, Mapping) or isinstance(q, Mapping):
        keys = sorted(set(p) | set(q))  # type: ignore[arg-type]
        pv = np.array([p.get(k, 0.0) for k in keys], dtype=float)  # type: ignore[union-attr]
        qv = np.array([q.get(k, 0.0) for k in keys], dtype=float)  # type: ignore[union-attr]
    else:
        pv, qv = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    if pv.sum() <= 0 or qv.sum() <= 0:
        raise ValueError("distributions must have positive mass")
    return float(jensenshannon(pv / pv.sum(), qv / qv.sum(), base=2) ** 2)


def wasserstein(a: Sequence[float], b: Sequence[float]) -> float:
    """1-D earth mover's distance, in the units of the values (e.g. minutes)."""
    return float(stats.wasserstein_distance(a, b))


@dataclass(frozen=True)
class DriftFlag:
    metric: str
    value: float
    threshold: float
    drifted: bool


def categorical_drift(ref: Mapping[str, float], cur: Mapping[str, float], threshold: float = 0.05) -> DriftFlag:
    v = js_divergence(ref, cur)
    return DriftFlag("js_divergence", v, threshold, v > threshold)


def continuous_drift(ref: Sequence[float], cur: Sequence[float], threshold: float) -> DriftFlag:
    """`threshold` is in the metric's own units; scale it to the reference spread
    (e.g. 0.1 x reference std) so the rule is meaningful for that quantity."""
    v = wasserstein(ref, cur)
    return DriftFlag("wasserstein", v, threshold, v > threshold)
