"""Multiple-comparison control across slices/metrics."""
from __future__ import annotations

from typing import Sequence

import numpy as np


def benjamini_hochberg(pvalues: Sequence[float], q: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg step-up procedure.

    Controls the expected false discovery rate at q (assuming independence or
    positive dependence). It does not remove false positives; it bounds their
    expected share among the rejections.

    Returns (reject_mask, adjusted_pvalues) in the original order.
    """
    p = np.asarray(pvalues, dtype=float)
    if p.ndim != 1 or p.size == 0:
        raise ValueError("pvalues must be a non-empty 1-D sequence")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("pvalues must lie in [0, 1]")
    m = p.size
    order = np.argsort(p)
    ranked = p[order]
    adj_sorted = ranked * m / (np.arange(m) + 1)
    # enforce monotonicity from the largest p downwards
    adj_sorted = np.minimum.accumulate(adj_sorted[::-1])[::-1]
    adj_sorted = np.clip(adj_sorted, 0, 1)
    adjusted = np.empty(m)
    adjusted[order] = adj_sorted
    return adjusted <= q, adjusted


def bonferroni(pvalues: Sequence[float], alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Family-wise error control; stricter than BH. Use for critical-slice gating."""
    p = np.asarray(pvalues, dtype=float)
    adjusted = np.clip(p * p.size, 0, 1)
    return adjusted <= alpha, adjusted
