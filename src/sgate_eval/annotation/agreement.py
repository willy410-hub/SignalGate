"""Inter-annotator agreement: Krippendorff's alpha and Cohen's kappa."""
from __future__ import annotations

from typing import Literal, Sequence

import numpy as np

Metric = Literal["nominal", "interval", "ordinal"]


def krippendorff_alpha(ratings: Sequence[Sequence[float]], metric: Metric = "nominal") -> float:
    """Krippendorff's alpha.

    `ratings` is units x coders; use float('nan') (or None) for a missing rating.
    Handles any number of coders and missing data. alpha = 1 is perfect agreement,
    0 is chance level. Units with fewer than two ratings carry no information and
    are ignored.
    """
    data = np.array(
        [[np.nan if v is None else v for v in row] for row in ratings], dtype=float
    )
    if data.ndim != 2:
        raise ValueError("ratings must be 2-D (units x coders)")
    units = [row[~np.isnan(row)] for row in data]
    units = [u for u in units if u.size >= 2]
    if not units:
        raise ValueError("need at least one unit with two or more ratings")

    cats = np.unique(np.concatenate(units))
    k = len(cats)
    index = {c: i for i, c in enumerate(cats)}
    o = np.zeros((k, k))
    for u in units:
        m = u.size
        for i in range(m):
            for j in range(m):
                if i != j:
                    o[index[u[i]], index[u[j]]] += 1.0 / (m - 1)
    n_c = o.sum(axis=1)
    n = n_c.sum()
    if k == 1:
        return 1.0  # no variation observed at all

    delta = np.zeros((k, k))
    for a in range(k):
        for b in range(k):
            if metric == "nominal":
                delta[a, b] = 0.0 if a == b else 1.0
            elif metric == "interval":
                delta[a, b] = (cats[a] - cats[b]) ** 2
            elif metric == "ordinal":
                lo, hi = min(a, b), max(a, b)
                delta[a, b] = (n_c[lo : hi + 1].sum() - (n_c[a] + n_c[b]) / 2) ** 2
            else:
                raise ValueError(f"unknown metric {metric}")

    d_o = (o * delta).sum()
    d_e = (np.outer(n_c, n_c) * delta).sum() / (n - 1)
    if d_e == 0:
        return 1.0
    return float(1.0 - d_o / d_e)


def cohens_kappa(a: Sequence, b: Sequence, weights: Literal[None, "linear", "quadratic"] = None) -> float:
    """Cohen's kappa for two raters on the same items (optionally weighted for ordinal labels)."""
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape or a.ndim != 1 or a.size == 0:
        raise ValueError("a and b must be equal-length non-empty 1-D sequences")
    cats = np.unique(np.concatenate([a, b]))
    k = len(cats)
    idx = {c: i for i, c in enumerate(cats)}
    conf = np.zeros((k, k))
    for x, y in zip(a, b):
        conf[idx[x], idx[y]] += 1
    conf /= conf.sum()
    expected = np.outer(conf.sum(axis=1), conf.sum(axis=0))
    i, j = np.indices((k, k))
    if weights is None:
        w = (i != j).astype(float)
    elif weights == "linear":
        w = np.abs(i - j) / max(k - 1, 1)
    elif weights == "quadratic":
        w = ((i - j) / max(k - 1, 1)) ** 2
    else:
        raise ValueError("weights must be None, 'linear' or 'quadratic'")
    denom = (w * expected).sum()
    if denom == 0:
        return 1.0
    return float(1.0 - (w * conf).sum() / denom)


def pairwise_kappa(labels: Sequence[Sequence], weights=None) -> dict[tuple[int, int], float]:
    """Cohen's kappa for every pair of coders. `labels` is coders x items (no missing values)."""
    out = {}
    m = len(labels)
    for i in range(m):
        for j in range(i + 1, m):
            out[(i, j)] = cohens_kappa(labels[i], labels[j], weights)
    return out
