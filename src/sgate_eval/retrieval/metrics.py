"""Ranking metrics with graded relevance (authoritative=2, superseded=1, other=0)."""
from __future__ import annotations

import math
from typing import Mapping, Sequence


def success_at_1(ranked: Sequence[str], rel: Mapping[str, int], target_grade: int = 2) -> float:
    """1 if the top result is the authoritative document, else 0."""
    return float(bool(ranked) and rel.get(ranked[0], 0) >= target_grade)


def recall_at_k(ranked: Sequence[str], rel: Mapping[str, int], k: int, target_grade: int = 2) -> float:
    return float(any(rel.get(d, 0) >= target_grade for d in ranked[:k]))


def reciprocal_rank(ranked: Sequence[str], rel: Mapping[str, int], target_grade: int = 2) -> float:
    for i, d in enumerate(ranked, 1):
        if rel.get(d, 0) >= target_grade:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[str], rel: Mapping[str, int], k: int = 10) -> float:
    """NDCG with graded gains (2^grade - 1) and log2 position discount."""
    def dcg(grades):
        return sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(grades))

    actual = dcg([rel.get(d, 0) for d in ranked[:k]])
    ideal = dcg(sorted(rel.values(), reverse=True)[:k])
    return actual / ideal if ideal > 0 else 0.0


def buried(ranked: Sequence[str], rel: Mapping[str, int], k: int = 10) -> float:
    """1 if the authoritative doc is retrieved within top-k but NOT ranked first."""
    return float(recall_at_k(ranked, rel, k) == 1.0 and success_at_1(ranked, rel) == 0.0)
