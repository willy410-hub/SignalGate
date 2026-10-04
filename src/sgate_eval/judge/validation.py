"""Validate an LLM-style judge BEFORE trusting its scores.

The judge is any callable. Checks implemented:
- agreement with trusted labels (Cohen's kappa)
- position bias in pairwise judging (swap A/B; a consistent judge flips its answer)
- verbosity bias (does it prefer the longer answer when quality is held equal?)
No real model is called here; tests use SIMULATED judges with known biases.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence, Tuple

import numpy as np

from ..annotation import cohens_kappa
from ..stats import wilson_interval

PairJudge = Callable[[str, str, str], str]  # (prompt, answer_a, answer_b) -> "A" | "B"


@dataclass(frozen=True)
class AgreementReport:
    kappa: float
    accuracy: float
    n: int
    passed: bool


def agreement_with_labels(judge_labels: Sequence, trusted_labels: Sequence, min_kappa: float = 0.6) -> AgreementReport:
    k = cohens_kappa(judge_labels, trusted_labels)
    acc = float(np.mean(np.asarray(judge_labels) == np.asarray(trusted_labels)))
    return AgreementReport(k, acc, len(judge_labels), k >= min_kappa)


@dataclass(frozen=True)
class PositionBiasReport:
    inconsistent_rate: float
    ci: Tuple[float, float]
    first_position_rate: float
    biased: bool


def position_bias(judge: PairJudge, items: Sequence[Tuple[str, str, str]], max_inconsistent: float = 0.1) -> PositionBiasReport:
    """For each (prompt, a, b) ask twice with positions swapped. A position-consistent
    judge picks the same *answer* both times. Only meaningful for pairwise judging."""
    inconsistent = first = 0
    for prompt, a, b in items:
        v1 = judge(prompt, a, b)           # a first
        v2 = judge(prompt, b, a)           # b first
        pick1 = a if v1 == "A" else b
        pick2 = b if v2 == "A" else a
        inconsistent += pick1 != pick2
        first += (v1 == "A") + (v2 == "A")
    n = len(items)
    lo, hi = wilson_interval(inconsistent, n)
    return PositionBiasReport(inconsistent / n, (lo, hi), first / (2 * n), bool(lo > max_inconsistent))


@dataclass(frozen=True)
class VerbosityReport:
    longer_win_rate: float
    ci: Tuple[float, float]
    biased: bool


def verbosity_bias(judge: PairJudge, equal_quality_pairs: Sequence[Tuple[str, str, str]]) -> VerbosityReport:
    """Each pair is (prompt, short_answer, long_answer) with the SAME quality by construction.
    An unbiased judge should prefer the longer one about half the time."""
    wins = 0
    for prompt, short, long_ in equal_quality_pairs:
        v1 = judge(prompt, short, long_)
        v2 = judge(prompt, long_, short)
        wins += (v1 == "B") + (v2 == "A")
    n = 2 * len(equal_quality_pairs)
    lo, hi = wilson_interval(wins, n)
    return VerbosityReport(wins / n, (lo, hi), bool(lo > 0.5))
