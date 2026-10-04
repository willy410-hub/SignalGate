"""Ongoing annotator quality control: hidden gold items and calibration gating."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence

from ..stats import wilson_interval
from .agreement import krippendorff_alpha


@dataclass(frozen=True)
class GoldCheck:
    annotator: str
    n_gold: int
    n_correct: int
    accuracy: float
    ci_high: float
    flagged: bool


def gold_audit(
    answers: Mapping[str, Mapping[str, object]],
    gold: Mapping[str, object],
    min_accuracy: float = 0.8,
    min_items: int = 10,
) -> List[GoldCheck]:
    """Score each annotator on hidden gold items.

    An annotator is flagged only when the UPPER Wilson bound on their accuracy is
    below `min_accuracy` (we are confident they are under the bar) and they have
    answered at least `min_items` gold items. Small samples are never flagged.
    """
    out: List[GoldCheck] = []
    for who, resp in answers.items():
        scored = [(k, v) for k, v in resp.items() if k in gold]
        n = len(scored)
        correct = sum(1 for k, v in scored if gold[k] == v)
        if n == 0:
            out.append(GoldCheck(who, 0, 0, float("nan"), float("nan"), False))
            continue
        _, hi = wilson_interval(correct, n)
        out.append(GoldCheck(who, n, correct, correct / n, hi, bool(n >= min_items and hi < min_accuracy)))
    return out


@dataclass(frozen=True)
class CalibrationResult:
    alpha: float
    threshold: float
    passed: bool
    action: str


def calibration_gate(ratings, metric="ordinal", threshold: float = 0.7) -> CalibrationResult:
    """Decide whether a calibration batch is good enough to scale production labeling."""
    a = krippendorff_alpha(ratings, metric)
    if a >= threshold:
        return CalibrationResult(a, threshold, True, "scale up; keep 5-10% overlap and hidden gold")
    return CalibrationResult(
        a, threshold, False,
        "do not scale: cluster the disagreements, rewrite the guideline/anchors, re-run calibration",
    )
