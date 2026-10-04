"""SIMULATED annotators.

Everything here generates synthetic labels from a known ground truth. It exists
to test the agreement and gold-audit code and to drive the RL environment. These
numbers are NOT human agreement data and must never be reported as such.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def simulate_annotators(
    truth: Sequence[int],
    accuracies: Sequence[float],
    n_classes: int,
    seed: int = 0,
    herding: float = 0.0,
) -> np.ndarray:
    """Return labels shaped (items, annotators).

    Each annotator is correct with its own probability; otherwise picks a wrong
    class at random. With herding > 0, annotators later in the list copy the
    running majority of earlier annotators with that probability regardless of
    the truth, which inflates agreement while hurting accuracy.
    """
    rng = np.random.default_rng(seed)
    truth = np.asarray(truth)
    n, m = len(truth), len(accuracies)
    out = np.zeros((n, m), dtype=int)
    for i in range(n):
        for j, acc in enumerate(accuracies):
            if j > 0 and herding > 0 and rng.random() < herding:
                vals, counts = np.unique(out[i, :j], return_counts=True)
                out[i, j] = vals[np.argmax(counts)]
            elif rng.random() < acc:
                out[i, j] = truth[i]
            else:
                wrong = [c for c in range(n_classes) if c != truth[i]]
                out[i, j] = rng.choice(wrong)
    return out
