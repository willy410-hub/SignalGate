"""Invest / iterate / stop: decide from a learning curve whether more data is worth it.

Fits score(n) = a - b * n**(-c) to (n, score) pairs, then bootstraps the *predicted gain from
buying `extra` more items*. The pre-set rule compares the LOWER bound of that gain to a
minimum worthwhile gain, so noisy curves do not trigger spending.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence, Tuple

import numpy as np
from scipy.optimize import curve_fit


class Action(str, Enum):
    INVEST = "invest"
    ITERATE = "iterate"   # fix labels/guidelines/task design first
    STOP = "stop"


def _curve(n, a, b, c):
    return a - b * np.power(n, -c)


def fit_curve(n: Sequence[float], score: Sequence[float]) -> Tuple[float, float, float]:
    n = np.asarray(n, float); score = np.asarray(score, float)
    popt, _ = curve_fit(_curve, n, score, p0=[score.max() + 0.05, 1.0, 0.5],
                        bounds=([0, 0, 0.05], [2, 50, 2]), maxfev=20000)
    return tuple(float(p) for p in popt)


@dataclass(frozen=True)
class InvestmentReport:
    action: Action
    expected_gain: float
    gain_ci: Tuple[float, float]
    asymptote: float
    reason: str


def recommend(
    n: Sequence[float], score: Sequence[float], extra: float, min_gain: float = 0.01,
    plateau_gap: float = 0.02, n_boot: int = 400, seed: int = 0, noise_sd: float | None = None,
) -> InvestmentReport:
    n = np.asarray(n, float); score = np.asarray(score, float)
    a, b, c = fit_curve(n, score)
    cur = n.max()
    gain = float(_curve(cur + extra, a, b, c) - _curve(cur, a, b, c))
    resid = score - _curve(n, a, b, c)
    sd = float(noise_sd if noise_sd is not None else max(resid.std(ddof=1), 1e-6))
    rng = np.random.default_rng(seed)
    gains = []
    for _ in range(n_boot):
        y = _curve(n, a, b, c) + rng.normal(0, sd, n.size)
        try:
            pa, pb, pc = fit_curve(n, y)
        except Exception:
            continue
        gains.append(_curve(cur + extra, pa, pb, pc) - _curve(cur, pa, pb, pc))
    lo, hi = (float(np.quantile(gains, 0.05)), float(np.quantile(gains, 0.95))) if gains else (float("nan"),) * 2
    gap_to_target = max(0.0, a - score.max())
    if lo >= min_gain:
        act, why = Action.INVEST, f"gain lower bound {lo:+.3f} >= {min_gain:.3f}"
    elif hi < min_gain:
        act = Action.STOP
        why = f"even the optimistic gain {hi:+.3f} < {min_gain:.3f}: curve has plateaued"
    else:
        act = Action.ITERATE
        why = f"gain uncertain (CI [{lo:+.3f}, {hi:+.3f}]); improve label quality or task design before buying more"
    return InvestmentReport(act, gain, (lo, hi), a, why)
