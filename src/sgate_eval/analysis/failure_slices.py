"""Root-cause helper: which attribute values are over-represented among failures?

For every (attribute, value) pair we compare the failure rate inside the slice to the
rest of the data with Fisher's exact test, correct all tests with Benjamini-Hochberg,
and rank survivors by risk ratio. It points to where to look; it does not prove cause.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
from scipy.stats import fisher_exact

from ..stats import benjamini_hochberg


@dataclass(frozen=True)
class SliceFinding:
    attribute: str
    value: Any
    n: int
    failures: int
    failure_rate: float
    rest_rate: float
    risk_ratio: float
    p_value: float
    adj_p: float
    significant: bool


def discover_failure_slices(
    records: Sequence[Mapping[str, Any]],
    failed_key: str,
    attributes: Sequence[str],
    min_n: int = 10,
    q: float = 0.05,
) -> List[SliceFinding]:
    fails = np.array([bool(r[failed_key]) for r in records])
    cand = []
    for attr in attributes:
        vals = np.array([r[attr] for r in records], dtype=object)
        for v in sorted(set(vals.tolist()), key=str):
            inside = vals == v
            n_in = int(inside.sum())
            n_out = len(records) - n_in
            if n_in < min_n or n_out < min_n:
                continue
            f_in, f_out = int(fails[inside].sum()), int(fails[~inside].sum())
            table = [[f_in, n_in - f_in], [f_out, n_out - f_out]]
            _, p = fisher_exact(table, alternative="greater")  # only slices WORSE than the rest
            r_in, r_out = f_in / n_in, f_out / n_out
            rr = r_in / r_out if r_out > 0 else float("inf")
            cand.append((attr, v, n_in, f_in, r_in, r_out, rr, float(p)))
    if not cand:
        return []
    reject, adj = benjamini_hochberg([c[-1] for c in cand], q)
    out = [SliceFinding(*c, float(a), bool(rj)) for c, a, rj in zip(cand, adj, reject)]
    return sorted(out, key=lambda f: (not f.significant, -f.risk_ratio if np.isfinite(f.risk_ratio) else -1e9, f.adj_p))
