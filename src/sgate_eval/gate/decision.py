"""The release gate: turns paired per-item scores into block / scoped release / canary."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from ..stats import benjamini_hochberg, compare_paired
from .config import PreRegistration

Scores = Tuple[Sequence[float], Sequence[float]]  # (candidate, baseline), paired by item


class Decision(str, Enum):
    BLOCK = "block"
    SCOPED_RELEASE = "scoped_release"
    CANARY = "canary"


@dataclass
class SliceReport:
    name: str
    critical: bool
    n: int
    mean_diff: float
    ci_low: float
    ci_high: float
    p_value: float
    adj_p: float
    significant: bool          # survives BH at q
    regression: bool           # significant AND worse than zero by more than min_effect
    breach: bool               # even the optimistic CI end is worse than tolerance
    underpowered: bool


@dataclass
class GateReport:
    decision: Decision
    reasons: List[str]
    overall: Dict[str, float]
    slices: List[SliceReport]
    excluded_slices: List[str] = field(default_factory=list)
    registration_hash: str = ""
    next_step: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["decision"] = self.decision.value
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def to_markdown(self) -> str:
        lines = [
            f"**Decision: `{self.decision.value}`**",
            "",
            *(f"- {r}" for r in self.reasons),
            "",
            f"Next step: {self.next_step}",
            "",
            "| slice | n | diff | 95% CI | adj p | flag |",
            "|---|---|---|---|---|---|",
        ]
        for s in self.slices:
            flag = "BREACH" if s.breach else "regression" if s.regression else "underpowered" if s.underpowered else "ok"
            lines.append(
                f"| {s.name}{' *' if s.critical else ''} | {s.n} | {s.mean_diff:+.4f} "
                f"| [{s.ci_low:+.4f}, {s.ci_high:+.4f}] | {s.adj_p:.3f} | {flag} |"
            )
        lines.append("")
        lines.append("`*` = critical slice. Registration hash: `" + self.registration_hash[:12] + "`")
        return "\n".join(lines)


def run_gate(
    reg: PreRegistration,
    overall: Scores,
    per_slice: Mapping[str, Scores],
    registered_hash: Optional[str] = None,
) -> GateReport:
    """Evaluate a candidate against a baseline under a frozen pre-registration.

    `registered_hash` should be the fingerprint recorded before the run; a mismatch
    means the rules were changed after seeing data, which is refused.
    """
    fp = reg.fingerprint()
    if registered_hash is not None and registered_hash != fp:
        raise ValueError("pre-registration changed after it was recorded; refusing to gate")

    missing = [s.name for s in reg.slices if s.name not in per_slice]
    if missing:
        raise ValueError(f"missing scores for pre-registered slices: {missing}")

    ov = compare_paired(*overall, n_boot=reg.n_boot, confidence=reg.confidence, seed=reg.seed)

    results = []
    for spec in reg.slices:
        cand, base = per_slice[spec.name]
        results.append((spec, compare_paired(cand, base, reg.n_boot, reg.confidence, reg.seed)))

    _, adj = benjamini_hochberg([r.p_value for _, r in results], q=reg.fdr_q)

    reports: List[SliceReport] = []
    for (spec, r), a in zip(results, adj):
        sig = bool(a <= reg.fdr_q)
        reports.append(
            SliceReport(
                name=spec.name, critical=spec.critical, n=r.n, mean_diff=r.mean_diff,
                ci_low=r.ci_low, ci_high=r.ci_high, p_value=r.p_value, adj_p=float(a),
                significant=sig,
                regression=bool(sig and r.mean_diff <= -reg.min_effect),
                breach=bool(r.ci_high < -spec.tolerance),
                underpowered=r.n < spec.min_n,
            )
        )

    reasons: List[str] = []
    overall_info = {"mean_diff": ov.mean_diff, "ci_low": ov.ci_low, "ci_high": ov.ci_high, "p_value": ov.p_value, "n": ov.n}

    crit_bad = [s for s in reports if s.critical and (s.breach or s.regression)]
    noncrit_reg = [s for s in reports if not s.critical and (s.regression or s.breach)]
    under = [s for s in reports if s.underpowered]
    gained = ov.ci_low >= reg.min_effect

    if crit_bad:
        for s in crit_bad:
            why = "breach" if s.breach else "confirmed regression"
            reasons.append(f"critical slice '{s.name}': {why} (diff {s.mean_diff:+.4f}, CI [{s.ci_low:+.4f}, {s.ci_high:+.4f}])")
        decision = Decision.BLOCK
        nxt = "Patch the failing critical slice, then re-run a slice-only eval with a larger sample before widening traffic."
        excluded: List[str] = []
    elif not gained:
        reasons.append(
            f"overall gain not confirmed: CI lower bound {ov.ci_low:+.4f} < min effect {reg.min_effect:+.4f}"
        )
        decision = Decision.BLOCK
        nxt = "Collect more items or iterate on the candidate; do not claim improvement yet."
        excluded = []
    elif noncrit_reg:
        excluded = [s.name for s in noncrit_reg]
        reasons.append(f"overall gain confirmed (CI lower bound {ov.ci_low:+.4f}) but non-critical slices regressed: {excluded}")
        decision = Decision.SCOPED_RELEASE
        nxt = "Release with the regressed slices excluded; fix them in a follow-up and re-evaluate."
    else:
        excluded = []
        reasons.append(f"overall gain confirmed (CI lower bound {ov.ci_low:+.4f} >= {reg.min_effect:+.4f}); no slice regression")
        if under:
            reasons.append(f"underpowered slices cannot be cleared: {[s.name for s in under]}")
        decision = Decision.CANARY
        stages = " -> ".join(f"{int(p * 100)}%" for p in reg.canary_stages)
        nxt = f"Canary at {stages} with pre-written rollback triggers; widen only if every window stays inside them."

    return GateReport(decision, reasons, overall_info, reports, excluded, fp, nxt)
