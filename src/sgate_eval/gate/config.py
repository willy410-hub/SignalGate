"""Pre-registration: the gate's rules are frozen and hashed BEFORE results are seen."""
from __future__ import annotations

import hashlib
import json
from typing import Dict, List

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SliceSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    critical: bool = False
    # largest acceptable absolute drop on this slice (metric units, positive number)
    tolerance: float = Field(0.01, ge=0)
    min_n: int = Field(30, ge=2)


class PreRegistration(BaseModel):
    """Everything that decides a release, fixed in advance."""

    model_config = ConfigDict(frozen=True)

    primary_metric: str
    slices: List[SliceSpec]
    min_effect: float = Field(0.005, ge=0, description="smallest overall gain worth acting on")
    fdr_q: float = Field(0.05, gt=0, lt=1)
    confidence: float = Field(0.95, gt=0, lt=1)
    n_boot: int = Field(10_000, ge=1000)
    seed: int = 0
    canary_stages: List[float] = [0.02, 0.10]

    @model_validator(mode="after")
    def _unique_slices(self) -> "PreRegistration":
        names = [s.name for s in self.slices]
        if len(set(names)) != len(names):
            raise ValueError("slice names must be unique")
        if not self.slices:
            raise ValueError("at least one slice is required")
        return self

    def fingerprint(self) -> str:
        """Stable SHA-256 of the canonical JSON; store it before running the eval."""
        payload = json.dumps(self.model_dump(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def slice_map(self) -> Dict[str, SliceSpec]:
        return {s.name: s for s in self.slices}
