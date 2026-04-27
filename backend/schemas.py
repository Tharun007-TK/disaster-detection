from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class InferenceResult(BaseModel):
    pre: str
    post: str
    checkpoint: str
    num_classes: int
    height: int
    width: int
    crs: Optional[str] = None
    pixel_counts: dict[str, int]
    damage_pct: dict[str, float]
    outputs: Optional[dict[str, str]] = None


class ResultSummary(BaseModel):
    id: str
    stem: str
    damage_pct: dict[str, float]
    mask_url: Optional[str] = None
    timestamp: Optional[str] = None


class PrecautionInfo(BaseModel):
    damage_class: int
    name: str
    color: str
    precautions: list[str]
    actions: list[str]
    evacuation: str
