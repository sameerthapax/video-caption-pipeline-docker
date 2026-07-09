from __future__ import annotations

from pydantic import BaseModel, Field


class VlmFramePlaceholder(BaseModel):
    status: str = "pending"
    description: str | None = None
    objects: list[str] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    setting: str | None = None
    mood: str | None = None


class FrameArtifact(BaseModel):
    frame_id: str
    timestamp: float = Field(ge=0)
    storage_path: str
    local_path: str
    selection_reasons: list[str] = Field(default_factory=list)
    scene_change_score: float | None = None
    vlm: VlmFramePlaceholder = Field(default_factory=VlmFramePlaceholder)


class SceneCandidateScore(BaseModel):
    timestamp: float = Field(ge=0)
    embedding_change: float | None = None
    histogram_change: float = Field(ge=0)
    pixel_change: float = Field(ge=0)
    raw_score: float = Field(ge=0)
    normalized_score: float = Field(ge=0)
    smoothed_score: float = Field(ge=0)
    selected: bool = False


class DedupeDecision(BaseModel):
    kept_timestamp: float = Field(ge=0)
    dropped_timestamp: float = Field(ge=0)
    reason: str
    replacement_timestamp: float | None = Field(default=None, ge=0)


class FinalFrameSelection(BaseModel):
    timestamp: float = Field(ge=0)
    selection_reasons: list[str] = Field(default_factory=list)
    scene_change_score: float | None = None


class FrameSamplingArtifact(BaseModel):
    job_id: str
    scene_scan_interval_seconds: float = 1.0
    embedding_available: bool = False
    fallback_reason: str | None = None
    all_candidate_scene_change_frames: list[SceneCandidateScore] = Field(default_factory=list)
    selected_top_scene_change_frames: list[float] = Field(default_factory=list)
    final_selected_frames: list[FinalFrameSelection] = Field(default_factory=list)
    deduplication_decisions: list[DedupeDecision] = Field(default_factory=list)
