from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class FrameArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    frame_id: str
    timestamp: float = Field(ge=0.0)
    local_path: str
    selection_reasons: list[str] = Field(default_factory=list)
    scene_change_score: float | None = None


class SceneCandidateScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timestamp: float = Field(ge=0.0)
    embedding_change: float | None = None
    histogram_change: float = Field(ge=0.0)
    pixel_change: float = Field(ge=0.0)
    raw_score: float = Field(ge=0.0)
    normalized_score: float = Field(ge=0.0)
    smoothed_score: float = Field(ge=0.0)
    selected: bool = False


class FrameExtractionArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str
    duration_seconds: float = Field(ge=0.0)
    target_frame_count: int = Field(ge=0)
    strategy: str
    scene_candidates: list[SceneCandidateScore] = Field(default_factory=list)
    frames: list[FrameArtifact] = Field(default_factory=list)
