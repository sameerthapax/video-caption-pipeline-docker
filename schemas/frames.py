from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class FrameArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    frame_id: str
    timestamp: float = Field(ge=0.0)
    local_path: str
    selection_reasons: list[str] = Field(default_factory=list)


class FrameExtractionArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str
    duration_seconds: float = Field(ge=0.0)
    target_frame_count: int = Field(ge=0)
    strategy: str
    frames: list[FrameArtifact] = Field(default_factory=list)
