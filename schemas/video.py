from __future__ import annotations

from pydantic import BaseModel, Field


class VideoMetadata(BaseModel):
    duration: float = Field(ge=0)
    fps: float | None = Field(default=None, ge=0)
    width: int | None = Field(default=None, ge=0)
    height: int | None = Field(default=None, ge=0)
    frame_count: int = Field(default=0, ge=0)
    has_audio: bool = False

