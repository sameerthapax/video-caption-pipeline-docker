from __future__ import annotations

from pydantic import BaseModel, Field

from schemas.frames import FrameArtifact
from schemas.transcription import TranscriptChunk
from schemas.video import VideoMetadata


class SegmentVlmSummary(BaseModel):
    status: str = "pending"
    setting: str | None = None
    people: list[str] = Field(default_factory=list)
    objects: list[str] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    mood: str | None = None
    short_visual_summary: str | None = None


class SamplingConfig(BaseModel):
    uniform_count: int = 8
    scene_change_count: int = 8
    safety_count: int = 4
    scene_scan_interval_seconds: float = 1.0
    min_scene_spacing_seconds: float = 4.0
    dedupe_timestamp_threshold_seconds: float = 0.5
    score_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "embedding_change": 0.6,
            "histogram_change": 0.3,
            "pixel_change": 0.1,
        }
    )


class TemporalSegment(BaseModel):
    segment_index: int
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    percent_range: str
    frames: list[FrameArtifact] = Field(default_factory=list)
    transcript_chunks: list[TranscriptChunk] = Field(default_factory=list)
    segment_vlm_summary: SegmentVlmSummary = Field(default_factory=SegmentVlmSummary)
    segment_ground_truth: dict = Field(default_factory=dict)


class TemporalSegmentsArtifact(BaseModel):
    job_id: str
    video_metadata: VideoMetadata
    sampling_config: SamplingConfig
    segments: list[TemporalSegment] = Field(default_factory=list)
