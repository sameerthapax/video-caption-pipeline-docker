from __future__ import annotations

from pydantic import BaseModel, Field


class MusicMetadata(BaseModel):
    no_audio: bool = False
    present: bool = False
    type: str = ""
    instrumentation: str = ""
    energy: str = ""
    description: str = ""


class ToneMetadata(BaseModel):
    no_audio: bool = False
    style: str = ""
    emotion: str = ""
    delivery: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class TranscriptChunk(BaseModel):
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    no_audio: bool = False
    text: str = ""
    expressive_transcript: str = ""
    music: MusicMetadata = Field(default_factory=MusicMetadata)
    tone: ToneMetadata = Field(default_factory=ToneMetadata)


class TranscriptionRequestArtifact(BaseModel):
    job_id: str
    source_audio_storage_path: str
    transcript_window_seconds: float = 5.0
    provider: str = "google_gemini"
    status: str = "pending"
    provider_metadata: dict[str, str | None] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)
    transcript_chunks: list[TranscriptChunk] = Field(default_factory=list)
