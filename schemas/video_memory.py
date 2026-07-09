from __future__ import annotations

from pydantic import BaseModel, Field


class MemoryGlobalSetting(BaseModel):
    current_location_type: str = ""
    stable_environment_details: list[str] = Field(default_factory=list)
    setting_changes: list[str] = Field(default_factory=list)


class PersistentSubject(BaseModel):
    subject_id: str
    type: str = "unknown"
    first_seen_segment: int = Field(ge=0, le=4)
    last_seen_segment: int = Field(ge=0, le=4)
    appearance_summary: str = ""
    clothing_summary: str = ""
    known_actions: list[str] = Field(default_factory=list)
    current_state: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class PersistentObject(BaseModel):
    object_id: str
    name: str = ""
    first_seen_segment: int = Field(ge=0, le=4)
    last_seen_segment: int = Field(ge=0, le=4)
    description: str = ""
    state_history: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class TimelineEntry(BaseModel):
    segment_index: int = Field(ge=0, le=4)
    start: float = Field(ge=0.0)
    end: float = Field(ge=0.0)
    summary: str = ""
    key_events: list[str] = Field(default_factory=list)


class SegmentMemoryEntry(BaseModel):
    segment_index: int = Field(ge=0, le=4)
    start: float = Field(default=0.0, ge=0.0)
    end: float = Field(default=0.0, ge=0.0)
    memory: str = ""
    screen_and_motion_summary: str = ""
    subjects_summary: str = ""
    objects_summary: str = ""
    audio_summary: str = ""
    uncertainties: list[str] = Field(default_factory=list)


class VideoMemory(BaseModel):
    job_id: str
    segments_processed: int = Field(default=0, ge=0, le=5)
    global_setting: MemoryGlobalSetting = Field(default_factory=MemoryGlobalSetting)
    persistent_subjects: list[PersistentSubject] = Field(default_factory=list)
    persistent_objects: list[PersistentObject] = Field(default_factory=list)
    timeline: list[TimelineEntry] = Field(default_factory=list)
    segment_memories: list[SegmentMemoryEntry] = Field(default_factory=list)
    unresolved_uncertainties: list[str] = Field(default_factory=list)
