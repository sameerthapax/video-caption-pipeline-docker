from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from schemas.frames import FrameArtifact
from schemas.transcription import TranscriptChunk


Clarity = Literal["high", "medium", "low"]
AudioClarity = Literal["high", "medium", "low", "none"]
EvidenceKind = Literal["visual", "audio", "both"]
EntityType = Literal["person", "animal", "object", "unknown"]


class EvidenceQuality(BaseModel):
    visual_clarity: Clarity = "medium"
    audio_clarity: AudioClarity = "none"
    limitations: list[str] = Field(default_factory=list)

    @field_validator("limitations", mode="before")
    @classmethod
    def _normalize_limitations(cls, value: Any) -> list[str]:
        return _normalize_string_list(value, preferred_keys=("description", "summary", "text", "label"))


class SegmentDescription(BaseModel):
    short_summary: str = ""
    detailed_visual_description: str = ""
    movement_and_flow: str = ""
    audio_summary: str = ""


class SegmentSetting(BaseModel):
    location_type: str = ""
    visual_environment: str = ""
    background_details: list[str] = Field(default_factory=list)
    changes_from_previous_segment: str = ""

    @field_validator("background_details", mode="before")
    @classmethod
    def _normalize_background_details(cls, value: Any) -> list[str]:
        return _normalize_string_list(value, preferred_keys=("description", "summary", "text", "label", "name"))


class SubjectAppearance(BaseModel):
    visible_features: list[str] = Field(default_factory=list)
    clothing: list[str] = Field(default_factory=list)
    colors: list[str] = Field(default_factory=list)
    accessories: list[str] = Field(default_factory=list)
    pose_or_posture: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    @field_validator("visible_features", "clothing", "colors", "accessories", mode="before")
    @classmethod
    def _normalize_lists(cls, value: Any) -> list[str]:
        return _normalize_string_list(value, preferred_keys=("description", "summary", "text", "label", "name"))


class SubjectEmotion(BaseModel):
    label: str = ""
    evidence: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class SubjectObservation(BaseModel):
    subject_id: str
    is_new: bool = True
    matched_previous_subject_id: str | None = None
    type: EntityType = "unknown"
    appearance: SubjectAppearance = Field(default_factory=SubjectAppearance)
    actions: list[str] = Field(default_factory=list)
    movement: str = ""
    facial_expression: str = ""
    emotion_or_tone: SubjectEmotion = Field(default_factory=SubjectEmotion)
    state_change: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    @field_validator("actions", mode="before")
    @classmethod
    def _normalize_actions(cls, value: Any) -> list[str]:
        return _normalize_string_list(value, preferred_keys=("description", "summary", "text", "label", "name"))


class ObjectObservation(BaseModel):
    object_id: str
    is_new: bool = True
    matched_previous_object_id: str | None = None
    name: str = ""
    description: str = ""
    location_in_scene: str = ""
    state: str = ""
    interaction: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class EventObservation(BaseModel):
    event_id: str
    approx_timestamp: float = Field(default=0.0, ge=0.0)
    description: str = ""
    subjects_involved: list[str] = Field(default_factory=list)
    objects_involved: list[str] = Field(default_factory=list)
    evidence: EvidenceKind = "visual"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    @field_validator("subjects_involved", "objects_involved", mode="before")
    @classmethod
    def _normalize_entities(cls, value: Any) -> list[str]:
        return _normalize_string_list(
            value,
            preferred_keys=("subject_id", "object_id", "name", "label", "description", "text"),
        )


class VisibleTextObservation(BaseModel):
    text: str = ""
    location: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class MusicObservation(BaseModel):
    present: bool = False
    type: str = ""
    instrumentation: str = ""
    energy: str = ""
    description: str = ""


class AudioToneObservation(BaseModel):
    style: str = ""
    emotion: str = ""
    delivery: str = ""
    evidence: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class AudioObservations(BaseModel):
    spoken_content_summary: str = ""
    music: MusicObservation = Field(default_factory=MusicObservation)
    tone: AudioToneObservation = Field(default_factory=AudioToneObservation)


class ContinuityUpdate(BaseModel):
    same_subjects_as_before: list[str] = Field(default_factory=list)
    same_objects_as_before: list[str] = Field(default_factory=list)
    new_subjects: list[str] = Field(default_factory=list)
    new_objects: list[str] = Field(default_factory=list)
    resolved_uncertainties: list[str] = Field(default_factory=list)
    new_uncertainties: list[str] = Field(default_factory=list)
    important_changes: list[str] = Field(default_factory=list)

    @field_validator(
        "same_subjects_as_before",
        "same_objects_as_before",
        "new_subjects",
        "new_objects",
        "resolved_uncertainties",
        "new_uncertainties",
        "important_changes",
        mode="before",
    )
    @classmethod
    def _normalize_lists(cls, value: Any) -> list[str]:
        return _normalize_string_list(
            value,
            preferred_keys=("subject_id", "object_id", "name", "label", "description", "summary", "text"),
        )


class MemoryUpdateForNextSegment(BaseModel):
    segment_memory: str = ""
    persistent_subjects: list[str] = Field(default_factory=list)
    persistent_objects: list[str] = Field(default_factory=list)
    persistent_setting: str = ""
    open_actions_or_context: list[str] = Field(default_factory=list)
    timeline_update: list[str] = Field(default_factory=list)

    @field_validator(
        "persistent_subjects",
        "persistent_objects",
        "open_actions_or_context",
        "timeline_update",
        mode="before",
    )
    @classmethod
    def _normalize_lists(cls, value: Any) -> list[str]:
        return _normalize_string_list(
            value,
            preferred_keys=("subject_id", "object_id", "name", "label", "description", "summary", "text"),
        )


class SegmentVlmResponse(BaseModel):
    status: str = "completed"
    errors: list[str] = Field(default_factory=list)
    raw_text: str | None = None
    segment_index: int = Field(ge=0, le=4)
    segment_start: float = Field(ge=0.0)
    segment_end: float = Field(ge=0.0)
    evidence_quality: EvidenceQuality = Field(default_factory=EvidenceQuality)
    segment_description: SegmentDescription = Field(default_factory=SegmentDescription)
    setting: SegmentSetting = Field(default_factory=SegmentSetting)
    subjects: list[SubjectObservation] = Field(default_factory=list)
    objects: list[ObjectObservation] = Field(default_factory=list)
    events: list[EventObservation] = Field(default_factory=list)
    visible_text: list[VisibleTextObservation] = Field(default_factory=list)
    audio_observations: AudioObservations = Field(default_factory=AudioObservations)
    continuity_update: ContinuityUpdate = Field(default_factory=ContinuityUpdate)
    memory_update_for_next_segment: MemoryUpdateForNextSegment = Field(default_factory=MemoryUpdateForNextSegment)


class VlmSegmentArtifactEntry(BaseModel):
    segment_index: int = Field(ge=0, le=4)
    input_frames: list[FrameArtifact] = Field(default_factory=list)
    input_transcript_chunks: list[TranscriptChunk] = Field(default_factory=list)
    vlm_response: SegmentVlmResponse


class VlmSegmentsArtifact(BaseModel):
    job_id: str
    segments: list[VlmSegmentArtifactEntry] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class GlobalSummaryTimelineEntry(BaseModel):
    segment_index: int = Field(ge=0, le=4)
    summary: str = ""
    key_events: list[str] = Field(default_factory=list)
    visual_facts: list[str] = Field(default_factory=list)
    transcript_facts: list[str] = Field(default_factory=list)
    continuity_notes: list[str] = Field(default_factory=list)

    @field_validator("key_events", "visual_facts", "transcript_facts", "continuity_notes", mode="before")
    @classmethod
    def _normalize_string_fields(cls, value: Any) -> list[str]:
        return _normalize_string_list(
            value,
            preferred_keys=("description", "summary", "text", "event_id", "name", "label"),
        )


class GlobalSpeakerObservation(BaseModel):
    speaker_label: str = ""
    speaking_style: str = ""
    tone_or_emotion: str = ""
    evidence: list[str] = Field(default_factory=list)
    appears_across_segments: list[int] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    @field_validator("evidence", mode="before")
    @classmethod
    def _normalize_evidence(cls, value: Any) -> list[str]:
        return _normalize_string_list(
            value,
            preferred_keys=("description", "summary", "text", "evidence", "label"),
        )


class GlobalTranscriptAlignmentEntry(BaseModel):
    segment_index: int = Field(ge=0, le=4)
    transcript_summary: str = ""
    visual_alignment: str = ""
    speaker_changes: list[str] = Field(default_factory=list)
    tone_notes: list[str] = Field(default_factory=list)
    mismatches_or_uncertainties: list[str] = Field(default_factory=list)

    @field_validator("speaker_changes", "tone_notes", "mismatches_or_uncertainties", mode="before")
    @classmethod
    def _normalize_string_fields(cls, value: Any) -> list[str]:
        return _normalize_string_list(
            value,
            preferred_keys=("description", "summary", "text", "note", "label"),
        )


class GlobalSceneChangeEntry(BaseModel):
    segment_index: int | None = None
    apparent_change: str = ""
    related_subjects: list[str] = Field(default_factory=list)
    related_objects: list[str] = Field(default_factory=list)

    @field_validator("related_subjects", "related_objects", mode="before")
    @classmethod
    def _normalize_related_entities(cls, value: Any) -> list[str]:
        return _normalize_string_list(
            value,
            preferred_keys=("subject_id", "object_id", "name", "label", "description", "text"),
        )


class GlobalContinuityEntry(BaseModel):
    segment_index: int | None = None
    continuity_note: str = ""
    continued_subjects: list[str] = Field(default_factory=list)
    new_subjects: list[str] = Field(default_factory=list)
    continued_objects: list[str] = Field(default_factory=list)
    new_objects: list[str] = Field(default_factory=list)

    @field_validator(
        "continued_subjects",
        "new_subjects",
        "continued_objects",
        "new_objects",
        mode="before",
    )
    @classmethod
    def _normalize_entities(cls, value: Any) -> list[str]:
        return _normalize_string_list(
            value,
            preferred_keys=("subject_id", "object_id", "name", "label", "description", "text"),
        )


class GlobalTrackedEntityEntry(BaseModel):
    object_id: str = ""
    name: str = ""
    tracking_summary: str = ""
    entity_type: str = ""
    appears_across_segments: list[int] = Field(default_factory=list)


class GlobalFactualSummary(BaseModel):
    factual_summary: str = ""
    detailed_ground_truth: str = ""
    detailed_timeline: list[GlobalSummaryTimelineEntry] = Field(default_factory=list)
    main_subjects: list[str] = Field(default_factory=list)
    main_objects: list[str] = Field(default_factory=list)
    setting_summary: str = ""
    audio_summary: str = ""
    transcript_visual_alignment: list[GlobalTranscriptAlignmentEntry] = Field(default_factory=list)
    speaker_analysis: list[GlobalSpeakerObservation] = Field(default_factory=list)
    scene_change_overview: list[GlobalSceneChangeEntry] = Field(default_factory=list)
    continuity_overview: list[GlobalContinuityEntry] = Field(default_factory=list)
    object_and_subject_tracking: list[GlobalTrackedEntityEntry] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    errors: list[str] = Field(default_factory=list)

    @field_validator("main_subjects", mode="before")
    @classmethod
    def _normalize_main_subjects(cls, value: Any) -> list[str]:
        return _normalize_string_list(
            value,
            preferred_keys=("subject_id", "name", "label", "description", "summary", "text"),
        )

    @field_validator("main_objects", mode="before")
    @classmethod
    def _normalize_main_objects(cls, value: Any) -> list[str]:
        return _normalize_string_list(
            value,
            preferred_keys=("object_id", "name", "label", "description", "summary", "text"),
        )

    @field_validator("scene_change_overview", mode="before")
    @classmethod
    def _normalize_scene_change_overview(cls, value: Any) -> list[dict[str, Any]]:
        return _normalize_typed_list(
            value,
            text_field="apparent_change",
            alias_keys=("summary", "description", "text"),
        )

    @field_validator("continuity_overview", mode="before")
    @classmethod
    def _normalize_continuity_overview(cls, value: Any) -> list[dict[str, Any]]:
        return _normalize_typed_list(
            value,
            text_field="continuity_note",
            alias_keys=("summary", "description", "text", "note"),
        )

    @field_validator("object_and_subject_tracking", mode="before")
    @classmethod
    def _normalize_object_tracking(cls, value: Any) -> list[dict[str, Any]]:
        return _normalize_typed_list(
            value,
            text_field="tracking_summary",
            alias_keys=("summary", "description", "text"),
            fallback_name_keys=("name", "label", "object_id"),
        )


def _normalize_typed_list(
    value: Any,
    *,
    text_field: str,
    alias_keys: tuple[str, ...],
    fallback_name_keys: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        value = [value]

    normalized: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, str):
            text = item.strip()
            if text:
                normalized.append({text_field: text})
            continue
        if isinstance(item, dict):
            candidate = dict(item)
            if not isinstance(candidate.get(text_field), str) or not candidate.get(text_field, "").strip():
                for key in alias_keys:
                    raw = item.get(key)
                    if isinstance(raw, str) and raw.strip():
                        candidate[text_field] = raw.strip()
                        break
            if (not isinstance(candidate.get(text_field), str) or not candidate.get(text_field, "").strip()) and fallback_name_keys:
                for key in fallback_name_keys:
                    raw = item.get(key)
                    if isinstance(raw, str) and raw.strip():
                        candidate[text_field] = raw.strip()
                        break
            if isinstance(candidate.get(text_field), str) and candidate[text_field].strip():
                candidate[text_field] = candidate[text_field].strip()
            else:
                candidate[text_field] = str(item)
            normalized.append(candidate)
            continue
        normalized.append({text_field: str(item)})
    return normalized


def _normalize_string_list(
    value: Any,
    *,
    preferred_keys: tuple[str, ...],
) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        value = [value]

    normalized: list[str] = []
    for item in value:
        if isinstance(item, str):
            text = item.strip()
            if text:
                normalized.append(text)
            continue
        if isinstance(item, dict):
            for key in preferred_keys:
                raw = item.get(key)
                if isinstance(raw, str) and raw.strip():
                    normalized.append(raw.strip())
                    break
            else:
                normalized.append(str(item))
            continue
        normalized.append(str(item))
    return normalized


class StyledCaptionVariant(BaseModel):
    style_name: Literal["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]
    caption: str = ""
    grounded_facts_used: list[str] = Field(default_factory=list)
    safety_notes: list[str] = Field(default_factory=list)


class FinalCaptionResult(BaseModel):
    job_id: str
    neutral_summary: str = ""
    formal_caption: str = ""
    sarcastic_caption: str = ""
    humorous_tech_caption: str = ""
    humorous_non_tech_caption: str = ""
    source_global_factual_summary_path: str = ""
    captions: dict[str, StyledCaptionVariant] = Field(default_factory=dict)
