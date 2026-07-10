from __future__ import annotations

from schemas.segments import TemporalSegment
from schemas.video_memory import (
    PersistentObject,
    PersistentSubject,
    SegmentMemoryEntry,
    TimelineEntry,
    VideoMemory,
)
from schemas.vlm import ObjectObservation, SegmentVlmResponse, SubjectObservation
from worker.config.settings import settings


def create_video_memory(*, job_id: str) -> VideoMemory:
    return VideoMemory(
        job_id=job_id,
        segment_memories=[SegmentMemoryEntry(segment_index=index) for index in range(settings.pipeline_segment_count)],
    )


def merge_segment_into_memory(*, memory: VideoMemory, segment: TemporalSegment, response: SegmentVlmResponse) -> VideoMemory:
    _merge_setting(memory=memory, response=response)
    normalized_subjects = [_normalize_subject(subject) for subject in response.subjects]
    normalized_objects = [_normalize_object(obj) for obj in response.objects]

    for subject in normalized_subjects:
        _merge_subject(memory=memory, subject=subject, segment_index=segment.segment_index)
    for obj in normalized_objects:
        _merge_object(memory=memory, obj=obj, segment_index=segment.segment_index)

    entry = SegmentMemoryEntry(
        segment_index=segment.segment_index,
        start=segment.start,
        end=segment.end,
        memory=response.memory_update_for_next_segment.segment_memory,
        screen_and_motion_summary=response.segment_description.movement_and_flow,
        subjects_summary=", ".join(item.subject_id for item in normalized_subjects),
        objects_summary=", ".join(item.object_id for item in normalized_objects),
        audio_summary=response.audio_observations.spoken_content_summary or response.segment_description.audio_summary,
        uncertainties=response.continuity_update.new_uncertainties[:],
    )
    memory.segment_memories[segment.segment_index] = entry

    timeline_entry = TimelineEntry(
        segment_index=segment.segment_index,
        start=segment.start,
        end=segment.end,
        summary=response.segment_description.short_summary,
        key_events=_dedupe_preserve_order(
            [
                event.description
                for event in response.events
                if event.description
            ] + response.memory_update_for_next_segment.timeline_update
        ),
    )
    memory.timeline = [item for item in memory.timeline if item.segment_index != segment.segment_index]
    memory.timeline.append(timeline_entry)
    memory.timeline.sort(key=lambda item: item.segment_index)

    resolved = set(response.continuity_update.resolved_uncertainties)
    memory.unresolved_uncertainties = [
        item for item in memory.unresolved_uncertainties if item not in resolved
    ]
    for item in _dedupe_preserve_order(response.continuity_update.new_uncertainties):
        if item and item not in memory.unresolved_uncertainties:
            memory.unresolved_uncertainties.append(item)

    memory.segments_processed = max(memory.segments_processed, segment.segment_index + 1)
    return memory


def _merge_setting(*, memory: VideoMemory, response: SegmentVlmResponse) -> None:
    if response.setting.location_type:
        memory.global_setting.current_location_type = response.setting.location_type
    for item in response.setting.background_details:
        if item and item not in memory.global_setting.stable_environment_details:
            memory.global_setting.stable_environment_details.append(item)


def _normalize_subject(subject: SubjectObservation) -> SubjectObservation:
    if subject.matched_previous_subject_id:
        subject.subject_id = subject.matched_previous_subject_id
        subject.is_new = False
    return subject


def _normalize_object(obj: ObjectObservation) -> ObjectObservation:
    if obj.matched_previous_object_id:
        obj.object_id = obj.matched_previous_object_id
        obj.is_new = False
    return obj


def _merge_subject(*, memory: VideoMemory, subject: SubjectObservation, segment_index: int) -> None:
    existing = next((item for item in memory.persistent_subjects if item.subject_id == subject.subject_id), None)
    appearance_summary = ", ".join(subject.appearance.visible_features)
    clothing_summary = ", ".join(subject.appearance.clothing)
    if existing is None:
        memory.persistent_subjects.append(
            PersistentSubject(
                subject_id=subject.subject_id,
                type=subject.type,
                first_seen_segment=segment_index,
                last_seen_segment=segment_index,
                appearance_summary=appearance_summary,
                clothing_summary=clothing_summary,
                known_actions=subject.actions[:],
                current_state=subject.state_change or subject.movement,
                confidence=subject.confidence,
            )
        )
        return
    existing.last_seen_segment = segment_index
    if appearance_summary:
        existing.appearance_summary = appearance_summary
    if clothing_summary:
        existing.clothing_summary = clothing_summary
    for action in subject.actions:
        if action and action not in existing.known_actions:
            existing.known_actions.append(action)
    if subject.state_change or subject.movement:
        existing.current_state = subject.state_change or subject.movement
    existing.confidence = max(existing.confidence, subject.confidence)


def _merge_object(*, memory: VideoMemory, obj: ObjectObservation, segment_index: int) -> None:
    existing = next((item for item in memory.persistent_objects if item.object_id == obj.object_id), None)
    if existing is None:
        memory.persistent_objects.append(
            PersistentObject(
                object_id=obj.object_id,
                name=obj.name,
                first_seen_segment=segment_index,
                last_seen_segment=segment_index,
                description=obj.description,
                state_history=[value for value in [obj.state, obj.interaction] if value],
                confidence=obj.confidence,
            )
        )
        return
    existing.last_seen_segment = segment_index
    if obj.name:
        existing.name = obj.name
    if obj.description:
        existing.description = obj.description
    for value in [obj.state, obj.interaction]:
        if value and value not in existing.state_history:
            existing.state_history.append(value)
    existing.confidence = max(existing.confidence, obj.confidence)


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        normalized = value.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered
