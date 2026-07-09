from __future__ import annotations

import logging

from pydantic import ValidationError

from prompts.global_summary_prompt import build_global_summary_prompt
from schemas.segments import TemporalSegmentsArtifact
from schemas.video_memory import VideoMemory
from schemas.vlm import GlobalFactualSummary, VlmSegmentsArtifact
from services.fireworks_client import FireworksClient, FireworksResponseFormatError

logger = logging.getLogger("video-caption-pipeline.worker")


async def generate_global_summary(
    *,
    client: FireworksClient,
    model: str,
    job_id: str,
    video_memory: VideoMemory,
    segment_artifact: VlmSegmentsArtifact,
    temporal_segments: TemporalSegmentsArtifact,
    all_frame_paths: list[str],
) -> GlobalFactualSummary:
    prompt = build_global_summary_prompt(
        job_id=job_id,
        video_memory=video_memory,
        segment_artifact=segment_artifact,
        temporal_segments=temporal_segments,
    )
    try:
        payload = await client.analyze_segment_with_images(
            model=model,
            prompt=prompt,
            image_paths=all_frame_paths,
            temperature=0.1,
        )
        return GlobalFactualSummary.model_validate(payload)
    except FireworksResponseFormatError as exc:
        logger.warning("Global summary JSON parse failed for job %s; attempting repair.", job_id)
        repaired = await client.generate_json(
            model=model,
            prompt=f"Repair this output into valid JSON without adding facts. Return JSON only.\n\n{exc.raw_text}",
            temperature=0.0,
        )
        return GlobalFactualSummary.model_validate(repaired)
    except ValidationError:
        logger.warning("Global summary schema validation failed for job %s; attempting repair.", job_id)
        repaired = await client.generate_json(
            model=model,
            prompt=(
                "Repair this output into valid JSON for the expected summary schema without adding facts. "
                "Flatten list fields like main_subjects and main_objects into arrays of strings. Return JSON only.\n\n"
                f"{payload}"
            ),
            temperature=0.0,
        )
        return GlobalFactualSummary.model_validate(repaired)


def fuse_segment_ground_truth(
    *,
    segment,
    visual_response: dict,
) -> dict:
    segment_description = _as_dict(visual_response.get("segment_description"))
    setting_payload = _as_dict(visual_response.get("setting"))
    audio_observations = _as_dict(visual_response.get("audio_observations"))
    continuity_update = _as_dict(visual_response.get("continuity_update"))
    evidence_quality = _as_dict(visual_response.get("evidence_quality"))

    transcript_facts = [
        chunk.expressive_transcript.strip() or chunk.text.strip()
        for chunk in segment.transcript_chunks
        if (chunk.expressive_transcript.strip() or chunk.text.strip())
    ]
    visual_subjects = [_pick_string(item, "subject_id") for item in visual_response.get("subjects", [])]
    visual_objects = [_pick_string(item, "object_id") for item in visual_response.get("objects", [])]
    key_events = [_pick_string(item, "description") for item in visual_response.get("events", [])]
    grounded_visual_facts = [
        value
        for value in [
            _string_value(segment_description, "short_summary"),
            _string_value(segment_description, "detailed_visual_description"),
            _string_value(segment_description, "movement_and_flow"),
        ]
        if value
    ]
    if _string_value(setting_payload, "visual_environment"):
        grounded_visual_facts.append(_string_value(setting_payload, "visual_environment"))
    if _string_value(audio_observations, "spoken_content_summary"):
        transcript_facts.append(_string_value(audio_observations, "spoken_content_summary"))

    same_people = _string_list(continuity_update.get("same_subjects_as_before"))
    new_people = _string_list(continuity_update.get("new_subjects"))
    same_objects = _string_list(continuity_update.get("same_objects_as_before"))
    new_objects = _string_list(continuity_update.get("new_objects"))
    continuity_notes = _string_list(continuity_update.get("important_changes"))

    setting_parts = [
        _string_value(setting_payload, "location_type"),
        _string_value(setting_payload, "visual_environment"),
    ]
    setting = " - ".join(part for part in setting_parts if part)
    scene_change = _string_value(setting_payload, "changes_from_previous_segment") or (
        continuity_notes[0] if continuity_notes else ""
    )
    uncertainties = _string_list(evidence_quality.get("limitations"))
    uncertainties.extend(_string_list(continuity_update.get("new_uncertainties")))

    return {
        "segment_index": segment.segment_index,
        "start": segment.start,
        "end": segment.end,
        "scene_summary": _string_value(segment_description, "short_summary"),
        "grounded_visual_facts": grounded_visual_facts,
        "grounded_transcript_facts": transcript_facts,
        "subjects": [item for item in visual_subjects if item],
        "objects": [item for item in visual_objects if item],
        "setting": setting,
        "scene_change": scene_change,
        "continuity": {
            "same_people": same_people,
            "new_people": new_people,
            "same_objects": same_objects,
            "new_objects": new_objects,
            "continuity_notes": continuity_notes,
        },
        "key_events": [item for item in key_events if item],
        "uncertainties": [item for item in uncertainties if item],
    }


def _as_dict(value) -> dict:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return {}


def _string_value(payload: dict, key: str) -> str:
    value = payload.get(key, "")
    return value.strip() if isinstance(value, str) else ""


def _string_list(value) -> list[str]:
    if not isinstance(value, list):
        return []
    results: list[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            results.append(item.strip())
    return results


def _pick_string(item, key: str) -> str:
    payload = _as_dict(item)
    value = payload.get(key, "")
    return value.strip() if isinstance(value, str) else ""
