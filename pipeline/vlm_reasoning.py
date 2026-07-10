from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import ValidationError

from prompts.segment_vlm_prompt import (
    build_segment_vlm_json_schema,
    build_segment_vlm_prompt,
    load_segment_vlm_system_prompt,
)
from schemas.segments import TemporalSegment
from schemas.video_memory import VideoMemory
from schemas.vlm import SegmentVlmResponse
from services.client_pool import get_openai_responses_client
from services.google_gemini_client import GoogleGeminiClient, GoogleGeminiResponseFormatError
from worker.config.settings import settings

logger = logging.getLogger("video-caption-pipeline.worker")


async def analyze_segment(
    *,
    client: GoogleGeminiClient,
    model: str,
    job_id: str,
    segment: TemporalSegment,
    memory: VideoMemory,
) -> SegmentVlmResponse:
    prompt = build_segment_vlm_prompt(job_id=job_id, segment=segment, memory=memory)
    system_prompt = load_segment_vlm_system_prompt()
    response_schema = build_segment_vlm_json_schema()
    image_paths = [frame.local_path for frame in segment.frames]
    try:
        payload = await client.analyze_segment_with_images(
            model=model,
            system_prompt=system_prompt,
            prompt=prompt,
            image_paths=image_paths,
            temperature=0.1,
            response_schema=response_schema,
            response_schema_name="segment_vlm_response",
        )
        return SegmentVlmResponse.model_validate(
            _normalize_segment_payload(
                payload,
                expected_segment_index=segment.segment_index,
                expected_start=segment.start,
                expected_end=segment.end,
                has_transcript=bool(segment.transcript_chunks),
                transcript_chunks=segment.transcript_chunks,
            )
        )
    except GoogleGeminiResponseFormatError as exc:
        logger.warning("Segment %s JSON parse failed for job %s; attempting repair.", segment.segment_index, job_id)
        repaired = await repair_json_response(client=client, model=model, raw_text=exc.raw_text, schema_name="segment_vlm_response")
        return SegmentVlmResponse.model_validate(
            _normalize_segment_payload(
                repaired,
                expected_segment_index=segment.segment_index,
                expected_start=segment.start,
                expected_end=segment.end,
                has_transcript=bool(segment.transcript_chunks),
                transcript_chunks=segment.transcript_chunks,
            )
        )
    except ValidationError:
        logger.warning("Segment %s schema validation failed for job %s; attempting repair.", segment.segment_index, job_id)
        repaired = await repair_json_response(
            client=client,
            model=model,
            raw_text=json.dumps(payload),
            schema_name="segment_vlm_response",
        )
        return SegmentVlmResponse.model_validate(
            _normalize_segment_payload(
                repaired,
                expected_segment_index=segment.segment_index,
                expected_start=segment.start,
                expected_end=segment.end,
                has_transcript=bool(segment.transcript_chunks),
                transcript_chunks=segment.transcript_chunks,
            )
        )


async def repair_json_response(*, client: GoogleGeminiClient, model: str, raw_text: str, schema_name: str) -> dict[str, Any]:
    prompt = f"""
Repair the following model output into valid JSON.

Rules:
- Preserve the original meaning.
- Do not add new facts.
- If a field is missing, use empty strings, empty arrays, null, false, or 0.0 as appropriate.
- Return JSON only.
- Target schema name: {schema_name}

Raw model output:
{raw_text}
""".strip()
    try:
        return await client.generate_json(model=model, prompt=prompt, temperature=0.0)
    except GoogleGeminiResponseFormatError:
        logger.warning("Gemini repair returned unusable JSON; falling back to OpenAI repair for %s.", schema_name)
        repair_client = get_openai_responses_client()
        return await repair_client.generate_json(model=settings.openai_final_caption_model, prompt=prompt)


def build_failed_segment_response(
    *,
    segment: TemporalSegment,
    error_message: str,
    raw_text: str | None = None,
) -> SegmentVlmResponse:
    return SegmentVlmResponse(
        status="failed",
        errors=[error_message],
        raw_text=raw_text,
        segment_index=segment.segment_index,
        segment_start=segment.start,
        segment_end=segment.end,
        evidence_quality={
            "visual_clarity": "low",
            "audio_clarity": "none" if not segment.transcript_chunks else "low",
            "limitations": [error_message],
        },
        continuity_update={
            "new_uncertainties": [f"segment_{segment.segment_index}_analysis_failed"],
        },
    )


def serialize_segment_error(exc: Exception) -> str:
    if isinstance(exc, GoogleGeminiResponseFormatError):
        return f"Google Gemini returned invalid JSON: {exc}"
    if isinstance(exc, json.JSONDecodeError):
        return f"JSON decode error: {exc}"
    return str(exc)


def _normalize_segment_payload(
    payload: dict[str, Any],
    *,
    expected_segment_index: int,
    expected_start: float,
    expected_end: float,
    has_transcript: bool,
    transcript_chunks: list[Any] | None = None,
) -> dict[str, Any]:
    payload = _coerce_payload_shape(payload)
    transcript_chunks = transcript_chunks or []
    unreadable_text_detected = _contains_unreadable_visible_text(payload.get("visible_text"))

    raw_segment_index = payload.get("segment_index")
    if isinstance(raw_segment_index, int):
        if raw_segment_index == expected_segment_index + 1:
            payload["segment_index"] = expected_segment_index
        elif raw_segment_index < 0 or raw_segment_index > 4:
            payload["segment_index"] = expected_segment_index
    else:
        payload["segment_index"] = expected_segment_index

    payload["segment_start"] = _normalize_numeric_value(payload.get("segment_start"), fallback=expected_start)
    payload["segment_end"] = _normalize_numeric_value(payload.get("segment_end"), fallback=expected_end)

    evidence_quality = payload.get("evidence_quality")
    if not isinstance(evidence_quality, dict):
        evidence_quality = {}
        payload["evidence_quality"] = evidence_quality
    evidence_quality["visual_clarity"] = _normalize_enum(
        evidence_quality.get("visual_clarity"),
        allowed={"high", "medium", "low"},
        fallback="medium",
    )
    evidence_quality["audio_clarity"] = _normalize_enum(
        evidence_quality.get("audio_clarity"),
        allowed={"high", "medium", "low", "none"},
        fallback=_derive_audio_clarity(transcript_chunks, has_transcript=has_transcript),
    )
    limitations = evidence_quality.get("limitations")
    if not isinstance(limitations, list):
        evidence_quality["limitations"] = _string_list(limitations)

    normalized_subjects: list[dict[str, Any]] = []
    for index, subject in enumerate(payload.get("subjects", []) if isinstance(payload.get("subjects"), list) else []):
        if isinstance(subject, dict):
            normalized_subjects.append(_normalize_subject(subject, index=index))
    payload["subjects"] = normalized_subjects

    normalized_objects: list[dict[str, Any]] = []
    for index, obj in enumerate(payload.get("objects", []) if isinstance(payload.get("objects"), list) else []):
        if isinstance(obj, dict):
            normalized_objects.append(_normalize_object(obj, index=index))
        elif isinstance(obj, str):
            normalized_objects.append(_normalize_object({"name": obj}, index=index))
    payload["objects"] = normalized_objects

    normalized_events: list[dict[str, Any]] = []
    for index, event in enumerate(payload.get("events", []) if isinstance(payload.get("events"), list) else []):
        if isinstance(event, dict):
            normalized_events.append(_normalize_event(event, index=index))
        elif isinstance(event, str):
            normalized_events.append(_normalize_event({"description": event}, index=index))
    payload["events"] = normalized_events

    payload["visible_text"] = _normalize_visible_text(payload.get("visible_text"))
    payload["audio_observations"] = _normalize_audio_observations(
        payload.get("audio_observations"),
        transcript_chunks=transcript_chunks,
    )
    payload["continuity_update"] = _normalize_continuity_update(payload.get("continuity_update"))
    if unreadable_text_detected:
        payload["continuity_update"]["new_uncertainties"] = _dedupe_preserve_order(
            [
                *payload["continuity_update"]["new_uncertainties"],
                "Visible text is present but unreadable.",
            ]
        )

    memory_update = payload.get("memory_update_for_next_segment")
    if isinstance(memory_update, dict):
        timeline_update = memory_update.get("timeline_update")
        if isinstance(timeline_update, list):
            memory_update["timeline_update"] = [_stringify_timeline_item(item) for item in timeline_update]
    payload["memory_update_for_next_segment"] = _normalize_memory_update(memory_update)
    setting_payload = payload.get("setting")
    if isinstance(setting_payload, dict):
        setting_payload["changes_from_previous_segment"] = ""
    return payload


def _coerce_payload_shape(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)

    if "summary" in normalized or "scene_details" in normalized or "main_events" in normalized or "segment_memory" in normalized:
        normalized = _inflate_minimal_segment_payload(normalized)

    top_visual = normalized.pop("visual_clarity", None)
    top_audio = normalized.pop("audio_clarity", None)
    top_limitations = normalized.pop("limitations", None)
    evidence_quality = normalized.get("evidence_quality")
    if not isinstance(evidence_quality, dict):
        evidence_quality = {}
    if top_visual is not None and "visual_clarity" not in evidence_quality:
        evidence_quality["visual_clarity"] = top_visual
    if top_audio is not None and "audio_clarity" not in evidence_quality:
        evidence_quality["audio_clarity"] = top_audio
    if top_limitations is not None and "limitations" not in evidence_quality:
        evidence_quality["limitations"] = top_limitations
    normalized["evidence_quality"] = evidence_quality

    description = normalized.get("segment_description")
    if isinstance(description, str):
        description = {"short_summary": description}
    elif not isinstance(description, dict):
        description = {}
    for source_key, target_key in (
        ("short_summary", "short_summary"),
        ("description", "short_summary"),
        ("detailed_visual_description", "detailed_visual_description"),
        ("movement_description", "movement_and_flow"),
        ("movement_and_flow", "movement_and_flow"),
        ("audio_description", "audio_summary"),
        ("audio_summary", "audio_summary"),
    ):
        value = normalized.pop(source_key, None)
        if value is not None and target_key not in description:
            description[target_key] = value
    normalized["segment_description"] = description

    setting = normalized.get("setting")
    if isinstance(setting, str):
        setting = {"visual_environment": setting}
    elif not isinstance(setting, dict):
        setting = {}
    for source_key, target_key in (
        ("location_type", "location_type"),
        ("visual_environment", "visual_environment"),
        ("background", "background_details"),
        ("background_details", "background_details"),
    ):
        value = normalized.pop(source_key, None)
        if value is not None and target_key not in setting:
            setting[target_key] = value
    setting.pop("changes_from_previous_segment", None)
    normalized["setting"] = setting

    return normalized


def _inflate_minimal_segment_payload(payload: dict[str, Any]) -> dict[str, Any]:
    setting = payload.get("setting")
    if not isinstance(setting, dict):
        setting = {}

    subjects = []
    for item in payload.get("subjects", []) if isinstance(payload.get("subjects"), list) else []:
        if not isinstance(item, dict):
            continue
        subject_id = _first_string(item.get("subject_id"), item.get("id"))
        matched_previous = _nullable_string(item.get("matched_previous_subject_id"))
        subjects.append(
            {
                "subject_id": subject_id or "subject_1",
                "is_new": matched_previous is None,
                "matched_previous_subject_id": matched_previous,
                "type": item.get("type", "unknown"),
                "appearance": {
                    "visible_features": _string_list(item.get("visible_features")),
                    "clothing": _string_list(item.get("clothing")),
                    "colors": _string_list(item.get("colors")),
                    "accessories": [],
                    "pose_or_posture": _first_string(item.get("pose_or_posture"), ""),
                    "confidence": 0.0,
                },
                "actions": _string_list(item.get("actions")),
                "movement": _first_string(item.get("state"), ""),
                "facial_expression": "",
                "emotion_or_tone": {"label": "", "evidence": "", "confidence": 0.0},
                "state_change": _first_string(item.get("state"), ""),
                "confidence": 0.0,
            }
        )

    objects = []
    for item in payload.get("objects", []) if isinstance(payload.get("objects"), list) else []:
        if not isinstance(item, dict):
            continue
        object_id = _first_string(item.get("object_id"), item.get("id"))
        matched_previous = _nullable_string(item.get("matched_previous_object_id"))
        objects.append(
            {
                "object_id": object_id or "object_1",
                "is_new": matched_previous is None,
                "matched_previous_object_id": matched_previous,
                "name": _first_string(item.get("name"), object_id or "object"),
                "description": _first_string(item.get("description"), ""),
                "location_in_scene": _first_string(item.get("location_in_scene"), ""),
                "state": _first_string(item.get("state"), ""),
                "interaction": _first_string(item.get("interaction"), ""),
                "confidence": 0.0,
            }
        )

    main_events = _string_list(payload.get("main_events"))
    focused_event = payload.get("focused_event") if isinstance(payload.get("focused_event"), dict) else {}
    events = []
    for index, description in enumerate(main_events):
        events.append(
            {
                "event_id": f"event_{index + 1}",
                "approx_timestamp": 0.0,
                "description": description,
                "subjects_involved": [],
                "objects_involved": [],
                "evidence": "visual",
                "confidence": 0.0,
            }
        )
    focused_description = _first_string(focused_event.get("description"), "")
    if focused_description:
        events.append(
            {
                "event_id": f"event_{len(events) + 1}",
                "approx_timestamp": 0.0,
                "description": focused_description,
                "subjects_involved": _string_list(focused_event.get("subjects_involved")),
                "objects_involved": _string_list(focused_event.get("objects_involved")),
                "evidence": "visual",
                "confidence": 0.0,
            }
        )

    return {
        "segment_index": payload.get("segment_index"),
        "segment_start": payload.get("segment_start"),
        "segment_end": payload.get("segment_end"),
        "evidence_quality": {
            "visual_clarity": payload.get("visual_clarity"),
            "audio_clarity": None,
            "limitations": payload.get("uncertainties", []),
        },
        "segment_description": {
            "short_summary": _first_string(payload.get("summary"), ""),
            "detailed_visual_description": _first_string(payload.get("scene_details"), ""),
            "movement_and_flow": focused_description or (main_events[0] if main_events else ""),
            "audio_summary": "",
        },
        "setting": {
            "location_type": _first_string(setting.get("location_type"), ""),
            "visual_environment": _first_string(setting.get("visual_environment"), ""),
            "background_details": setting.get("background_details", []),
            "changes_from_previous_segment": "",
        },
        "subjects": subjects,
        "objects": objects,
        "events": events,
        "visible_text": [
            {
                "text": _first_string(item.get("text"), "") if isinstance(item, dict) else "",
                "location": _first_string(item.get("location"), "") if isinstance(item, dict) else "",
                "confidence": 0.0,
            }
            for item in payload.get("visible_text", [])
            if isinstance(item, dict)
        ],
        "audio_observations": None,
        "continuity_update": {
            "same_subjects_as_before": [item["subject_id"] for item in subjects if not item["is_new"]],
            "same_objects_as_before": [item["object_id"] for item in objects if not item["is_new"]],
            "new_subjects": [item["subject_id"] for item in subjects if item["is_new"]],
            "new_objects": [item["object_id"] for item in objects if item["is_new"]],
            "resolved_uncertainties": [],
            "new_uncertainties": payload.get("uncertainties", []),
            "important_changes": payload.get("continuity_notes", []),
        },
        "memory_update_for_next_segment": {
            "segment_memory": _first_string(payload.get("segment_memory"), ""),
            "persistent_subjects": [item["subject_id"] for item in subjects],
            "persistent_objects": [item["object_id"] for item in objects],
            "persistent_setting": _first_string(setting.get("location_type"), setting.get("visual_environment"), ""),
            "open_actions_or_context": payload.get("uncertainties", []),
            "timeline_update": _dedupe_preserve_order(main_events),
        },
    }


def _normalize_subject(subject: dict[str, Any], *, index: int) -> dict[str, Any]:
    appearance = subject.get("appearance")
    if isinstance(appearance, str):
        appearance = {"visible_features": [appearance]}
    elif not isinstance(appearance, dict):
        appearance = {}

    description = _first_string(subject.get("description"), subject.get("appearance"))
    if description and not appearance.get("visible_features"):
        appearance["visible_features"] = [description]

    emotion = subject.get("emotion_or_tone")
    if isinstance(emotion, str):
        emotion = {"label": emotion}
    elif not isinstance(emotion, dict):
        emotion = {}
    emotion["confidence"] = _normalize_confidence(emotion.get("confidence"))

    return {
        "subject_id": _first_string(subject.get("subject_id"), subject.get("id"), f"subject_{index + 1}"),
        "is_new": bool(subject.get("is_new", True)),
        "matched_previous_subject_id": _nullable_string(subject.get("matched_previous_subject_id")),
        "type": _normalize_enum(
            subject.get("type"),
            allowed={"person", "animal", "object", "unknown"},
            fallback="unknown",
        ),
        "appearance": {
            "visible_features": _string_list(appearance.get("visible_features")),
            "clothing": _string_list(appearance.get("clothing")),
            "colors": _string_list(appearance.get("colors")),
            "accessories": _string_list(appearance.get("accessories")),
            "pose_or_posture": _first_string(appearance.get("pose_or_posture"), subject.get("pose"), ""),
            "confidence": _normalize_confidence(appearance.get("confidence")),
        },
        "actions": _string_list(subject.get("actions") if subject.get("actions") is not None else subject.get("action")),
        "movement": _first_string(subject.get("movement"), ""),
        "facial_expression": _first_string(subject.get("facial_expression"), ""),
        "emotion_or_tone": {
            "label": _first_string(emotion.get("label"), ""),
            "evidence": _first_string(emotion.get("evidence"), ""),
            "confidence": _normalize_confidence(emotion.get("confidence")),
        },
        "state_change": _first_string(subject.get("state_change"), ""),
        "confidence": _normalize_confidence(subject.get("confidence")),
    }


def _normalize_object(obj: dict[str, Any], *, index: int) -> dict[str, Any]:
    object_id = _first_string(obj.get("object_id"), obj.get("id"))
    name = _first_string(obj.get("name"), obj.get("object_name"), object_id or f"object_{index + 1}")
    return {
        "object_id": object_id or f"object_{index + 1}",
        "is_new": bool(obj.get("is_new", True)),
        "matched_previous_object_id": _nullable_string(obj.get("matched_previous_object_id")),
        "name": name,
        "description": _first_string(obj.get("description"), ""),
        "location_in_scene": _first_string(obj.get("location_in_scene"), obj.get("location"), ""),
        "state": _first_string(obj.get("state"), ""),
        "interaction": _first_string(obj.get("interaction"), obj.get("interaction_with_subject"), ""),
        "confidence": _normalize_confidence(obj.get("confidence")),
    }


def _normalize_event(event: dict[str, Any], *, index: int) -> dict[str, Any]:
    return {
        "event_id": _first_string(event.get("event_id"), event.get("id"), f"event_{index + 1}"),
        "approx_timestamp": _normalize_numeric_value(
            event.get("approx_timestamp", event.get("timestamp", event.get("start_time"))),
            fallback=0.0,
        ),
        "description": _first_string(event.get("description"), event.get("event_description"), event.get("name"), ""),
        "subjects_involved": _string_list(event.get("subjects_involved", event.get("involved_subjects"))),
        "objects_involved": _string_list(event.get("objects_involved", event.get("involved_objects"))),
        "evidence": _normalize_enum(
            event.get("evidence"),
            allowed={"visual", "audio", "both"},
            fallback="visual",
        ),
        "confidence": _normalize_confidence(event.get("confidence")),
    }


def _normalize_visible_text(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    normalized: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, str):
            if _looks_illegible_text(item):
                continue
            normalized.append({"text": item, "location": "", "confidence": 0.0})
        elif isinstance(item, dict):
            text = _first_string(item.get("text"), item.get("text_content"), "")
            if _looks_illegible_text(text):
                continue
            normalized.append(
                {
                    "text": text,
                    "location": _first_string(item.get("location"), item.get("position"), item.get("text_location"), ""),
                    "confidence": _normalize_confidence(item.get("confidence")),
                }
            )
    return normalized


def _normalize_audio_observations(value: Any, *, transcript_chunks: list[Any]) -> dict[str, Any]:
    if isinstance(value, dict):
        music = value.get("music") if isinstance(value.get("music"), dict) else {}
        tone = value.get("tone") if isinstance(value.get("tone"), dict) else {}
        return {
            "spoken_content_summary": _first_string(value.get("spoken_content_summary"), value.get("audio_transcript"), ""),
            "music": {
                "present": bool(music.get("present", False)),
                "type": _first_string(music.get("type"), ""),
                "instrumentation": _first_string(music.get("instrumentation"), ""),
                "energy": _first_string(music.get("energy"), ""),
                "description": _first_string(music.get("description"), ""),
            },
            "tone": {
                "style": _first_string(tone.get("style"), ""),
                "emotion": _first_string(tone.get("emotion"), ""),
                "delivery": _first_string(tone.get("delivery"), ""),
                "evidence": _first_string(tone.get("evidence"), ""),
                "confidence": _normalize_confidence(tone.get("confidence")),
            },
        }
    summary = "; ".join(_string_list(value))
    if not summary:
        summary = _derive_transcript_summary(transcript_chunks)
    return {
        "spoken_content_summary": summary,
        "music": {"present": False, "type": "", "instrumentation": "", "energy": "", "description": ""},
        "tone": {"style": "", "emotion": "", "delivery": "", "evidence": "", "confidence": 0.0},
    }


def _normalize_continuity_update(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        return {
            "same_subjects_as_before": [],
            "same_objects_as_before": [],
            "new_subjects": [],
            "new_objects": [],
            "resolved_uncertainties": [],
            "new_uncertainties": [],
            "important_changes": [value] if value else [],
        }
    if not isinstance(value, dict):
        value = {}
    important_changes = _string_list(value.get("important_changes"))
    important_changes.extend(_string_list(value.get("new_setting")))
    important_changes.extend(_string_list(value.get("previous_segment_status")))
    important_changes.extend(_string_list(value.get("notes")))
    return {
        "same_subjects_as_before": _dedupe_preserve_order(
            _string_list(value.get("same_subjects_as_before", value.get("tracked_subjects")))
        ),
        "same_objects_as_before": _dedupe_preserve_order(_string_list(value.get("same_objects_as_before"))),
        "new_subjects": _dedupe_preserve_order(_string_list(value.get("new_subjects"))),
        "new_objects": _dedupe_preserve_order(_string_list(value.get("new_objects"))),
        "resolved_uncertainties": _dedupe_preserve_order(_string_list(value.get("resolved_uncertainties"))),
        "new_uncertainties": _dedupe_preserve_order(_string_list(value.get("new_uncertainties"))),
        "important_changes": _dedupe_preserve_order(important_changes),
    }


def _normalize_memory_update(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        return {
            "segment_memory": value,
            "persistent_subjects": [],
            "persistent_objects": [],
            "persistent_setting": "",
            "open_actions_or_context": [],
            "timeline_update": [],
        }
    if not isinstance(value, dict):
        value = {}
    timeline_update = value.get("timeline_update")
    if isinstance(timeline_update, list):
        timeline_update = [_stringify_timeline_item(item) for item in timeline_update]
    else:
        timeline_update = _string_list(timeline_update)
    return {
        "segment_memory": _first_string(value.get("segment_memory"), ""),
        "persistent_subjects": _dedupe_preserve_order(_string_list(value.get("persistent_subjects"))),
        "persistent_objects": _dedupe_preserve_order(_string_list(value.get("persistent_objects"))),
        "persistent_setting": _first_string(value.get("persistent_setting"), value.get("setting"), ""),
        "open_actions_or_context": _dedupe_preserve_order(_string_list(value.get("open_actions_or_context"))),
        "timeline_update": _dedupe_preserve_order(timeline_update),
    }


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    if isinstance(value, list):
        items: list[str] = []
        for item in value:
            if isinstance(item, str):
                stripped = item.strip()
                if stripped:
                    items.append(stripped)
            elif isinstance(item, dict):
                for key in ("text", "description", "summary", "label", "name", "subject_id", "object_id"):
                    candidate = item.get(key)
                    if isinstance(candidate, str) and candidate.strip():
                        items.append(candidate.strip())
                        break
        return items
    if isinstance(value, dict):
        for key in ("text", "description", "summary", "label", "name"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return [candidate.strip()]
    return []


def _first_string(*values: Any) -> str:
    for value in values:
        if isinstance(value, str):
            stripped = value.strip()
            if stripped:
                return stripped
    return ""


def _nullable_string(value: Any) -> str | None:
    result = _first_string(value)
    return result or None


def _normalize_confidence(value: Any) -> float:
    normalized = _normalize_numeric_value(value, fallback=0.0)
    return max(0.0, min(1.0, normalized))


def _derive_transcript_summary(transcript_chunks: list[Any]) -> str:
    lines: list[str] = []
    for chunk in transcript_chunks:
        expressive = getattr(chunk, "expressive_transcript", "") or ""
        text = getattr(chunk, "text", "") or ""
        line = expressive.strip() or text.strip()
        if line:
            lines.append(line)
    return " ".join(lines[:2])


def _derive_audio_clarity(transcript_chunks: list[Any], *, has_transcript: bool) -> str:
    if not has_transcript:
        return "none"
    return "low" if _derive_transcript_summary(transcript_chunks) else "none"


def _normalize_numeric_value(value: Any, *, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _normalize_enum(value: Any, *, allowed: set[str], fallback: str) -> str:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in allowed:
            return normalized
        if "|" in normalized:
            first = normalized.split("|", 1)[0].strip()
            if first in allowed:
                return first
    return fallback


def _stringify_timeline_item(item: Any) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        summary = item.get("summary")
        if isinstance(summary, str) and summary.strip():
            return summary
        description = item.get("description")
        if isinstance(description, str) and description.strip():
            return description
        parts = []
        for key in ("segment_index", "start", "end"):
            if key in item:
                parts.append(f"{key}={item[key]}")
        return ", ".join(parts) if parts else json.dumps(item)
    return str(item)


def _contains_unreadable_visible_text(value: Any) -> bool:
    if not isinstance(value, list):
        return False
    for item in value:
        if isinstance(item, str) and _looks_illegible_text(item):
            return True
        if isinstance(item, dict):
            if _looks_illegible_text(_first_string(item.get("text"), item.get("text_content"), "")):
                return True
    return False


def _looks_illegible_text(value: str) -> bool:
    normalized = value.strip().lower()
    if not normalized:
        return False
    unreadable_tokens = {
        "illegible",
        "illegible text",
        "unreadable",
        "unreadable text",
        "blurred text",
        "text is unreadable",
    }
    return normalized in unreadable_tokens or "illegible" in normalized or "unreadable" in normalized


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
