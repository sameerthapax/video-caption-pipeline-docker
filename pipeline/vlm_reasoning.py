from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import ValidationError

from prompts.segment_vlm_prompt import build_segment_vlm_prompt
from schemas.segments import TemporalSegment
from schemas.video_memory import VideoMemory
from schemas.vlm import SegmentVlmResponse
from services.fireworks_client import FireworksClient, FireworksResponseFormatError

logger = logging.getLogger("video-caption-pipeline.worker")


async def analyze_segment(
    *,
    client: FireworksClient,
    model: str,
    job_id: str,
    segment: TemporalSegment,
    memory: VideoMemory,
    include_transcript: bool = False,
) -> SegmentVlmResponse:
    prompt = build_segment_vlm_prompt(job_id=job_id, segment=segment, memory=memory, include_transcript=include_transcript)
    image_paths = [frame.local_path for frame in segment.frames]
    try:
        payload = await client.analyze_segment_with_images(
            model=model,
            prompt=prompt,
            image_paths=image_paths,
            temperature=0.1,
        )
        return SegmentVlmResponse.model_validate(
            _normalize_segment_payload(
                payload,
                expected_segment_index=segment.segment_index,
                expected_start=segment.start,
                expected_end=segment.end,
                has_transcript=bool(segment.transcript_chunks),
            )
        )
    except FireworksResponseFormatError as exc:
        logger.warning("Segment %s JSON parse failed for job %s; attempting repair.", segment.segment_index, job_id)
        repaired = await repair_json_response(client=client, model=model, raw_text=exc.raw_text, schema_name="segment_vlm_response")
        return SegmentVlmResponse.model_validate(
            _normalize_segment_payload(
                repaired,
                expected_segment_index=segment.segment_index,
                expected_start=segment.start,
                expected_end=segment.end,
                has_transcript=bool(segment.transcript_chunks),
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
            )
        )


async def repair_json_response(*, client: FireworksClient, model: str, raw_text: str, schema_name: str) -> dict[str, Any]:
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
    return await client.generate_json(model=model, prompt=prompt, temperature=0.0)


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
    if isinstance(exc, FireworksResponseFormatError):
        return f"Fireworks returned invalid JSON: {exc}"
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
) -> dict[str, Any]:
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
        fallback="low" if has_transcript else "none",
    )

    for subject in payload.get("subjects", []) if isinstance(payload.get("subjects"), list) else []:
        if isinstance(subject, dict):
            subject["type"] = _normalize_enum(
                subject.get("type"),
                allowed={"person", "animal", "object", "unknown"},
                fallback="unknown",
            )

    for event in payload.get("events", []) if isinstance(payload.get("events"), list) else []:
        if isinstance(event, dict):
            event["evidence"] = _normalize_enum(
                event.get("evidence"),
                allowed={"visual", "audio", "both"},
                fallback="visual",
            )

    memory_update = payload.get("memory_update_for_next_segment")
    if isinstance(memory_update, dict):
        timeline_update = memory_update.get("timeline_update")
        if isinstance(timeline_update, list):
            memory_update["timeline_update"] = [_stringify_timeline_item(item) for item in timeline_update]
    return payload


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
