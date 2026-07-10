from __future__ import annotations

import logging
from typing import Any

from prompts.global_summary_prompt import (
    build_global_summary_prompt,
    build_progressive_chunk_prompt,
)
from prompts.segment_vlm_prompt import load_segment_vlm_system_prompt
from schemas.segments import TemporalSegmentsArtifact
from schemas.video_memory import VideoMemory
from schemas.vlm import GlobalFactualSummary, VlmSegmentsArtifact
from services.google_gemini_client import GoogleGeminiClient, GoogleGeminiResponseFormatError

logger = logging.getLogger("video-caption-pipeline.worker")


def build_segment_based_global_summary(
    *,
    job_id: str,
    temporal_segments: TemporalSegmentsArtifact,
    segment_artifact: VlmSegmentsArtifact,
    video_memory: VideoMemory,
    transcript_text: str,
) -> GlobalFactualSummary:
    fused_segments = [segment.segment_ground_truth for segment in temporal_segments.segments if segment.segment_ground_truth]
    segment_summaries = [item.get("scene_summary", "").strip() for item in fused_segments if item.get("scene_summary")]
    visual_facts = _unique_strings(
        fact
        for item in fused_segments
        for fact in item.get("grounded_visual_facts", [])
    )
    transcript_facts = _unique_strings(
        fact
        for item in fused_segments
        for fact in item.get("grounded_transcript_facts", [])
    )
    main_subjects = _unique_strings(
        subject
        for item in fused_segments
        for subject in item.get("subjects", [])
    )
    main_objects = _unique_strings(
        obj
        for item in fused_segments
        for obj in item.get("objects", [])
    )
    setting_candidates = _unique_strings(
        item.get("setting", "")
        for item in fused_segments
    )
    timeline_payload = []
    scene_changes = []
    continuity_overview = []
    tracking_entries = []
    uncertainties = _unique_strings(
        note
        for item in fused_segments
        for note in item.get("uncertainties", [])
    )

    for item in fused_segments:
        continuity = item.get("continuity", {})
        timeline_payload.append(
            {
                "segment_index": item.get("segment_index"),
                "summary": item.get("scene_summary", ""),
                "key_events": _unique_strings(item.get("key_events", [])),
                "visual_facts": _unique_strings(item.get("grounded_visual_facts", [])),
                "transcript_facts": _unique_strings(item.get("grounded_transcript_facts", [])),
                "continuity_notes": _unique_strings(continuity.get("continuity_notes", [])),
            }
        )
        if item.get("scene_change"):
            scene_changes.append(
                {
                    "segment_index": item.get("segment_index"),
                    "apparent_change": item.get("scene_change", ""),
                    "related_subjects": item.get("subjects", []),
                    "related_objects": item.get("objects", []),
                }
            )
        continuity_overview.append(
            {
                "segment_index": item.get("segment_index"),
                "continuity_note": item.get("scene_summary", ""),
                "continued_subjects": continuity.get("same_people", []),
                "new_subjects": continuity.get("new_people", []),
                "continued_objects": continuity.get("same_objects", []),
                "new_objects": continuity.get("new_objects", []),
            }
        )

    for subject in video_memory.persistent_subjects:
        tracking_entries.append(
            {
                "object_id": subject.subject_id,
                "name": subject.subject_id,
                "tracking_summary": subject.current_state or subject.appearance_summary or subject.subject_id,
                "entity_type": subject.type,
                "appears_across_segments": list(range(subject.first_seen_segment, subject.last_seen_segment + 1)),
            }
        )
    for obj in video_memory.persistent_objects:
        tracking_entries.append(
            {
                "object_id": obj.object_id,
                "name": obj.name or obj.object_id,
                "tracking_summary": obj.description or (obj.state_history[0] if obj.state_history else obj.object_id),
                "entity_type": "object",
                "appears_across_segments": list(range(obj.first_seen_segment, obj.last_seen_segment + 1)),
            }
        )

    transcript_alignment = []
    for segment in temporal_segments.segments:
        transcript_lines = [
            chunk.expressive_transcript.strip() or chunk.text.strip()
            for chunk in segment.transcript_chunks
            if (chunk.expressive_transcript.strip() or chunk.text.strip())
        ]
        if not transcript_lines:
            continue
        transcript_alignment.append(
            {
                "segment_index": segment.segment_index,
                "transcript_summary": " ".join(transcript_lines[:2]),
                "visual_alignment": segment.segment_ground_truth.get("scene_summary", ""),
                "speaker_changes": [],
                "tone_notes": [],
                "mismatches_or_uncertainties": segment.segment_ground_truth.get("uncertainties", []),
            }
        )

    factual_summary = " ".join(segment_summaries[:3]).strip()
    if not factual_summary:
        fallback_bits = []
        if visual_facts:
            fallback_bits.append(visual_facts[0])
        if transcript_facts:
            fallback_bits.append(transcript_facts[0])
        factual_summary = " ".join(fallback_bits).strip()

    detailed_parts = []
    if visual_facts:
        detailed_parts.append("Visual evidence: " + "; ".join(visual_facts[:8]) + ".")
    if transcript_facts:
        detailed_parts.append("Transcript evidence: " + "; ".join(transcript_facts[:6]) + ".")
    detailed_ground_truth = " ".join(detailed_parts).strip() or factual_summary

    if not setting_candidates and video_memory.global_setting.current_location_type:
        setting_candidates = [video_memory.global_setting.current_location_type]

    audio_summary = ""
    if transcript_facts:
        audio_summary = " ".join(transcript_facts[:2])
    elif transcript_text.strip():
        audio_summary = "Transcript available but sparse."

    payload = {
        "factual_summary": factual_summary,
        "detailed_ground_truth": detailed_ground_truth,
        "detailed_timeline": timeline_payload,
        "main_subjects": main_subjects,
        "main_objects": main_objects,
        "setting_summary": setting_candidates[0] if setting_candidates else "",
        "audio_summary": audio_summary,
        "transcript_visual_alignment": transcript_alignment,
        "speaker_analysis": [],
        "scene_change_overview": scene_changes,
        "continuity_overview": continuity_overview,
        "object_and_subject_tracking": tracking_entries,
        "uncertainties": _unique_strings([*uncertainties, *video_memory.unresolved_uncertainties]),
        "confidence": _derive_summary_confidence(segment_artifact=segment_artifact, factual_summary=factual_summary),
        "errors": [] if factual_summary else [f"segment_summary_empty:{job_id}"],
    }
    return GlobalFactualSummary.model_validate(payload)


async def generate_global_summary(
    *,
    client: GoogleGeminiClient,
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
            system_prompt=load_segment_vlm_system_prompt(),
            prompt=prompt,
            image_paths=all_frame_paths,
            temperature=0.1,
        )
        return GlobalFactualSummary.model_validate(payload)
    except GoogleGeminiResponseFormatError as exc:
        logger.warning("Global summary JSON parse failed for job %s; attempting repair.", job_id)
        repaired = await client.generate_json(
            model=model,
            prompt=f"Repair this output into valid JSON without adding facts. Return JSON only.\n\n{exc.raw_text}",
            temperature=0.0,
        )
        return GlobalFactualSummary.model_validate(repaired)


async def generate_verified_summary(
    *,
    client: GoogleGeminiClient,
    model: str,
    job_id: str,
    image_paths: list[str],
    transcript_text: str = "",
) -> GlobalFactualSummary:
    frame_chunks = _chunk_image_paths(image_paths, chunk_size=4)
    chunk_summaries: list[dict[str, Any]] = []
    for chunk_index, frame_chunk in enumerate(frame_chunks):
        chunk_payload = await client.analyze_segment_with_images(
            model=model,
            system_prompt=load_segment_vlm_system_prompt(),
            prompt=build_progressive_chunk_prompt(
                job_id=job_id,
                transcript_text=transcript_text,
                chunk_index=chunk_index,
                chunk_count=len(frame_chunks),
            ),
            image_paths=frame_chunk,
            temperature=0.1,
        )
        chunk_summaries.append(_normalize_chunk_summary(chunk_index, chunk_payload))
    merged_summary = _build_chunk_fallback_summary(
        job_id=job_id,
        chunk_summaries=chunk_summaries,
        transcript_text=transcript_text,
    )
    if not _summary_has_content(merged_summary):
        logger.warning("Deterministic chunk merge was empty for job %s.", job_id)
    return merged_summary


def _chunk_image_paths(image_paths: list[str], *, chunk_size: int) -> list[list[str]]:
    if chunk_size <= 0:
        return [image_paths]
    return [
        image_paths[index : index + chunk_size]
        for index in range(0, len(image_paths), chunk_size)
        if image_paths[index : index + chunk_size]
    ]


def _normalize_chunk_summary(chunk_index: int, payload: dict[str, Any]) -> dict[str, Any]:
    def normalize_string_list(value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]

    return {
        "chunk_index": chunk_index,
        "chunk_summary": str(payload.get("chunk_summary", "")).strip(),
        "key_details": normalize_string_list(payload.get("key_details")),
        "subjects": normalize_string_list(payload.get("subjects")),
        "objects": normalize_string_list(payload.get("objects")),
        "actions": normalize_string_list(payload.get("actions")),
        "setting": str(payload.get("setting", "")).strip(),
        "visible_text": normalize_string_list(payload.get("visible_text")),
        "uncertainties": normalize_string_list(payload.get("uncertainties")),
    }


def _summary_has_content(summary: GlobalFactualSummary) -> bool:
    content_fields = [
        summary.factual_summary,
        summary.detailed_ground_truth,
        summary.setting_summary,
        summary.audio_summary,
    ]
    if any(value.strip() for value in content_fields if isinstance(value, str)):
        return True
    if summary.main_subjects or summary.main_objects:
        return True
    if summary.scene_change_overview or summary.continuity_overview or summary.object_and_subject_tracking:
        return True
    return False


def _build_chunk_fallback_summary(
    *,
    job_id: str,
    chunk_summaries: list[dict[str, Any]],
    transcript_text: str,
) -> GlobalFactualSummary:
    chunk_summary_texts = [item["chunk_summary"] for item in chunk_summaries if item.get("chunk_summary")]
    key_details = _unique_strings(
        detail
        for item in chunk_summaries
        for detail in item.get("key_details", [])
    )
    subjects = _unique_strings(
        subject
        for item in chunk_summaries
        for subject in item.get("subjects", [])
    )
    objects = _unique_strings(
        obj
        for item in chunk_summaries
        for obj in item.get("objects", [])
    )
    settings = _unique_strings(
        item.get("setting", "")
        for item in chunk_summaries
    )
    actions = _unique_strings(
        action
        for item in chunk_summaries
        for action in item.get("actions", [])
    )
    visible_text = _unique_strings(
        text
        for item in chunk_summaries
        for text in item.get("visible_text", [])
    )
    uncertainties = _unique_strings(
        uncertainty
        for item in chunk_summaries
        for uncertainty in item.get("uncertainties", [])
    )

    factual_summary = " ".join(chunk_summary_texts[:3]).strip()
    if not factual_summary:
        fallback_bits = []
        if subjects:
            fallback_bits.append(f"Visible subjects include {', '.join(subjects[:3])}.")
        if actions:
            fallback_bits.append(f"Observed actions include {', '.join(actions[:3])}.")
        if settings:
            fallback_bits.append(f"The setting appears to be {settings[0]}.")
        factual_summary = " ".join(fallback_bits).strip()

    detailed_bits = chunk_summary_texts[:]
    if key_details:
        detailed_bits.append("Key visible details: " + "; ".join(key_details[:8]) + ".")
    if visible_text:
        detailed_bits.append("Visible text: " + "; ".join(visible_text[:6]) + ".")
    detailed_ground_truth = " ".join(bit for bit in detailed_bits if bit).strip()

    payload = {
        "factual_summary": factual_summary,
        "detailed_ground_truth": detailed_ground_truth or factual_summary,
        "main_subjects": subjects,
        "main_objects": objects,
        "setting_summary": settings[0] if settings else "",
        "audio_summary": "Transcript available." if transcript_text.strip() else "",
        "scene_change_overview": [
            {
                "segment_index": item["chunk_index"],
                "apparent_change": item["chunk_summary"],
                "related_subjects": item.get("subjects", []),
                "related_objects": item.get("objects", []),
            }
            for item in chunk_summaries
            if item.get("chunk_summary")
        ],
        "continuity_overview": [
            {
                "segment_index": item["chunk_index"],
                "continuity_note": item["chunk_summary"],
                "continued_subjects": item.get("subjects", []),
                "new_subjects": [],
                "continued_objects": item.get("objects", []),
                "new_objects": [],
            }
            for item in chunk_summaries
            if item.get("chunk_summary")
        ],
        "object_and_subject_tracking": [
            {
                "object_id": f"subject_{index + 1}",
                "name": subject,
                "tracking_summary": subject,
                "entity_type": "unknown",
                "appears_across_segments": [
                    item["chunk_index"]
                    for item in chunk_summaries
                    if subject in item.get("subjects", [])
                ],
            }
            for index, subject in enumerate(subjects[:6])
        ]
        + [
            {
                "object_id": f"object_{index + 1}",
                "name": obj,
                "tracking_summary": obj,
                "entity_type": "object",
                "appears_across_segments": [
                    item["chunk_index"]
                    for item in chunk_summaries
                    if obj in item.get("objects", [])
                ],
            }
            for index, obj in enumerate(objects[:6])
        ],
        "uncertainties": uncertainties,
        "confidence": 0.45 if factual_summary else 0.1,
        "errors": [] if factual_summary else [f"chunk_summary_empty:{job_id}"],
    }
    return GlobalFactualSummary.model_validate(payload)


def _unique_strings(values: Any) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for raw in values:
        text = str(raw).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return ordered


def _derive_summary_confidence(*, segment_artifact: VlmSegmentsArtifact, factual_summary: str) -> float:
    if not factual_summary:
        return 0.1
    if not segment_artifact.segments:
        return 0.2
    successful = sum(1 for item in segment_artifact.segments if item.vlm_response.status != "failed")
    ratio = successful / len(segment_artifact.segments)
    return round(max(0.25, min(0.85, 0.35 + (ratio * 0.5))), 2)


def _looks_illegible_text(value: str) -> bool:
    normalized = value.strip().lower()
    if not normalized:
        return False
    return normalized in {
        "illegible",
        "illegible text",
        "unreadable",
        "unreadable text",
        "blurred text",
        "text is unreadable",
    } or "illegible" in normalized or "unreadable" in normalized


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
    visual_subjects = [_subject_label(item) for item in visual_response.get("subjects", [])]
    visual_objects = [_object_label(item) for item in visual_response.get("objects", [])]
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
    scene_change = continuity_notes[0] if continuity_notes else ""
    uncertainties = _string_list(evidence_quality.get("limitations"))
    uncertainties.extend(_string_list(continuity_update.get("new_uncertainties")))
    raw_visible_text = [_pick_string(item, "text") for item in visual_response.get("visible_text", [])]
    if any(_looks_illegible_text(item) for item in raw_visible_text if item):
        uncertainties.append("Visible text is present but unreadable.")

    return {
        "segment_index": segment.segment_index,
        "start": segment.start,
        "end": segment.end,
        "scene_summary": _string_value(segment_description, "short_summary"),
        "grounded_visual_facts": _unique_strings(grounded_visual_facts),
        "grounded_transcript_facts": _unique_strings(transcript_facts),
        "subjects": _unique_strings(item for item in visual_subjects if item),
        "objects": _unique_strings(item for item in visual_objects if item),
        "setting": setting,
        "scene_change": scene_change,
        "continuity": {
            "same_people": _unique_strings(same_people),
            "new_people": _unique_strings(new_people),
            "same_objects": _unique_strings(same_objects),
            "new_objects": _unique_strings(new_objects),
            "continuity_notes": _unique_strings(continuity_notes),
        },
        "key_events": _unique_strings(item for item in key_events if item),
        "uncertainties": _unique_strings(item for item in uncertainties if item),
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


def _subject_label(item) -> str:
    payload = _as_dict(item)
    appearance = _as_dict(payload.get("appearance"))
    type_label = _string_value(payload, "type") or "subject"
    features = _string_list(appearance.get("visible_features"))
    clothing = _string_list(appearance.get("clothing"))
    parts = features[:2] + clothing[:1]
    if parts:
        return f"{type_label}: " + ", ".join(parts)
    return _pick_string(payload, "subject_id") or type_label


def _object_label(item) -> str:
    payload = _as_dict(item)
    name = _string_value(payload, "name")
    description = _string_value(payload, "description")
    if name and description:
        return f"{name}: {description}"
    return name or description or _pick_string(payload, "object_id")
