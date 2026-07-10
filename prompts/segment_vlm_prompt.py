from __future__ import annotations

import json
from pathlib import Path

from schemas.segments import TemporalSegment
from schemas.video_memory import VideoMemory


PROMPT_DIR = Path(__file__).resolve().parent


def load_segment_vlm_system_prompt() -> str:
    return (PROMPT_DIR / "perception_system.txt").read_text(encoding="utf-8").strip()


def build_segment_vlm_prompt(
    *,
    job_id: str,
    segment: TemporalSegment,
    memory: VideoMemory,
) -> str:
    frames_payload = [
        {
            "frame_id": frame.frame_id,
            "timestamp": frame.timestamp,
            "selection_reasons": frame.selection_reasons,
        }
        for frame in segment.frames
    ]
    continuity_context = {
        "last_segment_summary": _last_segment_summary(memory),
        "tracked_subjects": [
            {
                "subject_id": item.subject_id,
                "type": item.type,
                "appearance_summary": item.appearance_summary,
                "current_state": item.current_state,
            }
            for item in memory.persistent_subjects
        ],
        "tracked_objects": [
            {
                "object_id": item.object_id,
                "name": item.name,
                "description": item.description,
            }
            for item in memory.persistent_objects
        ],
        "open_uncertainties": memory.unresolved_uncertainties,
    }

    return f"""
Video id: {job_id}
Segment: {segment.segment_index + 1} of {max(1, len(memory.segment_memories))}
Time range: {segment.start}s to {segment.end}s

Analyze these {len(segment.frames)} sampled frames in chronological order.
Capture exact visible facts that help distinguish the clip:
- setting
- main subjects
- important visible objects and colors
- actions and changes over time
- visible text when clearly readable
- camera movement or viewpoint when obvious

Use prior continuity context only when the same visible subject or object clearly continues.
If something is unclear, put it in `uncertainties`.
If text is present but unreadable, do not put placeholder text in `visible_text`; mention that in `uncertainties`.
Return only valid JSON matching the required schema.

Sampled frames:
{json.dumps(frames_payload, indent=2)}

Prior continuity context:
{json.dumps(continuity_context, indent=2)}
""".strip()


def build_segment_vlm_json_schema() -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "segment_index": {"type": "integer"},
            "segment_start": {"type": "number"},
            "segment_end": {"type": "number"},
            "visual_clarity": {"type": "string", "enum": ["high", "medium", "low"]},
            "summary": {"type": "string"},
            "scene_details": {"type": "string"},
            "setting": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "location_type": {"type": "string"},
                    "visual_environment": {"type": "string"},
                    "background_details": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["location_type", "visual_environment", "background_details"],
            },
            "subjects": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "subject_id": {"type": "string"},
                        "matched_previous_subject_id": {"type": ["string", "null"]},
                        "type": {"type": "string", "enum": ["person", "animal", "object", "unknown"]},
                        "label": {"type": "string"},
                        "visible_features": {"type": "array", "items": {"type": "string"}},
                        "clothing": {"type": "array", "items": {"type": "string"}},
                        "colors": {"type": "array", "items": {"type": "string"}},
                        "pose_or_posture": {"type": "string"},
                        "actions": {"type": "array", "items": {"type": "string"}},
                        "state": {"type": "string"},
                    },
                    "required": [
                        "subject_id",
                        "matched_previous_subject_id",
                        "type",
                        "label",
                        "visible_features",
                        "clothing",
                        "colors",
                        "pose_or_posture",
                        "actions",
                        "state",
                    ],
                },
            },
            "objects": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "object_id": {"type": "string"},
                        "matched_previous_object_id": {"type": ["string", "null"]},
                        "name": {"type": "string"},
                        "description": {"type": "string"},
                        "location_in_scene": {"type": "string"},
                        "state": {"type": "string"},
                        "interaction": {"type": "string"},
                    },
                    "required": [
                        "object_id",
                        "matched_previous_object_id",
                        "name",
                        "description",
                        "location_in_scene",
                        "state",
                        "interaction",
                    ],
                },
            },
            "main_events": {"type": "array", "items": {"type": "string"}},
            "focused_event": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "description": {"type": "string"},
                    "subjects_involved": {"type": "array", "items": {"type": "string"}},
                    "objects_involved": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["description", "subjects_involved", "objects_involved"],
            },
            "visible_text": {"type": "array", "items": {"type": "object", "additionalProperties": False, "properties": {
                "text": {"type": "string"},
                "location": {"type": "string"},
            }, "required": ["text", "location"]}},
            "continuity_notes": {"type": "array", "items": {"type": "string"}},
            "segment_memory": {"type": "string"},
            "uncertainties": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "segment_index",
            "segment_start",
            "segment_end",
            "visual_clarity",
            "summary",
            "scene_details",
            "setting",
            "subjects",
            "objects",
            "main_events",
            "focused_event",
            "visible_text",
            "continuity_notes",
            "segment_memory",
            "uncertainties",
        ],
    }


def _last_segment_summary(memory: VideoMemory) -> str:
    for entry in reversed(memory.segment_memories):
        if entry.memory:
            return entry.memory
    return ""
