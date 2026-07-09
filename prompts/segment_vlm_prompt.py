from __future__ import annotations

import json

from schemas.segments import TemporalSegment
from schemas.video_memory import VideoMemory


def build_segment_vlm_prompt(
    *,
    job_id: str,
    segment: TemporalSegment,
    memory: VideoMemory,
    include_transcript: bool = False,
) -> str:
    frames_payload = [
        {
            "frame_id": frame.frame_id,
            "timestamp": frame.timestamp,
            "storage_path": frame.storage_path,
            "selection_reasons": frame.selection_reasons,
        }
        for frame in segment.frames
    ]
    transcript_payload = []
    if include_transcript:
        transcript_payload = [
            {
                "start": chunk.start,
                "end": chunk.end,
                "text": chunk.text,
                "expressive_transcript": chunk.expressive_transcript,
                "music": chunk.music.model_dump(),
                "tone": chunk.tone.model_dump(),
            }
            for chunk in segment.transcript_chunks
        ]
    prior_memory = {
        "segments_processed": memory.segments_processed,
        "global_setting": memory.global_setting.model_dump(),
        "persistent_subjects": [item.model_dump() for item in memory.persistent_subjects],
        "persistent_objects": [item.model_dump() for item in memory.persistent_objects],
        "timeline": [item.model_dump() for item in memory.timeline],
        "segment_memories": [item.model_dump() for item in memory.segment_memories if item.memory],
        "unresolved_uncertainties": memory.unresolved_uncertainties,
    }

    return f"""
You are analyzing segment {segment.segment_index + 1} of 5.
Job ID: {job_id}
This segment covers {segment.start}s to {segment.end}s.
Percent range: {segment.percent_range}

You are given selected video frames from this segment.
You are also given accumulated memory from previous segments.

Your job:
1. Analyze only the current segment frames.
2. Use previous memory only to maintain continuity.
3. Determine whether visible people, objects, or locations are the same as earlier.
4. Update the memory instead of creating duplicates unnecessarily.
5. Describe factual screen content, movement, visible behavior, emotion if visually clear, clothing, objects, setting, and timeline flow.
6. Do not invent unseen actions.
7. Do not infer identity, gender, relationship, profession, race, ethnicity, or private traits unless visually explicit.
8. If uncertain, use "unknown" or confidence < 0.6.
9. Return JSON only.
10. `segment_index` must be the exact zero-based index for this segment: {segment.segment_index}.
11. `segment_start` must be exactly {segment.start}.
12. `segment_end` must be exactly {segment.end}.
13. Use only these exact enum values:
   - `visual_clarity`: `high`, `medium`, `low`
   - `audio_clarity`: `high`, `medium`, `low`, `none`
   - `type`: `person`, `animal`, `object`, `unknown`
   - `evidence`: `visual`, `audio`, `both`

Important things to look for in every segment:
- setting/location
- visible people/subjects
- subject appearance
- clothing
- posture
- body movement
- facial expression if clearly visible
- emotion/mood if clearly supported
- objects
- object state changes
- interactions between people/objects
- actions and events
- camera movement or scene transition
- text visible on screen
- continuity with previous segments
- what changed from previous memory

Strict anti-hallucination rules:
- Only describe visible or audible evidence from the frames and transcript.
- If visibility is weak, say that it is unclear.
- If audio is missing or unclear, say that it is unclear.
- Do not guess missing details.
- Do not write styled captions, jokes, sarcasm, or commentary.

Return JSON with exactly this structure:
{json.dumps(_segment_response_shape(), indent=2)}

Current segment frames:
{json.dumps(frames_payload, indent=2)}

Accumulated memory from previous segments:
{json.dumps(prior_memory, indent=2)}

Current segment transcript chunks:
{json.dumps(transcript_payload, indent=2)}
""".strip()


def _segment_response_shape() -> dict:
    return {
        "segment_index": 0,
        "segment_start": 0.0,
        "segment_end": 0.0,
        "evidence_quality": {
            "visual_clarity": "high",
            "audio_clarity": "none",
            "limitations": [],
        },
        "segment_description": {
            "short_summary": "",
            "detailed_visual_description": "",
            "movement_and_flow": "",
            "audio_summary": "",
        },
        "setting": {
            "location_type": "",
            "visual_environment": "",
            "background_details": [],
            "changes_from_previous_segment": "",
        },
        "subjects": [
            {
                "subject_id": "person_1",
                "is_new": True,
                "matched_previous_subject_id": None,
                "type": "person",
                "appearance": {
                    "visible_features": [],
                    "clothing": [],
                    "colors": [],
                    "accessories": [],
                    "pose_or_posture": "",
                    "confidence": 0.0,
                },
                "actions": [],
                "movement": "",
                "facial_expression": "",
                "emotion_or_tone": {
                    "label": "",
                    "evidence": "",
                    "confidence": 0.0,
                },
                "state_change": "",
                "confidence": 0.0,
            }
        ],
        "objects": [
            {
                "object_id": "object_1",
                "is_new": True,
                "matched_previous_object_id": None,
                "name": "",
                "description": "",
                "location_in_scene": "",
                "state": "",
                "interaction": "",
                "confidence": 0.0,
            }
        ],
        "events": [
            {
                "event_id": "event_1",
                "approx_timestamp": 0.0,
                "description": "",
                "subjects_involved": [],
                "objects_involved": [],
                "evidence": "visual",
                "confidence": 0.0,
            }
        ],
        "visible_text": [{"text": "", "location": "", "confidence": 0.0}],
        "audio_observations": {
            "spoken_content_summary": "",
            "music": {
                "present": False,
                "type": "",
                "instrumentation": "",
                "energy": "",
                "description": "",
            },
            "tone": {
                "style": "",
                "emotion": "",
                "delivery": "",
                "evidence": "",
                "confidence": 0.0,
            },
        },
        "continuity_update": {
            "same_subjects_as_before": [],
            "same_objects_as_before": [],
            "new_subjects": [],
            "new_objects": [],
            "resolved_uncertainties": [],
            "new_uncertainties": [],
            "important_changes": [],
        },
        "memory_update_for_next_segment": {
            "segment_memory": "",
            "persistent_subjects": [],
            "persistent_objects": [],
            "persistent_setting": "",
            "open_actions_or_context": [],
            "timeline_update": [],
        },
    }
