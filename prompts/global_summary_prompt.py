from __future__ import annotations

import json

from schemas.segments import TemporalSegmentsArtifact
from schemas.video_memory import VideoMemory
from schemas.vlm import VlmSegmentsArtifact


def build_global_summary_prompt(
    *,
    job_id: str,
    video_memory: VideoMemory,
    segment_artifact: VlmSegmentsArtifact,
    temporal_segments: TemporalSegmentsArtifact,
) -> str:
    frames_payload = [
        {
            "segment_index": segment.segment_index,
            "frames": [
                {
                    "frame_id": frame.frame_id,
                    "timestamp": frame.timestamp,
                    "storage_path": frame.storage_path,
                }
                for frame in segment.frames
            ],
        }
        for segment in temporal_segments.segments
    ]
    transcript_payload = [
        {
            "segment_index": segment.segment_index,
            "transcript_chunks": [
                {
                    "start": chunk.start,
                    "end": chunk.end,
                    "text": chunk.text,
                    "expressive_transcript": chunk.expressive_transcript,
                }
                for chunk in segment.transcript_chunks
            ],
        }
        for segment in temporal_segments.segments
    ]
    return f"""
Create a factual, style-neutral video description.

Job ID: {job_id}

Rules:
- Use only evidence in video memory, fused per-segment truth, transcript chunks, provided segment frames, and segment reasoning outputs.
- Do not be funny or sarcastic.
- Do not invent details.
- Mention setting, subjects, objects, key actions, continuity, scene changes, and transcript-supported context.
- Compare transcript timing, speaking style, tone, and apparent speaker changes against the visual evidence and segment-level visual analysis.
- Be highly detailed because this output is the ground truth source for later caption generation.
- Distinguish what is directly visible, what is directly spoken, and where those two sources align or diverge.
- Identify recurring people, objects, and settings across segments and explain continuity changes.
- If the transcript suggests speech or tone but the visible speaker is uncertain, say that explicitly.
- If multiple people may be present, note whether speaker identity is visually confirmed or unconfirmed.
- If the video has no clear outcome, say what is visible instead of inventing one.
- No audio file is available here. Use only transcript text and prior fused evidence for audio-related statements.
- Return JSON only.

Return JSON with exactly this structure:
{json.dumps(_summary_shape(), indent=2)}

Video memory:
{video_memory.model_dump_json(indent=2)}

Segment reasoning outputs:
{segment_artifact.model_dump_json(indent=2)}

Fused temporal segments ground truth:
{temporal_segments.model_dump_json(indent=2)}

Transcript chunks:
{json.dumps(transcript_payload, indent=2)}

Segment frames:
{json.dumps(frames_payload, indent=2)}
""".strip()


def _summary_shape() -> dict:
    return {
        "factual_summary": "",
        "detailed_ground_truth": "",
        "detailed_timeline": [
            {
                "segment_index": 0,
                "summary": "",
                "key_events": [],
                "visual_facts": [],
                "transcript_facts": [],
                "continuity_notes": [],
            }
        ],
        "main_subjects": [],
        "main_objects": [],
        "setting_summary": "",
        "audio_summary": "",
        "transcript_visual_alignment": [
            {
                "segment_index": 0,
                "transcript_summary": "",
                "visual_alignment": "",
                "speaker_changes": [],
                "tone_notes": [],
                "mismatches_or_uncertainties": [],
            }
        ],
        "speaker_analysis": [
            {
                "speaker_label": "",
                "speaking_style": "",
                "tone_or_emotion": "",
                "evidence": [],
                "appears_across_segments": [],
                "confidence": 0.0,
            }
        ],
        "scene_change_overview": [
            {
                "segment_index": 0,
                "apparent_change": "",
                "related_subjects": [],
                "related_objects": [],
            }
        ],
        "continuity_overview": [
            {
                "segment_index": 0,
                "continuity_note": "",
                "continued_subjects": [],
                "new_subjects": [],
                "continued_objects": [],
                "new_objects": [],
            }
        ],
        "object_and_subject_tracking": [
            {
                "object_id": "",
                "name": "",
                "tracking_summary": "",
                "entity_type": "",
                "appears_across_segments": [],
            }
        ],
        "uncertainties": [],
        "confidence": 0.0,
    }
