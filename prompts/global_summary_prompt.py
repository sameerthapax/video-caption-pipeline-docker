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


def build_observation_prompt(*, job_id: str, transcript_text: str) -> str:
    transcript_block = transcript_text.strip() or "No transcript available."
    return f"""
You are analyzing sampled frames from a short video clip for job {job_id}.

Write a dense factual observation summary grounded only in the frames and optional transcript.

Rules:
- Be specific, but conservative.
- Prefer generic wording over guesses when a detail is unclear.
- Do not infer exact identities, brands, organizations, or locations unless clearly visible or stated in the transcript.
- Capture the overall setting, main subjects, key objects, main action, beginning-to-end flow, visible text, audio cues, scene changes, and uncertainties.
- Keep the summary compact enough to support a fast downstream captioning pass.
- Return JSON only.

Optional transcript:
{transcript_block}

Return JSON with exactly this shape:
{{
  "factual_summary": "2-4 concise sentences",
  "detailed_ground_truth": "slightly denser factual paragraph",
  "main_subjects": ["subject"],
  "main_objects": ["object"],
  "setting_summary": "setting",
  "audio_summary": "audio or speech summary",
  "scene_change_overview": [
    {{
      "segment_index": 0,
      "apparent_change": "change summary",
      "related_subjects": [],
      "related_objects": []
    }}
  ],
  "continuity_overview": [
    {{
      "segment_index": 0,
      "continuity_note": "continuity summary",
      "continued_subjects": [],
      "new_subjects": [],
      "continued_objects": [],
      "new_objects": []
    }}
  ],
  "object_and_subject_tracking": [
    {{
      "object_id": "entity_1",
      "name": "entity name",
      "tracking_summary": "tracking summary",
      "entity_type": "person|object|animal|unknown",
      "appears_across_segments": [0]
    }}
  ],
  "uncertainties": ["uncertain detail"],
  "confidence": 0.0
}}
""".strip()


def build_progressive_chunk_prompt(
    *,
    job_id: str,
    transcript_text: str,
    chunk_index: int,
    chunk_count: int,
) -> str:
    transcript_block = transcript_text.strip() or "No transcript available."
    return f"""
You are analyzing chunk {chunk_index + 1} of {chunk_count} from a sampled short video for job {job_id}.

Write a compact factual summary of only these frames.

Rules:
- Use only what is visible in these frames and the optional transcript.
- Capture the local setting, subjects, objects, actions, visible text, and what seems to change within this chunk.
- Preserve small but identifying details that may matter later.
- Be conservative when uncertain.
- Return JSON only.

Optional transcript:
{transcript_block}

Return JSON with exactly this shape:
{{
  "chunk_index": {chunk_index},
  "chunk_summary": "2-3 factual sentences",
  "key_details": ["important grounded detail"],
  "subjects": ["subject"],
  "objects": ["object"],
  "actions": ["action"],
  "setting": "setting summary",
  "visible_text": ["visible text"],
  "uncertainties": ["uncertain detail"]
}}
""".strip()


def build_progressive_aggregate_prompt(*, job_id: str, transcript_text: str, chunk_summaries_json: str) -> str:
    transcript_block = transcript_text.strip() or "No transcript available."
    return f"""
You are combining chunk-level visual summaries into one grounded full-video summary for job {job_id}.

Rules:
- Use only the chunk summaries and optional transcript.
- Preserve chronology and keep the most identifying grounded details.
- Merge repeated details instead of repeating them word-for-word.
- Do not invent facts that are not present in the chunk summaries.
- Keep the result concise but detailed enough to support style-specific caption generation.
- Return JSON only.

Optional transcript:
{transcript_block}

Chunk summaries:
{chunk_summaries_json}

Return JSON with exactly this structure:
{json.dumps(_summary_shape(), indent=2)}
""".strip()


def build_verification_prompt(*, draft_summary_json: str, transcript_text: str) -> str:
    transcript_block = transcript_text.strip() or "No transcript available."
    return f"""
You are verifying a draft grounded video summary against the sampled video frames and optional transcript.

Rules:
- If the draft contains unsupported, overstated, or overly specific claims, correct or remove them.
- Keep the result concise and useful for downstream caption generation.
- Preserve only details supported by the frames or transcript.
- Do not add new facts that are not grounded.
- Return JSON only in the same schema as the draft.

Optional transcript:
{transcript_block}

Draft summary JSON:
{draft_summary_json}
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
