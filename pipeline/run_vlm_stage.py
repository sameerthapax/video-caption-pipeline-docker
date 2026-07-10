from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from pipeline.audio_windows import build_transcript_windows, extract_audio_window_chunks
from pipeline.global_summary import build_segment_based_global_summary, fuse_segment_ground_truth
from pipeline.temporal_segments import assign_transcript_chunks_to_segments
from pipeline.video_memory import create_video_memory, merge_segment_into_memory
from pipeline.vlm_reasoning import analyze_segment, build_failed_segment_response, serialize_segment_error
from schemas.segments import TemporalSegment, TemporalSegmentsArtifact
from schemas.transcription import TranscriptionRequestArtifact, TranscriptChunk
from schemas.video_memory import VideoMemory
from schemas.vlm import SegmentVlmResponse, VlmSegmentArtifactEntry, VlmSegmentsArtifact
from services.client_pool import get_gemini_client
from services.google_gemini_client import GoogleGeminiClient, GoogleGeminiError
from worker.config.settings import settings


async def run_vlm_reasoning_stage(
    *,
    job_id: str,
    temporal_segments: TemporalSegmentsArtifact,
    artifacts_root: str,
    local_audio_path: str | None = None,
    persist_artifacts: bool = False,
) -> dict[str, Any]:
    if not settings.google_gemini_api_key and not settings.google_gemini_proxy_url:
        raise ValueError("GOOGLE_GEMINI_API_KEY or GOOGLE_GEMINI_PROXY_URL is required for the VLM reasoning stage.")
    if not settings.google_gemini_vision_model:
        raise ValueError("GOOGLE_GEMINI_VISION_MODEL is required for the VLM reasoning stage.")

    artifact_root = Path(artifacts_root)
    if persist_artifacts:
        artifact_root.mkdir(parents=True, exist_ok=True)

    gemini_client = get_gemini_client()

    transcript_task = asyncio.create_task(
        _run_transcript_branch(
            job_id=job_id,
            gemini_client=gemini_client,
            temporal_segments=temporal_segments,
            local_audio_path=local_audio_path,
            artifact_root=artifact_root,
            persist_artifacts=persist_artifacts,
        )
    )

    segment_artifact, video_memory = await _run_visual_branch(
        gemini_client=gemini_client,
        job_id=job_id,
        temporal_segments=temporal_segments,
        artifact_root=artifact_root,
        persist_artifacts=persist_artifacts,
    )

    transcription_request, transcript_buckets = await transcript_task

    for segment in temporal_segments.segments:
        segment.transcript_chunks = transcript_buckets[segment.segment_index]
    for entry in segment_artifact.segments:
        entry.input_transcript_chunks = transcript_buckets[entry.segment_index]

    for segment in temporal_segments.segments:
        visual_entry = next(item for item in segment_artifact.segments if item.segment_index == segment.segment_index)
        segment.segment_ground_truth = fuse_segment_ground_truth(
            segment=segment,
            visual_response=visual_entry.vlm_response.model_dump(),
        )

    global_summary = build_segment_based_global_summary(
        job_id=job_id,
        temporal_segments=temporal_segments,
        segment_artifact=segment_artifact,
        video_memory=video_memory,
        transcript_text=_flatten_transcript_chunks(transcript_buckets),
    )

    if persist_artifacts:
        (artifact_root / "temporal_segments.json").write_text(
            temporal_segments.model_dump_json(indent=2),
            encoding="utf-8",
        )
        (artifact_root / "video_memory.json").write_text(
            video_memory.model_dump_json(indent=2),
            encoding="utf-8",
        )
        (artifact_root / "global_factual_summary.json").write_text(
            global_summary.model_dump_json(indent=2),
            encoding="utf-8",
        )

    return {
        "transcription_request": transcription_request,
        "segment_artifact": segment_artifact,
        "temporal_segments": temporal_segments,
        "video_memory": video_memory,
        "global_summary": global_summary,
    }


async def _run_visual_branch(
    *,
    gemini_client: GoogleGeminiClient,
    job_id: str,
    temporal_segments: TemporalSegmentsArtifact,
    artifact_root: Path,
    persist_artifacts: bool,
) -> tuple[VlmSegmentsArtifact, VideoMemory]:
    memory = create_video_memory(job_id=job_id)
    segment_artifact = VlmSegmentsArtifact(job_id=job_id)
    successful_segments = 0

    for segment in temporal_segments.segments:
        response = await _process_segment(
            gemini_client=gemini_client,
            job_id=job_id,
            segment=segment,
            memory=memory,
        )
        if response.status != "failed":
            successful_segments += 1
        memory = merge_segment_into_memory(memory=memory, segment=segment, response=response)
        segment_artifact.segments.append(
            VlmSegmentArtifactEntry(
                segment_index=segment.segment_index,
                input_frames=segment.frames,
                input_transcript_chunks=segment.transcript_chunks,
                vlm_response=response,
            )
        )

    if successful_segments == 0:
        raise RuntimeError("All segment VLM calls failed; cannot generate a global factual summary.")

    if persist_artifacts:
        (artifact_root / "vlm_segments.json").write_text(
            segment_artifact.model_dump_json(indent=2),
            encoding="utf-8",
        )
    return segment_artifact, memory


async def _run_transcript_branch(
    *,
    job_id: str,
    gemini_client: GoogleGeminiClient,
    temporal_segments: TemporalSegmentsArtifact,
    local_audio_path: str | None,
    artifact_root: Path,
    persist_artifacts: bool,
) -> tuple[TranscriptionRequestArtifact, list[list[TranscriptChunk]]]:
    segment_duration = temporal_segments.video_metadata.duration / max(1, len(temporal_segments.segments))
    transcript_window_seconds = max(3.0, min(8.0, round(segment_duration / 2.0, 1)))
    transcript_windows = build_transcript_windows(
        temporal_segments.video_metadata.duration,
        window_seconds=transcript_window_seconds,
    )
    notes = [
        f"Transcript chunks are aligned to {transcript_window_seconds:.1f}-second windows.",
        "Transcript generation runs in parallel with frame extraction before segment reasoning begins.",
        f"Gemini transcription concurrency is capped at {settings.google_gemini_max_concurrency} windows.",
    ]
    transcript_chunks = transcript_windows
    status = "skipped"

    if local_audio_path and (settings.google_gemini_api_key or settings.google_gemini_proxy_url):
        extracted_windows = await asyncio.to_thread(
            extract_audio_window_chunks,
            source_audio_path=Path(local_audio_path),
            transcript_windows=transcript_windows,
        )
        semaphore = asyncio.Semaphore(max(1, settings.google_gemini_max_concurrency))
        transcript_chunks = await asyncio.gather(
            *[
                _transcribe_window_safe(
                    job_id=job_id,
                    gemini_client=gemini_client,
                    semaphore=semaphore,
                    audio_bytes=window.audio_bytes,
                    start=window.start,
                    end=window.end,
                )
                for window in extracted_windows
            ]
        )
        status = "completed"
    elif local_audio_path:
        notes.append("Audio was extracted, but neither GOOGLE_GEMINI_API_KEY nor GOOGLE_GEMINI_PROXY_URL was set, so transcript generation was skipped.")
    else:
        notes.append("No audio track was available, so transcript generation was skipped.")

    transcription_request = TranscriptionRequestArtifact(
        job_id=job_id,
        source_audio_storage_path=local_audio_path or "",
        provider="google_gemini",
        status=status,
        provider_metadata=gemini_client.build_transcription_request_metadata(),
        notes=notes,
        transcript_chunks=transcript_chunks,
    )
    if persist_artifacts:
        (artifact_root / "transcription_request.json").write_text(
            transcription_request.model_dump_json(indent=2),
            encoding="utf-8",
        )
    buckets = assign_transcript_chunks_to_segments(
        transcript_chunks=transcript_chunks,
        duration=temporal_segments.video_metadata.duration,
        segment_count=len(temporal_segments.segments),
    )
    return transcription_request, buckets


async def _process_segment(
    *,
    gemini_client: GoogleGeminiClient,
    job_id: str,
    segment: TemporalSegment,
    memory,
) -> SegmentVlmResponse:
    try:
        return await analyze_segment(
            client=gemini_client,
            model=settings.google_gemini_vision_model,
            job_id=job_id,
            segment=segment,
            memory=memory,
        )
    except Exception as exc:  # noqa: BLE001
        error_message = serialize_segment_error(exc)
        return build_failed_segment_response(segment=segment, error_message=error_message)


async def _transcribe_window_safe(
    *,
    job_id: str,
    gemini_client: GoogleGeminiClient,
    semaphore: asyncio.Semaphore,
    audio_bytes: bytes,
    start: float,
    end: float,
) -> TranscriptChunk:
    _ = job_id
    async with semaphore:
        try:
            return await gemini_client.transcribe_audio_window(audio_bytes=audio_bytes, start=start, end=end)
        except GoogleGeminiError:
            return TranscriptChunk(start=start, end=end, text="", expressive_transcript="")


def _collect_all_frame_paths(segments: list[TemporalSegment]) -> list[str]:
    seen: set[str] = set()
    paths: list[str] = []
    for segment in segments:
        for frame in segment.frames:
            if frame.local_path and frame.local_path not in seen:
                seen.add(frame.local_path)
                paths.append(frame.local_path)
    return paths


def _flatten_transcript_chunks(transcript_buckets: list[list[TranscriptChunk]]) -> str:
    lines: list[str] = []
    for bucket in transcript_buckets:
        for chunk in bucket:
            text = chunk.expressive_transcript.strip() or chunk.text.strip()
            if text:
                lines.append(f"{chunk.start:.1f}-{chunk.end:.1f}s: {text}")
    return "\n".join(lines)
