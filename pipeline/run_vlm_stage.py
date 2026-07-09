from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from pipeline.audio_windows import build_transcript_windows, extract_audio_window_files
from pipeline.global_summary import fuse_segment_ground_truth, generate_global_summary
from pipeline.temporal_segments import assign_transcript_chunks_to_segments
from pipeline.video_memory import create_video_memory, merge_segment_into_memory
from pipeline.vlm_reasoning import analyze_segment, build_failed_segment_response, serialize_segment_error
from schemas.segments import TemporalSegment, TemporalSegmentsArtifact
from schemas.transcription import TranscriptionRequestArtifact, TranscriptChunk
from schemas.vlm import SegmentVlmResponse, VlmSegmentArtifactEntry, VlmSegmentsArtifact
from services.fireworks_client import FireworksClient, FireworksConfig
from services.google_gemini_client import GoogleGeminiClient, GoogleGeminiConfig, GoogleGeminiError
from worker.config.settings import settings


async def run_vlm_reasoning_stage(
    *,
    job_id: str,
    local_temporal_segments_path: str,
    artifacts_root: str,
    local_audio_path: str | None = None,
    local_frame_sampling_path: str | None = None,
) -> dict[str, str]:
    if not settings.fireworks_api_key:
        raise ValueError("FIREWORKS_API_KEY is required for the VLM reasoning stage.")
    if not settings.fireworks_model:
        raise ValueError("FIREWORKS_MODEL is required for the VLM reasoning stage.")

    artifact_root = Path(artifacts_root)
    artifact_root.mkdir(parents=True, exist_ok=True)
    temporal_segments = TemporalSegmentsArtifact.model_validate_json(
        Path(local_temporal_segments_path).read_text(encoding="utf-8")
    )

    fireworks = FireworksClient(
        FireworksConfig(
            api_key=settings.fireworks_api_key,
            base_url=settings.fireworks_base_url,
            timeout_seconds=settings.fireworks_timeout_seconds,
            max_retries=settings.fireworks_max_retries,
        )
    )
    gemini_client = GoogleGeminiClient(
        GoogleGeminiConfig(
            api_key=settings.google_gemini_api_key or "",
            base_url=settings.google_gemini_base_url,
            model=settings.google_gemini_transcription_model,
            timeout_seconds=settings.google_gemini_timeout_seconds,
            max_retries=settings.google_gemini_max_retries,
        )
    )
    try:
        transcription_request, transcript_buckets, segment_artifact, video_memory = await _run_reasoning_branches(
            job_id=job_id,
            gemini_client=gemini_client,
            fireworks=fireworks,
            temporal_segments=temporal_segments,
            local_audio_path=local_audio_path,
            artifact_root=artifact_root,
        )

        for segment in temporal_segments.segments:
            segment.transcript_chunks = transcript_buckets[segment.segment_index]
            visual_entry = next(item for item in segment_artifact.segments if item.segment_index == segment.segment_index)
            segment.segment_ground_truth = fuse_segment_ground_truth(
                segment=segment,
                visual_response=visual_entry.vlm_response.model_dump(),
            )

        temporal_segments_path = artifact_root / "temporal_segments.json"
        video_memory_path = artifact_root / "video_memory.json"
        global_summary_path = artifact_root / "global_factual_summary.json"

        temporal_segments_path.write_text(temporal_segments.model_dump_json(indent=2), encoding="utf-8")
        video_memory_path.write_text(video_memory.model_dump_json(indent=2), encoding="utf-8")

        global_summary = await generate_global_summary(
            client=fireworks,
            model=settings.fireworks_model,
            job_id=job_id,
            video_memory=video_memory,
            segment_artifact=segment_artifact,
            temporal_segments=temporal_segments,
            all_frame_paths=_collect_all_frame_paths(temporal_segments.segments),
        )
        global_summary_path.write_text(global_summary.model_dump_json(indent=2), encoding="utf-8")

        artifact_paths = {
            "local_transcription_request_json": str(artifact_root / "transcription_request.json"),
            "local_vlm_segments_json": str(artifact_root / "vlm_segments.json"),
            "local_temporal_segments_json": str(temporal_segments_path),
            "local_video_memory_json": str(video_memory_path),
            "local_global_factual_summary_json": str(global_summary_path),
        }
        if local_frame_sampling_path:
            artifact_paths["local_frame_sampling_json"] = local_frame_sampling_path
        _ = transcription_request
        return artifact_paths
    finally:
        await gemini_client.aclose()
        await fireworks.aclose()


async def _run_reasoning_branches(
    *,
    job_id: str,
    gemini_client: GoogleGeminiClient,
    fireworks: FireworksClient,
    temporal_segments: TemporalSegmentsArtifact,
    local_audio_path: str | None,
    artifact_root: Path,
) -> tuple[TranscriptionRequestArtifact, list[list[TranscriptChunk]], VlmSegmentsArtifact, Any]:
    transcript_result, visual_result = await asyncio.gather(
        _run_transcript_branch(
            job_id=job_id,
            gemini_client=gemini_client,
            temporal_segments=temporal_segments,
            local_audio_path=local_audio_path,
            artifact_root=artifact_root,
        ),
        _run_visual_branch(
            fireworks=fireworks,
            job_id=job_id,
            temporal_segments=temporal_segments,
            artifact_root=artifact_root,
        ),
    )
    transcription_request, transcript_buckets = transcript_result
    segment_artifact, video_memory = visual_result
    return transcription_request, transcript_buckets, segment_artifact, video_memory


async def _run_visual_branch(
    *,
    fireworks: FireworksClient,
    job_id: str,
    temporal_segments: TemporalSegmentsArtifact,
    artifact_root: Path,
) -> tuple[VlmSegmentsArtifact, object]:
    memory = create_video_memory(job_id=job_id)
    segment_artifact = VlmSegmentsArtifact(job_id=job_id)
    successful_segments = 0

    for segment in temporal_segments.segments:
        response = await _process_segment(
            fireworks=fireworks,
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
                input_transcript_chunks=[],
                vlm_response=response,
            )
        )

    if successful_segments == 0:
        raise RuntimeError("All segment VLM calls failed; cannot generate a global factual summary.")

    vlm_segments_path = artifact_root / "vlm_segments.json"
    vlm_segments_path.write_text(segment_artifact.model_dump_json(indent=2), encoding="utf-8")
    return segment_artifact, memory


async def _run_transcript_branch(
    *,
    job_id: str,
    gemini_client: GoogleGeminiClient,
    temporal_segments: TemporalSegmentsArtifact,
    local_audio_path: str | None,
    artifact_root: Path,
) -> tuple[TranscriptionRequestArtifact, list[list[TranscriptChunk]]]:
    transcript_windows = build_transcript_windows(temporal_segments.video_metadata.duration, window_seconds=5.0)
    notes = [
        "Transcript chunks are aligned to 5-second windows.",
        "Transcript generation runs in parallel with visual-only segment analysis.",
        f"Gemini transcription concurrency is capped at {settings.google_gemini_max_concurrency} windows.",
    ]
    transcript_chunks = transcript_windows
    status = "skipped"

    if local_audio_path and settings.google_gemini_api_key:
        audio_windows_dir = artifact_root / "audio_windows"
        extracted_windows = await asyncio.to_thread(
            extract_audio_window_files,
            source_audio_path=Path(local_audio_path),
            output_dir=audio_windows_dir,
            transcript_windows=transcript_windows,
        )
        semaphore = asyncio.Semaphore(max(1, settings.google_gemini_max_concurrency))
        transcript_chunks = await asyncio.gather(
            *[
                _transcribe_window_safe(
                    job_id=job_id,
                    gemini_client=gemini_client,
                    semaphore=semaphore,
                    audio_path=window.path,
                    start=window.start,
                    end=window.end,
                )
                for window in extracted_windows
            ]
        )
        status = "completed"
    elif local_audio_path:
        notes.append("Audio was extracted, but GOOGLE_GEMINI_API_KEY was not set, so transcript generation was skipped.")
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
    transcription_request_path = artifact_root / "transcription_request.json"
    transcription_request_path.write_text(transcription_request.model_dump_json(indent=2), encoding="utf-8")
    buckets = assign_transcript_chunks_to_segments(
        transcript_chunks=transcript_chunks,
        duration=temporal_segments.video_metadata.duration,
        segment_count=len(temporal_segments.segments),
    )
    return transcription_request, buckets


async def _process_segment(
    *,
    fireworks: FireworksClient,
    job_id: str,
    segment: TemporalSegment,
    memory,
) -> SegmentVlmResponse:
    try:
        return await analyze_segment(
            client=fireworks,
            model=settings.fireworks_model or "",
            job_id=job_id,
            segment=segment,
            memory=memory,
            include_transcript=bool(segment.transcript_chunks),
        )
    except Exception as exc:  # noqa: BLE001
        error_message = serialize_segment_error(exc)
        return build_failed_segment_response(segment=segment, error_message=error_message)
async def _transcribe_window_safe(
    *,
    job_id: str,
    gemini_client: GoogleGeminiClient,
    semaphore: asyncio.Semaphore,
    audio_path: Path,
    start: float,
    end: float,
) -> TranscriptChunk:
    async with semaphore:
        try:
            return await gemini_client.transcribe_audio_window(audio_path=audio_path, start=start, end=end)
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
