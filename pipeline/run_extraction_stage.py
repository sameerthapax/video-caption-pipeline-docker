from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from pipeline.extract_frames import extract_selected_frames
from pipeline.frame_sampling import build_frame_sampling_artifact
from pipeline.probe_video import probe_video_metadata
from pipeline.scene_change import analyze_scene_changes
from pipeline.temporal_segments import build_temporal_segments_artifact
from schemas.frames import FrameSamplingArtifact
from schemas.segments import SamplingConfig, TemporalSegmentsArtifact
from worker.config.settings import settings


async def run_video_extraction_stage(
    *,
    job_id: str,
    local_video_path: str,
    local_audio_path: str | None,
    artifacts_root: str,
    persist_artifacts: bool = False,
) -> dict[str, Any]:
    artifact_root = Path(artifacts_root)
    frames_dir = artifact_root / "frames"
    video_path = Path(local_video_path)
    if persist_artifacts:
        artifact_root.mkdir(parents=True, exist_ok=True)
    video_metadata = probe_video_metadata(video_path)
    sampling_config = build_sampling_config(video_metadata.duration)
    frame_sampling = await _build_frame_sampling_async(
        job_id=job_id,
        local_video_path=video_path,
        video_metadata=video_metadata,
        sampling_config=sampling_config,
    )
    frames = await asyncio.to_thread(
        extract_selected_frames,
        job_id=job_id,
        video_path=video_path,
        output_dir=frames_dir,
        storage_prefix="frames",
        timestamps_with_reasons=[
            (item.timestamp, item.selection_reasons, item.scene_change_score)
            for item in frame_sampling.final_selected_frames
        ],
        max_width=settings.frame_extract_width,
    )
    temporal_segments = build_temporal_segments_artifact(
        job_id=job_id,
        video_metadata=video_metadata,
        sampling_config=sampling_config,
        frames=frames,
        transcript_chunks=[],
        segment_count=settings.pipeline_segment_count,
    )

    if persist_artifacts:
        (artifact_root / "frame_sampling.json").write_text(frame_sampling.model_dump_json(indent=2), encoding="utf-8")
        (artifact_root / "temporal_segments.json").write_text(
            temporal_segments.model_dump_json(indent=2),
            encoding="utf-8",
        )

    return {
        "frames": [frame.local_path for frame in frames],
        "frame_sampling": frame_sampling,
        "temporal_segments": temporal_segments,
        "local_video_path": str(video_path),
        "local_audio_path": local_audio_path or "",
        "video_duration_seconds": video_metadata.duration,
    }


def build_sampling_config(duration_seconds: float) -> SamplingConfig:
    target_total = max(
        settings.min_frames_per_video,
        min(
            settings.max_frames_per_video,
            int(round(duration_seconds / max(settings.target_seconds_per_frame, 1.0))) + 2,
        ),
    )
    scene_change_count = min(settings.max_scene_change_frames, max(1, target_total // 2))
    remaining = max(0, target_total - scene_change_count)
    uniform_count = min(settings.max_uniform_frames, max(1, remaining))
    remaining = max(0, remaining - uniform_count)
    safety_count = min(settings.max_safety_frames, max(1, remaining or 1))
    return SamplingConfig(
        uniform_count=uniform_count,
        scene_change_count=scene_change_count,
        safety_count=safety_count,
    )


async def _build_frame_sampling_async(
    *,
    job_id: str,
    local_video_path: Path,
    video_metadata,
    sampling_config: SamplingConfig,
) -> FrameSamplingArtifact:
    scene_change_result = await asyncio.to_thread(
        analyze_scene_changes,
        video_path=local_video_path,
        duration=video_metadata.duration,
    )
    return await asyncio.to_thread(
        build_frame_sampling_artifact,
        job_id=job_id,
        video_metadata=video_metadata,
        scene_change_result=scene_change_result,
        sampling_config=sampling_config,
    )
