from __future__ import annotations

import asyncio
from pathlib import Path

from pipeline.extract_frames import extract_selected_frames
from pipeline.frame_sampling import build_frame_sampling_artifact
from pipeline.probe_video import probe_video_metadata
from pipeline.scene_change import analyze_scene_changes
from pipeline.temporal_segments import build_temporal_segments_artifact
from schemas.segments import SamplingConfig
from worker.config.settings import settings


async def run_video_extraction_stage(
    *,
    job_id: str,
    local_video_path: str,
    local_audio_path: str | None,
    artifacts_root: str,
) -> dict[str, str | float | list[str]]:
    artifact_root = Path(artifacts_root)
    frames_dir = artifact_root / "frames"
    video_path = Path(local_video_path)
    sampling_config = SamplingConfig()

    artifact_root.mkdir(parents=True, exist_ok=True)
    video_metadata = probe_video_metadata(video_path)
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
    )

    frame_sampling_path = artifact_root / "frame_sampling.json"
    temporal_segments_path = artifact_root / "temporal_segments.json"
    frame_sampling_path.write_text(frame_sampling.model_dump_json(indent=2), encoding="utf-8")
    temporal_segments_path.write_text(temporal_segments.model_dump_json(indent=2), encoding="utf-8")

    return {
        "frames": [frame.local_path for frame in frames],
        "local_frame_sampling_json": str(frame_sampling_path),
        "local_temporal_segments_json": str(temporal_segments_path),
        "local_video_path": str(video_path),
        "local_audio_path": local_audio_path or "",
        "video_duration_seconds": video_metadata.duration,
    }


async def _build_frame_sampling_async(
    *,
    job_id: str,
    local_video_path: Path,
    video_metadata,
    sampling_config: SamplingConfig,
):
    _ = job_id
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
