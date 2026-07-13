from __future__ import annotations

from pathlib import Path

from schemas.frames import FrameArtifact, FrameExtractionArtifact
from schemas.video import VideoMetadata
from worker.config.settings import settings


def extract_frames_for_video(
    *,
    job_id: str,
    video_path: Path,
    output_dir: Path,
    video_metadata: VideoMetadata,
) -> FrameExtractionArtifact:
    output_dir.mkdir(parents=True, exist_ok=True)
    target_frame_count = compute_dynamic_frame_count(video_metadata.duration, settings.max_frames_per_video)
    uniform_frames = _extract_uniform_frames(
        video_path=video_path,
        frame_dir=output_dir / "uniform",
        frame_count=target_frame_count,
        width=settings.frame_extract_width,
        duration=video_metadata.duration,
    )
    if not uniform_frames:
        raise RuntimeError(f"Could not extract frames from {video_path}")

    return FrameExtractionArtifact(
        job_id=job_id,
        duration_seconds=video_metadata.duration,
        target_frame_count=target_frame_count,
        strategy="uniform",
        frames=uniform_frames,
    )


def compute_dynamic_frame_count(duration_seconds: float | None, max_frames: int) -> int:
    capped_max = max(6, min(max_frames, 12))
    if duration_seconds is None or duration_seconds <= 0:
        return capped_max
    if duration_seconds <= 15:
        return 6
    if duration_seconds >= 180:
        return capped_max
    ratio = duration_seconds / 180.0
    dynamic_count = 6 + int(round(ratio * (capped_max - 6)))
    return max(6, min(dynamic_count, capped_max))


def _extract_uniform_frames(*, video_path: Path, frame_dir: Path, frame_count: int, width: int, duration: float) -> list[FrameArtifact]:
    import cv2

    frame_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return []
    try:
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if total_frames <= 0 or fps <= 0 or frame_count <= 0:
            return []
        positions = _uniform_frame_positions(total_frames=total_frames, frame_count=frame_count)
        artifacts: list[FrameArtifact] = []
        for index, position in enumerate(positions, start=1):
            capture.set(cv2.CAP_PROP_POS_FRAMES, position)
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            resized = _resize_frame(frame=frame, max_width=width)
            output_path = frame_dir / f"frame_{index:03d}.jpg"
            if not cv2.imwrite(str(output_path), resized, [int(cv2.IMWRITE_JPEG_QUALITY), 88]):
                continue
            artifacts.append(
                FrameArtifact(
                    frame_id=f"frame_{index:03d}",
                    timestamp=_timestamp_for_position(position=position, fps=fps, duration=duration),
                    local_path=str(output_path),
                    selection_reasons=["uniform"],
                )
            )
        return artifacts
    finally:
        capture.release()


def _uniform_frame_positions(*, total_frames: int, frame_count: int) -> list[int]:
    if frame_count <= 1:
        return [max(0, total_frames // 2)]
    last_index = max(0, total_frames - 1)
    return [
        min(last_index, round(index * last_index / (frame_count - 1)))
        for index in range(frame_count)
    ]


def _timestamp_for_position(*, position: int, fps: float, duration: float) -> float:
    timestamp = position / fps if fps > 0 else 0.0
    if duration > 0:
        timestamp = min(timestamp, duration)
    return round(max(0.0, timestamp), 4)


def _resize_frame(*, frame, max_width: int):
    import cv2

    height, width = frame.shape[:2]
    if width <= max_width:
        return frame
    target_height = int(round(height * (max_width / width)))
    return cv2.resize(frame, (max_width, target_height), interpolation=cv2.INTER_AREA)
