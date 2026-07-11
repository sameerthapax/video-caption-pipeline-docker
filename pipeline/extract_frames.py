from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path

import numpy as np

from pipeline.scene_change import SceneChangeResult, analyze_scene_changes
from schemas.frames import FrameArtifact, FrameExtractionArtifact
from schemas.video import VideoMetadata
from worker.config.settings import settings

logger = logging.getLogger("video-caption-pipeline.worker")


def extract_frames_for_video(
    *,
    job_id: str,
    video_path: Path,
    output_dir: Path,
    video_metadata: VideoMetadata,
) -> FrameExtractionArtifact:
    output_dir.mkdir(parents=True, exist_ok=True)
    target_frame_count = compute_dynamic_frame_count(video_metadata.duration, settings.max_frames_per_video)

    if not settings.enable_planned_frame_extraction:
        anchor_frames = _extract_at_timestamps(
            video_path=video_path,
            frame_dir=output_dir / "anchor",
            width=settings.frame_extract_width,
            timestamps=[(timestamp, ["anchor"]) for timestamp in _sample_anchor_timestamps(
                video_metadata.duration,
                target_frame_count,
            )],
            score_lookup={},
        )
        if anchor_frames:
            return FrameExtractionArtifact(
                job_id=job_id,
                duration_seconds=video_metadata.duration,
                target_frame_count=target_frame_count,
                strategy="anchor",
                frames=anchor_frames,
            )
        uniform_frames = _extract_uniform_frames(
            video_path=video_path,
            frame_dir=output_dir / "uniform",
            max_frames=target_frame_count,
            width=settings.frame_extract_width,
            duration=video_metadata.duration,
        )
        if uniform_frames:
            return FrameExtractionArtifact(
                job_id=job_id,
                duration_seconds=video_metadata.duration,
                target_frame_count=target_frame_count,
                strategy="uniform",
                frames=uniform_frames,
            )
        raise RuntimeError(f"Could not extract frames from {video_path}")

    scene_result = analyze_scene_changes(
        video_path=video_path,
        duration=video_metadata.duration,
        config=None,
    )

    if settings.use_opencv_frames:
        opencv_frames = _extract_opencv_frames(
            video_path=video_path,
            frame_dir=output_dir / "opencv",
            max_frames=target_frame_count,
            width=settings.frame_extract_width,
        )
        if opencv_frames:
            return FrameExtractionArtifact(
                job_id=job_id,
                duration_seconds=video_metadata.duration,
                target_frame_count=target_frame_count,
                strategy="opencv",
                scene_candidates=scene_result.candidates,
                frames=_deduplicate_frame_artifacts(opencv_frames),
            )

    planned_frames = _extract_planned_frames(
        video_path=video_path,
        frame_dir=output_dir / "planned",
        max_frames=target_frame_count,
        width=settings.frame_extract_width,
        scene_result=scene_result,
        duration=video_metadata.duration,
    )
    if len(planned_frames) >= max(1, min(target_frame_count, settings.min_anchor_frames)):
        return FrameExtractionArtifact(
            job_id=job_id,
            duration_seconds=video_metadata.duration,
            target_frame_count=target_frame_count,
            strategy="planned",
            scene_candidates=scene_result.candidates,
            frames=_deduplicate_frame_artifacts(planned_frames),
        )

    if settings.use_scene_midpoint_frames:
        midpoint_frames = _extract_scene_midpoint_frames(
            video_path=video_path,
            frame_dir=output_dir / "scene_midpoint",
            max_frames=target_frame_count,
            width=settings.frame_extract_width,
        )
        if midpoint_frames:
            return FrameExtractionArtifact(
                job_id=job_id,
                duration_seconds=video_metadata.duration,
                target_frame_count=target_frame_count,
                strategy="scene_midpoint",
                scene_candidates=scene_result.candidates,
                frames=_deduplicate_frame_artifacts(midpoint_frames),
            )

    scene_frames = _extract_scene_frames(
        video_path=video_path,
        frame_dir=output_dir / "scene",
        max_frames=target_frame_count,
        width=settings.frame_extract_width,
    )
    if scene_frames:
        return FrameExtractionArtifact(
            job_id=job_id,
            duration_seconds=video_metadata.duration,
            target_frame_count=target_frame_count,
            strategy="scene",
            scene_candidates=scene_result.candidates,
            frames=_deduplicate_frame_artifacts(scene_frames),
        )

    uniform_frames = _extract_uniform_frames(
        video_path=video_path,
        frame_dir=output_dir / "uniform",
        max_frames=target_frame_count,
        width=settings.frame_extract_width,
        duration=video_metadata.duration,
    )
    if uniform_frames:
        return FrameExtractionArtifact(
            job_id=job_id,
            duration_seconds=video_metadata.duration,
            target_frame_count=target_frame_count,
            strategy="uniform",
            scene_candidates=scene_result.candidates,
            frames=_deduplicate_frame_artifacts(uniform_frames),
        )

    if planned_frames:
        return FrameExtractionArtifact(
            job_id=job_id,
            duration_seconds=video_metadata.duration,
            target_frame_count=target_frame_count,
            strategy="planned_fallback",
            scene_candidates=scene_result.candidates,
            frames=_deduplicate_frame_artifacts(planned_frames),
        )

    raise RuntimeError(f"Could not extract frames from {video_path}")


def compute_dynamic_frame_count(duration_seconds: float | None, max_frames: int) -> int:
    cap = min(max_frames, 12)
    if duration_seconds is None or duration_seconds <= 0:
        return cap
    if duration_seconds <= 30:
        return min(10, cap)
    if duration_seconds <= 60:
        return min(12, cap)
    if duration_seconds <= 90:
        return min(12, cap)
    return cap


def _extract_planned_frames(
    *,
    video_path: Path,
    frame_dir: Path,
    max_frames: int,
    width: int,
    scene_result: SceneChangeResult,
    duration: float,
) -> list[FrameArtifact]:
    timestamps = _build_planned_timestamps(
        duration=duration,
        max_frames=max_frames,
        scene_timestamps=scene_result.selected_timestamps,
    )
    if not timestamps:
        return []
    return _extract_at_timestamps(
        video_path=video_path,
        frame_dir=frame_dir,
        width=width,
        timestamps=timestamps,
        score_lookup={item.timestamp: item.smoothed_score for item in scene_result.candidates},
    )


def _build_planned_timestamps(*, duration: float, max_frames: int, scene_timestamps: list[float]) -> list[tuple[float, list[str]]]:
    candidates: list[tuple[float, str]] = []
    for timestamp in _sample_anchor_timestamps(duration, max_frames):
        candidates.append((timestamp, "anchor"))
    for timestamp in _sample_safety_timestamps(duration):
        candidates.append((timestamp, "safety"))
    scene_budget = max(1, min(len(scene_timestamps), max_frames // 2 or 1))
    for timestamp in scene_timestamps[:scene_budget]:
        candidates.append((timestamp, "scene_change"))

    merged: list[tuple[float, list[str]]] = []
    for timestamp, reason in sorted(candidates, key=lambda item: item[0]):
        if merged and abs(timestamp - merged[-1][0]) <= 0.35:
            merged[-1][1].append(reason)
            continue
        merged.append((timestamp, [reason]))
    return merged[:max_frames]


def _sample_anchor_timestamps(duration: float, max_frames: int) -> list[float]:
    if duration <= 0 or max_frames <= 0:
        return []
    if max_frames == 1:
        return [round(max(duration / 2.0, 0.0), 4)]
    start = min(0.5, max(duration * 0.05, 0.0))
    end = max(start, duration - 0.5)
    if end <= start:
        return [round(max(duration / 2.0, 0.0), 4)]
    step = (end - start) / max(1, max_frames - 1)
    return [round(start + (step * index), 4) for index in range(max_frames)]


def _sample_safety_timestamps(duration: float) -> list[float]:
    if duration <= 0:
        return []
    return sorted(
        {
            round(min(0.5, max(duration / 2.0, 0.0)), 4),
            round(min(duration, max(0.0, duration * 0.33)), 4),
            round(min(duration, max(0.0, duration * 0.66)), 4),
            round(max(0.0, duration - 0.5), 4),
        }
    )


def _extract_at_timestamps(
    *,
    video_path: Path,
    frame_dir: Path,
    width: int,
    timestamps: list[tuple[float, list[str]]],
    score_lookup: dict[float, float],
) -> list[FrameArtifact]:
    cv2 = _require_cv2()
    frame_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return []
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0:
            return []
        pending = [
            (index, timestamp, reasons, max(0, int(round(timestamp * fps))))
            for index, (timestamp, reasons) in enumerate(timestamps)
        ]
        pending.sort(key=lambda item: item[3])
        artifacts: list[FrameArtifact] = []
        frame_index = 0
        pending_index = 0
        while pending_index < len(pending):
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            while pending_index < len(pending) and frame_index >= pending[pending_index][3]:
                selected_index, timestamp, reasons, _ = pending[pending_index]
                resized = _resize_frame(frame=frame, max_width=width)
                output_path = frame_dir / f"frame_{selected_index + 1:03d}.jpg"
                if not cv2.imwrite(str(output_path), resized, [int(cv2.IMWRITE_JPEG_QUALITY), 88]):
                    break
                artifacts.append(
                    FrameArtifact(
                        frame_id=f"frame_{selected_index + 1:03d}",
                        timestamp=round(timestamp, 4),
                        local_path=str(output_path),
                        selection_reasons=sorted(set(reasons)),
                        scene_change_score=score_lookup.get(round(timestamp, 4), score_lookup.get(timestamp)),
                    )
                )
                pending_index += 1
            frame_index += 1
        return artifacts
    finally:
        capture.release()


def _extract_opencv_frames(*, video_path: Path, frame_dir: Path, max_frames: int, width: int) -> list[FrameArtifact]:
    cv2 = _require_cv2()
    frame_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return []
    try:
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if total_frames <= 0 or fps <= 0 or max_frames <= 0:
            return []
        if max_frames == 1:
            positions = [total_frames // 2]
        else:
            positions = [min(total_frames - 1, round(index * (total_frames - 1) / (max_frames - 1))) for index in range(max_frames)]
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
                    timestamp=round(position / fps, 4),
                    local_path=str(output_path),
                    selection_reasons=["opencv"],
                )
            )
        return artifacts
    finally:
        capture.release()


def _extract_uniform_frames(*, video_path: Path, frame_dir: Path, max_frames: int, width: int, duration: float) -> list[FrameArtifact]:
    frame_dir.mkdir(parents=True, exist_ok=True)
    if duration > 0:
        fps = min(max_frames / duration, 1.0)
    else:
        fps = 0.2
    output_pattern = frame_dir / "frame_%03d.jpg"
    command = [
        settings.ffmpeg_path,
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"fps={fps},scale={width}:-1",
        "-frames:v",
        str(max_frames),
        "-q:v",
        "3",
        str(output_pattern),
    ]
    if not _run_ffmpeg(command):
        return []
    files = sorted(frame_dir.glob("frame_*.jpg"))
    timestamps = _sample_anchor_timestamps(duration, len(files))
    return [
        FrameArtifact(
            frame_id=f"frame_{index:03d}",
            timestamp=timestamps[index - 1] if index - 1 < len(timestamps) else 0.0,
            local_path=str(path),
            selection_reasons=["uniform"],
        )
        for index, path in enumerate(files, start=1)
    ]


def _extract_scene_frames(*, video_path: Path, frame_dir: Path, max_frames: int, width: int) -> list[FrameArtifact]:
    frame_dir.mkdir(parents=True, exist_ok=True)
    output_pattern = frame_dir / "frame_%03d.jpg"
    command = [
        settings.ffmpeg_path,
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"select='gt(scene,0.12)',scale={width}:-1",
        "-fps_mode",
        "vfr",
        "-frames:v",
        str(max_frames),
        "-q:v",
        "3",
        str(output_pattern),
    ]
    if not _run_ffmpeg(command):
        return []
    return _load_frame_artifacts_from_disk(frame_dir=frame_dir, reason="scene")


def _extract_scene_midpoint_frames(*, video_path: Path, frame_dir: Path, max_frames: int, width: int) -> list[FrameArtifact]:
    frame_dir.mkdir(parents=True, exist_ok=True)
    duration = _probe_duration_seconds(video_path)
    if duration <= 0:
        return []
    segments = _detect_scene_segments(video_path=video_path, duration=duration)
    if not segments:
        return []
    if len(segments) > max_frames:
        if max_frames == 1:
            selected_indices = {len(segments) // 2}
        else:
            selected_indices = {round(index * (len(segments) - 1) / (max_frames - 1)) for index in range(max_frames)}
        segments = [segment for index, segment in enumerate(segments) if index in selected_indices]
    artifacts: list[FrameArtifact] = []
    for index, (start, end) in enumerate(segments, start=1):
        midpoint = start + ((end - start) / 2.0)
        output_path = frame_dir / f"frame_{index:03d}.jpg"
        command = [
            settings.ffmpeg_path,
            "-y",
            "-ss",
            f"{midpoint:.3f}",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-vf",
            f"scale={width}:-1",
            "-q:v",
            "2",
            str(output_path),
        ]
        if _run_ffmpeg(command) and output_path.exists():
            artifacts.append(
                FrameArtifact(
                    frame_id=f"frame_{index:03d}",
                    timestamp=round(midpoint, 4),
                    local_path=str(output_path),
                    selection_reasons=["scene_midpoint"],
                )
            )
    return artifacts


def _detect_scene_segments(*, video_path: Path, duration: float, threshold: float = 0.35) -> list[tuple[float, float]]:
    command = [
        settings.ffmpeg_path,
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"select='gt(scene,{threshold})',metadata=print",
        "-f",
        "null",
        "-",
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return [(0.0, duration)]
    combined_output = f"{result.stdout}\n{result.stderr}"
    timestamps: list[float] = []
    for match in re.finditer(r"pts_time:([\d.]+)", combined_output):
        timestamp = float(match.group(1))
        if 0 < timestamp < duration and (not timestamps or timestamp - timestamps[-1] > 1.0):
            timestamps.append(timestamp)
    segments: list[tuple[float, float]] = []
    current_start = 0.0
    for timestamp in timestamps:
        if timestamp > current_start + 0.5:
            segments.append((current_start, timestamp))
            current_start = timestamp
    if duration > current_start:
        segments.append((current_start, duration))
    return segments or [(0.0, duration)]


def _load_frame_artifacts_from_disk(*, frame_dir: Path, reason: str) -> list[FrameArtifact]:
    artifacts: list[FrameArtifact] = []
    for index, path in enumerate(sorted(frame_dir.glob("frame_*.jpg")), start=1):
        artifacts.append(
            FrameArtifact(
                frame_id=f"frame_{index:03d}",
                timestamp=0.0,
                local_path=str(path),
                selection_reasons=[reason],
            )
        )
    return artifacts


def _deduplicate_frame_artifacts(frame_artifacts: list[FrameArtifact]) -> list[FrameArtifact]:
    cv2 = _require_cv2()
    if len(frame_artifacts) <= 2:
        return frame_artifacts
    hashes: list[tuple[FrameArtifact, np.ndarray]] = []
    for artifact in frame_artifacts:
        image = cv2.imread(artifact.local_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return frame_artifacts
        resized = cv2.resize(image, (8, 8), interpolation=cv2.INTER_AREA)
        mean = float(np.mean(resized))
        hashes.append((artifact, resized >= mean))
    kept = [hashes[0][0]]
    last_hash = hashes[0][1]
    for artifact, current_hash in hashes[1:-1]:
        distance = int(np.count_nonzero(last_hash != current_hash))
        if distance >= settings.frame_dedupe_hash_threshold:
            kept.append(artifact)
            last_hash = current_hash
    kept.append(hashes[-1][0])
    return kept


def _resize_frame(*, frame, max_width: int):
    cv2 = _require_cv2()
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame
    target_height = int(round(height * (max_width / width)))
    return cv2.resize(frame, (max_width, target_height), interpolation=cv2.INTER_AREA)


def _probe_duration_seconds(video_path: Path) -> float:
    command = [
        settings.ffprobe_path,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        return float(result.stdout.strip() or 0.0)
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
        return 0.0


def _run_ffmpeg(command: list[str]) -> bool:
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        logger.debug("ffmpeg command failed", exc_info=True)
        return False


def _require_cv2():
    try:
        import cv2
    except ModuleNotFoundError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("opencv-python-headless is required for frame extraction.") from exc
    return cv2
