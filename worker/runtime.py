from __future__ import annotations

import logging
import shutil
import time
from contextlib import contextmanager
from pathlib import Path

from worker.config.settings import settings


def configure_logging() -> Path:
    logs_dir = settings.debug_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "pipeline.log"

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")

    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    return log_path


@contextmanager
def log_stage(logger: logging.Logger, stage_name: str):
    started_at = time.perf_counter()
    logger.info("Stage started: %s", stage_name)
    try:
        yield
    except Exception:
        elapsed = time.perf_counter() - started_at
        logger.exception("Stage failed: %s (%.2fs)", stage_name, elapsed)
        raise
    elapsed = time.perf_counter() - started_at
    logger.info("Stage completed: %s (%.2fs)", stage_name, elapsed)


def ensure_debug_directories(task_id: str) -> dict[str, Path]:
    base = settings.debug_root
    directories = {
        "original": base / "original" / task_id,
        "normalized": base / "normalized" / task_id,
        "audio": base / "audio" / task_id,
        "frames": base / "frames" / task_id,
        "json": base / "json" / task_id,
        "logs": base / "logs",
    }
    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)
    return directories


def persist_debug_artifacts(
    *,
    task_id: str,
    original_video_path: Path,
    normalized_video_path: Path,
    normalized_audio_path: Path | None,
    artifacts_root: Path,
) -> None:
    directories = ensure_debug_directories(task_id)
    if original_video_path.exists():
        shutil.copy2(original_video_path, directories["original"] / "original.mp4")
    if normalized_video_path.exists():
        shutil.copy2(normalized_video_path, directories["normalized"] / "normalized.mp4")
    if normalized_audio_path is not None and normalized_audio_path.exists():
        shutil.copy2(normalized_audio_path, directories["audio"] / "normalized.wav")

    frames_source = artifacts_root / "frames"
    if frames_source.exists():
        shutil.copytree(frames_source, directories["frames"], dirs_exist_ok=True)

    json_filenames = {
        "frame_sampling.json",
        "temporal_segments.json",
        "transcription_request.json",
        "vlm_segments.json",
        "video_memory.json",
        "global_factual_summary.json",
        "final_result.json",
    }
    for filename in json_filenames:
        source = artifacts_root / filename
        if source.exists():
            target_name = "transcript.json" if filename == "transcription_request.json" else filename
            shutil.copy2(source, directories["json"] / target_name)
