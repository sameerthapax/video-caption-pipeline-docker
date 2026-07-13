from __future__ import annotations

import logging
import shutil
from pathlib import Path

from services.process import ProcessExecutionError, run_command
from worker.config.settings import settings

logger = logging.getLogger("gemma-caption-pipe.worker")


def load_or_create_transcript(
    *,
    task_id: str,
    provided_transcript_text: str,
    transcript_source_path: Path | None,
    audio_path: Path | None,
    transcript_dir: Path,
) -> str:
    if provided_transcript_text.strip():
        return provided_transcript_text.strip()
    if transcript_source_path is not None and transcript_source_path.exists():
        return transcript_source_path.read_text(encoding="utf-8").strip()
    if not settings.enable_local_whisper or audio_path is None or not audio_path.exists():
        return ""
    transcript_dir.mkdir(parents=True, exist_ok=True)
    transcript_path = transcript_dir / f"{task_id}.txt"
    if transcript_path.exists() and not settings.force_transcription:
        return transcript_path.read_text(encoding="utf-8").strip()
    if shutil.which(settings.whisper_command) is None:
        logger.warning("Whisper command %s is not available; skipping local transcription.", settings.whisper_command)
        return ""
    command = [
        settings.whisper_command,
        str(audio_path),
        "--model",
        settings.whisper_model,
        "--output_dir",
        str(transcript_dir),
        "--output_format",
        "txt",
    ]
    if settings.whisper_language:
        command.extend(["--language", settings.whisper_language])
    try:
        run_command(args=command, timeout_seconds=settings.ffmpeg_timeout_seconds)
    except ProcessExecutionError:
        logger.warning("Local Whisper transcription failed for %s.", task_id, exc_info=True)
        return ""
    if transcript_path.exists():
        return transcript_path.read_text(encoding="utf-8").strip()
    return ""
