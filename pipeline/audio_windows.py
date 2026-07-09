from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from schemas.transcription import TranscriptChunk
from services.process import run_command
from worker.config.settings import settings


@dataclass(frozen=True)
class AudioWindowFile:
    start: float
    end: float
    path: Path
    mime_type: str = "audio/wav"


def build_transcript_windows(duration: float, window_seconds: float = 5.0) -> list[TranscriptChunk]:
    if duration <= 0:
        return []
    window_count = max(1, int(math.ceil(duration / window_seconds)))
    windows: list[TranscriptChunk] = []
    for index in range(window_count):
        start = round(index * window_seconds, 4)
        end = round(min(duration, start + window_seconds), 4)
        windows.append(TranscriptChunk(start=start, end=end))
    return windows


def extract_audio_window_files(
    *,
    source_audio_path: Path,
    output_dir: Path,
    transcript_windows: list[TranscriptChunk],
) -> list[AudioWindowFile]:
    if not transcript_windows:
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    output_pattern = output_dir / "window_%03d.wav"
    run_command(
        args=[
            settings.ffmpeg_path,
            "-y",
            "-i",
            str(source_audio_path),
            "-f",
            "segment",
            "-segment_time",
            "5",
            "-ac",
            "1",
            "-ar",
            "24000",
            "-c:a",
            "pcm_s16le",
            str(output_pattern),
        ],
        timeout_seconds=settings.ffmpeg_timeout_seconds,
    )
    output_files = sorted(output_dir.glob("window_*.wav"))
    if len(output_files) < len(transcript_windows):
        raise RuntimeError(
            f"Expected at least {len(transcript_windows)} transcript audio windows, found {len(output_files)}."
        )
    return [
        AudioWindowFile(start=window.start, end=window.end, path=output_files[index])
        for index, window in enumerate(transcript_windows)
    ]
