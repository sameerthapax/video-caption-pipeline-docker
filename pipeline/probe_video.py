from __future__ import annotations

import json
from pathlib import Path

from schemas.video import VideoMetadata
from services.process import ProcessExecutionError, run_command
from worker.config.settings import settings


def probe_video_metadata(video_path: Path) -> VideoMetadata:
    result = run_command(
        args=[
            settings.ffprobe_path,
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(video_path),
        ],
        timeout_seconds=settings.ffprobe_timeout_seconds,
    )
    payload = json.loads(result.stdout)
    format_data = payload.get("format") or {}
    streams = payload.get("streams") or []
    video_stream = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
    audio_stream = next((stream for stream in streams if stream.get("codec_type") == "audio"), None)
    if video_stream is None:
        raise ProcessExecutionError("ffprobe did not return a video stream.")

    duration = _to_float(format_data.get("duration")) or _to_float(video_stream.get("duration")) or 0.0
    fps = _parse_frame_rate(video_stream.get("avg_frame_rate"))
    frame_count = _to_int(video_stream.get("nb_frames"))
    if frame_count is None and fps is not None and duration > 0:
        frame_count = int(round(duration * fps))

    return VideoMetadata(
        duration=duration,
        fps=fps,
        width=_to_int(video_stream.get("width")),
        height=_to_int(video_stream.get("height")),
        frame_count=frame_count or 0,
        has_audio=audio_stream is not None,
    )


def _parse_frame_rate(value: str | None) -> float | None:
    if not value or value == "0/0":
        return None
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        denominator_value = float(denominator)
        if denominator_value == 0:
            return None
        return float(numerator) / denominator_value
    return float(value)


def _to_float(value: object) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _to_int(value: object) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None
