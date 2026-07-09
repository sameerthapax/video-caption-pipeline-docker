from __future__ import annotations

import json
import shutil
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path

from worker.config.settings import settings


class ProcessExecutionError(RuntimeError):
    pass


_ffmpeg_semaphore = threading.BoundedSemaphore(value=max(1, settings.ffmpeg_max_concurrency))


@dataclass(frozen=True)
class CompletedProcessResult:
    stdout: str
    stderr: str


def ensure_binary_exists(binary_path: str) -> None:
    if shutil.which(binary_path) is None:
        raise ProcessExecutionError(f"Required binary is not available: {binary_path}")


def run_command(*, args: list[str], timeout_seconds: int) -> CompletedProcessResult:
    ensure_binary_exists(args[0])
    try:
        semaphore = _ffmpeg_semaphore if _uses_ffmpeg(args[0]) else None
        if semaphore is None:
            completed = subprocess.run(
                args,
                check=True,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        else:
            with semaphore:
                completed = subprocess.run(
                    args,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds,
                )
    except FileNotFoundError as exc:
        raise ProcessExecutionError(f"Required binary is not available: {args[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise ProcessExecutionError(f"Command timed out after {timeout_seconds}s: {args[0]}") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() if exc.stderr else ""
        stdout = exc.stdout.strip() if exc.stdout else ""
        detail = stderr or stdout or f"Command failed: {args[0]}"
        raise ProcessExecutionError(detail) from exc

    return CompletedProcessResult(stdout=completed.stdout, stderr=completed.stderr)


def _uses_ffmpeg(binary_path: str) -> bool:
    binary_name = Path(binary_path).name.lower()
    return binary_name == "ffmpeg"


@dataclass(frozen=True)
class ProbeMetadata:
    format_name: str
    duration_seconds: float
    file_size_bytes: int
    width: int | None
    height: int | None
    fps: float | None
    video_codec: str | None
    audio_codec: str | None
    has_audio: bool


def probe_media(path: Path) -> ProbeMetadata:
    result = run_command(
        args=[
            settings.ffprobe_path,
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ],
        timeout_seconds=settings.ffprobe_timeout_seconds,
    )
    return parse_ffprobe_output(result.stdout)


def parse_ffprobe_output(payload: str) -> ProbeMetadata:
    data = json.loads(payload)
    format_payload = data.get("format") or {}
    streams = data.get("streams") or []

    video_stream = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
    audio_stream = next((stream for stream in streams if stream.get("codec_type") == "audio"), None)

    duration_raw = format_payload.get("duration") or (video_stream or {}).get("duration") or 0
    size_raw = format_payload.get("size") or 0
    fps = _parse_frame_rate((video_stream or {}).get("avg_frame_rate"))

    return ProbeMetadata(
        format_name=str(format_payload.get("format_name") or ""),
        duration_seconds=float(duration_raw or 0),
        file_size_bytes=int(size_raw or 0),
        width=_safe_int((video_stream or {}).get("width")),
        height=_safe_int((video_stream or {}).get("height")),
        fps=fps,
        video_codec=(video_stream or {}).get("codec_name"),
        audio_codec=(audio_stream or {}).get("codec_name"),
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


def _safe_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
