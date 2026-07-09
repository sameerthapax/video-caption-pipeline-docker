from __future__ import annotations

from pathlib import Path

from services.process import run_command
from worker.config.settings import settings


def normalize_video(*, input_video_path: Path, output_video_path: Path) -> Path:
    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        args=[
            settings.ffmpeg_path,
            "-y",
            "-i",
            str(input_video_path),
            "-vf",
            "scale='min(1280,iw)':-2,fps=30",
            "-c:v",
            "libx264",
            "-r",
            "30",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "veryfast",
            "-movflags",
            "+faststart",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            str(output_video_path),
        ],
        timeout_seconds=settings.ffmpeg_timeout_seconds,
    )
    return output_video_path
