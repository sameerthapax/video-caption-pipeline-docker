from __future__ import annotations

import mimetypes
from pathlib import Path
from urllib.parse import urlparse

import httpx


class DownloadError(RuntimeError):
    pass


def download_file(*, url: str, destination: Path, timeout_seconds: int = 300) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with httpx.Client(timeout=timeout_seconds, follow_redirects=True) as client:
            with client.stream("GET", url) as response:
                response.raise_for_status()
                with destination.open("wb") as handle:
                    for chunk in response.iter_bytes():
                        handle.write(chunk)
    except httpx.HTTPError as exc:
        raise DownloadError(f"Failed to download {url}: {exc}") from exc
    return destination


def guess_video_suffix(url: str) -> str:
    parsed = urlparse(url)
    suffix = Path(parsed.path).suffix.lower()
    if suffix:
        return suffix
    mime_type, _encoding = mimetypes.guess_type(url)
    if mime_type == "video/quicktime":
        return ".mov"
    if mime_type == "video/webm":
        return ".webm"
    if mime_type == "video/x-matroska":
        return ".mkv"
    return ".mp4"
