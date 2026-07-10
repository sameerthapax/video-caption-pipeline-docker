from __future__ import annotations

import io
import math
import wave
from dataclasses import dataclass
from pathlib import Path

from schemas.transcription import TranscriptChunk


@dataclass(frozen=True)
class AudioWindowChunk:
    start: float
    end: float
    audio_bytes: bytes
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


def extract_audio_window_chunks(
    *,
    source_audio_path: Path,
    transcript_windows: list[TranscriptChunk],
) -> list[AudioWindowChunk]:
    if not transcript_windows:
        return []
    with wave.open(str(source_audio_path), "rb") as source_wav:
        frame_rate = source_wav.getframerate()
        sample_width = source_wav.getsampwidth()
        channel_count = source_wav.getnchannels()
        frame_count = source_wav.getnframes()
        source_frames = source_wav.readframes(frame_count)

    chunks: list[AudioWindowChunk] = []
    bytes_per_frame = sample_width * channel_count
    for window in transcript_windows:
        start_frame = min(frame_count, max(0, int(window.start * frame_rate)))
        end_frame = min(frame_count, max(start_frame, int(math.ceil(window.end * frame_rate))))
        start_byte = start_frame * bytes_per_frame
        end_byte = end_frame * bytes_per_frame

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as window_wav:
            window_wav.setnchannels(channel_count)
            window_wav.setsampwidth(sample_width)
            window_wav.setframerate(frame_rate)
            window_wav.writeframes(source_frames[start_byte:end_byte])
        chunks.append(
            AudioWindowChunk(
                start=window.start,
                end=window.end,
                audio_bytes=buffer.getvalue(),
            )
        )
    return chunks
