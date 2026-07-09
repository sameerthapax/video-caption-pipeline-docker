from __future__ import annotations

from schemas.frames import FrameArtifact
from schemas.segments import SamplingConfig, TemporalSegment, TemporalSegmentsArtifact
from schemas.transcription import TranscriptChunk
from schemas.video import VideoMetadata


def build_segment_boundaries(duration: float, segment_count: int = 5) -> list[tuple[float, float, str]]:
    if duration <= 0 or segment_count <= 0:
        return []
    boundaries: list[tuple[float, float, str]] = []
    for index in range(segment_count):
        start = round(duration * (index / segment_count), 4)
        end = round(duration if index == segment_count - 1 else duration * ((index + 1) / segment_count), 4)
        boundaries.append((start, end, f"{index * 20}-{(index + 1) * 20}"))
    return boundaries


def assign_frames_to_segments(*, frames: list[FrameArtifact], duration: float, segment_count: int = 5) -> list[list[FrameArtifact]]:
    buckets = [[] for _ in range(segment_count)]
    boundaries = build_segment_boundaries(duration, segment_count)
    for frame in frames:
        for index, (start, end, _label) in enumerate(boundaries):
            is_last = index == segment_count - 1
            if start <= frame.timestamp < end or (is_last and frame.timestamp <= end):
                buckets[index].append(frame)
                break
    return buckets


def assign_transcript_chunks_to_segments(
    *,
    transcript_chunks: list[TranscriptChunk],
    duration: float,
    segment_count: int = 5,
) -> list[list[TranscriptChunk]]:
    buckets = [[] for _ in range(segment_count)]
    boundaries = build_segment_boundaries(duration, segment_count)
    for chunk in transcript_chunks:
        for index, (start, end, _label) in enumerate(boundaries):
            if chunk.end > start and chunk.start < end:
                buckets[index].append(chunk)
    return buckets


def build_temporal_segments_artifact(
    *,
    job_id: str,
    video_metadata: VideoMetadata,
    sampling_config: SamplingConfig,
    frames: list[FrameArtifact],
    transcript_chunks: list[TranscriptChunk],
) -> TemporalSegmentsArtifact:
    boundaries = build_segment_boundaries(video_metadata.duration, 5)
    frame_buckets = assign_frames_to_segments(frames=frames, duration=video_metadata.duration, segment_count=5)
    transcript_buckets = assign_transcript_chunks_to_segments(
        transcript_chunks=transcript_chunks,
        duration=video_metadata.duration,
        segment_count=5,
    )
    segments = [
        TemporalSegment(
            segment_index=index,
            start=start,
            end=end,
            percent_range=percent_range,
            frames=frame_buckets[index],
            transcript_chunks=transcript_buckets[index],
        )
        for index, (start, end, percent_range) in enumerate(boundaries)
    ]
    return TemporalSegmentsArtifact(
        job_id=job_id,
        video_metadata=video_metadata,
        sampling_config=sampling_config,
        segments=segments,
    )

