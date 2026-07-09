from __future__ import annotations

from dataclasses import dataclass, field

from schemas.frames import DedupeDecision, FinalFrameSelection, FrameSamplingArtifact
from pipeline.scene_change import SceneChangeResult
from schemas.segments import SamplingConfig
from schemas.video import VideoMetadata


@dataclass
class TimestampCandidate:
    timestamp: float
    reason: str
    scene_change_score: float | None = None


@dataclass
class MergedTimestamp:
    timestamp: float
    reasons: set[str] = field(default_factory=set)
    scene_change_score: float | None = None


def generate_uniform_timestamps(duration: float, count: int) -> list[float]:
    if duration <= 0 or count <= 0:
        return []
    start = min(0.5, max(duration / 2.0, 0.0))
    end = max(start, duration - 0.5)
    if count == 1 or start == end:
        return [round(start, 4)]
    step = (end - start) / (count - 1)
    return [round(start + (step * index), 4) for index in range(count)]


def generate_safety_timestamps(duration: float) -> list[float]:
    if duration <= 0:
        return []
    return [
        round(min(0.5, max(duration / 2.0, 0.0)), 4),
        round(min(duration, max(0.0, duration * 0.33)), 4),
        round(min(duration, max(0.0, duration * 0.66)), 4),
        round(max(0.0, duration - 0.5), 4),
    ]


def build_frame_sampling_artifact(
    *,
    job_id: str,
    video_metadata: VideoMetadata,
    scene_change_result: SceneChangeResult,
    sampling_config: SamplingConfig,
) -> FrameSamplingArtifact:
    target_frame_count = (
        sampling_config.uniform_count
        + sampling_config.scene_change_count
        + sampling_config.safety_count
    )
    candidates: list[TimestampCandidate] = []
    candidates.extend(
        TimestampCandidate(timestamp=value, reason="uniform")
        for value in generate_uniform_timestamps(video_metadata.duration, sampling_config.uniform_count)
    )
    candidates.extend(TimestampCandidate(timestamp=value, reason="safety") for value in generate_safety_timestamps(video_metadata.duration))
    score_lookup = {item.timestamp: item.smoothed_score for item in scene_change_result.candidates}
    selected_scene_timestamps = scene_change_result.selected_timestamps[: sampling_config.scene_change_count]
    candidates.extend(
        TimestampCandidate(timestamp=value, reason="scene_change", scene_change_score=score_lookup.get(value))
        for value in selected_scene_timestamps
    )
    rejected_scene_candidates = [
        TimestampCandidate(
            timestamp=item.timestamp,
            reason="scene_change_replacement",
            scene_change_score=item.smoothed_score,
        )
        for item in scene_change_result.candidates
        if item.timestamp not in set(selected_scene_timestamps)
    ]
    merged, dedupe_decisions = dedupe_timestamp_candidates(
        candidates=candidates,
        threshold_seconds=sampling_config.dedupe_timestamp_threshold_seconds,
        replacement_candidates=rejected_scene_candidates,
        target_count=target_frame_count,
    )

    return FrameSamplingArtifact(
        job_id=job_id,
        scene_scan_interval_seconds=sampling_config.scene_scan_interval_seconds,
        embedding_available=scene_change_result.embedding_available,
        fallback_reason=scene_change_result.fallback_reason,
        all_candidate_scene_change_frames=scene_change_result.candidates,
        selected_top_scene_change_frames=selected_scene_timestamps,
        final_selected_frames=[
            FinalFrameSelection(
                timestamp=round(item.timestamp, 4),
                selection_reasons=sorted(item.reasons),
                scene_change_score=item.scene_change_score,
            )
            for item in merged
        ],
        deduplication_decisions=dedupe_decisions,
    )


def dedupe_timestamp_candidates(
    *,
    candidates: list[TimestampCandidate],
    threshold_seconds: float,
    replacement_candidates: list[TimestampCandidate] | None = None,
    target_count: int | None = None,
) -> tuple[list[MergedTimestamp], list[DedupeDecision]]:
    sorted_candidates = sorted(candidates, key=lambda item: item.timestamp)
    merged: list[MergedTimestamp] = []
    decisions: list[DedupeDecision] = []
    dropped_candidates: list[TimestampCandidate] = []
    for candidate in sorted_candidates:
        if not merged:
            merged.append(
                MergedTimestamp(
                    timestamp=candidate.timestamp,
                    reasons={candidate.reason},
                    scene_change_score=candidate.scene_change_score,
                )
            )
            continue
        current = merged[-1]
        if abs(candidate.timestamp - current.timestamp) <= threshold_seconds:
            current.reasons.add(candidate.reason)
            current.scene_change_score = _max_score(current.scene_change_score, candidate.scene_change_score)
            dropped_candidates.append(candidate)
            decisions.append(
                DedupeDecision(
                    kept_timestamp=round(current.timestamp, 4),
                    dropped_timestamp=round(candidate.timestamp, 4),
                    reason="timestamp_within_threshold",
                )
            )
            continue
        merged.append(
            MergedTimestamp(
                timestamp=candidate.timestamp,
                reasons={candidate.reason},
                scene_change_score=candidate.scene_change_score,
            )
        )

    desired_count = target_count or len(candidates)
    replacement_pool = replacement_candidates or []
    used_replacement_timestamps: set[float] = set()

    for dropped_candidate in dropped_candidates:
        if len(merged) >= desired_count:
            break
        replacement = _find_best_replacement(
            dropped_candidate=dropped_candidate,
            merged=merged,
            replacement_pool=replacement_pool,
            threshold_seconds=threshold_seconds,
            used_timestamps=used_replacement_timestamps,
        )
        if replacement is None:
            continue
        used_replacement_timestamps.add(replacement.timestamp)
        merged.append(
            MergedTimestamp(
                timestamp=replacement.timestamp,
                reasons={replacement.reason},
                scene_change_score=replacement.scene_change_score,
            )
        )
        decisions.append(
            DedupeDecision(
                kept_timestamp=round(dropped_candidate.timestamp, 4),
                dropped_timestamp=round(dropped_candidate.timestamp, 4),
                replacement_timestamp=round(replacement.timestamp, 4),
                reason="replaced_with_rejected_scene_change",
            )
        )

    while len(merged) < desired_count:
        replacement = _find_best_replacement(
            dropped_candidate=None,
            merged=merged,
            replacement_pool=replacement_pool,
            threshold_seconds=threshold_seconds,
            used_timestamps=used_replacement_timestamps,
        )
        if replacement is None:
            break
        used_replacement_timestamps.add(replacement.timestamp)
        merged.append(
            MergedTimestamp(
                timestamp=replacement.timestamp,
                reasons={replacement.reason},
                scene_change_score=replacement.scene_change_score,
            )
        )
        decisions.append(
            DedupeDecision(
                kept_timestamp=round(replacement.timestamp, 4),
                dropped_timestamp=round(replacement.timestamp, 4),
                replacement_timestamp=round(replacement.timestamp, 4),
                reason="filled_from_rejected_scene_change_pool",
            )
        )

    merged.sort(key=lambda item: item.timestamp)
    return merged, decisions


def _find_best_replacement(
    *,
    dropped_candidate: TimestampCandidate | None,
    merged: list[MergedTimestamp],
    replacement_pool: list[TimestampCandidate],
    threshold_seconds: float,
    used_timestamps: set[float],
) -> TimestampCandidate | None:
    valid_candidates = [
        candidate
        for candidate in replacement_pool
        if candidate.timestamp not in used_timestamps
        and _is_timestamp_valid(candidate.timestamp, merged, threshold_seconds)
    ]
    if not valid_candidates:
        return None
    if dropped_candidate is not None:
        valid_candidates.sort(
            key=lambda candidate: (
                abs(candidate.timestamp - dropped_candidate.timestamp),
                -(candidate.scene_change_score or 0.0),
            )
        )
        return valid_candidates[0]
    valid_candidates.sort(
        key=lambda candidate: (-(candidate.scene_change_score or 0.0), candidate.timestamp)
    )
    return valid_candidates[0]


def _is_timestamp_valid(timestamp: float, merged: list[MergedTimestamp], threshold_seconds: float) -> bool:
    return all(abs(timestamp - item.timestamp) > threshold_seconds for item in merged)


def _max_score(left: float | None, right: float | None) -> float | None:
    if left is None:
        return right
    if right is None:
        return left
    return max(left, right)
