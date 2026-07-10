from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from schemas.frames import SceneCandidateScore

HISTOGRAM_BINS = (8, 8, 8)
GRAYSCALE_SIZE = (64, 64)


@dataclass(frozen=True)
class SceneScoreWeights:
    embedding_change: float = 0.6
    histogram_change: float = 0.3
    pixel_change: float = 0.1


@dataclass(frozen=True)
class SceneSamplingConfig:
    scan_interval_seconds: float = 1.0
    min_scene_spacing_seconds: float = 4.0
    max_selected: int = 8


@dataclass(frozen=True)
class SceneChangeResult:
    candidates: list[SceneCandidateScore]
    selected_timestamps: list[float]
    embedding_available: bool
    fallback_reason: str | None


def analyze_scene_changes(
    *,
    video_path: Path,
    duration: float,
    config: SceneSamplingConfig | None = None,
    weights: SceneScoreWeights | None = None,
) -> SceneChangeResult:
    cv2 = _require_cv2()
    sampling = config or SceneSamplingConfig()
    _ = weights or SceneScoreWeights()

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video for scene scanning: {video_path}")

    timestamps = _build_sample_timestamps(duration=duration, interval_seconds=sampling.scan_interval_seconds)
    previous_histogram: np.ndarray | None = None
    previous_gray: np.ndarray | None = None
    raw_hist: list[float] = []
    raw_pixel: list[float] = []

    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0:
            raise RuntimeError(f"Unable to determine FPS for scene scanning: {video_path}")

        target_frames = [max(0, int(round(timestamp * fps))) for timestamp in timestamps]
        frame_index = 0
        target_index = 0
        while target_index < len(target_frames):
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError(f"Failed to read frame near {timestamps[target_index]:.2f}s")
            if frame_index < target_frames[target_index]:
                frame_index += 1
                continue

            histogram = _compute_histogram(frame)
            gray = _compute_grayscale(frame)
            if previous_histogram is None or previous_gray is None:
                raw_hist.append(0.0)
                raw_pixel.append(0.0)
            else:
                raw_hist.append(max(0.0, float(cv2.compareHist(previous_histogram, histogram, cv2.HISTCMP_BHATTACHARYYA))))
                raw_pixel.append(float(np.mean(np.abs(gray.astype(np.float32) - previous_gray.astype(np.float32))) / 255.0))
            previous_histogram = histogram
            previous_gray = gray
            target_index += 1
            frame_index += 1
    finally:
        capture.release()

    raw_scores = _combine_fallback_scores(hist_scores=raw_hist, pixel_scores=raw_pixel)
    normalized_scores = _normalize_scores(raw_scores)
    smoothed_scores = _smooth_scores(normalized_scores)
    selected_indices = select_scene_change_indices(
        timestamps=timestamps,
        scores=smoothed_scores,
        min_spacing_seconds=sampling.min_scene_spacing_seconds,
        max_selected=sampling.max_selected,
    )
    selected_index_set = set(selected_indices)

    candidates = [
        SceneCandidateScore(
            timestamp=round(timestamps[index], 4),
            embedding_change=None,
            histogram_change=round(raw_hist[index], 6),
            pixel_change=round(raw_pixel[index], 6),
            raw_score=round(raw_scores[index], 6),
            normalized_score=round(normalized_scores[index], 6),
            smoothed_score=round(smoothed_scores[index], 6),
            selected=index in selected_index_set,
        )
        for index in range(len(timestamps))
    ]

    return SceneChangeResult(
        candidates=candidates,
        selected_timestamps=[round(timestamps[index], 4) for index in selected_indices],
        embedding_available=False,
        fallback_reason="TODO: add local CLIP/SigLIP embeddings or Fireworks embedding support for scene scoring.",
    )


def select_scene_change_indices(
    *,
    timestamps: list[float],
    scores: list[float],
    min_spacing_seconds: float,
    max_selected: int,
) -> list[int]:
    peaks = _find_local_peaks(scores)
    ranked = sorted(peaks, key=lambda index: scores[index], reverse=True)
    selected: list[int] = []
    for index in ranked:
        if all(abs(timestamps[index] - timestamps[chosen]) >= min_spacing_seconds for chosen in selected):
            selected.append(index)
        if len(selected) >= max_selected:
            break
    return sorted(selected, key=lambda index: timestamps[index])


def _build_sample_timestamps(*, duration: float, interval_seconds: float) -> list[float]:
    if duration <= 0:
        return []
    timestamps: list[float] = []
    current = min(0.5, max(duration - 0.001, 0.0))
    while current < duration:
        timestamps.append(round(current, 4))
        current += interval_seconds
    if not timestamps:
        timestamps.append(round(max(0.0, duration / 2.0), 4))
    return timestamps


def _compute_histogram(frame: np.ndarray) -> np.ndarray:
    cv2 = _require_cv2()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist([hsv], [0, 1, 2], None, HISTOGRAM_BINS, [0, 180, 0, 256, 0, 256])
    cv2.normalize(histogram, histogram)
    return histogram


def _compute_grayscale(frame: np.ndarray) -> np.ndarray:
    cv2 = _require_cv2()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, GRAYSCALE_SIZE, interpolation=cv2.INTER_AREA)


def _combine_fallback_scores(*, hist_scores: list[float], pixel_scores: list[float]) -> list[float]:
    return [
        (0.7 * hist_scores[index]) + (0.3 * pixel_scores[index])
        for index in range(len(hist_scores))
    ]


def _normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []
    max_score = max(scores)
    min_score = min(scores)
    if max_score == min_score:
        return [0.0 for _ in scores]
    return [(score - min_score) / (max_score - min_score) for score in scores]


def _smooth_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []
    if len(scores) < 3:
        return scores[:]
    padded = np.pad(np.asarray(scores, dtype=np.float32), (1, 1), mode="edge")
    kernel = np.asarray([0.25, 0.5, 0.25], dtype=np.float32)
    smoothed = np.convolve(padded, kernel, mode="valid")
    return [float(value) for value in smoothed]


def _find_local_peaks(scores: list[float]) -> list[int]:
    if not scores:
        return []
    peaks: list[int] = []
    for index, score in enumerate(scores):
        previous_score = scores[index - 1] if index > 0 else float("-inf")
        next_score = scores[index + 1] if index < len(scores) - 1 else float("-inf")
        if score > 0 and score >= previous_score and score >= next_score:
            peaks.append(index)
    return peaks


def _require_cv2():
    try:
        import cv2
    except ModuleNotFoundError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("opencv-python-headless is required for scene analysis.") from exc
    return cv2
