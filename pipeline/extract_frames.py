from __future__ import annotations

from pathlib import Path

import cv2

from schemas.frames import FrameArtifact, VlmFramePlaceholder


def build_frame_filename(*, frame_index: int, timestamp: float) -> str:
    return f"frame_{frame_index:02d}_{timestamp:.2f}s.jpg"


def extract_selected_frames(
    *,
    job_id: str,
    video_path: Path,
    output_dir: Path,
    storage_prefix: str,
    timestamps_with_reasons: list[tuple[float, list[str], float | None]],
    max_width: int = 640,
) -> list[FrameArtifact]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video for frame extraction: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[FrameArtifact] = []
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0:
            raise RuntimeError(f"Unable to determine FPS for frame extraction: {video_path}")

        pending = [
            (index, timestamp, reasons, scene_score, max(0, int(round(timestamp * fps))))
            for index, (timestamp, reasons, scene_score) in enumerate(timestamps_with_reasons)
        ]
        pending.sort(key=lambda item: item[4])

        frame_index = 0
        pending_index = 0
        while pending_index < len(pending):
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError(f"Failed to extract frame near {pending[pending_index][1]:.2f}s")

            while pending_index < len(pending) and frame_index >= pending[pending_index][4]:
                selected_index, timestamp, reasons, scene_score, _target_frame_index = pending[pending_index]
                resized = _resize_frame(frame=frame, max_width=max_width)
                filename = build_frame_filename(frame_index=selected_index, timestamp=timestamp)
                local_path = output_dir / filename
                if not cv2.imwrite(str(local_path), resized):
                    raise RuntimeError(f"Failed to write extracted frame to {local_path}")

                artifacts.append(
                    FrameArtifact(
                        frame_id=f"frame_{selected_index:02d}",
                        timestamp=round(timestamp, 4),
                        storage_path=f"{storage_prefix}/{filename}",
                        local_path=str(local_path),
                        selection_reasons=sorted(reasons),
                        scene_change_score=scene_score,
                        vlm=VlmFramePlaceholder(),
                    )
                )
                pending_index += 1

            frame_index += 1
    finally:
        capture.release()
    artifacts.sort(key=lambda item: item.timestamp)
    return artifacts


def _resize_frame(*, frame, max_width: int):
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame
    target_height = int(round(height * (max_width / width)))
    return cv2.resize(frame, (max_width, target_height), interpolation=cv2.INTER_AREA)
