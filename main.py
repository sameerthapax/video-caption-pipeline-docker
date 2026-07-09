from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
import shutil
from pathlib import Path

from pipeline.normalize import normalize_video
from pipeline.probe_video import probe_video_metadata
from pipeline.run_extraction_stage import run_video_extraction_stage
from pipeline.run_vlm_stage import run_vlm_reasoning_stage
from pipeline.style import STYLE_ORDER, style_captions
from schemas.tasks import CaptionTask, TaskResult
from services.local_files import download_file, guess_video_suffix
from services.openai_responses_client import OpenAIResponsesClient, OpenAIResponsesConfig
from services.process import ProcessExecutionError, run_command
from worker.config.settings import settings
from worker.runtime import configure_logging, log_stage, persist_debug_artifacts

logger = logging.getLogger("video-caption-hackathon-agent")


def main() -> None:
    log_path = configure_logging()
    logger.info("Pipeline logging initialized at %s", log_path)
    run()


def run() -> None:
    _ensure_required_settings()
    tasks_path = Path(settings.input_tasks_path)
    output_path = Path(settings.output_results_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tasks = _load_tasks(tasks_path)
    results = _process_tasks(tasks)

    output_payload = [result.model_dump(mode="json") for result in results]
    output_path.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")
    logger.info("Wrote %s task results to %s", len(results), output_path)


def _ensure_required_settings() -> None:
    missing: list[str] = []
    if not settings.fireworks_api_key:
        missing.append("FIREWORKS_API_KEY")
    if not settings.fireworks_model:
        missing.append("FIREWORKS_MODEL")
    if not settings.openai_api_key:
        missing.append("OPENAI_API_KEY")
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")


def _load_tasks(tasks_path: Path) -> list[CaptionTask]:
    if not tasks_path.exists():
        raise FileNotFoundError(f"Input task file does not exist: {tasks_path}")
    payload = json.loads(tasks_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Input task file must contain a JSON array.")
    tasks = [CaptionTask.model_validate(item) for item in payload]
    if not tasks:
        return []
    return tasks


def _process_tasks(tasks: list[CaptionTask]) -> list[TaskResult]:
    if not tasks:
        return []

    max_workers = max(1, settings.max_concurrent_jobs)
    logger.info("Processing %s task(s) with max_concurrent_jobs=%s", len(tasks), max_workers)

    if max_workers == 1 or len(tasks) == 1:
        return [_process_task(task) for task in tasks]

    ordered_results: list[TaskResult | None] = [None] * len(tasks)
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="caption-task") as executor:
        future_to_index = {
            executor.submit(_process_task, task): index
            for index, task in enumerate(tasks)
        }
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            ordered_results[index] = future.result()

    return [result for result in ordered_results if result is not None]


def _process_task(task: CaptionTask) -> TaskResult:
    supported_styles = set(STYLE_ORDER)
    requested_styles: list[str] = []
    for style_name in task.styles:
        if style_name not in supported_styles:
            raise ValueError(f"Unsupported style '{style_name}' for task {task.task_id}")
        if style_name not in requested_styles:
            requested_styles.append(style_name)
    if not requested_styles:
        requested_styles = list(STYLE_ORDER)

    task_root = Path(settings.worker_tmp_root) / task.task_id
    if task_root.exists():
        shutil.rmtree(task_root, ignore_errors=True)
    task_root.mkdir(parents=True, exist_ok=True)

    original_video_path = task_root / f"source{guess_video_suffix(str(task.video_url))}"
    normalized_video_path = task_root / "normalized.mp4"
    normalized_audio_path = task_root / "normalized.wav"
    artifacts_root = task_root / "artifacts"
    try:
        with log_stage(logger, f"{task.task_id}:download_video"):
            download_file(url=str(task.video_url), destination=original_video_path)
        with log_stage(logger, f"{task.task_id}:probe_original_video"):
            original_metadata = probe_video_metadata(original_video_path)
        _validate_video_constraints(task_id=task.task_id, video_path=original_video_path, duration=original_metadata.duration)
        with log_stage(logger, f"{task.task_id}:preprocess_media"):
            with ThreadPoolExecutor(max_workers=2, thread_name_prefix=f"{task.task_id}-preprocess") as executor:
                normalize_future = executor.submit(
                    _run_normalize_video_stage,
                    task_id=task.task_id,
                    original_video_path=original_video_path,
                    normalized_video_path=normalized_video_path,
                )
                extract_audio_future = executor.submit(
                    _run_extract_audio_stage,
                    task_id=task.task_id,
                    source_video_path=original_video_path,
                    output_audio_path=normalized_audio_path,
                    has_audio=original_metadata.has_audio,
                )
                normalize_future.result()
                local_audio_path = extract_audio_future.result()
        with log_stage(logger, f"{task.task_id}:extraction_stage"):
            extraction_artifacts = asyncio.run(
                run_video_extraction_stage(
                    job_id=task.task_id,
                    local_video_path=str(normalized_video_path),
                    local_audio_path=local_audio_path,
                    artifacts_root=str(artifacts_root),
                )
            )
        with log_stage(logger, f"{task.task_id}:reasoning_stage"):
            reasoning_artifacts = asyncio.run(
                run_vlm_reasoning_stage(
                    job_id=task.task_id,
                    local_temporal_segments_path=extraction_artifacts["local_temporal_segments_json"],
                    local_audio_path=local_audio_path,
                    local_frame_sampling_path=extraction_artifacts["local_frame_sampling_json"],
                    artifacts_root=str(artifacts_root),
                )
            )

        with log_stage(logger, f"{task.task_id}:caption_generation"):
            final_result, local_final_result_path = asyncio.run(
                _run_caption_generation(
                    task_id=task.task_id,
                    global_summary_path=Path(reasoning_artifacts["local_global_factual_summary_json"]),
                )
            )
        logger.info("Task %s completed. Final artifact at %s", task.task_id, local_final_result_path)
        captions = {style_name: final_result.captions[style_name].caption for style_name in requested_styles}
        return TaskResult(task_id=task.task_id, captions=captions)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Task %s failed. Falling back to safe captions.", task.task_id)
        fallback = _build_fallback_captions(task_id=task.task_id, requested_styles=requested_styles, error=str(exc))
        return TaskResult(task_id=task.task_id, captions=fallback)
    finally:
        if settings.debug_keep_temp:
            persist_debug_artifacts(
                task_id=task.task_id,
                original_video_path=original_video_path,
                normalized_video_path=normalized_video_path,
                normalized_audio_path=normalized_audio_path if normalized_audio_path.exists() else None,
                artifacts_root=artifacts_root,
            )
            logger.info("Keeping task temp directory at %s", task_root)
        else:
            shutil.rmtree(task_root, ignore_errors=True)


def _validate_video_constraints(*, task_id: str, video_path: Path, duration: float) -> None:
    size_mb = video_path.stat().st_size / (1024 * 1024)
    if size_mb > settings.max_video_size_mb:
        raise ValueError(
            f"Task {task_id} exceeds max input size: {size_mb:.1f}MB > {settings.max_video_size_mb}MB"
        )
    if duration > settings.max_video_duration_seconds:
        raise ValueError(
            f"Task {task_id} exceeds max duration: {duration:.1f}s > {settings.max_video_duration_seconds}s"
        )


def _extract_audio_if_present(*, video_path: Path, output_path: Path, has_audio: bool) -> str | None:
    if not has_audio:
        return None
    try:
        run_command(
            args=[
                settings.ffmpeg_path,
                "-y",
                "-i",
                str(video_path),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "24000",
                "-c:a",
                "pcm_s16le",
                str(output_path),
            ],
            timeout_seconds=settings.ffmpeg_timeout_seconds,
        )
    except ProcessExecutionError:
        logger.warning("Audio extraction failed for %s; continuing without transcript audio.", video_path)
        return None
    return str(output_path)


def _run_normalize_video_stage(*, task_id: str, original_video_path: Path, normalized_video_path: Path) -> Path:
    with log_stage(logger, f"{task_id}:normalize_video"):
        return normalize_video(
            input_video_path=original_video_path,
            output_video_path=normalized_video_path,
        )


def _run_extract_audio_stage(*, task_id: str, source_video_path: Path, output_audio_path: Path, has_audio: bool) -> str | None:
    with log_stage(logger, f"{task_id}:extract_audio"):
        return _extract_audio_if_present(
            video_path=source_video_path,
            output_path=output_audio_path,
            has_audio=has_audio,
        )


async def _run_caption_generation(*, task_id: str, global_summary_path: Path):
    client = OpenAIResponsesClient(
        OpenAIResponsesConfig(
            api_key=settings.openai_api_key or "",
            base_url=settings.openai_base_url,
            timeout_seconds=settings.openai_timeout_seconds,
            max_retries=settings.openai_max_retries,
            reasoning_effort=settings.openai_reasoning_effort,
            text_verbosity=settings.openai_text_verbosity,
        )
    )
    try:
        return await style_captions(
            client=client,
            model=settings.openai_final_caption_model,
            job_id=task_id,
            global_summary_path=global_summary_path,
        )
    finally:
        await client.aclose()


def _build_fallback_captions(*, task_id: str, requested_styles: list[str], error: str) -> dict[str, str]:
    base = (
        "The video could not be fully analyzed, so this caption stays conservative and avoids inventing details."
    )
    templates = {
        "formal": f"{base} Task {task_id} requires manual review.",
        "sarcastic": f"{base} Apparently the clip chose mystery mode and declined to cooperate.",
        "humorous_tech": f"{base} The caption pipeline hit an exception and returned a human-escalation feature.",
        "humorous_non_tech": f"{base} This one showed up like a blurry witness who forgot the whole story.",
    }
    logger.error("Fallback captions used for task %s: %s", task_id, error)
    return {style_name: templates[style_name] for style_name in requested_styles}


if __name__ == "__main__":
    main()
