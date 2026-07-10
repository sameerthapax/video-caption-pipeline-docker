from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
import logging
import os
import shutil
import stat
from pathlib import Path
from typing import Any


DEFAULT_STYLE_ORDER = ("formal", "sarcastic", "humorous_tech", "humorous_non_tech")
logger = logging.getLogger("video-caption-hackathon-agent")


@dataclass(slots=True)
class TaskSpec:
    task_id: str
    video_url: str
    styles: list[str]


def main() -> None:
    input_path = Path(_get_env_str("INPUT_TASKS_PATH", "/input/tasks.json"))
    output_path = Path(_get_env_str("OUTPUT_RESULTS_PATH", "/output/results.json"))
    raw_task_specs = _load_task_specs_best_effort(input_path)

    _configure_logging_safe()

    try:
        logger.info("Batch pipeline starting with input=%s output=%s", input_path, output_path)
        run(input_path=input_path, output_path=output_path)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Fatal startup failure. Writing fallback Track 2 results.")
        fallback_results = _build_fallback_results(
            raw_task_specs,
            error=f"fatal startup failure: {exc}",
        )
        try:
            _write_results(output_path, fallback_results)
        except Exception:  # noqa: BLE001
            logger.exception("Failed to write fallback results to %s", output_path)
    finally:
        from services.client_pool import close_pooled_async_clients

        close_pooled_async_clients()


def run(*, input_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tasks = _load_tasks(input_path)
    results = _process_tasks(tasks)
    _write_results(output_path, results)
    logger.info("Wrote %s task results to %s", len(results), output_path)


def _configure_logging_safe() -> None:
    try:
        from worker.runtime import configure_logging

        log_path = configure_logging()
        if log_path is not None:
            logger.info("Pipeline logging initialized at %s", log_path)
        else:
            logger.info("Pipeline logging initialized with stderr-only output.")
    except Exception as exc:  # noqa: BLE001
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
            force=True,
        )
        logger.warning("Falling back to stderr-only logging: %s", exc)


def _load_tasks(tasks_path: Path) -> list[Any]:
    from schemas.tasks import CaptionTask

    if not tasks_path.exists():
        raise FileNotFoundError(f"Input task file does not exist: {tasks_path}")

    payload = json.loads(tasks_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Input task file must contain a JSON array.")

    return [CaptionTask.model_validate(item) for item in payload]


def _load_task_specs_best_effort(tasks_path: Path) -> list[TaskSpec]:
    if not tasks_path.exists():
        return []

    try:
        payload = json.loads(tasks_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return []

    if not isinstance(payload, list):
        return []

    task_specs: list[TaskSpec] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            continue
        task_id = str(item.get("task_id") or f"task-{index + 1}")
        video_url = str(item.get("video_url") or "")
        styles = _normalize_requested_styles(item.get("styles"))
        task_specs.append(TaskSpec(task_id=task_id, video_url=video_url, styles=styles))
    return task_specs


def _process_tasks(tasks: list[Any]) -> list[dict[str, Any]]:
    if not tasks:
        return []

    from worker.config.settings import settings

    max_workers = max(1, settings.max_concurrent_jobs)
    logger.info("Processing %s task(s) with max_concurrent_jobs=%s", len(tasks), max_workers)

    if max_workers == 1 or len(tasks) == 1:
        return [_process_task(task) for task in tasks]

    ordered_results: list[dict[str, Any] | None] = [None] * len(tasks)
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="caption-task") as executor:
        future_to_index = {
            executor.submit(_process_task, task): index
            for index, task in enumerate(tasks)
        }
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            ordered_results[index] = future.result()

    return [result for result in ordered_results if result is not None]


def _process_task(task: Any) -> dict[str, Any]:
    from pipeline.normalize import normalize_video
    from pipeline.probe_video import probe_video_metadata
    from pipeline.run_extraction_stage import run_video_extraction_stage
    from pipeline.run_vlm_stage import run_vlm_reasoning_stage
    from pipeline.style import style_captions
    from services.client_pool import get_openai_responses_client
    from services.local_files import download_file, guess_video_suffix
    from services.process import ProcessExecutionError, run_command
    from worker.config.settings import settings
    from worker.runtime import log_stage, persist_debug_artifacts

    requested_styles = _normalize_requested_styles(getattr(task, "styles", []))
    unsupported_styles = [style_name for style_name in requested_styles if style_name not in DEFAULT_STYLE_ORDER]
    if unsupported_styles:
        return _build_task_result(
            task_id=str(task.task_id),
            captions=_build_fallback_captions(
                task_id=str(task.task_id),
                requested_styles=requested_styles,
                error=f"unsupported styles requested: {', '.join(unsupported_styles)}",
            ),
        )

    missing_dependencies = _missing_generation_dependencies(settings)
    if missing_dependencies:
        return _build_task_result(
            task_id=str(task.task_id),
            captions=_build_fallback_captions(
                task_id=str(task.task_id),
                requested_styles=requested_styles,
                error=f"missing external dependencies: {', '.join(missing_dependencies)}",
            ),
        )

    task_root = Path(settings.worker_tmp_root) / str(task.task_id)
    if task_root.exists():
        shutil.rmtree(task_root, ignore_errors=True)
    task_root.mkdir(parents=True, exist_ok=True)

    original_video_path = task_root / f"source{guess_video_suffix(str(task.video_url))}"
    normalized_video_path = task_root / "normalized.mp4"
    normalized_audio_path = task_root / "normalized.wav"
    artifacts_root = task_root / "artifacts"
    debug_persist_artifacts = settings.debug_keep_temp

    def extract_audio_if_present(*, video_path: Path, output_path: Path, has_audio: bool) -> str | None:
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

    def run_normalize_video_stage(*, task_id: str, source_video_path: Path, output_video_path: Path) -> Path:
        with log_stage(logger, f"{task_id}:normalize_video"):
            return normalize_video(
                input_video_path=source_video_path,
                output_video_path=output_video_path,
            )

    def run_extract_audio_stage(*, task_id: str, source_video_path: Path, output_audio_path: Path, has_audio: bool) -> str | None:
        with log_stage(logger, f"{task_id}:extract_audio"):
            return extract_audio_if_present(
                video_path=source_video_path,
                output_path=output_audio_path,
                has_audio=has_audio,
            )

    def run_extraction_stage_sync(*, task_id: str, source_video_path: Path, local_audio_path: str | None) -> dict[str, Any]:
        with log_stage(logger, f"{task_id}:extraction_stage"):
            return asyncio.run(
                run_video_extraction_stage(
                    job_id=str(task.task_id),
                    local_video_path=str(source_video_path),
                    local_audio_path=local_audio_path,
                    artifacts_root=str(artifacts_root),
                    persist_artifacts=debug_persist_artifacts,
                )
            )

    async def run_caption_generation(*, task_id: str, global_summary):
        return await style_captions(
            client=get_openai_responses_client(),
            model=settings.openai_final_caption_model,
            job_id=str(task.task_id),
            global_summary=global_summary,
            artifact_root=artifacts_root,
            persist_artifacts=debug_persist_artifacts,
        )

    try:
        with log_stage(logger, f"{task.task_id}:download_video"):
            download_file(url=str(task.video_url), destination=original_video_path)
        with log_stage(logger, f"{task.task_id}:probe_original_video"):
            original_metadata = probe_video_metadata(original_video_path)
        _validate_video_constraints(
            task_id=str(task.task_id),
            video_path=original_video_path,
            duration=original_metadata.duration,
            max_video_size_mb=settings.max_video_size_mb,
            max_video_duration_seconds=settings.max_video_duration_seconds,
        )
        analysis_video_path = original_video_path
        local_audio_path: str | None = None
        extraction_artifacts: dict[str, Any] | None = None
        if settings.enable_video_normalization:
            with log_stage(logger, f"{task.task_id}:preprocess_media"):
                with ThreadPoolExecutor(max_workers=2, thread_name_prefix=f"{task.task_id}-preprocess") as executor:
                    normalize_future = executor.submit(
                        run_normalize_video_stage,
                        task_id=str(task.task_id),
                        source_video_path=original_video_path,
                        output_video_path=normalized_video_path,
                    )
                    extract_audio_future = executor.submit(
                        run_extract_audio_stage,
                        task_id=str(task.task_id),
                        source_video_path=original_video_path,
                        output_audio_path=normalized_audio_path,
                        has_audio=original_metadata.has_audio,
                    )
                    analysis_video_path = normalize_future.result()
                    local_audio_path = extract_audio_future.result()
            extraction_artifacts = run_extraction_stage_sync(
                task_id=str(task.task_id),
                source_video_path=analysis_video_path,
                local_audio_path=local_audio_path,
            )
        else:
            with ThreadPoolExecutor(max_workers=2, thread_name_prefix=f"{task.task_id}-prepare") as executor:
                extraction_future = executor.submit(
                    run_extraction_stage_sync,
                    task_id=str(task.task_id),
                    source_video_path=analysis_video_path,
                    local_audio_path=None,
                )
                audio_future = executor.submit(
                    run_extract_audio_stage,
                    task_id=str(task.task_id),
                    source_video_path=original_video_path,
                    output_audio_path=normalized_audio_path,
                    has_audio=original_metadata.has_audio,
                )
                extraction_artifacts = extraction_future.result()
                local_audio_path = audio_future.result()
            extraction_artifacts["local_audio_path"] = local_audio_path or ""
        with log_stage(logger, f"{task.task_id}:reasoning_stage"):
            reasoning_artifacts = asyncio.run(
                run_vlm_reasoning_stage(
                    job_id=str(task.task_id),
                    temporal_segments=extraction_artifacts["temporal_segments"],
                    local_audio_path=local_audio_path,
                    artifacts_root=str(artifacts_root),
                    persist_artifacts=debug_persist_artifacts,
                )
            )
        with log_stage(logger, f"{task.task_id}:caption_generation"):
            final_result, local_final_result_path = asyncio.run(
                run_caption_generation(
                    task_id=str(task.task_id),
                    global_summary=reasoning_artifacts["global_summary"],
                )
            )
        if local_final_result_path:
            logger.info("Task %s completed. Final artifact at %s", task.task_id, local_final_result_path)
        else:
            logger.info("Task %s completed.", task.task_id)
        captions = {style_name: final_result.captions[style_name].caption for style_name in requested_styles}
        return _build_task_result(task_id=str(task.task_id), captions=captions)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Task %s failed. Falling back to safe captions.", task.task_id)
        return _build_task_result(
            task_id=str(task.task_id),
            captions=_build_fallback_captions(
                task_id=str(task.task_id),
                requested_styles=requested_styles,
                error=str(exc),
            ),
        )
    finally:
        if settings.debug_keep_temp:
            persist_debug_artifacts(
                task_id=str(task.task_id),
                original_video_path=original_video_path,
                normalized_video_path=analysis_video_path if analysis_video_path.exists() else normalized_video_path,
                normalized_audio_path=normalized_audio_path if normalized_audio_path.exists() else None,
                artifacts_root=artifacts_root,
            )
            logger.info("Keeping task temp directory at %s", task_root)
        else:
            shutil.rmtree(task_root, ignore_errors=True)


def _missing_generation_dependencies(settings: Any) -> list[str]:
    missing: list[str] = []
    if not settings.google_gemini_api_key and not settings.google_gemini_proxy_url:
        missing.append("GOOGLE_GEMINI_API_KEY or GOOGLE_GEMINI_PROXY_URL")
    if not settings.google_gemini_vision_model:
        missing.append("GOOGLE_GEMINI_VISION_MODEL")
    if not settings.openai_api_key and not settings.openai_proxy_url:
        missing.append("OPENAI_API_KEY or OPENAI_PROXY_URL")
    return missing


def _validate_video_constraints(
    *,
    task_id: str,
    video_path: Path,
    duration: float,
    max_video_size_mb: int,
    max_video_duration_seconds: int,
) -> None:
    size_mb = video_path.stat().st_size / (1024 * 1024)
    if size_mb > max_video_size_mb:
        raise ValueError(
            f"Task {task_id} exceeds max input size: {size_mb:.1f}MB > {max_video_size_mb}MB"
        )
    if duration > max_video_duration_seconds:
        raise ValueError(
            f"Task {task_id} exceeds max duration: {duration:.1f}s > {max_video_duration_seconds}s"
        )


def _build_fallback_results(task_specs: list[TaskSpec], *, error: str) -> list[dict[str, Any]]:
    return [
        _build_task_result(
            task_id=task_spec.task_id,
            captions=_build_fallback_captions(
                task_id=task_spec.task_id,
                requested_styles=task_spec.styles,
                error=error,
            ),
        )
        for task_spec in task_specs
    ]


def _build_task_result(*, task_id: str, captions: dict[str, str]) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "captions": captions,
    }


def _build_fallback_captions(*, task_id: str, requested_styles: list[str], error: str) -> dict[str, str]:
    base = (
        "The video could not be fully analyzed, so this caption stays conservative and avoids inventing details."
    )
    default_templates = {
        "formal": f"{base} Task {task_id} requires manual review.",
        "sarcastic": f"{base} Apparently the clip chose mystery mode and declined to cooperate.",
        "humorous_tech": f"{base} The caption pipeline hit an exception and returned a human-escalation feature.",
        "humorous_non_tech": f"{base} This one showed up like a blurry witness who forgot the whole story.",
    }
    captions: dict[str, str] = {}
    for style_name in requested_styles:
        captions[style_name] = default_templates.get(
            style_name,
            f"{base} Style '{style_name}' was requested, but the task fell back to a generic caption.",
        )
    logger.error("Fallback captions used for task %s: %s", task_id, error)
    return captions


def _normalize_requested_styles(value: Any) -> list[str]:
    requested_styles: list[str] = []
    if isinstance(value, list):
        for item in value:
            if not isinstance(item, str):
                continue
            style_name = item.strip()
            if style_name and style_name not in requested_styles:
                requested_styles.append(style_name)
    if not requested_styles:
        requested_styles = list(DEFAULT_STYLE_ORDER)
    return requested_styles


def _write_results(output_path: Path, results: list[dict[str, Any]]) -> None:
    payload = json.dumps(results, indent=2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.parent / f".{output_path.name}.tmp"
    try:
        temp_path.write_text(payload, encoding="utf-8")
        temp_path.replace(output_path)
        return
    except PermissionError:
        _make_path_writable(output_path.parent)
        _make_path_writable(output_path)
        try:
            if output_path.exists() and output_path.is_file():
                output_path.unlink()
        except OSError:
            pass
        temp_path.write_text(payload, encoding="utf-8")
        temp_path.replace(output_path)


def _make_path_writable(path: Path) -> None:
    try:
        if not path.exists():
            return
        current_mode = stat.S_IMODE(path.stat().st_mode)
        path.chmod(current_mode | stat.S_IWUSR)
    except OSError:
        return


def _get_env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    if value is None:
        return default
    stripped = value.strip()
    return stripped or default


if __name__ == "__main__":
    main()
