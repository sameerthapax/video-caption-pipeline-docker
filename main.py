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

from pipeline.caption_pipeline import CaptionPipeline
from pipeline.normalize import normalize_video
from pipeline.probe_video import probe_video_metadata
from pipeline.transcription import load_or_create_transcript
from schemas.caption import STYLE_ORDER, StyleName
from services.client_pool import close_pooled_async_clients, get_llm_client
from services.local_files import download_file, guess_video_suffix
from services.process import ProcessExecutionError, run_command
from worker.config.settings import settings


logger = logging.getLogger("video-caption-pipeline")


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
        logger.exception("Fatal startup failure. Writing fallback results.")
        fallback_results = _build_fallback_results(raw_task_specs, error=f"fatal startup failure: {exc}")
        try:
            _write_results(output_path, fallback_results)
        except Exception:  # noqa: BLE001
            logger.exception("Failed to write fallback results to %s", output_path)
    finally:
        close_pooled_async_clients()


def run(*, input_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tasks = _load_tasks(input_path)
    results = _process_tasks(tasks)
    _write_results(output_path, results)
    logger.info("Wrote %s task results to %s", len(results), output_path)


def _configure_logging_safe() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        force=True,
    )


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
    max_workers = max(1, settings.max_concurrent_jobs)
    logger.info("Processing %s task(s) with max_concurrent_jobs=%s", len(tasks), max_workers)
    if max_workers == 1 or len(tasks) == 1:
        return [_process_task(task) for task in tasks]

    ordered_results: list[dict[str, Any] | None] = [None] * len(tasks)
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="caption-task") as executor:
        future_to_index = {executor.submit(_process_task, task): index for index, task in enumerate(tasks)}
        for future in as_completed(future_to_index):
            ordered_results[future_to_index[future]] = future.result()
    return [result for result in ordered_results if result is not None]


def _process_task(task: Any) -> dict[str, Any]:
    requested_styles = _normalize_requested_styles(getattr(task, "styles", []))
    unsupported_styles = [style_name for style_name in requested_styles if style_name not in STYLE_ORDER]
    if unsupported_styles:
        return _build_task_result(
            task_id=str(task.task_id),
            captions=_build_fallback_captions(
                task_id=str(task.task_id),
                requested_styles=requested_styles,
                error=f"unsupported styles requested: {', '.join(unsupported_styles)}",
            ),
        )

    missing_dependencies = _missing_generation_dependencies()
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
    audio_path = task_root / "audio.wav"
    transcript_dir = task_root / "transcripts"
    artifact_root = task_root / "artifacts"

    try:
        download_file(url=str(task.video_url), destination=original_video_path)
        original_metadata = probe_video_metadata(original_video_path)
        _validate_video_constraints(
            task_id=str(task.task_id),
            video_path=original_video_path,
            duration=original_metadata.duration,
        )

        analysis_video_path = original_video_path
        if settings.enable_video_normalization:
            analysis_video_path = normalize_video(
                input_video_path=original_video_path,
                output_video_path=normalized_video_path,
            )
        analysis_metadata = probe_video_metadata(analysis_video_path)

        local_audio_path = _extract_audio_if_present(
            video_path=analysis_video_path,
            output_path=audio_path,
            has_audio=analysis_metadata.has_audio,
        )
        transcript_path = _download_optional_transcript(task=task, task_root=task_root)
        transcript_text = load_or_create_transcript(
            task_id=str(task.task_id),
            provided_transcript_text=str(getattr(task, "transcript_text", "") or ""),
            transcript_source_path=transcript_path,
            audio_path=Path(local_audio_path) if local_audio_path else None,
            transcript_dir=transcript_dir,
        )

        pipeline = CaptionPipeline(
            llm_client=get_llm_client(),
            artifact_root=artifact_root,
            persist_artifacts=settings.debug_keep_temp,
        )
        result = asyncio.run(
            pipeline.run(
                job_id=str(task.task_id),
                video_path=analysis_video_path,
                video_metadata=analysis_metadata,
                transcript_text=transcript_text,
                requested_styles=[style_name for style_name in requested_styles if style_name in STYLE_ORDER],
            )
        )
        captions = {style_name: result["captions"][style_name].caption for style_name in requested_styles}
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
            logger.info("Keeping task temp directory at %s", task_root)
        else:
            shutil.rmtree(task_root, ignore_errors=True)


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
                "16000",
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


def _download_optional_transcript(*, task: Any, task_root: Path) -> Path | None:
    transcript_url = getattr(task, "transcript_url", None)
    if not transcript_url:
        return None
    transcript_path = task_root / "transcript.txt"
    download_file(url=str(transcript_url), destination=transcript_path)
    return transcript_path


def _missing_generation_dependencies() -> list[str]:
    missing: list[str] = []
    if settings.llm_provider == "fireworks":
        if not settings.fireworks_proxy_url and not settings.fireworks_api_key:
            missing.append("FIREWORKS_PROXY_URL or FIREWORKS_API_KEY")
        if not settings.fireworks_model:
            missing.append("FIREWORKS_MODEL")
        return missing
    if settings.llm_provider == "openrouter":
        if not settings.openrouter_proxy_url and not settings.openrouter_api_key:
            missing.append("OPENROUTER_PROXY_URL or OPENROUTER_API_KEY")
        if not settings.openrouter_model:
            missing.append("OPENROUTER_MODEL")
        return missing
    missing.append("LLM_PROVIDER must be 'fireworks' or 'openrouter'")
    return missing


def _validate_video_constraints(*, task_id: str, video_path: Path, duration: float) -> None:
    size_mb = video_path.stat().st_size / (1024 * 1024)
    if size_mb > settings.max_video_size_mb:
        raise ValueError(f"Task {task_id} exceeds max input size: {size_mb:.1f}MB > {settings.max_video_size_mb}MB")
    if duration > settings.max_video_duration_seconds:
        raise ValueError(f"Task {task_id} exceeds max duration: {duration:.1f}s > {settings.max_video_duration_seconds}s")


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
    return {"task_id": task_id, "captions": captions}


def _build_fallback_captions(*, task_id: str, requested_styles: list[str], error: str) -> dict[str, str]:
    base = "The video could not be fully analyzed, so this caption stays conservative and avoids inventing details."
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
        requested_styles = list(STYLE_ORDER)
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
        # Some Docker-mounted output volumes allow writes but reject atomic replace.
        _make_path_writable(output_path.parent)
        _make_path_writable(output_path)
        try:
            output_path.write_text(payload, encoding="utf-8")
            try:
                if temp_path.exists():
                    temp_path.unlink()
            except OSError:
                pass
            return
        except PermissionError:
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
