from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover - local test environments may not install runtime deps
    def load_dotenv(*_args, **_kwargs):
        return False


load_dotenv("/app/.env", override=False)


class Settings:
    def __init__(self) -> None:
        self.app_name = self._get_str("APP_NAME", "video-caption-pipeline")

        self.input_tasks_path = self._get_str("INPUT_TASKS_PATH", "/input/tasks.json")
        self.output_results_path = self._get_str("OUTPUT_RESULTS_PATH", "/output/results.json")
        self.worker_tmp_root = self._get_str("WORKER_TMP_ROOT", "/tmp/video-caption-pipeline")

        self.debug_keep_temp = self._get_bool("DEBUG_KEEP_TEMP", False)
        self.max_concurrent_jobs = self._get_int("MAX_CONCURRENT_JOBS", 3)
        self.log_model_io = self._get_bool("LOG_MODEL_IO", True)

        self.caption_pipeline_mode = self._get_str("CAPTION_PIPELINE_MODE", "verified_scene")
        self.observation_caption_mode = self._get_str("OBSERVATION_CAPTION_MODE", "combined")
        self.run_judge_checks = self._get_bool("RUN_JUDGE_CHECKS", False)

        self.max_video_size_mb = self._get_int("MAX_VIDEO_SIZE_MB", 500)
        self.max_video_duration_seconds = self._get_int("MAX_VIDEO_DURATION_SECONDS", 180)

        self.frame_extract_width = self._get_int("FRAME_EXTRACT_WIDTH", 768)
        self.max_frames_per_video = self._get_int("MAX_FRAMES_PER_VIDEO", 12)
        self.min_anchor_frames = self._get_int("MIN_ANCHOR_FRAMES", 3)
        self.use_opencv_frames = self._get_bool("USE_OPENCV_FRAMES", False)
        self.use_scene_midpoint_frames = self._get_bool("USE_SCENE_MIDPOINT_FRAMES", False)
        self.scene_change_scan_interval_seconds = self._get_float("SCENE_CHANGE_SCAN_INTERVAL_SECONDS", 1.0)
        self.scene_change_min_spacing_seconds = self._get_float("SCENE_CHANGE_MIN_SPACING_SECONDS", 4.0)
        self.max_selected_scene_changes = self._get_int("MAX_SELECTED_SCENE_CHANGES", 8)
        self.frame_dedupe_hash_threshold = self._get_int("FRAME_DEDUPE_HASH_THRESHOLD", 6)

        self.ffmpeg_path = self._get_str("FFMPEG_PATH", "ffmpeg")
        self.ffprobe_path = self._get_str("FFPROBE_PATH", "ffprobe")
        self.ffmpeg_timeout_seconds = self._get_int("FFMPEG_TIMEOUT_SECONDS", 300)
        self.ffprobe_timeout_seconds = self._get_int("FFPROBE_TIMEOUT_SECONDS", 60)
        self.ffmpeg_max_concurrency = self._get_int("FFMPEG_MAX_CONCURRENCY", 2)

        self.enable_video_normalization = self._get_bool("ENABLE_VIDEO_NORMALIZATION", True)
        self.enable_local_whisper = self._get_bool("ENABLE_LOCAL_WHISPER", False)
        self.whisper_command = self._get_str("WHISPER_COMMAND", "whisper")
        self.whisper_model = self._get_str("WHISPER_MODEL", "base")
        self.whisper_language = self._get_optional_str("WHISPER_LANGUAGE")
        self.force_transcription = self._get_bool("FORCE_TRANSCRIPTION", False)

        self.vision_api_key = self._get_optional_str("VISION_API_KEY")
        self.vision_base_url = self._get_str("VISION_BASE_URL", "https://generativelanguage.googleapis.com")
        self.vision_proxy_url = self._get_str("VISION_PROXY_URL", "")
        self.vision_proxy_token = self._get_str("VISION_PROXY_TOKEN", "")
        self.vision_model = self._get_str("VISION_MODEL", "gemma-4-31b-it")
        self.vision_timeout_seconds = self._get_int("VISION_TIMEOUT_SECONDS", 90)
        self.vision_max_retries = self._get_int("VISION_MAX_RETRIES", 3)
        self.vision_max_concurrency = self._get_int("VISION_MAX_CONCURRENCY", 4)

        self.openai_api_key = self._get_optional_str("OPENAI_API_KEY")
        self.openai_base_url = self._get_str("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.openai_proxy_url = self._get_str("OPENAI_PROXY_URL", "")
        self.openai_proxy_token = self._get_str("OPENAI_PROXY_TOKEN", "")
        self.openai_caption_model = self._get_str("OPENAI_CAPTION_MODEL", "gpt-5.5")
        self.openai_judge_model = self._get_str("OPENAI_JUDGE_MODEL", self.openai_caption_model)
        self.openai_timeout_seconds = self._get_int("OPENAI_TIMEOUT_SECONDS", 60)
        self.openai_max_retries = self._get_int("OPENAI_MAX_RETRIES", 2)
        self.openai_temperature = self._get_float("OPENAI_TEMPERATURE", 0.2)
        self.openai_reasoning_effort = self._get_str("OPENAI_REASONING_EFFORT", "low")
        self.openai_text_verbosity = self._get_str("OPENAI_TEXT_VERBOSITY", "medium")

        self.caption_min_words = self._get_int("CAPTION_MIN_WORDS", 25)
        self.caption_max_words = self._get_int("CAPTION_MAX_WORDS", 60)

    @property
    def output_root(self) -> Path:
        return Path(self.output_results_path).parent

    @property
    def debug_root(self) -> Path:
        return self.output_root / "debug"

    @staticmethod
    def _get_optional_str(name: str) -> str | None:
        value = os.environ.get(name)
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    @staticmethod
    def _get_str(name: str, default: str) -> str:
        value = os.environ.get(name)
        if value is None:
            return default
        stripped = value.strip()
        return stripped or default

    @staticmethod
    def _get_int(name: str, default: int) -> int:
        value = os.environ.get(name)
        if value is None or not value.strip():
            return default
        return int(value)

    @staticmethod
    def _get_float(name: str, default: float) -> float:
        value = os.environ.get(name)
        if value is None or not value.strip():
            return default
        return float(value)

    @staticmethod
    def _get_bool(name: str, default: bool) -> bool:
        value = os.environ.get(name)
        if value is None or not value.strip():
            return default
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
        raise ValueError(f"Invalid boolean value for {name}: {value}")


settings = Settings()
