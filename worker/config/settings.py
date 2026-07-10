from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv("/app/.env", override=False)


class Settings:
    def __init__(self) -> None:
        self.app_name = self._get_str("APP_NAME", "video-caption-hackathon-agent")

        self.input_tasks_path = self._get_str("INPUT_TASKS_PATH", "/input/tasks.json")
        self.output_results_path = self._get_str("OUTPUT_RESULTS_PATH", "/output/results.json")
        self.worker_tmp_root = self._get_str("WORKER_TMP_ROOT", "/tmp/video-caption-agent")

        self.debug_keep_temp = self._get_bool("DEBUG_KEEP_TEMP", False)
        self.max_concurrent_jobs = self._get_int("MAX_CONCURRENT_JOBS", 3)
        self.simplified_pipeline_enabled = self._get_bool("SIMPLIFIED_PIPELINE_ENABLED", False)
        self.enable_video_normalization = self._get_bool("ENABLE_VIDEO_NORMALIZATION", False)
        self.pipeline_segment_count = self._get_int("PIPELINE_SEGMENT_COUNT", 3)

        self.max_video_size_mb = self._get_int("MAX_VIDEO_SIZE_MB", 500)
        self.max_video_duration_seconds = self._get_int("MAX_VIDEO_DURATION_SECONDS", 180)
        self.frame_extract_width = self._get_int("FRAME_EXTRACT_WIDTH", 640)
        self.min_frames_per_video = self._get_int("MIN_FRAMES_PER_VIDEO", 6)
        self.max_frames_per_video = self._get_int("MAX_FRAMES_PER_VIDEO", 12)
        self.target_seconds_per_frame = self._get_float("TARGET_SECONDS_PER_FRAME", 6.0)
        self.max_scene_change_frames = self._get_int("MAX_SCENE_CHANGE_FRAMES", 6)
        self.max_uniform_frames = self._get_int("MAX_UNIFORM_FRAMES", 4)
        self.max_safety_frames = self._get_int("MAX_SAFETY_FRAMES", 2)

        self.ffmpeg_path = self._get_str("FFMPEG_PATH", "ffmpeg")
        self.ffprobe_path = self._get_str("FFPROBE_PATH", "ffprobe")
        self.ffmpeg_timeout_seconds = self._get_int("FFMPEG_TIMEOUT_SECONDS", 300)
        self.ffprobe_timeout_seconds = self._get_int("FFPROBE_TIMEOUT_SECONDS", 60)
        self.ffmpeg_max_concurrency = self._get_int("FFMPEG_MAX_CONCURRENCY", 2)

        self.google_gemini_api_key = self._get_optional_str("GOOGLE_GEMINI_API_KEY")
        self.google_gemini_base_url = self._get_str("GOOGLE_GEMINI_BASE_URL", "https://generativelanguage.googleapis.com")
        self.google_gemini_proxy_url = self._get_str("GOOGLE_GEMINI_PROXY_URL", "")
        self.google_gemini_proxy_token = self._get_str("GOOGLE_GEMINI_PROXY_TOKEN", "")
        self.google_gemini_transcription_model = self._get_str("GOOGLE_GEMINI_TRANSCRIPTION_MODEL", "gemini-2.5-flash")
        self.google_gemini_vision_model = self._get_str("GOOGLE_GEMINI_VISION_MODEL", "gemma-4-31b-it")
        self.google_gemini_timeout_seconds = self._get_int("GOOGLE_GEMINI_TIMEOUT_SECONDS", 60)
        self.google_gemini_max_retries = self._get_int("GOOGLE_GEMINI_MAX_RETRIES", 3)
        self.google_gemini_max_concurrency = self._get_int("GOOGLE_GEMINI_MAX_CONCURRENCY", 4)

        self.fireworks_api_key = self._get_optional_str("FIREWORKS_API_KEY")
        self.fireworks_base_url = self._get_str("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1")
        self.fireworks_proxy_url = self._get_str("FIREWORKS_PROXY_URL", "")
        self.fireworks_proxy_token = self._get_str("FIREWORKS_PROXY_TOKEN", "")
        self.fireworks_model = self._get_optional_str("FIREWORKS_MODEL")
        self.fireworks_timeout_seconds = self._get_int("FIREWORKS_TIMEOUT_SECONDS", 90)
        self.fireworks_max_retries = self._get_int("FIREWORKS_MAX_RETRIES", 3)
        self.fireworks_max_concurrency = self._get_int("FIREWORKS_MAX_CONCURRENCY", 3)

        self.openai_api_key = self._get_optional_str("OPENAI_API_KEY")
        self.openai_base_url = self._get_str("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.openai_proxy_url = self._get_str("OPENAI_PROXY_URL", "")
        self.openai_proxy_token = self._get_str("OPENAI_PROXY_TOKEN", "")
        self.openai_final_caption_model = self._get_str("OPENAI_FINAL_CAPTION_MODEL", "gpt-5.4-mini-2026-03-17")
        self.openai_timeout_seconds = self._get_int("OPENAI_TIMEOUT_SECONDS", 45)
        self.openai_max_retries = self._get_int("OPENAI_MAX_RETRIES", 1)
        self.openai_temperature = self._get_float("OPENAI_TEMPERATURE", 0.2)
        self.openai_reasoning_effort = self._get_str("OPENAI_REASONING_EFFORT", "low")
        self.openai_text_verbosity = self._get_str("OPENAI_TEXT_VERBOSITY", "medium")
        self.caption_min_words = self._get_int("CAPTION_MIN_WORDS", 28)
        self.caption_max_words = self._get_int("CAPTION_MAX_WORDS", 50)
        self.log_model_io = self._get_bool("LOG_MODEL_IO", True)

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
