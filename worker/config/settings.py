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
        self.llm_provider = self._get_str("LLM_PROVIDER", "fireworks")

        self.max_video_size_mb = self._get_int("MAX_VIDEO_SIZE_MB", 500)
        self.max_video_duration_seconds = self._get_int("MAX_VIDEO_DURATION_SECONDS", 180)

        self.frame_extract_width = self._get_int("FRAME_EXTRACT_WIDTH", 768)
        self.max_frames_per_video = self._get_int("MAX_FRAMES_PER_VIDEO", 3)
        self.min_anchor_frames = self._get_int("MIN_ANCHOR_FRAMES", 3)
        self.enable_planned_frame_extraction = self._get_bool("ENABLE_PLANNED_FRAME_EXTRACTION", False)
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

        self.enable_video_normalization = self._get_bool("ENABLE_VIDEO_NORMALIZATION", False)
        self.enable_local_whisper = self._get_bool("ENABLE_LOCAL_WHISPER", False)
        self.whisper_command = self._get_str("WHISPER_COMMAND", "whisper")
        self.whisper_model = "large"
        self.whisper_language = self._get_optional_str("WHISPER_LANGUAGE")
        self.force_transcription = self._get_bool("FORCE_TRANSCRIPTION", False)

        self.fireworks_api_key = self._get_optional_str("FIREWORKS_API_KEY")
        self.fireworks_model = self._get_str("FIREWORKS_MODEL", "accounts/fireworks/models/kimi-k2p6")
        self.fireworks_base_url = self._get_str("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1")
        self.fireworks_proxy_url = self._get_str("FIREWORKS_PROXY_URL", "")
        self.fireworks_proxy_token = self._get_str("FIREWORKS_PROXY_TOKEN", "")
        self.openrouter_api_key = self._get_optional_str("OPENROUTER_API_KEY")
        self.openrouter_model = self._get_str("OPENROUTER_MODEL", "")
        self.openrouter_base_url = self._get_str("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        self.openrouter_proxy_url = self._get_str("OPENROUTER_PROXY_URL", "")
        self.openrouter_proxy_token = self._get_str("OPENROUTER_PROXY_TOKEN", "")
        self.fireworks_timeout_seconds = self._get_int("FIREWORKS_TIMEOUT_SECONDS", 180)
        self.fireworks_max_retries = self._get_int("FIREWORKS_MAX_RETRIES", 3)
        self.fireworks_reasoning_effort = self._get_str("FIREWORKS_REASONING_EFFORT", "none")
        self.fireworks_temperature = self._get_float("FIREWORKS_TEMPERATURE", 0.2)
        self.fireworks_creative_temperature = self._get_float("FIREWORKS_CREATIVE_TEMPERATURE", 0.75)
        self.fireworks_vision_max_tokens = self._get_int("FIREWORKS_VISION_MAX_TOKENS", 300)
        self.fireworks_caption_max_tokens = self._get_int("FIREWORKS_CAPTION_MAX_TOKENS", 180)
        self.fireworks_judge_max_tokens = self._get_int("FIREWORKS_JUDGE_MAX_TOKENS", 200)

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
