from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel

from services.async_limits import get_loop_semaphore
from schemas.transcription import MusicMetadata, ToneMetadata, TranscriptChunk
from worker.config.settings import settings

logger = logging.getLogger("video-caption-pipeline.worker")


class GoogleGeminiError(RuntimeError):
    pass


class GoogleGeminiConfig(BaseModel):
    api_key: str
    base_url: str = "https://generativelanguage.googleapis.com"
    model: str = "gemini-3.5-flash"
    timeout_seconds: int = 60
    max_retries: int = 3


class GoogleGeminiClient:
    def __init__(self, config: GoogleGeminiConfig) -> None:
        self.config = config
        self._client = httpx.AsyncClient(timeout=self.config.timeout_seconds)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> GoogleGeminiClient:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    def build_transcription_request_metadata(self) -> dict[str, str]:
        return {
            "provider": "google_gemini",
            "api": "interactions",
            "base_url": self.config.base_url,
            "model": self.config.model,
        }

    async def transcribe_audio_window(
        self,
        *,
        audio_path: Path,
        start: float,
        end: float,
    ) -> TranscriptChunk:
        payload = {
            "model": self.config.model,
            "input": [
                {"type": "text", "text": _build_transcription_prompt(start=start, end=end)},
                {
                    "type": "audio",
                    "data": base64.b64encode(audio_path.read_bytes()).decode("utf-8"),
                    "mime_type": "audio/wav",
                },
            ],
            "response_format": _transcript_response_schema(),
        }
        headers = {
            "x-goog-api-key": self.config.api_key,
            "Content-Type": "application/json",
        }
        endpoint = f"{self.config.base_url.rstrip('/')}/v1beta/interactions"
        semaphore = get_loop_semaphore(
            name="google_gemini_requests",
            limit=settings.google_gemini_max_concurrency,
        )

        last_error: Exception | None = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
                async with semaphore:
                    response = await self._client.post(endpoint, headers=headers, json=payload)
                if response.status_code >= 400:
                    raise GoogleGeminiError(response.text or "Google Gemini transcription request failed.")
                break
            except (httpx.HTTPError, GoogleGeminiError) as exc:
                last_error = exc
                logger.warning(
                    "Gemini transcription request failed for %.2f-%.2fs attempt %s/%s: %s",
                    start,
                    end,
                    attempt,
                    self.config.max_retries,
                    exc,
                )
                if attempt >= self.config.max_retries:
                    raise GoogleGeminiError(
                        f"Google Gemini transcription failed for {start:.2f}-{end:.2f}s after {self.config.max_retries} attempts."
                    ) from exc
                await _sleep_backoff(attempt)

        if last_error is not None and "response" not in locals():
            raise GoogleGeminiError(
                f"Google Gemini transcription failed for {start:.2f}-{end:.2f}s."
            ) from last_error

        response_payload = response.json()
        try:
            parsed = _extract_structured_payload(response_payload)
        except GoogleGeminiError as exc:
            raise GoogleGeminiError(
                f"{exc} Raw response: {json.dumps(response_payload)[:4000]}"
            ) from exc

        return TranscriptChunk(
            start=start,
            end=end,
            **_normalize_transcript_payload(parsed),
        )


async def _sleep_backoff(attempt: int) -> None:
    import asyncio

    await asyncio.sleep(min(2 ** (attempt - 1), 8))


def _build_transcription_prompt(*, start: float, end: float) -> str:
    return f"""
Analyze this audio window from a short-form video.

Window start: {start:.2f} seconds
Window end: {end:.2f} seconds

Return only valid JSON matching the provided schema.

Requirements:
1. Transcribe spoken words exactly when intelligible.
2. If there is no intelligible speech, set "text" to "".
3. Produce "expressive_transcript", which should include inline bracketed tags when relevant.
4. Use only these expressive tags when applicable:
   [laughter], [music], [applause], [speaker_change], [shouting], [crying], [gasp], [sigh], [slap_sound], [impact_sound], [beep], [unknown_sound], [unknown_speech_tone]
5. Keep expressive_transcript close to the spoken transcript, but include important non-speech events inline.
6. If there is no meaningful speech but there are sounds, expressive_transcript may contain only tags.
7. Detect whether music is present.
8. If music is present, describe its type, instrumentation, energy, and overall feel.
9. Infer the delivery and tone of the speaker if speech is present.
10. If the window is effectively silent with no speech and no music, set no_audio=true at the top level, music.no_audio=true, and tone.no_audio=true.
11. Set tone.confidence to a float between 0.0 and 1.0.
12. Use concise strings. Do not include markdown.
13. If uncertain, use [unknown_sound] and/or [unknown_speech_tone] as needed, keep fields concise, and lower confidence.
""".strip()


def _transcript_response_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "no_audio": {"type": "boolean"},
            "text": {"type": "string"},
            "expressive_transcript": {"type": "string"},
            "music": {
                "type": "object",
                "properties": {
                    "no_audio": {"type": "boolean"},
                    "present": {"type": "boolean"},
                    "type": {"type": "string"},
                    "instrumentation": {"type": "string"},
                    "energy": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["no_audio", "present", "type", "instrumentation", "energy", "description"],
            },
            "tone": {
                "type": "object",
                "properties": {
                    "no_audio": {"type": "boolean"},
                    "style": {"type": "string"},
                    "emotion": {"type": "string"},
                    "delivery": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                },
                "required": ["no_audio", "style", "emotion", "delivery", "confidence"],
            },
        },
        "required": ["no_audio", "text", "expressive_transcript", "music", "tone"],
    }


def _extract_output_text(payload: dict) -> str:
    if isinstance(payload.get("output_text"), str) and payload["output_text"].strip():
        return str(payload["output_text"])
    outputs = payload.get("outputs") or []
    for item in outputs:
        if item.get("type") == "text":
            return str(item.get("text") or "")
        if item.get("type") in {"json", "structured_data"} and item.get("json") is not None:
            return json.dumps(item.get("json"))
        if item.get("type") in {"json", "structured_data"} and item.get("data") is not None:
            return json.dumps(item.get("data"))
    for step in payload.get("steps") or []:
        if step.get("type") != "model_output":
            continue
        for item in step.get("content") or []:
            if item.get("type") == "text":
                return str(item.get("text") or "")
            if item.get("type") in {"json", "structured_data"} and item.get("json") is not None:
                return json.dumps(item.get("json"))
            if item.get("type") in {"json", "structured_data"} and item.get("data") is not None:
                return json.dumps(item.get("data"))
    return ""


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1] if "\n" in stripped else stripped
        if stripped.endswith("```"):
            stripped = stripped[:-3]
    return stripped.strip()


def _extract_structured_payload(payload: dict[str, Any]) -> dict[str, Any]:
    direct_json = _extract_json_object(payload)
    if direct_json is not None:
        return direct_json

    output_text = _extract_output_text(payload)
    if not output_text:
        raise GoogleGeminiError("Google Gemini transcription response did not contain parseable output.")
    try:
        parsed = json.loads(_strip_code_fences(output_text))
    except json.JSONDecodeError as exc:
        raise GoogleGeminiError(f"Google Gemini returned non-JSON transcription output: {output_text}") from exc
    if not isinstance(parsed, dict):
        raise GoogleGeminiError("Google Gemini transcription output was not a JSON object.")
    return parsed


def _extract_json_object(payload: Any) -> dict[str, Any] | None:
    if isinstance(payload, dict):
        if _looks_like_transcript_payload(payload):
            return payload
        for key in ("json", "data", "parsed", "response", "result"):
            if key in payload:
                found = _extract_json_object(payload[key])
                if found is not None:
                    return found
        for key in ("outputs", "steps", "content", "parts", "candidates"):
            if key in payload:
                found = _extract_json_object(payload[key])
                if found is not None:
                    return found
    if isinstance(payload, list):
        for item in payload:
            found = _extract_json_object(item)
            if found is not None:
                return found
    return None


def _looks_like_transcript_payload(payload: dict[str, Any]) -> bool:
    return {"text", "music", "tone"}.issubset(payload.keys())


def _normalize_transcript_payload(parsed: dict[str, Any]) -> dict[str, Any]:
    text = str(parsed.get("text") or "")
    expressive_transcript = str(parsed.get("expressive_transcript") or "")
    music_payload = dict(parsed.get("music") or {})
    tone_payload = dict(parsed.get("tone") or {})
    inferred_no_audio = (
        bool(parsed.get("no_audio"))
        or (
            not text.strip()
            and not expressive_transcript.strip()
            and not bool(music_payload.get("present"))
            and not any(str(music_payload.get(key) or "").strip() for key in ("type", "instrumentation", "energy", "description"))
            and not any(str(tone_payload.get(key) or "").strip() for key in ("style", "emotion", "delivery"))
            and float(tone_payload.get("confidence") or 0.0) == 0.0
        )
    )
    music_payload["no_audio"] = bool(music_payload.get("no_audio")) or inferred_no_audio
    tone_payload["no_audio"] = bool(tone_payload.get("no_audio")) or inferred_no_audio
    if inferred_no_audio:
        music_payload["present"] = False
        for key in ("type", "instrumentation", "energy", "description"):
            music_payload[key] = ""
        for key in ("style", "emotion", "delivery"):
            tone_payload[key] = ""
        tone_payload["confidence"] = 0.0
        text = ""
        expressive_transcript = ""
    elif not expressive_transcript.strip():
        expressive_transcript = text

    expressive_transcript = _normalize_expressive_tags(expressive_transcript)

    return {
        "no_audio": inferred_no_audio,
        "text": text,
        "expressive_transcript": expressive_transcript,
        "music": MusicMetadata(**music_payload),
        "tone": ToneMetadata(**tone_payload),
    }


ALLOWED_EXPRESSIVE_TAGS = {
    "laughter",
    "music",
    "applause",
    "speaker_change",
    "shouting",
    "crying",
    "gasp",
    "sigh",
    "slap_sound",
    "impact_sound",
    "beep",
    "unknown_sound",
    "unknown_speech_tone",
}


def _normalize_expressive_tags(value: str) -> str:
    normalized = value
    for raw_tag in _find_bracket_tokens(value):
        cleaned = raw_tag.strip().lower().replace(" ", "_")
        replacement = cleaned if cleaned in ALLOWED_EXPRESSIVE_TAGS else "unknown_sound"
        normalized = normalized.replace(f"[{raw_tag}]", f"[{replacement}]")
    return normalized.strip()


def _find_bracket_tokens(value: str) -> list[str]:
    tokens: list[str] = []
    current = ""
    inside = False
    for char in value:
        if char == "[":
            inside = True
            current = ""
            continue
        if char == "]" and inside:
            tokens.append(current)
            inside = False
            current = ""
            continue
        if inside:
            current += char
    return tokens
