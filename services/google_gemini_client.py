from __future__ import annotations

import base64
import json
import logging
from typing import Any
from pathlib import Path

import httpx
from pydantic import BaseModel

from services.async_limits import get_loop_semaphore
from schemas.transcription import MusicMetadata, ToneMetadata, TranscriptChunk
from worker.config.settings import settings

logger = logging.getLogger("video-caption-pipeline.worker")


class GoogleGeminiError(RuntimeError):
    pass


class GoogleGeminiResponseFormatError(GoogleGeminiError):
    def __init__(self, message: str, *, raw_text: str = "", payload: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.raw_text = raw_text
        self.payload = payload or {}


class GoogleGeminiConfig(BaseModel):
    api_key: str
    base_url: str = "https://generativelanguage.googleapis.com"
    proxy_url: str = ""
    proxy_token: str = ""
    model: str = "gemini-3.5-flash"
    timeout_seconds: int = 60
    max_retries: int = 3


class GoogleGeminiClient:
    def __init__(self, config: GoogleGeminiConfig) -> None:
        self.config = config
        self._client = httpx.AsyncClient(
            timeout=self.config.timeout_seconds,
            http2=True,
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=50),
        )

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
            "base_url": self.config.proxy_url or self.config.base_url,
            "model": self.config.model,
        }

    async def analyze_segment_with_images(
        self,
        *,
        model: str,
        system_prompt: str,
        prompt: str,
        image_paths: list[str],
        temperature: float = 0.1,
        response_schema: dict[str, Any] | None = None,
        response_schema_name: str = "response",
    ) -> dict[str, Any]:
        input_parts: list[dict[str, Any]] = [{"type": "text", "text": prompt.strip()}]
        for image_path in image_paths:
            input_parts.append(_image_input_part(image_path))
        return await self._request_structured_output(
            model=model,
            input_parts=input_parts,
            system_prompt=system_prompt,
            temperature=temperature,
            response_schema=response_schema,
            response_schema_name=response_schema_name,
        )

    async def generate_json(
        self,
        *,
        model: str,
        prompt: str,
        temperature: float = 0.0,
        response_schema: dict[str, Any] | None = None,
        response_schema_name: str = "response",
    ) -> dict[str, Any]:
        return await self._request_structured_output(
            model=model,
            input_parts=[{"type": "text", "text": prompt.strip()}],
            system_prompt="",
            temperature=temperature,
            response_schema=response_schema,
            response_schema_name=response_schema_name,
        )

    async def transcribe_audio_window(
        self,
        *,
        audio_bytes: bytes,
        start: float,
        end: float,
    ) -> TranscriptChunk:
        payload = {
            "model": self.config.model,
            "input": [
                {"type": "text", "text": _build_transcription_prompt(start=start, end=end)},
                {
                    "type": "audio",
                    "data": base64.b64encode(audio_bytes).decode("utf-8"),
                    "mime_type": "audio/wav",
                },
            ],
            "response_format": _transcript_response_schema(),
        }
        headers = {
            "Content-Type": "application/json",
        }
        if self.config.proxy_url:
            endpoint = self.config.proxy_url
            if self.config.proxy_token:
                headers["X-Proxy-Token"] = self.config.proxy_token
        else:
            headers["x-goog-api-key"] = self.config.api_key
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

    async def _request_structured_output(
        self,
        *,
        model: str,
        input_parts: list[dict[str, Any]],
        system_prompt: str,
        temperature: float,
        response_schema: dict[str, Any] | None,
        response_schema_name: str,
    ) -> dict[str, Any]:
        payload = _build_generate_content_payload(
            system_prompt=system_prompt,
            input_parts=input_parts,
            temperature=temperature,
        )
        headers = {
            "Content-Type": "application/json",
        }
        if self.config.proxy_url:
            endpoint = _build_generate_content_proxy_endpoint(self.config.proxy_url, model=model)
            if self.config.proxy_token:
                headers["X-Proxy-Token"] = self.config.proxy_token
        else:
            headers["x-goog-api-key"] = self.config.api_key
            endpoint = f"{self.config.base_url.rstrip('/')}/v1beta/models/{model}:generateContent"
        semaphore = get_loop_semaphore(
            name="google_gemini_requests",
            limit=settings.google_gemini_max_concurrency,
        )

        last_error: Exception | None = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
                if settings.log_model_io:
                    logger.info(
                        "Google Gemini request payload: %s",
                        json.dumps(_sanitize_request_payload(payload), ensure_ascii=False),
                    )
                async with semaphore:
                    response = await self._client.post(endpoint, headers=headers, json=payload)
                if response.status_code >= 400:
                    raise GoogleGeminiError(response.text or f"Google Gemini request failed for model {model}.")
                try:
                    response_payload = response.json()
                except ValueError as exc:
                    raw_text = response.text[:12000]
                    logger.warning(
                        "Google Gemini returned non-JSON HTTP 200 for model %s: %s",
                        model,
                        raw_text,
                    )
                    raise GoogleGeminiResponseFormatError(
                        "Google Gemini returned a non-JSON HTTP 200 response.",
                        raw_text=raw_text,
                    ) from exc
                if settings.log_model_io:
                    logger.info(
                        "Google Gemini raw response: %s",
                        json.dumps(response_payload, ensure_ascii=False)[:12000],
                    )
                try:
                    parsed = _extract_structured_payload(response_payload, expected_schema=response_schema)
                except GoogleGeminiResponseFormatError as exc:
                    logger.warning(
                        "Google Gemini parse failure for model %s. Raw response: %s",
                        model,
                        json.dumps(response_payload, ensure_ascii=False)[:12000],
                    )
                    raise GoogleGeminiResponseFormatError(
                        str(exc),
                        raw_text=exc.raw_text or _extract_output_text(response_payload),
                        payload=response_payload,
                    ) from exc
                if settings.log_model_io:
                    logger.info(
                        "Google Gemini parsed response: %s",
                        json.dumps(parsed, ensure_ascii=False),
                    )
                return parsed
            except (httpx.HTTPError, GoogleGeminiError) as exc:
                last_error = exc
                logger.warning(
                    "Gemini structured request failed for model %s attempt %s/%s: %s",
                    model,
                    attempt,
                    self.config.max_retries,
                    exc,
                )
                if attempt >= self.config.max_retries or isinstance(exc, GoogleGeminiResponseFormatError):
                    break
                await _sleep_backoff(attempt)

        if isinstance(last_error, GoogleGeminiError):
            raise last_error
        raise GoogleGeminiError(str(last_error) if last_error else "Google Gemini structured request failed.")


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


def _build_generate_content_proxy_endpoint(proxy_url: str, *, model: str) -> str:
    trimmed = proxy_url.rstrip("/")
    if "/v1beta/interactions" in trimmed:
        base = trimmed.rsplit("/v1beta/interactions", 1)[0]
        return f"{base}/v1beta/models/{model}:generateContent"
    if trimmed.endswith("/google"):
        return f"{trimmed}/v1beta/models/{model}:generateContent"
    if "/v1beta/models/" in trimmed and trimmed.endswith(":generateContent"):
        return trimmed
    return f"{trimmed}/v1beta/models/{model}:generateContent"


def _build_generate_content_payload(
    *,
    system_prompt: str,
    input_parts: list[dict[str, Any]],
    temperature: float,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "contents": [
            {
                "role": "user",
                "parts": [_to_generate_content_part(part) for part in input_parts],
            }
        ],
        "generationConfig": {
            "temperature": temperature,
        },
    }
    if system_prompt.strip():
        payload["systemInstruction"] = {
            "parts": [{"text": system_prompt.strip()}],
        }
    return payload


def _to_generate_content_part(part: dict[str, Any]) -> dict[str, Any]:
    part_type = str(part.get("type") or "")
    if part_type == "text":
        return {"text": str(part.get("text") or "")}
    if part_type == "image":
        return {
            "inline_data": {
                "mime_type": str(part.get("mime_type") or "image/jpeg"),
                "data": str(part.get("data") or ""),
            }
        }
    if part_type == "audio":
        return {
            "inline_data": {
                "mime_type": str(part.get("mime_type") or "audio/wav"),
                "data": str(part.get("data") or ""),
            }
        }
    return {"text": json.dumps(part, ensure_ascii=False)}


def _extract_output_text(payload: dict) -> str:
    if isinstance(payload.get("output_text"), str) and payload["output_text"].strip():
        return str(payload["output_text"])
    if isinstance(payload.get("text"), str) and payload["text"].strip():
        return str(payload["text"])
    if isinstance(payload.get("content"), str) and payload["content"].strip():
        return str(payload["content"])
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
    candidates = payload.get("candidates") or []
    for candidate in candidates:
        content = candidate.get("content") or {}
        if isinstance(content, dict):
            parts = content.get("parts") or []
            for part in parts:
                if isinstance(part, dict):
                    if isinstance(part.get("text"), str) and part["text"].strip():
                        return str(part["text"])
                    if part.get("inline_data") is not None:
                        inline_data = part.get("inline_data") or {}
                        if inline_data.get("data") is not None:
                            return json.dumps(inline_data.get("data"))
    for key in ("output", "response", "result", "data"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value
        if isinstance(value, dict):
            nested = _extract_output_text(value)
            if nested:
                return nested
    return ""


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1] if "\n" in stripped else stripped
        if stripped.endswith("```"):
            stripped = stripped[:-3]
    return stripped.strip()


def _extract_structured_payload(payload: dict[str, Any], expected_schema: dict[str, Any] | None = None) -> dict[str, Any]:
    direct_json = _extract_json_object(payload, expected_schema=expected_schema)
    if direct_json is not None:
        return direct_json

    output_text = _extract_output_text(payload)
    if not output_text:
        raise GoogleGeminiResponseFormatError(
            "Google Gemini response did not contain parseable output.",
            payload=payload,
        )
    try:
        parsed = json.loads(_strip_code_fences(output_text))
    except json.JSONDecodeError as exc:
        raise GoogleGeminiResponseFormatError(
            "Google Gemini returned non-JSON output.",
            raw_text=output_text,
            payload=payload,
        ) from exc
    if not isinstance(parsed, dict):
        raise GoogleGeminiResponseFormatError(
            "Google Gemini output was not a JSON object.",
            raw_text=output_text,
            payload=payload,
        )
    return parsed


def _extract_json_object(payload: Any, expected_schema: dict[str, Any] | None = None) -> dict[str, Any] | None:
    if isinstance(payload, dict):
        if _looks_like_expected_payload(payload, expected_schema):
            return payload
        for key in ("json", "data", "parsed", "response", "result", "output", "message", "content", "inline_data"):
            if key in payload:
                found = _extract_json_object(payload[key], expected_schema=expected_schema)
                if found is not None:
                    return found
        for key in ("outputs", "steps", "content", "parts", "candidates"):
            if key in payload:
                found = _extract_json_object(payload[key], expected_schema=expected_schema)
                if found is not None:
                    return found
    if isinstance(payload, list):
        for item in payload:
            found = _extract_json_object(item, expected_schema=expected_schema)
            if found is not None:
                return found
    if isinstance(payload, str):
        try:
            parsed = json.loads(_extract_json_substring(_strip_code_fences(payload)))
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, dict):
            if _looks_like_expected_payload(parsed, expected_schema) or not expected_schema:
                return parsed
    return None


def _looks_like_expected_payload(payload: dict[str, Any], expected_schema: dict[str, Any] | None) -> bool:
    if expected_schema and isinstance(expected_schema.get("properties"), dict):
        expected_keys = set(expected_schema["properties"].keys())
        return bool(expected_keys) and expected_keys.issubset(payload.keys())
    return {"text", "music", "tone"}.issubset(payload.keys())


def _image_input_part(image_path: str) -> dict[str, Any]:
    encoded = base64.b64encode(Path(image_path).read_bytes()).decode("utf-8")
    return {
        "type": "image",
        "data": encoded,
        "mime_type": "image/jpeg",
    }


def _sanitize_request_payload(payload: dict[str, Any]) -> dict[str, Any]:
    sanitized = dict(payload)
    if "input" in sanitized:
        sanitized_input: list[dict[str, Any]] = []
        for item in sanitized.get("input", []):
            if item.get("type") == "image":
                sanitized_input.append(
                    {
                        "type": "image",
                        "mime_type": item.get("mime_type", "image/jpeg"),
                        "data": f"<base64 length={len(item.get('data', ''))}>",
                    }
                )
            elif item.get("type") == "audio":
                sanitized_input.append(
                    {
                        "type": "audio",
                        "mime_type": item.get("mime_type", "audio/wav"),
                        "data": f"<base64 length={len(item.get('data', ''))}>",
                    }
                )
            else:
                sanitized_input.append(item)
        sanitized["input"] = sanitized_input

    if "contents" in sanitized:
        sanitized_contents: list[dict[str, Any]] = []
        for content in sanitized.get("contents", []):
            if not isinstance(content, dict):
                sanitized_contents.append(content)
                continue
            sanitized_parts: list[dict[str, Any]] = []
            for part in content.get("parts", []):
                if not isinstance(part, dict):
                    sanitized_parts.append(part)
                    continue
                inline_data = part.get("inline_data")
                if isinstance(inline_data, dict):
                    sanitized_parts.append(
                        {
                            "inline_data": {
                                "mime_type": inline_data.get("mime_type", "application/octet-stream"),
                                "data": f"<base64 length={len(str(inline_data.get('data', '')))}>",
                            }
                        }
                    )
                else:
                    sanitized_parts.append(part)
            sanitized_contents.append(
                {
                    **content,
                    "parts": sanitized_parts,
                }
            )
        sanitized["contents"] = sanitized_contents
    return sanitized


def _extract_json_substring(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


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
