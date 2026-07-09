from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel

from services.async_limits import get_loop_semaphore
from worker.config.settings import settings

logger = logging.getLogger("video-caption-pipeline.worker")


class FireworksError(RuntimeError):
    pass


class FireworksNonRetryableError(FireworksError):
    pass


class FireworksResponseFormatError(FireworksError):
    def __init__(self, message: str, *, raw_text: str = "", payload: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.raw_text = raw_text
        self.payload = payload or {}


class FireworksConfig(BaseModel):
    base_url: str = "https://api.fireworks.ai/inference/v1"
    api_key: str
    timeout_seconds: int = 90
    max_retries: int = 3


class FireworksClient:
    def __init__(self, config: FireworksConfig) -> None:
        self.config = config
        self._client = httpx.AsyncClient(timeout=self.config.timeout_seconds)
        self._image_uri_cache: dict[str, str] = {}

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> FireworksClient:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    def build_transcription_request_metadata(self) -> dict[str, str]:
        return {
            "provider": "fireworks",
            "api_style": "openai_compatible",
            "base_url": self.config.base_url,
            "model": "",
        }

    async def analyze_segment_with_images(
        self,
        *,
        model: str,
        prompt: str,
        image_paths: list[str],
        temperature: float = 0.1,
    ) -> dict:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for image_path in image_paths:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": self._image_data_uri(Path(image_path))},
                }
            )
        payload = {
            "model": model,
            "temperature": temperature,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": content}],
        }
        return await self._request_json(payload=payload)

    async def generate_json(
        self,
        *,
        model: str,
        prompt: str,
        temperature: float = 0.1,
    ) -> dict:
        payload = {
            "model": model,
            "temperature": temperature,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        }
        return await self._request_json(payload=payload)

    async def _request_json(self, *, payload: dict[str, Any]) -> dict:
        endpoint = f"{self.config.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        model_name = str(payload.get("model") or "")
        last_error: Exception | None = None
        semaphore = get_loop_semaphore(
            name="fireworks_requests",
            limit=settings.fireworks_max_concurrency,
        )
        for attempt in range(1, self.config.max_retries + 1):
            try:
                async with semaphore:
                    response = await self._client.post(endpoint, headers=headers, json=payload)
                if response.status_code >= 400:
                    if response.status_code == 404:
                        raise FireworksNonRetryableError(
                            f"Fireworks model not found or not accessible: {model_name}. Response: {response.text[:500]}"
                        )
                    raise FireworksError(
                        f"Fireworks request failed for model {model_name} with status {response.status_code}: {response.text[:500]}"
                    )
                response_payload = response.json()
                parsed = _extract_json_object(response_payload)
                if parsed is not None:
                    return parsed
                output_text = _extract_output_text(response_payload)
                if not output_text:
                    raise FireworksResponseFormatError(
                        "Fireworks response did not contain parseable output.",
                        payload=response_payload,
                    )
                try:
                    parsed_text = json.loads(_strip_code_fences(output_text))
                except json.JSONDecodeError as exc:
                    raise FireworksResponseFormatError(
                        "Fireworks returned non-JSON output.",
                        raw_text=output_text,
                        payload=response_payload,
                    ) from exc
                if not isinstance(parsed_text, dict):
                    raise FireworksResponseFormatError(
                        "Fireworks returned JSON that was not an object.",
                        raw_text=output_text,
                        payload=response_payload,
                    )
                return parsed_text
            except (httpx.HTTPError, FireworksError, FireworksResponseFormatError) as exc:
                last_error = exc
                logger.warning("Fireworks request attempt %s/%s failed: %s", attempt, self.config.max_retries, exc)
                if (
                    attempt >= self.config.max_retries
                    or isinstance(exc, FireworksResponseFormatError)
                    or isinstance(exc, FireworksNonRetryableError)
                ):
                    break
                await _sleep_backoff(attempt)
        if isinstance(last_error, FireworksResponseFormatError):
            raise last_error
        if isinstance(last_error, FireworksNonRetryableError):
            raise last_error
        raise FireworksError(str(last_error) if last_error else "Fireworks request failed.")

    def _image_data_uri(self, path: Path) -> str:
        cache_key = str(path.resolve())
        cached = self._image_uri_cache.get(cache_key)
        if cached is not None:
            return cached
        encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
        data_uri = f"data:image/jpeg;base64,{encoded}"
        self._image_uri_cache[cache_key] = data_uri
        return data_uri


async def _sleep_backoff(attempt: int) -> None:
    import asyncio

    await asyncio.sleep(min(2 ** (attempt - 1), 8))


def _extract_output_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    for choice in choices:
        message = choice.get("message") or {}
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content
        if isinstance(content, list):
            for item in content:
                if item.get("type") == "text" and item.get("text"):
                    return str(item["text"])
    if isinstance(payload.get("output_text"), str):
        return str(payload["output_text"])
    return ""


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1] if "\n" in stripped else stripped
        if stripped.endswith("```"):
            stripped = stripped[:-3]
    return stripped.strip()


def _extract_json_object(payload: Any) -> dict[str, Any] | None:
    if isinstance(payload, dict):
        if _looks_like_object_payload(payload):
            return payload
        for key in ("json", "data", "parsed", "response", "result"):
            if key in payload:
                found = _extract_json_object(payload[key])
                if found is not None:
                    return found
        for key in ("choices", "message", "content", "tool_calls"):
            if key in payload:
                found = _extract_json_object(payload[key])
                if found is not None:
                    return found
    if isinstance(payload, list):
        for item in payload:
            found = _extract_json_object(item)
            if found is not None:
                return found
    if isinstance(payload, str):
        try:
            parsed = json.loads(_strip_code_fences(payload))
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, dict):
            return parsed
    return None


def _looks_like_object_payload(payload: dict[str, Any]) -> bool:
    if "segment_index" in payload and "segment_description" in payload:
        return True
    if "factual_summary" in payload and "detailed_timeline" in payload:
        return True
    return False
