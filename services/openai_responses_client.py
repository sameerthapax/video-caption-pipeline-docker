from __future__ import annotations

import json
import logging
from typing import Any

import httpx
from pydantic import BaseModel

from services.async_limits import get_loop_semaphore
from worker.config.settings import settings

logger = logging.getLogger("video-caption-pipeline.worker")


class OpenAIResponsesError(RuntimeError):
    pass


class OpenAIResponsesResponseFormatError(OpenAIResponsesError):
    def __init__(self, message: str, *, raw_text: str = "", payload: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.raw_text = raw_text
        self.payload = payload or {}


class OpenAIResponsesConfig(BaseModel):
    api_key: str
    base_url: str = "https://api.openai.com/v1"
    timeout_seconds: int = 90
    max_retries: int = 3
    reasoning_effort: str = "medium"
    text_verbosity: str = "medium"


class OpenAIResponsesClient:
    def __init__(self, config: OpenAIResponsesConfig) -> None:
        self.config = config
        self._client = httpx.AsyncClient(timeout=self.config.timeout_seconds)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> OpenAIResponsesClient:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    async def generate_json(
        self,
        *,
        model: str,
        prompt: str,
    ) -> dict[str, Any]:
        payload = {
            "model": model,
            "input": prompt,
            "reasoning": {"effort": self.config.reasoning_effort},
            "text": {"verbosity": self.config.text_verbosity},
        }
        return await self._request_json(payload=payload)

    async def _request_json(self, *, payload: dict[str, Any]) -> dict[str, Any]:
        endpoint = f"{self.config.base_url.rstrip('/')}/responses"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        model_name = str(payload.get("model") or "")
        last_error: Exception | None = None
        semaphore = get_loop_semaphore(
            name="openai_responses_requests",
            limit=max(1, settings.max_concurrent_jobs),
        )
        for attempt in range(1, self.config.max_retries + 1):
            try:
                async with semaphore:
                    response = await self._client.post(endpoint, headers=headers, json=payload)
                if response.status_code >= 400:
                    raise OpenAIResponsesError(
                        f"OpenAI Responses request failed for model {model_name} with status {response.status_code}: {response.text[:500]}"
                    )
                response_payload = response.json()
                parsed = _extract_json_object(response_payload)
                if parsed is not None:
                    return parsed
                output_text = _extract_output_text(response_payload)
                if not output_text:
                    raise OpenAIResponsesResponseFormatError(
                        "OpenAI response did not contain parseable output.",
                        payload=response_payload,
                    )
                try:
                    parsed_text = json.loads(_strip_code_fences(output_text))
                except json.JSONDecodeError as exc:
                    raise OpenAIResponsesResponseFormatError(
                        "OpenAI returned non-JSON output.",
                        raw_text=output_text,
                        payload=response_payload,
                    ) from exc
                if not isinstance(parsed_text, dict):
                    raise OpenAIResponsesResponseFormatError(
                        "OpenAI returned JSON that was not an object.",
                        raw_text=output_text,
                        payload=response_payload,
                    )
                return parsed_text
            except (httpx.HTTPError, OpenAIResponsesError, OpenAIResponsesResponseFormatError) as exc:
                last_error = exc
                logger.warning("OpenAI response request attempt %s/%s failed: %s", attempt, self.config.max_retries, exc)
                if attempt >= self.config.max_retries or isinstance(exc, OpenAIResponsesResponseFormatError):
                    break
                await _sleep_backoff(attempt)
        if isinstance(last_error, OpenAIResponsesResponseFormatError):
            raise last_error
        raise OpenAIResponsesError(str(last_error) if last_error else "OpenAI Responses request failed.")


async def _sleep_backoff(attempt: int) -> None:
    import asyncio

    await asyncio.sleep(min(2 ** (attempt - 1), 8))


def _extract_output_text(payload: dict[str, Any]) -> str:
    if isinstance(payload.get("output_text"), str) and payload["output_text"].strip():
        return str(payload["output_text"])
    output_items = payload.get("output") or []
    for item in output_items:
        content = item.get("content") or []
        for block in content:
            if block.get("type") in {"output_text", "text"} and block.get("text"):
                return str(block["text"])
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
        if _looks_like_caption_bundle(payload):
            return payload
        for key in ("json", "data", "parsed", "response", "result"):
            if key in payload:
                found = _extract_json_object(payload[key])
                if found is not None:
                    return found
        for key in ("output", "content"):
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


def _looks_like_caption_bundle(payload: dict[str, Any]) -> bool:
    captions = payload.get("captions")
    if not isinstance(captions, dict):
        return False
    required_styles = {"formal", "sarcastic", "humorous_tech", "humorous_non_tech"}
    return required_styles.issubset(captions.keys())
