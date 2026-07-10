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
    proxy_url: str = ""
    proxy_token: str = ""
    timeout_seconds: int = 90
    max_retries: int = 3
    temperature: float = 0.2
    reasoning_effort: str = "medium"
    text_verbosity: str = "medium"


class OpenAIResponsesClient:
    def __init__(self, config: OpenAIResponsesConfig) -> None:
        self.config = config
        self._client = httpx.AsyncClient(
            timeout=self.config.timeout_seconds,
            http2=True,
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=50),
        )

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
            "temperature": self.config.temperature,
            "text": {"verbosity": self.config.text_verbosity},
        }
        return await self._request_json(payload=payload)

    async def _request_json(self, *, payload: dict[str, Any]) -> dict[str, Any]:
        endpoint = self.config.proxy_url or f"{self.config.base_url.rstrip('/')}/responses"
        headers = {"Content-Type": "application/json"}
        if self.config.proxy_url:
            if self.config.proxy_token:
                headers["X-Proxy-Token"] = self.config.proxy_token
        else:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        model_name = str(payload.get("model") or "")
        last_error: Exception | None = None
        semaphore = get_loop_semaphore(
            name="openai_responses_requests",
            limit=max(1, settings.max_concurrent_jobs),
        )
        for attempt in range(1, self.config.max_retries + 1):
            try:
                if settings.log_model_io:
                    logger.info(
                        "OpenAI request payload: %s",
                        json.dumps(_truncate_json_payload(payload), ensure_ascii=False),
                    )
                async with semaphore:
                    response = await self._client.post(endpoint, headers=headers, json=payload)
                if response.status_code >= 400:
                    raise OpenAIResponsesError(
                        f"OpenAI Responses request failed for model {model_name} with status {response.status_code}: {response.text[:500]}"
                    )
                response_payload = response.json()
                if settings.log_model_io:
                    logger.info(
                        "OpenAI raw response: %s",
                        json.dumps(_truncate_json_payload(response_payload), ensure_ascii=False),
                    )
                parsed = _extract_json_object(response_payload)
                if parsed is not None:
                    if settings.log_model_io:
                        logger.info(
                            "OpenAI parsed response: %s",
                            json.dumps(parsed, ensure_ascii=False),
                        )
                    return parsed
                output_text = _extract_output_text(response_payload)
                if not output_text:
                    raise OpenAIResponsesResponseFormatError(
                        "OpenAI response did not contain parseable output.",
                        payload=response_payload,
                    )
                try:
                    parsed_text = json.loads(_extract_json_substring(_strip_code_fences(output_text)))
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
                if settings.log_model_io:
                    logger.info(
                        "OpenAI parsed response: %s",
                        json.dumps(parsed_text, ensure_ascii=False),
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


def _extract_json_substring(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


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
            parsed = json.loads(_extract_json_substring(_strip_code_fences(payload)))
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, dict):
            return parsed
    return None


def _looks_like_caption_bundle(payload: dict[str, Any]) -> bool:
    captions = payload.get("captions")
    if not isinstance(captions, dict):
        return _looks_like_caption_variant(payload)
    required_styles = {"formal", "sarcastic", "humorous_tech", "humorous_non_tech"}
    return required_styles.issubset(captions.keys())


def _looks_like_caption_variant(payload: dict[str, Any]) -> bool:
    return isinstance(payload.get("style_name"), str) and isinstance(payload.get("caption"), str)


def _truncate_json_payload(payload: Any, *, max_string_length: int = 2000) -> Any:
    if isinstance(payload, dict):
        return {key: _truncate_json_payload(value, max_string_length=max_string_length) for key, value in payload.items()}
    if isinstance(payload, list):
        return [_truncate_json_payload(item, max_string_length=max_string_length) for item in payload]
    if isinstance(payload, str) and len(payload) > max_string_length:
        return payload[:max_string_length] + "...<truncated>"
    return payload
