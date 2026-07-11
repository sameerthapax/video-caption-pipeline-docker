from __future__ import annotations

import base64
import json
import logging
import mimetypes
from pathlib import Path
from typing import Any

try:
    import httpx
except ModuleNotFoundError:  # pragma: no cover - local test environments may not install runtime deps
    httpx = None
from pydantic import BaseModel

from services.async_limits import get_loop_semaphore
from worker.config.settings import settings

logger = logging.getLogger("video-caption-pipeline.worker")


class FireworksError(RuntimeError):
    pass


class FireworksResponseFormatError(FireworksError):
    def __init__(self, message: str, *, raw_text: str = "", payload: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.raw_text = raw_text
        self.payload = payload or {}


class FireworksConfig(BaseModel):
    api_key: str
    model: str = "accounts/fireworks/models/kimi-k2p6"
    base_url: str = "https://api.fireworks.ai/inference/v1"
    proxy_url: str = ""
    proxy_token: str = ""
    timeout_seconds: int = 90
    max_retries: int = 3
    reasoning_effort: str = "none"
    vision_max_tokens: int = 300
    caption_max_tokens: int = 180
    judge_max_tokens: int = 200


class FireworksClient:
    def __init__(self, config: FireworksConfig) -> None:
        self.config = config
        if httpx is None:
            raise RuntimeError("httpx is required to use FireworksClient.")
        self._client = httpx.AsyncClient(
            timeout=self.config.timeout_seconds,
            http2=True,
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=50),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def generate_text(
        self,
        *,
        prompt: str,
        system_prompt: str = "",
        image_paths: list[str] | None = None,
        temperature: float | None = 0.1,
        endpoint_path: str = "/chat/completions",
    ) -> str:
        payload = await self._build_payload(
            prompt=prompt,
            system_prompt=system_prompt,
            image_paths=image_paths or [],
            temperature=temperature,
            response_schema=None,
            response_schema_name="response",
            max_tokens=self._max_tokens_for_endpoint(endpoint_path),
        )
        response_payload = await self._request(payload=payload, endpoint_path=endpoint_path)
        output_text = _extract_output_text(response_payload).strip()
        if not output_text:
            raise FireworksResponseFormatError(
                "Fireworks returned an empty text response.",
                payload=response_payload,
            )
        return _strip_code_fences(output_text)

    async def generate_json(
        self,
        *,
        prompt: str,
        system_prompt: str = "",
        image_paths: list[str] | None = None,
        temperature: float | None = 0.1,
        response_schema: dict[str, Any] | None = None,
        response_schema_name: str = "response",
        endpoint_path: str = "/chat/completions",
    ) -> dict[str, Any]:
        payload = await self._build_payload(
            prompt=prompt,
            system_prompt=system_prompt,
            image_paths=image_paths or [],
            temperature=temperature,
            response_schema=response_schema,
            response_schema_name=response_schema_name,
            max_tokens=self._max_tokens_for_endpoint(endpoint_path),
        )
        response_payload = await self._request(payload=payload, endpoint_path=endpoint_path)
        parsed = _extract_json_object(response_payload)
        if parsed is not None:
            return parsed
        output_text = _extract_output_text(response_payload)
        if not output_text:
            raise FireworksResponseFormatError(
                f"Fireworks response did not contain parseable JSON for {response_schema_name}.",
                payload=response_payload,
            )
        try:
            parsed = json.loads(_extract_json_substring(_strip_code_fences(output_text)))
        except json.JSONDecodeError as exc:
            raise FireworksResponseFormatError(
                f"Fireworks returned non-JSON output for {response_schema_name}.",
                raw_text=output_text,
                payload=response_payload,
            ) from exc
        if not isinstance(parsed, dict):
            raise FireworksResponseFormatError(
                f"Fireworks returned JSON that was not an object for {response_schema_name}.",
                raw_text=output_text,
                payload=response_payload,
            )
        return parsed

    async def _build_payload(
        self,
        *,
        prompt: str,
        system_prompt: str,
        image_paths: list[str],
        temperature: float | None,
        response_schema: dict[str, Any] | None,
        response_schema_name: str,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        messages: list[dict[str, Any]] = []
        if system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})
        messages.append(
            {
                "role": "user",
                "content": await self._build_user_content(prompt=prompt, image_paths=image_paths),
            }
        )

        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if self.config.reasoning_effort:
            payload["reasoning_effort"] = self.config.reasoning_effort
        if response_schema is not None:
            payload["response_format"] = _build_response_format(schema=response_schema, schema_name=response_schema_name)
        return payload

    def _max_tokens_for_endpoint(self, endpoint_path: str) -> int:
        if endpoint_path.startswith("/vision"):
            return self.config.vision_max_tokens
        if endpoint_path.startswith("/judge"):
            return self.config.judge_max_tokens
        return self.config.caption_max_tokens

    async def _build_user_content(self, *, prompt: str, image_paths: list[str]) -> str | list[dict[str, Any]]:
        if not image_paths:
            return prompt.strip()
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt.strip()}]
        for image_path in image_paths:
            content.append(await self._build_image_part(Path(image_path)))
        return content

    async def _build_image_part(self, image_path: Path) -> dict[str, Any]:
        if not image_path.exists():
            raise FireworksError(f"Image path does not exist: {image_path}")
        mime_type = mimetypes.guess_type(str(image_path))[0] or "image/jpeg"
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime_type};base64,{encoded}",
            },
        }

    async def _request(self, *, payload: dict[str, Any], endpoint_path: str) -> dict[str, Any]:
        if self.config.proxy_url:
            endpoint = f"{self.config.proxy_url.rstrip('/')}{endpoint_path}"
        else:
            endpoint = f"{self.config.base_url.rstrip('/')}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.config.proxy_url:
            if self.config.proxy_token:
                headers["X-Proxy-Token"] = self.config.proxy_token
        else:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        last_error: Exception | None = None
        semaphore = get_loop_semaphore(
            name="fireworks_requests",
            limit=max(1, settings.max_concurrent_jobs),
        )
        for attempt in range(1, self.config.max_retries + 1):
            try:
                if settings.log_model_io:
                    logger.info("Fireworks request payload: %s", json.dumps(_truncate_json_payload(payload), ensure_ascii=False))
                async with semaphore:
                    response = await self._client.post(endpoint, headers=headers, json=payload)
                if response.status_code >= 400:
                    raise FireworksError(
                        f"Fireworks request failed for endpoint {endpoint_path} with status {response.status_code}: {response.text[:500]}"
                    )
                response_payload = response.json()
                if settings.log_model_io:
                    logger.info("Fireworks raw response: %s", json.dumps(_truncate_json_payload(response_payload), ensure_ascii=False))
                return response_payload
            except (httpx.HTTPError, ValueError, FireworksError) as exc:
                last_error = exc
                logger.warning("Fireworks request attempt %s/%s failed: %s", attempt, self.config.max_retries, exc)
                if attempt >= self.config.max_retries:
                    break
                await _sleep_backoff(attempt)
        raise FireworksError(str(last_error) if last_error else "Fireworks request failed.")


def _extract_output_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    for choice in choices:
        message = choice.get("message") or {}
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content
        if isinstance(content, list):
            text_parts: list[str] = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
                    text_parts.append(str(part["text"]))
            if text_parts:
                return "".join(text_parts)
    return ""


def _extract_json_object(payload: Any) -> dict[str, Any] | None:
    if isinstance(payload, dict):
        if _looks_like_payload_object(payload):
            return payload
        for key in ("json", "data", "parsed", "response", "result", "message", "content"):
            if key in payload:
                found = _extract_json_object(payload[key])
                if found is not None:
                    return found
        if "choices" in payload:
            found = _extract_json_object(payload["choices"])
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


def _looks_like_payload_object(payload: dict[str, Any]) -> bool:
    if isinstance(payload.get("style_name"), str) and isinstance(payload.get("caption"), str):
        return True
    interesting_keys = {"summary", "score", "feedback", "formal", "sarcastic", "humorous_tech", "humorous_non_tech"}
    return any(key in payload for key in interesting_keys)


def _truncate_json_payload(payload: Any, *, max_string_length: int = 2000) -> Any:
    if isinstance(payload, dict):
        return {key: _truncate_json_payload(value, max_string_length=max_string_length) for key, value in payload.items()}
    if isinstance(payload, list):
        return [_truncate_json_payload(item, max_string_length=max_string_length) for item in payload]
    if isinstance(payload, str) and len(payload) > max_string_length:
        return payload[:max_string_length] + "...<truncated>"
    return payload


def _build_response_format(*, schema: dict[str, Any], schema_name: str) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema_name,
            "strict": True,
            "schema": schema,
        },
    }


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


async def _sleep_backoff(attempt: int) -> None:
    import asyncio

    await asyncio.sleep(min(2 ** (attempt - 1), 8))
