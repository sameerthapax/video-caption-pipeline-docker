from __future__ import annotations

import asyncio
import ast
import base64
import json
import logging
import mimetypes
from pathlib import Path
import re
import time
from typing import Any

try:
    import httpx
except ModuleNotFoundError:  # pragma: no cover - local test environments may not install runtime deps
    httpx = None
from pydantic import BaseModel

from services.async_limits import get_loop_semaphore
from worker.config.settings import settings

logger = logging.getLogger("gemma-caption-pipe.worker")


class FireworksError(RuntimeError):
    pass


class FireworksResponseFormatError(FireworksError):
    def __init__(self, message: str, *, raw_text: str = "", payload: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.raw_text = raw_text
        self.payload = payload or {}


class FireworksConfig(BaseModel):
    provider_name: str = "fireworks"
    api_key: str
    model: str = "accounts/fireworks/models/kimi-k2p6"
    base_url: str = "https://api.fireworks.ai/inference/v1"
    proxy_url: str = ""
    proxy_token: str = ""
    timeout_seconds: int = 120
    queue_timeout_seconds: int = 60
    connect_timeout_seconds: int = 15
    write_timeout_seconds: int = 60
    pool_timeout_seconds: int = 15
    max_retries: int = 2
    reasoning_effort: str = "none"
    vision_max_tokens: int = 20_000
    caption_max_tokens: int = 20_000
    judge_max_tokens: int = 20_000


class FireworksClient:
    def __init__(self, config: FireworksConfig) -> None:
        self.config = config
        if httpx is None:
            raise RuntimeError("httpx is required to use FireworksClient.")
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=self.config.connect_timeout_seconds,
                read=self.config.timeout_seconds,
                write=self.config.write_timeout_seconds,
                pool=self.config.pool_timeout_seconds,
            ),
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
        try:
            response_payload = await self._request(payload=payload, endpoint_path=endpoint_path)
        except FireworksError as exc:
            fallback_payload = await self._fallback_unstructured_payload(
                error=exc,
                prompt=prompt,
                system_prompt=system_prompt,
                image_paths=image_paths or [],
                temperature=temperature,
                response_schema=response_schema,
                response_schema_name=response_schema_name,
                max_tokens=self._max_tokens_for_endpoint(endpoint_path),
                endpoint_path=endpoint_path,
            )
            if fallback_payload is None:
                raise
            response_payload = fallback_payload
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
            logger.warning(
                "Structured output parse failed for %s near char %s: %s | excerpt=%s",
                response_schema_name,
                exc.pos,
                exc.msg,
                _safe_excerpt(output_text),
            )
            heuristic = _heuristic_repair_json_text(output_text)
            if heuristic is not None:
                return heuristic
            repaired = await self._repair_json_output(
                raw_text=output_text,
                response_schema=response_schema,
                response_schema_name=response_schema_name,
                endpoint_path=endpoint_path,
            )
            if repaired is not None:
                return repaired
            raise FireworksResponseFormatError(
                f"Fireworks returned non-JSON output for {response_schema_name}.",
                raw_text=output_text,
                payload=response_payload,
            ) from exc
        if not isinstance(parsed, dict):
            logger.warning(
                "Structured output for %s parsed but was not an object. excerpt=%s",
                response_schema_name,
                _safe_excerpt(output_text),
            )
            heuristic = _heuristic_repair_json_text(output_text)
            if heuristic is not None:
                return heuristic
            repaired = await self._repair_json_output(
                raw_text=output_text,
                response_schema=response_schema,
                response_schema_name=response_schema_name,
                endpoint_path=endpoint_path,
            )
            if repaired is not None:
                return repaired
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
        if self._should_send_reasoning_effort():
            payload["reasoning_effort"] = self.config.reasoning_effort
        if response_schema is not None:
            payload["response_format"] = _build_response_format(schema=response_schema, schema_name=response_schema_name)
        if self.config.provider_name == "openrouter":
            payload["provider"] = {
                "require_parameters": True,
            }
        return payload

    def _should_send_reasoning_effort(self) -> bool:
        if not self.config.reasoning_effort:
            return False
        if self.config.provider_name == "openrouter" and self.config.reasoning_effort == "none":
            return False
        return True

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
        provider_name = (self.config.provider_name or "llm").strip()
        provider_label = provider_name.capitalize()
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
            name=f"{provider_name}_requests",
            limit=max(1, settings.max_concurrent_jobs),
        )
        for attempt in range(1, self.config.max_retries + 1):
            try:
                if settings.log_model_io:
                    logger.info("%s request payload: %s", provider_label, json.dumps(_sanitize_payload_for_logging(payload), ensure_ascii=False))
                response = await self._post_with_limit(
                    semaphore=semaphore,
                    endpoint=endpoint,
                    endpoint_path=endpoint_path,
                    headers=headers,
                    payload=payload,
                    provider_label=provider_label,
                    attempt=attempt,
                )
                if response.status_code >= 400:
                    if response.status_code == 400:
                        retry_payload = _payload_without_deprecated_temperature(
                            provider_name=provider_name,
                            payload=payload,
                            response_text=response.text,
                        )
                        if retry_payload is not None:
                            logger.warning("%s rejected temperature for endpoint %s. Retrying once without temperature.", provider_label, endpoint_path)
                            retry_response = await self._post_with_limit(
                                semaphore=semaphore,
                                endpoint=endpoint,
                                endpoint_path=endpoint_path,
                                headers=headers,
                                payload=retry_payload,
                                provider_label=provider_label,
                                attempt=attempt,
                            )
                            if retry_response.status_code < 400:
                                retry_payload_json = retry_response.json()
                                if settings.log_model_io:
                                    logger.info("%s raw response: %s", provider_label, json.dumps(retry_payload_json, ensure_ascii=False))
                                return retry_payload_json
                    raise FireworksError(
                        f"{provider_label} request failed for endpoint {endpoint_path} with status {response.status_code}: {response.text[:500]}"
                    )
                response_payload = response.json()
                if settings.log_model_io:
                    logger.info("%s raw response: %s", provider_label, json.dumps(response_payload, ensure_ascii=False))
                return response_payload
            except (httpx.HTTPError, ValueError, FireworksError) as exc:
                last_error = exc
                logger.warning("%s request attempt %s/%s failed: %s", provider_label, attempt, self.config.max_retries, exc)
                if attempt >= self.config.max_retries:
                    break
                await _sleep_backoff(attempt)
        raise FireworksError(str(last_error) if last_error else f"{provider_label} request failed.")

    async def _post_with_limit(
        self,
        *,
        semaphore: asyncio.Semaphore,
        endpoint: str,
        endpoint_path: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        provider_label: str,
        attempt: int,
    ) -> Any:
        queue_started = time.monotonic()
        try:
            await asyncio.wait_for(
                semaphore.acquire(),
                timeout=self.config.queue_timeout_seconds,
            )
        except TimeoutError as exc:
            raise FireworksError(
                f"{provider_label} request waited more than "
                f"{self.config.queue_timeout_seconds}s for an available request slot "
                f"at {endpoint_path}."
            ) from exc

        queued_seconds = time.monotonic() - queue_started
        request_started = time.monotonic()
        logger.info(
            "%s request started endpoint=%s model=%s attempt=%s/%s queued=%.1fs; waiting for provider response",
            provider_label,
            endpoint_path,
            self.config.model,
            attempt,
            self.config.max_retries,
            queued_seconds,
        )
        try:
            async with asyncio.timeout(self.config.timeout_seconds):
                response = await self._client.post(endpoint, headers=headers, json=payload)
        except (TimeoutError, httpx.TimeoutException) as exc:
            elapsed_seconds = time.monotonic() - request_started
            raise FireworksError(
                f"{provider_label} timed out at {endpoint_path} after "
                f"{elapsed_seconds:.1f}s ({type(exc).__name__})."
            ) from exc
        finally:
            semaphore.release()

        logger.info(
            "%s request finished endpoint=%s status=%s elapsed=%.1fs",
            provider_label,
            endpoint_path,
            response.status_code,
            time.monotonic() - request_started,
        )
        return response

    async def _fallback_unstructured_payload(
        self,
        *,
        error: FireworksError,
        prompt: str,
        system_prompt: str,
        image_paths: list[str],
        temperature: float | None,
        response_schema: dict[str, Any] | None,
        response_schema_name: str,
        max_tokens: int | None,
        endpoint_path: str,
    ) -> dict[str, Any] | None:
        if response_schema is None:
            return None
        if self.config.provider_name != "openrouter":
            return None
        message = str(error)
        if "No endpoints found that can handle the requested parameters" not in message:
            return None
        logger.warning(
            "OpenRouter rejected structured-output parameters for %s. Retrying without response_format and parsing JSON text locally.",
            response_schema_name,
        )
        unstructured_prompt = (
            prompt.strip()
            + "\n\nReturn exactly one JSON object that matches the requested format."
            + "\nDo not include markdown fences or any extra text."
        )
        fallback_payload = await self._build_payload(
            prompt=unstructured_prompt,
            system_prompt=system_prompt,
            image_paths=image_paths,
            temperature=temperature,
            response_schema=None,
            response_schema_name=response_schema_name,
            max_tokens=max_tokens,
        )
        fallback_payload.pop("provider", None)
        try:
            return await self._request(payload=fallback_payload, endpoint_path=endpoint_path)
        except FireworksError as fallback_error:
            logger.warning("OpenRouter unstructured fallback failed for %s: %s", response_schema_name, fallback_error)
            return None

    async def _repair_json_output(
        self,
        *,
        raw_text: str,
        response_schema: dict[str, Any] | None,
        response_schema_name: str,
        endpoint_path: str,
    ) -> dict[str, Any] | None:
        if response_schema is None:
            return None
        repair_prompt = (
            "Repair the following output into valid JSON that matches the required schema exactly.\n"
            "Do not add commentary. Do not wrap in markdown. Return only the repaired JSON object.\n\n"
            f"Schema name: {response_schema_name}\n"
            f"Invalid output:\n{raw_text}"
        )
        repair_payload = await self._build_payload(
            prompt=repair_prompt,
            system_prompt="You repair malformed JSON so it becomes valid strict JSON matching the requested schema.",
            image_paths=[],
            temperature=None,
            response_schema=response_schema,
            response_schema_name=f"{response_schema_name}_repair",
            max_tokens=self.config.caption_max_tokens,
        )
        try:
            repaired_response = await self._request(payload=repair_payload, endpoint_path="/caption/chat/completions")
        except FireworksError as exc:
            logger.warning("Structured output repair request failed for %s: %s", response_schema_name, exc)
            return None
        repaired = _extract_json_object(repaired_response)
        if repaired is not None:
            return repaired
        repaired_text = _extract_output_text(repaired_response)
        if not repaired_text:
            return None
        try:
            parsed = json.loads(_extract_json_substring(_strip_code_fences(repaired_text)))
        except json.JSONDecodeError:
            logger.warning(
                "Structured output repair parsing still failed for %s | excerpt=%s",
                response_schema_name,
                _safe_excerpt(repaired_text),
            )
            return None
        return parsed if isinstance(parsed, dict) else None


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
    interesting_keys = {
        "summary",
        "score",
        "feedback",
        "formal",
        "sarcastic",
        "humorous_tech",
        "humorous_non_tech",
        "candidate_1",
        "candidate_2",
        "selected_candidate",
        "scene_summary",
    }
    return any(key in payload for key in interesting_keys)


def _truncate_json_payload(payload: Any, *, max_string_length: int = 2000) -> Any:
    if isinstance(payload, dict):
        return {key: _truncate_json_payload(value, max_string_length=max_string_length) for key, value in payload.items()}
    if isinstance(payload, list):
        return [_truncate_json_payload(item, max_string_length=max_string_length) for item in payload]
    if isinstance(payload, str) and len(payload) > max_string_length:
        return payload[:max_string_length] + "...<truncated>"
    return payload


def _sanitize_payload_for_logging(payload: Any) -> Any:
    if isinstance(payload, dict):
        sanitized = {}
        for key, value in payload.items():
            if key == "image_url" and isinstance(value, dict):
                url = value.get("url")
                if isinstance(url, str) and url.startswith("data:"):
                    prefix, _, remainder = url.partition(",")
                    sanitized[key] = {
                        "url": f"{prefix},<base64:{len(remainder)} chars>"
                    }
                    continue
            sanitized[key] = _sanitize_payload_for_logging(value)
        return sanitized
    if isinstance(payload, list):
        return [_sanitize_payload_for_logging(item) for item in payload]
    return payload


def _safe_excerpt(text: str, *, max_length: int = 500) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) > max_length:
        return compact[:max_length] + "...<truncated>"
    return compact


def _payload_without_deprecated_temperature(*, provider_name: str, payload: dict[str, Any], response_text: str) -> dict[str, Any] | None:
    if provider_name != "openrouter":
        return None
    if "temperature" not in payload:
        return None
    if "`temperature` is deprecated for this model" not in response_text:
        return None
    retry_payload = dict(payload)
    retry_payload.pop("temperature", None)
    return retry_payload


def _heuristic_repair_json_text(text: str) -> dict[str, Any] | None:
    candidate = _extract_json_substring(_strip_code_fences(text))
    attempts = [
        candidate,
        re.sub(r'("|\]|\})(\s*)(")', r'\1,\2\3', candidate),
        re.sub(r'("|\]|\})(\s*)(\{)', r'\1,\2\3', candidate),
        re.sub(r',(\s*[}\]])', r'\1', candidate),
    ]
    for attempt in attempts:
        try:
            parsed = json.loads(attempt)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
        try:
            parsed = ast.literal_eval(attempt)
            if isinstance(parsed, dict):
                return json.loads(json.dumps(parsed))
        except (ValueError, SyntaxError):
            pass
    return None


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
