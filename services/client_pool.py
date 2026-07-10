from __future__ import annotations

import asyncio
from typing import Any

from services.fireworks_client import FireworksClient, FireworksConfig
from services.google_gemini_client import GoogleGeminiClient, GoogleGeminiConfig
from services.openai_responses_client import OpenAIResponsesClient, OpenAIResponsesConfig
from worker.config.settings import settings


from threading import Lock


_pool_lock = Lock()
_pooled_clients: list[Any] = []


def get_fireworks_client() -> FireworksClient:
    client = FireworksClient(
        FireworksConfig(
            api_key=settings.fireworks_api_key or "",
            base_url=settings.fireworks_base_url,
            proxy_url=settings.fireworks_proxy_url,
            proxy_token=settings.fireworks_proxy_token,
            timeout_seconds=settings.fireworks_timeout_seconds,
            max_retries=settings.fireworks_max_retries,
        )
    )
    _register_client(client)
    return client


def get_gemini_client() -> GoogleGeminiClient:
    client = GoogleGeminiClient(
        GoogleGeminiConfig(
            api_key=settings.google_gemini_api_key or "",
            base_url=settings.google_gemini_base_url,
            proxy_url=settings.google_gemini_proxy_url,
            proxy_token=settings.google_gemini_proxy_token,
            model=settings.google_gemini_transcription_model,
            timeout_seconds=settings.google_gemini_timeout_seconds,
            max_retries=settings.google_gemini_max_retries,
        )
    )
    _register_client(client)
    return client


def get_openai_responses_client() -> OpenAIResponsesClient:
    client = OpenAIResponsesClient(
        OpenAIResponsesConfig(
            api_key=settings.openai_api_key or "",
            base_url=settings.openai_base_url,
            proxy_url=settings.openai_proxy_url,
            proxy_token=settings.openai_proxy_token,
            timeout_seconds=settings.openai_timeout_seconds,
            max_retries=settings.openai_max_retries,
            temperature=settings.openai_temperature,
            reasoning_effort=settings.openai_reasoning_effort,
            text_verbosity=settings.openai_text_verbosity,
        )
    )
    _register_client(client)
    return client


def close_pooled_async_clients() -> None:
    with _pool_lock:
        clients = list(_pooled_clients)
        _pooled_clients.clear()

    for client in clients:
        try:
            asyncio.run(client.aclose())
        except RuntimeError:
            # Ignore shutdown edge cases during interpreter teardown.
            pass
        except Exception:
            pass


def _register_client(client: Any) -> None:
    with _pool_lock:
        _pooled_clients.append(client)
