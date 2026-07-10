from __future__ import annotations

import asyncio
from threading import Lock
from typing import Any

from services.openai_responses_client import OpenAIResponsesClient, OpenAIResponsesConfig
from services.vision_llm_client import VisionLlmClient, VisionLlmConfig
from worker.config.settings import settings


_pool_lock = Lock()
_pooled_clients: list[Any] = []


def get_vision_client() -> VisionLlmClient:
    client = VisionLlmClient(
        VisionLlmConfig(
            api_key=settings.vision_api_key or "",
            base_url=settings.vision_base_url,
            proxy_url=settings.vision_proxy_url,
            proxy_token=settings.vision_proxy_token,
            timeout_seconds=settings.vision_timeout_seconds,
            max_retries=settings.vision_max_retries,
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
            pass
        except Exception:
            pass


def _register_client(client: Any) -> None:
    with _pool_lock:
        _pooled_clients.append(client)
