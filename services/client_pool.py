from __future__ import annotations

import asyncio
from threading import Lock
from typing import Any

from services.fireworks_client import FireworksClient, FireworksConfig
from worker.config.settings import settings


_pool_lock = Lock()
_pooled_clients: list[Any] = []


def get_fireworks_client() -> FireworksClient:
    client = FireworksClient(
        FireworksConfig(
            provider_name="fireworks",
            api_key=settings.fireworks_api_key or "",
            model=settings.fireworks_model,
            base_url=settings.fireworks_base_url,
            proxy_url=settings.fireworks_proxy_url,
            proxy_token=settings.fireworks_proxy_token,
            timeout_seconds=settings.fireworks_timeout_seconds,
            max_retries=settings.fireworks_max_retries,
            reasoning_effort=settings.fireworks_reasoning_effort,
            vision_max_tokens=settings.fireworks_vision_max_tokens,
            caption_max_tokens=settings.fireworks_caption_max_tokens,
            judge_max_tokens=settings.fireworks_judge_max_tokens,
        )
    )
    _register_client(client)
    return client


def get_llm_client() -> FireworksClient:
    if settings.llm_provider == "openrouter":
        client = FireworksClient(
            FireworksConfig(
                provider_name="openrouter",
                api_key=settings.openrouter_api_key or "",
                model=settings.openrouter_model,
                base_url=settings.openrouter_base_url,
                proxy_url=settings.openrouter_proxy_url,
                proxy_token=settings.openrouter_proxy_token,
                timeout_seconds=settings.fireworks_timeout_seconds,
                max_retries=settings.fireworks_max_retries,
                reasoning_effort=settings.fireworks_reasoning_effort,
                vision_max_tokens=settings.fireworks_vision_max_tokens,
                caption_max_tokens=settings.fireworks_caption_max_tokens,
                judge_max_tokens=settings.fireworks_judge_max_tokens,
            )
        )
        _register_client(client)
        return client
    return get_fireworks_client()


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
