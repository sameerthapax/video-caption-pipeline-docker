import asyncio

from services.client_pool import get_llm_client
from services.fireworks_client import FireworksClient, FireworksConfig
from worker.config.settings import settings


def test_fireworks_config_accepts_proxy_settings():
    config = FireworksConfig(
        api_key="",
        base_url="https://api.fireworks.ai/inference/v1",
        proxy_url="https://worker.example",
        proxy_token="secret-token",
    )

    assert config.proxy_url == "https://worker.example"
    assert config.proxy_token == "secret-token"


def test_fireworks_payload_includes_model_and_omits_none_temperature():
    client = object.__new__(FireworksClient)
    client.config = FireworksConfig(api_key="test-key")

    async def run_test():
        try:
            return await client._build_payload(
                prompt="hello",
                system_prompt="",
                image_paths=[],
                temperature=None,
                response_schema=None,
                response_schema_name="response",
                max_tokens=180,
            )
        finally:
            pass

    payload = asyncio.run(run_test())

    assert "temperature" not in payload
    assert payload["model"] == "accounts/fireworks/models/kimi-k2p6"
    assert payload["max_tokens"] == 180
    assert payload["reasoning_effort"] == "none"


def test_fireworks_uses_role_specific_output_limits():
    client = object.__new__(FireworksClient)
    client.config = FireworksConfig(api_key="test-key")

    assert client._max_tokens_for_endpoint("/vision/chat/completions") == 300
    assert client._max_tokens_for_endpoint("/caption/chat/completions") == 180
    assert client._max_tokens_for_endpoint("/judge/chat/completions") == 200


def test_openrouter_payload_omits_none_reasoning_effort():
    client = object.__new__(FireworksClient)
    client.config = FireworksConfig(
        provider_name="openrouter",
        api_key="test-key",
        model="openai/gpt-5.6-sol",
        reasoning_effort="none",
    )

    async def run_test():
        return await client._build_payload(
            prompt="hello",
            system_prompt="",
            image_paths=[],
            temperature=0.2,
            response_schema=None,
            response_schema_name="response",
            max_tokens=180,
        )

    payload = asyncio.run(run_test())

    assert "reasoning_effort" not in payload


def test_client_pool_can_build_openrouter_client(monkeypatch):
    class DummyClient:
        def __init__(self, config):
            self.config = config

        async def aclose(self):
            return None

    monkeypatch.setattr(settings, "llm_provider", "openrouter")
    monkeypatch.setattr(settings, "openrouter_api_key", "test-key")
    monkeypatch.setattr(settings, "openrouter_model", "openai/gpt-5.6-sol")
    monkeypatch.setattr(settings, "openrouter_base_url", "https://openrouter.ai/api/v1")
    monkeypatch.setattr(settings, "openrouter_proxy_url", "https://worker.example/openrouter")
    monkeypatch.setattr(settings, "openrouter_proxy_token", "secret-token")
    monkeypatch.setattr("services.client_pool.FireworksClient", DummyClient)

    client = get_llm_client()

    try:
        assert client.config.provider_name == "openrouter"
        assert client.config.model == "openai/gpt-5.6-sol"
        assert client.config.proxy_url == "https://worker.example/openrouter"
    finally:
        asyncio.run(client.aclose())
