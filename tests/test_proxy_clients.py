import asyncio

from services.fireworks_client import FireworksClient, FireworksConfig


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
