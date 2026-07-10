from services.fireworks_client import FireworksConfig
from services.google_gemini_client import GoogleGeminiConfig
from services.openai_responses_client import OpenAIResponsesConfig


def test_fireworks_config_accepts_proxy_settings():
    config = FireworksConfig(
        api_key="",
        base_url="https://api.fireworks.ai/inference/v1",
        proxy_url="https://worker.example/fireworks",
        proxy_token="secret-token",
    )

    assert config.proxy_url == "https://worker.example/fireworks"
    assert config.proxy_token == "secret-token"


def test_openai_config_accepts_proxy_settings():
    config = OpenAIResponsesConfig(
        api_key="",
        base_url="https://api.openai.com/v1",
        proxy_url="https://worker.example/openai",
        proxy_token="secret-token",
    )

    assert config.proxy_url == "https://worker.example/openai"
    assert config.proxy_token == "secret-token"


def test_google_gemini_config_accepts_proxy_settings():
    config = GoogleGeminiConfig(
        api_key="",
        base_url="https://generativelanguage.googleapis.com",
        proxy_url="https://worker.example/google/v1beta/interactions",
        proxy_token="secret-token",
    )

    assert config.proxy_url == "https://worker.example/google/v1beta/interactions"
    assert config.proxy_token == "secret-token"
