from services.openai_responses_client import OpenAIResponsesConfig
from services.vision_llm_client import VisionLlmConfig


def test_openai_config_accepts_proxy_settings():
    config = OpenAIResponsesConfig(
        api_key="",
        base_url="https://api.openai.com/v1",
        proxy_url="https://worker.example/openai",
        proxy_token="secret-token",
    )

    assert config.proxy_url == "https://worker.example/openai"
    assert config.proxy_token == "secret-token"


def test_vision_config_accepts_proxy_settings():
    config = VisionLlmConfig(
        api_key="",
        base_url="https://generativelanguage.googleapis.com",
        proxy_url="https://worker.example/vision",
        proxy_token="secret-token",
    )

    assert config.proxy_url == "https://worker.example/vision"
    assert config.proxy_token == "secret-token"
