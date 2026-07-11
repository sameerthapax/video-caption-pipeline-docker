from services.fireworks_client import _build_response_format


def test_build_response_format_uses_strict_json_schema():
    schema = {"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"], "additionalProperties": False}

    payload = _build_response_format(schema=schema, schema_name="test_response")

    assert payload["type"] == "json_schema"
    assert payload["json_schema"]["name"] == "test_response"
    assert payload["json_schema"]["schema"] == schema
    assert payload["json_schema"]["strict"] is True
