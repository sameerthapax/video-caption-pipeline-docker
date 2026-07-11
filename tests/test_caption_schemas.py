from schemas.caption import (
    CaptionVariant,
    JudgeResult,
    ObservationResult,
    build_caption_variant_json_schema,
    build_combined_captions_json_schema,
    build_judge_json_schema,
    build_observation_json_schema,
)


def test_observation_result_accepts_source_shape():
    observation = ObservationResult.model_validate(
        {
            "summary": "A person walks through a park.",
            "setting": "a tree-lined park path",
            "subjects": ["one person"],
            "key_objects": ["trees", "paved path"],
            "actions": ["walking forward"],
            "timeline": ["beginning: person enters frame", "middle: person walks ahead", "end: person moves farther away"],
            "visible_text": [],
            "audio_or_speech": [],
            "uncertainties": ["exact location unclear"],
        }
    )

    assert observation.setting == "a tree-lined park path"
    assert observation.subjects == ["one person"]
    assert observation.uncertainties == ["exact location unclear"]


def test_combined_caption_schema_uses_requested_styles_only():
    schema = build_combined_captions_json_schema(["formal", "sarcastic"])

    assert set(schema["properties"].keys()) == {"formal", "sarcastic"}
    assert schema["properties"]["formal"] == {"type": "string"}
    assert schema["additionalProperties"] is False


def test_caption_variant_and_judge_result_validate():
    variant = CaptionVariant.model_validate(
        {
            "style_name": "formal",
            "caption": "A person walks through a tree-lined park path.",
        }
    )
    judge = JudgeResult.model_validate(
        {
            "accuracy": "pass",
            "tone": "pass",
            "notes": "Grounded and fits the requested style.",
        }
    )

    assert build_caption_variant_json_schema("formal")["properties"]["style_name"]["enum"] == ["formal"]
    assert build_observation_json_schema()["required"] == [
        "summary",
        "setting",
        "subjects",
        "key_objects",
        "actions",
        "timeline",
        "visible_text",
        "audio_or_speech",
        "uncertainties",
    ]
    assert build_judge_json_schema()["properties"]["accuracy"]["enum"] == ["pass", "fail"]
    assert variant.style_name == "formal"
    assert judge.accuracy == "pass"
    assert judge.tone == "pass"
    assert judge.notes == "Grounded and fits the requested style."
