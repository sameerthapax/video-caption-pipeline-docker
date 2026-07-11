from schemas.caption import (
    CaptionVariant,
    JudgeResult,
    ObservationResult,
    build_caption_variant_json_schema,
    build_combined_captions_json_schema,
)


def test_observation_result_accepts_expected_shape():
    observation = ObservationResult.model_validate(
        {
            "summary": "A person walks through a park.",
            "setting": {
                "location_type": "park",
                "environment": "outdoor path with trees",
                "time_of_day": "daytime",
                "weather": "clear",
            },
            "subjects": [{"type": "person", "count": "1", "description": "one person walking along the path"}],
            "key_objects": [
                {"object": "trees", "description": "trees line both sides of the path"},
                {"object": "path", "description": "a paved path through the park"},
            ],
            "relationships": ["person walking on path"],
            "actions": ["walking"],
            "timeline": {
                "beginning": "A person appears on the path.",
                "middle": "The person walks forward between the trees.",
                "end": "The person moves farther down the path.",
            },
            "camera": {"viewpoint": "eye-level", "movement": "static"},
            "visible_text": [],
            "audio_or_speech": [],
            "distinctive_details": ["tree-lined walkway"],
            "uncertainties": ["exact location unclear"],
        }
    )

    assert observation.setting.location_type == "park"
    assert observation.subjects[0].type == "person"
    assert observation.uncertainties == ["exact location unclear"]


def test_combined_caption_schema_uses_requested_styles_only():
    schema = build_combined_captions_json_schema(["formal", "sarcastic"])

    assert set(schema["properties"].keys()) == {"formal", "sarcastic"}
    assert schema["additionalProperties"] is False


def test_caption_variant_and_judge_result_validate():
    variant = CaptionVariant.model_validate(
        {
            "style_name": "formal",
            "caption": "A person walks through a tree-lined park path.",
            "grounded_facts_used": ["person walking", "park path"],
            "safety_notes": [],
        }
    )
    judge = JudgeResult.model_validate(
        {
            "accuracy": 0.95,
            "style_match": 0.9,
            "score": 0.9,
            "feedback": "Grounded and fits the requested style.",
        }
    )

    assert build_caption_variant_json_schema("formal")["properties"]["style_name"]["enum"] == ["formal"]
    assert variant.style_name == "formal"
    assert judge.score == 0.9
    assert judge.accuracy == 0.95
    assert judge.style_match == 0.9
    assert judge.feedback == "Grounded and fits the requested style."
