from schemas.caption import (
    CaptionCandidates,
    CaptionVariant,
    JudgeResult,
    ObservationResult,
    build_caption_candidates_json_schema,
    build_judge_json_schema,
    build_observation_json_schema,
)


def test_observation_result_accepts_source_shape():
    observation = ObservationResult.model_validate(
        {
            "scene_summary": "A person moves across an outdoor paved area.",
            "setting": "outdoor paved area in daylight",
            "subjects": ["person wearing a dark top and light shoes"],
            "actions": ["person moves across pavement"],
            "key_objects": ["pavement"],
            "timeline": [
                {"timestamp": "beginning", "observation": "The person is already visible."},
                {"timestamp": "middle", "observation": "The person continues moving forward."},
                {"timestamp": "end", "observation": "The person remains in the paved area."},
            ],
            "temporal_highlights": ["The person moves forward across the pavement."],
            "camera": "mostly static wide view",
            "caption_facts": ["One person is visible outdoors.", "The main action is the person's movement across pavement."],
        }
    )

    assert observation.setting == "outdoor paved area in daylight"
    assert observation.subjects[0] == "person wearing a dark top and light shoes"
    assert observation.timeline[1].timestamp == "middle"
    assert observation.camera == "mostly static wide view"


def test_caption_candidates_variant_and_judge_result_validate():
    candidates = CaptionCandidates.model_validate(
        {
            "candidate_1": "A person moves across an outdoor paved area.",
            "candidate_2": "One person crosses a paved outdoor area in daylight.",
        }
    )
    variant = CaptionVariant.model_validate(
        {
            "style_name": "formal",
            "caption": "A person moves across an outdoor paved area.",
        }
    )
    judge = JudgeResult.model_validate(
        {
            "selected_candidate": "candidate_2",
            "candidate_1": {
                "accuracy": "pass",
                "style": "fail",
                "accuracy_score": 0.93,
                "style_score": 0.62,
                "combined_score": 0.78,
                "notes": "Accurate but too conversational.",
            },
            "candidate_2": {
                "accuracy": "pass",
                "style": "pass",
                "accuracy_score": 0.97,
                "style_score": 0.94,
                "combined_score": 0.955,
                "notes": "Grounded and fits the requested style.",
            },
        }
    )

    assert build_observation_json_schema()["required"] == [
        "scene_summary",
        "setting",
        "subjects",
        "actions",
        "key_objects",
        "timeline",
        "temporal_highlights",
        "camera",
        "caption_facts",
    ]
    assert build_caption_candidates_json_schema()["required"] == ["candidate_1", "candidate_2"]
    assert build_judge_json_schema()["properties"]["candidate_1"]["properties"]["accuracy"]["enum"] == ["pass", "fail"]
    assert candidates.candidate_1 == "A person moves across an outdoor paved area."
    assert variant.style_name == "formal"
    assert judge.selected_candidate == "candidate_2"
    assert judge.candidate_1.style == "fail"
    assert judge.candidate_2.style == "pass"
    assert judge.candidate_2.combined_score == 0.955
