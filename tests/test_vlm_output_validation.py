import asyncio

from pipeline.vlm_reasoning import _normalize_segment_payload
from pipeline.run_vlm_stage import _process_segment
from pipeline.video_memory import create_video_memory
from schemas.segments import TemporalSegment
from schemas.vlm import SegmentVlmResponse


class FailingGeminiClient:
    async def analyze_segment_with_images(self, **_kwargs):
        raise RuntimeError("simulated segment failure")


def test_failed_segment_response_still_produces_valid_artifact_shape(monkeypatch):
    monkeypatch.setattr("pipeline.run_vlm_stage.settings.google_gemini_vision_model", "test-vlm")
    segment = TemporalSegment(segment_index=0, start=0.0, end=12.0, percent_range="0-20")
    response = asyncio.run(
        _process_segment(
            gemini_client=FailingGeminiClient(),
            job_id="job-1",
            segment=segment,
            memory=create_video_memory(job_id="job-1"),
        )
    )

    assert response.status == "failed"
    assert response.segment_index == 0
    assert isinstance(response.errors, list)
    assert response.evidence_quality.limitations


def test_normalize_segment_payload_coerces_flat_repair_shape():
    payload = {
        "segment_index": 1,
        "segment_start": 0,
        "segment_end": 6.0833,
        "visual_clarity": "high",
        "audio_clarity": "none",
        "limitations": ["No audio available"],
        "short_summary": "A woman sits at an office desk working on a computer.",
        "detailed_visual_description": "A woman is seated at a desk in an office using a computer.",
        "movement_description": "The camera is static while she types.",
        "audio_description": "No audio is present.",
        "location_type": "office",
        "visual_environment": "Modern office with desks and glass partitions.",
        "background_details": ["Potted plant", "Glass partitions"],
        "changes_from_previous_segment": "None (first segment)",
        "subjects": [
            {
                "id": "person_1",
                "type": "person",
                "appearance": "Dark hair in a high bun, light jacket, orange top.",
                "action": "Typing on keyboard",
                "emotion_or_tone": "focused",
                "confidence": 0.95,
            }
        ],
        "objects": [
            {"id": "object_1", "object_name": "computer monitor", "location": "Foreground right", "state": "On"},
            {"name": "keyboard", "location": "On desk"},
        ],
        "events": [{"name": "Woman working at computer", "start_time": 0.0, "involved_subjects": ["person_1"]}],
        "visible_text": [{"text_content": "HANYANG", "position": "left building", "confidence": "low"}],
        "audio_observations": [],
        "continuity_update": {"new_subjects": ["person_1"], "new_setting": "office"},
        "memory_update_for_next_segment": {"segment_memory": "Woman working at office desk.", "setting": "office"},
    }

    normalized = _normalize_segment_payload(
        payload,
        expected_segment_index=0,
        expected_start=0.0,
        expected_end=6.0833,
        has_transcript=False,
    )
    response = SegmentVlmResponse.model_validate(normalized)

    assert response.segment_index == 0
    assert response.evidence_quality.visual_clarity == "high"
    assert response.segment_description.movement_and_flow == "The camera is static while she types."
    assert response.setting.location_type == "office"
    assert response.subjects[0].subject_id == "person_1"
    assert response.subjects[0].appearance.visible_features
    assert response.subjects[0].actions == ["Typing on keyboard"]
    assert response.objects[0].name == "computer monitor"
    assert response.events[0].subjects_involved == ["person_1"]
    assert response.visible_text[0].text == "HANYANG"
    assert response.continuity_update.important_changes == ["office"]
    assert response.memory_update_for_next_segment.persistent_setting == "office"


def test_normalize_segment_payload_coerces_string_sections():
    payload = {
        "segment_index": 2,
        "segment_start": 8.2444,
        "segment_end": 12.3667,
        "segment_description": "kitten walking",
        "setting": "Outdoors under bushes",
        "subjects": [
            {
                "subject_id": "animal_1",
                "type": "animal",
                "description": "Orange fur, fluffy, green eyes, pink nose, white whiskers.",
                "actions": "Walking towards camera",
                "movement": "Walking forward",
            }
        ],
        "objects": ["Bushes/trees"],
        "events": ["Kitten walking towards camera"],
        "audio_observations": "No speech. No music indicated.",
        "continuity_update": "New subject: kitten.",
        "memory_update_for_next_segment": "Kitten walking in a garden or wooded outdoor area.",
    }

    normalized = _normalize_segment_payload(
        payload,
        expected_segment_index=2,
        expected_start=8.2444,
        expected_end=12.3667,
        has_transcript=False,
    )
    response = SegmentVlmResponse.model_validate(normalized)

    assert response.segment_description.short_summary == "kitten walking"
    assert response.setting.visual_environment == "Outdoors under bushes"
    assert response.subjects[0].appearance.visible_features == ["Orange fur, fluffy, green eyes, pink nose, white whiskers."]
    assert response.objects[0].name == "Bushes/trees"
    assert response.events[0].description == "Kitten walking towards camera"
    assert response.audio_observations.spoken_content_summary == "No speech. No music indicated."
    assert response.continuity_update.important_changes == ["New subject: kitten."]
    assert response.memory_update_for_next_segment.segment_memory == "Kitten walking in a garden or wooded outdoor area."


def test_normalize_segment_payload_inflates_minimal_vlm_shape():
    payload = {
        "segment_index": 0,
        "segment_start": 0.0,
        "segment_end": 2.0,
        "visual_clarity": "high",
        "summary": "A busy city street is shown from an elevated angle.",
        "scene_details": "Cars move along a multi-lane road beside apartment buildings and yellow trees.",
        "setting": {
            "location_type": "city street",
            "visual_environment": "urban road in daylight",
            "background_details": ["apartment buildings", "yellow trees"],
        },
        "subjects": [],
        "objects": [
            {
                "object_id": "object_1",
                "matched_previous_object_id": None,
                "name": "cars",
                "description": "moving traffic",
                "location_in_scene": "roadway",
                "state": "moving",
                "interaction": "",
            }
        ],
        "main_events": ["Traffic flows through the road."],
        "focused_event": {
            "description": "Cars move steadily through the intersection.",
            "subjects_involved": [],
            "objects_involved": ["object_1"],
        },
        "visible_text": [],
        "continuity_notes": ["First clear urban street view."],
        "segment_memory": "Urban street traffic with autumn trees.",
        "uncertainties": ["No distinct people visible."],
    }

    normalized = _normalize_segment_payload(
        payload,
        expected_segment_index=0,
        expected_start=0.0,
        expected_end=2.0,
        has_transcript=False,
    )
    response = SegmentVlmResponse.model_validate(normalized)

    assert response.segment_description.short_summary == "A busy city street is shown from an elevated angle."
    assert response.setting.location_type == "city street"
    assert response.objects[0].name == "cars"
    assert response.events[0].description == "Traffic flows through the road."
    assert response.events[1].objects_involved == ["object_1"]
    assert response.continuity_update.important_changes == ["First clear urban street view."]
    assert response.memory_update_for_next_segment.segment_memory == "Urban street traffic with autumn trees."
