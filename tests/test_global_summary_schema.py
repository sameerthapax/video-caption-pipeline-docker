from pipeline.global_summary import _build_chunk_fallback_summary, fuse_segment_ground_truth
from schemas.segments import TemporalSegment
from schemas.transcription import TranscriptChunk
from schemas.vlm import GlobalFactualSummary


def test_global_summary_schema_normalizes_mixed_inputs_to_typed_models():
    payload = {
        "factual_summary": "A screen recording shows a user editing visuals and browsing websites.",
        "scene_change_overview": [
            {"segment_index": 0, "apparent_change": "The view switches from an editing interface to a desktop."},
            {"segment_index": 1, "summary": "The screen changes to a browser page."},
        ],
        "continuity_overview": [
            "The same on-screen workflow continues across adjacent segments.",
        ],
        "object_and_subject_tracking": [
            {"object_id": "person_1", "tracking_summary": "The same cursor-driven workflow continues across segments."},
            {"object_id": "object_1", "name": "cursor"},
        ],
    }

    summary = GlobalFactualSummary.model_validate(payload)

    assert summary.scene_change_overview[0].apparent_change == "The view switches from an editing interface to a desktop."
    assert summary.scene_change_overview[1].apparent_change == "The screen changes to a browser page."
    assert summary.continuity_overview[0].continuity_note == "The same on-screen workflow continues across adjacent segments."
    assert summary.object_and_subject_tracking[0].tracking_summary == "The same cursor-driven workflow continues across segments."
    assert summary.object_and_subject_tracking[1].tracking_summary == "cursor"


def test_global_summary_schema_normalizes_nested_object_lists_to_strings():
    payload = {
        "factual_summary": "A kitten moves through garden foliage.",
        "main_subjects": [{"subject_id": "animal_1", "description": "orange kitten"}],
        "main_objects": [{"object_id": "object_1", "name": "green leaves"}],
        "detailed_timeline": [
            {
                "segment_index": 0,
                "summary": "The kitten looks around.",
                "key_events": [{"event_id": "event_1", "description": "The kitten pauses and stares ahead."}],
                "visual_facts": [{"description": "Orange fur stands out against green foliage."}],
                "transcript_facts": [],
                "continuity_notes": [{"summary": "The same kitten remains in view."}],
            }
        ],
    }

    summary = GlobalFactualSummary.model_validate(payload)

    assert summary.main_subjects == ["animal_1"]
    assert summary.main_objects == ["object_1"]
    assert summary.detailed_timeline[0].key_events == ["The kitten pauses and stares ahead."]
    assert summary.detailed_timeline[0].visual_facts == ["Orange fur stands out against green foliage."]
    assert summary.detailed_timeline[0].continuity_notes == ["The same kitten remains in view."]


def test_chunk_fallback_summary_produces_non_empty_grounded_summary():
    summary = _build_chunk_fallback_summary(
        job_id="job-1",
        transcript_text="",
        chunk_summaries=[
            {
                "chunk_index": 0,
                "chunk_summary": "Cars move along a city street beside tall buildings.",
                "key_details": ["yellow trees on one side", "fixed elevated view"],
                "subjects": ["cars"],
                "objects": ["buildings", "trees"],
                "actions": ["traffic moves steadily"],
                "setting": "urban street",
                "visible_text": [],
                "uncertainties": [],
            }
        ],
    )

    assert summary.factual_summary
    assert summary.detailed_ground_truth
    assert summary.setting_summary == "urban street"
    assert summary.main_subjects == ["cars"]


def test_fused_segment_ground_truth_uses_human_readable_subjects_objects():
    segment = TemporalSegment(
        segment_index=0,
        start=0.0,
        end=4.0,
        percent_range="0-33",
        transcript_chunks=[TranscriptChunk(start=0.0, end=2.0, text="Hello there", expressive_transcript="Hello there")],
    )

    fused = fuse_segment_ground_truth(
        segment=segment,
        visual_response={
            "segment_description": {
                "short_summary": "A woman works at a desk.",
                "detailed_visual_description": "A woman with a high bun sits at a white desk.",
                "movement_and_flow": "She types while facing a monitor.",
            },
            "setting": {
                "location_type": "office",
                "visual_environment": "bright office workspace",
            },
            "subjects": [
                {
                    "subject_id": "person_1",
                    "type": "person",
                    "appearance": {
                        "visible_features": ["high bun"],
                        "clothing": ["beige jacket"],
                    },
                }
            ],
            "objects": [
                {
                    "object_id": "object_1",
                    "name": "desk",
                    "description": "white desk with cables",
                }
            ],
            "events": [{"description": "She types at the desk."}],
            "audio_observations": {},
            "continuity_update": {},
            "evidence_quality": {},
        },
    )

    assert fused["subjects"] == ["person: high bun, beige jacket"]
    assert fused["objects"] == ["desk: white desk with cables"]
