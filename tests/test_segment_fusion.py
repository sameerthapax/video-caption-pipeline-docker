from pipeline.global_summary import fuse_segment_ground_truth
from schemas.segments import TemporalSegment
from schemas.transcription import TranscriptChunk


def test_segment_fusion_combines_visual_and_transcript_evidence_without_model_call():
    segment = TemporalSegment(
        segment_index=1,
        start=12.0,
        end=24.0,
        percent_range="20-40",
        transcript_chunks=[
            TranscriptChunk(
                start=12.0,
                end=17.0,
                text="A person opens the box",
                expressive_transcript="[speaker_change] A person opens the box",
            )
        ],
    )
    visual_response = {
        "segment_description": {
            "short_summary": "A person opens a cardboard box on a table.",
            "detailed_visual_description": "The person leans forward and lifts the top flap.",
            "movement_and_flow": "The action moves from holding the box to opening it.",
        },
        "setting": {
            "location_type": "indoor",
            "visual_environment": "A table in a simple room.",
            "changes_from_previous_segment": "The box is now being opened.",
        },
        "subjects": [{"subject_id": "person_1"}],
        "objects": [{"object_id": "box_1"}],
        "events": [{"description": "The box is opened."}],
        "audio_observations": {
            "spoken_content_summary": "Someone says the box is being opened."
        },
        "continuity_update": {
            "same_subjects_as_before": ["person_1"],
            "same_objects_as_before": ["box_1"],
            "important_changes": ["The closed box becomes open."],
            "new_uncertainties": ["Speaker identity is not visually confirmed."],
        },
        "evidence_quality": {
            "limitations": ["Hands partially block the box lid."]
        },
    }

    fused = fuse_segment_ground_truth(segment=segment, visual_response=visual_response)

    assert fused["segment_index"] == 1
    assert fused["scene_summary"] == "A person opens a cardboard box on a table."
    assert "A table in a simple room." in fused["grounded_visual_facts"]
    assert "[speaker_change] A person opens the box" in fused["grounded_transcript_facts"]
    assert "Someone says the box is being opened." in fused["grounded_transcript_facts"]
    assert fused["subjects"] == ["person_1"]
    assert fused["objects"] == ["box_1"]
    assert fused["continuity"]["same_people"] == ["person_1"]
    assert "Hands partially block the box lid." in fused["uncertainties"]
