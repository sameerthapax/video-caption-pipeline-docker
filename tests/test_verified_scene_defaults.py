from __future__ import annotations

from pipeline.caption_pipeline import (
    _build_observation_user_prompt,
    _calibrate_judge_result,
    _candidate_policy_violations,
    _clean_caption,
    _judge_passes,
    _needs_style_retry,
    format_timestamp_label,
)
from prompts.prompt_loader import _render_shuffle_blocks, render_prompt
from schemas.frames import FrameArtifact, FrameExtractionArtifact
from schemas.caption import JudgeResult
from worker.config.settings import Settings


def test_runtime_defaults(monkeypatch):
    monkeypatch.delenv("MAX_FRAMES_PER_VIDEO", raising=False)
    monkeypatch.delenv("ENABLE_VIDEO_NORMALIZATION", raising=False)

    configured = Settings()

    assert configured.app_name == "GemmaCaption-Pipe"
    assert configured.max_frames_per_video == 12
    assert configured.enable_video_normalization is False
    assert configured.caption_acceptance_threshold == 0.94


def test_observation_prompt_includes_ordered_frames_and_transcript():
    frame_artifact = FrameExtractionArtifact(
        job_id="clip-1",
        duration_seconds=12.5,
        target_frame_count=6,
        strategy="uniform",
        frames=[
            FrameArtifact(
                frame_id="frame_001",
                timestamp=0.5,
                local_path="/tmp/frame_001.jpg",
                selection_reasons=["uniform"],
            ),
            FrameArtifact(
                frame_id="frame_002",
                timestamp=6.25,
                local_path="/tmp/frame_002.jpg",
                selection_reasons=["uniform"],
            ),
        ],
    )

    prompt = _build_observation_user_prompt(
        job_id="clip-1",
        transcript_text="hello there",
        frame_artifact=frame_artifact,
    )

    assert "The attached images are video frames in chronological order." in prompt
    assert "- frame_001: 00:00.50" in prompt
    assert "- frame_002: 00:06.25" in prompt
    assert "Optional transcript:\nhello there" in prompt


def test_format_timestamp_label_uses_minute_second_format():
    assert format_timestamp_label(0.5) == "00:00.50"
    assert format_timestamp_label(65.2) == "01:05.20"


def test_caption_cleanup_decodes_wrappers_and_escapes():
    assert _clean_caption('"A kitten emerges from the bush."') == "A kitten emerges from the bush."
    assert _clean_caption("Line one\\nline two") == "Line one line two"
    assert _clean_caption("\\u2019") == "’"
    assert _clean_caption(": A mountain rises above the city.") == "A mountain rises above the city."
    assert _clean_caption("Coastal beach: Waves roll over a pebble shore.") == "Waves roll over a pebble shore."


def test_style_retry_only_checks_source_markers():
    assert _needs_style_retry("sarcastic", "A very normal scene.") is True
    assert _needs_style_retry("sarcastic", "Clearly, this is very serious business.") is False
    assert _needs_style_retry("humorous_tech", "The cat walks forward.") is True
    assert _needs_style_retry("humorous_tech", "The cat deploys into the yard with no rollback plan.") is False
    assert _needs_style_retry("formal", "A cat walks forward.") is False


def test_policy_flags_unsupported_precision_and_nontech_jargon():
    violations = _candidate_policy_violations(
        style_name="humorous_non_tech",
        candidates={
            "candidate_1": "The dog waits for 18 seconds before its deployment begins.",
            "candidate_2": "The dog finally crosses the sunny field like it remembered an appointment.",
        },
        observations={"caption_facts": ["A dog crosses a sunny field."]},
    )

    assert "contains technical jargon" in violations["candidate_1"]
    assert "unsupported numeric precision: 18" in violations["candidate_1"]
    assert "unsupported joke prop or event: appointment" in violations["candidate_2"]


def test_policy_allows_props_when_the_observations_support_them():
    violations = _candidate_policy_violations(
        style_name="humorous_non_tech",
        candidates={
            "candidate_1": "The dog finds the tennis ball, completing today's most important assignment.",
            "candidate_2": "The vegetables approach the pan like dinner is finally getting organized.",
        },
        observations={"caption_facts": ["A dog finds a tennis ball.", "Vegetables sit beside a pan."]},
    )

    assert violations["candidate_1"] == []
    assert violations["candidate_2"] == []


def test_judge_calibration_uses_weaker_dimension_and_local_selection(monkeypatch):
    monkeypatch.setattr("pipeline.caption_pipeline.settings.caption_acceptance_threshold", 0.94)
    result = JudgeResult.model_validate(
        {
            "selected_candidate": "candidate_1",
            "candidate_1": {
                "accuracy": "pass",
                "style": "pass",
                "accuracy_score": 0.98,
                "style_score": 0.90,
                "combined_score": 0.97,
                "notes": "Accurate but over-written.",
            },
            "candidate_2": {
                "accuracy": "pass",
                "style": "pass",
                "accuracy_score": 0.95,
                "style_score": 0.95,
                "combined_score": 0.91,
                "notes": "Simple and grounded.",
            },
        }
    )

    calibrated = _calibrate_judge_result(result, {"candidate_1": [], "candidate_2": []})

    assert calibrated.selected_candidate == "candidate_2"
    assert calibrated.candidate_1.combined_score == 0.90
    assert calibrated.candidate_2.combined_score == 0.95
    assert _judge_passes(calibrated.candidate_2) is True


def test_render_prompt_replaces_vision_output():
    rendered = render_prompt(
        "style_formal.txt",
        replacements={"VISION_OUTPUT": '{"scene_summary": "A person walks."}'},
    )

    assert "{{VISION_OUTPUT}}" not in rendered
    assert '{"scene_summary": "A person walks."}' in rendered


def test_shuffle_blocks_render_without_markers(monkeypatch):
    class ReverseRandom:
        def shuffle(self, items):
            items.reverse()

    monkeypatch.setattr("prompts.prompt_loader.random.SystemRandom", lambda: ReverseRandom())

    rendered = _render_shuffle_blocks("Before\n{{SHUFFLE_BLOCK:test}}\nalpha\nbeta\ngamma\n{{END_SHUFFLE_BLOCK}}\nAfter")

    assert "{{SHUFFLE_BLOCK:test}}" not in rendered
    assert "{{END_SHUFFLE_BLOCK}}" not in rendered
    assert rendered == "Before\ngamma\nbeta\nalpha\nAfter"
