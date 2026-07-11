from __future__ import annotations

from pipeline.caption_pipeline import (
    _build_direct_vision_prompt,
    _build_observation_user_prompt,
    _build_verified_scene_caption_prompt,
    _clean_caption,
    _needs_style_retry,
    _sanitize_observations,
)
from worker.config.settings import Settings


def test_runtime_defaults(monkeypatch):
    monkeypatch.delenv("MAX_FRAMES_PER_VIDEO", raising=False)
    monkeypatch.delenv("ENABLE_VIDEO_NORMALIZATION", raising=False)
    monkeypatch.delenv("ENABLE_PLANNED_FRAME_EXTRACTION", raising=False)

    configured = Settings()

    assert configured.max_frames_per_video == 3
    assert configured.enable_video_normalization is False
    assert configured.enable_planned_frame_extraction is False


def test_verified_scene_caption_prompt_requests_plain_text_and_prior_variety():
    prompt = _build_verified_scene_caption_prompt(
        style_name="sarcastic",
        verified_description="A dog crosses a lawn toward the camera.",
        prior_captions=["A dog crosses the lawn."],
    )

    assert "Output only the caption text" in prompt
    assert "Captions already written" in prompt
    assert "Return JSON" not in prompt
    assert "pipelines" in prompt


def test_verified_scene_caption_prompt_matches_source_style_examples():
    formal_prompt = _build_verified_scene_caption_prompt(
        style_name="formal",
        verified_description="A kitten walks slowly toward the camera.",
    )
    tech_prompt = _build_verified_scene_caption_prompt(
        style_name="humorous_tech",
        verified_description="Traffic moves along an avenue.",
    )

    assert "formal, professional, objective caption" in formal_prompt
    assert "The subject proceeds through the marked route without deviation." in formal_prompt
    assert "technology, software, programming, network, game engine, or debugging references" in tech_prompt
    assert "404: graceful landing not found." in tech_prompt


def test_caption_cleanup_decodes_wrappers_and_escapes():
    assert _clean_caption('"A kitten emerges from the bush."') == "A kitten emerges from the bush."
    assert _clean_caption("Line one\\nline two") == "Line one line two"
    assert _clean_caption("\\u2019") == "’"


def test_style_retry_only_checks_source_markers():
    assert _needs_style_retry("sarcastic", "A very normal scene.") is True
    assert _needs_style_retry("sarcastic", "Clearly, this is very serious business.") is False
    assert _needs_style_retry("humorous_tech", "The cat walks forward.") is True
    assert _needs_style_retry("humorous_tech", "The cat deploys into the yard with no rollback plan.") is False
    assert _needs_style_retry("formal", "A cat walks forward.") is False


def test_observation_sanitizer_removes_guessed_location_tail():
    observations = _sanitize_observations(
        {
            "summary": "A street scene.",
            "uncertainties": ["exact city uncertain but it may be Chicago"],
        }
    )

    assert observations["uncertainties"] == ["exact city uncertain."]


def test_observation_and_direct_vision_prompts_match_source_shape():
    observation_prompt = _build_observation_user_prompt(
        job_id="clip-1",
        transcript_text="hello there",
        frame_count=3,
    )
    direct_prompt = _build_direct_vision_prompt(
        requested_styles=["formal", "sarcastic"],
        transcript_text="hello there",
    )

    assert "Analyze these 3 sampled frames in chronological order." in observation_prompt
    assert "visible text, camera movement, and transcript-backed speech." in observation_prompt
    assert "Create one short, accurate caption for each requested style." in direct_prompt
    assert '"formal": "caption text"' in direct_prompt
    assert '"sarcastic": "caption text"' in direct_prompt
