from __future__ import annotations

from main import _build_task_result
from pipeline.caption_pipeline import (
    _build_verified_scene_caption_prompt,
    _clean_verified_caption,
    _find_unsupported_risk_terms,
    _needs_specificity_retry,
)
from worker.config.settings import Settings


def test_hackathon_runtime_defaults(monkeypatch):
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
    assert "grounded_facts_used" not in prompt
    assert "safety_notes" not in prompt
    assert "Return JSON" not in prompt


def test_verified_scene_creative_prompt_allows_figurative_meme_framing():
    prompt = _build_verified_scene_caption_prompt(
        style_name="humorous_non_tech",
        verified_description="A kitten walks slowly toward the camera.",
    )

    assert "clearly fictional comparisons" in prompt
    assert "first-person reactions" in prompt
    assert "meme framing" in prompt
    assert "playful invented context" in prompt
    assert "Prioritize a strong human caption over exhaustive visual coverage" in prompt
    assert "two to four concrete anchors" in prompt
    assert "explicitly include the main visible subject or action" in prompt
    assert "Use vague figurative wording instead of exact invented specifics" in prompt
    assert "not a vision-analysis report" in prompt
    assert "camera frustum" in prompt
    assert "one or two substantial sentences" in prompt
    assert "Avoid generic meme lines" in prompt
    assert "exact durations or years" in prompt
    assert "Safer figurative angles include landlord, NPCs" in prompt
    assert "do not contradict the verified description" in prompt


def test_verified_scene_formal_prompt_forbids_humor_and_invented_framing():
    prompt = _build_verified_scene_caption_prompt(
        style_name="formal",
        verified_description="A kitten walks slowly toward the camera.",
    )

    assert "Do not use humor, fictional framing, or invented context" in prompt
    assert "core visible subject, action, setting, background, and key objects" in prompt
    assert "one or two clean sentences" in prompt
    assert "jewelry, clothing, accessories" in prompt
    assert "time-of-day claims" in prompt
    assert "Avoid quoting visible text or full sign wording" in prompt
    assert "meme framing" not in prompt
    assert "Output only the caption text" in prompt
    assert "Return JSON" not in prompt


def test_final_result_contract_contains_only_task_id_and_caption_strings():
    result = _build_task_result(
        task_id="v1",
        captions={"formal": "A dog crosses a lawn."},
    )

    assert result == {
        "task_id": "v1",
        "captions": {"formal": "A dog crosses a lawn."},
    }
    assert all(isinstance(caption, str) for caption in result["captions"].values())


def test_verified_caption_cleanup_balances_model_added_quotes():
    assert _clean_verified_caption('"A kitten emerges from the bush."') == "A kitten emerges from the bush."
    assert _clean_verified_caption('A kitten emerges from the bush."') == "A kitten emerges from the bush."
    assert _clean_verified_caption("“A kitten emerges from the bush.”") == "A kitten emerges from the bush."
    assert _clean_verified_caption("The kitten's tail stays raised.") == "The kitten's tail stays raised."


def test_short_creative_caption_requires_two_scene_anchors():
    description = "A small kitten emerges from a green bush with its tail raised."

    assert _needs_specificity_retry(
        style_name="humorous_non_tech",
        caption="Fashionably late to the party again.",
        verified_description=description,
    )
    assert not _needs_specificity_retry(
        style_name="humorous_non_tech",
        caption="Kitten exits the bush, tail fully committed.",
        verified_description=description,
    )
    assert not _needs_specificity_retry(
        style_name="formal",
        caption="A kitten emerges.",
        verified_description=description,
    )


def test_creative_caption_must_retain_main_subject_or_action():
    description = "Traffic moves along an avenue beneath yellow ginkgo trees and green lights."

    assert _needs_specificity_retry(
        style_name="humorous_tech",
        caption="The yellow shader and green state machine keep the whole render loop suspiciously busy today.",
        verified_description=description,
    )
    assert not _needs_specificity_retry(
        style_name="humorous_tech",
        caption="Traffic keeps rendering beneath yellow ginkgo trees while green lights run the state machine.",
        verified_description=description,
    )


def test_creative_risk_filter_flags_only_unsupported_terms():
    description = "A kitten emerges from a green bush with its tail raised."

    assert _find_unsupported_risk_terms(
        style_name="sarcastic",
        caption="After three hours in the other room, the kitten exits the bush.",
        verified_description=description,
    ) == ["three hours", "other room"]
    assert _find_unsupported_risk_terms(
        style_name="humorous_non_tech",
        caption="The kitten emerges beside a visible treat bag.",
        verified_description=f"{description} A treat bag is visible beside it.",
    ) == []
    assert _find_unsupported_risk_terms(
        style_name="formal",
        caption="A spreadsheet is visible on the monitor.",
        verified_description=description,
    ) == []


def test_creative_risk_filter_catches_year_time_duration_and_distance_patterns():
    description = "Cars move along an avenue beneath yellow trees."

    risks = _find_unsupported_risk_terms(
        style_name="sarcastic",
        caption="Traffic has been here since 2017, took the wrong exit at 3 a.m., and stopped three blocks ago.",
        verified_description=description,
    )

    assert "2017" in risks
    assert "3 a.m." in risks
    assert "three blocks" in risks
    assert "wrong exit" in risks


def test_creative_risk_filter_allows_exact_verified_numeric_detail():
    description = "A clock reads 3 a.m. above the traffic scene."

    assert _find_unsupported_risk_terms(
        style_name="sarcastic",
        caption="Traffic crawls beneath a clock reading 3 a.m.",
        verified_description=description,
    ) == []


def test_creative_risk_filter_rejects_camera_religious_and_personal_judgment_phrases():
    description = "A fixed elevated angle shows a person typing beside a monitor while wearing a cross pendant."
    caption = "The fixed elevated angle makes the cross pendant a personality trait while Jesus blesses the camera frustum."

    risks = _find_unsupported_risk_terms(
        style_name="humorous_tech",
        caption=caption,
        verified_description=description,
    )

    assert "fixed elevated angle" in risks
    assert "cross pendant" in risks
    assert "personality trait" in risks
    assert "jesus" in risks
    assert "bless" in risks
    assert "camera frustum" in risks


def test_creative_unseen_props_are_allowed_only_when_verified():
    description = "A kitten walks from bushes toward the camera."

    assert _find_unsupported_risk_terms(
        style_name="humorous_non_tech",
        caption="The kitten arrives ready for the office party, snacks, and charcuterie.",
        verified_description=description,
    ) == ["charcuterie", "snacks", "office party"]
    assert _find_unsupported_risk_terms(
        style_name="humorous_non_tech",
        caption="The kitten approaches visible snacks beside a charcuterie board.",
        verified_description=f"{description} Snacks and a charcuterie board are visible.",
    ) == []


def test_final_unrelated_setup_and_accessory_terms_trigger_creative_retry():
    description = "A person types at a keyboard beside a monitor in an office with plants."
    caption = "The dev fixes one quick bug before the group project needs one slide, while a cross necklace supervises."

    risks = _find_unsupported_risk_terms(
        style_name="humorous_tech",
        caption=caption,
        verified_description=description,
    )

    assert "dev" in risks
    assert "one quick bug" in risks
    assert "group project" in risks
    assert "one slide" in risks
    assert "cross necklace" in risks


def test_dev_role_is_allowed_only_when_verified():
    assert "dev" not in _find_unsupported_risk_terms(
        style_name="humorous_tech",
        caption="The dev keeps the keyboard input loop running.",
        verified_description="A software developer types at a keyboard.",
    )


def test_formal_quote_filter_requires_text_and_readability_marker():
    caption = 'Traffic passes a building displaying "CENTRAL MARKET".'

    assert _find_unsupported_risk_terms(
        style_name="formal",
        caption=caption,
        verified_description="Traffic passes a commercial building with a sign.",
    ) == ["nonessential quoted sign text"]
    assert _find_unsupported_risk_terms(
        style_name="formal",
        caption=caption,
        verified_description='Traffic passes a building; "CENTRAL MARKET" is large, central, and clearly readable.',
    ) == []


def test_scene_fit_guidance_is_added_to_verified_caption_prompt():
    traffic_prompt = _build_verified_scene_caption_prompt(
        style_name="humorous_tech",
        verified_description="Traffic blurs beneath yellow ginkgo trees while green lights line the avenue.",
    )
    office_prompt = _build_verified_scene_caption_prompt(
        style_name="sarcastic",
        verified_description="A person is typing at a keyboard beside a monitor in an office.",
    )

    assert "Traffic scene fit" in traffic_prompt
    assert "Avoid landlord" in traffic_prompt
    assert "Office scene fit" in office_prompt
    assert "typing, keyboard, monitor" in office_prompt
    assert "about 30 to 65 words" in traffic_prompt


def test_landlord_is_limited_to_one_use_and_kitten_scenes():
    traffic_risks = _find_unsupported_risk_terms(
        style_name="sarcastic",
        caption="Traffic moves like a landlord beneath yellow trees.",
        verified_description="Traffic moves beneath yellow trees.",
    )
    kitten_first = _find_unsupported_risk_terms(
        style_name="sarcastic",
        caption="The kitten approaches like a tiny landlord with its tail raised.",
        verified_description="An orange kitten approaches with its tail raised.",
    )
    kitten_repeat = _find_unsupported_risk_terms(
        style_name="humorous_non_tech",
        caption="The kitten returns as the landlord of the bushes.",
        verified_description="An orange kitten approaches through bushes.",
        prior_captions=["The kitten approaches like a tiny landlord."],
    )

    assert "landlord" in traffic_risks
    assert "landlord" not in kitten_first
    assert "landlord" in kitten_repeat


def test_creative_sign_text_and_korea_retry_unless_text_is_central():
    caption = 'Traffic passes "KOREA" beneath yellow trees.'

    risks = _find_unsupported_risk_terms(
        style_name="humorous_non_tech",
        caption=caption,
        verified_description='Traffic passes a building where "KOREA" appears on a sign.',
    )
    central_risks = _find_unsupported_risk_terms(
        style_name="humorous_non_tech",
        caption='Traffic passes "CENTRAL MARKET" beneath yellow trees.',
        verified_description='"CENTRAL MARKET" is large, central, and clearly readable above the traffic.',
    )

    assert "korea" in risks
    assert 'quoted sign text: "KOREA"' in risks
    assert not any("quoted sign text" in risk for risk in central_risks)


def test_formal_focused_mental_state_and_long_creative_caption_retry():
    formal_risks = _find_unsupported_risk_terms(
        style_name="formal",
        caption="A focused person types at a keyboard.",
        verified_description="A person faces a monitor and types at a keyboard.",
    )
    long_caption = " ".join(["traffic"] + [f"detail{index}" for index in range(65)])
    creative_risks = _find_unsupported_risk_terms(
        style_name="humorous_tech",
        caption=long_caption,
        verified_description="Traffic moves along an avenue.",
    )

    assert "focused" in formal_risks
    assert "caption exceeds 65 words" in creative_risks
