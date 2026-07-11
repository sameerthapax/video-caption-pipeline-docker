from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any

from pipeline.extract_frames import extract_frames_for_video
from prompts.prompt_loader import load_prompt
from schemas.caption import CaptionVariant, JudgeResult, STYLE_ORDER, StyleName, build_combined_captions_json_schema, build_judge_json_schema, build_observation_json_schema
from schemas.frames import FrameExtractionArtifact
from schemas.video import VideoMetadata
from worker.config.settings import settings

logger = logging.getLogger("video-caption-pipeline.worker")

STYLE_PROMPTS = {
    "formal": "style_formal.txt",
    "sarcastic": "style_sarcastic.txt",
    "humorous_tech": "style_humorous_tech.txt",
    "humorous_non_tech": "style_humorous_non_tech.txt",
}

VISION_ENDPOINT = "/vision/chat/completions"
CAPTION_ENDPOINT = "/caption/chat/completions"
JUDGE_ENDPOINT = "/judge/chat/completions"

OBSERVATION_SCHEMA_PREVIEW = json.dumps(
    {
        "summary": "one factual overview sentence",
        "setting": "where the video appears to take place",
        "subjects": ["visible subject or object"],
        "key_objects": ["important visible objects, colors, signs, or environmental details"],
        "actions": ["important visible action"],
        "timeline": ["beginning: ...", "middle: ...", "end: ..."],
        "visible_text": ["text visible in frames, or empty array"],
        "audio_or_speech": ["relevant transcript or audio cue, or empty array"],
        "uncertainties": ["anything unclear or ambiguous, or empty array"],
    },
    indent=2,
)

CREATIVE_STYLES = {"sarcastic", "humorous_tech", "humorous_non_tech"}
TECH_STYLE_WORDS = {
    "api",
    "bug",
    "cache",
    "commit",
    "debug",
    "deploy",
    "latency",
    "log",
    "pipeline",
    "queue",
    "rollback",
    "runtime",
    "scheduler",
}
SARCASM_STYLE_MARKERS = {
    "apparently",
    "because",
    "clearly",
    "naturally",
    "of course",
    "obviously",
    "serious",
    "thrilling",
}
UNICODE_ESCAPE_SEQUENCE = re.compile(r"(?:\\u[0-9a-fA-F]{4})+")

VERIFIED_STYLE_PROMPTS = {
    "formal": (
        "Write a formal, professional, objective caption. Factual tone, no humor. "
        "Style example only: 'The subject proceeds through the marked route without deviation.'"
    ),
    "sarcastic": (
        "Write a sarcastic caption: dry, ironic, lightly mocking, grounded in the specific action described. "
        "Style example only: 'The subject surveys its kingdom of one bench with the confidence of a landlord.'"
    ),
    "humorous_tech": (
        "Write a funny caption using technology, software, programming, network, game engine, or debugging references. "
        "Style example only: '404: graceful landing not found.'"
    ),
    "humorous_non_tech": (
        "Write a funny everyday-humor caption with no technical jargon, relatable and light-hearted. "
        "Style example only: 'Confidence level: main character. Execution level: blooper reel.'"
    ),
}

STYLE_DESCRIPTIONS = {
    "formal": "Professional, objective, factual tone. No jokes, slang, sarcasm, or embellishment.",
    "sarcastic": "Dry, ironic, lightly mocking tone while staying true to the observed video.",
    "humorous_tech": "Funny with technology or programming references, but still grounded in the observations.",
    "humorous_non_tech": "Funny everyday humor for a general audience, with no technical jargon.",
}


class CaptionPipeline:
    def __init__(
        self,
        *,
        llm_client,
        artifact_root: Path,
        persist_artifacts: bool,
    ) -> None:
        self.llm_client = llm_client
        self.artifact_root = artifact_root
        self.persist_artifacts = persist_artifacts

    async def run(
        self,
        *,
        job_id: str,
        video_path: Path,
        video_metadata: VideoMetadata,
        transcript_text: str,
        requested_styles: list[StyleName],
    ) -> dict[str, Any]:
        frame_artifact = await asyncio.to_thread(
            extract_frames_for_video,
            job_id=job_id,
            video_path=video_path,
            output_dir=self.artifact_root / "frames",
            video_metadata=video_metadata,
        )
        self._persist_json("frames.json", frame_artifact.model_dump(mode="json"))
        if transcript_text.strip():
            self._persist_text("transcript.txt", transcript_text.strip())

        if settings.caption_pipeline_mode == "verified_scene":
            captions, observations, checks = await self._run_verified_scene_mode(
                frame_artifact=frame_artifact,
                transcript_text=transcript_text,
                requested_styles=requested_styles,
            )
        elif settings.caption_pipeline_mode == "direct_vision":
            captions, observations, checks = await self._run_direct_vision_mode(
                frame_artifact=frame_artifact,
                transcript_text=transcript_text,
                requested_styles=requested_styles,
            )
        elif settings.caption_pipeline_mode == "observation_first":
            captions, observations, checks = await self._run_observation_first_mode(
                job_id=job_id,
                frame_artifact=frame_artifact,
                transcript_text=transcript_text,
                requested_styles=requested_styles,
            )
        else:
            raise ValueError(f"Unsupported CAPTION_PIPELINE_MODE: {settings.caption_pipeline_mode}")

        self._persist_json(
            "final_result.json",
            {
                "job_id": job_id,
                "mode": settings.caption_pipeline_mode,
                "captions": {style_name: variant.caption for style_name, variant in captions.items()},
                "checks": {style_name: check.model_dump(mode="json") for style_name, check in checks.items()},
                "observations": observations,
            },
        )
        return {
            "captions": captions,
            "checks": checks,
            "observations": observations,
            "frame_artifact": frame_artifact,
        }

    async def _run_verified_scene_mode(
        self,
        *,
        frame_artifact: FrameExtractionArtifact,
        transcript_text: str,
        requested_styles: list[StyleName],
    ) -> tuple[dict[str, CaptionVariant], dict[str, Any], dict[str, JudgeResult]]:
        frame_paths = [frame.local_path for frame in frame_artifact.frames]
        draft = await self.llm_client.generate_text(
            prompt=_build_verified_scene_description_prompt(transcript_text=transcript_text),
            image_paths=frame_paths,
            temperature=0.2,
            endpoint_path=VISION_ENDPOINT,
        )
        verified_description = await self.llm_client.generate_text(
            prompt=_build_verified_scene_verification_prompt(draft=draft),
            image_paths=frame_paths,
            temperature=0.1,
            endpoint_path=VISION_ENDPOINT,
        )
        self._persist_text("verified_description.txt", verified_description)
        observations = {
            "summary": "Verified scene description.",
            "frame_count": len(frame_paths),
            "verified_description": verified_description,
        }

        captions: dict[str, CaptionVariant] = {}
        prior_captions: list[str] = []
        for style_name in requested_styles:
            prompt = _build_verified_scene_caption_prompt(
                style_name=style_name,
                verified_description=verified_description,
                prior_captions=prior_captions,
            )
            raw_caption = await self.llm_client.generate_text(
                prompt=prompt,
                temperature=settings.fireworks_temperature if style_name not in CREATIVE_STYLES else settings.fireworks_creative_temperature,
                endpoint_path=CAPTION_ENDPOINT,
            )
            caption = _clean_caption(raw_caption)
            captions[style_name] = CaptionVariant(style_name=style_name, caption=caption)
            prior_captions.append(caption)

        checks = await self._run_checks(
            requested_styles=requested_styles,
            observations=observations,
            captions=captions,
        )
        return captions, observations, checks

    async def _run_direct_vision_mode(
        self,
        *,
        frame_artifact: FrameExtractionArtifact,
        transcript_text: str,
        requested_styles: list[StyleName],
    ) -> tuple[dict[str, CaptionVariant], dict[str, Any], dict[str, JudgeResult]]:
        frame_paths = [frame.local_path for frame in frame_artifact.frames]
        payload = await self.llm_client.generate_json(
            prompt=_build_direct_vision_prompt(requested_styles=requested_styles, transcript_text=transcript_text),
            image_paths=frame_paths,
            temperature=settings.fireworks_creative_temperature,
            response_schema=build_combined_captions_json_schema(requested_styles),
            response_schema_name="direct_vision_captions",
            endpoint_path=VISION_ENDPOINT,
        )
        captions = {
            style_name: CaptionVariant(
                style_name=style_name,
                caption=_clean_caption(str(payload.get(style_name) or "")),
            )
            for style_name in requested_styles
        }
        for style_name in requested_styles:
            if not captions[style_name].caption:
                captions[style_name] = CaptionVariant(style_name=style_name, caption=_fallback_caption(style_name, {}))

        observations = {
            "summary": "Direct vision caption mode; captions generated from sampled frames.",
            "frame_count": len(frame_paths),
        }
        checks = await self._run_checks(
            requested_styles=requested_styles,
            observations=observations,
            captions=captions,
        )
        return captions, observations, checks

    async def _run_observation_first_mode(
        self,
        *,
        job_id: str,
        frame_artifact: FrameExtractionArtifact,
        transcript_text: str,
        requested_styles: list[StyleName],
    ) -> tuple[dict[str, CaptionVariant], dict[str, Any], dict[str, JudgeResult]]:
        frame_paths = [frame.local_path for frame in frame_artifact.frames]
        observations = await self.llm_client.generate_json(
            system_prompt=load_prompt("perception_system.txt"),
            prompt=_build_observation_user_prompt(
                job_id=job_id,
                transcript_text=transcript_text,
                frame_count=len(frame_paths),
            ),
            image_paths=frame_paths,
            temperature=settings.fireworks_temperature,
            response_schema=build_observation_json_schema(),
            response_schema_name="observations",
            endpoint_path=VISION_ENDPOINT,
        )
        observations = _sanitize_observations(observations)
        self._persist_json("observations.json", observations)

        async def generate_for_style(style_name: StyleName) -> tuple[StyleName, CaptionVariant]:
            prompt = load_prompt(STYLE_PROMPTS[style_name])
            caption_request = {
                "target_style": style_name,
                "style_requirement": STYLE_DESCRIPTIONS.get(style_name, ""),
                "strict_grounding_rules": [
                    "Use only summary, setting, subjects, key_objects, actions, timeline, visible_text, and audio_or_speech as factual evidence.",
                    "Never turn anything in uncertainties into a fact.",
                    "If the exact location, identity, motive, or text is uncertain, use generic wording instead of guessing.",
                    "Mention the main subject, setting, and primary action when supported.",
                    "Return only the final caption text.",
                ],
                "observations": observations,
            }
            temperature = settings.fireworks_creative_temperature if style_name in CREATIVE_STYLES else settings.fireworks_temperature
            caption = await self.llm_client.generate_text(
                prompt=json.dumps(caption_request, indent=2),
                system_prompt=prompt,
                temperature=temperature,
                endpoint_path=CAPTION_ENDPOINT,
            )
            cleaned = _clean_caption(caption)
            if _needs_style_retry(style_name, cleaned):
                cleaned = _clean_caption(
                    await self.llm_client.generate_text(
                        prompt=(
                            json.dumps(caption_request, indent=2)
                            + "\n\nRewrite the caption. It was too plain for the requested style. "
                            "Keep the same observed facts, but make the target style obvious. "
                            "Return only the rewritten caption text."
                        ),
                        system_prompt=prompt,
                        temperature=temperature,
                        endpoint_path=CAPTION_ENDPOINT,
                    )
                )
            return style_name, CaptionVariant(style_name=style_name, caption=cleaned)

        generated = await asyncio.gather(*(generate_for_style(style_name) for style_name in requested_styles))
        captions = {style_name: variant for style_name, variant in generated}

        checks = await self._run_checks(
            requested_styles=requested_styles,
            observations=observations,
            captions=captions,
        )
        return captions, observations, checks

    async def _run_checks(
        self,
        *,
        requested_styles: list[StyleName],
        observations: dict[str, Any],
        captions: dict[str, CaptionVariant],
    ) -> dict[str, JudgeResult]:
        if not settings.run_judge_checks:
            return {}

        async def judge_style(style_name: StyleName) -> tuple[StyleName, JudgeResult]:
            payload = await self.llm_client.generate_json(
                system_prompt=load_prompt("judge.txt"),
                prompt=json.dumps(
                    {
                        "target_style": style_name,
                        "observations": observations,
                        "caption": captions[style_name].caption,
                    },
                    indent=2,
                ),
                temperature=settings.fireworks_temperature,
                response_schema=build_judge_json_schema(),
                response_schema_name=f"{style_name}_judge",
                endpoint_path=JUDGE_ENDPOINT,
            )
            return style_name, JudgeResult.model_validate(payload)

        results = await asyncio.gather(*(judge_style(style_name) for style_name in requested_styles), return_exceptions=True)
        checks: dict[str, JudgeResult] = {}
        for result in results:
            if isinstance(result, Exception):
                logger.warning("Judge check failed: %s", result)
                continue
            style_name, judge = result
            checks[style_name] = judge
        self._persist_json("checks.json", {style_name: check.model_dump(mode="json") for style_name, check in checks.items()})
        return checks

    def _persist_json(self, filename: str, payload: Any) -> None:
        if not self.persist_artifacts:
            return
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        (self.artifact_root / filename).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _persist_text(self, filename: str, content: str) -> None:
        if not self.persist_artifacts:
            return
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        (self.artifact_root / filename).write_text(content.strip() + "\n", encoding="utf-8")


def _build_observation_user_prompt(*, job_id: str, transcript_text: str, frame_count: int) -> str:
    return (
        f"Video id: {job_id}\n"
        f"Optional transcript:\n{transcript_text or '[none provided]'}\n\n"
        f"Analyze these {frame_count} sampled frames in chronological order. "
        "Capture exact visible facts that would help a judge compare captions: "
        "setting, subjects, colors, countable objects, actions, scene changes, "
        "visible text, camera movement, and transcript-backed speech."
    )


def _build_direct_vision_prompt(*, requested_styles: list[StyleName], transcript_text: str) -> str:
    style_lines = "\n".join(f"- {style_name}" for style_name in requested_styles)
    return (
        "You are captioning a short video from sampled frames shown in chronological order.\n"
        "Create one short, accurate caption for each requested style.\n\n"
        f"Styles:\n{style_lines}\n\n"
        "Style guide:\n"
        "formal = factual and neutral.\n"
        "sarcastic = dry and ironic.\n"
        "humorous_tech = funny with a software/developer angle.\n"
        "humorous_non_tech = funny for a general audience, no tech jargon.\n\n"
        "Keep each caption one sentence. Mention the main subject/action/setting. "
        "Do not invent details not visible in the frames. Return only valid JSON.\n"
        f"Transcript if any: {transcript_text or 'none'}\n\n"
        "JSON shape:\n"
        + json.dumps({style_name: "caption text" for style_name in requested_styles}, indent=2)
    )


def _build_verified_scene_description_prompt(*, transcript_text: str) -> str:
    return (
        "These are frames sampled across a short video clip, in chronological order. "
        "Note the setting, main subjects, specific action or motion, camera/scene changes, "
        "and any readable on-screen text. Write 2-4 dense, factual sentences. "
        "Be specific, do not generalize, and do not mention frames or analysis. "
        "Only quote visible text if it is large, central, and clearly readable. "
        "Use generic human descriptions unless a specific identity is directly relevant and visually certain. "
        f"Optional transcript: {transcript_text or 'none'}"
    )


def _build_verified_scene_verification_prompt(*, draft: str) -> str:
    return (
        f"Here is a draft description of these video frames:\n{draft}\n\n"
        "Check it against the actual frames. If it is accurate and specific, repeat it unchanged. "
        "If anything is wrong, too generic, or unsupported, correct it. "
        "Remove exact quoted text, brand names, signs, ethnicity, identity labels, or location claims unless "
        "they are clearly visible and central in the frames. Prefer generic wording when unsure. "
        "Output only the final factual description. Do not mention frames, AI, uncertainty, or analysis."
    )


def _build_verified_scene_caption_prompt(
    *,
    style_name: StyleName,
    verified_description: str,
    prior_captions: list[str] | None = None,
) -> str:
    variety_note = ""
    if prior_captions:
        variety_note = (
            "\n\nCaptions already written for this clip in other styles. "
            "Use a different sentence structure and comedic angle: "
            + " | ".join(prior_captions)
        )
    return (
        f"{VERIFIED_STYLE_PROMPTS.get(style_name, STYLE_DESCRIPTIONS.get(style_name, 'Match the requested style.'))}\n\n"
        f"Factual description of the video:\n{verified_description}\n\n"
        "Write ONE caption, 25 to 60 words, as if you personally watched the video. "
        "Never mention computer vision, models, detection, frames, prompts, pipelines, or uncertainty. "
        "Do not invent details beyond the description. Do not quote signs, brands, or identity labels unless "
        "they are explicitly present in the factual description. Output only the caption text."
        f"{variety_note}"
    )


def _clean_caption(caption: str) -> str:
    text = str(caption).strip()

    def decode_unicode_escape(match: re.Match[str]) -> str:
        try:
            return json.loads(f'"{match.group(0)}"')
        except (json.JSONDecodeError, UnicodeDecodeError):
            return match.group(0)

    text = UNICODE_ESCAPE_SEQUENCE.sub(decode_unicode_escape, text)
    text = text.replace("\\n", " ").replace("\\r", " ").replace("\\t", " ")
    text = re.sub(r"\s+", " ", text).strip()

    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        text = text[1:-1].strip()
    return text


def _sanitize_observations(observations: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(observations)
    uncertainties = cleaned.get("uncertainties")
    if isinstance(uncertainties, list):
        cleaned["uncertainties"] = [_sanitize_uncertainty(str(item)) for item in uncertainties]
    return cleaned


def _sanitize_uncertainty(text: str) -> str:
    lowered = text.lower()
    if "exact city" in lowered or "exact location" in lowered:
        if any(marker in lowered for marker in ("suggest", "may be", "might be", "probably", "looks like")):
            return re.split(r"\bthough\b|\bbut\b|;|,", text, maxsplit=1, flags=re.IGNORECASE)[0].strip() + "."
    return text


def _needs_style_retry(style_name: StyleName, caption: str) -> bool:
    normalized = caption.lower()
    if style_name == "humorous_tech":
        return not any(word in normalized for word in TECH_STYLE_WORDS)
    if style_name == "sarcastic":
        return not any(marker in normalized for marker in SARCASM_STYLE_MARKERS)
    return False


def _fallback_caption(style_name: StyleName, observations: dict[str, Any]) -> str:
    summary = str(observations.get("summary") or observations.get("setting") or "").strip()
    subjects = ", ".join(str(item) for item in observations.get("subjects", [])[:2])
    actions = ", ".join(str(item) for item in observations.get("actions", [])[:2])
    base = summary or f"The video shows {subjects or 'visible subjects'} with {actions or 'visible activity'}."
    if style_name == "formal":
        return base
    if style_name == "sarcastic":
        return f"{base} A very serious moment for ordinary visual evidence."
    if style_name == "humorous_tech":
        return f"{base} The scene ships its visual update with no rollback needed."
    return f"{base} It is doing its best to make everyday motion look eventful."
