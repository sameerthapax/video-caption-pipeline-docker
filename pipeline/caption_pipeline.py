from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from pipeline.extract_frames import extract_frames_for_video
from prompts.prompt_loader import load_prompt
from schemas.caption import (
    STYLE_ORDER,
    CaptionVariant,
    JudgeResult,
    ObservationResult,
    StyleName,
    build_caption_variant_json_schema,
    build_combined_captions_json_schema,
    build_judge_json_schema,
    build_observation_json_schema,
)
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
CAPTION_SHARED_SYSTEM_PROMPT = load_prompt("caption_shared_system.txt")
CAPTION_STYLE_CONCURRENCY = 2
JUDGE_RETRY_THRESHOLD = 0.75
CREATIVE_RISK_PHRASES = (
    "three hours",
    "three blocks",
    "since 2019",
    "treat bag",
    "other room",
    "spreadsheet",
    "jesus",
    "bless",
    "45 minutes",
    "wrong exit",
    "spiritual",
    "frontal lobe",
    "existential",
    "maintenance request",
    "leak",
    "charcuterie",
    "snacks",
    "office party",
    "group chat",
    "personality trait",
    "camera frustum",
    "fixed elevated angle",
    "render distance",
    "draw distance",
    "monitor back panel is the real protagonist",
    "cross pendant",
    "group project",
    "one slide",
    "cross necklace",
    "monitor bezel",
    "life choices",
    "camera tilts",
    "fixed angle",
    "korea",
    "one quick bug",
)
CREATIVE_RISK_PATTERNS = (
    re.compile(r"\b(?:19|20)\d{2}\b", re.IGNORECASE),
    re.compile(
        r"\b\d+\s*(?:a\.m\.|p\.m\.|am|pm|minutes?|hours?|blocks?)(?=\s|[.,!?;:]|$)",
        re.IGNORECASE,
    ),
)
CREATIVE_ALWAYS_RETRY_PHRASES = (
    "personality trait",
    "camera frustum",
    "fixed elevated angle",
    "render distance",
    "draw distance",
    "monitor back panel is the real protagonist",
    "cross pendant",
    "cross necklace",
    "spiritual",
    "jesus",
    "bless",
    "monitor bezel",
    "life choices",
    "camera tilts",
    "fixed angle",
    "korea",
)
TECH_STYLE_WORDS = {
    "api",
    "bug",
    "cache",
    "commit",
    "compile",
    "debug",
    "deploy",
    "latency",
    "log",
    "lod",
    "packet",
    "pipeline",
    "process",
    "queue",
    "render",
    "rollback",
    "runtime",
    "scheduler",
}
SARCASM_STYLE_MARKERS = {
    "apparently",
    "as if",
    "because",
    "clearly",
    "naturally",
    "of course",
    "obviously",
    "i watched",
    "like",
    "pretending",
    "serious",
    "somehow",
    "thrilling",
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
                job_id=job_id,
                frame_artifact=frame_artifact,
                transcript_text=transcript_text,
                requested_styles=requested_styles,
            )
        elif settings.caption_pipeline_mode == "direct_vision":
            captions, observations, checks = await self._run_direct_vision_mode(
                job_id=job_id,
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
                "captions": {style_name: variant.model_dump(mode="json") for style_name, variant in captions.items()},
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
        job_id: str,
        frame_artifact: FrameExtractionArtifact,
        transcript_text: str,
        requested_styles: list[StyleName],
    ) -> tuple[dict[str, CaptionVariant], dict[str, Any], dict[str, JudgeResult]]:
        frame_paths = [frame.local_path for frame in frame_artifact.frames]
        description = await self.llm_client.generate_text(
            prompt=_build_verified_scene_description_prompt(transcript_text=transcript_text),
            image_paths=frame_paths,
            temperature=0.2,
            endpoint_path=VISION_ENDPOINT,
        )
        verified_description = await self.llm_client.generate_text(
            prompt=_build_verified_scene_verification_prompt(draft=description),
            image_paths=frame_paths,
            temperature=0.1,
            endpoint_path=VISION_ENDPOINT,
        )
        observations = {
            "summary": "Verified scene mode",
            "verified_description": verified_description,
            "frame_count": len(frame_paths),
        }
        self._persist_text("verified_description.txt", verified_description)

        captions = await self._generate_verified_scene_captions(
            requested_styles=requested_styles,
            verified_description=verified_description,
        )

        checks = await self._run_checks(
            requested_styles=requested_styles,
            observations=observations,
            captions=captions,
            frame_paths=frame_paths,
        )
        return captions, observations, checks

    async def _generate_verified_scene_captions(
        self,
        *,
        requested_styles: list[StyleName],
        verified_description: str,
    ) -> dict[str, CaptionVariant]:
        captions: dict[str, CaptionVariant] = {}
        prior_captions: list[str] = []
        for style_name in requested_styles:
            prompt = _build_verified_scene_caption_prompt(
                style_name=style_name,
                verified_description=verified_description,
                prior_captions=prior_captions,
            )
            caption = await self.llm_client.generate_text(
                prompt=prompt,
                temperature=None,
                endpoint_path=CAPTION_ENDPOINT,
            )
            caption = _clean_verified_caption(caption)
            if _needs_specificity_retry(
                style_name=style_name,
                caption=caption,
                verified_description=verified_description,
            ):
                retry = await self.llm_client.generate_text(
                    prompt=_build_specificity_retry_prompt(
                        original_prompt=prompt,
                        previous_caption=caption,
                    ),
                    temperature=None,
                    endpoint_path=CAPTION_ENDPOINT,
                )
                caption = _clean_verified_caption(retry)
            risky_terms = _find_unsupported_risk_terms(
                style_name=style_name,
                caption=caption,
                verified_description=verified_description,
                prior_captions=prior_captions,
            )
            if risky_terms:
                retry = await self.llm_client.generate_text(
                    prompt=_build_risk_retry_prompt(
                        style_name=style_name,
                        original_prompt=prompt,
                        previous_caption=caption,
                        risky_terms=risky_terms,
                    ),
                    temperature=None,
                    endpoint_path=CAPTION_ENDPOINT,
                )
                caption = _clean_verified_caption(retry)
            captions[style_name] = CaptionVariant(style_name=style_name, caption=caption)
            prior_captions.append(caption)
        return captions

    async def _run_direct_vision_mode(
        self,
        *,
        job_id: str,
        frame_artifact: FrameExtractionArtifact,
        transcript_text: str,
        requested_styles: list[StyleName],
    ) -> tuple[dict[str, CaptionVariant], dict[str, Any], dict[str, JudgeResult]]:
        _ = job_id
        frame_paths = [frame.local_path for frame in frame_artifact.frames]
        payload = await self.llm_client.generate_json(
            prompt=_build_direct_vision_prompt(requested_styles=requested_styles, transcript_text=transcript_text),
            image_paths=frame_paths,
            temperature=0.2,
            response_schema=build_combined_captions_json_schema(requested_styles),
            response_schema_name="direct_vision_captions",
            endpoint_path=VISION_ENDPOINT,
        )
        captions = {
            style_name: CaptionVariant.model_validate(payload[style_name])
            for style_name in requested_styles
        }
        observations = {
            "summary": "Direct vision caption mode",
            "frame_count": len(frame_paths),
            "transcript_used": bool(transcript_text.strip()),
        }
        checks = await self._run_checks(
            requested_styles=requested_styles,
            observations=observations,
            captions=captions,
            frame_paths=frame_paths,
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
        _ = job_id
        frame_paths = [frame.local_path for frame in frame_artifact.frames]
        observation_payload = await self.llm_client.generate_json(
            system_prompt=load_prompt("perception_system.txt"),
            prompt=_build_observation_user_prompt(job_id=job_id, transcript_text=transcript_text, frame_count=len(frame_paths)),
            image_paths=frame_paths,
            temperature=0.1,
            response_schema=build_observation_json_schema(),
            response_schema_name="observations",
            endpoint_path=VISION_ENDPOINT,
        )
        observations = ObservationResult.model_validate(observation_payload)
        self._persist_json("observations.json", observations.model_dump(mode="json"))

        if settings.observation_caption_mode not in {"combined", "per_style"}:
            raise ValueError(f"Unsupported OBSERVATION_CAPTION_MODE: {settings.observation_caption_mode}")
        captions = await self._generate_independent_style_captions(
            requested_styles=requested_styles,
            prompt_builder=lambda style_name: _build_per_style_caption_prompt(style_name=style_name, observations=observations),
        )

        checks = await self._run_checks(
            requested_styles=requested_styles,
            observations=observations.model_dump(mode="json"),
            captions=captions,
            frame_paths=frame_paths,
        )
        return captions, observations.model_dump(mode="json"), checks

    async def _run_checks(
        self,
        *,
        requested_styles: list[StyleName],
        observations: dict[str, Any],
        captions: dict[str, CaptionVariant],
        frame_paths: list[str],
    ) -> dict[str, JudgeResult]:
        if not settings.run_judge_checks:
            return {}

        async def judge_style(style_name: StyleName) -> JudgeResult:
            payload = await self.llm_client.generate_json(
                prompt=_build_judge_prompt(
                    observations=observations,
                    style_name=style_name,
                    caption_variant=captions[style_name],
                ),
                image_paths=frame_paths,
                temperature=None,
                response_schema=build_judge_json_schema(),
                response_schema_name=f"{style_name}_judge",
                endpoint_path=JUDGE_ENDPOINT,
            )
            result = JudgeResult.model_validate(payload)
            return result.model_copy(update={"score": min(result.accuracy, result.style_match)})

        tasks = {style_name: judge_style(style_name) for style_name in requested_styles}
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        checks: dict[str, JudgeResult] = {}
        judge_failures: dict[str, str] = {}
        for style_name, result in zip(tasks.keys(), results, strict=True):
            if isinstance(result, Exception):
                logger.warning("Judge check failed for style %s: %s", style_name, result)
                judge_failures[style_name] = str(result)
                continue
            checks[style_name] = result

        original_variants = {
            style_name: captions[style_name]
            for style_name, check in checks.items()
            if _needs_judge_retry(check)
        }
        retry_tasks = {
            style_name: self.llm_client.generate_json(
                system_prompt=_build_caption_system_prompt(style_name),
                prompt=_build_judge_retry_prompt(
                    observations=observations,
                    previous_variant=captions[style_name],
                    judge_result=check,
                ),
                temperature=None,
                response_schema=build_caption_variant_json_schema(style_name),
                response_schema_name=f"{style_name}_caption",
                endpoint_path=CAPTION_ENDPOINT,
            )
            for style_name, check in checks.items()
            if _needs_judge_retry(check)
        }
        retry_results = await asyncio.gather(*retry_tasks.values(), return_exceptions=True)
        successful_retries: list[StyleName] = []
        for style_name, result in zip(retry_tasks.keys(), retry_results, strict=True):
            if isinstance(result, Exception):
                logger.warning("Judge-guided caption retry failed for style %s: %s", style_name, result)
                continue
            try:
                captions[style_name] = CaptionVariant.model_validate(result)
            except ValidationError as exc:
                logger.warning("Judge-guided caption retry was invalid for style %s: %s", style_name, exc)
                continue
            successful_retries.append(style_name)

        rejudge_tasks = {style_name: judge_style(style_name) for style_name in successful_retries}
        rejudge_results = await asyncio.gather(*rejudge_tasks.values(), return_exceptions=True)
        for style_name, result in zip(rejudge_tasks.keys(), rejudge_results, strict=True):
            if isinstance(result, Exception):
                captions[style_name] = original_variants[style_name]
                logger.warning("Re-judge failed for style %s; restoring original caption: %s", style_name, result)
                continue
            checks[style_name] = result

        for style_name, check in checks.items():
            logger.info(
                "Judge check completed for style %s: accuracy=%.3f style_match=%.3f score=%.3f feedback=%s",
                style_name,
                check.accuracy,
                check.style_match,
                check.score,
                check.feedback,
            )
        self._persist_json("checks.json", {style_name: check.model_dump(mode="json") for style_name, check in checks.items()})
        if judge_failures:
            self._persist_json("judge_failures.json", judge_failures)
        return checks

    async def _generate_independent_style_captions(
        self,
        *,
        requested_styles: list[StyleName],
        prompt_builder,
    ) -> dict[str, CaptionVariant]:
        semaphore = asyncio.Semaphore(CAPTION_STYLE_CONCURRENCY)

        async def generate_for_style(style_name: StyleName) -> tuple[StyleName, CaptionVariant]:
            async with semaphore:
                prompt = prompt_builder(style_name)
                response_schema = build_caption_variant_json_schema(style_name)
                response_schema_name = f"{style_name}_caption"
                payload = await self.llm_client.generate_json(
                    system_prompt=_build_caption_system_prompt(style_name),
                    prompt=prompt,
                    temperature=None,
                    response_schema=response_schema,
                    response_schema_name=response_schema_name,
                    endpoint_path=CAPTION_ENDPOINT,
                )
                variant = CaptionVariant.model_validate(payload)
                if _needs_style_retry(style_name, variant.caption):
                    retry_payload = await self.llm_client.generate_json(
                        system_prompt=_build_caption_system_prompt(style_name),
                        prompt=_build_style_retry_prompt(
                            original_prompt=prompt,
                            previous_variant=variant,
                        ),
                        temperature=None,
                        response_schema=response_schema,
                        response_schema_name=response_schema_name,
                        endpoint_path=CAPTION_ENDPOINT,
                    )
                    variant = CaptionVariant.model_validate(retry_payload)
            return style_name, variant

        results = await asyncio.gather(*(generate_for_style(style_name) for style_name in requested_styles))
        return {style_name: variant for style_name, variant in results}

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
        "Capture exact visible facts that would help a judge compare captions."
    )


def _build_direct_vision_prompt(*, requested_styles: list[StyleName], transcript_text: str) -> str:
    style_lines = "\n".join(f"- {style_name}" for style_name in requested_styles)
    schema_preview = {
        style_name: {
            "style_name": style_name,
            "caption": "caption text",
            "grounded_facts_used": ["fact 1"],
            "safety_notes": [],
        }
        for style_name in requested_styles
    }
    return (
        "You are captioning a short video from sampled frames shown in chronological order.\n"
        "Create one vivid, grounded caption for each requested style.\n\n"
        f"Styles:\n{style_lines}\n\n"
        "Rules:\n"
        "- Keep every caption grounded in what is visible in the frames.\n"
        "- Use the transcript only when it supports a visible or audible fact.\n"
        "- Use one or two detailed sentences for formal captions and up to three connected sentences for styled captions.\n"
        "- Make humorous captions feel tailored to several visible details, with a clear setup and payoff.\n"
        "- Never invent motives, hidden events, or identities.\n"
        f"Transcript if any: {transcript_text or 'none'}\n\n"
        "Return strict JSON only in this shape:\n"
        f"{json.dumps(schema_preview, indent=2)}"
    )


def _build_verified_scene_description_prompt(*, transcript_text: str) -> str:
    return (
        "These are frames sampled across a short video clip, in chronological order. "
        "Note the setting, main subjects, specific action or motion, camera or scene changes, "
        "and any readable on-screen text. Write 2 to 4 dense factual sentences. "
        "Be specific, do not generalize, and do not mention frames or analysis. "
        "Use generic human descriptions unless a specific identity is visually certain. "
        f"Optional transcript: {transcript_text or 'none'}"
    )


def _build_verified_scene_verification_prompt(*, draft: str) -> str:
    return (
        f"Here is a draft description of these video frames:\n{draft}\n\n"
        "Check it against the actual frames. If it is accurate and specific, repeat it unchanged. "
        "If anything is wrong, too generic, too broad, or unsupported, correct it. "
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
            "Use a different sentence structure and, for creative styles, a different comedic angle: "
            + " | ".join(prior_captions)
        )
    if style_name == "formal":
        style_boundary = (
            "Stay factual, polished, and objective. Do not use humor, fictional framing, or invented context. "
            "Include the core visible subject, action, setting, background, and key objects when supported. Prefer "
            "one or two clean sentences, usually two when the description contains enough detail. Avoid risky minor "
            "specifics, including jewelry, clothing, accessories, exact lighting, or time-of-day claims, unless they "
            "are clearly visible, important, and explicitly supported by the factual description. Do not emphasize "
            "hairstyle or jewelry; prefer core actions and scene details. Prefer two clean sentences when enough detail exists. "
            "Avoid quoting visible text or full sign wording unless the factual description marks it as central, important, "
            "large, and unmistakably readable. Otherwise say that a sign or commercial building is visible, or omit it."
        )
    else:
        style_boundary = (
            "For sarcastic and humorous styles, clearly fictional comparisons, first-person reactions, meme framing, "
            "and playful invented context are allowed when they are obviously jokes. They must not be presented as "
            "literal visible facts. Prioritize a strong human caption over exhaustive visual coverage. Include two to "
            "four concrete anchors from the factual description: explicitly include the main visible subject or action, "
            "plus one to three anchors such as the setting, another visible action, "
            "distinctive object, lighting or color, background, or motion. Use those anchors naturally inside the joke; "
            "do not list every detail. Avoid generic meme lines that could describe many unrelated clips. Prefer one or "
            "two substantial sentences. A very short one-liner is acceptable only when highly specific to this clip. "
            "Do not invent unrelated context that replaces the actual scene, and do not contradict the verified description. "
            "Avoid unsupported concrete details such as exact durations or years, unseen objects or locations, food or "
            "treats, spreadsheets, meetings, religious references, medical or political references, and precise internal "
            "mental states. Do not use phrases such as 'three hours', 'since 2019', 'treat bag', 'other room', "
            "'spreadsheet', 'Jesus', 'bless', or '45 minutes' unless the factual description supports them. Safer "
            "figurative angles include landlord, NPCs, render loops, party entrances, deadline energy, infinite loops, "
            "dramatic entrances, and office questlines. If the joke could apply to many unrelated videos, rewrite it "
            "with more visible anchors. Use vague figurative wording instead of exact invented specifics. Sound like a "
            "person reacting to the clip, not a vision-analysis report. Avoid sterile camera or scene-parser language such "
            "as camera frustum, fixed elevated angle, render-distance shift, draw distance, viewpoint, foreground/background, "
            "or repeated LOD and biome references. Do not turn a monitor back panel into a protagonist."
        )
    scene_fit_guidance = _build_scene_fit_guidance(verified_description)
    length_guidance = {
        "formal": "Use one or two clean sentences.",
        "sarcastic": "Use one substantial sentence or two shorter sentences, about 25 to 55 words.",
        "humorous_tech": "Use one or two sentences, about 30 to 65 words.",
        "humorous_non_tech": "Use one or two sentences, about 25 to 60 words.",
    }[style_name]
    return (
        f"{load_prompt(STYLE_PROMPTS[style_name])}\n\n"
        f"Factual description of the video:\n{verified_description}\n\n"
        f"{style_boundary}\n\n"
        f"{scene_fit_guidance}\n\n"
        f"{length_guidance} Pick one joke angle and two to four visible anchors; do not overload the caption.\n\n"
        "Write ONE concise caption as if you watched the video. "
        "Never mention computer vision, models, detection, frames, prompts, pipelines, or uncertainty. "
        "Do not identify real people or assert exact locations, brands, quoted text, or unseen literal events unless "
        "they are explicitly present in the factual description. Output only the caption text."
        f"{variety_note}"
    )


def _clean_verified_caption(caption: str) -> str:
    text = caption.strip()
    if len(text) >= 2 and ((text.startswith('"') and text.endswith('"')) or (text.startswith("“") and text.endswith("”"))):
        text = text[1:-1].strip()
    if text.count('"') % 2:
        if text.endswith('"'):
            text = text[:-1].rstrip()
        elif text.startswith('"'):
            text = text[1:].lstrip()
    if (text.count("“") + text.count("”")) % 2:
        if text.endswith("”"):
            text = text[:-1].rstrip()
        elif text.startswith("“"):
            text = text[1:].lstrip()
    return text


def _needs_specificity_retry(*, style_name: StyleName, caption: str, verified_description: str) -> bool:
    if style_name == "formal":
        return False
    anchor_count = _count_scene_anchor_terms(caption=caption, verified_description=verified_description)
    has_main_anchor = _has_main_scene_anchor(caption=caption, verified_description=verified_description)
    if anchor_count < 2 or not has_main_anchor:
        return True
    word_count = len(re.findall(r"\b[\w'-]+\b", caption))
    return word_count < 12 and anchor_count < 3


def _find_unsupported_risk_terms(
    *,
    style_name: StyleName,
    caption: str,
    verified_description: str,
    prior_captions: list[str] | None = None,
) -> list[str]:
    if style_name == "formal":
        risks = _find_nonessential_formal_quotes(caption=caption, verified_description=verified_description)
        if re.search(r"\bfocused\b", caption, re.IGNORECASE):
            risks.append("focused")
        return risks
    normalized_caption = " ".join(caption.lower().split())
    normalized_description = " ".join(verified_description.lower().split())
    risky_terms = [phrase for phrase in CREATIVE_ALWAYS_RETRY_PHRASES if phrase in normalized_caption]
    risky_terms.extend(
        phrase
        for phrase in CREATIVE_RISK_PHRASES
        if phrase not in risky_terms and phrase in normalized_caption and phrase not in normalized_description
    )
    for pattern in CREATIVE_RISK_PATTERNS:
        for match in pattern.finditer(caption):
            matched_text = match.group(0)
            if matched_text.lower() not in normalized_description and matched_text.lower() not in risky_terms:
                risky_terms.append(matched_text.lower())
    if re.search(r"\bdev\b", caption, re.IGNORECASE) and not re.search(
        r"\b(?:dev|developer|programming|software engineer|coder)\b",
        verified_description,
        re.IGNORECASE,
    ):
        risky_terms.append("dev")
    if style_name != "humorous_tech" and re.search(r"\brender pass\b", caption, re.IGNORECASE):
        risky_terms.append("render pass")
    if re.search(r"\blandlord\b", caption, re.IGNORECASE):
        kitten_scene = bool(re.search(r"\b(?:kitten|cat|feline)\b", verified_description, re.IGNORECASE))
        landlord_already_used = any(
            re.search(r"\blandlord\b", prior_caption, re.IGNORECASE)
            for prior_caption in (prior_captions or [])
        )
        if not kitten_scene or landlord_already_used:
            risky_terms.append("landlord")
    for quote in _extract_quoted_phrases(caption):
        if quote.lower() in normalized_description and not _description_marks_text_central(verified_description):
            risky_terms.append(f'quoted sign text: "{quote}"')
    if len(re.findall(r"\b[\w'-]+\b", caption)) > 65:
        risky_terms.append("caption exceeds 65 words")
    return risky_terms


def _build_scene_fit_guidance(verified_description: str) -> str:
    description = verified_description.lower()
    if any(term in description for term in ("traffic", "avenue", "cars", "taxis", "bus")):
        return (
            "Traffic scene fit: favor traffic, motion blur, green lights, ginkgo trees, mountains, or towers. "
            "Good joke angles include city commute, NPCs, frame rate, a green-light loop, indifferent mountains, or "
            "overdressed ginkgo trees. Avoid landlord, exact sign text, and forced PR or merge-conflict jokes."
        )
    if any(term in description for term in ("kitten", "cat", "feline")):
        return (
            "Kitten scene fit: favor the orange kitten, bushes, raised tail, pink nose, dappled sunlight, or its walk "
            "toward the viewer. Tiny inspector, dramatic entrance, NPC aggro, or tail-flag jokes fit. Landlord may fit "
            "a tail-raised approach, but use it sparingly."
        )
    if any(term in description for term in ("office", "keyboard", "typing", "monitor", "desk")):
        return (
            "Office scene fit: favor typing, keyboard, monitor, mouse or cable, ceiling lights, plants, or glass "
            "partitions. Use office questline, keyboard-input loop, monitor drama, waiting mouse, or plants-as-witnesses. "
            "Avoid landlord, sign text, jewelry, religion, hairstyle jokes, and personal-appearance framing."
        )
    return "Use one scene-fit joke built from the main visible subject or action and a few distinctive anchors."


def _extract_quoted_phrases(value: str) -> list[str]:
    matches = re.findall(r'"([^"\n]+)"|“([^”\n]+)”', value)
    return [(straight or curly).strip() for straight, curly in matches if (straight or curly).strip()]


def _find_nonessential_formal_quotes(*, caption: str, verified_description: str) -> list[str]:
    quoted_text = _extract_quoted_phrases(caption)
    if not quoted_text:
        return []
    description = verified_description.lower()
    marked_readable = _description_marks_text_central(verified_description)
    for quote in quoted_text:
        text = quote.lower()
        if text not in description or not marked_readable:
            return ["nonessential quoted sign text"]
    return []


def _description_marks_text_central(verified_description: str) -> bool:
    description = verified_description.lower()
    central = "central" in description
    readable = any(marker in description for marker in ("clearly readable", "unmistakably readable"))
    prominent = "large" in description or "important" in description
    return central and readable and prominent


def _count_scene_anchor_terms(*, caption: str, verified_description: str) -> int:
    ignored = {
        "about", "after", "again", "against", "before", "being", "could", "from", "into", "just", "like",
        "more", "over", "really", "that", "their", "there", "these", "they", "this", "through", "very", "with",
    }

    def terms(value: str) -> set[str]:
        return {
            word
            for word in re.findall(r"[a-z0-9]+", value.lower())
            if len(word) >= 4 and word not in ignored
        }

    return len(terms(caption) & terms(verified_description))


def _has_main_scene_anchor(*, caption: str, verified_description: str) -> bool:
    main_clause = re.split(r"[.!?]", verified_description, maxsplit=1)[0]
    clause_terms = _meaningful_terms(main_clause)
    common_actions = {
        "approaches", "drives", "emerges", "moves", "typing", "types", "walks", "walking", "writes",
    }
    main_terms = clause_terms[:3] + [
        term
        for term in clause_terms[:10]
        if term.endswith("ing") or term in common_actions
    ]
    caption_terms = set(_meaningful_terms(caption))
    return any(term in caption_terms for term in main_terms)


def _meaningful_terms(value: str) -> list[str]:
    ignored = {
        "about", "after", "again", "against", "before", "being", "could", "from", "into", "just", "like",
        "more", "over", "really", "small", "that", "their", "there", "these", "they", "this", "through", "very",
        "with", "visible", "along", "beneath", "across", "toward", "towards", "under",
    }
    return [
        word
        for word in re.findall(r"[a-z0-9]+", value.lower())
        if len(word) >= 4 and word not in ignored
    ]


def _build_specificity_retry_prompt(*, original_prompt: str, previous_caption: str) -> str:
    return (
        f"{original_prompt}\n\nPrevious draft:\n{previous_caption}\n\n"
        "Rewrite the draft because it is too short, generic, or detached from the scene. Keep one strong joke, but "
        "explicitly include the main visible subject or action and one to three additional visible anchors from the "
        "factual description. Prefer one or two substantial sentences. Do not turn it into a list or a formal scene "
        "summary. Output only the revised caption text."
    )


def _build_risk_retry_prompt(
    *,
    style_name: StyleName,
    original_prompt: str,
    previous_caption: str,
    risky_terms: list[str],
) -> str:
    if style_name == "formal":
        revision_instruction = (
            "Keep the caption factual and polished, but remove nonessential quoted sign text and incidental details. "
            "Use a generic description such as a visible sign or commercial building unless the wording is central, "
            "important, and unmistakably readable. Preserve the main subject, action, setting, and key objects."
        )
    else:
        revision_instruction = (
            "Keep the caption funny and anchored, but remove noncentral sign text, repeated joke frames, camera-analysis "
            "wording, and overloaded detail. Use one strong scene-fit joke with two to four visible anchors."
        )
    return (
        f"{original_prompt}\n\nPrevious draft:\n{previous_caption}\n\n"
        f"Unsupported concrete details detected: {', '.join(risky_terms)}. "
        f"{revision_instruction} "
        "Output only the revised caption text."
    )


def _build_combined_caption_prompt(*, requested_styles: list[StyleName], observations: ObservationResult) -> str:
    style_descriptions = "\n".join(f"- {style_name}" for style_name in requested_styles)
    schema_preview = {
        style_name: {
            "style_name": style_name,
            "caption": "caption text",
            "grounded_facts_used": ["fact 1"],
            "safety_notes": [],
        }
        for style_name in requested_styles
    }
    return (
        "Write one final caption for each requested style from the structured video observations.\n\n"
        f"Requested styles:\n{style_descriptions}\n\n"
        "Return strict JSON only in this shape:\n"
        f"{json.dumps(schema_preview, indent=2)}\n\n"
        f"Observations:\n{json.dumps(observations.model_dump(mode='json'), indent=2)}"
    )


def _build_per_style_caption_prompt(*, style_name: StyleName, observations: ObservationResult) -> str:
    schema_preview = {
        "style_name": style_name,
        "caption": "caption text",
        "grounded_facts_used": ["fact 1"],
        "safety_notes": ["optional note"],
    }
    return (
        "Return JSON only in this shape:\n"
        f"{json.dumps(schema_preview, indent=2)}\n\n"
        f"Grounded video facts:\n{json.dumps(observations.model_dump(mode='json'), indent=2)}"
    )


def _build_caption_system_prompt(style_name: StyleName) -> str:
    return f"{CAPTION_SHARED_SYSTEM_PROMPT}\n\n{load_prompt(STYLE_PROMPTS[style_name])}"


def _needs_style_retry(style_name: StyleName, caption: str) -> bool:
    normalized_caption = caption.lower()
    if style_name == "humorous_tech":
        return not any(word in normalized_caption for word in TECH_STYLE_WORDS)
    if style_name == "sarcastic":
        return not any(marker in normalized_caption for marker in SARCASM_STYLE_MARKERS)
    return False


def _build_style_retry_prompt(*, original_prompt: str, previous_variant: CaptionVariant) -> str:
    return (
        f"{original_prompt}\n\n"
        f"Previous JSON response:\n{json.dumps(previous_variant.model_dump(mode='json'), indent=2)}\n\n"
        "Rewrite the previous caption because it was too plain for the requested style. Keep the same grounded "
        "facts, do not add new facts, and make the target style obvious. Return only the same JSON shape required "
        "above."
    )


def _needs_judge_retry(result: JudgeResult) -> bool:
    return result.accuracy < JUDGE_RETRY_THRESHOLD or result.style_match < JUDGE_RETRY_THRESHOLD


def _build_judge_retry_prompt(
    *,
    observations: dict[str, Any],
    previous_variant: CaptionVariant,
    judge_result: JudgeResult,
) -> str:
    schema_preview = {
        "style_name": previous_variant.style_name,
        "caption": "revised caption text",
        "grounded_facts_used": ["fact 1"],
        "safety_notes": [],
    }
    payload = {
        "observations": observations,
        "previous_caption": previous_variant.model_dump(mode="json"),
        "evaluation": judge_result.model_dump(mode="json"),
    }
    return (
        "Revise this caption once by applying the evaluator's general prompt guidance to the current grounded input. "
        "Preserve supported facts, remove or correct unsupported claims, and strengthen the requested style without "
        "inventing any new fact.\n\n"
        "Return strict JSON only in this shape:\n"
        f"{json.dumps(schema_preview, indent=2)}\n\n"
        f"Revision input:\n{json.dumps(payload, indent=2)}"
    )


def _build_judge_prompt(*, observations: dict[str, Any], style_name: StyleName, caption_variant: CaptionVariant) -> str:
    judge_prompt = load_prompt("judge.txt")
    payload = {
        "target_style": style_name,
        "observations": observations,
        "caption": caption_variant.caption,
        "grounded_facts_used": caption_variant.grounded_facts_used,
    }
    return f"{judge_prompt}\n\nJudge input:\n{json.dumps(payload, indent=2)}"
