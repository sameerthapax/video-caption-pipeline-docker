from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any

from pipeline.extract_frames import extract_frames_for_video
from prompts.prompt_loader import load_prompt, render_prompt
from schemas.caption import CandidateJudgeEvaluation, CaptionCandidates, CaptionVariant, JudgeResult, STYLE_ORDER, StyleName, build_caption_candidates_json_schema, build_judge_json_schema, build_observation_json_schema
from schemas.frames import FrameExtractionArtifact
from schemas.video import VideoMetadata
from worker.config.settings import settings

logger = logging.getLogger("gemma-caption-pipe.worker")

STYLE_PROMPTS = {
    "formal": "style_formal.txt",
    "sarcastic": "style_sarcastic.txt",
    "humorous_tech": "style_humorous_tech.txt",
    "humorous_non_tech": "style_humorous_non_tech.txt",
}

VISION_ENDPOINT = "/vision/chat/completions"
CAPTION_ENDPOINT = "/caption/chat/completions"
JUDGE_ENDPOINT = "/judge/chat/completions"

TECH_STYLE_WORDS = {
    "agent",
    "api",
    "bandwidth",
    "bug",
    "build",
    "cache",
    "code",
    "commit",
    "compiler",
    "database",
    "debug",
    "deploy",
    "inference",
    "kernel",
    "latency",
    "log",
    "model",
    "network",
    "node",
    "packet",
    "pipeline",
    "process",
    "queue",
    "rollback",
    "runtime",
    "scheduler",
    "server",
    "stack",
    "thread",
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
CREATIVE_STYLES = {"sarcastic", "humorous_tech", "humorous_non_tech"}
STYLE_WORD_LIMITS: dict[StyleName, tuple[int, int]] = {
    "formal": (10, 36),
    "sarcastic": (8, 30),
    "humorous_tech": (10, 32),
    "humorous_non_tech": (8, 30),
}
NUMBER_TOKEN = re.compile(r"(?<!\w)\d+(?:\.\d+)?%?(?!\w)")
UNSUPPORTED_PROP_TERMS = ("appointment", "oven", "pan", "snack", "tennis ball", "treat")
UNSUPPORTED_HISTORY_PHRASES = ("for the first time",)


class CaptionPipeline:
    def __init__(
        self,
        *,
        llm_client,
        judge_client,
        artifact_root: Path,
        persist_artifacts: bool,
    ) -> None:
        self.llm_client = llm_client
        self.judge_client = judge_client
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

        observations = await self._generate_observations(
            job_id=job_id,
            transcript_text=transcript_text,
            frame_artifact=frame_artifact,
        )
        self._persist_json("observations.json", observations)
        frame_paths = [frame.local_path for frame in frame_artifact.frames]

        async def generate_for_style(style_name: StyleName) -> tuple[StyleName, CaptionVariant, JudgeResult]:
            vision_output = json.dumps(observations, indent=2, ensure_ascii=False)
            prompt = render_prompt(
                STYLE_PROMPTS[style_name],
                replacements={"VISION_OUTPUT": vision_output},
            )
            temperature = settings.fireworks_creative_temperature
            prior_candidates: dict[str, str] | None = None
            prior_judge_result: JudgeResult | None = None
            final_candidates: dict[str, str] | None = None
            final_judge_result: JudgeResult | None = None

            for attempt_index in range(1, 4):
                prompt_text = _build_caption_candidates_user_prompt(
                    style_name=style_name,
                    prior_candidates=prior_candidates,
                    prior_judge_result=prior_judge_result,
                )
                payload = await self.llm_client.generate_json(
                    prompt=prompt_text,
                    system_prompt=prompt,
                    temperature=temperature,
                    response_schema=build_caption_candidates_json_schema(),
                    response_schema_name=f"{style_name}_candidates_attempt_{attempt_index}",
                    endpoint_path=CAPTION_ENDPOINT,
                )
                candidates = CaptionCandidates.model_validate(payload)
                candidate_payload = {
                    "candidate_1": _clean_caption(candidates.candidate_1),
                    "candidate_2": _clean_caption(candidates.candidate_2),
                }
                policy_violations = _candidate_policy_violations(
                    style_name=style_name,
                    candidates=candidate_payload,
                    observations=observations,
                )
                if all(policy_violations.values()):
                    retry_payload = await self.llm_client.generate_json(
                        prompt=(
                            prompt_text
                            + "\n\nBoth candidates violate caption policy:\n"
                            + json.dumps(policy_violations, indent=2)
                            + "\nKeep the facts unchanged and return two corrected candidates in the same JSON format."
                        ),
                        system_prompt=prompt,
                        temperature=temperature,
                        response_schema=build_caption_candidates_json_schema(),
                        response_schema_name=f"{style_name}_candidates_style_retry_{attempt_index}",
                        endpoint_path=CAPTION_ENDPOINT,
                    )
                    retried = CaptionCandidates.model_validate(retry_payload)
                    candidate_payload = {
                        "candidate_1": _clean_caption(retried.candidate_1),
                        "candidate_2": _clean_caption(retried.candidate_2),
                    }
                    policy_violations = _candidate_policy_violations(
                        style_name=style_name,
                        candidates=candidate_payload,
                        observations=observations,
                    )

                judge_result = await self._judge_style(
                    style_name=style_name,
                    observations=observations,
                    frame_paths=frame_paths,
                    candidates=candidate_payload,
                    policy_violations=policy_violations,
                )
                final_candidates = candidate_payload
                final_judge_result = judge_result
                selected_evaluation = _selected_evaluation(judge_result)
                if _judge_passes(selected_evaluation):
                    break
                prior_candidates = candidate_payload
                prior_judge_result = judge_result

            assert final_candidates is not None
            assert final_judge_result is not None
            selected_caption = final_candidates[final_judge_result.selected_candidate]
            return style_name, CaptionVariant(style_name=style_name, caption=selected_caption), final_judge_result

        generated = await asyncio.gather(*(generate_for_style(style_name) for style_name in requested_styles))
        captions = {style_name: variant for style_name, variant, _judge in generated}
        checks = {style_name: judge for style_name, _variant, judge in generated}
        self._persist_json("checks.json", {style_name: check.model_dump(mode="json") for style_name, check in checks.items()})

        self._persist_json(
            "final_result.json",
            {
                "job_id": job_id,
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

    async def _judge_style(
        self,
        *,
        style_name: StyleName,
        observations: dict[str, Any],
        frame_paths: list[str],
        candidates: dict[str, str],
        policy_violations: dict[str, list[str]],
    ) -> JudgeResult:
        payload = await self.judge_client.generate_json(
            system_prompt=load_prompt("judge.txt"),
            prompt=json.dumps(
                {
                    "target_style": style_name,
                    "style_checklist": _style_checklist_for(style_name),
                    "focus": [
                        "Accurate captions.",
                        "Correct requested style.",
                        "Specific video details.",
                        "No major hallucinations.",
                        "Complete outputs for all clips/styles.",
                    ],
                    "observations": observations,
                    "candidates": candidates,
                    "deterministic_policy_violations": policy_violations,
                },
                indent=2,
            ),
            image_paths=frame_paths,
            temperature=_judge_temperature(self.judge_client.config.provider_name, self.judge_client.config.model),
            response_schema=build_judge_json_schema(),
            response_schema_name=f"{style_name}_judge",
            endpoint_path=JUDGE_ENDPOINT,
        )
        result = JudgeResult.model_validate(payload)
        return _calibrate_judge_result(result, policy_violations)

    async def _generate_observations(
        self,
        *,
        job_id: str,
        transcript_text: str,
        frame_artifact: FrameExtractionArtifact,
    ) -> dict[str, Any]:
        base_prompt = _build_observation_user_prompt(
            job_id=job_id,
            transcript_text=transcript_text,
            frame_artifact=frame_artifact,
        )
        image_paths = [frame.local_path for frame in frame_artifact.frames]
        last_error: Exception | None = None
        for attempt_index in range(1, 4):
            prompt = base_prompt
            temperature = settings.fireworks_temperature
            if attempt_index > 1:
                prompt += (
                    "\n\nSTRICT OUTPUT REQUIREMENTS:\n"
                    "- Return exactly one valid JSON object.\n"
                    "- Do not include markdown fences.\n"
                    "- Do not include trailing commentary.\n"
                    "- Every required property must be present.\n"
                    "- Ensure commas and brackets are valid JSON syntax.\n"
                )
                temperature = 0.0
            try:
                return await self.llm_client.generate_json(
                    system_prompt=load_prompt("perception_system.txt"),
                    prompt=prompt,
                    image_paths=image_paths,
                    temperature=temperature,
                    response_schema=build_observation_json_schema(),
                    response_schema_name=f"observations_attempt_{attempt_index}",
                    endpoint_path=VISION_ENDPOINT,
                )
            except Exception as exc:
                last_error = exc
                logger.warning("Observation generation attempt %s failed for %s: %s", attempt_index, job_id, exc)
        if last_error is not None:
            raise last_error
        raise RuntimeError("Observation generation failed without an explicit error.")

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


def _build_observation_user_prompt(*, job_id: str, transcript_text: str, frame_artifact: FrameExtractionArtifact) -> str:
    frame_lines = "\n".join(
        f"- {frame.frame_id}: {format_timestamp_label(frame.timestamp)}"
        for frame in frame_artifact.frames
    )
    return (
        f"Video id: {job_id}\n"
        "The attached images are video frames in chronological order.\n"
        "Treat them as one video sequence. Use the frame labels and timestamps below to track time progression.\n\n"
        "Frame order:\n"
        f"{frame_lines}\n\n"
        f"Optional transcript:\n{transcript_text or '[none provided]'}\n"
    )


def format_timestamp_label(timestamp: float) -> str:
    total_seconds = max(0.0, float(timestamp))
    minutes = int(total_seconds // 60)
    seconds = total_seconds - (minutes * 60)
    return f"{minutes:02d}:{seconds:05.2f}"


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
    text = re.sub(r"^\s*:\s*", "", text)
    text = re.sub(r"^\s*(?:[A-Za-z][\w-]*\s*){1,3}:\s+(?=[A-Z])", "", text)
    return text


def _needs_style_retry(style_name: StyleName, caption: str) -> bool:
    normalized = caption.lower()
    if style_name == "humorous_tech":
        return not any(word in normalized for word in TECH_STYLE_WORDS)
    if style_name == "sarcastic":
        return not any(marker in normalized for marker in SARCASM_STYLE_MARKERS)
    return False


def _candidate_policy_violations(
    *,
    style_name: StyleName,
    candidates: dict[str, str],
    observations: dict[str, Any],
) -> dict[str, list[str]]:
    evidence = json.dumps(observations, ensure_ascii=False).lower()
    minimum_words, maximum_words = STYLE_WORD_LIMITS[style_name]
    violations: dict[str, list[str]] = {}
    for candidate_name, caption in candidates.items():
        issues: list[str] = []
        words = re.findall(r"\b[\w'-]+\b", caption)
        if len(words) < minimum_words or len(words) > maximum_words:
            issues.append(f"word count {len(words)} is outside {minimum_words}-{maximum_words}")
        if style_name == "humorous_tech" and _needs_style_retry(style_name, caption):
            issues.append("no clear computing analogy")
        if style_name == "humorous_non_tech" and any(_contains_word(caption, word) for word in TECH_STYLE_WORDS):
            issues.append("contains technical jargon")
        if re.match(r"^\s*[:;,-]", caption):
            issues.append("starts with stray punctuation")
        unsupported_props = [
            term
            for term in UNSUPPORTED_PROP_TERMS
            if _contains_word(caption, term) and term not in evidence
        ]
        if unsupported_props:
            issues.append("unsupported joke prop or event: " + ", ".join(unsupported_props))
        unsupported_history = [
            phrase
            for phrase in UNSUPPORTED_HISTORY_PHRASES
            if phrase in caption.lower() and phrase not in evidence
        ]
        if unsupported_history:
            issues.append("unsupported prior history: " + ", ".join(unsupported_history))
        unsupported_numbers = [token for token in NUMBER_TOKEN.findall(caption) if token.lower() not in evidence]
        if unsupported_numbers:
            issues.append("unsupported numeric precision: " + ", ".join(unsupported_numbers))
        violations[candidate_name] = issues
    return violations


def _contains_word(text: str, word: str) -> bool:
    return re.search(rf"\b{re.escape(word)}\w*\b", text, re.IGNORECASE) is not None


def _calibrate_judge_result(
    result: JudgeResult,
    policy_violations: dict[str, list[str]],
) -> JudgeResult:
    calibrated: dict[str, CandidateJudgeEvaluation] = {}
    for candidate_name in ("candidate_1", "candidate_2"):
        evaluation = getattr(result, candidate_name)
        score = min(evaluation.accuracy_score, evaluation.style_score)
        if evaluation.accuracy.lower() != "pass" or evaluation.style.lower() != "pass":
            score = min(score, 0.69)
        score = max(0.0, score - (0.08 * len(policy_violations.get(candidate_name, []))))
        calibrated[candidate_name] = evaluation.model_copy(update={"combined_score": round(score, 4)})
    selected_candidate = max(calibrated, key=lambda name: calibrated[name].combined_score)
    return result.model_copy(
        update={
            "selected_candidate": selected_candidate,
            "candidate_1": calibrated["candidate_1"],
            "candidate_2": calibrated["candidate_2"],
        }
    )


def _build_caption_candidates_user_prompt(
    style_name: StyleName,
    *,
    prior_candidates: dict[str, str] | None = None,
    prior_judge_result: JudgeResult | None = None,
) -> str:
    base_prompt = (
        f"Generate two distinct {style_name} caption candidates for this video.\n"
        "Return strict JSON only in this shape:\n"
        '{\n  "candidate_1": "caption text",\n  "candidate_2": "caption text"\n}\n'
        "Candidate 1 must be conservative and closely match the retired reference style.\n"
        "Candidate 2 may be more creative, but must remain equally grounded and concise.\n"
        "Name the visible scene before adding style. Use only facts in caption_facts or clearly repeated high-confidence observations.\n"
        "Do not invent exact counts, seconds, percentages, signs, brands, locations, intentions, or technical events.\n"
        "A joke may add metaphorical attitude, but never a new physical object, destination, prior event, or unseen action.\n"
        "Every concrete noun and physical action in either caption must be supported by the observations.\n"
        "Avoid making camera movement, filming, or model behavior the main joke.\n"
        "Both candidates must be accurate, complete, and stylistically valid.\n"
        "Caption values must start with words, not labels or leading punctuation.\n"
        "Do not return explanations."
    )
    if not prior_candidates or not prior_judge_result:
        return base_prompt
    selected_eval = _selected_evaluation(prior_judge_result)
    return (
        base_prompt
        + "\n\nPrevious candidates:\n"
        + json.dumps(prior_candidates, indent=2, ensure_ascii=False)
        + "\n\nJudge feedback:\n"
        + json.dumps(prior_judge_result.model_dump(mode="json"), indent=2, ensure_ascii=False)
        + f"\n\nThe selected candidate scored below {settings.caption_acceptance_threshold:.2f} calibrated quality. Generate two new candidates that fix every accuracy and style issue."
        + "\nUse the previous candidates only as negative or partial references. Do not repeat them."
        + "\nRemove unsupported details instead of paraphrasing them. Prefer a simpler caption when uncertain."
        + f"\nTarget both accuracy and style scores of at least {settings.caption_acceptance_threshold:.2f}. The last winning candidate scored {selected_eval.combined_score:.2f}."
    )


def _style_checklist_for(style_name: StyleName) -> str:
    if style_name == "formal":
        return "Formal: clear and professional."
    if style_name == "sarcastic":
        return "Sarcastic: sarcastic but still accurate."
    if style_name == "humorous_tech":
        return "Humorous tech: tech humor plus real video details."
    return "humorous_non_tech: funny, everyday humour with no technical jargon."


def _selected_evaluation(judge_result: JudgeResult):
    return getattr(judge_result, judge_result.selected_candidate)


def _judge_passes(evaluation: CandidateJudgeEvaluation) -> bool:
    return (
        evaluation.accuracy.lower() == "pass"
        and evaluation.style.lower() == "pass"
        and evaluation.accuracy_score >= settings.caption_acceptance_threshold
        and evaluation.style_score >= settings.caption_acceptance_threshold
        and evaluation.combined_score >= settings.caption_acceptance_threshold
    )


def _judge_temperature(provider_name: str, model_name: str) -> float | None:
    normalized_provider = (provider_name or "").strip().lower()
    if normalized_provider == "openrouter":
        return None
    return settings.fireworks_temperature
