from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Protocol

from prompts.prompt_loader import load_prompt
from schemas.vlm import FinalCaptionResult, GlobalFactualSummary, StyledCaptionVariant
from worker.config.settings import settings


STYLE_ORDER = ("formal", "sarcastic", "humorous_tech", "humorous_non_tech")
STYLE_PROMPTS = {
    "formal": "style_formal.txt",
    "sarcastic": "style_sarcastic.txt",
    "humorous_tech": "style_humorous_tech.txt",
    "humorous_non_tech": "style_humorous_non_tech.txt",
}


class CaptionGenerationClient(Protocol):
    async def generate_json(self, *, model: str, prompt: str) -> dict:
        ...


async def style_captions(
    *,
    client: CaptionGenerationClient,
    model: str,
    job_id: str,
    global_summary: GlobalFactualSummary,
    artifact_root: Path | None = None,
    persist_artifacts: bool = False,
) -> tuple[FinalCaptionResult, str]:
    if not _summary_has_content(global_summary):
        raise ValueError(f"No grounded summary content available for job {job_id}.")
    persisted_global_summary_path = str(artifact_root / "global_factual_summary.json") if artifact_root and persist_artifacts else ""
    caption_map = await _generate_caption_variants(
        client=client,
        model=model,
        job_id=job_id,
        global_summary=global_summary,
    )

    final_result = FinalCaptionResult(
        job_id=job_id,
        neutral_summary=global_summary.factual_summary,
        formal_caption=caption_map["formal"].caption,
        sarcastic_caption=caption_map["sarcastic"].caption,
        humorous_tech_caption=caption_map["humorous_tech"].caption,
        humorous_non_tech_caption=caption_map["humorous_non_tech"].caption,
        source_global_factual_summary_path=persisted_global_summary_path,
        captions=caption_map,
    )

    final_result_path = ""
    if artifact_root is not None and persist_artifacts:
        final_result_path = str(artifact_root / "final_result.json")
        Path(final_result_path).write_text(final_result.model_dump_json(indent=2), encoding="utf-8")
    return final_result, final_result_path


async def _generate_caption_variants(
    *,
    client: CaptionGenerationClient,
    model: str,
    job_id: str,
    global_summary: GlobalFactualSummary,
) -> dict[str, StyledCaptionVariant]:
    tasks = [
        _generate_single_style(
            client=client,
            model=model,
            job_id=job_id,
            style_name=style_name,
            global_summary=global_summary,
        )
        for style_name in STYLE_ORDER
    ]
    variants = await asyncio.gather(*tasks)
    return {variant.style_name: variant for variant in variants}


async def _generate_single_style(
    *,
    client: CaptionGenerationClient,
    model: str,
    job_id: str,
    style_name: str,
    global_summary: GlobalFactualSummary,
) -> StyledCaptionVariant:
    prompt = _build_style_prompt(job_id=job_id, style_name=style_name, global_summary=global_summary)
    payload = await client.generate_json(model=model, prompt=prompt)
    variant = StyledCaptionVariant.model_validate(payload)
    if variant.style_name != style_name:
        raise ValueError(f"Final caption variant style mismatch for '{style_name}'.")
    return variant


def _build_style_prompt(*, job_id: str, style_name: str, global_summary: GlobalFactualSummary) -> str:
    template = (
        load_prompt(STYLE_PROMPTS[style_name])
        .replace("{caption_min_words}", str(settings.caption_min_words))
        .replace("{caption_max_words}", str(settings.caption_max_words))
    )
    payload = {
        "job_id": job_id,
        "factual_summary": global_summary.factual_summary,
        "detailed_ground_truth": global_summary.detailed_ground_truth,
        "setting_summary": global_summary.setting_summary,
        "main_subjects": global_summary.main_subjects,
        "main_objects": global_summary.main_objects,
        "audio_summary": global_summary.audio_summary,
        "detailed_timeline": [item.model_dump() for item in global_summary.detailed_timeline[:3]],
        "scene_change_overview": [item.model_dump() for item in global_summary.scene_change_overview[:3]],
        "uncertainties": global_summary.uncertainties,
    }
    return f"{template}\n\nGrounded video facts:\n{json.dumps(payload, indent=2)}"


def _summary_has_content(global_summary: GlobalFactualSummary) -> bool:
    return any(
        [
            global_summary.factual_summary.strip(),
            global_summary.detailed_ground_truth.strip(),
            global_summary.setting_summary.strip(),
            global_summary.audio_summary.strip(),
            global_summary.main_subjects,
            global_summary.main_objects,
            global_summary.detailed_timeline,
        ]
    )
