from __future__ import annotations

from pathlib import Path
from typing import Protocol

from prompts.final_caption_prompt import build_final_caption_prompt
from schemas.vlm import FinalCaptionResult, GlobalFactualSummary, StyledCaptionVariant


STYLE_ORDER = ("formal", "sarcastic", "humorous_tech", "humorous_non_tech")


class CaptionGenerationClient(Protocol):
    async def generate_json(self, *, model: str, prompt: str) -> dict:
        ...


async def style_captions(
    *,
    client: CaptionGenerationClient,
    model: str,
    job_id: str,
    global_summary_path: Path,
) -> tuple[FinalCaptionResult, str]:
    global_summary = GlobalFactualSummary.model_validate_json(global_summary_path.read_text(encoding="utf-8"))

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
        source_global_factual_summary_path=str(global_summary_path),
        captions=caption_map,
    )

    final_result_path = global_summary_path.parent / "final_result.json"
    final_result_path.write_text(final_result.model_dump_json(indent=2), encoding="utf-8")
    return final_result, str(final_result_path)


async def _generate_caption_variants(
    *,
    client: CaptionGenerationClient,
    model: str,
    job_id: str,
    global_summary: GlobalFactualSummary,
) -> dict[str, StyledCaptionVariant]:
    prompt = build_final_caption_prompt(
        job_id=job_id,
        global_summary=global_summary,
    )
    payload = await client.generate_json(model=model, prompt=prompt)
    captions_payload = payload.get("captions")
    if not isinstance(captions_payload, dict):
        raise ValueError("Final caption response missing 'captions' object.")

    caption_map: dict[str, StyledCaptionVariant] = {}
    for style_name in STYLE_ORDER:
        if style_name not in captions_payload:
            raise ValueError(f"Final caption response missing '{style_name}' variant.")
        variant = StyledCaptionVariant.model_validate(captions_payload[style_name])
        if variant.style_name != style_name:
            raise ValueError(f"Final caption variant style mismatch for '{style_name}'.")
        caption_map[style_name] = variant
    return caption_map
