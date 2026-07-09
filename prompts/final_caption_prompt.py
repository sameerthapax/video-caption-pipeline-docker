from __future__ import annotations

import json

from schemas.vlm import GlobalFactualSummary


STYLE_INSTRUCTIONS = {
    "formal": "Professional, objective, factual tone with neutral wording and no jokes.",
    "sarcastic": "Dry, clearly ironic, lightly mocking tone with sharper phrasing and understated ridicule, without changing, shrinking, or inventing facts.",
    "humorous_tech": "Funny tone with clearly recognizable technology, software, automation, or programming jokes woven into the narration, while keeping all facts accurate and equally detailed.",
    "humorous_non_tech": "Funny everyday tone with playful non-technical comparisons and no programming jargon, while keeping all facts accurate and equally detailed.",
}

STYLE_ORDER = ("formal", "sarcastic", "humorous_tech", "humorous_non_tech")


def build_final_caption_prompt(*, job_id: str, global_summary: GlobalFactualSummary) -> str:
    ground_truth_json = json.dumps(global_summary.model_dump(mode="json"), indent=2)
    style_rules = "\n".join(f"- {style_name}: {STYLE_INSTRUCTIONS[style_name]}" for style_name in STYLE_ORDER)

    return f"""
You are generating the complete final caption set for video job {job_id}.

Goal:
- Produce four final captions from one shared ground-truth source.
- Keep the factual backbone aligned across every style.
- Preserve subject appearance, clothing, objects, scene changes, and timeline coverage in every variant.

Required output behavior:
- Generate one caption each for: formal, sarcastic, humorous_tech, humorous_non_tech.
- All four captions must describe the same key moments in the same order.
- Do not let the funny styles become shorter in substance than the formal caption.
- Make the tones genuinely distinct from one another. They should not read like the same caption with a few swapped adjectives.
- Keep the same core facts in every style:
  - who appears
  - visible appearance details such as hair, beard, glasses, clothing, and accessories when grounded
  - what the person is doing
  - the major objects, interfaces, websites, and settings
  - the scene transitions and final destination
- If the ground truth includes a more specific person description than "a man" or "a woman", use that richer grounded description in every style.
- If different segments may show different people, describe each segment directly instead of forcing identity continuity.
- Do not say "the same man," "the same person," or similar continuity claims unless the ground truth strongly supports that with matching appearance, clothing, and scene continuity.
- Tone may change, but factual coverage should remain comparable across all four outputs.
- Do not invent exact addresses, motives, emotions, dialogue, or other unsupported details.
- If a detail is uncertain in the ground truth, keep the uncertainty or omit the detail.
- Do not write analysis, explanations, or multiple options.

Style requirements:
{style_rules}

Grounding rules:
- Treat the provided global factual summary as the only ground truth source.
- Preserve factual meaning. Only tone, phrasing, and joke framing may change.
- Favor concrete grounded nouns over vague nouns. Example: prefer "a man with dark hair, a beard, and glasses wearing a black jacket over a white shirt" over "a man" when the ground truth supports it.
- For continuity across cuts, be conservative. If identity is not certain, say "a man appears" in that segment rather than claiming it is the same person from earlier.
- Cover the full video arc, not just the first or funniest moment.
- Keep each caption to one paragraph.

Tone separation rules:
- `formal` should read like a polished neutral description.
- `sarcastic` should sound noticeably more cutting and deadpan than the other styles.
- `humorous_tech` should include distinctly tech-flavored metaphors or jokes, such as UI, automation, modal, script, system, browser, or workflow framing when grounded and natural.
- `humorous_non_tech` should avoid tech-speak and instead use everyday funny phrasing, social comparisons, or errand-style humor.
- Do not reuse the same joke structure across multiple styles.
- Keep the facts aligned, but vary rhythm, framing, and comic angle so each style feels intentionally different.

Return JSON only with this exact shape:
{{
  "captions": {{
    "formal": {{
      "style_name": "formal",
      "caption": "single final caption",
      "grounded_facts_used": ["short fact 1", "short fact 2"],
      "safety_notes": ["short note about uncertainty or omitted unsupported details if relevant"]
    }},
    "sarcastic": {{
      "style_name": "sarcastic",
      "caption": "single final caption",
      "grounded_facts_used": ["short fact 1", "short fact 2"],
      "safety_notes": ["short note about uncertainty or omitted unsupported details if relevant"]
    }},
    "humorous_tech": {{
      "style_name": "humorous_tech",
      "caption": "single final caption",
      "grounded_facts_used": ["short fact 1", "short fact 2"],
      "safety_notes": ["short note about uncertainty or omitted unsupported details if relevant"]
    }},
    "humorous_non_tech": {{
      "style_name": "humorous_non_tech",
      "caption": "single final caption",
      "grounded_facts_used": ["short fact 1", "short fact 2"],
      "safety_notes": ["short note about uncertainty or omitted unsupported details if relevant"]
    }}
  }}
}}

Ground truth JSON:
{ground_truth_json}
""".strip()
