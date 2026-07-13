from __future__ import annotations

import random
import re
from pathlib import Path


PROMPT_DIR = Path(__file__).resolve().parent
SHUFFLE_BLOCK_PATTERN = re.compile(r"\{\{SHUFFLE_BLOCK(?::[^}]*)?\}\}(.*?)\{\{END_SHUFFLE_BLOCK\}\}", re.DOTALL)


def load_prompt(name: str) -> str:
    return (PROMPT_DIR / name).read_text(encoding="utf-8").strip()


def render_prompt(name: str, *, replacements: dict[str, str] | None = None) -> str:
    prompt = load_prompt(name)
    if replacements:
        for key, value in replacements.items():
            prompt = prompt.replace(f"{{{{{key}}}}}", value)
    return _render_shuffle_blocks(prompt).strip()


def _render_shuffle_blocks(prompt: str) -> str:
    rng = random.SystemRandom()

    def replace_block(match: re.Match[str]) -> str:
        body = match.group(1)
        items = [line.strip() for line in body.splitlines() if line.strip()]
        rng.shuffle(items)
        return "\n".join(items)

    return SHUFFLE_BLOCK_PATTERN.sub(replace_block, prompt)
