import asyncio
import json

from pipeline.style import style_captions
from schemas.vlm import GlobalFactualSummary


class FakeCaptionClient:
    def __init__(self):
        self.calls = 0

    async def generate_json(self, *, model: str, prompt: str):
        self.calls += 1
        if '"style_name": "formal"' in prompt:
            return {"style_name": "formal", "caption": "formal caption", "grounded_facts_used": ["fact 1"], "safety_notes": []}
        if '"style_name": "sarcastic"' in prompt:
            return {"style_name": "sarcastic", "caption": "sarcastic caption", "grounded_facts_used": ["fact 1"], "safety_notes": []}
        if '"style_name": "humorous_tech"' in prompt:
            return {"style_name": "humorous_tech", "caption": "humorous_tech caption", "grounded_facts_used": ["fact 1"], "safety_notes": []}
        return {"style_name": "humorous_non_tech", "caption": "humorous_non_tech caption", "grounded_facts_used": ["fact 1"], "safety_notes": []}


def test_style_captions_generates_all_variants_and_writes_json(tmp_path):
    summary = GlobalFactualSummary(
        factual_summary="A person opens a laptop and speaks to camera.",
        uncertainties=["Exact brand is unclear."],
    )

    client = FakeCaptionClient()
    final_result, local_path = asyncio.run(
        style_captions(
            client=client,
            model="test-model",
            job_id="job-1",
            global_summary=summary,
            artifact_root=tmp_path,
            persist_artifacts=True,
        )
    )

    assert final_result.formal_caption == "formal caption"
    assert final_result.sarcastic_caption == "sarcastic caption"
    assert final_result.humorous_tech_caption == "humorous_tech caption"
    assert final_result.humorous_non_tech_caption == "humorous_non_tech caption"
    assert json.loads(tmp_path.joinpath("final_result.json").read_text(encoding="utf-8"))["job_id"] == "job-1"
    assert local_path.endswith("final_result.json")
    assert client.calls == 4
