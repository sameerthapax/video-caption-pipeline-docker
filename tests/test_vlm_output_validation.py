import asyncio

from pipeline.run_vlm_stage import _process_segment
from pipeline.video_memory import create_video_memory
from schemas.segments import TemporalSegment


class FailingFireworksClient:
    async def analyze_segment_with_images(self, **_kwargs):
        raise RuntimeError("simulated segment failure")


def test_failed_segment_response_still_produces_valid_artifact_shape(monkeypatch):
    monkeypatch.setattr("pipeline.run_vlm_stage.settings.fireworks_model", "test-vlm")
    segment = TemporalSegment(segment_index=0, start=0.0, end=12.0, percent_range="0-20")
    response = asyncio.run(
        _process_segment(
            fireworks=FailingFireworksClient(),
            job_id="job-1",
            segment=segment,
            memory=create_video_memory(job_id="job-1"),
        )
    )

    assert response.status == "failed"
    assert response.segment_index == 0
    assert isinstance(response.errors, list)
    assert response.evidence_quality.limitations
