# Video Caption Pipeline

A containerized video understanding and caption generation pipeline for short-form clips.

The system downloads each input video, normalizes it, samples representative frames, optionally transcribes audio, performs segment-level visual reasoning, builds a global factual summary, and finally generates caption variants in multiple tones.

It is designed to be:

- evidence-first
- debuggable end to end
- resilient to partial model failures
- easy to run in a single Docker command

## What It Produces

For every task in `/input/tasks.json`, the pipeline returns a caption bundle in `/output/results.json`.

Each caption is grounded in a two-layer reasoning flow:

- Segment VLM analysis: frame-based reasoning for each temporal segment
- Global factual summary: a final image-backed fact synthesis across the whole clip

The final captions are generated from that factual summary, not directly from the raw video.

## Pipeline Overview

The runtime entrypoint is [main.py](/Users/sams/Desktop/video-caption-pipeline-docker/main.py:1).

High-level flow:

1. Download the source video
2. Probe the original media
3. Preprocess in parallel
4. Extract representative frames
5. Build temporal segments
6. Run transcript and visual reasoning branches in parallel
7. Merge segment evidence into video memory
8. Generate a global factual summary with frames
9. Generate final caption variants
10. Write outputs and debug artifacts

## Stage Breakdown

### 1. Preprocessing

The source clip is normalized into a stable analysis format:

- H.264 MP4
- capped at `1280px` width
- forced to `30fps`

Audio extraction runs concurrently with normalization when audio is present.

Relevant code:

- [main.py](/Users/sams/Desktop/video-caption-pipeline-docker/main.py:92)
- [pipeline/normalize.py](/Users/sams/Desktop/video-caption-pipeline-docker/pipeline/normalize.py:1)
- [pipeline/probe_video.py](/Users/sams/Desktop/video-caption-pipeline-docker/pipeline/probe_video.py:1)

### 2. Frame Sampling

The pipeline does not extract every frame.

Instead, it builds a compact evidence set from:

- uniform timestamps across the video
- safety timestamps
- scene-change candidates

Those timestamps are deduplicated and then extracted as JPEG frames.

Relevant code:

- [pipeline/scene_change.py](/Users/sams/Desktop/video-caption-pipeline-docker/pipeline/scene_change.py:1)
- [pipeline/frame_sampling.py](/Users/sams/Desktop/video-caption-pipeline-docker/pipeline/frame_sampling.py:1)
- [pipeline/extract_frames.py](/Users/sams/Desktop/video-caption-pipeline-docker/pipeline/extract_frames.py:1)

### 3. Temporal Segmentation

The video is split into `5` coarse temporal segments. Each selected frame is assigned to one of those segments.

Relevant code:

- [pipeline/temporal_segments.py](/Users/sams/Desktop/video-caption-pipeline-docker/pipeline/temporal_segments.py:1)

### 4. Transcript Branch

If audio exists and a Gemini key is available, the normalized audio is segmented into transcript windows and transcribed.

If there is no audio track, or transcription is disabled, the pipeline continues without failing the task.

Relevant code:

- [pipeline/audio_windows.py](/Users/sams/Desktop/video-caption-pipeline-docker/pipeline/audio_windows.py:1)
- [services/google_gemini_client.py](/Users/sams/Desktop/video-caption-pipeline-docker/services/google_gemini_client.py:1)
- [pipeline/run_vlm_stage.py](/Users/sams/Desktop/video-caption-pipeline-docker/pipeline/run_vlm_stage.py:19)

### 5. Segment VLM Reasoning

Each temporal segment is analyzed with:

- selected frames from that segment
- accumulated video memory from earlier segments
- transcript chunks for the segment when available

This stage produces structured segment-level visual reasoning, continuity updates, and memory updates for downstream summarization.

Relevant code:

- [pipeline/vlm_reasoning.py](/Users/sams/Desktop/video-caption-pipeline-docker/pipeline/vlm_reasoning.py:1)
- [prompts/segment_vlm_prompt.py](/Users/sams/Desktop/video-caption-pipeline-docker/prompts/segment_vlm_prompt.py:1)
- [schemas/vlm.py](/Users/sams/Desktop/video-caption-pipeline-docker/schemas/vlm.py:1)

### 6. Deterministic Segment Fusion

Per-segment fusion is no longer a separate model call.

The pipeline deterministically combines:

- transcript evidence
- structured segment VLM output
- continuity fields

into `segment_ground_truth` for each segment.

Relevant code:

- [pipeline/global_summary.py](/Users/sams/Desktop/video-caption-pipeline-docker/pipeline/global_summary.py:55)

### 7. Global Factual Summary

The global factual summary is the final fact-generation step.

It uses:

- accumulated video memory
- segment reasoning outputs
- fused temporal segment ground truth
- transcript chunks
- segment frame metadata
- the actual extracted frames

This is intentionally still image-backed.

Relevant code:

- [pipeline/global_summary.py](/Users/sams/Desktop/video-caption-pipeline-docker/pipeline/global_summary.py:13)
- [prompts/global_summary_prompt.py](/Users/sams/Desktop/video-caption-pipeline-docker/prompts/global_summary_prompt.py:8)

### 8. Caption Generation

The factual summary is passed to the final caption generator, which creates:

- `formal`
- `sarcastic`
- `humorous_tech`
- `humorous_non_tech`

Relevant code:

- [pipeline/style.py](/Users/sams/Desktop/video-caption-pipeline-docker/pipeline/style.py:1)
- [prompts/final_caption_prompt.py](/Users/sams/Desktop/video-caption-pipeline-docker/prompts/final_caption_prompt.py:1)
- [services/openai_responses_client.py](/Users/sams/Desktop/video-caption-pipeline-docker/services/openai_responses_client.py:1)

## Concurrency Model

The pipeline uses concurrency in a few important places:

- tasks can run concurrently across a batch
- normalization and audio extraction run in parallel per task
- transcript generation and visual reasoning run in parallel
- Gemini requests are concurrency-limited
- Fireworks requests are concurrency-limited
- ffmpeg subprocesses are globally bounded

Key settings live in [worker/config/settings.py](/Users/sams/Desktop/video-caption-pipeline-docker/worker/config/settings.py:1).

## Run

Build:

```bash
docker build -t video-caption-hackathon .
```

Run:

```bash
docker run --rm \
  -v "$(pwd)/input:/input" \
  -v "$(pwd)/output:/output" \
  video-caption-hackathon
```

No `--env-file` flag is required because the image can embed `.env.hackathon` at build time. Runtime environment variables still override embedded values.

## Input Format

The container expects `/input/tasks.json`:

```json
[
  {
    "task_id": "v1",
    "video_url": "https://storage.googleapis.com/amd-hackathon-clips/1860079-uhd_2560_1440_25fps.mp4",
    "styles": ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]
  }
]
```

Relevant schema:

- [schemas/tasks.py](/Users/sams/Desktop/video-caption-pipeline-docker/schemas/tasks.py:1)

## Output Format

The container writes `/output/results.json`:

```json
[
  {
    "task_id": "v1",
    "captions": {
      "formal": "...",
      "sarcastic": "...",
      "humorous_tech": "...",
      "humorous_non_tech": "..."
    }
  }
]
```

## Debug Artifacts

When `DEBUG_KEEP_TEMP=True`, the pipeline persists intermediate artifacts to `/output/debug`.

```text
debug/
  original/<task_id>/original.mp4
  normalized/<task_id>/normalized.mp4
  audio/<task_id>/normalized.wav
  frames/<task_id>/*.jpg
  json/<task_id>/
    frame_sampling.json
    temporal_segments.json
    transcript.json
    vlm_segments.json
    video_memory.json
    global_factual_summary.json
    final_result.json
  logs/pipeline.log
```

This makes it possible to inspect:

- which frames were chosen
- how the video was segmented
- what transcript windows were produced
- what each segment VLM call returned
- how the factual summary was built

When `DEBUG_KEEP_TEMP=False`, temporary task files are removed after completion.

## Configuration

All runtime configuration is loaded through [worker/config/settings.py](/Users/sams/Desktop/video-caption-pipeline-docker/worker/config/settings.py:1).

Load order:

1. runtime environment variables
2. embedded `/app/.env`

Common knobs:

- `DEBUG_KEEP_TEMP`
- `MAX_CONCURRENT_JOBS`
- `FFMPEG_MAX_CONCURRENCY`
- `FIREWORKS_MAX_CONCURRENCY`
- `GOOGLE_GEMINI_MAX_CONCURRENCY`
- `MAX_VIDEO_SIZE_MB`
- `MAX_VIDEO_DURATION_SECONDS`

## Reliability Notes

- If a segment VLM response is malformed, the pipeline attempts structured JSON repair before giving up.
- If a task still fails, the system returns conservative fallback captions instead of incomplete output.
- Transcript generation is optional; visual reasoning still runs without it.
- Global factual summary remains image-backed to preserve final evidence grounding.

## Repository Structure

```text
main.py
pipeline/
prompts/
schemas/
services/
tests/
worker/
```

Core directories:

- `pipeline/`: extraction, segmentation, reasoning, summarization
- `prompts/`: model prompts for segment analysis, factual summary, final captions
- `schemas/`: Pydantic contracts for all artifacts
- `services/`: API clients, downloads, subprocess helpers
- `worker/`: runtime settings, logging, debug artifact persistence

## Notes

- No secrets are hardcoded in Python source.
- The pipeline is built around structured artifacts, not opaque model text.
- The factual summary is the source of truth for final caption generation.
- The debug artifact trail is intended to make failures explainable, not mysterious.
