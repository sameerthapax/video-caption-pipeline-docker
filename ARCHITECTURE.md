# Architecture

The repo now uses a clip-level caption pipeline rather than the older segment-based reasoning stack.

Primary runtime:

- [main.py](/Users/sams/Desktop/video-caption-pipeline-docker/main.py:1)

Primary orchestrator:

- [pipeline/caption_pipeline.py](/Users/sams/Desktop/video-caption-pipeline-docker/pipeline/caption_pipeline.py:1)

## Execution Flow

```text
input/tasks.json
  -> main.py
    -> download video
    -> probe video
    -> normalize video
    -> extract audio
    -> optional local Whisper transcript
    -> extract frames
    -> run caption mode
    -> output/results.json
```

## Caption Modes

### Verified Scene

```text
frames
  -> describe scene
  -> verify description
  -> OpenAI writes captions one style at a time
  -> optional judge checks
```

### Direct Vision

```text
frames
  -> vision model writes all requested styles in one JSON object
  -> optional judge checks
```

### Observation First

```text
frames
  -> structured observations
  -> combined captions or per-style captions
  -> optional judge checks
```

## Frame Extraction

Frame extraction is handled by [pipeline/extract_frames.py](/Users/sams/Desktop/video-caption-pipeline-docker/pipeline/extract_frames.py:1).

It uses:

- dynamic frame budgets
- planned timestamps from anchor, safety, and scene-change moments
- optional OpenCV extraction
- optional scene-midpoint extraction
- scene-frame fallback
- uniform-frame fallback
- perceptual deduplication

Scene scoring remains in:

- [pipeline/scene_change.py](/Users/sams/Desktop/video-caption-pipeline-docker/pipeline/scene_change.py:1)

## Providers

Vision JSON and text generation:

- [services/vision_llm_client.py](/Users/sams/Desktop/video-caption-pipeline-docker/services/vision_llm_client.py:1)

OpenAI final caption and judge generation:

- [services/openai_responses_client.py](/Users/sams/Desktop/video-caption-pipeline-docker/services/openai_responses_client.py:1)

## Schemas

Simplified clip-level schemas:

- [schemas/caption.py](/Users/sams/Desktop/video-caption-pipeline-docker/schemas/caption.py:1)
- [schemas/frames.py](/Users/sams/Desktop/video-caption-pipeline-docker/schemas/frames.py:1)
- [schemas/tasks.py](/Users/sams/Desktop/video-caption-pipeline-docker/schemas/tasks.py:1)
- [schemas/video.py](/Users/sams/Desktop/video-caption-pipeline-docker/schemas/video.py:1)
