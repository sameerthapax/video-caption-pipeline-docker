# Architecture

This document describes the actual execution flow of the video caption pipeline, from task ingestion to final caption output.

## Top-Level Flow

```text
input/tasks.json
  -> main.py
    -> download video
    -> probe original media
    -> preprocess media
       -> normalize video
       -> extract audio
    -> extraction stage
       -> probe normalized video
       -> scan scene changes
       -> build frame sampling plan
       -> extract selected frames
       -> build temporal segments
    -> reasoning stage
       -> transcript branch
       -> visual branch
       -> deterministic segment fusion
       -> global factual summary
    -> caption generation
       -> formal
       -> sarcastic
       -> humorous_tech
       -> humorous_non_tech
    -> output/results.json
```

## Real Runtime Path

Primary entrypoint:

- [main.py](/Users/sams/Desktop/video-caption-pipeline-docker/main.py:1)

Main stage calls:

1. `_load_tasks()`
2. `_process_tasks()`
3. `_process_task()`
4. `run_video_extraction_stage()`
5. `run_vlm_reasoning_stage()`
6. `style_captions()`

## Task Lifecycle

### 1. Input Task

Each task contains:

- `task_id`
- `video_url`
- requested caption `styles`

Schema:

- [schemas/tasks.py](/Users/sams/Desktop/video-caption-pipeline-docker/schemas/tasks.py:1)

### 2. Download And Probe

The source video is downloaded locally and inspected before processing.

Code path:

- [services/local_files.py](/Users/sams/Desktop/video-caption-pipeline-docker/services/local_files.py:1)
- [pipeline/probe_video.py](/Users/sams/Desktop/video-caption-pipeline-docker/pipeline/probe_video.py:1)

Outputs:

- original local video file
- original media metadata

### 3. Preprocess Media

Two preprocessing steps run in parallel:

- video normalization
- audio extraction

Code path:

- [main.py](/Users/sams/Desktop/video-caption-pipeline-docker/main.py:118)
- [pipeline/normalize.py](/Users/sams/Desktop/video-caption-pipeline-docker/pipeline/normalize.py:1)

Outputs:

- `normalized.mp4`
- `normalized.wav` when audio exists

## Extraction Stage

Extraction is handled by:

- [pipeline/run_extraction_stage.py](/Users/sams/Desktop/video-caption-pipeline-docker/pipeline/run_extraction_stage.py:1)

### Scene Scan

The pipeline scans the normalized video at a coarse interval to score scene changes.

Code path:

- [pipeline/scene_change.py](/Users/sams/Desktop/video-caption-pipeline-docker/pipeline/scene_change.py:1)

This stage does not extract all frames. It only computes candidate timestamps and scene-change scores.

### Frame Sampling

A bounded evidence set is built from:

- uniform timestamps
- safety timestamps
- scene-change timestamps

Then timestamps are deduplicated and refilled from a replacement pool when possible.

Code path:

- [pipeline/frame_sampling.py](/Users/sams/Desktop/video-caption-pipeline-docker/pipeline/frame_sampling.py:1)

### Frame Extraction

Only the selected timestamps are extracted into JPEGs.

Code path:

- [pipeline/extract_frames.py](/Users/sams/Desktop/video-caption-pipeline-docker/pipeline/extract_frames.py:1)

### Temporal Segments

The video is divided into 5 temporal buckets, and extracted frames are assigned to those segments.

Code path:

- [pipeline/temporal_segments.py](/Users/sams/Desktop/video-caption-pipeline-docker/pipeline/temporal_segments.py:1)

Extraction artifacts:

- `frame_sampling.json`
- `temporal_segments.json`
- extracted frame files

## Reasoning Stage

Reasoning is handled by:

- [pipeline/run_vlm_stage.py](/Users/sams/Desktop/video-caption-pipeline-docker/pipeline/run_vlm_stage.py:1)

This stage has two branches that run in parallel.

```text
temporal_segments.json
  -> transcript branch
  -> visual branch
  -> deterministic fusion
  -> global factual summary
```

### Transcript Branch

If audio exists and Gemini is configured:

1. normalized audio is segmented into transcript windows
2. windows are transcribed concurrently
3. transcript chunks are assigned back to temporal segments

Code path:

- [pipeline/audio_windows.py](/Users/sams/Desktop/video-caption-pipeline-docker/pipeline/audio_windows.py:1)
- [services/google_gemini_client.py](/Users/sams/Desktop/video-caption-pipeline-docker/services/google_gemini_client.py:1)

Artifact:

- `transcription_request.json`

### Visual Branch

Each temporal segment is analyzed with Fireworks using:

- the selected frames for that segment
- previous video memory
- transcript context when available

Code path:

- [pipeline/vlm_reasoning.py](/Users/sams/Desktop/video-caption-pipeline-docker/pipeline/vlm_reasoning.py:1)
- [prompts/segment_vlm_prompt.py](/Users/sams/Desktop/video-caption-pipeline-docker/prompts/segment_vlm_prompt.py:1)
- [pipeline/video_memory.py](/Users/sams/Desktop/video-caption-pipeline-docker/pipeline/video_memory.py:1)

Artifact:

- `vlm_segments.json`

### Deterministic Segment Fusion

Per-segment fusion is not a separate model call.

It merges:

- transcript evidence
- structured segment VLM output
- continuity fields

into `segment_ground_truth`.

Code path:

- [pipeline/global_summary.py](/Users/sams/Desktop/video-caption-pipeline-docker/pipeline/global_summary.py:55)

Updated artifact:

- `temporal_segments.json`

### Global Factual Summary

This is the final fact-generation stage.

It uses:

- video memory
- fused segment truth
- segment reasoning outputs
- transcript chunks
- segment frame metadata
- actual extracted frames

Code path:

- [pipeline/global_summary.py](/Users/sams/Desktop/video-caption-pipeline-docker/pipeline/global_summary.py:13)
- [prompts/global_summary_prompt.py](/Users/sams/Desktop/video-caption-pipeline-docker/prompts/global_summary_prompt.py:8)

Artifact:

- `global_factual_summary.json`

## Caption Generation

Caption generation is handled by:

- [pipeline/style.py](/Users/sams/Desktop/video-caption-pipeline-docker/pipeline/style.py:1)
- [prompts/final_caption_prompt.py](/Users/sams/Desktop/video-caption-pipeline-docker/prompts/final_caption_prompt.py:1)
- [services/openai_responses_client.py](/Users/sams/Desktop/video-caption-pipeline-docker/services/openai_responses_client.py:1)

Input:

- `global_factual_summary.json`

Output:

- `final_result.json`
- final style captions returned in `results.json`

## Evidence Flow

```text
video
  -> normalized video
  -> sampled frames
  -> segment VLM outputs
  -> deterministic segment fusion
  -> global factual summary
  -> style captions
```

```text
audio
  -> normalized wav
  -> transcript windows
  -> transcript chunks
  -> segment transcript context
  -> global factual summary
  -> style captions
```

## Artifact Flow

```text
original.mp4
normalized.mp4
normalized.wav
frame_sampling.json
temporal_segments.json
transcription_request.json
vlm_segments.json
video_memory.json
global_factual_summary.json
final_result.json
results.json
```

## Concurrency

Concurrency exists at multiple levels.

### Task-Level

Multiple tasks can run in parallel.

Code path:

- [main.py](/Users/sams/Desktop/video-caption-pipeline-docker/main.py:69)

### Per-Task Preprocessing

These run concurrently:

- normalization
- audio extraction

Code path:

- [main.py](/Users/sams/Desktop/video-caption-pipeline-docker/main.py:118)

### Reasoning Branches

These run concurrently:

- transcript branch
- visual branch

Code path:

- [pipeline/run_vlm_stage.py](/Users/sams/Desktop/video-caption-pipeline-docker/pipeline/run_vlm_stage.py:101)

### Bounded External Work

The pipeline also limits:

- ffmpeg subprocess concurrency
- Fireworks request concurrency
- Gemini request concurrency

Code path:

- [services/process.py](/Users/sams/Desktop/video-caption-pipeline-docker/services/process.py:17)
- [services/async_limits.py](/Users/sams/Desktop/video-caption-pipeline-docker/services/async_limits.py:1)
- [worker/config/settings.py](/Users/sams/Desktop/video-caption-pipeline-docker/worker/config/settings.py:1)

## Schema Layers

The pipeline is driven by structured schemas.

Important schema files:

- [schemas/tasks.py](/Users/sams/Desktop/video-caption-pipeline-docker/schemas/tasks.py:1)
- [schemas/video.py](/Users/sams/Desktop/video-caption-pipeline-docker/schemas/video.py:1)
- [schemas/frames.py](/Users/sams/Desktop/video-caption-pipeline-docker/schemas/frames.py:1)
- [schemas/segments.py](/Users/sams/Desktop/video-caption-pipeline-docker/schemas/segments.py:1)
- [schemas/transcription.py](/Users/sams/Desktop/video-caption-pipeline-docker/schemas/transcription.py:1)
- [schemas/video_memory.py](/Users/sams/Desktop/video-caption-pipeline-docker/schemas/video_memory.py:1)
- [schemas/vlm.py](/Users/sams/Desktop/video-caption-pipeline-docker/schemas/vlm.py:1)

## Failure Handling

The pipeline is designed to degrade gracefully.

- malformed segment VLM JSON can be repaired
- transcript failures do not automatically fail the task
- task-level exceptions fall back to safe captions
- debug artifacts preserve enough state to inspect failures later

Relevant code:

- [pipeline/vlm_reasoning.py](/Users/sams/Desktop/video-caption-pipeline-docker/pipeline/vlm_reasoning.py:45)
- [main.py](/Users/sams/Desktop/video-caption-pipeline-docker/main.py:179)

## Summary

The actual pipeline is:

```text
task -> media normalization -> frame selection -> temporal segmentation
     -> parallel transcript + visual reasoning
     -> deterministic segment fusion
     -> image-backed global factual summary
     -> style-specific caption generation
```

That final separation is intentional:

- segment VLM builds local structured evidence
- global factual summary builds the final truth layer
- caption generation turns truth into tone
