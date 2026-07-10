# Video Caption Pipeline

This repo now uses a clip-level caption architecture with three explicit modes:

- `verified_scene`
- `direct_vision`
- `observation_first`

The runtime entrypoint is [main.py](/Users/sams/Desktop/video-caption-pipeline-docker/main.py:1).

## Batch Flow

For each task in `/input/tasks.json`, the pipeline:

1. downloads the video
2. probes the media
3. normalizes the video
4. extracts audio when available
5. optionally loads or creates a local Whisper transcript
6. extracts a compact set of frames
7. runs the configured caption mode
8. writes `/output/results.json`

## Modes

### `verified_scene`

1. extract frames
2. read transcript if available
3. describe the scene from frames with the vision model
4. verify that description against the frames
5. generate captions one style at a time with OpenAI
6. optionally run judge checks

Code path:

- [pipeline/caption_pipeline.py](/Users/sams/Desktop/video-caption-pipeline-docker/pipeline/caption_pipeline.py:1)

### `direct_vision`

1. extract frames
2. send frames directly to the vision model
3. request all styles in one JSON response
4. optionally run judge checks

### `observation_first`

1. extract frames
2. build structured factual observations with [prompts/perception_system.txt](/Users/sams/Desktop/video-caption-pipeline-docker/prompts/perception_system.txt:1)
3. generate captions either:
   - in one combined JSON call
   - or one style at a time from the style prompt files
4. optionally run judge checks

## Frame Strategy

The frame strategy is simplified, but it still preserves the repo’s stronger extraction technique:

- dynamic frame count by duration
- current planned timestamp extraction using anchor, safety, and scene-change timestamps
- optional OpenCV extraction
- optional scene-midpoint extraction
- fallback to scene frames
- fallback to uniform frames
- perceptual deduplication

Relevant code:

- [pipeline/extract_frames.py](/Users/sams/Desktop/video-caption-pipeline-docker/pipeline/extract_frames.py:1)
- [pipeline/scene_change.py](/Users/sams/Desktop/video-caption-pipeline-docker/pipeline/scene_change.py:1)

## Models

- Vision analysis uses the proxy-configured vision model, defaulting to `VISION_MODEL=gemma-4-31b-it`
- Final caption writing uses OpenAI, defaulting to `OPENAI_CAPTION_MODEL=gpt-5.5`
- Judge checks use `OPENAI_JUDGE_MODEL`

## Structured Output

Structured output is enforced in two places:

- the vision client sends `responseSchema` when requesting JSON
- the OpenAI Responses client uses strict `text.format = json_schema`

All JSON outputs are also validated with Pydantic models after the API call.

## Input

`/input/tasks.json` must be a JSON array:

```json
[
  {
    "task_id": "v1",
    "video_url": "https://example.com/video.mp4",
    "styles": ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]
  }
]
```

Optional task fields:

- `transcript_url`
- `transcript_text`

Schema:

- [schemas/tasks.py](/Users/sams/Desktop/video-caption-pipeline-docker/schemas/tasks.py:1)

## Output

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

## Key Config

- `CAPTION_PIPELINE_MODE=verified_scene|direct_vision|observation_first`
- `OBSERVATION_CAPTION_MODE=combined|per_style`
- `RUN_JUDGE_CHECKS=true|false`
- `VISION_PROXY_URL`
- `VISION_PROXY_TOKEN`
- `VISION_MODEL`
- `OPENAI_PROXY_URL`
- `OPENAI_PROXY_TOKEN`
- `OPENAI_CAPTION_MODEL`
- `ENABLE_LOCAL_WHISPER=true|false`

## Run

```bash
docker buildx build --platform linux/amd64 -t video-caption-pipeline .
docker run --rm -v "$(pwd)/input:/input" -v "$(pwd)/output:/output" video-caption-pipeline
```
