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
3. describe the scene from frames with Kimi K2.6 on Fireworks
4. verify that description against the frames
5. generate captions one style at a time through Fireworks
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

- All model traffic uses Fireworks OpenAI-compatible chat completions
- The Cloudflare Worker injects `accounts/fireworks/models/kimi-k2p6` for every role
- Vision, caption, and judge requests all use `FIREWORKS_MODEL`
- `LLM_PROVIDER=fireworks|openrouter` chooses which OpenAI-compatible upstream handles vision, caption, and judge requests
- For `openrouter`, the worker sends `OPENROUTER_MODEL` in the request body and the proxy forwards it unchanged

## Structured Output

Structured output is enforced in one place:

- the Fireworks client sends `response_format.type = json_schema` with `strict = true`

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
- `RUN_JUDGE_CHECKS=true|false` (defaults to `false`)
- `LLM_PROVIDER=fireworks|openrouter`
- `FIREWORKS_MODEL`
- `FIREWORKS_PROXY_URL`
- `FIREWORKS_PROXY_TOKEN`
- `FIREWORKS_API_KEY`
- `OPENROUTER_MODEL`
- `OPENROUTER_PROXY_URL`
- `OPENROUTER_PROXY_TOKEN`
- `OPENROUTER_API_KEY`
- `ENABLE_LOCAL_WHISPER=true|false`

## Run

```bash
docker buildx build --platform linux/amd64 -t video-caption-pipeline .
docker run --rm -v "$(pwd)/input:/input" -v "$(pwd)/output:/output" video-caption-pipeline
```
