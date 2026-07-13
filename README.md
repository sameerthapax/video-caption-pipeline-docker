# GemmaCaption-Pipe

GemmaCaption-Pipe is a Docker-first video captioning pipeline built around an observation-first workflow. It samples ordered frames from each video, uses Gemma to create structured visual evidence, generates style-specific caption candidates, and sends those candidates to an independent judge before writing the final result.

Supported caption styles:

- `formal`
- `sarcastic`
- `humorous_tech`
- `humorous_non_tech`

## Why Gemma

Gemma is the default model for both visual observation and caption generation. This was a deliberate engineering choice rather than using one large prompt for every stage.

- **Multimodal observation:** Gemma can inspect the ordered JPEG frames directly, allowing the pipeline to reason over a clip without uploading the original video to the model provider.
- **Structured intermediate output:** a compact JSON evidence packet separates visual perception from creative writing. This makes hallucinations easier to detect and keeps each caption grounded in the same facts.
- **Style range:** the same model can produce objective captions and three distinct humor personas when each style is isolated in its own prompt.
- **Practical throughput:** uniform sampling limits each clip to 6-12 frames, while independent style calls run concurrently. This keeps the workload predictable for batch processing.
- **Portable deployment:** Gemma is accessed through an OpenAI-compatible provider interface. The provider and model are configuration, not pipeline architecture.

The current default is `google/gemma-4-31b-it` through OpenRouter. Kimi K2.6 is configured as an independent judge by default so generation and evaluation do not share the same model bias. Both routes are configurable.

## Pipeline

```text
video URL
  -> download and probe
  -> uniformly sample 6-12 timestamped frames
  -> Gemma observation JSON
  -> two Gemma candidates for each requested style
  -> independent frame-aware judge
  -> retry weak styles with judge feedback, up to 3 quality rounds
  -> results.json
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the detailed data flow, schemas, concurrency model, and failure behavior.

## Quick Start

Requirements:

- Docker with `buildx`
- OpenRouter API key for the default Gemma route
- Fireworks API key for the default judge route

Create local configuration:

```bash
cp .env.example .env
```

Set `OPENROUTER_API_KEY` and `FIREWORKS_API_KEY` in `.env`, then create `input/tasks.json`:

```json
[
  {
    "task_id": "clip-1",
    "video_url": "https://example.com/video.mp4",
    "styles": ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]
  }
]
```

Build and run:

```bash
docker buildx build \
  --platform linux/amd64 \
  --tag gemma-caption-pipe:latest \
  --load \
  .

docker run --platform linux/amd64 --rm \
  --env-file .env \
  -v "$(pwd)/input:/input" \
  -v "$(pwd)/output:/output" \
  gemma-caption-pipe:latest
```

Results are written to `output/results.json`:

```json
[
  {
    "task_id": "clip-1",
    "captions": {
      "formal": "...",
      "sarcastic": "...",
      "humorous_tech": "...",
      "humorous_non_tech": "..."
    }
  }
]
```

Optional task fields are `transcript_url` and `transcript_text`. Input validation is defined in [schemas/tasks.py](schemas/tasks.py).

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `LLM_PROVIDER` | `openrouter` | Provider for vision and caption generation |
| `OPENROUTER_MODEL` | `google/gemma-4-31b-it` | Gemma model identifier |
| `OPENROUTER_API_KEY` | empty | Direct OpenRouter credential |
| `JUDGE_LLM_PROVIDER` | `fireworks` | Independent judge provider |
| `JUDGE_MODEL` | `accounts/fireworks/models/kimi-k2p6` | Judge model identifier |
| `FIREWORKS_API_KEY` | empty | Direct Fireworks credential |
| `MAX_FRAMES_PER_VIDEO` | `12` | Upper frame limit; the pipeline never uses fewer than 6 |
| `MAX_CONCURRENT_JOBS` | `3` | Concurrent video tasks and provider request slots |
| `CAPTION_ACCEPTANCE_THRESHOLD` | `0.94` | Required accuracy and style score |
| `FIREWORKS_MAX_RETRIES` | `2` | HTTP attempts per provider request |
| `FIREWORKS_TIMEOUT_SECONDS` | `120` | Hard wall-clock limit for one request |
| `LOG_MODEL_IO` | `false` | Logs sanitized model payloads and responses when enabled |

The complete safe template is [.env.example](.env.example). Despite the historical variable prefix, `FIREWORKS_*_MAX_TOKENS` and timeout controls are shared by both OpenAI-compatible clients.

## Optional Cloudflare Proxy

[caption-proxy](caption-proxy) is an optional Cloudflare Worker that keeps provider keys out of the Docker runtime. It authenticates callers with `X-Proxy-Token`, injects the upstream provider key, and forwards OpenAI-compatible requests.

```bash
cd caption-proxy
npm install
cp .dev.vars.example .dev.vars
npx wrangler secret put OPENROUTER_API_KEY
npx wrangler secret put FIREWORKS_API_KEY
npx wrangler secret put PROXY_TOKEN
npm run deploy
```

Set the deployed Worker URLs and matching proxy token in the root `.env`. Do not commit `.env`, `.dev.vars`, API keys, or proxy tokens. See [SECURITY.md](SECURITY.md).

For the included routing layout, `FIREWORKS_PROXY_URL` is the Worker origin and `OPENROUTER_PROXY_URL` is the same origin with `/openrouter` appended.

## Development

```bash
python3 -m pip install -r requirements.txt pytest
pytest -q
python3 -m compileall -q main.py pipeline schemas services worker

cd caption-proxy
npm install
npm run typecheck
```

## License

Licensed under the [Apache License 2.0](LICENSE).
