# GemmaCaption-Pipe Architecture

GemmaCaption-Pipe uses a staged, observation-first architecture. Perception, creative generation, and evaluation are separate model calls with strict JSON boundaries.

## Components

| Component | Responsibility |
| --- | --- |
| [main.py](main.py) | Batch entrypoint, task concurrency, downloads, output ordering, and safe fallbacks |
| [pipeline/extract_frames.py](pipeline/extract_frames.py) | Dynamic uniform frame sampling and timestamp labels |
| [pipeline/caption_pipeline.py](pipeline/caption_pipeline.py) | Observation, per-style generation, judging, and quality retries |
| [services/fireworks_client.py](services/fireworks_client.py) | Shared OpenAI-compatible multimodal client, schemas, timeouts, and transport retries |
| [services/client_pool.py](services/client_pool.py) | Generator and judge client construction |
| [schemas/caption.py](schemas/caption.py) | Observation, candidate, and judge response schemas |
| [prompts](prompts) | Perception, judge, and style-specific instructions |
| [caption-proxy](caption-proxy) | Optional authenticated Cloudflare forwarding proxy |

## End-to-End Flow

```text
/input/tasks.json
  |
  v
main.py ----------------------------------------------------+
  | download, validate, probe                               |
  v                                                         |
uniform frame extraction                                    |
  | 6-12 ordered JPEGs with timestamps                      |
  v                                                         |
Gemma vision pass                                            |
  | strict ObservationResult JSON                           |
  +----------------------+----------------------+------------+
  |                      |                      |
  v                      v                      v
formal candidates   sarcastic candidates   humor candidates
  |                      |                      |
  +----------------------+----------------------+
                         |
                         v
                independent frame-aware judge
                         |
              pass >= threshold or regenerate
                         |
                         v
                 /output/results.json
```

## Frame Sampling

The pipeline intentionally avoids scene detection and dense decoding. It samples frames uniformly across the full video so every request has predictable size and chronological coverage.

- Minimum: 6 frames
- Maximum: 12 frames
- Count: selected dynamically from video duration
- Width: 768 pixels by default
- Ordering: chronological, with explicit frame IDs and timestamps

This approach trades fine-grained action recognition for stable batch throughput and broad temporal coverage.

## Observation Contract

Gemma receives every sampled frame in one multimodal request. [prompts/perception_system.txt](prompts/perception_system.txt) asks for a compact evidence packet containing:

- scene summary and setting
- important subjects, actions, and objects
- beginning, middle, and end timeline
- temporal highlights
- camera behavior
- high-confidence caption facts

The response must satisfy `ObservationResult`. Caption generation consumes this JSON instead of independently reinterpreting the frames, which gives all four styles a shared factual base.

## Caption Generation

Each requested style runs independently and concurrently. The relevant style prompt receives the same observation JSON and returns two candidates:

```json
{
  "candidate_1": "conservative candidate",
  "candidate_2": "alternative candidate"
}
```

Candidate cleanup removes wrappers, escaped text, and accidental labels. Deterministic policy checks flag unsupported precision, technical jargon in non-technical humor, missing tech analogies, and common invented joke props.

Style prompts:

- [formal](prompts/style_formal.txt)
- [sarcastic](prompts/style_sarcastic.txt)
- [humorous tech](prompts/style_humorous_tech.txt)
- [humorous non-tech](prompts/style_humorous_non_tech.txt)

## Judging and Retries

The judge receives the original frames, observation JSON, target style, both candidates, and deterministic policy findings. It returns separate accuracy and style scores. The calibrated combined score is the lower dimension, so strong style cannot compensate for weak grounding.

A style passes only when all of the following meet `CAPTION_ACCEPTANCE_THRESHOLD`:

- accuracy status is `pass`
- style status is `pass`
- accuracy score
- style score
- combined score

Each style gets up to three quality rounds. Failed rounds feed the previous candidates and judge notes into the next generation request. If no round passes, the final round is returned.

Transport retries are separate from quality retries. Each HTTP request gets two attempts by default, a bounded queue wait, and a hard request timeout.

## Concurrency

`main.py` processes up to `MAX_CONCURRENT_JOBS` videos in a thread pool. Inside each video task, requested styles use `asyncio.gather`. Provider-specific semaphores cap active model requests, preventing style fan-out from exceeding the configured request limit.

Results retain the same order as the input tasks even though tasks execute concurrently.

## Provider Routing

The generator and judge are configured independently:

```text
vision + captions -> LLM_PROVIDER / OPENROUTER_MODEL
judge             -> JUDGE_LLM_PROVIDER / JUDGE_MODEL
```

Default routing uses Gemma through OpenRouter for observation and generation, then Kimi through Fireworks for independent judging. Either route can use a direct provider API key or the optional Cloudflare proxy.

The Cloudflare Worker never stores credentials in source or `wrangler.jsonc`. Provider keys and the caller token are Wrangler secrets. Non-sensitive model names remain ordinary Worker variables.

## Failure Behavior

- Invalid input fails validation before model calls.
- Downloads and media probes are bounded by configured constraints.
- Malformed model JSON is repaired heuristically, then by one schema-guided repair request when needed.
- Provider calls have queue and wall-clock timeouts.
- Task failures produce complete conservative fallback captions instead of breaking the full batch.
- Temporary task files are removed unless debug artifact retention is enabled.

## Output Boundary

Only the submission-facing structure is written to `/output/results.json`:

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

Observations, judge checks, and frame metadata remain internal artifacts unless debug persistence is enabled.
