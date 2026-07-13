#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${1:-gemma-caption-pipe:smoke}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT_DIR="$ROOT_DIR/input"
OUTPUT_DIR="$ROOT_DIR/output"

mkdir -p "$INPUT_DIR" "$OUTPUT_DIR"

cat > "$INPUT_DIR/tasks.json" <<'JSON'
[
  {
    "task_id": "v1",
    "video_url": "https://example.invalid/test.mp4",
    "styles": ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]
  }
]
JSON

rm -f "$OUTPUT_DIR/results.json"

docker buildx build --platform linux/amd64 --load -t "$IMAGE_NAME" "$ROOT_DIR"

docker run --platform linux/amd64 --rm \
  -v "$INPUT_DIR:/input" \
  -v "$OUTPUT_DIR:/output" \
  "$IMAGE_NAME"

python3 - "$OUTPUT_DIR/results.json" <<'PY'
import json
import pathlib
import sys

results_path = pathlib.Path(sys.argv[1])
if not results_path.exists():
    raise SystemExit("results.json was not created")

payload = json.loads(results_path.read_text(encoding="utf-8"))
if not isinstance(payload, list):
    raise SystemExit("results.json must contain a JSON array")
if len(payload) != 1:
    raise SystemExit(f"expected exactly 1 result, found {len(payload)}")

item = payload[0]
if sorted(item.keys()) != ["captions", "task_id"]:
    raise SystemExit(f"unexpected result keys: {sorted(item.keys())}")
if item["task_id"] != "v1":
    raise SystemExit(f"unexpected task_id: {item['task_id']}")

captions = item["captions"]
expected_styles = ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]
if sorted(captions.keys()) != sorted(expected_styles):
    raise SystemExit(f"unexpected caption styles: {sorted(captions.keys())}")
for style_name in expected_styles:
    if not isinstance(captions[style_name], str) or not captions[style_name].strip():
        raise SystemExit(f"caption for {style_name} must be a non-empty string")

print("Smoke test passed.")
PY
