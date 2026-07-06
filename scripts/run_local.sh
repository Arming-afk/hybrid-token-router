#!/usr/bin/env bash
# Mimic the judging harness: build the image, mount input/output, pass env through.
set -euo pipefail
cd "$(dirname "$0")/.."

: "${FIREWORKS_API_KEY:?set FIREWORKS_API_KEY}"
: "${FIREWORKS_BASE_URL:?set FIREWORKS_BASE_URL}"
: "${ALLOWED_MODELS:?set ALLOWED_MODELS}"

IMAGE=${IMAGE:-hybrid-token-router:dev}
TASKS=${TASKS:-tests/sample_tasks.json}

docker build -t "$IMAGE" .
mkdir -p out
docker run --rm \
  -v "$(pwd)/$TASKS:/input/tasks.json:ro" \
  -v "$(pwd)/out:/output" \
  -e FIREWORKS_API_KEY -e FIREWORKS_BASE_URL -e ALLOWED_MODELS \
  "$IMAGE" 2>&1 | tee out/run.log
echo "exit code: ${PIPESTATUS[0]}"

python -m json.tool out/results.json > /dev/null && echo "results.json: valid JSON"
python scripts/token_report.py out/run.log
