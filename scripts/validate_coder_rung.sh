#!/usr/bin/env bash
# Pre-submission validation for the coder-model rung (debug+codegen local).
# Run this on a machine with Docker + Ollama BEFORE scoring the coder-model image.
# It is the go/no-go gate that separates "code-complete" from "safe to submit".
#
# Usage:  bash scripts/validate_coder_rung.sh
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== 1. Offline test gate (must be fully green) ==="
python -m pytest tests/ -q
python tests/test_router.py

echo ""
echo "=== 2. Coder-model SEMANTIC eval (executes generated code) ==="
echo "Prereq (throttled to mimic the 2-vCPU/4GB grading box):"
echo "  docker run -d --name ollama-eval --cpus=2 -m 4g -p 11434:11434 -v ollama-eval:/root/.ollama ollama/ollama"
echo "  docker exec ollama-eval ollama pull qwen2.5-coder:3b"
echo ""
echo "GO/NO-GO: debug >=9/10 AND codegen >=9/10, SEMANTIC_MISS <=1 per category."
echo "SEMANTIC_MISS is a wrong answer the verifiers would have submitted (run-16 class)."
python scripts/eval_local_code.py qwen2.5-coder:3b debug codegen

echo ""
echo "=== 3. Throttled full-harness rehearsal (build + run in the grading shape) ==="
echo "Builds the coder-model image, runs with a bogus key so remote fails fast;"
echo "read the LOCAL/LOCAL_FAIL/LOCAL_SKIP funnel + DONE elapsed_s < 510."
FIREWORKS_API_KEY=bogus \
FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1 \
ALLOWED_MODELS="gemma-4-26b-a4b-it,minimax-m3,kimi-k2p7-code" \
TASKS=tests/sample_tasks.json bash scripts/run_local.sh

echo ""
echo "=== 4. Network-isolation DQ audit (organizer rule) ==="
echo "Runtime must make NO outbound call except FIREWORKS_BASE_URL; localhost Ollama exempt."
echo "Run the image under --network=none: results.json must still be valid + exit 0,"
echo "the only ERROR lines being connection failures toward FIREWORKS_BASE_URL."
echo "  docker run --rm --network=none --cpus=2 -m 4g \\"
echo "    -e FIREWORKS_API_KEY=bogus -e FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1 \\"
echo "    -e ALLOWED_MODELS=kimi-k2p7-code \\"
echo "    -v \"\$PWD/tests/sample_tasks.json:/input/tasks.json\" -v \"\$PWD/out:/output\" \\"
echo "    hybrid-token-router:dev"
echo ""
echo "If all four pass, the coder-model image is safe to score. Keep the anchor"
echo "  ghcr.io/arming-afk/hybrid-token-router:7bb50c4  ready to re-submit."
