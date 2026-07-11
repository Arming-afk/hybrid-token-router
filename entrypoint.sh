#!/bin/sh
# Start the Ollama sidecar, warm the model in the background, run the agent.
# Fail-open: if Ollama cannot start, main.py's startup probe sees it missing and
# every task takes the proven remote-only path — the agent itself must never die
# because of the sidecar. The agent starts immediately (60s readiness rule);
# the model load happens concurrently with the first remote-bound work.

(ollama serve > /tmp/ollama.log 2>&1 &) || true

# Warm the SAME model main.py will call (LOCAL_MODEL, default matches the Dockerfile
# bake). Warming the wrong model leaves the first real task to pay the ~57s cold
# start while holding the serialized local lock — the run-19 starvation trigger.
LOCAL_MODEL="${LOCAL_MODEL:-qwen2.5-coder:3b}"
(
  i=0
  while [ $i -lt 30 ]; do
    if curl -s http://localhost:11434/api/version > /dev/null 2>&1; then
      # one tiny generation forces the ~2GB model load now, not on task one
      curl -s http://localhost:11434/api/generate \
        -d "{\"model\":\"${LOCAL_MODEL}\",\"prompt\":\"hi\",\"stream\":false,\"keep_alive\":\"30m\",\"options\":{\"num_predict\":1}}" \
        > /dev/null 2>&1 || true
      break
    fi
    i=$((i + 1))
    sleep 1
  done
) &

exec python -m src.main
