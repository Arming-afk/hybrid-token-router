#!/bin/sh
# Start the Ollama sidecar, warm the model in the background, run the agent.
# Fail-open: if Ollama cannot start, main.py's startup probe sees it missing and
# every task takes the proven remote-only path — the agent itself must never die
# because of the sidecar. The agent starts immediately (60s readiness rule);
# the model load happens concurrently with the first remote-bound work.

# One serialized caller (main.py's LOCAL_LOCK) — parallel slots would only
# multiply the KV cache allocation (num_ctx x parallel) on the 4GB box.
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_MAX_LOADED_MODELS=1

(ollama serve > /tmp/ollama.log 2>&1 &) || true

# Warm with the EXACT payload src/local.py builds (same model, same runner
# options — num_ctx/num_thread). In Ollama, a call whose runner options differ
# from the loaded runner's forces a full model RELOAD; the old warmup sent no
# options, so the first real task threw the warm runner away and paid the ~57s
# cold start while holding the serialized local lock (the run-19 starvation
# trigger). Building the payload from the same code that makes production calls
# means the two can never drift apart again.
(
  WARMUP_PAYLOAD="$(python -c 'from src import local; print(local.warmup_payload())')"
  i=0
  while [ $i -lt 30 ]; do
    if curl -s http://localhost:11434/api/version > /dev/null 2>&1; then
      curl -s http://localhost:11434/api/generate \
        -d "$WARMUP_PAYLOAD" \
        > /dev/null 2>&1 || true
      break
    fi
    i=$((i + 1))
    sleep 1
  done
) &

exec python -m src.main
