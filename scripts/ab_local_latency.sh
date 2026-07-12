#!/usr/bin/env bash
# A/B the local-runner latency hypotheses under the grading box's constraints
# (--cpus=2 --memory=4g on a many-core host = CPU-quota container, the exact
# environment where llama.cpp's detected-core-count threading oversubscribes).
#
# Runs the SAME image twice with a bogus Fireworks key (every remote call fails
# fast, so the log isolates pure local behavior):
#   A: LOCAL_NUM_THREAD=16  — simulates the OLD behavior (threads = detected host
#      cores) under the 2-cpu quota
#   B: LOCAL_NUM_THREAD=2   — the rung's pin
# Compare the LOCAL/LOCAL_FAIL 'elapsed' values per call.
set -uo pipefail
cd "$(dirname "$0")/.."

IMAGE=${IMAGE:-hybrid-token-router:rung}
TASKS=${TASKS:-tests/sample_tasks.json}

HOST_DIR="$(pwd)"
if command -v cygpath > /dev/null 2>&1; then
  HOST_DIR="$(cygpath -w "$HOST_DIR")"
  export MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*'
fi

mkdir -p out
for run in "A:16" "B:2"; do
  name="${run%%:*}"; threads="${run##*:}"
  echo "=== RUN $name: LOCAL_NUM_THREAD=$threads (quota --cpus=2, 16-core host) ==="
  docker run --rm --cpus=2 --memory=4g --platform linux/amd64 \
    -v "$HOST_DIR/$TASKS:/input/tasks.json:ro" \
    -v "$HOST_DIR/out:/output" \
    -e FIREWORKS_API_KEY=bogus-key-local-only \
    -e FIREWORKS_BASE_URL=http://127.0.0.1:9 \
    -e ALLOWED_MODELS="minimax-m3,kimi-k2p7-code,gemma-4-26b-a4b-it" \
    -e LOCAL_NUM_THREAD="$threads" \
    "$IMAGE" 2>&1 | tee "out/ab_$name.log" | grep -E "LOCAL|DONE" || true
  echo
done

echo "=== SUMMARY (elapsed seconds per local call) ==="
python - <<'EOF'
import json, re
for name in "AB":
    ok, fail = [], []
    for line in open(f"out/ab_{name}.log", encoding="utf-8", errors="replace"):
        m = re.match(r"(LOCAL|LOCAL_FAIL) (\{.*\})", line.strip())
        if not m:
            continue
        d = json.loads(m.group(2))
        if "elapsed" not in d:
            continue
        (ok if m.group(1) == "LOCAL" else fail).append((d.get("category"), d["elapsed"]))
    print(f"--- RUN {name}: {len(ok)} LOCAL ok, {len(fail)} LOCAL_FAIL ---")
    for tag, rows in (("ok", ok), ("FAIL", fail)):
        for cat, el in rows:
            print(f"  {tag:4} {cat:14} {el:6.1f}s")
EOF
