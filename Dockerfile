FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Ollama sidecar for the local-first path (local tokens score zero). zstd is
# required by the ollama install script for binary decompression; curl stays for
# the entrypoint's warmup probe.
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates zstd && \
    curl -fsSL https://ollama.com/install.sh | sh && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Which local model to bake in. qwen2.5-coder:3b carries the code categories that
# plain qwen2.5:3b could not (Track B offline eval, docs/eval-results.md: debug
# 10/10, codegen 9/10 with EXECUTED assertions; ~equal on sentiment/ner/summ). The
# 4GB grading box holds one 3B model, so it's a swap, not a second model. Override
# at build time (`--build-arg LOCAL_MODEL=qwen2.5:3b`) to rebuild the safe anchor.
ARG LOCAL_MODEL=qwen2.5-coder:3b

# Bake the model weights into the image at build time — nothing downloads at
# runtime, so the container is ready well inside the 60s rule.
RUN ollama serve > /tmp/ollama_build.log 2>&1 & \
    OLLAMA_PID=$! && \
    sleep 5 && \
    ollama pull "$LOCAL_MODEL" && \
    kill $OLLAMA_PID

COPY src/ src/
COPY entrypoint.sh .
# .gitattributes asks for LF, but that's a checkout-time normalization that some
# Windows git configs (core.autocrlf=true, no reliable eol=lf override observed
# on this box) don't honor consistently -- a CRLF entrypoint.sh breaks the
# shebang once COPY'd into this Linux image (`exec ./entrypoint.sh: no such file
# or directory`, container never starts). Strip any \r at build time so the
# image is correct regardless of what line endings the host checkout produced.
RUN sed -i 's/\r$//' entrypoint.sh && chmod +x entrypoint.sh

# The harness injects only FIREWORKS_*; these defaults ARE the local-path config.
# LOCAL_CATEGORIES="" disables local entirely (pure remote = the proven config).
# Runner-pin rung R1 (2026-07-12, docs/eval-results.md "Runner-pin rung"): with the
# thread/num_ctx fixes local calls actually LAND for the first time, so category
# choice is now a real semantic-risk decision, not a no-op. local-5 = the verified
# set: sentiment/ner/summarization (run-18-proven 19/19) + debug/codegen (coder
# model, offline 10/10 debug & 9/10 codegen on EXECUTED assertions). factual/math/
# logic stay on kimi (~165-250 tok/task, terse): the quota-repro caught qwen2.5-
# coder:3b answering an easy math word problem confidently WRONG through the
# format verifier (run D, t9) — the exact silent-miss class that sank run 16.
# All-local (add factual,math,logic back) is the R3 moonshot, a separate gamble.
# Anchor for recovery if this drops below the gate: 6c8dcc4 (94.7-100% @ 3.9-4.1k).
ARG LOCAL_MODEL=qwen2.5-coder:3b
ENV LOCAL_MODEL=${LOCAL_MODEL}
ENV LOCAL_CATEGORIES="sentiment,ner,summarization,debug,codegen"
# Batching (Phase E) scored a token REGRESSION for this config (run 24: 5,067 vs
# the 3.9-4.1k anchor) and buys nothing under all-local (almost nothing hits
# remote to batch). Disable it in the image; the code path stays behind the flag.
ENV BATCHING_ENABLED="false"

ENTRYPOINT ["./entrypoint.sh"]
