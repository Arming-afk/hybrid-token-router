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
RUN chmod +x entrypoint.sh

# The harness injects only FIREWORKS_*; these defaults ARE the local-path config.
# LOCAL_CATEGORIES="" disables local entirely (pure remote = the proven config).
# Bisect ladder: run 17 (sentiment,ner) 17/19; run 18 (+summarization) 100% @ 4,178.
# This rung adds debug,codegen on the coder model — the ~1k lever toward sub-3k,
# gated on Track B's offline execution eval + the run-19 lock-starvation fix.
ARG LOCAL_MODEL=qwen2.5-coder:3b
ENV LOCAL_MODEL=${LOCAL_MODEL}
ENV LOCAL_CATEGORIES="sentiment,ner,summarization,debug,codegen"

ENTRYPOINT ["./entrypoint.sh"]
