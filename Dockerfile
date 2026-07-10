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

# Bake the model weights into the image at build time — nothing downloads at
# runtime, so the container is ready well inside the 60s rule.
RUN ollama serve > /tmp/ollama_build.log 2>&1 & \
    OLLAMA_PID=$! && \
    sleep 5 && \
    ollama pull qwen2.5:3b && \
    kill $OLLAMA_PID

COPY src/ src/
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# The harness injects only FIREWORKS_*; these defaults ARE the local-path config.
# LOCAL_CATEGORIES="" disables local entirely (pure remote = the proven config).
# Run 16 (five local categories) failed at 15/19 — bisecting: safest pair first,
# widen one category per passing run.
ENV LOCAL_CATEGORIES="sentiment,ner"

ENTRYPOINT ["./entrypoint.sh"]
