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
    ollama pull qwen2.5-coder:3b && \
    kill $OLLAMA_PID

COPY src/ src/
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# The harness injects only FIREWORKS_*; these defaults ARE the local-path config.
# LOCAL_CATEGORIES="" disables local entirely (pure remote = the proven config).
# Bisect ladder: run 16 (five cats on PLAIN qwen2.5:3b) failed 15/19; run 17
# (sentiment,ner) 17/19; run 18 (+summarization) 19/19. This rung swaps the model
# to the coder sibling — offline-proven on debug (10/10) and codegen (9/10),
# parity on the easy categories — and re-adds the code categories it can carry.
ENV LOCAL_MODEL="qwen2.5-coder:3b"
ENV LOCAL_CATEGORIES="sentiment,ner,summarization,debug,codegen"

ENTRYPOINT ["./entrypoint.sh"]
