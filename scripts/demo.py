"""Demo web chatbot: type a question, see which category the router picks and
which model/path would answer it — and get a real answer when a backend is up.

Presentation aid only, never shipped: the judged image copies src/ and
entrypoint.sh, not scripts/. Reuses src/ strictly read-only and mirrors
main.py's solve_task order (deterministic arithmetic -> deterministic logic ->
classify -> local-first -> remote with LARGE escalation), so what the demo
shows is what production does.

Batch mode (Phase E showcase): tick "batch" and enter one question per line to
watch src/batching.py group remote-pending tasks into per-tier chunks and
answer a whole chunk with ONE model call, with per-task parse/verify and solo
fallback. The production image ships BATCHING_ENABLED=false (run 24 measured
batching at 5,067 tokens vs the 3,853 anchor — the packed prompt re-bills every
instruction and the model answers longer), so this mode demos the machinery,
not the deployed default.

Run from the repo root:
    python scripts/demo.py                    # zero env needed: routing-only mode
    python scripts/demo.py --route-only       # never call a model, even with env set
    python scripts/demo.py --env .env.eval    # real answers via the practice arena
With FIREWORKS_API_KEY + FIREWORKS_BASE_URL set it answers remotely; with an
Ollama sidecar running it answers local categories for zero tokens.
"""
import asyncio
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# --env <file> loads a dotenv-style file (the repo's .env / .env.eval convention;
# PowerShell has no `set -a; . ./.env`). Shell-set variables still win.
if "--env" in sys.argv:
    try:
        env_path = Path(sys.argv[sys.argv.index("--env") + 1])
        lines = env_path.read_text(encoding="utf-8").splitlines()
    except (IndexError, OSError) as error:
        sys.exit(f"--env: {error}")
    for line in lines:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))

# The real Track 1 model list (tests/test_models.py) so build_tiers() works with
# zero setup; a caller-provided env still wins when demoing a different list.
os.environ.setdefault(
    "ALLOWED_MODELS",
    "minimax-m3,kimi-k2p7-code,gemma-4-31b-it,gemma-4-26b-a4b-it,gemma-4-31b-it-nvfp4",
)

from src import batching, client, local, models, prompts, router  # noqa: E402

HOST, PORT = "127.0.0.1", 8765
# --route-only: classification + target model only, guaranteed zero model calls
# (presentation mode — immune to slow endpoints and missing models).
ROUTE_ONLY = "--route-only" in sys.argv
REMOTE_OK = (not ROUTE_ONLY) and bool(
    os.environ.get("FIREWORKS_API_KEY") and os.environ.get("FIREWORKS_BASE_URL"))
TIERS = models.build_tiers()
REMOTE_CALL_CEILING = 90  # complete() can legitimately take ~25s x retries
BATCH_MAX_QUESTIONS = 8  # demo guard; production chunks are capped in batching.py
# What the judged image actually ships (Dockerfile: ENV BATCHING_ENABLED="false",
# after run 24 measured the batch prompt costing MORE tokens than solo calls).
IMAGE_BATCHING_DEFAULT = False

# --- backends ---------------------------------------------------------------

# One event loop for the process lifetime: client.py caches an AsyncOpenAI bound
# to the first loop that uses it, so per-request asyncio.run() would hand request
# #2 a client tied to a dead loop.
LOOP = asyncio.new_event_loop()
threading.Thread(target=LOOP.run_forever, daemon=True, name="asyncio-loop").start()

# Serialized like production: the grading box only really runs one local
# generation at a time, and the demo should show that behavior.
LOCAL_LOCK = threading.Lock()

_ollama = {"up": False, "checked": 0.0}
_OLLAMA_TTL = 20.0
_probe_lock = threading.Lock()


def ollama_up(force: bool = False) -> bool:
    if ROUTE_ONLY:
        return False
    now = time.monotonic()
    with _probe_lock:
        if force or now - _ollama["checked"] > _OLLAMA_TTL:
            _ollama["up"] = local.is_available(timeout=1.0)
            _ollama["checked"] = now
        return _ollama["up"]


def safe_complete(model: str, messages: list[dict], max_tokens: int):
    """(text, usage, error): failures become data so the UI can show them —
    e.g. gemma-26b 404ing on the public key demos the escalation path."""
    try:
        future = asyncio.run_coroutine_threadsafe(
            client.complete(model, messages, max_tokens), LOOP)
        text, usage = future.result(timeout=REMOTE_CALL_CEILING)
        return text, usage, None
    except Exception as error:
        return "", {"prompt_tokens": 0, "completion_tokens": 0, "truncated": False}, \
            f"{type(error).__name__}: {str(error)[:200]}"


# --- decision chain (mirrors src/main.py solve_task) -------------------------

def answer_question(prompt: str) -> dict:
    t0 = time.perf_counter()
    steps: list[str] = []
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    result = {
        "category": None, "classifier": None, "path": None, "tier": None,
        "model": None, "spec": None, "answer": None, "engine": "none",
        "usage": None, "local": None, "escalated": None, "steps": steps,
        # Cached flag only: a live probe here would tax every request ~2s on
        # Windows (localhost tries ::1 then 127.0.0.1, 1s timeout each) even for
        # categories that never touch local. Step 4 re-probes when it matters.
        "timings_ms": {}, "env": {"remote": REMOTE_OK, "ollama": _ollama["up"]},
        "error": None,
    }

    def merge_usage(u: dict) -> None:
        usage["prompt_tokens"] += u.get("prompt_tokens", 0)
        usage["completion_tokens"] += u.get("completion_tokens", 0)

    def done(**overrides) -> dict:
        result.update(overrides)
        result["timings_ms"]["total"] = round((time.perf_counter() - t0) * 1000)
        return result

    # 1. Pure arithmetic never touches any model (production runs this first,
    # before classification, so it works even when the router reads it as factual).
    det = router.deterministic_math_answer(prompt)
    if det is not None:
        steps.append("deterministic arithmetic: HIT — solved in-process, 0 tokens")
        return done(category="math", classifier="deterministic",
                    path="deterministic", answer=det, engine="in-process solver")
    steps.append("deterministic arithmetic: miss")

    # 2. Transitive-ordering logic puzzles ("A is older than B... who is
    # youngest?") also answer deterministically — 0 tokens AND immune to model
    # error; the solver self-gates to a unique extreme over a clean total order.
    logic = router.deterministic_logic_answer(prompt)
    if logic is not None:
        steps.append("deterministic logic: HIT — total order resolved in-process, 0 tokens")
        return done(category="logic", classifier="deterministic",
                    path="deterministic", answer=logic, engine="in-process solver")
    steps.append("deterministic logic: miss")

    # 3. Classify: regex rules first, the production LLM fallback when reachable.
    t = time.perf_counter()
    category = router.classify(prompt)
    classifier = "regex"
    if category is None:
        steps.append("regex classify: ambiguous (mixed math+logic signals)")
        if REMOTE_OK:
            text, fb_usage, err = safe_complete(
                TIERS["SMALL"], router.fallback_messages(prompt), 2)
            merge_usage(fb_usage)
            if err:
                category, classifier = "factual", "ambiguous-default"
                steps.append(f"LLM fallback failed ({err}) -> defaulting to factual")
            else:
                category, classifier = router.parse_fallback_letter(text), "llm-fallback"
                steps.append(f"LLM fallback on SMALL ({TIERS['SMALL']}): {category}")
        else:
            category, classifier = "factual", "ambiguous-default"
            steps.append("no API key -> production would ask the SMALL model; defaulting to factual")
    else:
        steps.append(f"regex classify: {category}")
    result["timings_ms"]["classify"] = round((time.perf_counter() - t) * 1000)
    result["category"], result["classifier"] = category, classifier

    # The routing target is shown even when nothing can execute it.
    messages, max_tokens, tier = prompts.render(category, prompt)
    result.update(tier=tier, model=TIERS[tier], spec={
        "instruction": prompts.SPEC[category]["instruction"], "max_tokens": max_tokens})

    # 4. Local-first: a verified local answer costs zero tokens.
    if category in local.LOCAL_CATEGORIES:
        if ROUTE_ONLY:
            steps.append(f"local-first: '{category}' is local-eligible — production tries "
                         f"{local.MODEL} in-image first for 0 tokens (skipped: route-only mode)")
        elif ollama_up():
            steps.append(f"local-first: '{category}' is local-eligible, Ollama up -> trying {local.MODEL}")
            t = time.perf_counter()
            try:
                with LOCAL_LOCK:
                    text = local.generate(prompt, category)
            except local.LocalError as error:
                result["local"] = {"attempted": True, "verify_ok": False,
                                   "verify_reason": f"generate failed: {str(error)[:120]}"}
                steps.append(f"local generate FAILED ({str(error)[:80]}) -> fail-open to remote")
            else:
                ok, reason = local.verify(prompt, category, text)
                result["local"] = {"attempted": True, "verify_ok": ok, "verify_reason": reason}
                if ok:
                    result["timings_ms"]["local"] = round((time.perf_counter() - t) * 1000)
                    steps.append("local answer passed verification -> 0 tokens")
                    return done(path="local", answer=text, engine=f"ollama {local.MODEL}")
                steps.append(f"local answer REJECTED ({reason}) -> fail-open to remote")
            result["timings_ms"]["local"] = round((time.perf_counter() - t) * 1000)
        else:
            steps.append(f"local-first: '{category}' is local-eligible but Ollama is not running")

    # 5. Remote. Batching context: with BATCHING_ENABLED this task would queue
    # into a per-tier chunk (see batch mode); the image ships it off after run 24
    # measured the packed prompt costing MORE tokens, so production calls solo.
    steps.append(f"batching: would join a {tier}-tier chunk if enabled — image ships "
                 f"BATCHING_ENABLED=false (run 24: 5,067 vs 3,853 tokens), so solo call")
    if not REMOTE_OK:
        note = "route-only mode" if ROUTE_ONLY else "no FIREWORKS_API_KEY/BASE_URL"
        steps.append(f"{note} -> route shown, not executed "
                     f"(would call {TIERS[tier]}, max_tokens={max_tokens})")
        return done(path="route-only")
    t = time.perf_counter()
    steps.append(f"remote call: {tier} ({TIERS[tier]}), max_tokens={max_tokens}")
    text, call_usage, err = safe_complete(TIERS[tier], messages, max_tokens)
    merge_usage(call_usage)
    engine_model = TIERS[tier]
    if err:
        steps.append(f"remote call failed: {err}")
    if not text.strip() or call_usage.get("truncated"):
        reason = "truncated at max_tokens" if text.strip() else "blank or error"
        retry_tier = "LARGE" if tier != "LARGE" else "MEDIUM"
        retry_max = max(max_tokens, 700)
        steps.append(f"escalating to {retry_tier} ({TIERS[retry_tier]}), "
                     f"max_tokens={retry_max} ({reason})")
        retry_text, retry_usage, retry_err = safe_complete(TIERS[retry_tier], messages, retry_max)
        merge_usage(retry_usage)
        result["escalated"] = {"to_tier": retry_tier, "model": TIERS[retry_tier], "reason": reason}
        if retry_text.strip():
            # An empty rescue must not erase a truncated-but-present first answer.
            text, engine_model = retry_text, TIERS[retry_tier]
        elif retry_err:
            steps.append(f"escalation failed too: {retry_err}")
    result["timings_ms"]["remote"] = round((time.perf_counter() - t) * 1000)
    answer = prompts.postprocess(category, text)
    if not answer:
        return done(path="remote", usage=usage, error=err or "no answer from any tier")
    return done(path="remote", answer=answer, engine=f"fireworks:{engine_model}", usage=usage)


# --- batch mode (mirrors src/main.py phase 2: run_batches/try_batch) ----------

def answer_batch(prompt_list: list[str]) -> dict:
    """Run several questions through the production pipeline shape: per-task
    zero-token phase first (both solvers + classify), then everything still
    pending goes through batching.group_and_chunk -> one call per chunk ->
    per-task parse/verify -> solo fallback for any failed slice. Local-first is
    reported but not executed here (a chunk's worth of serialized 25-40s local
    calls would stall a live demo); the single-question mode demos local live."""
    t0 = time.perf_counter()
    steps: list[str] = []
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    tasks: dict[str, dict] = {}
    pending: list[dict] = []

    def merge_usage(u: dict) -> None:
        usage["prompt_tokens"] += u.get("prompt_tokens", 0)
        usage["completion_tokens"] += u.get("completion_tokens", 0)

    # Phase 1: the zero-token per-task chain (solve_task order, local as note only).
    for i, prompt in enumerate(prompt_list, 1):
        task_id = f"q{i}"
        entry = {"task_id": task_id, "prompt": prompt, "category": None,
                 "path": None, "answer": None, "verify": None, "source": None}
        tasks[task_id] = entry
        det = router.deterministic_math_answer(prompt)
        if det is not None:
            entry.update(category="math", path="deterministic", answer=det, source="in-process solver")
            steps.append(f"{task_id}: deterministic arithmetic HIT (0 tokens)")
            continue
        logic = router.deterministic_logic_answer(prompt)
        if logic is not None:
            entry.update(category="logic", path="deterministic", answer=logic, source="in-process solver")
            steps.append(f"{task_id}: deterministic logic HIT (0 tokens)")
            continue
        category = router.classify(prompt) or "factual"
        entry["category"] = category
        note = ""
        if category in local.LOCAL_CATEGORIES:
            note = f" (production tries {local.MODEL} locally first — skipped in batch demo)"
        steps.append(f"{task_id}: classified {category} -> pending remote{note}")
        pending.append({"task_id": task_id, "category": category, "prompt": prompt})

    # Phase 2: group + chunk + one call per chunk (batching.py, the real code).
    chunk_reports = []
    chunks = batching.group_and_chunk(pending)
    if pending:
        steps.append(f"phase 2: {len(pending)} pending task(s) -> {len(chunks)} chunk(s) "
                     f"via group_and_chunk (per-tier, capped)")
    for chunk in chunks:
        tier = prompts.SPEC[chunk[0]["category"]]["tier"]
        ids = [item["task_id"] for item in chunk]
        report = {"tier": tier, "model": TIERS[tier], "task_ids": ids,
                  "usage": None, "elapsed_ms": None, "error": None, "solo": len(chunk) == 1}
        chunk_reports.append(report)

        if len(chunk) == 1:  # a batch of one is not a batch: production goes solo
            item = chunk[0]
            steps.append(f"chunk [{item['task_id']}]: single task -> solo path (no batch format)")
            _solo_answer(item, tasks, steps, merge_usage, report)
            continue

        messages, max_tokens = batching.build_batch_messages(chunk)
        if not REMOTE_OK:
            steps.append(f"chunk {ids}: route-only — would send ONE {tier} call "
                         f"({TIERS[tier]}, max_tokens={max_tokens}) for {len(chunk)} tasks")
            for item in chunk:
                tasks[item["task_id"]].update(path="route-only", source=f"{tier} batch (not executed)")
            continue
        t = time.perf_counter()
        steps.append(f"chunk {ids}: ONE {tier} call ({TIERS[tier]}, max_tokens={max_tokens})")
        text, call_usage, err = safe_complete(TIERS[tier], messages, max_tokens)
        merge_usage(call_usage)
        report["usage"] = {k: call_usage.get(k, 0) for k in ("prompt_tokens", "completion_tokens")}
        report["elapsed_ms"] = round((time.perf_counter() - t) * 1000)
        if err or not text.strip() or call_usage.get("truncated"):
            report["error"] = err or ("truncated" if text.strip() else "blank reply")
            steps.append(f"chunk {ids}: batch call failed ({report['error']}) -> "
                         f"every task falls back to solo (production behavior)")
            for item in chunk:
                _solo_answer(item, tasks, steps, merge_usage, report)
            continue
        parsed = batching.parse_batch_response(text, ids)
        for item in chunk:
            slice_ = parsed[item["task_id"]]
            if slice_["complete"]:
                ok, reason = batching.verify_slice(item["prompt"], item["category"], slice_["text"])
            else:
                ok, reason = False, "incomplete slice"
            entry = tasks[item["task_id"]]
            entry["verify"] = reason
            if ok:
                entry.update(path="batch", answer=prompts.postprocess(item["category"], slice_["text"]),
                             source=f"{tier} batch · {TIERS[tier]}")
                steps.append(f"{item['task_id']}: batch slice parsed + verified OK")
            else:
                steps.append(f"{item['task_id']}: slice rejected ({reason}) -> solo fallback")
                _solo_answer(item, tasks, steps, merge_usage, report)

    calls = sum(1 for r in chunk_reports if not r["solo"] and r["usage"] is not None)
    return {
        "tasks": [tasks[f"q{i}"] for i in range(1, len(prompt_list) + 1)],
        "chunks": chunk_reports, "steps": steps, "usage": usage,
        "image_batching": IMAGE_BATCHING_DEFAULT,
        "timings_ms": {"total": round((time.perf_counter() - t0) * 1000)},
        "summary": f"{len(prompt_list)} tasks -> {calls} batch call(s) "
                   f"+ {sum(1 for t_ in tasks.values() if t_['path'] == 'solo-fallback')} solo fallback(s), "
                   f"{sum(1 for t_ in tasks.values() if t_['path'] == 'deterministic')} solved free",
    }


def _solo_answer(item: dict, tasks: dict, steps: list, merge_usage, report: dict) -> None:
    """The exact solo remote leg (solve_remote shape: call, escalate on blank/
    truncation) for one task — the fallback that keeps any batch failure no
    worse than the pre-batching baseline."""
    entry = tasks[item["task_id"]]
    if not REMOTE_OK:
        entry.update(path="route-only", source="solo (not executed)")
        return
    messages, max_tokens, tier = prompts.render(item["category"], item["prompt"])
    text, call_usage, err = safe_complete(TIERS[tier], messages, max_tokens)
    merge_usage(call_usage)
    engine = TIERS[tier]
    if not text.strip() or call_usage.get("truncated"):
        retry_tier = "LARGE" if tier != "LARGE" else "MEDIUM"
        retry_text, retry_usage, _ = safe_complete(TIERS[retry_tier], messages, max(max_tokens, 700))
        merge_usage(retry_usage)
        steps.append(f"{item['task_id']}: solo escalated to {retry_tier}")
        if retry_text.strip():
            text, engine = retry_text, TIERS[retry_tier]
    answer = prompts.postprocess(item["category"], text)
    if answer:
        entry.update(path="solo-fallback", answer=answer, source=f"solo · {engine}")
    else:
        entry.update(path="solo-fallback", source=f"solo · {engine}",
                     answer=None, verify=err or "no answer from any tier")


# --- HTTP server -------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    def _send(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: dict) -> None:
        self._send(json.dumps(payload, ensure_ascii=False).encode(),
                   "application/json; charset=utf-8")

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(PAGE.encode(), "text/html; charset=utf-8")
        elif self.path == "/api/status":
            self._send_json({"tiers": TIERS, "remote": REMOTE_OK, "ollama": ollama_up(),
                             "route_only": ROUTE_ONLY,
                             "local_categories": sorted(local.LOCAL_CATEGORIES),
                             "image_batching": IMAGE_BATCHING_DEFAULT})
        else:
            self.send_error(404)

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
        except Exception:
            body = {}
        try:
            if self.path == "/api/ask":
                prompt = body.get("prompt", "")
                if not isinstance(prompt, str) or not prompt.strip():
                    self._send_json({"error": "empty prompt"})
                    return
                self._send_json(answer_question(prompt.strip()))
            elif self.path == "/api/batch":
                raw = body.get("prompts", [])
                prompt_list = [p.strip() for p in raw if isinstance(p, str) and p.strip()]
                if not prompt_list:
                    self._send_json({"error": "no questions — one per line"})
                    return
                if len(prompt_list) > BATCH_MAX_QUESTIONS:
                    self._send_json({"error": f"batch demo caps at {BATCH_MAX_QUESTIONS} questions"})
                    return
                self._send_json(answer_batch(prompt_list))
            else:
                self.send_error(404)
        except Exception as error:  # the UI must always get JSON, never a 500 page
            self._send_json({"error": f"{type(error).__name__}: {str(error)[:300]}"})

    def log_message(self, fmt, *args):
        pass  # keep the console to our own startup prints


def main() -> None:
    print(f"tiers: {json.dumps(TIERS)}")
    if ROUTE_ONLY:
        print("route-only mode: no model is ever called; every answer shows the routing decision")
    else:
        print(f"remote (Fireworks): {'ON' if REMOTE_OK else 'off — routing-only for remote categories'}")
        print(f"ollama local:       {'ON' if ollama_up(force=True) else 'off'}"
              f"  (local categories: {', '.join(sorted(local.LOCAL_CATEGORIES)) or 'none'})")
    print("batching (Phase E): demo-able in batch mode; production image ships it OFF (run 24)")
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"\ndemo ready -> http://{HOST}:{PORT}   (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        # Close the SDK client inside its loop; GC after the loop is gone
        # crashes the interpreter at exit on Windows (see src/client.py).
        try:
            asyncio.run_coroutine_threadsafe(client.aclose(), LOOP).result(timeout=5)
        except Exception:
            pass
        LOOP.call_soon_threadsafe(LOOP.stop)


# --- page --------------------------------------------------------------------

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Hybrid Token Router — demo</title>
<style>
  :root {
    --bg: #0f1117; --panel: #181b24; --panel2: #1f2330; --border: #2a2f3e;
    --text: #e6e9f0; --dim: #8b93a7; --accent: #7aa2ff;
  }
  * { box-sizing: border-box; margin: 0; }
  body {
    background: var(--bg); color: var(--text); height: 100vh;
    display: flex; flex-direction: column;
    font: 15px/1.5 "Segoe UI", system-ui, sans-serif;
  }
  header {
    padding: 14px 20px; background: var(--panel); border-bottom: 1px solid var(--border);
    display: flex; flex-wrap: wrap; gap: 10px 18px; align-items: center;
  }
  header h1 { font-size: 17px; font-weight: 600; margin-right: auto; }
  header h1 small { color: var(--dim); font-weight: 400; margin-left: 8px; }
  .dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 5px; }
  .status { font-size: 13px; color: var(--dim); }
  .tierchips { width: 100%; display: flex; flex-wrap: wrap; gap: 6px; font-size: 12px; }
  .tierchips span {
    background: var(--panel2); border: 1px solid var(--border);
    border-radius: 20px; padding: 2px 10px; color: var(--dim);
  }
  .tierchips b { color: var(--text); font-weight: 600; }
  #chat { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 14px; }
  .msg { max-width: 72%; border-radius: 14px; padding: 10px 14px; white-space: pre-wrap; overflow-wrap: anywhere; }
  .user { align-self: flex-end; background: #2b4bb3; border-bottom-right-radius: 4px; }
  .bot  { align-self: flex-start; background: var(--panel); border: 1px solid var(--border); border-bottom-left-radius: 4px; }
  .bot.wide { max-width: 92%; }
  .badges { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 8px; }
  .badge {
    font-size: 11px; font-weight: 700; letter-spacing: .4px;
    border-radius: 4px; padding: 2px 8px; color: #10131b; text-transform: uppercase;
  }
  .badge.model, .badge.tokens { background: var(--panel2); color: var(--dim); font-weight: 500; text-transform: none; }
  .badge.esc { background: #facc15; }
  .cat-factual { background: #a78bfa; } .cat-math { background: #22d3ee; }
  .cat-sentiment { background: #f472b6; } .cat-summarization { background: #fbbf24; }
  .cat-ner { background: #34d399; } .cat-debug { background: #f87171; }
  .cat-logic { background: #60a5fa; } .cat-codegen { background: #a3e635; }
  .path-deterministic { background: #22c55e; } .path-local { background: #3b82f6; color: #fff; }
  .path-remote { background: #f97316; } .path-route-only { background: #6b7280; color: #fff; }
  .path-batch { background: #c084fc; } .path-solo-fallback { background: #fb923c; }
  .noanswer { color: var(--dim); font-style: italic; }
  .error { color: #f87171; }
  details { margin-top: 8px; font-size: 13px; color: var(--dim); }
  details summary { cursor: pointer; user-select: none; }
  details ol { margin: 6px 0 6px 20px; }
  details .kv { margin: 2px 0; }
  details .kv b { color: var(--text); font-weight: 600; }
  .batchtask { border-top: 1px solid var(--border); padding-top: 8px; margin-top: 8px; }
  .batchtask .q { color: var(--dim); font-size: 13px; margin-bottom: 4px; }
  form { display: flex; gap: 10px; padding: 14px 20px; background: var(--panel); border-top: 1px solid var(--border); align-items: flex-end; }
  #input, #batchinput {
    flex: 1; background: var(--panel2); border: 1px solid var(--border); border-radius: 10px;
    color: var(--text); padding: 10px 14px; font: inherit; outline: none;
  }
  #batchinput { display: none; resize: vertical; min-height: 84px; }
  #input:focus, #batchinput:focus { border-color: var(--accent); }
  button {
    background: var(--accent); color: #10131b; border: 0; border-radius: 10px;
    padding: 10px 22px; font: inherit; font-weight: 600; cursor: pointer;
  }
  button:disabled { opacity: .5; cursor: default; }
  .mode { display: flex; align-items: center; gap: 6px; color: var(--dim); font-size: 13px; user-select: none; white-space: nowrap; padding-bottom: 10px; }
</style>
</head>
<body>
<header>
  <h1>Hybrid Token Router <small>routing demo — ask anything</small></h1>
  <span class="status" id="st-remote"><span class="dot" style="background:#6b7280"></span>Fireworks</span>
  <span class="status" id="st-ollama"><span class="dot" style="background:#6b7280"></span>Ollama</span>
  <div class="tierchips" id="tiers"></div>
</header>
<div id="chat"></div>
<form id="form">
  <input id="input" placeholder="Ask a question… e.g. What is 15% of 240?" autocomplete="off" autofocus>
  <textarea id="batchinput" placeholder="One question per line — the demo groups them into per-tier chunks and answers each chunk with ONE model call (Phase E; Ctrl+Enter to send)"></textarea>
  <label class="mode"><input type="checkbox" id="batchmode"> batch</label>
  <button id="send" type="submit">Send</button>
</form>
<script>
const chat = document.getElementById("chat");
const form = document.getElementById("form");
const input = document.getElementById("input");
const batchinput = document.getElementById("batchinput");
const batchmode = document.getElementById("batchmode");
const send = document.getElementById("send");

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
}
function scroll() { chat.scrollTop = chat.scrollHeight; }

fetch("/api/status").then(r => r.json()).then(s => {
  const dot = (on) => `<span class="dot" style="background:${on ? "#22c55e" : "#6b7280"}"></span>`;
  if (s.route_only) {
    document.getElementById("st-remote").innerHTML = dot(false) + "route-only mode — no model calls";
    document.getElementById("st-ollama").innerHTML = "";
  } else {
    document.getElementById("st-remote").innerHTML = dot(s.remote) + "Fireworks " + (s.remote ? "connected" : "off");
    document.getElementById("st-ollama").innerHTML = dot(s.ollama) + "Ollama " + (s.ollama ? "up" : "off");
  }
  const chips = document.getElementById("tiers");
  for (const [tier, model] of Object.entries(s.tiers)) {
    const chip = el("span"); chip.innerHTML = `<b>${tier}</b> ${model}`;
    chips.appendChild(chip);
  }
  const lc = el("span"); lc.innerHTML = `<b>LOCAL</b> ${s.local_categories.join(", ") || "—"}`;
  chips.appendChild(lc);
  const bc = el("span");
  bc.innerHTML = `<b>BATCH</b> Phase E — ${s.image_batching ? "ON in image" : "off in image (run 24), demo via ☑ batch"}`;
  chips.appendChild(bc);
});

batchmode.addEventListener("change", () => {
  const on = batchmode.checked;
  input.style.display = on ? "none" : "";
  // Explicit "block": clearing the inline style would fall back to the
  // stylesheet's `#batchinput { display: none }` and hide BOTH fields.
  batchinput.style.display = on ? "block" : "none";
  (on ? batchinput : input).focus();
});
batchinput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) form.requestSubmit();
});

function renderBot(d) {
  const bubble = el("div", "msg bot");
  if (d.error && !d.category) {           // hard failure before routing
    bubble.appendChild(el("div", "error", d.error));
    return bubble;
  }
  const badges = el("div", "badges");
  badges.appendChild(el("span", "badge cat-" + d.category, d.category));
  badges.appendChild(el("span", "badge path-" + d.path, d.path));
  if (d.path === "deterministic") {
    badges.appendChild(el("span", "badge model", "in-process solver · 0 tokens"));
  } else if (d.path === "local") {
    badges.appendChild(el("span", "badge model", d.engine + " · 0 tokens"));
  } else if (d.model) {
    badges.appendChild(el("span", "badge model", d.tier + " · " + d.model));
  }
  if (d.escalated) badges.appendChild(el("span", "badge esc", "→ " + d.escalated.to_tier));
  if (d.usage) badges.appendChild(el("span", "badge tokens",
      "↑" + d.usage.prompt_tokens + " ↓" + d.usage.completion_tokens + " tokens"));
  bubble.appendChild(badges);

  if (d.answer) {
    bubble.appendChild(el("div", null, d.answer));
  } else if (d.path === "route-only") {
    bubble.appendChild(el("div", "noanswer",
      "Routing decision — production sends this to " + d.model +
      " (" + d.tier + ", max_tokens=" + d.spec.max_tokens + ")."));
  } else {
    bubble.appendChild(el("div", "error", d.error || "no answer"));
  }

  const details = el("details");
  details.appendChild(el("summary", null, "details"));
  const steps = el("ol");
  for (const s of d.steps || []) steps.appendChild(el("li", null, s));
  details.appendChild(steps);
  const kv = (k, v) => { const row = el("div", "kv"); row.innerHTML = "<b>" + k + ":</b> "; row.appendChild(document.createTextNode(v)); details.appendChild(row); };
  kv("classifier", d.classifier);
  if (d.spec) { kv("instruction", d.spec.instruction); kv("max_tokens", d.spec.max_tokens); }
  if (d.local) kv("local verify", (d.local.verify_ok ? "passed" : "rejected") + " — " + d.local.verify_reason);
  if (d.escalated) kv("escalation", d.escalated.to_tier + " (" + d.escalated.model + ") — " + d.escalated.reason);
  kv("timing", Object.entries(d.timings_ms).map(([k, v]) => k + " " + v + "ms").join(", "));
  bubble.appendChild(details);
  return bubble;
}

function renderBatch(d) {
  const bubble = el("div", "msg bot wide");
  if (d.error) { bubble.appendChild(el("div", "error", d.error)); return bubble; }
  const badges = el("div", "badges");
  badges.appendChild(el("span", "badge path-batch", "batch mode"));
  badges.appendChild(el("span", "badge model", d.summary));
  badges.appendChild(el("span", "badge tokens",
      "↑" + d.usage.prompt_tokens + " ↓" + d.usage.completion_tokens + " tokens total"));
  bubble.appendChild(badges);
  for (const c of d.chunks) {
    if (c.solo) continue;
    const line = el("div", "noanswer",
      "chunk [" + c.task_ids.join(", ") + "] → 1 call on " + c.tier + " (" + c.model + ")" +
      (c.usage ? " · ↑" + c.usage.prompt_tokens + " ↓" + c.usage.completion_tokens : "") +
      (c.elapsed_ms ? " · " + c.elapsed_ms + "ms" : "") +
      (c.error ? " · FAILED: " + c.error : ""));
    bubble.appendChild(line);
  }
  for (const t of d.tasks) {
    const box = el("div", "batchtask");
    box.appendChild(el("div", "q", t.task_id + " · " + t.prompt));
    const b = el("div", "badges");
    if (t.category) b.appendChild(el("span", "badge cat-" + t.category, t.category));
    b.appendChild(el("span", "badge path-" + t.path, t.path));
    if (t.source) b.appendChild(el("span", "badge model", t.source));
    box.appendChild(b);
    if (t.answer) box.appendChild(el("div", null, t.answer));
    else box.appendChild(el("div", "noanswer", t.verify || "route shown, not executed"));
    bubble.appendChild(box);
  }
  const details = el("details");
  details.appendChild(el("summary", null, "details"));
  const steps = el("ol");
  for (const s of d.steps || []) steps.appendChild(el("li", null, s));
  details.appendChild(steps);
  bubble.appendChild(details);
  return bubble;
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const isBatch = batchmode.checked;
  let url, payload, shown;
  if (isBatch) {
    const lines = batchinput.value.split("\n").map(l => l.trim()).filter(Boolean);
    if (!lines.length) return;
    url = "/api/batch"; payload = { prompts: lines }; shown = lines.join("\n");
    batchinput.value = "";
  } else {
    const prompt = input.value.trim();
    if (!prompt) return;
    url = "/api/ask"; payload = { prompt }; shown = prompt;
    input.value = "";
  }
  chat.appendChild(el("div", "msg user", shown));
  input.disabled = true; batchinput.disabled = true; send.disabled = true;
  const pending = el("div", "msg bot noanswer", isBatch ? "batching…" : "routing…");
  chat.appendChild(pending); scroll();
  try {
    const r = await fetch(url, { method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload) });
    const data = await r.json();
    chat.replaceChild(isBatch ? renderBatch(data) : renderBot(data), pending);
  } catch (err) {
    pending.className = "msg bot error"; pending.textContent = "request failed: " + err;
  }
  input.disabled = false; batchinput.disabled = false; send.disabled = false;
  (isBatch ? batchinput : input).focus(); scroll();
});
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
