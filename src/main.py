"""Entrypoint: read /input/tasks.json, route each task, write /output/results.json.

Invariants the harness scores on:
- /output/results.json must exist and be valid JSON even if calls fail or the deadline hits.
- Exit 0 whenever results were written; a missing/invalid input file exits non-zero.
"""
import asyncio
import json
import os
import sys
import time

from . import client, local, models, prompts, router

INPUT_PATH = os.environ.get("INPUT_PATH", "/input/tasks.json")
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "/output/results.json")
DEADLINE_SECONDS = 8.5 * 60  # harness kills at 10 min; leave margin to write output
CONCURRENCY = 6  # gentler on the proxy's rate limits; 429-hit answers come back empty

# Local-first: verified local answers cost zero tokens. Enabled at startup only if
# the Ollama sidecar responds (fail-open: without it, behavior is the proven
# remote-only config). Local generations are serialized — the 2 vCPU grading box
# can only really run one at a time — and skipped when the global budget runs low,
# because a remote fallback takes ~2-7s while a local attempt can take ~15s.
LOCAL_ENABLED = False
LOCAL_LOCK = asyncio.Lock()
LOCAL_MIN_REMAINING_SECONDS = 150

START = time.monotonic()

# One entry per successful API call; dumped to inference_log.json next to the
# results. The guide's "no inference log is required for Track 2" phrasing implies
# Track 1 expects one. Single event loop -> plain list append is safe.
CALL_LOG: list[dict] = []


def log(event: str, **fields) -> None:
    print(event + " " + json.dumps(fields), file=sys.stderr, flush=True)


async def try_complete(task_id: str, category: str, model: str,
                       messages: list[dict], max_tokens: int) -> tuple[str, bool]:
    """Return (text, truncated). Truncated answers are likely wrong (code cut
    mid-function, missing Answer line) — callers escalate them like blanks."""
    try:
        text, usage = await client.complete(model, messages, max_tokens)
        log("USAGE", task_id=task_id, category=category, model=model, **usage)
        CALL_LOG.append({"task_id": task_id, "category": category, "model": model, **usage})
        return text, bool(usage.get("truncated"))
    except Exception as error:
        log("ERROR", task_id=task_id, model=model, error=str(error)[:200])
        return "", False


async def llm_classify(task_id: str, prompt: str, tiers: dict) -> str:
    # The 2-token cap makes finish_reason=length expected here — ignore the flag.
    text, _ = await try_complete(task_id, "router", tiers["SMALL"],
                                 router.fallback_messages(prompt), max_tokens=2)
    return router.parse_fallback_letter(text)


async def try_local(task_id: str, category: str, prompt: str) -> str:
    """One serialized local attempt; returns "" whenever the remote path should run."""
    remaining = DEADLINE_SECONDS - (time.monotonic() - START)
    if remaining < LOCAL_MIN_REMAINING_SECONDS:
        log("LOCAL_SKIP", task_id=task_id, note="global budget low")
        return ""
    async with LOCAL_LOCK:
        try:
            text = await asyncio.to_thread(local.generate, prompt, category)
        except local.LocalError as error:
            log("LOCAL_FAIL", task_id=task_id, error=str(error)[:120])
            return ""
    ok, reason = local.verify(prompt, category, text)
    if not ok:
        log("LOCAL_REJECTED", task_id=task_id, reason=reason)
        return ""
    log("LOCAL", task_id=task_id, category=category, chars=len(text))
    return text


async def solve_task(task: dict, tiers: dict, sem: asyncio.Semaphore, results: dict) -> None:
    task_id, prompt = task["task_id"], task["prompt"]
    async with sem:
        # Pure arithmetic never touches the API: solved in-process for zero tokens,
        # before classification so it works even when the router reads "847 x 23"
        # as factual.
        det = router.deterministic_math_answer(prompt)
        if det is not None:
            log("LOCAL", task_id=task_id, solver="deterministic-arithmetic")
            results[task_id] = det
            return
        category = router.classify(prompt)
        if category is None:
            category = await llm_classify(task_id, prompt, tiers)
        if LOCAL_ENABLED and category in local.LOCAL_CATEGORIES:
            text = await try_local(task_id, category, prompt)
            if text:
                results[task_id] = text
                return
        messages, max_tokens, tier = prompts.render(category, prompt)
        text, truncated = await try_complete(task_id, category, tiers[tier], messages, max_tokens)
        if not text.strip() or truncated:
            retry_tier = "LARGE" if tier != "LARGE" else "MEDIUM"
            # LARGE may be a reasoning model whose billed hidden thinking competes with
            # the visible answer for max_tokens; give the rescue attempt enough room.
            retry_max = max(max_tokens, 700)
            retry_text, _ = await try_complete(task_id, category, tiers[retry_tier],
                                               messages, retry_max)
            # An empty rescue must not erase a truncated-but-present first answer.
            if retry_text.strip():
                text = retry_text
        results[task_id] = prompts.postprocess(category, text)


async def run(tasks: list[dict], results: dict) -> None:
    global LOCAL_ENABLED
    LOCAL_ENABLED = bool(local.LOCAL_CATEGORIES) and local.is_available()
    log("LOCAL_STATUS", enabled=LOCAL_ENABLED, categories=sorted(local.LOCAL_CATEGORIES))
    tiers = models.build_tiers()
    log("TIERS", **tiers)
    sem = asyncio.Semaphore(CONCURRENCY)
    jobs = [solve_task(task, tiers, sem, results) for task in tasks
            if isinstance(task.get("prompt"), str) and task["prompt"].strip()]
    remaining = DEADLINE_SECONDS - (time.monotonic() - START)
    try:
        await asyncio.wait_for(asyncio.gather(*jobs, return_exceptions=True), timeout=remaining)
    except asyncio.TimeoutError:
        log("DEADLINE", note="deadline reached, writing partial results")
    finally:
        await client.aclose()


def write_results(tasks: list[dict], results: dict) -> None:
    out_dir = os.path.dirname(OUTPUT_PATH)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    payload = [{"task_id": t["task_id"], "answer": results.get(t["task_id"], "")} for t in tasks]
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)


def write_inference_log() -> None:
    # Best-effort: the log must never endanger results.json (which is already written).
    try:
        totals = {
            "prompt_tokens": sum(c.get("prompt_tokens", 0) for c in CALL_LOG),
            "completion_tokens": sum(c.get("completion_tokens", 0) for c in CALL_LOG),
            "calls": len(CALL_LOG),
        }
        path = os.path.join(os.path.dirname(OUTPUT_PATH) or ".", "inference_log.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"calls": CALL_LOG, "totals": totals}, f, ensure_ascii=False)
    except Exception as error:
        log("WARN", note="could not write inference log", error=str(error)[:200])


def main() -> None:
    with open(INPUT_PATH, encoding="utf-8") as f:
        raw = json.load(f)
    # A malformed entry must never take down the run: drop anything without a task_id
    # (nothing to key the answer on) and let promptless tasks fall through as "".
    tasks = [t for t in raw if isinstance(t, dict) and t.get("task_id")]
    if len(tasks) != len(raw):
        log("SKIPPED", invalid_entries=len(raw) - len(tasks))
    results = {task["task_id"]: "" for task in tasks}
    try:
        asyncio.run(run(tasks, results))
    except Exception as error:
        log("FATAL", error=str(error)[:300])
    write_results(tasks, results)
    write_inference_log()
    log("DONE", tasks=len(tasks), answered=sum(1 for a in results.values() if a),
        elapsed_s=round(time.monotonic() - START, 1))


if __name__ == "__main__":
    main()
