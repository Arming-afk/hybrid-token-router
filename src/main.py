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

from . import batching, client, local, models, prompts, router

INPUT_PATH = os.environ.get("INPUT_PATH", "/input/tasks.json")
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "/output/results.json")
DEADLINE_SECONDS = 8.5 * 60  # harness kills at 10 min; leave margin to write output
CONCURRENCY = 6  # gentler on the proxy's rate limits; 429-hit answers come back empty

# Batching (Phase E): tasks that still need a remote call after the zero-token
# paths (deterministic solver, local-first) collect into `pending` instead of
# calling the remote leg inline, then go out in grouped multi-task calls
# (src/batching.py) -- validated against a real model on the public endpoint
# (scripts/probe_batch_format.py) before this was wired in here. Any task whose
# batch slice is missing/malformed/unverified falls back to solve_remote, so no
# task is ever worse off than the pre-batching one-call-per-task baseline.
# Fail-open kill switch: set BATCHING_ENABLED=false to get that exact baseline
# back with zero code-path changes.
BATCHING_ENABLED = os.environ.get("BATCHING_ENABLED", "true").strip().lower() != "false"
# Reserved wall-clock so phase 2 (batches) always gets a real shot even if
# phase 1's local-first attempts are still churning through the serialized
# LOCAL_LOCK on a tight budget -- the always-remote categories (factual/math/
# logic) are exactly where batching pays off every run.
#
# 200s (a first guess) was WRONG -- validated wrong on the 106-task rehearsal
# (docs/eval-results.md, Phase E): reserving 200s of the 510s budget cut phase
# 1 down to ~310s, hit ITS OWN "phase 1 deadline reached" cutoff, and answered
# only 14/106 (vs the pre-batching baseline's 25) -- tasks got cancelled
# mid-flight waiting on LOCAL_LOCK that would otherwise have finished. The real
# public-endpoint probe (scripts/probe_batch_format.py) measured actual batch
# calls at 1.2-5.9s each; even several chunks running through CONCURRENCY=6
# realistically need seconds, not minutes. 60s gives ~10x margin over that
# measurement while giving phase 1 back the time it actually needs.
PHASE2_RESERVE_SECONDS = 60

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


async def try_complete_batch(task_ids: list[str], categories: list[str], model: str,
                             messages: list[dict], max_tokens: int) -> tuple[str, bool]:
    """Batch analogue of try_complete: one API call answering multiple tasks at
    once. Returns (text, truncated) exactly like try_complete; the caller (
    try_batch) is responsible for per-task parsing/fallback since a truncated
    or empty batch reply must escalate every task in it, not just one."""
    try:
        text, usage = await client.complete(model, messages, max_tokens)
        log("BATCH_USAGE", task_ids=task_ids, categories=categories, model=model, **usage)
        CALL_LOG.append({"task_id": task_ids, "category": categories, "model": model, **usage})
        return text, bool(usage.get("truncated"))
    except Exception as error:
        log("BATCH_ERROR", task_ids=task_ids, model=model, error=str(error)[:200])
        return "", False


async def llm_classify(task_id: str, prompt: str, tiers: dict) -> str:
    # The 2-token cap makes finish_reason=length expected here — ignore the flag.
    text, _ = await try_complete(task_id, "router", tiers["SMALL"],
                                 router.fallback_messages(prompt), max_tokens=2)
    return router.parse_fallback_letter(text)


def _remaining() -> float:
    return DEADLINE_SECONDS - (time.monotonic() - START)


async def try_local(task_id: str, category: str, prompt: str) -> str:
    """One serialized local attempt; returns "" whenever the remote path should run."""
    if _remaining() < LOCAL_MIN_REMAINING_SECONDS:
        log("LOCAL_SKIP", task_id=task_id, category=category, note="global budget low")
        return ""
    async with LOCAL_LOCK:
        # Re-check AFTER acquiring the lock: local generations are serialized, so a
        # task can wait minutes in this queue behind slower ones (run 19's -5 was
        # this starvation — long timeouts let one task hold the lock while others
        # queued past the deadline and returned blank). Bail to remote rather than
        # start a fresh (up to 60s) generation on a budget that already ran out.
        if _remaining() < LOCAL_MIN_REMAINING_SECONDS:
            log("LOCAL_SKIP", task_id=task_id, category=category, note="budget low after lock wait")
            return ""
        started = time.monotonic()
        try:
            text = await asyncio.to_thread(local.generate, prompt, category)
        except local.LocalError as error:
            log("LOCAL_FAIL", task_id=task_id, category=category,
                elapsed=round(time.monotonic() - started, 1), error=str(error)[:120])
            return ""
        elapsed = round(time.monotonic() - started, 1)
    ok, reason = local.verify(prompt, category, text)
    if not ok:
        log("LOCAL_REJECTED", task_id=task_id, category=category,
            elapsed=elapsed, reason=reason)
        return ""
    log("LOCAL", task_id=task_id, category=category, chars=len(text), elapsed=elapsed)
    return text


async def solve_remote(task_id: str, category: str, prompt: str, tiers: dict, results: dict) -> None:
    """One task's remote leg — identical to the pre-batching solo path. This is
    the fallback target for any batch slice that's missing, malformed, or fails
    verification, so no task is ever worse off than the one-call-per-task
    baseline it replaces."""
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


async def try_batch(chunk: list[dict], tiers: dict, results: dict) -> None:
    """One chunk of 1+ same-tier tasks. A size-1 chunk skips the batch format
    entirely (zero parsing risk, zero overhead, for zero savings otherwise). A
    whole-batch failure (exception, empty reply, or a cap-truncated completion —
    which can't be trusted for ANY task in it, unlike a single missing slice)
    escalates every item to solve_remote; a good reply is parsed and verified
    per task, with only the failing tasks escalated."""
    if len(chunk) == 1:
        item = chunk[0]
        await solve_remote(item["task_id"], item["category"], item["prompt"], tiers, results)
        return

    tier = prompts.SPEC[chunk[0]["category"]]["tier"]
    task_ids = [item["task_id"] for item in chunk]
    categories = [item["category"] for item in chunk]
    messages, max_tokens = batching.build_batch_messages(chunk)
    text, truncated = await try_complete_batch(task_ids, categories, tiers[tier], messages, max_tokens)

    if not text.strip() or truncated:
        for item in chunk:
            await solve_remote(item["task_id"], item["category"], item["prompt"], tiers, results)
        return

    parsed = batching.parse_batch_response(text, task_ids)
    for item in chunk:
        slice_ = parsed[item["task_id"]]
        if slice_["complete"]:
            ok, reason = batching.verify_slice(item["prompt"], item["category"], slice_["text"])
        else:
            ok, reason = False, "incomplete slice"
        if ok:
            log("BATCH", task_id=item["task_id"], category=item["category"], chars=len(slice_["text"]))
            results[item["task_id"]] = prompts.postprocess(item["category"], slice_["text"])
        else:
            log("BATCH_FALLBACK", task_id=item["task_id"], category=item["category"], reason=reason)
            await solve_remote(item["task_id"], item["category"], item["prompt"], tiers, results)


async def run_batches(pending: list[dict], tiers: dict, sem: asyncio.Semaphore, results: dict) -> None:
    chunks = batching.group_and_chunk(pending)

    async def _run_one(chunk: list[dict]) -> None:
        async with sem:
            await try_batch(chunk, tiers, results)

    await asyncio.gather(*(_run_one(chunk) for chunk in chunks), return_exceptions=True)


async def solve_task(task: dict, tiers: dict, sem: asyncio.Semaphore, results: dict,
                     pending: list[dict]) -> None:
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
        if BATCHING_ENABLED:
            pending.append({"task_id": task_id, "category": category, "prompt": prompt})
        else:
            await solve_remote(task_id, category, prompt, tiers, results)


async def run(tasks: list[dict], results: dict) -> None:
    global LOCAL_ENABLED
    LOCAL_ENABLED = bool(local.LOCAL_CATEGORIES) and local.is_available()
    log("LOCAL_STATUS", enabled=LOCAL_ENABLED, categories=sorted(local.LOCAL_CATEGORIES))
    log("BATCHING_STATUS", enabled=BATCHING_ENABLED)
    tiers = models.build_tiers()
    log("TIERS", **tiers)
    sem = asyncio.Semaphore(CONCURRENCY)
    pending: list[dict] = []
    jobs = [solve_task(task, tiers, sem, results, pending) for task in tasks
            if isinstance(task.get("prompt"), str) and task["prompt"].strip()]
    try:
        phase1_budget = _remaining()
        if BATCHING_ENABLED:
            phase1_budget = max(0.0, phase1_budget - PHASE2_RESERVE_SECONDS)
        try:
            await asyncio.wait_for(asyncio.gather(*jobs, return_exceptions=True), timeout=phase1_budget)
        except asyncio.TimeoutError:
            log("DEADLINE", note="phase 1 deadline reached, writing partial results")
        if BATCHING_ENABLED and pending:
            try:
                await asyncio.wait_for(run_batches(pending, tiers, sem, results), timeout=_remaining())
            except asyncio.TimeoutError:
                log("DEADLINE", note="phase 2 (batches) deadline reached, writing partial results")
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
