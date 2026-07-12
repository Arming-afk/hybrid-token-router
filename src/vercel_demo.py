"""Vercel demo backend helpers.

This module exposes the same routing/demo behavior as scripts/demo.py for a
serverless deployment on Vercel. The static UI lives in public/index.html and
calls the API endpoints under /api.
"""

import asyncio
import os
import time

from . import batching, client, local, models, prompts, router

os.environ.setdefault(
    "ALLOWED_MODELS",
    "minimax-m3,kimi-k2p7-code,gemma-4-31b-it,gemma-4-26b-a4b-it,gemma-4-31b-it-nvfp4",
)

ROUTE_ONLY = os.environ.get("ROUTE_ONLY", "false").strip().lower() == "true"
REMOTE_OK = (not ROUTE_ONLY) and bool(
    os.environ.get("FIREWORKS_API_KEY") and os.environ.get("FIREWORKS_BASE_URL")
)
TIERS = models.build_tiers()
BATCH_MAX_QUESTIONS = 8

_ollama = {"up": False, "checked": 0.0}
_OLLAMA_TTL = 20.0


def ollama_up(force: bool = False) -> bool:
    if ROUTE_ONLY:
        return False
    now = time.monotonic()
    if force or now - _ollama["checked"] > _OLLAMA_TTL:
        _ollama["up"] = local.is_available(timeout=1.0)
        _ollama["checked"] = now
    return _ollama["up"]


async def _safe_complete(model: str, messages: list[dict], max_tokens: int) -> tuple[str, dict, str | None]:
    try:
        text, usage = await client.complete(model, messages, max_tokens)
        return text, usage, None
    except Exception as error:
        return "", {"prompt_tokens": 0, "completion_tokens": 0, "truncated": False}, \
            f"{type(error).__name__}: {str(error)[:200]}"


def _wrapped_safe_complete(model: str, messages: list[dict], max_tokens: int) -> tuple[str, dict, str | None]:
    # One asyncio.run per call means one event loop per call; the cached
    # AsyncOpenAI in client.py binds to the loop that first used it, so it must
    # be closed before that loop dies or the NEXT call gets a client tied to a
    # dead loop ("Event loop is closed"). Per-call close costs a TLS handshake;
    # correctness over warm-start reuse for a demo.
    async def call() -> tuple[str, dict, str | None]:
        try:
            return await _safe_complete(model, messages, max_tokens)
        finally:
            await client.aclose()
    return asyncio.run(call())


def _done(result: dict, usage: dict, t0: float, **overrides) -> dict:
    result.update(overrides)
    result["timings_ms"]["total"] = round((time.perf_counter() - t0) * 1000)
    if result.get("usage") is None:
        result["usage"] = usage if usage["prompt_tokens"] or usage["completion_tokens"] else None
    return result


def answer_question(prompt: str) -> dict:
    t0 = time.perf_counter()
    steps: list[str] = []
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    result = {
        "category": None, "classifier": None, "path": None, "tier": None,
        "model": None, "spec": None, "answer": None, "engine": "none",
        "usage": None, "local": None, "escalated": None, "steps": steps,
        "timings_ms": {}, "env": {"remote": REMOTE_OK, "ollama": _ollama["up"]},
        "error": None,
    }

    def merge_usage(u: dict) -> None:
        usage["prompt_tokens"] += u.get("prompt_tokens", 0)
        usage["completion_tokens"] += u.get("completion_tokens", 0)

    def done(**overrides) -> dict:
        return _done(result, usage, t0, **overrides)

    det = router.deterministic_math_answer(prompt)
    if det is not None:
        steps.append("deterministic arithmetic: HIT — solved in-process, 0 tokens")
        return done(category="math", classifier="deterministic",
                    path="deterministic", answer=det, engine="in-process solver")
    steps.append("deterministic arithmetic: miss")

    logic = router.deterministic_logic_answer(prompt)
    if logic is not None:
        steps.append("deterministic logic: HIT — total order resolved in-process, 0 tokens")
        return done(category="logic", classifier="deterministic",
                    path="deterministic", answer=logic, engine="in-process solver")
    steps.append("deterministic logic: miss")

    category = router.classify(prompt)
    classifier = "regex"
    if category is None:
        steps.append("regex classify: ambiguous (mixed math+logic signals)")
        text, fb_usage, err = _wrapped_safe_complete(
            TIERS["SMALL"], router.fallback_messages(prompt), 2)
        merge_usage(fb_usage)
        if err:
            category, classifier = "factual", "ambiguous-default"
            steps.append(f"LLM fallback failed ({err}) -> defaulting to factual")
        else:
            category, classifier = router.parse_fallback_letter(text), "llm-fallback"
            steps.append(f"LLM fallback on SMALL ({TIERS['SMALL']}): {category}")
    else:
        steps.append(f"regex classify: {category}")
    result["timings_ms"]["classify"] = round((time.perf_counter() - t0) * 1000)
    result["category"], result["classifier"] = category, classifier

    messages, max_tokens, tier = prompts.render(category, prompt)
    result.update(tier=tier, model=TIERS[tier], spec={
        "instruction": prompts.SPEC[category]["instruction"], "max_tokens": max_tokens})

    if category in local.LOCAL_CATEGORIES and not ROUTE_ONLY:
        if ollama_up():
            steps.append(f"local-first: '{category}' is local-eligible, Ollama up -> trying {local.MODEL}")
            try:
                text = local.generate(prompt, category)
            except local.LocalError as error:
                result["local"] = {"attempted": True, "verify_ok": False,
                                    "verify_reason": f"generate failed: {str(error)[:120]}"}
                steps.append(f"local generate FAILED ({str(error)[:80]}) -> fail-open to remote")
            else:
                ok, reason = local.verify(prompt, category, text)
                result["local"] = {"attempted": True, "verify_ok": ok, "verify_reason": reason}
                if ok:
                    result["timings_ms"]["local"] = round((time.perf_counter() - t0) * 1000)
                    steps.append("local answer passed verification -> 0 tokens")
                    return done(path="local", answer=text, engine=f"ollama {local.MODEL}")
                steps.append(f"local answer REJECTED ({reason}) -> fail-open to remote")
            result["timings_ms"]["local"] = round((time.perf_counter() - t0) * 1000)
        else:
            steps.append(f"local-first: '{category}' is local-eligible but Ollama is not running")

    steps.append(f"batching: would join a {tier}-tier chunk if enabled — image ships "
                 f"BATCHING_ENABLED=false (run 24: 5,067 vs 3,853 tokens), so solo call")
    if not REMOTE_OK:
        note = "route-only mode" if ROUTE_ONLY else "no FIREWORKS_API_KEY/BASE_URL"
        steps.append(f"{note} -> route shown, not executed (would call {TIERS[tier]}, max_tokens={max_tokens})")
        return done(path="route-only")

    t = time.perf_counter()
    steps.append(f"remote call: {tier} ({TIERS[tier]}), max_tokens={max_tokens}")
    text, call_usage, err = _wrapped_safe_complete(TIERS[tier], messages, max_tokens)
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
        retry_text, retry_usage, retry_err = _wrapped_safe_complete(
            TIERS[retry_tier], messages, retry_max)
        merge_usage(retry_usage)
        result["escalated"] = {"to_tier": retry_tier, "model": TIERS[retry_tier], "reason": reason}
        if retry_text.strip():
            text, engine_model = retry_text, TIERS[retry_tier]
        elif retry_err:
            steps.append(f"escalation failed too: {retry_err}")
    result["timings_ms"]["remote"] = round((time.perf_counter() - t) * 1000)
    answer = prompts.postprocess(category, text)
    if not answer:
        return done(path="remote", usage=usage, error=err or "no answer from any tier")
    return done(path="remote", answer=answer, engine=f"fireworks:{engine_model}", usage=usage)


def answer_batch(prompt_list: list[str]) -> dict:
    t0 = time.perf_counter()
    steps: list[str] = []
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    tasks: dict[str, dict] = {}
    pending: list[dict] = []

    def merge_usage(u: dict) -> None:
        usage["prompt_tokens"] += u.get("prompt_tokens", 0)
        usage["completion_tokens"] += u.get("completion_tokens", 0)

    if len(prompt_list) > BATCH_MAX_QUESTIONS:
        return {"error": f"batch demo caps at {BATCH_MAX_QUESTIONS} questions"}

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

        if len(chunk) == 1:
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
        text, call_usage, err = _wrapped_safe_complete(TIERS[tier], messages, max_tokens)
        merge_usage(call_usage)
        report["usage"] = {k: call_usage.get(k, 0) for k in ("prompt_tokens", "completion_tokens")}
        report["elapsed_ms"] = round((time.perf_counter() - t) * 1000)
        if err or not text.strip() or call_usage.get("truncated"):
            report["error"] = err or ("truncated" if text.strip() else "blank reply")
            steps.append(f"chunk {ids}: batch call failed ({report['error']}) -> every task falls back to solo (production behavior)")
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
            if ok:
                steps.append(f"{item['task_id']}: batch slice parsed + verified OK")
                tasks[item["task_id"]].update(path="batch", answer=prompts.postprocess(item["category"], slice_["text"]),
                                                source=f"{tier} batch · {TIERS[tier]}")
            else:
                steps.append(f"{item['task_id']}: slice rejected ({reason}) -> solo fallback")
                _solo_answer(item, tasks, steps, merge_usage, report)

    calls = sum(1 for r in chunk_reports if not r["solo"] and r["usage"] is not None)
    summary = (f"{len(prompt_list)} tasks -> {calls} batch call(s) "
               f"+ {sum(1 for t_ in tasks.values() if t_['path'] == 'solo-fallback')} solo fallback(s), "
               f"{sum(1 for t_ in tasks.values() if t_['path'] == 'deterministic')} solved free")
    return _done({"tasks": [tasks[f"q{i}"] for i in range(1, len(prompt_list) + 1)],
                  "chunks": chunk_reports, "steps": steps,
                  "usage": usage, "image_batching": False,
                  "timings_ms": {}, "summary": summary}, usage, t0)


def _solo_answer(item: dict, tasks: dict, steps: list[str], merge_usage, report: dict) -> None:
    entry = tasks[item["task_id"]]
    if not REMOTE_OK:
        entry.update(path="route-only", source="solo (not executed)")
        return
    messages, max_tokens, tier = prompts.render(item["category"], item["prompt"])
    text, call_usage, err = _wrapped_safe_complete(TIERS[tier], messages, max_tokens)
    merge_usage(call_usage)
    engine = TIERS[tier]
    if not text.strip() or call_usage.get("truncated"):
        retry_tier = "LARGE" if tier != "LARGE" else "MEDIUM"
        retry_text, retry_usage, _ = _wrapped_safe_complete(TIERS[retry_tier], messages, max(max_tokens, 700))
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


def status_payload() -> dict:
    return {
        "tiers": TIERS,
        "remote": REMOTE_OK,
        "ollama": ollama_up(force=True),
        "route_only": ROUTE_ONLY,
        "local_categories": sorted(local.LOCAL_CATEGORIES),
        "image_batching": False,
    }
