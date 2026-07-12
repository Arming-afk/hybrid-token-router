"""Local inference sidecar (Ollama + qwen2.5:3b) and its zero-token answer verifiers.

Contest rule: "Local models and tokens used locally count as zero for the final
score." Every task in LOCAL_CATEGORIES is attempted here first; a verified local
answer costs 0 tokens. Anything that fails — the call, the timeout, or a verifier —
falls back to the normal Fireworks path in main.py (fail-open: with Ollama absent
the agent behaves exactly like the proven remote-only config).

Design facts this module leans on (validated by the 100%-at-3753 competitor whose
architecture this adapts, and sized for the 2 vCPU / 4 GB grading box):
- qwen2.5:3b on 2 vCPUs generates ~12-30 tok/s; short answers finish in 5-15s.
- num_ctx 8192 so long summarization passages don't get silently truncated
  (KV cache cost ~300MB, safe in 4GB).
- done_reason == "length" means the answer was cut mid-thought: never submit it.
- Local tokens are free, so local instructions optimize for JUDGE satisfaction,
  not brevity: sentiment keeps its justification, debug keeps the bug sentence.
- Small local models fail silently (plausible wrong answers, no error) — that is
  what the verifiers are for: anything provably wrong escalates to Fireworks.
"""
import ast
import json
import os
import re
import urllib.error
import urllib.request

BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL = os.environ.get("LOCAL_MODEL", "qwen2.5:3b")
# First call pays the model load from disk (~2GB into RAM); later calls don't.
# Measured 2026-07-11 (2-core pin): a loaded-under-contention first call took 57s
# while the same prompt warm took 3.4s. The entrypoint warms the model in the
# background, but a task can still race it, so the first real call gets a wide
# budget. Later calls are compute-only: even 870-word passages answer in ~5s.
FIRST_CALL_TIMEOUT = 60.0
CALL_TIMEOUT = 15.0
# debug/codegen answer with a full fenced code block (see _CATEGORY_INSTRUCTIONS),
# a materially longer completion than a sentiment label or an NER line list, even
# on a short prompt. The base/scaled timeout above is tuned to PROMPT length
# (prompt-eval time); it has no term for COMPLETION length (generation time), so
# code categories were timing out at scale even on short prompts (Phase D 106-task
# rehearsal, docs/eval-results.md: 7/7 attempted debug+codegen calls timed out at
# 15.6-16.3s, right at the untuned ceiling, vs 6-14s for sentiment/ner/summarization
# on comparable prompt lengths). Give code categories a higher floor and ceiling;
# main.py's LOCAL_MIN_REMAINING_SECONDS re-check after the lock is the guard
# against this starving the queue on a tight overall budget.
#
# 30.0/55.0 is a validated sweet spot, not an arbitrary first guess -- tested
# both directions on the same 106-task rehearsal (docs/eval-results.md "Phase D
# follow-up" sections):
# - 30.0/55.0: fixed debug outright (0/13 -> 5/13 local success, 18.6-25.6s/call).
#   codegen still 0/13 but got 1 attempt through the queue (vs 0 before).
# - 45.0/70.0 (tried and REJECTED): made things WORSE, not better -- debug
#   dropped to 4/13 and codegen dropped to ZERO attempts (0/13, fully skipped).
#   Longer ceiling -> each code call holds LOCAL_LOCK longer -> fewer tasks fit
#   through the serialized queue before the budget runs out -> total answered
#   fell 25 -> 23. This is a real queueing trade-off (higher per-call service
#   time trades against total queue throughput), not free headroom -- do not
#   raise further without a way to shrink the per-call time too (e.g. a lower
#   NUM_PREDICT specifically for codegen), which is the next thing to try.
CODE_CALL_TIMEOUT = 30.0
CODE_MAX_CALL_TIMEOUT = 55.0
# Adaptive per-call ceiling: base + a measured per-KB slope, capped. Long
# summarization passages get more wall-clock than a one-line sentiment prompt
# WITHOUT a flat 45s that would let one task hold the serialized LOCAL_LOCK and
# starve the queue (run 19's -5). Derived from the ~5s/870-word measurement with
# generous margin; main.py re-checks the deadline budget after the lock anyway.
SECONDS_PER_KB = 8.0
MAX_CALL_TIMEOUT = 40.0
KEEP_ALIVE = "30m"  # never unload the model mid-run
NUM_PREDICT = 450  # generation cap: bounds CPU time; length-hits escalate instead
# num_ctx sizes the KV cache. A flat 8192 costs several hundred MB of the 4 GB
# grading box's RAM AND slows prompt-eval on every call — even the short
# factual/math/sentiment prompts that need a fraction of it. Under memory
# pressure that slowdown pushes local calls past their timeout → fail open to
# paid remote, which is the leading suspect for why all-local didn't shed tokens
# (docs: run 25 was a wash). Size it to the prompt instead: 2048 covers a ~6k-char
# prompt + full generation; long summarization passages get the big window only
# when they actually need it.
NUM_CTX_SMALL = 2048
NUM_CTX_LARGE = 8192
NUM_CTX_LARGE_CHARS = 5000  # prompts longer than this get the large window

# Categories allowed to try local-first. The MODULE default is the proven-safe
# rung (sentiment,ner,summarization → run 18, 100% @ 4,178); anyone importing
# local.py without the deployment env gets that behavior. The DEPLOYMENT (Dockerfile
# ENV) opts into the aggressive rung by adding debug,codegen — safe only with the
# coder model (Track B: debug 10/10, codegen 9/10 on executed assertions; plain
# qwen2.5:3b was algorithmically wrong here, sinking run 16) AND the run-19
# lock-starvation fix in main.py. factual/math/logic stay remote (kimi) — proven
# 18/19. Empty env disables local entirely (pure remote = the proven config).
_raw = os.environ.get("LOCAL_CATEGORIES", "sentiment,ner,summarization")
LOCAL_CATEGORIES = {c.strip() for c in _raw.split(",") if c.strip()}

_BASE_INSTRUCTION = (
    "Follow the task instructions exactly and answer directly. Be concise and "
    "correct; no preamble. If the task asks multiple things, answer every part. "
    "If an exact count is required (e.g. exactly N words), draft, count the units "
    "one by one, and revise until it matches; output only the final answer.\n\n"
)

# Per-category guidance for the local model. Local tokens are free, so these aim
# at judge satisfaction, not token thrift. The NER wording encodes the reference
# repo's hard-won lessons (classify by what the thing IS, keep relative dates,
# one category per entity) and is applied only to NER so the example cannot bleed
# into unrelated categories.
_CATEGORY_INSTRUCTIONS = {
    "sentiment": (
        "Start with one overall label — Positive, Negative, Neutral, or Mixed — "
        "then one short justification. Use Mixed only when clearly both-sided.\n\n"
    ),
    "ner": (
        "List every named entity as 'Label: value', one per line, using the labels "
        "the task requests (person, organization, location, date). Classify each "
        "item by what it fundamentally IS: a place is Location even after a company "
        "name; a person keeps Person even with a title; dates include relative ones "
        "like 'last week'. One label per entity; skip labels with no matches.\n\n"
    ),
    "summarization": (
        "Output only the summary. Obey any stated length or format constraint "
        "exactly; if none is stated, use at most 3 sentences.\n\n"
    ),
    "debug": (
        "State the bug in one short sentence, then give the fully corrected code "
        "in one fenced code block. No other commentary.\n\n"
    ),
    "codegen": (
        "Give the complete working code in one fenced code block, then at most one "
        "short sentence. No introduction.\n\n"
    ),
    # factual/math/logic added for the all-local rung. Wording mirrors the proven
    # remote SPEC so a local answer meets the same judge contract; math/logic MUST
    # end with an 'Answer:' line so the verifier can gate them (a factual answer
    # cannot be verified for correctness — that is the accepted risk of this rung).
    "factual": (
        "Give a correct, clear, self-contained answer in under 120 words.\n\n"
    ),
    "math": (
        "Work through it in brief steps, then end with 'Answer: <value>' on its "
        "own line.\n\n"
    ),
    "logic": (
        "Reason in brief numbered steps, checking each constraint, then end with "
        "'Answer: <value>' on its own line.\n\n"
    ),
}


class LocalError(RuntimeError):
    pass


_first_call_done = False


def is_available(timeout: float = 3.0) -> bool:
    """Fast probe used once at startup; connection-refused returns immediately."""
    try:
        req = urllib.request.Request(f"{BASE_URL}/api/version", method="GET")
        with urllib.request.urlopen(req, timeout=timeout):
            return True
    except Exception:
        return False


def _timeout_for(prompt_chars: int, first_call: bool, category: str = "") -> float:
    """Scale the per-call timeout with passage length, bounded. First call also
    absorbs the model-load cold start. debug/codegen get a higher floor/ceiling
    (see CODE_CALL_TIMEOUT) because their completions run long regardless of
    prompt length."""
    base, ceiling = (
        (CODE_CALL_TIMEOUT, CODE_MAX_CALL_TIMEOUT) if category in ("debug", "codegen")
        else (CALL_TIMEOUT, MAX_CALL_TIMEOUT)
    )
    scaled = min(base + SECONDS_PER_KB * prompt_chars / 1024.0, ceiling)
    return max(scaled, FIRST_CALL_TIMEOUT) if first_call else scaled


def generate(prompt: str, category: str) -> str:
    """Blocking Ollama call (main.py runs it in a thread, serialized — the 2 vCPU
    box effectively processes one local generation at a time anyway)."""
    global _first_call_done
    timeout = _timeout_for(len(prompt), not _first_call_done, category)
    instruction = _BASE_INSTRUCTION + _CATEGORY_INSTRUCTIONS.get(category, "")
    full_prompt = instruction + prompt
    num_ctx = NUM_CTX_LARGE if len(full_prompt) > NUM_CTX_LARGE_CHARS else NUM_CTX_SMALL
    payload = {
        "model": MODEL,
        "prompt": full_prompt,
        "stream": False,
        "keep_alive": KEEP_ALIVE,
        "options": {"temperature": 0, "num_ctx": num_ctx, "num_predict": NUM_PREDICT},
    }
    req = urllib.request.Request(
        f"{BASE_URL}/api/generate", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.load(resp)
    except Exception as error:
        raise LocalError(f"ollama call failed: {str(error)[:120]}") from error
    _first_call_done = True
    text = (body.get("response") or "").strip()
    if not text:
        raise LocalError("empty response")
    if body.get("done_reason") == "length":
        raise LocalError("truncated at num_predict (incomplete answer)")
    return text


# --- zero-token verifiers -------------------------------------------------------
# Principle: only reject what is PROVABLY wrong — over-blocking burns tokens on
# needless fallbacks; under-blocking submits silent wrong answers to the judge.

_HEDGES = (
    "i don't know", "i do not know", "cannot answer", "can't answer",
    "i'm not sure", "i am not sure", "as an ai", "insufficient information",
    "unable to determine", "cannot be determined", "does not apply",
)

_SENTIMENT_LABELS = re.compile(r"\b(positive|negative|neutral|mixed)\b", re.I)
_NER_LINE = re.compile(r"^\s*[A-Za-z][\w ]{0,20}:\s*\S", re.M)
_FENCE = re.compile(r"```[a-zA-Z0-9]*[ \t]*\n(.*?)```", re.S)
_WORD_LIMIT = re.compile(
    r"(?:exactly|in|within|under|at most|no more than|maximum of)\s+(\d+)\s+words?", re.I)
_SENTENCE_LIMIT = re.compile(r"\b(?:in\s+|exactly\s+)?(one|two|three|four|1|2|3|4)\s+sentences?\b", re.I)
_SENTENCE_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4}
_FN_NAME = re.compile(
    r"function\s+(?:called\s+|named\s+)?[`'\"]([A-Za-z_]\w*)[`'\"]|[`'\"]([A-Za-z_]\w*)\s*\(")


def _is_hedged(answer: str) -> bool:
    low = answer.lower()
    return any(h in low for h in _HEDGES)


def _is_degenerate(answer: str) -> bool:
    words = answer.split()
    if len(words) < 12:
        return False
    from collections import Counter
    top = Counter(w.lower() for w in words).most_common(1)[0][1]
    return top / len(words) > 0.4


def _extract_code(text: str) -> str:
    match = _FENCE.search(text)
    if match:
        return match.group(1).strip()
    if "```" in text:  # unclosed fence
        tail = re.search(r"```[a-zA-Z0-9]*[ \t]*\n?(.*)", text, re.S)
        if tail:
            return tail.group(1).strip()
    return ""


# Periods that do not end a sentence: dotted acronyms (U.S., e.g.), common
# abbreviations, and single-letter initials before a capitalized name. Masking
# them keeps the counter from rejecting good answers; erring lenient is the right
# direction — over-counting burns paid remote fallbacks on correct output.
_ABBREV = re.compile(
    r"\b(?:[A-Za-z]\.){2,}"
    r"|\b(?:Dr|Mr|Mrs|Ms|Prof|Rev|Gen|Sen|St|No|Fig|vs|etc|approx|Inc|Corp|Ltd|Co|Jr|Sr)\."
    r"|\b[A-Z]\.(?=\s+[A-Z][a-z])")


def _sentence_count(text: str) -> int:
    masked = _ABBREV.sub(lambda m: m.group(0).replace(".", ""), text.strip())
    parts = re.split(r"[.!?]+(?:\s|$)", masked)
    return len([p for p in parts if p.strip()])


def _code_ok(prompt: str, answer: str, is_debug: bool) -> tuple[bool, str]:
    code = _extract_code(answer)
    if not code:
        return False, "no fenced code block"
    looks_python = bool(re.search(r"\bpython\b", prompt, re.I)) or bool(
        re.search(r"^\s*(def |import |from \w+ import|class \w)", code, re.M))
    if looks_python:
        try:
            ast.parse(code)
        except Exception:
            return False, "code does not parse"
    fn = _FN_NAME.search(prompt)
    if fn:
        name = fn.group(1) or fn.group(2)
        if name and name not in code:
            return False, f"requested function '{name}' missing"
    if is_debug:
        original = _extract_code(prompt)
        if original and code.strip() == original.strip():
            return False, "returned the buggy code unchanged"
    return True, "ok"


def verify(prompt: str, category: str, answer: str) -> tuple[bool, str]:
    """(ok, reason). ok=False -> main.py escalates to the Fireworks path."""
    text = (answer or "").strip()
    if not text:
        return False, "empty"
    if _is_hedged(text):
        return False, "hedged non-answer"
    if _is_degenerate(text):
        return False, "degenerate repetition"

    if category == "sentiment":
        if not _SENTIMENT_LABELS.search(text):
            return False, "no sentiment label"
    elif category == "ner":
        if not _NER_LINE.search(text):
            return False, "no 'Label: value' lines"
    elif category == "summarization":
        limit = _WORD_LIMIT.search(prompt)
        if limit and len(text.split()) > int(limit.group(1)):
            return False, f"exceeds {limit.group(1)}-word limit"
        sentences = _SENTENCE_LIMIT.search(prompt)
        if sentences:
            allowed = _SENTENCE_WORDS.get(sentences.group(1).lower()) or int(sentences.group(1))
            if _sentence_count(text) > allowed:
                return False, f"exceeds {allowed}-sentence limit"
        # This is the first run showing the judge REAL qwen2.5:3b summaries (run 18
        # scored while summarization mostly fell back to remote). Two conservative
        # provable-non-summary checks, both erring lenient:
        prompt_words = prompt.split()
        answer_words = text.split()
        # (1) length ratio: a "summary" longer than 0.8x a >80-word source is not one.
        if len(prompt_words) > 80 and len(answer_words) > 0.8 * len(prompt_words):
            return False, "answer nearly as long as source (not a summary)"
        # (2) echo: the answer opening copied verbatim from the source is not a
        # summary. Compare on alphanumerics only so punctuation differences
        # ("division." vs "division this") don't hide a real echo.
        if len(answer_words) >= 15:
            def _norm(s):
                return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()
            opening = _norm(" ".join(answer_words[:15]))
            if opening and opening in _norm(prompt):
                return False, "answer echoes the source verbatim"
    elif category in ("math", "logic"):
        # Can't verify correctness, but CAN require the instructed contract: an
        # 'Answer:' line carrying a real value. A local answer missing it (the 3B
        # rambled or ran out) fails open to the remote tier rather than shipping a
        # half-answer. Pure-arithmetic math never reaches here — main.py's
        # deterministic solver answers it for zero tokens first.
        answer_line = re.search(r"(?im)^\s*answer\s*:\s*(\S.*)$", text)
        if not answer_line:
            return False, "no 'Answer:' line"
        if not re.search(r"[0-9A-Za-z]", answer_line.group(1)):
            return False, "empty 'Answer:' value"
    elif category in ("debug", "codegen"):
        ok, reason = _code_ok(prompt, text, is_debug=(category == "debug"))
        if not ok:
            return False, reason

    return True, "ok"
