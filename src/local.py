"""Local inference sidecar (Ollama + qwen2.5-coder:3b) and its zero-token answer verifiers.

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
# qwen2.5-coder:3b replaced the plain sibling after offline evals (2026-07-11):
# debug 10/10 and codegen 9/10 semantically correct (run 16's killers), at parity
# with qwen2.5:3b on sentiment/ner/summarization. One model only — 4GB RAM cannot
# hold two 3B models, and a swap costs 10-20s per switch on the grading box.
MODEL = os.environ.get("LOCAL_MODEL", "qwen2.5-coder:3b")
# First call pays the model load from disk (~2GB into RAM); later calls don't.
# Measured 2026-07-11 (2-core pin): a loaded-under-contention first call took 57s
# while the same prompt warm took 3.4s — 25s loses the race against the
# entrypoint's background warmup and burns one remote spend per run.
FIRST_CALL_TIMEOUT = 60.0
CALL_TIMEOUT = 15.0
# Measured (2-core pin, full RAM): even 870-word passages answer in ~5s warm, so
# 15s is enough for compute alone. The raised cap is insurance for the grading
# box's 4GB RAM (KV-cache pressure can stall long prompts there — not testable
# natively); it costs nothing when answers land in 5s, and the worst case is
# bounded by main.py's LOCAL_MIN_REMAINING_SECONDS budget guard.
SUMMARIZATION_CALL_TIMEOUT = 45.0
NUM_PREDICT = 450  # generation cap: bounds CPU time; length-hits escalate instead

# Categories allowed to try local-first. Run 16 (all five local on PLAIN
# qwen2.5:3b) scored 15/19 — code categories were the poison. Run 17
# (sentiment,ner) and run 18 (+summarization, 19/19) rebuilt trust one rung at
# a time. This rung re-adds debug+codegen on the CODER model, which passed the
# offline semantic eval (generated code executed against assertions) that the
# plain model failed. factual/math/logic stay remote (kimi) — proven 18/19.
# Empty env disables local entirely.
_raw = os.environ.get("LOCAL_CATEGORIES", "sentiment,ner,summarization,debug,codegen")
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


def generate(prompt: str, category: str) -> str:
    """Blocking Ollama call (main.py runs it in a thread, serialized — the 2 vCPU
    box effectively processes one local generation at a time anyway)."""
    global _first_call_done
    timeout = SUMMARIZATION_CALL_TIMEOUT if category == "summarization" else CALL_TIMEOUT
    if not _first_call_done:
        timeout = max(timeout, FIRST_CALL_TIMEOUT)
    instruction = _BASE_INSTRUCTION + _CATEGORY_INSTRUCTIONS.get(category, "")
    payload = {
        "model": MODEL,
        "prompt": instruction + prompt,
        "stream": False,
        "options": {"temperature": 0, "num_ctx": 8192, "num_predict": NUM_PREDICT},
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
# them keeps the counter from rejecting good answers; erring lenient is the
# right direction — over-counting burns paid remote fallbacks on correct output.
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
    elif category in ("debug", "codegen"):
        ok, reason = _code_ok(prompt, text, is_debug=(category == "debug"))
        if not ok:
            return False, reason

    return True, "ok"
