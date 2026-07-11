"""Pack multiple tasks into one LLM call (yassai-style batching, Phase E).

Pure functions only — no network, no imports from client.py/main.py. Validated
offline against a real model on the public Fireworks endpoint before this file
was written (scripts/probe_batch_format.py): a real model reliably follows the
distinct-marker format below and does not bleed one task's instruction/content
into another's answer, PROVIDED reasoning_effort="none" is sent on the call
(src/client.py already sends this on every call, so main.py gets it for free).

Design: only the pending, remote-eligible leftover after the existing
zero-token paths (deterministic solver, local-first) gets batched — those paths
in main.py/local.py are completely untouched by this module. Any task whose
batch slice is missing, truncated, or fails verification must fall back to a
solo call in main.py; this module only ever reports what it found, it never
guesses.
"""
import re

from . import local, prompts

# Bin-packing caps per tier: whichever bound (task count or combined max_tokens)
# is hit first closes the current chunk. Provisional starting points from the
# real-endpoint probe (both the 3-task CODE batch and 3-task SMALL batch stayed
# well under the ~25s per-request ceiling) — revisit if real batches run bigger.
CHUNK_LIMITS = {
    "CODE": {"max_tasks": 3, "max_tokens": 1300},
    "SMALL": {"max_tasks": 4, "max_tokens": 900},
}
_DEFAULT_CHUNK_LIMITS = {"max_tasks": 2, "max_tokens": 800}

# Per-task overhead of the @@TASK:id@@/@@ENDTASK:id@@/INSTRUCTION/PROMPT wrapper
# text itself, on top of the category's own instruction+prompt tokens.
MARKER_OVERHEAD_TOKENS = 40

_ANSWER_LINE_RE = re.compile(r"(?im)^\s*answer\s*:\s*\S")


def group_and_chunk(pending: list[dict]) -> list[list[dict]]:
    """pending items: {"task_id", "category", "prompt"}. Groups by
    prompts.SPEC[category]["tier"], then greedily bin-packs each tier group in
    input order, bounded by that tier's max task count and combined max_tokens.
    Every task_id from `pending` appears in exactly one output chunk. A chunk
    of length 1 is a valid result — callers route those straight to the solo
    path rather than wrapping a single task in the batch format."""
    by_tier: dict[str, list[dict]] = {}
    for item in pending:
        tier = prompts.SPEC[item["category"]]["tier"]
        by_tier.setdefault(tier, []).append(item)

    chunks: list[list[dict]] = []
    for tier, items in by_tier.items():
        limits = CHUNK_LIMITS.get(tier, _DEFAULT_CHUNK_LIMITS)
        current: list[dict] = []
        current_tokens = 0
        for item in items:
            item_tokens = prompts.SPEC[item["category"]]["max_tokens"] + MARKER_OVERHEAD_TOKENS
            if current and (len(current) + 1 > limits["max_tasks"]
                             or current_tokens + item_tokens > limits["max_tokens"]):
                chunks.append(current)
                current, current_tokens = [], 0
            current.append(item)
            current_tokens += item_tokens
        if current:
            chunks.append(current)
    return chunks


_SYSTEM_TEMPLATE = (
    "You will receive {n} independent tasks. Each is delimited by "
    "@@TASK:<id>@@ ... @@ENDTASK:<id>@@ and carries its own INSTRUCTION line. "
    "Follow ONLY that task's instruction for that task's answer — never let "
    "one task's wording, content, or format influence another task's answer.\n\n"
    "Answer every task. For each one, output exactly:\n"
    "@@ANSWER:<id>@@\n<answer, following that task's instruction>\n@@ENDANSWER:<id>@@\n\n"
    "Reuse the exact <id> token shown for each task. Do not skip a task. "
    "Do not add any text outside these blocks."
)


def build_batch_messages(chunk: list[dict]) -> tuple[list[dict], int]:
    """chunk has 2+ items, all the same tier (caller enforces this — a size-1
    chunk should go through the solo path instead). Returns (messages,
    max_tokens) ready for client.complete()."""
    system = _SYSTEM_TEMPLATE.format(n=len(chunk))
    blocks = [
        f"@@TASK:{item['task_id']}@@\n"
        f"INSTRUCTION: {prompts.SPEC[item['category']]['instruction']}\n"
        f"PROMPT: {item['prompt']}\n"
        f"@@ENDTASK:{item['task_id']}@@"
        for item in chunk
    ]
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n\n".join(blocks)},
    ]
    max_tokens = (sum(prompts.SPEC[item["category"]]["max_tokens"] for item in chunk)
                  + MARKER_OVERHEAD_TOKENS * len(chunk))
    return messages, max_tokens


def parse_batch_response(raw_text: str, task_ids: list[str]) -> dict[str, dict]:
    """Return {task_id: {"text": str, "complete": bool}} for every task_id
    given — every id is always a key, even if nothing was found (text="",
    complete=False). "complete" is True only when BOTH the ANSWER and matching
    ENDANSWER markers for that id were found; a slice recovered by falling
    through to the next marker or end-of-string (truncation, or a dropped
    ENDANSWER) is "complete"=False and must never be trusted by callers,
    regardless of how well-formed the partial text looks."""
    text = raw_text or ""
    result: dict[str, dict] = {}
    for task_id in task_ids:
        escaped = re.escape(task_id)
        strict = re.search(rf"@@ANSWER:{escaped}@@\s*\n?(.*?)@@ENDANSWER:{escaped}@@", text, re.S)
        if strict:
            result[task_id] = {"text": strict.group(1).strip(), "complete": True}
            continue
        loose = re.search(rf"@@ANSWER:{escaped}@@\s*\n?(.*?)(?=@@ANSWER:|\Z)", text, re.S)
        result[task_id] = {
            "text": loose.group(1).strip() if loose else "",
            "complete": False,
        }
    return result


def verify_slice(prompt: str, category: str, text: str) -> tuple[bool, str]:
    """(ok, reason). Reuses local.verify's existing hedge/degenerate check plus
    its real category branches for sentiment/ner/summarization/debug/codegen
    unmodified. math/logic have no branch there (never part of
    LOCAL_CATEGORIES) so batch slices additionally require an 'Answer: <value>'
    line, matching prompts.SPEC's own instructed contract for those two
    categories — stricter than today's unverified solo path, deliberately."""
    ok, reason = local.verify(prompt, category, text)
    if not ok:
        return ok, reason
    if category in ("math", "logic") and not _ANSWER_LINE_RE.search(text):
        return False, "missing 'Answer:' line"
    return True, "ok"
