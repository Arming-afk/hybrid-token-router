"""Per-category prompt templates, output caps, and model tiers.

This is the main tuning surface: every token in an instruction must earn its place,
and every tier bump must be justified by a failed eval at the cheaper tier.

Current values mirror the gate-PASSING config of the 2nd-place Track 1 entry
(KaananeTaha/AMD-AI-Hackathon, analyzed 2026-07-09; full rationale in
docs/eval-results.md): factual/math/logic on LARGE (minimax-m3, made safe by
client.py's reasoning_effort="none"), debug/codegen on the CODE tier
(kimi-k2p7-code), sentiment/summarization/ner on SMALL. factual doubles as the
router's misroute default, so it deliberately sits on the strongest tier — any
false positive lands on the most capable model instead of failing the gate.
Their measured cost on the real harness models: ~150 tokens/task, all correct.

Stage 2 (after the 84.2%/5273-token gate-passing run, image 6f01e64), one cut per
submission so each result cleanly measures one variable:
- Cut #1 trimmed instruction filler only; it saved 13 tokens (5273 -> 5260) against
  an input-side prediction of ~140, proving the counted tokens are overwhelmingly
  on the OUTPUT side. Wording micro-trims are a dead end but harmless, so it stays.
- Cut #2 shrank factual's output budget ("under 120 words" -> "1-2 sentences") and
  scored **73.7% (14/19), a real regression from 84.2% (16/19)** — reverted. Factual
  has the highest task share and is the router's misroute default on LARGE, so it
  carries an outsized share of the accuracy gate; "1-2 sentences" was too terse for
  explanatory factual prompts ("explain how X works") to satisfy the judge. Back to
  the proven "under 120 words". See docs/eval-results.md for the run history.
- Cut #3 (probe-backed, see the token-cost probe in docs/eval-results.md): math and
  logic capped at "2 short steps" — the public-endpoint probe showed minimax-m3 with
  reasoning_effort="none" answers correctly with even zero visible steps, so verbose
  step-by-step output was pure token cost; codegen additionally forbids comments.
  Debug is deliberately untouched: the judge's category description is "identifying
  bugs AND providing corrected implementations", so the bug sentence stays.
"""

_BASE = "English. Terse; no preamble."

SPEC = {
    "factual": {
        "tier": "LARGE",
        "max_tokens": 300,
        "instruction": f"{_BASE} Answer clearly in under 120 words.",
    },
    "math": {
        "tier": "LARGE",
        # Cap unchanged on purpose: it only bounds a disobedient long answer (billed,
        # not truncated); the token cut comes from the instruction, the single
        # variable this submission measures.
        "max_tokens": 400,
        "instruction": f"{_BASE} At most 2 short steps, then 'Answer: <value>' on its own line.",
    },
    "sentiment": {
        "tier": "SMALL",
        "max_tokens": 120,
        "instruction": (
            f"{_BASE} Label positive, negative, or neutral, then justify in one line."
        ),
    },
    "summarization": {
        "tier": "SMALL",
        "max_tokens": 220,
        "instruction": (
            f"{_BASE} Output only the summary; obey any stated length/format constraint."
        ),
    },
    "ner": {
        "tier": "SMALL",
        "max_tokens": 260,
        "instruction": (
            f"{_BASE} One entity per line as 'label: value'; labels: person, "
            f"organization, location, date."
        ),
    },
    "debug": {
        "tier": "CODE",
        "max_tokens": 520,
        "instruction": (
            f"{_BASE} Name the bug in one sentence, then the corrected code in one "
            f"fenced block."
        ),
    },
    "logic": {
        "tier": "LARGE",
        "max_tokens": 420,
        "instruction": (
            f"{_BASE} At most 2 short steps, then 'Answer: <value>' on its own line."
        ),
    },
    "codegen": {
        "tier": "CODE",
        "max_tokens": 520,
        "instruction": (
            f"{_BASE} Only the code, in one fenced block, correct and self-contained. "
            f"No comments."
        ),
    },
}

CATEGORIES = list(SPEC)


def render(category: str, prompt: str) -> tuple[list[dict], int, str]:
    # Instruction as a system message, task as the user message — the exact message
    # shape the gate-passing reference ran through this proxy and these models.
    spec = SPEC[category]
    messages = [
        {"role": "system", "content": spec["instruction"]},
        {"role": "user", "content": prompt},
    ]
    return messages, spec["max_tokens"], spec["tier"]


def postprocess(category: str, text: str) -> str:
    # The judge scores the full answer as-is: the reference entry passed the gate
    # handing over "brief steps + Answer: <value>" untouched, so the old stripping
    # to the text after "Answer:" only added risk. Kept as the single seam where
    # any future output rewriting would go (category is part of that interface).
    return text.strip()
