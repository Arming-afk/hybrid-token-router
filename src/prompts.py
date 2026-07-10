"""Per-category prompt templates, output caps, and model tiers.

This is the main tuning surface: every token in an instruction must earn its place,
and every tier bump must be justified by a failed eval at the cheaper tier.

Current values descend from the gate-PASSING config of the 2nd-place Track 1 entry
(KaananeTaha/AMD-AI-Hackathon, analyzed 2026-07-09; full rationale in
docs/eval-results.md), reshaped by the stage-2 cuts below: today NOTHING sits on
LARGE (minimax-m3 bills a ~100/call hidden prompt tax) — factual/math/logic and
debug/codegen all run on the CODE tier (kimi-k2p7-code; MEDIUM's gemma-31b has no
serverless deployment and 404s at grading), sentiment/summarization/ner on SMALL.
factual remains the router's misroute default; since the 2026-07-10 router
hardening (dev set 187/187) that insurance is carried by routing quality rather
than by the biggest model. LARGE is still used by main.py's empty-completion and
truncation escalation retry.

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
  Gate held at 5085 tokens (−175).
- Cut #4 tried factual "under 60 words" and scored **73.7% — the same 14/19 as the
  "1-2 sentences" failure** — reverted. Factual's accuracy cliff sits between 60 and
  120 words; "under 120 words" is the proven floor. Do NOT squeeze factual's output
  again; the remaining token fat is structural (the minimax prompt tax — a Phase C
  tier move), not instructional.
- Phase C move #1: math+logic LARGE -> MEDIUM, instructions and caps unchanged (the
  tier is the single variable). Sheds the ~100/call minimax prompt tax on those
  tasks (see the token-cost probe); predicted −400–600 from 5085. Chosen over
  factual->MEDIUM because factual has failed the gate twice on completeness and is
  the router's misroute default, while math/logic are judged on a verifiable
  'Answer: <value>' line. MEDIUM (gemma-4-31b-it) is unreachable with the public
  key (404, re-probed 2026-07-10), so this is a blind experiment: if the gate
  falls, revert and re-submit the cut #3 anchor image (86cf241) immediately.
- Moonshot (2026-07-10, after run 13's 94.7%/5095 and rank 1 posting 2600/84%):
  strategy shift — ranking is by tokens among gate-passers, so accuracy above the
  gate is wasted spend; run 13's +2-task headroom (bought by the router hardening)
  is deliberately converted into token cuts, several at once. Bundle: factual ->
  MEDIUM at "under 50 words" (kills the minimax tax on the biggest category AND
  halves its output; the old 60-word failures predate the router fix, whose
  misroutes were likely the real casualties), sentiment -> label only, debug ->
  corrected code only (accepting the judge-intent risk previously dodged),
  summarization defaults to <=3 sentences when the task states no length. math/
  logic keep their 2 short steps ON PURPOSE: the steps double as chain-of-thought
  for the MEDIUM model — answer-only would save ~150 tokens and risk math
  correctness. Caps are now hard budget enforcers (billed = generated, so a tight
  cap bounds worst-case spend), sized ~1.5-2x the instructed length so an obedient
  answer never truncates. Rollback anchor: image 94619a7 (94.7%, 5095).
- Moonshot fix (resubmission after run 14's INFRA_ERROR): factual/math/logic tier
  MEDIUM -> CODE (kimi). Community-confirmed intel (docs/eval-results.md, LocalFirst
  section): gemma-4-31b-it has NO serverless support and 404s at grading too, so
  "MEDIUM" was silently escalating every call to LARGE minimax via the blank-answer
  retry — still paying the prompt tax this bundle meant to kill, plus a wasted
  roundtrip. kimi is proven alive in every scored run (debug/codegen ran on it),
  has no prompt tax, and the 100%-at-3753 competitor answered every remote category
  on it. Also new: pure-arithmetic prompts short-circuit to a deterministic
  in-process solver (0 tokens, router.deterministic_math_answer), and truncated
  answers (finish_reason=length) escalate instead of being submitted.
"""

_BASE = "English. Terse; no preamble."

SPEC = {
    "factual": {
        # CODE (kimi): no minimax prompt tax, and — unlike MEDIUM's gemma-31b, which
        # has no serverless deployment and 404s at grading — actually reachable.
        # Misroute-default duty rides on routing quality (hardened router, 187/187).
        "tier": "CODE",
        # Cap ~1.5x the instructed 50 words so an obedient answer never truncates.
        "max_tokens": 120,
        "instruction": f"{_BASE} Answer clearly in under 50 words.",
    },
    "math": {
        # The 2 short steps stay: they double as chain-of-thought; answer-only
        # would save ~150 tokens/run at real correctness risk. Pure-arithmetic
        # prompts never get here — main.py short-circuits them to the free
        # deterministic solver first.
        "tier": "CODE",
        "max_tokens": 150,
        "instruction": f"{_BASE} At most 2 short steps, then 'Answer: <value>' on its own line.",
    },
    "sentiment": {
        "tier": "SMALL",
        "max_tokens": 30,
        "instruction": (
            f"{_BASE} One word: positive, negative, or neutral."
        ),
    },
    "summarization": {
        "tier": "SMALL",
        "max_tokens": 220,
        "instruction": (
            f"{_BASE} Output only the summary; obey any stated length/format "
            f"constraint, else at most 3 sentences."
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
        # Bug-name sentence dropped in the moonshot: the judge's category description
        # says "identify AND correct", so this knowingly spends gate headroom.
        "tier": "CODE",
        "max_tokens": 450,
        "instruction": (
            f"{_BASE} Only the corrected code, in one fenced block. No comments."
        ),
    },
    "logic": {
        # Same shape as math: steps kept as chain-of-thought; CODE for the same
        # reachability reason (the LocalFirst competitor routed seating puzzles to
        # kimi specifically because it solves them reliably).
        "tier": "CODE",
        "max_tokens": 150,
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
