"""Per-category prompt templates, output caps, and model tiers.

This is the main tuning surface: every token in an instruction must earn its place,
and every tier bump must be justified by a failed eval at the cheaper tier.

Current values are the GATE-FIRST config after the first scored run failed the
accuracy gate at 47.4% (2026-07-08): every category runs on MEDIUM (the largest
non-reasoning model) with generous caps, because truncated or weak answers fail
the judge outright and rank-by-tokens only exists after the gate is passed.
Once a submission passes, walk categories back down (SMALL, tighter caps) one
resubmission at a time and keep the cheapest config that still passes.
"""
import re

SPEC = {
    "factual": {
        # TODO(person2): eval SMALL vs MEDIUM on real gemma — dropping to SMALL is the
        # biggest token saving but a real accuracy risk. Kept at MEDIUM until measured.
        "tier": "MEDIUM",
        "max_tokens": 250,
        "instruction": "Answer accurately in at most 3 sentences.",
    },
    "math": {
        "tier": "MEDIUM",
        "max_tokens": 450,
        "instruction": (
            'Show at most 3 brief calculation steps, then end with exactly: "Answer: <value>"'
        ),
    },
    "sentiment": {
        "tier": "MEDIUM",
        "max_tokens": 120,
        "instruction": 'Reply: "<label> - <one short sentence of justification>"',
    },
    "summarization": {
        "tier": "MEDIUM",
        "max_tokens": 300,
        "instruction": (
            "Follow the stated length/format constraint exactly. Output only the summary."
        ),
    },
    "ner": {
        "tier": "MEDIUM",
        "max_tokens": 300,
        "instruction": (
            "List one entity per line as: <entity> | <PERSON|ORG|LOCATION|DATE>\n"
            "No other text."
        ),
    },
    "debug": {
        "tier": "MEDIUM",
        "max_tokens": 800,
        "instruction": (
            "State the bug in one sentence, then give the full corrected code in one "
            "code block. No other explanation."
        ),
    },
    "logic": {
        "tier": "MEDIUM",
        "max_tokens": 500,
        "instruction": (
            "Work through the constraints in at most 5 short lines, then end with "
            'exactly: "Answer: <solution>"'
        ),
    },
    "codegen": {
        "tier": "MEDIUM",
        "max_tokens": 800,
        "instruction": "Output only the code in one code block. No explanation before or after.",
    },
}

CATEGORIES = list(SPEC)


def render(category: str, prompt: str) -> tuple[list[dict], int, str]:
    spec = SPEC[category]
    messages = [{"role": "user", "content": f"{prompt}\n\n{spec['instruction']}"}]
    return messages, spec["max_tokens"], spec["tier"]


_ANSWER_MARKER = re.compile(r"answer\s*:", re.I)


def postprocess(category: str, text: str) -> str:
    # math/logic answers end with "Answer: X"; hand the judge just the final answer.
    # Matched case-insensitively since models don't always follow the exact casing
    # asked for in the instruction.
    if category in ("math", "logic"):
        matches = list(_ANSWER_MARKER.finditer(text))
        if matches:
            answer = text[matches[-1].end():].strip()
            if answer:
                return answer
    return text.strip()
