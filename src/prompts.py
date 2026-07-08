"""Per-category prompt templates, output caps, and model tiers.

This is the main tuning surface: every token in an instruction must earn its place,
and every tier bump must be justified by a failed eval at the cheaper tier.
"""

SPEC = {
    "factual": {
        # TODO(person2): eval SMALL vs MEDIUM on real gemma — dropping to SMALL is the
        # biggest token saving but a real accuracy risk. Kept at MEDIUM until measured.
        "tier": "MEDIUM",
        "max_tokens": 200,
        "instruction": "Answer accurately in at most 3 sentences.",
    },
    "math": {
        "tier": "MEDIUM",
        "max_tokens": 300,
        "instruction": (
            'Show at most 3 brief calculation steps, then end with exactly: "Answer: <value>"'
        ),
    },
    "sentiment": {
        "tier": "SMALL",
        "max_tokens": 60,
        "instruction": 'Reply: "<label> - <one short sentence of justification>"',
    },
    "summarization": {
        "tier": "SMALL",
        "max_tokens": 250,
        "instruction": (
            "Follow the stated length/format constraint exactly. Output only the summary."
        ),
    },
    "ner": {
        "tier": "SMALL",
        "max_tokens": 250,
        "instruction": (
            "List one entity per line as: <entity> | <PERSON|ORG|LOCATION|DATE>\n"
            "No other text."
        ),
    },
    "debug": {
        "tier": "MEDIUM",
        "max_tokens": 600,
        "instruction": (
            "State the bug in one sentence, then give the full corrected code in one "
            "code block. No other explanation."
        ),
    },
    "logic": {
        "tier": "MEDIUM",
        "max_tokens": 350,
        "instruction": (
            "Work through the constraints in at most 5 short lines, then end with "
            'exactly: "Answer: <solution>"'
        ),
    },
    "codegen": {
        "tier": "MEDIUM",
        "max_tokens": 600,
        "instruction": "Output only the code in one code block. No explanation before or after.",
    },
}

CATEGORIES = list(SPEC)


def render(category: str, prompt: str) -> tuple[list[dict], int, str]:
    spec = SPEC[category]
    messages = [{"role": "user", "content": f"{prompt}\n\n{spec['instruction']}"}]
    return messages, spec["max_tokens"], spec["tier"]


def postprocess(category: str, text: str) -> str:
    # math/logic answers end with "Answer: X"; hand the judge just the final answer.
    if category in ("math", "logic"):
        marker = text.rfind("Answer:")
        if marker != -1:
            answer = text[marker + len("Answer:"):].strip()
            if answer:
                return answer
    return text.strip()
