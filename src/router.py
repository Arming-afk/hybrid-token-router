"""Classify tasks into the 8 categories: cheap regexes first, LLM fallback only when ambiguous."""
import re

# Categories whose prompts name the operation explicitly -- checked first, in this order
# (a sentiment task may say "in one sentence", so sentiment must beat summarization).
R = {
    "sentiment": re.compile(
        r"\bsentiment\b"
        r"|\b(classify|label|categori[sz]e)\b.{0,60}\b(positive|negative|neutral)\b"
        r"|\bpositive\b.{0,40}\bnegative\b",
        re.I,
    ),
    "summarization": re.compile(
        r"\bsummari[sz]e\b|\bsummary\b|\btl;?dr\b|\bcondense\b|\bshorten\b"
        r"|\bin (one|two|three|\d+) (sentence|word|bullet)",
        re.I,
    ),
    "ner": re.compile(
        r"named entit|\bNER\b|\bentit(y|ies)\b"
        r"|\b(extract|identify|find|list|tag)\b.{0,80}"
        r"\b(person|people|organi[sz]ation|compan(y|ies)|location|date)s?\b",
        re.I,
    ),
}

# Code tokens are case-sensitive on purpose: prose "Function" at sentence start is rare,
# and lowercase keywords are a strong signal of actual code.
CODE = re.compile(r"```|\bdef |\bfunction\b|\bclass \w|=>|\breturn\b|print\(|console\.log")
DEBUG = re.compile(
    r"\b(bug|bugs|buggy|fix|debug|broken|incorrect|wrong|crash(es)?|fails?|error)\b"
    r"|doesn'?t work|not working",
    re.I,
)
WRITE = re.compile(r"\b(write|implement|create|build|generate|develop)\b", re.I)
MATH = re.compile(
    r"\bcalculate\b|\bhow (many|much)\b|\bpercent(age)?\b|%|\btotal\b|\baverage of\b"
    r"|\bcompound\b|\bprofit\b|\binterest\b|\bproject(ed|ion)s?\b|\bsum\b|\bprice\b|\bcost\b",
    re.I,
)
LOGIC = re.compile(
    r"\b(lying|liar|truth|knight|knave)\b"
    r"|\bwho (is|are|was|has|sits|owns|gets|likes|finished|came|wins|won)\b"
    r"|seat(ed|ing)|arrangement|\bclues?\b|\bconstraints?\b|\bdeduce\b"
    r"|\beach\b.{0,30}\b(different|exactly one)\b|\b(all|some|no) \w+ are\b",
    re.I,
)


def classify(prompt: str) -> str | None:
    """Return a category name, or None when ambiguous (caller should use the LLM fallback)."""
    for category, rx in R.items():
        if rx.search(prompt):
            return category
    if CODE.search(prompt):
        if DEBUG.search(prompt):
            return "debug"
        if WRITE.search(prompt):
            return "codegen"
        return None  # code present but intent unclear
    has_math = bool(MATH.search(prompt))
    has_logic = bool(LOGIC.search(prompt))
    if has_math and has_logic:
        return None
    if has_math:
        return "math"
    if has_logic:
        return "logic"
    return "factual"


LETTER_TO_CATEGORY = {
    "A": "factual",
    "B": "math",
    "C": "sentiment",
    "D": "summarization",
    "E": "ner",
    "F": "debug",
    "G": "logic",
    "H": "codegen",
}


def fallback_messages(prompt: str) -> list[dict]:
    return [{
        "role": "user",
        "content": (
            "Classify into one letter: A factual B math C sentiment D summarize "
            "E ner F debug G logic H codegen.\n"
            f"Task: {prompt[:400]}\nLetter:"
        ),
    }]


def parse_fallback_letter(text: str) -> str:
    for char in text.strip().upper():
        if char in LETTER_TO_CATEGORY:
            return LETTER_TO_CATEGORY[char]
    return "factual"
