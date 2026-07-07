"""Classify tasks into the 8 categories: cheap regexes first, LLM fallback only when ambiguous.

Design notes (each rule maps to a real misroute the dev set exercises):
- Explicit-operation categories (sentiment -> summarization -> ner) are checked first,
  because their instruction word is unambiguous even when other keywords co-occur.
- Code is detected before math/logic. Debug vs codegen is decided by intent: a WRITE
  verb ("write/implement/create") means new code (codegen) unless the prompt also
  describes existing broken code (STRONG_FIX), in which case it is a repair (debug).
- Math requires BOTH a digit and a math signal, so factual questions that merely use
  "how much/many" or names a math concept ("compound interest") stay factual.
- Logic keys off puzzle structure (clues, seating, syllogisms, rankings), NOT a bare
  "who is", which is overwhelmingly a factual lead-in ("Who is the CEO of Apple?").
"""
import re

# --- explicitly-named operations, checked in this order -----------------------
R = {
    "sentiment": re.compile(
        r"\bsentiment\b"
        r"|\b(classify|label|categori[sz]e|determine|identify|rate|analy[sz]e)\b"
        r".{0,40}\b(positive|negative|neutral)\b"
        r"|\bpositive[,/ ]+negative[,/ ]+(or +)?neutral\b"
        r"|\bpositive or negative\b"
        r"|\bhow (do|does|would) (you|people|customers|users|they|the (customer|reviewer|user))"
        r" feel\b"
        r"|\bwhat do people think\b"
        r"|\bsatisfied or dissatisfied\b"
        r"|\bhappy or unhappy\b",
        re.I,
    ),
    "summarization": re.compile(
        r"\bsummari[sz]e\b|\bsummary\b|\btl;?dr\b|\bcondense\b|\bshorten\b|\bcompress\b"
        r"|\brecap\b|\bgist\b|\bboil\b.{0,30}\bdown to\b"
        r"|\bin (one|two|three|four|five|\d+) (sentence|word|bullet|line)"
        r"|\b(one|single|1)[ -]?(sentence|line)\b",
        re.I,
    ),
    "ner": re.compile(
        r"named entit|\bNER\b"
        r"|\b(extract|identify|find|list|tag|label|pull out|pick out)\b.{0,80}"
        r"\b(persons?|people|organi[sz]ations?|\borgs?\b|compan(y|ies)|locations?|places?"
        r"|cit(y|ies)|countr(y|ies)|dates?|entit(y|ies))\b"
        r"|\bwho (is|are) (mentioned|named|listed|referenced)\b"
        r"|\b(which|what)\b.{0,60}\b(mentioned|referenced|named|listed|appear)\b",
        re.I,
    ),
}

# --- code detection -----------------------------------------------------------
# Strong signals are almost certainly literal code. Weak signals ("function",
# "return", a language name) only count as code when paired with a write/fix intent,
# so "Explain what a function is" stays factual.
STRONG_CODE = re.compile(
    r"```|\bdef \w|=>|print\(|console\.log|System\.out|#include|\bimport \w"
    r"|\)\s*\{|\bfor\b[^.\n]{0,40}:\s|\bwhile\b[^.\n]{0,30}:|\bSELECT\b[^.\n]{0,80}\bFROM\b"
)
WEAK_CODE = re.compile(
    r"\bfunction\b|\breturn\b|\bclass \w|\bmethod\b|\bvariable\b|\bloop\b|\bregex\b"
    r"|\bquery\b|\bscript\b|\balgorithm\b|\bsort\b|\bdecorator\b|\bSQL\b"
    r"|\berror handling\b|\bexception\b|\bhandler\b|\bendpoint\b|\bparser\b|\bAPI\b"
    r"|\bin (python|java|javascript|typescript|c\+\+|go|rust|ruby)\b",
    re.I,
)
WRITE = re.compile(r"\b(write|implement|create|build|generate|develop|complete)\b", re.I)
# STRONG_FIX names existing broken code and beats a co-occurring WRITE verb.
STRONG_FIX = re.compile(
    r"\bfix\b|\bdebug\b|\bbugs?\b|\bbuggy\b|\bbroken\b|off-by-one"
    r"|doesn'?t work|not working|never (terminates|returns|works|ends)"
    r"|should\b.{0,50}\bbut\b|why (does|do|is|are|doesn'?t)",
    re.I,
)
# FIX is the wider repair vocabulary, only consulted once no WRITE verb is present.
FIX = re.compile(
    STRONG_FIX.pattern + r"|\bcorrect(ed)?\b|\bincorrect\b|\bcrash(es|ed)?\b|\bfails?\b|\berror\b",
    re.I,
)

# --- math ---------------------------------------------------------------------
DIGIT = re.compile(r"\d")
MATH_SIGNAL = re.compile(
    r"\bcalculate\b|\bcompute\b|\bconvert\b|\bhow (many|much|old|far|fast|long)\b"
    r"|\bpercent(age)?\b|%|\baverage of\b|\bsum of\b|\btotal\b|\bcompound\b|\binterest\b"
    r"|\bdiscount|\bprofit\b|\bsales tax\b|\bper (hour|day|week|month|year|item|unit|person)\b"
    r"|\bspeed\b|\bratio\b|\bproportion\b|\btimes\b|\bdivided\b|\bmultipl|\bplus\b|\bminus\b"
    r"|\barea\b|\bperimeter\b|\bvolume\b|[+×÷]",
    re.I,
)

# --- logic --------------------------------------------------------------------
STRONG_LOGIC = re.compile(
    r"\b(lying|liar|knight|knave)\b|truth[- ]?tell|\btruth\b|\bclues?\b|\bconstraints?\b"
    r"|\bdeduce\b|\bseat(ed|ing|s)?\b|\barrangement\b|\beach\b.{0,40}\bdifferent\b"
    r"|\blogic puzzle\b",
    re.I,
)
WEAK_LOGIC = re.compile(
    r"\brow of\b|\bnext to\b|\badjacent\b|\b(directly |immediately )?(left|right) of\b"
    r"|\bfinish(es|ed)?\b.{0,20}\b(before|after|first|last)\b"
    r"|\bwho (finished|came|won|wins|placed|ranked|owns|is next)\b"
    r"|\bwho is (the )?(tallest|shortest|oldest|youngest|fastest|slowest|biggest|smallest"
    r"|richest|first|last)\b"
    r"|\bmore than\b.{0,30}\bless than\b|\bfirst to last\b"
    r"|\brank\b.{0,40}\b(first|last|order|from|to)\b|\b(all|some|no)\b \w+ \bare\b",
    re.I,
)


def classify(prompt: str) -> str | None:
    """Return a category name, or None when ambiguous (caller should use the LLM fallback)."""
    for category, rx in R.items():
        if rx.search(prompt):
            return category

    if STRONG_CODE.search(prompt) or (
        WEAK_CODE.search(prompt) and (WRITE.search(prompt) or FIX.search(prompt))
    ):
        if WRITE.search(prompt) and not STRONG_FIX.search(prompt):
            return "codegen"
        if FIX.search(prompt):
            return "debug"
        if WRITE.search(prompt):
            return "codegen"
        return None  # code pasted with no clear fix/write intent

    strong_logic = bool(STRONG_LOGIC.search(prompt))
    has_logic = strong_logic or bool(WEAK_LOGIC.search(prompt))
    has_math = bool(DIGIT.search(prompt)) and bool(MATH_SIGNAL.search(prompt))
    if has_math and has_logic:
        return "logic" if strong_logic else None
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


_LETTER = re.compile(r"\b([A-H])\b")


def parse_fallback_letter(text: str) -> str:
    # Match a standalone letter, not any A-H character: a stray "H" inside a word like
    # "The" must not be read as a class. Unrecognized output -> the safe factual default.
    match = _LETTER.search(text.upper())
    return LETTER_TO_CATEGORY[match.group(1)] if match else "factual"
