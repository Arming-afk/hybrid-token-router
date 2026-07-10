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
        # Bare "sentiment" is decisive except in "sentiment analysis" as an NLP term
        # ("What is sentiment analysis?" is factual); "sentiment analysis on/of <text>"
        # is still a labeling task.
        r"\bsentiment\b(?!\s+analysis\b(?!\s+(on|of)\b))"
        r"|\b(classify|label|categori[sz]e|determine|identify|rate|analy[sz]e)\b"
        r".{0,40}\b(positive|negative|neutral)\b"
        r"|\bpositive[,/ ]+negative[,/ ]+(or +)?neutral\b"
        r"|\bpositive or negative\b"
        r"|\bhow (do|does|would) (you|people|customers|users|they|the (customer|reviewer|user))"
        r" feel\b"
        r"|\bwhat do people think\b"
        r"|\bsatisfied or dissatisfied\b"
        r"|\bhappy or unhappy\b"
        # "tone/mood/emotion" of a text is a sentiment task only when paired with a
        # classification verb AND aimed at a given text (deixis like "of this comment"
        # or a quoted/colon-introduced passage). "What is the mood of a minor key" and
        # "analyze the mood of the Romantic period" are factual questions.
        r"|\b(classify|determine|identify|what is|what'?s|analy[sz]e|judge|assess)"
        r"(?=.{0,25}\b(tone|mood|emotion|sentiment)\b)"
        r"(?=.{0,90}(?:\bof (?:this|these|the following|the (?:text|message|review"
        r"|comment|tweet|feedback|post|email|statement|passage|paragraph))\b|[:\"']))"
        # emotion-word either/or pairs beyond the fixed positive/negative wording
        r"|\b(happy|glad|angry|sad|upset|excited|frustrated|pleased|annoyed)\s+or\s+"
        r"(happy|glad|angry|sad|upset|excited|frustrated|pleased|annoyed|disappointed)\b",
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
        r"|\b(extract|identify|find|list|tag|label|pull out|pick out|highlight|name)\b.{0,80}"
        r"\b(persons?|people|organi[sz]ations?|\borgs?\b|compan(y|ies)|locations?|places?"
        r"|cit(y|ies)|countr(y|ies)|dates?|entit(y|ies))\b"
        r"|\b(persons?|people|organi[sz]ations?|companies|locations?)\b.{0,60}\bnamed\b"
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
    r"\bcode\b|\bsnippet\b|\bfunction\b|\breturn\b|\bclass \w|\bmethod\b|\bvariable\b|\bloop\b|\bregex\b"
    r"|\bquery\b|\bscript\b|\balgorithm\b|\bsort\b|\bdecorator\b|\bSQL\b"
    r"|\berror handling\b|\bexception\b|\bhandler\b|\bendpoint\b|\bparser\b|\bAPI\b"
    r"|\bin (python|java|javascript|typescript|c\+\+|go|rust|ruby)\b",
    re.I,
)
WRITE = re.compile(
    r"\b(write|implement|create|build|generate|develop|complete)\b"
    # Soft request verbs count as write-intent only when their object is a code
    # artifact: "give me a function" is codegen, but "provide an overview of the
    # quicksort algorithm" is a factual question.
    r"|\b(give me|provide|show me)\s+(?:(?:a|an|the|some)\s+)?(?:[\w-]+\s+){0,2}?"
    r"(?:function|method|script|program|class|snippet|implementation|regex|query"
    r"|decorator|endpoint|parser|code)\b",
    re.I,
)
# STRONG_FIX names existing broken code and beats a co-occurring WRITE verb.
STRONG_FIX = re.compile(
    r"\bfix\b|\bdebug\b|\bbugs?\b|\bbuggy\b|\bbroken\b|off-by-one"
    r"|doesn'?t work|not working|never (terminates|returns|works|ends|stops)"
    r"|should\b.{0,50}\bbut\b"
    r"|\bspot the (problem|bug|error|issue)\b|\bwhat'?s? (is )?wrong\b|\bwhat is wrong\b"
    # throws/raises marks a bug report only when a SPECIFIC artifact does the throwing
    # ("this code throws...", "my script raises..."). "a program throws an exception"
    # is conceptual (factual), and "write a function that raises ValueError" is a spec.
    r"|\b(?:this|my|it|below|following|the (?:code|function|script|program|snippet|loop"
    r"|method|query))\b.{0,60}\b(?:throws?|raises?|throwing|raising)\b"
    r".{0,30}\b\w*(?:error|exception)s?\b"
    r"|\bcorrected (version|implementation|code)\b"
    # "why does/do/is/are" alone is an ordinary explanatory question ("why do we use
    # X?"); only treat it as a bug report when it's paired with a failure/behavior word.
    r"|why (does|do|is|are|doesn'?t)\b.{0,40}\b(work|returns?|crash(es)?|fails?|raise"
    r"s?|throws?|breaks?|output|print|behave|terminate|loop|error|exception)\b",
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
    r"\bcalculate\b|\bcompute\b|\bconvert\b|\bhow (many|much|far|fast|long)\b"
    r"|\bpercent(age)?\b|%|\baverage of\b|\bsum of\b|\btotal\b|\bcompound\b|\binterest\b"
    r"|\bdiscount|\bprofit\b|\bsales tax\b|\bper (hour|day|week|month|year|item|unit|person)\b"
    r"|\bspeed\b|\bratio\b|\bproportion\b|\btimes\b|\bdivided?\b|\bmultipl|\bplus\b|\bminus\b"
    r"|\barea\b|\bperimeter\b|\bvolume\b|[+×÷]"
    r"|\bhalf of\b|\bquarter of\b|\bdozen\b|\bdouble[sd]?\b|\btriple[sd]?\b|\btwice\b"
    r"|\bsquare root\b|\bsquared\b|\bcubed\b|\bremainder\b|\bquotient\b|\bproduct of\b"
    r"|\bround(ed)?\b.{0,20}\bdecimal"
    # word-problem phrasings that describe an arithmetic relation between numbers.
    # Age problems key off the RELATION ("years older", "as old as", "in 5 years"),
    # not "how old" itself — "How old is the US Constitution, signed in 1787?" is
    # factual trivia despite containing a digit.
    r"|\badds? (up )?to\b|\bdiffer by\b|\bincreases? by\b"
    r"|\bdecreases? by\b|\bgrows? by\b"
    r"|\byears? (older|younger)\b|\bas old as\b|\bin \d+ years?\b",
    re.I,
)

# --- logic --------------------------------------------------------------------
STRONG_LOGIC = re.compile(
    r"\b(lying|liar|knight|knave)\b|truth[- ]?tell|\btruth\b|\bclues?\b|\bconstraints?\b"
    r"|\bdeduce\b|\bseat(ed|ing)\b|\barrangement\b|\beach\b.{0,40}\bdifferent\b"
    r"|\blogic puzzle\b",
    re.I,
)
# Excludes common factual-trivia subjects ("tallest mountain in the world") so a bare
# superlative question about a real-world thing doesn't get mistaken for a puzzle about
# named people/items being ranked.
NOT_TRIVIA_SUBJECT = (
    r"(?!.{0,25}\b(mountain|animal|country|countries|ocean|river|planet|building|city"
    r"|structure|lake|desert|bird|fish|tree|species|continent|state|volcano|waterfall)\b)"
)
WEAK_LOGIC = re.compile(
    # "in a row" is a seating/lineup signal only with an arrangement verb — "won the
    # most championships in a row" is factual trivia. Same for "sits/stands in":
    # require a positional object ("sits in the middle"), so "the organ sits in the
    # chest cavity" stays factual.
    r"\brow of\b"
    r"|\b(?:sit|sits|sitting|stand|stands|standing|seated|are|is|placed|arranged"
    r"|lined? up)\s+in a row\b"
    r"|\bsits? in\b(?=\s+(?:the\s+)?(?:middle|front|back|center|centre|row|seat"
    r"|position|chair|first|second|third|last))"
    r"|\bstands? in\b(?=\s+(?:the\s+)?(?:middle|front|back|line|queue|row|position))"
    r"|\badjacent\b"
    r"|\b(directly |immediately )?(left|right) of\b"
    r"|\bfinish(es|ed)?\b.{0,20}\b(before|after|first|last)\b"
    r"|\bwho (finished|came|won|wins|placed|ranked|owns|is next)\b(?!\s+to\b)"
    r"|\bwho is (the )?" + NOT_TRIVIA_SUBJECT + r"(tallest|shortest|oldest|youngest|fastest|slowest|biggest|smallest"
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
        # Literal code pasted with no write intent is almost always a "here's my code,
        # fix it" request; debug is a far safer default than an LLM fallback that can
        # silently degrade to factual on an empty/rate-limited response.
        return "debug"

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
