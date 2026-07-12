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


# --- deterministic arithmetic (zero tokens) -------------------------------------
# Pure-calculation prompts ("What is 847 x 23?", "Calculate 15% of 240") are solved
# in-process: no API call, no billed tokens, no hallucination risk. The charset gate
# rejects anything containing narrative words, so word problems structurally cannot
# trigger it and flow through normal routing untouched.
_ARITH_LEAD = re.compile(r"^(?:what\s+is|what'?s|calculate|compute|evaluate|solve)\b[: ]*", re.I)
_PERCENT_OF = re.compile(r"^([\d,]+(?:\.\d+)?)\s*(?:%|percent)\s+of\s+([\d,]+(?:\.\d+)?)$", re.I)

_NUM = r"([\d,]+(?:\.\d+)?)"
# Anchored full-string patterns: each matches ONLY a pure-calculation phrasing and
# computes directly, so no narrative residue reaches _eval_arith. Order matters only
# in that each is tried on the full lead-stripped text; a word problem matches none.
_SQRT = re.compile(rf"^(?:the\s+)?square\s+root\s+of\s+{_NUM}$", re.I)
_SUM_OF = re.compile(rf"^(?:the\s+)?sum\s+of\s+{_NUM}\s+and\s+{_NUM}$", re.I)
_PRODUCT_OF = re.compile(rf"^(?:the\s+)?product\s+of\s+{_NUM}\s+and\s+{_NUM}$", re.I)
_DIFFERENCE_OF = re.compile(rf"^(?:the\s+)?difference\s+between\s+{_NUM}\s+and\s+{_NUM}$", re.I)
_ADD = re.compile(rf"^add\s+{_NUM}\s+(?:and|to)\s+{_NUM}$", re.I)
_SUBTRACT_FROM = re.compile(rf"^subtract\s+{_NUM}\s+from\s+{_NUM}$", re.I)  # B - A
_PERCENT_OFF = re.compile(rf"^{_NUM}\s*(?:%|percent)\s+off\s+(?:of\s+)?{_NUM}$", re.I)
_PERCENT_CHANGE = re.compile(
    rf"^(increase|decrease)\s+{_NUM}\s+by\s+{_NUM}\s*(?:%|percent)$", re.I)
_MOD = re.compile(rf"^{_NUM}\s+mod(?:ulo|ulus)?\s+{_NUM}$", re.I)
_REMAINDER = re.compile(
    rf"^(?:the\s+)?remainder\s+(?:when\s+)?{_NUM}\s+is\s+divided\s+by\s+{_NUM}$", re.I)
_HALF_OF = re.compile(rf"^(?:one\s+)?half\s+of\s+{_NUM}$", re.I)
_FRACTION_OF = re.compile(rf"^(?:a\s+)?(third|quarter|fifth|tenth)\s+of\s+{_NUM}$", re.I)
_FRACTIONS = {"third": 1 / 3, "quarter": 0.25, "fifth": 0.2, "tenth": 0.1}

# Word/symbol power operators: substituted before the charset gate. 'squared'/'cubed'
# expand to '** 2'/'** 3'; '^' and 'to the power of' become '**'. Non-numeric residue
# still fails the charset gate, so "the power struggle" never reaches the evaluator.
_POWER_OPS = [
    ("to the power of", "**"), ("squared", "** 2"), ("cubed", "** 3"), ("^", "**"),
]
_WORD_OPS = [
    ("multiplied by", "*"), ("divided by", "/"), ("times", "*"),
    ("plus", "+"), ("minus", "-"), (" x ", " * "), ("×", "*"), ("÷", "/"),
]
_ARITH_CHARSET = re.compile(r"^[\d\s+\-*/.(),]+$")

# Hard bounds so no crafted input can hang the process on a huge exponentiation.
_MAX_ABS_EXPONENT = 12
_MAX_ABS_BASE = 10 ** 6


def _num(s: str) -> float:
    return float(s.replace(",", ""))


def _isqrt_or_none(n: float) -> float | None:
    if n < 0 or not float(n).is_integer():
        return None
    root = int(round(n ** 0.5))
    for candidate in (root - 1, root, root + 1):
        if candidate >= 0 and candidate * candidate == int(n):
            return float(candidate)
    return None


def _solve_named(text: str) -> str | None:
    """Anchored named-operation patterns. Returns a formatted answer or None."""
    m = _SQRT.match(text)
    if m:
        root = _isqrt_or_none(_num(m.group(1)))
        return None if root is None else f"Answer: {_format_number(root)}"
    m = _SUM_OF.match(text) or _ADD.match(text)
    if m:
        return f"Answer: {_format_number(_num(m.group(1)) + _num(m.group(2)))}"
    m = _PRODUCT_OF.match(text)
    if m:
        return f"Answer: {_format_number(_num(m.group(1)) * _num(m.group(2)))}"
    m = _DIFFERENCE_OF.match(text)
    if m:
        return f"Answer: {_format_number(abs(_num(m.group(1)) - _num(m.group(2))))}"
    m = _SUBTRACT_FROM.match(text)
    if m:  # 'subtract A from B' = B - A
        return f"Answer: {_format_number(_num(m.group(2)) - _num(m.group(1)))}"
    m = _PERCENT_OFF.match(text)
    if m:
        base = _num(m.group(2))
        return f"Answer: {_format_number(base * (1 - _num(m.group(1)) / 100.0))}"
    m = _PERCENT_CHANGE.match(text)
    if m:
        base, pct = _num(m.group(2)), _num(m.group(3))
        sign = 1 if m.group(1).lower() == "increase" else -1
        return f"Answer: {_format_number(base * (1 + sign * pct / 100.0))}"
    m = _MOD.match(text) or _REMAINDER.match(text)
    if m:
        b = _num(m.group(2))
        if b == 0:
            return None
        return f"Answer: {_format_number(_num(m.group(1)) % b)}"
    m = _HALF_OF.match(text)
    if m:
        return f"Answer: {_format_number(_num(m.group(1)) * 0.5)}"
    m = _FRACTION_OF.match(text)
    if m:
        return f"Answer: {_format_number(_num(m.group(2)) * _FRACTIONS[m.group(1).lower()])}"
    return None


def _eval_arith(expr: str) -> float | None:
    """AST-whitelist arithmetic: + - * / and parentheses only. No eval()."""
    import ast

    def walk(node):
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = walk(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and isinstance(
                node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
            left, right = walk(node.left), walk(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Pow):
                # Bound both operands so no input can hang on a huge exponentiation.
                if abs(right) > _MAX_ABS_EXPONENT or abs(left) > _MAX_ABS_BASE:
                    raise ValueError("exponent out of bounds")
                return left ** right
            if right == 0:
                raise ZeroDivisionError
            return left / right
        raise ValueError("disallowed node")

    try:
        return walk(ast.parse(expr, mode="eval"))
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError):
        return None


def _format_number(value: float) -> str:
    if abs(value) < 1e15 and float(value).is_integer():
        return str(int(value))
    return f"{round(value, 6):g}"


def deterministic_math_answer(prompt: str) -> str | None:
    """Return the computed answer for a pure-arithmetic prompt, else None."""
    text = (prompt or "").strip().strip("?!. \t\n").lower()
    text = _ARITH_LEAD.sub("", text).strip().strip("?!. ")
    percent = _PERCENT_OF.match(text)
    if percent:
        pct = float(percent.group(1).replace(",", ""))
        base = float(percent.group(2).replace(",", ""))
        return f"Answer: {_format_number(base * pct / 100.0)}"
    # Anchored named operations (sum of, subtract A from B, square root, mod, ...):
    # these match whole-string phrasings a narrative prompt cannot, and never leave
    # residue for the evaluator.
    named = _solve_named(text)
    if named is not None:
        return named
    for word, symbol in _POWER_OPS:
        text = text.replace(word, symbol)
    for word, symbol in _WORD_OPS:
        text = text.replace(word, symbol)
    if not _ARITH_CHARSET.match(text):
        return None
    expr = text.replace(",", "")
    # A lone number ("What is 42?") is not a calculation — require an operator.
    if not any(op in expr for op in "+-*/"):
        return None
    value = _eval_arith(expr)
    if value is None:
        return None
    return f"Answer: {_format_number(value)}"


# --- Deterministic logic: transitive-ordering puzzles ---------------------------
# Only the clean "A is <comparative> than B ... who is the <superlative>?" shape,
# on ONE consistent dimension, resolving to a UNIQUE extreme over a TOTAL order.
# Anything else — mixed dimensions, a broken chain, a syllogism, a truth-teller
# puzzle, a conditional — returns None and pays the normal path. A wrong solver
# answer would cost accuracy, so every branch here is biased hard toward None.
#
# Each dimension: (comparative words meaning LEFT > RIGHT, comparatives meaning
# LEFT < RIGHT, superlatives asking for the MAX, superlatives asking for the MIN).
_ORDER_DIMS = [
    ({"older", "elder"}, {"younger"}, {"oldest", "eldest"}, {"youngest"}),
    ({"taller"}, {"shorter"}, {"tallest"}, {"shortest"}),
    ({"heavier"}, {"lighter"}, {"heaviest"}, {"lightest"}),
    ({"bigger", "larger"}, {"smaller"}, {"biggest", "largest"}, {"smallest"}),
    ({"faster"}, {"slower"}, {"fastest"}, {"slowest"}),
    ({"richer", "wealthier"}, {"poorer"}, {"richest", "wealthiest"}, {"poorest"}),
    ({"stronger"}, {"weaker"}, {"strongest"}, {"weakest"}),
]
_COMPARE_RE = re.compile(
    r"\b([A-Z][a-z]*)\s+(?:is|was|comes?|finished|ran|scored)?\s*"
    r"(older|elder|younger|taller|shorter|heavier|lighter|bigger|larger|smaller|"
    r"faster|slower|richer|wealthier|poorer|stronger|weaker)\s+than\s+([A-Z][a-z]*)\b")
_SUPERLATIVE_RE = re.compile(
    r"\bwho\b.*?\b(oldest|eldest|youngest|tallest|shortest|heaviest|lightest|"
    r"biggest|largest|smallest|fastest|slowest|richest|wealthiest|poorest|"
    r"strongest|weakest)\b", re.I)


def deterministic_logic_answer(prompt: str) -> str | None:
    """Solve a single-dimension transitive-ordering puzzle, else None."""
    text = prompt or ""
    sup = _SUPERLATIVE_RE.search(text)
    pairs = _COMPARE_RE.findall(text)
    if not sup or not pairs:
        return None
    superlative = sup.group(1).lower()

    # The one dimension must be consistent across every comparison AND the question.
    dim = next((d for d in _ORDER_DIMS
                if superlative in d[2] or superlative in d[3]), None)
    if dim is None:
        return None
    greater_w, lesser_w, max_super, _min_super = dim
    want_max = superlative in max_super

    # Build "greater-than" edges (winner > loser) from each comparison.
    edges: list[tuple[str, str]] = []
    people: set[str] = set()
    for left, comp, right in pairs:
        comp = comp.lower()
        if comp in greater_w:
            edges.append((left, right))
        elif comp in lesser_w:
            edges.append((right, left))
        else:
            return None  # comparison on a different dimension → bail
        people.update((left, right))

    # Transitive closure, then require a TOTAL order: the "greater-than" counts
    # must be exactly the permutation 0..N-1 (a unique rank for everyone).
    n = len(people)
    if n < 2:
        return None
    greater_than: dict[str, set[str]] = {p: set() for p in people}
    for a, b in edges:
        greater_than[a].add(b)
    for _ in range(n):  # propagate reachability to a fixed point
        for a in people:
            for b in list(greater_than[a]):
                greater_than[a] |= greater_than[b]
    counts = {p: len(greater_than[p]) for p in people}
    if sorted(counts.values()) != list(range(n)):
        return None  # not a clean total order → ambiguous, defer

    target_rank = n - 1 if want_max else 0
    winners = [p for p, c in counts.items() if c == target_rank]
    if len(winners) != 1:
        return None
    return winners[0]


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
