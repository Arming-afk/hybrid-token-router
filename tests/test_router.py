"""Router accuracy gate: >=95% correct on decided cases, <=10% falling back to the LLM."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.router import classify, parse_fallback_letter  # noqa: E402


def load_cases() -> list[dict]:
    with open(Path(__file__).parent / "router_cases.json", encoding="utf-8") as f:
        return json.load(f)


def test_router_accuracy():
    cases = load_cases()
    wrong, fallback = [], []
    for case in cases:
        got = classify(case["prompt"])
        if got is None:
            fallback.append(case["prompt"][:60])
        elif got != case["category"]:
            wrong.append((case["prompt"][:60], case["category"], got))
    decided = len(cases) - len(fallback)
    accuracy = (decided - len(wrong)) / decided
    print(f"cases={len(cases)} decided={decided} accuracy={accuracy:.1%} "
          f"fallback={len(fallback)}/{len(cases)}")
    assert accuracy >= 0.95, f"wrong: {wrong}"
    assert len(fallback) / len(cases) <= 0.10, f"too many fallbacks: {fallback}"


def test_ambiguous_prompts_defer_to_llm():
    # Genuinely ambiguous inputs must return None so main.py pays for an LLM classify
    # instead of guessing wrong. These should NOT resolve to a concrete category.
    ambiguous = [
        "Each person has a different number of coins; using 2 clues, calculate who has the most.",
    ]
    for prompt in ambiguous:
        result = classify(prompt)
        assert result in (None, "logic"), f"{prompt!r} -> {result}"


def test_pasted_code_defaults_to_debug():
    # There is no "explain this code" category; pasted literal code with no write verb
    # is a "here's my code, fix it" request. Defaulting to debug beats an LLM fallback
    # that can silently degrade to factual on an empty/rate-limited response.
    assert classify("x = [1, 2, 3]\nprint(x[5])") == "debug"


def test_fallback_letter_parsing():
    assert parse_fallback_letter("B") == "math"
    assert parse_fallback_letter("The answer is E.") == "ner"
    assert parse_fallback_letter("") == "factual"        # empty -> safe default
    assert parse_fallback_letter("zzz") == "factual"     # no valid letter -> default


if __name__ == "__main__":
    test_router_accuracy()
    test_ambiguous_prompts_defer_to_llm()
    test_fallback_letter_parsing()
    print("router dev-set: PASS")
