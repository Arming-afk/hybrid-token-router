"""Unit tests for prompts.render()/postprocess()."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.prompts import CATEGORIES, SPEC, postprocess, render  # noqa: E402

VALID_TIERS = {"SMALL", "MEDIUM", "LARGE", "CODE"}


def test_render_returns_tier_and_cap_from_spec():
    for category in CATEGORIES:
        messages, max_tokens, tier = render(category, "some prompt")
        assert max_tokens == SPEC[category]["max_tokens"]
        assert tier == SPEC[category]["tier"]
        assert tier in VALID_TIERS
        assert messages == [
            {"role": "system", "content": SPEC[category]["instruction"]},
            {"role": "user", "content": "some prompt"},
        ]


def test_every_instruction_demands_english():
    # Harness hard rule: all responses must be in English.
    for category in CATEGORIES:
        assert "English" in SPEC[category]["instruction"], category


def test_postprocess_passes_answers_through_untouched():
    # The gate-passing reference config hands the judge the full answer, steps and
    # "Answer:" line included; postprocess must only trim whitespace.
    text = "Step 1: add.\nStep 2: multiply.\nAnswer: 42"
    assert postprocess("math", text) == text
    assert postprocess("logic", "  reasoning...\nanswer :  Carol  ") == "reasoning...\nanswer :  Carol"


def test_postprocess_strips_whitespace_for_all_categories():
    for category in CATEGORIES:
        assert postprocess(category, "  body  ") == "body"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
    print("prompts: PASS")
