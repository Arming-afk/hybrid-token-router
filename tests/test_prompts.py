"""Unit tests for prompts.render()/postprocess()."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.prompts import CATEGORIES, SPEC, postprocess, render  # noqa: E402


def test_render_returns_tier_and_cap_from_spec():
    for category in CATEGORIES:
        messages, max_tokens, tier = render(category, "some prompt")
        assert max_tokens == SPEC[category]["max_tokens"]
        assert tier == SPEC[category]["tier"]
        assert messages == [{"role": "user",
                             "content": f"some prompt\n\n{SPEC[category]['instruction']}"}]


def test_postprocess_extracts_after_last_answer_marker():
    text = "Step 1: add.\nStep 2: multiply.\nAnswer: 42"
    assert postprocess("math", text) == "42"


def test_postprocess_is_case_insensitive_and_tolerates_no_space():
    assert postprocess("math", "reasoning...\nANSWER:7") == "7"
    assert postprocess("logic", "reasoning...\nanswer :  Carol") == "Carol"


def test_postprocess_uses_last_marker_when_multiple_present():
    # A model that echoes "Answer:" while reasoning, then gives the real one at the end.
    text = "If Answer: were X that would be wrong.\nAnswer: Y"
    assert postprocess("math", text) == "Y"


def test_postprocess_falls_back_to_full_text_when_marker_missing():
    assert postprocess("math", "  42  ") == "42"


def test_postprocess_falls_back_when_marker_has_no_content_after_it():
    assert postprocess("math", "Answer:") == "Answer:"


def test_postprocess_leaves_non_math_logic_categories_untouched():
    assert postprocess("factual", "  Paris.  ") == "Paris."
    assert postprocess("codegen", "Answer: not stripped here") == "Answer: not stripped here"


if __name__ == "__main__":
    test_render_returns_tier_and_cap_from_spec()
    test_postprocess_extracts_after_last_answer_marker()
    test_postprocess_is_case_insensitive_and_tolerates_no_space()
    test_postprocess_uses_last_marker_when_multiple_present()
    test_postprocess_falls_back_to_full_text_when_marker_missing()
    test_postprocess_falls_back_when_marker_has_no_content_after_it()
    test_postprocess_leaves_non_math_logic_categories_untouched()
    print("prompts: PASS")
