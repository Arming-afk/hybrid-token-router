"""Unit tests for local.py's zero-token verifiers (pure functions, no network)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.local import (  # noqa: E402
    CALL_TIMEOUT, CODE_CALL_TIMEOUT, CODE_MAX_CALL_TIMEOUT, CODEGEN_NUM_PREDICT,
    FIRST_CALL_TIMEOUT, MAX_CALL_TIMEOUT, NUM_PREDICT, LOCAL_CATEGORIES,
    _num_predict_for, _timeout_for, verify,
)


def test_default_local_categories_are_the_current_bisect_rung():
    assert LOCAL_CATEGORIES == {"sentiment", "ner", "summarization"}
    for kept_remote in ("factual", "math", "logic", "debug", "codegen"):
        assert kept_remote not in LOCAL_CATEGORIES


def test_empty_and_hedged_answers_fail():
    assert verify("q", "sentiment", "")[0] is False
    assert verify("q", "factual", "I'm not sure, as an AI I cannot answer that.")[0] is False


def test_degenerate_repetition_fails():
    assert verify("q", "summarization", "the same " * 30)[0] is False


def test_sentiment_needs_a_label():
    assert verify("Classify the sentiment.", "sentiment", "Positive - praises battery.")[0] is True
    assert verify("Classify the sentiment.", "sentiment", "The review praises the battery.")[0] is False


def test_ner_needs_label_value_lines():
    good = "Person: Tim Cook\nLocation: Paris\nDate: June 2021"
    assert verify("Extract entities.", "ner", good)[0] is True
    assert verify("Extract entities.", "ner", "Tim Cook went to Paris in June 2021.")[0] is False


def test_summarization_respects_stated_limits():
    prompt = "Summarize in one sentence: ..."
    assert verify(prompt, "summarization", "One tidy sentence.")[0] is True
    assert verify(prompt, "summarization", "First sentence. Second sentence. Third.")[0] is False
    prompt_words = "Summarize in under 5 words: ..."
    assert verify(prompt_words, "summarization", "Far too many words in this answer.")[0] is False


def test_timeout_scales_with_length_and_is_bounded():
    # Short prompt, warm: near the base timeout.
    assert _timeout_for(50, first_call=False) == CALL_TIMEOUT + 8.0 * 50 / 1024.0
    # A ~2.5KB passage scales up but stays under the ceiling.
    mid = _timeout_for(2560, first_call=False)
    assert CALL_TIMEOUT < mid <= MAX_CALL_TIMEOUT
    # A huge passage is capped, never unbounded.
    assert _timeout_for(100_000, first_call=False) == MAX_CALL_TIMEOUT
    # First call absorbs the cold start regardless of length.
    assert _timeout_for(50, first_call=True) == FIRST_CALL_TIMEOUT


def test_code_categories_get_a_higher_timeout_floor_and_ceiling():
    # debug/codegen produce a full code block regardless of prompt length, so
    # their base/ceiling are both higher than the default category timeout.
    assert _timeout_for(50, first_call=False, category="debug") == (
        CODE_CALL_TIMEOUT + 8.0 * 50 / 1024.0)
    assert _timeout_for(50, first_call=False, category="codegen") > (
        _timeout_for(50, first_call=False))
    assert _timeout_for(100_000, first_call=False, category="debug") == CODE_MAX_CALL_TIMEOUT
    assert CODE_MAX_CALL_TIMEOUT > MAX_CALL_TIMEOUT
    # Non-code categories are unaffected (default category arg preserves old behavior).
    assert _timeout_for(50, first_call=False, category="sentiment") == (
        _timeout_for(50, first_call=False))


def test_codegen_gets_a_lower_generation_cap_than_other_categories():
    assert CODEGEN_NUM_PREDICT < NUM_PREDICT
    assert _num_predict_for("codegen") == CODEGEN_NUM_PREDICT
    for other in ("sentiment", "ner", "summarization", "debug", "factual", "math", "logic"):
        assert _num_predict_for(other) == NUM_PREDICT


def test_summarization_rejects_non_summaries():
    source = " ".join(f"word{i}" for i in range(120))  # 120-word source
    # An answer nearly as long as the source is not a summary.
    too_long = " ".join(f"word{i}" for i in range(110))
    assert verify(source, "summarization", too_long)[0] is False
    # A genuine short summary of a long source passes.
    assert verify(source, "summarization", "A brief three word summary here.")[0] is True


def test_summarization_rejects_verbatim_echo():
    source = ("The quarterly report shows that revenue rose twelve percent while "
              "costs fell across every division this fiscal year, and management "
              "expects the trend to continue into the next two quarters as well.")
    # First 15+ words copied verbatim from the source is an echo, not a summary.
    echo = "The quarterly report shows that revenue rose twelve percent while costs fell across every division."
    assert verify(source, "summarization", echo)[0] is False


def test_sentence_counter_ignores_abbreviations():
    prompt = "Summarize in one sentence: ..."
    # One real sentence that contains abbreviations must not be over-counted.
    ok = "The U.S. team, led by Dr. Smith, shipped the product on time."
    assert verify(prompt, "summarization", ok)[0] is True


def test_code_answers_must_parse_and_contain_requested_function():
    prompt = "Write a Python function called 'reverse_string' that reverses a string."
    good = "```python\ndef reverse_string(s):\n    return s[::-1]\n```"
    bad_syntax = "```python\ndef reverse_string(s:\n    return s[::-1\n```"
    wrong_name = "```python\ndef flip(s):\n    return s[::-1]\n```"
    no_fence = "def reverse_string(s): return s[::-1]"
    assert verify(prompt, "codegen", good)[0] is True
    assert verify(prompt, "codegen", bad_syntax)[0] is False
    assert verify(prompt, "codegen", wrong_name)[0] is False
    assert verify(prompt, "codegen", no_fence)[0] is False


def test_debug_rejects_unchanged_buggy_code():
    prompt = "Fix this code:\n```python\ndef add(a, b):\n    return a - b\n```"
    unchanged = "The bug is subtraction.\n```python\ndef add(a, b):\n    return a - b\n```"
    fixed = "The bug is subtraction.\n```python\ndef add(a, b):\n    return a + b\n```"
    assert verify(prompt, "debug", unchanged)[0] is False
    assert verify(prompt, "debug", fixed)[0] is True


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("PASS", fn.__name__)
    print(f"\ntest_local: {len(fns)} passed")
