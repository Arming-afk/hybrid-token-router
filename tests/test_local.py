"""Unit tests for local.py's zero-token verifiers (pure functions, no network)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.local import LOCAL_CATEGORIES, verify  # noqa: E402


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


def test_sentence_count_ignores_abbreviations_and_initials():
    prompt = "Summarize in one sentence: ..."
    abbrev = ("The U.S. transit authority, led by Dr. Lee and J. Smith of Acme Inc., "
              "approved the plan (e.g. new corridors).")
    assert verify(prompt, "summarization", abbrev)[0] is True
    two_real = "The plan was approved. Work begins in September."
    assert verify(prompt, "summarization", two_real)[0] is False


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
