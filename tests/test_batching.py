"""Unit tests for batching.py's pure batch-format functions (no network)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.batching import (  # noqa: E402
    CHUNK_LIMITS, build_batch_messages, group_and_chunk, parse_batch_response,
    verify_slice,
)
from src.prompts import SPEC  # noqa: E402


def _wrap(task_id: str, text: str) -> str:
    return f"@@ANSWER:{task_id}@@\n{text}\n@@ENDANSWER:{task_id}@@"


def test_well_formed_batch_extracts_every_slice_as_complete():
    raw = "\n\n".join([_wrap("t1", "answer one"), _wrap("t2", "answer two"), _wrap("t3", "answer three")])
    parsed = parse_batch_response(raw, ["t1", "t2", "t3"])
    assert parsed["t1"] == {"text": "answer one", "complete": True}
    assert parsed["t2"] == {"text": "answer two", "complete": True}
    assert parsed["t3"] == {"text": "answer three", "complete": True}


def test_missing_task_block_does_not_affect_the_others():
    raw = "\n\n".join([_wrap("t1", "answer one"), _wrap("t3", "answer three")])  # t2 dropped
    parsed = parse_batch_response(raw, ["t1", "t2", "t3"])
    assert parsed["t1"]["complete"] is True
    assert parsed["t3"]["complete"] is True
    assert parsed["t2"] == {"text": "", "complete": False}


def test_truncated_last_block_is_never_trusted_even_if_plausible():
    # No @@ENDANSWER:t2@@ at all -- model's completion was cut off mid-answer.
    raw = _wrap("t1", "answer one") + "\n\n@@ANSWER:t2@@\nthis looks like a real answer but got cut off"
    parsed = parse_batch_response(raw, ["t1", "t2"])
    assert parsed["t1"] == {"text": "answer one", "complete": True}
    assert parsed["t2"]["complete"] is False
    # The loose fallback still recovers the text for diagnostics, but complete=False
    # means callers must never trust it as an answer.
    assert "cut off" in parsed["t2"]["text"]


def test_reordered_and_noisy_output_still_recovered():
    raw = (
        "Sure, here are the answers:\n\n"
        + _wrap("t2", "second") + "\n\n" + _wrap("t1", "first")
        + "\n\nHope that helps!"
    )
    parsed = parse_batch_response(raw, ["t1", "t2"])
    assert parsed["t1"] == {"text": "first", "complete": True}
    assert parsed["t2"] == {"text": "second", "complete": True}


def test_completely_absent_id_reports_missing_not_a_crash():
    parsed = parse_batch_response("no markers of any kind here", ["t1"])
    assert parsed["t1"] == {"text": "", "complete": False}


def test_verify_slice_reuses_local_verify_for_its_categories():
    # sentiment: local.verify requires a label -- unchanged behavior via batching.
    assert verify_slice("Classify sentiment.", "sentiment", "Positive - great battery.")[0] is True
    assert verify_slice("Classify sentiment.", "sentiment", "The battery is good.")[0] is False


def test_verify_slice_requires_answer_line_for_math_and_logic():
    good = "Step 1: compute.\nAnswer: 42"
    assert verify_slice("2+2*20", "math", good)[0] is True
    assert verify_slice("2+2*20", "math", "Step 1: compute.\nThe result is 42.")[0] is False
    assert verify_slice("seating puzzle", "logic", "Reasoning...\nAnswer: Bob")[0] is True
    assert verify_slice("seating puzzle", "logic", "Reasoning... Bob sits left.")[0] is False


def test_verify_slice_factual_only_gets_generic_checks():
    # No category-specific branch for factual in local.verify -- any non-hedged,
    # non-degenerate text passes, exactly like today's unverified solo path.
    assert verify_slice("What is a hash table?", "factual", "A data structure mapping keys to buckets.")[0] is True
    assert verify_slice("What is a hash table?", "factual", "I'm not sure, as an AI I cannot know.")[0] is False


def test_group_and_chunk_covers_every_task_id_exactly_once():
    pending = [
        {"task_id": f"t{i}", "category": cat, "prompt": "p"}
        for i, cat in enumerate(["factual", "math", "logic", "debug", "codegen",
                                  "sentiment", "ner", "summarization"])
    ]
    chunks = group_and_chunk(pending)
    seen = [item["task_id"] for chunk in chunks for item in chunk]
    assert sorted(seen) == sorted(item["task_id"] for item in pending)
    assert len(seen) == len(set(seen))


def test_group_and_chunk_respects_tier_caps():
    # 5 CODE-tier tasks (debug/codegen at 520 max_tokens each) must not all land
    # in one chunk if that would exceed CHUNK_LIMITS["CODE"].
    pending = [{"task_id": f"c{i}", "category": "codegen", "prompt": "p"} for i in range(5)]
    chunks = group_and_chunk(pending)
    limits = CHUNK_LIMITS["CODE"]
    for chunk in chunks:
        assert len(chunk) <= limits["max_tasks"]
        total = sum(SPEC[item["category"]]["max_tokens"] + 40 for item in chunk)
        assert total <= limits["max_tokens"]


def test_group_and_chunk_keeps_tiers_separate():
    pending = [
        {"task_id": "a", "category": "factual", "prompt": "p"},   # CODE
        {"task_id": "b", "category": "sentiment", "prompt": "p"},  # SMALL
    ]
    chunks = group_and_chunk(pending)
    assert len(chunks) == 2
    tiers_seen = {SPEC[chunk[0]["category"]]["tier"] for chunk in chunks}
    assert tiers_seen == {"CODE", "SMALL"}


def test_singleton_tier_group_still_produces_a_chunk():
    # Caller (main.py) is responsible for routing a length-1 chunk to the solo
    # path instead of the batch format -- group_and_chunk itself just reports
    # the grouping faithfully.
    pending = [{"task_id": "only", "category": "factual", "prompt": "p"}]
    chunks = group_and_chunk(pending)
    assert chunks == [[{"task_id": "only", "category": "factual", "prompt": "p"}]]


def test_build_batch_messages_embeds_every_task_and_its_own_instruction():
    chunk = [
        {"task_id": "t1", "category": "factual", "prompt": "What is X?"},
        {"task_id": "t2", "category": "math", "prompt": "2+2"},
    ]
    messages, max_tokens = build_batch_messages(chunk)
    assert messages[0]["role"] == "system"
    user_content = messages[1]["content"]
    assert "@@TASK:t1@@" in user_content and "@@ENDTASK:t1@@" in user_content
    assert "@@TASK:t2@@" in user_content and "@@ENDTASK:t2@@" in user_content
    assert SPEC["factual"]["instruction"] in user_content
    assert SPEC["math"]["instruction"] in user_content
    assert "What is X?" in user_content and "2+2" in user_content
    assert max_tokens > SPEC["factual"]["max_tokens"] + SPEC["math"]["max_tokens"]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("PASS", fn.__name__)
    print(f"\ntest_batching: {len(fns)} passed")
