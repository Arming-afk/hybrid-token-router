"""Tier assignment must survive the real-world model-id shapes we can't see until launch."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import models  # noqa: E402


def tiers(allowed: str) -> dict:
    os.environ["ALLOWED_MODELS"] = allowed
    return models.build_tiers()


def test_three_distinct_sizes():
    t = tiers("acc/llama-v3p1-8b-instruct,acc/llama-v3p3-70b-instruct,acc/qwen3-235b-a22b")
    assert t["SMALL"].endswith("8b-instruct")
    assert t["MEDIUM"].endswith("70b-instruct")
    assert t["LARGE"].endswith("235b-a22b")


def test_single_model_fills_all_tiers():
    t = tiers("acc/qwen3-30b-a3b")
    assert t["SMALL"] == t["MEDIUM"] == t["LARGE"] == "acc/qwen3-30b-a3b"


def test_moe_counts_total_params():
    # mixtral-8x7b must read as 56b (not 7b), so it outranks a plain 8b model.
    assert models._size_b("mixtral-8x7b") == 56.0
    assert models._size_b("mixtral-8x22b") == 176.0
    t = tiers("acc/llama-8b,acc/mixtral-8x7b,acc/qwen-235b")
    assert t["SMALL"].endswith("llama-8b")
    assert t["MEDIUM"].endswith("mixtral-8x7b")
    assert t["LARGE"].endswith("qwen-235b")


def test_active_params_id_uses_larger_number():
    assert models._size_b("qwen3-235b-a22b") == 235.0


def test_unparseable_name_treated_as_large():
    assert models._size_b("deepseek-v3") == models._UNKNOWN_SIZE
    t = tiers("acc/llama-8b,acc/deepseek-v3")
    assert t["SMALL"].endswith("llama-8b")
    assert t["LARGE"].endswith("deepseek-v3")


def test_reasoning_model_never_small_or_medium():
    t = tiers("acc/llama-8b,acc/llama-70b,acc/deepseek-r1")
    assert not models._is_reasoning(t["SMALL"])
    assert not models._is_reasoning(t["MEDIUM"])
    assert t["LARGE"].endswith("deepseek-r1")  # reasoning allowed only in the retry tier


def test_all_reasoning_does_not_crash():
    t = tiers("acc/deepseek-r1-distill-qwen-7b,acc/deepseek-r1-distill-llama-70b")
    assert t["SMALL"].endswith("7b")
    assert t["LARGE"].endswith("70b")


def test_reasoning_detection_patterns():
    for ident in ["deepseek-r1", "qwen-qwq-32b", "some-thinking-model", "acc/o1-mini"]:
        assert models._is_reasoning(ident), ident
    for ident in ["llama-v3p1-8b", "qwen3-235b-a22b", "deepseek-v3"]:
        assert not models._is_reasoning(ident), ident


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("PASS", fn.__name__)
    print(f"\ntest_models: {len(fns)} passed")
