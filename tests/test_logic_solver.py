"""Tests for router.deterministic_logic_answer — the transitive-ordering solver.

The must-NOT-fire cases matter most: a solver that returns a confidently wrong
answer would COST accuracy, the opposite of a zero-token free win. Every
ambiguous, mixed-dimension, broken-chain, or non-ordering prompt must return None
so the task pays the normal model path.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.router import deterministic_logic_answer as solve  # noqa: E402


def test_fires_on_clean_ordering_chains():
    assert solve("Tom is older than Sara. Sara is older than Bill. Who is the youngest?") == "Bill"
    assert solve("Tom is older than Sara. Sara is older than Bill. Who is the oldest?") == "Tom"
    assert solve("A is heavier than B. C is lighter than B. Who is the heaviest?") == "A"
    assert solve("Maya is faster than Noah. Noah is faster than Omar. Who is the slowest?") == "Omar"
    assert solve("Lena is taller than Kim. Kim is taller than Joe. Who is the shortest?") == "Joe"


def test_returns_none_on_non_ordering_puzzles():
    # syllogism, truth-teller, conditional — not a transitive ordering
    assert solve("If all roses are flowers and all flowers need water, do all roses need water?") is None
    assert solve("Anna always tells the truth. Anna says Ben is lying. Is Ben lying?") is None
    assert solve("If it is raining then the ground is wet. The ground is wet. Is it raining?") is None
    assert solve("What is the capital of France?") is None
    assert solve("") is None


def test_returns_none_on_broken_or_ambiguous_orders():
    # two disconnected chains → no unique extreme
    assert solve("A is older than B. C is older than D. Who is the youngest?") is None
    # mixed dimensions → bail
    assert solve("A is older than B. B is taller than C. Who is the youngest?") is None
    # a comparison but no superlative question
    assert solve("Tom is older than Sara. Is Tom old?") is None
    # superlative dimension doesn't match the comparison dimension
    assert solve("A is older than B. B is older than C. Who is the tallest?") is None


def test_none_when_extreme_not_unique():
    # only a partial constraint, extreme not determined over all named people
    assert solve("A is older than B. A is older than C. Who is the youngest?") is None


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("PASS", fn.__name__)
    print(f"\ntest_logic_solver: {len(fns)} passed")
