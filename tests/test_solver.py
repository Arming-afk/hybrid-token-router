"""Unit tests for router.deterministic_math_answer (the zero-token arithmetic solver).

Two duties per pattern: (1) it computes the RIGHT value on pure-calculation prompts,
and (2) narrative word problems still return None so they flow to normal routing.
The second duty is the important one — a false positive here submits a wrong answer
for free, which is worse than paying for a correct one. History (run 2 vs 4): any
widening that leaks into narrative prompts costs judge tasks.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.router import deterministic_math_answer as solve  # noqa: E402


def _val(prompt):
    out = solve(prompt)
    if out is None:
        return None
    assert out.startswith("Answer: "), out
    return out[len("Answer: "):]


# --- existing behavior must not regress ----------------------------------------
def test_existing_arithmetic_still_works():
    assert _val("What is 847 x 23?") == "19481"
    assert _val("Calculate 15% of 240") == "36"
    assert _val("What is 12 plus 30?") == "42"
    assert _val("compute (2 + 3) * 4") == "20"


def test_narrative_word_problems_still_return_none():
    # The core safety contract: anything with prose is None.
    assert solve("If John has 5 apples and eats 2, how many remain?") is None
    assert solve("What is 2 plus 2 apples?") is None
    assert solve("A train travels 60 miles in 2 hours, what is its speed?") is None
    assert solve("What is the capital of France?") is None
    assert solve("What is 42?") is None  # a lone number is not a calculation


# --- powers --------------------------------------------------------------------
def test_powers():
    assert _val("What is 5 squared?") == "25"
    assert _val("What is 3 cubed?") == "27"
    assert _val("Calculate 2 to the power of 10") == "1024"
    assert _val("What is 2 ^ 8?") == "256"


def test_powers_are_bounded_and_do_not_hang():
    # |exponent| <= 12, |base| <= 1e6 — anything larger must refuse, not compute.
    assert solve("What is 10 to the power of 1000?") is None
    assert solve("What is 999999999 squared?") is None


# --- roots ---------------------------------------------------------------------
def test_square_root_integer_only():
    assert _val("What is the square root of 144?") == "12"
    # Non-integer roots refuse rather than emit a rounded decimal.
    assert solve("What is the square root of 2?") is None


# --- word fractions ------------------------------------------------------------
def test_word_fractions():
    assert _val("What is half of 50?") == "25"
    assert _val("What is a third of 90?") == "30"
    assert _val("Calculate a quarter of 100") == "25"


# --- sum / product / difference / add / subtract -------------------------------
def test_sum_and_product_and_difference():
    assert _val("What is the sum of 12 and 30?") == "42"
    assert _val("What is the product of 6 and 7?") == "42"
    assert _val("What is the difference between 10 and 3?") == "7"
    assert _val("What is the difference between 3 and 10?") == "7"  # abs


def test_add_and_subtract_including_the_reversal():
    assert _val("Add 40 and 2") == "42"
    # 'subtract A from B' = B - A — the classic reversal bug.
    assert _val("Subtract 8 from 50") == "42"


# --- percent variants ----------------------------------------------------------
def test_percent_variants():
    assert _val("What is 20% off 50?") == "40"
    assert _val("Increase 200 by 10%") == "220"
    assert _val("Decrease 200 by 10%") == "180"


# --- modulo --------------------------------------------------------------------
def test_modulo_does_not_touch_bare_percent():
    assert _val("What is 17 mod 5?") == "2"
    assert _val("What is the remainder when 17 is divided by 5?") == "2"
    # bare % is still percent-of, never modulo
    assert _val("What is 50% of 8?") == "4"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("PASS", fn.__name__)
    print(f"\ntest_solver: {len(fns)} passed")
