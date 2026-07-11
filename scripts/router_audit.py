"""Router stress-test report: per-category accuracy, misroute confusion matrix,
danger partition, and the deterministic-solver false-positive cross-check.

Usage:
    python scripts/router_audit.py [case_file ...]

Defaults to tests/router_cases.json (the 187-case dev set) when no file is given.
Pass multiple files (e.g. tests/router_cases.json tests/router_cases_fresh.json)
to audit each independently and get a combined solver cross-check across all of
them. Exit code is non-zero only if the solver cross-check finds a false
positive (a non-math prompt the deterministic solver answers) — that is a
correctness bug regardless of which file it came from. Per-file accuracy is
reported but does not fail the script: the dev set's 95%/10% thresholds are
enforced by tests/test_router.py; router_cases_fresh.json is a stress probe,
not a gate.
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.router import classify, deterministic_math_answer  # noqa: E402

DANGEROUS_TARGETS = {"sentiment", "ner", "summarization", "debug", "codegen"}
SAFE_TARGETS = {"factual"}
DEFAULT_FILE = "tests/router_cases.json"


def load_cases(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def audit_file(path: str) -> list[dict]:
    cases = load_cases(path)
    wrong, fallback = [], []
    confusion: Counter = Counter()
    danger, safe, other = [], [], []
    per_category_total: Counter = Counter()
    per_category_wrong: Counter = Counter()

    for case in cases:
        expected, prompt = case["category"], case["prompt"]
        per_category_total[expected] += 1
        got = classify(prompt)
        if got is None:
            fallback.append(prompt[:70])
            continue
        if got != expected:
            wrong.append((prompt[:70], expected, got))
            per_category_wrong[expected] += 1
            confusion[(expected, got)] += 1
            entry = f"{expected} -> {got}: {prompt[:70]!r}"
            if got in DANGEROUS_TARGETS:
                danger.append(entry)
            elif got in SAFE_TARGETS:
                safe.append(entry)
            else:
                other.append(entry)

    decided = len(cases) - len(fallback)
    accuracy = (decided - len(wrong)) / decided if decided else 0.0
    fallback_rate = len(fallback) / len(cases) if cases else 0.0

    print(f"\n=== {path} ===")
    print(f"cases={len(cases)} decided={decided} accuracy={accuracy:.1%} "
          f"fallback={len(fallback)}/{len(cases)} ({fallback_rate:.1%})")
    gate = accuracy >= 0.95 and fallback_rate <= 0.10
    print(f"dev-set gate (>=95% decided-accuracy, <=10% fallback): "
          f"{'PASS' if gate else 'FAIL (informational on non-dev files)'}")

    print("\nPer-category accuracy:")
    for cat in sorted(per_category_total):
        total = per_category_total[cat]
        w = per_category_wrong[cat]
        print(f"  {cat:15s} {total - w:3d}/{total:3d}  ({(total - w) / total:.1%})")

    if confusion:
        print("\nConfusion matrix (expected -> got: count):")
        for (expected, got), count in sorted(confusion.items(), key=lambda x: -x[1]):
            print(f"  {expected:15s} -> {got:15s}: {count}")

    print(f"\nDanger partition: DANGEROUS-FP={len(danger)}  SAFE={len(safe)}  "
          f"OTHER(math/logic)={len(other)}")
    if danger:
        print("  DANGEROUS-FP cases (misrouted into a narrow-format category):")
        for entry in danger:
            print(f"    {entry}")
    if fallback:
        print(f"  fallback (deferred to LLM, not a misroute): {len(fallback)} case(s)")

    return [{"category": c["category"], "prompt": c["prompt"]} for c in cases]


def solver_cross_check(all_cases: list[dict]) -> bool:
    """Any non-math prompt the deterministic solver answers is a false positive:
    it would silently substitute a wrong/irrelevant number for real classification
    and answering. Must be empty."""
    false_positives = []
    for case in all_cases:
        if case["category"] == "math":
            continue
        answer = deterministic_math_answer(case["prompt"])
        if answer is not None:
            false_positives.append((case["category"], case["prompt"][:70], answer))

    print("\n=== Deterministic solver cross-check (combined across all files) ===")
    if false_positives:
        print(f"FAIL: {len(false_positives)} non-math prompt(s) answered by the solver:")
        for cat, prompt, answer in false_positives:
            print(f"  [{cat}] {prompt!r} -> {answer}")
        return False
    print(f"PASS: solver returned None on all {sum(1 for c in all_cases if c['category'] != 'math')} "
          f"non-math prompts across {len(all_cases)} total cases.")
    return True


def main() -> int:
    files = sys.argv[1:] or [DEFAULT_FILE]
    all_cases: list[dict] = []
    for path in files:
        all_cases.extend(audit_file(path))
    ok = solver_cross_check(all_cases)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
