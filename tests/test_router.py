"""Router accuracy gate: >=95% correct on decided cases, <=10% falling back to the LLM."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.router import classify  # noqa: E402


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


if __name__ == "__main__":
    test_router_accuracy()
    print("router dev-set: PASS")
