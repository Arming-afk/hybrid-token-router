"""Person 1, Task 3: measure the 2-token LLM fallback classifier on the SMALL model.

The router defers ambiguous prompts to a 2-max-token letter classification on the SMALL
model. This script checks how trustworthy that is by running it over the labeled dev set
and comparing the parsed letter to the gold category. Gate: >=80% accuracy. Below that,
rework fallback_messages() (keeping it short) or accept more misroutes.

Run (needs a real FIREWORKS_BASE_URL):
    set -a; . ./.env; set +a
    python scripts/measure_fallback.py            # all dev cases
    python scripts/measure_fallback.py 5          # 5 per category (fewer tokens)
"""
import asyncio
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import client, models, router  # noqa: E402

CATEGORY_TO_LETTER = {v: k for k, v in router.LETTER_TO_CATEGORY.items()}


def load(per_category: int | None) -> list[dict]:
    cases = json.loads((Path(__file__).resolve().parents[1] / "tests" / "router_cases.json")
                        .read_text(encoding="utf-8"))
    if per_category is None:
        return cases
    seen: Counter = Counter()
    picked = []
    for c in cases:
        if seen[c["category"]] < per_category:
            picked.append(c)
            seen[c["category"]] += 1
    return picked


async def main(per_category: int | None) -> None:
    small = models.build_tiers()["SMALL"]
    cases = load(per_category)
    print(f"model={small}  cases={len(cases)}\n")

    async def classify_one(case: dict) -> tuple[str, str, int]:
        text, usage = await client.complete(small, router.fallback_messages(case["prompt"]), 2)
        got = router.parse_fallback_letter(text)
        return case["category"], got, usage["prompt_tokens"] + usage["completion_tokens"]

    results = await asyncio.gather(*(classify_one(c) for c in cases))

    correct = sum(1 for gold, got, _ in results if gold == got)
    tokens = sum(t for _, _, t in results)
    confusion: dict[str, Counter] = defaultdict(Counter)
    for gold, got, _ in results:
        confusion[gold][got] += 1

    print("per-category accuracy (gold -> predictions):")
    for gold in router.LETTER_TO_CATEGORY.values():
        preds = confusion[gold]
        n = sum(preds.values())
        if not n:
            continue
        hit = preds[gold]
        wrong = {k: v for k, v in preds.items() if k != gold}
        print(f"  {gold:14s} {hit}/{n}" + (f"  miss->{wrong}" if wrong else ""))

    acc = correct / len(results)
    print(f"\noverall accuracy = {acc:.1%} ({correct}/{len(results)})   tokens = {tokens}")
    print("GATE: PASS (>=80%)" if acc >= 0.80 else "GATE: FAIL (<80%) -- rework fallback prompt")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else None
    asyncio.run(main(n))
