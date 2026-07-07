"""Person 2: measure pass rate and answer-token cost per (category, tier, template) config.

For each category it runs the eval set through candidate configs, grades each answer with
a cheap SMALL-model judge, and reports the cheapest config that clears the pass threshold.
The seeded configs target the three open questions:
  (a) math/logic  : MEDIUM with short steps   vs  LARGE answering directly
  (b) sentiment/ner/summarization : does SMALL hold, or is MEDIUM needed?
  (c) factual     : can it drop from MEDIUM to SMALL?

Judge tokens do NOT count toward the contest score; only the answer call's tokens do, so
ranking is by average answer tokens among configs that pass.

Prerequisite: run on integrated code (merge person1/router-tiering first) so build_tiers()
returns the corrected tiers, then:
    set -a; . ./.env; set +a
    python scripts/eval_matrix.py            # all categories
    python scripts/eval_matrix.py math logic # only some
"""
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import client, models  # noqa: E402
from src.prompts import CATEGORIES, SPEC  # noqa: E402

EVAL_DIR = ROOT / "tests" / "eval"
PASS_THRESHOLD = 0.90          # eval-set proxy for the real accuracy gate
CONCURRENCY = 8
RESULTS_DOC = ROOT / "docs" / "eval-results.md"

DIRECT_MATH = 'Give only the final result as "Answer: <value>", with no explanation.'
DIRECT_LOGIC = 'Give only the final answer as "Answer: <solution>", with no explanation.'


def _cfg(label: str, tier: str, instruction: str, max_tokens: int) -> dict:
    return {"label": label, "tier": tier, "instruction": instruction, "max_tokens": max_tokens}


def configs_for(cat: str) -> list[dict]:
    ins, mx = SPEC[cat]["instruction"], SPEC[cat]["max_tokens"]
    if cat == "math":
        return [_cfg("MEDIUM/steps", "MEDIUM", ins, mx), _cfg("LARGE/direct", "LARGE", DIRECT_MATH, 64)]
    if cat == "logic":
        return [_cfg("MEDIUM/steps", "MEDIUM", ins, mx), _cfg("LARGE/direct", "LARGE", DIRECT_LOGIC, 64)]
    if cat in ("debug", "codegen"):
        return [_cfg("MEDIUM/default", "MEDIUM", ins, mx), _cfg("SMALL/default", "SMALL", ins, mx)]
    # factual, sentiment, ner, summarization -> is SMALL enough, or is MEDIUM needed?
    return [_cfg("SMALL/default", "SMALL", ins, mx), _cfg("MEDIUM/default", "MEDIUM", ins, mx)]


def load_eval(cat: str) -> list[dict]:
    return json.loads((EVAL_DIR / f"{cat}.json").read_text(encoding="utf-8"))


async def judge(model: str, prompt: str, answer: str, expected: str) -> bool:
    messages = [{"role": "user", "content": (
        f"Task: {prompt}\nReference answer: {expected}\nCandidate answer: {answer}\n\n"
        "Does the candidate answer satisfy the reference? Answer yes or no."
    )}]
    text, _ = await client.complete(model, messages, 3)
    return text.strip().lower().startswith("y")


async def run_config(cfg: dict, items: list[dict], model: str, judge_model: str,
                     sem: asyncio.Semaphore) -> dict:
    async def one(item: dict) -> tuple[bool, int]:
        async with sem:
            messages = [{"role": "user", "content": f"{item['prompt']}\n\n{cfg['instruction']}"}]
            try:
                answer, usage = await client.complete(model, messages, cfg["max_tokens"])
            except Exception:
                return False, 0
            tokens = usage["prompt_tokens"] + usage["completion_tokens"]
            ok = await judge(judge_model, item["prompt"], answer, item["expected"])
            return ok, tokens

    outcomes = await asyncio.gather(*(one(i) for i in items))
    passed = sum(1 for ok, _ in outcomes if ok)
    tokens = [t for _, t in outcomes]
    return {
        "label": cfg["label"], "tier": cfg["tier"],
        "pass_rate": passed / len(outcomes),
        "avg_tokens": sum(tokens) / len(tokens),
    }


async def main(categories: list[str]) -> None:
    tiers = models.build_tiers()
    judge_model = tiers["SMALL"]
    sem = asyncio.Semaphore(CONCURRENCY)
    print(f"tiers={json.dumps(tiers)}  judge={judge_model}\n")

    lines = ["| category | config | model | pass rate | avg answer tokens | pick |",
             "|---|---|---|---|---|---|"]
    for cat in categories:
        items = load_eval(cat)
        rows = [await run_config(c, items, tiers[c["tier"]], judge_model, sem)
                for c in configs_for(cat)]
        passing = [r for r in rows if r["pass_rate"] >= PASS_THRESHOLD]
        best = min(passing, key=lambda r: r["avg_tokens"]) if passing else None
        for r in rows:
            pick = "  <= cheapest pass" if r is best else ("  (fails gate)" if r["pass_rate"] < PASS_THRESHOLD else "")
            lines.append(f"| {cat} | {r['label']} | {tiers[r['tier']]} | {r['pass_rate']:.0%} "
                         f"| {r['avg_tokens']:.0f} |{pick} |")
        if not passing:
            lines.append(f"| {cat} | — | — | — | — | NONE PASS: bump tier or add few-shot |")

    table = "\n".join(lines)
    print(table)
    _write_results(table)


def _write_results(table: str) -> None:
    start, end = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
    if not RESULTS_DOC.exists():
        return
    doc = RESULTS_DOC.read_text(encoding="utf-8")
    if start in doc and end in doc:
        head, rest = doc.split(start, 1)
        _, tail = rest.split(end, 1)
        RESULTS_DOC.write_text(f"{head}{start}\n\n{table}\n\n{end}{tail}", encoding="utf-8")
        print(f"\nwrote table into {RESULTS_DOC.relative_to(ROOT)}")


if __name__ == "__main__":
    cats = [c for c in sys.argv[1:] if c in CATEGORIES] or CATEGORIES
    asyncio.run(main(cats))
