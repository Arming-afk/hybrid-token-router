"""Offline SEMANTIC eval of a local Ollama model on the debug/codegen eval sets.

Purpose (leaderboard note, docs/eval-results.md): breaking 3k tokens needs the code
categories local, and run 16 proved plain qwen2.5:3b can't carry them. A coder-tuned
sibling (e.g. qwen2.5-coder:3b) was never tested. This script tests any local model
OFFLINE — zero submissions, zero API tokens — before we risk a scored run.

For each eval item it runs the PRODUCTION local pipeline (local.generate with the
category instruction, then local.verify), and then goes one step further than the
in-image verifiers can: it EXECUTES the generated code against assertions derived
from the item's 'expected' — the semantic check whose absence sank run 16.
Per-call latency is recorded; to mirror the grading box, run Ollama throttled:

    docker run -d --name ollama-eval --cpus=2 -m 4g -p 11434:11434 \
        -v ollama-eval:/root/.ollama ollama/ollama
    docker exec ollama-eval ollama pull qwen2.5-coder:3b

Usage: python scripts/eval_local_code.py <model> [category ...]
       (default model qwen2.5:3b, categories debug codegen)

Reading the report: PASS = correct AND verifier-approved (safe to serve locally).
SEMANTIC_MISS = wrong answer the verifiers would have submitted — the dangerous
class; more than ~1/10 per category means do NOT widen local to it.
VERIFIER_REJECT / GEN_FAIL = falls back to remote in production (safe, just not
free). A model is submission-worthy for a category at roughly >=9/10 PASS.
"""
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

MODEL = sys.argv[1] if len(sys.argv) > 1 else "qwen2.5:3b"
CATEGORIES = sys.argv[2:] or ["debug", "codegen"]
os.environ["LOCAL_MODEL"] = MODEL

from src import local  # noqa: E402

local.MODEL = MODEL  # the module reads env at import time; force the override

PICK_HELPER = """
def _pick(name=None):
    cands = {k: v for k, v in list(globals().items())
             if callable(v) and not k.startswith('_') and type(v).__name__ == 'function'}
    if name and name in cands:
        return cands[name]
    return list(cands.values())[-1]
"""

# index -> ("exec", assertions) | ("grep", regex) | ("execwrap", template).
# Keyed by position in tests/eval/<category>.json; update together with those files.
CHECKS = {
    "debug": {
        0: ("exec", "assert total([1,2,3]) == 6"),
        1: ("grep", r"a\s*\+\s*b"),
        2: ("execwrap", "nums=[1,2,3,4,5,6]\n{code}\nassert evens == [2,4,6]"),
        3: ("exec", "assert rev('abc') == 'cba'"),
        4: ("grep", r"i\s*<\s*arr\.length"),
        5: ("exec", "assert first([]) is None\nassert first([5]) == 5"),
        6: ("exec", "assert avg([2,4]) == 3\navg([])"),
        7: ("grep", r"count\s*(\+=\s*1|=\s*count\s*\+\s*1)"),
        8: ("exec", "assert fact(5) == 120\nassert fact(0) == 1"),
        9: ("exec", "assert get_count({}, 'k') == 0\nassert get_count({'k':3}, 'k') == 3"),
    },
    "codegen": {
        0: ("exec", "f=_pick('factorial')\nassert f(5)==120\nassert f(0)==1"),
        1: ("exec", "f=_pick('is_palindrome')\nassert f('racecar')\nassert not f('hello')"),
        2: ("exec", "f=_pick()\nassert f([3,7,2])==7"),
        3: ("exec", "f=_pick('fizzbuzz')\nassert f(3)=='Fizz'\nassert f(5)=='Buzz'\nassert f(15)=='FizzBuzz'\nassert f(7) in (7,'7')"),
        4: ("exec", "f=_pick()\nassert f('hello')==2"),
        5: ("exec", "f=_pick()\nassert f([1,3],[2,4])==[1,2,3,4]"),
        6: ("exec", "f=_pick('nth_fibonacci')\nassert f(6)==8\nassert f(0)==0\nassert f(1)==1"),
        7: ("exec", "f=_pick()\nassert f([1,2,1,3,2])==[1,2,3]"),
        8: ("exec", "f=_pick('is_prime')\nassert f(7)\nassert not f(9)\nassert not f(1)"),
        9: ("exec", "f=_pick()\nassert f([2,None,4])==3\nf([])\nf([None])"),
    },
}


def run_code_check(kind: str, spec: str, answer: str) -> tuple[bool, str]:
    code = local._extract_code(answer)
    if kind == "grep":
        target = code or answer
        hit = bool(re.search(spec, target))
        return hit, "pattern ok" if hit else "pattern missing"
    if not code:
        return False, "no code block"
    if kind == "execwrap":
        script = spec.replace("{code}", code) + "\nprint('SEMANTIC_OK')"
    else:
        script = code + "\n" + PICK_HELPER + "\n" + spec + "\nprint('SEMANTIC_OK')"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(script)
        path = f.name
    try:
        proc = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=8)
        if "SEMANTIC_OK" in proc.stdout:
            return True, "ok"
        err = (proc.stderr.strip().splitlines() or ["no output"])[-1]
        return False, err[:100]
    except subprocess.TimeoutExpired:
        return False, "execution timeout"
    finally:
        os.unlink(path)


def main() -> None:
    print(f"model={MODEL} categories={CATEGORIES}")
    grand = {}
    for category in CATEGORIES:
        with open(REPO / "tests" / "eval" / f"{category}.json", encoding="utf-8") as f:
            items = json.load(f)
        rows = []
        for i, item in enumerate(items):
            t0 = time.monotonic()
            try:
                answer = local.generate(item["prompt"], category)
                latency = time.monotonic() - t0
            except local.LocalError as e:
                rows.append((i, "GEN_FAIL", time.monotonic() - t0, str(e)[:80]))
                continue
            ok, reason = local.verify(item["prompt"], category, answer)
            if not ok:
                rows.append((i, "VERIFIER_REJECT", latency, reason))
                continue
            kind, spec = CHECKS[category][i]
            sem_ok, sem_reason = run_code_check(kind, spec, answer)
            rows.append((i, "PASS" if sem_ok else "SEMANTIC_MISS", latency, sem_reason))
        passes = sum(1 for r in rows if r[1] == "PASS")
        misses = sum(1 for r in rows if r[1] == "SEMANTIC_MISS")
        rejects = sum(1 for r in rows if r[1] in ("VERIFIER_REJECT", "GEN_FAIL"))
        lat = [r[2] for r in rows]
        grand[category] = {"pass": passes, "semantic_miss": misses,
                           "safe_fallback": rejects, "n": len(items)}
        print(f"\n== {category}: {passes}/{len(items)} pass | {misses} SEMANTIC MISSES (dangerous) | "
              f"{rejects} rejected->remote (safe) | latency avg {sum(lat)/len(lat):.1f}s max {max(lat):.1f}s")
        for i, status, latency, reason in rows:
            print(f"  [{i}] {status:16s} {latency:5.1f}s  {reason}")
    print("\nSUMMARY " + json.dumps(grand))


if __name__ == "__main__":
    main()
