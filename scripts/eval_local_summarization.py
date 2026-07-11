"""Throttled repro for run 18's -21 anomaly (docs/eval-results.md, run 18 lessons).

Hypothesis under test: long summarization passages make local prompt-eval slow on
the 2 vCPU grading box, blowing local.CALL_TIMEOUT (15s) -> LOCAL_FAIL -> silent
fail-open to remote, which is why widening local to summarization shed only 21
tokens instead of several hundred.

Runs the production local pipeline (local.generate + local.verify) on the
summarization eval set PLUS synthesized long passages, with the timeout raised to
120s so we observe TRUE latency instead of just "timed out", then compares each
call against the 15s production cap. Mirror the grading box first:

    docker run -d --name ollama-eval --cpus=2 -m 4g -p 11434:11434 \
        -v ollama-eval:/root/.ollama ollama/ollama
    docker exec ollama-eval ollama pull qwen2.5:3b

Usage: python scripts/eval_local_summarization.py [model]

Reading the report: items marked OVER 15s CAP are the ones production silently
pays remote tokens for. If only the long passages blow the cap, the fix is a
summarization-specific timeout (bounded by main.py's serialized local budget);
if even short ones do, the box is slower than assumed and local summarization
needs rethinking.
"""
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

MODEL = sys.argv[1] if len(sys.argv) > 1 else "qwen2.5:3b"
os.environ["LOCAL_MODEL"] = MODEL

from src import local  # noqa: E402

local.MODEL = MODEL

PARA = (
    "The regional transit authority announced a comprehensive overhaul of its bus "
    "network on Tuesday, the first major redesign in two decades. Under the plan, "
    "twelve low-ridership routes will be consolidated into six high-frequency "
    "corridors, with buses arriving every eight minutes during peak hours instead "
    "of every twenty. Officials said the change follows an eighteen-month study "
    "that logged boarding data at every stop and found that two thirds of riders "
    "were concentrated on one third of the routes. Community groups praised the "
    "frequency gains but warned that riders in outlying neighborhoods would face "
    "longer walks to reach the new corridors, in some cases more than half a mile. "
    "The authority responded by promising an on-demand shuttle pilot covering the "
    "three most affected districts, funded by a state accessibility grant. Fares "
    "will not change in the first year, though officials left open the possibility "
    "of a restructured zone system later. Construction of new shelters and signage "
    "begins in September, and the full network switch is scheduled for January, "
    "with a three-month transition period during which both old and new route "
    "numbers will be displayed. "
)

LONG_TASKS = [
    ("long-2x (~350 words)", "Summarize in one sentence: " + PARA * 2),
    ("long-3x (~520 words)", "Summarize in 2-3 sentences: " + PARA * 3),
    ("long-5x (~870 words)", "Summarize the main points: " + PARA * 5),
]


def run(label: str, prompt: str) -> None:
    t0 = time.monotonic()
    saved_first, saved_call = local.FIRST_CALL_TIMEOUT, local.CALL_TIMEOUT
    local.FIRST_CALL_TIMEOUT = local.CALL_TIMEOUT = 120.0
    try:
        text = local.generate(prompt, "summarization")
        latency = time.monotonic() - t0
        ok, reason = local.verify(prompt, "summarization", text)
        verdict = "would PASS" if ok else f"verifier reject: {reason}"
        over = "OVER 15s CAP -> LOCAL_FAIL in prod" if latency > 15 else "fits 15s cap"
        print(f"{label:22s} {latency:6.1f}s  [{over}]  {verdict}  answer={text[:80]!r}")
    except local.LocalError as e:
        print(f"{label:22s} {time.monotonic()-t0:6.1f}s  GEN_FAIL: {e}")
    finally:
        local.FIRST_CALL_TIMEOUT, local.CALL_TIMEOUT = saved_first, saved_call


def main() -> None:
    print(f"model={MODEL} (timeout raised to 120s to observe true latency; prod cap is 15s)")
    print("NOTE: the very first call also pays the ~2GB model load - that is the")
    print("cold-start number; compare it against FIRST_CALL_TIMEOUT (25s).")
    with open(REPO / "tests" / "eval" / "summarization.json", encoding="utf-8") as f:
        items = json.load(f)
    for i, item in enumerate(items[:4]):
        run(f"evalset-short [{i}]", item["prompt"])
    for label, prompt in LONG_TASKS:
        run(label, prompt)


if __name__ == "__main__":
    main()
