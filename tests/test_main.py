"""Failure drills for the entrypoint (README checklist #4), no network needed.

The invariant under test: /output/results.json exists, is valid JSON, and carries
every task_id — no matter what the API, the input file, or the clock does.
"""
import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("ALLOWED_MODELS", "acc/tiny-1b,acc/mid-7b,acc/big-70b")

from src import client, main, router  # noqa: E402


async def echo_complete(model, messages, max_tokens):
    return f"answer from {model}", {"prompt_tokens": 1, "completion_tokens": 1}


last_run_dir = None  # tmp dir of the most recent run_pipeline call, for log assertions


def run_pipeline(tasks_payload, fake_complete):
    """Run main.main() against a fake client.complete; return parsed results.json."""
    global last_run_dir
    tmp = tempfile.mkdtemp()
    last_run_dir = tmp
    in_path, out_path = os.path.join(tmp, "tasks.json"), os.path.join(tmp, "results.json")
    with open(in_path, "w", encoding="utf-8") as f:
        json.dump(tasks_payload, f)
    saved = (client.complete, main.INPUT_PATH, main.OUTPUT_PATH)
    client.complete, main.INPUT_PATH, main.OUTPUT_PATH = fake_complete, in_path, out_path
    main.CALL_LOG.clear()
    try:
        main.main()
        with open(out_path, encoding="utf-8") as f:
            return json.load(f)
    finally:
        client.complete, main.INPUT_PATH, main.OUTPUT_PATH = saved


def test_happy_path_answers_every_task():
    tasks = [{"task_id": "a", "prompt": "What is the capital of France?"},
             {"task_id": "b", "prompt": "Summarize this in one sentence: hello."}]
    rows = run_pipeline(tasks, echo_complete)
    assert [r["task_id"] for r in rows] == ["a", "b"]
    assert all(r["answer"].startswith("answer from ") for r in rows)


def test_inference_log_written_next_to_results():
    tasks = [{"task_id": "a", "prompt": "What is the capital of France?"}]
    run_pipeline(tasks, echo_complete)
    with open(os.path.join(last_run_dir, "inference_log.json"), encoding="utf-8") as f:
        log = json.load(f)
    assert log["totals"]["calls"] == 1
    assert log["calls"][0]["task_id"] == "a"
    assert "prompt_tokens" in log["calls"][0]


def test_empty_tasks_file_writes_empty_list():
    assert run_pipeline([], echo_complete) == []


def test_all_calls_failing_still_writes_all_ids():
    async def broken(model, messages, max_tokens):
        raise RuntimeError("bad api key")
    rows = run_pipeline([{"task_id": "a", "prompt": "x"}, {"task_id": "b", "prompt": "y"}],
                        broken)
    assert [r["task_id"] for r in rows] == ["a", "b"]
    assert all(r["answer"] == "" for r in rows)


def test_malformed_entries_do_not_crash():
    tasks = [
        {"task_id": "good", "prompt": "What is 2?"},
        {"task_id": "no-prompt"},
        {"prompt": "no task_id"},
        "not even a dict",
        {"task_id": "blank-prompt", "prompt": "   "},
    ]
    rows = run_pipeline(tasks, echo_complete)
    assert [r["task_id"] for r in rows] == ["good", "no-prompt", "blank-prompt"]
    by_id = {r["task_id"]: r["answer"] for r in rows}
    assert by_id["good"] and by_id["no-prompt"] == "" and by_id["blank-prompt"] == ""


def test_empty_completion_retries_on_larger_tier():
    calls = []

    async def empty_then_text(model, messages, max_tokens):
        calls.append(model)
        return ("" if len(calls) == 1 else "recovered"), {}

    rows = run_pipeline([{"task_id": "a", "prompt": "What is the capital of France?"}],
                        empty_then_text)
    assert rows[0]["answer"] == "recovered"
    assert calls == ["acc/big-70b", "acc/mid-7b"]  # factual=LARGE, retry falls back to MEDIUM


def test_ambiguous_prompt_pays_llm_classifier_first():
    calls = []
    saved_classify = router.classify
    router.classify = lambda prompt: None  # force every task down the fallback path

    async def classifier_then_answer(model, messages, max_tokens):
        calls.append((model, max_tokens))
        return ("B" if max_tokens == 2 else "42"), {}

    try:
        rows = run_pipeline([{"task_id": "a", "prompt": "mystery"}], classifier_then_answer)
    finally:
        router.classify = saved_classify
    assert calls[0] == ("acc/tiny-1b", 2)  # 2-token classify on SMALL
    assert rows[0]["answer"] == "42"


def test_missing_allowed_models_still_writes_output():
    saved = os.environ.pop("ALLOWED_MODELS")
    try:
        rows = run_pipeline([{"task_id": "a", "prompt": "x"}], echo_complete)
    finally:
        os.environ["ALLOWED_MODELS"] = saved
    assert rows == [{"task_id": "a", "answer": ""}]


def test_deadline_writes_partial_results():
    async def stalls(model, messages, max_tokens):
        await asyncio.sleep(30)
        return "too late", {}

    saved_deadline = main.DEADLINE_SECONDS
    main.DEADLINE_SECONDS = (time.monotonic() - main.START) + 0.3  # ~0.3s of budget left
    try:
        rows = run_pipeline([{"task_id": "a", "prompt": "x"}], stalls)
    finally:
        main.DEADLINE_SECONDS = saved_deadline
    assert rows == [{"task_id": "a", "answer": ""}]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("PASS", fn.__name__)
    print(f"\ntest_main: {len(fns)} passed")
