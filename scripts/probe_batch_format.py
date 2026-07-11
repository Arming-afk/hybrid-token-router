"""Probe the PUBLIC Fireworks endpoint with a real multi-task batch prompt.

Answers one question before any of src/batching.py or the src/main.py rewrite
gets written: does a real model actually follow our @@TASK@@/@@ANSWER@@ marker
format when asked to answer several tasks in one completion? This never runs
inside the graded pipeline or the Docker image -- it is a manual, one-off
diagnostic, the same role scratchpad/probe_tokens.py played for the earlier
per-category token-cost probe (docs/eval-results.md, "Token-cost probe on the
public endpoint").

Track 1's contest model IDs (kimi-k2p7-code, minimax-m3, ...) only exist behind
the judging proxy and 404 on the public API (see .env's own note on this) -- this
probe therefore calls a same-family public model as an approximation. Treat
results as a strong signal, not ground truth for the judging proxy itself.

Usage:
    python scripts/probe_batch_format.py
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openai import OpenAI  # noqa: E402

from src import batching  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# Public-endpoint-reachable candidates (Track 1's own model IDs 404 here). Same
# family as our production CODE tier (kimi) first, general model as fallback.
PROBE_MODELS = [
    "accounts/fireworks/models/kimi-k2p6",
    "accounts/fireworks/models/deepseek-v4-pro",
]


def _load_dotenv() -> None:
    """Populate os.environ from .env for any var not already set (env wins)."""
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def run_probe(client: OpenAI, model: str, label: str, chunk: list[dict]) -> None:
    # Reuses the EXACT production format/parsing code (src/batching.py), so this
    # probe keeps testing the real thing if that module ever changes.
    messages, max_tokens = batching.build_batch_messages(chunk)
    print(f"\n=== {label} ({len(chunk)} tasks) on {model} ===")
    print(f"requested max_tokens={max_tokens}")
    started = time.monotonic()
    try:
        # reasoning_effort="none": matches src/client.py's production call exactly.
        # Without it, a reasoning-capable model can burn the whole max_tokens budget
        # on visible chain-of-thought instead of the structured answer (this is the
        # documented reason client.py sends it -- see src/client.py's REASONING_EFFORT
        # comment).
        response = client.chat.completions.create(
            model=model, messages=messages, max_tokens=max_tokens, temperature=0,
            reasoning_effort="none",
        )
    except Exception as error:
        print(f"CALL FAILED: {str(error)[:300]}")
        return
    elapsed = time.monotonic() - started
    text = response.choices[0].message.content or ""
    usage = response.usage
    print(f"elapsed={elapsed:.1f}s finish_reason={response.choices[0].finish_reason} "
          f"prompt_tokens={usage.prompt_tokens} completion_tokens={usage.completion_tokens}")
    print("--- raw response ---")
    print(text)
    print("--- parsed slices ---")
    task_ids = [item["task_id"] for item in chunk]
    parsed = batching.parse_batch_response(text, task_ids)
    for item in chunk:
        slice_ = parsed[item["task_id"]]
        if slice_["complete"]:
            preview = slice_["text"][:100].replace("\n", " / ")
            print(f"  {item['task_id']} ({item['category']}): FOUND "
                  f"({len(slice_['text'])} chars) -> {preview!r}")
        else:
            print(f"  {item['task_id']} ({item['category']}): MISSING/INCOMPLETE "
                  f"(recovered text: {slice_['text'][:60]!r})")
        if slice_["complete"]:
            ok, reason = batching.verify_slice(item["prompt"], item["category"], slice_["text"])
            print(f"      verify_slice: {'OK' if ok else 'REJECTED'} ({reason})")


def _pick_model(client: OpenAI) -> str | None:
    for candidate in PROBE_MODELS:
        try:
            client.chat.completions.create(
                model=candidate, messages=[{"role": "user", "content": "hi"}], max_tokens=1,
            )
            return candidate
        except Exception as error:
            print(f"{candidate}: unreachable ({str(error)[:120]})")
    return None


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _load_dotenv()
    for key in ("FIREWORKS_API_KEY", "FIREWORKS_BASE_URL"):
        if not os.environ.get(key):
            print(f"Missing {key} -- set it in .env or the environment.", file=sys.stderr)
            sys.exit(1)

    client = OpenAI(api_key=os.environ["FIREWORKS_API_KEY"], base_url=os.environ["FIREWORKS_BASE_URL"])

    model = _pick_model(client)
    if model is None:
        print("\nNo probe model reachable on the public endpoint -- listing what IS available...")
        try:
            for m in client.models.list().data:
                print(" ", m.id)
        except Exception as error:
            print(f"models.list() also failed: {str(error)[:200]}")
        sys.exit(1)
    print(f"Using public-endpoint model: {model} (approximation of the judging proxy's model)")

    def _t(task_id, category, prompt):
        return {"task_id": task_id, "category": category, "prompt": prompt}

    code_tasks = [
        _t("t1", "factual", "Explain how a hash table achieves average O(1) lookup."),
        _t("t2", "math", "A laptop costs $1,200 and is discounted 15%, then a 7% tax "
                         "is added. Calculate the final price."),
        _t("t7", "logic", "Three colleagues sit in a row. Priya is not on the left end. "
                          "Marco is directly right of Priya. Jonas is not in the middle. "
                          "Who sits where?"),
    ]
    small_tasks = [
        _t("t3", "sentiment", "Classify the sentiment of this review and justify: "
                              "'Setup took five minutes and it has run flawlessly since.'"),
        _t("t4", "summarization", "Summarize the following in one sentence: The city "
                                  "council voted on Tuesday to expand the bike lane "
                                  "network by 40 kilometres over the next two years, "
                                  "citing a 30 percent rise in cycling commuters since "
                                  "2024 and pressure from residents to reduce downtown "
                                  "congestion."),
        _t("t5", "ner", "Extract all named entities (person, organization, location, "
                        "date) from: On 14 February, Elena Ruiz of Nordwind AG "
                        "presented in Oslo."),
    ]

    run_probe(client, model, "CODE-tier batch (factual+math+logic)", code_tasks)
    run_probe(client, model, "SMALL-tier batch (sentiment+summarization+ner)", small_tasks)


if __name__ == "__main__":
    main()
