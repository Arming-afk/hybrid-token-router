"""Summarize a run log: per-category token usage, and the local-inference funnel
(DETERMINISTIC / LOCAL / LOCAL_FAIL->REMOTE / LOCAL_REJECTED->REMOTE /
LOCAL_SKIP->REMOTE / REMOTE / NO_ANSWER per task) with fail/reject reason counts.

Usage:
    python scripts/token_report.py [path] [--per-task]

Reads stderr-style log lines of the form `EVENT {"task_id": ..., ...}` as emitted
by src/main.py: USAGE, LOCAL, LOCAL_FAIL, LOCAL_REJECTED, LOCAL_SKIP, ERROR.
"""
import argparse
import json
import sys
from collections import defaultdict

EVENTS = ("USAGE", "LOCAL", "LOCAL_FAIL", "LOCAL_REJECTED", "LOCAL_SKIP", "ERROR")

# Local-path events are mutually exclusive per task (try_local makes one attempt):
# whichever of these appears is the outcome of the local leg, if any was tried.
LOCAL_LEG_EVENTS = ("LOCAL_FAIL", "LOCAL_REJECTED", "LOCAL_SKIP")


def parse_lines(path: str) -> list[tuple[str, dict]]:
    events = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            for event in EVENTS:
                marker = line.find(event + " {")
                if marker == -1:
                    continue
                try:
                    payload = json.loads(line[marker + len(event) + 1:])
                except json.JSONDecodeError:
                    continue
                events.append((event, payload))
                break
    return events


def build_task_index(events: list[tuple[str, dict]]) -> dict[str, list[tuple[str, dict]]]:
    by_task = defaultdict(list)
    for event, payload in events:
        task_id = payload.get("task_id")
        if task_id is not None:
            by_task[task_id].append((event, payload))
    return by_task


def task_summary(task_events: list[tuple[str, dict]]) -> dict:
    """Resolve one task's category, disposition, and token totals from its events."""
    prompt_tokens = sum(p.get("prompt_tokens", 0) for e, p in task_events if e == "USAGE")
    completion_tokens = sum(p.get("completion_tokens", 0) for e, p in task_events if e == "USAGE")

    category = None
    for event, payload in task_events:
        cat = payload.get("category")
        if cat and cat != "router":
            category = cat
            break

    non_router_usage = [p for e, p in task_events if e == "USAGE" and p.get("category") != "router"]
    det = next((p for e, p in task_events if e == "LOCAL" and p.get("solver")), None)
    local_ok = next((p for e, p in task_events if e == "LOCAL" and not p.get("solver")), None)
    leg = next(((e, p) for e, p in task_events if e in LOCAL_LEG_EVENTS), None)

    if det is not None:
        disposition = "DETERMINISTIC"
    elif local_ok is not None:
        disposition = "LOCAL"
    elif leg is not None:
        event, payload = leg
        landed_remote = bool(non_router_usage)
        disposition = f"{event}->REMOTE" if landed_remote else "NO_ANSWER"
    elif non_router_usage:
        disposition = "REMOTE"
    else:
        disposition = "NO_ANSWER"

    reason = None
    if leg is not None:
        event, payload = leg
        reason = payload.get("error") or payload.get("reason") or payload.get("note")

    return {
        "category": category or "unknown",
        "disposition": disposition,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "reason": reason,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", default="out/run.log")
    parser.add_argument("--per-task", action="store_true",
                        help="print one row per task_id with its resolved disposition")
    args = parser.parse_args()

    events = parse_lines(args.path)
    by_task = build_task_index(events)
    tasks = {task_id: task_summary(task_events) for task_id, task_events in by_task.items()}

    # --- classic per-category token table (kept for backward compatibility) -------
    per_category = defaultdict(lambda: {"calls": 0, "prompt": 0, "completion": 0})
    for event, payload in events:
        if event != "USAGE":
            continue
        row = per_category[payload["category"]]
        row["calls"] += 1
        row["prompt"] += payload.get("prompt_tokens", 0)
        row["completion"] += payload.get("completion_tokens", 0)

    total = 0
    print(f"{'category':<15}{'calls':>6}{'prompt':>9}{'compl':>9}{'total':>9}")
    for category, row in sorted(per_category.items()):
        subtotal = row["prompt"] + row["completion"]
        total += subtotal
        print(f"{category:<15}{row['calls']:>6}{row['prompt']:>9}{row['completion']:>9}{subtotal:>9}")
    print(f"{'TOTAL':<15}{'':>6}{'':>9}{'':>9}{total:>9}")

    # --- local-inference funnel, per category ---------------------------------
    funnel = defaultdict(lambda: defaultdict(int))
    funnel_tokens = defaultdict(int)
    for summary in tasks.values():
        funnel[summary["category"]][summary["disposition"]] += 1
        funnel_tokens[summary["category"]] += summary["prompt_tokens"] + summary["completion_tokens"]

    dispositions = ["DETERMINISTIC", "LOCAL", "LOCAL_FAIL->REMOTE", "LOCAL_REJECTED->REMOTE",
                    "LOCAL_SKIP->REMOTE", "REMOTE", "NO_ANSWER"]
    print(f"\n{'category':<15}" + "".join(f"{d:>25}" for d in dispositions) + f"{'tokens':>9}")
    for category in sorted(funnel):
        row = funnel[category]
        print(f"{category:<15}" + "".join(f"{row.get(d, 0):>25}" for d in dispositions)
              + f"{funnel_tokens[category]:>9}")

    # --- fail/reject reason histogram (the Phase A decision instrument) -------
    reasons = defaultdict(int)
    for summary in tasks.values():
        if summary["disposition"] not in ("DETERMINISTIC", "LOCAL", "REMOTE") and summary["reason"]:
            reasons[summary["reason"]] += 1
    if reasons:
        print("\nLocal fail/reject/skip reasons:")
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            print(f"  {count:3d}x  {reason}")

    if args.per_task:
        print(f"\n{'task_id':<24}{'category':<15}{'disposition':<24}{'tokens':>8}  reason")
        for task_id, summary in sorted(tasks.items()):
            tok = summary["prompt_tokens"] + summary["completion_tokens"]
            print(f"{task_id:<24}{summary['category']:<15}{summary['disposition']:<24}"
                  f"{tok:>8}  {summary['reason'] or ''}")


if __name__ == "__main__":
    main()
