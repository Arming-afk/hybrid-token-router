"""Summarize per-category token usage from a run log containing USAGE lines."""
import json
import sys
from collections import defaultdict


def main(path: str) -> None:
    per_category = defaultdict(lambda: {"calls": 0, "prompt": 0, "completion": 0})
    with open(path, encoding="utf-8") as f:
        for line in f:
            marker = line.find("USAGE ")
            if marker == -1:
                continue
            entry = json.loads(line[marker + len("USAGE "):])
            row = per_category[entry["category"]]
            row["calls"] += 1
            row["prompt"] += entry["prompt_tokens"]
            row["completion"] += entry["completion_tokens"]

    total = 0
    print(f"{'category':<15}{'calls':>6}{'prompt':>9}{'compl':>9}{'total':>9}")
    for category, row in sorted(per_category.items()):
        subtotal = row["prompt"] + row["completion"]
        total += subtotal
        print(f"{category:<15}{row['calls']:>6}{row['prompt']:>9}{row['completion']:>9}{subtotal:>9}")
    print(f"{'TOTAL':<15}{'':>6}{'':>9}{'':>9}{total:>9}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "out/run.log")
