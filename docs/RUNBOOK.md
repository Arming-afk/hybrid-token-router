# Submission runbook (Person 3)

The leaderboard scores the **LATEST run only** (unlimited total, **10/hour per team**).
This file is the operational side of that rule. The scored history itself lives in
`docs/eval-results.md` — this runbook only tells you what to do.

## Rollback anchor — memorize this

Best proven run: **run 5 — 84.2%, 5,273 tokens, gate PASSED.**

```
ghcr.io/arming-afk/hybrid-token-router:6f01e64
```

Images are tagged by git SHA and immutable: **rolling back = submitting that existing
image reference on the portal. No rebuild, no merge, no waiting for CI.** Update this
anchor (and this file) whenever a newer run both passes the gate and costs fewer tokens.

## Experiment cycle (one cut per run)

1. Merge exactly one instruction/config change to `main` → CI pushes `ghcr.io/...:<sha>`.
2. Submit that image. Record `(sha, config, accuracy, tokens)` in `eval-results.md`
   **immediately** — the token delta vs the previous run is that category's cost share.
3. Score comes back:
   - **Gate passed, tokens down** → this run is the new rollback anchor; update this file.
   - **Anything else** → submit the rollback anchor image NOW, before analyzing anything.
     Never leave a failed experiment as the latest run.
4. Budget check: each failed experiment consumes 2 submissions (experiment + rollback).
   At 10/hour that is at most ~5 experiment cycles per hour — plan sequences, not bursts.

## Freeze window (final hours before the deadline)

- **T−90 min: last experimental submission goes out.** Nothing new after this.
- **T−60 min: submit the best-known config** (the anchor above) regardless of what the
  last experiment scored — do not wait for its result to come back first.
- Keep ≥2 submissions of the hourly quota in reserve during the final hour in case the
  final submission errors (PULL_ERROR etc.) and must be re-sent.
- After the final submission: verify the portal shows the expected image SHA. Touch
  nothing afterwards.

## Current state (update as runs land)

- `main` right now = run-5 config **+ cut #1 (input filler trim, unmeasured in
  isolation)**. The next submission of `main` doubles as cut #1's isolated measurement:
  expected ≈ run 5's accuracy with a ~13-token saving. If it scores below run 5's
  84.2%, cut #1 was not neutral — revert it and re-anchor on `6f01e64`.
