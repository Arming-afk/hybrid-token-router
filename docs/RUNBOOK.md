# Submission runbook (Person 3)

The leaderboard scores the **LATEST run only** (unlimited total, **10/hour per team**).
This file is the operational side of that rule. The scored history itself lives in
`docs/eval-results.md` — this runbook only tells you what to do.

## Rollback anchor — memorize this

Best proven run: **run 18 — 100% (19/19), 4,178 tokens, gate PASSED.**

```
ghcr.io/arming-afk/hybrid-token-router:7bb50c4
```

Images are tagged by git SHA and immutable: **rolling back = submitting that existing
image reference on the portal. No rebuild, no merge, no waiting for CI.** Update this
anchor (and this file) whenever a newer run both passes the gate and costs fewer tokens.

Note (2026-07-11, run 20): re-saving this exact image scored 73.7% on the fresh judge
set, not 100% — the drop is the judge prompts changing, not this image regressing (see
`docs/eval-results.md` run 20 bisect verdict). It is still the right rollback anchor:
it is the newest config that is bit-identical to a proven-passing run, and the fresh-set
number moves with the judge set for every candidate image, not just this one.

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

**Deadline: 2026-07-12 18:00 ICT** (moved from the original 2026-07-11 23:00; see
`docs/ENDGAME-PLAN.md`). Scoring turnaround ~20 min.

- **16:30 ICT: last experimental submission goes out.** Nothing new after this.
- **17:00 ICT: submit the best-known config** (the anchor above) regardless of what the
  last experiment scored — do not wait for its result to come back first.
- Keep ≥2 submissions of the hourly quota in reserve during the final hour in case the
  final submission errors (PULL_ERROR etc.) and must be re-sent.
- After the final submission: verify the portal shows the expected image SHA. Touch
  nothing afterwards.

## Pre-submission audit (run before every submission, not just the final one)

```bash
docker run --rm --network=none --cpus=2 -m 4g \
  -v "$(pwd)/tests/sample_tasks.json:/input/tasks.json:ro" \
  -v "$(pwd)/out:/output" \
  hybrid-token-router:dev
```

- Must still write a valid `results.json` and exit 0. The only `ERROR` lines allowed
  are connection failures toward `FIREWORKS_BASE_URL` (with `--network=none` even that
  host is unreachable, so expect every remote call to fail — this checks that failure
  is handled gracefully, not that remote succeeds). Any other outbound attempt is an
  organizer-rule disqualification; localhost Ollama (`localhost:11434`) is explicitly
  exempt.
- `grep -rn "http" src/` may only surface the `FIREWORKS_BASE_URL` env var reference and
  `localhost:11434` — any other literal host is a red flag, stop and investigate before
  submitting.
- This audit was SKIPPED on 2026-07-11 during coder-rung validation (`docs/eval-results.md`
  "Coder-rung validation" section) because docker was unusable on that machine's
  connection at the time; the risk was accepted because the entrypoint/network shape was
  unchanged from prior graded images. Re-run it for real before the final submission if
  docker becomes available — do not let it stay skipped through the freeze window.

## Current state (update as runs land)

- Rollback anchor is run 18, `7bb50c4` (see above) — **100% (19/19) on the judge set as
  it stood 2026-07-10, 73.7% on the judge set as it stood 2026-07-11 (run 20 bisect);
  both numbers are the same bits, the difference is the prompts, not the image.**
- Submission queue per `docs/ENDGAME-PLAN.md`: (1) Phase A local-timeout fix — already
  shipped (`0f9c19a`) and folded into the coder-rung image `6e4fa3a`; (2) Phase C solver
  widening — already shipped (`df2f2c1`), also folded into `6e4fa3a`; (3) Phase B router
  audit — completed 2026-07-11, **no code change** (see `docs/eval-results.md`
  "Phase B: router audit on fresh wording" — zero DANGEROUS-FP found, router is not the
  cause of the run 19-21 losses); (4) Phase D scale rehearsal — see
  `docs/eval-results.md` for the latest 106-task throttled-docker measurement, if one
  landed.
- Open question carried into the extended window: run 21 (`6e4fa3a`, ALL-IN coder rung)
  scored 78.9% on the fresh judge set, still below the gate bracket `(78.9%, 84.2%]`.
  With Phase B clearing the router, the remaining suspects are local semantic misses
  (qwen2.5-coder:3b silently wrong under a verifier that checks format, not meaning) and
  genuine remote-model misses on the new phrasings.
