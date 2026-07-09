# hybrid-token-router

AMD hackathon Track 1 entry: answer 8 task categories through Fireworks AI with the
fewest total tokens. Scoring is a two-stage gate — pass the LLM-Judge accuracy
threshold, then rank ascending by tokens counted at the judging proxy.

## How it works

1. `src/main.py` reads `/input/tasks.json`, classifies every task locally (free),
   fires all API calls concurrently, and always writes valid `/output/results.json`
   before an 8.5-minute internal deadline (harness kills at 10).
2. `src/router.py` picks one of 8 categories with regexes; only ambiguous prompts
   pay for a 2-token LLM classification on the smallest model.
3. `src/models.py` sorts `ALLOWED_MODELS` (published on launch day) by parsed
   parameter count into SMALL/MEDIUM/LARGE tiers, plus a CODE tier for
   code-specialized models (serves debug/codegen; falls back to MEDIUM).
4. `src/prompts.py` maps each category to a tier, a terse instruction, and a
   `max_tokens` cap. **This file is the main tuning surface.**

## Team ownership

| Area | Files | Owner |
|---|---|---|
| Router & tiering | `router.py`, `models.py`, `tests/router_cases.json` | Person 1 |
| Prompts & model selection | `prompts.py` + per-category eval sets | Person 2 |
| Infra & submission | `main.py`, `client.py`, `Dockerfile`, `scripts/` | Person 3 |

Shared interfaces (change only by agreement): the 8 category names in
`prompts.CATEGORIES`, `models.build_tiers() -> dict`, and
`prompts.render(category, prompt) -> (messages, max_tokens, tier)`.

## Run locally

Without Docker (Windows-friendly):

```bash
pip install -r requirements.txt
export FIREWORKS_API_KEY=... FIREWORKS_BASE_URL=... ALLOWED_MODELS=...
INPUT_PATH=tests/sample_tasks.json OUTPUT_PATH=out/results.json python -m src.main
```

Full harness simulation (Docker):

```bash
./scripts/run_local.sh          # builds image, mounts input/output, prints token report
```

Offline regression gates (no network needed) — run all five before any submission:

```bash
python tests/test_router.py     # >=95% accuracy, <=10% LLM fallback on the dev set
python tests/test_models.py     # tier assignment against real-world model-id shapes
python tests/test_prompts.py    # SPEC invariants
python tests/test_client.py     # retry/timeout/reasoning_effort policy
python tests/test_main.py       # entrypoint failure drills
```

## Pre-submission checklist

1. `python tests/test_router.py` passes.
2. `./scripts/run_local.sh` — exit code 0, `out/results.json` valid, every task_id present.
3. Token report shows no category burning unexpectedly (compare against last run).
4. Failure drills: empty tasks file, bad API key — must still write valid JSON and exit 0.
5. Time a ~50-task run; keep well under 10 minutes.

## Submission rules (per the portal, verified 2026-07-10)

- Submissions are **unlimited in total but rate-limited to 10 per hour per team**
  (guide wording: "submissions are rate-limited to 10 per hour per team").
- The leaderboard scores the **LATEST run only**, not the best. Two consequences:
  - After any experimental submission that scores worse, **immediately re-submit the
    best-known config** so we never sit on a bad score.
  - Before the deadline, the **final submission must be the best config** — schedule
    a freeze window; no experiments after it.
- One variable per experimental submission; log every scored run (config, accuracy,
  tokens) in `docs/eval-results.md` — the token delta doubles as that category's
  cost measurement.

## Rules encoded in this repo

- All calls go through `FIREWORKS_BASE_URL` (`src/client.py`) — bypassing the proxy
  scores zero.
- No hardcoded model IDs, keys, or `.env` in the image; everything comes from the
  environment at runtime.
- No cached/hardcoded answers — evaluation uses unseen prompt variants.
