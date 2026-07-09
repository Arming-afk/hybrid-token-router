# Eval results — per-category model & prompt selection (Person 2)

This file records the numbers behind every value in `src/prompts.py` `SPEC`. Nothing in
`SPEC` should be changed without a row here justifying it.

## Status (2026-07-08): local eval is BLOCKED — no access to the real Track 1 models

Per the participant guide, `FIREWORKS_API_KEY`, `FIREWORKS_BASE_URL`, and `ALLOWED_MODELS`
are **injected by the harness only at evaluation time** ("provided by the harness — use
this key, not your own"). We verified with a personal Fireworks key that:

- The pipeline is correct: a raw call returns HTTP 200 with real `usage`, confirming
  `client.py`, the base URL, and the `accounts/fireworks/models/<id>` ID format all work.
- **The personal key cannot reach the real Track 1 models.** `models.list()` returns only
  unrelated models (`gpt-oss-120b`, `kimi-k2p6`, `glm-5p1`, `deepseek-v4-pro`, …) — none of
  `gemma-4-*`, `minimax-m3`, `kimi-k2p7-code`.
- Every model the personal key *can* reach is a **reasoning model** (returns
  `reasoning_content`; with a tight `max_tokens` it spends the whole budget thinking and
  emits empty `content`) and the account is rate-limited (429s even on spaced calls). These
  are the *opposite* profile of the real gemma `-it` tiers, so their numbers are not just
  imprecise — they are misleading and must not drive `SPEC`.

**Conclusion:** `scripts/eval_matrix.py` cannot produce trustworthy numbers before the
harness runs. Strategy is therefore **validate the pipeline (done) + hand-tune `SPEC` from
public model facts**, and defer measured tuning to real harness submissions (10/hour).

### What we know about the real tiers (basis for hand-tuning)
- **SMALL `gemma-4-26b-a4b-it`** — 26B MoE, 4B active. Instruction-tuned (`-it`), *not* a
  reasoning model → answers land directly in `content`, so tight `max_tokens` is safe.
- **MEDIUM `gemma-4-31b-it`** — 31B dense, instruction-tuned, non-reasoning.
- **LARGE `minimax-m3`** — unknown size; reasoning status UNVERIFIED (see follow-ups).
- **`kimi-k2p7-code`** — code-specialized, unused by the 3-tier scheme.

## Scored submission history (the only real accuracy numbers we have)

The harness gives one number per submission — treat each run as one eval data point.

| # | image | config | accuracy | verdict |
|---|---|---|---|---|
| 1 | `60a82aa` | original guesses: easy cats on SMALL (4B-active), tight caps | **47.4%** | gate FAILED |
| 2 | `bcf1e8f` | pre-PR#9 router + math/logic MEDIUM, caps raised, 429 handling | **68.4%** | gate FAILED |
| 3 | `4d5e32b` | PR #9 router widening + math/logic → LARGE (minimax-m3, 2000-tok cap) | **57.9%** | gate FAILED, regression |
| 4 | `215e22a` | PR #9 router widening (kept) + math/logic reverted to MEDIUM | **63.2%** | gate FAILED, partial recovery |

Readings: 47.4% ≈ 9/19, 68.4% ≈ 13/19, 57.9% ≈ 11/19, 63.2% ≈ 12/19 — judge set is ~19 tasks.

### Root cause of the run 2→3 regression, now cleanly decomposed (2026-07-09)

Run 3 shipped two changes at once (PR #9 router + PR #10 math/logic→LARGE), so the drop
looked like a confound. Run 4 isolates them — it keeps PR #9's router and only reverts the
model tier back to run 2's config. That gives two clean same-variable comparisons:

- **Run 2 vs run 4** (router changed, model config identical): 68.4% → 63.2%, **−1 task**.
  Attributable to PR #9's router widening alone — its 169-case dev set scored 99.4%, but
  that dev set doesn't cover the real judge prompts; at least one of the widened patterns
  (sentiment tone/mood, WRITE soft-verbs, STRONG_FIX failure-phrasings, math word-relations,
  logic seating phrasings, or the bare-code-defaults-to-debug change) is a new false
  positive on the actual eval set. **Still needs isolation within router.py** — not yet
  identified which specific pattern. Router owner should stress-test each widened group
  independently before the next router change ships.
- **Run 3 vs run 4** (router identical, model config changed): 57.9% → 63.2%, **+1 task**.
  Attributable to math/logic on LARGE (`minimax-m3`) without reasoning suppression. Matches
  a documented finding from a public reference implementation of the same contest
  (`KaananeTaha/AMD-AI-Hackathon`): "minimax-m3 is a reasoning model — without
  `reasoning_effort="none"` it burns the whole token budget on hidden reasoning and returns
  a BLANK answer on hard prompts." Our `client.py` never sent that param.

Together these two ~1-task effects account for the full 68.4%→57.9% drop.

### client.py fix (2026-07-09, committed locally, not yet pushed — tests green)

Independent of the router question, comparing against the same reference repo surfaced a
second bug: `CALL_TIMEOUT_SECONDS` was 60s, but the Participant Guide's general rules (page
8, never fully read by our team until now) state **"Response time per request must be under
30 seconds"** as a hard rule. Any call running 30–60s was likely already being
discarded/penalized by the harness while we sat there waiting for nothing.

Fix (`src/client.py`, branch `fix/client-timeout-reasoning-effort`, rebased onto `215e22a`):
- `CALL_TIMEOUT_SECONDS` 60 → 25 (matches the reference repo's validated value).
- `reasoning_effort="none"` sent by default on every call (not just LARGE — `main.py`'s
  blank-answer fallback escalates *any* category to LARGE as a rescue, so minimax-m3 is in
  the hot path regardless of math/logic's own tier). A model that rejects the param as
  unknown (400) is remembered per-model and skipped on subsequent calls, retried once free.
- Covered by `tests/test_client.py` (5 cases, stubbed SDK, no network) — all green, plus the
  full existing gate suite (router 99.4%/0%, models 9/9, prompts, main 8/8) still passes.

This fix is orthogonal to the router question and should be pushed on top of `215e22a`
regardless of how the router false-positive investigation goes — it doesn't reopen the
LARGE-tier question, just removes two concrete bugs (hard-rule violation + blank-answer
mechanism). Reopening math/logic→LARGE with the reasoning_effort fix applied is a separate,
later experiment once the router false positive is found and fixed.

Token cost is deliberately ignored until the gate passes.

### Decisions deferred to a real eval (do NOT guess these blind)
- `factual`: currently MEDIUM. Biggest token-saving opportunity is dropping to SMALL, but
  it's a genuine accuracy risk — needs a real gemma eval. Left at MEDIUM (safe) with a TODO.
- `math` / `logic`: kept at MEDIUM (SMALL 4B-active most likely to fail multi-step here).
- `debug` / `codegen`: whether `kimi-k2p7-code` beats `gemma-4-31b-it` on cost/accuracy.

## Method

`scripts/eval_matrix.py` runs each category's eval set (`tests/eval/<category>.json`,
~10 items with expected answers) through candidate configs, grades each answer with a
cheap SMALL-model judge, and reports pass rate and **average answer-call tokens** (the
quantity the contest actually scores — judge tokens are not counted). Within a category,
the cheapest config that clears the pass threshold (default 90% on the eval set) wins.

The seeded configs target the three open questions:
- **(a) math / logic** — MEDIUM with short reasoning steps vs LARGE answering directly.
- **(b) sentiment / NER / summarization** — does SMALL hold, or is MEDIUM needed?
- **(c) factual** — can it drop from MEDIUM to SMALL?

## Prerequisites to reproduce

1. Merge `person1/router-tiering` into the branch first, so `models.build_tiers()`
   returns the corrected tiers for the real models — otherwise MEDIUM resolves to the
   FP4-quantized `gemma-4-31b-it-nvfp4` and the numbers below are not comparable.
2. Provide real credentials (local dev): `set -a; . ./.env; set +a`.
3. Run: `python scripts/eval_matrix.py` (append category names to run a subset).

Real Track 1 tiers (post-merge): SMALL=`gemma-4-26b-a4b-it`,
MEDIUM=`gemma-4-31b-it`, LARGE=`minimax-m3`.

## Results

Regenerated by `scripts/eval_matrix.py` (do not edit the block by hand):

<!-- RESULTS:START -->

_Not yet run — needs FIREWORKS_BASE_URL. Run the command above to populate this table._

<!-- RESULTS:END -->

## Decisions log

Fill one row per `SPEC` field once the table above exists. Example shape:

| category | chosen tier | chosen max_tokens | few-shot? | evidence |
|---|---|---|---|---|
| _pending_ | | | | |

## Open follow-ups for the eval

- **Reasoning-token check (from Person 1):** confirm empirically that `minimax-m3` and
  `kimi-k2p7-code` do not emit billed reasoning traces before assigning either to a
  category. If one does, keep it out of the hot path entirely.
- **kimi-k2p7-code is code-specialized and currently unused.** Add it as an explicit
  candidate for `debug`/`codegen` in `configs_for()` and compare pass rate / tokens
  against MEDIUM. Routing a category to a specific model beyond SMALL/MEDIUM/LARGE
  touches the shared tier interface — agree with the team first.
- **max_tokens trimming (Task 4):** once a tier is fixed per category, lower `max_tokens`
  to the smallest value that never truncates a correct answer on the eval set (keep ~30%
  headroom), and re-confirm pass rate.
