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
| 5 | `6f01e64` | 2nd-place proven config: LARGE+`reasoning_effort=none` for factual/math/logic, CODE tier for debug/codegen, SMALL for the rest, 25s timeout, no stripping | **84.2%** | **gate PASSED** (5273 tokens) |
| 6 | `6b9aefa` | Stage 2 cut #1: instruction filler trim only | (accuracy not recorded) | tokens **5260** (−13 vs predicted ~140) |
| 7 | `2518cf4` | Stage 2 cut #2: factual → "1-2 sentences" | **73.7%** | gate FAILED, regression from run 5 |
| 8 | `c7120f0` | cut #2 reverted (factual back to "under 120 words"), cut #1 kept | | |
| 9 | _(queued)_ | cut #3: math/logic → "at most 2 short steps" + codegen "No comments." (probe-backed; predicted −300–550) | | |
| 10 | _(queued)_ | cut #4: factual → "under 60 words" (untested middle between proven-120 and failed-1-2-sentences; predicted −100–200) | | |

Readings: 47.4% ≈ 9/19, 68.4% ≈ 13/19, 57.9% ≈ 11/19, 63.2% ≈ 12/19, 84.2% ≈ 16/19,
73.7% ≈ 14/19 — judge set is ~19 tasks. **Gate threshold is bracketed at (73.7%, 84.2%]**
— tightest evidence so far on where it actually sits.

### Run 5→7 regression: cut #2 (factual → "1-2 sentences") cost ~2 tasks (2026-07-09)

Cut #1 rode its own scored run (6) and only moved tokens −13, so cut #2 in run 7 is
cleanly isolated as the regression: a semantic cut to answer length (not just wording)
on the highest-task-share category that also doubles as the router's misroute default
on LARGE. Reverted cut #2 back to the run-5-proven "under 120 words"; kept cut #1.
Cut #1's own lesson: it saved 13 tokens against an input-side prediction of ~140 —
run-to-run token noise is ~±100, so **cuts below ~200 predicted tokens are unmeasurable**
and wording micro-trims are a dead end.

### Token-cost probe on the public endpoint (2026-07-10) — numbers for Stage 2 cuts

`scratchpad/probe_tokens.py` hit the PUBLIC Fireworks endpoint with the team key using
our exact production instructions (system+user shape, `reasoning_effort="none"`,
temperature 0). Public deployment may differ from the judges' proxy — treat as strong
approximation, not ground truth. All six answers were CORRECT.

| probe | model | prompt | completion | note |
|---|---|---|---|---|
| math, current "brief steps" | minimax-m3 | 163 | **38** | steps + Answer line |
| math, answer-only | minimax-m3 | 158 | **7** | correct without any steps |
| logic, current "numbered steps" | minimax-m3 | 176 | **30** | |
| logic, answer-only | minimax-m3 | 164 | **4** | correct without any steps |
| debug, current (bug sentence + code) | kimi-k2p7-code | 77 | **69** | |
| debug, code-only | kimi-k2p7-code | 73 | **43** | −38% |

Findings:
1. **`reasoning_effort="none"` suppresses minimax completely** (`reasoning_len=0` on
   every call) and correctness held even with zero visible steps on these samples.
   minimax OUTPUT is cheap — completions are not where run 5's 5273 went.
2. **minimax bills ~+95–100 hidden PROMPT tokens per call.** Identical-size content
   costs ~77 prompt tokens on kimi but ~160+ on minimax; the variable part tracks
   instruction length 1:1, so the fixed overhead is ~100/call (hidden preamble or
   tokenizer, either way billed). With factual+math+logic ≈ 7–8 judge tasks on LARGE,
   that's **~700–800 tokens/run of structural overhead tied to the tier mapping**.
3. kimi-k2p7-code is now reachable on the public key (was 429-locked on 2026-07-08)
   and is terse; "code only" cuts debug by ~38% on the sample.

Implications for reaching <4000 total (from the 5260 baseline, need ≈ −1300):
- Phase B (instruction cuts, one per submission, probe-backed): math/logic to
  "at most 2 short steps" or answer-only (−250–500); debug to code-only, codegen
  "no comments" (−100–150); factual to "under 60 words" (−100–200; "1-2 sentences"
  is proven too far, 4× that budget is the untested middle).
- Phase C (structural, if B lands short): move ONE of factual|math+logic off minimax
  to MEDIUM to shed the ~100/call prompt tax (−200–300 plus shorter gemma answers);
  factual→MEDIUM sacrifices the misroute-insurance design, math+logic→MEDIUM re-tests
  what run 2 could not isolate. Each is its own scored experiment.
- Expected landing: B alone ≈ 4400–4700; B + one C move ≈ 3900–4300.

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

### Adopting the 2nd-place gate-PASSING config (2026-07-09, run 5 candidate)

The public reference repo compared above (`KaananeTaha/AMD-AI-Hackathon`) turned out to be
the **2nd-place Track 1 entry** (confirmed by the team 2026-07-09). Its config therefore
*passed the accuracy gate* — every value in it is stronger evidence than anything we can
measure locally (the gemma tiers remain unreachable outside the judges' proxy). It also
directly answers all three questions this file had deferred:

| question | their proven answer |
|---|---|
| (a) math/logic: MEDIUM vs LARGE | **LARGE `minimax-m3` — but only with `reasoning_effort="none"`** (without it: blank answers, which is what sank our run 3) |
| (b) sentiment/NER/summarization on SMALL | **yes, SMALL holds** (`gemma-4-26b-a4b-it`) |
| (c) factual on SMALL | **no — they run factual on the strongest model**, deliberately: factual is the router's misroute default, so any false positive lands on the most capable model (gate insurance) |

Their measured cost on the real harness models: 8 practice tasks ≈ 1220 tokens (~150/task),
all correct — that's also the token-efficiency bar once the gate passes.

Changes adopted (branch `fix/client-timeout-reasoning-effort`, on top of the client.py fix):
- `models.py`: new **CODE tier** — code-specialized ids (`code`/`coder`) are excluded from
  the general tiers and serve debug/codegen; falls back to MEDIUM if no code model exists.
  Side benefit: LARGE was previously an env-order-dependent tie between two unsized MoEs
  (`minimax-m3` vs `kimi-k2p7-code`); it is now deterministically `minimax-m3`.
- `prompts.py` SPEC — the reference's category→tier/caps verbatim:

| category | tier (model) | max_tokens (was) |
|---|---|---|
| factual | LARGE minimax-m3 | 300 (250, MEDIUM) |
| math | LARGE minimax-m3 | 400 (450, MEDIUM) |
| logic | LARGE minimax-m3 | 420 (500, MEDIUM) |
| debug | CODE kimi-k2p7-code | 520 (800, MEDIUM) |
| codegen | CODE kimi-k2p7-code | 520 (800, MEDIUM) |
| sentiment | SMALL gemma-4-26b-a4b-it | 120 (120, MEDIUM) |
| summarization | SMALL gemma-4-26b-a4b-it | 220 (300, MEDIUM) |
| ner | SMALL gemma-4-26b-a4b-it | 260 (300, MEDIUM) |

- Instructions rewritten to their proven wording, with the shared base
  "English only. Be concise; no preamble." (the guide's all-tracks rule requires English
  responses; ours never said so). Delivered as a **system message** (their exact shape).
- **`postprocess()` stripping dropped**: they passed the gate handing the judge the full
  "brief steps + Answer: <value>" text untouched; stripping only added risk.
- `main.py` writes **`/output/inference_log.json`** (calls + token totals, best-effort).
  The guide's "No inference log is required for Track 2" phrasing implies Track 1 expects
  one; the reference writes it too.

Known open risk this does NOT address: PR #9's router widening still costs ~1 task
(run 2 vs 4). Mitigated here by the factual-on-LARGE misroute-insurance design; finding
the specific false-positive pattern in `router.py` remains open for the router owner.

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

| category | chosen tier | chosen max_tokens | few-shot? | evidence |
|---|---|---|---|---|
| factual | LARGE, "under 120 words" | 300 | no | 2nd-place gate pass (84.2%); "1-2 sentences" regressed to 73.7% (run 6), reverted |
| math | LARGE | 400 | no | 2nd-place gate pass (with `reasoning_effort="none"`) |
| logic | LARGE | 420 | no | 2nd-place gate pass (with `reasoning_effort="none"`) |
| debug | CODE | 520 | no | 2nd-place gate pass on `kimi-k2p7-code` |
| codegen | CODE | 520 | no | 2nd-place gate pass on `kimi-k2p7-code` |
| sentiment | SMALL | 120 | no | 2nd-place gate pass on `gemma-4-26b-a4b-it` |
| summarization | SMALL | 220 | no | 2nd-place gate pass on `gemma-4-26b-a4b-it` |
| ner | SMALL | 260 | no | 2nd-place gate pass on `gemma-4-26b-a4b-it` |

Evidence class: an entry that *passed the real accuracy gate* (finished 2nd) using these
exact values on the exact judge models — stronger than any local proxy eval we can run.
Await run 5's score before touching any value; if it passes, tighten caps one category
per submission using their ~150 tokens/task as the target.

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
