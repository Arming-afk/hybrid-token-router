# Endgame optimization plan (handoff)

**Deadline: 2026-07-12 18:00 ICT** (moved from the original 2026-07-11 23:00 — extra
runway confirmed 2026-07-11). Scoring turnaround ~20 min. Freeze: 16:30 last
experiment / 17:00 submit the anchor / keep ≥2 submissions of the 10/hour quota in
reserve.

Status: we are #9 at 4,178 tokens @ 100% (run 18, anchor image
`ghcr.io/arming-afk/hybrid-token-router:7bb50c4`). Top entries sit at 1.8–2.2k.
Realistic ceiling from this plan: ~3.3–3.8k ≈ rank 5–6.

Organizer rules that shape everything here: final rankings **re-score on NEW
randomized prompts** (task count may change) — architectural properties carry,
prompt-specific tuning is overfitting; local/zero-API strategies are legal;
outbound calls other than `FIREWORKS_BASE_URL` = disqualification (localhost
Ollama exempt); leaderboard scores the LATEST run only.

Work order: **Phase 0 → A-repro → (B while docker runs) → A-fix → C → B-fixes if
any → D.** Cut from the bottom if time runs out. One variable per submission,
always per `docs/RUNBOOK.md` discipline.

---

## Phase 0 — Measurement tooling (no submission, ~1.5h, prerequisite for A & D)

1. **Extend `scripts/token_report.py`** (today it only parses `USAGE` lines, so a
   local-answered task is invisible and a local-failed-then-remote task looks like
   a plain remote task). Parse `LOCAL` / `LOCAL_FAIL` / `LOCAL_REJECTED` /
   `LOCAL_SKIP` / `ERROR` too (all emitted in `src/main.py:66-90`, all carry
   `task_id`) and join per task into a disposition:
   `DETERMINISTIC | LOCAL | LOCAL_FAIL->REMOTE | LOCAL_REJECTED->REMOTE |
   LOCAL_SKIP->REMOTE | REMOTE | NO_ANSWER`.
   - Gotcha: LOCAL_FAIL/REJECTED/SKIP carry **no `category` field** — resolve the
     category from a success `LOCAL` line or the first `USAGE` line whose category
     is not `"router"` (the llm_classify call logs `category="router"` under the
     same task_id; exclude it from category resolution but still count its tokens).
   - New output: per-category local-funnel table (det/local/fail/rej/skip/remote
     calls/tokens), a histogram of fail & reject **reasons** (this is the Phase A
     decision instrument), and a `--per-task` flag.
2. **Add `elapsed=<seconds>`** to the `LOCAL` / `LOCAL_FAIL` / `LOCAL_REJECTED`
   log lines in `try_local` (`src/main.py:62-79`). Stderr only, scoring-neutral —
   it turns the repro from pass/fail into a seconds-vs-passage-length measurement
   that calibrates the timeout formula. Ships bundled with Phase A.
3. **Author `tests/repro_summarization_long.json`** (harness shape
   `{"task_id","prompt"}`): 3 short controls (~100–150 chars — these must stay
   `LOCAL`) + 9–11 long passages graduated ~500/800/1200/1800/2500 chars, mixing
   explicit word limits, sentence limits, and no constraint. The existing
   `tests/eval/summarization.json` passages are only ~130 chars — far too short to
   reproduce the timeout.
4. **Repro command** (Git Bash; `scripts/run_local.sh` already does the docker
   build, `--cpus=2 -m 4g`, mount rewriting, `tee out/run.log`, token report):

   ```bash
   FIREWORKS_API_KEY=bogus \
   FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1 \
   ALLOWED_MODELS="gemma-4-26b-a4b-it,gemma-4-31b-it,minimax-m3,kimi-k2p7-code" \
   TASKS=tests/repro_summarization_long.json bash scripts/run_local.sh
   ```

   The bogus key makes remote calls fail fast — irrelevant, we only read the
   LOCAL* funnel. Run twice: the first run pays image build + model load.

## Phase A — Summarization local-timeout fix (biggest token lever, highest gate risk, ~3h)

Background (the “−21 anomaly”, `docs/eval-results.md`): run 18 added summarization
to the local categories but saved only 21 tokens vs run 17 — hypothesis: long
passages make local prompt-eval slow on the 2-vCPU judge box, blowing the flat 15s
`CALL_TIMEOUT` (`src/local.py:31`) → `LOCAL_FAIL` → fail-open to remote, paying
the full passage prompt tokens + completion every time. Summarization is the most
expensive remote call per task, so this is the last big lever.

**A1. Throttled repro first** — mandated by the doc before any code change.

**A2. Decision gate:**

| Repro observation on long passages | Action |
|---|---|
| `LOCAL_FAIL` "timed out", elapsed ≈ 15s wall | A3: timeout scaling |
| `LOCAL_REJECTED` (verifier reasons) | Fix only the verifier (`src/local.py:220-228`) — the timeout is innocent |
| `LOCAL` success even at 2500 chars | Try `keep_alive` alone once; if still clean, **stop — do not ship a blind fix** (the anchor already scores 100%) |
| Even short controls fail | Startup/entrypoint problem, not local.py |

**A3. The fix** (all one submission variable — “local time management”):

- `_timeout_for(prompt_chars, first_call)` in `src/local.py`: base 15s (25s first
  call) + `SECONDS_PER_KB × chars/1024`, ceiling ≤60s. **Derive `SECONDS_PER_KB`
  from the measured `elapsed`-vs-length slope with ~2× margin — do not guess.**
- Add `"keep_alive": "30m"` to the Ollama options payload so the model never
  unloads mid-run.
- **Re-check the remaining budget AFTER acquiring `LOCAL_LOCK`** in `try_local`
  (`src/main.py:64-67` checks before queuing on the lock; with a 60s ceiling and a
  bigger final task set, a task can wait minutes in the queue and then start a long
  generation on a stale check — exactly the organizers' "task count may change"
  warning).
- Strengthen the summarization verifier conservatively (a no-op today since
  everything times out before verification → same submission variable):
  (1) length-ratio: if the prompt is >80 words and the answer exceeds 0.8× the
  prompt's word count, it is provably not a summary → reject;
  (2) echo check: answer's first ~15 words appearing verbatim in the prompt →
  reject.

**A4. Quality gate before submitting**: re-run the repro; long passages must now
log `LOCAL ... elapsed=...`. Then **read every local summary manually** against
its passage. Run 18's 100% was scored while summarization mostly fell back to
remote gemma — this fix shows the judge real qwen2.5:3b summaries for the first
time, and the verifiers catch format, not semantics (run 16 lesson). If quality
looks wrong, **do not ship**; the anchor already scores 100%.

**A5. Unit tests**: extend `tests/test_local.py` — `_timeout_for` (short → base,
2.5KB → scaled, huge → ceiling) and the new verifier checks (near-passage-length
answer rejected; normal 3-sentence summary of a long passage accepted; echo
rejected). Full suite green: `python -m pytest tests/ -q`.

## Phase B — Router generalization audit (offline, free, ~2–3h; parallel with A's docker waits)

This is rehearsal for exactly what the final re-score does. History lesson
(run 2 vs 4): regex-widening scored 99.4% on the dev set and still LOST a task on
the real judge — **precision-first is the law**. Misroutes INTO narrow-format
categories cascade into wrong-format answers and LARGE-escalation token bloat;
misroutes into factual land safely (factual-on-kimi absorbs anything).

1. **New `scripts/router_audit.py`**: CLI taking case-file paths (default
   `tests/router_cases.json`, 187 cases of `{"category","prompt"}`). Per file:
   per-category accuracy (same definitions as `tests/test_router.py:16-30` —
   decided-accuracy ≥95%, fallback ≤10%), a misroute confusion matrix with prompt
   snippets, a **danger partition** (`DANGEROUS-FP` = misroute into
   sentiment/ner/summarization/debug/codegen; `SAFE` = fell to factual/fallback),
   and a **solver cross-check**: run `deterministic_math_answer` over every case
   prompt — any non-arithmetic prompt returning an answer must be listed (must be
   empty; doubles as the Phase C safety check).
2. **New `tests/router_cases_fresh.json`**: ~25–50 cases × 8 categories in NEW
   wording not lifted from the dev set — indirect phrasing ("could you tell me
   whether this customer walked away satisfied"), keyword-free intents,
   mixed-intent traps ("summarize the bug in this code"). Timebox: 2h.
3. **Decision gate**: DANGEROUS-FPs found → fix ONLY the demonstrated patterns and
   only by **tightening**, then re-run the audit on BOTH files (dev set must stay
   ≥95% / ≤10%). SAFE-only misroutes → record in eval-results.md, no code change,
   no submission. **Router is Person 1's — hand over the report and have them
   review any diff before merge.**

## Phase C — Deterministic solver widening (zero-token, cannot overfit by construction, ~2h)

Current solver (`src/router.py:204-275`) handles only `N% of M`, the four word
operators, and `+ - * /` with parentheses behind a strict charset gate. Every
addition below is either an anchored full-string regex (like `_PERCENT_OF`) or a
word→symbol substitution whose non-numeric residue still fails the charset gate —
so narrative word problems keep flowing to normal routing untouched.

1. **Tests first — new `tests/test_solver.py`** (no solver unit tests exist
   anywhere today): positives per new pattern AND narrative negatives
   ("If John has 5 apples and eats 2, how many remain?" → None;
   "What is 2 plus 2 apples?" → None).
2. **Widening, in value order:**
   - Powers: `squared`→`**2`, `cubed`→`**3`, `to the power of`→`**`, `^`→`**`;
     allow `ast.Pow` in `_eval_arith` **with hard bounds** (|exponent| ≤ 12,
     |base| ≤ 10⁶) so no input can hang the process.
   - `square root of N` — answer only when the result is an exact integer.
   - Word fractions: `half of`→`0.5 *`, `a third of`→`(1/3) *`,
     `quarter of`→`0.25 *`.
   - `sum of A and B`, `product of A and B`, `difference between A and B` (=abs),
     `add A and B`, and **`subtract A from B` = B−A (test the reversal — classic
     bug)**.
   - Percent variants (anchored): `X% off M` → `M(1−x/100)`;
     `increase/decrease N by X%` → `N(1±x/100)`.
   - `A mod B` / `remainder when A is divided by B` — never touch bare `%`
     (owned by `_PERCENT_OF`).
3. **Cross-checks**: full pytest; `router_audit.py` on both case files (solver
   cross-check section must be empty); for `tests/eval/math.json` prompts the
   solver now catches, the computed value must equal the `expected` field.

## Phase D — Scale rehearsal + RUNBOOK reconciliation (no submission, ~1.5h)

1. **New `tests/scale_tasks_100.json`**: ~100 tasks, 12–13/category — reuse
   `tests/load_tasks.json` (50) + the Phase 0 long-passage set + new fill (unique
   task_ids). Must include long summarization, pure arithmetic (solver hits), and
   a few ambiguous prompts (LLM-fallback path).
2. **Rehearse** through `run_local.sh` (throttled docker, bogus key). Measure via
   the extended token report: `DONE elapsed_s` vs the 510s deadline, `LOCAL_SKIP`
   count (budget-guard behavior at 100 tasks), serialized local throughput,
   `results.json` valid + exit 0. If LOCAL_SKIP starves summarization long before
   the deadline, **document it — do not gamble a config change on deadline day.**
3. **Fix `docs/RUNBOOK.md`** (it is stale):
   - Rollback anchor run 5/`6f01e64` → **run 18
     `ghcr.io/arming-afk/hybrid-token-router:7bb50c4`, 100% (19/19), 4,178**.
   - Replace the "Current state" section (still describes cut #1) with the run-18
     state and the A/C/B submission queue.
   - Add a "Pre-submission audit" section: `docker run --rm --network=none
     --cpus=2 -m 4g ...` must still write valid results.json and exit 0, with the
     only ERROR lines being connection failures toward `FIREWORKS_BASE_URL`
     (organizer rule: any other outbound = DQ; localhost Ollama exempt); plus
     `grep -rn "http" src/` may only surface the env var and `localhost:11434`.
   - Pin the freeze window to the real close: 21:30 / 22:00 / ≥2 quota reserve /
     verify the portal shows the expected SHA after submitting.

## Submission playbook

| # | What | Expected tokens | Gate risk | Slot rationale |
|---|---|---|---|---|
| 1 | A: summarization local fix | −200–400 | Medium-high (first real qwen summaries shown to the judge) | Biggest lever AND riskiest — needs the most wall-clock for a rollback + re-anchor cycle; must not sit near the freeze window |
| 2 | C: solver widening | 0–150 (depends on unseen mix) | Low (charset-gated, fully tested) | Cheap re-score robustness; even a null token delta can't hurt the anchor |
| 3 (cond.) | B: router FP fixes | ~0 | Low if tighten-only | Only if the fresh set demonstrates DANGEROUS-FPs; Person 1 reviews |
| — | D | — | — | No submission; doc commits ride along with any run |

Standing rules per run (see `docs/RUNBOOK.md`): one variable per submission;
record `(sha, config, accuracy, tokens)` in `docs/eval-results.md` immediately;
on any non-improvement **submit the anchor image immediately, before analyzing**
(a failed experiment burns 2 of the 10/hour quota → ~5 cycles/hour ceiling). The
22:00 closer is itself one judge roll (±1 task noise) — from 19/19, a −1 roll is
94.7% and still clears the gate.

Out of scope (noted for later, NOT in this plan):

- Breaking 3k requires debug/codegen local via a coder-tuned 3B — testable OFFLINE
  against `tests/eval/{debug,codegen}.json` before any submission risk (see the
  leaderboard snapshot analysis in eval-results.md).
- **yassai-style batching (multiple tasks → one call)**: their `yassai-summary.md`
  writeup shows 19 tasks answered in just **2 calls** to MiniMax-M3 (one DIRECT
  batch for factual/sentiment/summary/NER/code, one PYTHON-tool batch for
  math/logic) — separable from their local offload, which their own writeup marks
  as still-unshipped at that stage. This is their single biggest token lever and
  distinct from anything in Phases A–D: it needs a real rewrite of `src/main.py`'s
  per-task call loop ([main.py:14-24](../src/main.py#L14), currently 1 call = 1
  task, `CONCURRENCY=6`) into batched prompt construction + per-task answer
  parsing + per-task verification/fallback. Ruled too risky to start same-day
  against the original 23:00 close; **with the deadline now 2026-07-12 18:00
  ICT there may be enough runway to scope it properly** — give it its own phase
  (tests first, offline-measured batch-parse correctness before any submission
  risk) rather than squeezing it into the existing A–D queue.

## Verification checklist

- Every phase: `python -m pytest tests/ -q` fully green (49 existing + new).
- Phase 0/A: before/after funnel tables from the repro — long passages flip from
  `LOCAL_FAIL->REMOTE` to `LOCAL`; short controls unchanged.
- Phase B/C: `router_audit.py` green on BOTH case files; solver cross-check empty.
- Phase D: 100-task rehearsal finishes < 510s, results.json valid, exit 0,
  `--network=none` audit passes.
- Before every submission: note the git SHA, confirm CI pushed the image, and keep
  `ghcr.io/arming-afk/hybrid-token-router:7bb50c4` ready to re-submit.
