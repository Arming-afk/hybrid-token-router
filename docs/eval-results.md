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

## Leaderboard snapshot (2026-07-10 ~23:00) — we are #9 at 4,178 @ 100%

| # | entry | team | tokens | acc | scored |
|---|---|---|---|---|---|
| 1 | Metis | Kingdom of Science | **1,797** | 94.7% | Jul 10 22:45 |
| 2 | Kestrel v0.68 | SoloPlayer | 1,798 | 89.5% | Jul 10 20:54 |
| 3 | yassai | Solo Stack | 2,228 | **100%** | Jul 10 21:40 |
| 4 | YOLOAI_v6 | YoloAI | 2,664 | 84.2% | Jul 10 16:03 |
| 5 | TokenRouter prove or escalate | Sprint Sprint Win | 3,562 | 89.5% | Jul 10 20:50 |
| 6 | Git it done v27 | GitCommit_and_Pray | 3,682 | 84.2% | Jul 10 23:02 |
| 7 | LocalFirst-4 | jae | 3,753 | 100% | Jul 10 16:13 |
| 8 | Divine v15 | Divine | 3,779 | 84.2% | Jul 10 19:08 |

**Update (2026-07-11 ~01:50, user-reported):** yassai has dropped again —
**1,292 tokens @ 94.7%** (was 2,228 @ 100%). Their public writeup (written at their
4,826 cloud-only stage) closes with "further reduction needs a reliable zero-token
local offload" — this drop is that offload landing, and they visibly sold one judge
task (100% → 18/19) for it. Implication: the podium zone is now ~1.3–1.8k,
unreachable for our architecture today; the Track A/B ceiling (~2.5–3.8k) fights
for top 5. Plan unchanged — the final re-score on fresh prompts favors robust
architectures over margins tuned to the current 19-task set.

What this snapshot proves:
- **High accuracy at 1,8–2,2k tokens is achievable** (Metis 94.7%, yassai 100%).
  The "cheap = sold accuracy" model is falsified; the top entries must be running
  broad local coverage that stays accurate — plausibly a better/specialized local
  model (e.g. a coder-tuned 3B for code categories) or stronger verification.
- **Scoring turnaround is now ~20 minutes** (Git it done: submitted 22:42,
  scored 23:02) — many more experiment cycles fit before the Jul 11 23:00 close
  than the endgame plan assumed.
- Our realistic ceiling without new local coverage: summarization fix + cold-start
  fix ≈ 3,3–3,8k → around rank 5–6. Breaking 3k needs code categories local,
  which run 16 poisoned for plain qwen2.5:3b — but a coder-tuned sibling was
  never tested and can be evaluated OFFLINE against tests/eval/{debug,codegen}.json
  before risking a submission.

### In flight (2026-07-11, person2): offline eval tooling for the next two levers

Two runnable-by-anyone scripts are now in `scripts/` so the blocked measurements can
run on whichever machine first gets the models downloaded (the throttled
`docker run --cpus=2 -m 4g ... ollama/ollama` setup is in each script's docstring):

- `scripts/eval_local_code.py <model>` — the coder-model question: runs
  debug/codegen eval items through the PRODUCTION local pipeline
  (local.generate + local.verify), then **executes the generated code against
  per-item assertions** — the semantic check whose absence sank run 16. Meant for
  `qwen2.5-coder:3b` (never tested; the only path below 3k). ≥9/10 PASS per
  category = worth a scored run; the SEMANTIC_MISS count is the widen/don't-widen
  signal.
- `scripts/eval_local_summarization.py` — the −21-anomaly repro: true local latency
  (timeout raised to 120s) on short eval passages + synthesized long ones vs the
  15s production cap; the first call also measures the cold-start model load.

Also identified by code reading (no measurement yet): the entrypoint's warmup races
the agent — local-eligible tasks arriving in the first ~25s can LOCAL_FAIL to paid
remote while the model is still loading. Candidate fix: gate the first local
attempts on warmup completion (wait, don't fail open) — cheap, and part of the
"cold-start fix" in the ceiling estimate above. Status on person2's machine: Ollama
image download cancelled midway (bandwidth); scripts are ready to fire elsewhere.

### Track A measurement (2026-07-11 ~04:20, person3 machine, ollama pinned to 2 cores)

`scripts/eval_local_summarization.py` on native Windows ollama (qwen2.5:3b), all
ollama processes affinity-pinned to 2 logical cores to mimic the grading box
(caveat: full RAM — the 4GB cap is NOT reproduced; docker pull of the image kept
failing on this connection, weights fetched via resumable curl instead):

| item | latency | verdict |
|---|---|---|
| short [0] (first call) | **57.0s** | model load under contention — would LOCAL_FAIL at the old 25s cap |
| short [1] | 3.0s | genuine verifier reject: 12 words vs a 10-word limit (correct escalation) |
| short [2]–[3] | 3.1s | pass |
| long 350 / 520 / 870 words | 3.6 / 4.4 / 5.2s | pass — all far inside the 15s cap |

Re-timing item [0] warm: **3.4s** (load_duration 0.6s) — the 57s was pure model
load, not compute. Conclusions:

1. **The long-passage-timeout hypothesis for the −21 anomaly is DISPROVED** on
   CPU grounds: 870-word prompt-eval takes ~5s at 2 cores, 3× inside the cap.
2. **Cold-start is the confirmed real risk**: a first call under load took 57s;
   the old FIRST_CALL_TIMEOUT=25 loses that race → raised to 60s.
3. The `_sentence_count` verifier over-counts on abbreviations (U.S., Dr., e.g.,
   single-letter initials) — confirmed offline, rejects correct one-sentence
   answers → abbreviation masking added.
4. Remaining candidate causes for −21, not separable without the real box:
   verifier rejections in prod (abbrev + genuine word-limit misses), 4GB RAM
   pressure (untestable natively; SUMMARIZATION_CALL_TIMEOUT=45s added as free
   insurance), or simply few/short summarization tasks in the judge set.

Rung 3 = these three fixes, LOCAL_CATEGORIES unchanged (sentiment,ner,summarization).
All fail-open: zero accuracy risk vs run 18's 100%; token saving materializes only
if prod summarization was bleeding to remote via causes 2–3.

### Track B measurement (2026-07-11 ~06:30): qwen2.5-coder:3b carries the code categories

`scripts/eval_local_code.py qwen2.5-coder:3b` (native ollama pinned to 2 cores,
generated code EXECUTED against per-item assertions):

- **debug: 10/10 PASS**, 0 semantic misses, latency avg 4.1s / max 8.1s
- **codegen: 9/10 PASS**, 1 semantic miss (average-ignoring-None item: generated
  code divides by zero on an empty list), latency avg 4.3s
- 19/20 total ≥ the 14/15 gate → run 16's "code categories are poison" verdict is
  confirmed to be a PLAIN-qwen2.5:3b property, not a 3B-local property.

Single-model comparison on the easy categories (both models, same eval sets,
manual re-judge of scorer artifacts): sentiment base 9/10 vs coder 8/10 (the one
extra flub is a contradictory label+justification); ner ≈7/10 BOTH (different
miss profiles; the automated scorer's comma-split values understated both);
summarization equal by eyeball. **Verdict: swap to the coder model outright** —
4GB RAM cannot hold two 3B models and a swap costs 10-20s serialized.

**Rung 4** (this commit): LOCAL_MODEL=qwen2.5-coder:3b, LOCAL_CATEGORIES=
sentiment,ner,summarization,debug,codegen. factual/math/logic stay on kimi.
Validated end-to-end against the real local sidecar with remote stubbed dead:
all five categories answered locally through the production pipeline (the
summarization answer contained "U.S." and passed the one-sentence verifier —
the rung-3 abbrev fix live). Expected landing ~3.0–3.5k if the code categories
hold on the judge set; risk budget is run 18's +3-task headroom above the gate.

## Organizer clarification (2026-07-10) — read before choosing the final submission

Announced on the contest channel:

1. **Local-only / zero-API-call strategies are explicitly legitimate.** Our
   in-image Ollama path breaks no rules.
2. **Final rankings re-score submissions on NEW randomized prompts after the
   close.** Every accuracy number in the table below measures fit to the
   CURRENT 19-task set only. Prompt-specific margins (razor-thin caps, wording
   tuned to observed tasks) may not carry; category-level and architectural
   properties (router regexes, deterministic solver, verifiers + fail-open
   local) do carry. Task COUNT may also change — the serialized local path and
   the 150s local-budget guard matter if the final set is bigger.
3. **No network isolation, but manual audits: routing inference outside
   Fireworks = disqualification.** Localhost Ollama is fine under point 1;
   runtime must make no outbound calls except `FIREWORKS_BASE_URL`
   (verifiable with `docker run --network=none`).
4. **Tie-breaker for equal tokens/accuracy: TBD by organizers.**

Consequence for the endgame: pick the final image for **robustness on unseen
prompts**, not for its score on the current set.

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
| 9 | `86cf241` | cut #3: math/logic → "at most 2 short steps" + codegen "No comments." | (pass; % not recorded) | tokens **5085** (−175 vs 5260; predicted −300–550) |
| 10 | `f4c9742` | cut #4: factual → "under 60 words" (middle between proven-120 and failed-1-2-sentences) | **73.7%** | gate FAILED — reverted |
| 11 | `86cf241` (resubmitted) | rollback to cut #3 anchor after cut #4's failure — IDENTICAL code to run 9 | **~78.9%** (15/19) | **identical code lost ~1 judge task vs 84.2%** — accuracy noise is ±1 task and spans the gate |
| 12 | `3a7f0e7` | Phase C move #1: math+logic → MEDIUM, instructions/caps unchanged (sheds the ~100/call minimax prompt tax; predicted −400–600 from 5085; blind — gemma-4-31b-it 404s on the public key, re-probed 2026-07-10) | **~78.9%** (15/19) | gate FAILED — but same roll as the anchor's own re-roll (run 11), so **inconclusive as accuracy evidence** |

| 13 | `94619a7` | router hardening (5 false-positive classes fixed, dev 187/187) + Phase C move #1 + cut #3 — submitted as "latest tag at submit time", not the intended `3a7f0e7` re-roll | **94.7%** (18/19) | **gate PASSED, 5095 tokens — best accuracy ever** |
| 14 | `1a4947a` | **Moonshot bundle** (rank 1 posted 2600/84% — accuracy above the gate is wasted spend, so run 13's +2 headroom is converted to cuts): factual → MEDIUM + "under 50 words" (cap 120); sentiment → label only (cap 30); debug → corrected code only, no comments (cap 450); summarization → ≤3-sentence default; math/logic caps → 150 (2 steps kept as CoT); nothing on LARGE | **INFRA_ERROR** | not scored — image verified pullable (200) and CI green, so most likely a premature submit inside the ~30s build window or a transient harness failure |
| 15 | `de9bbf4` | **Moonshot, fixed**: factual/math/logic MEDIUM → **CODE (kimi)** (gemma-31b 404s at grading; kimi is scored-run-proven and tax-free) + deterministic arithmetic solver (0 tokens) + truncation escalation | **94.7%** (18/19) | **gate PASSED, 4548 tokens — best on both axes** |

| 16 | `e9838d0` | **Local-first architecture**: Ollama + qwen2.5:3b in-image; 5 categories local behind zero-token verifiers; factual/math/logic stay on kimi | **78.9%** (15/19) | gate FAILED — ~2-3 silent local wrong answers slipped past the verifiers |
| 17 | `263dd54` | Local narrowed to **sentiment,ner** only (the shortest-output, most-verifiable categories); summarization/debug/codegen back to the proven remote path | **89.5%** (17/19) | **gate PASSED, 4,199 tokens — new best token count; new endgame anchor** |
| 18 | `7bb50c4` | Ladder rung 2: local widened to **sentiment,ner,summarization** (its word/sentence-limit verifiers already exist) | **100%** (19/19) | **gate PASSED, 4,178 tokens — best on both axes, first 100%; new endgame anchor** — but only −21 vs run 17: summarization likely fell back to remote (see lessons) |

Run 18 lessons (2026-07-10):
- **First 100% (19/19), 4,178 tokens — new endgame anchor `7bb50c4`.**
- **The −21 anomaly**: +summarization was predicted to shed several hundred
  remote tokens but shed 21. Most plausible cause: long summarization passages
  make local prompt-eval slow on the 2 vCPU box, blowing the 15s `CALL_TIMEOUT`
  → `LOCAL_FAIL` → fail-open to remote. Needs a throttled local repro
  (docker `--cpus=2 -m 4g`, bogus API key, watch LOCAL/LOCAL_FAIL lines)
  before any code change.
- **Bisect verdict on run 16**: run 16 (5 cats local) = 15/19; run 18
  (3 cats local) = 19/19 on the same judge set — the only config delta is
  debug+codegen local. The ~3-4 lost tasks were code answers that passed
  ast.parse/fn-name verifiers while being semantically wrong. **Do NOT widen
  local to debug/codegen** — qwen2.5:3b can't carry them.
- Token math: model choice barely moves the score (tokens, not dollars) — the
  only big lever left is making summarization actually answer locally; the
  conventional remote trims (math/logic answer-only, cap shaves) are ~100-200.

Run 17 lessons (2026-07-10):
- **sentiment+ner local is safe**: 17/19 @ 4,199, −349 tokens vs run 15. The −1
  task vs run 15's 18/19 is inside the proven ±1 noise band, and from 17/19 even
  a −1 re-roll (16/19 = 84.2%) still clears the gate.
- **New endgame anchor: `263dd54`** — beats `de9bbf4` on tokens (the ranking
  axis) while passing; supersedes run 16's "re-save `de9bbf4`" standing recovery.
- Submission mechanics: the portal would NOT accept re-saving `de9bbf4` while
  `263dd54` was pending — a "best" entry cannot be parked. Whatever is submitted
  LAST before close (2026-07-11 23:00) is what counts, so the final submission
  must itself be a proven-passing image; never let an experiment be the closer.

Run 16 lesson (2026-07-10): verifiers catch format and syntax, not semantics — a
confidently wrong local label/summary/algorithm passes every programmatic check. With
~12 of 19 tasks answered locally, 2-3 semantic misses ≈ the observed 15/19. This is
exactly the LocalFirst 68.4% experience; their recovery (and now ours) is category
bisection through the re-scoring loop: shrink LOCAL_CATEGORIES to the safest set,
re-measure, widen one category at a time. Standing recovery: re-save `de9bbf4`
(4,548 @ 94.7%) whenever a passing entry needs to be on the board.

Run 15 lessons (2026-07-10):
- **−547 tokens at unchanged 18/19.** The moonshot's headroom-selling cuts (factual
  "under 50 words", sentiment label-only, debug code-only) cost NOTHING measurable —
  person2's hypothesis holds: the old factual-squeeze failures (73.7% × 2) were
  casualties of the pre-hardening router era and/or minimax-vs-kimi behavior, not of
  the length budget itself. The +2 headroom is still unspent.
- kimi now carries factual/math/logic/debug/codegen; SMALL carries the rest; minimax
  and gemma-31b are fully out of the hot path.
- Remaining conventional levers are small (~200–400 total: math/logic answer-only on
  kimi, factual→SMALL, cap trims). Going meaningfully below ~4,100 requires the
  local-inference architecture (LocalFirst section above): local tokens count as zero.
- Leaderboard reference points: rank 1 = 2,664 @ 84.2%; LocalFirst = 3,753 @ 100%;
  us = 4,548 @ 94.7%.

Run 13 lessons (2026-07-10):
- **The router fix is validated on the real judge set**: +2–3 tasks over the 15/19 band,
  far outside ±1 noise. Factual misroutes were real and expensive, exactly as the hunt
  predicted.
- **Phase C's token saving is real but masked**: repaired misroutes now route INTO
  factual on LARGE (minimax tax + full 120-word answers), adding back roughly what
  math/logic→MEDIUM saved. Net 5095 ≈ run 9's 5085, at +2 tasks more accuracy.
- **Operational**: no `latest` tag exists on GHCR (release.yml pushes SHA tags only,
  digests immutable — verified). What ran was determined by the tag typed into the form:
  the newest SHA at submit time, not the older tag assumed in the plan. ALWAYS pin and
  double-check the exact SHA in the form before saving; record the submitted tag with
  the result.
- With 18/19 there is finally **+2 tasks of headroom above the gate** — token cuts can
  resume without every experiment being a coin flip.

### Competitor intel: jaeyooniee/track1-hybrid-routing-agent — 100% @ 3,753 tokens (2026-07-10)

Full analysis of the repo that beats us on both axes. Three findings that matter to us:

1. **Local inference counts as ZERO tokens** (it's in the official rules) and they built
   their whole architecture on it: Ollama + qwen2.5:3b baked into the image (3.6GB),
   5 of 8 categories answered locally for free, plus a deterministic Python arithmetic
   evaluator for pure-calculation prompts (0 tokens, no hallucination). Only
   factual/math/logic and measured-unreliable puzzle patterns hit Fireworks (kimi-first,
   reasoning off, compact per-category system prompts). Their journey: 68.4% FAIL
   (local answers silently wrong) → verifiers added → 100% @ 3,753.
2. **gemma-4-31b-it and -nvfp4 reportedly have NO serverless support** (their code cites
   multi-participant Discord consensus + the Fireworks dashboard) — they 404 everywhere,
   likely including the grading env. If true, **our MEDIUM tier never existed**: runs
   12/13's math/logic→MEDIUM actually 404'd (free) and fell back to LARGE minimax via
   main.py's blank-answer retry. This cleanly explains run 12 ≈ run 11 anchor (15/19
   both, since nothing effectively changed) and run 13's tokens ≈ run 9's. Every future
   tier decision should treat ALLOWED_MODELS' usable set as: minimax-m3, kimi-k2p7-code,
   gemma-4-26b-a4b-it.
3. **Zero-token verifiers + pay-for-intelligence-on-failure**: they verify every answer
   programmatically (hedge phrases, math answer must contain a number, ast.parse for
   Python, requested-function-name present, buggy code returned unchanged, summary
   length/bullet counts, degenerate output) and only retry with reasoning ON when
   verification fails. finish_reason=length is treated as an error, never submitted.
   They also measured **kimi hides ~60% of completion in reasoning_content** unless
   reasoning_effort="none" (189→113, 195→117) — our client already sends it (validated).

Notable: their comments credit "the #2 team at 94.7%" — us — for the "Answer: <value>"
last-line pattern and the reasoning-off-with-retry design they adopted. The competitive
gap is NOT prompt tuning; it is the local-zero rule plus verification.

Methodology constraint (confirmed 2026-07-10): **failed runs do not show a token count**
on the results page. A token-cut experiment that drops below the gate returns ZERO
information — accuracy is noise-ambiguous AND the token saving stays invisible. Every
future experiment must be designed to pass the gate, or it is a wasted submission.

Run 12 lesson (2026-07-10): 15/19 is exactly what the UNCHANGED anchor rolled in run 11,
so Phase C move #1 shows no detectable accuracy cost — the score can't distinguish
"MEDIUM lost a task" from the anchor's own noise. The token saving (predicted −400–600)
is the real payoff and stays unverified until a passing roll banks it (failed runs
don't rank). Threshold evidence tightens: 78.9% fails ⇒ bracket **(78.9%, 84.2%]**,
consistent with exactly 80% (16/19). Operational note: if the platform counts only the
latest saved submission, a failed run may be what's currently standing — banking a
passing roll has priority over further experiments.

Run 11 lesson (2026-07-10): the SAME image that scored 84.2% (run 9) scored ~78.9% on
resubmission — **run-to-run accuracy noise is ±1 judge task on identical code**, so the
gate is stochastic for our config, and single-run deltas of one task (~5.3pt) cannot
distinguish a real regression from noise. Only ≥2-task moves (like the two factual
failures at 73.7%) are signal. If run 11 failed the gate, the threshold bracket narrows
to (78.9%, 84.2%] — consistent with a threshold of exactly 80% (need 16/19). End-game
consequence for the freeze window: the final submission is itself a judge-roll; leave
enough quota to re-submit the anchor if the final run rolls a 15/19.

Cut #4 lesson: 73.7% is the exact same 14/19 as the "1-2 sentences" failure — most
likely the same ~2 factual tasks fail whenever factual's output budget drops below its
cliff, which now sits somewhere in (60, 120] words. "Under 120 words" is the proven
floor; factual's output is load-bearing for the judge (explanatory completeness, not
just the fact). **Phase B is closed**: best state is cut #3 (`86cf241`-equivalent,
5085 tokens, gate held). The path to <4000 now runs exclusively through Phase C
(shedding the ~100/call minimax prompt tax by moving categories off LARGE).

Cut #3 lesson: the actual saving (−175) undershot the probe-based prediction — real
math/logic answers under the old "brief steps" wording were already shorter than the
probe's worst case, and/or the judge set holds fewer math/logic tasks than the ~5
assumed. Output-side instruction cuts on math/logic are now largely exhausted; what
remains is factual verbosity (cut #4 measures it) and the structural minimax prompt
tax (~700–800/run) that only a Phase C tier move can touch.

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
