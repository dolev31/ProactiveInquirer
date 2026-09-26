# Which data trains this, and which data measures it

This is the decision record. **It holds no counts**, on purpose: numbers here would drift from
the code within a week, and the code can already print them.

```bash
pi suites map        # every suite, its role, and whether its numbers may carry a p-value
pi suites validate   # construct each adapter, take a view, retrieve once, join gold
pi suites audit      # disagreements between the registry, the loader and the preregistration
```

The declarations live in `src/pi_run/suites.py` and are enforced by
`tests/test_suites_registry.py`. If this file and that one disagree, that one is right.

## The two roles are independent

A suite is **mined** (its `train` split is exported into `data/rl/`), **measured** (its `test`
split reaches a paper table), both, or neither. These are not opposites, and the first draft
of the registry used a single `role` enum and could not express MuSiQue, which is both and
which the central claim rests on.

| | mined | measured | why |
|---|---|---|---|
| musique | yes | yes | the only real suite carrying depth ≥ 2; the vertical claim |
| strategyqa | yes | yes | decomposition is strategic and unstated. **NOT the horizontal claim** — MEASURED 2026-09-19: 0.0% of its 2,290 tasks have `n_facets` >= 2, so `facet_breadth`'s denominator is 1 everywhere and breadth cannot vary. See `artifacts/horizontal_axis_instrument_20260919/` |
| wiki2 | yes | yes | see below — it carries no claim of its own and is still load-bearing |
| drgym | **never** | yes | breadth at depth 0, judge-derived; secondary — eval-only by decision (user, 2026-09-15), and no shipped training file ever held a drgym row. **Cannot carry a `facet_breadth` endpoint** — MEASURED 2026-09-19: `n_prereq_edges` = 0 and `n_facets` = 0 on 100% of its 976 tasks; its multiplicity (mean 16.6) is all depth-0 seed nodes, i.e. structure entailed by x alone rather than evidence-seeking branching |
| tau2 | **never** | yes | the zero-shot transfer target |
| tau2_retail | yes | yes | a separate suite id exactly so tau2 can stay untouched |
| tau2_airline | yes | yes | the second trainable tau2 domain; upstream's own 30/20 split, so a public "train" trace can never land on a task we call test |
| convlog | yes | intended | real sessions; replay-only, and the evaluator is not written |
| synth | no | yes | closed form; a deviation is a harness bug, not a finding |
| userbench | no | yes | one of the two benchmarks we did not build |
| frames | **never** | yes | the other one; external multi-hop, scored on answers only |
| musique_x2 | **never** | yes | composed pairs of held-out MuSiQue TEST tasks (two questions, two facets, one constituent of depth ≥ 2); eval-only because its content is test content; exploratory. Built by `pi_eval.build.compose_build` into an isolated root; declared in `artifacts/composed_pairs_20260923/DECLARATION.md`. Composed natural tasks, never a natural benchmark |
| pare | no | no | cut |

## Six things worth knowing before changing any of it

**wiki2 cannot simply be dropped.** Its depth ≤ 1 structure makes the vertical axis
degenerate, which reads like a reason to leave it out of the paper. It is not: wiki2 sits in
the **pool** of six preregistered secondary endpoints — the four kill-switch coverage
contrasts (`inquirer_noevidence`, `parallel_replay`, `verbosity`, `compute_matched`), plus
`rnr_resolve` and `frontier_auc`. Removing it changes the denominator of six declared tests.
That is an amendment, to be made before the seal, not an editorial choice afterwards. The
`mechanism` axis in the registry exists to name this role.

**convlog may not carry a p-value.** It was built after `pi_eval.prereg` was written, so it
is not named in it. It is marked `exploratory`: point estimate and CI, never a p-value, until
the preregistration is amended. This is the field that stops a number reaching a confirmatory
table because it happened to be available.

**Three roles moved on 2026-09-15 (T20), and the document they answer to moved with them.**
The registry now describes the programme the paper is RUNNING — the trained one, whose stage 1
is a separate sealed file — while `pi_eval.prereg`'s module-level endpoints are the PROMPTED
era's and are not amended by it.

| suite | was | is | why |
|---|---|---|---|
| `tau2_retail` | exploratory | **confirmatory** | its fork pairs are the trained programme's second confirmatory family: 34 fork points over 25 task ids, 102 pairs, an exact two-sided sign test on follow-up turns paired with `tau_reward`. That is the preregistration being amended, by a NEW stage file. |
| `tau2` | confirmatory | **exploratory** | banking's success sits at the floor, it carries zero legal fork points, and its own endpoint was already declared DIRECTIONAL (MDE 22.1pp at n=97) — so the registry was promising a claim the preregistration itself said the suite is not powered to make. |
| `drgym` | confirmatory | **exploratory** | eval-only and judge-derived (decisions D2/D8): a confirmatory row here spends multiplicity on a transfer claim whose effect must clear the sigma_J noise floor before it may be cited at all. |

**The role is declared, not derived.** `pi_eval.prereg.SUITE_ROLES` is where a suite's
`confirmatory | exploratory | calibration` status is written down, and `default_stage1()`
seals it as `suite_roles`. It used to be read off the endpoint tuples, which conflates two
different statements — that a document *defines* a test on a suite, and that it *authorises a
confirmatory claim* from it. They come apart in both directions: `tau2` has a primary endpoint
here whose own rationale already says "DIRECTIONAL, NOT POWERED", and `tau2_retail`'s fork pair
is confirmatory in the trained programme whose endpoints are sealed in its *own* stage file.
The endpoints, primaries, margins and seeds are untouched by the role change.

`pi suites audit` cross-checks the registry's roles against a stage 1 and **prints which one**:
`--stage1 <path>` if given, else a sealed `prereg/stage1.json`, else the module's own
declarations. All three agree today —

```text
$ pi suites audit
registry audit: clean (endpoint roles checked against pi_eval.prereg (no sealed stage 1))
$ pi suites audit --stage1 artifacts/prereg_draft/stage1.draft.json
registry audit: clean (endpoint roles checked against .../stage1.draft.json)
```

— and a payload written before the field existed (the draft is one) falls back to its endpoint
suites, so the checker can still read the document it was added to agree with.

**tau2 is eval-only and `split_of` enforces it.** `EVAL_ONLY_SUITES` returns `test` for every
task, so no exporter can emit a training row from it. It is defined **once**, in
`pinq.splitting`; `pinq_train.split` and `pi_run.serve.score` re-export it and
`tests/test_serve.py` pins that the three names are the same object. It used to be three
literals kept equal by a test, which is a duplication waiting for its first divergence.

**drgym joined that set on 2026-09-15, and it is the only member that joined by a decision
rather than by a property of its dataset.** tau2 cannot be split (97 tasks), userbench would
read as an external validation, FRAMES ships no train split at all — each of those is an
argument about the benchmark. drgym is trainable and was simply decided against: it was designated to carry
the horizontal endpoint, and a training row drawn from it contaminates the suite whose number
that endpoint is. (CORRECTED 2026-09-19: drgym cannot in fact carry a `facet_breadth` endpoint --
`n_facets` is 0 on 100% of its tasks, so the metric is undefined there. The eval-only decision
stands on its own grounds; only the stated reason about the horizontal endpoint was wrong.) Two facts make the change cheap and one makes it necessary. Cheap: the
registry already said `mines_training_data: no`, and **no shipped training file ever held a
drgym row** — `data/rl/*.jsonl` grep to zero and `data/rl/sft.manifest.json` names only
musique, strategyqa, synth and wiki2. Necessary: `runs.parquet` carries drgym runs stamped
`split=train` from campaigns that predate the decision, so "we did not happen to mine it" was
a fact about the last export and not about the next one. The stale stamps are **left alone**
— `split` is not in `pinq.ids.SEMANTIC_FIELDS`, so rewriting them would move no `run_id` and
buy nothing; what changed is that `assert_trainable` and `assert_evaluable` now refuse them by
name, and every drgym run stamped from here on is `test`.

**FRAMES is scored on its answers and nothing else, and that is a property of the
benchmark.** It ships no need-graph and never will, so every coverage, depth, facet and
frontier-Q metric is *undefined* there rather than zero. `pi score` takes an explicit branch
for it (`pi_eval.score.GOLD_FREE_SUITES`, declared beside `SuiteSpec.gold_free`, and
`pi suites audit` fails if the two disagree) which emits `answer_correct`, `answer_token_f1`,
`answer_token_recall`, `answer_hedged`, `answer_exact_match`, `answer_n_words` and the
cost/turn counts, and emits no structural metric at all. The branch is a literal and not
"this suite has no graphs on disk", because that condition is also what a *failed gold build*
looks like, and those two must not be scored the same way. See the frames section below.

**For tau2 and tau2_retail, a `view()` that succeeds is the bug.** The only task text
available outside the Orchestrator is the customer's roleplay script, which inverts the
agent's role and hands over the user-private partition the ceiling is defined over. So
`Tau2Suite.view()` raises `Tau2NeedsOrchestrator`, and `pi suites validate` asserts the
refusal rather than reporting it as a failure — the same shape as `GoldAccessError` being the
firewall working. The first validator written here called that door and reported the lock as
a break-in.

## FRAMES: the external benchmark, and exactly what it can say

FRAMES (Google DeepMind, NAACL 2025, [arXiv:2409.12941](https://arxiv.org/abs/2409.12941),
Apache-2.0; `google/frames-benchmark` on Hugging Face at commit `429d8fd1`). 824 hand-written
multi-hop questions, each shipping the 2-11 Wikipedia articles that together answer it. In the
authors' "oracle articles" setting those articles are a closed per-question paragraph pool,
which is the same shape as musique's, so it runs through the existing loop, retriever,
Inquirer, Drafter and Answerer with no change beyond the adapter.

**Role: measured, never mined.** `frames` is in `pinq.splitting.EVAL_ONLY_SUITES`, so
`split_of` returns `test` for every task and `assert_trainable` refuses it. Unlike tau2 and
userbench there is not even a train split upstream to be tempted by -- which is a reason to
declare it here rather than to rely on that absence, because "upstream ships no train split"
is a property of today's dataset card and this is a property of the code.

**What it can measure.** The answer endpoint on questions that genuinely require a chain
(`answer_correct`, `answer_token_f1`, `answer_token_recall`, `answer_hedged`), the cost and
turn counts, and the success-versus-spend frontier. The claim it supports is the one the
in-domain suites cannot: *the trained policy's gain is not a property of the suites it was
mined from*.

**What it cannot measure, and this is not a gap to be filled.** No `evidence_coverage`, no
`cad`/`dwr`, no facet breadth, no latent-need rate, no `rnr_*`, no frontier-Q, no
`stop_overshoot` -- all of them are functions of a need-graph FRAMES does not have. It also
has no user simulator, so it carries no follow-up endpoint. The depth metrics stay on
musique; the user axis stays on the tau2 fork pairs.

**No p-value.** `endpoint_status` is `exploratory`: FRAMES was chosen on 2026-09-11, long
after `pi_eval.prereg` was written, so it gets a point estimate and a clustered CI and
nothing else until the preregistration is amended. The judge-protocol accuracy the FRAMES
paper reports needs that same amendment -- `judge_derived_suites()` and `judge_arm_pairs()`
both read the seal, so no judge fires on frames today.

**Three build decisions that are measurements, not preferences** (argued at length in
`pi_eval/build/frames_build.py`):

- *Rendered HTML, not `prop=extracts`.* The plain-text extract API silently drops every
  table. Measured on `List_of_tallest_buildings_in_New_York_City`: the extract is 17,625
  bytes of lead prose and the gold answer to that question (`37th`) is not in it. 236 of the
  824 questions (28.6%) are tagged `Tabular reasoning`, so a plain-text pool would make those
  unanswerable by construction. The builder emits one paragraph per table row with the column
  headers inlined.
- *The revision live at 2024-09-01, not today's.* FRAMES was released in September 2024 and
  several prompts say "as of August 2024" in words. The current revision of that same article
  is dated 2026-08-27. Every page is pinned to its `oldid` and the revid is in the manifest.
- *A per-page sha256, verified at load.* No other adapter here checks a hash at load and none
  needs to; this is the only suite whose source is live and mutable. `corpus_hash` pins the
  directory *name*, which catches a rebuild and cannot catch a byte edited in place.
  `FramesSuite` raises `CorpusTampered` on a mismatch.

**`word_cap` is 50, not musique's 30.** Measured over the 824 gold answers: 97.3% are <= 30
words, 99.5% are <= 50, maximum 206. `answer_correct` is a token-subsequence test, so a gold
answer longer than the cap can never be contained in an answer and scores a structural zero --
22 tasks at a cap of 30, 4 at 50. Those four are `frames_0324` (206 words, an enumeration of
Nobel laureates in Physics 1901-1920), `frames_0335` (55), `frames_0811` (54) and
`frames_0813` (66). Their `answer_correct` is not interpretable and `answer_token_recall` is
the number to read for them.

**The build, as measured** (`scripts/build_frames_corpus.py --root .`, corpus
`ad2841f391ba94b9`, 2026-09-12):

```
tasks      824
articles   wanted 2518  fetched 2511  failed 7
paragraphs 260923  per task min 4 median 255 max 2844
bytes      corpus 100,484,499  raw cache 78,135,228
wall       26.4 min total (23.9 first pass + 2.5 resuming the failures)
```

Every one of the 2,511 pages is pinned to a revision at or before 2024-09-01 (latest
`rev_timestamp` in the manifest: `2024-08-31T23:57:17Z`). The build is resumable and
idempotent: a third run refetched nothing, took 6 seconds and produced the same corpus hash.

The 7 remaining failures are real and named in the manifest: four Wikipedia stubs that render
no paragraph over the 80-character floor, and three titles that no longer exist -- one of
which, `Pokémon (NOT REQUIRED, BUT HELPFUL)`, is an editorial note upstream wrote into its own
link column. They leave 7 tasks one article short -- `frames_0021`, `frames_0023`,
`frames_0141`, `frames_0199`, `frames_0540`, `frames_0580`, `frames_0644`. That is a *corpus*
gap and not a gold gap (all 824 have a gold answer), so it is deliberately not recorded in the
registry's `gold_gap_tasks`, whose meaning is "has no gold graph and is supposed to have
none"; it lives in the registry note and in the manifest's `tasks_with_missing_pages`. Five more links are not article URLs at all (a search box, a `Module:` page,
a `Category:` page, a `w.wiki` short link, a `simple.wikipedia.org` page) and are refused
rather than guessed at.

**Two upstream link defects cost 92 articles until they were found,** and both are the kind
that reads downstream as a policy failure rather than a build gap:

- 7 rows pack two or three URLs into a *single* `wiki_links` element, comma-separated. Read as
  one string the whole thing becomes one nonexistent title and all of its articles vanish.
- 1 row is double-URL-encoded (`Men%2527s`), so one unquote pass leaves a title that does not
  resolve.

A third was self-inflicted and worth recording: the first full build ran 8 concurrent fetches
with a linear backoff and lost **84 of 2,474 articles to HTTP 429**, leaving 59 tasks with a
short pool. Honouring `Retry-After` and dropping to 3-4 workers took that to zero.

**The pool is two orders of magnitude larger than musique's,** and any figure putting the two
frontiers side by side has to say so: ~20 paragraphs per musique task against several hundred
here -- measured: median 255 paragraphs per task, maximum 2,844. At a fixed `k=5` that is a far
harder needle, which is a property of the benchmark and not of the policy.

**Contamination, measured rather than asserted:**

```bash
python scripts/check_frames_contamination.py --root . --frames-root .
```

```
FRAMES questions          824
  musique      corpus 38f5afb69fb7ea18     tasks    800  train    456
  strategyqa   corpus ee8291a32bb761de     tasks   2290  train   1376
  wiki2        corpus 309cefe9519b79c5     tasks  12576  train   7636
train questions compared  9468
normalised exact matches  0
near-duplicates (J>=0.8)  0
max Jaccard observed      0.3889
```

Zero exact matches and zero near-duplicates over 7.8M question pairs. The maximum Jaccard
actually observed is printed on purpose: a near-duplicate check that reports zero is
otherwise indistinguishable from one that is broken, and 0.3889 -- two unrelated questions
that happen to share the multi-hop phrasebook -- is what makes the zero a measurement.

## Where the training data actually comes from

Rollouts on a suite's `train` split are forked at states where a real choice existed
(`frontier_size >= 2`), the candidates are ranked outcome → quickest → gain, and same-state
pairs become DPO rows. `pi suites map` prints which suites feed this; `pi train status`
prints what has been exported.

Two properties of the current export are worth stating because neither is visible from the
row counts:

- It is overwhelmingly one suite. Resuming the strategyqa and wiki2 fork campaigns is the fix,
  and it is a diversity argument rather than a volume one: a policy trained on one suite's
  phrasing is a policy that learned that suite.
- Anticipation only separates a preference pair at states where evidence has already arrived.
  Forks concentrated at turn 0 sample the states where the target behaviour cannot occur.

## What is not ready, and what each blocker actually needs

Run `pi suites validate` for the current list. The standing blockers are of different kinds,
and the distinction matters because they have different fixes:

- **drgym** — NOT BLOCKED, and this file said it was. The tier3 grid ran on 2026-08-28 for
  $16.87: 342 runs, all `ok`, 112 per arm, scored. The claim came from two mistakes made on
  the same day: a `grep` over `manifest.json` that missed on whitespace and reported zero
  drgym runs, and `pi suites validate` probing the retriever with the raw task question when
  the replay cache holds the sub-queries the runs actually issued. Both are fixed. What does
  remain on drgym is analysis hygiene rather than data collection: three or four distinct
  `scorer_hash` values are pooled in the eligible set, and its runs are split
  train 201 / dev 51 / test 90 — which the split clause below now resolves: 174 train and 51
  dev rows passed the split-blind predicate and none of them can reach a table any more.
  (**Re-measured 2026-09-15** on the current `runs.parquet`, after drgym was made eval-only,
  that split-blind count was train 174 / dev 51 / test 81. Eval-only stopped those rows being
  MINED — the export doors refuse drgym by name — and the split clause is what stops them
  reaching a *table*; they were two different repairs and both have now landed.)
- **tau2_retail** — two of its tasks have no gold graph and would score as a silent miss.
- **convlog** — replay-only by construction. `ConvlogSuite` implements one method, is not a
  `TaskSuite`, and `load_suite` cannot reach it; the replay evaluator that would grade two
  policies against one rendered prompt is described in `render.py` and not implemented. The
  registry marks convlog as measured, which records an intent the code does not yet satisfy.

## The rule the whole file serves

Every arm runs against a `test` split it never trained on, every mined suite exports only
`train`, and the two facts are enforced in five independent places listed in
`src/pinq_train/split.py`. `pi suites audit` fails if a suite is ever declared both a training
source and eval-only — the one combination that voids the strongest claim in the paper and
leaves no other trace.

## Eligibility filters on split — decided 2026-09-15

`pi_eval.report.ELIGIBLE` decides which runs may reach a table. It filtered on status, gold
exposure, dirty trees, pilots, canaries, the firewall and reconciliation — **and not on
split**, and neither did any confirmatory grid; only `tier1_trained.yaml` and
`latent_trainset.yaml` declare one. It now carries `AND r.split = 'test'`.

```bash
pi suites eligibility     # prints what the clause removes, per suite; exits 0
```

Nothing was contaminated while every arm was prompted: a policy that never trained cannot have
trained on these tasks. Two things break at the first trained arm, and both are silent:

- `inquirer_trained` against `inquirer_prompted` on rows drawn from `train` is a comparison
  on the trained arm's own training data;
- `tier1_trained` declares `split: test` and the confirmatory grids declare none, so the two
  grids measure different populations and comparing across them is invalid whether or not
  anything is contaminated.

**The decision, and why this one.** The alternative was `split: test` in each confirmatory
grid: narrower, it would have left existing tables alone, and it is one forgotten line away
from recurring in the next grid anyone writes. A predicate is enforced everywhere and cannot
be forgotten by a grid, which is the argument `report.py` already makes for itself in its own
first paragraph — eligibility is a predicate, not a habit. `pi suites eligibility` had exited
1 since the day it was written, and a standing refusal nobody can clear is a refusal people
learn to ignore.

**What it cost, measured the day it landed** over `scores/parquet/runs.parquet`:

| | train | dev | test |
|---|---|---|---|
| eligible before the clause | 34,155 | 1,976 | 1,537 |
| eligible after | 0 | 0 | 1,537 |

By suite, the rows that left: musique 14,152 train + 1,368 dev, strategyqa 14,056 + 360,
wiki2 4,548 + 248, tau2_airline 493 train, tau2_retail 906 train. What remains eligible is
tau2_telecom 1,020, tau2 236, tau2_airline 229, tau2_golden 36, musique 8, wiki2 8.

**Every table rendered before this date was computed over a different population and must be
re-rendered.** The prompted-era tables are re-rendered from the test grids anyway, so no grid
is re-run: the grids already ran these tasks, and what changed is which of their rows may be
reported.

Two consequences to keep in view, neither of them a contamination:

- **strategyqa and tau2_retail now report nothing** — 16,092 and 906 rows respectively pass
  `pi suites eligibility`'s core predicate (the one without the two reconciliation clauses,
  which is why its counts run higher than the table above) and not one of them is test-split.
  That is a gap in the evidence, and the command names it on every run; it is closed by
  running those suites on `split: test`, not by widening the predicate.
- **drgym's 201 train-stamped runs** were refused by name at export and stayed table-eligible
  without this clause: 174 of them passed the split-blind core predicate. They are ineligible
  now. (Under the full predicate drgym contributes nothing today for a second, independent
  reason — all 342 of its rows carry `reconciled_tokens = FALSE` and `reconciled_docs =
  FALSE`.)

## tau2 airline and retail do not get their split the same way (2026-09-15)

`pinq.splitting.split_of` reads `FIXED_SPLITS` for **tau2_airline** and hashes the id for
**tau2_retail**. That asymmetry is deliberate and it is a measurement, not a preference.

The fork benchmark's evaluation grid is the distinct `task_id`s of the paired-fork runs on
disk -- arms `inquirer_prompted`/`self_ask`, `split='test'`, in `scores/parquet/runs.parquet`.
20 task ids for airline, 25 for retail; 102 pairs per domain (34 fork points x 3 seeds). Every
one of them must be `test`, or the headline is computed over tasks a trained arm may have seen.

**Airline needed the table.** Measured on this checkout, the hash splitter placed 14 of the 20
airline grid tasks in `train` and 2 in `dev`, leaving 4 in `test`:

    train  6 8 16 18 19 22 25 26 29 30 32 35 37 45
    dev    24 44
    test   2 13 31 48

Upstream's `domains/airline/split_tasks.json` (30 train / 20 test) puts all 20 in `test`, and
the runs on disk are stamped `split=test`. The public trajectories the prefixes are cut from
were produced against that split, so hashing ids ourselves lets a prefix drawn from an upstream
"train" trace land on a task we call `test` -- contamination that `assert_trainable` cannot see,
because it asks our own `split_of`. The 20 grid tasks are *exactly* upstream's test half, which
`tests/test_airline_fixed_split.py` asserts against the file rather than against a literal.

**Retail must not have it, and the same measurement is why.** Upstream ships a retail
`split_tasks.json` too (74 train / 40 test), but 13 of the 25 retail grid tasks sit in its
TRAIN half:

    104 105 15 19 22 37 67 76 84 88 91 95 98

The hash already places all 25 in `test` -- which is how the 204 retail fork runs came to be
stamped `split=test` in the first place. Adopting upstream's file for retail would *reclassify
rows that already exist* and contaminate the grid the table exists to protect. The defect points
opposite ways in the two domains, so the remedy does too.

An id a fixed-split suite does not assign raises `UnknownFixedSplitTask` rather than falling
back to the hash: a fallback would hand a task a split upstream never gave it and would be
indistinguishable from a typo in the table. No dev bucket is invented for airline -- upstream
ships train/test only.

**This is not a claim that retail or airline was trained on.** Under D2 neither contributes to
this paper's training data; the registry's `mines_training_data` flag records what the suite is
*for*, and the split rule records how its tasks would be partitioned if it were.
