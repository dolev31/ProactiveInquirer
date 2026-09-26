# Measuring whether the agent is getting more proactive

Two axes, both load-bearing, and they answer different questions. A gain on one is not a gain on
the other, and a claim that does not say which is not a claim.

    VERTICAL    did it pursue a need that only became NAMEABLE after earlier evidence arrived?
    HORIZONTAL  did it cover independent needs the user never stated?

Everything below is computed from `(run directory, gold graph)` and nothing else. `score_run`
takes `(run, graph, turns, evidence, env_calls, ledger, records, answer)` -- all of it already on
disk -- so **every metric here can be added later and re-scored over runs that have already
finished**. Adding one changes `scorer_hash`, which is the intended guard: re-score, never
backfill.

## Vertical

Already implemented and populated on tau2 (airline depths `{0:204, 1:129, 2:29}`, retail
`{0:592, 1:278, 2:122, 3:23}`).

| metric | reads | what a gain means |
|---|---|---|
| `dwr` | depth-weighted recall, weights preregistered | needs resolved, weighted toward deep ones |
| `cad` | C@d, indexed by depth | where in the chain the coverage actually is |
| `max_depth_reached` | deepest resolved node, -1 if none | the policy reached a level at all |
| `latent_discovery_rate` | latent needs resolved / latent needs that ever became reachable | it took the openings it was given |
| `newly_reachable_share` | share taken the turn they appeared | PROMPTNESS -- it pursued the chain rather than stumbling on it later |
| `precedence_violation_rate` | v resolved before its prerequisite u (lower is better) | a policy that genuinely follows the chain cannot violate |

`newly_reachable_share` and `precedence_violation_rate` were nominated here as the pair that
separates real vertical proactivity from luck, on the argument that a policy resolving deep nodes by
accident scores on `max_depth_reached` and fails both. **THAT CLAIM IS WITHDRAWN 2026-09-18 and the
measurement that withdraws it is in `artifacts/testsplit_plan_metrics_20260918/`.** Read on the
held-out test split at matched cost, trained against prompted at the same base, the pair does not
discriminate and one half of it points the other way:

- `newly_reachable_share` is NULL on every suite. musique +0.00581 [-0.07461, +0.08333] at n=86,
  wiki2 +0.01293 [-0.08621, +0.09483] at n=58, strategyqa -0.03571 [-0.17857, +0.00000] at n=14 with
  an upper endpoint of exactly zero. It is emitted only where the arm's own trajectory makes it
  defined, so 56% to 94% of paired cells are dropped per suite and the surviving population is
  selected on both arms' behaviour. It decides nothing in either direction.
- `precedence_violation_rate` excludes zero in the ADVERSE direction on musique:
  **+0.06530 [+0.01182, +0.12687] at n=134**, and lower is better, so the trained policy violates
  prerequisite order MORE than the policy it was trained from. strategyqa +0.00000
  [-0.05556, +0.03968] at n=63 and wiki2 +0.01989 [-0.00852, +0.06600] at n=88 both span zero.

The instrument can take the other sign on this population -- 19 distinct per-task deltas on musique,
and the unmatched strategyqa cell is -0.05778 [-0.12342, -0.00444] -- so the adverse cell is a
finding about the policy and not a degenerate contrast. What the same record measures on the depth
side is a gain at equal cost (`dwr` +0.13764 on musique and +0.08484 on wiki2, `max_depth_reached`
+0.29830 and +0.07530, all excluding zero), which is a separate measurement and is not an offset:
depth coverage improves and prerequisite ORDER gets worse, on the same runs.

So neither of these two is a luck discriminator on the evidence available. Nothing here licenses a
replacement pair. A quantity nominated as a discriminator has to be measured as one before it is
described as one, and this pair was not.

## Horizontal

`facet_breadth` was the only horizontal metric and it is defined over `gold_facets`. **MEASURED:
tau2 carries zero facets on 43 of 43 airline graphs and 112 of 112 retail graphs**, so the whole
family scored `0 / 0` -- a number that reads as "no breadth" rather than "not measured here" --
while vertical was fully populated on the same graphs. Two axes claimed as crucial, one of them
unmeasurable.

| metric | reads | what a gain means |
|---|---|---|
| `breadth_components` | independent lines of inquiry resolved | it thought of separate things, not one thing deeply |
| `breadth_recall` | components touched / components that EXIST | normalised, so one line out of nine cannot score 1.0 |
| `breadth_singleton_share` | share of components that are lone nodes | HEALTH CHECK, not a claim: high means breadth is measuring annotation gaps |
| `facet_breadth` | distinct gold facets touched | kept for suites that have facets; NaN-shaped on tau2 |

A component is a weakly-connected group of the prerequisite DAG. Two needs joined by a chain are
one line of inquiry -- you had to know the first to name the second -- and two needs in different
components are independent things the agent had to think of separately. Only PREREQUISITE edges
merge; merging on any other kind would turn an annotation choice into a breadth claim.

Measured on the real graphs: airline `min 1 / median 2 / max 28` components per task, retail
`min 1 / median 5 / max 11`. So the axis has range to move on.

Not a count of resolved roots: six needs down one chain and six across six chains give the same
node count always, and can give the same root count. Components are what separates the axes.

**WITHHELD ON A GRAPH WITH NO CHAINS, and NaN rather than 0 on a graph with no nodes.** A graph
carrying no prerequisite edge at all has one component per node, so "independent lines of
inquiry" there counts annotation gaps: airline task 18 has 28 components over 28 nodes, 90% of
airline components and 85% of retail ones are singletons, and only 49% of airline graphs (72% of
retail) carry even one edge. The scorer emits no breadth row for such a graph, and
`breadth_singleton_share` rides alongside the ones it does emit so the fragmentation is in the
table rather than buried in the numerator.

## The joint claim, which is the one the paper actually makes

Neither axis alone says the agent got better. A policy that interrogates the customer to
exhaustion scores well on both.

| metric | status | why |
|---|---|---|
| `user_followups_given_success` | implemented | follow-ups on runs that COMPLETED the task; absent on failures, because a failure scoring 0 follow-ups would take the best anticipation score in the table for having failed |
| `n_user_followups` | implemented | the anticipation endpoint. **The 11.9% banking reduction recorded here is WITHDRAWN 2026-09-17 and must not be quoted.** It came from the parquet path, which still lacks `environment.prefix_user_turns` although the schema and compaction have declared it, so the metric there computes user turns minus one and charges every fork for its prefix: 306 of 384 scored fork runs are inflated by a mean of 2.45 turns (max 6). Paired at task and seed that population is also fork-mismatched on 99 of 359 cells (28%), including 83 fork-against-unfork pairings. The prefix fallback DOES cancel in `fork_report`, which reads status files where the field is present on all 408 published fork runs with zero nulls, so fork-report levels are unaffected. Requires a compaction re-run before any figure from the parquet path is quoted |
| `tau_reward` | implemented | did it actually finish |

The claim is a PAIR: more needs resolved per user engagement, at equal or better success. Reported
as two numbers, never collapsed into one -- a ratio hides which half moved.

## Stopping

More proactivity is only a gain if the agent also knows when to stop, and the unit of that
question is a **decision point, not a run**. One episode offers the policy `n_asks + 1`
decisions, `t = 0..n_asks`; a run-level summary collapses them into the last one, so a run that
asked six questions after it was already done and then stopped scored a perfect 1.0 with the six
wasted questions invisible.

    DONE AXIS   frontier_q#t >= 1 - 1e-12 -- the required-evidence coverage the policy HOLDS
                when it decides at t. Undefined points are SKIPPED, never filed as "not done".
    STOP AXIS   ASK at every t < n_asks. STOP at t = n_asks ONLY under stop_reason =
                'policy_stop'; under 'budget' / 'max_turns' the HARNESS halted the episode and
                the policy was never consulted, so that state carries no decision.

| metric | reads | what it is for |
|---|---|---|
| `stop2x2_n_done` | decision points where required coverage was already complete | the denominator of P(STOP \| done) |
| `stop2x2_n_stop_at_done` | of those, the ones where it stopped | the numerator |
| `stop2x2_n_not_done` | decision points with required evidence still missing | the denominator of P(ASK \| not done) |
| `stop2x2_n_ask_at_not_done` | of those, the ones where it asked | the numerator |
| `stop2x2_n_forced_stops` | final states the harness halted | EXCLUDED from the cells, counted so the exclusion is visible |
| `stop2x2_asks_after_done` | questions bought at states already done | the cost the two cells hide, in the unit an operator pays in |

`stop_undershoot` is **not** the done axis and never was one: it is `max(0, k* - k_hat)` with
`k*` the argmax of a ladder that is monotone in `k` and `k_hat` its last index, so `k* <= k_hat`
identically and the metric is 0 on every run ever scored -- 1,395/1,395 on each gate parquet, and
1 distinct value on every store on disk including 224,532 rows of `scores/parquet`. NO positive
control for it exists anywhere in this project, so it is STRUCTURALLY DETERMINED and must never be
reported as a null result. Its "not done" cell was empty by construction.

Counted per state and emitted per run, so a table may sum them over any grouping. The same cells
are computed at gate time by `pinq_train.gate._stop_2x2`; import-linter contracts forbid the two
packages from sharing code in either direction, so `pi_eval.metrics.stopping` is a deliberate
second implementation and `tests/test_stop_2x2_in_the_scorer.py` pins the two equal cell by cell.
**No row at all** when no decision point carried a coverage reading — six zeros there would read
as a policy that decided correctly on a task that was never measured.

## Filtering: what a row must carry to be selected on later

Every field below is already written per turn (`turns.jsonl`) or per run (`status.json`), so a
filter can be written after the fact:

    per turn    turn_idx, question, target, action_kind, depth_pred, n_new, new_uids,
                parent_uids, retrieved_uids, subset_hash_before/after, policy_conf, spec_bits
    per run     n_asks, n_turns, n_user_turns, n_prefix_user_turns, n_env_calls_continuation,
                stop_reason, terminated_prematurely, n_errors_agent/user, foreign_trace_sha,
                foreign_prefix_k, arm_id, seed, split, code_version, upstream_pins
    per unit    first_turn_idx on every evidence unit -- WHEN a fact first arrived

Joining `new_uids` to `gold_ev_uids` gives each turn the DEPTH and DISCOVERABILITY of what it
resolved, which is what makes an ask classifiable as vertical or horizontal after the fact. That
join is why nothing here needs recording now: `first_turn_idx` plus the gold depth already say
whether a need was taken the turn it became reachable.

## Rehydrated runs: an answer-text metric can read the wrong text

`scripts/restore_corpus.py` rebuilt 21,001+ run directories after the 2026-09-03 `rm -rf`, from
`scores/parquet` and the LLM response cache, and stamps each one
`manifest.json["rehydrated_from"] = "scores/parquet + cache"`. This is carried as a SIDECAR
(`scripts/rehydrated_answers/sidecar.py`, `tests/test_rehydrated_sidecar.py`), a two-column
parquet built by reading `manifest.json` directly, NOT as a column on `runs.parquet` itself: a
first attempt added `("rehydrated", _bool)` to `pi_eval.schema.RUNS` and was measured, before
merging, to move `schema_hash()` -> `metric_defs_hash()` -> `scorer_hash()` (`score.py`'s own
formula) for every already-scored row in the shared store, not only the 786 this lane
investigated (hash pairs in `artifacts/rehydrated_answers_20260918/RESULT.md`). `graph_hash`,
`matcher_hash` and `METRIC_DEFS_VERSION` do not read `pi_eval.schema` and were unaffected; the
column was reverted for exactly the two that did.

**The mechanism.** The rebuild writes `outcome.json["answer"]["text"]` from
`cache[request_sha]["text"]` -- the ANSWERER call's RAW response -- but
`outcome.json["answer"]["n_words"]` from this same run's ORIGINAL `answer_n_words` in
`runs.parquet`, the count AFTER `FrozenLLMAnswerer.answer` (`pinq_expt/components.py`) applies
`capped = " ".join(text.split()[:word_cap])`. The two fields of one file can come from different
processing stages of different samples: `text` can still carry a reasoning preamble or run past
the run's own `word_cap`, and -- because the response cache is first-writer-wins under a race
(`src/pi_run/cache.py`'s own docstring; the same mechanism as "the ledger records the discarded
sample") -- the cache entry today is not always the exact sample this run's own call received, in
either direction. **MEASURED 2026-09-18/19** (`artifacts/rehydrated_answers_20260918/RESULT.md`):
of 22,605 rehydrated runs, 786 hold `n_words != len(text.split())` -- self-contradicting on
nothing more than their own word count. By suite (contradicting / rehydrated, all splits):
tau2 672/1,199 (test only; `suite_id` `"tau2"` is BANKING specifically, not
`tau2_airline`/`tau2_retail`/`tau2_telecom`/`tau2_golden`, which carry zero rehydrated runs of any
kind), musique 106/17,639, strategyqa 4/3,327, drgym 3/342, synth 1/92, wiki2 0/6.

**A cheap re-derivation was tried and does not reliably work.** Re-applying the answerer's own
capping to the cached text (`scripts/rehydrated_answers/recount.py`) reproduces the original
`n_words` in only 2 of 20 sampled contradicting runs (10%) -- confirming the mechanism above is
dominated by the cache-race case, which no amount of re-capping can recover, not only by the
preamble/overrun case, which it can. The script is kept as a diagnostic and is not a repair.

**THE RULE.** Every `needs_answer=True` metric in `pi_eval.score.METRICS` (20 names, incl.
`answer_token_f1`, `answer_token_recall`, `answer_exact_match`, `answer_correct`,
`answer_wellformed`, `answer_n_words`, `citation_support`, `rnr_use`, `keypoint_recall`,
`keypoint_contradiction_rate`, `kpr_incremental`, `kpr_repetition_gap`, `answer_hedged`,
`answer_token_precision`, and the five `quality_*` judge dimensions) excludes a run with
`rehydrated=True`. Not only the 786 measured self-contradictions: a rehydrated run that
currently agrees on word count has been shown only to hold text of the scored LENGTH, never
shown to hold the scored TEXT. `task_success` is `needs_answer=False` (it reads only evidence
coverage, `pi_eval/score.py` line ~638) and is UNAFFECTED, correcting an earlier note
(`artifacts/compaction_loss_20260918/RESULT.md` section 4) that listed it as changed. Evidence-,
turn- and coverage-based metrics (`dwr`, `cad`, `evidence_coverage`, `n_turns`, `n_asks`, the stop
2x2, ...) read `turns.jsonl`/`evidence.jsonl`, which the rebuild copied unchanged from this run's
own original `turns.parquet`/evidence rows, and are unaffected by any of the above.

**Exposure check.** Every `artifacts/*/run_ids*.{txt,tsv}` population file on disk (140 files
across 18 directories, discovered by glob rather than by the task names alone -- a `.txt`-only
glob would silently miss `frames`/`frames_frontier`/`frontier_strategyqa`'s `.tsv` files) was
joined against both the rehydrated set and the contradicting set: zero intersection in every
file, including banking's own 291 published test-split tau2 ids (a later `tier2_confirmatory`
grid, disjoint from the pre-2026-09-03 pool) despite tau2/banking holding 672 of the 786
contradictions overall. `contributions_on_test`'s one dynamic (non-file) population -- the
teacher-pin check, `grid_name = 'tier1_trained_qa_teacher'`, musique/strategyqa/wiki2,
split='test' -- is also zero: 0 of 1,219. `tau2 forks_test` draws on `suite_id IN (tau2_airline,
tau2_retail)`, which has zero rehydrated runs of any kind. Full command output:
`artifacts/rehydrated_answers_20260918/RESULT.md`.

## Rules any addition here must obey

- **A label is a property of one state and one observed object, never of a trajectory.**
  FutureConversationCoverage was deleted because a need the agent PREEMPTS never appears as a
  later utterance, so it scored a miss on exactly the success case.
- **Absent is not zero.** A suite with no user simulator, a task with no gold, an ungraded reward
  basis: all NaN or omitted. A fabricated 0 is indistinguishable from a measurement.
- **Cross-arm follow-up comparison is legal on LIVE simulator runs and illegal on logs.** Each
  arm generates its own user turns live; a log's turns exist because of what that agent did.
- **Do not reintroduce the deleted five** without reading why they went: Anticipation Depth
  (non-monotone), Proactive Precision as stated (circular), `P(QU>0)` (a property of ordering),
  Stop Regret in quality units (a perfect stopper shows regret), FutureConversationCoverage.
- **A rehydrated run's answer text is not a measurement, even where it agrees with `n_words`.**
  The `rehydrated` sidecar (`scripts/rehydrated_answers/sidecar.py`, kept OFF `runs.parquet` --
  see "Rehydrated runs" above for why) means the directory was rebuilt from the compaction and
  the response cache, not produced by the run; agreement on word count shows the cached text is
  of the scored LENGTH, never that it is the scored TEXT. Every `needs_answer=True` metric
  excludes it; evidence, turns and coverage are the run's own original measurements and are
  unaffected.
  See "Rehydrated runs" above.
