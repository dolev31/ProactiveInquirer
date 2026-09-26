"""Turn recorded trajectories into an SFT / preference dataset.

WHY THIS READS PARQUET AND NOT pi_eval. `pinq_train` must never import the gold package --
the trainer reaches scores only over HTTP, through `pi_run.serve`. So the exporter consumes
the columnar artifacts the scoring stage already wrote and does its own split enforcement.

WHAT AN EXAMPLE IS. One decision point: the serialized state the policy actually saw, and
the action it took, with a scalar value attached. Rejection-sampling SFT keeps the argmax ASK
when its phi clears the judge noise floor.

WHAT THE STOP LABEL IS, AND THE BELIEF THAT WAS WRONG. This docstring used to end: "below
that margin the honest label is STOP, because 'this question was not measurably better than
not asking' is exactly what the policy should learn from a tie." That reads a property of the
SAMPLES as a property of the STATE. On a state where required evidence is still missing and
every one of eight candidates happened to ask badly, it taught the policy to stop exactly
where it should have kept going -- and it did, on 66.8% of the live SFT.

Whether stopping is right is a GOLD question: was the required evidence already in hand
before this decision? Rows carry that now (`done_before`, from `pi_run.cmd_train`), so the
rule asks it (`stop_rule = "gold_coverage_v1"`):

  * done before the decision            -> STOP        (`label_rule="stop_done"`)
  * not done, best ASK clears the floor -> that ASK    (`label_rule="ask_clears_floor"`)
  * not done, nothing clears the floor  -> NO ROW      (`n_no_target_dropped`)

The third case is the point: a state where every sample missed is evidence about the sampler,
and the exporter has no target to teach there. The reward's own stop term already punishes
the undershoot at rung 3.

AND WHAT THAT SILENCE COST, MEASURED. The third case is 24,538 states on
`data/rl/sft.manifest.json`, and it is right about an IMITATION target and wrong about the
policy it produced: re-read per decision point (docs/TRAINING.md 5.3.1-5.3.2, 2026-09-15),
every rung-1 checkpoint scores `P(ASK | not done)` 0.83-0.91 against the prompted base's
0.96 -- they stop at 9-17% of states where required evidence was still missing. Nothing in
the dataset ever said "you are not done, so do not stop, even though no sampled question was
measurably good", because those states emit no row at all.

A PREFERENCE CAN SAY IT WHERE A TARGET CANNOT. `stop_rule = "gold_coverage_v2"` is an
opt-in PAIRS rule (`export_pairs` only; `export_sft` is untouched and its manifest still
reads v1, because the SFT rule did not change). At a not-done state where at least one
candidate asked and NO candidate stopped, it emits ONE pair -- best-available ASK chosen
over `STOP_ACTION_JSON` -- counted in `n_stop_pairs_synth_notdone`. It asserts strictly less
than a target would: not "ask this question", which the floor says is unsupported, but
"asking beat stopping here", which `done_before is False` alone establishes. v1 remains the
default of record; see docs/TRAINING.md 3.2.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from pinq.actions import STOP_ACTION_JSON, action_kind_of
from pinq_train.split import assert_evaluable, assert_trainable, id_set_hash


@dataclass(frozen=True, slots=True)
class Example:
    suite_id: str
    task_id: str
    run_id: str
    turn_idx: int
    state_text: str  # exactly what the policy was shown
    action_json: str  # exactly what it emitted
    value: float  # phi_hat or terminal reward
    is_stop: bool
    candidate_rank: int | None = None
    # LATENT-NEED LABELS, resolved gold-side in `pi_run.cmd_train.latent_fields` and arriving
    # here as plain numbers -- `pinq_train` may never import `pi_eval` (contract 1). They are
    # dataclass FIELDS because `write_jsonl` serialises `asdict(it)`: a key that is only on
    # the row dict is computed, carried through the exporter, and then silently dropped on
    # the floor at write time.
    latent_depth: int = -1
    is_latent: bool = False
    newly_reachable: bool = False
    frontier_size: int = 0
    # `latent_depth == -1` means NO required node was resolved at this turn, and `is_latent`
    # folds that into False -- indistinguishable from a genuine depth-0 root, on 48.8% of
    # rows. `has_gold_node` is the distinction: "no identified need" is not "the need was
    # stated in the task".
    has_gold_node: bool = False
    # Did the TASK TEXT already name this need? `gold_depth >= 1` was documented as "never
    # nameable from the task statement alone" and is not: on 21 of 99 depth-3 items, three
    # independent raters unanimously said the task named it, because musique refers to its
    # intermediates descriptively. None means NOT MEASURED (no task text reached the
    # exporter), never "not nameable" -- defaulting to False would assert the stronger claim.
    nameable_from_task: bool | None = None
    # The policy's OWN latency claim. Audited, never scored, and deliberately absent from
    # `action_json` so the loss mask never supervises a content hash.
    parent_uids: tuple[str, ...] = ()

    # ---- PROVENANCE (AGENTS.md rule 1). Not optional and not defaulted to "": a row that
    # cannot name the instrument that scored it is not a training row. `rows_from_run` has
    # always produced these; they were absent HERE, so `asdict()` dropped them at write time
    # and the export on disk carried nine keys.
    scorer_hash: str = ""
    graph_version: str = ""
    matcher_id: str = ""
    reward_weights_sha: str = ""
    # THE INSTRUMENT: model pin + rendered prompts, collapsed to one digest. Two rows are
    # comparable on their questions only if this matches.
    pins_sha: str = ""
    # False on every EXPORTED row by construction -- a leaked row is refused -- but carried so
    # an audit of the raw rows can count what was rejected rather than trusting the manifest.
    leaks_gold_answer: bool = False
    # Contamination auditing. `split` is the RUN's stamp; `template_id` is what binds
    # near-duplicate MuSiQue permutations to one side of the wall (see MusiqueSuite).
    split: str = ""
    template_id: str | None = None
    # ---- REWARD, DECOMPOSED. `value` alone cannot be re-weighted: a future reward that
    # prices redundancy differently would have to re-score every run to recover rho.
    # `phi_tilde` is the ACCEPT key and `value` the RANK key -- two different questions that
    # were once asked with one number (see export_sft). Both are carried so a consumer can
    # re-derive either decision without re-scoring.
    phi_tilde: float = float("nan")
    rho: float = 0.0
    # ---- WHICH PARENT THIS FORKED FROM. A candidate is a separate run; without these the
    # row cannot be traced back to the decision point it was sampled at.
    branch_of_run_id: str | None = None
    branch_turn_idx: int | None = None
    # ---- EPISODE OUTCOME. The measurement that motivated adding these: over 448 musique
    # runs, complete evidence coexists with a wrong answer often enough that ranking on
    # per-turn evidence gain alone accepts trajectories that retrieved everything and still
    # failed. `answer_hedged` is carried beside `answer_correct` because hedge rate varies
    # 16.7%-83.3% ACROSS ARMS and silently drives every answer-quality number.
    evidence_coverage: float = float("nan")
    answer_correct: float = float("nan")
    answer_hedged: float = float("nan")
    # ---- THE QUICKEST-PATH SIGNAL. Turns from this state until all required gold evidence
    # was in hand; -1 when the run never got there. Per-turn gain scores "gain now" and
    # cannot distinguish a question that opens the chain from one that pays the same now and
    # dead-ends.
    turns_to_complete: int = -1
    # ---- THE BUDGET COHORT. None on rows that predate the field. Carried so a consumer can
    # tell a cap-8 candidate from a cap-24 one; the labels above depend on it.
    budget_cap: int | None = None
    max_turns: int | None = None
    # ---- WHICH BRANCH OF THE STOP RULE PRODUCED THIS TARGET. "stop_done" | "ask_clears_floor".
    # Without it a STOP row is indistinguishable from an ASK row that happened to be a STOP
    # candidate, and no audit can separate "gold says the task was finished" from any other
    # reason a STOP might appear. "" on rows written before the rule existed.
    label_rule: str = ""
    # 1/sqrt(rows in this task and kind), normalised to mean 1 within the kind. Balances
    # exposure across tasks WITHOUT deleting rows: the per-task cap at 12 removed 65% of ASK
    # rows and 70% of the evidence-bearing ones, every one a state no kept row represented.
    # A trainer that ignores the field trains on everything, which is the right default.
    sample_weight: float = 1.0
    # ---- THE GOLD STATE THE LABEL WAS DERIVED FROM: required-evidence coverage BEFORE this
    # decision, and whether that was complete. None means UNKNOWN (a legacy row), never False:
    # an unknown read as "not done" would let the exporter teach an ASK on a finished task.
    coverage_before: float = float("nan")
    done_before: bool | None = None
    # ---- THE ANSWER-BEARING NODE'S OWN EVIDENCE, before this decision (`pi_run.cmd_train`,
    # rule from `pi_eval.answer_node`). None is UNKNOWN -- the graph named no answer node whose
    # evidence could be checked -- and is NOT False, by the same asymmetry as `done_before`.
    # CARRIED EVEN WHEN IT DID NOT DECIDE THE LABEL: the v1 file and the answer-node file differ
    # only in `label_rule`/`is_stop`, so an auditor holding one of them can only tell which rule
    # a row obeyed if BOTH gold facts are on it. `sft_stop_label` on the manifest says which one
    # was in force.
    answer_node_covered_before: bool | None = None
    # ---- WHICH CODE, AND WHICH POLICY RENDERED THIS PROMPT. The corpus spans two loop cohorts
    # (see `conf/cohorts/fixed_loop.json`) and several arms; without these on the row, the
    # fixed-only ablation and the arm ablation cannot be selected from the artifact at all.
    # Appended last, in the same order as on PreferencePair, where the position is load-bearing.
    # `""` is "not recorded", never a guess: an audit must be able to tell an unstamped row from
    # one that ran under the fixed loop.
    code_version: str = ""
    arm_id: str = ""


@dataclass(frozen=True, slots=True)
class PreferencePair:
    """Two candidates from the SAME state.

    Same-state pairing is the point: the within-state contrast cancels V(s_t) exactly,
    whereas a terminal-reward policy gradient has to estimate it across tasks whose
    difficulty variance dwarfs the treatment effect.
    """

    suite_id: str
    task_id: str
    run_id: str
    turn_idx: int
    state_text: str
    chosen_json: str
    rejected_json: str
    margin: float
    len_delta: int
    # Carried so a preference pair can be filtered to the states the claim is about.
    latent_depth: int = -1
    is_latent: bool = False
    newly_reachable: bool = False
    # The two candidate RUNS being compared -- not the parent `run_id` above. This is the
    # only way to identify which candidates a pair came from, and the join key onto
    # `judgments.parquet` (keyed `run_id_a`/`run_id_b`) for human preference annotation.
    chosen_run_id: str = ""
    rejected_run_id: str = ""
    pair_id: str = ""
    # ---- PROVENANCE. Same reasoning as Example; see there.
    scorer_hash: str = ""
    graph_version: str = ""
    matcher_id: str = ""
    reward_weights_sha: str = ""
    pins_sha: str = ""
    split: str = ""
    template_id: str | None = None
    branch_of_run_id: str | None = None
    branch_turn_idx: int | None = None
    frontier_size: int = 0
    # Outcome of the CHOSEN candidate's episode. A pair whose winner retrieved everything and
    # still answered wrong is a pair that teaches the wrong lesson; carrying this is what
    # lets a consumer filter on it without re-joining against scores.parquet.
    evidence_coverage: float = float("nan")
    answer_correct: float = float("nan")
    answer_hedged: float = float("nan")
    turns_to_complete: int = -1
    # ---- THE REJECTED SIDE. Without these a pair describes only its winner, and the central
    # question -- does the chosen candidate pursue a latent need more often than the rejected
    # one? -- cannot be answered from the artifact at all. Measured on the 1,525-pair export
    # before this existed: ZERO pairs were joinable to their candidates, because `export_sft`
    # stamps the PARENT run id (the state key) and only the pair carries candidate ids.
    #
    # -1 and NaN mean UNKNOWN, and are deliberately not False/0: a row that carried no label
    # must not be recorded as measured-and-negative, or an audit cannot tell the two apart.
    rejected_is_latent: bool | None = None
    # THE ANTICIPATION KEY'S INPUT ON THE LOSING SIDE. Without it the artifact records the key
    # on the winner only, and no audit can check a single decision `_anticipates` made -- nor,
    # on the CONTROL export, whether ranking on outcome and speed selected for anticipation at
    # all. None means the row carried no label, never False: measured-negative and unmeasured
    # are different facts and an audit must be able to tell them apart.
    rejected_newly_reachable: bool | None = None
    rejected_latent_depth: int = -1
    rejected_turns_to_complete: int = -1
    rejected_answer_correct: float = float("nan")
    rejected_evidence_coverage: float = float("nan")
    rejected_phi_tilde: float = float("nan")
    # ---- THE BUDGET COHORT both sides ran under (equal by construction: the exporter refuses
    # a pair whose sides differ). None on pairs that predate the field.
    budget_cap: int | None = None
    max_turns: int | None = None
    # ---- WHAT KIND OF DECISION THIS PAIR IS ABOUT. "ask_ask": two questions, ranked on what
    # they retrieved -- every pair in every export before this one. "ask_stop": a question
    # against stopping, which is the decision the paper is about and which the artifact could
    # not express at all. They are NOT interchangeable in training: an ask_stop pair has a
    # ~60-character length asymmetry that no guard can remove (it is what the two actions
    # ARE), so the rung-2 loader selects on this field rather than pretending the categories
    # are one population.
    pair_kind: str = "ask_ask"
    # "recorded" (a candidate actually stopped at this state) | "synthesised" (no candidate
    # stopped and gold says the state was already done) | "synthesised_notdone" (no candidate
    # stopped and gold says the state was NOT done, so the ASK is the chosen side -- the
    # `gold_coverage_v2` rule) | "" for ask_ask. A synthesised side is not a rollout and has
    # no run id; this is how a consumer tells.
    #
    # THE TWO SYNTHESISED VALUES ARE OPPOSITE DIRECTIONS AND MUST NEVER BE POOLED. Both carry
    # `pair_kind="ask_stop_synth"` -- the STOP side is the same derived constant, and
    # `pair_kind_of` re-derives the kind from the payloads, which cannot tell them apart -- so
    # this field is the only witness to which way the pair points. Summing them would report a
    # stopping dataset and an anti-stopping one as one number.
    stop_source: str = ""
    # WHICH KEY ORDERED THIS PAIR (`DECIDED_BY`), and whether the exporter's rule ordered it
    # at all (`LABEL_SOURCES`). Both are recorded rather than left re-derivable, because the
    # re-derivation depends on a precedence that can change under it.
    label_source: str = "rule"
    decided_by: str = ""
    # ---- THE GOLD STATE THE DIRECTION WAS DERIVED FROM. `done_before` is what makes STOP
    # right; without it on the artifact an auditor cannot check the direction of a single
    # ask_stop pair. None on ask_ask pairs and on legacy rows.
    coverage_before: float = float("nan")
    done_before: bool | None = None
    # ---- WHICH CODE, AND WHICH POLICY. Appended LAST and deliberately so: `_pair` constructs a
    # PreferencePair with its first twelve arguments POSITIONAL, and a field inserted anywhere
    # earlier would silently rebind `margin`, `len_delta` or the latent labels -- a corruption
    # with no exception and no failing type check. `""` is "not recorded", never a guess.
    code_version: str = ""
    arm_id: str = ""


@dataclass
class ExportManifest:
    n_examples: int = 0
    n_pairs: int = 0
    # Pairs refused because both sides asked the SAME question. Counted, never silent: a
    # guard that drops rows without saying how many is a guard nobody can audit.
    n_identical_dropped: int = 0
    # Pairs dropped because their state was already at MAX_PAIRS_PER_STATE.
    n_over_cap_dropped: int = 0
    # Pairs refused because the two candidates were produced under different prompts or a
    # different model pin, so the contrast is not about the question alone.
    n_cross_instrument_dropped: int = 0
    # CROSS-COHORT. The same fact one level down: the two sides ran under different LOOP code.
    # `pins_sha` cannot see it -- the prompt template is byte-identical across the fix; what
    # changed is what got rendered into it (the OLD loop wrote a stale per-turn answer into the
    # history block of ~63% of evidence-bearing prompts). A pair asserts "at this state, A beats
    # B", which is a claim about the QUESTIONS only if everything else matched, and the measured
    # precedent is the prompt-era straddle: 266 pairs (18.9%), newer side chosen 35.3% of the
    # time (p = 1e-6) with the outcome-driven flips among them balanced -- a systematic cohort
    # difference ranked as if it were a difference in the question.
    #
    # THE RULE IS COHORT MEMBERSHIP, NOT COMMIT IDENTITY (see `_cross_cohort`), so this count
    # is only readable next to the definition that produced it: the same rows under a cohort
    # file with one more sha in it report a different number.
    n_cross_code_version_dropped: int = 0
    # WHICH COHORT FILE DECIDED. sha256 of the bytes of `conf/cohorts/fixed_loop.json` (or
    # whatever `--cohort-file` named). `""` means NO cohort file decided this export and the
    # guard fell back to comparing commits -- a fact about the artifact, not a default.
    cohort_file_sha: str = ""
    # WHICH COHORT THIS FILE HOLDS, and whether turn-0 rows were admitted from the other one.
    # "any" is the historical behaviour and the default, so an old manifest and a new one say
    # the same thing about the same file. These two plus `cohort_file_sha` are what makes
    # `n_cohort_refused` readable: the same corpus under a different membership set reports a
    # different number, and without the declaration a reader cannot tell which rule ran.
    cohort: str = "any"
    turn0_any_cohort: bool = False
    # INPUT rows the cohort selection refused, counted where the split refusal and the answer
    # leak are counted -- per row consumed, before any state is grouped. Reconciles with
    # sum(not cohort_keeps(r)) over the rows the exporter was handed, in the split it accepted.
    n_cohort_refused: int = 0
    # CROSS-CAP. Two candidates at one state from different budget cohorts. Their fork-turn
    # actions are comparable; their episode labels are not, and the label is what orders a
    # pair. Counted rather than hidden in the state key so the manifest says how many pairs
    # the two cohorts would otherwise have formed.
    n_cross_cap_dropped: int = 0
    n_states_mixed_cap: int = 0
    # Candidates refused because the QUESTION already contained the gold answer. Counted
    # loudly: this is the one contamination the reward actively rewards.
    n_answer_leak_dropped: int = 0
    # Which measure the length guard applied. "question" = `question_len_delta` (the parsed
    # question text); the pre-2026-09-01 exports measured the whole action_json and are
    # distinguishable by this field's absence from their manifests.
    len_delta_on: str = "question"
    # WHICH ORDERING PRODUCED THIS FILE. "outcome" is the CONTROL -- nothing in the ranking
    # references a latent-need label, so `scripts/validate_pairs.py`'s latent-pursuit check is
    # a genuine measurement on it. "anticipation" adds `_anticipates` to the precedence, which
    # makes that same check true by construction. The two are different artifacts and must
    # never share a filename or a manifest.
    rank_rule: str = "outcome"
    # Pairs the rule refused and a rater majority re-admitted, under `--rank rater` only. The
    # rule's own refusal counters (`n_below_margin_dropped`, `n_over_cap_dropped`) keep
    # counting every refusal, so the three artifacts stay comparable; these are the
    # re-admitted subset. The LENGTH guard is never forgiven -- on that bucket the
    # rater-preferred side is the longer question 71.7% of the time.
    # Candidates that share a state KEY but were rendered from different evidence. A fork
    # replays its parent's prefix and the retriever does not guarantee the same set twice, so
    # two rounds of forks off one parent can produce two different prompts -- measured, 5,536
    # characters against 3,573 at one state.
    #
    # THIS HAS NEVER FIRED. On the live corpus every such round also changed the model/prompt
    # pin, so `pins_sha` separates them first: the count of (state, pins_sha) combinations
    # carrying more than one rendered state is zero, and enabling this guard changed nothing.
    # It is kept because `pins_sha` covers the PIN and not the retrieval RESULT, and a round
    # that re-forks an already-forked state under an unchanged pin would drift invisibly --
    # `render_state` checks each row against its own hash, so every row would be correct, and
    # rung 2 trusts the exporter to have emitted one state per pair.
    # WHICH preference set produced a rank_rule="rater" ordering. Two artifacts can both be
    # "rater" and be different experiments -- one ordered by A6 candidate rankings, one by
    # A7's `reaches` axis -- and without this they are indistinguishable after the fact.
    # Over the SET, so a file renamed or re-sorted still identifies the same ordering.
    rater_prefs_sha: str = ""
    n_cross_state_dropped: int = 0
    n_rater_rescued_below_margin: int = 0
    n_rater_rescued_over_cap: int = 0
    # Pairs the RATER key actually ordered. A6 covers the states it was sampled over, not the
    # corpus, so a `rank="rater"` export leaves most pairs on the mechanical precedence. A key
    # that silently did nothing on 95% of the file would otherwise read as a ranking decision.
    n_rater_ordered: int = 0
    n_below_margin_dropped: int = 0
    n_len_dropped: int = 0
    # Rows whose `answer_correct` was blanked to NaN by the binary-gold hedge guard in
    # `pi_run.cmd_train._outcome_fields`: a hedged answer on a yes/no gold, where
    # `contains_answer` credits the refusal's own "no" (250 of 508 "correct" gold-"no" labels on
    # the 1,110 strategyqa pairs were hedges). Counted once per row at the point the exporter
    # reads its input -- before split refusal, the leak drop and same-state grouping -- so it
    # reconciles exactly with sum(binary_gold_hedged) over the rows file. A guard that blanks a
    # value without saying how often is a guard nobody can audit.
    n_binary_gold_hedged_nan: int = 0
    # ---- THE SFT STOP RULE, counted. `n_stop_done_before_dedupe` is states gold says were
    # already complete; `n_no_target_dropped` is states where evidence was still missing and no
    # candidate cleared the floor -- previously ALL of these were exported as STOP, which is
    # how `stop_share` reached 0.668. `n_done_before_unknown` is states whose rows predate
    # `done_before` (no gold-side signal, so no STOP can be derived).
    # ---- ASK-vs-STOP PAIRS, by direction and by origin. `n_ask_stop_undecided` is the
    # honest third outcome: the task was not done (so STOP is not right) AND the question
    # missed the noise floor (so ASK is not right either) -- no pair, rather than a coin flip.
    n_ask_stop_pairs: int = 0
    n_stop_chosen: int = 0
    n_ask_chosen_over_stop: int = 0
    n_ask_stop_undecided: int = 0
    # RENAMED from `n_stop_pairs_recorded`. The BELIEF that changed, not the code: the old name
    # read as "recorded ask_stop pairs in this file" and it never counted that. It is incremented
    # while the pair sits in `here`, and `MAX_PAIRS_PER_STATE` then trims `here`.
    #
    # MEASURED on the shipped export: manifest 1,958 against 1,592 rows carrying
    # `pair_kind == "ask_stop"` in `data/rl/pairs.jsonl` -- a 366-pair gap that two inventory
    # tables quote as a row count. THE STAGE IS THE CAP, NOT A DEDUPE: `export_pairs` runs no
    # dedupe at all (`n_exact_duplicate_dropped` is 0 on that manifest), and `n_stop_pairs_synth`
    # is 32,683 in the manifest AND 32,683 on disk, because a synthetic pair is appended outside
    # the cap. The synth counter therefore needs no qualifier and does not get one.
    n_stop_pairs_recorded_before_cap: int = 0
    n_stop_pairs_synth: int = 0
    # ---- THE `gold_coverage_v2` RULE, counted. Its OWN counter and not a share of
    # `n_stop_pairs_synth`: that one is STOP-chosen-because-done and this one is
    # ASK-chosen-because-not-done, and the two pull the policy in opposite directions.
    # Outside the per-state cap like its sibling, so this equals the row count exactly and
    # needs no `_before_cap` qualifier.
    n_stop_pairs_synth_notdone: int = 0
    # NOT-DONE states the rule could emit nothing at, because every candidate stopped and
    # there was no ASK to put on the chosen side. Incremented ONLY under v2: under v1 a
    # not-done state was never a candidate for synthesis, and a non-zero value on a v1
    # manifest would describe a rule that export did not run.
    n_notdone_states_no_ask: int = 0
    # Two STOP candidates at one state. Byte-identical by construction, so counting them as
    # `n_identical_dropped` would read as "both asked the same question", which is not what
    # happened.
    n_stop_stop_dropped: int = 0
    # RENAMED from `n_stop_done`, for the same reason and a different stage: it is incremented
    # as each STOP example is appended, and the EXACT-DEDUPE pass below then collapses byte
    # identical (task, state, action) repeats. MEASURED on the shipped export: manifest 36,381
    # against 29,876 rows with `is_stop: true` in `data/rl/sft.jsonl`, with
    # `n_exact_duplicate_dropped` at 19,463 -- so here the dedupe is real and the name says so.
    #
    # IT COUNTS STOP ROWS, whichever gold fact produced them. On an `sft_stop_label ==
    # "answer_node_covered_before"` manifest the "done_before" in this name is HISTORY, not a
    # claim about the rows: most of them sit on states that were not done. Read it with
    # `sft_stop_label`, never alone. Not renamed, because the name is load-bearing in every
    # manifest already written and a field that means one thing under two spellings is worse
    # than one whose spelling has to be read in context.
    n_stop_done_before_dedupe: int = 0
    n_no_target_dropped: int = 0
    n_done_before_unknown: int = 0
    # ---- WHICH GOLD FACT DECIDED STOP. `"done_before"` is the rule of record (pooled required-
    # evidence completeness); `"answer_node_covered_before"` is the L6.1 variant, which asks the
    # same question of the ANSWER-BEARING node alone. STAMPED ON THE MANIFEST because the two
    # files are otherwise indistinguishable -- same rows, same states, same provenance, a
    # different label on a subset of them -- and a checkpoint that cannot name which one it was
    # fitted on has no readable contrast.
    sft_stop_label: str = "done_before"
    # States whose candidates carry no `answer_node_covered_before` at all: the graph named no
    # answer node whose evidence could be checked (or the rows predate the field). Counted on
    # BOTH labels' manifests, because under `done_before` it measures how much of the corpus the
    # variant could never have relabelled, and under the variant it is the population that fell
    # through to the ASK branch for want of a gold fact rather than because of one.
    n_answer_node_unknown: int = 0
    # Candidates at one state disagreeing about it. Same treatment as `n_state_done_disagree`:
    # counted and skipped, never fatal (see that field for the 57-minute export this rule cost).
    n_answer_node_state_disagree: int = 0
    # Coverage says the state was complete while the matcher's frontier still holds an askable
    # node. Coverage decides (it is what the reward and `_complete_at` use); the disagreement
    # is counted so the two instruments can be reconciled rather than silently differing.
    n_done_frontier_disagree: int = 0
    # STATES REFUSED because their candidates contradicted each other about `done_before`.
    # Counted and skipped rather than fatal to the export: a 206,088-row export once died at
    # the 57-minute mark on ONE such state, and re-reading it afterwards found all 32
    # candidates in agreement -- the exporter had read those directories while a fork worker
    # was still writing them. The guard was right; killing the whole export was not.
    # `collect_rows` sets the standard this follows: a unit that cannot be exported is
    # counted, not hidden. A NON-ZERO value here on a quiet tree means a real inconsistency;
    # during a live fork campaign it means the export raced a writer.
    n_state_done_disagree: int = 0
    # THE PER-TASK CAP on `export_sft`. None = off, which is the default: a cap changes what
    # the checkpoint imitates, so it is never silent. MEASURED reason it exists: within-task
    # distinct-3 collapses as a task contributes more questions (0.81 at 2-3 per task, 0.13 at
    # 50+), and 127 tasks contributing 50 or more dragged the live export to 0.638 against a
    # 0.65 floor. The same rows capped at 20 score 0.666, at 10 score 0.701.
    max_examples_per_task: int | None = None
    n_over_task_cap_dropped: int = 0
    # Byte-identical (task, state_text, action) repeats. MEASURED: 14,275 of 25,734 uncapped
    # ASK rows, from parents re-run across days and forks-of-forks replaying one prefix under
    # different state keys. Removing them raises within-task distinct-3 (0.671 -> 0.792).
    n_exact_duplicate_dropped: int = 0
    # THE RULE AND THE SHAPE, on the artifact. An SFT file written under a different stop rule
    # is a different dataset, and the manifest is where a reader finds out which one they have.
    # One of `STOP_RULES`. On an SFT manifest this always reads `gold_coverage_v1`: v2 is a
    # PAIRS rule and `export_sft` has no branch it changes, so stamping v2 there would claim
    # the SFT rows were derived differently than they were.
    stop_rule: str = "gold_coverage_v1"
    stop_action_json: str = STOP_ACTION_JSON
    # WHICH POLICIES' SAMPLES THIS FILE HOLDS, and how many runs the allowlist refused. Filled
    # by `pi_run.cmd_train`, which is the only place that walks run directories; an exporter
    # given rows cannot know an arm was refused before a row existed. Empty/0 when a caller used
    # `export_sft` directly, which is honest: it applied no allowlist.
    included_arms: tuple[str, ...] = ()
    n_arm_refused: int = 0
    suites: list[str] = field(default_factory=list)
    train_id_set_hash: str = ""
    # WHERE THE ID SET ITSELF IS, relative to this manifest. `train_id_set_hash` alone proves
    # membership only to a reader who already HOLDS the set, and until this field existed the
    # set was built, hashed and dropped on the floor inside one function call -- which is why
    # `pinq_train.split` could call the hash "layer 3, so a later run can prove which ids a
    # checkpoint saw" while nothing was able to prove anything. `write_jsonl` writes the file
    # and fills this in; "" means an artifact written before the sidecar existed.
    train_ids_file: str = ""
    refused: dict[str, int] = field(default_factory=dict)
    margin_threshold: float = 0.0
    len_delta_max: int = 40
    # WHICH SPLIT THIS FILE HOLDS. "train" by default, so an old manifest and a new one say the
    # same thing about the same file. A dev export is a different artifact with a different
    # filename, and this is what a loader reads to refuse training on it -- `train_id_set_hash`
    # keeps its name because every consumer on disk reads it, and it is the id set of whatever
    # split this says.
    split: str = "train"


def _assert_split(row: dict, split: str) -> None:
    """The split gate both exporters pass every row through.

    ONE function, because two would be two chances for the dev file to enforce something the
    train file does not. `assert_trainable` is untouched and still refuses everything that is not
    train, which is what layer 2 of the firewall is; `assert_evaluable` is its held-out twin and
    must be told WHICH held-out split, since "not train" would let a test task ride into a dev
    export on the grounds that it, too, is held out.
    """
    if split == "train":
        assert_trainable(row["suite_id"], row["task_id"], row.get("template_id"))
    else:
        assert_evaluable(row["suite_id"], row["task_id"], row.get("template_id"), split=split)


def state_key(r: dict) -> tuple | None:
    """The STATE a row belongs to, or None if the row is not a decision at a sampled state.

    A candidate branch is a separate RUN by construction -- that is what stops it
    overwriting its parent's directory (see RunManifest.semantic_hash) -- so keying on
    run_id put every candidate in a group of one, and C(1,2) = 0. That is half of why
    `pairs.jsonl` has always been 0 lines; the other half was that nothing produced
    candidates at all. Candidates of one state are identified by the PARENT they forked and
    the turn they forked at.

    ONLY THE ROW AT THE FORK COUNTS. A branch also records the prefix it replayed (turns
    before the fork, identical across candidates) and its own continuation (turns after,
    each downstream of a DIFFERENT decision). Grouping the continuation would assert a
    same-state comparison that is not one -- exactly what `state_at` refuses to do.

    `is None` rather than `or`: branch_turn_idx 0 is the commonest fork and is falsy.
    """
    parent = r.get("branch_of_run_id")
    b_turn = r.get("branch_turn_idx")
    if parent and b_turn is not None:
        if int(r["turn_idx"]) != int(b_turn):
            return None
        return (r["suite_id"], r["task_id"], str(parent), int(b_turn))
    return (r["suite_id"], r["task_id"], r["run_id"], r["turn_idx"])


# NO STATE MAY DOMINATE THE LOSS. `export_pairs` emits all C(n,2) pairs per state, which is
# safe only under a FIXED n. MEASURED on the 1,525-pair export, n is not fixed: implied
# candidate counts run 2..12, pairs per state 1..74, and the top TEN states carry 19.8% of the
# dataset. n varies because candidates FAIL -- a rate-limited or errored branch is simply
# absent -- so the count a state gets is an accident of the proxy, not a property of the state.
#
# 12 sits just above the observed median of 8, so a typical state is untouched while the 74-
# and 48-pair tails are cut. With ~170 states that bounds any single state at about 1% of the
# dataset.
MAX_PAIRS_PER_STATE = 12


def _f(v: object) -> float:
    """float or NaN. NaN means UNKNOWN and 0.0 means measured-zero; conflating them would make
    an audit read an unlabelled row as a failure."""
    try:
        return float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")


def _action_text(x: str) -> str:
    """The text a candidate is ABOUT: the parsed question, normalised.

    Whitespace collapsed and casefolded, not raw JSON bytes: two renderings of one question
    -- a different separator, a doubled space -- are one question, and the policy could not
    learn the difference between them.

    Falls back to the whole normalised string when the action does not parse, which is the
    conservative direction for both consumers: an unparseable pair that happens to be
    identical is still noise, and unparseable junk is measured whole, not waved through.
    """
    try:
        obj = json.loads(x)
    except (TypeError, ValueError):
        return " ".join(str(x).split()).casefold()
    if isinstance(obj, dict):
        for key in ("question", "text"):
            if obj.get(key):
                return " ".join(str(obj[key]).split()).casefold()
        # NO QUESTION FIELD -> compare the WHOLE payload, not the empty string. Falling
        # back to "" made every action without a `question` key compare equal to every
        # other, which collapsed unrelated candidates into one and dropped real pairs.
        return json.dumps(obj, sort_keys=True, separators=(",", ":")).casefold()
    return " ".join(str(obj).split()).casefold()


def _same_action(a: str, b: str) -> bool:
    """Do these two candidates ask the same thing? Compared on `_action_text`."""
    return _action_text(a) == _action_text(b)


# PORTED VERBATIM from convlog-work 306fd52 (the pair-paraphrase guard, calibrated on the
# first exported ask_ask pairs; tests/test_pairs_paraphrase_guard.py came with it), which never
# reached main. RULES amendment 8 names it for the tau2 rerun's exploratory near-exact repeat
# rate; main's exporter does not call it.
_QUESTION_STOPWORDS = frozenset(
    "the a an of for on and or what is are with to its in at by this that their your my "
    # REQUEST AND POLITENESS WORDS carry no need. "Could you provide the user's email address or
    # full name and zip code for verification?" scored 0.46 against its plain twin because these
    # counted as content; the broader-ask case (real extra needs) stays a contrast at the same
    # threshold once they do not.
    "could can would please kindly provide tell me find check look up retrieve get pull show list".split()
)
# CALIBRATED on the first six ask_ask pairs exported: three real pairs (record vs customer) at
# token-set Jaccard 0.00, three rewordings of one order lookup at 1.00, 0.62 and 0.50. Trigram
# overlap read the same three at 0.06, 0.06 and 0.05 -- reordering a clause changes every trigram
# and no meaning. The threshold sits between the populations, not on either.
PARAPHRASE_JACCARD = 0.5
# AND A MINIMUM SHARED COUNT. The ratio alone dropped "where did he die?" against "when did he
# die?" (0.60): different needs -- a place and a date -- sharing three function-like words. The
# three real paraphrases share five or six content tokens (the order id, "status", "items",
# "shipping", "address"). Four separates a reworded lookup from a short question with one word
# changed, and leaves the calibration set exactly where the ratio put it.
PARAPHRASE_MIN_SHARED = 4


def question_tokens(action_json: str) -> frozenset[str]:
    """The content words of a candidate's question: lowercased, punctuation stripped, stopwords
    removed. Compared as a SET, because word order is what a paraphrase changes."""
    import re as _re

    text = _action_text(action_json)
    # An underscore inside a FIELD NAME is a separator: `product_id` and `product id` are the
    # same two words. MEASURED on pilot D: two wordings of one item lookup passed the guard at
    # Jaccard 0.36 because the field names were joined in one and spaced in the other. An
    # IDENTIFIER stays whole: `noah_ito_3850` is one token, and splitting it added three
    # tokens to one side of a pair that sits at the threshold.
    words: list[str] = []
    for w in _re.sub(r"[^a-z0-9_ ]", "", text.lower()).split():
        words.extend(w.split("_") if "_" in w and not any(c.isdigit() for c in w) else [w])
    return frozenset(w for w in words if w and w not in _QUESTION_STOPWORDS)


def is_paraphrase(a: str, b: str) -> bool:
    """Two candidates asking the same thing in different words. A pair of them is ranked by
    retrieval luck -- which units the same lookup happened to return -- not by the question, and
    training on it teaches the model to prefer one wording of a lookup over another."""
    ta, tb = question_tokens(a), question_tokens(b)
    if not (ta | tb):
        return True
    shared = ta & tb
    return len(shared) >= PARAPHRASE_MIN_SHARED and len(shared) / len(ta | tb) >= PARAPHRASE_JACCARD


def question_len_delta(a: str, b: str) -> int:
    """The length guard's measure, shared with rung 2's load-time re-check.

    Measured on the QUESTION, not the whole `action_json`: the guard exists to block
    "longer question wins", and the rationale beside the question is 60-120 chars of
    sample-varying prose that says nothing the pair is about. Measured before this change,
    the whole-JSON delta killed ~74% of same-state candidate pairs while kept pairs hugged
    the cap (median 17 of 40) -- rationale wording variance masquerading as verbosity.

    ONE function for both the exporter and `rung2_dpo.train.assert_length_guard`: two
    implementations of one guard drift, and the load-time check rejecting the exporter's
    own artifact is exactly the failure mode.
    """
    return abs(len(_action_text(a)) - len(_action_text(b)))


def _best_available_ask(asks: Sequence[dict]) -> dict:
    """The highest-valued ASK at a state, with a tie-break that does not read input order.

    `export_pairs` sorts its candidates with `sorted(..., key=-value)`, which is STABLE: on
    equal values the winner is whichever row the caller happened to hand over first, and the
    same corpus read in a different directory order exports a different pair. That is
    tolerable where the tie only reorders a C(n,2) enumeration that emits every combination
    anyway; it is not tolerable here, where the tie DECIDES the one pair the state gets.

    The tie-break is a digest of the candidate's identity, the same instrument
    `MAX_PAIRS_PER_STATE` (`pair_id`) and the per-task cap (`_task_cap_key`) use, and for the
    same reason: it is arbitrary with respect to value, length and diversity, so it cannot
    quietly select for any of them, and it is stable across runs and across input order.
    """
    return min(
        asks,
        key=lambda c: (
            -float(c["value"]),
            hashlib.sha256(
                f"{c.get('run_id') or ''}|{c.get('action_json') or ''}".encode()
            ).hexdigest(),
        ),
    )


def _caps(row: dict) -> tuple[int | None, int | None]:
    """The budget cohort a candidate ran under. See the cross-cap guard in `export_pairs`."""
    b, m = row.get("budget_cap"), row.get("max_turns")
    return (None if b is None else int(b), None if m is None else int(m))


COHORT_MODES = ("any", "fixed", "old")


def validate_cohort(mode: str, cohort: frozenset[str], turn0_any: bool) -> None:
    """Refuse a cohort SELECTION that cannot mean what it says. Called by both exports.

    Three ways to ask for nothing and be told you got something:

      * an unknown mode -- there is no default reading of "fixed_loop";
      * a mode with no membership set -- `fixed` would then keep no row and `old` every row,
        under a manifest saying a cohort decided. `pi_run.cmd_train._cohort` already refuses a
        file naming no members for the same reason; this is the library-side half, because
        `export_sft` is also called directly;
      * `turn0_any_cohort` under `any`, where every row is kept anyway. A flag that decides
        nothing while the manifest records that it was asked for is the "value that silently
        does not apply" failure, one layer down from a `--config` key nobody consumes.
    """
    if mode not in COHORT_MODES:
        raise ValueError(f"cohort_mode={mode!r}: expected one of {COHORT_MODES}")
    if mode != "any" and not cohort:
        raise ValueError(
            f"cohort_mode={mode!r} with no members: the membership set is empty, so this would "
            "keep nothing (fixed) or everything (old) while the manifest says a cohort decided "
            "the file. Pass the set `scripts/cohort_ids.py` wrote."
        )
    if turn0_any and mode == "any":
        raise ValueError(
            "turn0_any_cohort under cohort_mode='any': every row is kept already, so the flag "
            "decides nothing and the manifest would record a rule nobody applied. It is only "
            "meaningful with 'fixed' or 'old'."
        )


def cohort_keeps(row: dict, *, mode: str, cohort: frozenset[str], turn0_any: bool) -> bool:
    """Is this row IN the selected loop cohort? The one predicate, shared by both exports.

    `old` IS THE COMPLEMENT OF `fixed`, not a second list: a row with no `code_version` ran
    under no recorded code, which is not the fixed loop, so it is old. That asymmetry is
    deliberate and is the opposite of `_cross_cohort`'s, where an unstamped row is UNKNOWN and
    refuses to pair -- there the question is "are these two the same population", here it is
    "is this row one the fixed loop produced", and the honest answer to the second is no.

    TURN 0 IS CLEAN IN BOTH LOOPS. The OLD loop's defect was that the answer written into a
    turn's history block never saw the documents fetched to answer it; at turn 0 nothing has
    been asked and no evidence has been fetched, so there is no such answer and the rendered
    prompt is byte-identical under either loop. `turn0_any` admits those rows -- MEASURED on
    the shipped export, 4,115 ASK rows and 4,721 ask_ask pairs that are otherwise discarded
    for a defect that cannot have touched them.
    """
    if mode == "any":
        return True
    if turn0_any and int(row.get("turn_idx") or 0) == 0:
        return True
    member = str(row.get("code_version") or "") in cohort
    return member if mode == "fixed" else not member


def _cross_cohort(hi: dict, lo: dict, cohort: frozenset[str]) -> bool:
    """Did these two candidates run under DIFFERENT loop cohorts? See the guard in
    `export_pairs`.

    COHORT MEMBERSHIP, NOT COMMIT IDENTITY. What the guard protects against is a PRE-fix
    candidate (stale per-turn answers rendered into its history block) being ranked against a
    POST-fix one. Two commits that are both descendants of the loop fix -- or both ancestors of
    it -- rendered the same prompt, so a pair across them is a claim about the QUESTIONS, which
    is what a preference pair is allowed to be. Comparing shas refused those too: measured on
    the 2026-09-12 prompted-only control export, 448 pairs (1.1% of 38,858), all of them states
    whose candidates were forked under two commits of ONE cohort.

    `cohort` is the membership set, loaded once from `conf/cohorts/fixed_loop.json` by the
    caller (see `pi_run.cmd_train._cohort`) rather than re-derived here: membership is a git
    ancestry fact, and the trainer runs on a box that may hold no repository at all.

    EMPTY IS UNKNOWN, NOT OLD. An empty or missing `code_version` is refused against any
    stamped row -- including a row that is itself outside the cohort -- because "we could not
    check" and "we checked and it is outside" are different facts; a bare
    `(a in cohort) != (b in cohort)` would call them one population. Two unstamped rows still
    pair, the asymmetry `pins_sha` and `_caps` already apply.

    AN EMPTY COHORT FALLS BACK TO THE COMMIT. With no cohort definition in hand, no sha's
    membership is known, so the strict rule is the only defensible one -- and it errs toward
    refusing. `ExportManifest.cohort_file_sha` is how a reader tells the two regimes apart.
    """
    a = str(hi.get("code_version") or "")
    b = str(lo.get("code_version") or "")
    if not a or not b:
        return bool(a) != bool(b)
    if not cohort:
        return a != b
    return (a in cohort) != (b in cohort)


def _provenance(row: dict) -> dict:
    """The fields AGENTS.md rule 1 requires, pulled off a row and REFUSED when absent.

    Deliberately not defaulted. A `scorer_hash=""` is worse than the missing-field bug it
    replaces: an empty string looks like provenance to every downstream consumer and to
    `train_split_violations`, so a row that cannot name its instrument would be silently
    poolable with rows that can. Raising here is what makes the export fail loudly at the
    moment the row builder stops producing a field.
    """
    out: dict = {}
    for k in ("scorer_hash", "graph_version"):
        v = row.get(k)
        if not v:
            raise ValueError(
                f"row {row.get('suite_id')}/{row.get('task_id')}@{row.get('turn_idx')} has no "
                f"{k!r}. A training row that cannot name the instrument that scored it is not "
                "a training row (AGENTS.md rule 1)."
            )
        out[k] = str(v)
    out["matcher_id"] = str(row.get("matcher_id") or "")
    # SOFT, like `matcher_id` and unlike `scorer_hash`. Hard would be the stronger claim and a
    # false one: runs recorded before these were stamped exist in the corpus and are legitimately
    # unlabelled. `""` means "ran under no recorded code", which an audit must be able to tell
    # apart from "ran under the fixed loop" -- so it is never defaulted to a sha.
    out["code_version"] = str(row.get("code_version") or "")
    out["arm_id"] = str(row.get("arm_id") or "")
    out["reward_weights_sha"] = str(row.get("reward_weights_sha") or "")
    out["pins_sha"] = str(row.get("pins_sha") or "")
    out["split"] = str(row.get("split") or "")
    out["template_id"] = row.get("template_id")
    out["branch_of_run_id"] = row.get("branch_of_run_id")
    bt = row.get("branch_turn_idx")
    out["branch_turn_idx"] = None if bt is None else int(bt)
    for k in ("budget_cap", "max_turns"):
        v = row.get(k)
        out[k] = None if v is None else int(v)
    for k in ("evidence_coverage", "answer_correct", "answer_hedged"):
        v = row.get(k)
        out[k] = float("nan") if v is None else float(v)
    ttc = row.get("turns_to_complete")
    out["turns_to_complete"] = -1 if ttc is None else int(ttc)
    return out


#: The gold facts a STOP label may be derived from. `export_sft(stop_label=...)` names one.
#: NOT free-form: the value is stamped on the manifest and rides into every downstream reading
#: of the file, so a typo must be a refusal here rather than a third dataset nobody can name.
SFT_STOP_LABELS = ("done_before", "answer_node_covered_before")


def _state_flag(cands: Sequence[dict], field: str) -> bool | None:
    """A gold BOOLEAN that is a property of the STATE, read off the candidates at that state.

    `done_before` -- and `answer_node_covered_before` with it -- is a property of the STATE, so
    every candidate at one state must agree: they were all rendered from the same evidence
    prefix. Two that disagree were not, and no label may be derived from them -- that is a
    corrupt group, not a close call.

    None when no candidate carries the field: a legacy row set, or a task whose graph names no
    checkable answer node. Unknown is not "not done" and not "not covered"; the caller must
    never turn it into a STOP.
    """
    seen = {bool(c[field]) for c in cands if c.get(field) is not None}
    if not seen:
        return None
    if len(seen) > 1:
        c = cands[0]
        raise ValueError(
            f"state {c.get('suite_id')}/{c.get('task_id')}@{c.get('turn_idx')}: candidates "
            f"disagree about `{field}`, which is a property of the state they share. They "
            "were not rendered from the same evidence; no label can be derived from them."
        )
    return next(iter(seen))


def _state_done(cands: Sequence[dict]) -> bool | None:
    """`_state_flag` on `done_before`. Kept as a name because the pairs exporter reads it."""
    return _state_flag(cands, "done_before")


def _task_cap_key(e: Example) -> str:
    """Stable per-example digest used to choose WHICH examples a capped task keeps.

    Not by value: keeping the highest-valued examples selects exactly the questions the reward
    already likes, which is the confound `assert_diverse` exists to catch. Not by diversity
    either -- that optimises the metric directly, which is worse than failing it. A hash of the
    example's identity is arbitrary with respect to both, and it is stable across runs and
    across input order, so two exports of one corpus keep the same rows.
    """
    return hashlib.sha256(f"{e.suite_id}|{e.task_id}|{e.run_id}|{e.turn_idx}".encode()).hexdigest()


def export_sft(
    rows: Iterable[dict],
    *,
    margin_threshold: float,
    stop_action_json: str = STOP_ACTION_JSON,
    max_examples_per_task: int | None = None,
    split: str = "train",
    cohort: frozenset[str] = frozenset(),
    cohort_file_sha: str = "",
    cohort_mode: str = "any",
    turn0_any_cohort: bool = False,
    stop_label: str = "done_before",
) -> tuple[list[Example], ExportManifest]:
    """rows: one dict per candidate decision point, already joined against scores.

    THE COHORT IS A ROW FILTER AND IT RUNS BEFORE THE STATE IS GROUPED, not after the example
    is built. Two consequences, both wanted: a state whose every candidate is refused never
    reaches the argmax at all (so no target is derived from a prompt the broken loop rendered),
    and `sample_weight` is renormalised over the rows the file actually holds rather than over
    a superset -- a weight whose mean is 1 on rows that were then deleted is not mean 1.

    `stop_label` NAMES THE GOLD FACT THAT DECIDES STOP, and changes nothing else -- not the
    population, not the cohort, not the floor, not the argmax, not the dedupe. `"done_before"`
    is the rule of record. `"answer_node_covered_before"` is lane L6.1's variant: STOP where the
    ANSWER-BEARING node's evidence is in hand, whatever the pooled coverage.

    THE DIRECTION OF THAT SWAP IS FIXED AND IS NOT A CHOICE. An answer node is a REQUIRED node,
    so its gold spans are a subset of the required set and `done_before is True` implies
    `answer_node_covered_before is True` (`tests/test_answer_node_stop_label.py::
    test_done_before_true_forces_the_answer_node_covered`). The variant's STOP set is therefore
    a SUPERSET of v1's: relabelling can move a state ASK -> STOP and, except where the answer
    node is unknown, can never move one STOP -> ASK. Anyone reporting this file's label shift
    must report it that way round.
    """
    if stop_label not in SFT_STOP_LABELS:
        raise ValueError(
            f"stop_label={stop_label!r}: expected one of {SFT_STOP_LABELS}. The value is stamped "
            "on the manifest and is how a checkpoint names the rule it was fitted on, so an "
            "unrecognised one is refused rather than recorded."
        )
    out: list[Example] = []
    validate_cohort(cohort_mode, cohort, turn0_any_cohort)
    man = ExportManifest(
        margin_threshold=margin_threshold,
        split=split,
        cohort_file_sha=cohort_file_sha,
        cohort=cohort_mode,
        turn0_any_cohort=turn0_any_cohort,
        sft_stop_label=stop_label,
    )
    refused: dict[str, int] = {}
    ids: list[str] = []

    by_state: dict[tuple, list[dict]] = {}
    for r in rows:
        if r.get("binary_gold_hedged"):
            man.n_binary_gold_hedged_nan += 1  # per row consumed; see ExportManifest
        try:
            _assert_split(r, split)
        except Exception as exc:
            key = type(exc).__name__ + ":" + r["suite_id"]
            refused[key] = refused.get(key, 0) + 1
            continue
        # BEFORE the leak guard, because the cohort decides whether this row is in the
        # population at all; a row the cohort refused was never a candidate to leak.
        if not cohort_keeps(r, mode=cohort_mode, cohort=cohort, turn0_any=turn0_any_cohort):
            man.n_cohort_refused += 1
            continue
        if r.get("leaks_gold_answer"):
            # Not an imitation target: teaching the policy to emit a string it could only
            # produce by having memorised the benchmark is worse than teaching it nothing.
            man.n_answer_leak_dropped += 1
            continue
        key = state_key(r)
        if key is None:
            continue  # a branch row away from its fork: not a same-state comparison
        by_state.setdefault(key, []).append(r)

    for (suite, task, run, turn), cands in sorted(by_state.items()):
        cands = sorted(cands, key=lambda c: -float(c["value"]))
        if len({_caps(c) for c in cands}) > 1:
            # Counted only: the fork-turn action and its per-turn value are cohort-invariant,
            # so the SFT target is unaffected. A reader still deserves to know the state
            # held two budget cohorts.
            man.n_states_mixed_cap += 1

        try:
            done = _state_done(cands)
        except ValueError:
            man.n_state_done_disagree += 1
            continue
        if done is None:
            man.n_done_before_unknown += 1
        # READ ON BOTH LABELS, and counted on both. Under `done_before` it decides nothing and
        # is carried onto the row for the audit; under the variant it decides STOP. A disagreeing
        # state is refused on EITHER label rather than only on the one that reads it: the two
        # files must hold the same states, or the label shift between them is confounded with a
        # population difference and no count of it means anything.
        try:
            answer_covered = _state_flag(cands, "answer_node_covered_before")
        except ValueError:
            man.n_answer_node_state_disagree += 1
            continue
        if answer_covered is None:
            man.n_answer_node_unknown += 1
        coverage = next(
            (float(c["coverage_before"]) for c in cands if c.get("coverage_before") is not None),
            float("nan"),
        )
        # The gold fact `stop_label` names. `None` (unknown) falls through to the ASK branch
        # exactly as an unknown `done_before` always has: a STOP is only ever taught from a
        # gold fact that was actually read.
        stop_here = done if stop_label == "done_before" else answer_covered
        if stop_here:
            # THE STATE IS FINISHED, WHATEVER THE SAMPLES DID. Every remaining question is
            # redundancy or noise, and the gain one of them happens to show is measured over
            # evidence the task did not need. `frontier_size` is a SECOND opinion here -- a
            # matcher statement over uids against coverage's statement over required units --
            # and it is counted, never decisive: the two came apart on the graphs whose
            # required nodes sat behind an optional prerequisite.
            # GUARDED ON `done`, NOT ON THE BRANCH. It compares the matcher's frontier against
            # COVERAGE's claim that the state is complete, so it is only a disagreement when
            # coverage made that claim. Under v1 the branch and the claim coincide and this
            # changes nothing; under the variant, counting every answer-node STOP here would
            # report a frontier "disagreement" on states nothing ever called done.
            if done and any(int(c.get("frontier_size", 0)) > 0 for c in cands):
                man.n_done_frontier_disagree += 1
            man.n_stop_done_before_dedupe += 1
            ids.append(f"{suite}/{task}")
            out.append(
                Example(
                    suite_id=suite,
                    task_id=task,
                    run_id=run,
                    turn_idx=turn,
                    state_text=cands[0]["state_text"],
                    action_json=stop_action_json,
                    value=float(cands[0]["value"]),
                    is_stop=True,
                    candidate_rank=0,
                    latent_depth=int(cands[0].get("latent_depth", -1)),
                    is_latent=bool(cands[0].get("is_latent", False)),
                    newly_reachable=bool(cands[0].get("newly_reachable", False)),
                    frontier_size=int(cands[0].get("frontier_size", 0)),
                    has_gold_node=bool(cands[0].get("has_gold_node", False)),
                    nameable_from_task=cands[0].get("nameable_from_task"),
                    parent_uids=(),
                    phi_tilde=float("nan"),
                    leaks_gold_answer=False,
                    rho=0.0,
                    # NAMED BY THE RULE THAT FIRED, not by the branch it fired in. Under the
                    # variant this row's justification is "the answer node's evidence was in
                    # hand", which on a majority of such states is NOT "the task was done"; a
                    # row that said `stop_done` there would assert a gold fact no field on it
                    # supports, and every offline reading keyed on `label_rule` would pool two
                    # different claims.
                    label_rule=(
                        "stop_done" if stop_label == "done_before" else "stop_answer_node_covered"
                    ),
                    coverage_before=coverage,
                    # THE FACT, not the branch. This read `done_before=True` -- true by
                    # construction under v1 and a LIE under the variant, where a STOP row can
                    # sit on an incomplete state. Both gold fields now carry what gold said.
                    done_before=done,
                    answer_node_covered_before=answer_covered,
                    **_provenance(cands[0]),
                )
            )
            continue

        # NOT DONE. A recorded STOP candidate is not a target here and never enters the
        # argmax: its value is 0.0 by construction (`pi_run.cmd_train.STOP_VALUE`), which
        # outranks every negative-valued ASK, and exporting it would teach stopping on an
        # unfinished task for the reason the rule above exists to refuse.
        asks = [c for c in cands if not c.get("is_stop")]
        best = asks[0] if asks else None
        if best is None:
            man.n_no_target_dropped += 1
            continue
        ids.append(f"{suite}/{task}")

        # RANK BY VALUE, ACCEPT BY PHI. Two different questions, and they were being asked with
        # one number.
        #
        # `margin_threshold` is `tau + 1.5 * sigma_J * sqrt(2)`, where tau is "the 35th
        # percentile of pilot phi_LOO" and sigma_J is the judge's SD -- so it is expressed in
        # PHI units. `value` is `w_phi * phi_tilde - w_red * rho - c_ret`: phi rescaled by a
        # weight, minus a redundancy penalty, minus a fixed retrieval cost. Comparing them comes
        # apart in both directions -- with w_phi > 1 a question well below the noise floor
        # clears it, and a question genuinely above the floor is rejected for having also cost
        # a retrieval.
        #
        # The accept test asks "was this question measurably better than not asking?", which is
        # a statement about phi alone; cost belongs in the objective, which is rung 3's job.
        # Ranking stays on `value`: among candidates that clear the floor, prefer the one whose
        # cost-adjusted value is highest.
        phi = best.get("phi_tilde")
        clears = float(phi if phi is not None else best["value"]) > margin_threshold
        if not clears:
            # EVERY SAMPLE MISSED, AND THE TASK IS NOT DONE. That is a fact about the sampler,
            # not about the state: the right action here is some question none of these eight
            # candidates asked. There is no target to teach, so the state is dropped and
            # counted. It is NOT a STOP -- see the module docstring.
            man.n_no_target_dropped += 1
            ids.pop()
            continue
        out.append(
            Example(
                suite_id=suite,
                task_id=task,
                run_id=run,
                turn_idx=turn,
                state_text=best["state_text"],
                action_json=best["action_json"],
                value=float(best["value"]),
                is_stop=False,
                candidate_rank=0,
                latent_depth=int(best.get("latent_depth", -1)),
                is_latent=bool(best.get("is_latent", False)),
                newly_reachable=bool(best.get("newly_reachable", False)),
                frontier_size=int(best.get("frontier_size", 0)),
                has_gold_node=bool(best.get("has_gold_node", False)),
                nameable_from_task=best.get("nameable_from_task"),
                parent_uids=tuple(best.get("parent_uids") or ()),
                phi_tilde=float(
                    best.get("phi_tilde") if best.get("phi_tilde") is not None else float("nan")
                ),
                # Always False on an exported row -- a leaked row is refused above -- but
                # declared so the field exists for an audit of raw rows.
                leaks_gold_answer=bool(best.get("leaks_gold_answer", False)),
                rho=float(best.get("rho") or 0.0),
                label_rule="ask_clears_floor",
                coverage_before=coverage,
                done_before=done,
                answer_node_covered_before=answer_covered,
                **_provenance(best),
            )
        )

    # ---- EXACT DEDUPE, before any cap. One demonstration per (task, prompt, target): a
    # parent re-run on another day, or a fork-of-fork replaying the same prefix, reaches the
    # same bytes under a different state key, and that is one row, not two. Deterministic
    # representative (smallest `_task_cap_key`), so file order cannot change the artifact.
    seen: dict[tuple, Example] = {}
    for e in out:
        k = (e.suite_id, e.task_id, e.state_text, e.action_json)
        cur = seen.get(k)
        if cur is None or _task_cap_key(e) < _task_cap_key(cur):
            if cur is not None:
                man.n_exact_duplicate_dropped += 1
            seen[k] = e
        else:
            man.n_exact_duplicate_dropped += 1
    out = sorted(seen.values(), key=lambda e: (e.suite_id, e.task_id, e.run_id, e.turn_idx))

    # ---- SAMPLE WEIGHT, 1/sqrt(rows in task) per kind, mean 1 within the kind.
    per: dict[tuple[str, str, bool], int] = {}
    for e in out:
        per[e.suite_id, e.task_id, e.is_stop] = per.get((e.suite_id, e.task_id, e.is_stop), 0) + 1
    raw = [1.0 / math.sqrt(per[e.suite_id, e.task_id, e.is_stop]) for e in out]
    for kind in (False, True):
        idx = [i for i, e in enumerate(out) if e.is_stop is kind]
        if not idx:
            continue
        mean = sum(raw[i] for i in idx) / len(idx)
        for i in idx:
            out[i] = dataclasses.replace(out[i], sample_weight=raw[i] / mean)
    ids = [f"{e.suite_id}/{e.task_id}" for e in out]

    # ---- THE PER-TASK CAP, now OPTIONAL and off by default: it deleted a quarter of the
    # scarce class to flatten 107 against 2 while the gate it served sat at 0.79 over a 0.65
    # floor. Kept as a flag so the old artifact is reproducible and the ablation is one switch.
    man.max_examples_per_task = max_examples_per_task
    if max_examples_per_task is not None:
        if max_examples_per_task < 1:
            raise ValueError(
                f"max_examples_per_task={max_examples_per_task}: a cap below 1 would empty the "
                "dataset while reporting a manifest."
            )
        buckets: dict[tuple[str, str, bool], list[Example]] = {}
        for e in out:
            buckets.setdefault((e.suite_id, e.task_id, e.is_stop), []).append(e)
        kept: list[Example] = []
        for group in buckets.values():
            if len(group) > max_examples_per_task:
                man.n_over_task_cap_dropped += len(group) - max_examples_per_task
                group = sorted(group, key=_task_cap_key)[:max_examples_per_task]
            kept.extend(group)
        out = sorted(kept, key=lambda e: (e.suite_id, e.task_id, e.run_id, e.turn_idx))
        ids = [f"{e.suite_id}/{e.task_id}" for e in out]

    man.n_examples = len(out)
    man.suites = sorted({e.suite_id for e in out})
    man.train_id_set_hash = id_set_hash(ids)
    man.refused = refused
    return out, man


def _quicker(a: dict, b: dict) -> bool:
    """True when `a` reaches complete evidence in FEWER turns than `b`.

    -1 means the run never got there, which is NOT "zero turns away" -- ranking on it would
    make a run that never finished look instant. Either side unknown means the comparison is
    unavailable and the caller falls through to gain.
    """
    x, y = a.get("turns_to_complete"), b.get("turns_to_complete")
    if x is None or y is None:
        return False
    xi, yi = int(x), int(y)
    if xi < 0 or yi < 0:
        return False
    return xi < yi


RANK_RULES = ("outcome", "anticipation", "rater")

# WHICH STOP RULE AN EXPORT RAN UNDER, recorded on the manifest as `stop_rule`.
#
# "gold_coverage_v1" is the DEFAULT OF RECORD -- every shipped artifact, and the rule
# `export_sft` implements. "gold_coverage_v2" changes `export_pairs` ONLY: it adds the
# not-done contrast the corpus has never held (see the module docstring), and touches no
# other branch, so a v2 export is a v1 export plus `n_stop_pairs_synth_notdone` rows.
#
# VALIDATED rather than free-form, for the reason `--rank` is: a typo that fell through to
# the default would write a manifest naming a rule the export did not run, and the manifest
# is the only place a reader can find out which dataset they have.
STOP_RULES = ("gold_coverage_v1", "gold_coverage_v2")

# Which key ordered a pair. Until this existed the only record was the file-level `rank_rule`,
# so the per-key agreement table had to be re-derived by replaying the precedence over the
# carried fields -- a derivation that breaks silently the day the precedence changes.
DECIDED_BY = (
    "outcome",
    "anticipation",
    "rater",
    "quickest",
    "gain",
    "stop_done",
    "ask_clears_floor",
    # `gold_coverage_v2` only. The gold fact is `done_before is False` and NOTHING about the
    # sample -- deliberately not "ask_clears_floor", which is the claim the floor refused to
    # support at exactly these states. A consumer that wants the corpus as it was can exclude
    # this one key (`DPOConfig.exclude_decided_by`) and recover the v1 ordering.
    "not_done",
)
# "rule": the exporter's own ordering. "rater": a pair the rule could not order and a rater
# majority could. See tests/test_pairs_rater_rescue.py for the measurement that motivates it.
LABEL_SOURCES = ("rule", "rater")


def rater_prefs_sha(prefs: dict) -> str:
    """The 16-hex identity of a PREFERENCE SET. Public, because `pi train verify-prefs` recomputes
    it from the file on disk and compares.

    OVER THE SET, NOT THE FILE. A renamed, re-sorted or re-serialised preference file identifies
    the same ordering, so this is a statement about the experiment rather than about bytes.

    ONE IMPLEMENTATION, called by `export_pairs` and by the verifier. Two implementations of one
    digest drift, and a verifier that disagrees with the exporter it verifies is worse than none:
    it fails on correct files and teaches people to ignore it.
    """
    return hashlib.sha256(
        "\n".join(sorted(f"{w}<{'|'.join(sorted(k))}" for k, w in prefs.items())).encode()
    ).hexdigest()[:16]


def _rater_prefers(a: dict, b: dict, prefs: dict) -> bool:
    """True when a rater majority put `a`'s candidate above `b`'s.

    Keyed on the unordered PAIR of candidate run ids, which is A6's own unit ("the UNIT is one
    unordered candidate PAIR, never the whole ranking"). A pair with no verdict returns False
    from both directions and falls through to the mechanical precedence unchanged -- the same
    way an unknown outcome or an unmeasured `newly_reachable` does.
    """
    ra, rb = str(a.get("run_id") or ""), str(b.get("run_id") or "")
    if not ra or not rb:
        return False
    return prefs.get(frozenset((ra, rb))) == ra


def _anticipates(a: dict, b: dict) -> bool:
    """True when `a` took a need at the moment it became nameable and `b` did not.

    `newly_reachable` is a TRAJECTORY property, not a property of the retrieval: the turn
    resolved a required need at depth > 0 whose last prerequisite landed on the immediately
    previous turn. That is the vertical claim stated per turn. The weaker `is_latent` ("the
    turn resolved something at depth > 0") is a property of what came back and is close to
    tautological with phi, which is why it is not the key.

    None on either side is UNKNOWN and never orders a pair -- the same rule `_outcome_prefers`
    applies to a NaN, and for the same reason: reading an unlabelled row as a negative would
    invert pairs on exactly the rows nobody thought to check.

    A NO-OP AT TURN-0 FORKS. A depth-0 need has no prerequisite, so no turn-0 action is ever
    `newly_reachable`. 46% of the live states are turn-0 forks, so this key cannot fire on
    about half the dataset.
    """
    x, y = a.get("newly_reachable"), b.get("newly_reachable")
    if x is None or y is None:
        return False
    return bool(x) and not bool(y)


def _prefers(a: dict, b: dict, *, rank: str = "outcome", rater_prefs: dict | None = None) -> bool:
    """Should `a` beat `b` on something per-turn gain cannot see?

    THE PRECEDENCE, in order, each firing only when its signal is known on both sides and
    actually differs:

      1. OUTCOME       -- a candidate that answered the task beats one that did not, whatever
         it cost. `turn_values` contains no task term at all, so without this a candidate that
         retrieved more and answered wrong outranks one that answered right.
      2. ANTICIPATION  -- only under `rank="anticipation"`. See `_anticipates`, and see
         `ExportManifest.rank_rule` for why this is a flag and not the default: it makes the
         latent-pursuit check in `scripts/validate_pairs.py` true by construction, so the
         control export must keep existing.
      3. QUICKEST      -- among candidates that AGREE above, fewer remaining turns wins. phi
         scores gain NOW and cannot separate a question that opens the chain from one that
         pays the same now and dead-ends. Measured across runs of one musique task, the
         fastest reaches complete evidence at a median of 0.5 turns against the slowest at 4.0.
      4. GAIN          -- otherwise, unchanged.

    Speed never overrides outcome: reaching gold sooner is worthless if the task was not
    answered. Anticipation never overrides outcome either, and sits ABOVE speed because speed
    is a proxy for the thing anticipation measures directly.
    """
    return _decide(a, b, rank=rank, rater_prefs=rater_prefs)[0]


def _decide(
    a: dict, b: dict, *, rank: str = "outcome", rater_prefs: dict | None = None
) -> tuple[bool, str]:
    """`_prefers`, plus the NAME of the key that decided -- one of `DECIDED_BY`.

    Split out so every pair can record what ordered it. `_prefers` is this function's first
    element and nothing about its contract changes: the old tail `return _quicker(a, b)`
    returned False when speed could not separate the two, and so does `(False, "gain")`.
    """
    if _outcome_prefers(a, b):
        return True, "outcome"
    if _outcome_prefers(b, a):
        return False, "outcome"
    if rank == "anticipation":
        if _anticipates(a, b):
            return True, "anticipation"
        if _anticipates(b, a):
            return False, "anticipation"
    if rank == "rater" and rater_prefs:
        # Below outcome for the same reason anticipation is: a candidate whose episode
        # answered the task beats one whose did not, whatever a reader preferred.
        if _rater_prefers(a, b, rater_prefs):
            return True, "rater"
        if _rater_prefers(b, a, rater_prefs):
            return False, "rater"
    if _quicker(a, b):
        return True, "quickest"
    if _quicker(b, a):
        return False, "quickest"
    # Nothing above could separate them, so the value sort that produced this ordering stands.
    return False, "gain"


def _outcome_prefers(a: dict, b: dict) -> bool:
    """True when episode outcome says `a` should beat `b`, overriding per-turn gain.

    NaN IS NOT A FAILURE. `answer_correct` is NaN on every suite with no gold answer, and
    reading that as 0.0 would invert every pair on those suites -- a silent, total corruption
    of the training signal on exactly the suites nobody would think to check. Unknown compares
    equal to unknown and never orders a pair.
    """
    x, y = a.get("answer_correct"), b.get("answer_correct")
    if x is None or y is None:
        return False
    fx, fy = float(x), float(y)
    if fx != fx or fy != fy:  # NaN on either side: outcome is unknown, gain decides
        return False
    return fx > fy


def _is_stop_row(row: dict) -> bool:
    """Is this candidate a STOP? Read off the ACTION BYTES, not the row's `is_stop` flag.

    The bytes are what a training example contains, so deriving the kind from anything else
    lets a row claim one thing and teach another. `action_kind_of` is the policy parser's own
    normalisation (`pinq.actions`), so the exporter and the loop can never disagree.
    """
    return action_kind_of(row.get("action_json")) == "stop"


def _ask_stop_direction(ask: dict, stop: dict, margin_threshold: float) -> str:
    """Which side of an ASK-vs-STOP pair wins: "stop", "ask", or "" for neither.

    STOP NEVER WINS ON VALUE. A STOP's value is 0.0 by construction (it retrieves nothing and
    pays nothing), so on any state whose questions all came out negative -- each cost a
    retrieval and gained little -- a value comparison hands STOP every pair, and the policy
    learns to stop whenever asking is expensive. The STOP direction therefore requires a GOLD
    reason: the required evidence was already in hand (`done_before`), or the outcome key
    decides. The ASK direction requires the same noise floor `export_sft` uses: below it the
    question was not measurably better than not asking, and the pair would assert a preference
    the measurement does not support.

    Outcome is checked FIRST, as it is in `_prefers`: a candidate whose episode answered the
    task beats one whose episode did not, whatever the coverage of the state says.
    """
    return _ask_stop_decide(ask, stop, margin_threshold)[0]


def _ask_stop_decide(ask: dict, stop: dict, margin_threshold: float) -> tuple[str, str]:
    """`_ask_stop_direction`, plus the name of the gold fact that decided it."""
    if _outcome_prefers(stop, ask):
        return "stop", "outcome"
    if _outcome_prefers(ask, stop):
        return "ask", "outcome"
    if ask.get("done_before") is True:
        return "stop", "stop_done"
    phi = ask.get("phi_tilde")
    if phi is not None and float(phi) > margin_threshold:
        return "ask", "ask_clears_floor"
    return "", ""


def export_pairs(
    rows: Iterable[dict],
    *,
    margin_threshold: float,
    len_delta_max: int = 40,
    rank: str = "outcome",
    rater_prefs: dict | None = None,
    split: str = "train",
    cohort: frozenset[str] = frozenset(),
    cohort_file_sha: str = "",
    cohort_mode: str = "any",
    turn0_any_cohort: bool = False,
    stop_rule: str = "gold_coverage_v1",
) -> tuple[list[PreferencePair], ExportManifest]:
    """Same-state preference pairs, with a LENGTH GUARD.

    Without the guard the preference model learns "longer question wins", which is the same
    confound that makes a naive LLM judge unusable -- imported straight into the policy. It
    is not hypothetical here: measured on 164 musique runs, `answer_token_f1` is
    rank-correlated with answer length at -0.934 within `inquirer_prompted` alone.

    THE ONLY PAIRING IMPLEMENTATION. `pinq.sampling` produces candidates and deliberately
    does not pair them; two pairing rules would be free to disagree about what a preference
    is.

    ALL C(n,2) PAIRS PER STATE, which is safe ONLY while every state is sampled with the
    SAME candidate count. Under a fixed n each state contributes an equal C(n,2), so no
    state is over-weighted. If the sampler is ever given a per-state n, a state with 8
    candidates contributes 28 pairs against another's 1 and dominates the loss in proportion
    to how many candidates it happened to get -- at which point this needs a per-state cap,
    not a bigger margin.
    """
    validate_cohort(cohort_mode, cohort, turn0_any_cohort)
    if rank not in RANK_RULES:
        raise ValueError(f"rank={rank!r}: expected one of {RANK_RULES}")
    if stop_rule not in STOP_RULES:
        raise ValueError(f"stop_rule={stop_rule!r}: expected one of {STOP_RULES}")
    if rank == "rater" and not rater_prefs:
        raise ValueError(
            "rank='rater' with no rater_prefs: the key would order nothing and the export "
            "would be byte-identical to the control while claiming a different rank_rule."
        )
    man = ExportManifest(
        margin_threshold=margin_threshold,
        len_delta_max=len_delta_max,
        rank_rule=rank,
        split=split,
        cohort_file_sha=cohort_file_sha,
        cohort=cohort_mode,
        turn0_any_cohort=turn0_any_cohort,
        stop_rule=stop_rule,
    )
    if rank == "rater" and rater_prefs:
        man.rater_prefs_sha = rater_prefs_sha(rater_prefs)
    refused: dict[str, int] = {}
    by_state: dict[tuple, list[dict]] = {}
    for r in rows:
        if r.get("binary_gold_hedged"):
            man.n_binary_gold_hedged_nan += 1  # per row consumed; see ExportManifest
        try:
            _assert_split(r, split)
        except Exception as exc:
            key = type(exc).__name__ + ":" + r["suite_id"]
            refused[key] = refused.get(key, 0) + 1
            continue
        # SELECTING a cohort, where `_cross_cohort` below only refuses to MIX them. A pair
        # carries its winner's `code_version` (see `_pair`), so filtering the candidates is
        # what makes that field true of every pair in the file -- and it also stops a refused
        # candidate from being the loser half of a pair it does not appear in.
        if not cohort_keeps(r, mode=cohort_mode, cohort=cohort, turn0_any=turn0_any_cohort):
            man.n_cohort_refused += 1
            continue
        key = state_key(r)
        if key is None:
            continue  # a branch row away from its fork: not a same-state comparison
        by_state.setdefault(key, []).append(r)

    pairs: list[PreferencePair] = []
    ids: list[str] = []
    for (suite, task, run, turn), cands in sorted(by_state.items()):
        try:
            state_done = _state_done(cands)
        except ValueError:
            # See `ExportManifest.n_state_done_disagree`: the state is refused, the export
            # continues. Refused BEFORE any pair is built, so a contradictory state cannot
            # contribute an ask_stop direction derived from one side's reading of the gold.
            man.n_state_done_disagree += 1
            continue
        here: list[PreferencePair] = []
        # Pairs the rule refused and a rater majority re-admitted; empty unless rank=rater.
        rescued: list[PreferencePair] = []
        cands = sorted(cands, key=lambda c: -float(c["value"]))
        for i in range(len(cands)):
            for j in range(i + 1, len(cands)):
                hi, lo = cands[i], cands[j]
                # ---- ASK vs STOP, decided on gold rather than on gain. Handled before the
                # ask_ask machinery because almost none of it applies: two STOPs are not an
                # identical QUESTION, a STOP has no question to measure a length delta
                # against, and a value margin is exactly the thing that must not decide the
                # direction. The instrument, cohort and leak guards DO apply and are shared.
                hs, ls = _is_stop_row(hi), _is_stop_row(lo)
                if hs and ls:
                    man.n_stop_stop_dropped += 1
                    continue
                if hs or ls:
                    ask, stop = (lo, hi) if hs else (hi, lo)
                    if ask.get("leaks_gold_answer"):
                        man.n_answer_leak_dropped += 1
                        continue
                    if str(hi.get("pins_sha") or "") != str(lo.get("pins_sha") or ""):
                        man.n_cross_instrument_dropped += 1
                        continue
                    if _cross_cohort(hi, lo, cohort):
                        man.n_cross_code_version_dropped += 1
                        continue
                    if _caps(hi) != _caps(lo):
                        man.n_cross_cap_dropped += 1
                        continue
                    # A recorded STOP replays the same prefix as every other candidate.
                    if str(hi.get("state_text") or "") != str(lo.get("state_text") or ""):
                        man.n_cross_state_dropped += 1
                        continue
                    # NAMED `gold_key`, NOT `stop_rule`. It was `stop_rule` and that is now the
                    # name of this function's rule PARAMETER, so the old local would rebind it
                    # mid-loop and silently switch every later state back to v1 -- a wrong
                    # dataset under a manifest naming the rule the caller asked for.
                    direction, gold_key = _ask_stop_decide(ask, stop, margin_threshold)
                    if not direction:
                        man.n_ask_stop_undecided += 1
                        continue
                    win, lose = (stop, ask) if direction == "stop" else (ask, stop)
                    # The MAGNITUDE is the question's own gain against doing nothing: what the
                    # ASK was worth, not a difference of two values one of which is a constant.
                    margin = abs(float(ask["value"]) - float(stop["value"]))
                    here.append(
                        _pair(
                            suite,
                            task,
                            run,
                            turn,
                            win,
                            lose,
                            margin=margin,
                            # RECORDED, so the length delta is what the two actions actually
                            # were. No guard: see `pair_kind` on PreferencePair.
                            len_delta=question_len_delta(win["action_json"], lose["action_json"]),
                            pair_kind="ask_stop",
                            stop_source="recorded",
                            decided_by=gold_key,
                        )
                    )
                    man.n_ask_stop_pairs += 1
                    man.n_stop_pairs_recorded_before_cap += 1
                    if direction == "stop":
                        man.n_stop_chosen += 1
                    else:
                        man.n_ask_chosen_over_stop += 1
                    ids.append(f"{suite}/{task}")
                    continue
                # OUTCOME BREAKS THE TIE THE GAIN CANNOT SEE. `value` is per-turn evidence
                # gain and contains no task term at all (`turn_values` drops the one
                # `reward_of` computes), so without this a candidate that retrieved more and
                # answered wrong outranks one that answered right. Applied ONLY when the two
                # disagree about the outcome: when they agree -- both right, both wrong, or
                # both unknown -- gain remains the ordering, because the state is informative
                # about which question was better even when the episode was not.
                if rank == "rater" and rater_prefs:
                    ra, rb = str(hi.get("run_id") or ""), str(lo.get("run_id") or "")
                    if frozenset((ra, rb)) in rater_prefs:
                        man.n_rater_ordered += 1
                flipped, decided_by = _decide(lo, hi, rank=rank, rater_prefs=rater_prefs)
                if flipped:
                    hi, lo = lo, hi
                margin = float(hi["value"]) - float(lo["value"])
                if flipped:
                    # The preference is real but its MAGNITUDE is not a gain difference; the
                    # gain runs the other way. Report the magnitude so the accept threshold
                    # sees a preference rather than a negative number it would filter away.
                    margin = abs(margin)
                # IDENTICAL TEXT IS NOT A PREFERENCE. Measured on the 1,521-pair export: 95 of
                # 159 states (60%) contained two candidates whose question was byte-identical.
                # They cleared `margin_threshold` because their VALUES differ despite the text
                # being the same -- phi_tilde is computed over the evidence a turn actually
                # retrieved, and the retriever does not guarantee the same set for two runs of
                # one query. So the export asserted "prefer X over X" with a margin on it.
                #
                # NOT the length guard, which blocks a pair whose sides differ too MUCH. This
                # blocks a pair whose sides do not differ at all.
                # SAME INSTRUMENT, OR NO PAIR. A pair asserts "at this state, A beats B",
                # which is a claim about the QUESTIONS only if everything else about the two
                # rollouts matched. MEASURED after `answerer_frozen` was fixed: 266 pairs
                # (18.9%) straddled the two prompt eras -- states branched once before and
                # once after -- and the newer side was chosen just 35.3% of the time
                # (p = 1e-6) while the outcome-driven flips among them were balanced. So the
                # skew was some OTHER systematic difference between the cohorts, ranked as if
                # it were a difference in the question.
                #
                # An unstamped row is refused against a stamped one rather than assumed equal
                # (the direction `_provenance` already takes), but two unstamped rows still
                # pair: legacy exports are internally consistent and refusing them would
                # delete data for a difference that is not there.
                # A QUESTION CARRYING THE GOLD ANSWER WOULD WIN. Naming the answer retrieves
                # the gold passage perfectly, so a leaked candidate earns maximum evidence gain
                # and takes the pair -- contamination concentrating in exactly the candidates
                # selected as best. Measured: gpt-5.6-terra writes the answer into its turn-0
                # query on 12% of musique tasks, gpt-oss-120b on 0%.
                if hi.get("leaks_gold_answer") or lo.get("leaks_gold_answer"):
                    man.n_answer_leak_dropped += 1
                    continue
                if str(hi.get("pins_sha") or "") != str(lo.get("pins_sha") or ""):
                    man.n_cross_instrument_dropped += 1
                    continue
                # SAME LOOP, OR NO PAIR -- the loop COHORT, not the commit: two commits that
                # are both descendants of the fix, or both ancestors of it, rendered the same
                # prompt. The same asymmetry as the line above: an unstamped row is refused
                # against a stamped one, two unstamped rows still pair. See `_cross_cohort`.
                if _cross_cohort(hi, lo, cohort):
                    man.n_cross_code_version_dropped += 1
                    continue
                # SAME BUDGET COHORT, OR NO PAIR. The fork-turn actions are comparable across
                # caps -- same state, same seeds, same request bytes -- but the label that
                # orders a pair is episode-level (`answer_correct`, `turns_to_complete`,
                # `evidence_coverage`) and depends on how far the continuation was allowed to
                # run. A cap-24 candidate beats a cap-8 twin on outcome by construction, and
                # that would be recorded as a preference over the question. `pins_sha` does
                # not carry the cap, so this guard is the only thing standing in the way.
                # None-vs-int refuses (an unknown cohort against a known one); None-vs-None
                # pairs (legacy rows), the same asymmetry the pins guard applies.
                if _caps(hi) != _caps(lo):
                    man.n_cross_cap_dropped += 1
                    continue
                # SAME STATE KEY IS NOT THE SAME STATE. `_pair` writes the winner's
                # `state_text`, so a drifted loser would have its action scored against a
                # prompt it never saw, and the within-state cancellation rung 2 depends on
                # would not hold. Refused, not repaired: it is the one invariant the method
                # cannot lose.
                if str(hi.get("state_text") or "") != str(lo.get("state_text") or ""):
                    man.n_cross_state_dropped += 1
                    continue
                if _same_action(hi["action_json"], lo["action_json"]):
                    man.n_identical_dropped += 1
                    continue
                dl = question_len_delta(hi["action_json"], lo["action_json"])
                if margin <= margin_threshold:
                    man.n_below_margin_dropped += 1
                    # THE PAIR THE RULE CANNOT ORDER AND A RATER CAN. Our own score is at
                    # chance on this bucket (50.2% agreement with the A6 majority, n=287
                    # after excluding 198 exact ties) while the raters' unanimity is
                    # unchanged from the exported pairs -- so the verdict is signal the
                    # margin is blind to, not noise the margin was filtering out.
                    #
                    # THE LENGTH GUARD IS STILL ENFORCED HERE, and must be: on
                    # length-refused pairs the rater-preferred side is the LONGER question
                    # 71.7% [67.2, 75.7] of the time, 79.6% when unanimous. That bucket's
                    # agreement IS a length preference, and rescuing it would import the
                    # rung-0 GEPA failure into the label. The margin guard runs before the
                    # length guard, so the check is re-stated rather than inherited.
                    if rank == "rater" and decided_by == "rater" and dl <= len_delta_max:
                        rescued.append(
                            _pair(
                                suite,
                                task,
                                run,
                                turn,
                                hi,
                                lo,
                                margin=margin,
                                len_delta=dl,
                                label_source="rater",
                                decided_by="rater",
                            )
                        )
                        man.n_rater_rescued_below_margin += 1
                        ids.append(f"{suite}/{task}")
                    continue
                if dl > len_delta_max:
                    man.n_len_dropped += 1
                    continue
                here.append(
                    _pair(
                        suite,
                        task,
                        run,
                        turn,
                        hi,
                        lo,
                        margin=margin,
                        len_delta=dl,
                        decided_by=decided_by,
                    )
                )
                ids.append(f"{suite}/{task}")

        # TRIMMED DETERMINISTICALLY BY pair_id, which hashes the state and the two candidate
        # run ids. Top-k by margin would keep the widest gaps and drop every close call --
        # exactly the contrasts a preference model has most to learn from.
        if len(here) > MAX_PAIRS_PER_STATE:
            man.n_over_cap_dropped += len(here) - MAX_PAIRS_PER_STATE
            ordered = sorted(here, key=lambda p: p.pair_id)
            here, cut = ordered[:MAX_PAIRS_PER_STATE], ordered[MAX_PAIRS_PER_STATE:]
            # The cap bounds C(n,2) growth from an accidental n; it is not a quality
            # judgment, and the cap-refused bucket is indistinguishable from the exported
            # one on rater unanimity (42.8% against 41.3%) and on how often our own score
            # agrees with the rater (59.2% against 59.1%). A pair the cap cut that a rater
            # majority ordered is re-admitted, OUTSIDE the cap for the same reason the
            # synthetic STOP sits outside it: it carries a different fact, and counting it
            # inside would either zero this counter by construction or displace a rule pair.
            if rank == "rater":
                for cp in cut:
                    if cp.decided_by == "rater":
                        rescued.append(dataclasses.replace(cp, label_source="rater"))
                        man.n_rater_rescued_over_cap += 1
        pairs.extend(here)
        pairs.extend(rescued)

        # ---- ONE SYNTHETIC STOP PAIR, only where gold says the state was already done and no
        # candidate actually stopped there. That is the commonest shape in the data: the task
        # was finished and all eight samples asked anyway, so the state contributes nothing at
        # all to the decision the paper is about. The synthetic side is the STOP CONSTANT, not
        # a fabricated rollout -- there is exactly one way to stop -- and it is marked
        # `stop_source="synthesised"` so no consumer mistakes it for a measured episode.
        #
        # ONE per state, not one per candidate: n copies of the same STOP against n questions
        # would weight a single gold fact by however many samples the state happened to get.
        # OUTSIDE the per-state cap for the same reason -- the cap exists to stop one state
        # dominating with C(n,2) ASK pairs, and this is one pair carrying a different fact.
        # NEVER on an unfinished state: inventing a STOP there fabricates the very label the
        # dataset is short of.
        asks = [c for c in cands if not _is_stop_row(c)]
        if state_done and asks and not any(_is_stop_row(c) for c in cands):
            best_ask = asks[0]
            synth = dict(best_ask)
            synth["run_id"] = ""
            synth["action_json"] = STOP_ACTION_JSON
            synth["value"] = 0.0
            synth["phi_tilde"] = float("nan")
            synth["is_stop"] = True
            pairs.append(
                _pair(
                    suite,
                    task,
                    run,
                    turn,
                    synth,
                    best_ask,
                    margin=abs(float(best_ask["value"])),
                    len_delta=question_len_delta(STOP_ACTION_JSON, best_ask["action_json"]),
                    # ITS OWN KIND, not "ask_stop". A gold-derived statement and a STOP a
                    # candidate actually took are different evidence, and the first
                    # outnumbered the second 18,227 to 627 on the live corpus -- 73% of the
                    # dataset, one 18-byte chosen side. The rung-2 loader defaults to the
                    # SAMPLED kinds so this is opt-in.
                    pair_kind="ask_stop_synth",
                    stop_source="synthesised",
                    # The only fact that ever justifies a synthetic STOP: gold says the
                    # required evidence was already in hand before this decision.
                    decided_by="stop_done",
                )
            )
            man.n_ask_stop_pairs += 1
            man.n_stop_pairs_synth += 1
            man.n_stop_chosen += 1
            ids.append(f"{suite}/{task}")

        # ---- THE NOT-DONE CONTRAST (`stop_rule="gold_coverage_v2"`, opt-in). The mirror of
        # the block above and the lesson the corpus has never held: gold says the required
        # evidence was NOT yet in hand, so whatever these candidates asked, asking beat
        # stopping. Measured consequence of its absence -- every rung-1 checkpoint stops at
        # 9-17% of not-done states (docs/TRAINING.md 5.3.1-5.3.2).
        #
        # IT DOES NOT NEED A GOOD QUESTION, WHICH IS WHY IT EXISTS. `export_sft` drops these
        # states (24,538 of them, `n_no_target_dropped`) because no candidate cleared the
        # noise floor and there is no question worth IMITATING. A preference asserts strictly
        # less: not "ask this", but "asking beat stopping here", and `done_before is False`
        # establishes that on its own. The chosen side is the best AVAILABLE ASK -- the least
        # bad statement of "some question" -- and its `phi_tilde` rides on the row, so a
        # consumer that disagrees can filter on it.
        #
        # `is False`, NOT `not state_done`: None is UNKNOWN (a legacy row with no gold-side
        # signal) and reading it as "not done" would fabricate the direction on exactly the
        # rows nobody can check -- the asymmetry `_state_done` and `export_sft` enforce.
        #
        # NEVER WHERE A CANDIDATE STOPPED. There the STOP on the table is MEASURED: the state's
        # contrast is the recorded pair above, or the rule's recorded refusal to order it
        # (`n_ask_stop_undecided`). Appending a side marked "synthesised" whose bytes a rollout
        # did emit would be a false provenance claim, and the state would carry two contrasts
        # about one decision. The guard is therefore on the CANDIDATE, which is strictly
        # stronger than "no recorded ask_stop pair survived" and needs no knowledge of which
        # guard dropped it.
        if stop_rule == "gold_coverage_v2" and state_done is False:
            if not asks:
                # Every candidate stopped. There is no ASK to put on the chosen side, and
                # inventing a question is the one thing a synthetic side may never do: there
                # is exactly one way to stop and no canonical way to ask.
                man.n_notdone_states_no_ask += 1
            elif not any(_is_stop_row(c) for c in cands):
                best_ask = _best_available_ask(asks)
                synth_stop = dict(best_ask)
                synth_stop["run_id"] = ""
                synth_stop["action_json"] = STOP_ACTION_JSON
                synth_stop["value"] = 0.0
                synth_stop["phi_tilde"] = float("nan")
                synth_stop["is_stop"] = True
                pairs.append(
                    _pair(
                        suite,
                        task,
                        run,
                        turn,
                        best_ask,
                        synth_stop,
                        # The question's own gain against doing nothing, exactly as the
                        # done-state synth measures it. It can be 0.0 or the magnitude of a
                        # NEGATIVE value here -- a question that cost a retrieval and gained
                        # little -- and that is honest: the pair's warrant is the gold fact,
                        # never the size of the gap. Recorded, never gated on.
                        margin=abs(float(best_ask["value"])),
                        len_delta=question_len_delta(best_ask["action_json"], STOP_ACTION_JSON),
                        # THE SAME KIND as the done-state synth, because the STOP side is the
                        # same derived constant and `pair_kind_of` re-derives from the payloads.
                        # `stop_source` is the only witness to the DIRECTION; see PreferencePair.
                        pair_kind="ask_stop_synth",
                        stop_source="synthesised_notdone",
                        # The gold fact, and nothing about the sample. Deliberately not
                        # "ask_clears_floor": that is the claim the floor refused to support at
                        # exactly these states.
                        decided_by="not_done",
                    )
                )
                man.n_ask_stop_pairs += 1
                man.n_stop_pairs_synth_notdone += 1
                man.n_ask_chosen_over_stop += 1
                ids.append(f"{suite}/{task}")
    man.n_pairs = len(pairs)
    man.suites = sorted({p.suite_id for p in pairs})
    man.train_id_set_hash = id_set_hash(ids)
    man.refused = refused
    return pairs, man


def _pair(
    suite: str,
    task: str,
    run: str,
    turn: int,
    win: dict,
    lose: dict,
    *,
    margin: float,
    len_delta: int,
    pair_kind: str = "ask_ask",
    stop_source: str = "",
    label_source: str = "rule",
    decided_by: str = "",
) -> PreferencePair:
    """The ONE PreferencePair constructor, shared by the ask_ask loop, the ask_stop branch and
    the synthetic STOP. Three constructors would be three chances for a field to be filled on
    one path and dropped on another -- the exact way `rejected_*` was missing for 1,525 pairs.
    """
    chosen_run_id = str(win.get("run_id") or "")
    rejected_run_id = str(lose.get("run_id") or "")
    pair_id = hashlib.sha256(
        f"{suite}|{task}|{run}|{turn}|{chosen_run_id or 'synth-stop'}|{rejected_run_id}"
        f"|{pair_kind}".encode()
    ).hexdigest()[:16]
    return PreferencePair(
        suite,
        task,
        run,
        turn,
        win["state_text"],
        win["action_json"],
        lose["action_json"],
        margin,
        len_delta,
        int(win.get("latent_depth", -1)),
        bool(win.get("is_latent", False)),
        bool(win.get("newly_reachable", False)),
        chosen_run_id=chosen_run_id,
        rejected_run_id=rejected_run_id,
        pair_id=pair_id,
        frontier_size=int(win.get("frontier_size", 0)),
        rejected_is_latent=(None if lose.get("is_latent") is None else bool(lose["is_latent"])),
        rejected_newly_reachable=(
            None if lose.get("newly_reachable") is None else bool(lose["newly_reachable"])
        ),
        rejected_latent_depth=int(lose.get("latent_depth", -1)),
        rejected_turns_to_complete=int(
            lose.get("turns_to_complete") if lose.get("turns_to_complete") is not None else -1
        ),
        rejected_answer_correct=_f(lose.get("answer_correct")),
        rejected_evidence_coverage=_f(lose.get("evidence_coverage")),
        rejected_phi_tilde=_f(lose.get("phi_tilde")),
        pair_kind=pair_kind,
        stop_source=stop_source,
        label_source=label_source,
        decided_by=decided_by,
        coverage_before=_f(win.get("coverage_before")),
        done_before=(None if win.get("done_before") is None else bool(win["done_before"])),
        **_provenance(win),
    )


def train_ids_name(train_id_set_hash: str) -> str:
    """The sidecar's filename, derived from the hash it holds the preimage of.

    NAMED BY THE HASH, not by the dataset. Two exports of one id set differ byte-for-byte
    (timestamps, counters) and are the same training set, so `sft.ids.txt` beside `sft.jsonl`
    would be overwritten by the next export and a checkpoint registered against the old one
    would silently be checked against the new ids. The hash in the name means a file either IS
    the set a row names or is not there at all -- and `pi_eval.report.train_split_violations`
    finds it by that name from the hash a run was stamped with, with no index in between.
    """
    return f"train_ids.{train_id_set_hash[:16]}.txt"


def write_jsonl(path: Path, items: Sequence[object], manifest: ExportManifest) -> Path:
    """The dataset, its manifest, and the ID SET the manifest's hash names.

    THE SIDECAR IS WHAT MAKES LAYER 3 TRUE. `train_id_set_hash` was computed, written into the
    manifest, and the list it was computed from was dropped on the floor -- so the claim in
    `pinq_train.split`'s docstring, that a later run can "prove which ids a checkpoint saw",
    had no preimage anywhere on disk to prove it against. A sha256 is a proof of membership
    only to a reader HOLDING the set.

    RECOMPUTED FROM THE ITEMS BEING WRITTEN, not copied from the manifest. The file is the
    thing a contamination check will trust, so it is derived from the rows that are actually
    going into the artifact; a manifest handed in from elsewhere, or one whose hash was taken
    before a filter ran, disagrees and is REFUSED rather than written. A sidecar whose name
    says one id set and whose contents are another is worse than no sidecar at all.
    """
    ids = sorted({f"{it.suite_id}/{it.task_id}" for it in items})  # type: ignore[attr-defined]
    recomputed = id_set_hash(ids)
    if recomputed != manifest.train_id_set_hash:
        raise ValueError(
            f"{path}: the manifest's train_id_set_hash is {manifest.train_id_set_hash!r} but "
            f"the {len(items)} items being written hash to {recomputed!r}. The sidecar is the "
            "preimage a contamination check reads; writing one that disagrees with the hash "
            "stamped on every run would make the check answer about the wrong id set."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for it in items:
            f.write(json.dumps(asdict(it), sort_keys=True) + "\n")
    manifest.train_ids_file = train_ids_name(manifest.train_id_set_hash)
    (path.parent / manifest.train_ids_file).write_text("".join(f"{i}\n" for i in ids))
    (path.parent / f"{path.stem}.manifest.json").write_text(
        json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n"
    )
    return path
