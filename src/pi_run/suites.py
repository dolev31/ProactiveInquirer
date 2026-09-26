"""The benchmark registry: what each suite is FOR, and which half of the paper it serves.

WHY THIS EXISTS. Until now a suite's role lived in four places that could not see each
other: `worker.load_suite` knew how to build it, `pinq.splitting.EVAL_ONLY_SUITES` knew
whether it could be trained on, `conf/grids/*.yaml` knew which arms it ran, and
`pi_eval.prereg` knew which endpoint it carried. Nothing said, in one place, "musique is
mined for training on its train split and reports the vertical endpoint on its test split".
So the question a reviewer asks first --- *which data trained this, and which data measured
it* --- could only be answered by reading four trees and hoping they agreed.

They did not always agree. `EVAL_ONLY_SUITES` is written out three times (here via
`pinq.splitting`, in `pinq_train.split`, in `pi_run.serve.score`) because contract 4 forbids
`pi_run` importing `pinq_train`; two of those three could import `pinq.splitting` and did
not. Adding one eval-only suite therefore meant editing three literals and three test
literals, and the only thing standing between that and a training-set leak was a reviewer
noticing.

WHAT THIS IS NOT. Not a plugin system and not a loader. `worker.load_suite` still constructs
adapters, and it stays a flat if-chain for the reason its own docstring gives. This module
holds only the declarations a human made and a machine can check, and `audit()` is what
checks them. A registry that can drift from the code is worse than no registry, because it
reads like documentation and is not.

THE TWO ROLE FLAGS ARE INDEPENDENT, AND THAT IS THE POINT. `mines_training_data` and
`carries_endpoints` are not opposites: musique is both (train split is exported, test split
is measured), tau2 is only the second. Collapsing them into one "role" enum was the first
draft and it could not express musique, which is the suite the paper's central claim rests
on.

THE FIRST THING THIS FOUND. The draft above said wiki2 was "only the first" -- a training
source carrying no endpoint, on the grounds that its depth<=1 structure makes the vertical
axis degenerate. That is false, and the audit is what said so: wiki2 sits in the POOL of six
preregistered secondary endpoints, including all four kill-switch contrasts. Dropping it
from the paper is therefore an amendment to six declared tests, not an editorial choice
about which suite is interesting, and it has to be made before the seal rather than after
the numbers are visible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

from pinq.splitting import EVAL_ONLY_SUITES

# How a suite's tasks arrive. This decides what "validate" even means for it: a corpus-backed
# suite can be checked against a content hash, a self-sourced one can only be checked against
# an env var pointing at somebody else's checkout.
Source = Literal["corpus", "self", "logs", "gym"]

# Which axis of the claim the suite is evidence for.
#
# `mechanism` is the one that is easy to miss and expensive to get wrong: it marks a suite
# that sits inside a POOLED preregistered contrast (the kill-switch family --
# inquirer_noevidence, parallel_replay, verbosity, compute_matched -- plus rnr_resolve and
# frontier_auc). A suite can carry no claim of its own and still be load-bearing there,
# because dropping it changes the denominator of six declared tests. wiki2 is exactly that
# suite, and treating its depth<=1 degeneracy as a reason to drop it from the paper would
# silently rewrite those six pools.
#
# `calibration` is not an axis of the claim at all -- it is the suite whose every value is
# known in closed form, so a deviation there is a bug in the harness rather than a finding.
Axis = Literal["vertical", "horizontal", "cost", "mechanism", "calibration"]

Status = Literal["live", "planned", "cut"]

# What a number from this suite is ALLOWED to be called in the paper.
#
# The distinction is not cosmetic. `confirmatory` means the (suite, metric, contrast) is
# declared in pi_eval.prereg and may carry a p-value; `exploratory` means it is measurable
# today but was never declared, so it gets a point estimate and a CI and no p-value. Two
# suites that this project measures -- tau2_retail and convlog -- were BUILT AFTER the
# preregistration was written and are therefore exploratory until it is amended. Recording
# that here is what stops a retail number appearing in a confirmatory table because it
# happened to be available.
EndpointStatus = Literal["confirmatory", "exploratory", "calibration", "none"]

# How a rollout on this suite is actually driven, which decides what "validate" may assert.
#
# THIS FIELD EXISTS BECAUSE THE FIRST VALIDATOR CALLED THE WRONG DOOR AND REPORTED THE LOCK
# AS A BREAK-IN. `Tau2Suite.view()` raises `Tau2NeedsOrchestrator` on purpose: the only task
# text available outside the Orchestrator is the customer's roleplay script, which inverts
# the agent's role and hands over the user-private partition that the whole ceiling argument
# is defined over. A tau2 view that SUCCEEDS is the defect. So for an `orchestrator` suite
# the check is inverted -- the refusal is the passing condition, exactly as `GoldAccessError`
# is for the firewall.
DrivenBy = Literal["loop", "orchestrator", "replay", "gym"]


@dataclass(frozen=True)
class SuiteSpec:
    """One suite's declared role. Frozen because a role that can be mutated at runtime is a
    role that will be mutated at runtime, in a worker, halfway through a sweep."""

    suite_id: str
    title: str
    source: Source
    status: Status

    # THE TWO INDEPENDENT ROLE FLAGS.
    # mines_training_data: its `train` split is exported into data/rl/ for SFT/DPO/GRPO.
    # carries_endpoints:   its `test` split appears in a paper table.
    mines_training_data: bool
    carries_endpoints: bool

    axis: tuple[Axis, ...]
    endpoint_status: EndpointStatus
    driven_by: DrivenBy

    required_env: tuple[str, ...]
    note: str
    # Tasks that HAVE no gold graph and are supposed to have none. Named, not counted: a
    # count would let a new gap hide behind an old one. `validate` subtracts exactly these
    # and still reports anything else, because a blocker that is always there and always
    # expected is a blocker people learn to scroll past -- the same failure as declaring an
    # env var that does not exist.
    gold_gap_tasks: tuple[str, ...] = ()

    # True when retrieval is served from a RECORDED cache rather than a live index. For such
    # a suite an ad-hoc probe query missing the cache says nothing: the cache holds the
    # sub-queries the recorded runs actually issued, not the raw task questions. Treating
    # that miss as a blocker is how `validate` reported drgym as unrunnable while 342 of its
    # runs sat scored on disk. The honest check for a replay suite is whether recorded runs
    # exist, which is what `validate` reports instead.
    retrieval_is_replay: bool = False

    # False when the suite's reward does NOT come from a need-graph. UserBench is scored
    # inside its own gym -- set membership against the task's best/correct option ids -- so it
    # ships no GoldGraph and never will. `validate` was reporting "255 tasks have no gold
    # graph, they would score as a silent miss" for a suite where that is the design. This is
    # the fourth false blocker this command has produced; each one was a check that could not
    # tell an expected condition from a defect.
    uses_gold_graph: bool = True

    # False when `pi score` must take the ANSWERS-ONLY branch: no need-graph exists, so every
    # coverage, depth, facet and frontier metric is UNDEFINED rather than zero, and a 0.0
    # there is a measurement nobody took.
    #
    # THIS IS NOT A DUPLICATE OF `uses_gold_graph`, and collapsing the two was the first
    # draft. They are read by different processes for different decisions: `uses_gold_graph`
    # tells `pi suites validate` not to go looking for graphs it will not find, and userbench
    # sets it because it is scored inside its own gym and `pi score` never sees it at all.
    # `gold_free` tells the SCORER which branch to take on runs it does see. A suite can be
    # the first without being the second; frames is both, userbench is only the first.
    #
    # `pi_eval.score.GOLD_FREE_SUITES` is the scorer's copy, and `audit()` checks the two
    # agree. Neither module imports the other: this one must stay importable in a rollout
    # worker with PI_GOLD_ROOT unset, which is why the set is passed in, exactly as
    # `audit_against_prereg` takes the preregistration's.
    gold_free: bool = False

    @property
    def eval_only(self) -> bool:
        """Derived, never declared. Declaring it here as a third copy of the same fact is the
        defect this module was written to remove; `pinq.splitting` is the definition."""
        return self.suite_id in EVAL_ONLY_SUITES


# ------------------------------------------------------------------------------ the registry
#
# Ordered the way the paper uses them: the claim first, then the transfer targets, then the
# instruments that only bound or calibrate.

REGISTRY: Mapping[str, SuiteSpec] = {
    "musique": SuiteSpec(
        suite_id="musique",
        title="MuSiQue (multi-hop QA)",
        source="corpus",
        status="live",
        mines_training_data=True,
        carries_endpoints=True,
        axis=("vertical", "mechanism"),
        endpoint_status="confirmatory",
        driven_by="loop",
        required_env=(),
        note=(
            "The vertical suite, and the only real one carrying depth>=2: its prerequisite "
            "edges are MuSiQue's own human-authored `#N` placeholders, so the latent label "
            "needs no annotator. Hop-stratified on purpose -- head-N truncation yields all "
            "2-hop tasks and not one depth-2 node."
        ),
    ),
    "musique_x2": SuiteSpec(
        suite_id="musique_x2",
        title="Composed MuSiQue pairs (two held-out test tasks per task)",
        source="corpus",
        status="live",
        mines_training_data=False,
        carries_endpoints=True,
        axis=("vertical", "horizontal"),
        endpoint_status="exploratory",
        driven_by="loop",
        required_env=(),
        note=(
            "Each task joins two held-out MuSiQue TEST tasks (not in the checkpoint's training "
            "ids, no shared template, title or answer), both questions stated, pools unioned, "
            "gold graphs unioned with no cross-pair edge -- so exactly two facets, and every "
            "pair carries a constituent of depth >= 2. Covering the second question's chain is "
            "horizontal; each chain's depth is vertical. Both question orders are separate "
            "tasks and one cluster (`template_id` is the pair). Built by "
            "pi_eval.build.compose_build into an isolated root, declared in "
            "artifacts/composed_pairs_20260923/DECLARATION.md before any data. EVAL-ONLY "
            "because its content is test content. EXPLORATORY: built after the "
            "preregistration, so a point estimate and a CI, never a p-value. Framing: composed "
            "natural tasks, never a natural benchmark."
        ),
    ),
    "strategyqa": SuiteSpec(
        suite_id="strategyqa",
        title="StrategyQA (implicit decomposition)",
        source="corpus",
        status="live",
        mines_training_data=True,
        carries_endpoints=True,
        axis=("horizontal", "mechanism"),
        endpoint_status="confirmatory",
        driven_by="loop",
        required_env=(),
        note=(
            "The horizontal suite: the decomposition is strategic and NOT stated in the "
            "question. Binary answers, so it carries no answer-quality endpoint -- coverage "
            "and facet breadth only."
        ),
    ),
    "wiki2": SuiteSpec(
        suite_id="wiki2",
        title="2WikiMultihopQA",
        source="corpus",
        status="live",
        mines_training_data=True,
        carries_endpoints=True,
        axis=("mechanism",),
        endpoint_status="confirmatory",
        driven_by="loop",
        required_env=(),
        note=(
            "Depth <= 1 on dev, so it can carry no claim of its OWN -- and it is still "
            "load-bearing, because it sits in the pool of six preregistered secondary "
            "endpoints (the four kill-switch coverage contrasts, rnr_resolve, frontier_auc). "
            "Also the training-diversity source that matters most: mining stopped at 480 "
            "tasks that are 97% musique, and a policy trained on one suite's phrasing is a "
            "policy that learned that suite. BOTH roles at once is exactly the case the two "
            "independent flags exist to express."
        ),
    ),
    "drgym": SuiteSpec(
        suite_id="drgym",
        title="DeepResearchGym (open-ended research)",
        source="corpus",
        status="live",
        mines_training_data=False,
        carries_endpoints=True,
        axis=("horizontal", "mechanism"),
        endpoint_status="exploratory",
        driven_by="loop",
        retrieval_is_replay=True,
        required_env=("DRGYM_API_KEY",),
        note=(
            "Breadth over 16k benchmark-author key points, all at depth 0 -- so it is "
            "horizontal evidence and cannot speak to the vertical claim. Judge-derived, "
            "which is why it is the suite whose sigma_J had to be measured before any of "
            "its numbers were reportable. Offline replay by default; PI_DRGYM_ONLINE=1 "
            "re-queries the hosted index.\n"
            "EVAL-ONLY BY DECISION (user, 2026-09-15), not by a property of the dataset the "
            "way frames is. `mines_training_data` was ALREADY False here, so this flag did "
            "not move and `audit()` was clean before and after; what changed is that the "
            "suite entered `pinq.splitting.EVAL_ONLY_SUITES`, which is the difference "
            "between a role nobody had exercised and a refusal the code makes. MEASURED the "
            "same day: no shipped training file holds a drgym row -- `grep -c drgym` is 0 on "
            "each of data/rl/{sft,pairs,a6_preferences,reaches_preferences,latency_relabel}"
            ".jsonl, and data/rl/sft.manifest.json names only musique, strategyqa, synth and "
            "wiki2 -- so no dataset changes. What DOES change is runs.parquet: 201 drgym "
            "runs over 57 tasks were stamped `split=train` (and 51 over 17 `dev`) before the "
            "decision, and every one of them is now ineligible for a training export. Those "
            "stamps are left alone rather than rewritten: `split` is not in "
            "`pinq.ids.SEMANTIC_FIELDS`, so no run_id moves.\n"
            # The SECOND decision of the same day, and a different field. The note above moved
            # `mines_training_data` / EVAL_ONLY_SUITES; this one moves `endpoint_status`. They
            # are kept side by side because a role change whose reason is not written next to
            # it is what this registry exists to prevent, and collapsing them would lose which
            # flag each one actually moved.
            "ROLE CHANGED 2026-09-15 (T20): endpoint_status confirmatory -> exploratory, per "
            "the stage-1 draft and decisions D2/D8. Eval-only (above) and judge-derived, so a "
            "confirmatory row here would spend multiplicity on a transfer claim whose effect "
            "must clear the sigma_J noise floor before it may be cited at all. Rows keep a "
            "point estimate and a CI; they carry no p-value."
        ),
    ),
    "tau2": SuiteSpec(
        suite_id="tau2",
        title="tau2-bench (banking_knowledge)",
        source="self",
        status="live",
        mines_training_data=False,
        carries_endpoints=True,
        axis=("vertical", "cost", "mechanism"),
        endpoint_status="exploratory",
        driven_by="orchestrator",
        required_env=("TAU2_DATA_DIR",),
        note=(
            "The zero-shot transfer target, and the only suite where the interaction cost is "
            "a real user simulator's turns rather than a proxy. 86% of its gold assistant "
            "actions are the read-then-unlock-then-call prerequisite edge, so it carries the "
            "vertical claim in an action-consequential environment. EVAL-ONLY: 97 tasks "
            "cannot be split, and transfer is a stronger claim than a within-suite gain. "
            "ROLE CHANGED 2026-09-15: confirmatory -> exploratory, per the stage-1 draft. "
            "Banking's success sits at the floor, it carries ZERO legal fork points, and its "
            "follow-up numbers are a transfer row -- and the endpoint was already declared "
            "DIRECTIONAL (MDE 22.1pp at n=97, simulated through the test that actually runs), "
            "so the registry was promising a confirmatory claim the preregistration itself "
            "had already said this suite is not powered to make."
        ),
    ),
    "tau2_retail": SuiteSpec(
        suite_id="tau2_retail",
        title="tau2-bench (retail)",
        source="self",
        status="live",
        mines_training_data=True,
        carries_endpoints=True,
        axis=("vertical", "cost"),
        endpoint_status="confirmatory",
        driven_by="orchestrator",
        # Upstream's choice, not a build gap. Both ship "actions": [] because the correct
        # outcome is a database that does NOT change -- 24's customer asks to cancel a grill
        # and then keeps it, 57's item has already shipped -- and this builder's nodes are the
        # argument values gold actions required. Zero actions, zero nodes. The gold replay
        # skips the same two and reproduces 112 of 112.
        gold_gap_tasks=("24", "57"),
        required_env=("TAU2_DATA_DIR",),
        note=(
            "A SEPARATE suite_id, not a flag on tau2, exactly so that tau2 can stay "
            "eval-only while this one is trainable. Registering it as `tau2` would have "
            "destroyed the transfer claim silently. "
            "ROLE CHANGED 2026-09-15: exploratory -> confirmatory, per the stage-1 draft. "
            "This registry entry was written after pi_eval.prereg and the fork machinery "
            "landed between them: the retail FORK PAIRS are the draft's second confirmatory "
            "family -- 34 fork points over 25 task ids, 102 pairs, an exact two-sided sign "
            "test on follow-up turns paired with tau_reward, and the pair is claimed only "
            "when both members point the same way. Measured sd 6.3992 over 102 pairs -> "
            "permutation MDE 1.775 follow-ups, so it needs all three seeds."
        ),
    ),
    "tau2_airline": SuiteSpec(
        suite_id="tau2_airline",
        title="tau2-bench (airline)",
        source="self",
        status="live",
        mines_training_data=True,
        carries_endpoints=True,
        axis=("vertical", "cost"),
        endpoint_status="exploratory",
        driven_by="orchestrator",
        # Upstream ships "actions": [] for these seven. Measured from tasks.json; the gold
        # builder's nodes are the argument values gold actions required, so zero actions means
        # zero nodes. Named rather than counted -- a count lets the next gap hide behind these.
        gold_gap_tasks=("0", "10", "26", "28", "31", "34", "46"),
        required_env=("TAU2_DATA_DIR",),
        note=(
            "THE TOOL DOMAIN WE MINE. The only tau2 domain whose upstream task split this "
            "repo adopts verbatim (split_tasks.json, 30 train / 20 test), which is why it is "
            "the one that can be trained on without contaminating a public trace: "
            "`pinq.splitting.FIXED_SPLITS` reads that file rather than hashing ids, so a "
            "prefix taken from a public train trace can never land on a task we call test. "
            "MEASURED: the hash splitter put 14 of the 20 tasks the fork benchmark evaluates "
            "over into `train`. Built after `pi_eval.prereg` was written, so exploratory: "
            "point estimate and CI, never a p-value, until the preregistration is amended."
        ),
    ),
    "convlog": SuiteSpec(
        suite_id="convlog",
        title="Real agent-user sessions (private)",
        source="logs",
        status="live",
        mines_training_data=True,
        carries_endpoints=True,
        axis=("cost",),
        endpoint_status="exploratory",
        driven_by="replay",
        # NO ENV VAR. The first draft declared `PI_CONVLOG_CONSENT_SHA`, which does not exist
        # anywhere in src/, scripts/ or .env -- the consent gate is the `--consent-sha` FLAG
        # on the build, checked against the sha256 of docs/consent/convlog.md, with no flag to
        # skip it. Declaring a fictional variable made `validate` report a blocker that no one
        # could clear, which is how a readiness check teaches people to ignore it.
        required_env=(),
        note=(
            "The only source containing a person reacting to an agent. Replay-only: a "
            "candidate policy is compared by replaying held-out states, NEVER by rerunning a "
            "conversation, because a need the agent preempts never appears as a later "
            "utterance. Trains STOP far more than ASK -- 83% STOP, and B4 put the ceiling on "
            "the other half at 0.47% of silent states. Never released (D16).\n"
            "NOT YET AN EVALUATION SUITE IN CODE: ConvlogSuite implements only "
            "template_id_of, is not a TaskSuite, and load_suite cannot reach it. The replay "
            "evaluator that would grade two policies against one rendered prompt is "
            "described in render.py and not implemented, so `carries_endpoints` here records "
            "an intent the code does not yet satisfy -- and `audit()` cannot see that, which "
            "is why it is written down."
        ),
    ),
    "synth": SuiteSpec(
        suite_id="synth",
        title="Synthetic need-graph suite",
        source="corpus",
        status="live",
        mines_training_data=False,
        carries_endpoints=True,
        axis=("calibration",),
        endpoint_status="calibration",
        driven_by="loop",
        required_env=(),
        note=(
            "The need-graph is generated FIRST and the corpus derived from it, so coverage, "
            "depth and per-question value all have closed-form answers. Its endpoint is a "
            "calibration claim, not a finding: a deviation here is a harness bug."
        ),
    ),
    "userbench": SuiteSpec(
        suite_id="userbench",
        title="UserBench (Salesforce travelgym, external)",
        source="gym",
        status="live",
        mines_training_data=False,
        carries_endpoints=True,
        axis=("cost",),
        endpoint_status="exploratory",
        driven_by="gym",
        uses_gold_graph=False,
        required_env=("USERBENCH_DIR",),
        note=(
            "The one benchmark here that this project did not construct -- environment, "
            "user simulator, reward ladder and 85/15 split are all Salesforce's, published "
            "before this work -- which is what makes it the answer to 'you measured "
            "yourself with your own ruler'. 255 test tasks over travel22/33/44 at commit "
            "80506d2. EVAL-ONLY for a stronger reason than tau2's: upstream ships 2,651 "
            "TRAIN scenarios, so training on it is one config line away and the resulting "
            "number would still look like an external validation. Built after the "
            "preregistration, so exploratory. Its user simulator is an LLM call that "
            "returns None on failure and is scored as reward 0.0 with a canned reply, so "
            "every result must carry a zero from the failure counter in "
            "pinq_adapters.userbench.simulator or it is not admissible."
        ),
    ),
    "frames": SuiteSpec(
        suite_id="frames",
        title="FRAMES (Google DeepMind, external)",
        source="corpus",
        status="live",
        mines_training_data=False,
        carries_endpoints=True,
        axis=("vertical", "cost"),
        endpoint_status="exploratory",
        driven_by="loop",
        uses_gold_graph=False,
        gold_free=True,
        required_env=(),
        # NO `gold_gap_tasks`, DELIBERATELY, AND THE FIELD'S OWN DOCSTRING IS WHY: it means
        # "tasks that HAVE no gold graph and are supposed to have none". All 824 frames tasks
        # have a gold graph -- the answer. What seven of them have is an incomplete PARAGRAPH
        # POOL, which is a corpus gap, not a gold gap, and putting it in this field would have
        # made `validate` subtract a set it does not apply to while hiding a real gold gap
        # behind it later. The seven are named in the note below and in the build manifest's
        # `tasks_with_missing_pages`, which is the machine-readable source.
        note=(
            "The second benchmark here this project did not construct: 824 hand-written "
            "multi-hop questions, each shipping the Wikipedia articles that answer it, and "
            "the authors' own ablation puts iterative multi-step retrieval at 66% against "
            "single-shot at 47% -- so the vertical structure we claim to exploit is present "
            "and was measured by someone else. In the oracle-articles setting it is the same "
            "closed per-question pool as musique, two orders of magnitude larger: ~20 "
            "paragraphs per musique task against hundreds here.\n"
            "EVAL-ONLY, and unlike tau2 or userbench there is no train split upstream to be "
            "tempted by. `axis` says vertical in the sense of the ANSWER endpoint on "
            "questions that need a chain; it carries NO depth metric, because it has no "
            "need-graph -- see `gold_free`. EXPLORATORY because it was chosen on 2026-09-11, "
            "long after the preregistration was written, so it gets a point estimate and a "
            "CI and never a p-value until that is amended.\n"
            "SEVEN TASKS RUN WITH AN INCOMPLETE POOL, measured on the 2026-09-12 build "
            "(corpus ad2841f391ba94b9): frames_0021, frames_0023, frames_0141, frames_0199, "
            "frames_0540, frames_0580, frames_0644. Each is short one linked article -- four "
            "are Wikipedia stubs that render no paragraph over the 80-character floor, three "
            "are titles that no longer exist, one of those being the editorial note "
            "'Pokemon (NOT REQUIRED, BUT HELPFUL)' that upstream wrote into its own link "
            "column. They are a CORPUS gap and not a gold gap, which is why they are not in "
            "`gold_gap_tasks`; the build manifest's `tasks_with_missing_pages` is the source."
        ),
    ),
    "pare": SuiteSpec(
        suite_id="pare",
        title="PARE (proactive-agents research environment)",
        source="self",
        status="cut",
        mines_training_data=False,
        carries_endpoints=False,
        axis=(),
        endpoint_status="none",
        driven_by="orchestrator",
        required_env=("PARE_BENCHMARK_SPLITS_DIR",),
        note=(
            "CUT. Grid E was never run: PareActuator.attach_validation has no caller, so the "
            "oracle verdict it exists to produce is unreachable. Left registered rather than "
            "deleted so that `audit()` reports it as cut instead of as missing, and so the "
            "decision is visible rather than inferred from an absence."
        ),
    ),
}


# --------------------------------------------------------------------------- derived views
#
# Each of these answers one of the two questions the plan asks: what do we mine, and what do
# we measure on. They are functions rather than precomputed constants so that a suite whose
# status changes cannot leave a stale tuple behind at import time.


def training_sources() -> tuple[str, ...]:
    """Suites whose `train` split is exported into data/rl/. The answer to 'where does the
    training data come from'."""
    return tuple(
        s.suite_id for s in REGISTRY.values() if s.status == "live" and s.mines_training_data
    )


def evaluation_suites() -> tuple[str, ...]:
    """Suites whose `test` split reaches a paper table. The answer to 'what is this measured
    on'."""
    return tuple(
        s.suite_id for s in REGISTRY.values() if s.status == "live" and s.carries_endpoints
    )


def transfer_suites() -> tuple[str, ...]:
    """Evaluation suites the policy provably never trained on. This is the subset whose
    numbers are a transfer claim rather than a within-suite gain."""
    return tuple(s for s in evaluation_suites() if s in EVAL_ONLY_SUITES)


def suites_for_axis(axis: Axis) -> tuple[str, ...]:
    return tuple(s.suite_id for s in REGISTRY.values() if s.status == "live" and axis in s.axis)


class RegistryDrift(RuntimeError):
    """The registry disagrees with the code it describes.

    Raised by `audit()` rather than returned, because a registry that is merely *reported* as
    wrong is a registry someone reads past.
    """


def audit(loadable: frozenset[str], gold_free: frozenset[str] | None = None) -> list[str]:
    """Cross-check the declarations against the code, and return every disagreement.

    `loadable` is the set of suite_ids `pi_run.worker.load_suite` can actually construct; it
    is passed in rather than imported so this module stays importable in a worker that has
    none of the suite extras installed. `gold_free` is `pi_eval.score.GOLD_FREE_SUITES`,
    passed in for the stronger version of the same reason: importing pi_eval here would work
    today and would be one refactor away from a GoldAccessError at import time in every
    worker. Omit it to skip that one check rather than to pass it vacuously.

    Returns a list of human-readable problems, empty when the registry is consistent. The
    caller decides whether that is a failure -- `pi suites audit` prints them, the test
    asserts the list is empty.
    """
    problems: list[str] = []

    live = {s.suite_id for s in REGISTRY.values() if s.status == "live"}

    # A suite the runner can build but nobody declared a role for. This is the direction that
    # actually happens: an adapter lands, a grid runs it, and no one says what it is for.
    for sid in sorted(loadable - set(REGISTRY)):
        problems.append(f"{sid}: load_suite can build it, but it is not in the registry")

    # A live suite the runner cannot build. convlog is the deliberate exception: it is
    # replay-only and never goes through load_suite.
    for sid in sorted(live - loadable - {"convlog"}):
        problems.append(f"{sid}: declared live, but load_suite cannot construct it")

    # THE ONE THAT MATTERS. A suite may not be mined for training and be eval-only. This is
    # the training-set leak, stated as an assertion instead of as a comment.
    for spec in REGISTRY.values():
        if spec.mines_training_data and spec.eval_only:
            problems.append(
                f"{spec.suite_id}: declared as a training source AND is in EVAL_ONLY_SUITES. "
                f"One of the two is wrong, and if it is the split the transfer claim is void."
            )
        if spec.status == "cut" and (spec.mines_training_data or spec.carries_endpoints):
            problems.append(f"{spec.suite_id}: status is cut but it still declares a role")
        if spec.status == "live" and not (spec.mines_training_data or spec.carries_endpoints):
            problems.append(
                f"{spec.suite_id}: declared live but is neither mined nor measured -- "
                f"say what it is for, or mark it cut"
            )
        if spec.carries_endpoints and not spec.axis:
            problems.append(
                f"{spec.suite_id}: carries an endpoint but names no axis, so a reader cannot "
                f"tell which claim its number supports"
            )
        if spec.gold_free and spec.uses_gold_graph:
            problems.append(
                f"{spec.suite_id}: declared gold_free but also uses_gold_graph -- one of the "
                f"two is wrong, and if it is gold_free the scorer will emit depth metrics "
                f"over a graph that has no nodes"
            )

    # THE SCORER'S COPY OF THE SAME FACT. Two trees, one truth, and neither may import the
    # other: a registry that says answers-only while the scorer computes a need-graph
    # (or the reverse) is a wrong number in whichever direction disagrees.
    if gold_free is not None:
        declared = {s.suite_id for s in REGISTRY.values() if s.gold_free}
        for sid in sorted(declared - set(gold_free)):
            problems.append(
                f"{sid}: registry says gold_free, pi_eval.score.GOLD_FREE_SUITES does not -- "
                f"`pi score` will emit structural metrics over a graph with no nodes"
            )
        for sid in sorted(set(gold_free) - declared):
            problems.append(
                f"{sid}: pi_eval.score.GOLD_FREE_SUITES names it, the registry does not "
                f"declare it gold_free -- its endpoints are being silently suppressed"
            )

    return problems


def audit_against_prereg(confirmatory_suites: frozenset[str]) -> list[str]:
    """Check the declared `endpoint_status` against what `pi_eval.prereg` actually declares.

    The set is passed in rather than imported for the same reason `audit()` takes `loadable`:
    this module must stay importable in a rollout worker, and a worker has `PI_GOLD_ROOT`
    unset by design. Importing `pi_eval` there would work today and would be one refactor
    away from a `GoldAccessError` at import time in every worker.

    Both directions are checked, and the second is the one that bites:

      * a suite declared `confirmatory` that prereg does not name -- the registry is
        promising a p-value the preregistration never authorised;
      * a suite prereg names that the registry calls `exploratory` -- the paper is about to
        under-claim, or, far worse, a suite was quietly added to a pool after the design was
        written and nobody noticed that it moved a declared denominator.
    """
    problems: list[str] = []
    for spec in REGISTRY.values():
        if spec.status != "live":
            continue
        named = spec.suite_id in confirmatory_suites
        if spec.endpoint_status == "confirmatory" and not named:
            problems.append(
                f"{spec.suite_id}: declared confirmatory, but pi_eval.prereg names no "
                f"endpoint for it -- it may not carry a p-value"
            )
        if spec.endpoint_status == "exploratory" and named:
            problems.append(
                f"{spec.suite_id}: declared exploratory, but pi_eval.prereg DOES name it. "
                f"Either the registry is stale or a suite entered a declared pool after the "
                f"design was written"
            )
    return problems
