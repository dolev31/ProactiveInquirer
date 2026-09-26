"""`pi train` -- the training track's five commands: export, rung0, rung1, rung2, status.

WHY EVERY REFERENCE TO `pinq_train` IN THIS FILE IS A STRING
============================================================
Import-linter contract 4 says NOTHING may import `pinq_train`, and `pi_run` is one of its
source modules. That contract is not bureaucracy: `pi_run` reaches `pi_eval` (the scoring
handlers do, by design), so a static edge from `pi_run` to `pinq_train` would put a
`pinq_train -> pi_run -> pi_eval` path into the module graph and the gold firewall would be
one transitive hop from broken. grimp sees imports inside functions as well as at module
scope, so a "lazy" `import pinq_train` would break it exactly as loudly.

So this module loads the trainer through `importlib.import_module(...)` by NAME. That is not a
loophole around the contract, it is the contract's intended shape: the trainer is a separate
program that this CLI launches, in the same sense that `pi run` launches workers. The direction
that matters -- `pinq_train` must never reach `pi_eval` -- is unaffected, and
`tests/test_train_cli.py` asserts this file contains no static import of it.

WHERE THE ROWS COME FROM, AND WHY THE STATE IS RE-RENDERED RATHER THAN REMEMBERED
=================================================================================
`pinq_train.export.dataset` takes rows and enforces the split; it does not know how to read a
run directory, and it must not learn, because reading one requires the suite adapters and the
corpus. This module supplies the rows.

The `state_text` of a row is the EXACT prompt the Inquirer saw, re-rendered from the recorded
artifacts through `pinq.promptlib` -- the same template, the same `render_evidence` and
`render_history`, the same placebo user-channel fragment. Three facts make that exact rather
than approximate:

  * `Evidence` is canonically ordered and deduplicated by uid, so the evidence block is a
    function of the retrieved uid SET and nothing else -- which is precisely what
    `Turn.subset_hash_before` records;
  * `Turn.response_text` holds the Drafter's actual reply text, which is what `render_history`
    renders, so the history is reproduced verbatim;
  * `state.draft` is `None` for every turn INSIDE the inquiry loop -- `pinq.loop` calls
    `drafter.draft()` only after the loop exits -- so the draft slot is always
    "(no draft yet)". There is no draft to reconstruct and therefore nothing to approximate.

`--verify-state` re-derives `subset_hash_before` from the reconstructed evidence and refuses
the run if it disagrees, so a reconstruction that silently drifted from the recorded state is
caught rather than exported.

WHY THE EXPORTER SCORES IN-PROCESS AND THE TRAINER DOES NOT
===========================================================
`pi train export` is a GOLD-SIDE command, like `pi score`: it needs `PI_GOLD_ROOT` and it calls
`pi_run.serve.score.handle_score` directly. The seam exists so the TRAINER cannot read gold,
not so that a dataset build cannot. Rungs 0-3 never call this path; they reach the same
function over HTTP through `pinq_train.client.SeamClient`, whose constructor REFUSES to run in
a process where `PI_GOLD_ROOT` is set. Two processes, two roles, one measurement function.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from pinq.actions import STOP_ACTION_JSON, ask_action_json

# The Inquirer template every trained/prompted/frontier family shares. Rung 0 optimises it and
# writes the winner into an overlay directory; nothing here edits the shipped file.
INQUIRER_PROMPT = "inquirer_prompted"

# The placebo paragraph every arm except `inquirer_may_ask_user` renders. Token-matched to the
# real one, so opening the user channel changes what the prompt says and not how long it is.
USER_CHANNEL_FRAGMENT = "fragment_user_channel_placebo"

DEFAULT_OUT = "data/rl"
# WHICH `code_version` VALUES ARE ONE LOOP. Written by `scripts/cohort_ids.py` from git
# ancestry over the corpus's distinct shas, and committed, so no consumer needs an object
# store. The exporter's cross-cohort guard reads it; `pi train export` records its sha256 on
# the pairs manifest as `cohort_file_sha`.
DEFAULT_COHORT_FILE = "conf/cohorts/fixed_loop.json"
DEFAULT_SCORE_URL = "http://127.0.0.1:8077"

# Arms whose questions were chosen where gold is readable. Refused by name in addition to the
# manifest's `gold_exposed` flag, because a ceiling arm exported by accident is oracle
# distillation that no ablation could recover from.
GOLD_EXPOSED_ARMS = frozenset({"gold_evidence", "oracle_vreq"})

# WHOSE SAMPLES TRAIN THE HEADLINE. A scope decision, not a contamination one -- which is why
# it is a default a flag can widen rather than a refusal by class.
#
# MEASURED (2026-09-11, `data/rl/sft.jsonl` joined to `scores/parquet/runs.parquet`): 35,833
# SFT rows are `inquirer_prompted`'s, 8,399 are `self_inquire`'s and ~250 come from the other
# comparator arms; on the pairs side 7,286 ask_ask pairs are prompted-vs-prompted against
# 3,145 self_inquire-vs-self_inquire. So a quarter of the ASK supervision belonged to a
# DIFFERENT policy, and the headline checkpoint had partly distilled the comparator it is
# later measured against -- a reviewer's first question, with no answer on the artifact.
#
# The refusal lives in `rows_from_run` and not in a post-hoc filter because a filter cannot
# restore a row the walk never built, and because `ExportManifest.n_arm_refused` must count
# what scope cost. `--include-arm` re-admits an arm by name for the '+ancestor rows'
# ablation, and the manifest records which arms were admitted, so the two artifacts are
# distinguishable by reading them.
DEFAULT_TRAINABLE_ARMS = frozenset({"inquirer_prompted"})


def _train(name: str) -> Any:
    """Load a `pinq_train` module BY NAME. See this module's docstring for why."""
    try:
        return importlib.import_module(f"pinq_train.{name}")
    except ModuleNotFoundError as exc:  # pragma: no cover - only without an editable install
        raise SystemExit(
            f"cannot load pinq_train.{name}: {exc}. Install the repo editable "
            '(`uv pip install -e ".[dev]"`); the training track ships in the same tree.'
        ) from None


# --------------------------------------------------------------------------- run artifacts


def _read_json(p: Path) -> dict[str, Any]:
    return json.loads(p.read_text())


def _read_jsonl(p: Path) -> list[dict[str, Any]]:
    if not p.exists():
        return []
    return [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]


class PoolUnavailable(RuntimeError):
    """A suite whose per-task evidence pool cannot be enumerated offline."""


def _task_units(suite: Any, task_id: str) -> Sequence[Any]:
    direct = getattr(suite, "units", None)
    if callable(direct):
        return tuple(direct(task_id))
    pool = getattr(suite.retriever(task_id), "_units", None)
    if pool is not None:
        return tuple(pool)
    raise PoolUnavailable(
        f"{type(suite).__name__} exposes neither units(task_id) nor a retriever holding its "
        "unit pool, so the state the policy saw cannot be re-rendered from a uid. A suite "
        "whose pool is remote (drgym) can never be exported this way; that is a property of "
        "the suite, not a bug here."
    )


class SuiteCache:
    """Suite adapters, built once per suite. A corpus load per run directory would dominate."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._suites: dict[str, Any] = {}
        self._units: dict[tuple[str, str], dict[str, Any]] = {}
        self._views: dict[tuple[str, str], Any] = {}
        self._answer_nodes: dict[tuple[str, str], tuple[frozenset[str], ...]] = {}

    def suite(self, suite_id: str) -> Any:
        if suite_id not in self._suites:
            from pi_run.serve.rollout import resolve_corpus
            from pi_run.worker import load_suite

            self._suites[suite_id] = load_suite(suite_id, str(resolve_corpus(self.root, suite_id)))
        return self._suites[suite_id]

    def units(self, suite_id: str, task_id: str) -> dict[str, Any]:
        """uid -> EvidenceUnit for one task's whole pool.

        THE SUITE PROTOCOL HAS NO `units()`. `pinq.protocols` requires `view`, `retriever`
        and `task_ids` and nothing else, because a rollout only ever needs units the
        RETRIEVER hands it; only `ParagraphSuite` happens to expose the pool directly. The
        exporter does need the whole pool -- it has a uid and must recover the text that uid
        named -- so it tries the direct accessor first and falls back to the retriever's own
        unit list, which every retriever in this repo holds because it must search over it.
        Widening the protocol would mean editing every adapter for the benefit of one
        offline tool, so the fallback is here and the failure is explicit rather than an
        AttributeError three frames down.
        """
        key = (suite_id, task_id)
        if key not in self._units:
            self._units[key] = {u.uid: u for u in _task_units(self.suite(suite_id), task_id)}
        return self._units[key]

    def view(self, suite_id: str, task_id: str) -> Any:
        key = (suite_id, task_id)
        if key not in self._views:
            self._views[key] = self.suite(suite_id).view(task_id)
        return self._views[key]

    def answer_node_uid_sets(
        self, suite_id: str, task_id: str, graph: Any
    ) -> tuple[frozenset[str], ...]:
        """The task's answer-bearing node(s), as one gold-uid set each. Cached per task.

        A PROPERTY OF THE TASK, and the walk sees ~35 runs per task, so recomputing it per RUN
        would re-run `contains_answer` over every required node's evidence 35 times for an
        answer that cannot have changed. Empty means "no checkable answer node" and is cached
        too: the absence is as stable as the presence.

        CACHED ON THE INSTANCE AND NOT AT MODULE SCOPE, deliberately. `_graphs` is a module
        cache keyed by (suite, version, gold root), which is right for a graph FILE; this is
        derived from whichever graph object the caller holds, and a module cache would let one
        test's doctored graph answer another test's question about the real one -- the same
        (suite, task) key, a different graph, no error anywhere.
        """
        from pi_eval.answer_node import answer_node_uid_sets, answer_nodes_for_task

        key = (suite_id, task_id)
        if key not in self._answer_nodes:
            units = self.units(suite_id, task_id)
            picked = answer_nodes_for_task(graph, {uid: u.text for uid, u in units.items()})
            self._answer_nodes[key] = answer_node_uid_sets(graph, picked.node_ids)
        return self._answer_nodes[key]


# THE VALUE OF A RECORDED STOP. A STOP retrieves nothing, so its evidence gain, redundancy
# and retrieval charge are all zero: `w_phi*0 - w_red*0 - 0`. Not a tunable; the reward
# formula evaluated at the action that pays nothing and gains nothing. Whether a STOP was
# RIGHT is a gold question (`done_before`), answered by the exporters, never by this number.
STOP_VALUE: float = 0.0


class ArmNotAllowed(RuntimeError):
    """This run's arm is outside the export's allowlist. See `DEFAULT_TRAINABLE_ARMS`.

    A RAISE, not an empty return. `collect_rows` counts exceptions by class, so a function that
    returned [] here would be indistinguishable from a run holding no decision points, and the
    skip table -- the only place the loss is visible -- would stay silent about a quarter of the
    corpus. Distinct from `StateMismatch` because the remedy is different: nothing is wrong with
    the run, it is simply not this dataset's population.
    """


class StateMismatch(RuntimeError):
    """The reconstructed evidence set does not hash to what the run recorded.

    Fatal for that run rather than a warning: an export whose `state_text` is not the prompt
    the policy saw trains the policy on inputs it will never receive, and nothing downstream
    -- not the loss, not the eval -- can tell you that happened.
    """


def render_state(
    cache: SuiteCache,
    suite_id: str,
    task_id: str,
    turns: Sequence[dict[str, Any]],
    upto: int,
    *,
    verify: bool = True,
    expect_hash: str | None = None,
) -> str:
    """Re-render the Inquirer's prompt as it stood before `turns[upto]` was decided.

    `upto == len(turns)` renders the state AFTER the last recorded turn: the state a recorded
    STOP was decided in. No turn carries a `subset_hash_before` for it, so the caller passes
    the last turn's `subset_hash_after` as `expect_hash` and the same guard applies.
    """
    from pinq import promptlib
    from pinq.types import Ask, Evidence, Turn
    from pinq_expt.components import render_evidence, render_history

    by_uid = cache.units(suite_id, task_id)
    view = cache.view(suite_id, task_id)

    held: list[Any] = []
    history: list[Turn] = []
    for prior in turns[:upto]:
        held.extend(by_uid[u] for u in prior.get("retrieved_uids") or () if u in by_uid)
        history.append(
            Turn(
                turn_idx=int(prior["turn_idx"]),
                action=Ask(text=str(prior.get("question", "")), rationale=""),
                # `response_sha` is the pre-rename key. Run directories already on disk carry
                # it, and they are still readable: the field was renamed because it holds
                # TEXT, not because its contents changed.
                response_text=str(prior.get("response_text") or prior.get("response_sha") or ""),
            )
        )
    evidence = Evidence.of(tuple(held))
    if upto < len(turns):
        where = f"turn {turns[upto].get('turn_idx')}"
    elif turns:
        where = f"the STOP after turn {turns[-1].get('turn_idx')}"
    else:
        where = "the STOP before any turn"
    if verify:
        if expect_hash is not None:
            recorded = str(expect_hash)
        elif upto < len(turns):
            recorded = str(turns[upto].get("subset_hash_before", ""))
        else:
            raise ValueError(
                f"render_state(upto={upto}) over {len(turns)} turns: no turn carries a "
                "subset_hash_before for this state; pass expect_hash."
            )
        if recorded and evidence.subset_hash != recorded:
            raise StateMismatch(
                f"{suite_id}/{task_id} {where}: reconstructed "
                f"evidence hashes to {evidence.subset_hash[:12]} but the run recorded "
                f"{recorded[:12]}. The rendered state would not be the state the policy saw."
            )
    # D_t. THE COMMENT THAT USED TO BE HERE DESCRIBED A LOOP THAT NO LONGER EXISTS.
    #
    # It read: "`pinq.loop` calls drafter.draft() only AFTER the inquiry loop exits, so no turn
    # inside the loop ever saw a draft. This is reconstruction, not approximation." That was
    # true of the pre-fix loop, and the fix that made D_t real -- drafting INSIDE the loop so
    # the next act() can see it -- is the whole two-agent mechanism. `loop.py` now stamps
    # `draft_sha` on every turn it drafted for.
    #
    # Measured on the recorded runs: 149 of 335 turns carry a non-empty draft_sha, including 26
    # at turn 0. Rendering "(no draft yet)" for those produces a `state_text` that is NOT the
    # prompt the policy saw, on 44% of turns -- and the guard above cannot catch it, because it
    # re-derives `subset_hash_before` (evidence) and says nothing about the draft.
    #
    # The draft TEXT is not persisted (only its sha), so exact reconstruction is impossible from
    # the current artifacts. This REFUSES rather than substituting, for the same reason the
    # evidence guard raises: silently training on a prompt the policy never saw is worse than
    # not training. To make these turns exportable, persist the draft text on the Turn the way
    # `response_text` already is, or resolve it from the response cache by request_sha.
    # THE DRAFT THE POLICY ACTUALLY SAW, or a refusal -- never a substitute.
    #
    # `Turn.draft_text` now persists it (the sha alone cannot reconstruct a prompt), so a turn
    # recorded by a current sweep renders exactly what `act()` was shown. A turn recorded BEFORE
    # that field existed carries a draft_sha and no text: those are still refused, because
    # rendering "(no draft yet)" for them would train on a prompt the policy never saw, and
    # `collect_rows` counts the refusal rather than silently shrinking the dataset.
    # THE PREVIOUS TURN, NOT THIS ONE. `pinq.loop` computes the draft AFTER turn i's
    # retrieval and stamps it on Turn i -- "refreshed INSIDE the loop so the NEXT act() sees
    # it" -- so `turns[i]` carries D_{i+1}, the belief held when deciding turn i+1.
    #
    # Reading `turns[upto]` here rendered the draft produced BY the very retrieval the
    # predicted action caused: the future, inside the prompt used to predict it. The guard
    # above cannot catch it, because it verifies `subset_hash_before` while that draft was
    # computed from `subset_hash_after` -- the two halves of the rendered state came from
    # opposite sides of the decision.
    #
    # Provable at turn 0: `run_loop` calls `act()` with `State(view=view)` and no draft at
    # all, yet 138 of 174 recorded turn-0 rows carry a draft_sha. Turn 0 has no predecessor,
    # so it correctly renders "(no draft yet)" and needs no draft to be exportable.
    prev = turns[upto - 1] if upto > 0 else {}
    recorded_draft = str(prev.get("draft_sha", ""))
    draft_text = str(prev.get("draft_text", ""))
    if recorded_draft and not draft_text:
        raise StateMismatch(
            f"{suite_id}/{task_id} {where}: the run recorded a draft "
            f"({recorded_draft[:12]}) on the PRECEDING turn but carries no draft_text -- it "
            "predates that field. Rendering '(no draft yet)' here would train on a prompt the "
            "policy never saw. Re-run the sweep to record draft_text, or export only "
            "draft-free turns."
        )
    return promptlib.render(
        INQUIRER_PROMPT,
        question=view.question,
        instructions=view.instructions,
        evidence=render_evidence(evidence),
        draft=draft_text or "(no draft yet)",
        history=render_history(history),
        user_channel=promptlib.load(USER_CHANNEL_FRAGMENT).strip(),
    )


def action_json(turn: dict[str, Any]) -> str:
    """The action as the policy emitted it, in the parser's canonical key order.

    Both shapes come from `pinq.actions`, the one place they are defined. This function used
    to serialise a STOP with a `rationale` key on a branch nothing could reach -- a STOP is
    never recorded as a turn -- while the SFT exporter wrote a different, compact STOP. Now
    that recorded STOPs become rows (see `rows_from_run`), the branch is live and must agree
    with every other writer byte for byte.
    """
    if str(turn.get("action_kind")) != "ask":
        return STOP_ACTION_JSON
    return ask_action_json(str(turn.get("question", "")), str(turn.get("rationale", "")))


def self_reported_parents(turn: dict[str, Any]) -> tuple[str, ...]:
    """`Ask.parent_uids`: the policy's OWN claim about what made it aware of this need.

    The Inquirer prompt asks for exactly this -- "cite in parent_uids the units whose content
    made you aware of this need, or leave it empty if the need is stated in the task itself" --
    so an empty tuple is the policy asserting depth 0 and a non-empty one is it asserting
    latency. MEASURED across recorded runs: 45% of ask-turns carry one, and on musique the
    share is monotone in turn index (0% at turn 0, 37.5% by turn 7), exactly as the semantics
    predict.

    It is the only latency signal here produced BY THE POLICY rather than derived from gold,
    which makes it the natural thing to validate the automatic label against -- and it is
    SELF-REPORTED, so it is audited and never scored, the treatment `pinq.types.Ask` already
    prescribes.

    DELIBERATELY NOT IN `action_json`. That string is the SFT target and `rung1_sft.mask`
    supervises exactly its tokens, so a content hash in there would take most of the
    supervised positions and train the policy to memorise uids; it would also silently move
    `len_delta`, which is the length guard's whole job.
    """
    return tuple(str(u) for u in (turn.get("parent_uids") or ()))


_GRAPHS: dict[tuple[str, str, str], dict[str, Any]] = {}


def _graphs(suite_id: str, graph_version: str) -> dict[str, Any]:
    """`load_graphs` re-reads the whole suite file, and the exporter calls it once per RUN.

    musique ships 800 graphs and the tree holds ~900 run directories, so an uncached read is
    720,000 graph constructions for one export. Memoised on (GOLD ROOT, suite, version), which is
    what `SuiteCache` already does for corpora and for the same reason.

    THE ROOT IS IN THE KEY AND WAS NOT. `load_graphs` reads `gold_root() / "graphs" / suite /
    ...`, so the root is an input to the answer; leaving it out of the key meant a process that
    ever saw two gold roots served the first one's graphs for the second one's runs. Inside one
    export that is invisible -- a process has one root -- which is exactly why it survived.

    AND THE FAILURE MODE IS A SILENT ZERO. `rows_from_run` turns a missing graph into
    `NoGoldForTask`; `collect_rows` counts that as a skip and carries on. So the export finishes,
    exits 0, writes a manifest and holds no rows: the shape of every bug this repository is
    organised against. Found when a dev export built its corpus after another process had warmed
    the cache and every dev run was refused `no_gold` for a task that exists.
    """
    from pi_eval.gold import gold_root, load_graphs

    key = (str(gold_root()), suite_id, graph_version)
    if key not in _GRAPHS:
        _GRAPHS[key] = load_graphs(suite_id, graph_version)
    return _GRAPHS[key]


def _match_records(run_dir: Path, graph: Any, turns: list[dict[str, Any]]) -> list[Any]:
    """Run the REAL matcher over a minimal trajectory view of the recorded turns.

    Reimplementing "a node is resolved when all its gold uids were retrieved" here would be a
    second declaration of the matching rule, free to disagree with `MechanicalMatcher` -- the
    shape this repository keeps finding bugs in. The matcher needs only `.turns` (turn_idx,
    action.text, retrieved_uids) and `.outcome.answer.cited_unit_ids`, all of which a run
    directory already holds, so it is driven rather than copied.
    """
    from pi_eval.matcher.base import MechanicalMatcher

    class _A:
        def __init__(self, text):
            self.text = text

    class _T:
        def __init__(self, row):
            self.turn_idx = int(row.get("turn_idx", 0))
            self.action = _A(str(row.get("question", "")))
            self.retrieved_uids = tuple(row.get("retrieved_uids") or ())

    outcome = _read_json(run_dir / "outcome.json") if (run_dir / "outcome.json").exists() else {}
    cited = tuple((outcome.get("answer") or {}).get("cited_unit_ids") or ())

    class _Ans:
        cited_unit_ids = cited

    class _Out:
        answer = _Ans()

    class _Traj:
        pass

    traj = _Traj()
    traj.turns = [_T(r) for r in turns]
    traj.outcome = _Out()
    return MechanicalMatcher().match(graph=graph, trajectory=traj, run_id=run_dir.name)


@dataclass(frozen=True, slots=True)
class BranchState:
    """A decision point where more than one required need was askable."""

    turn_idx: int
    frontier_size: int
    # The deepest need resolved AT this turn, as the run actually played it. Carried for
    # prioritisation only -- a fork does not have to follow the recorded choice.
    latent_depth: int


def branchable_states(graph: Any, records: Any, *, n_turns: int) -> list[BranchState]:
    """Turns worth forking: those where a CHOICE existed. Earliest first.

    `frontier_size >= 2` is the whole criterion. With one need askable, every candidate is a
    paraphrase of the same question and the resulting preference pair says nothing about WHICH
    need to pursue -- it ranks phrasing, which is not the claim.

    WHY NOT `newly_reachable`, which the first pass used. A need is "newly reachable" when its
    last prerequisite landed on the previous turn. A ROOT need is on the frontier from t=0, so
    nothing ever makes it appear and turn 0 can never satisfy that predicate. Measured on the
    built musique corpus: 44% of tasks offer more than one frontier need at t=0, against 27%
    at t=1 and 20% at t>=2. The criterion excluded exactly the states where the choice is
    widest, which is also where a wrong choice is least recoverable -- an early fork dominates
    a later one, because every later state is downstream of it.

    Returns [] rather than raising when the graph carries no usable depth (drgym ships 0 edges
    and null `gold_depth` on all 16,156 nodes): "nothing to branch here" is the right answer
    for a flat graph, not an error.
    """
    from pi_eval.metrics.latent import latent_labels

    labels = latent_labels(graph, records, turn_idxs=tuple(range(max(0, int(n_turns)))))
    out: list[BranchState] = []
    for t in sorted(labels.per_turn):
        if t < 0 or t >= n_turns:
            continue
        row = labels.per_turn[t]
        if row.frontier_size < 2:
            continue
        out.append(
            BranchState(
                turn_idx=int(t),
                frontier_size=int(row.frontier_size),
                latent_depth=int(row.latent_depth),
            )
        )
    return out


def _task_text(cache: Any, suite_id: str, task_id: str) -> str:
    """The task statement, for nameability. Empty on any failure, which yields
    `nameable_from_task=None` rather than a false negative."""
    try:
        return str(getattr(cache.view(suite_id, task_id), "question", "") or "")
    except Exception:
        return ""


def latent_fields(
    graph: Any, records: Any, *, turn_idx: int, task_text: str = ""
) -> dict[str, Any]:
    """The latent-need labels for one decision point.

    Attached HERE, gold-side, because `pinq_train` may never import `pi_eval` (contract 1);
    they cross to the trainer as plain numbers on the row dict.

    `has_gold_node` and `nameable_from_task` exist because `is_latent` alone was carrying two
    different failures. `latent_depth == -1` means nothing was resolved at this turn and was
    folded into `is_latent=False`, indistinguishable from a depth-0 root -- 54.1% of rows. And
    `gold_depth >= 1` was documented as "never nameable from the task statement", which three
    A4 raters unanimously contradicted on 21 of 99 depth-3 items, because musique names its
    intermediates descriptively. Both are now separate fields rather than one overloaded flag.
    """
    from pi_eval.metrics.latent import latent_labels

    t = latent_labels(graph, records, task_text=task_text).per_turn[int(turn_idx)]
    return {
        "latent_depth": t.latent_depth,
        "is_latent": t.is_latent,
        "newly_reachable": t.newly_reachable,
        "frontier_size": t.frontier_size,
        "has_gold_node": t.has_gold_node,
        "nameable_from_task": t.nameable_from_task,
    }


def rows_from_run(
    run_dir: Path,
    cache: SuiteCache,
    *,
    graph_version: str,
    weights: Any,
    verify_state: bool = True,
    include_arms: frozenset[str] = DEFAULT_TRAINABLE_ARMS,
    split: str = "train",
) -> list[dict[str, Any]]:
    """One run directory -> one row per decision point, with a measured per-turn value.

    A decision point is every recorded ASK and, when the run ended because the policy chose
    to (`stop_reason == "policy_stop"`), the STOP itself. `pinq.loop` records what was
    retrieved and a STOP retrieves nothing, so it is not a `Turn` -- but it IS the decision
    this dataset exists to teach. Before this function emitted it, 1,988 of 23,583 ok fork
    candidates (8.4%) -- the ones that stopped at the fork turn -- produced no row and
    vanished from both exporters, uncounted. The STOP row is rendered from the evidence after
    the last turn and verified against that turn's `subset_hash_after`; its value is
    `STOP_VALUE`; its outcome fields are the episode's real ones, because a recorded STOP has
    a real `outcome.json`. A budget or max_turns stop is the harness's decision, not the
    policy's, and gets no row: teaching STOP there would teach the cap.
    """
    from pi_run.serve.score import handle_score, score_heldout
    from pinq.wire import ScoreRequest

    reward = _train("reward")
    manifest = _read_json(run_dir / "manifest.json")
    turns = sorted(_read_jsonl(run_dir / "turns.jsonl"), key=lambda t: int(t["turn_idx"]))
    suite_id, task_id = str(manifest["suite_id"]), str(manifest["task_id"])
    arm = str(manifest.get("arm_id", ""))
    # GOLD EXPOSURE IS CHECKED FIRST, AND THE ORDER IS LOAD-BEARING. A ceiling arm is outside the
    # allowlist too, so whichever guard runs first is the reason that reaches the skip table.
    # Contamination and scope must stay distinguishable there: collapsing the first into the
    # second would quietly downgrade "oracle distillation" to "not in this run's arm list".
    if arm in GOLD_EXPOSED_ARMS:
        raise StateMismatch(
            f"{run_dir.name}: arm {manifest.get('arm_id')!r} chose its questions where gold is "
            "readable. Exporting it is oracle distillation wearing an experiment's clothes."
        )
    if arm not in include_arms:
        raise ArmNotAllowed(
            f"{run_dir.name}: arm {arm!r} is outside this export's allowlist "
            f"({', '.join(sorted(include_arms))}). The headline trains on the prompted policy's "
            "own filtered samples; pass --include-arm to build the +ancestor-rows ablation."
        )
    # TWO DOORS, ONE MEASUREMENT. `handle_score` is the trainer-facing one and refuses anything
    # that is not `train`; `score_heldout` is the gold-side one that only accepts a held-out run
    # and is not reachable over the wire. Both delegate to `pi_run.serve.score._measure`, so a
    # dev number and a train number carry the same `scorer_hash` and are poolable.
    req = ScoreRequest(episode_id=run_dir.name, mode="prefix_ladder")
    resp = (
        handle_score(req, runs_root=run_dir.parent, graph_version=graph_version)
        if split == "train"
        else score_heldout(
            req,
            split=split,  # type: ignore[arg-type]
            runs_root=run_dir.parent,
            graph_version=graph_version,
        )
    )
    # A run with no turns holds a decision point only if the policy stopped on the empty
    # state. Anything else with no turns (an error, a zero cap) has nothing to teach.
    if not turns and resp.stop_reason != "policy_stop":
        return []
    values = reward.turn_values(resp, weights)
    by_turn = {v.turn_idx: v for v in values}

    # THE LATENT LABELS, gold-side. `pinq_train` may never import `pi_eval` (contract 1), so
    # depth and reachability are resolved here and cross to the trainer as plain numbers.
    from pi_run.serve.score import NoGoldForTask

    graph = _graphs(suite_id, graph_version).get(task_id)
    if graph is None:  # handle_score would already have refused; belt and braces
        raise NoGoldForTask(f"{run_dir.name}: no gold graph for {suite_id}/{task_id}")
    records = _match_records(run_dir, graph, turns)
    outcome_fields = _outcome_fields(run_dir, graph)
    # The episode's FINAL required-evidence coverage. `resp.potential` is that quantity at
    # each prefix and is monotone, so the last point is the episode value; taking it from the
    # ladder we already have avoids a second, independently-driftable definition.
    outcome_fields["evidence_coverage"] = (
        float(resp.potential[-1]) if resp.potential else float("nan")
    )
    complete_at = _complete_at(resp.potential)
    # COVERAGE BEFORE EACH DECISION. `potential[i]` is required-evidence coverage before the
    # turn at position i, and `potential[len(turns)]` is what a STOP was decided on. The
    # episode-final `evidence_coverage` above cannot say whether a state was already done: it
    # labels a state done exactly when the ASK taken there is what completed it.
    ladder = [float(q) for q in resp.potential]

    def before(pos: int) -> float:
        return ladder[pos] if pos < len(ladder) else float("nan")

    # THE ANSWER-BEARING NODE'S OWN EVIDENCE, BEFORE EACH DECISION. Same shape as `ladder`,
    # narrowed from the pooled required set to one node -- see `_answer_node_ladder` and
    # `pi_eval.answer_node`. Empty when the graph names no checkable answer node, in which case
    # the field is ABSENT from every row of this run rather than False.
    answer_ladder = _answer_node_ladder(graph, cache, suite_id, task_id, turns)

    def answer_node_field(pos: int) -> dict[str, Any]:
        # NO CLAMP. `answer_ladder` holds exactly `len(turns) + 1` entries, the same shape as
        # `resp.potential`, so every decision position 0..len(turns) indexes a real one. A
        # clamp here would have turned a shape bug into a plausible boolean on the STOP row.
        if not answer_ladder:
            return {}
        return {"answer_node_covered_before": bool(answer_ladder[pos])}

    def turns_to_complete(pos: int) -> int:
        # TURNS FROM HERE UNTIL ALL REQUIRED EVIDENCE WAS IN HAND. -1 when the run never got
        # there, 0 once it already has it. Per-turn phi scores "gain now" and cannot separate
        # a question that opens the chain from one that pays the same now and dead-ends; this
        # is the signal that can.
        return -1 if complete_at is None else max(0, complete_at - pos)

    # Everything an episode stamps on each of its decision points.
    episode: dict[str, Any] = {
        "suite_id": suite_id,
        "task_id": task_id,
        "template_id": manifest.get("template_id"),
        "run_id": str(manifest["run_id"]),
        "scorer_hash": resp.scorer_hash,
        "graph_version": resp.graph_version,
        # WHICH INSTRUMENTS. matcher_id rides into matcher_hash -> scorer_hash, so a row
        # scored under mechanical_v2 can never be pooled with one scored under v3; carrying
        # it means a consumer can tell without re-deriving the chain.
        "matcher_id": _matcher_id(),
        "reward_weights_sha": weights.sha,
        # THE INSTRUMENT THAT PRODUCED THIS ROLLOUT: model pin + every prompt it rendered.
        # Two candidates at one state are only comparable on their QUESTIONS if these match;
        # see export_pairs for the 266 cross-era pairs that motivated it. Derived from the
        # manifest rather than re-read from the environment, so a re-export of an old run
        # reports what that run actually used.
        "pins_sha": _pins_sha(manifest),
        # The RUN's stamp, not the exporter's re-derivation. `split_disagreement` exists to
        # prove the two agree; carrying the stamp is what lets it.
        "split": manifest.get("split"),
        # WHICH LOOP RENDERED THIS PROMPT. The corpus spans two code cohorts: in the OLD one a
        # question was resolved against evidence gathered before its own retrieval merged, so
        # ~63% of evidence-bearing prompts carry a stale per-turn answer in their history block;
        # the FIXED loop is 3ffa972 and its descendants (`conf/cohorts/fixed_loop.json`). That is
        # a covariate shift in the PROMPT, not label noise, so the primary trains on both -- but
        # the fixed-only ablation cannot be selected if no row says which loop it came from, and
        # `code_version` rode only in `semantic_hash`, reaching no artifact.
        #
        # WHICH POLICY ASKED. `pins_sha` carries the model pin and the prompt hashes and is
        # therefore equal across arms that share them, so it cannot separate `inquirer_prompted`
        # from `self_inquire`. The allowlist refuses at the walk; this is what lets a consumer
        # audit an already-written file.
        #
        # From the MANIFEST, so a re-export of an old run reports what that run actually used.
        "code_version": manifest.get("code_version"),
        "arm_id": manifest.get("arm_id"),
        **outcome_fields,
        # THE JOIN KEY for rung 2. A candidate branch is a separate run, so without these the
        # export groups every candidate alone and produces no pair. See
        # `pinq_train.export.dataset.state_key`.
        "branch_of_run_id": manifest.get("branch_of_run_id"),
        "branch_turn_idx": manifest.get("branch_turn_idx"),
        # THE CAPS RIDE ON EVERY ROW. Two candidates at one state may now come from different
        # budget cohorts (see `candidate_specs`); the exporter refuses to pair across cohorts,
        # and a consumer can filter on this without re-joining the manifests.
        "budget_cap": manifest.get("budget_cap"),
        "max_turns": manifest.get("max_turns"),
        # HOW THE EPISODE ENDED, on every row of it. "policy_stop" is the policy's decision;
        # "budget" and "max_turns" are the harness's. A consumer that wants only episodes the
        # policy finished filters here without re-reading status.json.
        "stop_reason": resp.stop_reason,
    }

    out: list[dict[str, Any]] = []
    for i, t in enumerate(turns):
        v = by_turn.get(int(t["turn_idx"]))
        if v is None:
            continue
        out.append(
            {
                **episode,
                "turn_idx": int(t["turn_idx"]),
                "state_text": render_state(cache, suite_id, task_id, turns, i, verify=verify_state),
                "action_json": action_json(t),
                # The policy's OWN latency claim. A FIELD, never part of the target: the loss
                # mask supervises action_json's tokens, and a content hash there would train
                # the policy to memorise uids. Audited, never scored.
                "parent_uids": list(self_reported_parents(t)),
                **latent_fields(
                    graph,
                    records,
                    turn_idx=int(t["turn_idx"]),
                    task_text=_task_text(cache, suite_id, task_id),
                ),
                # RANKING key: cost-aware, w_phi*phi - w_red*rho - c_ret.
                "value": float(v.value),
                # ACCEPT key: raw phi. `margin_threshold` is tau + 1.5*sigma_J*sqrt(2), and tau
                # is "the 35th percentile of pilot phi_LOO" -- so the threshold lives in PHI
                # units and must be compared against phi, not against a weighted,
                # cost-subtracted composite. See export.dataset.
                "phi_tilde": float(v.phi_tilde),
                "rho": float(v.rho),
                # DOES THE QUESTION ALREADY CONTAIN THE ANSWER? Computed here because the
                # comparison is against `gold_answer`, which a rollout worker cannot read.
                # A leaked question retrieves the gold passage perfectly, so the reward
                # REWARDS it -- this is the one contamination the pipeline cannot detect on
                # its own. See pi_eval.metrics.quality.leaks_answer for the measurement.
                "leaks_gold_answer": _leaks(graph, t),
                "turns_to_complete": turns_to_complete(int(t["turn_idx"])),
                "is_stop": False,
                "coverage_before": before(i),
                "done_before": _done(before(i)),
                **answer_node_field(i),
            }
        )
    if resp.stop_reason == "policy_stop":
        # THE RECORDED STOP. Rendered on the evidence after the last turn and verified against
        # that turn's `subset_hash_after` -- the state the policy saw when it chose to stop.
        # A run that stopped before asking anything decided on the empty state; there is no
        # hash to verify, and nothing to drift.
        n = len(turns)
        out.append(
            {
                **episode,
                "turn_idx": n,
                "state_text": render_state(
                    cache,
                    suite_id,
                    task_id,
                    turns,
                    n,
                    verify=verify_state,
                    expect_hash=str(turns[-1].get("subset_hash_after", "")) if turns else "",
                ),
                "action_json": STOP_ACTION_JSON,
                "parent_uids": [],
                **latent_fields(
                    graph, records, turn_idx=n, task_text=_task_text(cache, suite_id, task_id)
                ),
                "value": STOP_VALUE,
                "phi_tilde": 0.0,
                "rho": 0.0,
                "leaks_gold_answer": False,
                "turns_to_complete": turns_to_complete(n),
                "is_stop": True,
                "coverage_before": before(n),
                "done_before": _done(before(n)),
                **answer_node_field(n),
            }
        )
    return out


def _answer_node_ladder(
    graph: Any,
    cache: SuiteCache,
    suite_id: str,
    task_id: str,
    turns: Sequence[Mapping[str, Any]],
) -> tuple[bool, ...]:
    """Was the ANSWER-BEARING node's evidence in hand before each decision? Empty if unknowable.

    `len(turns) + 1` entries, the same shape as `resp.potential`, built the same way: accumulate
    each turn's `retrieved_uids` and evaluate before every position. Entry `i` is the state the
    turn at position `i` was decided in and the last entry is the state a recorded STOP saw.

    THE NODE IS L1.11'S, NOT A SECOND RULE. `pi_eval.answer_node.answer_nodes_for_task` is the
    module that lane published its numbers from, moved here from `scripts/answer_node_coverage/`
    unchanged so `src/` can import it (that package now re-exports it). Re-stating the rule
    would mean the dataset's answer node and the diagnostic's answer node could drift into two
    different things under one name.

    THE TEXT COMES FROM THE UNIT POOL, not from a second read of `tasks.jsonl`. `SuiteCache.
    units` already holds `uid -> EvidenceUnit` for the task -- the identical objects
    `render_state` rebuilds the prompt from -- so the containment test for RULE 1 is run against
    the same bytes the policy was shown. The diagnostic lane brute-forced `unit_uid` over the
    public corpus tree to get there; that is the same map by a longer road, and two roads is one
    more than can be kept in agreement.

    EMPTY, NEVER ALL-FALSE, when the graph names no answer node whose evidence can be checked
    (`answer_node_uid_sets` drops evidence-free nodes). The caller omits the field entirely for
    such a run: "not checkable" is not "not covered".

    A POOL THAT CANNOT BE LOADED IS NOT FATAL HERE. `render_state` already raises
    `PoolUnavailable` for such a suite before this runs, so on the export path this cannot fire;
    a caller that skipped verification would get an unknown field rather than a dead run,
    because the answer node is an added label and no existing row depends on it.
    """
    try:
        uid_sets = cache.answer_node_uid_sets(suite_id, task_id, graph)
    except PoolUnavailable:
        return ()
    if not uid_sets:
        return ()

    from pi_eval.answer_node import answer_node_covered

    seen: set[str] = set()
    out = [answer_node_covered(seen, uid_sets)]
    for t in turns:
        seen |= set(t.get("retrieved_uids") or ())
        out.append(answer_node_covered(seen, uid_sets))
    return tuple(out)


def _leaks(graph: Any, turn: Mapping[str, Any]) -> bool:
    """True when this turn's QUESTION names the task's gold answer.

    Verbatim recall only; a paraphrase is invisible. That bounds the damage rather than
    removing it, which is why a frontier generator stays under measurement rather than being
    trusted.
    """
    from pi_eval.metrics.quality import leaks_answer

    q = str(turn.get("question") or "")
    gold = getattr(graph, "answer", "") or ""
    if not q or not gold:
        return False
    return leaks_answer(q, gold, tuple(getattr(graph, "gold_aliases", ()) or ()))


def _pins_sha(manifest: Mapping[str, Any]) -> str:
    """Stable digest of (model_pin_hash, prompt_hashes) -- the instrument a rollout ran under.

    Both already ride in `semantic_hash`, so runs that differ here already have different
    run_ids; this collapses them into one comparable field so the exporter can group by it
    without re-deriving run identity.

    Empty when the manifest carries neither, which is how a legacy run reads. Two such rows
    still pair with each other -- they are internally consistent -- but not with a stamped one.
    """
    from pinq.ids import canon, h

    model = str(manifest.get("model_pin_hash") or "")
    prompts = manifest.get("prompt_hashes") or {}
    if not model and not prompts:
        return ""
    return h("pins", canon({"model": model, "prompts": dict(sorted(prompts.items()))}))[:16]


def _matcher_id() -> str:
    """The LIVE matcher's id, never a literal.

    `tests/test_report_render.py` once hard-coded "mechanical_v2" in a fixture and every row
    vanished from T4 the day the matcher was bumped, because `report.MATCHER_ID` filters on
    whatever the matcher currently calls itself. Reading it from the class is what keeps a
    bump from silently emptying a table.
    """
    from pi_eval.matcher.base import MechanicalMatcher

    return str(MechanicalMatcher.matcher_id)


def _complete_at(potential: Sequence[float]) -> int | None:
    """Index of the TURN that first brought required-evidence coverage to 1.0, or None.

    `potential[0]` is coverage before any turn and `potential[i + 1]` is coverage after turn
    i, so the completing turn is one less than the ladder index. Coverage is monotone (see
    pi_run.serve.score), so the first crossing is the only one.
    """
    for k, q in enumerate(potential):
        if k and _done(q):
            return k - 1
    return None


def _done(coverage: float) -> bool:
    """Required-evidence coverage is complete: the ONE threshold `_complete_at` and the rows'
    `done_before` share, so "done" cannot mean two things. NaN (no gold) is not done."""
    return float(coverage) >= 1.0 - 1e-12


def _outcome_fields(run_dir: Path, graph: Any) -> dict[str, Any]:
    """Episode-level outcome, carried onto every turn row of that episode.

    WHY A TURN ROW CARRIES AN EPISODE NUMBER. The preference pairs rank candidates by per-turn
    evidence gain, which cannot see whether the episode ultimately answered the question. Over
    448 musique runs, complete evidence coexists with a wrong answer often enough that ranking
    on gain alone accepts trajectories that retrieved everything and still failed. Carrying the
    outcome lets a consumer filter or reweight on it without re-joining scores.parquet.

    `answer_hedged` rides along because hedge rate varies 16.7%-83.3% across arms and drives
    every answer-quality number with it; a consumer that filters on `answer_correct` without
    seeing it would be filtering on prompt compliance and not know.
    """
    from pi_eval.metrics import quality

    nan = float("nan")
    out = {
        "evidence_coverage": nan,
        "answer_correct": nan,
        "answer_hedged": nan,
        "binary_gold_hedged": False,
    }
    try:
        oc = _read_json(run_dir / "outcome.json")
    except Exception:
        return out
    ans = oc.get("answer")
    text = ans.get("text", "") if isinstance(ans, dict) else str(ans or "")
    if not text:
        return out
    out["answer_hedged"] = 1.0 if quality.is_hedged(text) else 0.0
    gold = getattr(graph, "answer", "") or ""
    if gold:
        out["answer_correct"] = quality.contains_answer(
            text, gold, tuple(getattr(graph, "gold_aliases", ()) or ())
        )
        # A HEDGE ON A YES/NO GOLD IS UNKNOWN, NOT CORRECT. `contains_answer` matches a
        # normalised token sequence, and strategyqa's gold is the WORD "yes"/"no" with aliases
        # ("true","yes")/("false","no") (pi_eval.build.strategyqa_build). So the refusal "There
        # is no evidence in the retrieved documents that this is the case." scores 1.0 on every
        # gold-"no" task and 0.0 on every gold-"yes" one: the same non-answer, graded by gold
        # polarity alone. `_outcome_prefers` ranks on this value at precedence 1, so those pairs
        # were ordered by which candidate happened to hedge. MEASURED on the 1,110 strategyqa
        # pairs: 250 of 508 "correct" gold-"no" labels were hedges, against 4 of 245 on
        # gold-"yes".
        #
        # HEDGE-GATED, NOT BLANKET. NaN on every binary-gold row would discard 245 legitimately
        # committed answers to kill 4 false ones; gating on the hedge discards the hedges and
        # nothing else. A committed "No, Aristotle did not use a laptop." keeps its 1.0. NaN on
        # BOTH polarities: the 0.0 a hedge gets on gold-"yes" is not a measured wrong, it is the
        # identical non-answer seen from the other side, and blanking one side only would keep
        # the label polarity-dependent -- the defect's own shape. NaN is what `_outcome_prefers`
        # already reads as unknown, so gain decides.
        #
        # NOT IN `contains_answer` OR `pi_eval.score`. The scorer's hash carries no answer term,
        # so a changed scorer would make one `scorer_hash` denote two measurements. This
        # exporter recomputes the value independently and can change it without that lie. The
        # flag rides on the row and both exporters count it (`n_binary_gold_hedged_nan`).
        binary_gold = quality.normalize(gold) in {"yes", "no", "true", "false"}
        if binary_gold and out["answer_hedged"] == 1.0:
            out["answer_correct"] = nan
            out["binary_gold_hedged"] = True
    return out


def collect_rows(
    runs_root: Path,
    root: Path,
    *,
    graph_version: str,
    weights: Any,
    verify_state: bool = True,
    on_skip: Any = None,
    include_arms: frozenset[str] = DEFAULT_TRAINABLE_ARMS,
    split: str = "train",
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Walk `runs_root` in sorted order. A run that cannot be exported is COUNTED, not hidden."""
    from pi_run.serve.score import EpisodeNotFound, NoGoldForTask, SplitRefused

    cache = SuiteCache(root)
    rows: list[dict[str, Any]] = []
    skipped: dict[str, int] = {}

    def skip(reason: str, detail: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1
        if on_skip is not None:
            on_skip(reason, detail)

    for d in sorted(p for p in runs_root.iterdir() if (p / "manifest.json").exists()):
        try:
            rows.extend(
                rows_from_run(
                    d,
                    cache,
                    graph_version=graph_version,
                    weights=weights,
                    verify_state=verify_state,
                    include_arms=include_arms,
                    split=split,
                )
            )
        except SplitRefused as exc:
            skip("split_refused", f"{d.name}: {exc}")
        except ArmNotAllowed as exc:
            skip("arm_not_allowed", f"{d.name}: {exc}")
        except NoGoldForTask as exc:
            skip("no_gold", f"{d.name}: {exc}")
        except (EpisodeNotFound, FileNotFoundError, KeyError) as exc:
            skip("incomplete_run", f"{d.name}: {type(exc).__name__}: {exc}")
        except StateMismatch as exc:
            skip("state_mismatch", f"{d.name}: {exc}")
        except PoolUnavailable as exc:
            skip("pool_unavailable", f"{d.name}: {exc}")
        except Exception as exc:  # noqa: BLE001 - an actionable line, not a traceback
            skip(type(exc).__name__, f"{d.name}: {exc}")
    return rows, skipped


# --------------------------------------------------------------------------- pi train export


def _require_gold(a: argparse.Namespace) -> int | None:
    if getattr(a, "gold_root", None):
        os.environ["PI_GOLD_ROOT"] = str(Path(a.gold_root).resolve())
    if not os.environ.get("PI_GOLD_ROOT"):
        print(
            "PI_GOLD_ROOT is unset. `pi train export` is a GOLD-SIDE command -- it attaches a "
            "measured value to every decision point -- so it must be able to read gold. Pass "
            "--gold-root or export it. (The TRAINER never does this: it asks a scoring server.)",
            file=sys.stderr,
        )
        return 2
    return None


def _rater_prefs(path: str | None) -> dict | None:
    """Load A6 majority preferences as {frozenset(winner, loser): winner}.

    Only lines carrying BOTH candidate run ids are usable: a verdict whose candidates cannot
    be named cannot be joined to a pair, and silently dropping it while counting it would
    overstate what the key ordered.
    """
    if not path:
        return None
    out: dict = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        w, lo = d.get("winner_run"), d.get("loser_run")
        if w and lo:
            out[frozenset((str(w), str(lo)))] = str(w)
    return out


def _cohort(path: str | None) -> tuple[frozenset[str], str]:
    """The FIXED-loop cohort, and the sha256 of the file that defined it.

    LOADED HERE AND PASSED DOWN. `pinq_train.export.dataset` takes the membership SET, not a
    path: membership is a git-ancestry fact computed once by `scripts/cohort_ids.py`, and the
    trainer that imports that module runs on a rented box which may hold no repository and no
    `conf/` at all. A library that read a checkout-relative path would work here and fail there.

    The sha256 is of the file's BYTES and rides onto the manifest, because
    `n_cross_code_version_dropped` is unreadable without the definition that produced it.

    A file naming NO members is refused rather than returned: an empty set is how the exporter
    spells "no cohort definition", and it falls back to comparing commits -- so accepting one
    here would restore the old behaviour under a flag that says it was replaced.
    """
    from pi_run.manifest import repo_root

    p = Path(path) if path else repo_root() / DEFAULT_COHORT_FILE
    blob = p.read_bytes()
    members = json.loads(blob).get("members") or []
    if not members:
        raise ValueError(f"{p} names no `members`")
    return frozenset(str(m) for m in members), hashlib.sha256(blob).hexdigest()


def _weights(a: argparse.Namespace) -> Any:
    reward = _train("reward")
    # `headroom_normalised` pays for the fraction of REMAINING coverage a turn captured. OFF
    # unless asked for: it changes the SCALE of phi_tilde, so `--tau` must be re-estimated
    # from headroom-normalised pilot values before an export under it is trained on.
    w = reward.RewardWeights(
        tau=a.tau,
        c_ret=a.tau,
        headroom_normalised=bool(getattr(a, "headroom_normalised", False)),
    )
    w.validate()
    return w


def split_disagreement(rows: Sequence[dict[str, Any]], split: str = "train") -> dict[str, int]:
    """Rows the RUN's stamped split calls `split` and the EXPORTER's bucket function does not.

    THE DEFECT THIS ONCE DESCRIBED IS FIXED, AND THE COUNT IS NOW STRUCTURALLY ZERO.
    This docstring used to open "THERE ARE TWO BUCKET FUNCTIONS IN THIS REPOSITORY AND THEY DO
    NOT AGREE" and quote a ~40% recall loss. There is now ONE definition: `pinq.splitting`.
    `pi_run.manifest` delegates to it and `pinq_train.split` delegates to it, so the runner's
    stamp and the exporter's re-check cannot disagree, and the WARNING below can never fire.

    Kept, not deleted, because it is cheap and it is the check that would catch a future
    re-divergence -- someone re-implementing a bucket in either module. A count that is zero
    because the invariant holds is worth keeping; a count that is zero because nobody looks is
    not, which is why this says which one it is.
    """
    bucket = _train("split")
    counts: dict[str, int] = {}
    for r in rows:
        if bucket.split_of(r["suite_id"], r["task_id"], r.get("template_id")) != split:
            counts[str(r["suite_id"])] = counts.get(str(r["suite_id"]), 0) + 1
    return counts


def cmd_train_promote(a: argparse.Namespace) -> int:
    """Re-attach every rater verdict to a pairs file. See `pi_run.promote`."""
    from pi_run.promote import AnnotationLoss, QuarantineDisagreement, promote

    try:
        blocks = promote(
            a.pairs,
            a.annotations,
            a.quarantine,
            out=a.out,
            allow_loss=a.allow_loss,
        )
    except AnnotationLoss as exc:
        print(f"REFUSED: {exc}")
        return 3
    except QuarantineDisagreement as exc:
        print(f"REFUSED: {exc}")
        return 2

    out = Path(a.out or a.pairs)
    mpath = out.with_suffix("").with_suffix(".manifest.json")
    mpath = out.parent / (out.name.replace(".jsonl", "") + ".manifest.json")
    if mpath.exists():
        man = json.loads(mpath.read_text())
        man.update(blocks)
        mpath.write_text(json.dumps(man, indent=1, sort_keys=True) + "\n")
    print(f"promoted {out}  rows={blocks['promote']['n_rows']}")
    for k in ("a7_campaign", "a6_campaign", "a5_campaign"):
        print(f"  {k}: {blocks[k]['pairs_annotated']} pairs annotated")
    print(f"  q_distinctness: {blocks['promote']['q_distinctness']}")
    return 0


def _stamp_arms(man: Any, include_arms: frozenset[str], skipped: Mapping[str, int]) -> None:
    """Record the population on the manifest.

    Set HERE and not inside `export_sft`/`export_pairs`: those take rows and have never seen a
    run directory, so they cannot know an arm was refused before a row existed. The walk that
    refused it is in this module, and the count would otherwise be visible only on a terminal.
    """
    man.included_arms = tuple(sorted(include_arms))
    man.n_arm_refused = int(skipped.get("arm_not_allowed", 0))


def cmd_train_export(a: argparse.Namespace) -> int:
    rc = _require_gold(a)
    if rc is not None:
        return rc
    from pi_run.manifest import repo_root

    dataset = _train("export.dataset")
    rung1 = _train("rung1_sft")

    root = Path(a.root).resolve() if a.root else repo_root()
    runs_root = Path(a.runs_root).resolve() if a.runs_root else root / "runs"
    if not runs_root.exists():
        print(f"no runs under {runs_root}", file=sys.stderr)
        return 1
    weights = _weights(a)
    margin = rung1.margin_threshold(a.tau, a.sigma_j)

    # `default=None` on an `append` action, resolved here: argparse appends to a MUTABLE
    # default, so a list default would accumulate across parses and half-apply the allowlist.
    include_arms = frozenset(a.include_arm) if a.include_arm else DEFAULT_TRAINABLE_ARMS
    rows, skipped = collect_rows(
        runs_root,
        root,
        graph_version=a.graph_version,
        weights=weights,
        verify_state=not a.no_verify_state,
        on_skip=(lambda why, detail: print(f"  skip [{why}] {detail}")) if a.verbose else None,
        include_arms=include_arms,
        split=a.split,
    )
    # A DEV EXPORT IS A DIFFERENT ARTIFACT AND GETS A DIFFERENT PATH AND A DIFFERENT NAME. Both,
    # not either: a subdirectory alone would leave two files called `sft.jsonl` on disk, and a
    # suffix alone would leave the dev file beside the train one in the directory every loader
    # globs. `pairs.jsonl` vs `pairs.anticipation.jsonl` already exists for this reason -- a
    # training run and a validation run reading different bytes under one path is the failure.
    out_dir = Path(a.out).resolve() if a.out else root / DEFAULT_OUT
    suffix = ""
    if a.split != "train":
        out_dir = out_dir / a.split
        suffix = f".{a.split}"
    print(f"runs {runs_root}  rows {len(rows)}  margin_threshold {margin:.4f}")
    print(f"  arms {sorted(include_arms)}  refused {skipped.get('arm_not_allowed', 0)} runs")
    n_stop = sum(1 for r in rows if r.get("is_stop"))
    n_stop_at_fork = sum(
        1
        for r in rows
        if r.get("is_stop")
        and r.get("branch_turn_idx") is not None
        and int(r["turn_idx"]) == int(r["branch_turn_idx"])
    )
    print(f"  recorded STOP rows {n_stop}  at a fork turn {n_stop_at_fork}")
    disagree = split_disagreement(rows, a.split)
    if disagree:
        print(
            f"  WARNING: {sum(disagree.values())}/{len(rows)} rows {disagree} are {a.split!r} "
            f"under pi_run.manifest.split_of (which stamped the run) and NOT {a.split!r} under "
            "pinq_train.split.bucket (which the exporter re-checks). The two hash differently. "
            "The intersection is exported, which is safe but lossy; see split_disagreement()."
        )
    if skipped:
        print("  not exported: " + ", ".join(f"{k}={v}" for k, v in sorted(skipped.items())))

    # LOADED ONCE, BEFORE EITHER EXPORT, AND FATAL IF IT CANNOT BE READ.
    #
    # WAS LOADED INSIDE THE `pairs` BRANCH, which is why `data/rl/sft.manifest.json` ships
    # `cohort_file_sha: ""` while every pairs manifest beside it carries the sha. On a pairs
    # manifest that empty value means something -- "no cohort definition was in hand, and the
    # guard fell back to comparing commits" -- so on the SFT manifest it read as a claim about
    # the export that was not true of it: the file simply had never been opened. A provenance
    # field that is empty for two different reasons cannot be read for either.
    try:
        cohort, cohort_sha = _cohort(a.cohort_file)
    except (OSError, ValueError) as exc:
        print(
            f"--cohort-file: {exc}. The cross-cohort guard has no definition to read. "
            "Point it at a file `scripts/cohort_ids.py` wrote.",
            file=sys.stderr,
        )
        return 2
    print(f"  cohort {len(cohort)} shas  cohort_file_sha={cohort_sha[:12]}")
    if a.turn0_any_cohort and a.cohort == "any":
        print(
            "--turn0-any-cohort is only meaningful with --cohort fixed or --cohort old. Under "
            "'any' every row is kept already, so the flag would decide nothing while the "
            "manifest recorded that it was asked for.",
            file=sys.stderr,
        )
        return 2
    print(f"  cohort selection {a.cohort}  turn0_any_cohort={bool(a.turn0_any_cohort)}")

    if a.kind in ("sft", "both"):
        items, man = dataset.export_sft(
            rows,
            margin_threshold=margin,
            max_examples_per_task=a.max_examples_per_task,
            split=a.split,
            cohort=cohort,
            cohort_file_sha=cohort_sha,
            cohort_mode=a.cohort,
            turn0_any_cohort=bool(a.turn0_any_cohort),
            stop_label=a.sft_stop_label,
        )
        _stamp_arms(man, include_arms, skipped)
        # A DIFFERENT LABEL IS A DIFFERENT DATASET AND GETS A DIFFERENT FILENAME. Same argument
        # as the dev suffix above and as `pairs.jsonl` vs `pairs.anticipation.jsonl`: two files
        # with the same name under one directory is how a training run and a validation run come
        # to read different bytes under one path.
        stem = "sft" if a.sft_stop_label == "done_before" else "sft.answer_node"
        path = dataset.write_jsonl(out_dir / f"{stem}{suffix}.jsonl", items, man)
        if a.stop_rule != "gold_coverage_v1":
            # SAID OUT LOUD, because with --kind both the two manifests in one directory will
            # name DIFFERENT stop rules and that looks like a bug. It is not: v2 adds a pairs
            # branch and changes nothing `export_sft` does, so stamping v2 on the SFT manifest
            # would claim those rows were derived differently than they were.
            print(
                f"      NOTE: --stop-rule {a.stop_rule} is a PAIRS rule. The SFT rows and this "
                "manifest are unchanged and still read stop_rule=gold_coverage_v1."
            )
        print(f"sft   {path}  n={man.n_examples}  suites={man.suites}")
        print(f"      train_id_set_hash={man.train_id_set_hash[:16]}  refused={man.refused}")
        print(f"      cohort={man.cohort}  n_cohort_refused={man.n_cohort_refused}")
        n_stop = sum(1 for it in items if it.is_stop)
        print(
            f"      sft_stop_label={man.sft_stop_label}  stop_rows={n_stop}"
            f"  stop_share={(n_stop / len(items) if items else float('nan')):.4f}"
        )
        print(
            f"      n_answer_node_unknown={man.n_answer_node_unknown}"
            f"  n_answer_node_state_disagree={man.n_answer_node_state_disagree}"
            f"  n_done_before_unknown={man.n_done_before_unknown}"
        )
    if a.kind in ("pairs", "both"):
        # The cohort definition is loaded ABOVE, before either export: carrying on without one
        # would silently fall back to comparing COMMITS, which is the behaviour --cohort-file
        # exists to replace -- 448 refused pairs on the 2026-09-12 control export.
        pairs, man = dataset.export_pairs(
            rows,
            margin_threshold=margin,
            len_delta_max=a.len_delta_max,
            rank=a.rank,
            rater_prefs=_rater_prefs(a.rater_prefs),
            split=a.split,
            cohort=cohort,
            cohort_file_sha=cohort_sha,
            cohort_mode=a.cohort,
            turn0_any_cohort=bool(a.turn0_any_cohort),
            stop_rule=a.stop_rule,
        )
        _stamp_arms(man, include_arms, skipped)
        # TWO ORDERINGS, TWO FILENAMES. `pairs.jsonl` is the CONTROL (rank_rule="outcome"),
        # the only export the latent-pursuit check may be quoted on; ranking on anticipation
        # makes that check true by construction. Writing both to one name would let a training
        # run and a validation run read different bytes under one path.
        name = f"pairs{suffix}.jsonl" if a.rank == "outcome" else f"pairs.{a.rank}{suffix}.jsonl"
        path = dataset.write_jsonl(out_dir / name, pairs, man)
        print(f"pairs {path}  n={man.n_pairs}  suites={man.suites}")
        print(f"      cohort={man.cohort}  n_cohort_refused={man.n_cohort_refused}")
        print(
            f"      stop_rule={man.stop_rule}  synth_done={man.n_stop_pairs_synth}  "
            f"synth_notdone={man.n_stop_pairs_synth_notdone}  "
            f"notdone_states_no_ask={man.n_notdone_states_no_ask}"
        )
        if man.n_pairs == 0:
            print(
                "      0 pairs: a preference pair needs TWO candidates at one state. Recorded "
                "runs hold one action per state, so pairs come from rung 1's candidate "
                "sampling (N=8 per state), not from a finished sweep."
            )
    return 0


# --------------------------------------------------------------------------- pi train rung0


# --- track C: verify-prefs ---
def cmd_train_verify_prefs(a: argparse.Namespace) -> int:
    """Does the preference file on disk still produce the digest the pairs manifest names?

    `rater_prefs_sha` is the only field distinguishing two `rank_rule="rater"` artifacts -- A6
    candidate rankings against A7's `reaches` axis -- and it proves an ordering came from a named
    preference set only while somebody checks. The files under `data/rl/` have been rewritten by
    promotion campaigns since the exports ran (one dropped 2,188 of 2,429 preferences) and nothing
    in the tree would have said so.

    Exit 2 on mismatch, not 1: a CI job must be able to tell "the labels moved under the dataset"
    apart from "something broke". READ-ONLY; it never rewrites either file.
    """
    man_path = Path(a.pairs_manifest)
    man = json.loads(man_path.read_text())
    want = str(man.get("rater_prefs_sha") or "")
    prefs = _rater_prefs(a.prefs) or {}
    got = _train("export.dataset").rater_prefs_sha(prefs)

    print(f"pairs manifest  {man_path}")
    print(f"  rank_rule     {man.get('rank_rule', '?')}  n_pairs {man.get('n_pairs', '?')}")
    print(f"  recorded sha  {want or '(none)'}")
    print(f"prefs           {a.prefs}")
    print(f"  {len(prefs)} verdicts read  ->  sha {got}")
    if not want:
        print(
            "MISMATCH: this manifest names no preference set (rater_prefs_sha is empty), so it "
            "was not ordered by a rater key. Comparing against an empty digest would let every "
            "file 'verify' against the control."
        )
        return 2
    if want != got:
        print(
            f"MISMATCH: the export was ordered by preference set {want}, the file on disk is "
            f"{got}. The pairs file's ordering cannot be reproduced from this file; find the "
            "preference set the export actually used before quoting anything ranked by it."
        )
        return 2
    print(f"MATCH: {got} -- this file is the preference set that ordered that export.")
    return 0


# --- end track C: verify-prefs ---


def cmd_train_rung0(a: argparse.Namespace) -> int:
    from pi_run.manifest import repo_root
    from pinq import promptlib

    gepa = _train("rung0_gepa")
    root = Path(a.root).resolve() if a.root else repo_root()
    cfg = gepa.Rung0Config(
        suites=tuple(a.suite) if a.suite else ("musique", "wiki2"),
        n_tasks=a.n_tasks,
        n_val_tasks=a.n_val_tasks,
        n_candidates=a.candidates,
        n_generations=a.generations,
        minibatch=a.minibatch,
        seed=a.seed,
        policy_model=a.policy_model or "",
        max_prompt_tokens=(
            None
            if a.max_prompt_tokens == "none"
            else "family"
            if a.max_prompt_tokens == "family"
            else int(a.max_prompt_tokens)
        ),
    )
    # BEFORE anything is spent. A search that discovers its own price after eight hours has
    # already spent it.
    print(cfg.projected_cost(root).render())
    print(f"  config_sha        {cfg.sha[:16]}")
    print(f"  seed prompt       {INQUIRER_PROMPT}.txt  sha {promptlib.sha(INQUIRER_PROMPT)[:16]}")
    if a.dry_run:
        print("\ndry run: nothing was rolled out and nothing was spent.")
        return 0
    if not a.yes:
        print(
            "\nrefusing to start without --yes. Re-run with --yes once the projected cost "
            "above is acceptable.",
            file=sys.stderr,
        )
        return 2
    if not a.task_ids:
        print(
            "--task-ids is required to run: rung 0 scores candidates on a FIXED task set, and "
            "a set chosen afresh each generation makes the Pareto front incomparable between "
            "them. Pass a file with one `suite_id/task_id` per line.",
            file=sys.stderr,
        )
        return 2
    keys = _read_task_keys(gepa, Path(a.task_ids))
    # TWO SERVERS, because one cannot serve both endpoints: a gold-set process REFUSES
    # /rollout (the firewall is re-asserted there and is "not catchable into a warning") and a
    # gold-unset process 503s on /score. `--rollout-url` defaults to `--score-url`, so a
    # single-server invocation is unchanged.
    client = _train("client").SeamClient(a.score_url, rollout_url=getattr(a, "rollout_url", None))
    evaluator = gepa.SeamEvaluator(
        client=client, cfg=cfg, overlay_root=Path(a.overlay_root or (root / "cache" / "rung0"))
    )
    # NORMALISED, because both defaults were wrong against a litellm proxy and failed AFTER
    # the seed generation had been paid for: LITELLM_BASE_URL is the proxy ROOT (its
    # OpenAI-compatible routes live under /v1, so the raw root gave HTTP 403), and
    # PI_MODEL_INQUIRER carries litellm's `openai/` PROVIDER prefix, which a direct POST to the
    # proxy rejects. An explicit --reflector-base-url / --reflector-model is normalised too,
    # since the same two mistakes are just as easy to make by hand.
    reflector = gepa.OpenAIReflector(
        base_url=gepa.openai_base(a.reflector_base_url or os.environ.get("LITELLM_BASE_URL", "")),
        model=gepa.proxy_model_name(a.reflector_model or os.environ.get("PI_MODEL_INQUIRER", "")),
        api_key=os.environ.get("LITELLM_API_KEY", "local"),
    )
    result = gepa.run_search(
        cfg,
        # Survives a stop: rollouts already resume, and this carries the SEARCH state too.
        checkpoint_dir=a.out,
        seed_prompt=promptlib.load(INQUIRER_PROMPT),
        tasks=_slice_per_suite(keys, cfg.suites, 0, cfg.n_tasks),
        # SLICED to n_val_tasks, not "whatever is left". `Rung0Config.n_rollouts` prices
        # n_val_tasks held-out passes and the pre-spend estimate is printed from it, so taking
        # the entire remainder made the quoted cost wrong by however many task ids the file
        # happened to carry.
        val_tasks=_slice_per_suite(keys, cfg.suites, cfg.n_tasks, cfg.n_val_tasks),
        evaluate=evaluator,
        reflect=reflector,
    )
    out = gepa.write_winner(a.out or (root / "conf" / "prompts" / "rung0"), cfg, result)
    print(
        f"winner {result.winner.cid[:12]}  mean {result.winner_mean:.4f} "
        f"(baseline {result.baseline_mean:.4f})  -> {out}"
    )
    return 0


class EmptyTaskIds(RuntimeError):
    """No task ids were selected, so there is nothing to score a candidate prompt on."""


def select_train_ids(
    suite_id: str,
    task_ids: Sequence[str],
    *,
    template_of: Any,
    limit: int | None = None,
) -> list[str]:
    """The TRAIN-split ids of a suite, in the suite's own order.

    TRAIN ONLY, and this is the point of the command. Rung 0 selects a prompt by SCORING
    candidates, so every task it touches is burned for evaluation -- and unlike a pilot,
    nothing downstream records that it was: there is no `rung0_flag` on a run and no
    equivalent of `burned_pilot_ids` in the seal. Restricting the input is the only place
    that contamination can be prevented.

    A PREFIX, never a sample. The same `--limit` must select the same tasks on every
    machine, or two rung-0 searches are not comparable and neither is reproducible.
    """
    from pinq.splitting import split_of

    picked = [t for t in task_ids if split_of(suite_id, t, template_of(t)) == "train"]
    return picked[:limit] if limit is not None else picked


def write_task_ids(
    path: Path,
    pairs: Iterable[tuple[str, str]],
    *,
    split: str = "train",
) -> int:
    """Write `suite_id/task_id` lines for `pi train rung0 --task-ids`. Returns the count.

    Deduplicated and sorted: a repeated id would weight one task twice in a candidate's
    score, and an order that depended on the caller would make two identical searches
    incomparable.

    REFUSES on empty. An empty file makes rung 0 score every candidate on nothing, find them
    all equal, and report a winner anyway.
    """
    rows = sorted({(str(su), str(ti)) for su, ti in pairs})
    if not rows:
        raise EmptyTaskIds(
            f"no {split}-split task ids selected, so rung 0 would score every candidate "
            "prompt on an empty task set and still report a winner."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    header = f"# {len(rows)} {split}-split task ids, written by `pi train task-ids`\n"
    path.write_text(header + "".join(f"{su}/{ti}\n" for su, ti in rows))
    return len(rows)


def _slice_per_suite(keys: Sequence[Any], suites: Sequence[str], offset: int, n: int) -> list[Any]:
    """`n` task keys from EACH suite, starting `offset` into that suite's own run.

    NOT `keys[offset * len(suites) : (offset + n) * len(suites)]`, which is what rung 0 did.
    That is `n` per suite only if the file interleaves suites, and `write_task_ids` sorts by
    (suite, task) -- so it groups them. The 2026-09-02 search asked for 12 musique + 12 wiki2
    and got 24 musique and 0 wiki2, while recording both suites in its manifest.

    Refuses a shortfall rather than returning fewer: a candidate scored on 8 tasks and one
    scored on 12 are not comparable, and the mean hides it.
    """
    out: list[Any] = []
    for suite in suites:
        of_suite = [k for k in keys if k.suite_id == suite]
        window = of_suite[offset : offset + n]
        if len(window) < n:
            raise ValueError(
                f"{suite}: asked for {n} task ids at offset {offset} but only "
                f"{max(0, len(of_suite) - offset)} remain of {len(of_suite)}"
            )
        out.extend(window)
    return out


def _read_task_keys(gepa: Any, path: Path) -> list[Any]:
    keys = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        suite, _, task = line.partition("/")
        keys.append(gepa.TaskKey(suite_id=suite, task_id=task))
    return keys


# --- track A: tokstats ---


def cmd_train_tokstats(a: argparse.Namespace) -> int:
    """Real-token percentiles over an exported dataset: the measurement that sets `max_seq_len`.

    The counting lives in `pinq_train.rung1_sft.tokstats` and is reached BY NAME, like every other
    trainer call in this file: `pi_run` may not import `pinq_train` (import-linter contract 4), and
    grimp sees function-level imports too.
    """
    ts = _train("rung1_sft.tokstats")
    try:
        tok = ts.load_tokenizer(a.tokenizer)
        st = ts.stats_from_jsonl(a.dataset, tok, field=a.field)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"tokstats failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(st.render())
    print(json.dumps({"field": a.field, **st.as_dict()}, indent=2, sort_keys=True))
    return 0


# --- track A: tokstats ---


# --------------------------------------------------------------------------- rung1 / rung2


# --- track F: cli ---
# THE DEFAULTS LIVE HERE, ONCE, rather than in `add_argument(default=...)`, because every flag
# they back has to be able to say "not given" -- that is what lets `--config` fill it in and an
# explicit flag override it. Resolution order is FLAG > --config file > this dict.
#
# A knob that argparse does not expose is a knob that gets set in a python REPL, and a number
# typed into a REPL is provenance nothing records. `pi train rung1 --max-seq-len` was the only
# shape knob either rung had, so the 8B grid could not be written on a command line at all.
#
# EVERY PLAIN HYPERPARAMETER BELOW IS READ FROM `SFTConfig`/`LoraSpec` THEMSELVES, not retyped:
# a literal here that happens to equal the dataclass's field default is a second place the same
# number can go stale in. `reasoning_effort` and `stop_weight` are the exception and stay literal
# `None`: their authority is `SHA_OMIT_WHEN_NONE` (and, for `reasoning_effort`,
# byte-identical-to-serving), independent of whatever `SFTConfig` itself defaults to.
# `curriculum` IS read through: `SHA_OMIT_WHEN_DEFAULT = {"curriculum": SHUFFLED}` is anchored to
# the very constant that is `SFTConfig.curriculum`'s own field default, so the two are the same
# authority already and must not be allowed to drift apart.
def _rung1_defaults(rung1: Any) -> dict[str, Any]:
    sft = {f.name: f.default for f in fields(rung1.SFTConfig)}
    lora = {f.name: f.default for f in fields(rung1.LoraSpec)}
    return {
        "epochs": sft["epochs"],
        "seed": sft["seed"],
        "max_seq_len": sft["max_seq_len"],
        "learning_rate": sft["learning_rate"],
        "per_device_batch": sft["per_device_batch"],
        "grad_accum": sft["grad_accum"],
        "lora_r": lora["r"],
        "bf16": sft["bf16"],
        "chat_template": sft["chat_template"],
        # HARMONY ONLY, AND None IS A REAL VALUE HERE. `None` means "whatever the model's template
        # defaults to" -- which is what the server renders when the gateway sends no
        # `chat_template_kwargs`, so it is the only setting that is byte-identical to serving by
        # default. Listed with a `None` fallback so `--config` accepts the key: a field that is in
        # `cfg.sha` and that no grid file can set is this function's own failure mode, one layer up.
        # NOT read from `SFTConfig.reasoning_effort`'s field default -- see the module note above.
        "reasoning_effort": None,
        # --- stop weight ---
        # ABSENT, not 1.0, and that distinction is the whole of `SHA_OMIT_WHEN_NONE`: a rung-1 run
        # trained before this knob existed must keep its recorded `config_sha`. NOT read from the
        # dataclass default, for the same reason as `reasoning_effort`.
        "stop_weight": None,
        # --- end stop weight ---
        # --- curriculum ---
        # "shuffled" IS A VALUE, not an absence: it names the null arm -- every epoch fully
        # shuffled -- which is what every rung-1 run of record trained under, and
        # `SHA_OMIT_WHEN_DEFAULT` keeps it out of `cfg.sha` for exactly that reason.
        "curriculum": sft["curriculum"],
        # --- end curriculum ---
        # --- resume ---
        "resume": sft["resume"],
        "save_steps": sft["save_steps"],
        # --- end resume ---
    }


# --- track F: cli --- (see the rung-1 note above; read from `DPOConfig` the same way)
# `include_pair_kinds` and `loss_type` stay literal `None`: here `None` means "defer to
# `rung2.SAMPLED_PAIR_KINDS`/`rung2.DEFAULT_LOSS_TYPE` at cfg-construction time" (see
# `cmd_train_rung2` below) -- a different value from `DPOConfig`'s own field default, so reading
# them from the dataclass would be reading the wrong authority, not the same one more safely.
def _rung2_defaults(rung2: Any) -> dict[str, Any]:
    d = {f.name: f.default for f in fields(rung2.DPOConfig)}
    return {
        "epochs": d["epochs"],
        "seed": d["seed"],
        "max_seq_len": d["max_seq_len"],
        "max_prompt_len": d["max_prompt_len"],
        "learning_rate": d["learning_rate"],
        "per_device_batch": d["per_device_batch"],
        "grad_accum": d["grad_accum"],
        "beta": d["beta"],
        "bf16": d["bf16"],
        "include_pair_kinds": None,  # None means "the config's own default", not "no kinds"
        "reference": d["reference"],
        # --- objective ---
        # WHAT THE LOSS IS, which until now no command line could say. `None` means "the config's
        # own default" -- `DEFAULT_LOSS_TYPE`, i.e. plain DPO -- and not "no loss"; the two optional
        # knobs are `None` because ABSENT is a real value for them (they are omitted from `cfg.sha`
        # at it, so a run trained before they existed keeps its identity).
        "loss_type": None,
        "loss_weights": None,
        "ld_alpha": None,
        # --- end objective ---
        # --- pair floors ---
        # Both have been `DPOConfig` fields, and therefore in `cfg.sha`, since they were written,
        # and nothing could set either from a command line: an ablation on them had to be typed into
        # a python REPL, which is provenance nobody records. Read from the dataclass's own defaults
        # ("" applies no diversity filter, 0.0 refuses no STOP share).
        "min_q_distinctness": d["min_q_distinctness"],
        "min_stop_chosen_share": d["min_stop_chosen_share"],
        # --- end pair floors ---
        # --- gain key ---
        # WHICH DECIDER ORDERED THE PAIR. A `DPOConfig` field and in `cfg.sha` since it was
        # written, and settable only by `--exclude-decided-by` until now -- which made A11's
        # `dpo_outcome_only` arm (E48) unlaunchable, for the reason the block below names:
        # `run_experiment.sh` builds a fixed argument list from the experiment row and has no
        # flag channel, so the arm could be run by hand on a laptop and not by a queued node.
        # Read from the dataclass (`()` -- exclude nobody), so a config built without the key
        # hashes exactly what every finished rung-2 run hashed; it is NOT in
        # `SHA_OMIT_WHEN_NONE`, so `()` is a hashed value and `None`/`""` here would rename
        # every one of them.
        "exclude_decided_by": d["exclude_decided_by"],
        # --- end gain key ---
        # --- matched n ---
        # Literal `None` for the same reason as `loss_weights`: ABSENT is a real value here and
        # its authority is `SHA_OMIT_WHEN_NONE`, not whatever `DPOConfig` happens to default to.
        # Listed here, and not only as flags, because `scripts/hpc/run_experiment.sh` invokes
        # this command with `--config` and a fixed argument list it builds from the experiment
        # row -- a knob no config file can set is a knob no queued experiment can use.
        "subsample_pairs": None,
        "subsample_seed": None,
        # --- end matched n ---
        # --- resume ---
        "resume": d["resume"],
        "save_steps": d["save_steps"],
        # --- end resume ---
    }


def _resolve_options(a: argparse.Namespace, defaults: dict[str, Any]) -> dict[str, Any]:
    """FLAG > `--config` file > `defaults`, with every key named in one of the three.

    A key the parser does not know is REFUSED rather than ignored: a typo in a grid file would
    otherwise train at the default under a `cfg.sha` that names the value nobody applied --
    the same failure as a flag nothing forwards, one layer up. Keys starting with `_` are
    documentation (`_why_max_seq_len` says where the number came from) and are skipped, so the
    file stays readable without having to be stripped before it can be used.
    """
    conf: dict[str, Any] = {}
    path = getattr(a, "config", None)
    if path:
        raw = _read_json(Path(path))
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: expected a JSON object of hyperparameters")
        unknown = sorted(k for k in raw if not k.startswith("_") and k not in defaults)
        if unknown:
            raise ValueError(
                f"{path} sets {unknown}, which this command has no flag for. Expected a subset "
                f"of {sorted(defaults)}. A key nobody consumes is a value that silently does "
                "not apply, under a config sha that says it did."
            )
        conf = {k: v for k, v in raw.items() if not k.startswith("_")}
    out: dict[str, Any] = {}
    for key, fallback in defaults.items():
        given = getattr(a, key, None)
        out[key] = given if given not in (None, [], ()) else conf.get(key, fallback)
    return out


# --- end track F: cli ---


# --- objective ---
def _loss_weights(value: Any) -> tuple[float, ...] | None:
    """`--loss-weights 1.0,0.5` and the same key in a `--config` file, to one tuple.

    BOTH SPELLINGS, because `_resolve_options` returns whichever of the two won and the config
    must not be able to tell which: a grid file writes a JSON list, a command line writes a
    comma-separated string, and a knob that accepts one but not the other is a knob half the
    sweeps cannot express. Converted here rather than by argparse's `type=`, so the refusal
    comes out of `cmd_train_rung2`'s one PREFLIGHT FAILED path instead of two.

    The LENGTH is not checked here -- `DPOConfig.validate()` owns that, because the number it
    has to match is a field of the config and not of the command line.
    """
    if value is None or value == "":
        return None
    parts = [p.strip() for p in value.split(",")] if isinstance(value, str) else list(value)
    try:
        return tuple(float(p) for p in parts if p != "")
    except (TypeError, ValueError):
        raise ValueError(
            f"loss_weights={value!r}: expected comma-separated floats, one per loss type "
            "(e.g. --loss-weights 1.0,0.5), or a JSON list of numbers in a --config file."
        ) from None


# --- end objective ---


# --- seed ---
def _add_seed_flag(p: argparse.ArgumentParser) -> None:
    """`--seed`, identical on both rungs and in both DEFAULTS maps.

    ONE FUNCTION FOR THE TWO RUNGS, like `_add_resume_flags`, because the answer is the same on
    both: a seed is a property of the sweep, not of the objective. Both dataclasses have carried
    `seed: int = 0` since they were written, both forward it to the HF/TRL arguments and both
    have it in `cfg.sha` -- and nothing could set it, so three "seeds" of an arm would have been
    three runs of seed 0 under one `config_sha`: one run reported three times.

    `default=None`, like every other knob in this block, so "not given" stays distinguishable
    from "given the default" and `--config` can fill it.
    """
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="the training seed (default 0), forwarded to the HF/TRL arguments as `seed`. IN "
        "cfg.sha: an arm run at three seeds is three checkpoints, and they have to be "
        "distinguishable by their config alone.",
    )


# --- end seed ---


# --- resume ---
def _add_resume_flags(p: argparse.ArgumentParser, out: str) -> None:
    """`--resume/--no-resume` and `--save-steps`, identical on both rungs.

    ONE FUNCTION FOR THE TWO RUNGS because the answer must be the same on both: a node is
    preempted, not a rung. `default=None` on both, like every other knob in the track-F block,
    so "not given" stays distinguishable from "given the default" and `--config` can fill it.
    """
    p.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=f"resume from the highest-numbered checkpoint-N/ under --out (default {out}) that "
        "carries a trainer_state.json. ON by default: cluster jobs are preempted and a "
        "resubmission that restarts from step 0 pays for the whole run again while its loss "
        "curve looks exactly like a first run's. --no-resume starts fresh, which is what a "
        "changed dataset or changed hyperparameters need. NOT in cfg.sha -- a resumed run and "
        "the run it continues are one experiment; the manifest records `resumed_from`.",
    )
    p.add_argument(
        "--save-steps",
        type=int,
        default=None,
        help="optimiser steps between checkpoints (default 200, about 3,200 rows at the "
        "shipped effective batch of 16). This bounds what a preemption costs. IN cfg.sha: it "
        "changes what --out contains.",
    )


# --- end resume ---


def cmd_train_rung1(a: argparse.Namespace) -> int:
    rung1 = _train("rung1_sft")
    try:
        opt = _resolve_options(a, _rung1_defaults(rung1))
    except (ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"PREFLIGHT FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    cfg = rung1.SFTConfig(
        base_model=a.base_model or "",
        dataset=a.dataset,
        out_dir=a.out,
        tau=a.tau,
        sigma_j=a.sigma_j,
        epochs=opt["epochs"],
        max_seq_len=opt["max_seq_len"],
        learning_rate=opt["learning_rate"],
        per_device_batch=opt["per_device_batch"],
        grad_accum=opt["grad_accum"],
        seed=opt["seed"],
        # alpha = 2r is a COUPLING, not a second knob: decoupling them means a rank sweep also
        # silently sweeps the adapter's effective learning rate. `LoraSpec.validate` refuses it.
        lora=rung1.LoraSpec(r=opt["lora_r"], alpha=2 * opt["lora_r"]),
        bf16=opt["bf16"],
        chat_template=opt["chat_template"],
        reasoning_effort=opt["reasoning_effort"],
        # --- stop weight ---
        stop_weight=opt["stop_weight"],
        # --- end stop weight ---
        # --- curriculum ---
        curriculum=opt["curriculum"],
        # --- end curriculum ---
        # --- resume ---
        resume=opt["resume"],
        save_steps=opt["save_steps"],
        # --- end resume ---
    )
    try:
        rows = rung1.read_jsonl(cfg.dataset)
        report = rung1.preflight(cfg, rows)
    # --- track A: heldout except ---
    # `HeldOutDataset` is a RuntimeError like `ModeCollapse`, so without it here a dev dataset
    # exits on a traceback instead of the one line that says what went wrong.
    except (
        FileNotFoundError,
        ValueError,
        rung1.ModeCollapse,
        rung1.HeldOutDataset,
    ) as exc:
        print(f"PREFLIGHT FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    # --- track A: heldout except ---
    print(json.dumps(report, indent=2, sort_keys=True))
    if not a.train:
        print(
            "\npreflight only. --train needs a CUDA device; this project has none "
            "(Apple M1 Max), so no run of the trainer has ever completed here."
        )
        return 0
    out = rung1.train(cfg, acknowledge_untested=a.acknowledge_untested)
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


def _merged_base(path: str) -> tuple[str, str]:
    """Resolve `--merged-base DIR` to (base_model, merged_base_sha).

    The sha is READ from the merge's own manifest rather than typed on the command line: a
    provenance field an operator retypes is a provenance field that drifts, and the one thing
    it has to survive is the day someone re-merges and reuses yesterday's flag.
    """
    man = Path(path) / "merge.manifest.json"
    if not man.is_file():
        raise FileNotFoundError(
            f"{man} does not exist, so this directory cannot say which base and adapter "
            "produced it. Build it with `pi train merge --base-model ... --adapter ... "
            f"--out {path}`."
        )
    return str(path), str(_read_json(man).get("output_sha") or "")


def cmd_train_rung2(a: argparse.Namespace) -> int:
    rung2 = _train("rung2_dpo")
    try:
        opt = _resolve_options(a, _rung2_defaults(rung2))
        loss_weights = _loss_weights(opt["loss_weights"])
    except (ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"PREFLIGHT FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    # `--merged-base` is the MERGED path's flag and nothing else. Under the other two modes
    # nothing is merged, and the consequence differs: under 'adapter' the directory would be
    # silently ignored while the operator believed it had selected the reference policy; under
    # 'base' it RESOLVES INTO base_model, so the run would load rung 1's merged weights as its
    # starting policy under the name of the arm defined by not having one.
    if a.merged_base and opt["reference"] != "merged":
        # WHY, per mode. The two refusals are the same rule and NOT the same mistake: under
        # 'adapter' the directory would be read and ignored, under 'base' it would be LOADED as
        # the base -- i.e. the merged arm running under the ablation's name and cfg.sha.
        why = {
            "adapter": "Under 'adapter' the reference policy is rung 1's LoRA loaded frozen "
            "beside the trainable copy and nothing is merged; pass --adapter artifacts/rung1 "
            "instead.",
            "base": "Under 'base' nothing is merged and nothing is loaded on top: pi_ref IS "
            "the untrained base, which is the whole of what the dpo_from_base ablation "
            "measures. A merge output named here resolves INTO base_model, so the run would "
            "start from rung 1's weights under the name of the arm that has no rung 1 in it; "
            "pass --base-model instead.",
        }[opt["reference"]]
        print(
            f"PREFLIGHT FAILED: --merged-base is only meaningful with --reference merged "
            f"(got --reference {opt['reference']}). {why}",
            file=sys.stderr,
        )
        return 1
    try:
        base_model, merged_sha = (
            _merged_base(a.merged_base) if a.merged_base else (a.base_model or "", "")
        )
    except FileNotFoundError as exc:
        print(f"PREFLIGHT FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    cfg = rung2.DPOConfig(
        base_model=base_model,
        adapter=a.adapter or "",
        pairs=a.pairs,
        out_dir=a.out,
        beta=opt["beta"],
        epochs=opt["epochs"],
        max_seq_len=opt["max_seq_len"],
        max_prompt_len=opt["max_prompt_len"],
        learning_rate=opt["learning_rate"],
        per_device_batch=opt["per_device_batch"],
        grad_accum=opt["grad_accum"],
        seed=opt["seed"],
        bf16=opt["bf16"],
        include_pair_kinds=(
            tuple(opt["include_pair_kinds"])
            if opt["include_pair_kinds"]
            else rung2.SAMPLED_PAIR_KINDS
        ),
        reference=opt["reference"],
        # A merge HAPPENED only if a merge output was named. `--reference merged` without
        # `--merged-base` is the "merged elsewhere, or a full-weight fine-tune" reading, where
        # base_model is already the policy and there is no merge manifest to reconcile.
        merge_adapter=opt["reference"] == "merged" and bool(a.merged_base),
        merged_base_sha=merged_sha,
        label_smoothing=a.label_smoothing,
        include_label_sources=tuple(a.include_label_source) or rung2.LABEL_SOURCES,
        exclude_decided_by=tuple(opt["exclude_decided_by"]),
        include_code_versions=tuple(a.include_code_version),
        # --- objective ---
        # Empty means "the config's own default", as with include_pair_kinds: the default is
        # named ONCE, in the dataclass, and read from there rather than retyped here -- a
        # second spelling of a default is a second thing to keep in step with `cfg.sha`.
        loss_type=tuple(opt["loss_type"]) if opt["loss_type"] else rung2.DEFAULT_LOSS_TYPE,
        loss_weights=loss_weights,
        ld_alpha=opt["ld_alpha"],
        # --- end objective ---
        # --- pair floors ---
        min_q_distinctness=opt["min_q_distinctness"],
        min_stop_chosen_share=opt["min_stop_chosen_share"],
        # --- end pair floors ---
        # --- matched n ---
        subsample_pairs=opt["subsample_pairs"],
        subsample_seed=opt["subsample_seed"],
        # --- end matched n ---
        # --- resume ---
        resume=opt["resume"],
        save_steps=opt["save_steps"],
        # --- end resume ---
    )
    try:
        # EVERY filter the trainer will apply, through the one function that maps a config to
        # its rows. This call used to omit `include_label_sources`, so preflight reported a
        # row set the trainer never saw -- one cfg.sha over two datasets.
        rows, drops = rung2.rows_for(cfg)
        report = rung2.preflight(cfg, rows, drops=drops)
    except (
        FileNotFoundError,
        ValueError,
        rung2.NoPairs,
        rung2.TooFewStopPairs,
        # A target above what the filters left. On this path, not a traceback: the operator
        # reading a queue log has to see WHICH filter took the rows, not a stack.
        rung2.TooFewPairsToSubsample,
        rung2.HeldOutDataset,
    ) as exc:
        print(f"PREFLIGHT FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    if not a.train:
        print("\npreflight only. --train needs a CUDA device; this project has none.")
        return 0
    # The rows go WITH the config: the trainer optimises the list preflight just reported,
    # not a second read of the file.
    out = rung2.train(cfg, acknowledge_untested=a.acknowledge_untested, rows=rows, drops=drops)
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


# --- track B: merge ---
def cmd_train_merge(a: argparse.Namespace) -> int:
    """Fold rung 1's LoRA into the base weights and print the sha rung 2 must declare.

    Loaded by name like every other trainer entry point here (contract 4 -- see this module's
    docstring). This is the only command in the file that needs the `[train]` extra installed.
    """
    merge = _train("merge")
    try:
        sha = merge.merge_adapter(a.base_model, a.adapter, a.out)
    except (FileNotFoundError, merge.NothingMerged) as exc:
        print(f"MERGE FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"out_dir": a.out, "merged_base_sha": sha}, indent=2, sort_keys=True))
    print(f"\npass it to rung 2 as:  pi train rung2 --merged-base {a.out} --adapter {a.adapter}")
    return 0


# --- end track B: merge ---


# --------------------------------------------------------------------------- pi train status


def _importable(name: str) -> bool:
    from importlib.util import find_spec

    try:
        return find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _split_inventory(runs_root: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not runs_root.exists():
        return counts
    for d in sorted(p for p in runs_root.iterdir() if (p / "manifest.json").exists()):
        try:
            m = _read_json(d / "manifest.json")
        except (OSError, json.JSONDecodeError):
            continue
        suite, split = str(m.get("suite_id", "?")), str(m.get("split", "?"))
        key = f"{suite}/{split}" + ("/gold_exposed" if m.get("gold_exposed") else "")
        counts[key] = counts.get(key, 0) + 1
    return counts


def cmd_train_task_ids(a: argparse.Namespace) -> int:
    """Write the train-split id file `pi train rung0 --task-ids` requires.

    Nothing produced this file, and rung 0 refuses without it -- so the rung that optimises
    the Inquirer prompt could not be run at all. `conf/prompts/rung0/` is absent for the
    same reason, which means no arm runs an optimised prompt today.
    """
    from pi_run.cli import _resolve_corpus, _template_of
    from pi_run.manifest import repo_root
    from pi_run.worker import load_suite

    root = Path(a.root).resolve() if a.root else repo_root()
    pairs: list[tuple[str, str]] = []
    for suite_id in a.suite or ["musique"]:
        try:
            suite = load_suite(suite_id, str(_resolve_corpus(root, suite_id, None)))
        except Exception as exc:
            print(f"cannot load suite {suite_id!r}: {exc}", file=sys.stderr)
            return 1
        ids = select_train_ids(
            suite_id,
            list(suite.task_ids()),
            template_of=lambda t, _s=suite: _template_of(_s, t),
            limit=a.limit,
        )
        print(f"  {suite_id}: {len(ids)} train-split ids")
        pairs.extend((suite_id, t) for t in ids)

    out = Path(a.out).resolve() if a.out else root / "data" / "rl" / "rung0_tasks.txt"
    try:
        n = write_task_ids(out, pairs)
    except EmptyTaskIds as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {n} ids -> {out}")
    return 0


def cmd_train_status(a: argparse.Namespace) -> int:
    from pi_run.manifest import repo_root

    root = Path(a.root).resolve() if a.root else repo_root()
    runs_root = Path(a.runs_root).resolve() if a.runs_root else root / "runs"
    split = _train("split")
    rung3 = _train("rung3_grpo")

    print("ladder")
    for name, need, note in (
        ("rung0  GEPA prompt search", (), "SHIPS. No GPU: rollouts and scores go over the seam."),
        ("rung1  rejection-sampling SFT", ("torch", "peft", "transformers"), "LoRA, masked loss."),
        ("rung2  DPO on same-state pairs", ("torch", "trl", "peft"), "Reuses rung 1's candidates."),
        (
            "rung3  GRPO",
            ("torch", "vllm"),
            f"OPTIONAL, UNVALIDATED, hard kill {rung3.KILL_DATE.isoformat()}. train() raises.",
        ),
    ):
        missing = [m for m in need if not _importable(m)]
        state = "ready" if not missing else f"deps missing: {', '.join(missing)} (extra: train)"
        print(f"  {name:<32} {state}")
        print(f"  {'':<32} {note}")

    print("\nsplit")
    print(f"  eval-only suites (never trainable): {sorted(split.EVAL_ONLY_SUITES)}")
    print(
        f"  buckets: 0-{split.TRAIN_MAX} train, {split.TRAIN_MAX + 1}-{split.DEV_MAX} dev, "
        f"{split.DEV_MAX + 1}-99 test"
    )

    inv = _split_inventory(runs_root)
    print(f"\nrecorded runs under {runs_root}")
    if not inv:
        print("  (none)")
    for key, n in sorted(inv.items()):
        mark = "  exportable" if key.endswith("/train") else "  NOT exportable"
        print(f"  {key:<28} {n:>5}{mark}")

    print("\nartifacts")
    for label, path in (
        ("sft dataset", root / DEFAULT_OUT / "sft.jsonl"),
        ("preference pairs", root / DEFAULT_OUT / "pairs.jsonl"),
        # THE DEV HALF. Selection reads these; the trainers refuse them. Their absence used to
        # be invisible until `pi train eval-offline` failed on a missing path.
        ("dev sft (gate)", root / DEFAULT_OUT / "dev" / "sft.dev.jsonl"),
        ("dev pairs (gate)", root / DEFAULT_OUT / "dev" / "pairs.dev.jsonl"),
        ("rung0 winning prompt", root / "conf" / "prompts" / "rung0" / f"{INQUIRER_PROMPT}.txt"),
        ("rung1 adapter", root / "artifacts" / "rung1"),
        ("rung2 adapter", root / "artifacts" / "rung2"),
    ):
        if path.exists():
            n = len(path.read_text().splitlines()) if path.is_file() else -1
            extra = f"  ({n} lines)" if n >= 0 else ""
            print(f"  {label:<22} {path}{extra}")
        else:
            print(f"  {label:<22} {path}  -- absent")

    _print_track_d_status(root)
    return 0


# --------------------------------------------------------------------------- registration


def register(sub: argparse._SubParsersAction) -> None:
    """Attach `pi train` to the top-level dispatch table."""
    p = sub.add_parser("train", help="the training ladder: export, rung0, rung1, rung2, status")
    s = p.add_subparsers(dest="sub", required=True)

    e = s.add_parser("export", help="recorded runs -> SFT rows / preference pairs (NEEDS gold)")
    e.add_argument("--kind", choices=("sft", "pairs", "both"), default="sft")
    e.add_argument("--runs-root", default=None)
    e.add_argument("--root", default=None)
    e.add_argument("--out", default=None, help=f"default <root>/{DEFAULT_OUT}")
    e.add_argument("--gold-root", default=None, help="sets PI_GOLD_ROOT for this process")
    e.add_argument("--graph-version", default="v1")
    # THE NAMES ARE SPELLED HERE A SECOND TIME. `build_parser` may not import `pinq_train`
    # (contract 4) and argparse needs the choices at parser-construction time;
    # `tests/test_answer_node_stop_label.py::test_the_cli_choices_are_the_exporter_s_own` joins
    # this tuple to `pinq_train.export.dataset.SFT_STOP_LABELS` through the dynamic loader.
    e.add_argument(
        "--sft-stop-label",
        choices=("done_before", "answer_node_covered_before"),
        default="done_before",
        help=(
            "WHICH GOLD FACT DECIDES THE SFT STOP TARGET. 'done_before' (default, the rule of "
            "record) is pooled required-evidence completeness. 'answer_node_covered_before' "
            "asks the same question of the ANSWER-BEARING node alone (lane L1.11's rule, "
            "`pi_eval.answer_node`): the diagnostic is that the trained policy stops when "
            "POOLED coverage looks complete while the node that names the answer is still "
            "open. Nothing else changes -- same population, same cohort, same floor, same "
            "argmax. An answer node is a required node, so its spans are a subset of the "
            "required set and the variant's STOP set is a SUPERSET of the default's: this can "
            "move a state ASK -> STOP and, except where the answer node is unknown, never the "
            "other way. Stamped on the manifest as `sft_stop_label`. PAIRS ARE UNAFFECTED "
            "(see --stop-rule, a different flag for a different exporter)."
        ),
    )
    e.add_argument(
        "--split",
        choices=("train", "dev"),
        default="train",
        help=(
            "which split to mine. 'dev' goes through a gold-side scorer that is NOT reachable "
            "over the wire and writes <out>/dev/{sft,pairs}.dev.jsonl. There is deliberately no "
            "'test' here: a checkpoint is selected on dev, and the test split is not touched "
            "until one checkpoint per arm has been chosen."
        ),
    )
    e.add_argument(
        "--include-arm",
        action="append",
        default=None,
        help=(
            "repeatable arm allowlist; default "
            f"{sorted(DEFAULT_TRAINABLE_ARMS)}. MEASURED: 19%% of the shipped sft.jsonl and 30%% "
            "of its ask_ask pairs came from `self_inquire`, so the default keeps the headline "
            "on the prompted policy's own samples and this flag builds the +ancestor-rows "
            "ablation. Recorded on the manifest as `included_arms`."
        ),
    )
    e.add_argument(
        "--cohort",
        choices=("any", "fixed", "old"),
        default="any",
        help=(
            "WHICH LOOP RENDERED THE ROWS THIS FILE MAY HOLD. 'fixed' keeps only rows whose "
            "`code_version` is a member of --cohort-file; 'old' keeps only rows that are not "
            "(an unstamped row ran under no recorded code, so it is old). 'any' is the "
            "historical behaviour and the default. Under the OLD loop the answer written into "
            "a turn's history block never saw the documents fetched to answer it, so an "
            "evidence-bearing prompt it rendered is not one the policy will ever see. The "
            "count refused is on the manifest as `n_cohort_refused`, beside the cohort file's "
            "sha that defined it."
        ),
    )
    e.add_argument(
        "--turn0-any-cohort",
        action="store_true",
        help=(
            "with --cohort fixed/old, keep turn-0 rows from either cohort. At turn 0 nothing "
            "has been asked and no evidence has been fetched, so there is no stale per-turn "
            "answer and the rendered prompt is identical under both loops. Only meaningful "
            "with a cohort selection; under --cohort any it is refused rather than recorded."
        ),
    )
    e.add_argument(
        "--cohort-file",
        default=None,
        help=(
            "json naming the FIXED-loop cohort (`members`), default "
            f"<root>/{DEFAULT_COHORT_FILE}. A preference pair is refused when one candidate is "
            "INSIDE the cohort and the other is outside it; two commits of the same loop pair "
            "normally, and an unstamped candidate is refused against any stamped one. The "
            "file's sha256 is recorded on the manifest as `cohort_file_sha`."
        ),
    )
    e.add_argument(
        "--rater-prefs",
        default=None,
        help=(
            "jsonl of A6-derived majority preferences (winner_run/loser_run), required by "
            "--rank rater. Each line orders one unordered candidate pair; pairs with no "
            "verdict keep the mechanical order and are excluded from n_rater_ordered."
        ),
    )
    e.add_argument(
        "--max-examples-per-task",
        type=int,
        default=None,
        help=(
            "cap the SFT rows any one task contributes (per kind). OFF by default. Measured: "
            "within-task distinct-3 falls from 0.81 at 2-3 questions per task to 0.13 at 50+, "
            "and the uncapped live export failed the 0.65 floor at 0.638 while the same rows "
            "capped at 20 score 0.666."
        ),
    )
    e.add_argument(
        "--rank",
        choices=("outcome", "anticipation", "rater"),
        default="outcome",
        help=(
            "how a pair is ordered when the outcome ties. 'outcome' (default) is the CONTROL "
            "export -- nothing in the ranking references a latent-need label, so the "
            "latent-pursuit check is a genuine measurement on it. 'anticipation' adds "
            "`newly_reachable` above speed and writes pairs.anticipation.jsonl."
        ),
    )
    e.add_argument(
        "--stop-rule",
        choices=("gold_coverage_v1", "gold_coverage_v2"),
        default="gold_coverage_v1",
        help=(
            "which STOP rule the PAIRS export runs under, recorded on the manifest as "
            "`stop_rule`. 'gold_coverage_v1' (default, and the rule of record for every "
            "shipped artifact): a state gold says was done and where nothing stopped gets one "
            "synthesised STOP-chosen pair, and a not-done state gets only what its candidates "
            "sampled. 'gold_coverage_v2' adds the mirror -- at a not-done state where at least "
            "one candidate asked and none stopped, one pair with the best-available ASK chosen "
            "over STOP, counted in `n_stop_pairs_synth_notdone`. That contrast is absent from "
            "the corpus: those states clear no noise floor, so the SFT rule drops them "
            "(`n_no_target_dropped`), and every rung-1 checkpoint stops at 9-17 percent of "
            "not-done states. Affects --kind pairs ONLY; the SFT rule is unchanged and its "
            "manifest still reads gold_coverage_v1."
        ),
    )
    e.add_argument(
        "--tau",
        type=float,
        default=0.05,
        help="the formalism's stopping threshold; MEASURE it with reward.tau_from_pilot",
    )
    e.add_argument("--sigma-j", type=float, default=0.0, help="the judge's measured sigma")
    e.add_argument("--len-delta-max", type=int, default=40)
    e.add_argument(
        "--headroom-normalised",
        action="store_true",
        help="pay for the fraction of REMAINING coverage a turn captured, not of the whole. "
        "Changes the SCALE of phi_tilde, so --tau must be re-estimated from "
        "headroom-normalised pilot values before an export under it is trained on.",
    )
    e.add_argument(
        "--no-verify-state",
        action="store_true",
        help="skip the subset_hash check on the reconstructed state (not recommended)",
    )
    e.add_argument("--verbose", action="store_true", help="print every skipped run and why")
    e.set_defaults(fn=cmd_train_export)

    pr = s.add_parser(
        "promote",
        help="re-attach rater verdicts to an exported pairs file (idempotent, by identity)",
    )
    pr.add_argument("--pairs", default="data/rl/pairs.jsonl")
    pr.add_argument(
        "--annotations",
        required=True,
        help="the IMMUTABLE annotation root (bundles/, round*/, a6/). data/rl is never the "
        "system of record for a verdict; this is.",
    )
    pr.add_argument("--quarantine", default="conf/annotations/quarantine.json")
    pr.add_argument("--out", default=None, help="default: rewrite --pairs in place")
    pr.add_argument(
        "--allow-loss",
        action="store_true",
        help="write even though rows would lose an annotation they currently carry. Only when "
        "that is intended -- it usually means the annotation root is missing a campaign.",
    )
    pr.set_defaults(fn=cmd_train_promote)

    # --- track C: verify-prefs ---
    vp = s.add_parser(
        "verify-prefs",
        help="recompute a pairs manifest's rater_prefs_sha from a preference file (read-only)",
    )
    vp.add_argument(
        "--pairs-manifest", required=True, help="e.g. data/rl/pairs.rater.manifest.json"
    )
    vp.add_argument("--prefs", required=True, help="the winner_run/loser_run jsonl it should name")
    vp.set_defaults(fn=cmd_train_verify_prefs)
    # --- end track C: verify-prefs ---

    r0 = s.add_parser("rung0", help="GEPA prompt search over the Inquirer template (no GPU)")
    r0.add_argument("--suite", action="append", default=[])
    r0.add_argument("--n-tasks", type=int, default=40)
    r0.add_argument("--n-val-tasks", type=int, default=20)
    r0.add_argument(
        "--max-prompt-tokens",
        default="family",
        metavar="family|none|N",
        help="parity budget for a mutated prompt: 'family' derives it from the base prompt's "
        "PARITY_FAMILY (default, keeps the winner shippable into an arm comparison), an "
        "integer sets it, 'none' removes it -- an unconstrained search optimises length",
    )
    r0.add_argument("--candidates", type=int, default=8, help="candidates per generation")
    r0.add_argument("--generations", type=int, default=6)
    r0.add_argument("--minibatch", type=int, default=8)
    r0.add_argument("--seed", type=int, default=0)
    r0.add_argument("--task-ids", default=None, help="file of `suite_id/task_id` lines")
    r0.add_argument("--score-url", default=DEFAULT_SCORE_URL)
    r0.add_argument("--policy-model", default=None)
    r0.add_argument("--reflector-base-url", default=None)
    r0.add_argument("--reflector-model", default=None)
    r0.add_argument("--overlay-root", default=None)
    r0.add_argument("--out", default=None, help="default <root>/conf/prompts/rung0")
    r0.add_argument("--root", default=None)
    r0.add_argument("--dry-run", action="store_true", help="print the projected cost and stop")
    r0.add_argument("--yes", action="store_true", help="required to actually spend money")
    r0.add_argument(
        "--rollout-url",
        default=None,
        help="rollout/retrieve server; defaults to --score-url. A scoring server has "
        "PI_GOLD_ROOT set and REFUSES /rollout, so a real search needs both.",
    )
    r0.set_defaults(fn=cmd_train_rung0)

    r1 = s.add_parser("rung1", help="rejection-sampling SFT: preflight always, --train needs a GPU")
    r1.add_argument("--base-model", default=None)
    r1.add_argument("--dataset", default=f"{DEFAULT_OUT}/sft.jsonl")
    r1.add_argument("--out", default="artifacts/rung1")
    r1.add_argument("--tau", type=float, default=0.05)
    r1.add_argument("--sigma-j", type=float, default=0.0)
    # --- track F: cli ---
    # `default=None` on every knob below, so that "not given" is distinguishable from "given
    # the default": that is what lets --config fill it in and a flag override it. The values
    # are computed by `_rung1_defaults()`.
    r1.add_argument("--config", default=None, help="a JSON grid; flags override it")
    r1.add_argument("--epochs", type=int, default=None)
    r1.add_argument("--max-seq-len", type=int, default=None)
    r1.add_argument("--learning-rate", type=float, default=None)
    r1.add_argument("--per-device-batch", type=int, default=None)
    r1.add_argument("--grad-accum", type=int, default=None)
    r1.add_argument(
        "--lora-r", type=int, default=None, help="LoRA rank; alpha is 2r and is not separate"
    )
    r1.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=None)
    r1.add_argument(
        "--chat-template",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="--no-chat-template trains on the raw concatenation, which is NOT what the server "
        "renders; every dev NLL is then measured on a prompt the policy never sees.",
    )
    r1.add_argument(
        "--reasoning-effort",
        choices=("low", "medium", "high"),
        default=None,
        help="harmony (gpt-oss) only: sets the 'Reasoning: ...' line in the SYSTEM message. "
        "Refused on a ChatML base, where the template has no such variable and would drop it. "
        "Unset means the template's own default, which is what the server renders.",
    )
    # --- stop weight ---
    r1.add_argument(
        "--stop-weight",
        type=float,
        default=None,
        help="multiply every STOP row's sample_weight by this before the accumulation window is "
        "normalised. STOP is read off the ACTION BYTES (pinq.actions.action_kind_of), which is "
        "what the mask supervises. Must be > 0. UNSET is not 1.0: at the absent default the "
        "weights are the ones every finished rung-1 run trained on and cfg.sha is unmoved; set, "
        "it enters cfg.sha. The preflight reports the STOP share OF THE LOSS it buys.",
    )
    # --- end stop weight ---
    # --- curriculum ---
    # THE THREE NAMES ARE SPELLED HERE A SECOND TIME, because `build_parser` may not import
    # `pinq_train` (contract 4) and argparse needs the choices at parser-construction time.
    # `tests/test_train_cli.py::test_the_curriculum_choices_are_the_ones_the_trainer_knows`
    # joins this tuple to `pinq_train.rung1_sft.CURRICULA` through the dynamic loader, so a
    # name added on one side and not the other fails there rather than on a rented card.
    r1.add_argument(
        "--curriculum",
        choices=("shuffled", "depth_easy_first", "depth_hard_first"),
        default=None,
        help="the order the SFT rows are shown in. `shuffled` (default) reshuffles every "
        "epoch and is the null arm; `depth_easy_first` shows epoch 1 as depth-0, then "
        "depth-1, then depth>=2 rows and shuffles the epochs after it, and "
        "`depth_hard_first` is the same schedule reversed. STOP rows carry no target depth "
        "and are dealt across the strata so each one holds the file's STOP share. Exposure is "
        "identical in all three -- every row once per epoch. IN cfg.sha when it is not the "
        "default: it changes what the checkpoint saw and when.",
    )
    # --- end curriculum ---
    # --- end track F: cli ---
    _add_seed_flag(r1)
    # --- resume ---
    _add_resume_flags(r1, "artifacts/rung1")
    # --- end resume ---
    r1.add_argument("--train", action="store_true", help="run the trainer (needs CUDA)")
    r1.add_argument(
        "--acknowledge-untested",
        action="store_true",
        help="required by --train: no run of this trainer has ever completed in this project",
    )
    r1.set_defaults(fn=cmd_train_rung1)

    # --- track A: tokstats ---
    tk = s.add_parser(
        "tokstats",
        help="real-token percentiles over an exported dataset (sets --max-seq-len)",
    )
    tk.add_argument("--dataset", required=True, help="the exported jsonl, streamed not loaded")
    tk.add_argument("--tokenizer", required=True, help="AutoTokenizer name or path; NEEDS [train]")
    tk.add_argument(
        "--field",
        default="state_text",
        help="which field to measure. The default is the STATE alone: the action and the chat "
        "template's header sit on top of the number this prints.",
    )
    tk.set_defaults(fn=cmd_train_tokstats)
    # --- track A: tokstats ---

    r2 = s.add_parser(
        "rung2", help="DPO on same-state pairs: preflight always, --train needs a GPU"
    )
    r2.add_argument(
        "--reference",
        choices=("adapter", "merged", "base"),
        default=None,
        help="how pi_ref is obtained. 'adapter' (default) loads rung 1's LoRA twice onto one "
        "base -- a trainable copy and a frozen one -- so nothing is merged and nothing can be "
        "rounded away; a bf16 fold lost 44.7%% of rung 1's update (docs/TRAINING.md 6.1). "
        "'merged' is the fused-checkpoint path and needs --merged-base. 'base' is the "
        "dpo_from_base ABLATION: no rung 1 anywhere, a fresh LoRA on the untrained base and "
        "pi_ref that same untrained base, i.e. what the preference label buys with no "
        "imitation in front of it. It refuses both --adapter and --merged-base, and its KL is "
        "anchored to a different model, so its beta and learning rate do not transfer from "
        "the adapter arm (docs/TRAINING.md 6.1).",
    )
    r2.add_argument(
        "--merged-base",
        default=None,
        help="the output directory of `pi train merge`; only with --reference merged. Rung 2's "
        "reference policy is then rung 1 with the fresh LoRA disabled, which is only true if "
        "rung 1 is IN the weights; the sha is read from that directory's merge.manifest.json "
        "into cfg.sha.",
    )
    r2.add_argument("--base-model", default=None, help="only without --merged-base")
    r2.add_argument(
        "--adapter",
        default=None,
        help="rung 1's LoRA. PROVENANCE under a merge -- which adapter went into the merged "
        "base -- not something this command loads.",
    )
    r2.add_argument("--pairs", default=f"{DEFAULT_OUT}/pairs.jsonl")
    r2.add_argument("--out", default="artifacts/rung2")
    # --- track F: cli --- (see the rung-1 block; values are computed by `_rung2_defaults()`)
    r2.add_argument("--config", default=None, help="a JSON grid; flags override it")
    r2.add_argument("--beta", type=float, default=None)
    r2.add_argument("--epochs", type=int, default=None)
    r2.add_argument("--max-seq-len", type=int, default=None)
    r2.add_argument("--max-prompt-len", type=int, default=None)
    r2.add_argument("--learning-rate", type=float, default=None)
    r2.add_argument("--per-device-batch", type=int, default=None)
    r2.add_argument("--grad-accum", type=int, default=None)
    r2.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=None)
    r2.add_argument(
        "--include-pair-kind",
        dest="include_pair_kinds",
        action="append",
        default=[],
        help="repeatable: which DECISION to train. Default is the sampled kinds (ask_ask, "
        "ask_stop); the primary arm is ask_ask alone, because the 236 STOP-winning pairs "
        "cannot teach stopping and the 1,356 ASK-winning ones import the length confound.",
    )
    # --- objective ---
    # WHAT THE LOSS IS. `loss_type` has been in `cfg.sha` since rung 2 was written and no flag
    # could set it, so every arm this repository has run was plain DPO whether or not that was
    # the intention. The names are NOT `choices=` here: the accepted set is `rung2.LOSS_TYPES`,
    # which lives behind the `pinq_train` import this module may not make (contract 4), so
    # `DPOConfig.validate()` is what refuses a bad one -- with the list, and before anything
    # loads.
    r2.add_argument(
        "--loss-type",
        dest="loss_type",
        action="append",
        default=[],
        help="repeatable: the DPO loss terms, summed with --loss-weights. Default sigmoid "
        "(plain DPO). `--loss-type sigmoid --loss-type sft` is the RPO/MPO shape -- the "
        "preference term plus an NLL anchor on the chosen side. Refused, with the list the "
        "installed trl dispatches, if it names one that trl does not.",
    )
    r2.add_argument(
        "--loss-weights",
        default=None,
        help="comma-separated floats, ONE PER --loss-type (e.g. 1.0,0.5). Omitted means trl's "
        "own equal weighting, which is what every run trained before this flag existed used.",
    )
    r2.add_argument(
        "--ld-alpha",
        type=float,
        default=None,
        help="LD-DPO: the weight on the log-probabilities of the tokens PAST the shared prefix "
        "of the two sides. 1.0 applies no weighting, 0.0 masks that tail entirely; omitted is "
        "plain DPO. A second lever on the length confound the |delta len| guard bounds from "
        "the other side.",
    )
    # --- end objective ---
    # --- pair floors ---
    # Two `DPOConfig` fields that have been in `cfg.sha` from the start with no way to set them
    # from a command line. See `_rung2_defaults()`. The `choices` below spell `rung2.Q_DISTINCTNESS`
    # out rather than importing it (contract 4); `DPOConfig.validate()` checks the value against
    # the real tuple, so a drift between the two is a refusal and not a silently ignored filter.
    r2.add_argument(
        "--min-q-distinctness",
        choices=("paraphrase", "related", "different"),
        default=None,
        help="floor on the question-diversity sidecar, ordered paraphrase < related < "
        "different. Default applies no filter. MEASURED on the interim export: 9.0%% of pairs "
        "are paraphrases -- two renderings of one question, ordered by which retrieval came "
        "back better. Refused if no row carries the field.",
    )
    r2.add_argument(
        "--min-stop-chosen-share",
        type=float,
        default=None,
        help="floor on n_stop_chosen / n_pairs; preflight refuses the run below it. MEASURED "
        "on data/rl/pairs.jsonl: among ask_stop pairs ASK is chosen 1,356 times and STOP 236, "
        "so a run that includes ask_stop to teach stopping trains 'keep asking' and n_pairs "
        "cannot tell. Default 0.0 refuses nothing.",
    )
    # --- end pair floors ---
    # --- matched n ---
    # THE MECHANISM A MATCHED CONTROL IS MADE OF, and the only selector on this command that is
    # not a predicate over a row's fields. `default=None` is ABSENT and not "": an unset flag
    # that arrives as an empty string is a value the config has to parse, and `cfg.sha` would
    # then have to decide whether "" means the same thing as absent.
    r2.add_argument(
        "--subsample-pairs",
        type=int,
        default=None,
        help="train on a random n of the pairs the filters kept, instead of all of them: the "
        "matched control of an ablation (A11 trains dpo_control at dpo_outcome_only's n). "
        "REFUSES when n exceeds what survived -- an arm that quietly trained on fewer is a "
        "matched control that is not matched. Requires --subsample-seed; both enter cfg.sha.",
    )
    r2.add_argument(
        "--subsample-seed",
        type=int,
        default=None,
        help="WHICH draw of --subsample-pairs rows. No default: tying it to --seed would make "
        "three seed replicates three datasets, and 0 would hide the choice. The draw is over "
        "`pair_id`, so it depends on the row SET and not on the file's order.",
    )
    # --- end matched n ---
    # --- end track F: cli ---
    r2.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
        help="conservative DPO: the share of preference labels assumed flipped. The rule "
        "agrees with the A6 raters ~60%% of the time, so the control arm is run at 0.2.",
    )
    r2.add_argument(
        "--include-label-source",
        action="append",
        default=[],
        help="repeatable allowlist over `label_source` (rule, rater); default both. This is "
        "the file the pair came from, NOT who ordered it -- see --exclude-decided-by.",
    )
    r2.add_argument(
        "--exclude-decided-by",
        action="append",
        default=[],
        help="repeatable: drop pairs ordered by this decider. 532 rows on pairs.rater.jsonl "
        "carry label_source=rule and decided_by=rater, so filtering the label source alone "
        "excludes none of them.",
    )
    r2.add_argument(
        "--include-code-version",
        action="append",
        default=[],
        help="repeatable cohort allowlist. Rows without a `code_version` are dropped under "
        "it and counted separately; empty means no filter.",
    )
    _add_seed_flag(r2)
    # --- resume ---
    _add_resume_flags(r2, "artifacts/rung2")
    # --- end resume ---
    r2.add_argument("--train", action="store_true", help="run the trainer (needs CUDA)")
    r2.add_argument("--acknowledge-untested", action="store_true")
    r2.set_defaults(fn=cmd_train_rung2)

    # --- track B: merge ---
    mg = s.add_parser(
        "merge",
        help="merge rung 1's LoRA into the base weights (rung 2's reference policy; needs the "
        "[train] extra)",
    )
    mg.add_argument("--base-model", required=True)
    mg.add_argument("--adapter", required=True, help="rung 1's LoRA directory")
    mg.add_argument("--out", required=True, help="directory to write the merged model into")
    mg.set_defaults(fn=cmd_train_merge)
    # --- end track B: merge ---

    ti = s.add_parser(
        "task-ids",
        help="write the TRAIN-split id file rung0 --task-ids needs (nothing else produced it)",
    )
    ti.add_argument("--suite", action="append", default=None, help="repeatable; default musique")
    ti.add_argument("--limit", type=int, default=None, help="prefix length per suite")
    ti.add_argument("--out", default=None, help="default data/rl/rung0_tasks.txt")
    ti.add_argument("--root", default=None)
    ti.set_defaults(fn=cmd_train_task_ids)

    sc = s.add_parser(
        "sample-candidates",
        help="fork a recorded run into N candidate continuations per turn (rung 2's inputs)",
    )
    sc.add_argument("--run-id", required=True, help="the PARENT run to fork")
    sc.add_argument(
        "--turns",
        required=True,
        help="comma-separated turn indices to branch at, e.g. 1,3",
    )
    sc.add_argument("--n", type=int, default=4, help="candidates per turn (default 4)")
    sc.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="sampling temperature. At 0 every candidate is identical and the pair is empty.",
    )
    sc.add_argument("--runs-root", default=None)
    sc.add_argument("--cache-root", default=None)
    sc.add_argument("--concurrency", type=int, default=None)
    sc.add_argument("--spend-cap", type=float, default=None)
    sc.add_argument(
        "--budget-cap",
        type=int,
        default=None,
        help="retrieval-call cap for the continuation; default inherits the parent. In "
        "semantic_hash, so a re-fork at a new cap is a distinct run, never a resume.",
    )
    sc.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="turn cap for the continuation; default inherits the parent. Raise it WITH "
        "--budget-cap: every ask charges one retrieval, so whichever is lower binds.",
    )
    sc.add_argument("--dry-run", action="store_true", help="print the plan, dispatch nothing")
    sc.set_defaults(fn=cmd_train_sample_candidates)

    fs = s.add_parser(
        "frontier-states",
        help="which (run, turn) pairs are worth forking: those where a CHOICE existed",
    )
    fs.add_argument("--runs-root", default=None)
    fs.add_argument("--suite", default="musique")
    fs.add_argument("--graph-version", default="v1")
    fs.add_argument("--gold-root", default=None, help="sets PI_GOLD_ROOT for this process")
    fs.add_argument(
        "--max-turn",
        type=int,
        default=2,
        help="ignore states later than this. 'Prioritising early': an early fork dominates a "
        "later one, because every later state is downstream of it.",
    )
    fs.add_argument("--min-frontier", type=int, default=2, help="a fork needs a real choice")
    fs.add_argument("--limit", type=int, default=0, help="0 = no cap")
    fs.add_argument(
        "--per-task",
        type=int,
        default=1,
        help="at most this many states per TASK, so one task cannot eat the budget",
    )
    fs.add_argument("--shell", action="store_true", help="emit runnable sample-candidates lines")
    fs.add_argument("--budget-cap", type=int, default=None, help="passed through to --shell lines")
    fs.add_argument("--max-turns", type=int, default=None, help="passed through to --shell lines")
    fs.add_argument("--n", type=int, default=8, help="candidates per state, for --shell")
    fs.set_defaults(fn=cmd_train_frontier_states)

    st = s.add_parser("status", help="the ladder, the split, what is on disk, what is missing")
    st.add_argument("--root", default=None)
    st.add_argument("--runs-root", default=None)
    st.set_defaults(fn=cmd_train_status)

    _register_track_d(s)


def cmd_train_frontier_states(a: argparse.Namespace) -> int:
    """List the decision points worth forking, earliest and widest first.

    A fork buys signal only where more than one required need was askable: with a frontier of
    one, every candidate paraphrases the same question. Ordering is (turn, -frontier_size) so
    a capped budget spends on the earliest, widest choices -- which is where a wrong turn is
    least recoverable, every later state being downstream of it.
    """
    if a.gold_root:
        os.environ["PI_GOLD_ROOT"] = str(Path(a.gold_root).resolve())
    runs_root = Path(a.runs_root) if a.runs_root else Path.cwd() / "runs"
    graphs = _graphs(a.suite, a.graph_version)

    found: list[tuple[int, int, str, str, int]] = []  # turn, -size, run_id, task_id, depth
    scanned = skipped = 0
    for d in sorted(runs_root.iterdir()):
        if not (d / "manifest.json").exists():
            continue
        try:
            man = _read_json(d / "manifest.json")
        except Exception:
            skipped += 1
            continue
        if str(man.get("suite_id")) != a.suite or str(man.get("status", "ok")) not in ("", "ok"):
            continue
        # A fork of a fork is not a fresh choice: it inherits its parent's prefix.
        if man.get("branch_of_run_id"):
            continue
        if str(man.get("arm_id", "")) in GOLD_EXPOSED_ARMS:
            continue
        task_id = str(man.get("task_id"))
        graph = graphs.get(task_id)
        if graph is None:
            continue
        try:
            turns = sorted(_read_jsonl(d / "turns.jsonl"), key=lambda t: int(t["turn_idx"]))
            if not turns:
                continue
            records = _match_records(d, graph, turns)
            states = branchable_states(graph, records, n_turns=len(turns))
        except Exception:
            skipped += 1
            continue
        scanned += 1
        for st in states:
            if st.turn_idx > a.max_turn or st.frontier_size < a.min_frontier:
                continue
            found.append((st.turn_idx, -st.frontier_size, d.name, task_id, st.latent_depth))

    found.sort()
    # AT MOST `--per-task` STATES PER TASK. Sibling candidates at two turns of the same task
    # are not independent draws, and without this one heavily-branched task eats the budget.
    seen: dict[str, int] = {}
    picked: list[tuple[int, int, str, str, int]] = []
    for row in found:
        t = row[3]
        if seen.get(t, 0) >= a.per_task:
            continue
        seen[t] = seen.get(t, 0) + 1
        picked.append(row)
        if a.limit and len(picked) >= a.limit:
            break

    if a.shell:
        caps = ""
        if a.budget_cap is not None:
            caps += f" --budget-cap {a.budget_cap}"
        if a.max_turns is not None:
            caps += f" --max-turns {a.max_turns}"
        for turn, _neg, run_id, _task, _d in picked:
            print(
                f"pi train sample-candidates --run-id {run_id} --turns {turn} "
                f"--n {a.n} --temperature 1.0 --concurrency 4{caps}"
            )
        return 0

    print(f"scanned {scanned} {a.suite} runs ({skipped} unreadable), {len(found)} states found")
    print(
        f"selected {len(picked)} across {len(seen)} tasks "
        f"(max_turn={a.max_turn}, min_frontier={a.min_frontier}, per_task={a.per_task})"
    )
    by_turn: dict[int, int] = {}
    for turn, neg, _r, _t, _d in picked:
        by_turn[turn] = by_turn.get(turn, 0) + 1
    for turn in sorted(by_turn):
        print(f"  turn {turn}: {by_turn[turn]} states")
    for turn, neg, run_id, task, depth in picked[:15]:
        print(f"    t={turn} frontier={-neg} depth={depth} {run_id[:16]} {task}")
    return 0


# ------------------------------------------------------- pi train sample-candidates


def _parent_manifest(runs_root: Path, run_id: str) -> dict:
    p = runs_root / run_id / "manifest.json"
    if not p.exists():
        raise FileNotFoundError(f"no run at {p.parent}")
    return _read_json(p)


def _n_recorded_turns(runs_root: Path, run_id: str) -> int:
    p = runs_root / run_id / "turns.jsonl"
    if not p.exists():
        return 0
    return sum(1 for line in p.read_text(encoding="utf-8").splitlines() if line.strip())


def candidate_specs(
    *,
    runs_root: Path,
    cache_root: str,
    parent_run_id: str,
    turns: Sequence[int],
    n_candidates: int,
    code_version: str,
    dirty: bool,
    dirty_files: tuple[str, ...] = (),
    temperature: float,
    budget_cap: int | None = None,
    max_turns: int | None = None,
) -> list:
    """One UnitSpec per (branch turn, candidate). The parent's cell, forked.

    Every field but the seed, the branch point and the two caps comes from the PARENT's
    manifest: a candidate that differed in arm, task or k would not be a continuation of the
    same state, and the pair built from it would compare two different experiments.

    THE CAPS MAY BE OVERRIDDEN, AND THIS PARAGRAPH USED TO SAY THEY MAY NOT. It said a
    candidate "that differed in caps would not be a continuation of the same state". That is
    false, and the reason is the firewall's own argument: the policy is budget-blind by type
    (`Inquirer.act(s)` takes no ledger, `FORBIDDEN_PLACEHOLDERS` keep the cap out of every
    prompt), so the STATE at the fork turn is identical whatever cap the continuation runs
    under. What the cap changes is the episode-level LABEL -- whether the candidate reached
    complete evidence, and how many turns it took. Measured overnight: 93% of candidates were
    forked at cap 8 (inherited from latent_trainset.yaml) and 53% of them stopped on the
    budget, so `turns_to_complete` was -1 on 71% of pairs and the preference label fell back
    to per-turn gain. `None` inherits the parent, which keeps every existing caller exact.

    Both values are in `semantic_hash`, so a re-fork at a larger cap is a distinct run: it is
    not skipped by resume-by-existence and cannot overwrite the cap-8 candidates. The
    exporter's cross-cap guard keeps the two cohorts from being paired with each other.

    Candidate 0 is deliberately included even though `candidate_seed(s, 0) == s` reproduces
    the parent's own continuation: it is the on-policy sample, and a preference pair needs it
    as the baseline. It gets its own run directory only because `branch_of_run_id` and
    `branch_turn_idx` are in semantic_hash -- without them it would collide with the parent.
    """
    from pi_run.worker import UnitSpec
    from pinq.sampling import candidate_seed

    # READ, NOT RETYPED: the literal fallbacks below used to duplicate `UnitSpec.max_turns`'s
    # and `UnitSpec.k`'s own field defaults by coincidence of value. A change to either default
    # on `UnitSpec` now reaches this fallback automatically instead of silently diverging from it.
    unit_spec_defaults = {f.name: f.default for f in fields(UnitSpec)}

    m = _parent_manifest(runs_root, parent_run_id)
    n_turns = _n_recorded_turns(runs_root, parent_run_id)
    # The manifest records the corpus directory's BASENAME, not a path -- absolute paths are
    # kept out of artifacts (scripts/check_no_home_paths.sh). Rejoin it to the corpora root
    # the same way `pi run` resolves it, or the worker looks for `<hash>/tasks.jsonl`
    # relative to the cwd and a branch dies on FileNotFoundError.
    corpus_dir = str(m.get("corpus_dir") or "")
    if corpus_dir and not Path(corpus_dir).exists():
        candidate = runs_root.parent / "data" / "corpora" / str(m["suite_id"]) / corpus_dir
        if candidate.exists():
            corpus_dir = str(candidate)
    out = []
    cap = int(m["budget_cap"]) if m.get("budget_cap") is not None else None
    if budget_cap is not None:
        cap = int(budget_cap)
    mt = (
        int(max_turns)
        if max_turns is not None
        else int(m.get("max_turns") or unit_spec_defaults["max_turns"])
    )
    for t in turns:
        if not 0 <= t < n_turns:
            raise ValueError(
                f"turn {t} is outside the parent's recorded trajectory of {n_turns} turns; "
                "branching past the end would sample a decision the policy never faced."
            )
        if mt <= t:
            # A fork at turn t with max_turns <= t replays the prefix and never acts. That is
            # not a candidate; it is the parent's prefix wearing a new run id.
            raise ValueError(
                f"max_turns={mt} is not above the fork turn {t}; the candidate would end "
                "before it could act."
            )
        for i in range(n_candidates):
            out.append(
                UnitSpec(
                    suite_id=str(m["suite_id"]),
                    corpus_dir=corpus_dir,
                    task_id=str(m["task_id"]),
                    arm_id=str(m["arm_id"]),
                    seed=int(m["seed"]),
                    runs_root=str(runs_root),
                    cache_root=cache_root,
                    max_turns=mt,
                    k=int(m.get("k") or unit_spec_defaults["k"]),
                    budget_cap=cap,
                    code_version=code_version,
                    dirty=dirty,
                    dirty_files=dirty_files,
                    branch_of_run_id=parent_run_id,
                    branch_turn_idx=t,
                    branch_seed=candidate_seed(int(m["seed"]), i),
                    # Temperature 0 would make every candidate identical and the pair empty.
                    sampling={"temperature": float(temperature)},
                )
            )
    return out


def cmd_train_sample_candidates(a: argparse.Namespace) -> int:
    """Fork a recorded run into N candidate continuations per chosen turn.

    This is the driver rung 2 never had. `pairs.jsonl` has been 0 lines not because the
    export was broken but because nothing ever produced a SECOND continuation from one
    state: the primitives in `pinq.sampling` had no production caller.
    """
    from pi_run.cache import cache_root as _cache_root
    from pi_run.manifest import git_info
    from pi_run.sweep import run_sweep

    runs_root = Path(a.runs_root) if a.runs_root else Path.cwd() / "runs"
    gi = git_info(str(Path.cwd()))
    turns = [int(t) for t in str(a.turns).split(",") if t.strip() != ""]
    try:
        specs = candidate_specs(
            runs_root=runs_root,
            cache_root=str(_cache_root(a.cache_root)),
            parent_run_id=a.run_id,
            turns=turns,
            n_candidates=a.n,
            code_version=gi.sha,
            dirty=gi.dirty,
            dirty_files=gi.dirty_files,
            temperature=a.temperature,
            budget_cap=a.budget_cap,
            max_turns=a.max_turns,
        )
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    if a.dry_run:
        print(
            json.dumps(
                {
                    "parent": a.run_id,
                    "turns": turns,
                    "n_candidates": a.n,
                    "units": len(specs),
                    "temperature": a.temperature,
                    "budget_cap": specs[0].budget_cap if specs else None,
                    "max_turns": specs[0].max_turns if specs else None,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    results = run_sweep(specs, concurrency=a.concurrency, cap_usd=a.spend_cap)
    ok = sum(1 for r in results if str(r.get("status")) == "ok")
    print(
        json.dumps(
            {
                "parent": a.run_id,
                "units": len(results),
                "ok": ok,
                "failed": len(results) - ok,
                "budget_cap": specs[0].budget_cap if specs else None,
                "max_turns": specs[0].max_turns if specs else None,
                "run_ids": sorted({str(r.get("run_id")) for r in results if r.get("run_id")}),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if ok == len(results) else 1


# --- track D: eval-offline, gate ---
#
# Checkpoint selection, which is a different activity from measurement and is kept in its own
# block for that reason. `eval-offline` is Tier A (free, deterministic, on the dev SFT/pair
# files); `gate` is Tier B (the dev rollouts, read back out of parquet). Neither may touch the
# test split, and both refuse rather than warn when asked to.
#
# Both trainer modules are reached through `_train(...)`, i.e. by NAME, for the reason this
# file's header gives: a static `pi_run -> pinq_train` edge would put `pinq_train -> pi_run ->
# pi_eval` in the module graph and leave the gold firewall one hop from broken.


def _dev_rows_or_refuse(path: Path, label: str) -> list[dict] | int:
    """Read a JSONL artifact and REFUSE it unless every row is `split == "dev"`.

    The mirror of the trainers' `HeldOutDataset` refusal, pointing the other way. These numbers
    SELECT a checkpoint; computed over train rows they would select on the training set, and
    computed over test rows they would burn the held-out split before it is used once. Neither
    failure leaves a mark in the output JSON, so neither is allowed to happen quietly.
    """
    if not path.exists():
        print(f"{label}: {path} does not exist", file=sys.stderr)
        return 2
    rows: list[dict] = []
    for i, line in enumerate(path.read_text().splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        split = str(row.get("split", ""))
        if split != "dev":
            print(
                f"{label}: {path} row {i} has split={split!r}, not 'dev'. Refusing: this "
                "command selects a checkpoint, so a train row would select on the training "
                "set and a test row would burn the held-out split before it is read once.",
                file=sys.stderr,
            )
            return 2
        rows.append(row)
    return rows


def _print_track_d_status(root: Path) -> None:
    """The checkpoint registry and the diversity floor, printed where they are read.

    THE FLOOR IS PRINTED WITH ITS CALIBRATION, not on its own. `DISTINCT3_FLOOR = 0.55` gates
    rung-1 SFT with the message "mode collapse: the checkpoint's questions stopped depending on
    the state", and on musique it is UNREACHABLE -- the benchmark's own task questions score
    0.4685 and the gold answer key 0.2250, so a policy asking exactly the gold sub-questions,
    perfectly state-dependent by construction, would be rejected. Printing 0.55 without that
    number invites someone to read it as a target they are failing to hit. The gate's floor is
    a different statistic: within-task rather than pooled, where our own export scores 0.777.
    """
    from pinq_adapters.llm import checkpoints

    # BY NAME, like every other reference to the trainer in this file: grimp sees a
    # function-level `import pinq_train` exactly as loudly as a module-level one.
    floor = _train("rung1_sft.collapse").DISTINCT3_FLOOR

    print("\ncheckpoint registry")
    path = os.environ.get("PI_CHECKPOINTS") or str(root / "conf" / "checkpoints.json")
    try:
        reg = checkpoints.registry(path)
    except ValueError as exc:
        print(f"  {path}\n  REFUSED: {exc}")
        return
    print(f"  {path}  ({len(reg)} registered)")
    if not reg:
        print("    (none) -- every model id pins adapter_sha=None, so no run_id moves.")
        print("    An entry for an id that has ALREADY run renames every run that used it.")
    for name, e in sorted(reg.items()):
        print(f"    {name:<24} rung {e['rung']}  {e['base_model']}  {str(e['adapter_sha'])[:12]}")

    print("\ndiversity floors")
    print(
        f"  pooled distinct-3 floor (rung 1 gate) ....... {floor}\n"
        "    UNREACHABLE ON MUSIQUE: the corpus's own 800 task questions score 0.4685 and the\n"
        "    gold answer key 0.2250, so a perfectly state-dependent policy fails it. A failure\n"
        "    against this floor on musique is not evidence of mode collapse.\n"
        "  within-task distinct-3 floor (pi train gate)  0.65\n"
        "    The reachable one, and the one that separates state-dependence from repetition:\n"
        "    our own SFT export scores 0.777 there."
    )


# The files a directory must hold to be a tokenizer in its own right. `tokenizer.json` is the
# fast tokenizer; `tokenizer_config.json` alone (a slow/sentencepiece checkpoint) is enough for
# `AutoTokenizer`, so either is accepted rather than both demanded.
_TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "spiece.model")


def tokenizer_source(checkpoint: Path) -> str:
    """Where `--checkpoint`'s tokenizer lives: the directory itself, or the base it adapts.

    `--checkpoint` is documented as "a local model/adapter directory" and the adapter half did
    not work. MEASURED on transformers 5.17.0 against rung 1's output:
    `AutoModelForCausalLM.from_pretrained(adapter_dir)` SUCCEEDS -- 5.x reads
    `adapter_config.json`, resolves `base_model_name_or_path` and applies the LoRA -- while
    `AutoTokenizer.from_pretrained(adapter_dir)` raises `ValueError: Couldn't instantiate the
    backend tokenizer`, because peft's `save_pretrained` writes no tokenizer files. Tier A, the
    free check that chooses a checkpoint before any rollout is bought, was therefore usable
    only on a merged full model -- and `pinq_train.merge.MERGE_DTYPE` is the measurement of why
    merging is not free.

    A LoRA cannot change the tokenizer, so the base's IS the right one; the only question was
    where to look, and the adapter records the answer itself.

    REFUSES rather than defaulting when neither is present. A checkpoint scored under the wrong
    tokenizer yields a perfectly plausible NLL over token ids that mean something else, and
    nothing downstream could see it.
    """
    if any((checkpoint / f).exists() for f in _TOKENIZER_FILES):
        return str(checkpoint)
    cfg = checkpoint / "adapter_config.json"
    if cfg.is_file():
        base = str(_read_json(cfg).get("base_model_name_or_path") or "")
        if base:
            return base
    raise FileNotFoundError(
        f"{checkpoint} holds no tokenizer ({', '.join(_TOKENIZER_FILES)}) and no "
        "adapter_config.json naming a base model to borrow one from. Scoring a checkpoint "
        "under some other tokenizer would produce a plausible NLL over token ids that mean "
        "something else, so this refuses instead of guessing."
    )


def eval_offline_load_kwargs(n_visible_gpus: int | None = None) -> dict:
    """Where and in what precision `pi train eval-offline` loads a checkpoint. PURE in
    `n_visible_gpus`, mirroring `pinq_train.rung1_sft.train.base_model_load_kwargs`.

    A bare `from_pretrained(path)` under transformers 5.x resolves to `torch.get_default_device()`
    (cpu) and `torch.get_default_dtype()` (fp32). MEASURED 2026-09-14 on the cluster: that made
    one Tier A pass over the 1,000-row dev sample take hours on the CPU of an A100 node, while
    the same pass took 21 minutes once the checkpoint evaluator forced the torch defaults to
    cuda/bf16 (scripts/hpc/eval_checkpoints.sh). Elsewhere (the Mac smoke: fp32 on MPS/CPU) the
    library defaults stand, so the recorded Mac verdicts do not change.

    ONE VISIBLE GPU keeps the unchanged path this rung has always run: `device_map="cuda"`
    pins the whole checkpoint onto that one device, which is every 8B-and-smaller Tier A pass
    of record.

    MORE THAN ONE GPU MUST NOT ALSO PIN TO ONE DEVICE. MEASURED 2026-09-16, cluster job 778746
    (rung1-32b-headline): with two GPUs visible this used to return `device_map="cuda"`
    regardless -- the same string `base_model_load_kwargs` never uses above one card -- and the
    Qwen3-32B checkpoint (trained fine, with `device_map="auto"`, on those same two cards) loaded
    back onto device 0 ALONE for eval; nvidia-smi recorded 76,815 MiB on one card and 4 MiB on
    the other, and `pinq_train.eval_offline._logprob`'s `model(batch).logits.float()` died with
    CUDA OOM. `device_map="auto"` above one visible GPU is the same fix `base_model_load_kwargs`
    already made for training, applied to the eval side of the same checkpoint.

    THE DTYPE IS STATED, not left to `device_map="auto"`'s own resolution, for the reason
    `base_model_load_kwargs` states it: transformers computes the split FROM the dtype, and an
    unstated dtype on a bf16 checkpoint upcasts to fp32 before splitting -- doubling the memory
    the split has to fit, on the exact path job 778746 just OOM'd.
    """
    if n_visible_gpus is None:
        import torch  # the trainer venv only; the pure decision below needs no torch

        n_visible_gpus = torch.cuda.device_count()
    # "bfloat16" as a string: transformers resolves dtype names itself, and keeping torch out of
    # the decision lets the main venv (no torch) test it.
    if n_visible_gpus > 1:
        return {"device_map": "auto", "dtype": "bfloat16"}
    if n_visible_gpus == 1:
        return {"device_map": "cuda", "dtype": "bfloat16"}
    return {}


def cmd_train_eval_offline(a: argparse.Namespace) -> int:
    """Tier A: dev NLL, the STOP 2x2, and dev pair accuracy, from a checkpoint on disk."""
    eo = _train("eval_offline")

    sft = _dev_rows_or_refuse(Path(a.dev_sft), "--dev-sft")
    if isinstance(sft, int):
        return sft
    pairs = _dev_rows_or_refuse(Path(a.dev_pairs), "--dev-pairs")
    if isinstance(pairs, int):
        return pairs

    ckpt = Path(a.checkpoint)
    if not ckpt.exists():
        print(f"--checkpoint {ckpt} does not exist", file=sys.stderr)
        return 2

    try:
        tok_src = tokenizer_source(ckpt)
    except FileNotFoundError as exc:
        print(f"--checkpoint {ckpt}: {exc}", file=sys.stderr)
        return 2
    # PROVENANCE BEFORE THE WEIGHTS LOAD, and before a single forward pass. `--registry-name`
    # refuses a name that does not hash to these weights, and the refusal is worth nothing if it
    # arrives 105 minutes into a run that has already written a file. It also costs one sha256 of
    # a 350 MB file, which is seconds, against an evaluation measured in hours.
    from pi_run.manifest import repo_root

    root = repo_root()
    try:
        prov = eo.provenance(
            checkpoint=ckpt,
            dev_sft=a.dev_sft,
            sft_rows=sft,
            dev_pairs=a.dev_pairs,
            pair_rows=pairs,
            # VERBATIM `sys.argv`, not a reconstruction from the namespace. A reconstruction
            # prints the flags argparse understood, which is precisely the set that cannot
            # contain the flag someone forgot; `--reference-ask` missing is the failure this
            # field exists to make visible from the file alone.
            argv=list(sys.argv),
            repo_root=root,
            registry_path=(
                a.registry
                or os.environ.get("PI_CHECKPOINTS")
                or str(root / "conf" / "checkpoints.json")
            ),
            registry_name=a.registry_name,
            reference_ask=a.reference_ask,
            tokenizer=tok_src,
        )
    except eo.AdapterIdentityMismatch as exc:
        print(f"eval-offline REFUSED: {exc}", file=sys.stderr)
        return 2

    # transformers/torch live behind this call and are NOT a dependency of this CLI file. Imported
    # after the provenance check, which needs neither, so a refused name exits 2 on a machine
    # without them instead of failing on the import first.
    tr = importlib.import_module("transformers")
    tok = tr.AutoTokenizer.from_pretrained(tok_src)
    model = tr.AutoModelForCausalLM.from_pretrained(str(ckpt), **eval_offline_load_kwargs()).eval()
    render, logprob = eo.torch_logprob_fn(
        model, tok, chat_template=bool(a.chat_template), enable_thinking=bool(a.enable_thinking)
    )

    out = {
        "provenance": prov,
        "checkpoint": str(ckpt),
        # WHICH TOKENIZER SCORED IT. An adapter directory borrows the base's (see
        # tokenizer_source); two runs of this command over one checkpoint under two
        # tokenizers would report two NLLs and nothing else would say why.
        "tokenizer": tok_src,
        "dev_sft": str(a.dev_sft),
        "dev_pairs": str(a.dev_pairs),
        "n_sft_rows": len(sft),
        "n_pairs": len(pairs),
        "chat_template": bool(a.chat_template),
        "enable_thinking": bool(a.enable_thinking),
        "sft_nll": eo.sft_nll(sft, render=render, logprob=logprob),
        # `default=None` on an `append` action, resolved here rather than in the parser:
        # argparse appends to a MUTABLE default, so a list default accumulates across parses.
        "stop_confusion": eo.stop_confusion(
            sft,
            render=render,
            logprob=logprob,
            reference_ask=a.reference_ask,
            condition_fields=tuple(a.stop_condition_field or ("done_before",)),
        ),
        "pair_accuracy": eo.pair_accuracy(pairs, render=render, logprob=logprob),
    }
    text = json.dumps(out, indent=2, sort_keys=True)
    Path(a.out).write_text(text + "\n")
    print(text)
    return 0


def cmd_train_gate(a: argparse.Namespace) -> int:
    """Tier B: the online dev gate. Non-zero exit on any failed criterion, by design.

    `EmptyArm` (T19d), `IncompleteArm` (2026-09-17), `UnscoredArm` (2026-09-17) and
    `UnderfilledArm` (2026-09-17) are caught here, not inside `run_gate`, for the same reason
    every other refusal in this file is a CLI-level `except`: the trainer module raises a
    plain exception by NAME (contract 4 forbids a static `pinq_train` import here), and this
    is the one place that both knows the shape of a terminal-friendly refusal and holds
    `--out`, so it is also the one place that can guarantee nothing is written there on either
    path.
    """
    gate = _train("gate")
    try:
        verdict = gate.run_gate(
            parquet_dir=Path(a.parquet_dir),
            grid_name=a.grid_name,
            baseline_grid_names=tuple(a.baseline_grid_name or ()) or (a.grid_name,),
            checkpoint_arm=a.checkpoint_arm,
            baseline_arm=a.baseline_arm,
            baseline_model_id=a.baseline_model_id,
            checkpoint_model_id=a.checkpoint_model_id,
            length_margin=float(a.length_margin),
            distinct_floor=float(a.distinct_floor),
            scorer_hash=a.scorer_hash,
            bootstrap_seed=int(a.bootstrap_seed),
            n_resamples=int(a.n_resamples),
            coverage_rule=a.coverage_rule,
            grids_root=Path(a.grids_root) if a.grids_root else None,
        )
    except gate.EmptyArm as exc:
        print(f"GATE REFUSED: {exc}", file=sys.stderr)
        return 2
    except gate.IncompleteArm as exc:
        print(f"GATE REFUSED: {exc}", file=sys.stderr)
        return 2
    except gate.UnscoredArm as exc:
        print(f"GATE REFUSED: {exc}", file=sys.stderr)
        return 2
    except gate.UnderfilledArm as exc:
        print(f"GATE REFUSED: {exc}", file=sys.stderr)
        return 2
    text = json.dumps(verdict, indent=2, sort_keys=True)
    Path(a.out).write_text(text + "\n")
    print(text)
    return 0 if verdict["passed"] else 1


def _register_track_d(s: argparse._SubParsersAction) -> None:
    """Attach `eval-offline` and `gate`. Called from `register()`; kept here so track D is one
    contiguous block and a merge with another track touches no shared line."""
    eo = s.add_parser(
        "eval-offline",
        help="Tier A: dev NLL, STOP 2x2 and pair accuracy for a checkpoint (no rollouts)",
    )
    eo.add_argument("--checkpoint", required=True, help="a local model/adapter directory")
    eo.add_argument("--dev-sft", required=True, help="data/rl/dev/sft.dev.jsonl")
    eo.add_argument("--dev-pairs", required=True, help="data/rl/dev/pairs.dev.jsonl")
    eo.add_argument("--out", required=True, help="where the JSON verdict is written")
    eo.add_argument(
        "--reference-ask",
        default=None,
        help="the ASK string a STOP row is compared against. Without it, STOP rows are "
        "skipped and counted rather than scored against nothing.",
    )
    eo.add_argument(
        "--stop-condition-field",
        action="append",
        default=None,
        help="repeatable: WHICH GOLD FACT the stop 2x2 is keyed on. Default `done_before`, "
        "pooled required-evidence completeness, which is what every verdict written before this "
        "flag existed means. `answer_node_covered_before` keys it on the ANSWER-BEARING node "
        "instead (lane L1.11's rule). The rows are scored ONCE and bucketed per field, so a "
        "second table costs nothing; the FIRST field named also fills the flat "
        "`p_stop_given_done` / `p_ask_given_not_done` keys, so old readers keep working and "
        "keep meaning what they did. A row lacking a named field is skipped and counted under "
        "that condition only -- unknown is not 'not done'.",
    )
    eo.add_argument("--chat-template", action="store_true", default=True)
    eo.add_argument("--no-chat-template", dest="chat_template", action="store_false")
    eo.add_argument("--enable-thinking", action="store_true", default=False)
    eo.add_argument(
        "--registry",
        default=None,
        help="the checkpoint registry to read (default: $PI_CHECKPOINTS, else "
        "conf/checkpoints.json). Its row supplies base_model, rung and init_from for the "
        "provenance block.",
    )
    eo.add_argument(
        "--registry-name",
        default=None,
        help="the registry row these weights are claimed to be. REFUSES unless the row's "
        "adapter_sha equals sha256(adapter_model.safetensors) of --checkpoint: a wrong name "
        "would put a wrong base model and a wrong lineage in the output and both would look "
        "right. Omitted, the block still records every registry name whose adapter_sha "
        "matches these weights.",
    )
    eo.set_defaults(fn=cmd_train_eval_offline)

    g = s.add_parser(
        "gate",
        help="Tier B: the online dev gate over compacted parquet; exits 1 on any failure",
    )
    g.add_argument("--parquet-dir", required=True, help="the `pi compact` output directory")
    g.add_argument("--grid-name", required=True, help="the grid the CHECKPOINT ran under")
    g.add_argument(
        "--baseline-grid-name",
        action="append",
        default=None,
        help="the grid(s) the BASELINE ran under; repeatable. Defaults to --grid-name. The "
        "dev baselines carry grid_name dev_baseline_* while the checkpoint carries "
        "dev_select_*, so one name cannot select both.",
    )
    g.add_argument(
        "--grids-root",
        default=None,
        help="directory of grid YAML files (e.g. conf/grids), read to check the checkpoint "
        "arm's paired task count against its own grid's declared n_tasks/task_ids, and to "
        "stamp k_matches_baseline. Optional: omitted, both stamps read null with a reason "
        "and neither check raises.",
    )
    g.add_argument("--checkpoint-arm", default="inquirer_trained")
    g.add_argument("--baseline-arm", default="inquirer_prompted")
    g.add_argument(
        "--baseline-model-id",
        default=None,
        help="restrict the baseline to runs whose INQUIRER was this model. Resolved through "
        "calls.parquet (actor='inquirer', model), because runs.parquet carries only "
        "model_pin_hash and no model name.",
    )
    g.add_argument(
        "--checkpoint-model-id",
        default=None,
        help="restrict the checkpoint side to runs whose INQUIRER was this model, the mirror "
        "of --baseline-model-id. Without it, several checkpoints compacted into one store "
        "under the same --checkpoint-arm and --grid-name (the shape a shared isolated store "
        "produces) are pooled into one average. Resolved through calls.parquet "
        "(actor='inquirer', model), same as the baseline side.",
    )
    g.add_argument("--length-margin", type=float, default=0.10)
    g.add_argument("--distinct-floor", type=float, default=0.65)
    g.add_argument(
        "--scorer-hash",
        default=None,
        help="pin the scorer. Default: the modal hash among the selected runs, printed.",
    )
    g.add_argument(
        "--coverage-rule",
        choices=("cap8", "matched_cost"),
        default="matched_cost",
        help="what the COVERAGE criterion subtracts (D9). matched_cost (default): the "
        "baseline's OWN PREFIX at the trained run's own stop k, read from scores.frontier_q#k "
        "-- legitimate because Inquirer.act(s) is budget-blind, so the first k questions of a "
        "cap-8 run are what a cap-k run would have asked. cap8: the baseline's whole cap-8 "
        "episode, which charges a checkpoint for stopping early and credits it nothing for the "
        "cost it saved. The cap-8 delta is reported under either rule; only the selected one is "
        "gated, and the verdict records which in `coverage_rule`. matched_cost also moves the "
        "criteria that fail for the same confound: facet_breadth and cad_ge2 become report-only "
        "(`facet_rule`, `cad_rule`), length_equivalence becomes one-sided non-inferiority "
        "(`length_rule`), and the STOP 2x2 becomes report-only against the base that never "
        "stops (`stop_rule`). Every cell and interval stays in the JSON; cap8 changes none of "
        "them. See docs/TRAINING.md 5.3.2.",
    )
    g.add_argument("--bootstrap-seed", type=int, default=0)
    g.add_argument("--n-resamples", type=int, default=1000)
    g.add_argument("--out", required=True, help="where the JSON verdict is written")
    g.set_defaults(fn=cmd_train_gate)


# --- end track D ---
