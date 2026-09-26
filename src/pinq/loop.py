"""run_loop: the ONLY place ASK/STOP is sequenced.

Keeping this in one short function is what makes the whole system auditable: every arm,
every ablation and every control differs only in which Inquirer/Drafter is passed in.
"""

from __future__ import annotations

from typing import Sequence

from .budget import BudgetExceeded, BudgetLedger
from .protocols import Actuator, Answerer, Drafter, Inquirer, Retriever
from .types import (
    Ask,
    EvidenceUnit,
    Outcome,
    State,
    Stop,
    StopReason,
    TaskView,
    Trajectory,
    Turn,
)


def run_loop(
    *,
    view: TaskView,
    inquirer: Inquirer,
    retriever: Retriever,
    drafter: Drafter,
    answerer: Answerer,
    ledger: BudgetLedger,
    actuator: Actuator | None = None,
    max_turns: int = 16,
    k: int = 5,
    seed: int = 0,
    allow_user_target: bool = False,
    draft_every_turn: bool = True,
    branch_at: int | None = None,
    branch_seed: int | None = None,
) -> Trajectory:
    """`branch_at`/`branch_seed` fork this rollout at a turn: rung 2's candidate emission.

    From turn `branch_at` onward the Inquirer is reset with `branch_seed`, so turns before it
    are byte-identical to the unbranched run and everything after is a different continuation
    of the SAME state. That is the object a preference pair is defined over.

    THE PREFIX IS RE-EXECUTED, NOT REPLAYED. Handing the loop a State rebuilt from a recorded
    trajectory would put the prefix's usage in the trajectory and not in the ledger, and
    `reconcile` compares exactly those two sums -- so every branch would die on
    ReconcileError. Re-executing costs approximately nothing (the requests are byte-identical,
    so they are cache hits) and keeps usage attribution, the budget frontier and the
    subset_hash chain true by construction.

    Only the INQUIRER is re-seeded. `Drafter.draft()` must stay pure in
    (view, evidence.subset_hash, seed) -- phi_LOO, the prefix ladder and the stop test are all
    undefined without it -- so the drafting seed is the run's, not the branch's.
    """
    if branch_seed is not None and branch_at is None:
        raise ValueError(
            "branch_seed without branch_at: the fork point is what makes a branch a branch. "
            "Ignoring it would write a run labelled as a candidate that is not one."
        )
    inquirer.reset(view, seed)
    state = State(view=view)
    stop_reason: StopReason = "max_turns"
    # Advanced to `ledger.n_calls` every time a Turn is stamped, so it always points at the
    # end of the last ATTRIBUTED window. Whatever follows -- the act() that returned STOP, an
    # act() that broke on a budget or a malformed action, the final draft, the Answerer -- is
    # terminal usage rather than usage attributed to nothing.
    terminal_mark = ledger.n_calls

    for i in range(max_turns):
        ledger.set_turn(i)
        if branch_at is not None and i == branch_at and branch_seed is not None:
            # THE FORK. Everything before this line ran on the run's own seed and therefore
            # reproduces the parent exactly; everything after is the candidate continuation.
            inquirer.reset(view, branch_seed)
        mark = ledger.n_calls  # window start: everything charged from here belongs to turn i
        action = inquirer.act(state)  # sees State only: no budget, no turn hint

        if isinstance(action, Stop):
            stop_reason = "policy_stop"
            break

        if not isinstance(action, Ask):
            stop_reason = "error"
            break

        if action.target != "kb" and not allow_user_target:
            # The user channel is closed by construction in every confirmatory arm, and the
            # attempt is charged as a wasted turn. This is what makes "the Inquirer is not
            # eliciting user utterances" provable rather than asserted.
            ledger.record("rejected_user_asks", 1)
            state = state.with_turn(
                Turn(
                    turn_idx=i,
                    action=action,
                    subset_hash_before=state.evidence.subset_hash,
                    subset_hash_after=state.evidence.subset_hash,
                    usage=ledger.usage_since(mark),
                )
            )
            terminal_mark = ledger.n_calls
            continue

        try:
            ledger.charge_retrieval(1)
        except BudgetExceeded:
            stop_reason = "budget"
            break

        before = state.evidence.subset_hash
        units: Sequence[EvidenceUnit] = tuple(retriever.search(action.text, k))
        ledger.note_docs(u.doc_id for u in units)
        known = state.evidence.uids

        # THE ASK IS ANSWERED FROM WHAT THE ASK RETRIEVED. `state.evidence` here is the set held
        # BEFORE this turn, so resolving against it asked the Drafter to answer a question while
        # withholding the documents just fetched to answer it -- every answer one turn stale.
        #
        # MEASURED on 680 asks from 79 retail runs at 828a720: 55 asks that retrieved a unit came
        # back "No evidence has been retrieved ... so this cannot be answered", and on turn 0,
        # where nothing is yet held, 48% were refused that way. The refusal does not stay in the
        # run: `render_history` renders it into the Inquirer's next prompt as `A1: No evidence has
        # been retrieved`, so the live policy re-asks for what it already holds, and `state_text`
        # carries the same false record into every SFT row built from the run.
        #
        # `components.py::_queries` already states the contract from the other side -- "run_loop
        # already retrieved for the ask itself, and re-issuing it would spend a second call on the
        # same units" -- which is why the plain Drafter does not re-search. The loop simply never
        # handed the units over. The two halves have disagreed since d739909, the initial commit.
        asked = state.evidence.with_units(units)
        response, resolved = drafter.resolve(view, action, asked, seed=seed, ledger=ledger)
        merged = resolved

        # WHAT THIS ASK SURFACED, FROM BOTH RETRIEVAL PATHS.
        #
        # `retrieved_uids` used to be `tuple(u.uid for u in units)` -- the loop's own search and
        # nothing else -- while state evidence grew from `merged`, which also carries whatever
        # `drafter.resolve()` retrieved. `QueryExpansionDrafter._sub_retrieve` does exactly that,
        # charges the ledger for it, and `query_expansion` is a preregistered stage-1 arm.
        #
        # So the two records of "what this run retrieved" disagreed, and four consumers split
        # across the gap:
        #   * pi_eval.score's coverage/phi_LOO base is `retrieved | evidence` (both paths) while
        #     its budget-frontier ladder accumulates `retrieved_uids` (one path). Measured on one
        #     run with one sub-retrieved gold unit: evidence_coverage 1.0, terminal frontier_q
        #     0.5. The same run, the same definition of coverage, two numbers -- and the
        #     understatement lands ONLY on the arm whose mechanism is extra retrieval.
        #   * Trajectory.prefix() rebuilds evidence with `only(keep_uids)` from retrieved_uids,
        #     so prefix(len(turns)) was not the identity and phi_prefix_marginal understated the
        #     true marginal by whatever resolve() had found.
        #   * evidence_rows() stamps first_turn_idx from retrieved_uids, so sub-retrieved units
        #     were written with first_turn_idx = None, attributable to no turn on the spend axis.
        #   * reward.py and qvalue read it as ev(q_t) -- and the sub-queries were fired BECAUSE
        #     of this ask, so they belong to it.
        #
        # Fixed here rather than in each consumer, because the disagreement is not a scoring
        # choice: it is one fact recorded twice. Ordered loop-first so the sequence remains the
        # retriever's ranking (drgym relies on that), deduplicated, and byte-identical for every
        # Drafter whose _queries() is empty -- which is all of them but this one.
        # BEYOND the ask's own units, which `resolved` now also carries: without the second
        # clause every loop-retrieved uid would read as a sub-retrieval. `retrieved_uids` below
        # dedupes either way, but this keeps `sub_added` meaning what its name says.
        asked_uids = {u.uid for u in units}
        sub_added = tuple(u for u in resolved.uids if u not in known and u not in asked_uids)
        retrieved_uids = tuple(dict.fromkeys(tuple(u.uid for u in units) + sub_added))
        new_uids = tuple(u for u in retrieved_uids if u not in known)

        # D_t. The draft is refreshed INSIDE the loop so the NEXT act() sees it, and BEFORE
        # the Turn is stamped so its tokens land in this turn's usage window.
        #
        # This is not bookkeeping: s_t = (x, D_t, E_t, H_t) is the object the whole method is
        # defined over, and drafting only after the loop left `s.draft is None` on every call
        # to act(). The prompt rendered "(no draft yet)" every turn and 0 of 186 recorded
        # turns carried a draft_sha -- the Inquirer was interrogating a RETRIEVER, not a
        # Drafter, and the two-agent claim was not implemented at all.
        #
        # Drafting after the stamp was the SECOND half of the same mistake: the draft was
        # charged to the ledger and attributed to no turn, so Turn.usage under-reported by
        # about 40% (measured live: 30 ledger calls, 18 inside turns) and the budget frontier
        # -- whose x-axis is cumulative Turn.usage -- was plotted against the wrong spend.
        #
        # It costs one drafter call per turn. That is the mechanism, not an overhead: an
        # Inquirer that cannot see what the Drafter currently believes cannot ask the question
        # that belief makes necessary. `draft_every_turn=False` restores the old behaviour for
        # a cheap ablation, and is recorded on the Turn so a run made that way is identifiable.
        draft = drafter.draft(view, merged, seed=seed, ledger=ledger) if draft_every_turn else None

        state = state.with_turn(
            Turn(
                turn_idx=i,
                action=action,
                response_text=response,
                retrieved_uids=retrieved_uids,
                new_uids=new_uids,
                draft_sha=(draft.sha if draft is not None else ""),
                draft_text=(draft.text if draft is not None else ""),
                subset_hash_before=before,
                subset_hash_after=merged.subset_hash,
                # The REAL cost of this turn: the Inquirer's own call, whatever the Drafter
                # spent resolving it, and the redraft it triggered.
                usage=ledger.usage_since(mark),
            ),
            units=merged.units,
        )
        if draft is not None:
            state = state.with_draft(draft)
        terminal_mark = ledger.n_calls

    # Final draft. Recomputed unconditionally: draft() is pure in
    # (view, evidence.subset_hash, seed), so when the loop already produced one for this exact
    # evidence set the cache returns it and this costs nothing.
    draft = drafter.draft(view, state.evidence, seed=seed, ledger=ledger)
    state = state.with_draft(draft)

    env_calls: tuple = ()
    hashes: dict[str, str] = {}
    native: dict[str, float] = {}
    if actuator is not None:
        env_calls = tuple(actuator.execute(draft.tool_plan, turn_idx=len(state.history)))
        hashes = dict(actuator.final_hashes())
        native = dict(actuator.native())

    answer = answerer.answer(view, state.evidence, draft, seed=seed, ledger=ledger)

    return Trajectory(
        view=view,
        turns=state.history,
        evidence=state.evidence,
        outcome=Outcome(answer=answer, env_calls=env_calls, env_final_hashes=hashes, native=native),
        usage=ledger.usage,
        stop_reason=stop_reason,
        calls=tuple(ledger.calls),
        terminal_usage=ledger.usage_since(terminal_mark),
        final_draft=draft,
    )
