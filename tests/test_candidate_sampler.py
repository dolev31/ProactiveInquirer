"""The rung-2 candidate sampler: N actions at ONE state, so DPO has pairs to prefer between.

WHAT WAS BROKEN, MEASURED

  * `RolloutRequest.sampling` exists on the wire type and `_to_spec` DROPPED IT. A caller
    asking for temperature 1.0 silently got 0.0 and no error.
  * `worker.py` hardcoded `MeteredClient(ledger, temperature=0.0)`, so two calls at one
    state returned the identical action -- and the second was a cache hit, because the
    request bytes were identical.
  * `Turn.candidate_id` and `Turn.candidate_rank` exist end to end (pinq.types ->
    worker persistence -> pinq_train.export.dataset) and were NEVER ASSIGNED.

WHY THE SAMPLE INDEX RIDES ON `seed` AND NOT A NEW FIELD

`MeteredClient.request_payload` puts every extra kwarg into the payload, and
`complete` dispatches that payload verbatim -- so an invented `candidate_index` kwarg
would change the cache key AND be sent to the provider, which rejects it. `seed` is
already in the payload, already a real sampling control, and already part of the cache
key, so varying it per candidate changes the sample and the key coherently with no new
plumbing at the client boundary.
"""

from __future__ import annotations

import pytest

# --------------------------------------------------------------- the dropped sampling field


def test_rollout_request_sampling_reaches_the_unit_spec(tmp_path) -> None:
    from pi_run.serve.rollout import build_spec
    from pinq.wire import RolloutRequest

    corpus = tmp_path / "data" / "corpora" / "synth" / "deadbeef"
    corpus.mkdir(parents=True)
    (corpus / "tasks.jsonl").write_text("")

    req = RolloutRequest(
        task_id="t",
        suite_id="synth",
        sampling={"temperature": 0.9, "top_p": 0.95},
    )
    spec = build_spec(req, root=tmp_path, runs_root=tmp_path, cache_root=tmp_path)
    assert dict(spec.sampling) == {"temperature": 0.9, "top_p": 0.95}


def test_a_spec_with_no_sampling_is_greedy() -> None:
    """Greedy stays the default: every confirmatory arm depends on it."""
    from pi_run.worker import sampling_temperature

    assert sampling_temperature({}) == 0.0
    assert sampling_temperature({"top_p": 0.9}) == 0.0


def test_the_spec_temperature_is_what_the_client_gets() -> None:
    from pi_run.worker import sampling_temperature

    assert sampling_temperature({"temperature": 1.2}) == 1.2


# ------------------------------------------------------------------- candidate seeds


def test_candidate_seeds_are_distinct_and_derived_from_the_run_seed() -> None:
    from pinq.sampling import candidate_seed

    seeds = [candidate_seed(7, i) for i in range(8)]
    assert len(set(seeds)) == 8, "colliding seeds would make two candidates one cache entry"
    assert candidate_seed(7, 0) == 7, "candidate 0 IS the greedy/base rollout, unchanged"


def test_candidate_seeds_are_deterministic() -> None:
    from pinq.sampling import candidate_seed

    assert candidate_seed(3, 5) == candidate_seed(3, 5)


def test_different_run_seeds_do_not_collide() -> None:
    """Two runs of the same task must not share candidate cache entries."""
    from pinq.sampling import candidate_seed

    a = {candidate_seed(0, i) for i in range(16)}
    b = {candidate_seed(1, i) for i in range(16)}
    assert not (a & b) - {0, 1}, f"overlap: {sorted((a & b) - {0, 1})}"


def test_a_candidate_seed_changes_the_cache_key() -> None:
    """The whole point: without this the second sample is a cache hit on the first."""
    from pi_run.cache import CachingClient
    from pinq.sampling import candidate_seed

    class _Inner:
        model = "m"

        def request_payload(self, *, role, messages, seed, max_tokens=None, **kw):
            return {"model": role, "messages": list(messages), "seed": seed, **kw}

    c = CachingClient(_Inner())
    msgs = [{"role": "user", "content": "same state"}]
    keys = {c.key_for(role="inquirer", messages=msgs, seed=candidate_seed(0, i)) for i in range(4)}
    assert len(keys) == 4


# ------------------------------------------------------------------- candidate ids


def test_candidate_id_is_stable_for_a_state_and_index() -> None:
    from pinq.sampling import candidate_id

    a = candidate_id(run_id="r", turn_idx=2, index=1)
    assert a == candidate_id(run_id="r", turn_idx=2, index=1)


def test_candidate_id_separates_index_turn_and_run() -> None:
    from pinq.sampling import candidate_id

    base = candidate_id(run_id="r", turn_idx=2, index=1)
    assert base != candidate_id(run_id="r", turn_idx=2, index=2)
    assert base != candidate_id(run_id="r", turn_idx=3, index=1)
    assert base != candidate_id(run_id="q", turn_idx=2, index=1)


def test_rank_orders_by_score_descending_with_a_deterministic_tiebreak() -> None:
    """DPO needs a strict order: equal scores must not rank nondeterministically."""
    from pinq.sampling import rank_candidates

    cands = [
        {"candidate_id": "c", "score": 0.5},
        {"candidate_id": "a", "score": 0.9},
        {"candidate_id": "b", "score": 0.5},
    ]
    ranked = rank_candidates(cands)
    assert [c["candidate_rank"] for c in ranked] == [0, 1, 2]
    assert [c["candidate_id"] for c in ranked] == ["a", "b", "c"], "ties break on id"


def test_ranking_is_stable_under_input_order() -> None:
    from pinq.sampling import rank_candidates

    cands = [{"candidate_id": x, "score": 0.5} for x in ("b", "a", "c")]
    assert [c["candidate_id"] for c in rank_candidates(cands)] == ["a", "b", "c"]
    assert [c["candidate_id"] for c in rank_candidates(list(reversed(cands)))] == ["a", "b", "c"]


def test_a_nan_score_sorts_last_and_does_not_crash() -> None:
    """An ungraded candidate is not the best one."""
    from pinq.sampling import rank_candidates

    ranked = rank_candidates(
        [{"candidate_id": "a", "score": float("nan")}, {"candidate_id": "b", "score": 0.1}]
    )
    assert ranked[0]["candidate_id"] == "b"


def test_a_sampled_run_is_a_DIFFERENT_measurement_from_a_greedy_one(monkeypatch) -> None:
    """Temperature reaches `ModelPin.sampling_sha`, so it reaches run identity.

    This is what makes sampling safe to add: a temperature-0.9 rollout can never be
    confused with, resumed over, or cache-shared with the greedy confirmatory run of the
    same cell, because `pin.key` -> `model_pin_hash` -> `run_id` all move with it.
    """
    monkeypatch.setenv("PI_MODEL_INQUIRER", "openai/test-model")
    from pinq.budget import BudgetLedger
    from pinq_adapters.llm.litellm_client import MeteredClient

    greedy = MeteredClient(BudgetLedger(cap=None), temperature=0.0).pin("inquirer")
    sampled = MeteredClient(BudgetLedger(cap=None), temperature=0.9).pin("inquirer")

    assert greedy.sampling_sha != sampled.sampling_sha
    assert greedy.key != sampled.key


# ------------------------------------------------------- reconstructing the state at turn t


def _traj(n_turns: int = 3):
    """A trajectory whose turns carry distinguishable drafts and evidence."""
    from pinq.types import Ask, Evidence, EvidenceUnit, Outcome, TaskView, Trajectory, Turn, Usage

    units = [
        EvidenceUnit(
            uid=f"u{i}",
            corpus_id="synth",
            doc_id=f"d{i}",
            span=(0, 1),
            title=f"t{i}",
            text=f"text {i}",
        )
        for i in range(n_turns)
    ]
    view = TaskView(
        task_id="t",
        suite_id="synth",
        question="q?",
        instructions="",
        corpus_id="synth",
        corpus_hash="c",
        word_cap=30,
    )
    turns = []
    held: list = []
    for i in range(n_turns):
        before = Evidence.of(tuple(held)).subset_hash
        held.append(units[i])
        turns.append(
            Turn(
                turn_idx=i,
                action=Ask(text=f"ask {i}", rationale=""),
                response_text=f"resp {i}",
                retrieved_uids=(f"u{i}",),
                new_uids=(f"u{i}",),
                draft_text=f"D_{i + 1}",
                draft_sha=f"sha{i}",
                subset_hash_before=before,
                subset_hash_after=Evidence.of(tuple(held)).subset_hash,
            )
        )
    return Trajectory(
        view=view,
        turns=tuple(turns),
        evidence=Evidence.of(tuple(units)),
        outcome=Outcome(),
        usage=Usage(),
        stop_reason="policy_stop",
    )


def test_state_at_zero_has_no_draft_and_no_history() -> None:
    """`run_loop` starts with `State(view=view)`. Anything else is not what act() saw."""
    from pinq.sampling import state_at

    s = state_at(_traj(), 0)
    assert s.draft is None
    assert s.history == ()
    assert s.evidence.uids == frozenset()


def test_state_at_t_carries_the_PREVIOUS_turns_draft() -> None:
    """turns[i] carries D_{i+1}: the draft is computed after turn i's retrieval so the NEXT
    act() sees it. The same off-by-one that was live in cmd_train.render_state."""
    from pinq.sampling import state_at

    assert state_at(_traj(), 1).draft.text == "D_1"
    assert state_at(_traj(), 2).draft.text == "D_2"


def test_state_at_t_holds_exactly_the_evidence_recorded_before_that_turn() -> None:
    from pinq.sampling import state_at

    traj = _traj()
    for t in range(len(traj.turns)):
        assert state_at(traj, t).evidence.subset_hash == traj.turns[t].subset_hash_before


def test_state_at_t_refuses_when_the_reconstruction_disagrees() -> None:
    """Silently branching from the wrong state produces preference pairs for a decision that
    was never faced."""
    import dataclasses

    from pinq.sampling import StateMismatch, state_at

    traj = _traj()
    bad = dataclasses.replace(traj.turns[1], subset_hash_before="not-the-real-hash")
    traj = dataclasses.replace(traj, turns=(traj.turns[0], bad, traj.turns[2]))
    with pytest.raises(StateMismatch):
        state_at(traj, 1)


def test_state_at_t_history_is_the_preceding_turns() -> None:
    from pinq.sampling import state_at

    s = state_at(_traj(), 2)
    assert [t.turn_idx for t in s.history] == [0, 1]


def test_state_at_refuses_an_out_of_range_turn() -> None:
    from pinq.sampling import state_at

    with pytest.raises(IndexError):
        state_at(_traj(3), 3)


# ------------------------------------------------------------------- the branching driver


class _SeedEcho:
    """An Inquirer whose action records the seed it was reset with.

    Deterministic, LLM-free, and enough to prove the branch varies the sample: a real
    policy varies because the seed reaches `llm.complete(seed=...)`, which this stands in
    for exactly.
    """

    def __init__(self) -> None:
        self.seen: list[int] = []
        self._seed = 0

    def reset(self, view, seed: int) -> None:
        self._seed = seed
        self.seen.append(seed)

    def act(self, s):
        from pinq.types import Ask

        return Ask(text=f"question@seed={self._seed}", rationale="")


def test_branching_produces_n_distinct_candidates() -> None:
    from pinq.sampling import sample_actions

    inq = _SeedEcho()
    cands = sample_actions(_traj(), 1, inquirer=inq, run_seed=0, n=4, run_id="r")
    assert len(cands) == 4
    assert len({c["question"] for c in cands}) == 4, "every candidate is a different sample"


def test_candidate_zero_reproduces_the_base_rollout_seed() -> None:
    """candidate_seed(s, 0) == s, so branching does not silently re-run greedy under a new id."""
    from pinq.sampling import sample_actions

    inq = _SeedEcho()
    sample_actions(_traj(), 1, inquirer=inq, run_seed=7, n=3, run_id="r")
    assert inq.seen[0] == 7


def test_every_candidate_carries_its_id_rank_and_turn() -> None:
    from pinq.sampling import sample_actions

    cands = sample_actions(_traj(), 2, inquirer=_SeedEcho(), run_seed=0, n=3, run_id="r")
    assert all(c["turn_idx"] == 2 for c in cands)
    assert len({c["candidate_id"] for c in cands}) == 3
    assert sorted(c["candidate_index"] for c in cands) == [0, 1, 2]


def test_branching_reuses_ONE_reconstructed_state() -> None:
    """All candidates must face the SAME state, or they are not comparable and DPO's
    'same prompt, two completions' premise is false."""
    from pinq.sampling import sample_actions, state_at

    traj = _traj()
    cands = sample_actions(traj, 1, inquirer=_SeedEcho(), run_seed=0, n=3, run_id="r")
    want = state_at(traj, 1).evidence.subset_hash
    assert {c["subset_hash_before"] for c in cands} == {want}


def test_a_stop_candidate_is_recorded_not_dropped() -> None:
    """STOP is an action. Dropping it would make the sampler's pairs conditional on asking."""
    from pinq.sampling import sample_actions
    from pinq.types import Stop

    class _Stopper(_SeedEcho):
        def act(self, s):
            return Stop(reason="policy_stop")

    cands = sample_actions(_traj(), 1, inquirer=_Stopper(), run_seed=0, n=2, run_id="r")
    assert len(cands) == 2
    assert all(c["action"] == "stop" for c in cands)
    assert all(c["question"] == "" for c in cands)


def test_branching_refuses_a_state_it_cannot_reconstruct() -> None:
    import dataclasses

    from pinq.sampling import StateMismatch, sample_actions

    traj = _traj()
    bad = dataclasses.replace(traj.turns[1], subset_hash_before="wrong")
    traj = dataclasses.replace(traj, turns=(traj.turns[0], bad, traj.turns[2]))
    with pytest.raises(StateMismatch):
        sample_actions(traj, 1, inquirer=_SeedEcho(), run_seed=0, n=2, run_id="r")


def test_n_of_one_is_the_base_rollout_unchanged() -> None:
    from pinq.sampling import sample_actions

    inq = _SeedEcho()
    cands = sample_actions(_traj(), 1, inquirer=inq, run_seed=5, n=1, run_id="r")
    assert len(cands) == 1 and inq.seen == [5]
