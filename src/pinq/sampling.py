"""Candidate sampling primitives: N actions at ONE state, ranked, paired.

Rung 2 (DPO) needs two actions taken from the SAME state that scored differently. Nothing
here calls a model or touches the network -- these are the pure parts, so sampling and
ordering are testable without a provider and identical in a replay.

PAIRING LIVES IN `pinq_train.export.dataset.export_pairs`, NOT HERE. That function already
groups by (suite, task, run, turn), enforces the margin, and carries a LENGTH GUARD --
without which the preference model learns "longer question wins", the same confound that
makes a naive judge unusable, imported straight into the policy. A second pairing
implementation here would be two declarations of one rule, free to disagree; this module
produces the candidates it consumes and stops there.

WHY THE SAMPLE INDEX RIDES ON `seed`

`MeteredClient.request_payload` puts every extra kwarg into the payload and `complete`
dispatches that payload verbatim, so an invented `candidate_index` kwarg would change the
cache key AND be sent to a provider that rejects it. `seed` is already in the payload,
already a legitimate sampling control and already part of the cache key, so deriving the
candidate's seed from (run_seed, index) makes the sample and the key vary together with no
new field at the client boundary.

`pinq` is stdlib-only; this module keeps that.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping

from .ids import h
from .types import Ask, Draft, State, Trajectory

# 2**32 keeps the value inside what an OpenAI-compatible `seed` accepts.
_SEED_SPACE = 2**32


def candidate_seed(run_seed: int, index: int) -> int:
    """The seed for candidate `index` of a rollout run at `run_seed`.

    Candidate 0 IS the base rollout: `candidate_seed(s, 0) == s`, so sampling a state does
    not silently re-run the greedy trajectory under a different identity, and a one-candidate
    sample is byte-identical to no sampling at all.

    Derived by hashing rather than by `run_seed + index`, which collides across runs: seed 0
    candidate 1 and seed 1 candidate 0 would be the same request and share a cache entry, so
    two different runs' candidates would silently be one measurement.
    """
    if index == 0:
        return int(run_seed)
    return int(h("candidate-seed", str(int(run_seed)), str(int(index)))[:8], 16) % _SEED_SPACE


def candidate_id(*, run_id: str, turn_idx: int, index: int) -> str:
    """A stable identity for one candidate at one state.

    Includes the run and the turn, not just the index: `candidate_rank` is meaningful only
    WITHIN a state, and an id that repeated across turns would let a join collapse two
    different decisions into one.
    """
    return h("candidate", run_id, str(int(turn_idx)), str(int(index)))[:16]


def _sort_key(c: Mapping[str, Any]) -> tuple[int, float, str]:
    """(is_unscored, -score, id). NaN sorts LAST -- an ungraded candidate is not the best."""
    score = c.get("score")
    try:
        value = float(score)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return (1, 0.0, str(c.get("candidate_id", "")))
    if math.isnan(value):
        return (1, 0.0, str(c.get("candidate_id", "")))
    return (0, -value, str(c.get("candidate_id", "")))


def rank_candidates(candidates: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Best first, with `candidate_rank` stamped. Ties break on `candidate_id`.

    THE TIEBREAK IS NOT COSMETIC. Two candidates that scored identically are common -- the
    same question phrased twice retrieves the same evidence -- and a sort that left their
    order to the input would produce a different `candidate_rank` on every export, so the
    same dataset would hash differently and `train_id_set_hash` would stop identifying it.
    """
    ordered = sorted((dict(c) for c in candidates), key=_sort_key)
    for rank, c in enumerate(ordered):
        c["candidate_rank"] = rank
    return ordered


def _score_of(c: Mapping[str, Any]) -> float:
    try:
        return float(c.get("score"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")


class StateMismatch(RuntimeError):
    """The reconstructed state is not the state the policy faced."""


def state_at(traj: Trajectory, t: int) -> State:
    """s_t, rebuilt from a recorded trajectory, for branching a new decision at turn t.

    THE DRAFT COMES FROM turns[t-1], NOT turns[t]. `pinq.loop` computes the draft AFTER
    turn i's retrieval and stamps it on Turn i -- "refreshed INSIDE the loop so the NEXT
    act() sees it" -- so turns[i] carries D_{i+1}. Reading turns[t] would hand the policy a
    summary of the consequences of the action it is about to choose. The identical off-by-one
    was live in `pi_run.cmd_train.render_state`; the two reconstructions are kept honest by
    the same rule.

    Turn 0 gets `draft=None`, because `run_loop` opens with `State(view=view)` and the
    policy provably saw no draft there.

    VERIFIED, NOT ASSUMED. The rebuilt evidence must hash to the `subset_hash_before` the
    run recorded for turn t. Branching from a state the policy never faced produces
    preference pairs for a decision that was never made, and nothing downstream could tell.
    """
    if not 0 <= t < len(traj.turns):
        raise IndexError(f"turn {t} is outside the recorded trajectory of {len(traj.turns)}")

    prefix = traj.prefix(t)
    recorded = str(traj.turns[t].subset_hash_before or "")
    if recorded and prefix.evidence.subset_hash != recorded:
        raise StateMismatch(
            f"turn {t}: reconstructed evidence hashes to "
            f"{prefix.evidence.subset_hash[:12]} but the run recorded {recorded[:12]}. "
            "Branching here would sample actions for a state the policy never faced."
        )

    draft: Draft | None = None
    if t > 0:
        prior = traj.turns[t - 1]
        if prior.draft_sha and not prior.draft_text:
            raise StateMismatch(
                f"turn {t}: the preceding turn recorded a draft ({prior.draft_sha[:12]}) but "
                "carries no draft_text -- it predates that field. Branching would sample "
                "against a prompt the policy never saw."
            )
        if prior.draft_text:
            draft = Draft(text=prior.draft_text)

    return State(
        view=traj.view,
        evidence=prefix.evidence,
        draft=draft,
        history=tuple(traj.turns[:t]),
    )


def sample_actions(
    traj: Trajectory,
    t: int,
    *,
    inquirer: Any,
    run_seed: int,
    n: int,
    run_id: str,
) -> list[dict[str, Any]]:
    """N actions sampled at the ONE state the policy faced at turn `t`.

    THE BRANCHING DRIVER. The state is reconstructed once and reused for every candidate:
    DPO's premise is "one prompt, two completions", and re-deriving the state per candidate
    would let a bug make them face different prompts while still looking like a pair.

    The candidate index reaches the model through `reset(view, candidate_seed(run_seed, i))`
    and NOT through a new parameter on `act`. `Inquirer.act(s)` takes State and nothing else
    -- a budget- or index-aware policy makes prefix-k non-exchangeable with a true short run
    -- and the base policy already threads its reset seed into
    `llm.complete(seed=self._seed + self._n_asks)`. Since a branch calls `act` once, every
    candidate's request seed is exactly `candidate_seed(run_seed, i)`: distinct by
    construction, so distinct cache keys and distinct samples.

    A STOP is recorded like any other action. Dropping it would make every pair conditional
    on the policy having chosen to ask, which is the decision being studied.
    """
    state = state_at(traj, t)
    out: list[dict[str, Any]] = []
    for i in range(max(0, int(n))):
        seed = candidate_seed(run_seed, i)
        inquirer.reset(traj.view, seed)
        action = inquirer.act(state)
        out.append(
            {
                "candidate_id": candidate_id(run_id=run_id, turn_idx=t, index=i),
                "candidate_index": i,
                "turn_idx": t,
                "seed": seed,
                "action": "ask" if isinstance(action, Ask) else "stop",
                "question": action.text if isinstance(action, Ask) else "",
                "subset_hash_before": state.evidence.subset_hash,
            }
        )
    return out
