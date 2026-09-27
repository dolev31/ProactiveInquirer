"""The reward every rung optimises. One definition, imported by rungs 1, 2 and 3.

    R = w_task*Q(E_K)
      + sum_t [ w_phi*phi_tilde_t - w_red*rho_t ]
      + w_stop*sigma(tau)
      - c_ret*N_ret - c_tok*T/1000 - c_lat*L/60 - lambda_fmt*N_malformed

Four properties are load-bearing. Each has a test in tests/test_train_reward.py, and each
exists because the obvious alternative is gameable.

1. PHI IS OVER RETRIEVED EVIDENCE, NEVER OVER QUESTION TEXT. `Phi(s_t)` is need-coverage over
   the EvidenceUnits the retriever actually returned, which the server computes and this
   module only weights. A policy therefore cannot move phi by rewriting a question to look
   more like a gold need — it can only move it by causing different documents to come back.
   Had phi been a matcher score over question strings, the optimal policy would be a
   paraphrase generator, and it would have scored well on the metric the paper reports.

2. PHI IS POTENTIAL-BASED: phi_tilde_t = Phi(s_{t+1}) - Phi(s_t). Potential-based shaping
   (Ng, Harada & Russell 1999) leaves the optimal policy unchanged, so the shaping term
   cannot invent a preference that the task reward does not already have; it only makes the
   credit arrive at the turn that earned it instead of at the end of the episode.

3. THE SHAPING SUM IS CAPPED AT Q(E_K) - Q(E_0). Splitting one useful question into five
   fragments must not pay five times. With Phi = Q the differences telescope and the cap is
   already tight; the renormalisation is what keeps the guarantee when Phi and Q are
   different scales (coverage vs answer quality), which is the configuration a later rung is
   most likely to try. Fragmentation is then strictly punished, because each fragment still
   pays c_ret.

4. THE STOP TERM CANNOT BE BOUGHT. c_ret alone makes "STOP at t=0" the cheapest policy in
   the space, so a stop term is needed to punish undershoot — but a naive "+1 for stopping"
   pays a policy that asks one junk question and stops. sigma is therefore 0 unless the
   episode's questions, on average, cleared tau: stopping is rewarded only for a policy that
   was doing something worth stopping.

WHAT IS DELIBERATELY NOT HERE. `phi_LOO` is a leave-one-out value and is NOT available to a
deployed policy, so training on it trains on an oracle; the server returns it for diagnostics
and this module ignores it. `usd` is not a term at all: it is a function of a price table
that changes under the experiment. `c_lat` defaults to 0.0 for the same family of reasons —
wall time is a machine artifact (see CONTRIBUTING.md), and a reward that depends on it produces a
checkpoint that cannot be reproduced on a different host.

WHY THE SERVER DOES NOT DO THIS. `pi_run` may not import `pinq_train` (contract 4), so the
weights physically cannot live server-side. That constraint is also the right design:
re-weighting a finished rollout set is arithmetic over stored numbers, exactly as re-pricing
a sweep is arithmetic over stored token counts. A server-computed reward would make every
weight change a re-score of every episode.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import asdict, dataclass, replace
from typing import Sequence

from pinq.ids import canon, h
from pinq.wire import ScoreResponse

# The stop reason that means the POLICY decided. Anything else (budget, max_turns, error) is
# the harness deciding, and paying a policy for the cap the experiment announced is how STOP
# becomes confounded with the budget — the confound pinq.types.State exists to prevent.
POLICY_STOP = "policy_stop"


@dataclass(frozen=True, slots=True)
class RewardWeights:
    """Every coefficient in one auditable object, with `sha` for the manifest.

    `tau` is not a free knob: it is the formalism's own stopping threshold, estimated as the
    35th percentile of pilot `phi_LOO` (`tau_from_pilot`), and `c_ret` is set equal to it.
    That identity is the point — the price of a retrieval call IS the value below which the
    formalism says to stop, so a policy that asks a question worth less than tau loses more
    than it gains, without any extra tuning.
    """

    w_task: float = 1.0
    w_phi: float = 0.5
    w_red: float = 0.1
    w_stop: float = 0.05
    c_ret: float = 0.05
    c_tok: float = 0.002  # per 1,000 tokens
    c_lat: float = 0.0  # per minute; 0.0 on purpose — see the module docstring
    lambda_fmt: float = 0.25
    tau: float = 0.05
    # Pay for the fraction of REMAINING coverage a turn captured, not of the whole. OFF by
    # default: it changes the scale of phi_tilde, so `tau`/`c_ret` must be re-estimated from
    # headroom-normalised pilot values. See `headroom_potential` for the measurement that
    # motivates it and for why the obvious phi/(1-Q) form is unsound.
    headroom_normalised: bool = False

    @property
    def sha(self) -> str:
        """Provenance: a stored reward value must name the weights that produced it."""
        return h("reward", canon(asdict(self)))

    def validate(self) -> None:
        neg = {k: v for k, v in asdict(self).items() if v < 0}
        if neg:
            raise ValueError(
                f"negative reward coefficients {neg}: signs are already in the formula, so a "
                "negative weight silently flips a term's meaning."
            )
        if self.c_ret != self.tau:
            raise ValueError(
                f"c_ret={self.c_ret} != tau={self.tau}. The retrieval price IS the stopping "
                "threshold; decoupling them means the policy's stop rule and its cost model "
                "disagree, and neither one is the formalism's."
            )

    @classmethod
    def from_pilot(cls, phi_loo_values: Sequence[float], **over: float) -> "RewardWeights":
        t = tau_from_pilot(phi_loo_values)
        return replace(cls(**over), tau=t, c_ret=t)  # type: ignore[arg-type]


def tau_from_pilot(values: Sequence[float], *, q: float = 0.35) -> float:
    """tau := the q-th percentile of pilot phi_LOO. Pilot ids are burned for exactly this
    reason: data used to choose a threshold cannot also be used to test it."""
    vals = sorted(float(v) for v in values)
    if not vals:
        raise ValueError("no pilot phi_LOO values: tau must be measured, never guessed")
    idx = min(len(vals) - 1, max(0, int(round(q * (len(vals) - 1)))))
    return vals[idx]


# --------------------------------------------------------------------------- shaping


def shaped_phi(potential: Sequence[float], *, cap: float | None = None) -> tuple[float, ...]:
    """Potential-based, non-negative, and capped so fragmentation cannot pay twice.

    `cap` defaults to `potential[-1] - potential[0]`, which is `Q(E_K) - Q(E_0)` when the
    server's Phi is the coverage Q — the usual case, where the differences telescope and the
    cap binds with equality. Passing a different cap is what keeps the guarantee when Phi and
    Q are measured on different scales.
    """
    p = [float(x) for x in potential]
    if len(p) < 2:
        return ()
    raw = [max(0.0, b - a) for a, b in zip(p, p[1:])]
    total = sum(raw)
    limit = (p[-1] - p[0]) if cap is None else float(cap)
    limit = max(0.0, limit)
    if total > limit and total > 0.0:
        scale = limit / total
        return tuple(x * scale for x in raw)
    return tuple(raw)


def headroom_potential(potential: Sequence[float], *, eps: float = 1e-6) -> tuple[float, ...]:
    """Psi = -log(1 - Q): a potential whose increments are the HEADROOM-normalised gain.

    WHY THIS EXISTS. `phi_tilde` is a gain over a FIXED gold set, so a later question
    mechanically has less left to find, and the reward pays it less for being late. MEASURED
    over 361 non-STOP decision points on 13 tasks (the SFT export joined to matches.parquet
    and gold depth):

        Spearman(value, turn_idx)                = -0.643
        Spearman(value, gold depth)              = +0.208   95% CI [+0.065, +0.483]
        Spearman(value, depth | turn fixed)      = +0.507   95% CI [+0.241, +0.751]
        mean value  turn 0 -> turn 5             = +0.2016 -> +0.0045   (45x)
        mean value  depth 0 -> depth 2 at turn 0 = +0.115  -> +0.228    (2x)

    So depth is ALREADY rewarded at rho ~ +0.5 once turn is held fixed, and an "ask early"
    gradient roughly 20x larger points the other way. A latent need is reachable only AFTER
    the evidence that reveals it, so the two oppose by construction and a policy sees the
    diluted +0.208. The denominator is the cause, not a missing depth term.

    WHY NOT phi/(1 - Q), THE OBVIOUS FORM. Dividing each increment by the remaining headroom
    breaks both load-bearing properties in this module's docstring: property 2 requires the
    shaping term to be Phi(s_{t+1}) - Phi(s_t) for SOME Phi, and a ratio of differences is
    not one, so Ng-Harada-Russell invariance is lost and the shaping can invent a preference
    the task reward does not have; property 3's telescoping cap then stops binding, and
    splitting one question into five pays five times again.

    -log(1 - Q) keeps both, because it is a potential. Its increment is

        Psi(s_{t+1}) - Psi(s_t) = -log(1 - (Q_{t+1} - Q_t) / (1 - Q_t))

    a strictly increasing function of the fraction of REMAINING coverage the turn captured.
    `shaped_phi`'s default cap `p[-1] - p[0]` still binds with equality on the transformed
    ladder, so fragmentation is still unprofitable.

    THE UNITS CHANGE, and that is why this is opt-in and rides in `RewardWeights.sha`. `tau`
    is the 35th percentile of pilot phi_LOO and `c_ret == tau`; under this transform both
    must be re-estimated from headroom-normalised pilot values (`RewardWeights.from_pilot`
    does that automatically, since it takes whatever values it is given).

    `eps` clamps Q into [0, 1 - eps]: -log(0) is +inf, and a reward that depends on a clip
    that is not there is a reward that returns inf on a perfect episode. A scorer emitting a
    Q outside [0, 1] is broken, and log() would raise rather than say so.
    """
    out: list[float] = []
    for q in potential:
        x = min(max(float(q), 0.0), 1.0 - eps)
        out.append(-math.log(1.0 - x))
    return tuple(out)


def redundancy(n_retrieved: int, n_new: int) -> float:
    """rho_t in [0, 1]: the share of this turn's retrieval that was already held.

    Computed from `Turn.retrieved_uids` / `Turn.new_uids`, the objective record of what came
    back — never from the policy's own account of what it was asking about.
    """
    if n_retrieved <= 0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - (n_new / n_retrieved)))


def stop_indicator(
    *,
    stop_reason: str,
    phi: Sequence[float],
    tau: float,
) -> float:
    """sigma(tau) in {-1, 0, +1}. See property 4 in the module docstring.

     0   the harness stopped the episode (budget / max_turns / error). Not the policy's
         decision, so it is not the policy's credit.
     0   no questions were asked. There is no evidence about stopping quality, and this is
         also what stops "STOP immediately" from collecting the bonus for free.
    -1   stopped while the last question was still returning more than tau: undershoot.
    +1   stopped after the returns flattened, by a policy whose questions on average cleared
         tau. Both halves are required, or one junk question buys the bonus.
    """
    if stop_reason != POLICY_STOP or not phi:
        return 0.0
    last = float(phi[-1])
    if last > tau:
        return -1.0
    return 1.0 if statistics.fmean(phi) >= tau else 0.0


# --------------------------------------------------------------------------- the reward


@dataclass(frozen=True, slots=True)
class TurnReward:
    turn_idx: int
    phi_tilde: float
    rho: float

    @property
    def shaped(self) -> float:
        return self.phi_tilde - self.rho


@dataclass(frozen=True, slots=True)
class RewardBreakdown:
    """Every term, kept separately. `total` is a sum of the fields below it and nothing else.

    Storing the breakdown rather than the scalar is what makes a re-weighting a spreadsheet
    operation instead of a re-run, and what lets a training curve be decomposed after the
    fact into "it learned to cover more" versus "it learned to stop sooner".
    """

    episode_id: str
    total: float
    task_term: float
    shaping_term: float
    redundancy_term: float
    stop_term: float
    retrieval_cost: float
    token_cost: float
    latency_cost: float
    malformed_cost: float
    q_terminal: float
    q_initial: float
    sigma: float
    turns: tuple[TurnReward, ...] = ()
    weights_sha: str = ""
    scorer_hash: str = ""
    graph_version: str = ""

    def as_dict(self) -> dict[str, object]:
        d = asdict(self)
        d["turns"] = [asdict(t) for t in self.turns]
        return d


def reward_of(resp: ScoreResponse, weights: RewardWeights | None = None) -> RewardBreakdown:
    """Score components -> one reward. Pure arithmetic; no I/O, no gold, no network."""
    w = weights or RewardWeights()
    w.validate()
    if not resp.ok or not resp.supported:
        raise ValueError(
            f"{resp.episode_id}: the scorer did not produce a usable measurement "
            f"(ok={resp.ok}, supported={resp.supported}, note={resp.note!r}). Rewarding an "
            "unsupported response would put a fabricated number into a gradient."
        )
    ladder = headroom_potential(resp.potential) if w.headroom_normalised else resp.potential
    phi = shaped_phi(ladder)
    rhos = [redundancy(t.n_retrieved, t.n_new) for t in resp.turns]
    # A response whose turn list and potential ladder disagree is a corrupt measurement, not
    # something to zip-and-truncate: silently dropping the tail would silently drop reward.
    if resp.potential and len(phi) != len(resp.turns):
        raise ValueError(
            f"{resp.episode_id}: {len(resp.turns)} turns but a potential ladder of "
            f"{len(resp.potential)} points ({len(phi)} steps). The scorer and the trajectory "
            "disagree about how many decisions were made."
        )
    turns = tuple(
        TurnReward(turn_idx=t.turn_idx, phi_tilde=p, rho=r)
        for t, p, r in zip(resp.turns, phi, rhos)
    )

    q0 = float(resp.potential[0]) if resp.potential else 0.0
    sigma = stop_indicator(stop_reason=resp.stop_reason, phi=phi, tau=w.tau)

    task_term = w.w_task * resp.q_terminal
    shaping_term = w.w_phi * sum(phi)
    redundancy_term = w.w_red * sum(rhos)
    stop_term = w.w_stop * sigma
    retrieval_cost = w.c_ret * resp.n_ret
    token_cost = w.c_tok * (resp.tok_total / 1000.0)
    latency_cost = w.c_lat * (resp.wall_ms / 60_000.0)
    malformed_cost = w.lambda_fmt * resp.n_malformed

    total = (
        task_term
        + shaping_term
        - redundancy_term
        + stop_term
        - retrieval_cost
        - token_cost
        - latency_cost
        - malformed_cost
    )
    return RewardBreakdown(
        episode_id=resp.episode_id,
        total=total,
        task_term=task_term,
        shaping_term=shaping_term,
        redundancy_term=redundancy_term,
        stop_term=stop_term,
        retrieval_cost=retrieval_cost,
        token_cost=token_cost,
        latency_cost=latency_cost,
        malformed_cost=malformed_cost,
        q_terminal=resp.q_terminal,
        q_initial=q0,
        sigma=sigma,
        turns=turns,
        weights_sha=w.sha,
        scorer_hash=resp.scorer_hash,
        graph_version=resp.graph_version,
    )


@dataclass(frozen=True, slots=True)
class ValueRow:
    """One decision point, ready for `pinq_train.export.dataset`.

    `value` is the per-turn credit the exporter ranks candidates by: the shaped phi net of
    redundancy and of the retrieval call it cost. It is the PREFIX MARGINAL, which is legal
    as a training reward (the bias is shared within a state's candidate group) and illegal as
    a reported metric — see pi_eval.metrics.qvalue for why the paper reports phi_LOO instead.
    """

    turn_idx: int
    value: float
    phi_tilde: float
    rho: float


def turn_values(resp: ScoreResponse, weights: RewardWeights | None = None) -> list[ValueRow]:
    w = weights or RewardWeights()
    br = reward_of(resp, w)
    return [
        ValueRow(
            turn_idx=t.turn_idx,
            value=w.w_phi * t.phi_tilde - w.w_red * t.rho - w.c_ret,
            phi_tilde=t.phi_tilde,
            rho=t.rho,
        )
        for t in br.turns
    ]


DEFAULT_WEIGHTS = RewardWeights()
