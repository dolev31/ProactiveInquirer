"""In what ORDER rung 1 shows its rows. A pure function over plain dicts, and nothing else.

WHAT THIS IS FOR. The headline recipe shuffles every epoch, which is the null: the policy sees
a depth-3 need and a depth-0 need with equal probability at step 0. This module is the one
alternative the plan asks for -- show the shallow needs first, or the deep ones first, for ONE
epoch -- so that "does the order matter" is a question with a measured answer rather than an
assumption baked into a shuffle nobody chose.

WHY IT LIVES OUTSIDE `train.py`. The ordering IS the intervention. If it were computed inside a
`Trainer` subclass it would be the one part of the ablation that no test on a laptop could
reach, which is the failure `train.py`'s own docstring is written against. Everything here is
a function of `(rows, curriculum, seed, epochs)`; `tests/test_curriculum.py` drives all of it
with no torch installed.

THE AXIS IS THE TARGET'S DEPTH, NOT THE STATE'S. A row supervises one action. For an ASK, the
row's `latent_depth` is the depth of the gold node that ask resolved -- the thing the policy had
to anticipate -- so it is the axis. For a STOP there is no such node: `latent_depth` on a STOP
row describes the state it stopped in, and reading it as "the depth of this row's ask" would
sort a class of rows by a number that means something else.

WHY THE STOP ROWS ARE DEALT IN PROPORTION AND NOT ONE-THIRD EACH. They have to be spread, or a
curriculum's first block would carry a different STOP share from the block after it and the arm
would be an ordering intervention AND a STOP-share schedule, reported as one number. An equal
three-way deal spreads them but does NOT equalise the share: with a_k ASK rows in stratum k and
S STOP rows dealt S/3 each, stratum k's STOP share is (S/3)/(a_k + S/3), which is the file's
share only when the depths are balanced -- and on the real corpus they are not. Dealing s_k =
S * a_k / A gives (S/A)/(1 + S/A) = S/(A+S) in every stratum, exactly, for any depth
distribution. So "uniformly at random" is the RANDOMNESS (a uniform permutation decides which
STOP row goes where, because a STOP row carries no depth signal to decide it by) and the
proportion is the invariant it is drawn under. `tests/test_curriculum.py::
test_every_stratum_carries_the_global_stop_share` pins this on a deliberately unbalanced
corpus: MEASURED there, the proportional deal lands every stratum within 0.00024 of the file's
0.400000, while the equal deal gives 0.286 / 0.444 / 0.571 -- off by as much as 0.171.

WHAT IS NOT AN ORDERING KEY. The suite. Rows of every domain are shuffled together inside a
stratum: a suite-blocked epoch would be a domain curriculum wearing a depth curriculum's name,
and the two could not be told apart afterwards from anything the checkpoint records.
"""

from __future__ import annotations

import random
from typing import Any, Mapping, Sequence

from pinq.actions import action_kind_of

SHUFFLED = "shuffled"
EASY_FIRST = "depth_easy_first"
HARD_FIRST = "depth_hard_first"
# THE NULL FIRST, and it is a real member rather than "no curriculum": the arm that shuffles is
# the arm the other two are read against, so it has a name, a spelling on the command line and
# a row in the table. `SFTConfig.curriculum` defaults to it.
CURRICULA = (SHUFFLED, EASY_FIRST, HARD_FIRST)

# S0 = depth 0 (and the -1 sentinel), S1 = depth 1, S2 = depth >= 2. THREE rather than one per
# depth because depth >= 3 is 4.6% of the ASK rows: a stratum that small is a block of a few
# hundred rows whose first-epoch position is decided by sampling noise.
N_STRATA = 3
# Which stratum each curriculum's FIRST epoch opens with. Epochs 2..n are fully shuffled under
# every curriculum -- see `order_examples`.
FIRST_EPOCH_ORDER: dict[str, tuple[int, ...]] = {
    EASY_FIRST: (0, 1, 2),
    HARD_FIRST: (2, 1, 0),
}


def _is_stop(row: Mapping[str, Any]) -> bool:
    """Is this row's SUPERVISED ACTION a STOP? Read off the action bytes, like `row_weight`.

    `pinq.actions.action_kind_of` is the policy parser's own normalisation, which is why
    `train.is_stop_row` calls it too: the exporter, the loop, the loss weighting and this
    ordering cannot then disagree about what a row is. It is reached from `pinq.actions` here
    rather than from `train.is_stop_row` because `train` imports THIS module; both are one call
    to the same function, so there is no second predicate to drift.

    NOT the `is_stop` FLAG. The flag is set beside the bytes rather than derived from them, and
    a row whose flag says STOP while its bytes say ASK would be dealt into an arbitrary stratum
    while the loss supervised a question -- the exact disagreement `export.dataset._is_stop_row`
    already refuses to trust. They agree on all 43,837 rows of `data/rl/sft.jsonl` today.
    """
    return action_kind_of(row.get("action_json")) == "stop"


def ask_stratum(row: Mapping[str, Any]) -> int:
    """The stratum of a row whose target is an ASK, from its `latent_depth`. Pure.

    -1 IS A SENTINEL, NOT A DEPTH. It means no required node was resolved at this turn, so
    there is no depth to order it by; it joins S0, where it is the same "nothing deeper was
    needed here" case as a genuine root. A missing key, a null, and anything that is not an
    integer are the same case -- an unreadable label is an unknown one, and defaulting it into
    S2 would put the rows nothing could label at the hard end of every curriculum.
    """
    try:
        depth = int(row.get("latent_depth", -1))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        depth = -1
    if depth <= 0:
        return 0
    return 1 if depth == 1 else 2


def _quotas(total: int, weights: Sequence[int]) -> list[int]:
    """Split `total` across the strata in proportion to `weights`, by largest remainder.

    INTEGERS, AND THEY SUM TO `total`. Rounding each share independently loses or gains rows,
    and a deal that does not conserve its input is an exposure change hiding inside an ordering
    change. The remainder goes to the largest fractional parts, ties to the lower stratum, so
    the split is a function of the counts alone.
    """
    if total <= 0:
        return [0] * len(weights)
    mass = sum(weights)
    if mass <= 0:
        # NO ASK ROW ANYWHERE -- a fully collapsed dataset, which `dataset_report` refuses
        # before this is ever reached. There is no ASK mass to be proportional to, so the deal
        # is as even as it can be; the alternative is a division by zero in a corner that the
        # mode-collapse gate has already rejected.
        weights = [1] * len(weights)
        mass = len(weights)
    exact = [total * w / mass for w in weights]
    out = [int(x) for x in exact]
    short = total - sum(out)
    for s in sorted(range(len(weights)), key=lambda s: (-(exact[s] - out[s]), s))[:short]:
        out[s] += 1
    return out


def _assign(rows: Sequence[Mapping[str, Any]], rng: random.Random) -> list[int]:
    strata = [0] * len(rows)
    stops: list[int] = []
    ask_counts = [0] * N_STRATA
    for i, row in enumerate(rows):
        if _is_stop(row):
            stops.append(i)
            continue
        s = ask_stratum(row)
        strata[i] = s
        ask_counts[s] += 1
    # WHICH stop row goes where is random; HOW MANY go where is not. See the module docstring.
    rng.shuffle(stops)
    at = 0
    for s, quota in enumerate(_quotas(len(stops), ask_counts)):
        for i in stops[at : at + quota]:
            strata[i] = s
        at += quota
    return strata


def assign_strata(rows: Sequence[Mapping[str, Any]], seed: int) -> list[int]:
    """Each row's stratum, 0/1/2, at this seed. Deterministic, and pure in `(rows, seed)`.

    THE SAME ASSIGNMENT `order_examples` USES, because it is the first thing drawn from the
    same `random.Random(seed)` stream. Exposed so the ordering can be checked against a stratum
    label computed outside it -- a test that asked the orderer where it put a row would be
    asking the thing under test.
    """
    return _assign(rows, random.Random(seed))


def order_examples(
    rows: Sequence[Mapping[str, Any]], curriculum: str, seed: int, epochs: int
) -> list[int]:
    """The index sequence the trainer walks: `epochs` passes over `rows`, in curriculum order.

    EXPOSURE-MATCHED BY CONSTRUCTION. Every index appears exactly `epochs` times under every
    curriculum, so the arms differ in ORDER and in nothing else. A schedule that re-weighted by
    repeating a stratum would be a different dataset, and the difference it bought in the
    checkpoint could not be attributed to ordering.

    ONLY THE FIRST EPOCH IS ORDERED. Epochs 2..n are full shuffles under every curriculum:
    repeating the ramp every epoch would confound "shallow first" with "shallow n times, each
    time right before the deep rows", and the LR schedule's tail would then always sit on one
    stratum. With the shipped `epochs=2` this is "one ordered epoch, then the null's epoch".

    DETERMINISTIC IN `seed`, and in nothing else. One `random.Random(seed)` stream draws the
    STOP deal first and then each epoch's shuffle, so two runs of one recipe produce one
    ordering -- an arm whose order moved between runs would be unreproducible while every
    recorded field still matched.
    """
    if curriculum not in CURRICULA:
        raise ValueError(
            f"curriculum={curriculum!r}: not one of {list(CURRICULA)}. Refused rather than "
            "defaulted: an unrecognised name would train the shuffled null arm under a "
            "cfg.sha that names an ablation, and the arm would report a difference of zero."
        )
    rng = random.Random(seed)
    strata = _assign(rows, rng)
    n = len(rows)
    out: list[int] = []
    for epoch in range(int(epochs)):
        if epoch == 0 and curriculum != SHUFFLED:
            for s in FIRST_EPOCH_ORDER[curriculum]:
                block = [i for i in range(n) if strata[i] == s]
                rng.shuffle(block)
                out.extend(block)
        else:
            block = list(range(n))
            rng.shuffle(block)
            out.extend(block)
    return out
