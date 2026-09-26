"""Human-centred metrics: what the Inquirer found that a person never asked for.

This replaces FutureConversationCoverage, which was not merely noisy but BACKWARDS. In a
logged conversation, U_3 exists BECAUSE of the assistant's turn-2 reply. A need the Inquirer
PREEMPTS therefore never appears as a later user utterance at all, so FCC scores a miss on
exactly the success case it was meant to reward. No amount of smoothing fixes a metric whose
sign is wrong.

What replaces it:
  * needs are annotated, not utterances: a need satisfied in the final answer counts as
    covered whether or not anyone ever "asked" it;
  * needs are partitioned into DISCOVERABLE-FROM-KB and USER-PRIVATE, because no autonomous
    inquirer can ever recover "I plan to move in three years". The private share is the
    CEILING on any such system and is the single most important number for the framing;
  * ADR measures what the system found that the human did not, rated for usefulness
    independently. That, not FCC, is what supports "exceeds what the user thought to ask".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from pi_eval.matcher.base import MatchRecord

_RANK = {"none": 0, "ask": 1, "resolve": 2, "use": 3}


@dataclass(frozen=True, slots=True)
class Ceiling:
    n_total: int
    n_kb: int
    n_private: int
    n_unknown: int

    @property
    def private_share(self) -> float:
        """The hard ceiling: the fraction of needs no autonomous inquirer could ever reach."""
        return self.n_private / self.n_total if self.n_total else float("nan")

    @property
    def attainable(self) -> float:
        return self.n_kb / self.n_total if self.n_total else float("nan")


def discoverability_ceiling(graph) -> Ceiling:
    kb = sum(1 for n in graph.gold_nodes if n.gold_discoverability == "kb")
    priv = sum(1 for n in graph.gold_nodes if n.gold_discoverability == "user_private")
    unk = sum(1 for n in graph.gold_nodes if n.gold_discoverability == "unknown")
    return Ceiling(len(graph.gold_nodes), kb, priv, unk)


def anticipated_discovery_rate(
    records: Sequence[MatchRecord], graph, *, level: str = "resolve"
) -> dict[str, float]:
    """ADR = |V_found \\ V_human| / |V_annotated|, over KB-discoverable needs.

    Restricted to KB-discoverable needs, because crediting a system for failing to read the
    user's mind — or for "finding" something only the user could have told it — measures
    nothing.

    WHY THE DENOMINATOR IS THE ANNOTATED SUBSET, AND WHY AN UNANNOTATED GRAPH RETURNS NaN.
    `V_human` is `gold_human_asked`, which is a HUMAN ANNOTATION and is `None` on every node
    this repository has built. The obvious implementation --

        beyond = [n for n in found if n.gold_human_asked is False]
        return len(beyond) / len(universe)

    -- filters `None is False` to nothing and returns a hard **0.0**. Not NaN, not an error:
    a clean, plausible, publishable zero that reads as "the system anticipated nothing the
    user asked for", on every task, whatever the policy did. It is the exact shape this
    codebase exists to refuse — a well-formed number computed from an absent measurement —
    and it would have gone into a table under a metric whose whole purpose is to support
    "exceeds what the user thought to ask".

    So: nodes with no annotation are not evidence of anything and are excluded from BOTH
    halves. With none annotated the metric is NOT COMPUTABLE and says so, and
    `pi_eval.score` emits no row rather than a zero. `n_annotated` is returned alongside so
    the denominator is never a mystery, and `annotation_coverage` says how much of the
    universe the estimate rests on. Likewise `mean_usefulness_beyond_human` rests on
    `n_rated` -- the count of `beyond` needs that actually carry a `gold_usefulness_rating`,
    which can be zero even when `n_beyond_human` is not, because a usefulness campaign can
    lag the `gold_human_asked` annotation it depends on.

    WHY THE ANTICIPATED SIDE IS RATED TOO. `mean_usefulness_beyond_human` on its own cannot
    support "exceeds what the user thought to ask": 4.5 out of 5 is a claim only against what
    the same annotators rated the needs they DID anticipate. So the reference level is
    computed here as `mean_usefulness_anticipated`, over `found AND gold_human_asked is
    True`. Both sides condition on `found`, which varies anticipation and holds discovery
    constant -- without that conditioning the contrast would also carry "the system found it
    at all", which is a different claim and one `rnr_resolve` already makes.

    This also constrains the ANNOTATION INSTRUMENT, not just the arithmetic: usefulness must
    be elicited on every shown need, never only on the unticked ones. Asking for a rating on
    exactly one branch makes that branch more work, and the cheap branch there is "I would
    have asked about this" -- which drives ADR toward 0.0, the null this metric exists to
    test. The cost of an answer may not depend on the answer.
    """
    want = _RANK[level]
    by_id = {r.node_id: r for r in records}
    universe = [
        n
        for n in graph.gold_nodes
        if n.gold_partition in ("required", "optional") and n.gold_discoverability == "kb"
    ]
    base = {
        "adr": float("nan"),
        "n_universe": len(universe),
        "n_annotated": 0,
        "n_found": 0,
        "n_beyond_human": 0,
        "annotation_coverage": float("nan"),
        "mean_usefulness_beyond_human": float("nan"),
        "n_rated": 0,
        "mean_usefulness_anticipated": float("nan"),
        "n_rated_anticipated": 0,
    }
    if not universe:
        return base

    annotated = [n for n in universe if n.gold_human_asked is not None]
    if not annotated:
        # Absent, and it must LOOK absent. See the docstring.
        base["annotation_coverage"] = 0.0
        return base

    found = [n for n in annotated if n.gold_node_id in by_id and by_id[n.gold_node_id].rank >= want]
    beyond = [n for n in found if n.gold_human_asked is False]
    rated = [n.gold_usefulness_rating for n in beyond if n.gold_usefulness_rating is not None]
    anticipated = [n for n in found if n.gold_human_asked is True]
    rated_anticipated = [
        n.gold_usefulness_rating for n in anticipated if n.gold_usefulness_rating is not None
    ]
    return {
        "adr": len(beyond) / len(annotated),
        "n_universe": len(universe),
        "n_annotated": len(annotated),
        "n_found": len(found),
        "n_beyond_human": len(beyond),
        "annotation_coverage": len(annotated) / len(universe),
        "mean_usefulness_beyond_human": (sum(rated) / len(rated)) if rated else float("nan"),
        "n_rated": len(rated),
        "mean_usefulness_anticipated": (sum(rated_anticipated) / len(rated_anticipated))
        if rated_anticipated
        else float("nan"),
        "n_rated_anticipated": len(rated_anticipated),
    }


def user_burden(ledger_spent: dict[str, float], n_turns: int) -> float:
    """Fraction of turns spent trying to ask the USER rather than the knowledge source.

    In the confirmatory configuration this must be 0 by construction — the loop rejects
    target='user' — so a non-zero value is a bug alarm, not a finding.
    """
    return ledger_spent.get("rejected_user_asks", 0.0) / n_turns if n_turns else 0.0


# --------------------------------------------------------------------------- the user channel

# `Ask.target` (pinq.types.Ask) is `{"kb","user"}` and reaches here on `turns.parquet`. The
# string is duplicated rather than imported for the same reason as the tau2 tool names in
# `metrics.environment`: the firewall forbids gold-side code importing the rollout package.
# `tests/test_user_private_ask.py` holds the two equal.
USER_TARGET = "user"

# What an ask can be ABOUT. `dropped` nodes are not annotated needs, so a question naming one
# is unattributed rather than misdirected.
_SCORABLE = ("required", "optional")


def user_private_ask(turns, graph) -> dict[str, float]:
    """Did the policy ask the USER only for what only the user knows, and for all of it?

        precision = P(the need is user-private | the policy asked the user)
        recall    = P(the policy asked the user | the need is user-private AND required)

    THIS IS THE ONLY PAIR THAT SEPARATES THE TWO CHANNELS. `rnr_ask`, `adr` and the coverage
    family are all indifferent to whether a question went to the retriever or to the customer,
    so none of them can support "proactive without harassing". `Ask.target` is what makes the
    distinction objective: it is set by the policy, gated by `run_loop(allow_user_target=...)`
    off the arm's own declaration, and written to `turns.parquet`. It is a record of where a
    question was sent, not an opinion about what it sounded like.

    THE ASK INSTRUMENT IS `matcher.base.asked_about` AND NOT A NEW RULE. That function already
    decides "did this question mention this need", under ASK_MIN_COVERAGE and ASK_MIN_TERMS,
    and those constants ride into matcher_hash -> scorer_hash. A second term-overlap rule here
    would be a second instrument answering the same question under a different threshold.

    WHY THE RUNGS ARE NOT USED. A `MatchRecord` reports the BEST rung a need reached, and a
    user-private need that happens to carry gold spans comes back "resolve" with
    `matched_turn_idx` pointing at the RETRIEVAL turn that fetched them. Joining on that turn
    would attribute a retrieval to the user channel. The questions are therefore read directly,
    per turn, which is also the only way to know which channel each one went down.

    THE HAZARD IS NOT LIVE AT THE OPERATIVE VERSION, AND THE DECISION IS KEPT ANYWAY.
    Measured: at v1 -- `pi_eval.score.DEFAULT_GRAPH_VERSION`, the version this code actually
    runs -- `0 of tau2_retail's 219` user-private needs carry `gold_ev_uids`, so no
    user-private need can come back "resolve" through a retrieval turn and the join above could
    not misattribute anything. The figure this paragraph used to cite, 146 of 252, is TRUE and
    is a v7 figure; v7 CANNOT BE LOADED (see below), so that number is unreachable through
    `load_graphs` and was reproducible only by parsing the raw JSONL. The rungs are still not
    used, because the reason is robustness to the version moving -- not a live failure. At v7
    the hazard applies to 146 of 252 nodes (58%), and a join written against v1's emptiness
    would break silently the day the default moved.

    ANY USER-PRIVATE COUNT IS VERSION-LOAD-BEARING. The recall denominator below is 219 at v1
    and 252 at v7 for tau2_retail -- a 15% move, all `required` in both -- so a user-private n
    quoted without its graph_version is not a result (CLAUDE.md rule 1).

    AND v2..v7 CANNOT BE LOADED AT ALL. `load_graphs` raises
    `TypeError: GoldNode.__init__() got an unexpected keyword argument 'gold_h1_label'`
    (pi_eval/gold.py, the `_tuplify` genexpr) on FOURTEEN files across three suites:
    tau2_retail v2-v7, tau2_airline v2-v7, tau2_telecom v2-v3. No tracked file writes OR reads
    `gold_h1_label` -- `git grep gold_h1_label` is empty -- so those graphs can be neither
    loaded nor regenerated from this tree, and every gold-dependent measurement here is
    narrowed to v1 whether or not it says so. Recorded, not worked around: adding the field to
    `GoldNode` would make files this tree cannot produce readable, which is a gold-build
    question and not a metric one.

    THREE DENOMINATORS, ALL RETURNED, BECAUSE TWO OF THEM CAN BE EMPTY:

      `n_user_asks`   user-target asks in this run. ZERO for every arm but
                      `inquirer_may_ask_user`, whose policy rewrites the target to "kb"
                      otherwise. With no user ask, recall is 0 by construction rather than by
                      measurement -- entailed by the arm table -- so BOTH rates are NaN here
                      and `pi_eval.score` writes the count alone.
      `n_attributed`  user-target asks naming at least one annotated need: the precision
                      denominator. An ask that names nothing is not evidence that the policy
                      asked the user for something the KB held; it is evidence of nothing, and
                      counting it as a miss would let a chatty greeting drive the metric.
      `n_universe`    REQUIRED user-private needs: the recall denominator. EMPTY on tau2
                      (banking), where all 3,637 user-private nodes are `optional` and none is
                      required -- the same 3,637 that `cmd_gold` reports as depth-orphaned. A
                      0/0 rendered 0.0 there would say "the policy asked the user for none of
                      what it needed" on the one domain where the question cannot be asked.
                      Well formed at v1 on tau2_airline (58) and tau2_retail (219).

    Optional user-private needs count toward PRECISION and not toward RECALL: asking the
    customer for something optional but genuinely private is not harassment, while failing to
    ask for it is not a miss the task ever required.
    """
    from pi_eval.matcher.base import asked_about

    scorable = [n for n in graph.gold_nodes if n.gold_partition in _SCORABLE]
    private = {n.gold_node_id for n in scorable if n.gold_discoverability == "user_private"}
    universe = {
        n.gold_node_id
        for n in scorable
        if n.gold_partition == "required" and n.gold_discoverability == "user_private"
    }
    user_asks = [
        t
        for t in turns
        if str(t.get("action_kind") or "") == "ask" and str(t.get("target") or "") == USER_TARGET
    ]
    named = [
        {n.gold_node_id for n in scorable if asked_about(n, str(t.get("question") or "").upper())}
        for t in user_asks
    ]
    attributed = [ids for ids in named if ids]
    precise = [ids for ids in attributed if ids & private]
    reached = {nid for ids in named for nid in ids} & universe
    return {
        "precision": (len(precise) / len(attributed)) if attributed else float("nan"),
        "recall": (len(reached) / len(universe)) if (universe and user_asks) else float("nan"),
        "n_user_asks": float(len(user_asks)),
        "n_attributed": float(len(attributed)),
        "n_universe": float(len(universe)),
        "n_reached": float(len(reached)),
        "n_private": float(len(private)),
    }
