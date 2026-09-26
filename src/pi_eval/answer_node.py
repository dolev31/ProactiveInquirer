"""The per-task ANSWER-BEARING NODE, defined from the gold graph alone.

THE GRAPH DOES NOT MARK ONE. `GoldNode` has no `is_answer` field and `GoldGraph.gold_answer`
lives at the GRAPH level, not pinned to any single node -- confirmed by reading `pi_eval.gold`
(this suite's own schema) rather than assumed. So "the answer node" has to be operationalised,
and the brief names two candidate rules. Both are implemented; which one applies is decided per
task, not chosen once for a whole suite.

RULE 1 (primary): the required node whose GOLD EVIDENCE TEXT contains the gold answer, tested
with the scorer's own `pi_eval.metrics.quality.contains_answer` (token-sequence containment
after normalization -- the identical instrument `score_run` uses for `answer_correct`, not a
second hand-rolled string match).

"Gold evidence text" means the corpus passage(s) named by the node's `gold_ev_uids`, resolved
via `corpus_text.py` -- NOT the node's own `gold_text`. Verified on a live example before
committing to this: musique task `2hop__1007_59746`'s deepest required node's `gold_text` is
"when did #1 become a publicly traded company", which does not contain the gold answer "1980"
at all; its evidence passage does. A rule keyed on `gold_text` would silently find nothing on
the very shape of task (multi-hop bridge) the hypothesis is about.

RULE 2 (fallback): the deepest required node(s) on the answer path (`gold_depth`, already
computed by `pi_eval.gold.compute_depths` with AND/longest-path semantics and stamped onto
every node at build time -- not re-derived here).

WHY A FALLBACK IS NECESSARY, MEASURED, NOT ASSUMED. Every one of StrategyQA's 2,290 v1 graphs
has a gold answer that normalizes to "yes"/"no"/"true"/"false" (checked over the full gold
population before writing this rule); 1,295 of wiki2's 12,576 do too (comparison-type
sub-questions). Two problems, not one:

  * the node that would answer a yes/no question -- the comparison itself ("Is #2 part of
    #1?") -- is annotated OPTIONAL with `gold_ev_uids == ()`: there is no required node whose
    evidence could contain the word "yes" even in principle.
  * testing containment of "yes"/"no"/"true"/"false" against arbitrary evidence prose is not a
    weaker version of the same test, it is a DIFFERENT and broken one: these are among the
    commonest tokens in English, so `contains_answer` would fire on unrelated passages and the
    "hit" would carry no information about whether the retrieved evidence was the one that
    settled the question.

So RULE 1 is never attempted when `graph.answer` normalizes to a boolean token -- the fallback
is unconditional there, not a last resort. On every other task RULE 1 is tried first and RULE 2
is used only if it finds no qualifying required node (e.g. the answer's surface form in the
evidence disagrees with `gold_answer`'s formatting -- a unit conversion, a paraphrase).

If several nodes qualify under whichever rule fired (a tie at the max depth, or several nodes
whose evidence independently names the answer), ALL of them form the task's answer-node set,
and "the answer node is covered" means ANY of them is -- the brief's own instruction for the
several-answer-node case.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from pi_eval.gold import GoldGraph, GoldNode
from pi_eval.metrics.quality import contains_answer, normalize

BOOL_ANSWERS = frozenset({"yes", "no", "true", "false"})


@dataclass(frozen=True, slots=True)
class AnswerNodeResult:
    node_ids: tuple[str, ...]
    rule: str
    max_depth: int | None
    n_evidence_uids_unresolved: int

    @property
    def is_empty(self) -> bool:
        return not self.node_ids


# Rule labels, exhaustive -- `run.py` tallies tasks by this string, so a new branch here must
# add a new label rather than reuse one, or the tally silently mixes two different reasons.
RULE_EVIDENCE_HIT = "evidence_contains_answer"
RULE_BOOLEAN_FALLBACK = "boolean_answer_depth_fallback"
RULE_NO_HIT_FALLBACK = "no_evidence_hit_depth_fallback"
RULE_ALL_REQUIRED_FALLBACK = "all_required_orphaned_depth"
RULE_NO_REQUIRED_NODES = "no_required_nodes"


def node_evidence_text(node: GoldNode, uid_to_text: Mapping[str, str]) -> tuple[str, int]:
    """Concatenated corpus text of `node`'s gold evidence uids, and how many did not resolve.

    A node with zero ev uids (StrategyQA's optional comparison nodes, and any orphaned
    required node) returns `("", 0)` -- not an error, just nothing to search.
    """
    parts: list[str] = []
    n_missing = 0
    for uid in node.gold_ev_uids:
        text = uid_to_text.get(uid)
        if text is None:
            n_missing += 1
        else:
            parts.append(text)
    return " ".join(parts), n_missing


def answer_nodes_for_task(graph: GoldGraph, uid_to_text: Mapping[str, str]) -> AnswerNodeResult:
    required = graph.required()
    if not required:
        return AnswerNodeResult((), RULE_NO_REQUIRED_NODES, None, 0)

    ans_is_boolean = normalize(graph.answer) in BOOL_ANSWERS
    n_missing_total = 0

    if not ans_is_boolean:
        hits: list[str] = []
        for n in required:
            text, n_missing = node_evidence_text(n, uid_to_text)
            n_missing_total += n_missing
            if text and contains_answer(text, graph.answer, graph.gold_aliases):
                hits.append(n.gold_node_id)
        if hits:
            return AnswerNodeResult(tuple(sorted(hits)), RULE_EVIDENCE_HIT, None, n_missing_total)

    depths = [n.gold_depth for n in required if n.gold_depth is not None]
    if depths:
        max_depth = max(depths)
        deepest = tuple(sorted(n.gold_node_id for n in required if n.gold_depth == max_depth))
        rule = RULE_BOOLEAN_FALLBACK if ans_is_boolean else RULE_NO_HIT_FALLBACK
        return AnswerNodeResult(deepest, rule, max_depth, n_missing_total)

    # every required node is orphaned (gold_depth is None on all of them): no depth signal at
    # all. Falls back to "any required node", which is the weakest, most conservative answer-
    # node set -- reported as its own rule so it is never silently folded into a real depth
    # read.
    return AnswerNodeResult(
        tuple(sorted(n.gold_node_id for n in required)),
        RULE_ALL_REQUIRED_FALLBACK,
        None,
        n_missing_total,
    )


def answer_nodes_for_suite(
    graphs: Mapping[str, GoldGraph], uid_text_by_task: Mapping[str, Mapping[str, str]]
) -> dict[str, AnswerNodeResult]:
    """`task_key -> AnswerNodeResult`, over every task in `graphs`."""
    return {
        task_key: answer_nodes_for_task(graph, uid_text_by_task.get(task_key, {}))
        for task_key, graph in graphs.items()
    }


def rule_tally(results: Mapping[str, AnswerNodeResult]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in results.values():
        out[r.rule] = out.get(r.rule, 0) + 1
    return out


def gold_uid_partition(
    graph: GoldGraph, answer_node_ids: Sequence[str]
) -> tuple[frozenset[str], frozenset[str]]:
    """(answer_uids, nonanswer_uids): the required-evidence gold uid set, partitioned by
    whether it belongs to a task's designated answer node(s). Disjoint by construction and
    together equal `pi_eval.score._gold_uids(graph)` -- required nodes only, same as the
    published `evidence_coverage` denominator, asserted by
    `tests/test_answer_node_coverage.py::test_gold_uid_partition_reunites_to_the_scorers_own_gold_uids`.
    """
    ans_ids = set(answer_node_ids)
    answer_uids: set[str] = set()
    nonanswer_uids: set[str] = set()
    for n in graph.required():
        target = answer_uids if n.gold_node_id in ans_ids else nonanswer_uids
        target.update(n.gold_ev_uids)
    # A uid could in principle be shared by two required nodes (not seen in practice, but not
    # forbidden by the schema); if one of the two is an answer node, credit it to the answer
    # side rather than letting it land in both/neither silently.
    nonanswer_uids -= answer_uids
    return frozenset(answer_uids), frozenset(nonanswer_uids)


def answer_node_uid_sets(
    graph: GoldGraph, answer_node_ids: Sequence[str]
) -> tuple[frozenset[str], ...]:
    """One gold-evidence uid set PER answer node, evidence-bearing nodes only.

    PER NODE AND NOT POOLED, and the difference is the whole point of the several-node case.
    `gold_uid_partition` unions the answer nodes' spans, which is the right denominator for a
    coverage SHARE; "the answer node is covered" is L1.11's `ANY of them is`, and a union would
    silently promote that to ALL of them -- a strictly harder condition on exactly the tasks
    (StrategyQA's 43% depth ties, musique's multi-hit evidence) where the rule is weakest.

    A node with no `gold_ev_uids` is DROPPED rather than contributing the empty set. The empty
    set is a subset of every evidence state, so keeping it would make such a node covered before
    the policy retrieved anything -- an answer that is always already there. A task left with no
    sets at all is one whose answer node cannot be checked, which the caller must report as
    unknown and never as uncovered.
    """
    ans = set(answer_node_ids)
    return tuple(
        frozenset(n.gold_ev_uids)
        for n in graph.required()
        if n.gold_node_id in ans and n.gold_ev_uids
    )


def answer_node_covered(
    seen_uids: frozenset[str] | set[str], uid_sets: Sequence[frozenset[str]]
) -> bool:
    """Is at least one answer node's gold evidence entirely in hand?

    CONTAINMENT, which is `done_before`'s own convention narrowed to one node: `done_before` is
    `evidence_coverage(seen, required) >= 1.0`, i.e. `required <= seen`, and this is that test
    with `required` replaced by a single answer node's spans. Deliberately NOT the matcher's
    `match_kind in (resolve, use)`: that asks which question resolved the node, a property of
    the trajectory, where an SFT state label may only depend on the evidence prefix every
    candidate at that state shares.

    Callers must not reach this with an empty `uid_sets` -- see `answer_node_uid_sets`; `any()`
    over nothing is False, and False here would read as "the answer evidence is missing" on a
    task where it was never checkable.
    """
    seen = frozenset(seen_uids)
    return any(s <= seen for s in uid_sets)
