"""MOVED. The answer-bearing-node rule now lives at `pi_eval.answer_node`.

WHY IT MOVED, AND WHAT DID NOT. Lane L6.1 needs this rule inside `pi_run.cmd_train`'s
decision-row export, and `src/` may not import `scripts/` -- so the rule had to become an
importable library module. It is the SAME FILE, moved by `git mv` with no edit to any rule:
`git log --follow src/pi_eval/answer_node.py` shows one content commit (L1.11's) and the
rename. `pi_eval` is its home because it reads gold (`pi_eval.gold`) and scores containment
with the scorer's own instrument (`pi_eval.metrics.quality.contains_answer`); the firewall
already allows `pi_run` to import `pi_eval` and still forbids `pinq_train` from doing so.

This shim exists so `scripts.answer_node_coverage.{lib,run}` and
`tests/test_answer_node_coverage.py` -- the L1.11 diagnostic lane, whose published numbers
must stay reproducible from the paths its RESULT.md names -- keep importing the name they
always did, and so a reader who follows that RESULT.md's "Tooling and tests" section lands
here and is told where the rule went rather than finding nothing.
"""

from __future__ import annotations

from pi_eval.answer_node import (  # noqa: F401
    BOOL_ANSWERS,
    RULE_ALL_REQUIRED_FALLBACK,
    RULE_BOOLEAN_FALLBACK,
    RULE_EVIDENCE_HIT,
    RULE_NO_HIT_FALLBACK,
    RULE_NO_REQUIRED_NODES,
    AnswerNodeResult,
    answer_nodes_for_suite,
    answer_nodes_for_task,
    gold_uid_partition,
    node_evidence_text,
    rule_tally,
)

__all__ = [
    "BOOL_ANSWERS",
    "RULE_ALL_REQUIRED_FALLBACK",
    "RULE_BOOLEAN_FALLBACK",
    "RULE_EVIDENCE_HIT",
    "RULE_NO_HIT_FALLBACK",
    "RULE_NO_REQUIRED_NODES",
    "AnswerNodeResult",
    "answer_nodes_for_suite",
    "answer_nodes_for_task",
    "gold_uid_partition",
    "node_evidence_text",
    "rule_tally",
]
