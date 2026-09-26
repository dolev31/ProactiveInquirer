"""Retrieval sensitivity: the cheapest check that can end a suite, run before any LLM call.

THE FAILURE THIS EXISTS FOR. On a closed pool of 20 short paragraphs, BM25 for a gold
sub-question may return the same top-k as BM25 for the full multi-hop question. If it does,
then `drafter_only`, `self_inquire`, `inquirer_prompted`, `parallel_replay` and `random_q` all
converge on an identical `evidence.subset_hash`, every paired delta is 0.00 +/- 0.00, and no
amount of prompt engineering changes it -- you have spent two weeks measuring a constant.

It is more likely than any interesting negative result, because it requires no scientific
truth to occur: only a pool too small to discriminate. And it MASQUERADES AS A NULL, which is
the dangerous part. You would read the flat table and start debugging the policy.

The check costs zero tokens, zero dollars and runs in seconds, so there is no budget at which
skipping it is rational. It is gold-side because it needs the gold sub-questions, and it must
never run inside a rollout worker.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Protocol, Sequence


class _Retriever(Protocol):
    def search(self, query: str, k: int) -> Sequence[object]: ...


@dataclass(frozen=True, slots=True)
class TaskProbe:
    task_id: str
    n_subquestions: int
    n_distinct_sets: int
    subq_only_hits: int  # gold units a sub-question finds that the full question misses
    full_question_hits: int
    gold_units_total: int
    # Units the sub-questions returned AT ALL. Zero across a suite means the probe is
    # inapplicable there, not that the suite failed -- see ProbeResult.applicable.
    subq_retrieved_any: int = 0

    @property
    def has_subq_only_evidence(self) -> bool:
        return self.subq_only_hits > 0


@dataclass(frozen=True, slots=True)
class ProbeResult:
    suite_id: str
    n_tasks: int
    k: int
    median_distinct_sets: float
    frac_tasks_with_subq_only_evidence: float
    min_median_distinct: float
    min_frac_subq_only: float
    tasks: tuple[TaskProbe, ...] = ()

    @property
    def applicable(self) -> bool:
        """Whether gold node text is usable AS A RETRIEVAL QUERY on this suite.

        It is on MuSiQue/StrategyQA/2Wiki, where a node IS a sub-question. It is NOT on synth
        or tau2, where a node is a fact or a document title and the thing that actually keys
        retrieval (an unguessable token, a policy phrase) lives in the document rather than in
        the need text. Reporting FAIL there would be the instrument mistaking its own
        inapplicability for a property of the suite -- exactly the error this module exists to
        prevent elsewhere.
        """
        return any(t.subq_retrieved_any > 0 for t in self.tasks)

    @property
    def passed(self) -> bool:
        return (
            self.applicable
            and self.median_distinct_sets >= self.min_median_distinct
            and self.frac_tasks_with_subq_only_evidence >= self.min_frac_subq_only
        )

    @property
    def verdict(self) -> str:
        if self.passed:
            return "PASS"
        if self.tasks and not self.applicable:
            return (
                "INAPPLICABLE: gold node text is not a retrieval query on this suite "
                "(sub-questions returned nothing at all). Probe a suite whose nodes ARE "
                "sub-questions, or supply suite-specific queries."
            )
        why = []
        if self.median_distinct_sets < self.min_median_distinct:
            why.append(
                f"median distinct top-{self.k} sets {self.median_distinct_sets:.2f} < "
                f"{self.min_median_distinct}: queries do not discriminate"
            )
        if self.frac_tasks_with_subq_only_evidence < self.min_frac_subq_only:
            why.append(
                f"only {self.frac_tasks_with_subq_only_evidence:.0%} of tasks have evidence a "
                f"sub-question finds and the full question misses (need "
                f"{self.min_frac_subq_only:.0%}): there is nothing for inquiry to add"
            )
        return "FAIL: " + "; ".join(why)

    @property
    def consequence(self) -> str:
        return (
            "Drop k and re-probe the same afternoon. If it still fails, this suite cannot "
            "discriminate between arms and a flat results table would be an artifact of the "
            "retriever, not a null result. Demote the suite rather than spending on it."
        )


def probe_task(
    *,
    task_id: str,
    question: str,
    subquestions: Sequence[str],
    gold_uids: frozenset[str],
    retriever: _Retriever,
    k: int = 5,
) -> TaskProbe:
    def uids(q: str) -> frozenset[str]:
        return frozenset(getattr(u, "uid", "") for u in retriever.search(q, k))

    full = uids(question)
    sets = {full}
    subq_union: set[str] = set()
    for sq in subquestions:
        s = uids(sq)
        sets.add(s)
        subq_union |= s

    return TaskProbe(
        task_id=task_id,
        n_subquestions=len(subquestions),
        # The full question's own set counts: if every sub-question reproduces it, that is 1.
        n_distinct_sets=len(sets),
        subq_only_hits=len((subq_union & gold_uids) - full),
        full_question_hits=len(full & gold_uids),
        gold_units_total=len(gold_uids),
        subq_retrieved_any=len(subq_union),
    )


def probe(
    tasks: Sequence[TaskProbe],
    *,
    suite_id: str,
    k: int = 5,
    min_median_distinct: float = 2.0,
    min_frac_subq_only: float = 0.30,
) -> ProbeResult:
    if not tasks:
        return ProbeResult(suite_id, 0, k, 0.0, 0.0, min_median_distinct, min_frac_subq_only)
    med = statistics.median(t.n_distinct_sets for t in tasks)
    frac = sum(1 for t in tasks if t.has_subq_only_evidence) / len(tasks)
    return ProbeResult(
        suite_id=suite_id,
        n_tasks=len(tasks),
        k=k,
        median_distinct_sets=float(med),
        frac_tasks_with_subq_only_evidence=frac,
        min_median_distinct=min_median_distinct,
        min_frac_subq_only=min_frac_subq_only,
        tasks=tuple(tasks),
    )
