"""Deterministic, LLM-free components.

They exist so the loop, the ledger, the prefix ladder and every metric can be exercised in
CI with no network, no keys and no dollars — and so a failing metric test points at the
metric rather than at model sampling noise.
"""

from __future__ import annotations

import re

from pinq.budget import BudgetLedger
from pinq.types import (
    Action,
    Answer,
    Ask,
    Draft,
    Evidence,
    State,
    Stop,
    TaskView,
)

TOKEN_RE = re.compile(r"\b[KD][0-9A-F]{8}\b")


class EchoDrafter:
    """resolve() is a no-op over evidence; draft() is a PURE function of (view, subset_hash).

    Purity is not a nicety here: phi_LOO, the prefix ladder and the stop test are all
    undefined without it, so the reference implementation is pure by construction and
    tests/test_purity.py holds every real drafter to the same standard.
    """

    def resolve(self, view: TaskView, ask: Ask, ev: Evidence, *, seed: int, ledger: BudgetLedger):
        return ("", ev)

    def draft(self, view: TaskView, ev: Evidence, *, seed: int, ledger: BudgetLedger) -> Draft:
        values = sorted({m for u in ev.units for m in re.findall(r"\bV\d+\b", u.text)})
        return Draft(text=" ".join(values))


class FrozenAnswerer:
    """Word-capped and arm-blind: it never sees the arm id or the budget, so an answer's
    surface cannot betray which condition produced it."""

    prompt_hash = "frozen-answerer-v1"

    def answer(
        self, view: TaskView, ev: Evidence, draft: Draft | None, *, seed: int, ledger: BudgetLedger
    ) -> Answer:
        text = " ".join((draft.text if draft else "").split()[: view.word_cap])
        return Answer(
            text=text,
            evidence_hash=ev.subset_hash,
            cited_unit_ids=tuple(u.uid for u in ev.units),
            n_words=len(text.split()),
            stop_reason="policy_stop",
        )


class ChainInquirer:
    """A competent reference policy: it reads the tokens it has already seen and asks for the
    next one. Optimal on the synthetic suite, which makes it the ceiling against which the
    degenerate policies below are compared."""

    policy_id = "chain"

    def __init__(self, max_asks: int = 12) -> None:
        self.max_asks = max_asks

    def reset(self, view: TaskView, seed: int) -> None:
        self._asked: set[str] = set()
        self._seen: list[str] = TOKEN_RE.findall(view.question.upper())

    def act(self, s: State) -> Action:
        known = list(self._seen)
        for u in s.evidence.units:
            known += TOKEN_RE.findall(u.text.upper())
        for tok in known:
            if tok not in self._asked:
                self._asked.add(tok)
                return Ask(text=f"What does record {tok} say?")
        return Stop()


class NeverAsk:
    """The degenerate default: the Drafter alone. Not a strawman — it is the comparator."""

    policy_id = "never_ask"

    def reset(self, view: TaskView, seed: int) -> None:
        return None

    def act(self, s: State) -> Action:
        return Stop()


class BreadthOnlyInquirer:
    """Asks only about tokens stated in the question: depth-1 restricted.

    If this matches the chain policy, vertical proactivity contributes nothing — which is
    exactly the inquirer_depth1 ablation, made concrete and free to run.
    """

    policy_id = "depth1"

    def reset(self, view: TaskView, seed: int) -> None:
        self._queue = list(TOKEN_RE.findall(view.question.upper()))

    def act(self, s: State) -> Action:
        if self._queue:
            return Ask(text=f"What does record {self._queue.pop(0)} say?")
        return Stop()


class ShallowWideInquirer:
    """Every facet, one hop deep: breadth with the depth axis pinned at 1.

    BreadthOnlyInquirer cannot serve as the horizontal corner. It asks only about the tokens the
    question states, and those are the depth-0 frontier, which belongs to NO facet by
    construction -- so it scores facet_breadth 0, not FACETS. Touching a facet in this suite
    requires one hop. This policy takes exactly that hop on every facet and then stops, which is
    the shallow-wide corner the vertical corner below is compared against.

    Stopping after one hop is not a turn cap: the tokens revealed by the seed records are
    captured ONCE, so tokens revealed at depth 1 and beyond are never queued. A cap would make
    the policy's depth a property of the budget instead of a property of the policy.
    """

    policy_id = "shallow_wide"

    def reset(self, view: TaskView, seed: int) -> None:
        self._seeds = list(TOKEN_RE.findall(view.question.upper()))
        self._asked: set[str] = set()
        self._hop1: list[str] | None = None

    def act(self, s: State) -> Action:
        for tok in self._seeds:
            if tok not in self._asked:
                self._asked.add(tok)
                return Ask(text=f"What does record {tok} say?")
        if self._hop1 is None:
            self._hop1 = [t for u in s.evidence.units for t in TOKEN_RE.findall(u.text.upper())]
        for tok in self._hop1:
            if tok not in self._asked:
                self._asked.add(tok)
                return Ask(text=f"What does record {tok} say?")
        return Stop()


class DeepNarrowInquirer:
    """ONE facet, all the way down: depth with the breadth axis pinned at 1.

    The counterpart of ShallowWideInquirer. It takes the first token the question states and
    follows only what that chain reveals, so it resolves one node at every depth and one facet
    in total. Together the two make the horizontal and vertical axes separately falsifiable on a
    single task, which no real gold suite here supports -- none carries a facet denominator above
    one and depth->=2 nodes in the same tasks.

    It follows only the continuation token of the record it last asked for, not every token in
    the evidence set. Retrieval returns k documents per ask, so a policy that chased every token
    it saw would wander into other facets and its breadth would become a property of k.
    """

    policy_id = "deep_narrow"

    def reset(self, view: TaskView, seed: int) -> None:
        toks = TOKEN_RE.findall(view.question.upper())
        self._chain: list[str] = toks[:1]  # the FIRST facet's seed, and nothing else
        self._asked: set[str] = set()

    def act(self, s: State) -> Action:
        # Extend the chain from evidence already in hand. A record's body names itself first and
        # its successor second ("Record K... To continue, consult record K..."), so a unit whose
        # OWN token this policy asked for is the one entitled to extend the chain. Distractor
        # notes name only themselves and so extend nothing.
        for u in s.evidence.units:
            found = TOKEN_RE.findall(u.text.upper())
            if len(found) >= 2 and found[0] in self._asked and found[1] not in self._chain:
                self._chain.append(found[1])
        for tok in self._chain:
            if tok not in self._asked:
                self._asked.add(tok)
                return Ask(text=f"What does record {tok} say?")
        return Stop()


class VerbatimInquirer:
    """The corpus-suite baseline: ask the question as given, then follow up on what came back.

    ChainInquirer cannot serve here. It navigates by unlock TOKENS, which exist only in the
    synthetic suite; on MuSiQue or 2Wiki it finds none in the question, stops on turn zero
    and retrieves nothing, so a loop test written around it would pass while proving nothing.

    This policy is deliberately weak — one verbatim query plus one query per document title
    it has already seen. It is a floor for the real suites, not a ceiling: it can only reach
    a need whose entity is named in the question or in a paragraph it already holds, which
    is exactly the depth-1 horizon a competent Inquirer is supposed to beat.
    """

    policy_id = "verbatim"

    def __init__(self, max_asks: int = 4) -> None:
        self.max_asks = max_asks

    def reset(self, view: TaskView, seed: int) -> None:
        self._question = view.question
        self._asked: set[str] = set()
        self._n = 0

    def act(self, s: State) -> Action:
        if self._n >= self.max_asks:
            return Stop()
        self._n += 1
        if self._n == 1:
            return Ask(text=self._question)
        # Evidence.units is canonically ordered by uid, so the follow-up sequence is a pure
        # function of which units are held — no dependence on retrieval order.
        for u in s.evidence.units:
            if u.title not in self._asked:
                self._asked.add(u.title)
                return Ask(text=f"{u.title}: {self._question}", parent_uids=(u.uid,))
        return Stop()
