"""Budget accounting.

Exactly ONE currency is hard-capped. Everything else is recorded and asserted equal across
arms at aggregate time. Single-phase check-and-charge: this is a single-threaded loop and
you cannot un-spend tokens, so a two-phase reserve/settle would be ceremony.

The ledger is per (task, arm, seed) process, so it needs no locking at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .types import CallTelemetry, Usage

HARD_CURRENCY = "retrieval_calls"

# Recorded but never hard-capped. wall_ms and usd are deliberately ABSENT from the
# cross-arm parity assertion: an arm with more turns takes more wall time, so asserting
# parity there fails on day one and teaches you to disable the check.
PARITY_CURRENCIES = ("tok_prompt", "tok_completion", "tok_total", "retrieval_calls", "unique_docs")


class BudgetExceeded(RuntimeError):
    """Raised BEFORE dispatch. Policies must not catch it; run_loop does."""


@dataclass
class LedgerRow:
    currency: str
    charged: float
    cumulative: float
    turn_idx: int
    cap: float | None
    hard: bool


class BudgetLedger:
    def __init__(self, cap: int, *, unique_doc_cap: int | None = None) -> None:
        self.cap = cap
        self.unique_doc_cap = unique_doc_cap
        self.spent: dict[str, float] = {}
        self._docs: set[str] = set()
        self.rows: list[LedgerRow] = []
        self.calls: list[CallTelemetry] = []
        self._turn_idx = 0
        # Added to every turn index this ledger stamps. Zero for a single rollout, which is
        # every suite but tau2. A tau2 unit runs ONE run_loop per user message against ONE
        # shared ledger, and run_loop counts turns from 0 each time -- so without this the
        # ledger's turn_idx restarts while `merge_trajectories` renumbers Turn.turn_idx
        # sequentially, and the two are joined on that column. Rows from the second rollout
        # would be attributed to the first rollout's turns.
        self.turn_base = 0

    def set_turn(self, turn_idx: int) -> None:
        self._turn_idx = self.turn_base + turn_idx

    def _bump(self, currency: str, amount: float, cap: float | None, hard: bool) -> None:
        cum = self.spent.get(currency, 0.0) + amount
        self.spent[currency] = cum
        self.rows.append(LedgerRow(currency, amount, cum, self._turn_idx, cap, hard))

    def check(self, amount: float = 1.0) -> None:
        """Pre-dispatch check against the single hard cap."""
        if self.spent.get(HARD_CURRENCY, 0.0) + amount > self.cap:
            raise BudgetExceeded(
                f"{HARD_CURRENCY}: {self.spent.get(HARD_CURRENCY, 0.0) + amount} > {self.cap}"
            )

    def charge_retrieval(self, amount: float = 1.0) -> None:
        self.check(amount)
        self._bump(HARD_CURRENCY, amount, self.cap, True)

    def note_docs(self, doc_ids: Iterable[str]) -> int:
        """Meter UNIQUE documents, not calls.

        Without this a policy free-rides on the cache: repeating a query costs no retrieval
        call at the provider but still delivers evidence into the draft.
        """
        new = set(doc_ids) - self._docs
        if new:
            self._docs |= new
            self._bump("unique_docs", float(len(new)), self.unique_doc_cap, True)
            if self.unique_doc_cap is not None and len(self._docs) > self.unique_doc_cap:
                raise BudgetExceeded(f"unique_docs: {len(self._docs)} > {self.unique_doc_cap}")
        return len(new)

    def record_call(self, call: CallTelemetry) -> None:
        """Recorded, never capped. Parity across arms is asserted at aggregate time."""
        self.calls.append(call)
        u = Usage.from_call(call)
        for currency, amount in (
            ("tok_prompt", u.tok_prompt),
            ("tok_completion", u.tok_completion),
            ("tok_reasoning", u.tok_reasoning),
            ("tok_cached", u.tok_cached),
            ("tok_total", u.tok_total),
            ("usd", u.usd),
            ("wall_ms", u.wall_ms),
            ("llm_calls", 1),
        ):
            if amount:
                self._bump(currency, float(amount), None, False)

    def record(self, currency: str, amount: float = 1.0) -> None:
        self._bump(currency, amount, None, False)

    @property
    def usage(self) -> Usage:
        return self.usage_since(0)

    def usage_since(self, index: int) -> Usage:
        """Usage accumulated by calls recorded at or after `index`.

        This is what lets the driver stamp a REAL per-turn usage on each Turn. Without it
        Turn.usage is uniformly zero, per-turn cost attribution is impossible, the frontier
        has no cumulative-spend x-axis, and reconcile() becomes tautological -- it would
        compare the ledger against a sum of zeros and could never fail.
        """
        total = Usage()
        for c in self.calls[index:]:
            total = total + Usage.from_call(c)
        return total

    @property
    def n_calls(self) -> int:
        return len(self.calls)

    @property
    def unique_docs(self) -> int:
        return len(self._docs)

    def reconcile(self, turn_usage: Usage, terminal_usage: Usage | None = None) -> bool:
        """CI assertion substrate: per-turn usage PLUS terminal usage must equal the ledger.

        An unmetered nested retrieval inside Drafter.resolve fails this, which is how a budget
        cheat becomes a red build instead of a silently inflated arm.

        `terminal_usage` is not a loophole, it is what makes the check bite. Without it the
        comparison was `ledger == sum(turns)`, and a real rollout charges three things to no
        turn at all -- the act() that returned STOP, the final draft, and the Answerer -- so
        the equality was FALSE on every LLM arm and TRUE only on the llm_free arms whose
        ledgers are empty. Measured on a live 2-unit run: 30 ledger calls, 18 inside turns.
        A check that cannot pass where it matters is not a check; it is a field named
        `tokens_ok` that always says False and is therefore never read.
        """
        led = self.usage
        total = turn_usage + (terminal_usage or Usage())
        return (
            led.tok_prompt == total.tok_prompt
            and led.tok_completion == total.tok_completion
            and led.n_calls == total.n_calls
        )
