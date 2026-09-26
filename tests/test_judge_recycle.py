"""Recycling the judge's client must not lose a cent of its spend.

MEASURED decay within one judging process: 47 -> 13 calls/min on one pass, 71 -> 22 on
another. A FRESH PROCESS at the same provider, same cache and twice the concurrency ran at
71/min, so the loss is in-process. Cache sharding was ruled out (256 dirs, ~125 files each).
The root cause is unproven; the remedy is empirical, and these tests pin the ONE property
that must hold whether or not it turns out to help.

That property is accounting continuity. `_Judge.spend()` is what `pi score` reports and what
`--judge-spend-cap` enforces, and it reads the ledger. Rebuilding the client around a NEW
ledger would zero the spend at every recycle: the cap would never trigger, the reported
judge bill would be only the calls since the last rebuild, and both failures are silent.
"""

from __future__ import annotations

from typing import Any

from pinq.budget import BudgetLedger
from pinq.types import CallTelemetry


def _tel(usd: float, *, hit: bool = False) -> CallTelemetry:
    return CallTelemetry(
        call_id="c",
        actor="judge",
        model="m",
        provider="p",
        request_sha="rq",
        response_sha="rs",
        tok_prompt=10,
        tok_completion=5,
        usd=usd,
        cache_hit=hit,
    )


class _Client:
    """Each instance is a distinct object, so a swap is observable."""

    def __init__(self, ledger: BudgetLedger, tag: str) -> None:
        self.ledger, self.tag = ledger, tag

    def complete(self, **kw: Any) -> tuple[str, CallTelemetry]:
        tel = _tel(0.001)
        self.ledger.record_call(tel)
        return self.tag, tel

    def request_payload(self, **kw: Any) -> dict[str, Any]:
        return {"tag": self.tag}

    def pin(self, role: str) -> Any:
        return self.tag


def _judge_with_rebuild() -> Any:
    from pi_run.judge_client import _Judge

    ledger = BudgetLedger(cap=10**9)
    seq = {"n": 0}

    def rebuild() -> _Client:
        seq["n"] += 1
        return _Client(ledger, f"client{seq['n']}")

    return _Judge(rebuild(), ledger, rebuild=rebuild)


def test_spend_survives_a_recycle() -> None:
    """The whole point. A new ledger here would silently disarm --judge-spend-cap."""
    j = _judge_with_rebuild()
    j.complete()
    j.complete()
    before = j.spend()
    assert before["n_calls"] == 2

    assert j.recycle() is True
    j.complete()

    after = j.spend()
    assert after["n_calls"] == 3, f"lost calls across recycle: {before} -> {after}"
    assert after["usd"] == round(before["usd"] + 0.001, 6)


def test_the_client_is_actually_replaced() -> None:
    """Otherwise the remedy is a no-op that would still 'pass' an A/B by luck."""
    j = _judge_with_rebuild()
    assert j.complete()[0] == "client1"
    j.recycle()
    assert j.complete()[0] == "client2"


def test_a_judge_with_no_rebuild_declines_rather_than_crashing() -> None:
    """`judge_replay` is cache-only and has nothing to rebuild.

    A recycle hook that raised there would take down every CI re-score, which is the one
    judging path that must never need the network.
    """
    from pi_run.judge_client import _Judge

    ledger = BudgetLedger(cap=10**9)
    j = _Judge(_Client(ledger, "replay"), ledger)
    assert j.recycle() is False
    assert j.complete()[0] == "replay"


def test_recycles_are_counted() -> None:
    j = _judge_with_rebuild()
    assert j.recycles == 0
    j.recycle()
    j.recycle()
    assert j.recycles == 2


def test_the_real_judge_factory_supplies_a_rebuild(tmp_path) -> None:
    """The production path must actually be recyclable, not just the test double."""
    from pi_run.judge_client import judge

    j = judge(
        {"PI_MODEL_JUDGE": "openai/aws/gpt-oss-120b", "PI_CACHE_ROOT": str(tmp_path)},
        cache_root=str(tmp_path),
    )
    inner_before = j._inner
    assert j.recycle() is True
    assert j._inner is not inner_before
    assert j._ledger is not None


# ------------------------------------------------------------------ the wiring


class _RecordingLLM:
    """A judge client that counts recycles, without importing one."""

    def __init__(self) -> None:
        self.recycles = 0

    def recycle(self) -> bool:
        self.recycles += 1
        return True


class _FakeGrader:
    """Shaped like the REAL `_Grader`: `_grade(run, family, order)`, `_cache`, `_llm`.

    A double that took `_grade(key)` and held its own counters was written once in this
    project and produced twelve green tests against a signature production does not have.
    """

    def __init__(self, llm: Any) -> None:
        self._llm = llm
        self._cache: dict[tuple[str, str, str], tuple[Any, ...]] = {}
        self._result = type("R", (), {"n_calls_failed": 0, "unjudged": []})()
        self.budget_checks = 0

    def check_budget(self) -> None:
        self.budget_checks += 1

    def _grade(self, run: Any, family: str, order: str) -> tuple[Any, ...]:
        return ()


class _Run:
    def __init__(self, i: int) -> None:
        self.run_id = f"r{i}"


def _items(n: int) -> list[tuple[Any, str, str]]:
    return [(_Run(i), "keypoint", "ab") for i in range(n)]


def test_warm_grader_recycles_every_n_calls(monkeypatch) -> None:
    """The remedy has to actually fire during a long pass, which is when decay happens."""
    from pi_eval.judges.harness import warm_grader

    monkeypatch.setenv("PI_JUDGE_RECYCLE_CALLS", "10")
    llm = _RecordingLLM()
    g = _FakeGrader(llm)
    warm_grader(g, _items(35), workers=2, batch=5)
    assert llm.recycles == 3, f"expected 3 recycles over 35 calls at every 10, got {llm.recycles}"


def test_recycling_can_be_switched_off(monkeypatch) -> None:
    """0 means never. An operator must be able to disable an empirical remedy."""
    from pi_eval.judges.harness import warm_grader

    monkeypatch.setenv("PI_JUDGE_RECYCLE_CALLS", "0")
    llm = _RecordingLLM()
    warm_grader(_FakeGrader(llm), _items(50), workers=2, batch=5)
    assert llm.recycles == 0


def test_a_judge_without_recycle_is_left_alone(monkeypatch) -> None:
    """`pi_eval` must not import `pi_run`, so the hook is duck-typed.

    A grader whose llm has no `recycle` (every test double, and `judge_replay`'s client)
    must warm normally rather than raise.
    """
    from pi_eval.judges.harness import warm_grader

    monkeypatch.setenv("PI_JUDGE_RECYCLE_CALLS", "10")
    g = _FakeGrader(object())
    warm_grader(g, _items(25), workers=2, batch=5)
    assert len(g._cache) == 25
