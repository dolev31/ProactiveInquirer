"""The judge client `pi score` reaches a model through, and the only implementation of the
`PI_JUDGE_CLIENT` hook.

WHY THIS FILE IS IN `pi_run` AND NOT IN `pi_eval`. `pi_eval` depends on `pinq` and nothing
else first-party — that is what keeps the gold-only package free of the runner, the providers
and their transitive pins, and it is one of the four things the import contracts enforce. But
`pi score` IS the process that judges, so it has to reach a model somehow. The resolution is a
DEPLOYMENT hook: `PI_JUDGE_CLIENT="module:factory"`, resolved by string at call time inside
`pi_eval.score.judge_client_from_env`. No static import edge is created in either direction,
so this file exists without a single contract moving.

WHAT WAS ACTUALLY WRONG. `PI_JUDGE_CLIENT` appeared in the tree exactly once, as the NAME of
an environment variable. Nothing implemented it. So `pi score` took the documented
"no client, and here is why" path on every invocation, `judgments.parquet` stayed empty, and
`kpr_incremental` — a PRIMARY endpoint, P3 — emitted zero rows. The failure was honest (it
said `skipped: PI_JUDGE_CLIENT unset`) and completely silent to anyone reading a table, since
a metric with no rows is a metric that simply is not there.

THREE PROPERTIES, EACH LOAD-BEARING.

  * CACHED, through the same content-addressed store the rollouts use. Re-scoring under a new
    `scorer_hash` is meant to be a groupby, never a re-roll; a judge that re-billed on every
    `pi score` would make re-scoring cost money and would therefore stop happening.
  * METERED. The judge's spend is real and belongs in the campaign's accounting rather than
    off the books. `spend()` reports it; `pi score` prints it.
  * REPLAY-ONLY IN CI. `judge_replay()` raises on a cache miss instead of dispatching, so a
    test that would have made a network call is a stack trace on line one rather than an
    invoice.
"""

from __future__ import annotations

import os
from typing import Any, Mapping

from pinq.budget import BudgetLedger

# The judge is a ROLE, and it resolves through the same ROLE_ENV table as every other role, so
# `PI_MODEL_JUDGE` means one thing across the runner and the scorer. The client refuses to
# default a pin, which is what stops two arms being judged by different models.
JUDGE_ROLE = "judge"

# Judging must be deterministic for the same reason drafting must: `sigma_j` is estimated from
# test-retest of identical and paraphrased pairs, and a judge sampling at t>0 would inflate the
# identical-pair variance and widen every band built on it.
JUDGE_TEMPERATURE = 0.0


class _Judge:
    """A `pi_eval.judges._llm.JudgeLLM`, plus the two things the protocol does not carry:
    what it spent, and how often the cache saved a call."""

    def __init__(self, inner: Any, ledger: BudgetLedger, *, rebuild: Any = None) -> None:
        self._inner = inner
        self._ledger = ledger
        self._rebuild = rebuild
        self.recycles = 0

    def recycle(self) -> bool:
        """Rebuild the client stack, keeping the ledger. False when there is nothing to rebuild.

        MEASURED: judging decays inside one process -- 47 -> 13 calls/min on one pass, 71 -> 22
        on another -- while a FRESH PROCESS at the same provider, same cache and twice the
        concurrency held 71/min. Cache sharding was ruled out (256 dirs, ~125 files each). The
        root cause is unproven, so this is an empirical remedy and is measured, not assumed.

        THE LEDGER IS NOT REBUILT. `spend()` reads it and `--judge-spend-cap` enforces from it,
        so a fresh ledger would zero the judge's bill at every recycle: the cap would stop
        triggering and the reported spend would cover only the calls since the last rebuild.
        Both failures are silent, which is why the test for it is the first one in the file.

        `judge_replay` has no rebuild by construction -- it is cache-only and must never need
        the network -- so it declines instead of raising, or every CI re-score would break.
        """
        if self._rebuild is None:
            return False
        self._inner = self._rebuild()
        self.recycles += 1
        return True

    def complete(self, **kw: Any) -> tuple[str, Any]:
        return self._inner.complete(**kw)

    def request_payload(self, **kw: Any) -> dict[str, Any]:
        return self._inner.request_payload(**kw)

    def pin(self, role: str = JUDGE_ROLE) -> Any:
        return self._inner.pin(role)

    # ---------------------------------------------------------------- accounting

    def spend(self) -> dict[str, float]:
        """As-if and billed, the same split the sweep reports. Their difference is the cache."""
        calls = list(self._ledger.calls)
        return {
            "usd": round(sum(float(c.usd) for c in calls), 6),
            "usd_billed": round(sum(float(c.usd) for c in calls if not c.cache_hit), 6),
            "n_calls": len(calls),
            "cache_hits": sum(1 for c in calls if c.cache_hit),
            "tok_total": sum(c.tok_prompt + c.tok_completion + c.tok_reasoning for c in calls),
        }


def _ledger() -> BudgetLedger:
    # An unreachable cap. The hard cap is `retrieval_calls` and a judge performs none; capping
    # judging by token count would silently truncate a scoring pass and leave a table with an
    # arbitrary subset of its rows judged.
    return BudgetLedger(cap=10**9)


def judge(env: Mapping[str, str] | None = None, *, cache_root: str | None = None) -> _Judge:
    """The production judge: metered, cached, temperature-pinned. `PI_JUDGE_CLIENT` points here.

    PI_JUDGE_CLIENT=pi_run.judge_client:judge
    PI_MODEL_JUDGE=<the pin>
    """
    from pi_run.cache import CachingClient, DiskCache
    from pi_run.cache import cache_root as _root
    from pinq_adapters.llm.litellm_client import MeteredClient

    e = dict(os.environ if env is None else env)
    ledger = _ledger()
    root = cache_root or _root(e.get("PI_CACHE_ROOT"))

    def _build() -> Any:
        # The SAME ledger and the SAME cache root every time, so a recycle changes only the
        # HTTP client underneath -- not what has been spent, and not what has been cached.
        return CachingClient(
            MeteredClient(ledger, temperature=JUDGE_TEMPERATURE, env=e),
            DiskCache(root),
            ledger=ledger,
        )

    return _Judge(_build(), ledger, rebuild=_build)


def judge_replay(env: Mapping[str, str] | None = None, *, cache_root: str | None = None) -> _Judge:
    """Cache-only. A miss RAISES rather than dispatching.

        PI_JUDGE_CLIENT=pi_run.judge_client:judge_replay

    This is what CI and a re-score should use. A judged metric recomputed from the cache is
    reproducible by anyone holding the cache; one recomputed by re-querying a model is a new
    measurement wearing an old `scorer_hash`.
    """
    from pi_run.cache import DiskCache, ReplayClient
    from pi_run.cache import cache_root as _root
    from pinq_adapters.llm.litellm_client import MeteredClient

    e = dict(os.environ if env is None else env)
    ledger = _ledger()
    inner = MeteredClient(ledger, temperature=JUDGE_TEMPERATURE, env=e)
    replay = ReplayClient(
        inner, DiskCache(cache_root or _root(e.get("PI_CACHE_ROOT"))), ledger=ledger
    )
    return _Judge(replay, ledger)
