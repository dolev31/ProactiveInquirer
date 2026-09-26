"""Fan out (task, arm, seed) units across processes and reassemble them BY KEY.

Two rules, both learned the expensive way:

  1. RESULTS ARE REASSEMBLED BY KEY, NEVER BY COMPLETION ORDER. `as_completed` yields in
     whatever order the OS scheduler produced, and zipping that against the input list
     silently mislabels every row — an error that produces a plausible table rather than a
     crash, and would survive review.

  2. PARALLELISM IS PROCESSES, NOT THREADS. Each unit owns its BudgetLedger, so there is
     nothing to share and nothing to lock; the process boundary is what makes that literally
     true rather than a convention people forget under deadline.

Resumption is inherited from the worker: a unit whose status.json exists returns
status='resumed' without re-running, so re-issuing the identical sweep command after a crash
is the correct recovery procedure and costs one stat() per unit.
"""

from __future__ import annotations

import dataclasses
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable, Iterable, Mapping, Sequence

from pi_run.worker import UnitSpec, run_unit

DEFAULT_CONCURRENCY = 8

# READ, NOT RETYPED: `plan()`'s own `max_turns`/`k`/`budget_cap`/`concurrency` defaults below
# used to duplicate these four `UnitSpec` field defaults by coincidence of value. This module's
# one live caller (`pi_run.cli`) always passes all four explicitly, so this path currently
# resolves nothing in production -- but a future caller that omits one should still fall back
# to `UnitSpec`'s own default, not to a second, independently-maintained copy of it.
_UNIT_SPEC_DEFAULTS = {f.name: f.default for f in dataclasses.fields(UnitSpec)}


class SpendCapExceeded(RuntimeError):
    """The campaign spend cap was reached. Remaining units were cancelled, not run."""


class SpendCapInvalid(ValueError):
    """A spend cap that cannot be honoured as written."""


def spend_cap(explicit: float | None = None) -> float | None:
    """The campaign cap in dollars, or None for uncapped. AN EXPLICIT ARGUMENT WINS.

    A CAMPAIGN cap, checked in the parent as results return, because the per-unit ledger
    cannot see a runaway sweep: 15,600 units each costing a defensible $0.007 is $109 and no
    single unit ever looks wrong.

    Same precedence rule as `max_concurrency`, for the same reason. A cap left in .env from a
    cautious first evening is a number nobody re-reads, and the failure it produces -- a grid
    stopping two thirds of the way through with a line buried above the summary -- looks like
    the grid finishing. `--spend-cap` states this invocation's budget where the invocation is
    read; PI_SPEND_CAP_USD remains the guard rail for an unattended run.
    """
    # ZERO MEANS ZERO. This read `if float(explicit) > 0 else None`, so `--spend-cap 0` --
    # the most natural way to type "spend nothing", and the obvious way to ask for a
    # rehearsal -- returned None, which means UNCAPPED. Worse, because an explicit argument
    # wins, it also discarded any PI_SPEND_CAP_USD guard rail that was set. Measured before
    # the fix: with PI_SPEND_CAP_USD=5.00, `spend_cap(0.0)` returned None while
    # `spend_cap(None)` returned 5.0, so adding the flag REMOVED the protection.
    #
    # A negative cap is refused rather than reinterpreted. There is no sensible reading of a
    # negative budget, and quietly picking one is exactly how 0 came to mean "unlimited".
    if explicit is not None:
        v = float(explicit)
        if v < 0:
            raise SpendCapInvalid(
                f"--spend-cap {explicit} is negative. Use 0 to run nothing billable, or a "
                "positive dollar amount. It is not a way to disable the cap; omit the flag "
                "for that."
            )
        return v
    raw = os.environ.get("PI_SPEND_CAP_USD")
    if raw:
        try:
            v = float(raw)
            return v if v >= 0 else None
        except ValueError:
            pass
    return None


def max_concurrency(explicit: int | None = None) -> int:
    """Resolve the worker count. AN EXPLICIT ARGUMENT WINS; PI_MAX_CONCURRENCY is the default.

    The precedence used to be the other way round, and that is worse than it looks. A throttle
    set once in .env for one VPN-flaky evening then silently outranked `--concurrency 16`
    forever after: `PI_MAX_CONCURRENCY=2 pi run --concurrency 16` resolved to 2, printed
    `concurrency=2` on a line nobody re-reads, and turned a 9-hour sweep into a 74-hour one
    with no error anywhere. An env var is a DEFAULT for the unattended case; a flag is an
    instruction for this invocation, and an instruction must beat a default.

    The binding constraint on a real sweep is provider TPM rather than CPU, which is why this
    is an explicit number and never os.cpu_count().
    """
    if explicit is not None:
        return max(1, int(explicit))
    raw = os.environ.get("PI_MAX_CONCURRENCY")
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    return DEFAULT_CONCURRENCY


def billed_usd(res: Mapping[str, Any]) -> float:
    """The INVOICE for one unit result: money the provider will actually charge.

    Two subtractions, both of which the naive `usage["usd"]` gets wrong in the same direction:

      * A RESUMED unit spent its dollars on an earlier day. Counting them again meant a sweep
        that had already hit its cap could never be resumed past it -- the second invocation
        re-read the same historical totals, tripped the cap in the first few stat() calls, and
        cancelled every unit that had not yet run. A cap that makes recovery impossible is a
        cap that gets deleted from .env under deadline, which is the failure mode this whole
        mechanism exists to avoid.

      * A CACHE HIT is charged AS-IF by the ledger, deliberately: `_telemetry_from` bills a hit
        its recorded tokens so that two arms stay comparable no matter how warm the cache
        happened to be. That is the right convention for a parity table and the wrong one for a
        spend cap, which is about the invoice. Runs written before `usd_billed` existed fall
        back to the as-if number, which over-counts rather than under-counts.
    """
    if str(res.get("status")) == "resumed":
        return 0.0
    v = res.get("usd_billed")
    if v is None:
        return float((res.get("usage") or {}).get("usd", 0.0))
    return float(v)


def progress_printer(
    total: int, *, every: int = 25, stream: Any = None
) -> Callable[[dict[str, Any]], None]:
    """A callback for `run_sweep(on_result=...)`.

    `run_sweep` has always accepted one and `cli.py` passed none, so a three-hour sweep printed
    its header, went silent, and printed its summary -- during which the only way to tell a
    wedged tunnel from slow progress was to count files in runs/. Silence is not neutral on a
    paid job: it is what turns a 20-minute diagnosis into a 3-hour one.

    Prints done/total, ok/err, the billed dollars, an ETA extrapolated from elapsed wall time,
    and the LAST DISTINCT error message -- distinct because 300 copies of one 429 traceback is
    noise and one is a diagnosis.
    """
    import sys as _sys
    import time as _time

    out = stream if stream is not None else _sys.stderr
    t0 = _time.perf_counter()
    seen = {"n": 0, "ok": 0, "err": 0, "usd": 0.0, "last": ""}

    def _cb(res: dict[str, Any]) -> None:
        seen["n"] += 1
        st = str(res.get("prior_status") or res.get("status", "?"))
        if res.get("error"):
            seen["err"] += 1
            msg = str(res.get("error"))[:120]
            if msg != seen["last"]:
                seen["last"] = msg
        elif st == "ok":
            seen["ok"] += 1
        seen["usd"] += billed_usd(res)
        n = int(seen["n"])
        if n % every and n != total:
            return
        el = _time.perf_counter() - t0
        eta = (el / n) * (total - n) if n else 0.0
        tail = f"  last-err: {seen['last']}" if seen["last"] else ""
        print(
            f"  [{n}/{total}] ok={seen['ok']} err={seen['err']} "
            f"${seen['usd']:.4f} elapsed={_fmt_hms(el)} eta={_fmt_hms(eta)}{tail}",
            file=out,
            flush=True,
        )

    return _cb


def _fmt_hms(seconds: float) -> str:
    s = int(max(0.0, seconds))
    return f"{s // 3600:d}h{(s % 3600) // 60:02d}m{s % 60:02d}s"


def plan(
    *,
    suite_id: str,
    corpus_dir: str,
    task_ids: Sequence[str],
    arm_ids: Sequence[str],
    seeds: Sequence[int],
    runs_root: str,
    cache_root: str,
    max_turns: int = _UNIT_SPEC_DEFAULTS["max_turns"],
    k: int = _UNIT_SPEC_DEFAULTS["k"],
    budget_cap: int | None = _UNIT_SPEC_DEFAULTS["budget_cap"],
    pilot: bool = False,
    resume: bool = True,
    code_version: str = "",
    dirty: bool = True,
    dirty_files: tuple[str, ...] = (),
    concurrency: int = _UNIT_SPEC_DEFAULTS["concurrency"],
    questions: Mapping[str, Sequence[str]] | None = None,
    # arm_id -> its own {task_id: [q, ...]}, overriding `questions` for that arm. The two
    # ceiling arms bound different things and must not be seeded from one file.
    questions_by_arm: Mapping[str, Mapping[str, Sequence[str]]] | None = None,
    # seed -> {task_id: [q, ...]}. A replay arm must replay the questions of the run it is
    # PAIRED with, and the treatment asks different questions at each seed; keyed by task
    # alone, a replay at seed 1 was handed seed 0's list. See
    # pinq_expt.policies.controls.recorded_questions.
    questions_by_seed: Mapping[int, Mapping[str, Sequence[str]]] | None = None,
    ask_counts: Mapping[str, int] | None = None,
    timeout_s: int | None = None,
    grid_sha256: str = "",
    grid_name: str = "",
    exploratory: bool = False,
) -> list[UnitSpec]:
    """The cross product, in a fixed order.

    Task-major rather than arm-major: with common random numbers across arms, adjacent units
    share a task and therefore share cache entries, so the arms of one task warm each other
    instead of evicting each other.
    """
    return [
        UnitSpec(
            suite_id=suite_id,
            corpus_dir=corpus_dir,
            task_id=tid,
            arm_id=aid,
            seed=seed,
            # The WHOLE pool reaches every unit, not just this task's entry. parallel_replay
            # wants its own task's questions, but random_q must draw from OTHER tasks -- a
            # random-question arm sampled from the task's own pool is not a null, it is a
            # weaker copy of the treatment. Slicing per task here made random_q refuse with
            # EmptyPool on every unit, which was the guard working against my own plumbing.
            questions={
                k: tuple(v)
                for k, v in (
                    (questions_by_arm or {}).get(aid)
                    or (questions_by_seed or {}).get(seed)
                    or questions
                    or {}
                ).items()
            },
            ask_counts=dict(ask_counts or {}),
            runs_root=runs_root,
            cache_root=cache_root,
            max_turns=max_turns,
            k=k,
            budget_cap=budget_cap,
            pilot=pilot,
            resume=resume,
            code_version=code_version,
            dirty=dirty,
            dirty_files=dirty_files,
            concurrency=concurrency,
            timeout_s=timeout_s,
            grid_sha256=grid_sha256,
            grid_name=grid_name,
            exploratory=exploratory,
        )
        for tid in task_ids
        for aid in arm_ids
        for seed in seeds
    ]


def drain_uncollected(futures: Mapping[Any, Any], by_key: dict[Any, Any]) -> float:
    """Record the units that were ALREADY RUNNING when the cap tripped. Returns their spend.

    `cancel_futures=True` drops only futures that had not started. The ones already in
    flight -- up to `workers` of them -- run to completion and bill, but the `as_completed`
    loop has already broken, so nothing ever read their results. Their run directories were
    written to disk and their money was spent, while the printed "$N billed" omitted both.
    A cap that under-reports its own overshoot is worse than no cap, because the number it
    prints is the one the operator writes down.

    Called after `shutdown(wait=True, ...)`, so every non-cancelled future is finished and
    `result()` returns immediately. A unit that RAISED spent tokens too, but has no result
    to record; it is skipped rather than allowed to take the rest of the accounting with it.
    """
    extra = 0.0
    for fut, key in futures.items():
        if key in by_key or fut.cancelled():
            continue
        try:
            res = fut.result(timeout=0)
        except BaseException:
            continue
        by_key[key] = res
        extra += billed_usd(res)
    return extra


def run_sweep(
    specs: Iterable[UnitSpec],
    *,
    concurrency: int | None = None,
    cap_usd: float | None = None,
    on_result: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    specs = list(specs)
    if not specs:
        return []
    workers = min(max_concurrency(concurrency), len(specs))

    by_key: dict[tuple, dict[str, Any]] = {}
    cap = spend_cap(cap_usd)
    spent = 0.0
    stopped = False
    overshoot_usd = 0.0
    if workers == 1:
        # In-process for a single worker: a pool of one buys nothing and costs a spawn plus
        # an unreadable traceback when a unit raises.
        for spec in specs:
            res = run_unit(spec)
            by_key[spec.key] = res
            spent += billed_usd(res)
            if on_result:
                on_result(res)
            if cap is not None and spent >= cap:
                stopped = True
                break
    else:
        ex = ProcessPoolExecutor(max_workers=workers)
        try:
            futures = {ex.submit(run_unit, spec): spec.key for spec in specs}
            for fut in as_completed(futures):
                key = futures[fut]
                res = fut.result()  # a firewall/reconcile failure propagates, by design
                by_key[key] = res
                spent += billed_usd(res)
                if on_result:
                    on_result(res)
                if cap is not None and spent >= cap:
                    # break, not raise: the finally below cancels every pending future, and
                    # the units already finished are real results worth keeping.
                    stopped = True
                    break
        finally:
            # cancel_futures is the whole point. `with ProcessPoolExecutor(...)` calls
            # shutdown(wait=True) WITHOUT it, so a firewall failure on unit 5 of 15,600 ran
            # the remaining 15,595 to completion before the traceback printed -- hours of
            # tokens burned after the results were already known to be void.
            ex.shutdown(wait=True, cancel_futures=True)
            # ...and the ones that were already RUNNING are not cancellable. They finished
            # and billed; collect them so the reported total is the invoice.
            overshoot_usd = drain_uncollected(futures, by_key)
            spent += overshoot_usd
    if stopped:
        over = (
            f" (including ${overshoot_usd:.4f} from units already in flight)"
            if overshoot_usd
            else ""
        )
        print(
            f"SPEND CAP: stopped after {len(by_key)}/{len(specs)} units, "
            f"${spent:.4f} billed >= ${cap:.2f}{over}. Remaining units were "
            f"CANCELLED, not run. A cap is a STOP-LOSS, not a ceiling: units already running "
            f"when it trips cannot be cancelled, so the total can exceed the cap by up to one "
            f"unit per worker. Re-issue the same command to continue: finished units resume "
            f"and, since a resumed unit bills nothing, the cap does not re-trip on them."
        )
    # Spec order, by key. Never the completion order the futures arrived in. Units that were
    # cancelled have no entry, so they are simply absent rather than silently zero-valued.
    return [by_key[spec.key] for spec in specs if spec.key in by_key]


def summarize(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Summarize a sweep, with FAILURE MADE LOUD.

    `resumed` deliberately does not hide the underlying outcome. A resumed unit that had
    previously errored is still an errored unit, and a sweep in which every unit failed once
    reported `{"resumed": N}` and read as a success -- the single most dangerous shape a
    summary can take, because the next stage then produces an empty table rather than an
    error. So the terminal status is resolved for resumed units too, `failed` is a top-level
    field, and `ok` is False whenever anything failed.
    """
    counts: dict[str, int] = {}
    terminal: dict[str, int] = {}
    errors: list[dict[str, str]] = []
    for r in results:
        st = str(r.get("status", "?"))
        counts[st] = counts.get(st, 0) + 1
        # A resumed unit reports the status it was resumed FROM, not "resumed".
        eff = str(r.get("prior_status") or r.get("status", "?")) if st == "resumed" else st
        if st == "resumed" and r.get("error"):
            eff = "error"
        terminal[eff] = terminal.get(eff, 0) + 1
        if r.get("error"):
            errors.append(
                {
                    "run_id": str(r.get("run_id", "")),
                    "key": "/".join(str(x) for x in (r.get("key") or [])),
                    "error": str(r.get("error"))[:200],
                }
            )
    failed = terminal.get("error", 0) + terminal.get("timeout", 0)
    # THE MARGIN, BEFORE IT BITES. A timeout excludes a unit, and the arms are not equally
    # exposed: measured live, the treatment runs 2.4-5x the baseline's wall time, so it is the
    # only arm that can realistically trip the cap. Reporting the slowest unit against the
    # limit lets an operator see the headroom shrinking on a cheap sweep instead of discovering
    # it as missing rows in an expensive one.
    from pi_run.worker import unit_timeout_s

    limit = unit_timeout_s()
    slowest = max((int(r.get("wall_ms", 0)) for r in results), default=0) / 1000.0
    margin: dict[str, Any] = {
        "slowest_unit_s": round(slowest, 1),
        "unit_timeout_s": limit,
        "headroom": round(limit / slowest, 1) if slowest and limit else None,
    }
    if limit and slowest and slowest > 0.5 * limit:
        margin["warning"] = (
            f"the slowest unit took {slowest:.0f}s against a {limit}s cap. A timeout EXCLUDES "
            "the unit, and the arms are not equally exposed -- raise --unit-timeout before the "
            "next sweep rather than after."
        )
    reconciled = sum(1 for r in results if r.get("reconciled") is True)
    dev = sum(1 for r in results if str(r.get("run_id", "")).startswith("dev-"))
    return {
        "units": len(results),
        "ok": failed == 0,
        "failed": failed,
        "by_status": dict(sorted(counts.items())),
        "by_terminal_status": dict(sorted(terminal.items())),
        # Distinct messages only: 36 copies of one traceback is noise, one is a diagnosis.
        "distinct_errors": sorted({e["error"] for e in errors})[:5],
        "example_failures": errors[:3],
        "reconciled": reconciled,
        "dev_runs": dev,
        "timeout_margin": margin,
        # `usd` is the AS-IF total (cache hits charged, resumed units included): the number
        # that keeps arms comparable. `usd_billed` is the invoice. They differ by exactly the
        # cache, so printing both is also the cache's savings report.
        "usd": round(sum(float((r.get("usage") or {}).get("usd", 0.0)) for r in results), 6),
        "usd_billed": round(sum(billed_usd(r) for r in results), 6),
        "tok_total": sum(int((r.get("usage") or {}).get("tok_total", 0)) for r in results),
    }
