"""Run forked tau2 rollouts from imported public trace prefixes.

WHY A SCRIPT AND NOT `pi run`. A fork's unit is addressed by (task, trace, cut point), not by
(task, split, n) -- `pi run` enumerates a suite's task ids, and two forks of one task are
different units it has no way to name. The specs are built here and handed to the same
`run_tau2_unit` the sweep uses, so a forked run is byte-identical in shape to any other: same
seven files, same status dict, same resume sentinel.

TWO SOURCES OF FORK POINTS, and they are not interchangeable.

  --forkpoints FILE          A candidate POOL: every legal cut some producer found. It is
                             filtered by `--min-k`/`--max-k` and sampled down to `--n`, because
                             one trace can offer eight cut points and another three, and taking
                             the first N would draw most of them from a handful of long
                             dialogues. This is convlog-work's own input format.

  --forkpoints-from-runs DIR A RECORD: the cuts a campaign already ran, read back out of the
                             fork manifests it wrote. Not a pool, so it is neither filtered nor
                             sampled -- passing `--n`, `--min-k` or `--max-k` with it is refused
                             rather than silently launching a different campaign than the flags
                             name. This is how the recorded retail and airline campaigns are
                             re-launched (or re-checked) without the fork-point producer, which
                             lives in `pi_run/traces/` and is not on main.

DEEP CUTS ARE THE POINT, for the pool. `--min-k` defaults above the opening exchange because a
fork at k=2 is nearly a fresh rollout: the states worth forking to are the mid-dialogue ones
where evidence is already in hand, which is exactly what our own rollouts -- all starting at
turn 0 -- can never reach.

--dry-run PRINTS THE UNITS AND LAUNCHES NONE. Launching is the only irreversible thing here: a
campaign is fork points x arms x seeds of paid rollout, and a stale points file or a wrong k
window is invisible until the bill arrives.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def sample_round_robin(points: Sequence[Mapping[str, Any]], n: int) -> list[dict]:
    """Draw across tasks first, then depth, so a few long dialogues cannot dominate."""
    by_task: dict[str, list[dict]] = collections.defaultdict(list)
    for p in sorted(points, key=lambda x: -int(x["k"])):
        by_task[str(p["task_id"])].append(dict(p))
    out: list[dict] = []
    while len(out) < n and any(by_task.values()):
        for tid in sorted(by_task):
            if by_task[tid] and len(out) < n:
                out.append(by_task[tid].pop(0))
    return out


def in_k_window(points: Iterable[Mapping[str, Any]], *, min_k: int, max_k: int) -> list[dict]:
    return [dict(p) for p in points if min_k <= int(p["k"]) <= max_k]


def pool_points(raw: Any, *, path: str = "") -> list[dict]:
    """Fork points, from either shape `--forkpoints` has ever been asked to load.

    convlog-work's own format is a bare JSON list. `conf/forks/tau2_retail_dev.*.json` are
    dicts carrying provenance metadata (selection rule, k histograms, the sha256 of the pool
    they were drawn from) with the actual point list under `fork_points` -- not `points`, not
    `pool`, not `data`. Before this function existed, handing the wrapped shape straight to
    `in_k_window` raised `TypeError: string indices must be integers, not 'str'`: it was
    iterating the dict's own string keys instead of the list of points nested inside it.
    """
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict) and isinstance(raw.get("fork_points"), list):
        return raw["fork_points"]
    where = f" ({path})" if path else ""
    shape = f"dict with keys {sorted(raw.keys())}" if isinstance(raw, dict) else type(raw).__name__
    raise ValueError(
        f"--forkpoints{where} is neither a bare JSON list nor a dict with a list under "
        f"'fork_points' -- got a {shape}"
    )


def _recorded_forks(
    runs_root: str | Path, *, suite: str, split: str, arms: Sequence[str]
) -> list[dict]:
    """The fork runs of one campaign: this suite, this split, THESE ARMS, status ok.

    THE ARM FILTER IS NOT A CONVENIENCE. `runs/` accumulates forks from every campaign that has
    ever run -- the retail tree holds cuts under five different arms -- so an unfiltered read
    returns 174 points where the recorded two-arm campaign ran 34, and a `--dry-run` checked
    against the report would disagree with it for a reason that looks like a bug in the report.
    This is the same selection `pi_eval.fork_report` pairs on, so the two agree by construction.

    Status ok for the same reason it is required there: a run that errored is not evidence that
    its cut produced a comparable continuation.
    """
    out: list[dict] = []
    wanted = set(arms)
    for m in sorted(Path(runs_root).glob("*/manifest.json")):
        try:
            d = json.loads(m.read_text())
        except (OSError, ValueError):
            continue
        if (
            d.get("suite_id") != suite
            or str(d.get("split")) != split
            or d.get("arm_id") not in wanted
            or not d.get("foreign_trace_sha")
        ):
            continue
        st = m.parent / "status.json"
        if not st.is_file():
            continue
        try:
            status = json.loads(st.read_text())
        except (OSError, ValueError):
            continue
        if status.get("status") not in (None, "ok"):
            continue
        out.append(d)
    return out


def points_from_runs(
    runs_root: str | Path, *, suite: str, split: str, arms: Sequence[str] = ()
) -> list[dict]:
    """The fork points a campaign ALREADY RAN, recovered from the manifests it wrote.

    `foreign_trace_sha` and `foreign_prefix_k` are in SEMANTIC_FIELDS, so every fork run
    records the exact cut it was launched at -- which makes the run directories a faithful
    record of the campaign and not merely of its results.

    A run with no `foreign_trace_sha` is a full-task run and contributes NOTHING: a point built
    from one would launch a fresh rollout into the fork table, which is the failure the whole
    fork path is arranged to prevent.

    Returns the DISTINCT (trace_sha, k, task_id) triples, sorted, so two invocations name the
    same campaign in the same order -- a `--dry-run` that was checked and the launch that
    follows it must be the same set.
    """
    seen = {
        (str(d["foreign_trace_sha"]), int(d["foreign_prefix_k"]), str(d["task_id"]))
        for d in _recorded_forks(runs_root, suite=suite, split=split, arms=arms)
    }
    return [{"trace_sha": s, "k": k, "task_id": t} for s, k, t in sorted(seen)]


def source_runs(
    runs_root: str | Path, *, suite: str, split: str, arms: Sequence[str] = ()
) -> list[str]:
    """The run ids the points above were read out of. Printed as provenance so a dry run can be
    compared against `provenance.run_ids_sha` in docs/reports/forks_tau2_<suite>_test.json,
    which digests exactly this set of ids in exactly this way."""
    return sorted(
        str(d["run_id"]) for d in _recorded_forks(runs_root, suite=suite, split=split, arms=arms)
    )


def build_specs(
    points: Sequence[Mapping[str, Any]],
    *,
    suite: str,
    arms: Sequence[str],
    seeds: Sequence[int],
    max_turns: int = 16,
    budget_cap: int = 16,
    k: int | None = None,
    code_version: str = "",
    dirty: bool = True,
    dirty_files: tuple[str, ...] = (),
    runs_root: str = "runs",
    cache_root: str | None = None,
    prompt_variant: str = "",
) -> list[Any]:
    """The cross product, in a fixed order: points x arms x seeds.

    THE ARM IS CHECKED HERE, before anything is launched. `run_tau2_unit` would raise on a
    typo too -- once per unit, after the sweep had already started paying for the correct ones.

    `code_version`/`dirty` are computed ONCE by the caller (see `_git_state`) and stamped
    uniformly here, mirroring `pi_run.cli`/`pi_run.sweep.plan`'s normal path exactly --
    `UnitSpec.dirty` defaults `True`, so a caller that does not pass this gets `dev-` runs
    even out of a clean tree, which is the defect this parameter closes.

    `k` (retrieval top-k) is OMITTED from the UnitSpec(...) call below when the caller does
    not pass one, rather than defaulted here to some launcher-local number -- this used to
    default to 4 while `UnitSpec`'s own class default is 5, so every fork spec this launcher
    ever built silently carried a retrieval-k no other caller of UnitSpec uses, in a field
    nobody passed and no flag could set. See `--k`'s help text in `main()`.
    """
    from pi_run.worker import UnitSpec
    from pinq_expt import arms as arm_table

    for arm in arms:
        arm_table.get(arm)  # raises UnknownArm; a typo costs a message, not a campaign

    return [
        UnitSpec(
            suite_id=suite,
            task_id=str(p["task_id"]),
            arm_id=arm,
            seed=int(seed),
            corpus_dir="",
            cache_root=cache_root or os.environ.get("PI_CACHE_ROOT", "cache"),
            runs_root=runs_root,
            code_version=code_version,
            foreign_trace_sha=str(p["trace_sha"]),
            foreign_prefix_k=int(p["k"]),
            budget_cap=budget_cap,
            max_turns=max_turns,
            dirty=dirty,
            dirty_files=dirty_files,
            # A FOREIGN-PREFIX RUN IS EXPLORATORY BY CONSTRUCTION: it continues someone else's
            # dialogue, so its follow-up count is measured from a different origin than a full
            # run's. `ELIGIBLE` also refuses it via foreign_trace_sha; this makes the manifest
            # say so on its own.
            exploratory=True,
            # EMPTY IS THE SHIPPED PROMPT SET, which is what the recorded campaigns rendered.
            # A named variant reaches the manifest as `prompt_variant_id`, so its runs cannot
            # be pooled with theirs by accident.
            prompt_variant=prompt_variant,
            # k IS OMITTED WHEN THE CALLER DOESN'T PASS ONE, not defaulted here to some
            # launcher-local number: UnitSpec's own class default must then apply, so this
            # launcher cannot silently diverge from every other caller of UnitSpec the way
            # it did before (k=4 here vs UnitSpec's own 5 -- every one of the 204 recorded
            # retail dev runs carries 5).
            **({"k": k} if k is not None else {}),
        )
        for p in points
        for arm in arms
        for seed in seeds
    ]


def _run_unit(spec: Any) -> dict:
    """Indirection so `--dry-run` can be tested by replacing this with a detonator."""
    from pi_run.stages.tau2_runner import run_tau2_unit

    return run_tau2_unit(spec)


def _git_state(root: Path = ROOT) -> Any:
    """The tree's identity, computed once -- mirrors the normal sweep path (`pi_run.cli`,
    `pi_run.sweep.plan`), which computes `git_info(root)` once and forwards BOTH `gi.sha` and
    `gi.dirty` from the same call. The old `_git_head()` only ever fetched the sha by hand and
    never looked at dirtiness at all, which is the other half of why every fork spec came out
    `dev-` regardless of the tree's real state (`UnitSpec.dirty` defaults `True`).

    A seam, not a bare call, so a test can report a clean or dirty tree without a real git
    process or a real working tree to match it -- `monkeypatch.setattr(launcher, "_git_state",
    ...)`, the same indirection `_run_unit` gives `--dry-run`.
    """
    from pi_run.manifest import git_info

    return git_info(str(root))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--forkpoints", help="a candidate POOL of legal cuts, as JSON")
    src.add_argument(
        "--forkpoints-from-runs",
        help="a runs/ tree; replays the cuts its fork manifests recorded, unsampled",
    )
    ap.add_argument("--suite", default="tau2_airline")
    ap.add_argument("--split", default="test", help="only with --forkpoints-from-runs")
    ap.add_argument("--arm", action="append", default=None, help="repeatable")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    # DEFAULTS ARE None SO AN EXPLICIT FLAG IS DISTINGUISHABLE FROM AN UNPASSED ONE, which is
    # what lets --forkpoints-from-runs refuse sampling instead of quietly ignoring it. Same
    # pattern, same reason, as `pi run`'s `resolved_run_defaults`.
    ap.add_argument("--n", type=int, default=None, help="fork points to sample (pool only)")
    ap.add_argument("--min-k", type=int, default=None)
    ap.add_argument("--max-k", type=int, default=None)
    # MEASURED: 60-65% of forks on every suite end at the 16-turn cap, so the cap binds
    # everywhere -- but airline still succeeds 48% under it while retail succeeds 0% while
    # making 34 env calls where its source trace needed 3. The cap is therefore not what
    # separates them, and raising it is an experiment rather than a fix. Exposed so that
    # experiment can be run without editing the driver.
    ap.add_argument("--max-turns", type=int, default=16)
    ap.add_argument("--budget-cap", type=int, default=16)
    ap.add_argument(
        "--k",
        type=int,
        default=None,
        help=(
            "retrieval top-k handed to every unit's Inquirer. Unset defers to UnitSpec's "
            "own class default, so this launcher cannot silently diverge from every other "
            "caller of UnitSpec the way it used to (build_specs() hardcoded k=4 here while "
            "UnitSpec's own default is 5, and the 204 recorded retail dev runs all carry 5)."
        ),
    )
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument(
        "--prompt-variant",
        default="",
        help=(
            "a named prompt variant: 'tau2_base' (tau2_runner.TAU2_BASE_VARIANT -- the shared "
            "Inquirer prompt plus commit 3052546's routing paragraph, 'a question reaches "
            "records, never the customer', which this line of history never merged) or "
            "'tau2_stop' (TAU2_STOP_VARIANT -- tau2_base with the WHEN TO STOP paragraph "
            "replaced, and nothing else, so stop-vs-base isolates stopping). Unset renders the "
            "shipped prompts, which is what the recorded campaigns rendered; the name lands in "
            "the manifest as prompt_variant_id, which is in SEMANTIC_FIELDS"
        ),
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="print the units that would be launched, and launch none of them",
    )
    a = ap.parse_args(argv)
    arms = a.arm or ["inquirer_prompted", "inquirer_may_ask_user"]

    if not a.forkpoints and not a.forkpoints_from_runs:
        ap.error("one of --forkpoints or --forkpoints-from-runs is required")

    provenance = ""
    if a.forkpoints_from_runs:
        sampling = [f for f, v in (("--n", a.n), ("--min-k", a.min_k), ("--max-k", a.max_k)) if v]
        if sampling:
            ap.error(
                f"{', '.join(sampling)} cannot be used with --forkpoints-from-runs: those points "
                "are a RECORD of a campaign that ran, not a pool to draw from, and filtering "
                "them would launch a different campaign than the one being replayed"
            )
        chosen = points_from_runs(a.forkpoints_from_runs, suite=a.suite, split=a.split, arms=arms)
        ids = source_runs(a.forkpoints_from_runs, suite=a.suite, split=a.split, arms=arms)
        digest = hashlib.sha256("\n".join(ids).encode()).hexdigest()[:16]
        provenance = f"  source fork runs: {len(ids)}  run_ids_sha {digest}"
    else:
        raw = json.load(open(a.forkpoints))
        pool = pool_points(raw, path=a.forkpoints)
        if isinstance(raw, dict):
            # AN EXPLICIT SELECTION, not a candidate pool to draw from. Someone already chose
            # exactly these points (see conf/forks/tau2_retail_dev.selected34.json's own
            # `selection_rule`) -- --min-k 8 / --n 12 are POOL-path defaults tuned for
            # convlog-work's raw candidate lists, and silently re-applying them here would
            # launch a fraction of a named campaign with the missing points visible nowhere
            # (measured: 34 points -> 12, with no warning). Default to using every point; if an
            # explicit flag combination would still drop any, refuse instead of truncating.
            k_flags_given = a.min_k is not None or a.max_k is not None
            n_given = a.n is not None
            if not k_flags_given and not n_given:
                chosen = list(pool)
                window = pool
                provenance = f"  explicit selection: {len(pool)} points (no k window applied)"
            else:
                window = in_k_window(
                    pool,
                    min_k=a.min_k if a.min_k is not None else 0,
                    max_k=a.max_k if a.max_k is not None else 10**9,
                )
                dropped_by_window = len(pool) - len(window)
                n = a.n if a.n is not None else len(window)
                chosen = sample_round_robin(window, n)
                dropped_by_n = len(window) - len(chosen)
                if dropped_by_window or dropped_by_n:
                    given_k_flags = [
                        f for f, v in (("--min-k", a.min_k), ("--max-k", a.max_k)) if v is not None
                    ]
                    culprits = []
                    if dropped_by_window:
                        culprits.append(
                            f"{dropped_by_window} point(s) fall outside the k window "
                            f"({'/'.join(given_k_flags)})"
                        )
                    if dropped_by_n:
                        culprits.append(f"{dropped_by_n} point(s) dropped by --n {a.n}")
                    ap.error(
                        f"--forkpoints names an explicit selection of {len(pool)} point(s) "
                        f"({a.forkpoints}); {'; '.join(culprits)} would be excluded. An "
                        "explicit selection's points ARE the campaign -- pass a flag "
                        "combination that keeps all of them (e.g. --min-k 0 --max-k <max> "
                        f"--n {len(pool)}), or omit these flags entirely to use every point."
                    )
                provenance = (
                    f"  explicit selection: {len(pool)} points, {len(window)} in the k window"
                )
        else:
            window = in_k_window(
                pool,
                min_k=a.min_k if a.min_k is not None else 8,
                max_k=a.max_k if a.max_k is not None else 40,
            )
            chosen = sample_round_robin(window, a.n if a.n is not None else 12)
            provenance = f"  pool: {len(pool)} points, {len(window)} in the k window"

    if not chosen:
        print(
            f"no fork points for {a.suite}/{a.split} from "
            f"{a.forkpoints_from_runs or a.forkpoints}; nothing to launch",
            file=sys.stderr,
        )
        return 2

    gi = _git_state()
    specs = build_specs(
        chosen,
        suite=a.suite,
        arms=arms,
        seeds=a.seeds,
        max_turns=a.max_turns,
        budget_cap=a.budget_cap,
        k=a.k,
        code_version=gi.sha,
        dirty=gi.dirty,
        dirty_files=gi.dirty_files,
        runs_root=a.runs_root,
        prompt_variant=a.prompt_variant,
    )

    # THE ROOTS MUST BE ABSOLUTE, CHECKED HERE BECAUSE THIS IS WHERE THE CLASS LIVES.
    # `--runs-root` defaults to "runs" and `cache_root` falls back to `$PI_CACHE_ROOT`, and
    # `.env` -- which every lane sources, because a missing gateway key hangs in retry backoff
    # instead of erroring -- sets BOTH to relative values (`./runs`, `./cache`). A relative root
    # resolves against whatever directory the caller happens to be standing in.
    #
    # MEASURED 2026-09-19: a launcher validated an absolute runs root, then sourced `.env` for
    # the key, and the file's `./runs` silently replaced the approved value. Twelve shards wrote
    # 117 run directories into a pinned worktree, where no reporting path looks; the campaign
    # reported every unit `ok` and cost almost nothing, and both were true. Separately, a cache
    # probe once read a 16% hit rate against a cache holding 97.1% for the same reason, and that
    # was reported as a finding about the cache. It was a finding about the path.
    #
    # Refused here rather than documented, because the next launcher's author will not read the
    # documentation and WILL pass a relative root by omission.
    #
    # SCOPED TO A LAUNCH, NOT A PLAN. This check originally ran before the `--dry-run` return
    # below, and `--runs-root` defaults to the relative "runs", so every caller that asked only
    # to SEE a plan got exit 4 instead of one -- including this repository's own launch preflight
    # and 15 tests in test_run_tau2_forks.py, which is how the over-reach was found. A dry run
    # writes no run directory and opens no cache, so a caller-relative root cannot mislocate
    # anything and refusing it buys nothing. The guard is unchanged for anything that launches.
    if not a.dry_run:
        for name, val in (("--runs-root", a.runs_root), ("cache_root", specs[0].cache_root)):
            if not os.path.isabs(str(val)):
                print(
                    f"REFUSING TO LAUNCH: {name}={val!r} is not absolute. It resolves against the "
                    f"caller's working directory ({os.getcwd()}), so these runs would land "
                    "somewhere the reporting path never looks and no error would be raised. "
                    "Pass an absolute path.",
                    file=sys.stderr,
                )
                return 4

    # A SELF-CHECK ON THIS WIRING, not on the operator. build_specs() above is supposed to
    # stamp gi.sha/gi.dirty onto every spec uniformly; if it -- or UnitSpec's own default --
    # ever regresses to ignoring what was passed, the failure is silent: every launched run is
    # quietly `dev-` on a clean tree (or quietly not, on a dirty one) with nothing to say so.
    # Refuse to launch anything rather than let that defect come back a second time unnoticed.
    bad_dirty = [s for s in specs if s.dirty != gi.dirty or s.dirty_files != gi.dirty_files]
    bad_code = [s for s in specs if s.code_version != gi.sha]
    if bad_dirty or bad_code:
        print(
            f"REFUSING TO LAUNCH: build_specs() produced specs whose git identity disagrees "
            f"with the tree's own. git_info({ROOT}) reports dirty={gi.dirty} sha={gi.sha!r}; "
            f"{len(bad_dirty)}/{len(specs)} spec(s) carry a different dirty flag and "
            f"{len(bad_code)}/{len(specs)} spec(s) carry a different code_version. This is a "
            "self-check on the launcher's own wiring, not a data problem -- nothing launched.",
            file=sys.stderr,
        )
        return 3

    print(f"{a.suite} {a.split}: fork points: {len(chosen)}  arms {arms}  seeds {a.seeds}")
    print(provenance)
    print(f"  retrieval k={specs[0].k}")
    print(f"  would launch {len(specs)} units")
    for s in specs:
        print(
            f"    trace {s.foreign_trace_sha[:16]}  k={s.foreign_prefix_k:<3} "
            f"task={s.task_id:<5} seed={s.seed}  {s.arm_id}"
        )
    if a.dry_run:
        print("  --dry-run: nothing launched")
        return 0

    counts: collections.Counter = collections.Counter()
    for s in specs:
        try:
            st = str(_run_unit(s).get("status"))
        except Exception as exc:  # noqa: BLE001 - one fork's failure is a counted row
            st = f"raised:{type(exc).__name__}"
        counts[st] += 1
        print(
            f"  task {s.task_id:>5} k={s.foreign_prefix_k:<3} seed={s.seed} {s.arm_id:24s} {st}",
            flush=True,
        )
    print("status counts:", dict(counts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
