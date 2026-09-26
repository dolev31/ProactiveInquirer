"""Does the FORK grading path still reproduce a known-good grade?

WHY THIS EXISTS. `pi verify tau2` replays each task's OWN answer key against its own
`initial_state` and reports 112/112 on retail. It never seeds gold from a foreign prefix, so it
validates the FLAT grading path and says nothing about the forked one. The published tau2
transfer rewards (retail 0.392 vs 0.167, airline 0.500 vs 0.529) were produced BEFORE
`40b3f95` ("load a foreign prefix, fork the task, and grade against the fork"), and a campaign
re-run at HEAD on the SAME 34 points, arms, seeds and caps scores zero on every control run --
joint probability 6.8e-7 under the published rates. Either HEAD grades forks correctly and those
cited numbers are not the quantity the paper says they are, or HEAD regressed. Under CLAUDE.md
rule 1 a cited number whose grader cannot reproduce it has no provenance, so this has to be
answerable, and no instrument in the tree could answer it.

WHY IT IS GOLD-VS-GOLD AND NOT A RUN RE-GRADE. A run's tau2 message transcript is NOT persisted
-- `outcome.json` keeps `env_calls` with `result_digest` only, `transcript_digest`, and
`env_final_hashes` -- so `_reward_of(sim, ...)`, which consumes `sim.messages`, cannot be replayed
from disk. What CAN be held fixed is the GOLD trajectory: apply the task's own answer key on top
of the fork prefix and ask whether the grader still returns a match. That is the fork analogue of
`replay_gold`, the trajectory is fixed by construction, and the expected answer is known (1.0), so
a failure is a statement about the grader and not about any policy.

WHY IT IS NOT IN pi_eval. A tau2 reward comes from upstream comparing DATABASE STATES, not from
our need graphs; nothing here reads `PI_GOLD_ROOT`. Putting a diagnostic that needs no gold inside
the gold-only module would make it require the gold root and grow the gold surface for nothing.
It imports `pi_run.stages.tau2_runner` and no gold module.

POSITIVE CONTROL RUNS FIRST AND ITS FAILURE IS FATAL. `--positive-control N` replays N non-fork
tasks through this same code and must reproduce, because an instrument that cannot reproduce a
case whose answer we already know cannot be believed on a case we do not.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def gold_actions(task):
    crit = getattr(task, "evaluation_criteria", None)
    acts = list(getattr(crit, "actions", None) or []) if crit else []
    return [{"name": a.name, "args": dict(a.arguments), "requestor": a.requestor} for a in acts]


def grade_gold_on(suite, tid, prefix):
    """Apply the task's OWN answer key with the environment seeded by `prefix`.

    prefix == [] is the flat path (the positive control). A non-empty prefix is the fork path.

    THE PREFIX IS DELIVERED THROUGH THE TASK, NOT BY CALLING set_state HERE. `Tau2Actuator`
    already seeds `initial_state.{initialization_data,initialization_actions,message_history}`
    for the task it is handed, and `forked_task` is exactly a copy whose `message_history` is
    the prefix -- the same object the Orchestrator and the evaluator both consume. Reaching
    into `Environment.set_state` by hand duplicated that seeding under a different signature
    and produced a TypeError that this script's first draft mis-reported as a grader failure.

    Returns (db_reward, n_actions, error).
    """
    from pi_run.stages.tau2_runner import forked_task
    from pinq_adapters.tau2.actuator import Tau2Actuator

    task = suite.tau2_task_object(tid)
    if prefix:
        task = forked_task(task, prefix)
    acts = gold_actions(task)
    if not acts:
        return None, 0, "NO_ACTIONS"
    act = Tau2Actuator(
        suite.environment(tid), task=task, domain=suite.domain, env_kwargs=suite.env_kwargs(tid)
    )
    act.execute(acts, turn_idx=0)
    return act.native().get("db_reward"), len(acts), ""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--suite", default="tau2_retail")
    ap.add_argument("--forkpoints", help="a recovered selection JSON (conf/forks/*.json)")
    ap.add_argument(
        "--positive-control",
        type=int,
        default=8,
        help="non-fork tasks to replay first; 0 disables (NOT recommended)",
    )
    ap.add_argument("--limit", type=int, default=0, help="fork points to test (0 = all)")
    a = ap.parse_args(argv)

    from pi_run.stages.tau2_runner import load_prefix
    from pi_run.worker import load_suite

    sink = io.StringIO()
    with contextlib.redirect_stdout(sink):
        suite = load_suite(a.suite, "")

    out = {"suite": a.suite, "positive_control": {}, "fork": {}}

    # ---------------------------------------------------------------- positive control
    if a.positive_control:
        ids = list(suite.task_ids())[: a.positive_control]
        rows = []
        for tid in ids:
            with contextlib.redirect_stdout(sink):
                try:
                    rw, n, err = grade_gold_on(suite, tid, [])
                except Exception as exc:  # noqa: BLE001
                    rw, n, err = None, 0, f"{type(exc).__name__}: {exc}"
            rows.append({"task_id": tid, "db_reward": rw, "n_actions": n, "error": err[:200]})
        graded = [r for r in rows if r["db_reward"] is not None]
        repro = [r for r in graded if r["db_reward"] == 1.0]
        out["positive_control"] = {
            "n": len(rows),
            "graded": len(graded),
            "reproduced": len(repro),
            "skipped_no_actions": sum(1 for r in rows if r["error"] == "NO_ACTIONS"),
            "failures": [r for r in graded if r["db_reward"] != 1.0][:5],
            "ok": bool(graded) and len(repro) == len(graded),
        }
        if not out["positive_control"]["ok"]:
            out["verdict"] = (
                "POSITIVE CONTROL FAILED -- this instrument cannot reproduce the "
                "flat grading that `pi verify tau2` reports, so its fork verdict "
                "is not believable. No conclusion about the fork path."
            )
            print(json.dumps(out, indent=2, sort_keys=True))
            return 2

    # ---------------------------------------------------------------- the fork path
    if a.forkpoints:
        pts = json.loads(Path(a.forkpoints).read_text())["fork_points"]
        if a.limit:
            pts = pts[: a.limit]
        rows = []
        for p in pts:

            class _S:  # the two fields load_prefix reads
                foreign_trace_sha = str(p["trace_sha"])
                foreign_prefix_k = int(p["k"])

            with contextlib.redirect_stdout(sink):
                try:
                    prefix = load_prefix(_S())
                    rw, n, err = grade_gold_on(suite, str(p["task_id"]), prefix)
                except Exception as exc:  # noqa: BLE001
                    prefix, rw, n, err = [], None, 0, f"{type(exc).__name__}: {exc}"
            rows.append(
                {
                    "task_id": str(p["task_id"]),
                    "k": int(p["k"]),
                    "prefix_len": len(prefix),
                    "db_reward": rw,
                    "n_actions": n,
                    "error": err[:200],
                }
            )
        graded = [r for r in rows if r["db_reward"] is not None]
        repro = [r for r in graded if r["db_reward"] == 1.0]
        out["fork"] = {
            "n": len(rows),
            "graded": len(graded),
            "reproduced": len(repro),
            "not_reproduced": [r for r in graded if r["db_reward"] != 1.0][:10],
            "errors": [r for r in rows if r["db_reward"] is None and r["error"] != "NO_ACTIONS"][
                :5
            ],
            "ok": bool(graded) and len(repro) == len(graded),
        }
        # THREE OUTCOMES, NOT TWO. `graded == 0` means THIS SCRIPT could not grade anything,
        # which is a fact about the instrument; reporting it as "the grader is broken" is the
        # same laundering `cmd_verify_tau2` documents for its own skip list, and this script's
        # first draft did exactly that off a TypeError of its own making.
        if out["fork"]["graded"] == 0:
            out["verdict"] = (
                "INCONCLUSIVE -- this instrument graded 0 of "
                f"{out['fork']['n']} fork points; see fork.errors. This says nothing about the "
                "grader. No conclusion either way."
            )
        elif out["fork"]["ok"]:
            out["verdict"] = (
                "FORK GRADING REPRODUCES the answer key on top of the prefix on "
                f"{out['fork']['reproduced']}/{out['fork']['graded']} graded points: the grader "
                "is sound on the fork path at this revision, so a zero reward on a forked run "
                "is a measurement of the policy."
            )
        else:
            out["verdict"] = (
                "FORK GRADING DOES NOT REPRODUCE its own answer key on "
                f"{out['fork']['graded'] - out['fork']['reproduced']}/{out['fork']['graded']} "
                "graded points: tau_reward on a forked run at this revision measures the "
                "harness, not the policy."
            )
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0 if out.get("fork", {}).get("ok", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
