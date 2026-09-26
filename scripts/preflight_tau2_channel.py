"""Did the policy's questions actually get ANSWERED? A precondition for any tau2 sweep.

WHY THIS EXISTS, MEASURED. 87 fork units ran to `status: ok` -- reconciled, graded, writing all
seven files -- while `src/pinq/prompts/retriever_select.txt` was absent from the tree and
`pinq_adapters/tau2/tool_retriever.py` raised `MissingPrompt` on every question it was asked to
turn into a read-only tool call. Not one of those units produced a single piece of evidence from
an Inquirer ask, and NOTHING NOTICED. A run that asks 16 questions and receives nothing is
indistinguishable, in `status`, `n_asks`, `n_turns`, `stop_reason`, `wall_ms`, `usd` and the
reconciliation gates, from a run that asks 16 and learns from each. The failure even looked like a
result: the policy never stopped, which read as a transfer finding and was reported as one.

`pi verify tau2` PASSED 112/112 throughout, at both revisions, because gold replay executes
recorded gold actions against the task's own initial state and never invokes the policy or its
retrieval channel -- a dead channel and a live one produce identical replays. A check that
structurally cannot fail on the question produces a FALSE CLEARANCE, which is worse than no check,
because "we verified the environment" is what stops anyone looking further.

THREE ASSERTIONS, on a real unit, none of which anything else makes:

  1. an ASK was ANSWERED                -- some ask carries a non-empty `retrieved_uids`
  2. the evidence set is NON-EMPTY      -- the read came back and was attached to the trajectory
  3. a REQUIRED need was RESOLVED       -- the evidence intersects a required gold node's uids

Each is necessary and none is sufficient alone: (1) without (2) is a read whose result is dropped,
(2) without (3) is evidence that answers nothing the task needed. And (1) is the one that was
missing: on retail the DRAFTER's agentic tool calls attach evidence and resolve needs even while
the Inquirer's own channel is dead, so (2) and (3) both PASS on a broken unit. Only (1) sees it.

ASSERTION 1 IS `retrieved_uids`, NOT THE `mutating` FLAG. A first draft of this file asserted "a
read-only tool call executed", read from `outcome.json`'s `env_calls`. That is unmeasurable on this
path: `env_calls_from` derives the log from the ORCHESTRATOR's transcript and stamps
`mutating=True` on every call by documented design ("conservatively; the orchestrator does not
expose the flag"), so the field is a constant there and the check could only ever return false.
That was the third vacuous instrument found in one night, and it is why each assertion below names
the field it reads and why the non-vacuity evidence is recorded here:

  MEASURED over fork runs, from run directories --
    published revisions (922060c/828a720/401b8fe): retail 1833 of 2882 asks answered,
                                                   airline 1299 of 1923 answered
    HEAD 1d6d0f9 with retriever_select.txt absent: retail 0 of 657, airline 0 of 571

  So the field DOES take both values, 100% zero is a real measurement, and every ask in all 87
  units of the killed campaign went unanswered.

THIS READS GOLD (assertion 3) and is therefore an OPERATOR precondition, not something a rollout
worker may run: `PI_GOLD_ROOT` must be set here and must be UNSET in a worker. It is a standalone
script for exactly that reason -- nothing imports it.

EXIT 0 only if all three hold on at least one unit. Non-zero means the channel is broken and a
campaign must not start.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def check_run(run_dir: Path, graph_version: str = "v1") -> dict:
    """The three assertions for one finished run directory."""
    from pi_eval.gold import load_graphs

    man = json.loads((run_dir / "manifest.json").read_text())
    st = json.loads((run_dir / "status.json").read_text())
    out: dict = {
        "run_id": man.get("run_id"),
        "suite_id": man.get("suite_id"),
        "arm_id": man.get("arm_id"),
        "task_id": man.get("task_id"),
        "foreign_prefix_k": man.get("foreign_prefix_k"),
        "status": st.get("status"),
        "n_asks": st.get("n_asks"),
        "stop_reason": st.get("stop_reason"),
    }

    # ---- 1. an ask was ANSWERED: some ask carries a non-empty retrieved_uids
    turns_f = run_dir / "turns.jsonl"
    asks, answered = 0, 0
    if turns_f.is_file():
        for line in turns_f.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if str(r.get("action_kind")) != "ask":
                continue
            asks += 1
            if r.get("retrieved_uids"):
                answered += 1
    oc = run_dir / "outcome.json"
    calls = json.loads(oc.read_text()).get("env_calls") or [] if oc.is_file() else []
    out["n_asks_recorded"] = asks
    out["n_asks_answered"] = answered
    out["n_env_calls"] = len(calls)
    out["a1_an_ask_was_answered"] = answered >= 1

    # ---- 2. the evidence set is non-empty
    ev = run_dir / "evidence.jsonl"
    uids: set[str] = set()
    if ev.is_file():
        for line in ev.read_text().splitlines():
            if line.strip():
                u = json.loads(line).get("uid")
                if u:
                    uids.add(str(u))
    out["n_evidence_uids"] = len(uids)
    out["a2_evidence_non_empty"] = len(uids) >= 1

    # ---- 3. a required need was resolved by that evidence
    graphs = load_graphs(str(man.get("suite_id")), graph_version)
    g = graphs.get(str(man.get("task_id")))
    if g is None:
        out["a3_required_need_resolved"] = False
        out["a3_note"] = f"no gold graph for task {man.get('task_id')} at {graph_version}"
        out["n_required"] = 0
        out["n_required_resolved"] = 0
    else:
        req = [n for n in g.gold_nodes if n.gold_partition == "required"]
        hit = [n for n in req if uids & set(n.gold_ev_uids or ())]
        # A required node with NO gold_ev_uids can never be resolved by evidence, so it is
        # excluded from the denominator rather than counted as a miss -- absent is not failing.
        span = [n for n in req if n.gold_ev_uids]
        out["n_required"] = len(req)
        out["n_required_with_uids"] = len(span)
        out["n_required_resolved"] = len(hit)
        out["a3_required_need_resolved"] = len(hit) >= 1
        if not span:
            out["a3_note"] = (
                "NO required gold node on this task carries gold_ev_uids, so assertion 3 cannot "
                "be satisfied by any policy here; pick a task where it can, or this run cannot "
                "validate the channel"
            )
    # THREE OUTCOMES PER UNIT, NOT TWO. A task on which NO required gold node carries
    # `gold_ev_uids` cannot satisfy assertion 3 under any policy, so a False there is a property
    # of the task and not of the channel. Reporting it as a failure would block a sweep for the
    # wrong reason -- and reporting it as a pass would clear the channel without testing it.
    out["a3_decidable"] = bool(out.get("n_required_with_uids", 0))
    if not out["a3_decidable"]:
        out["ok"] = False
        out["unsuitable"] = True
    else:
        out["unsuitable"] = False
        out["ok"] = bool(
            out["a1_an_ask_was_answered"]
            and out["a2_evidence_non_empty"]
            and out["a3_required_need_resolved"]
        )
    return out


def suitable_tasks(suite_id: str, graph_version: str = "v1") -> list[str]:
    """Task ids on which assertion 3 CAN be satisfied: at least one required gold node carrying
    `gold_ev_uids`. Printed when every supplied unit is unsuitable, so the operator picks a fork
    point that can actually validate the channel instead of retrying blind."""
    from pi_eval.gold import load_graphs

    out = []
    for tid, g in sorted(load_graphs(suite_id, graph_version).items()):
        if any(n.gold_partition == "required" and n.gold_ev_uids for n in g.gold_nodes):
            out.append(str(tid))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dirs", nargs="+", help="finished run directories to check")
    ap.add_argument("--graph-version", default="v1")
    a = ap.parse_args(argv)

    rows = []
    for d in a.run_dirs:
        p = Path(d)
        try:
            rows.append(check_run(p, a.graph_version))
        except Exception as exc:  # noqa: BLE001 - a run we cannot check is not a pass
            rows.append({"run_dir": str(p), "error": f"{type(exc).__name__}: {exc}", "ok": False})
    passed = [r for r in rows if r.get("ok")]
    unsuitable = [r for r in rows if r.get("unsuitable")]
    if not passed and unsuitable and len(unsuitable) == len(rows):
        suite = str(rows[0].get("suite_id") or "tau2_retail")
        ok_tasks = suitable_tasks(suite, a.graph_version)
        print(
            json.dumps(
                {
                    "units": rows,
                    "n_passed": 0,
                    "verdict": (
                        "INCONCLUSIVE -- every unit supplied is on a task where NO required gold "
                        "node carries gold_ev_uids, so assertion 3 is undecidable there. This says "
                        "nothing about the channel. Re-run the preflight on a task that can "
                        "satisfy it."
                    ),
                    "suitable_task_ids": ok_tasks[:40],
                    "n_suitable": len(ok_tasks),
                },
                indent=2,
            )
        )
        return 2
    verdict = (
        f"CHANNEL LIVE: {len(passed)}/{len(rows)} unit(s) had an ask ANSWERED, attached evidence and "
        "resolved a required need. A tau2 sweep may start."
        if passed
        else "CHANNEL DEAD OR UNVALIDATED: no unit satisfied all three assertions. A tau2 sweep "
        "must NOT start; see the per-unit flags."
    )
    print(json.dumps({"units": rows, "n_passed": len(passed), "verdict": verdict}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
