"""What the pre-change post-simulate harvest would do to a recorded fork campaign.

    python scripts/replay_post_hoc_harvest.py --suite tau2_retail --split test

`_attach_env_evidence` charges a finished dialogue's transcript tool calls through
`meter_env_calls`, against the same hard cap `run_loop` already charged the in-loop Asks to.
Before this branch that charge went through `charge_retrieval`, which RAISES when the sum
exceeds the cap -- and the raise is caught by `run_tau2_unit`'s own `except Exception`, which
writes `status: error` over a unit that ran to completion and produced a real reward. Nothing is
prevented: the Orchestrator executed those calls before the meter ever ran.

This prints, per arm, how many recorded runs that would delete, and what the deletion does to
the endpoint -- because the count alone understates it. The arm that calls more tools loses more
runs, and its expensive runs are its long dialogues, so the deletion filters the endpoint by its
own value. It exists so the figure in `tests/test_post_hoc_harvest_meter.py` and in the paper's
reproducibility caveat can be re-derived rather than trusted, and so it can be re-run against a
different suite, split or contrast.

TWO INDEPENDENT INSTRUMENTS, CROSS-CHECKED ON EVERY ROW, because a single one cannot tell a
finding from an instrument error.

  A) `ledger.jsonl`. Its `retrieval_calls` rows split by `hard`: `charge_retrieval` writes hard
     rows carrying the cap, and the harvest's `ledger.record` writes non-hard rows with no cap.
     `record("retrieval_calls", ...)` has exactly one call site in `src/` -- `meter_env_calls`
     itself -- so a non-hard row IS a harvest charge. The ordering that makes the pre-harvest
     total readable (no in-loop charge after a harvest charge) is checked, not assumed.

  B) `outcome.json`. Its `env_calls` are the recorded transcript; the count of `ok=True` calls is
     exactly what `meter_env_calls` charges.

A run recorded WITHOUT the fix has no non-hard rows to read, so instrument A reports zero for it
and the cross-check against B fails loudly rather than reporting a clean zero.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def rows_for(
    runs_root: Path, *, suite: str, split: str, arms: tuple[str, str]
) -> tuple[list[dict], list[str]]:
    """Returns `(rows, incomplete)`. `runs/` is SHARED and live -- other sessions write into it
    while this runs -- so a matching run directory may be missing a file this needs. Those are
    returned and reported, never dropped in silence: a shrinking denominator that nothing prints
    is how a live campaign turns into a wrong n."""
    out: list[dict] = []
    incomplete: list[str] = []
    for m in sorted(runs_root.glob("*/manifest.json")):
        d = json.loads(m.read_text())
        if d.get("suite_id") != suite or str(d.get("split")) != split:
            continue
        if d.get("arm_id") not in arms or not d.get("foreign_trace_sha"):
            continue
        need = [m.parent / n for n in ("status.json", "ledger.jsonl", "outcome.json")]
        if not all(f.is_file() for f in need):
            incomplete.append(
                f"{m.parent.name} (missing {[f.name for f in need if not f.is_file()]})"
            )
            continue
        s = json.loads(st.read_text()) if (st := m.parent / "status.json") else {}
        led = [
            json.loads(x) for x in (m.parent / "ledger.jsonl").read_text().splitlines() if x.strip()
        ]
        rc = [r for r in led if r.get("currency") == "retrieval_calls"]
        in_loop = [r for r in rc if r.get("hard")]
        harvest = [r for r in rc if not r.get("hard")]
        seen_harvest = False
        order_ok = True
        for r in rc:
            if not r.get("hard"):
                seen_harvest = True
            elif seen_harvest:
                order_ok = False
        oc = json.loads((m.parent / "outcome.json").read_text())
        out.append(
            {
                "run_id": d["run_id"],
                "arm": d["arm_id"],
                "seed": d.get("seed"),
                "trace": d.get("foreign_trace_sha"),
                "k": d.get("foreign_prefix_k"),
                "cap": d.get("budget_cap"),
                "code_version": d.get("code_version", ""),
                "status": s.get("status"),
                "pre": in_loop[-1]["cumulative"] if in_loop else 0.0,
                "n_harvest": len(harvest),
                "n_ok_env": sum(1 for c in (oc.get("env_calls") or []) if c.get("ok")),
                "order_ok": order_ok,
                "follow_ups": int(s["n_user_turns"]) - int(s.get("n_prefix_user_turns") or 0),
            }
        )
    return out, incomplete


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default=str(ROOT / "runs"))
    ap.add_argument("--suite", default="tau2_retail")
    ap.add_argument("--split", default="test")
    ap.add_argument("--treatment", default="inquirer_prompted")
    ap.add_argument("--control", default="self_ask")
    ap.add_argument("--emit-fixture", action="store_true", help="print the pinned test table")
    a = ap.parse_args()
    arms = (a.treatment, a.control)
    rows, incomplete = rows_for(Path(a.runs_root), suite=a.suite, split=a.split, arms=arms)
    if not rows:
        print(f"no runs matched {a.suite}/{a.split} in {a.runs_root}", file=sys.stderr)
        return 2
    print(f"incomplete run dirs skipped (reported, not dropped): {len(incomplete)}")
    for x in incomplete:
        print("   ", x)

    ids = sorted(r["run_id"] for r in rows)
    print(f"population: {len(rows)} runs")
    print(f"run_ids_sha: {hashlib.sha256(chr(10).join(ids).encode()).hexdigest()[:16]}")
    print("code_versions:", dict(Counter(r["code_version"][:7] for r in rows)))
    print("statuses:", dict(Counter(r["status"] for r in rows)))
    print("caps:", dict(Counter(r["cap"] for r in rows)))
    print("ledger ordering ok on every row:", all(r["order_ok"] for r in rows))
    dis = [r for r in rows if r["n_harvest"] != r["n_ok_env"]]
    print(f"instrument disagreement: {len(dis)} of {len(rows)}")
    for r in dis[:10]:
        print("   ", r["run_id"], r["arm"], "ledger", r["n_harvest"], "env", r["n_ok_env"])
    if dis:
        print("REFUSING to report: the two instruments disagree", file=sys.stderr)
        return 3

    refused = [r for r in rows if r["pre"] + r["n_harvest"] > r["cap"]]
    print()
    print(f"the pre-change harvest refuses: {len(refused)} of {len(rows)}")
    print("  by arm:", dict(Counter(r["arm"] for r in refused)))
    print("  denominators:", dict(Counter(r["arm"] for r in rows)))
    for arm in arms:
        s = [r for r in rows if r["arm"] == arm]
        print(
            f"  {arm:20s} mean charged = {statistics.mean(r['pre'] + r['n_harvest'] for r in s):6.3f}"
            f"  (in-loop {statistics.mean(r['pre'] for r in s):6.3f}"
            f" + harvest {statistics.mean(r['n_harvest'] for r in s):6.3f})"
            f"  cap {s[0]['cap']}"
        )

    pairs: dict[tuple, dict] = {}
    for r in rows:
        pairs.setdefault((r["trace"], r["k"], r["seed"]), {})[r["arm"]] = r
    full = {k: v for k, v in pairs.items() if len(v) == 2}
    lost = {r["run_id"] for r in refused}
    surv = {k: v for k, v in full.items() if all(x["run_id"] not in lost for x in v.values())}
    print()
    print(f"pairs {len(full)} -> {len(surv)}")
    print(f"dialogue clusters {len({k[0] for k in full})} -> {len({k[0] for k in surv})}")
    print(f"fork points {len({k[:2] for k in full})} -> {len({k[:2] for k in surv})}")

    def endpoint(label: str, ps: dict) -> None:
        if not ps:
            print(f"{label:32s} (empty)")
            return
        d = [v[a.treatment]["follow_ups"] - v[a.control]["follow_ups"] for v in ps.values()]
        print(
            f"{label:32s} pairs={len(d):3d}  treatment="
            f"{statistics.mean(v[a.treatment]['follow_ups'] for v in ps.values()):6.3f}"
            f"  control={statistics.mean(v[a.control]['follow_ups'] for v in ps.values()):6.3f}"
            f"  diff_mean={statistics.mean(d):+.4f}"
            f"  fewer/more={sum(x < 0 for x in d)}/{sum(x > 0 for x in d)}"
        )

    print()
    print("follow-up turns, treatment minus control:")
    endpoint("  as published", full)
    endpoint("  surviving a pre-change re-run", surv)
    endpoint("  in the pairs it would delete", {k: v for k, v in full.items() if k not in surv})

    if a.emit_fixture:
        traces: dict[str, str] = {}
        print()
        for r in sorted(rows, key=lambda x: x["run_id"]):
            traces.setdefault(r["trace"], f"T{len(traces)}")
            print(
                '    ("%s", "%s", %d, "%s", %d, %d, %d),'
                % (
                    r["run_id"],
                    r["arm"],
                    r["seed"],
                    traces[r["trace"]],
                    r["k"],
                    int(r["pre"]),
                    r["n_harvest"],
                )
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
