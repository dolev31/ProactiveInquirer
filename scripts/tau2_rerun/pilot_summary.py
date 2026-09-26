"""Read a tau2 rerun PILOT root: is the comparator degenerate under the enforced cap?

WHY THE PILOT EXISTS. Replaying history at code 132e3e8, the enforcement lane found that the
inquirer arms' asks consumed all 16 budget units before any tool call on nearly every run. If the
rerun's comparator does the same, the downstream agent executes almost no tool calls and the
primary contrast is degenerate by construction. So this reports, per suite x pin x variant and
NEVER pooled across any of them: asks, executed env calls, refused calls, stop reasons and
task success -- plus the two degeneracy shares (runs with ZERO executed env calls, runs whose asks
reached the cap).

THE ENFORCEMENT FIELDS ARE READ, NEVER GUESSED. The enforcement lane writes `budget_enforcement`
("refuse_and_tell" on retail and airline), `budget_cap`, `n_refused_budget` and
`refused_budget_calls` (a list, ok branch only) into every tau2 status.json. Refused calls are not
env_calls rows, and `n_env_calls` counts executed calls only. A record that lacks a field is
REFUSED, not read as zero: a schema miss returns None in the shape of a finding.
`len(refused_budget_calls)` is a second route to `n_refused_budget`; disagreements are counted and
printed, not hidden.

It reads only a root whose top LAUNCH.json has a pilot setting and whose every manifest carries
pilot_flag=true -- this is the pilot's own readout, never a path by which a campaign root is read.

    python scripts/tau2_rerun/pilot_summary.py --root /abs/pilot-root [--out summary.json]

Exit 0 with the table; 3 if the root or any record fails a check above.
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping

_spec = importlib.util.spec_from_file_location("tau2_rerun", Path(__file__).with_name("rerun.py"))
rerun = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("tau2_rerun", rerun)  # a @dataclass resolves its module through sys.modules
_spec.loader.exec_module(rerun)

REQUIRED = ("budget_enforcement", "budget_cap", "n_refused_budget", "refused_budget_calls")
EXPECTED_MODE = {"tau2_retail": "refuse_and_tell", "tau2_airline": "refuse_and_tell"}


class Refused(RuntimeError):
    pass


def load(root: Path) -> list[dict[str, Any]]:
    """Every unit under <root>/<suite>/<pin>/<run_id>/, with its suite and pin from the layout
    and both checked against the unit's own manifest."""
    top = root / "LAUNCH.json"
    if not top.is_file():
        raise Refused(f"{root} has no LAUNCH.json; not a launcher root")
    launch = json.loads(top.read_text())
    if not launch.get("pilot"):
        raise Refused(f"{root} is not a pilot root (LAUNCH.json pilot={launch.get('pilot')!r})")
    rows = []
    for suite in launch["suites"]:
        for pin, _arm in rerun.PINS:
            for st in sorted((root / suite / pin).glob("*/status.json")):
                s = json.loads(st.read_text())
                mf = st.parent / "manifest.json"
                if not mf.is_file():
                    raise Refused(f"{st.parent} has a status.json and no manifest.json")
                m = json.loads(mf.read_text())
                inq = ((m.get("pins") or {}).get("inquirer") or {}).get("model_id")
                if m.get("pilot_flag") is not True or inq != pin:
                    raise Refused(f"{mf}: pilot_flag={m.get('pilot_flag')!r}, inquirer={inq!r}")
                rows.append({"suite": suite, "pin": pin, "variant": s["key"][9], "s": s})
    return rows


def _m(xs: list[float]) -> dict[str, float | None]:
    if not xs:
        return {"mean": None, "median": None}
    return {"mean": round(statistics.fmean(xs), 3), "median": statistics.median(xs)}


def summarise(rows: list[Mapping[str, Any]]) -> tuple[dict[str, Any], list[str]]:
    cells: dict[str, list[Mapping[str, Any]]] = collections.defaultdict(list)
    for r in rows:
        cells[f"{r['suite']}|{r['pin']}|{r['variant']}"].append(r)
    problems: list[str] = []
    out: dict[str, Any] = {}
    for key, rs in sorted(cells.items()):
        suite = key.split("|")[0]
        ok = [r["s"] for r in rs if r["s"].get("status") == "ok"]
        for s in ok:
            missing = [f for f in REQUIRED if f not in s]
            if missing:
                problems.append(f"{key} run {s.get('run_id')}: missing {missing}")
                continue
            if s["budget_enforcement"] != EXPECTED_MODE[suite]:
                problems.append(
                    f"{key} run {s.get('run_id')}: budget_enforcement={s['budget_enforcement']!r}"
                )
            if s["budget_cap"] != rerun.BUDGET_CAP:
                problems.append(f"{key} run {s.get('run_id')}: budget_cap={s['budget_cap']!r}")
        if problems:
            continue
        asks = [int(s["n_asks"]) for s in ok]
        env = [int(s["n_env_calls"]) for s in ok]
        refused = [int(s["n_refused_budget"]) for s in ok]
        n = len(ok)
        out[key] = {
            "n_units": len(rs),
            "status": dict(collections.Counter(str(r["s"].get("status")) for r in rs)),
            "n_ok": n,
            "asks": _m(asks),
            "n_env_calls": _m(env),
            "n_refused_budget": _m(refused),
            "share_zero_env_calls": round(sum(e == 0 for e in env) / n, 3) if n else None,
            "share_asks_at_cap": round(sum(a >= rerun.BUDGET_CAP for a in asks) / n, 3)
            if n
            else None,
            "share_any_refused": round(sum(x > 0 for x in refused) / n, 3) if n else None,
            "stop_reason": dict(collections.Counter(str(s.get("stop_reason")) for s in ok)),
            "tau_reward_gt0": f"{sum(float((s.get('native') or {}).get('tau_reward') or 0) > 0 for s in ok)}/{n}",
            "refused_list_len_disagrees": sum(
                len(s["refused_budget_calls"] or []) != int(s["n_refused_budget"]) for s in ok
            ),
        }
    return out, problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    try:
        rows = load(Path(a.root))
    except Refused as e:
        print(f"pilot_summary: REFUSING: {e}", file=sys.stderr)
        return 3
    table, problems = summarise(rows)
    for p in problems:
        print(f"pilot_summary: REFUSING: {p}", file=sys.stderr)
    if problems:
        return 3
    for key, c in table.items():
        print(
            f"{key}\n  units {c['n_units']} {c['status']}  asks {c['asks']}  env_calls "
            f"{c['n_env_calls']}  refused {c['n_refused_budget']}\n  zero-env-call share "
            f"{c['share_zero_env_calls']}  asks-at-cap share {c['share_asks_at_cap']}  any-refused "
            f"share {c['share_any_refused']}  success {c['tau_reward_gt0']}  stop {c['stop_reason']}"
            + (
                f"  REFUSED-LIST DISAGREES x{c['refused_list_len_disagrees']}"
                if c["refused_list_len_disagrees"]
                else ""
            )
        )
    if a.out:
        Path(a.out).write_text(json.dumps(table, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
