"""When would `BudgetGate` have refused a call in a recorded tau2 fork campaign?

    python scripts/replay_budget_gate.py --runs-root <checkout>/runs

THIS REPLAYS THE SHARED-CAP RULE of ba48ab0..e5e8206 (asks and tool calls on one ledger, one
cap), NOT RULES amendment 7's split budgets, under which the gate counts tool calls on its own
counter and asks never refuse a call. Its counts answer "what would the shared cap have done to
the recorded runs" and nothing about a split-budget re-run.

Read-only over run directories. For every recorded tau2_retail / tau2_airline fork run
(`split == test`, `status == ok`), the dialogue's two kinds of in-dialogue debit are put back in
the order they happened, then walked through a real `BudgetLedger` under the gate's rule:

    an Ask       (`run_loop.charge_retrieval`)  charged if it fits, else the rollout would have
                                                 stopped on `budget` -- the FIRST DIVERGENCE is
                                                 then an ask, not a refusal
    a tool call  (the Orchestrator's)            refused if the ledger is spent, charged if it
                                                 executed ok, free if the environment rejected it
    a GENERIC tool call (conf/tau2/tool_types.json)  never gated, never charged

A fork's inherited prefix is its starting state and is not charged (the gate never sees upstream's
`set_state` replay of it). `--charge-prefix` prints the other accounting beside it: the one the
post-hoc meter actually applied, which charged every ok call in the transcript, prefix and GENERIC
calls included. `1stGEN_old` counts runs whose FIRST refusal under the rule before the GENERIC
exemption (ba48ab0) was a GENERIC tool -- the runs that exemption changes at the first bite.
`--summary-arms/--summary-codes` total the groups of one named population (e.g. the 408-run
transfer population) so its sums are printed rather than added by hand.

HOW THE ORDER IS RECOVERED, since no single file records it.

  * `ledger.jsonl` is append-only in charge order. Its `retrieval_calls` rows with `hard` are the
    Asks. Its `llm_calls` rows are one per `calls.jsonl` row, in the same order, so each can be
    labelled with the call's `actor`; every rollout ends with exactly one `answerer` call
    (`run_loop` answers last), which closes it.
  * `outcome.json`'s `env_calls` carry `turn_idx` = user messages so far. The driver runs one
    rollout per user message, starting at the fork's last prefix user turn, so rollout r's plan
    executed at `turn_idx == n_prefix_user_turns + r`, after that rollout's Asks and before the
    next rollout's.

  A run on which either half does not hold -- answerer count != `n_assistant_rollouts`, an own
  call whose turn maps to no rollout, the two charge instruments disagreeing -- is REPORTED as
  unreconstructable with its reason, never guessed.

WHAT THE COUNTS MEAN. Everything up to the first bite is exact: the record is what the policy did
while the gate would have been silent. After it, the record is counterfactual (the policy never
saw a refusal), so "refused calls" is the number of calls IN THE RECORDED DIALOGUE the gate would
not have executed, holding the rest of the record fixed. It is a statement about the recorded
runs, not a prediction of a gated re-run.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pinq.budget import BudgetExceeded, BudgetLedger  # noqa: E402

TOOL_TYPES = json.loads((ROOT / "conf" / "tau2" / "tool_types.json").read_text())
DOMAIN = {"tau2_retail": "retail", "tau2_airline": "airline"}


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def events_of(run_dir: Path, status: dict, arm: str) -> tuple[list[tuple], str]:
    """`(events, why_not)`. Each event is ("ask",) or ("call", idx, tool_name, ok, turn_idx)."""
    led = _jsonl(run_dir / "ledger.jsonl")
    calls = _jsonl(run_dir / "calls.jsonl")
    env = json.loads((run_dir / "outcome.json").read_text()).get("env_calls") or []
    u0 = int(status.get("n_prefix_user_turns") or 0)

    rc = [r for r in led if r.get("currency") == "retrieval_calls"]
    seen_harvest = False
    for r in rc:
        if not r.get("hard"):
            seen_harvest = True
        elif seen_harvest:
            return [], "an in-loop charge follows a harvest charge"
    n_harvest = sum(1 for r in rc if not r.get("hard"))
    n_ok = sum(1 for c in env if c.get("ok"))
    if n_harvest != n_ok:
        return [], f"instruments disagree: {n_harvest} harvest rows vs {n_ok} ok env calls"

    # Asks per rollout, from the ledger's order.
    per_rollout: list[int] = []
    asks = 0
    k = 0
    for r in led:
        cur = r.get("currency")
        if cur == "retrieval_calls" and r.get("hard"):
            asks += 1
        elif cur == "llm_calls":
            if k >= len(calls):
                return [], "more llm_calls ledger rows than calls.jsonl rows"
            if calls[k].get("actor") == "answerer":
                per_rollout.append(asks)
                asks = 0
            k += 1
    if k != len(calls):
        return [], f"{len(calls)} calls.jsonl rows vs {k} llm_calls ledger rows"
    if asks:
        return [], "asks charged after the last answerer call"
    n_roll = int(status.get("n_assistant_rollouts") or 0)
    if len(per_rollout) != n_roll:
        return [], f"{len(per_rollout)} answerer calls vs n_assistant_rollouts {n_roll}"

    own = [c for c in env if int(c["turn_idx"]) >= u0]
    by_rollout: dict[int, list[dict]] = defaultdict(list)
    for c in own:
        r = int(c["turn_idx"]) - u0
        if n_roll and not 0 <= r < n_roll:
            return [], f"own call at turn {c['turn_idx']} maps to no rollout (u0={u0})"
        by_rollout[r].append(c)

    events: list[tuple] = []
    idx = 0
    if not n_roll:
        # No pinq rollout at all (the stock agent): its only debits are its tool calls.
        for c in own:
            idx += 1
            events.append(("call", idx, c["tool_name"], bool(c["ok"]), int(c["turn_idx"])))
        return events, ""
    for r, n_asks in enumerate(per_rollout):
        events.extend([("ask",)] * n_asks)
        for c in by_rollout.get(r, ()):
            idx += 1
            events.append(("call", idx, c["tool_name"], bool(c["ok"]), int(c["turn_idx"])))
    return events, ""


def replay(
    events: list[tuple], cap: int, *, domain: str, precharge: int = 0, exempt_generic: bool = True
) -> dict:
    """Walk `events` under the gate's rule. `exempt_generic=False` is the rule at ba48ab0, which
    gated and charged GENERIC tools too; it is kept to count what the exemption changed."""
    led = BudgetLedger(cap=cap)
    for _ in range(precharge):
        led.record("retrieval_calls", 1.0)  # the post-hoc meter's own row shape: never raises
    out = {
        "first": None,
        "first_name": None,
        "first_kind": None,
        "refused": 0,
        "refused_writes": 0,
        "asks_blocked": 0,
    }
    for ev in events:
        if ev[0] == "call" and exempt_generic and TOOL_TYPES[domain].get(ev[2]) == "GENERIC":
            continue  # not retrieval: neither gated nor charged
        try:
            led.check(1.0)
            fits = True
        except BudgetExceeded:
            fits = False
        if ev[0] == "ask":
            if fits:
                led.charge_retrieval(1.0)
            else:
                out["asks_blocked"] += 1
                if out["first_kind"] is None:
                    out["first_kind"] = "ask"
            continue
        _k, idx, name, ok, _turn = ev
        if not fits:
            out["refused"] += 1
            if TOOL_TYPES[domain].get(name) == "WRITE":
                out["refused_writes"] += 1
            if out["first"] is None:
                out["first"] = idx
                out["first_name"] = name
            if out["first_kind"] is None:
                out["first_kind"] = "call"
        elif ok:
            led.charge_retrieval(1.0)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default=str(ROOT / "runs"))
    ap.add_argument("--split", default="test")
    ap.add_argument(
        "--charge-prefix", action="store_true", help="also replay the meter's accounting"
    )
    ap.add_argument("--summary-arms", default="", help="comma list: arms of a population to total")
    ap.add_argument(
        "--summary-codes", default="", help="comma list: code_version prefixes of that population"
    )
    a = ap.parse_args()

    rows: list[dict] = []
    bad: Counter = Counter()
    bad_examples: dict[str, str] = {}
    for m in sorted(Path(a.runs_root).glob("*/manifest.json")):
        try:
            d = json.loads(m.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        suite = d.get("suite_id")
        if suite not in DOMAIN or str(d.get("split")) != a.split or not d.get("foreign_trace_sha"):
            continue
        run_dir = m.parent
        try:
            s = json.loads((run_dir / "status.json").read_text())
        except (OSError, json.JSONDecodeError):
            bad["no readable status.json"] += 1
            continue
        if s.get("status") != "ok":
            continue
        key = (suite, d.get("arm_id"), str(d.get("code_version") or "")[:12])
        try:
            events, why = events_of(run_dir, s, str(d.get("arm_id")))
        except (OSError, json.JSONDecodeError, KeyError) as exc:
            events, why = [], f"{type(exc).__name__}"
        row = {
            "key": key,
            "run_id": d.get("run_id"),
            "why": why,
            "recorded_overrun": "post_hoc_budget_overrun" in (s.get("spent") or {}),
        }
        if not why:
            cap = int(d.get("budget_cap") or 16)
            row["gate"] = replay(events, cap, domain=DOMAIN[suite])
            row["old"] = replay(events, cap, domain=DOMAIN[suite], exempt_generic=False)
            if a.charge_prefix:
                env = json.loads((run_dir / "outcome.json").read_text()).get("env_calls") or []
                u0 = int(s.get("n_prefix_user_turns") or 0)
                pre = sum(1 for c in env if c.get("ok") and int(c["turn_idx"]) < u0)
                row["meter"] = replay(
                    events, cap, domain=DOMAIN[suite], precharge=pre, exempt_generic=False
                )
        else:
            bad[why.split(":")[0]] += 1
            bad_examples.setdefault(why.split(":")[0], f"{d.get('run_id')} {why}")
        rows.append(row)

    print(f"runs matched (ok, split={a.split}, forks, tau2_retail|tau2_airline): {len(rows)}")
    print("unreconstructable, by reason:", dict(bad) or "none")
    for k, ex in bad_examples.items():
        print("   e.g.", ex)
    print()
    hdr = (
        f"{'suite':13s} {'arm':22s} {'code':12s} {'n':>4s} {'rec_ovr':>7s} {'recon':>5s}"
        f" {'bitten':>6s} {'1st=ask':>7s} {'>=1ref':>6s} {'refused':>7s} {'ref_W':>5s}"
        f" {'runs_W':>6s} {'1stGEN_old':>10s}"
        f"  first refused call index (own calls, 1-based): min/med/max"
    )
    if a.charge_prefix:
        hdr += "  | meter: >=1ref refused"
    print(hdr)

    def line_for(label: str, g: list[dict]) -> str:
        ok = [r for r in g if not r["why"]]
        gate = [r["gate"] for r in ok]
        old = [r["old"] for r in ok]
        generic = {
            n
            for dom in TOOL_TYPES.values()
            if isinstance(dom, dict)
            for n, t in dom.items()
            if t == "GENERIC"
        }
        firsts = sorted(x["first"] for x in gate if x["first"] is not None)
        dist = (
            f"{firsts[0]}/{statistics.median(firsts):g}/{firsts[-1]}"
            f"  {dict(sorted(Counter(firsts).items()))}"
            if firsts
            else "-"
        )
        line = (
            f"{label} {len(g):4d}"
            f" {sum(r['recorded_overrun'] for r in g):7d} {len(ok):5d}"
            f" {sum(x['first_kind'] is not None for x in gate):6d}"
            f" {sum(x['first_kind'] == 'ask' for x in gate):7d}"
            f" {sum(x['refused'] > 0 for x in gate):6d} {sum(x['refused'] for x in gate):7d}"
            f" {sum(x['refused_writes'] for x in gate):5d}"
            f" {sum(x['refused_writes'] > 0 for x in gate):6d}"
            f" {sum(x['first_name'] in generic for x in old):10d}  {dist}"
        )
        if a.charge_prefix:
            meter = [r["meter"] for r in ok]
            line += (
                f"  | {sum(x['refused'] > 0 for x in meter):6d}"
                f" {sum(x['refused'] for x in meter):7d}"
            )
        return line

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r["key"]].append(r)
    for key in sorted(groups, key=lambda k: tuple(str(x) for x in k)):
        print(line_for(f"{key[0]:13s} {str(key[1]):22s} {key[2]:12s}", groups[key]))
    if a.summary_arms and a.summary_codes:
        arms = set(a.summary_arms.split(","))
        codes = tuple(a.summary_codes.split(","))
        keys = [k for k in groups if k[1] in arms and k[2].startswith(codes)]
        print()
        print(f"population: arms {sorted(arms)} x codes {sorted(codes)} -> {len(keys)} groups")
        for suite in sorted({k[0] for k in keys}) + ["(both)"]:
            g = [r for k in keys if suite in ("(both)", k[0]) for r in groups[k]]
            print(line_for(f"{suite:13s} {'(total)':22s} {'':12s}", g))
    print()
    print(
        "columns: n = ok fork runs; rec_ovr = runs whose ledger recorded post_hoc_budget_overrun;"
    )
    print(
        "  recon = reconstructable; bitten = the cap binds somewhere (ask blocked or call refused);"
    )
    print("  1st=ask = first bite is an Ask (rollout would stop on budget before any refusal);")
    print(
        "  >=1ref = runs with >=1 refused tool call; refused = refused calls; ref_W = of them WRITE"
    )
    print("  (conf/tau2/tool_types.json); runs_W = runs with >=1 refused WRITE;")
    print(
        "  1stGEN_old = runs whose first refusal WITHOUT the GENERIC exemption was a GENERIC tool."
    )
    print("  All columns but 1stGEN_old and meter apply the exemption.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
