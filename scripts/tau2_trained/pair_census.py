"""Can a tau2 consequence-labelled preference set be built from the fork rollouts on disk?

    python scripts/tau2_trained/pair_census.py --suite tau2_retail --out census.json
    python scripts/tau2_trained/pair_census.py --suite tau2_retail --split train --strict

WHY THIS EXISTS AS A CENSUS AND NOT AS AN EXPORTER. Lane L4.2 set out to fine-tune the
Inquirer on tau2 retail *consequences* -- pairs of continuations from one fork prefix, ordered
by `db_reward`, then by customer follow-up turns. Four properties of the recorded population
decide whether such a file can be written at all, and none of them is visible from a row
count. This script measures all four and writes them beside the pair yield, so the decision to
build or not to build is made on numbers rather than on the plan.

THE FOUR, each printed as its own block:

1. WHERE THE ROLLOUTS ARE. `split` on a tau2 fork row, and whether its fork key is even
   present. `prefix_k` is NULL on every tau2 fork row in `runs.parquet` (the tau2 fork key is
   `foreign_prefix_k`, which lives only on the run's own manifest -- see
   `pinq_train.gate._fork_key`), so a census keyed on the parquet column silently groups every
   cut of one dialogue together and reads far too few states.

2. WHETHER TWO ARMS CAN SHARE A STATE. `may_ask_user` is not an action the policy picks at a
   shared state: it is a constructor flag on the arm (`pinq_expt.arms` builds
   `PromptedInquirer(..., may_ask_user=True)` for exactly one arm), it forces `target="kb"` in
   the parser for every other arm (`policies/base.py:_action_from`), and it swaps a whole
   PROMPT FRAGMENT -- the real `fragment_user_channel` against a token-matched placebo. So the
   two arms are shown different prompts and cannot be two continuations of one state. This
   block reads the fragment each arm's manifest pins, which is the check rather than the claim.

3. WHETHER THE STATE CAN BE RE-RENDERED AT ALL. The Inquirer's prompt has six holes. Five are
   pinned by the run itself (the template and the fragment by `manifest.prompt_hashes`, the
   question verbatim on `rollouts.jsonl`, and -- at a fork's FIRST decision only -- an empty
   evidence set, an empty history and no preceding draft). This block reports which template
   sha each run pinned and whether that sha exists in the working tree, because a population
   whose prompt bytes are not in the tree cannot be re-rendered from it.

4. WHETHER THE SHIPPED STATE GUARD CAN FIRE HERE. `cmd_train.render_state(verify=True)`
   re-derives the evidence set and compares it to the turn's recorded `subset_hash_before`.
   At a fork's first decision nothing has been retrieved, so both sides are the empty-set hash
   and the guard compares a constant to itself. A guard that cannot fail is not a check, so
   this block reports the share of first turns whose `subset_hash_before` IS the empty hash
   rather than leaving the guard's pass unexamined.

THE PAIR YIELD ITSELF is reported under two rules, never one. LOOSE groups on the fork point
alone; STRICT adds everything a same-state pair must also share -- the recorded question bytes,
the whole `prompt_hashes` map, `model_pin_hash` and `budget_cap`. The two numbers differ by
more than a factor of four, and reporting only the loose one would count pairs that
`rung2_dpo.train.assert_same_state` and the exporter's own `n_cross_state_dropped` guard both
refuse.

Reads run directories and `runs.parquet` only. Touches no gold, writes nothing but its report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

# The arms that record an Inquirer decision. `drafter_only` forks exist and hold no turn, and
# `reference_agent` is the replayed upstream trace rather than a policy, so neither can supply
# a continuation to compare.
POLICY_ARMS: tuple[str, ...] = ("inquirer_prompted", "inquirer_may_ask_user", "self_ask")

# The exporter's own length guard, so this census and `pi train export` agree on what a pair
# is. `pinq_train.export.dataset.question_len_delta` measures the QUESTION text, not the
# action JSON, and `ExportManifest.len_delta_max` defaults to the same 40.
LEN_DELTA_MAX = 40


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _first_line_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open() as fh:
            line = fh.readline()
    except OSError:
        return None
    if not line.strip():
        return None
    try:
        return json.loads(line)
    except ValueError:
        return None


def load_forks(
    parquet: Path, runs_root: Path, suite: str, arms: Sequence[str]
) -> list[dict[str, Any]]:
    """One record per fork continuation that recorded at least one Inquirer decision.

    THE FORK KEY COMES OFF THE MANIFEST. `runs.parquet` carries a `prefix_k` column and it is
    NULL on every tau2 fork row; `foreign_prefix_k` is the tau2 cut depth and it is written
    only to `manifest.json`/`status.json`. Joining on the parquet column instead would key
    every cut of one dialogue to the same group.
    """
    import duckdb

    quoted = ",".join(f"'{a}'" for a in arms)
    rows = (
        duckdb.connect()
        .execute(
            f"""SELECT run_id, arm_id, seed, task_id, split, foreign_trace_sha, model_pin_hash,
                   budget_cap, code_version, dirty
            FROM '{parquet}'
            WHERE suite_id = '{suite}' AND status = 'ok'
              AND foreign_trace_sha IS NOT NULL AND foreign_trace_sha <> ''
              AND arm_id IN ({quoted})"""
        )
        .fetchall()
    )

    out: list[dict[str, Any]] = []
    for rid, arm, seed, task, split, trace, pin, cap, code_version, dirty in rows:
        d = runs_root / str(rid)
        status = _read_json(d / "status.json")
        manifest = _read_json(d / "manifest.json")
        turn0 = _first_line_json(d / "turns.jsonl")
        rollout0 = _first_line_json(d / "rollouts.jsonl")
        if status is None or manifest is None or turn0 is None or rollout0 is None:
            continue
        native = status.get("native") or {}
        n_user = status.get("n_user_turns")
        n_prefix = status.get("n_prefix_user_turns")
        out.append(
            {
                "run_id": rid,
                "arm_id": arm,
                "seed": seed,
                "task_id": str(task),
                "split": split,
                "trace": trace,
                "k": status.get("foreign_prefix_k"),
                "model_pin_hash": pin,
                "budget_cap": cap,
                "code_version": code_version,
                "dirty": bool(dirty),
                "prompt_hashes": manifest.get("prompt_hashes") or {},
                "view_question": rollout0.get("view_question"),
                "action_kind": turn0.get("action_kind"),
                "target": turn0.get("target"),
                "question": str(turn0.get("question") or ""),
                "subset_hash_before": str(turn0.get("subset_hash_before") or ""),
                "db_reward": native.get("db_reward"),
                "tau_reward": native.get("tau_reward"),
                "followups": (
                    None if n_user is None or n_prefix is None else int(n_user) - int(n_prefix)
                ),
                "n_turns": status.get("n_turns"),
                "n_asks": status.get("n_asks"),
                # THE DOLLAR FIELD THAT IS NOT SHORT. `usd_billed` reads 0.0 on a third of
                # these units; `usage.usd` agrees with `spent.usd` and with the parquet's own
                # `usd` column on every one of them. A campaign costed from `usd_billed`
                # understates itself.
                "usd_billed": float(status.get("usd_billed") or 0.0),
                "usage_usd": float((status.get("usage") or {}).get("usd") or 0.0),
                "user_sim_usd": float(status.get("user_sim_usd") or 0.0),
            }
        )
    return out


def state_key(rec: dict[str, Any], *, strict: bool) -> tuple:
    """The grouping key. LOOSE is the fork point; STRICT is everything a pair must share.

    `pins_sha` is not recomputed here: `model_pin_hash` and the whole `prompt_hashes` map are
    the two halves it is built from, and carrying them separately says which half differs when
    a group splits.
    """
    key: tuple = (rec["trace"], rec["k"])
    if strict:
        key = key + (
            rec["view_question"],
            json.dumps(rec["prompt_hashes"], sort_keys=True),
            rec["model_pin_hash"],
            rec["budget_cap"],
        )
    return key


def _norm(text: str) -> str:
    return " ".join(str(text).split()).casefold()


def decide(a: dict[str, Any], b: dict[str, Any]) -> tuple[str, str]:
    """(winner_run_id, decided_by) for two continuations of one state, or ("", "").

    THE LADDER, in the order lane L4.2 declared: the environment's own reward first, then
    fewer customer follow-up turns at equal reward, then the shorter continuation. `db_reward`
    rather than `tau_reward` because `tau_reward` is a GATED COPY of it -- the actuator writes
    it only when the task's `reward_basis` names DB (`pinq_adapters.tau2.actuator`), so
    ordering on `tau_reward` would silently drop every task graded on another basis.
    """
    if a["db_reward"] is None or b["db_reward"] is None:
        return "", ""
    if a["db_reward"] != b["db_reward"]:
        hi = a if a["db_reward"] > b["db_reward"] else b
        return hi["run_id"], "db_reward"
    if (
        a["followups"] is not None
        and b["followups"] is not None
        and a["followups"] != b["followups"]
    ):
        hi = a if a["followups"] < b["followups"] else b
        return hi["run_id"], "followups"
    if a["n_turns"] is not None and b["n_turns"] is not None and a["n_turns"] != b["n_turns"]:
        hi = a if a["n_turns"] < b["n_turns"] else b
        return hi["run_id"], "quickest"
    return "", ""


def pair_yield(recs: Iterable[dict[str, Any]], *, strict: bool) -> dict[str, Any]:
    """Candidate pairs at each state, and what each guard removes, counted separately.

    One pooled "dropped" figure would report the mix as much as the effect: identical
    questions and the length guard remove different things for different reasons, and the
    identical-question count is the one that says whether more seeds would help.
    """
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for r in recs:
        groups[state_key(r, strict=strict)].append(r)

    counts: Counter[str] = Counter()
    decided_by: Counter[str] = Counter()
    tasks: set[str] = set()
    for members in groups.values():
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = members[i], members[j]
                counts["candidate_pairs"] += 1
                if _norm(a["question"]) == _norm(b["question"]):
                    counts["identical_question"] += 1
                    continue
                if abs(len(a["question"]) - len(b["question"])) > LEN_DELTA_MAX:
                    counts["len_guard_dropped"] += 1
                    continue
                winner, by = decide(a, b)
                if not winner:
                    counts["undecided"] += 1
                    continue
                counts["kept"] += 1
                decided_by[by] += 1
                if a["arm_id"] != b["arm_id"]:
                    counts["kept_cross_arm"] += 1
                if a["target"] != b["target"]:
                    counts["kept_cross_target"] += 1
                tasks.add(f"{a['task_id']}")
    return {
        "states": len(groups),
        "states_with_two_or_more": sum(1 for v in groups.values() if len(v) >= 2),
        **dict(counts),
        "decided_by": dict(decided_by),
        "task_ids_in_kept_pairs": len(tasks),
    }


def channel_block(recs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Which user-channel fragment each arm's own manifest pins, and how many asks went where.

    The claim this replaces is "the routing decision can be supervised by pairing a
    user-directed ask against a tool-directed one". It cannot, and this is the measurement
    that says so rather than the argument.
    """
    frag: dict[str, Counter] = defaultdict(Counter)
    tgt: Counter = Counter()
    for r in recs:
        names = sorted(k for k in r["prompt_hashes"] if k.startswith("fragment_user_channel"))
        frag[r["arm_id"]][",".join(names) or "(none)"] += 1
        tgt[(r["arm_id"], str(r["target"]))] += 1
    return {
        "fragment_pinned_by_arm": {a: dict(c) for a, c in sorted(frag.items())},
        "first_action_target_by_arm": {f"{a}/{t}": n for (a, t), n in sorted(tgt.items())},
    }


def template_block(recs: Sequence[dict[str, Any]], root: Path) -> dict[str, Any]:
    """Which Inquirer template sha each run pinned, and whether it is in the working tree.

    A population whose prompt bytes are not in the tree cannot be re-rendered from the tree,
    and the failure is silent: rendering the tree's version produces a state_text that is not
    the prompt the policy saw, and no shipped guard looks at the template.
    """
    prompts = root / "src" / "pinq" / "prompts"
    in_tree = {_sha(p.read_text()): p.name for p in prompts.glob("*.txt") if p.is_file()}
    seen: Counter = Counter()
    for r in recs:
        for name, sha in sorted(r["prompt_hashes"].items()):
            if name.startswith(("inquirer", "self_ask", "self_inquire")):
                seen[(name, str(sha))] += 1
    rows = []
    for (name, sha), n in sorted(seen.items(), key=lambda kv: -kv[1]):
        rows.append(
            {
                "template": name,
                "sha": sha,
                "runs": n,
                "in_working_tree": sha in in_tree,
                "tree_file": in_tree.get(sha, ""),
            }
        )
    return {
        "templates_pinned": rows,
        "runs_on_a_template_absent_from_the_tree": sum(
            r["runs"] for r in rows if not r["in_working_tree"]
        ),
    }


def guard_block(recs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Could `render_state`'s evidence guard have failed on these states?

    An invariance check that cannot detect a change is indistinguishable from a clean pass.
    """
    from pinq.ids import subset_hash

    empty = subset_hash(())
    c: Counter = Counter()
    for r in recs:
        sb = r["subset_hash_before"]
        c["empty" if sb == empty else ("missing" if not sb else "non_empty")] += 1
    return {
        "empty_evidence_subset_hash": empty,
        "first_turn_subset_hash_before": dict(c),
        "guard_is_vacuous_on": c["empty"],
        "guard_could_have_fired_on": c["non_empty"],
    }


def cost_block(recs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """The dollar field that is not short, measured on this population."""
    n = len(recs)
    zero = sum(1 for r in recs if r["usd_billed"] == 0.0)
    billed = sum(r["usd_billed"] for r in recs)
    usage = sum(r["usage_usd"] for r in recs)
    sim = sum(r["user_sim_usd"] for r in recs)
    return {
        "units": n,
        "usd_billed_is_zero_on": zero,
        "usd_billed_is_zero_share": round(zero / n, 4) if n else None,
        "total_usd_billed": round(billed, 2),
        "total_usage_usd": round(usage, 2),
        "total_user_sim_usd": round(sim, 2),
        "usage_over_billed": round(usage / billed, 3) if billed else None,
        "mean_usd_per_unit": round((usage + sim) / n, 4) if n else None,
    }


def census(
    *, parquet: Path, runs_root: Path, root: Path, suite: str, arms: Sequence[str]
) -> dict[str, Any]:
    recs = load_forks(parquet, runs_root, suite, arms)
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in recs:
        by_split[str(r["split"])].append(r)
    report: dict[str, Any] = {
        "suite": suite,
        "arms": list(arms),
        "fork_continuations_with_a_recorded_decision": len(recs),
        "by_split": {k: len(v) for k, v in sorted(by_split.items())},
        "first_action_kind": dict(Counter(r["action_kind"] for r in recs)),
        "distinct_tasks": len({r["task_id"] for r in recs}),
        "distinct_traces": len({r["trace"] for r in recs}),
        "user_channel": channel_block(recs),
        "templates": template_block(recs, root),
        "state_guard": guard_block(recs),
        "cost": cost_block(recs),
        "pairs": {},
    }
    for split, rows in sorted(by_split.items()):
        report["pairs"][split] = {
            "loose": pair_yield(rows, strict=False),
            "strict": pair_yield(rows, strict=True),
        }
    return report


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--suite", default="tau2_retail")
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--runs-root", default=None, help="default <root>/runs")
    ap.add_argument("--parquet", default=None, help="default <root>/scores/parquet/runs.parquet")
    ap.add_argument("--arm", action="append", default=None, help=f"default {list(POLICY_ARMS)}")
    ap.add_argument("--out", default=None, help="write the report as JSON here too")
    a = ap.parse_args(argv)

    root = Path(a.root).resolve()
    runs_root = Path(a.runs_root).resolve() if a.runs_root else root / "runs"
    parquet = (
        Path(a.parquet).resolve() if a.parquet else root / "scores" / "parquet" / "runs.parquet"
    )
    if not parquet.exists():
        print(f"no runs.parquet at {parquet}; run `pi compact` first", file=sys.stderr)
        return 2

    rep = census(
        parquet=parquet,
        runs_root=runs_root,
        root=root,
        suite=a.suite,
        arms=tuple(a.arm) if a.arm else POLICY_ARMS,
    )
    text = json.dumps(rep, indent=2, sort_keys=True)
    print(text)
    if a.out:
        Path(a.out).write_text(text + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
