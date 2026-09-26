"""Symmetric matched-cost contrasts for the structured-baselines campaign, locked before reading.

WHAT THIS READS. The clean grid `conf/grids/structured_baselines_20260922.yaml` (test split,
musique and strategyqa, 200 tasks, seeds 0 and 1, cap 8) run in four invocations from ONE pinned
worktree (code_version a61c4f4be3e9) through ONE base URL, differing only in the questioner:

    A  runs/           inquirer_prompted, par2_rag   @ gpt-oss-120b
    B  runs/           inquirer_trained              @ qwen3-8b-dpo-stacked-notdone-both
    C  runs_8b/        inquirer_prompted, par2_rag   @ qwen3-8b-base
    D  runs_frontier/  inquirer_prompted, par2_rag   @ claude-opus-5

Each root is scored into its own store by `score.sh`, so one ARM here is (store, arm_id, the
inquirer model it must carry). `arm_id` alone does not name an arm: `inquirer_prompted` exists in
three stores at three models, which is how a comparator pooled across pins once made a finding
look stable (memory: comparator-by-arm-id-pools-model-pins). The inquirer model is read from each
run's own manifest, `pins.inquirer.model_id`, and a run carrying any other model is a refusal.

THE RULE is the paper's Table 1 rule and nothing else: seed-matched `(suite, task, seed)` pairing,
BOTH arms read at `min(k_a, k_b)` (the shared, smaller question count), seeds averaged into the
task, a paired bootstrap over tasks. It is `symmetric_completed.seed_matched_symmetric`, imported,
not re-implemented. `matched_cost.contrast` truncates only the comparator and is not used.

THE LOCKS, both required before any cell here is reported:

1. PUBLISHED-CELL LOCK. On the completed held-out cohort this reader must reproduce the paper's
   Table 1 coverage cells (`artifacts/baseline_completion_20260920/three.json`, symmetric deltas)
   and its depth-weighted recall cells (`artifacts/plan_metrics_completed_20260920/
   symmetric_completed.json`) to `--tol`. That coverage record was computed from the scorer's
   `frontier_q` ladder and this reader rebuilds coverage from `turns.parquet`, so agreement is
   two routes to one quantity, not one route run twice.
2. INSTRUMENT LOCK. On every store read here, each metric rebuilt at `k = n_asks` must equal the
   scorer's own stored value run by run, on value AND on presence (`sweep.terminal_check`'s
   rule). A metric that fails is withheld for that store, never reported with a footnote.

Resamples follow the repository's rule for bounds near zero: 1k, 10k, and 50k at three bootstrap
seeds, with a cell DECIDED only if every 50k interval excludes zero on the same side.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
for _p in (
    REPO / "src",
    REPO / "scripts",
    REPO / "scripts" / "plan_metrics_symmetric",
    REPO / "scripts" / "plan_metrics_completed",
):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import duckdb  # noqa: E402
from plan_metrics_symmetric import ladder  # noqa: E402

# ONE MODULE OBJECT FOR `ladder`, or a registered metric is invisible. `sweep` imports it as
# `plan_metrics_symmetric.ladder` and `symmetric_completed` as bare `ladder`; Python loads those
# as two modules with two METRIC_FNS dicts, so `sweep.registered_metrics` added evidence_coverage
# to one and `seed_matched_symmetric` looked it up in the other (KeyError, found on first run).
# Aliasing before the second import makes both names the same object.
sys.modules["ladder"] = ladder

import sweep  # noqa: E402
from symmetric_completed import seed_matched_symmetric  # noqa: E402

assert sweep.ladder is ladder, "sweep bound a different ladder module"

SUITES: tuple[str, ...] = ("musique", "strategyqa")
METRICS: tuple[str, ...] = (
    "evidence_coverage",
    "dwr",
    "max_depth_reached",
    "facet_breadth_scorer",
    "precedence_violation_rate",
)
COVERAGE_METRICS = frozenset({"evidence_coverage"})
# The ladder registers the scorer's COUNT under a different name than the scorer stores it, because
# `ladder.METRIC_FNS["facet_breadth"]` is a RATIO (a different function under the same name).
STORED_NAME: Mapping[str, str] = {"facet_breadth_scorer": "facet_breadth"}
RESAMPLES: tuple[tuple[int, int], ...] = (
    (1000, 0),
    (10000, 0),
    (50000, 101),
    (50000, 202),
    (50000, 303),
)


class Refusal(SystemExit):
    """A population or instrument that fails a check. Exits 2, distinct from a crash."""

    def __init__(self, msg: str) -> None:
        super().__init__(2)
        self.msg = msg

    def __str__(self) -> str:
        return self.msg


@dataclass(frozen=True)
class ArmSpec:
    label: str
    store: Path
    runs_root: Path
    arm_id: str
    inquirer_model: str


def parse_arm(spec: str) -> ArmSpec:
    """`label=store_dir:runs_root:arm_id:inquirer_model`."""
    label, _, rest = spec.partition("=")
    parts = rest.split(":")
    if not label or len(parts) < 4:
        raise argparse.ArgumentTypeError(f"bad --arm {spec!r}")
    store, runs_root, arm_id = parts[0], parts[1], parts[2]
    model = ":".join(parts[3:])
    return ArmSpec(label, Path(store), Path(runs_root), arm_id, model)


def _digest(ids: Sequence[str]) -> str:
    h = hashlib.sha256()
    for rid in sorted(ids):
        h.update(rid.encode())
        h.update(b"\n")
    return h.hexdigest()


# ------------------------------------------------------------------------ population checks


def population_failures(rows: Sequence[Mapping[str, Any]], spec: ArmSpec) -> list[str]:
    """Every reason this arm's population cannot be read. Empty means readable.

    `rows` are runs.parquet rows joined with each run's manifest pins, one per run. Pure, so each
    check has a test that makes it fire (tests/test_structured_baselines_read.py).
    """
    if not rows:
        return [f"{spec.label}: no ok runs of arm {spec.arm_id} in {spec.store}"]
    fail: list[str] = []
    models = sorted({str(r["inquirer_model"]) for r in rows})
    if models != [spec.inquirer_model]:
        fail.append(
            f"{spec.label}: inquirer models {models}, expected only {spec.inquirer_model!r} "
            "(an arm_id pooled across pins is not one arm)"
        )
    cvs = sorted({str(r["code_version"]) for r in rows})
    if len(cvs) != 1:
        fail.append(f"{spec.label}: {len(cvs)} code_versions {cvs}")
    if any(bool(r["is_dev_run"]) or str(r["run_id"]).startswith("dev-") for r in rows):
        fail.append(f"{spec.label}: dev- runs present (training-only, never an eval arm)")
    if any(bool(r["dirty"]) for r in rows):
        fail.append(f"{spec.label}: dirty runs present")
    if any(bool(r.get("canary_hit")) for r in rows):
        fail.append(f"{spec.label}: a canary hit, gold leaked into a request")
    if any(r.get("firewall_ok") is False for r in rows):
        fail.append(f"{spec.label}: firewall_ok is False on some run")
    keys = [(r["suite_id"], r["task_id"], int(r["seed"])) for r in rows]
    if len(keys) != len(set(keys)):
        fail.append(f"{spec.label}: more than one run at a (suite, task, seed) cell")
    return fail


def contrast_failures(a: Sequence[Mapping[str, Any]], b: Sequence[Mapping[str, Any]]) -> list[str]:
    """Two arms are one experiment only at one code_version and one base URL per role."""
    fail: list[str] = []
    cv = {str(r["code_version"]) for r in a} | {str(r["code_version"]) for r in b}
    if len(cv) != 1:
        fail.append(f"contrast spans {len(cv)} code_versions {sorted(cv)}")
    urls = {str(r["base_url_sha"]) for r in a} | {str(r["base_url_sha"]) for r in b}
    if len(urls) != 1:
        fail.append(f"contrast spans {len(urls)} base_url_shas {sorted(urls)}")
    frozen = {str(r["frozen_prompts"]) for r in a} | {str(r["frozen_prompts"]) for r in b}
    if len(frozen) != 1:
        fail.append("the frozen roles' prompts differ across the two arms")
    return fail


def load_arm(spec: ArmSpec, suites: Sequence[str]) -> list[dict[str, Any]]:
    con = duckdb.connect()
    ph = ",".join("?" for _ in suites)
    df = con.execute(
        "SELECT run_id, suite_id, task_id, seed, n_asks, code_version, dirty, is_dev_run, "
        "canary_hit, firewall_ok, split, status, corpus_hash "
        f"FROM read_parquet('{(spec.store / 'runs.parquet').as_posix()}') "
        f"WHERE arm_id = ? AND suite_id IN ({ph}) AND status = 'ok'",
        [spec.arm_id, *suites],
    ).fetchdf()
    rows: list[dict[str, Any]] = []
    for r in df.to_dict("records"):
        if r["split"] != "test":
            raise Refusal(f"{spec.label}: run {r['run_id']} is split {r['split']!r}, not test")
        man_path = spec.runs_root / r["run_id"] / "manifest.json"
        if not man_path.is_file():
            raise Refusal(f"{spec.label}: no manifest at {man_path}")
        man = json.loads(man_path.read_text())
        pins = man["pins"]
        r["inquirer_model"] = pins["inquirer"]["model_id"]
        base_urls = {p["base_url_sha"] for p in pins.values()}
        r["base_url_sha"] = ",".join(sorted(base_urls))
        ph_ = man["prompt_hashes"]
        r["frozen_prompts"] = json.dumps(
            {k: v for k, v in sorted(ph_.items()) if k.startswith(("answerer", "drafter"))}
        )
        rows.append(r)
    return rows


# ------------------------------------------------------------------------------ instrument


def instrument_check(
    metric: str,
    arms: Mapping[str, ladder.RunLadder],
    graphs: Mapping[str, Any],
    stored: Mapping[tuple[str, str], float],
    tol: float = 1e-9,
) -> dict[str, int]:
    """`sweep.terminal_check` with the stored name mapped, on value AND presence."""
    fn = ladder.METRIC_FNS[metric]
    name = STORED_NAME.get(metric, metric)
    agree = differ = only_here = only_stored = 0
    for rid, lad in arms.items():
        graph = graphs.get(lad.task_id)
        if graph is None:
            continue
        mine = fn(lad.at(lad.n_asks), graph)
        theirs = stored.get((rid, name))
        if math.isnan(mine) and theirs is None:
            continue
        if math.isnan(mine):
            only_stored += 1
        elif theirs is None:
            only_here += 1
        elif abs(mine - theirs) <= tol:
            agree += 1
        else:
            differ += 1
    return {
        "agree": agree,
        "differ": differ,
        "emitted_here_absent_in_store": only_here,
        "in_store_absent_here": only_stored,
    }


def instrument_ok(check: Mapping[str, int]) -> bool:
    return (
        check["agree"] > 0
        and check["differ"] == 0
        and check["emitted_here_absent_in_store"] == 0
        and check["in_store_absent_here"] == 0
    )


def _ladders(
    store: Path, run_ids: Sequence[str], graphs: Mapping[str, Any]
) -> tuple[dict[str, ladder.RunLadder], dict[str, ladder.RunLadder]]:
    return (
        ladder.load_run_ladders(store, run_ids),
        sweep.load_coverage_ladders(store, run_ids, graphs),
    )


def _task_signs(
    metric: str,
    a: Mapping[str, ladder.RunLadder],
    b: Mapping[str, ladder.RunLadder],
    graphs: Mapping[str, Any],
    seeds: Mapping[str, int],
) -> dict[str, Any]:
    """Per-task sign counts and mean shared k, over exactly seed_matched_symmetric's pairing."""
    fn = ladder.METRIC_FNS[metric]
    a_by = {(lad.suite_id, lad.task_id, int(seeds[r])): lad for r, lad in a.items()}
    b_by = {(lad.suite_id, lad.task_id, int(seeds[r])): lad for r, lad in b.items()}
    acc: dict[str, tuple[list[float], list[float]]] = {}
    ks: list[int] = []
    for key in sorted(set(a_by) & set(b_by)):
        graph = graphs.get(key[1])
        if graph is None:
            continue
        k = min(a_by[key].n_asks, b_by[key].n_asks)
        av, bv = fn(a_by[key].at(k), graph), fn(b_by[key].at(k), graph)
        if math.isnan(av) or math.isnan(bv):
            continue
        ks.append(k)
        la, lb = acc.setdefault(key[1], ([], []))
        la.append(av)
        lb.append(bv)
    wins = losses = ties = 0
    for la, lb in acc.values():
        d = sum(la) / len(la) - sum(lb) / len(lb)
        wins += d > 0
        losses += d < 0
        ties += d == 0
    return {
        "a_gt_b": wins,
        "a_lt_b": losses,
        "tie": ties,
        "mean_k_common": (sum(ks) / len(ks)) if ks else None,
    }


def read_cell(
    metric: str,
    a: Mapping[str, ladder.RunLadder],
    b: Mapping[str, ladder.RunLadder],
    graphs: Mapping[str, Any],
    seeds: Mapping[str, int],
) -> dict[str, Any]:
    readings = [
        {
            "n_boot": nb,
            "seed": sd,
            **seed_matched_symmetric(metric, a, b, graphs, seeds, seed=sd, n_boot=nb),
        }
        for nb, sd in RESAMPLES
    ]
    fifty = [x for x in readings if x["n_boot"] == 50000]
    excludes = all(x["ci_lo"] > 0 for x in fifty) or all(x["ci_hi"] < 0 for x in fifty)
    return {
        "readings": readings,
        "verdict": "DECIDED" if excludes else "SPANS_ZERO",
        "signs": _task_signs(metric, a, b, graphs, seeds),
    }


def read_pooled_cell(
    metric: str,
    members: Sequence[Mapping[str, ladder.RunLadder]],
    b: Mapping[str, ladder.RunLadder],
    graphs: Mapping[str, Any],
    seeds: Mapping[str, int],
) -> dict[str, Any]:
    """read_cell for several training seeds of one recipe, averaged within the task."""
    p = str(REPO / "scripts" / "seed_identity")
    if p not in sys.path:
        sys.path.insert(0, p)
    from table1_by_seed import pooled_symmetric  # lazy: that module imports this one

    readings = [
        {
            "n_boot": nb,
            "seed": sd,
            **pooled_symmetric(metric, list(members), b, graphs, seeds, seed=sd, n_boot=nb),
        }
        for nb, sd in RESAMPLES
    ]
    fifty = [x for x in readings if x["n_boot"] == 50000]
    excludes = all(x["ci_lo"] > 0 for x in fifty) or all(x["ci_hi"] < 0 for x in fifty)
    return {
        "readings": readings,
        "verdict": "DECIDED" if excludes else "SPANS_ZERO",
        "pooled": True,
    }


# ------------------------------------------------------------------------------------ lock


def published_lock(cohort: Path, graphs_by_suite: Mapping[str, Any], tol: float) -> dict[str, Any]:
    """Reproduce Table 1's coverage and dwr cells on their own population, or refuse."""
    cov = json.loads((REPO / "artifacts/baseline_completion_20260920/three.json").read_text())
    dwr = json.loads(
        (REPO / "artifacts/plan_metrics_completed_20260920/symmetric_completed.json").read_text()
    )
    store = cohort / "scores_parquet"
    seeds = _seed_map(store)
    out: dict[str, Any] = {}
    with sweep.registered_metrics({**sweep.EXTRA_METRIC_FNS, **sweep.DIAGNOSTIC_METRIC_FNS}):
        for suite in ("musique", "strategyqa", "wiki2"):
            graphs = graphs_by_suite[suite]
            tr = _ids(cohort / "cohort" / f"run_ids.trained.{suite}.txt")
            pr = _ids(cohort / "cohort" / f"run_ids.prompted.{suite}.txt")
            run_l, cov_l = _ladders(store, tr + pr, graphs)
            for metric, want in (
                ("evidence_coverage", cov["complete"]["by_suite"][suite]["symmetric"]["delta"]),
                (
                    "dwr",
                    next(x for x in dwr["completed"][f"dwr::{suite}"] if x["n_boot"] == 10000)[
                        "delta"
                    ],
                ),
            ):
                lad = cov_l if metric in COVERAGE_METRICS else run_l
                got = seed_matched_symmetric(
                    metric,
                    {r: lad[r] for r in tr if r in lad},
                    {r: lad[r] for r in pr if r in lad},
                    graphs,
                    seeds,
                    seed=0,
                    n_boot=1000,
                )
                ok = abs(got["delta"] - want) <= tol and got["n"] == 200
                out[f"{metric}::{suite}"] = {
                    "published": want,
                    "reproduced": got["delta"],
                    "n": got["n"],
                    "abs_diff": abs(got["delta"] - want),
                    "verdict": "LOCKED" if ok else "FAILED",
                }
    failed = [k for k, v in out.items() if v["verdict"] != "LOCKED"]
    if failed:
        raise Refusal(f"published-cell lock FAILED on {failed}: {json.dumps(out, indent=1)}")
    return out


def _ids(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]


def _seed_map(store: Path) -> dict[str, int]:
    con = duckdb.connect()
    return {
        r[0]: int(r[1])
        for r in con.execute(
            f"SELECT run_id, seed FROM read_parquet('{(store / 'runs.parquet').as_posix()}')"
        ).fetchall()
    }


# ------------------------------------------------------------------------------------ main


def run(
    arms: Sequence[ArmSpec],
    contrasts: Sequence[tuple[str, str]],
    *,
    cohort: Path,
    graph_version: str,
    tol: float,
    pools: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    """`pools` maps a label to member arm labels (training seeds of one recipe). A pool may only be
    the A side of a contrast; its cells average every member's pairs within the task
    (seed_identity/table1_by_seed.pooled_symmetric), each member keyed separately so two training
    seeds never collide on a (suite, task, rollout seed) key. Every member passes the population
    and instrument checks on its own."""
    pools = dict(pools or {})
    known = {a.label for a in arms}
    for la, lb in contrasts:  # validated before any I/O, so a bad plan costs nothing
        if lb in pools:
            raise Refusal(f"a pool may only be the A side of a contrast: {la},{lb}")
    for lab, ms in pools.items():
        if len(ms) < 2 or any(m not in known for m in ms):
            raise Refusal(f"pool {lab} needs two or more known members, got {ms}")
    from pi_eval.gold import load_graphs

    graphs_by_suite = {s: load_graphs(s, graph_version) for s in ("musique", "strategyqa", "wiki2")}
    lock = published_lock(cohort, graphs_by_suite, tol)

    by_label = {a.label: a for a in arms}
    rows = {a.label: load_arm(a, SUITES) for a in arms}
    pop_fail = [f for a in arms for f in population_failures(rows[a.label], a)]
    if pop_fail:
        raise Refusal("population checks failed:\n  " + "\n  ".join(pop_fail))

    seeds: dict[str, int] = {}
    for a in arms:
        for r in rows[a.label]:
            seeds[r["run_id"]] = int(r["seed"])

    out: dict[str, Any] = {
        "rule": "seed-matched (suite, task, seed); both arms at min(k_a, k_b); seeds averaged "
        "into the task; paired bootstrap over tasks (symmetric_completed.seed_matched_symmetric)",
        "published_lock": lock,
        "arms": {},
        "instrument": {},
        "contrasts": {},
    }
    ladders: dict[tuple[str, str], tuple[dict, dict]] = {}
    with sweep.registered_metrics({**sweep.EXTRA_METRIC_FNS, **sweep.DIAGNOSTIC_METRIC_FNS}):
        for a in arms:
            for suite in SUITES:
                ids = [r["run_id"] for r in rows[a.label] if r["suite_id"] == suite]
                ladders[(a.label, suite)] = _ladders(a.store, ids, graphs_by_suite[suite])
                stored = sweep._stored_values(a.store, ids)
                sub = [r for r in rows[a.label] if r["suite_id"] == suite]
                cov_at_stop = [
                    stored[(r["run_id"], "evidence_coverage")]
                    for r in sub
                    if (r["run_id"], "evidence_coverage") in stored
                ]
                out["arms"].setdefault(a.label, {})[suite] = {
                    "store": str(a.store.relative_to(REPO))
                    if a.store.is_relative_to(REPO)
                    else str(a.store),
                    "arm_id": a.arm_id,
                    "inquirer_model": a.inquirer_model,
                    "n_runs": len(sub),
                    "n_tasks": len({r["task_id"] for r in sub}),
                    "run_id_sha256": _digest(ids),
                    "code_version": sub[0]["code_version"] if sub else None,
                    "base_url_sha": sub[0]["base_url_sha"] if sub else None,
                    "mean_n_asks": sum(r["n_asks"] for r in sub) / len(sub) if sub else None,
                    "mean_coverage_at_own_stop": sum(cov_at_stop) / len(cov_at_stop)
                    if cov_at_stop
                    else None,
                }
                for metric in METRICS:
                    run_l, cov_l = ladders[(a.label, suite)]
                    lad = cov_l if metric in COVERAGE_METRICS else run_l
                    out["instrument"].setdefault(a.label, {}).setdefault(suite, {})[metric] = (
                        instrument_check(metric, lad, graphs_by_suite[suite], stored)
                    )

        for la, lb in contrasts:
            members = list(pools.get(la, [la]))
            a_rows = [r for m in members for r in rows[m]]
            cf = contrast_failures(a_rows, rows[lb])
            name = f"{la}_minus_{lb}"
            if cf:
                out["contrasts"][name] = {"refused": cf}
                continue
            cell: dict[str, Any] = {
                "a": la,
                "a_members": members,
                "b": lb,
                "a_model": sorted({by_label[m].inquirer_model for m in members}),
                "b_model": by_label[lb].inquirer_model,
            }
            for suite in SUITES:
                for metric in METRICS:
                    ok_a = all(instrument_ok(out["instrument"][m][suite][metric]) for m in members)
                    ok_b = instrument_ok(out["instrument"][lb][suite][metric])
                    key = f"{metric}::{suite}"
                    if not (ok_a and ok_b):
                        cell[key] = {"withheld": "instrument lock failed on one arm"}
                        continue
                    idx = 1 if metric in COVERAGE_METRICS else 0
                    if la in pools:
                        cell[key] = read_pooled_cell(
                            metric,
                            [ladders[(m, suite)][idx] for m in members],
                            ladders[(lb, suite)][idx],
                            graphs_by_suite[suite],
                            seeds,
                        )
                    else:
                        cell[key] = read_cell(
                            metric,
                            ladders[(la, suite)][idx],
                            ladders[(lb, suite)][idx],
                            graphs_by_suite[suite],
                            seeds,
                        )
            out["contrasts"][name] = cell
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument(
        "--arm",
        action="append",
        type=parse_arm,
        required=True,
        help="label=store_dir:runs_root:arm_id:inquirer_model (repeatable)",
    )
    ap.add_argument(
        "--contrast",
        action="append",
        required=True,
        help="labelA,labelB -- delta is A minus B (repeatable)",
    )
    ap.add_argument(
        "--pool",
        action="append",
        default=[],
        help="label=memberA,memberB -- training seeds of one recipe, pooled within the task; "
        "usable only as the A side of --contrast (repeatable)",
    )
    ap.add_argument("--cohort", type=Path, default=REPO / "artifacts/completed_cohort_20260922")
    ap.add_argument("--graph-version", default="v1")
    ap.add_argument("--tol", type=float, default=1e-9)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if not os.environ.get("PI_GOLD_ROOT"):
        print(
            "read_contrasts: REFUSING: PI_GOLD_ROOT unset; scoring-stage readers need gold",
            file=sys.stderr,
        )
        return 3
    labels = {a.label for a in args.arm}
    pools: dict[str, list[str]] = {}
    for spec in args.pool:
        lab, _, mem = spec.partition("=")
        ms = [m for m in mem.split(",") if m]
        if not lab or len(ms) < 2 or any(m not in labels for m in ms) or lab in labels:
            print(f"read_contrasts: bad --pool {spec!r}", file=sys.stderr)
            return 3
        pools[lab] = ms
    labels |= set(pools)
    pairs = []
    for c in args.contrast:
        la, _, lb = c.partition(",")
        if la not in labels or lb not in labels:
            print(f"read_contrasts: unknown arm label in {c!r}", file=sys.stderr)
            return 3
        pairs.append((la, lb))
    try:
        out = run(
            args.arm,
            pairs,
            cohort=args.cohort,
            graph_version=args.graph_version,
            tol=args.tol,
            pools=pools,
        )
    except Refusal as r:
        print(f"read_contrasts: REFUSING: {r}", file=sys.stderr)
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, sort_keys=True, default=str) + "\n")
    print(f"read_contrasts: wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
