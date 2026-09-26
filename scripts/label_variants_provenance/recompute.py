"""Provenance for the four-label-source table, or a finding that it does not reproduce.

WHY THIS FILE EXISTS. `artifacts/label_variants_heldout_20260919/RESULT.md` reports that the
symmetric matched-cost `evidence_coverage` gain over the prompted 8B base excludes zero on all
three held-out suites for four preference-trained checkpoints that differ in label source, and
the paper leans on it for "the gain follows the preference stage ... rather than the label
source". That record names no store, no scorer_hash, no script and no run-id list, so under
CLAUDE.md rule 1 it is not yet a result. This script re-derives it with provenance, LOCKS against
already-published values before printing anything new, and only then answers the question the
paper's sentence actually asks -- whether the four arms differ from EACH OTHER -- with direct
paired arm-versus-arm contrasts, because overlapping per-arm intervals are not that test and
matched-cost deltas against a moving base rung do not subtract.

POPULATION. Every arm is resolved through `pinq_train.gate._select_runs` (via
`scripts/stopping_answer_test/lib.load_arm_runs`), which matches the MODEL through
`calls.parquet`. That alone is not enough on the shared store: under grid
`tier1_trained_qa_base` the model `qwen3-8b-base` carries two pins over three code versions, so
the selection is then restricted to `split`, `NOT dirty`, `NOT gold_exposed`, `budget_cap` and a
`code_version` prefix, and the script REFUSES unless each arm is then exactly one model pin and
one code version with unique (suite, task, seed) keys. What each filter removed is printed.

ESTIMAND. Symmetric matched cost: per (suite, task, seed) both arms are read at
`k = min(k_a, k_b)` off their own prefix ladder, seeds are averaged into the task, the paired
per-task difference is bootstrapped over tasks (BCa). Two independent routes compute it:

  route A  `plan_metrics_completed.symmetric_completed.seed_matched_symmetric` over the coverage
           ladders `plan_metrics_symmetric.sweep.load_coverage_ladders` rebuilds from
           `turns.parquet`;
  route B  the `symmetric` block of `pinq_train.gate._matched_cost`, which reads the STORED
           `scores.frontier_q#k` ladder directly (added 2026-09-19 beside the asymmetric rule,
           which it does not alter).

Route A's turn-derived ladder is also checked RUNG BY RUNG against the stored `frontier_q#k` for
every run and every k in 0..n_asks. Route B is the INTERVAL OF RECORD -- it is the code every
published figure locked against here came from -- and route A must agree with it on n and the
point to 1e-12; its bounds are compared and reported but can differ by an order statistic on a
lattice-valued cell for a floating-point reason measured in `cross_check`. `ladder._pair_keys`
(the N:M cross-product pairing) is not used, and the gate's ASYMMETRIC figure appears only in
lock (c) and as context.

THE LOCKS, all read before any new number:
  (b) the paper's printed selected-arm cells (`artifacts/baseline_completion_20260920/three.json`,
      completed comparator; the body prints +0.108 [+0.074, +0.145]) on their own store;
  (c) the gate's ASYMMETRIC verdicts for three of the four arms from a different scoring pass
      (`artifacts/label_ordering_test_20260918/verdicts`), on this script's population -- does
      the store read here score these runs as that pass did;
  (d) the ARM-VS-ARM path, against the published seed-vs-seed symmetric cells of
      `artifacts/seedrep_gate_20260919` at 10,000 and 50,000 resamples.
The script STOPS (exit 2), printing no four-arm table, unless every cell of (b), (c) and (d)
reproduces.

(a) WAS A GATE AND IS NOW A RECORDED, NON-GATING CHECK (decision of 2026-09-23). It compared this
reader against the twelve cells of `artifacts/label_variants_heldout_20260919`, encoding the belief
that those cells are this estimand's output on this population. The first run of this script
(commit 34ef5ee) refuted that belief rather than the reader: (a) failed on all twelve POINT
estimates while (b), (c) and (d) reproduced exactly, and the record's own quoted levels are not
this population's (base musique asks 6.59 against 6.09, coverage 0.8321 against 0.8019). The
record was therefore WITHDRAWN. (a) is still computed and printed against every cell, labelled as
the withdrawn record, so the audit trail shows the disagreement on every run; it no longer gates,
because validating the reader is what (b), (c) and (d) do, on published values.

TWO LADDER MODULE OBJECTS. `sweep` imports `plan_metrics_symmetric.ladder`; `symmetric_completed`
imports the same file as top-level `ladder`. `sweep.registered_metrics` registers
`evidence_coverage` into the first only, so `seed_matched_symmetric` would raise KeyError.
`coverage_registered` puts sweep's own function into BOTH tables for the duration of the run,
refusing to shadow an existing entry, and asserts they hold the same function object.

RESAMPLE RULE (post-lock only). Every bound is read at 1,000, 10,000 and 50,000 resamples (seed
0). A cell with any bound within 0.01 of zero is re-read at 50,000 under seeds 1 and 2 too. A cell
is `excludes_zero_*` only if EVERY reading excludes zero on the same side, `includes_zero` only if
every reading includes it, and `undecided` otherwise.

USAGE (gold-side; run from the repository root; reads only, writes only under --out-dir):

    PI_GOLD_ROOT="$PWD/data/gold" .venv/bin/python \\
        scripts/label_variants_provenance/recompute.py \\
        --out-dir artifacts/label_variants_provenance_20260922
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import itertools
import json
import math
import platform
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "plan_metrics_symmetric"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# `sweep` is imported top-level FIRST so `symmetric_completed`'s own `from sweep import ...`
# binds this same module object rather than a second copy.
import sweep  # noqa: E402
from plan_metrics_completed import symmetric_completed  # noqa: E402
from stopping_answer_test import lib as stop_lib  # noqa: E402

from pi_eval.gold import load_graphs  # noqa: E402
from pinq_train import gate  # noqa: E402

METRIC = "evidence_coverage"
SUITES = ("musique", "strategyqa", "wiki2")
GRID = "tier1_trained_qa_base"
BASE = "base"
TRAINED = ("control", "rater", "reaches", "stopcontrast")
ARMS: dict[str, tuple[str, str]] = {
    BASE: ("inquirer_prompted", "qwen3-8b-base"),
    "control": ("inquirer_trained", "qwen3-8b-dpo-headline-control"),
    "rater": ("inquirer_trained", "qwen3-8b-dpo-headline-rater"),
    "reaches": ("inquirer_trained", "qwen3-8b-dpo-headline-reaches"),
    "stopcontrast": ("inquirer_trained", "qwen3-8b-dpo-headline-stopcontrast"),
}
SELECTED = "qwen3-8b-dpo-stacked-notdone-both"
SEEDREP_A, SEEDREP_B = f"{SELECTED}-s1", f"{SELECTED}-s2"

MAIN_READINGS: tuple[tuple[int, int], ...] = ((1000, 0), (10000, 0), (50000, 0))
STABILITY_READINGS: tuple[tuple[int, int], ...] = ((50000, 1), (50000, 2))
NEAR_ZERO = 0.01
DELTA_TOL = 1e-12
CI_TOL = 1e-9
RUNG_TOL = 1e-12
ROUTE_A_BOUND_TOL = 1e-3  # see `cross_check`: bounds of the two routes, not of record vs here

# The record's twelve cells, as printed (4 dp), at "10,000 resamples with a fixed seed".
RECORD_CELLS: dict[tuple[str, str], tuple[float, float, float]] = {
    ("control", "musique"): (0.1227, 0.0885, 0.1577),
    ("control", "strategyqa"): (0.0651, 0.0309, 0.1007),
    ("control", "wiki2"): (0.0506, 0.0206, 0.0825),
    ("rater", "musique"): (0.1248, 0.0910, 0.1596),
    ("rater", "strategyqa"): (0.0622, 0.0263, 0.0983),
    ("rater", "wiki2"): (0.0550, 0.0244, 0.0875),
    ("reaches", "musique"): (0.1308, 0.0963, 0.1665),
    ("reaches", "strategyqa"): (0.0625, 0.0274, 0.0986),
    ("reaches", "wiki2"): (0.0481, 0.0181, 0.0800),
    ("stopcontrast", "musique"): (0.1263, 0.0917, 0.1615),
    ("stopcontrast", "strategyqa"): (0.0520, 0.0145, 0.0899),
    ("stopcontrast", "wiki2"): (0.0462, 0.0144, 0.0794),
}


class Refused(RuntimeError):
    """A precondition failed. The message is the finding; nothing downstream is printed."""


# ------------------------------------------------------------------------------ plumbing


def rel(p: Path) -> str:
    p = Path(p).resolve()
    return str(p.relative_to(REPO)) if p.is_relative_to(REPO) else str(p)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def digest_ids(ids: Sequence[str]) -> str:
    h = hashlib.sha256()
    for rid in sorted(ids):
        h.update(rid.encode() + b"\n")
    return h.hexdigest()


def read_ids(p: Path) -> list[str]:
    return [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.strip()


READER_PATHS = (
    "scripts/label_variants_provenance/recompute.py",
    "scripts/plan_metrics_completed/symmetric_completed.py",
    "scripts/plan_metrics_symmetric/sweep.py",
    "scripts/plan_metrics_symmetric/ladder.py",
    "scripts/stopping_answer_test/lib.py",
    "src/pinq_train/gate.py",
    "src/pi_eval",
)


def reader_identity() -> dict[str, Any]:
    status = git("status", "--porcelain", "--", *READER_PATHS)
    return {
        "head": git("rev-parse", "HEAD"),
        "reader_paths": list(READER_PATHS),
        "reader_paths_clean_against_head": status == "",
        "reader_paths_status": status.splitlines(),
        "blob_sha": {p: git("hash-object", p) for p in READER_PATHS if (REPO / p).is_file()},
    }


@contextlib.contextmanager
def coverage_registered() -> Iterator[Callable[..., float]]:
    """`sweep`'s `evidence_coverage`, visible to both ladder module objects (module docstring)."""
    fn = sweep.EXTRA_METRIC_FNS[METRIC]
    tables = {id(m.METRIC_FNS): m.METRIC_FNS for m in (sweep.ladder, symmetric_completed.ladder)}
    for t in tables.values():
        if METRIC in t:
            raise Refused(f"a ladder module already defines {METRIC!r}; refusing to shadow it")
    for t in tables.values():
        t[METRIC] = fn
    try:
        assert symmetric_completed.ladder.METRIC_FNS[METRIC] is sweep.ladder.METRIC_FNS[METRIC]
        yield fn
    finally:
        for t in tables.values():
            t.pop(METRIC, None)


# ---------------------------------------------------------------------------- population


def _meta(con, ids: Sequence[str]) -> dict[str, dict[str, Any]]:
    cols = (
        "run_id, suite_id, task_id, seed, split, dirty, gold_exposed, code_version, "
        "model_pin_hash, budget_cap, n_asks, n_turns, stop_reason"
    )
    rows = gate._rows(con, f"SELECT {cols} FROM runs WHERE run_id IN {gate._in(ids)}")
    return {str(r["run_id"]): r for r in rows}


def select_arm(
    con,
    arm_id: str,
    model_id: str,
    *,
    split: str,
    code_prefix: str,
    budget_cap: int = 8,
    grid: str = GRID,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    runs = stop_lib.load_arm_runs(con, arm_id=arm_id, model_id=model_id, grid_name=grid)
    meta = _meta(con, [r["run_id"] for r in runs])
    before = Counter(
        (
            str(meta[r["run_id"]]["model_pin_hash"])[:16],
            str(meta[r["run_id"]]["code_version"])[:12],
            str(meta[r["run_id"]]["split"]),
            bool(meta[r["run_id"]]["dirty"]),
        )
        for r in runs
    )
    removed: dict[str, int] = {}
    keep = list(runs)
    filters: list[tuple[str, Callable[[Mapping[str, Any]], bool]]] = [
        (f"split = {split!r}", lambda m: m["split"] == split),
        ("dirty IS FALSE", lambda m: m["dirty"] is False),
        ("gold_exposed IS NOT TRUE", lambda m: m["gold_exposed"] is not True),
        (f"budget_cap = {budget_cap}", lambda m: m["budget_cap"] == budget_cap),
        (
            f"code_version LIKE '{code_prefix}%'",
            lambda m: str(m["code_version"]).startswith(code_prefix),
        ),
    ]
    for name, pred in filters:
        nxt = [r for r in keep if pred(meta[r["run_id"]])]
        removed[name] = len(keep) - len(nxt)
        keep = nxt
    for r in keep:
        m = meta[r["run_id"]]
        r["n_asks"], r["stop_reason"], r["n_turns"] = m["n_asks"], m["stop_reason"], m["n_turns"]
    pins = sorted({str(meta[r["run_id"]]["model_pin_hash"]) for r in keep})
    codes = sorted({str(meta[r["run_id"]]["code_version"]) for r in keep})
    keys = Counter((r["suite_id"], r["task_id"], int(r["seed"])) for r in keep)
    dups = [k for k, n in keys.items() if n > 1]
    if len(pins) != 1 or len(codes) != 1 or dups:
        raise Refused(
            f"{model_id}: after filtering, {len(pins)} model pins {[p[:16] for p in pins]}, "
            f"{len(codes)} code versions, {len(dups)} duplicated (suite, task, seed) keys. "
            "A contrast over this selection would pool configurations."
        )
    per_suite = {}
    for s in SUITES:
        rs = [r for r in keep if r["suite_id"] == s]
        per_suite[s] = {
            "runs": len(rs),
            "tasks": len({r["task_id"] for r in rs}),
            "seeds": sorted({int(r["seed"]) for r in rs}),
        }
    census = {
        "arm_id": arm_id,
        "model_id": model_id,
        "grid_name": grid,
        "n_via_select_runs": len(runs),
        "census_before_filters": [
            {"pin16": k[0], "code12": k[1], "split": k[2], "dirty": k[3], "n": n}
            for k, n in sorted(before.items())
        ],
        "removed_by_filter": removed,
        "n_kept": len(keep),
        "model_pin_hash": pins[0],
        "code_version": codes[0],
        "per_suite": per_suite,
        "run_id_digest_sha256": digest_ids([r["run_id"] for r in keep]),
    }
    return keep, census


def arm_id_only_tell(con, *, suite: str, split: str) -> dict[str, Any]:
    """What selecting the comparator by `arm_id` alone would have pooled (any grid)."""
    rows = gate._rows(
        con,
        "SELECT count(*) AS n, count(DISTINCT model_pin_hash) AS pins FROM runs "
        f"WHERE arm_id = 'inquirer_prompted' AND status = 'ok' AND split = {gate._q(split)} "
        f"AND suite_id = {gate._q(suite)}",
    )
    return {"suite": suite, "split": split, "n_runs": rows[0]["n"], "n_pins": rows[0]["pins"]}


def scorer_of(con, ids: Sequence[str]) -> tuple[str, str]:
    rows = gate._rows(
        con,
        "SELECT scorer_hash, graph_version, count(DISTINCT run_id) AS n FROM scores "
        f"WHERE run_id IN {gate._in(ids)} GROUP BY 1, 2",
    )
    if len(rows) != 1 or int(rows[0]["n"]) != len(set(ids)):
        raise Refused(
            f"the population's score rows are not one (scorer_hash, graph_version) covering "
            f"every run: {[(str(r['scorer_hash'])[:16], r['graph_version'], r['n']) for r in rows]}"
            f" over {len(set(ids))} runs"
        )
    return str(rows[0]["scorer_hash"]), str(rows[0]["graph_version"])


def coverage_ladders(
    store: Path, runs: Sequence[Mapping[str, Any]], graph_version: str
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """suite -> run_id -> route-A ladder, and suite -> gold graphs. Refuses a silent skip."""
    ladders: dict[str, dict[str, Any]] = {}
    graphs_by_suite: dict[str, dict[str, Any]] = {}
    for s in SUITES:
        rs = [r for r in runs if r["suite_id"] == s]
        if not rs:
            continue
        graphs = load_graphs(s, graph_version)
        missing = sorted({r["task_id"] for r in rs} - set(graphs))
        if not graphs or missing:
            raise Refused(f"{s}: {len(graphs)} gold graphs, {len(missing)} tasks lack one")
        lad = sweep.load_coverage_ladders(store, [r["run_id"] for r in rs], graphs)
        if len(lad) != len(rs):
            raise Refused(f"{s}: {len(rs) - len(lad)} runs produced no coverage ladder")
        ladders[s], graphs_by_suite[s] = lad, graphs
    return ladders, graphs_by_suite


def only(ladders: Mapping[str, Any], runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ids = {r["run_id"] for r in runs}
    return {rid: lad for rid, lad in ladders.items() if rid in ids}


# ---------------------------------------------------------------------- instrument checks


def rung_check(
    ladders: Mapping[str, Any],
    stored: Mapping[str, Mapping[int, float]],
    graphs: Mapping[str, Any],
    fn: Callable[..., float],
) -> dict[str, int]:
    """Route A's turn-derived ladder against the stored `frontier_q#k`, every run, every k."""
    agree_ = differ = presence = absent_both = extra_stored = 0
    for rid, lad in ladders.items():
        g = graphs[lad.task_id]
        st = stored.get(rid, {})
        extra_stored += sum(1 for k in st if k > lad.n_asks or k < 0)
        for k in range(lad.n_asks + 1):
            mine = fn(lad.at(k), g)
            theirs = st.get(k)
            if math.isnan(mine) and theirs is None:
                absent_both += 1
            elif math.isnan(mine) or theirs is None:
                presence += 1
            elif abs(mine - float(theirs)) <= RUNG_TOL:
                agree_ += 1
            else:
                differ += 1
    return {
        "rungs_agree": agree_,
        "rungs_differ": differ,
        "rungs_presence_mismatch": presence,
        "rungs_absent_both": absent_both,
        "stored_rungs_outside_0_to_n_asks": extra_stored,
    }


# ----------------------------------------------------------------------------- estimation


def route_a(ck, ba, graphs, seeds, *, n_boot: int, seed: int) -> dict[str, Any]:
    r = symmetric_completed.seed_matched_symmetric(
        METRIC, ck, ba, graphs, seeds, seed=seed, n_boot=n_boot
    )
    return {k: r[k] for k in ("delta", "ci_lo", "ci_hi", "n", "n_pairs", "pairs_dropped")}


def route_b(con, ck_runs, ba_runs, scorer_hash, *, n_boot: int, seed: int) -> dict[str, Any]:
    mc = gate._matched_cost(
        con,
        ckpt=ck_runs,
        base=ba_runs,
        ck_keys=gate._by_key(ck_runs),
        ba_keys=gate._by_key(ba_runs),
        scorer_hash=scorer_hash,
        seed=seed,
        n_resamples=n_boot,
    )
    out = {}
    for s, v in mc["by_suite"].items():
        sym = v["symmetric"]
        out[s] = {
            "symmetric": {k: sym[k] for k in ("delta", "ci_lo", "ci_hi", "n_tasks")},
            "mean_k_common": sym["mean_k_common"],
            "asymmetric": {k: v[k] for k in ("delta", "ci_lo", "ci_hi", "n_tasks")},
            "cap8_coverage_delta": v["cap8_coverage_delta"],
            "trained_mean_n_asks": v["trained_mean_n_asks"],
            "baseline_mean_n_asks": v["baseline_mean_n_asks"],
        }
    return out


def _n(r: Mapping[str, Any]) -> Any:
    return r.get("n", r.get("n_tasks"))


def same(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    """Two readings of one cell agree: n, point to DELTA_TOL, both bounds to CI_TOL."""
    return (
        _n(a) == _n(b)
        and abs(a["delta"] - b["delta"]) <= DELTA_TOL
        and abs(a["ci_lo"] - b["ci_lo"]) <= CI_TOL
        and abs(a["ci_hi"] - b["ci_hi"]) <= CI_TOL
    )


def cross_check(ra: Mapping[str, Any], rb: Mapping[str, Any]) -> dict[str, Any]:
    """Route A against route B on one cell.

    The POINT and n must agree to DELTA_TOL: both routes read the same per-run rungs (checked
    rung by rung) and average the same pairs. The BOUNDS are compared but not required to agree
    to CI_TOL, and this is measured, not assumed: route A takes mean(a) - mean(b) per task and
    route B mean(a - b), which differ by ~1e-17, and on a lattice-valued cell many bootstrap
    replicates TIE with the point estimate, so that rounding flips `rep < theta_hat` for some of
    them, moves the BCa bias correction and shifts an endpoint by an order statistic. Measured on
    seedrep's published s1-vs-s2 cells: points equal to 2e-19, bounds apart by up to 5.3e-5.
    Route B is therefore the interval of record (it is the code every locked published figure
    came from); a bound gap above ROUTE_A_BOUND_TOL would be a real disagreement and fails.
    """
    gap = max(abs(ra["ci_lo"] - rb["ci_lo"]), abs(ra["ci_hi"] - rb["ci_hi"]))
    point_ok = _n(ra) == _n(rb) and abs(ra["delta"] - rb["delta"]) <= DELTA_TOL
    return {
        "point_agrees": point_ok,
        "max_bound_gap": gap,
        "ok": point_ok and gap <= ROUTE_A_BOUND_TOL,
    }


def mean_k(ck, ba, seeds) -> dict[str, float]:
    """Mean own asks of each side, and mean shared k, over the seed-matched pairs."""
    a = {(x.suite_id, x.task_id, int(seeds[r])): x for r, x in ck.items()}
    b = {(x.suite_id, x.task_id, int(seeds[r])): x for r, x in ba.items()}
    keys = sorted(set(a) & set(b))
    if not keys:
        return {"k_a": float("nan"), "k_b": float("nan"), "k_common": float("nan")}
    return {
        "k_a": sum(a[k].n_asks for k in keys) / len(keys),
        "k_b": sum(b[k].n_asks for k in keys) / len(keys),
        "k_common": sum(min(a[k].n_asks, b[k].n_asks) for k in keys) / len(keys),
    }


def classify(readings: Sequence[Mapping[str, Any]]) -> str:
    sides = set()
    for r in readings:
        if r["ci_lo"] > 0:
            sides.add("pos")
        elif r["ci_hi"] < 0:
            sides.add("neg")
        else:
            sides.add("zero")
    if sides == {"pos"}:
        return "excludes_zero_positive"
    if sides == {"neg"}:
        return "excludes_zero_negative"
    if sides == {"zero"}:
        return "includes_zero"
    return "undecided"


def read_pair(con, ck_runs, ba_runs, scorer_hash) -> dict[str, dict[str, Any]]:
    """Every suite of one (A minus B) contrast under the resample rule, off route B."""
    res = {
        (nb, sd): route_b(con, ck_runs, ba_runs, scorer_hash, n_boot=nb, seed=sd)
        for nb, sd in MAIN_READINGS
    }
    near = {
        s: any(
            min(abs(res[k][s]["symmetric"]["ci_lo"]), abs(res[k][s]["symmetric"]["ci_hi"]))
            < NEAR_ZERO
            for k in MAIN_READINGS
        )
        for s in SUITES
    }
    if any(near.values()):
        for nb, sd in STABILITY_READINGS:
            res[(nb, sd)] = route_b(con, ck_runs, ba_runs, scorer_hash, n_boot=nb, seed=sd)
    cells = {}
    for s in SUITES:
        keys = list(MAIN_READINGS) + (list(STABILITY_READINGS) if near[s] else [])
        readings = [{"n_boot": nb, "seed": sd, **res[(nb, sd)][s]["symmetric"]} for nb, sd in keys]
        ctx = res[(10000, 0)][s]
        signs = {b: sorted({(r[b] > 0) - (r[b] < 0) for r in readings}) for b in ("ci_lo", "ci_hi")}
        cells[s] = {
            "readings": readings,
            "near_zero": near[s],
            "bound_signs_over_all_readings": signs,
            "sign_stable": all(len(v) == 1 for v in signs.values()),
            "verdict": classify(readings),
            "mean_k_common": ctx["mean_k_common"],
            "context_10k": {
                "asymmetric_gate_rule": ctx["asymmetric"],
                "cap8_unmatched_delta": ctx["cap8_coverage_delta"],
            },
        }
    return cells


def r4(x: float) -> float:
    return float(f"{x:.4f}")


def fmt(r: Mapping[str, Any]) -> str:
    return f"{r['delta']:+.4f} [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]"


# ------------------------------------------------------------------------ levels and basis


def levels(runs: Sequence[Mapping[str, Any]], cov: Mapping[str, float]) -> dict[str, Any]:
    """Per suite: runs, mean asks and mean evidence_coverage over runs. Every task here carries
    exactly two runs (checked by `select_arm`'s key uniqueness and 400 runs / 200 tasks), so the
    run mean equals the seed-folded task mean."""
    out = {}
    for s in SUITES:
        rs = [r for r in runs if r["suite_id"] == s]
        vals = [cov[r["run_id"]] for r in rs if r["run_id"] in cov]
        out[s] = {
            "n_runs": len(rs),
            "n_with_coverage": len(vals),
            "mean_asks": sum(int(r["n_asks"] or 0) for r in rs) / len(rs) if rs else float("nan"),
            "mean_coverage": sum(vals) / len(vals) if vals else float("nan"),
        }
    return out


def basis_overlap(
    con, base_runs: Sequence[Mapping[str, Any]], scorer_hash: str, *, root: Path
) -> dict[str, Any]:
    """Is this table's comparator the SAME runs as the paper's Table 1 comparator?

    Table 1 reads `artifacts/completed_cohort_20260922` (its prompted run-id lists). Reports, per
    suite, run-id overlap, (task, seed) key overlap and task overlap against this population's
    base; each side's pin and code version; and, for Table 1's comparator runs that also sit in
    the store read here, whether `n_asks` and `evidence_coverage` agree across the two scorer
    hash labels.
    """
    store = root / "scores_parquet"
    ccon = gate._con(store)
    c_hash, _ = scorer_of(
        ccon, [i for s in SUITES for i in read_ids(root / "cohort" / f"run_ids.prompted.{s}.txt")]
    )
    c_cov = gate._metric_by_run(ccon, METRIC, scorer_hash=c_hash)
    shared_cov = gate._metric_by_run(con, METRIC, scorer_hash=scorer_hash)
    out: dict[str, Any] = {"table1_store": rel(store), "table1_scorer_hash": c_hash, "suites": {}}
    for s in SUITES:
        t1_ids = read_ids(root / "cohort" / f"run_ids.prompted.{s}.txt")
        t1 = _meta(ccon, t1_ids)
        here = [r for r in base_runs if r["suite_id"] == s]
        here_meta = _meta(con, [r["run_id"] for r in here])
        in_shared = _meta(con, t1_ids)
        agree_asks = sum(
            1 for i in in_shared if int(in_shared[i]["n_asks"] or 0) == int(t1[i]["n_asks"] or 0)
        )
        both_cov = [i for i in in_shared if i in shared_cov and i in c_cov]
        max_cov_gap = max((abs(shared_cov[i] - c_cov[i]) for i in both_cov), default=float("nan"))
        out["suites"][s] = {
            "n_table1_comparator_runs": len(t1_ids),
            "n_this_base_runs": len(here),
            "run_id_overlap": len({r["run_id"] for r in here} & set(t1_ids)),
            "task_seed_key_overlap": len(
                {(r["task_id"], int(r["seed"])) for r in here}
                & {(m["task_id"], int(m["seed"])) for m in t1.values()}
            ),
            "task_overlap": len({r["task_id"] for r in here} & {m["task_id"] for m in t1.values()}),
            "table1_pins": sorted({str(m["model_pin_hash"])[:16] for m in t1.values()}),
            "table1_code_versions": sorted({str(m["code_version"])[:12] for m in t1.values()}),
            "this_pins": sorted({str(m["model_pin_hash"])[:16] for m in here_meta.values()}),
            "this_code_versions": sorted({str(m["code_version"])[:12] for m in here_meta.values()}),
            "table1_runs_present_in_this_store": len(in_shared),
            "of_those_n_asks_equal": agree_asks,
            "of_those_coverage_compared": len(both_cov),
            "of_those_max_abs_coverage_gap_across_hash_labels": max_cov_gap,
            "table1_comparator_levels_at_its_own_hash": {
                "mean_asks": sum(int(m["n_asks"] or 0) for m in t1.values()) / len(t1),
                "mean_coverage": sum(c_cov[i] for i in t1_ids) / len(t1_ids),
            },
        }
    return out


# ---------------------------------------------------------------------------------- locks


def lock_b(*, root: Path, published: Path, graph_version: str) -> dict[str, Any]:
    """The paper's printed selected-arm cells, on their own store, at 10,000 resamples seed 0."""
    store = root / "scores_parquet"
    con = gate._con(store)
    ck, ck_c = select_arm(con, "inquirer_trained", SELECTED, split="test", code_prefix="3ae099d0")
    ba, ba_c = select_arm(
        con, "inquirer_prompted", "qwen3-8b-base", split="test", code_prefix="3ae099d0"
    )
    lists = {
        "trained": sorted(
            i for s in SUITES for i in read_ids(root / "cohort" / f"run_ids.trained.{s}.txt")
        ),
        "prompted": sorted(
            i for s in SUITES for i in read_ids(root / "cohort" / f"run_ids.prompted.{s}.txt")
        ),
    }
    selection_equals_cohort_lists = (
        sorted(r["run_id"] for r in ck) == lists["trained"]
        and sorted(r["run_id"] for r in ba) == lists["prompted"]
    )
    sh, gv = scorer_of(con, [r["run_id"] for r in ck + ba])
    seeds = {r["run_id"]: int(r["seed"]) for r in ck + ba}
    lad, graphs = coverage_ladders(store, ck + ba, graph_version)
    rb = route_b(con, ck, ba, sh, n_boot=10000, seed=0)
    pub = json.loads(published.read_text())["complete"]["by_suite"]
    cells = {}
    for s in SUITES:
        ra = route_a(only(lad[s], ck), only(lad[s], ba), graphs[s], seeds, n_boot=10000, seed=0)
        target = pub[s]["symmetric"]
        cells[s] = {
            "published": {k: target[k] for k in ("delta", "ci_lo", "ci_hi", "n_tasks")},
            "route_a": ra,
            "route_b": rb[s]["symmetric"],
            "route_a_equals_published_exactly": same(ra, target),
            "route_b_equals_published": same(rb[s]["symmetric"], target),
            "route_a_vs_route_b": cross_check(ra, rb[s]["symmetric"]),
        }
        cells[s]["verdict"] = (
            "LOCKED"
            if cells[s]["route_b_equals_published"] and cells[s]["route_a_vs_route_b"]["ok"]
            else "FAILED"
        )
    return {
        "store": rel(store),
        "published": rel(published),
        "scorer_hash": sh,
        "graph_version": gv,
        "selection": {"trained": ck_c, "prompted": ba_c},
        "selection_equals_cohort_run_id_lists": selection_equals_cohort_lists,
        "cells": cells,
    }


def lock_c_reference(verdicts_dir: Path) -> dict[str, dict[str, Any]]:
    """The gate's own ASYMMETRIC verdicts for three of the arms, from a different scoring pass."""
    out = {}
    for label in ("control", "rater", "reaches"):
        p = verdicts_dir / f"{ARMS[label][1]}.matched_cost.json"
        d = json.loads(p.read_text())
        boot = d.get("bootstrap") or {}
        for s, v in d["matched_cost"]["by_suite"].items():
            out[f"{label}::{s}"] = {
                "delta": v["delta"],
                "ci_lo": v["ci_lo"],
                "ci_hi": v["ci_hi"],
                "n_tasks": v["n_tasks"],
                "scorer_hash": d["selection"]["scorer_hash"],
                "bootstrap": boot,
                "file": rel(p),
            }
    return out


def lock_d(*, root: Path, graph_version: str) -> dict[str, Any]:
    """The arm-vs-arm symmetric path against seedrep's published s1-vs-s2 cells."""
    store = root / "scores_parquet"
    con = gate._con(store)
    a, a_c = select_arm(con, "inquirer_trained", SEEDREP_A, split="test", code_prefix="3ae099d0")
    b, b_c = select_arm(con, "inquirer_trained", SEEDREP_B, split="test", code_prefix="3ae099d0")
    lists_equal = sorted(r["run_id"] for r in a) == sorted(
        i for s in SUITES for i in read_ids(root / "run_ids" / f"run_ids.{SEEDREP_A}.{s}.txt")
    ) and sorted(r["run_id"] for r in b) == sorted(
        i for s in SUITES for i in read_ids(root / "run_ids" / f"run_ids.{SEEDREP_B}.{s}.txt")
    )
    sh, gv = scorer_of(con, [r["run_id"] for r in a + b])
    seeds = {r["run_id"]: int(r["seed"]) for r in a + b}
    lad, graphs = coverage_ladders(store, a + b, graph_version)
    name = f"{SEEDREP_A} vs {SEEDREP_B}"
    ladder_pub = json.loads((root / "resample_ladder.json").read_text())["arm_vs_arm"][name]
    contrast_pub = json.loads((root / "contrast.json").read_text())["arm_vs_arm"][name]["by_suite"]
    readings = ((10000, 0), (50000, 0), (50000, 1), (50000, 2))
    rb = {(nb, sd): route_b(con, a, b, sh, n_boot=nb, seed=sd) for nb, sd in readings}
    cells = {}
    for s in SUITES:
        la, lb = only(lad[s], a), only(lad[s], b)
        rows = []
        for pub in ladder_pub[s]:
            nb, sd = int(pub["n_resamples"]), int(pub["seed"])
            if (nb, sd) not in readings:
                continue
            got_b = rb[(nb, sd)][s]["symmetric"]
            ra = route_a(la, lb, graphs[s], seeds, n_boot=nb, seed=sd)
            want = {**pub, "n_tasks": got_b["n_tasks"]}  # the ladder file carries no n
            rows.append(
                {
                    "n_boot": nb,
                    "seed": sd,
                    "published": pub,
                    "route_b": got_b,
                    "route_b_matches": same(got_b, want),
                    "route_a": ra,
                    "route_a_vs_route_b": cross_check(ra, got_b),
                }
            )
        b10 = rb[(10000, 0)][s]["symmetric"]
        cells[s] = {
            "readings": rows,
            "route_b_10k_equals_contrast_json": same(b10, contrast_pub[s]),
        }
        cells[s]["verdict"] = (
            "LOCKED"
            if len(rows) == len(readings)
            and all(r["route_b_matches"] and r["route_a_vs_route_b"]["ok"] for r in rows)
            and cells[s]["route_b_10k_equals_contrast_json"]
            else "FAILED"
        )
    return {
        "store": rel(store),
        "scorer_hash": sh,
        "graph_version": gv,
        "pair": name,
        "selection": {"a": a_c, "b": b_c},
        "selection_equals_run_id_lists": lists_equal,
        "cells": cells,
    }


# ----------------------------------------------------------------------------------- main


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store", type=Path, default=REPO / "scores" / "parquet")
    ap.add_argument("--split", default="test")
    ap.add_argument("--code-prefix", default="107ef2221a55")
    ap.add_argument("--budget-cap", type=int, default=8)
    ap.add_argument("--graph-version", default="v1")
    ap.add_argument("--runs-root", type=Path, default=REPO / "runs")
    ap.add_argument(
        "--lock-b-root", type=Path, default=REPO / "artifacts" / "completed_cohort_20260922"
    )
    ap.add_argument(
        "--lock-b-published",
        type=Path,
        default=REPO / "artifacts" / "baseline_completion_20260920" / "three.json",
    )
    ap.add_argument(
        "--lock-c-verdicts",
        type=Path,
        default=REPO / "artifacts" / "label_ordering_test_20260918" / "verdicts",
    )
    ap.add_argument(
        "--lock-d-root", type=Path, default=REPO / "artifacts" / "seedrep_gate_20260919"
    )
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    rec: dict[str, Any] = {"argv": list(argv if argv is not None else sys.argv[1:])}

    store_files = ("runs", "scores", "turns", "calls")
    sha_start = {f: sha256_file(a.store / f"{f}.parquet") for f in store_files}
    rec["provenance"] = {
        "store": rel(a.store),
        "store_sha256_at_start": sha_start,
        "reader": reader_identity(),
        "python": platform.python_version(),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gold_graph_sha256": {
            s: sha256_file(REPO / "data" / "gold" / "graphs" / s / f"{a.graph_version}.jsonl")
            for s in SUITES
        },
    }
    rd = rec["provenance"]["reader"]
    print(
        f"store {rel(a.store)}  head {rd['head'][:12]}  "
        f"reader paths clean against head: {rd['reader_paths_clean_against_head']}"
    )

    # ---- population
    con = gate._con(a.store)
    runs: dict[str, list[dict]] = {}
    pop: dict[str, Any] = {}
    for label, (arm_id, model_id) in ARMS.items():
        runs[label], pop[label] = select_arm(
            con,
            arm_id,
            model_id,
            split=a.split,
            code_prefix=a.code_prefix,
            budget_cap=a.budget_cap,
        )
    pop["_tell_arm_id_only"] = [arm_id_only_tell(con, suite=s, split=a.split) for s in SUITES]
    rec["population"] = pop
    print(
        f"\n== population (grid {GRID}, split {a.split}, code {a.code_prefix}*, "
        f"budget_cap {a.budget_cap})"
    )
    for label in ARMS:
        c = pop[label]
        ps = " ".join(
            f"{s}={v['runs']}r/{v['tasks']}t/seeds{v['seeds']}" for s, v in c["per_suite"].items()
        )
        rm = {k: v for k, v in c["removed_by_filter"].items() if v}
        print(
            f"  {label:12s} {c['model_id']:36s} pin {c['model_pin_hash'][:16]} "
            f"code {c['code_version'][:12]} select_runs={c['n_via_select_runs']} "
            f"kept={c['n_kept']} removed={rm}"
        )
        print(f"  {'':12s} {ps}")
    for t in pop["_tell_arm_id_only"]:
        print(
            f"  tell: arm_id='inquirer_prompted' alone, {t['suite']} {t['split']}, any grid: "
            f"{t['n_runs']} runs over {t['n_pins']} model pins"
        )

    all_runs = [r for label in ARMS for r in runs[label]]
    all_ids = [r["run_id"] for r in all_runs]
    scorer_hash, gv_scores = scorer_of(con, all_ids)
    if gv_scores != a.graph_version:
        raise Refused(f"scores carry graph_version {gv_scores}, expected {a.graph_version}")
    rec["provenance"]["scorer_hash"] = scorer_hash
    rec["provenance"]["graph_version"] = gv_scores
    print(f"\n  scorer_hash {scorer_hash}  graph_version {gv_scores}  runs {len(all_ids)}")

    run_dir = a.out_dir / "run_ids"
    run_dir.mkdir(parents=True, exist_ok=True)
    for label in ARMS:
        path = run_dir / f"{ARMS[label][1]}.txt"
        path.write_text("\n".join(sorted(r["run_id"] for r in runs[label])) + "\n")
        pop[label]["run_id_list"] = rel(path)

    # ---- instrument checks
    checks: dict[str, Any] = {}
    pr = stop_lib.population_report(con, run_ids=all_ids, scorer_hash=scorer_hash)
    checks["population_report"] = {k: v for k, v in pr.items() if not isinstance(v, list)}
    tnd = stop_lib.verify_turns_not_dropped(
        a.runs_root, {r["run_id"]: int(r["n_turns"] or 0) for r in all_runs}
    )
    checks["turns_not_dropped"] = {k: v for k, v in tnd.items() if not isinstance(v, list)}
    vl = stop_lib.verify_ladder_exists_for_asking_runs(con, all_runs, scorer_hash=scorer_hash)
    checks["ladder_exists_for_asking_runs"] = {k: v for k, v in vl.items() if k != "bad_run_ids"}
    stored = gate._coverage_ladder(con, all_runs, scorer_hash=scorer_hash)
    cov = gate._metric_by_run(con, METRIC, scorer_hash=scorer_hash)
    n_asks = {r["run_id"]: int(r["n_asks"] or 0) for r in all_runs}
    checks["terminal_rung_equals_evidence_coverage_checked"] = (
        gate._check_ladder_is_the_coverage_column(all_ids, stored, cov, n_asks)
    )
    seeds = {r["run_id"]: int(r["seed"]) for r in all_runs}

    with coverage_registered() as cov_fn:
        lad, graphs = coverage_ladders(a.store, all_runs, a.graph_version)
        checks["rung_by_rung_route_a_vs_stored_frontier_q"] = {
            s: rung_check(lad[s], stored, graphs[s], cov_fn) for s in SUITES
        }
        rec["instrument_checks"] = checks
        print("\n== instrument checks")
        for k, v in checks.items():
            print(f"  {k}: {v}")
        rungs = checks["rung_by_rung_route_a_vs_stored_frontier_q"].values()
        if (
            pr["n_scored"] != len(set(all_ids))
            or tnd["n_bad"] != 0
            or vl["n_bad"] != 0
            or any(v["rungs_differ"] or v["rungs_presence_mismatch"] for v in rungs)
        ):
            raise Refused(f"instrument check failed: {checks}")

        def L(label: str, s: str) -> dict[str, Any]:
            return only(lad[s], runs[label])

        # ---- route B at the lock setting, once per arm (also feeds lock c)
        rb10 = {
            label: route_b(con, runs[label], runs[BASE], scorer_hash, n_boot=10000, seed=0)
            for label in TRAINED
        }
        rb1 = {
            label: route_b(con, runs[label], runs[BASE], scorer_hash, n_boot=1000, seed=0)
            for label in ("control", "rater", "reaches")
        }

        # ---- (a): WITHDRAWN RECORD, printed for the audit trail, not a gate (module docstring)
        lock_a: dict[str, Any] = {}
        print(
            "\n== (a) WITHDRAWN RECORD, printed for the audit trail, not a gate: "
            "artifacts/label_variants_heldout_20260919 against this reader, 10,000 resamples seed 0"
        )
        for label in TRAINED:
            for s in SUITES:
                ra = route_a(L(label, s), L(BASE, s), graphs[s], seeds, n_boot=10000, seed=0)
                b = rb10[label][s]["symmetric"]
                want = RECORD_CELLS[(label, s)]
                got = (r4(b["delta"]), r4(b["ci_lo"]), r4(b["ci_hi"]))
                key = f"{label}::{s}"
                lock_a[key] = {
                    "record_4dp": list(want),
                    "route_b": b,
                    "route_a": ra,
                    "route_a_vs_route_b": cross_check(ra, b),
                    "reproduced_4dp": list(got),
                    "delta_minus_record": b["delta"] - want[0],
                    "verdict": "MATCHES_RECORD" if got == want else "DIFFERS_FROM_RECORD",
                    "gating": False,
                }
                print(
                    f"  {label:12s} {s:10s} withdrawn record {want[0]:+.4f} [{want[1]:+.4f}, "
                    f"{want[2]:+.4f}]  this reader {fmt(b)} n={b['n_tasks']}  route A point "
                    f"agrees={lock_a[key]['route_a_vs_route_b']['point_agrees']}  "
                    f"{lock_a[key]['verdict']}"
                )
        rec["withdrawn_record_audit"] = {
            "source": "artifacts/label_variants_heldout_20260919/RESULT.md",
            "status": "WITHDRAWN 2026-09-23; printed for the audit trail, not a gate",
            "cells": lock_a,
        }

        # ---- LOCK (b)
        lb = lock_b(root=a.lock_b_root, published=a.lock_b_published, graph_version=a.graph_version)
        rec["lock_b"] = lb
        print(
            f"\n== LOCK (b): the paper's printed selected-arm cells ({lb['published']}), "
            f"store {lb['store']}, scorer {lb['scorer_hash'][:16]}, 10,000 resamples seed 0; "
            f"selection == cohort run-id lists: {lb['selection_equals_cohort_run_id_lists']}"
        )
        for s, c in lb["cells"].items():
            p, b = c["published"], c["route_b"]
            print(
                f"  {s:10s} published {p['delta']!r} [{p['ci_lo']!r}, {p['ci_hi']!r}] "
                f"n={p['n_tasks']}"
            )
            print(
                f"  {'':10s} route B   {b['delta']!r} [{b['ci_lo']!r}, {b['ci_hi']!r}] "
                f"n={b['n_tasks']}  equal={c['route_b_equals_published']}  route A equal="
                f"{c['route_a_equals_published_exactly']}  {c['verdict']}"
            )

        # ---- LOCK (c)
        ref = lock_c_reference(a.lock_c_verdicts)
        lc = {}
        for key, want in sorted(ref.items()):
            label, s = key.split("::")
            got = rb1[label][s]["asymmetric"]
            lc[key] = {"reference": want, "here": got, "match": same(got, want)}
        rec["lock_c_cross_store_asymmetric"] = lc
        n_c = sum(v["match"] for v in lc.values())
        any_ref = next(iter(ref.values()))
        print(
            f"\n== LOCK (c): gate asymmetric verdicts of another scoring pass (scorer "
            f"{any_ref['scorer_hash'][:16]}, {any_ref['bootstrap']}) on this population, "
            f"route B 1,000 resamples seed 0: {n_c} of {len(lc)} reproduce exactly"
        )
        for key, v in lc.items():
            print(
                f"  {key:22s} reference {fmt(v['reference'])}  here {fmt(v['here'])}  "
                f"{'match' if v['match'] else 'DIFFERS'}"
            )

        # ---- LOCK (d)
        ld = lock_d(root=a.lock_d_root, graph_version=a.graph_version)
        rec["lock_d"] = ld
        print(
            f"\n== LOCK (d): arm-vs-arm path, {ld['pair']}, store {ld['store']}, scorer "
            f"{ld['scorer_hash'][:16]}; selection == run-id lists: "
            f"{ld['selection_equals_run_id_lists']}"
        )
        for s, c in ld["cells"].items():
            reads = "  ".join(
                f"{r['n_boot'] // 1000}k/s{r['seed']} B{'=' if r['route_b_matches'] else 'X'}"
                f" A-point{'=' if r['route_a_vs_route_b']['point_agrees'] else 'X'}"
                f" A-gap {r['route_a_vs_route_b']['max_bound_gap']:.1e}"
                for r in c["readings"]
            )
            print(
                f"  {s:10s} {reads}  contrast.json 10k B equal="
                f"{c['route_b_10k_equals_contrast_json']}  {c['verdict']}"
            )

        failed = [f"(b) {s}" for s, c in lb["cells"].items() if c["verdict"] != "LOCKED"]
        failed += [f"(c) {k}" for k, v in lc.items() if not v["match"]]
        if len(lc) != 9:
            failed.append(f"(c) expected 9 reference cells, read {len(lc)}")
        failed += [f"(d) {s}" for s, c in ld["cells"].items() if c["verdict"] != "LOCKED"]
        rec["lock_summary"] = {
            "gating_locks": ["b", "c", "d"],
            "withdrawn_record_cells_matching": sum(
                v["verdict"] == "MATCHES_RECORD" for v in lock_a.values()
            ),
            "a_route_a_points_agree": sum(
                v["route_a_vs_route_b"]["point_agrees"] for v in lock_a.values()
            ),
            "a_route_a_max_bound_gap": max(
                v["route_a_vs_route_b"]["max_bound_gap"] for v in lock_a.values()
            ),
            "b_cells_locked": sum(c["verdict"] == "LOCKED" for c in lb["cells"].values()),
            "c_cells_matched": n_c,
            "d_cells_locked": sum(c["verdict"] == "LOCKED" for c in ld["cells"].values()),
            "failed": failed,
        }
        print(f"\n== lock summary: {rec['lock_summary']}")
        if failed:
            rec["stopped"] = (
                f"lock failed: {failed}. By the lock rule no four-arm table and no arm-vs-arm "
                "contrast is computed."
            )
            finish(rec, a, store_files, sha_start, t0)
            print(f"\nSTOPPED: {rec['stopped']}")
            return 2

        # ---- post-lock: levels, the basis question, then the contrasts
        rec["levels"] = {label: levels(runs[label], cov) for label in ARMS}
        print(
            f"\n== per-arm levels over runs (= seed-folded task means), scorer {scorer_hash[:16]}"
        )
        for label in ARMS:
            print(
                f"  {label:12s} "
                + "  ".join(
                    f"{s} asks {v['mean_asks']:.4f} cov {v['mean_coverage']:.4f} "
                    f"(n={v['n_runs']}/{v['n_with_coverage']})"
                    for s, v in rec["levels"][label].items()
                )
            )
        bo = basis_overlap(con, runs[BASE], scorer_hash, root=a.lock_b_root)
        rec["basis_vs_table1_comparator"] = bo
        print(
            f"\n== basis: this base against Table 1's comparator ({bo['table1_store']}, "
            f"scorer {bo['table1_scorer_hash'][:16]})"
        )
        for s, v in bo["suites"].items():
            print(
                f"  {s:10s} run_id overlap {v['run_id_overlap']}/{v['n_this_base_runs']} vs "
                f"{v['n_table1_comparator_runs']}  (task, seed) overlap "
                f"{v['task_seed_key_overlap']}  task overlap {v['task_overlap']}  | this pin "
                f"{v['this_pins']} code {v['this_code_versions']}  Table 1 pin {v['table1_pins']} "
                f"code {v['table1_code_versions']}"
            )
            lv = v["table1_comparator_levels_at_its_own_hash"]
            print(
                f"  {'':10s} Table 1 comparator runs also in {rel(a.store)}: "
                f"{v['table1_runs_present_in_this_store']}; n_asks equal "
                f"{v['of_those_n_asks_equal']}; coverage compared "
                f"{v['of_those_coverage_compared']}, max |gap| across hash labels "
                f"{v['of_those_max_abs_coverage_gap_across_hash_labels']:.3g}; Table 1 "
                f"comparator levels asks {lv['mean_asks']:.4f} cov {lv['mean_coverage']:.4f}"
            )

        # ---- each arm against the base, then arm versus arm, under the resample rule,
        # intervals off route B, route A as the point-level cross-check at 10k seed 0
        def contrast(x: str, y: str) -> dict[str, dict[str, Any]]:
            cells = read_pair(con, runs[x], runs[y], scorer_hash)
            for s, cell in cells.items():
                ra = route_a(L(x, s), L(y, s), graphs[s], seeds, n_boot=10000, seed=0)
                b10 = next(q for q in cell["readings"] if (q["n_boot"], q["seed"]) == (10000, 0))
                cell["route_a_vs_route_b_10k"] = cross_check(ra, b10)
                cell["mean_k"] = mean_k(L(x, s), L(y, s), seeds)
                print_cell(f"{x}-{y}", s, cell)
            return cells

        print("\n== each arm minus the prompted base, symmetric matched cost (route B)")
        rec["vs_base"] = {label: contrast(label, BASE) for label in TRAINED}
        print("\n== arm minus arm, symmetric matched cost, paired (suite, task, seed) (route B)")
        rec["arm_vs_arm"] = {
            f"{x}-{y}": contrast(x, y) for x, y in itertools.combinations(TRAINED, 2)
        }

    av = {f"{n}::{s}": c["verdict"] for n, arm in rec["arm_vs_arm"].items() for s, c in arm.items()}
    rec["summary"] = {
        "vs_base_verdicts": dict(
            Counter(c["verdict"] for arm in rec["vs_base"].values() for c in arm.values())
        ),
        "arm_vs_arm_verdicts": dict(Counter(av.values())),
        "arm_vs_arm_separated": sorted(k for k, v in av.items() if v.startswith("excludes")),
        "arm_vs_arm_undecided": sorted(k for k, v in av.items() if v == "undecided"),
        "route_a_cross_check_ok_everywhere": all(
            c["route_a_vs_route_b_10k"]["ok"]
            for part in ("vs_base", "arm_vs_arm")
            for arm in rec[part].values()
            for c in arm.values()
        ),
    }
    print("\n== summary")
    for k, v in rec["summary"].items():
        print(f"  {k}: {v}")
    return finish(rec, a, store_files, sha_start, t0)


def print_cell(name: str, s: str, cell: Mapping[str, Any]) -> None:
    rd = {(x["n_boot"], x["seed"]): x for x in cell["readings"]}
    k = cell["mean_k"]
    print(
        f"  {name:22s} {s:10s} 1k {fmt(rd[(1000, 0)])}  10k {fmt(rd[(10000, 0)])}  "
        f"50k {fmt(rd[(50000, 0)])}  n={rd[(10000, 0)]['n_tasks']}  "
        f"k {k['k_a']:.2f}/{k['k_b']:.2f}->{k['k_common']:.2f}  {cell['verdict']}  "
        f"routeA ok={cell['route_a_vs_route_b_10k']['ok']}"
    )
    if cell["near_zero"]:
        print(
            f"  {'':33s} NEAR ZERO -> 50k s1 {fmt(rd[(50000, 1)])}  50k s2 {fmt(rd[(50000, 2)])}"
            f"  bound signs over all readings {cell['bound_signs_over_all_readings']}"
            f"  sign stable={cell['sign_stable']}"
        )


def finish(rec: dict[str, Any], a: argparse.Namespace, files, sha_start, t0: float) -> int:
    sha_end = {f: sha256_file(a.store / f"{f}.parquet") for f in files}
    rec["provenance"]["store_sha256_at_end"] = sha_end
    rec["provenance"]["store_unchanged_during_run"] = sha_end == sha_start
    rec["provenance"]["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    rec["provenance"]["elapsed_s"] = round(time.time() - t0, 1)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / "numbers.json").write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n")
    print(f"\n  store unchanged during run: {rec['provenance']['store_unchanged_during_run']}")
    print(f"  wrote {rel(a.out_dir / 'numbers.json')}  ({rec['provenance']['elapsed_s']} s)")
    if not rec["provenance"]["store_unchanged_during_run"]:
        print("WARNING: a store file changed during the run; the numbers are not attributable.")
        return 3
    return 2 if rec.get("stopped") else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Refused as e:
        print(f"\nREFUSED: {e}")
        raise SystemExit(4)
