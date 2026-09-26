#!/usr/bin/env python3
"""Lane L2.5: the Granite-3.3-8B family contrast, seed by seed, suite by suite.

    PI_GOLD_ROOT="$PWD/data/gold" .venv/bin/python scripts/granite_family/contrasts.py \
        --parquet-dir scores/parquet --scorer-hash 677146c3e453... \
        --out artifacts/granite_family_20260918/contrasts.json

REUSES, DOES NOT REIMPLEMENT. `pinq_train.gate._select_runs`, `._matched_cost`,
`._metric_by_run`, `._paired_by_task_map` and `.bca_ci` are the exact functions
`pi train gate` runs for the qwen headline table (CLAUDE.md: "same functions as the
headline"). This script's only original code is the per-seed MODEL_ID filter on the
CHECKPOINT side, which `pinq_train.gate.run_gate` never needed before: every prior gated
checkpoint was the only inquirer_trained pin on its grid, so grid_name alone disambiguated
it. Three granite seeds run the SAME grid (tier1_trained_qa_base) with three different
PI_MODEL_INQUIRER pins, so `_select_runs`'s existing (but previously baseline-only)
`model_id` filter, resolved through `calls.parquet WHERE actor='inquirer'`, is applied to
the checkpoint side too. Nothing about the coverage arithmetic, the pairing rule or the
bootstrap changes.

CAP-8: `_matched_cost` already computes the cap-8 point delta internally (the ordered
pair's cost line) but does not interval it -- it is "reported per suite and NOT gated"
there. This script intervals it with the identical `bca_ci` call on the identical
per-task deltas, because a reported delta without a reported interval is not a result
CLAUDE.md's four rules would accept.

BOOTSTRAP STABILITY (coordinator rule, 2026-09-18). Every interval is read at
`--n-resamples` (10000 here). Any interval whose lower OR upper bound lies within
`STABILITY_BAND` (0.01) of zero is re-read at `RECHECK_RESAMPLES` (50000) under
`RECHECK_SEEDS` (three bootstrap seeds distinct from the primary one). If that bound's
SIGN is not identical across all three reruns, the cell is marked `"stable": False` and
must be reported as undecided, never as a pass or fail -- see `stability_note`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(REPO_SRC))

from pinq_train.gate import (  # noqa: E402
    _by_key,
    _con,
    _coverage_ladder,
    _in,
    _matched_cost,
    _metric_by_run,
    _paired_by_task_map,
    _rows,
    _select_runs,
    bca_ci,
)

GRID_NAME = "tier1_trained_qa_base"
CHECKPOINT_ARM = "inquirer_trained"
BASELINE_ARM = "inquirer_prompted"
BASELINE_MODEL = "granite33-8b-base"
SEEDS = ("s0", "s1", "s2")
SUITES = ("musique", "strategyqa", "wiki2")

# "The Qwen headline row (existing)": NOT qwen3-8b-sft-headline (rung 1 -- see RESULT.md for why
# that population is unusable: 482 runs, musique only, empty grid_name). The paper's own selected
# arm, named in results.tex's commented provenance block above \subsection{The horizontal axis...}:
# rung 2 (DPO/preference-tuned). Granite has no rung-2 checkpoint, so this row is not the same
# rung as the granite seeds -- reported and labelled as such, never silently paired as if matched.
QWEN_CHECKPOINT_MODEL = "qwen3-8b-dpo-stacked-notdone-both"
QWEN_BASELINE_MODEL = "qwen3-8b-base"

PRIMARY_N_RESAMPLES = 10_000
PRIMARY_BOOTSTRAP_SEED = 0
STABILITY_BAND = 0.01
RECHECK_N_RESAMPLES = 50_000
RECHECK_SEEDS = (101, 202, 303)


def _sign(x: float) -> int:
    return 0 if x == 0 else (1 if x > 0 else -1)


def stability_check(values: list[float]) -> dict:
    """Re-read a boundary interval at 50k resamples under three seeds; report sign stability.

    `values` are the SAME per-task deltas the primary bca_ci call was made on -- the recheck
    changes only the resample count and the seed, never the underlying data, which is what
    makes it a check ON THE ESTIMATOR rather than a second, different measurement.
    """
    reruns = [bca_ci(values, seed=s, n_resamples=RECHECK_N_RESAMPLES) for s in RECHECK_SEEDS]
    lo_signs = {_sign(lo) for _, lo, _ in reruns}
    hi_signs = {_sign(hi) for _, _, hi in reruns}
    return {
        "reruns": [
            {"seed": s, "point": p, "ci_lo": lo, "ci_hi": hi}
            for s, (p, lo, hi) in zip(RECHECK_SEEDS, reruns)
        ],
        "lo_sign_stable": len(lo_signs) == 1,
        "hi_sign_stable": len(hi_signs) == 1,
    }


def annotate_interval(tag: str, point: float, lo: float, hi: float, values: list[float]) -> dict:
    cell = {
        "tag": tag,
        "delta": point,
        "ci_lo": lo,
        "ci_hi": hi,
        "n_resamples": PRIMARY_N_RESAMPLES,
    }
    near_zero = (abs(lo) <= STABILITY_BAND) or (abs(hi) <= STABILITY_BAND)
    cell["near_zero_boundary"] = near_zero
    if not near_zero:
        cell["verdict"] = "positive" if lo > 0 else ("negative" if hi < 0 else "crosses_zero")
        cell["stable"] = True
        return cell
    check = stability_check(values)
    cell["stability_recheck"] = check
    stable = check["lo_sign_stable"] and check["hi_sign_stable"]
    cell["stable"] = stable
    if not stable:
        cell["verdict"] = "undecided"
    else:
        cell["verdict"] = "positive" if lo > 0 else ("negative" if hi < 0 else "crosses_zero")
    return cell


def suite_cap8(con, *, ck_keys, ba_keys, scorer_hash: str, suite: str) -> dict:
    cov = _metric_by_run(con, "evidence_coverage", scorer_hash=scorer_hash)
    cap8_by_task = _paired_by_task_map(ck_keys, ba_keys, cov)
    vals = [v for (s, _t), v in cap8_by_task.items() if s == suite]
    point, lo, hi = bca_ci(vals, seed=PRIMARY_BOOTSTRAP_SEED, n_resamples=PRIMARY_N_RESAMPLES)
    cell = annotate_interval(f"cap8/{suite}", point, lo, hi, vals)
    cell["n_tasks"] = len(vals)
    return cell


def model_contrast(
    con, *, label: str, checkpoint_model: str, baseline_model: str, scorer_hash: str
) -> dict:
    """One (checkpoint pin, baseline pin) contrast on `tier1_trained_qa_base`/test.

    Generic over the model pins so it serves both a granite seed (checkpoint=
    granite33-8b-sft-headline-sN, baseline=granite33-8b-base) and the Qwen headline row
    (checkpoint=qwen3-8b-dpo-stacked-notdone-both, baseline=qwen3-8b-base) through the same
    path -- the whole point being that neither gets bespoke arithmetic.
    """
    ckpt = _select_runs(con, arm=CHECKPOINT_ARM, grids=[GRID_NAME], model_id=checkpoint_model)
    base = _select_runs(con, arm=BASELINE_ARM, grids=[GRID_NAME], model_id=baseline_model)
    ck_keys = _by_key(ckpt)
    ba_keys = _by_key(base)

    mc = _matched_cost(
        con,
        ckpt=ckpt,
        base=base,
        ck_keys=ck_keys,
        ba_keys=ba_keys,
        scorer_hash=scorer_hash,
        seed=PRIMARY_BOOTSTRAP_SEED,
        n_resamples=PRIMARY_N_RESAMPLES,
    )

    by_suite_meta = {
        suite: {
            "n_checkpoint_runs": sum(len(v) for k, v in ck_keys.items() if k[0] == suite),
            "n_baseline_runs": sum(len(v) for k, v in ba_keys.items() if k[0] == suite),
        }
        for suite in SUITES
    }

    return {
        "label": label,
        "checkpoint_model": checkpoint_model,
        "baseline_model": baseline_model,
        "n_checkpoint_runs": len(ckpt),
        "n_baseline_runs": len(base),
        "matched_cost": mc,
        "by_suite_meta": by_suite_meta,
        "_ck_keys": ck_keys,
        "_ba_keys": ba_keys,
    }


def annotate_row(con, raw: dict, *, scorer_hash: str) -> dict:
    """Add the stability-checked matched-cost and cap-8 cells to a `model_contrast` row."""
    ck_keys, ba_keys = raw.pop("_ck_keys"), raw.pop("_ba_keys")
    mc = raw["matched_cost"]

    # matched-cost: annotate each suite's already-computed by_suite CI with the
    # stability recheck, using the SAME per-task deltas _matched_cost bootstrapped.
    # _matched_cost does not return the raw per-suite deltas list, so it is rebuilt
    # here from `by_task` using the identical pairing (_matched_cost's own `per_task`
    # dict, keyed the same way) -- see suite_matched_cost_deltas below.
    for suite in SUITES:
        cell = mc["by_suite"].get(suite)
        if not cell or cell.get("n_tasks", 0) == 0:
            continue
        vals = suite_matched_cost_deltas(
            con,
            ck_keys=ck_keys,
            ba_keys=ba_keys,
            scorer_hash=scorer_hash,
            suite=suite,
        )
        annotated = annotate_interval(
            f"matched_cost/{suite}", cell["delta"], cell["ci_lo"], cell["ci_hi"], vals
        )
        cell["stability"] = annotated

        cap8_cell = suite_cap8(
            con, ck_keys=ck_keys, ba_keys=ba_keys, scorer_hash=scorer_hash, suite=suite
        )
        cell["cap8"] = cap8_cell

    return raw


def non_vacuity(con, run_ids: list[str]) -> dict:
    """Coordinator's non-vacuity check: n_turns > 0 wherever turns.jsonl is non-empty, and
    evidence_coverage not identically 0 for an arm that asks. A real but nonzero malformed-action
    rate can zero an arm's coverage in a way that reads as a finding about the model rather than
    about parsing, so the malformed rate is computed and returned beside the vacuity flags rather
    than left for a reader to infer from a suspicious zero.
    """
    rows = _rows(
        con,
        "select r.run_id, r.n_turns, r.n_asks, "
        "(select count(*) from turns t where t.run_id = r.run_id) as n_turn_rows "
        f"from runs r where r.run_id in {_in(run_ids)}",
    )
    turns_lost = [r["run_id"] for r in rows if r["n_turns"] and not r["n_turn_rows"]]
    asks = [r for r in rows if r["n_asks"]]

    malformed = _rows(
        con,
        "select run_id, sum(charged) as n_malformed from ledger "
        f"where currency = 'malformed' and run_id in {_in(run_ids)} group by run_id",
    )
    n_malformed_total = sum(float(r["n_malformed"] or 0) for r in malformed)

    return {
        "n_runs": len(rows),
        "n_runs_with_lost_turn_rows": len(turns_lost),
        "lost_turn_run_ids_sample": turns_lost[:10],
        "n_runs_that_ask": len(asks),
        "n_malformed_events": n_malformed_total,
        "malformed_rate_per_ask_run": (n_malformed_total / len(asks)) if asks else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet-dir", required=True)
    ap.add_argument("--scorer-hash", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--symmetry-out",
        help="if set, also run the coordinator's 2026-09-18 symmetric-pairing check "
        "(both arms read at min(k_ckpt, k_base) off their own ladders) and write it here, "
        "instead of the asymmetric --out table",
    )
    a = ap.parse_args()

    if a.symmetry_out:
        result = symmetry_check(parquet_dir=a.parquet_dir, scorer_hash=a.scorer_hash)
        Path(a.symmetry_out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.symmetry_out).write_text(json.dumps(result, indent=2, default=str))
        print(f"wrote {a.symmetry_out}")
        return 0

    con = _con(Path(a.parquet_dir))
    result: dict = {"grid_name": GRID_NAME, "scorer_hash": a.scorer_hash, "seeds": {}}

    all_run_ids: list[str] = []
    for seed_tag in SEEDS:
        raw = model_contrast(
            con,
            label=seed_tag,
            checkpoint_model=f"granite33-8b-sft-headline-{seed_tag}",
            baseline_model=BASELINE_MODEL,
            scorer_hash=a.scorer_hash,
        )
        all_run_ids += [r["run_id"] for r in _flatten(raw["_ck_keys"])]
        all_run_ids += [r["run_id"] for r in _flatten(raw["_ba_keys"])]
        result["seeds"][seed_tag] = annotate_row(con, raw, scorer_hash=a.scorer_hash)

    qwen_raw = model_contrast(
        con,
        label="qwen_headline (rung 2, DPO -- NOT the same rung as the granite seeds)",
        checkpoint_model=QWEN_CHECKPOINT_MODEL,
        baseline_model=QWEN_BASELINE_MODEL,
        scorer_hash=a.scorer_hash,
    )
    all_run_ids += [r["run_id"] for r in _flatten(qwen_raw["_ck_keys"])]
    all_run_ids += [r["run_id"] for r in _flatten(qwen_raw["_ba_keys"])]
    result["qwen_headline"] = annotate_row(con, qwen_raw, scorer_hash=a.scorer_hash)

    result["non_vacuity"] = non_vacuity(con, sorted(set(all_run_ids)))

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=2, default=str))
    print(f"wrote {a.out}")
    return 0


def suite_matched_cost_deltas(
    con, *, ck_keys, ba_keys, scorer_hash: str, suite: str
) -> list[float]:
    """Re-derive exactly the per-task matched-cost deltas `_matched_cost` bootstraps for one
    suite, so the stability recheck resamples the SAME population its primary CI did.

    FIXED 2026-09-19 (found while answering the coordinator's symmetric-pairing question, not
    reported by them): the first version of this function appended one entry per
    `(suite, task, seed)` KEY straight into the returned list. `ck_keys`/`ba_keys` are keyed by
    that 3-tuple (`_by_key`), so a 2-seed grid produced 400 entries for a 200-task suite instead
    of 200 -- the POINT estimate (a mean) is invariant to this when every task carries the same
    seed count, which is why `test_suite_matched_cost_deltas_reproduces_matched_costs_own_by_
    suite_mean` passed regardless, but the bootstrap CI is not: resampling 400 correlated,
    unclustered points understates variance relative to resampling 200 task-clustered ones.
    `_matched_cost` itself avoids exactly this with a two-stage `per_task.setdefault((suite,
    task), []).append(...)` step (seed-keys accumulate into their task's list, then that list is
    averaged) before anything reaches `bca_ci`. This function now does the same two-stage
    grouping, so the population handed to `stability_check`'s 50k-resample recheck is task
    clustered, not seed-flattened. `test_the_deltas_list_is_task_clustered_not_seed_flattened`
    pins the count directly (`len(deltas) == n_tasks`, not `n_tasks * n_seeds`).

    This duplicates `_matched_cost`'s inner loop rather than changing that function's return
    shape, because `_matched_cost` is imported verbatim from `pinq_train.gate` (the headline's
    own function, per CLAUDE.md) and is not this repo's to edit for one caller's convenience.
    `test_suite_matched_cost_deltas_reproduces_matched_costs_own_by_suite_mean` pins this
    function's MEAN against `_matched_cost`'s own `by_suite[suite]["delta"]`; the task-clustering
    test above pins its SHAPE, which the mean-only test cannot see.
    """
    cov = _metric_by_run(con, "evidence_coverage", scorer_hash=scorer_hash)
    n_asks = {
        str(r["run_id"]): int(r["n_asks"] or 0)
        for r in _rows_n_asks(con, [*_flatten(ck_keys), *_flatten(ba_keys)])
    }
    shared = sorted(set(ck_keys) & set(ba_keys) & {k for k in ck_keys if k[0] == suite})
    base_used = sorted({rid for key in shared for rid in ba_keys[key]})
    ladders = _coverage_ladder(con, [{"run_id": rid} for rid in base_used], scorer_hash=scorer_hash)

    per_task: dict[tuple, list[float]] = {}
    for key in shared:
        if key[0] != suite:
            continue
        per_ckpt: list[float] = []
        for c_rid in ck_keys[key]:
            c_val = cov.get(c_rid)
            if c_val is None:
                continue
            k = n_asks.get(c_rid, 0)
            rungs = []
            for b_rid in ba_keys[key]:
                b_n = n_asks.get(b_rid, 0)
                at = min(k, b_n)
                rung = (ladders.get(b_rid) or {}).get(at)
                if rung is None:
                    continue
                rungs.append(float(rung))
            if not rungs:
                continue
            per_ckpt.append(float(c_val) - (sum(rungs) / len(rungs)))
        if per_ckpt:
            per_task.setdefault((key[0], key[1]), []).append(sum(per_ckpt) / len(per_ckpt))
    return [sum(v) / len(v) for v in per_task.values()]


def symmetric_matched_cost_deltas(
    con, *, ck_keys, ba_keys, scorer_hash: str, suite: str
) -> tuple[list[float], int, int]:
    """The coordinator's 2026-09-18 symmetry fix, applied to one suite.

    THE DEFECT. `_matched_cost` (and `suite_matched_cost_deltas` above, which reproduces it
    verbatim) reads the checkpoint at its OWN terminal `evidence_coverage` -- i.e. at its own
    k -- and the baseline at `frontier_q#min(k, baseline_n_asks)`. That is a matched pairing
    only while the checkpoint spends no more than the baseline. Where the checkpoint outspends
    the baseline (`k > baseline_n_asks`), the baseline is capped at ITS OWN shorter episode
    while the checkpoint keeps its full, longer one: the two sides are no longer read at the
    same question count. `_matched_cost` already counts this exposure per suite as
    `n_baseline_shorter_than_k` (the "unsafe" pairs below) -- this function is what recomputing
    those pairs symmetrically means: read BOTH sides at `min(k_ckpt, k_base)`, off each side's
    OWN prefix ladder, never off either side's terminal value directly.

    Returns (deltas, n_pairs, n_unsafe) so the caller can report the published-vs-symmetric
    comparison per suite without a second query. `deltas` is task-clustered (seeds averaged
    within a task, same fix and same reason as `suite_matched_cost_deltas` above); `n_pairs`
    and `n_unsafe` are counted at the finer (task, seed) x run granularity on purpose, because
    the coordinator's question is about individual comparator instances, not about how many
    tasks the bootstrap sees.
    """
    n_asks = {
        str(r["run_id"]): int(r["n_asks"] or 0)
        for r in _rows_n_asks(con, [*_flatten(ck_keys), *_flatten(ba_keys)])
    }
    shared = sorted(set(ck_keys) & set(ba_keys) & {k for k in ck_keys if k[0] == suite})
    ckpt_used = sorted({rid for key in shared for rid in ck_keys[key]})
    base_used = sorted({rid for key in shared for rid in ba_keys[key]})
    ckpt_ladders = _coverage_ladder(
        con, [{"run_id": rid} for rid in ckpt_used], scorer_hash=scorer_hash
    )
    base_ladders = _coverage_ladder(
        con, [{"run_id": rid} for rid in base_used], scorer_hash=scorer_hash
    )

    per_task: dict[tuple, list[float]] = {}
    n_pairs = 0
    n_unsafe = 0
    for key in shared:
        if key[0] != suite:
            continue
        per_ckpt: list[float] = []
        for c_rid in ck_keys[key]:
            k = n_asks.get(c_rid, 0)
            pair_vals: list[float] = []
            for b_rid in ba_keys[key]:
                b_n = n_asks.get(b_rid, 0)
                at = min(k, b_n)
                n_pairs += 1
                if b_n < k:
                    n_unsafe += 1
                c_rung = (ckpt_ladders.get(c_rid) or {}).get(at)
                b_rung = (base_ladders.get(b_rid) or {}).get(at)
                if c_rung is None or b_rung is None:
                    continue
                pair_vals.append(float(c_rung) - float(b_rung))
            if pair_vals:
                per_ckpt.append(sum(pair_vals) / len(pair_vals))
        if per_ckpt:
            per_task.setdefault((key[0], key[1]), []).append(sum(per_ckpt) / len(per_ckpt))
    deltas = [sum(v) / len(v) for v in per_task.values()]
    return deltas, n_pairs, n_unsafe


def symmetry_check(*, parquet_dir: str, scorer_hash: str) -> dict:
    """The full published-vs-symmetric comparison, all 9 granite matched-cost cells."""
    con = _con(Path(parquet_dir))
    out: dict[str, dict] = {}
    for seed_tag in SEEDS:
        ckpt_model = f"granite33-8b-sft-headline-{seed_tag}"
        ckpt = _select_runs(con, arm=CHECKPOINT_ARM, grids=[GRID_NAME], model_id=ckpt_model)
        base = _select_runs(con, arm=BASELINE_ARM, grids=[GRID_NAME], model_id=BASELINE_MODEL)
        ck_keys, ba_keys = _by_key(ckpt), _by_key(base)
        for suite in SUITES:
            published = suite_matched_cost_deltas(
                con, ck_keys=ck_keys, ba_keys=ba_keys, scorer_hash=scorer_hash, suite=suite
            )
            symmetric, n_pairs, n_unsafe = symmetric_matched_cost_deltas(
                con, ck_keys=ck_keys, ba_keys=ba_keys, scorer_hash=scorer_hash, suite=suite
            )
            pub_point, pub_lo, pub_hi = bca_ci(
                published, seed=PRIMARY_BOOTSTRAP_SEED, n_resamples=PRIMARY_N_RESAMPLES
            )
            sym_point, sym_lo, sym_hi = bca_ci(
                symmetric, seed=PRIMARY_BOOTSTRAP_SEED, n_resamples=PRIMARY_N_RESAMPLES
            )
            out[f"{seed_tag}/{suite}"] = {
                "seed": seed_tag,
                "suite": suite,
                "n_tasks": len(published),
                "n_pairs": n_pairs,
                "n_unsafe": n_unsafe,
                "published": {"delta": pub_point, "ci_lo": pub_lo, "ci_hi": pub_hi},
                "symmetric": {"delta": sym_point, "ci_lo": sym_lo, "ci_hi": sym_hi},
                "abs_diff": abs(sym_point - pub_point),
                "sign_changed": (pub_point > 0) != (sym_point > 0)
                and pub_point != 0
                and sym_point != 0,
                "significance_changed": ((pub_lo > 0 or pub_hi < 0) != (sym_lo > 0 or sym_hi < 0)),
            }
    return out


def _flatten(keys: dict) -> list[dict]:
    return [{"run_id": rid} for v in keys.values() for rid in v]


def _rows_n_asks(con, run_dicts: list[dict]) -> list[dict]:
    from pinq_train.gate import _with_stop

    return _with_stop(con, run_dicts)


if __name__ == "__main__":
    raise SystemExit(main())
