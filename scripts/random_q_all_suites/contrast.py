#!/usr/bin/env python3
"""Lane L1.8: the retrieval-volume control (`random_q`) contrasted against `inquirer_trained`
and `inquirer_prompted`, on all three test suites, at matched question count and at cap 8.

    PI_GOLD_ROOT="$PWD/data/gold" .venv/bin/python -m scripts.random_q_all_suites.contrast \
        --parquet-dir scores/parquet --out artifacts/random_q_all_suites_20260918/contrast.json

Reuses `pinq_train.gate._matched_cost`, `_by_key` and `bca_ci` rather than re-deriving the
matched-k ladder read or the BCa bootstrap: `pi_eval` is gold-only and this module must not
import it (import-linter contract 3), and gate.py's docstring on `_matched_cost` already
explains why the prefix ladder is a legitimate matched-cost comparator.

WHY SELECTION HERE IS NOT `gate._select_runs`. That helper filters on `runs.grid_name`, and
the `random_q` rows this lane reads carry an EMPTY grid_name (launched by explicit `pi run`
flags, not `--sweep`) -- see `artifacts/launch_controls_20260918/RESULT.md` and
`artifacts/nulls_and_determinism_20260918/README.md`. A grid-name selector returns nothing for
them. `select()` below filters on `(arm_id, suite_id, split, status, dirty, budget_cap, seed,
code_version)` and, where an arm pools more than one checkpoint pin under one arm_id, on the
inquirer model id read through `calls.parquet` -- the same disambiguation `_select_runs` uses
for the same reason (`nulls_and_determinism_20260918/README.md` section 4's two-pin warning).

THE TRAINED PIN IS THE SAME MODEL ON EVERY SUITE: `qwen3-8b-dpo-stacked-notdone-both`, the
checkpoint this lane's brief names and the one `artifacts/testsplit_qa/TESTSPLIT_QA.md` reads
against `qwen3-8b-base` on all three suites (musique n=200, strategyqa n=200, wiki2 n=166,
test, cap 8, code_version 3ae099d0e9f08f6654d5e259ffc0850232f8e70a). `qwen3-8b-sft-headline` is
a DIFFERENT real checkpoint that also carries `inquirer_trained` rows on musique at this commit
-- `TESTSPLIT_QA.md`'s own two-pin warning is that musique's `inquirer_trained` pools 482
sft-headline rows with 400 dpo-stacked-notdone-both rows under the one arm_id, and reading the
arm without a `model_id` filter averages the two. An earlier version of `TRAINED_MODEL_BY_SUITE`
here substituted sft-headline for musique alone, which would have contrasted a different
treatment on musique than on the other two suites under one row label; corrected so `model_id`
disambiguates to the SAME pin everywhere, matching the brief and matching TESTSPLIT_QA.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]

SUITES: tuple[str, ...] = ("musique", "strategyqa", "wiki2")
SEEDS: tuple[int, ...] = (0, 1)
BUDGET_CAP = 8

# The code_version the EXISTING musique/strategyqa random_q test rows carry, and that this
# lane's wiki2 sweep was pinned to match (artifacts/random_q_all_suites_20260918/RESULT.md).
RANDOM_Q_CODE_VERSION = "107ef2221a55229ac7ba60a54ebfc9849971bde0"
# The balanced reference cell: single-valued 200/200 per suite per seed, verified in
# artifacts/nulls_and_determinism_20260918/README.md section 4.
PROMPTED_CODE_VERSION = "3ae099d0e9f08f6654d5e259ffc0850232f8e70a"
TRAINED_CODE_VERSION = "3ae099d0e9f08f6654d5e259ffc0850232f8e70a"
TRAINED_MODEL_BY_SUITE: Mapping[str, str] = {
    "musique": "qwen3-8b-dpo-stacked-notdone-both",
    "strategyqa": "qwen3-8b-dpo-stacked-notdone-both",
    "wiki2": "qwen3-8b-dpo-stacked-notdone-both",
}
# 52 run ids from artifacts/nulls_and_determinism_20260918/EXCLUDE_random_q_zero_ask.eligible.txt:
# zero-ask `random_q` rows that reached `ok` with no reference for their task, i.e. a
# `drafter_only` trajectory wearing this arm's label (commit 6c040cd fixed the arm; these rows
# predate the fix). Only the 50 at RANDOM_Q_CODE_VERSION, musique, are actually reachable by
# `select()`'s other filters; the file is read whole because that is the record of provenance.
DEFAULT_EXCLUDE_FILE = (
    REPO_ROOT
    / "artifacts"
    / "nulls_and_determinism_20260918"
    / "EXCLUDE_random_q_zero_ask.eligible.txt"
)


def load_exclude_run_ids(path: Path = DEFAULT_EXCLUDE_FILE) -> frozenset[str]:
    if not path.exists():
        return frozenset()
    return frozenset(x.strip() for x in path.read_text().splitlines() if x.strip())


def select(
    con,
    *,
    arm_id: str,
    suite_id: str,
    code_version: str,
    model_id: str | None = None,
    exclude_run_ids: frozenset[str] = frozenset(),
    seeds: Sequence[int] = SEEDS,
    budget_cap: int = BUDGET_CAP,
) -> list[dict]:
    """The `(run_id, suite_id, task_id, seed)` rows for one arm's one pin on one suite.

    Every filter here is a fact this lane measured about the shared store, not a default:
    split=test (train/dev are training-only, never an eval arm), status=ok, dirty=false (a
    dirty tree stamps `dev-`, training-only), budget_cap=8 and code_version pinned exactly
    (mixing commits silently voids the pin the way `nulls_and_determinism_20260918` warns
    against), seed restricted to the two this lane ran (musique carries a third seed at
    `TRAINED_CODE_VERSION` that the other two suites do not).
    """
    seed_list = ", ".join(str(int(s)) for s in seeds)
    where = (
        f"r.arm_id = '{arm_id}' AND r.suite_id = '{suite_id}' AND r.split = 'test' "
        f"AND r.status = 'ok' AND r.dirty = false AND r.budget_cap = {int(budget_cap)} "
        f"AND r.code_version = '{code_version}' AND r.seed IN ({seed_list})"
    )
    if model_id:
        where += (
            f" AND r.run_id IN (SELECT run_id FROM calls WHERE actor = 'inquirer' "
            f"AND model = '{model_id}')"
        )
    rows = con.execute(
        f"SELECT r.run_id, r.suite_id, r.task_id, r.seed FROM runs r WHERE {where} "
        "ORDER BY r.suite_id, r.task_id, r.seed, r.run_id"
    ).fetchdf()
    out = [dict(rec) for rec in rows.to_dict("records")]
    if exclude_run_ids:
        out = [r for r in out if r["run_id"] not in exclude_run_ids]
    return out


def run_contrast(
    parquet_dir: Path,
    *,
    scorer_hash: str,
    seed: int = 0,
    n_resamples: int = 10_000,
    exclude_run_ids: frozenset[str] | None = None,
) -> dict:
    from pinq_train.gate import _by_key, _con, _matched_cost

    con = _con(parquet_dir)
    excl = load_exclude_run_ids() if exclude_run_ids is None else exclude_run_ids

    random_q = {
        suite: select(
            con,
            arm_id="random_q",
            suite_id=suite,
            code_version=RANDOM_Q_CODE_VERSION,
            exclude_run_ids=excl,
        )
        for suite in SUITES
    }
    prompted = {
        suite: select(
            con, arm_id="inquirer_prompted", suite_id=suite, code_version=PROMPTED_CODE_VERSION
        )
        for suite in SUITES
    }
    trained = {
        suite: select(
            con,
            arm_id="inquirer_trained",
            suite_id=suite,
            code_version=TRAINED_CODE_VERSION,
            model_id=TRAINED_MODEL_BY_SUITE[suite],
        )
        for suite in SUITES
    }

    out: dict = {"comparisons": {}, "selection": {}}
    for suite in SUITES:
        out["selection"][suite] = {
            "random_q_n_rows": len(random_q[suite]),
            "random_q_n_tasks": len({r["task_id"] for r in random_q[suite]}),
            "inquirer_prompted_n_rows": len(prompted[suite]),
            "inquirer_trained_n_rows": len(trained[suite]),
            "inquirer_trained_model": TRAINED_MODEL_BY_SUITE[suite],
        }

    for label, ckpt_by_suite in (("inquirer_trained", trained), ("inquirer_prompted", prompted)):
        all_ckpt = [r for suite in SUITES for r in ckpt_by_suite[suite]]
        all_base = [r for suite in SUITES for r in random_q[suite]]
        if not all_ckpt or not all_base:
            out["comparisons"][label] = {
                "error": "empty arm",
                "n_ckpt": len(all_ckpt),
                "n_base": len(all_base),
            }
            continue
        ck_keys, ba_keys = _by_key(all_ckpt), _by_key(all_base)
        mc = _matched_cost(
            con,
            ckpt=all_ckpt,
            base=all_base,
            ck_keys=ck_keys,
            ba_keys=ba_keys,
            scorer_hash=scorer_hash,
            seed=seed,
            n_resamples=n_resamples,
        )
        mc.pop("deltas", None)
        out["comparisons"][f"{label}_vs_random_q"] = mc

        # NEAR-ZERO ROBUSTNESS. A single bootstrap seed's CI on a small effect is not enough to
        # call: rerun at 50k resamples across three seeds and require the sign AND the
        # zero-exclusion verdict to agree across all three, on every cell (pooled and per
        # suite) whose 10k reading is near zero. Cells that are not near zero (e.g. the
        # pre-existing +0.5151 pooled trained-vs-random_q reading) are left at 10k/seed 0: this
        # check exists for the borderline case, not to re-run everything at 5x the cost.
        candidates = [("pooled", mc["pooled"]), *mc["by_suite"].items()]
        near_zero_cells = [(where, cell) for where, cell in candidates if is_near_zero(cell)]
        if near_zero_cells:
            robustness = {}
            for where, _cell in near_zero_cells:
                if where == "pooled":
                    ckpt_r, base_r = all_ckpt, all_base
                else:
                    ckpt_r = [r for r in all_ckpt if r["suite_id"] == where]
                    base_r = [r for r in all_base if r["suite_id"] == where]
                by_seed = resample_at_seeds(
                    con,
                    ckpt=ckpt_r,
                    base=base_r,
                    scorer_hash=scorer_hash,
                    seeds=(0, 1, 2),
                    n_resamples=50_000,
                    suite=None if where == "pooled" else where,
                )
                robustness[where] = {"by_seed": by_seed, **sign_stability(by_seed)}
            out["comparisons"][f"{label}_vs_random_q"]["near_zero_50k_3seed"] = robustness
    return out


def is_near_zero(cell: Mapping) -> bool:
    """A cell's 10k CI does not exclude zero -- the trigger for the heavier 50k/3-seed check.

    `cell` is one `_matched_cost` `by_suite[...]` entry or its `pooled` entry: a mapping with at
    least `ci_lo`/`ci_hi`. Missing bounds (an empty cell) are NOT near zero -- there is nothing
    to re-check, and reporting "near zero" for "no data" would be a fabricated reading.
    """
    lo, hi = cell.get("ci_lo"), cell.get("ci_hi")
    if lo is None or hi is None:
        return False
    return lo <= 0 <= hi


def sign_stability(by_seed: Sequence[Mapping]) -> dict:
    """Do independent bootstrap seeds agree on the sign and on whether zero is excluded?

    `by_seed` is one dict per seed with at least `delta`, `ci_lo`, `ci_hi` (as produced by
    `resample_at_seeds`). Point estimates are seed-INVARIANT (a mean is a mean, `bca_ci`'s own
    docstring), so a real sign disagreement across seeds can only mean too few tasks carry the
    contrast for the bootstrap to have converged -- which is exactly the case this function
    exists to catch rather than silently report as one seed's number.
    """
    signs = {d["delta"] >= 0 for d in by_seed if d.get("delta") is not None}
    excludes_zero = {
        not (d["ci_lo"] <= 0 <= d["ci_hi"])
        for d in by_seed
        if d.get("ci_lo") is not None and d.get("ci_hi") is not None
    }
    return {
        "sign_stable": len(signs) <= 1,
        "zero_exclusion_stable": len(excludes_zero) <= 1,
        "n_seeds": len(by_seed),
    }


def resample_at_seeds(
    con,
    *,
    ckpt: Sequence[Mapping],
    base: Sequence[Mapping],
    scorer_hash: str,
    seeds: Sequence[int],
    n_resamples: int,
    suite: str | None = None,
) -> list[dict]:
    """Re-run `_matched_cost`'s pooled reading at each seed in `seeds`.

    Thin orchestration over `pinq_train.gate._matched_cost`, which already has its own tests;
    this does not re-derive the bootstrap, only repeats it. `suite=None` pools every suite in
    `ckpt`/`base` (matching the pooled cell); a suite name restricts to rows already filtered to
    that suite by the caller (`ckpt`/`base` are expected pre-filtered, so this never re-derives
    the suite filter differently from the cell it is re-checking).
    """
    from pinq_train.gate import _by_key, _matched_cost

    ck_keys, ba_keys = _by_key(ckpt), _by_key(base)
    out = []
    for s in seeds:
        mc = _matched_cost(
            con,
            ckpt=ckpt,
            base=base,
            ck_keys=ck_keys,
            ba_keys=ba_keys,
            scorer_hash=scorer_hash,
            seed=s,
            n_resamples=n_resamples,
        )
        cell = mc["pooled"] if suite is None else mc["by_suite"].get(suite, {})
        out.append(
            {
                "seed": s,
                "delta": cell.get("delta"),
                "ci_lo": cell.get("ci_lo"),
                "ci_hi": cell.get("ci_hi"),
                "n_tasks": cell.get("n_tasks", 0),
            }
        )
    return out


def _print_table(result: dict) -> None:
    print(
        f"{'comparison':28s} {'suite':11s} {'n_tasks':>8s} {'matched_delta':>22s} "
        f"{'cap8_delta':>10s} {'cap8_n':>7s}"
    )
    for label, mc in result["comparisons"].items():
        if "error" in mc:
            print(f"{label:28s}  ({mc['error']}: ckpt={mc['n_ckpt']} base={mc['n_base']})")
            continue
        for suite, cell in mc["by_suite"].items():
            ci = (
                f"[{cell['ci_lo']:+.4f},{cell['ci_hi']:+.4f}]"
                if cell["ci_lo"] is not None
                else "[n/a]"
            )
            delta = f"{cell['delta']:+.4f}" if cell["delta"] is not None else "n/a"
            cap8 = (
                f"{cell['cap8_coverage_delta']:+.4f}"
                if cell["cap8_coverage_delta"] is not None
                else "n/a"
            )
            print(
                f"{label:28s} {suite:11s} {cell['n_tasks']:8d} {delta:>10s} {ci:>12s} "
                f"{cap8:>10s} {cell['cap8_n_tasks']:7d}"
            )
        p = mc["pooled"]
        pci = f"[{p['ci_lo']:+.4f},{p['ci_hi']:+.4f}]" if p["ci_lo"] is not None else "[n/a]"
        pdelta = f"{p['delta']:+.4f}" if p["delta"] is not None else "n/a"
        print(f"{label:28s} {'pooled':11s} {p['n_tasks']:8d} {pdelta:>10s} {pci:>12s}")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet-dir", default=str(REPO_ROOT / "scores" / "parquet"))
    ap.add_argument("--scorer-hash", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-resamples", type=int, default=10_000)
    ap.add_argument("--out", default=None, help="write the JSON result here")
    a = ap.parse_args(argv)

    result = run_contrast(
        Path(a.parquet_dir), scorer_hash=a.scorer_hash, seed=a.seed, n_resamples=a.n_resamples
    )
    _print_table(result)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(result, indent=2, sort_keys=True))
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
