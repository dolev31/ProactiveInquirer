"""The seed-replicate contrast: matched-cost AND cap-8 coverage deltas, pins side by side.

Lane L2.2 (2026-09-18) turns the headline's single training seed into a three-seed claim.
`pinq_train.gate._matched_cost` is the function the headline's own numbers came from -- it
produces the matched-cost delta AND the cap-8 delta in one call, paired on (suite, task, seed)
with seeds averaged into the task before the BCa bootstrap (`scripts/matched_cost.py`'s own
docstring names the same convention: task-clustered BCa, seeds folded in first). This module
calls it directly, once per (pin, suite), selecting each side by `model_id` through
`calls.parquet` -- exactly `_select_runs`'s own existing filter -- so that pooling several
checkpoints under one shared `grid_name` never happens. The test-split contrast run here needs
no isolated parquet directory the way the dev gate does (see `isolate_parquet.py`): `_select_
runs`'s `model_id` filter is enough on its own to keep pins apart within the shared store,
because nothing downstream of this module's `pin_contrast` re-selects without that filter.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:  # a checkout without an editable install
    sys.path.insert(0, str(REPO / "src"))
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:  # `scripts/` is not a package; sibling import by path
    sys.path.insert(0, str(HERE))

from stability import (  # noqa: E402
    DEFAULT_RESEED_RESAMPLES,
    DEFAULT_SEEDS,
    bound_stability,
    is_near_zero,
)

from pinq_train.gate import _by_key, _con, _matched_cost, _select_runs  # noqa: E402


def pin_contrast(
    con,
    *,
    checkpoint_model_id: str,
    baseline_model_id: str,
    checkpoint_grids: Sequence[str] | None = None,
    baseline_grids: Sequence[str] | None = None,
    grid_name: str | None = None,
    checkpoint_arm: str = "inquirer_trained",
    baseline_arm: str = "inquirer_prompted",
    scorer_hash: str,
    seed: int = 0,
    n_resamples: int = 10000,
) -> dict[str, Any]:
    """Matched-cost + cap-8 coverage deltas for one pin, split per suite by `_matched_cost`'s
    own `by_suite`. Also carries the exact run_id list both sides selected, so a caller can
    write it out as this artifact's provenance file (CLAUDE.md rule 1: a run_id list, not a
    filter description, is what makes a number reproducible).

    CHECKPOINT AND BASELINE CAN NEED DIFFERENT `grid_name`s. MEASURED: the existing
    qwen3-8b-sft-headline (s0) test-split rows carry `grid_name == ""` (the "grid file was
    lost" shape `artifacts/launch_controls_20260918/RESULT.md` already documented for other
    arms at this commit), while the baseline they pair against carries
    `grid_name == "tier1_trained_qa_base"`. A single shared `grid_name` cannot select both, so
    `checkpoint_grids`/`baseline_grids` are independent; `grid_name` is kept as a convenience
    that sets both to `[grid_name]` when the two sides DO share one (the common case).
    """
    if grid_name is not None:
        checkpoint_grids = checkpoint_grids if checkpoint_grids is not None else [grid_name]
        baseline_grids = baseline_grids if baseline_grids is not None else [grid_name]
    if not checkpoint_grids or not baseline_grids:
        raise ValueError(
            "pin_contrast: pass grid_name=, or both checkpoint_grids= and baseline_grids="
        )
    ckpt = _select_runs(
        con, arm=checkpoint_arm, grids=list(checkpoint_grids), model_id=checkpoint_model_id
    )
    base = _select_runs(
        con, arm=baseline_arm, grids=list(baseline_grids), model_id=baseline_model_id
    )
    if not ckpt:
        raise ValueError(
            f"no {checkpoint_arm!r} rows for model_id={checkpoint_model_id!r} under grid(s) "
            f"{list(checkpoint_grids)!r}"
        )
    if not base:
        raise ValueError(
            f"no {baseline_arm!r} rows for model_id={baseline_model_id!r} under grid(s) "
            f"{list(baseline_grids)!r}"
        )
    ck_keys, ba_keys = _by_key(ckpt), _by_key(base)
    mc = _matched_cost(
        con,
        ckpt=ckpt,
        base=base,
        ck_keys=ck_keys,
        ba_keys=ba_keys,
        scorer_hash=scorer_hash,
        seed=seed,
        n_resamples=n_resamples,
    )
    mc["run_ids"] = sorted({r["run_id"] for r in ckpt} | {r["run_id"] for r in base})
    mc["checkpoint_model_id"] = checkpoint_model_id
    mc["baseline_model_id"] = baseline_model_id
    mc["n_checkpoint_runs"] = len(ckpt)
    mc["n_baseline_runs"] = len(base)

    _attach_stability(
        mc,
        recompute=lambda rs, n: _matched_cost(
            con,
            ckpt=ckpt,
            base=base,
            ck_keys=ck_keys,
            ba_keys=ba_keys,
            scorer_hash=scorer_hash,
            seed=rs,
            n_resamples=n,
        ),
    )
    return mc


def _attach_stability(mc: dict[str, Any], *, recompute) -> None:
    """A near-zero `ci_lo`/`ci_hi` is read again at 50k resamples under three more seeds (see
    `stability.py`). One shared reseed pass covers every suite AND the pooled row, since
    `_matched_cost` computes all of them in a single call -- so the extra cost is at most three
    more calls total per pin, not three per cell.

    Mutates `mc` in place: each flagged `by_suite[suite]` and, if flagged, `pooled` gets a
    `stability` key with the reseed record; every other cell gets `stability: None`.
    """
    targets: list[tuple[str, dict]] = [(s, bs) for s, bs in mc["by_suite"].items()]
    targets.append(("__pooled__", mc["pooled"]))
    any_flagged = any(is_near_zero(t["ci_lo"]) or is_near_zero(t["ci_hi"]) for _, t in targets)
    reseed_cache: dict[int, dict[str, Any]] = {}

    def reseed_full(rs: int, n: int) -> dict[str, Any]:
        if rs not in reseed_cache:
            reseed_cache[rs] = recompute(rs, n)
        return reseed_cache[rs]

    for suite, cell in targets:

        def reseed_cell(rs: int, n: int, suite=suite) -> tuple[float, float]:
            r = reseed_full(rs, n)
            block = r["pooled"] if suite == "__pooled__" else r["by_suite"][suite]
            return (block["ci_lo"], block["ci_hi"])

        cell["stability"] = (
            bound_stability(
                reseed_cell,
                primary_lo=cell["ci_lo"],
                primary_hi=cell["ci_hi"],
                seeds=DEFAULT_SEEDS,
                n_resamples=DEFAULT_RESEED_RESAMPLES,
            )
            if any_flagged
            else None
        )


def _fmt_cell(bs: dict[str, Any] | None) -> str:
    if bs is None:
        return "(no rows)"
    flag = ""
    stab = bs.get("stability")
    if stab is not None and not stab["all_stable"]:
        flag = " UNDECIDED(sign unstable @50k x3)"
    return f"{bs['delta']:+.4f} [{bs['ci_lo']:+.4f},{bs['ci_hi']:+.4f}] n={bs['n_tasks']}{flag}"


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet-dir", default=str(REPO / "scores/parquet"))
    ap.add_argument(
        "--grid-name",
        default="tier1_trained_qa_base",
        help="used for both sides unless overridden below",
    )
    ap.add_argument(
        "--checkpoint-grid",
        action="append",
        default=None,
        help="repeatable; overrides --grid-name for the checkpoint side",
    )
    ap.add_argument(
        "--baseline-grid",
        action="append",
        default=None,
        help="repeatable; overrides --grid-name for the baseline side",
    )
    ap.add_argument("--baseline-model-id", default="qwen3-8b-base")
    ap.add_argument(
        "--pin",
        action="append",
        required=True,
        metavar="LABEL=MODEL_ID",
        help="repeatable, e.g. --pin s0=qwen3-8b-sft-headline --pin s1=qwen3-8b-sft-headline-s1",
    )
    ap.add_argument("--scorer-hash", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-resamples", type=int, default=10000)
    ap.add_argument("--out", default=None, help="write the full per-pin JSON here")
    a = ap.parse_args(argv)

    con = _con(Path(a.parquet_dir))
    results: dict[str, Any] = {}
    for spec in a.pin:
        label, _, model_id = spec.partition("=")
        results[label] = pin_contrast(
            con,
            checkpoint_model_id=model_id,
            baseline_model_id=a.baseline_model_id,
            checkpoint_grids=a.checkpoint_grid or [a.grid_name],
            baseline_grids=a.baseline_grid or [a.grid_name],
            scorer_hash=a.scorer_hash,
            seed=a.seed,
            n_resamples=a.n_resamples,
        )

    suites = sorted({s for r in results.values() for s in r["by_suite"]})
    labels = list(results)
    print(f"{'suite':11s} " + " ".join(f"{label:>42s}" for label in labels))
    for suite in suites:
        cells = [_fmt_cell(results[label]["by_suite"].get(suite)).rjust(42) for label in labels]
        print(f"{suite:11s} " + " ".join(cells))

    if a.out:
        Path(a.out).write_text(json.dumps(results, indent=2, sort_keys=True))
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
