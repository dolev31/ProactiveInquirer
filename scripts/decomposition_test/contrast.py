"""Lane L2.3: the held-out (test-split) counterpart of appendix_training.tex's
``tab:train-coverage``, computed with the SAME pairing and bootstrap machinery the headline
used -- ``pinq_train.gate._matched_cost`` -- not a reimplementation of it.

WHY THIS FILE EXISTS RATHER THAN A CALL TO ``pinq_train.gate.run_gate``. ``run_gate``'s public
signature accepts a ``baseline_model_id`` filter (the two-pin protocol: the same
``inquirer_prompted`` arm run against both the teacher and the base model) but has **no
checkpoint-side model filter** -- ``_select_runs(con, arm=checkpoint_arm, grids=[grid_name],
model_id=None)`` is called with ``model_id=None`` unconditionally. That is correct for
``run_gate``'s normal caller, which points ``parquet_dir`` at a directory `pi compact` built
from ONE checkpoint's own runs-root, so the missing filter is redundant there, never wrong.

It is silently wrong against the SHARED ``scores/parquet`` store this lane reads, where many
checkpoints share one ``grid_name`` (``tier1_trained_qa_base``) and one ``arm_id``
(``inquirer_trained``). Measured 2026-09-18: calling ``run_gate`` against the shared store for
the Stacked checkpoint alone selected 7,200 checkpoint rows, not the 1,200 that belong to it --
every OTHER trained arm sharing that grid_name was pooled into the same "checkpoint" side, and
`_matched_cost`'s per-key average silently blended them into what looked like a Stacked-only
number. Musique's ``n_baseline_shorter_than_k`` moved from 41 to 246 and ``delta`` moved from
0.1230 to 0.1246 -- a plausible number with no error message anywhere. ``select_and_contrast``
below is the fix: it calls ``_select_runs`` directly with an explicit ``model_id`` on BOTH
sides, exactly as ``run_gate`` already does for the baseline. ``tests/test_decomposition_test.py``
encodes this as the failing-without-the-fix case (CONTRIBUTING.md rule 2).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from pinq_train.gate import (
    _by_key,
    _con,
    _coverage_ladder,
    _in,
    _matched_cost,
    _mean,
    _metric_by_run,
    _rows,
    _select_runs,
    _with_stop,
    bca_ci,
)


class Vacuous(RuntimeError):
    """The coordinator's non-vacuity check failed (2026-09-18 correction): a population that
    looks selectable and pairable but carries no real signal, most likely because the shared
    store's compaction defect (turns/evidence/calls rows dropped for 21,001 runs while the run
    directories on disk still hold them) reached this population's SCORING even though its
    `runs.parquet` rows are intact. Raised rather than returned so a caller cannot forget to
    check it -- see `EmptyArm`/`IncompleteArm` in `pinq_train.gate` for the same pattern.
    """


# --------------------------------------------------------------------------------------------
# The mapping (Lane L2.3 step 1). Every entry is evidenced in
# artifacts/decomposition_test_20260918/RESULT.md; kept here as the one place both the launcher
# and the table renderer read it from, so the two cannot silently disagree about which
# registry name a paper row means.
#
# "arm" is the paper's own name for the row (tab:train-recipes / tab:train-coverage).
# "model_id" is the registry key in conf/checkpoints.json -- what PI_MODEL_INQUIRER is set to.
# "is_candidate" mirrors the paper's own text: "The four candidates are Stop-weight,
# Persistence-only, Mixed and Stacked. Question-preference is the policy Stacked starts from
# ... and is listed because the mechanism argument below rests on it" (appendix_training.tex,
# tab:train-recipes caption) -- i.e. Question-preference and Reference are shown but are not
# "candidates" in the paper's own vocabulary.
ARM_MAP: dict[str, dict[str, Any]] = {
    "Reference": {
        "model_id": "qwen3-8b-sft-headline",
        "is_candidate": False,
        "evidence": (
            "conf/checkpoints.json _second_row_added: 'the headline SFT run: 14,302 examples, "
            "inquirer_prompted only, 2 epochs; best/by_nll selects that same final tag'. "
            "appendix_training.tex:26 'Reference | base model | 14,302 decisions | nothing "
            "(the comparison point)' -- the example count matches exactly. rung=1, "
            "dataset_sha=c8b89f4d... (the shipped headline export's train_id_set_hash)."
        ),
    },
    "Stop-weight": {
        "model_id": "qwen3-8b-sft-sw01",
        "is_candidate": True,
        "evidence": (
            "Same dataset_sha as Reference (c8b89f4d..., the identical 14,302-decision corpus) "
            "but a DIFFERENT training_manifest_sha (a11dda4e... vs 7220e885...) -- same data, "
            "different recipe, matching appendix_training.tex:27 'stop rows down-weighted to "
            "0.1'. Name 'sw01' = stop-weight 0.1; siblings sw025/sw05 exist at rung 1 with the "
            "same dataset_sha, matching the caption's cross-reference to Table~train-lever's "
            "'0.1 row' as one cell read twice with the Stop-weight row here."
        ),
    },
    "Persistence-only": {
        "model_id": "qwen3-8b-dpo-headline-notdone-only",
        "is_candidate": True,
        "evidence": (
            "rung=2, init_from=qwen3-8b-sft-headline (Reference), matching "
            "appendix_training.tex:30 'Reference | ... | preference on stop contrasts only'. "
            "dataset_sha d6750e34... shared only with notdone-both (the 'notdone' pair-export "
            "family); 'only' = the 18,919 synthesised contrasts and nothing else, matching the "
            "text: 'carries no signal at all about which of two questions is better'."
        ),
    },
    "Mixed": {
        "model_id": "qwen3-8b-dpo-headline-notdone-both",
        "is_candidate": True,
        "evidence": (
            "rung=2, init_from=qwen3-8b-sft-headline (Reference), matching "
            "appendix_training.tex:31 'Reference | ... | preference on both kinds at once'. "
            "dataset_sha d6750e34... IDENTICAL to Persistence-only's (same 'notdone' export "
            "family) and to Stacked's -- 'both' = both the persistence and the question-pair "
            "signal, i.e. the 31,473-pair Mixed set."
        ),
    },
    "Stacked": {
        "model_id": "qwen3-8b-dpo-stacked-notdone-both",
        "is_candidate": True,
        "evidence": (
            "rung=2, dataset_sha d6750e34... IDENTICAL to Mixed's (the SAME 31,473-pair file, "
            "matching appendix_training.tex:32 'the Mixed set, from a trained start'), but "
            "init_from=qwen3-8b-dpo-headline-control -- NOT qwen3-8b-sft-headline. That is the "
            "one structural difference the paper describes between Mixed and Stacked ('differ "
            "only in the policy they start from'), reproduced exactly in the registry's lineage "
            "graph: Stacked's init_from IS the node the paper calls Question-preference."
        ),
    },
    "Question-preference": {
        "model_id": "qwen3-8b-dpo-pooled-control",
        "is_candidate": False,
        "evidence": (
            "tab:train-prov's own caption: 'The Question-preference row at 8B is fitted to the "
            "pooled preference export rather than the export the Stacked arm's own starting "
            "policy used, so it is the same intervention on a sibling corpus and NOT the "
            "identical starting checkpoint.' The literal node Stacked starts from "
            "(qwen3-8b-dpo-headline-control, init_from=qwen3-8b-sft-headline, dataset_sha "
            "c1444a776...) is therefore NOT what tab:train-coverage's Question-preference row "
            "reads. qwen3-8b-dpo-pooled-control is rung=2, init_from=qwen3-8b-sft (the pooled "
            "family's own base), dataset_sha f3632ac0... (the pooled export) -- the disclosed "
            "sibling. Measured 2026-09-18: qwen3-8b-dpo-pooled-control has ZERO split=test rows "
            "on any suite (dev only, 132 musique + 333 strategyqa, no wiki2 at all), so this "
            "row is reported from dev only in the held-out table; qwen3-8b-dpo-headline-control "
            "(the literal Stacked parent) already has full split=test coverage under ITS OWN "
            "identity and is named separately as an uncounted opportunity, not conflated with "
            "this row."
        ),
    },
}

# Registry names present but explicitly OUT OF SCOPE for this lane, with the reason, so a
# reader does not have to re-derive why they are absent from ARM_MAP.
OUT_OF_SCOPE: dict[str, str] = {
    "qwen3-8b-sft-headline-stop": (
        "Same training_manifest_sha and dataset_sha as qwen3-8b-sft-headline (Reference) -- "
        "the SAME training run, a different checkpoint SELECTION. conf/checkpoints.json "
        "_second_row_added: 'best/by_nll selects that same final tag ... best/by_stop2x2 "
        "selects step 1200'; docs/serve/README.md Lane L0.3: 'qwen3-8b-sft-headline-stop "
        "existed only as rung1-8b-headline/best/by_stop2x2 (TAG=1200)'. Not a row of "
        "tab:train-recipes or tab:train-coverage. Dev-only (measured), fills no table cell; "
        "flagged for Fable rather than spent on unilaterally."
    ),
    "qwen3-8b-sft-headline-s1": "Seed replicate of Reference. Owned by another lane (task brief).",
    "qwen3-8b-sft-headline-s2": "Seed replicate of Reference. Owned by another lane (task brief).",
}


def assert_non_vacuous(
    parquet_dir: Path,
    runs_root: Path,
    run_ids: Sequence[str],
    *,
    scorer_hash: str,
) -> dict[str, Any]:
    """The coordinator's rule, added 2026-09-18: before trusting a contrast over this
    population, confirm (a) every run whose `turns.jsonl` is non-empty ON DISK also has
    `n_turns > 0` in the parquet just built from it, and (b) among runs that asked at least
    one question, `evidence_coverage` is not identically zero. Both are cheap, and both are
    exactly what the lost-rows compaction defect would produce: a real `turns.jsonl` next to a
    zero-stamped parquet row, or asks that score no evidence at all.

    Raises `Vacuous` rather than returning a bool -- a caller that reads a return value can
    forget to check it; a raise cannot be skipped silently.
    """
    con = _con(Path(parquet_dir))
    ids = list(run_ids)
    rows = _rows(con, f"SELECT run_id, n_turns, n_asks FROM runs WHERE run_id IN {_in(ids)}")
    by_id = {str(r["run_id"]): r for r in rows}
    runs_root = Path(runs_root)

    zero_turns_nonempty_file = []
    for rid in ids:
        turns_path = runs_root / rid / "turns.jsonl"
        file_nonempty = turns_path.exists() and turns_path.stat().st_size > 0
        n_turns = int((by_id.get(rid) or {}).get("n_turns") or 0)
        if file_nonempty and n_turns == 0:
            zero_turns_nonempty_file.append(rid)

    cov = _metric_by_run(con, "evidence_coverage", scorer_hash=scorer_hash)
    asking = [rid for rid in ids if int((by_id.get(rid) or {}).get("n_asks") or 0) > 0]
    nonzero_cov_among_asking = [rid for rid in asking if (cov.get(rid) or 0.0) > 0.0]

    report = {
        "n_run_ids": len(ids),
        "n_missing_from_runs_table": len(ids) - len(rows),
        "n_zero_turns_but_nonempty_file": len(zero_turns_nonempty_file),
        "zero_turns_but_nonempty_file_sample": zero_turns_nonempty_file[:20],
        "n_asking": len(asking),
        "n_nonzero_coverage_among_asking": len(nonzero_cov_among_asking),
    }
    if zero_turns_nonempty_file:
        raise Vacuous(
            f"{len(zero_turns_nonempty_file)} of {len(ids)} runs have n_turns=0 in the "
            f"parquet but a non-empty turns.jsonl on disk -- the lost-rows compaction defect. "
            f"report={report}"
        )
    if asking and not nonzero_cov_among_asking:
        raise Vacuous(
            f"{len(asking)} runs asked at least one question and NONE has nonzero "
            f"evidence_coverage. report={report}"
        )
    return report


def select_and_contrast(
    parquet_dir: Path,
    *,
    checkpoint_model_id: str,
    baseline_model_id: str,
    checkpoint_arm: str = "inquirer_trained",
    baseline_arm: str = "inquirer_prompted",
    grid_name: str = "tier1_trained_qa_base",
    baseline_grid_names: Sequence[str] = ("tier1_trained_qa_base",),
    scorer_hash: str,
    seed: int = 0,
    n_resamples: int = 10_000,
) -> dict[str, Any]:
    """The matched-cost AND cap-8 contrast for ONE checkpoint against ONE baseline pin.

    Thin wrapper around ``_select_runs`` + ``_by_key`` + ``_matched_cost`` -- the exact
    functions ``run_gate`` composes -- with an explicit ``model_id`` filter on the CHECKPOINT
    side that ``run_gate`` itself does not offer (see module docstring).
    """
    con = _con(Path(parquet_dir))
    ckpt = _select_runs(con, arm=checkpoint_arm, grids=[grid_name], model_id=checkpoint_model_id)
    base = _select_runs(
        con, arm=baseline_arm, grids=list(baseline_grid_names), model_id=baseline_model_id
    )
    ck_keys, ba_keys = _by_key(ckpt), _by_key(base)
    result = _matched_cost(
        con,
        ckpt=ckpt,
        base=base,
        ck_keys=ck_keys,
        ba_keys=ba_keys,
        scorer_hash=scorer_hash,
        seed=seed,
        n_resamples=n_resamples,
    )
    result["selection"] = {
        "checkpoint_model_id": checkpoint_model_id,
        "baseline_model_id": baseline_model_id,
        "checkpoint_arm": checkpoint_arm,
        "baseline_arm": baseline_arm,
        "grid_name": grid_name,
        "baseline_grid_names": list(baseline_grid_names),
        "scorer_hash": scorer_hash,
        "seed": seed,
        "n_resamples": n_resamples,
        "n_checkpoint_runs": len(ckpt),
        "n_baseline_runs": len(base),
        "n_checkpoint_keys": len(ck_keys),
        "n_baseline_keys": len(ba_keys),
    }
    return result


# --------------------------------------------------------------------------------------------
# Coordinator's rule (added mid-lane): every printed interval is read at 10k resamples; any
# bound within NEAR_ZERO_TOL of zero is ALSO read at 50k resamples across three bootstrap
# seeds, and a sign that is not stable across those reads is reported as undecided rather than
# as a pass or fail. Mirrors appendix_training.tex's own practice after tab:train-coverage:
# "seven distinct cells carry a gated lower bound within 0.01 of zero ... Four of the seven
# give the same sign on every seed at every count. Three do not."
NEAR_ZERO_TOL = 0.01
RESEED_N_RESAMPLES = 50_000
RESEEDS: tuple[int, ...] = (0, 1, 2)


def stability_recheck(
    parquet_dir: Path,
    *,
    checkpoint_model_id: str,
    baseline_model_id: str,
    by_suite: dict[str, dict[str, Any]],
    checkpoint_arm: str = "inquirer_trained",
    baseline_arm: str = "inquirer_prompted",
    grid_name: str = "tier1_trained_qa_base",
    baseline_grid_names: Sequence[str] = ("tier1_trained_qa_base",),
    scorer_hash: str,
    tol: float = NEAR_ZERO_TOL,
    big_n: int = RESEED_N_RESAMPLES,
    reseeds: Sequence[int] = RESEEDS,
) -> dict[str, dict[str, Any]]:
    """For every suite whose 10k-resample CI bound sits within ``tol`` of zero, re-derive the
    interval at ``big_n`` resamples across ``reseeds`` and report whether the near-zero bound's
    SIGN is stable. Suites whose bounds are not near zero are reported ``checked: False`` and
    are not re-derived (the coordinator's rule only re-reads a bound close enough to matter).
    """
    out: dict[str, dict[str, Any]] = {}
    for suite, cell in by_suite.items():
        lo, hi = cell.get("ci_lo"), cell.get("ci_hi")
        near = [b for b in (lo, hi) if b is not None and abs(b) <= tol]
        if not near:
            out[suite] = {"checked": False, "stable": True, "verdict": "as computed"}
            continue
        signs: list[int] = []
        reads: list[dict[str, float]] = []
        for rs in reseeds:
            r = select_and_contrast(
                parquet_dir,
                checkpoint_model_id=checkpoint_model_id,
                baseline_model_id=baseline_model_id,
                checkpoint_arm=checkpoint_arm,
                baseline_arm=baseline_arm,
                grid_name=grid_name,
                baseline_grid_names=baseline_grid_names,
                scorer_hash=scorer_hash,
                seed=rs,
                n_resamples=big_n,
            )
            cell_rs = r["by_suite"][suite]
            reads.append({"seed": rs, "ci_lo": cell_rs["ci_lo"], "ci_hi": cell_rs["ci_hi"]})
            # The bound that was near zero at 10k is the one whose SIGN decides the verdict;
            # track it specifically rather than both bounds, so a comfortably-negative lower
            # bound next to a near-zero upper bound cannot mask the upper bound's own flip.
            bound_rs = cell_rs["ci_lo"] if (lo is not None and abs(lo) <= tol) else cell_rs["ci_hi"]
            signs.append(1 if bound_rs >= 0 else -1)
        stable = len(set(signs)) == 1
        out[suite] = {
            "checked": True,
            "near_zero_bound": "ci_lo" if (lo is not None and abs(lo) <= tol) else "ci_hi",
            "reads_at_50k": reads,
            "signs": signs,
            "stable": stable,
            "verdict": "as computed" if stable else "undecided",
        }
    return out


# --------------------------------------------------------------------------------------------
# Arm-versus-arm: `_matched_cost` reads the CHECKPOINT at its own full terminal value
# (`evidence_coverage`, gate.py:988/1004: ``k = n_asks.get(c_rid, 0)`` then
# ``deltas.append(float(c_val) - _mean(rungs))`` with no truncation of ``c_val``) and the
# BASELINE at ``gate.py:994`` ``at = min(k, b_n)`` -- truncated DOWN to the checkpoint's k only
# when the baseline asks at least that many questions. That is a matched comparison exactly when
# the checkpoint spends LESS than the baseline (every headline contrast: trained ~2-3 against
# prompted ~6 or teacher ~5.3). Pointed the other way -- a HIGHER-asking arm as "checkpoint"
# against a lower-asking arm as "baseline" -- the baseline is never truncated (`min(k, b_n) = b_n`
# whenever `b_n < k`) and the checkpoint keeps its own full value: both sides land at their own
# natural stopping point, which is not a matched-cost reading at all. Measured 2026-09-18: this
# is not only an arm-vs-arm hazard. Persistence-only's OWN contrast against the plain
# `qwen3-8b-base` baseline in this lane's main table (ss5) has `n_baseline_shorter_than_k` of
# 116/200 (MuSiQue), 138/200 (StrategyQA, 69%), 90/200 (wiki2) -- Persistence-only's mean k
# (6.4-6.9) sits AT OR ABOVE the untrained baseline's own mean asking rate (6.0-6.1) on two of
# three suites, so the majority of StrategyQA task-pairs in that "matched-cost" cell are actually
# reading both arms at their own full length, not at a shared matched budget.
#
# `select_and_contrast_symmetric` is the fix for exactly this direction: BOTH sides are read off
# their own coverage ladder at ``min(k_a, k_b)``, so neither ever keeps an untruncated value the
# other side does not also get truncated to. Perfectly antisymmetric by construction
# (``min(k_a, k_b) == min(k_b, k_a)``), so `arm_a` vs `arm_b` is exactly the negative of `arm_b`
# vs `arm_a` -- unlike `select_and_contrast`, which is not antisymmetric because only one side is
# ever truncated.
def select_and_contrast_symmetric(
    parquet_dir: Path,
    *,
    arm_a_model_id: str,
    arm_b_model_id: str,
    arm_a_arm: str = "inquirer_trained",
    arm_b_arm: str = "inquirer_trained",
    grid_name: str = "tier1_trained_qa_base",
    scorer_hash: str,
    seed: int = 0,
    n_resamples: int = 10_000,
) -> dict[str, Any]:
    """Delta = arm_a's coverage minus arm_b's, BOTH read at ``min(k_a, k_b)`` per paired unit."""
    con = _con(Path(parquet_dir))
    a_runs = _select_runs(con, arm=arm_a_arm, grids=[grid_name], model_id=arm_a_model_id)
    b_runs = _select_runs(con, arm=arm_b_arm, grids=[grid_name], model_id=arm_b_model_id)
    a_keys, b_keys = _by_key(a_runs), _by_key(b_runs)
    shared = sorted(set(a_keys) & set(b_keys))

    n_asks = {str(r["run_id"]): int(r["n_asks"] or 0) for r in _with_stop(con, [*a_runs, *b_runs])}
    a_used = sorted({rid for key in shared for rid in a_keys[key]})
    b_used = sorted({rid for key in shared for rid in b_keys[key]})
    a_ladders = _coverage_ladder(con, [{"run_id": rid} for rid in a_used], scorer_hash=scorer_hash)
    b_ladders = _coverage_ladder(con, [{"run_id": rid} for rid in b_used], scorer_hash=scorer_hash)

    per_task: dict[tuple, list[float]] = {}
    k_common_by_suite: dict[str, list[float]] = {}
    for key in shared:
        suite = str(key[0])
        deltas = []
        for a_rid in a_keys[key]:
            k_a = n_asks.get(a_rid, 0)
            for b_rid in b_keys[key]:
                k_b = n_asks.get(b_rid, 0)
                k_common = min(k_a, k_b)
                val_a = (a_ladders.get(a_rid) or {}).get(k_common)
                val_b = (b_ladders.get(b_rid) or {}).get(k_common)
                if val_a is None or val_b is None:
                    continue
                deltas.append(float(val_a) - float(val_b))
                k_common_by_suite.setdefault(suite, []).append(float(k_common))
        if deltas:
            per_task.setdefault((suite, key[1]), []).append(_mean(deltas))

    by_task = {k: _mean(v) for k, v in per_task.items()}
    by_suite: dict[str, Any] = {}
    for suite in sorted({k[0] for k in by_task}):
        vals = [v for k, v in by_task.items() if k[0] == suite]
        point, lo, hi = bca_ci(vals, seed=seed, n_resamples=n_resamples)
        by_suite[suite] = {
            "n_tasks": len(vals),
            "delta": point,
            "ci_lo": lo,
            "ci_hi": hi,
            "mean_k_common": _mean(k_common_by_suite.get(suite, [0.0])),
        }
    return {
        "by_suite": by_suite,
        "selection": {
            "arm_a_model_id": arm_a_model_id,
            "arm_b_model_id": arm_b_model_id,
            "grid_name": grid_name,
            "scorer_hash": scorer_hash,
            "seed": seed,
            "n_resamples": n_resamples,
            "n_a_runs": len(a_runs),
            "n_b_runs": len(b_runs),
        },
    }
