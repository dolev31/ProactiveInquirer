"""The paired-fork engagement report: the protocol behind the paper's headline, re-runnable.

WHAT IS MEASURED. A fork point is a reference trajectory prefix (`foreign_trace_sha`,
`foreign_prefix_k`). At each fork point and seed, one treatment continuation is paired with one
control continuation. Follow-ups are the customer turns the continuation needed beyond the
prefix: `n_user_turns - n_prefix_user_turns`. The difference (treatment - control) is averaged
per seed and pooled; a two-sided exact sign test counts pairs where the treatment needed fewer
against pairs where it needed more; `tau_reward` is reported beside it, never folded in.

THE PAIR IS NOT THE INDEPENDENT UNIT, AND `paired_engagement`'s POOLED p TREATS IT AS ONE.
`pooled.sign_test_p` counts every (trace, k, seed) pair as one exchangeable observation.
Measured over the runs the paper reports, by `scripts/report_forks.py --clusters`:

    tau2_retail  test   102 pairs   34 fork points   32 traces   25 tasks
    tau2_airline test   102 pairs   34 fork points   30 traces   17 tasks

Three seeds of one prefix are three re-runs of one prefix, and two prefixes cut from one
recorded dialogue are two views of one dialogue. Counting 95 informative pairs where there are
32 dialogues does not make the evidence stronger; it makes the statement about it wrong.
`clustered_engagement` recomputes the same contrast at each of those four resolutions and names
the unit beside every number, so which one a table quotes is a visible choice. Use it, not
`pooled.sign_test_p`, for anything published.

The estimand at a level is the UNWEIGHTED MEAN OVER CLUSTERS, matching
`pi_eval.stats.inference.cluster_bootstrap` -- so the point, the interval and both p-values
describe one quantity. A pooled mean would let the one retail trace carrying two prefixes vote
twice while a single-prefix trace votes once.

WHAT THE PREFIX-FALLBACK DEFECT DOES AND DOES NOT TOUCH. `pi_run.compact` only recently began
writing `n_prefix_user_turns` to the parquet, and `pi_eval.metrics.environment.user_followups`
never reads it: on the parquet path a fork's follow-ups are `n_user_turns - 1`, charging it for
its own prefix. THIS REPORT IS NOT ON THAT PATH -- `scripts/report_forks.py` reads `status.json`
directly, where the field is present on all 408 published fork runs (measured: zero null, zero
zero). And the paired difference is immune either way: the two arms at a fork point continue
from the SAME prefix, so the subtrahend is a within-pair constant and cancels exactly. Measured
on both suites: 0/102 fork points where the arms disagree on `n_prefix_user_turns`, and the 102
per-pair differences are bit-identical under both formulas. The LEVELS are not immune -- under
the fallback both arms inflate by exactly the mean prefix (retail +2.2647, airline +1.4118) --
which is why a level is only ever reported here beside the difference that survives it.

WHY IT LIVES HERE AND NOT IN `report.py`. `ELIGIBLE` excludes forks by design (they inherit a
foreign prefix and must never pool with our own arms under one task_id). This protocol compares
forks with forks, paired at the same prefix, so the exclusion does not apply -- but neither do the
main tables' provenance records, so this report writes its own: run ids, code versions, seeds.
"""

from __future__ import annotations

import hashlib
import math
import statistics
from typing import Any, Iterable, Mapping, Sequence

from pi_eval.stats.inference import paired_difference

# alpha=0.05 two-sided, power=0.80. Spelled out rather than imported so the floor below is
# readable without chasing a constant: 1.959964 + 0.841621.
_Z_ALPHA_2 = 1.959963984540054
_Z_POWER = 0.8416212335729143


def sign_test_p(fewer: int, more: int) -> float:
    """Two-sided exact sign test over the informative pairs (ties dropped). NaN when none."""
    n = fewer + more
    if n == 0:
        return float("nan")
    k = min(fewer, more)
    p = 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, p)


def follow_ups(run: Mapping[str, Any]) -> int:
    return int(run.get("n_user_turns") or 0) - int(run.get("n_prefix_user_turns") or 0)


def _pair_forks(
    runs: Sequence[Mapping[str, Any]], *, treatment: str, control: str
) -> tuple[dict[tuple, dict[str, Mapping[str, Any]]], int, int]:
    """Pair runs on (foreign_trace_sha, foreign_prefix_k, seed); one run per arm per key.

    A key with two runs of one arm is refused rather than picked from: which one "counts"
    would be a choice the reader cannot see. Runs whose status is not ok are ignored.

    ONE PAIRING, TWO REPORTS. `paired_engagement` and `clustered_engagement` must agree on
    which 102 pairs exist or their p-values are not comparable, so the pairing lives here and
    is called twice rather than written twice. A hand copy has already cost this repository a
    silent four-field divergence between `run_tau2_unit` and `worker.run_unit`.

    Returns (complete pairs, unpaired run count, non-fork run count).
    """
    by_key: dict[tuple, dict[str, Mapping[str, Any]]] = {}
    n_not_forks = 0
    for r in runs:
        if r.get("status") not in (None, "ok") or r.get("arm_id") not in (treatment, control):
            continue
        # FORKS ONLY. A run with no foreign prefix is a full-task run, not a continuation; it
        # is outside this protocol, and two of them would collide on the empty key.
        if not r.get("foreign_trace_sha"):
            n_not_forks += 1
            continue
        key = (r.get("foreign_trace_sha"), r.get("foreign_prefix_k"), r.get("seed"))
        slot = by_key.setdefault(key, {})
        if r["arm_id"] in slot:
            raise ValueError(
                f"two {r['arm_id']} runs at fork {key}: {slot[r['arm_id']]['run_id']} and {r['run_id']}"
            )
        slot[r["arm_id"]] = r
    pairs = {k: v for k, v in by_key.items() if len(v) == 2}
    n_unpaired = sum(len(v) for v in by_key.values() if len(v) != 2)
    return pairs, n_unpaired, n_not_forks


def paired_engagement(
    runs: Sequence[Mapping[str, Any]], *, treatment: str, control: str
) -> dict[str, Any]:
    """The report as published: per seed and pooled over pairs.

    `pooled.sign_test_p` IS THE ANTICONSERVATIVE NUMBER described in the module docstring. It is
    kept, unchanged, so the recomputation can be printed beside the value it replaces rather
    than quietly substituted for it. Anything published takes its p from
    `clustered_engagement`.
    """
    pairs, n_unpaired, n_not_forks = _pair_forks(runs, treatment=treatment, control=control)
    per_seed: dict[Any, dict[str, Any]] = {}
    pooled_diff: list[float] = []
    codes: dict[str, int] = {}
    ids: list[str] = []
    for seed in sorted({k[2] for k in pairs}, key=lambda x: (x is None, x)):
        ps = [v for k, v in pairs.items() if k[2] == seed]
        t = [follow_ups(v[treatment]) for v in ps]
        c = [follow_ups(v[control]) for v in ps]
        d = [a - b for a, b in zip(t, c)]
        pooled_diff += d
        for v in ps:
            for arm in (treatment, control):
                codes[str(v[arm].get("code_version") or "")[:7]] = (
                    codes.get(str(v[arm].get("code_version") or "")[:7], 0) + 1
                )
                ids.append(str(v[arm]["run_id"]))
        fewer, more = sum(x < 0 for x in d), sum(x > 0 for x in d)
        per_seed[seed] = {
            "n_pairs": len(ps),
            "follow_ups": {"treatment": statistics.mean(t), "control": statistics.mean(c)},
            "diff_mean": statistics.mean(d),
            "fewer": fewer,
            "more": more,
            "sign_test_p": sign_test_p(fewer, more),
            "reward": {
                "treatment": statistics.mean(
                    float(v[treatment].get("tau_reward") or 0.0) for v in ps
                ),
                "control": statistics.mean(float(v[control].get("tau_reward") or 0.0) for v in ps),
            },
        }
    fewer, more = sum(x < 0 for x in pooled_diff), sum(x > 0 for x in pooled_diff)
    allp = list(pairs.values())
    pooled = {
        "n_pairs": len(allp),
        "diff_mean": statistics.mean(pooled_diff) if pooled_diff else float("nan"),
        "fewer": fewer,
        "more": more,
        "sign_test_p": sign_test_p(fewer, more),
        "reward": {
            "treatment": statistics.mean(float(v[treatment].get("tau_reward") or 0.0) for v in allp)
            if allp
            else float("nan"),
            "control": statistics.mean(float(v[control].get("tau_reward") or 0.0) for v in allp)
            if allp
            else float("nan"),
        },
    }
    return {
        "treatment": treatment,
        "control": control,
        "n_pairs": len(allp),
        "n_unpaired": n_unpaired,
        "n_not_forks": n_not_forks,
        "per_seed": per_seed,
        "pooled": pooled,
        "provenance": {
            "code_versions": codes,
            "seeds": sorted(per_seed, key=lambda x: (x is None, x)),
            "run_ids_sha": hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()[:16],
            "n_runs": len(ids),
        },
    }


# ------------------------------------------------------------------- clustering

#: cluster_by -> (level name, human name of the unit). Ordered coarsest-last, so a report reads
#: down from the resolution the paper published to the resolution the data actually has.
_LEVELS: tuple[tuple[str, str], ...] = (
    ("pair", "(foreign_trace_sha, foreign_prefix_k, seed)"),
    ("fork_point", "(foreign_trace_sha, foreign_prefix_k)"),
    ("trace", "foreign_trace_sha"),
    ("task", "task_id"),
)


def _cluster_of(key: tuple, run: Mapping[str, Any], level: str) -> str:
    """The cluster id a pair belongs to at `level`. Pairs are keyed (trace, k, seed)."""
    if level == "pair":
        return repr(key)
    if level == "fork_point":
        return repr(key[:2])
    if level == "trace":
        return str(key[0])
    if level == "task":
        # `task_id` comes off the RUN, not the key: a trace belongs to exactly one task (measured
        # 0 traces mapping to >1 task on both suites), but the key does not carry it.
        return str(run.get("task_id"))
    raise ValueError(f"unknown cluster level {level!r}")


def clustered_engagement(
    runs: Sequence[Mapping[str, Any]],
    *,
    treatment: str,
    control: str,
    n_boot: int = 10_000,
    n_perm: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    """The same paired-fork contrast at four clustering resolutions, each one named.

    At every level the per-pair differences are averaged WITHIN a cluster and the cluster mean is
    the observation. Three numbers come out of each level and all three are at that level:

      * `sign_test_p` -- the two-sided EXACT sign test on the cluster means. This is the
        like-for-like replacement for `pooled.sign_test_p`: same test, coarser unit.
      * `estimate` -- `pi_eval.stats.inference.paired_difference` with the cluster map, i.e. a
        BCa interval resampling clusters and a cluster-level sign-FLIP permutation p. Reported
        beside the exact test because they answer the same question two ways; if they disagree
        the disagreement is the finding, not a detail to pick between.
      * `diff_mean` -- the unweighted mean over clusters, the estimand both of the above use.

    The `pair` level reproduces the published pooled number exactly, which is the point: nothing
    here is a different measurement, only a differently counted one.
    """
    pairs, n_unpaired, n_not_forks = _pair_forks(runs, treatment=treatment, control=control)
    diffs = {k: float(follow_ups(v[treatment]) - follow_ups(v[control])) for k, v in pairs.items()}
    levels: dict[str, Any] = {}
    for level, unit in _LEVELS:
        cmap = {repr(k): _cluster_of(k, pairs[k][treatment], level) for k in pairs}
        groups: dict[str, list[float]] = {}
        for k, d in diffs.items():
            groups.setdefault(cmap[repr(k)], []).append(d)
        means = [statistics.fmean(v) for v in groups.values()]
        fewer, more = sum(x < 0 for x in means), sum(x > 0 for x in means)
        est = paired_difference(
            {repr(k): float(follow_ups(pairs[k][treatment])) for k in pairs},
            {repr(k): float(follow_ups(pairs[k][control])) for k in pairs},
            clusters=cmap,
            n_boot=n_boot,
            n_perm=n_perm,
            seed=seed,
        )
        levels[level] = {
            "unit": unit,
            "n_units": len(groups),
            "n_pairs": len(diffs),
            "diff_mean": statistics.fmean(means) if means else float("nan"),
            "fewer": fewer,
            "more": more,
            "ties": len(means) - fewer - more,
            "sign_test_p": sign_test_p(fewer, more),
            "estimate": {
                "point": est.point,
                "ci_lo": est.ci_lo,
                "ci_hi": est.ci_hi,
                "sign_flip_p": est.p_value,
                "n_clusters": len(groups),
                "method": est.method,
                "note": est.note,
            },
        }
    ids = sorted(str(v[arm]["run_id"]) for v in pairs.values() for arm in (treatment, control))
    return {
        "treatment": treatment,
        "control": control,
        "n_pairs": len(pairs),
        "n_unpaired": n_unpaired,
        "n_not_forks": n_not_forks,
        "levels": levels,
        "published_unit": "pair",
        "recomputed_unit": "trace",
        "provenance": {
            "code_versions": _code_census(pairs, (treatment, control)),
            "seeds": sorted({k[2] for k in pairs}, key=lambda x: (x is None, x)),
            "run_ids_sha": hashlib.sha256("\n".join(ids).encode()).hexdigest()[:16],
            "n_runs": len(ids),
            "prefix_field_present": all(
                v[arm].get("n_prefix_user_turns") is not None
                for v in pairs.values()
                for arm in (treatment, control)
            ),
            "arms_share_the_prefix_at_every_fork_point": all(
                v[treatment].get("n_prefix_user_turns") == v[control].get("n_prefix_user_turns")
                for v in pairs.values()
            ),
        },
    }


def _code_census(
    pairs: Mapping[tuple, Mapping[str, Mapping[str, Any]]], arms: Sequence[str]
) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in pairs.values():
        for arm in arms:
            k = str(v[arm].get("code_version") or "")[:7]
            out[k] = out.get(k, 0) + 1
    return out


# --------------------------------------------------- duplicated-cell selection


def select_one_per_cell(
    rows: Iterable[Mapping[str, Any]],
    *,
    prefer_code_version: str,
    cell_fields: Sequence[str] = ("task_id", "seed", "arm_id"),
) -> dict[str, Any]:
    """Keep ONE row per (task, seed, arm) cell, by a rule that cannot read the outcome.

    WHY A RULE AND NOT A MEAN. A population with a duplicated cell mixes implementations, and
    averaging the duplicates does not unmix them -- it silently weights the cells that happen to
    have been re-run twice as heavily as the cells that were not. Measured on the airline
    eligible test population: 31 of 198 cells carry two rows, every one of them a
    `5800f8db`/`4b8b842c` pair, and NONE differs in `model_pin_hash` or `grid_name`.

    THE RULE READS `code_version` AND NOTHING ELSE, then `run_id` to break a tie. It never reads
    a score, a reward, a token count or a status. That is the whole reason it is a rule: the
    majority version, the earliest version and the version with full task coverage are the same
    version here, so no judgement is exercised and none can be smuggled in later.

    DEDUPLICATION IS NOT RESTRICTION, and the difference is a population, not a nicety. A cell
    whose ONLY row is the non-preferred version is KEPT on a fallback -- it is one observation of
    that cell, and the cell is not less real for having been run once on the other version.
    Restricting the population to `prefer_code_version` instead deletes such cells outright:
    measured on the airline population that is 175 rows at 58/60/57 per arm, against 198 rows at
    66/66/66 here -- 20 tasks either way, so the cost is 23 whole CELLS and an arm imbalance, not
    task coverage. THE 58/60/57 IMBALANCE IS THAT DELETION AND NOTHING ELSE: 18 of the 23 are
    task 2 seeds 3-8, a six-seed block that only ever ran under `acf42554` (balanced 6/6/6, so
    not itself the imbalance), and the other 5 are task 13, where the `5800f8db` campaign ran
    `inquirer_may_ask_user` at all three seeds, `drafter_only` at one, and `inquirer_prompted` at
    none, and `4b8b842c` filled the gaps. That is a PARTIAL CAMPAIGN, not more duplicates.

    Returns the kept rows and THE RECORD: the rule as a sentence, the cells it touched, the
    cells kept on a fallback, and the counts in and out. A selection whose record is a comment
    is a selection a reader cannot audit.
    """
    cells: dict[tuple, list[Mapping[str, Any]]] = {}
    for r in rows:
        cells.setdefault(tuple(r.get(f) for f in cell_fields), []).append(r)
    kept: list[Mapping[str, Any]] = []
    touched: list[tuple] = []
    fallback: list[tuple] = []
    for key in sorted(cells, key=repr):
        group = cells[key]
        pref = [
            r for r in group if str(r.get("code_version") or "").startswith(prefer_code_version)
        ]
        pick = sorted(pref or group, key=lambda r: str(r.get("run_id")))[0]
        kept.append(pick)
        if len(group) > 1:
            touched.append(key)
        if not pref:
            fallback.append(key)
    n_in = sum(len(v) for v in cells.values())
    return {
        "rule": (
            f"one row per {'+'.join(cell_fields)} cell: prefer code_version {prefer_code_version} "
            "(the majority version, the earliest version, and the only version covering every "
            "task -- all three agree, so the choice needs no judgement); on a cell that version "
            "does not occupy, keep the lowest run_id. The rule reads code_version and run_id "
            "only, never an outcome."
        ),
        "prefer_code_version": prefer_code_version,
        "cell_fields": list(cell_fields),
        "n_rows_in": n_in,
        "n_rows_kept": len(kept),
        "n_dropped": n_in - len(kept),
        "n_cells": len(cells),
        "n_cells_touched": len(touched),
        "n_cells_kept_on_a_fallback": len(fallback),
        "cells_touched": [list(k) for k in touched],
        "fallback_cells": [list(k) for k in fallback],
        "kept": kept,
    }


# ---------------------------------------------------- minimum detectable effect


def minimum_detectable_effect(
    diffs_by_cluster: Mapping[str, float], *, alpha: float = 0.05, power: float = 0.80
) -> dict[str, Any]:
    """The smallest paired effect this many CLUSTERS can detect, from the OBSERVED paired sd.

        MDE = (z_{1-alpha/2} + z_{power}) * sd_d / sqrt(n_clusters)

    IN THE UNITS OF THE ENDPOINT, because `sd_d` is. No standardised effect size, no rule of
    thumb, no borrowed sd: `sd_d` is the sample sd of the per-cluster paired differences that
    were actually measured.

    THE n IS THE CLUSTER COUNT AND THAT IS THE ENTIRE POINT. The airline confirmatory endpoint
    reads as 229 runs; it is 20 tasks, and the replicate seeds within a task add precision to a
    task's own mean, not independent tasks. Computing this floor at 229 understates it by
    sqrt(229/20) = 3.4x, which is the difference between "underpowered" and "fine".

    Two-sided, normal approximation. At n=20 the exact paired-t value is roughly 6% larger, so
    this is the OPTIMISTIC floor -- if the answer is already "too big to be useful", the t
    correction cannot rescue it. NaN below two clusters: one observation has no variance, and
    inventing one is how an underpowered endpoint gets published as an adequate one.
    """
    vals = [float(v) for v in diffs_by_cluster.values() if not math.isnan(float(v))]
    n = len(vals)
    if n < 2:
        return {
            "alpha": alpha,
            "power": power,
            "n_clusters": n,
            "sd": float("nan"),
            "var": float("nan"),
            "observed_mean": vals[0] if vals else float("nan"),
            "mde": float("nan"),
            "method": "paired normal approximation, two-sided; NaN: fewer than 2 clusters",
        }
    sd = statistics.stdev(vals)
    return {
        "alpha": alpha,
        "power": power,
        "n_clusters": n,
        "sd": sd,
        "var": sd**2,
        "observed_mean": statistics.fmean(vals),
        "mde": (_Z_ALPHA_2 + _Z_POWER) * sd / math.sqrt(n),
        "method": (
            f"MDE = (z_{{1-alpha/2}}={_Z_ALPHA_2:.6f} + z_{{power}}={_Z_POWER:.6f}) * sd/sqrt(n); "
            "sd is the observed per-cluster paired sd; two-sided normal approximation "
            "(the exact paired-t floor is ~6% larger at n=20)"
        ),
    }
