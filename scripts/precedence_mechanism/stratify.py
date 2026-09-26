"""Stratified precedence-violation contrast: does the trained-vs-prompted gap concentrate on
parents the trained policy's own base can already answer closed-book?

THE ESTIMAND MATCHES `precedence_violation_rate` EXACTLY, restricted to a stratum of parent
nodes. Input is `events.QualifyingEdge` -- one row per prerequisite edge with BOTH parent and
child present in the arm's matched window, tagged `is_violation` -- which is that metric's
own `total`/`viol` population read out edge-by-edge instead of collapsed into one scalar per
run (see `events.py`'s docstring). Restricting to a stratum and re-aggregating reproduces the
suite-level matched delta when the stratum is "everything"; that equality is the lock in
`tests/test_precedence_mechanism.py`.

A STRATUM KEY IS `(task_id, parent_node_id)`, NEVER A BARE `parent_node_id`. MuSiQue node ids
are LOCAL to a task -- `_graph` in `pi_eval.build.musique_build` assigns every task's own
decomposition steps `s1, s2, s3, ...` from 1, so "s1" on one task and "s1" on another are
unrelated gold nodes that happen to share a string. MEASURED on the live matched-cost
population: 199 violating edges reduce to just 3 distinct BARE node ids ("s1".."s3") but 102
distinct `(task_id, node_id)` pairs (364 over the full qualifying-edge population, violated
or not). Stratifying, or caching a probe answer, on the bare id would silently pool dozens of
unrelated questions under one answerability verdict -- exactly the "pooled-vs-per-unit" defect
this repo's operating rules name.

A SECOND, DIFFERENT DEFECT LIVES ONE LEVEL UP FROM THIS FIX, RECORDED HERE SO IT IS NOT
REDISCOVERED. Probing answerability only for the 102 parents that appear in a VIOLATED edge
makes "answerable" true only where a violation already happened -- selected on its own
outcome. The stratum key above being correct does not protect against that; only probing
every one of the 364 qualifying-edge parents (violated or not) does. See
`artifacts/precedence_mechanism_20260918/RESULT.md`'s "Bug 2" for the retracted number this
produced (interaction `+0.2668` at n=17) and the corrected one that replaced it.

SEEDS ARE AVERAGED INTO THE TASK BEFORE THE TASK ENTERS THE BOOTSTRAP, exactly as
`testsplit_plan_metrics_20260918/RESULT.md`'s own estimator does: first a per-(task, seed)
rate, then the mean over that task's available seeds.

THE INTERACTION IS NOT A SUBTRACTION OF TWO PUBLISHED INTERVALS. Per the operating rule this
repo has hit before ("Matched-cost deltas do not subtract" -- two overlapping CIs are not a
difference test, and the converse), `interaction_bca` draws ONE joint resample of tasks per
replicate and recomputes BOTH stratum deltas from that same draw, so the reported interval is
of the actual estimator (delta_answerable - delta_not_answerable) and not an eyeballed gap
between two independently-resampled numbers.

EVERY INTERVAL IS RE-CHECKED FOR SIGN STABILITY WHEN A BOUND IS NEAR ZERO. Per the added
brief rule (measured: at 1k resamples 3 of 7 near-zero-gated cells flip a bound's sign across
bootstrap seeds, one printed pass dies at 10k): the primary read is 10,000 resamples, seed 0.
Any interval with a bound within 0.01 of zero is ALSO recomputed at 50,000 resamples across
three more seeds (1, 2, 3); if that bound's sign is not the same in all three, the estimate is
UNDECIDED and must not be read as a pass or a fail. `check_stability` and `StabilityCheck` do
this uniformly for both the per-stratum estimates and the interaction.
"""

from __future__ import annotations

import random
import statistics
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Sequence

# Acklam's normal-quantile approximation and the BCa percentile reader: reused rather than
# retyped, because a second hand-transcribed copy of `_phi_inv` is how a transcription bug
# would enter unnoticed. `cluster_bootstrap` in the same package is the only other user.
from pi_eval.stats.inference import Estimate, _percentile, _phi, _phi_inv, paired_difference

from .events import ARM_PROMPTED, ARM_TRAINED, QualifyingEdge

ParentKey = tuple[str, str]  # (task_id, parent_node_id)
Predicate = Callable[[ParentKey], bool]

NEAR_ZERO_TOL = 0.01
STABILITY_SEEDS = (1, 2, 3)
STABILITY_N_BOOT = 50_000


def task_rates(
    edges: Sequence[QualifyingEdge], arm: str, in_stratum: Predicate
) -> dict[str, float]:
    """{task_id: violation rate}, one arm, restricted to edges whose PARENT satisfies
    `in_stratum` on its `(task_id, parent_node_id)` key. Per-(task, seed) rate first, then
    the mean over that task's available seeds."""
    per_seed: dict[tuple[str, int], list[int]] = defaultdict(list)
    for e in edges:
        if e.arm != arm or not in_stratum((e.task_id, e.parent_node_id)):
            continue
        per_seed[(e.task_id, e.seed)].append(1 if e.is_violation else 0)
    seed_rates: dict[str, list[float]] = defaultdict(list)
    for (task_id, _seed), vals in per_seed.items():
        seed_rates[task_id].append(statistics.fmean(vals))
    return {t: statistics.fmean(v) for t, v in seed_rates.items()}


def n_qualifying(edges: Sequence[QualifyingEdge], arm: str, in_stratum: Predicate) -> int:
    return sum(1 for e in edges if e.arm == arm and in_stratum((e.task_id, e.parent_node_id)))


# --------------------------------------------------------------------------- stability


@dataclass(frozen=True, slots=True)
class StabilityCheck:
    checked: bool
    stable: bool | None  # None when the primary interval never triggered a check
    bounds: tuple[float, ...]  # the re-derived bound(s) at 50k, one per extra seed
    note: str


def check_stability(
    recompute: Callable[[int, int], tuple[float, float]],
    lo: float,
    hi: float,
    *,
    seeds: Sequence[int] = STABILITY_SEEDS,
    n_boot: int = STABILITY_N_BOOT,
    tol: float = NEAR_ZERO_TOL,
) -> StabilityCheck:
    """`recompute(n_boot, seed) -> (lo, hi)` re-run at a bigger, differently-seeded bootstrap.
    Checked only when the PRIMARY (10k, seed 0) interval has a bound within `tol` of zero --
    that is the regime the added rule measured as unreliable. `stable=False` means the sign of
    the near-zero bound moved across the three reseeds: the verdict on that interval is
    UNDECIDED, not a pass or a fail, regardless of which side of zero the primary read landed."""
    near_lo = abs(lo) <= tol
    near_hi = abs(hi) <= tol
    if not near_lo and not near_hi:
        return StabilityCheck(False, None, (), "neither bound within 0.01 of zero at 10k/seed0")

    los: list[float] = []
    his: list[float] = []
    for s in seeds:
        lo2, hi2 = recompute(n_boot, s)
        los.append(lo2)
        his.append(hi2)

    stable = True
    parts = []
    if near_lo:
        base_negative = lo <= 0
        flips = [(v <= 0) != base_negative for v in los]
        stable = stable and not any(flips)
        parts.append(f"lo@10k/0={lo:+.4f} lo@50k/{list(seeds)}={[f'{v:+.4f}' for v in los]}")
    if near_hi:
        base_negative = hi <= 0
        flips = [(v <= 0) != base_negative for v in his]
        stable = stable and not any(flips)
        parts.append(f"hi@10k/0={hi:+.4f} hi@50k/{list(seeds)}={[f'{v:+.4f}' for v in his]}")
    return StabilityCheck(True, stable, tuple(los + his), "; ".join(parts))


# --------------------------------------------------------------------------- per-stratum


@dataclass(frozen=True, slots=True)
class StratumResult:
    name: str
    n_edges_trained: int
    n_edges_prompted: int
    estimate: Estimate
    stability: StabilityCheck


def stratified_contrast(
    edges: Sequence[QualifyingEdge], answerable: frozenset[ParentKey]
) -> dict[str, StratumResult]:
    strata: dict[str, Predicate] = {
        "answerable": lambda p: p in answerable,
        "not_answerable": lambda p: p not in answerable,
    }
    out: dict[str, StratumResult] = {}
    for name, pred in strata.items():
        a = task_rates(edges, ARM_TRAINED, pred)
        b = task_rates(edges, ARM_PROMPTED, pred)
        est = paired_difference(a, b)  # primary: 10,000 resamples, seed 0 (this fn's default)

        def recompute(n_boot: int, seed: int, a=a, b=b) -> tuple[float, float]:
            e = paired_difference(a, b, n_boot=n_boot, seed=seed)
            return e.ci_lo, e.ci_hi

        stability = check_stability(recompute, est.ci_lo, est.ci_hi)
        out[name] = StratumResult(
            name=name,
            n_edges_trained=n_qualifying(edges, ARM_TRAINED, pred),
            n_edges_prompted=n_qualifying(edges, ARM_PROMPTED, pred),
            estimate=est,
            stability=stability,
        )
    return out


# --------------------------------------------------------------------------- interaction


@dataclass(frozen=True, slots=True)
class Interaction:
    point: float
    ci_lo: float
    ci_hi: float
    n_tasks: int
    stability: StabilityCheck


def _task_diffs(edges: Sequence[QualifyingEdge], in_stratum: Predicate) -> dict[str, float]:
    a = task_rates(edges, ARM_TRAINED, in_stratum)
    b = task_rates(edges, ARM_PROMPTED, in_stratum)
    return {t: a[t] - b[t] for t in sorted(set(a) & set(b))}


def _interaction_ci(
    diffs_a: dict[str, float],
    diffs_n: dict[str, float],
    *,
    n_boot: int,
    seed: int,
    alpha: float = 0.05,
) -> tuple[float, float, float, int]:
    all_tasks = sorted(set(diffs_a) | set(diffs_n))

    def theta(tasks: Sequence[str]) -> float:
        da = [diffs_a[t] for t in tasks if t in diffs_a]
        dn = [diffs_n[t] for t in tasks if t in diffs_n]
        if not da or not dn:
            return float("nan")
        return statistics.fmean(da) - statistics.fmean(dn)

    if not all_tasks:
        return float("nan"), float("nan"), float("nan"), 0

    theta_hat = theta(all_tasks)
    rng = random.Random(seed)
    n = len(all_tasks)
    # THE CALLER'S LIST ORDER IS NOT AN INPUT TO THE ANSWER: `all_tasks` is already a sorted
    # list of ids (strings sort deterministically), so indices drawn by the RNG are stable
    # across runs and across however `edges` happened to arrive -- the same discipline
    # `cluster_bootstrap` documents at length for the reason it matters.
    reps = [
        v
        for v in (theta([all_tasks[rng.randrange(n)] for _ in range(n)]) for _ in range(n_boot))
        if v == v
    ]
    if not reps:
        return theta_hat, float("nan"), float("nan"), n

    jack = [v for i in range(n) if (v := theta(all_tasks[:i] + all_tasks[i + 1 :])) == v]
    lo, hi = _bca_endpoints(theta_hat, reps, jack, alpha)
    return theta_hat, lo, hi, n


def interaction_bca(
    edges: Sequence[QualifyingEdge],
    answerable: frozenset[ParentKey],
    *,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Interaction:
    """BCa interval for (delta_answerable - delta_not_answerable). See module docstring for
    why this is a joint resample and not a subtraction of two `stratified_contrast` cells, and
    for the near-zero re-seed rule this also applies."""
    diffs_a = _task_diffs(edges, lambda p: p in answerable)
    diffs_n = _task_diffs(edges, lambda p: p not in answerable)

    point, lo, hi, n = _interaction_ci(diffs_a, diffs_n, n_boot=n_boot, seed=seed, alpha=alpha)

    def recompute(n_boot: int, seed: int) -> tuple[float, float]:
        _, lo2, hi2, _ = _interaction_ci(diffs_a, diffs_n, n_boot=n_boot, seed=seed, alpha=alpha)
        return lo2, hi2

    stability = (
        check_stability(recompute, lo, hi)
        if lo == lo and hi == hi
        else StabilityCheck(False, None, (), "nan")
    )
    return Interaction(point, lo, hi, n, stability)


def _bca_endpoints(
    theta_hat: float, reps: Sequence[float], jack: Sequence[float], alpha: float
) -> tuple[float, float]:
    prop = sum(1 for r in reps if r < theta_hat) / len(reps)
    z0 = _phi_inv(min(max(prop, 1e-9), 1 - 1e-9))
    if len(jack) > 1:
        jbar = statistics.fmean(jack)
        num = sum((jbar - x) ** 3 for x in jack)
        den = 6 * (sum((jbar - x) ** 2 for x in jack) ** 1.5)
        a = num / den if den else 0.0
    else:
        a = 0.0

    def adj(q: float) -> float:
        z = _phi_inv(q)
        denom = 1 - a * (z0 + z)
        return _phi(z0 + (z0 + z) / denom) if denom else q

    return _percentile(reps, adj(alpha / 2)), _percentile(reps, adj(1 - alpha / 2))


# --------------------------------------------------------------------------- logistic (secondary)


@dataclass(frozen=True, slots=True)
class LogisticResult:
    """`is_violation ~ arm * answerable`, cluster-robust (sandwich) SEs on task_id -- the
    standard `statsmodels` covariance estimator for a binary outcome with within-task
    correlation, reused rather than hand-rolled (the repo's own rule against a second,
    untested implementation of shared statistical machinery).

    `arm=1` is `inquirer_trained`, `answerable=1` is qwen3-8b-base >= 1.0 on that parent. The
    `arm:answerable` coefficient is the log-odds analogue of `interaction_bca`'s
    (answerable delta - not-answerable delta): if its CI excludes 0, the trained arm's excess
    violation rate is concentrated on answerable parents on the ODDS scale too.
    """

    n_edges: int
    n_tasks: int
    cells: dict[str, dict[str, int]]
    converged: bool
    coef: dict[str, float]
    se: dict[str, float]
    ci_lo: dict[str, float]
    ci_hi: dict[str, float]
    note: str

    def to_dict(self) -> dict:
        return {
            "n_edges": self.n_edges,
            "n_tasks": self.n_tasks,
            "cells": self.cells,
            "converged": self.converged,
            "coef": self.coef,
            "se": self.se,
            "ci_lo": self.ci_lo,
            "ci_hi": self.ci_hi,
            "note": self.note,
        }

    def __str__(self) -> str:
        if not self.coef:
            return f"n/a ({self.note})"
        term = "arm:answerable"
        c = self.coef.get(term, float("nan"))
        lo = self.ci_lo.get(term, float("nan"))
        hi = self.ci_hi.get(term, float("nan"))
        return (
            f"arm:answerable coef {c:+.4f} [{lo:+.4f}, {hi:+.4f}] "
            f"(converged={self.converged}, n_edges={self.n_edges}, n_tasks={self.n_tasks})"
        )


def logistic_violation_model(
    edges: Sequence[QualifyingEdge], answerable: frozenset[ParentKey]
) -> LogisticResult:
    """Secondary to the BCa contrast above, not a replacement: same population
    (`QualifyingEdge` rows), same two factors, a different functional form and a different
    standard-error estimator, so a reader can see whether the two agree rather than trusting
    either alone."""
    import pandas as pd
    import statsmodels.formula.api as smf

    rows = [
        {
            "is_violation": int(e.is_violation),
            "arm": 1 if e.arm == ARM_TRAINED else 0,
            "answerable": 1 if (e.task_id, e.parent_node_id) in answerable else 0,
            "task_id": e.task_id,
        }
        for e in edges
    ]
    df = pd.DataFrame(rows)

    cells: dict[str, dict[str, int]] = {}
    for (arm, ans), g in df.groupby(["arm", "answerable"]):
        name = f"arm={'trained' if arm else 'prompted'}/answerable={bool(ans)}"
        cells[name] = {"n_edges": int(len(g)), "n_tasks": int(g["task_id"].nunique())}

    n_edges = len(df)
    n_tasks = int(df["task_id"].nunique())

    if df.empty or df["is_violation"].nunique() < 2:
        return LogisticResult(
            n_edges, n_tasks, cells, False, {}, {}, {}, {}, "degenerate outcome, not fit"
        )

    try:
        fit = smf.logit("is_violation ~ arm * answerable", data=df).fit(
            disp=0, cov_type="cluster", cov_kwds={"groups": df["task_id"]}
        )
    except Exception as exc:  # noqa: BLE001 - reported, not swallowed
        return LogisticResult(
            n_edges, n_tasks, cells, False, {}, {}, {}, {}, f"{type(exc).__name__}: {exc}"
        )

    converged = bool(fit.mle_retvals.get("converged", True))
    ci = fit.conf_int()
    note = "" if converged else "optimizer did not report convergence; coefficients unreliable"
    return LogisticResult(
        n_edges=n_edges,
        n_tasks=n_tasks,
        cells=cells,
        converged=converged,
        coef={k: float(v) for k, v in fit.params.items()},
        se={k: float(v) for k, v in fit.bse.items()},
        ci_lo={k: float(v) for k, v in ci[0].items()},
        ci_hi={k: float(v) for k, v in ci[1].items()},
        note=note,
    )
