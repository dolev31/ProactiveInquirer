"""Inference machinery. Pure stdlib, so the whole statistics layer is testable without scipy.

Design commitments, each of which exists because the naive alternative fails a specific way:

* PAIRED throughout. Arms are run on the same tasks with common random numbers, so the
  correct null is exchangeability of the SIGN of the within-task difference, not equality of
  two independent means.
* Sign-flip PERMUTATION for p-values. It makes no distributional assumption, which matters
  because per-task quality differences are neither normal nor symmetric.
* Task-clustered BCa BOOTSTRAP for intervals, resampling TASKS. Resampling rollouts would
  understate the SE by roughly sqrt(1 + (m-1)*ICC); for tau2, whose tasks derive from a small
  number of templates, resampling must cluster at the TEMPLATE level or the SE is wrong by
  ~1.6x at m=4, ICC=0.5.
* EXACT McNemar for binary outcomes. The chi-square approximation is unreliable at the
  discordant-pair counts a 97-task suite produces.
* BH-FDR over a DECLARED secondary family. Everything outside the preregistered primary and
  that family is exploratory and gets a CI with no p-value at all.
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class Estimate:
    point: float
    ci_lo: float
    ci_hi: float
    p_value: float | None
    n: int
    method: str
    note: str = ""

    @property
    def significant(self) -> bool:
        return self.p_value is not None and self.p_value < 0.05

    def __str__(self) -> str:
        p = "n/a" if self.p_value is None else f"{self.p_value:.4f}"
        return f"{self.point:+.4f} [{self.ci_lo:+.4f}, {self.ci_hi:+.4f}] p={p} n={self.n}"


# ----------------------------------------------------------------- permutation


def sign_flip_p(diffs: Sequence[float], *, n_perm: int = 10_000, seed: int = 0) -> float:
    """Two-sided sign-flip permutation test on paired differences.

    Zero differences carry no sign information and are kept in the statistic but never
    flipped, which is the standard treatment and avoids inflating power on ties.
    """
    d = [x for x in diffs if not math.isnan(x)]
    if not d:
        return float("nan")
    obs = abs(statistics.fmean(d))
    rng = random.Random(seed)
    hits = 0
    for _ in range(n_perm):
        flipped = statistics.fmean([x if rng.random() < 0.5 else -x for x in d])
        if abs(flipped) >= obs - 1e-15:
            hits += 1
    # +1/+1 keeps the estimate strictly inside (0,1): a permutation p of exactly 0 is a
    # statement the test cannot make.
    return (hits + 1) / (n_perm + 1)


# ----------------------------------------------------------------- bootstrap


def _percentile(xs: Sequence[float], q: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    pos = q * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def _phi(z: float) -> float:
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def _phi_inv(p: float) -> float:
    """Acklam's rational approximation; accurate to ~1e-9, plenty for CI endpoints."""
    if p <= 0:
        return -math.inf
    if p >= 1:
        return math.inf
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    q = p - 0.5
    r = q * q
    return (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
        * q
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    )


def _nan_last(x: float) -> tuple[int, float]:
    """Sort key for one value. NaN compares false against everything, so a plain `sorted` of a
    list holding one falls back on the input order this module is removing."""
    return (1, 0.0) if math.isnan(x) else (0, x)


def _unit_order_key(u: Sequence[float]) -> tuple[int, tuple[tuple[int, float], ...]]:
    return (len(u), tuple(_nan_last(x) for x in u))


def cluster_bootstrap(
    units: Sequence[Sequence[float]],
    stat: Callable[[Sequence[float]], float] = statistics.fmean,
    *,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """BCa interval, resampling CLUSTERS (each `units[i]` is one task's or template's values).

    Bias-corrected and accelerated rather than plain percentile because the statistic here is
    a difference of correlated scores whose bootstrap distribution is routinely skewed, and
    percentile intervals under-cover in exactly that case.

    THE ANSWER DOES NOT DEPEND ON THE ORDER `units` ARRIVES IN. Units, and the values inside
    each unit, are canonicalised before resampling, because the RNG draws indices and the
    caller's order therefore used to move the endpoints -- see the comment on the sort. `stat`
    must be order-insensitive (a mean is); a unit is a multiset of that cluster's values.

    THE ESTIMAND IS THE UNWEIGHTED MEAN OVER CLUSTERS OF `stat` WITHIN EACH CLUSTER -- not
    `stat` over the pooled values. The two coincide only where EVERY cluster is a singleton, and
    whether that holds is a property of whether the suite TEMPLATES its tasks rather than a list
    of suites worth memorising: `report` clusters on
    `COALESCE(NULLIF(template_id, ''), task_id)`, so a suite emitting no template_id has
    cluster == task by construction, and a suite emitting one does not. MEASURED over all ten
    suites in `scores/parquet/runs.parquet` (2026-09-17), and the correspondence is exact:

      emits template_id, clusters NOT singletons: musique 589 tasks / 560 clusters,
                                                  tau2 97 / 18, tau2_golden 12 / 5
      emits none, singleton by construction:      drgym, strategyqa, synth, tau2_airline,
                                                  tau2_retail, tau2_telecom, wiki2

    THIS SENTENCE USED TO READ "which is every suite but tau2". That was FALSE for musique and
    for tau2_golden, and the assurance is why nobody checked: a published table reported POOLED
    means while describing itself as cluster estimates over as many clusters as distinct tasks,
    and both halves of that description were wrong. The claim is now checked rather than
    asserted -- `tests/test_stats.py::test_the_singleton_suites_are_exactly_those_emitting_no_
    template` fails, naming the suite, if the correspondence ever breaks.

    HOW MUCH IT MATTERS, so a reader can judge it for their own use rather than trusting a word
    like "negligible". On the frontier musique population -- 200 tasks in 186 clusters, because
    172 templates hold one task and 14 hold two -- pooled against cluster mean, on the real
    values in `artifacts/frontier/scores_parquet.trained`:

      evidence_coverage      0.839417 pooled  vs  0.840233 clustered   (+0.000816)
      facet_breadth          0.942000         vs  0.944355             (+0.002355)
      newly_reachable_share  0.467694         vs  0.479497             (+0.011803)

    Under a thousandth on coverage and over eleven on newly_reachable_share, at a ratio of only
    1.08 tasks per cluster -- so the size of the gap depends on the metric, not just on the
    imbalance, and "too small to matter" is not a safe default. tau2, at 5.4 tasks per cluster,
    is the case the rest of this docstring measures.

    tau2's 97 tasks fall in 18 templates of measured sizes
    [27,12,10,8,7,6,4,4,4,3,2,2,2,2,1,1,1,1] -- one template is 28% of the suite. Pooling
    weights that template 27x. Two consequences, both measured on exactly those sizes:

      1. It disagreed with its own p-value. `paired_difference` sign-flips CLUSTER MEANS (that
         permutation is the one that is exact under the cluster-exchangeability null), so a
         single `Estimate` carried a point and CI for the pooled mean beside a p for the
         cluster mean. Under a true null the two verdicts contradicted each other on 9.5% of
         datasets at ICC=0 and 14.0-14.5% at ICC>=0.3. In a concrete draw where the 27-task
         template favours the treatment they had OPPOSITE SIGNS: point +0.0553, tested
         quantity -0.0320, p=0.63.
      2. The pooled interval is anticonservative on its own terms. Resampling clusters with
         replacement makes a size-weighted mean lurch as the 27-task cluster is drawn twice or
         not at all. False-positive rate of the 95% CI under a true null, T=600, n_boot=3000:
         pooled 8.2% (ICC=0), 14.5% (0.3), 14.7% (0.5), 15.5% (0.8); cluster-mean 10.0, 7.0,
         6.0, 6.5%.

    So this is not merely "make the two halves agree" -- the half that moved is also the half
    that was miscalibrated.
    """
    units = [u for u in units if u]
    if not units:
        return float("nan"), float("nan"), float("nan")

    # THE CALLER'S LIST ORDER IS NOT AN INPUT TO THE ANSWER.
    #
    # The RNG draws unit INDICES, so `units[rng.randrange(n)]` reads a different unit for the
    # same draw when the caller built its list in a different order. The point estimate is a
    # mean over clusters and never moved; the REPS did, and with them z0, the acceleration and
    # both percentiles. Measured on FRAMES `answer_correct`, 824 singleton clusters,
    # n_boot=10000, seed=0: run-id order against task order moved three of four published
    # endpoints by exactly one 1/824 step (base8b [0.1723, 0.2269] against [0.1711, 0.2257]).
    # An interval that depends on an ordering no record states is reproducible by accident.
    #
    # Every caller that arrives through `paired_difference` was always safe -- it groups over
    # `sorted(set(a) & set(b))`. The exposed callers are the SINGLE-ARM ones
    # (`report.level_estimate`, `scripts/matched_cost.level`, `pinq_train.gate.bca_ci`), which
    # take whatever order their rows arrived in, i.e. parquet row order.
    #
    # A resampling unit is a MULTISET of that cluster's values, so both levels are canonicalised
    # here: the values inside each unit, then the units themselves. Sorting inside a unit is
    # what makes the fix complete rather than partial -- reordering values within a cluster
    # changes its sort key, so canonicalising only the outer list would leave the same defect
    # reachable one level down. `stat` is applied to the unit as a whole and every caller passes
    # the default mean, for which a multiset is the whole of the input.
    units = sorted((sorted(u, key=_nan_last) for u in units), key=_unit_order_key)

    def theta(us: Sequence[Sequence[float]]) -> float:
        return statistics.fmean([stat(u) for u in us])

    theta_hat = theta(units)
    rng = random.Random(seed)
    n = len(units)
    reps: list[float] = []
    for _ in range(n_boot):
        reps.append(theta([units[rng.randrange(n)] for _ in range(n)]))
    if not reps:
        return theta_hat, float("nan"), float("nan")

    prop = sum(1 for r in reps if r < theta_hat) / len(reps)
    z0 = _phi_inv(min(max(prop, 1e-9), 1 - 1e-9))

    # jackknife over clusters -> acceleration. Same estimand as the point and the reps.
    jack: list[float] = []
    for i in range(n):
        rest = [u for j, u in enumerate(units) if j != i]
        if rest:
            jack.append(theta(rest))
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

    return theta_hat, _percentile(reps, adj(alpha / 2)), _percentile(reps, adj(1 - alpha / 2))


def paired_difference(
    a: Mapping[str, float],
    b: Mapping[str, float],
    *,
    clusters: Mapping[str, str] | None = None,
    n_boot: int = 10_000,
    n_perm: int = 10_000,
    seed: int = 0,
) -> Estimate:
    """The workhorse: Delta = mean(a - b) over shared tasks, with a clustered CI and a
    sign-flip p. `clusters` maps task_id -> template_id for suites (like tau2) whose tasks
    are not independent."""
    keys = sorted(set(a) & set(b))
    diffs = {k: a[k] - b[k] for k in keys if not (math.isnan(a[k]) or math.isnan(b[k]))}
    if not diffs:
        return Estimate(float("nan"), float("nan"), float("nan"), None, 0, "paired", "no overlap")

    groups: dict[str, list[float]] = {}
    for k, v in diffs.items():
        groups.setdefault(clusters.get(k, k) if clusters else k, []).append(v)
    units = list(groups.values())

    point, lo, hi = cluster_bootstrap(units, n_boot=n_boot, seed=seed)
    # THE PERMUTATION UNIT IS THE CLUSTER, exactly as the bootstrap unit is.
    #
    # This used to be `sign_flip_p(list(diffs.values()))` -- the raw PER-TASK differences, with
    # the cluster map discarded -- while the CI two lines above resampled clusters. One
    # `Estimate` then carried a CI at cluster resolution and a p at task resolution, and its own
    # note printed both numbers side by side without anything reconciling them.
    #
    # Sign-flipping per task assumes each task's sign is independently exchangeable, which is
    # the assumption this module's docstring exists to deny. Tasks sharing a template share
    # their sign, so the effective n for the permutation is the number of TEMPLATES. Measured
    # by simulation under a true null with 25 templates x 4 tasks: the per-task flip rejects at
    # nominal alpha=0.05 in 14.3% of datasets at ICC=0.3, 21.7% at ICC=0.5 and 30.5% at
    # ICC=0.8, against 5.5/5.7/5.0% for the cluster-level flip. It produced rows reading
    # p=0.0018 beside a 95% CI of [-0.0002, +0.6597] -- significant next to an interval
    # containing zero, in a preregistered uncorrected primary.
    #
    # Flipping the sign of a cluster MEAN is the standard cluster-level permutation and is
    # exact under the cluster-exchangeability null. With singleton clusters -- every suite but
    # tau2 mints no template_id -- it reduces to the per-task flip, so nothing moves where
    # nothing was wrong.
    # ONE ESTIMAND. `cluster_bootstrap` and this line both reduce a cluster to its mean, so the
    # point, the interval and the p all describe the same quantity: the unweighted mean over
    # clusters. Fixing only the p (as I first did) left an Estimate whose CI and whose p
    # answered different questions -- see cluster_bootstrap's docstring for the measurements.
    cluster_means = [sum(v) / len(v) for v in units]
    p = sign_flip_p(cluster_means, n_perm=n_perm, seed=seed)
    note = (
        f"{len(units)} clusters over {len(diffs)} tasks; "
        "estimand = unweighted mean over clusters; BCa + sign-flip both at cluster level"
    )
    return Estimate(point, lo, hi, p, len(diffs), "paired-signflip-BCa", note)


# ----------------------------------------------------------------- binary


def mcnemar_exact(
    a: Mapping[str, int],
    b: Mapping[str, int],
    *,
    clusters: Mapping[str, str] | None = None,
    n_perm: int = 10_000,
    seed: int = 0,
) -> Estimate:
    """Exact McNemar on discordant pairs -- WHEN THE PAIRS ARE INDEPENDENT.

    Only the discordant cells carry information: tasks both arms get right (or both wrong) say
    nothing about which arm is better, which is why n here is b01+b10 and not the task count.
    On a 97-task suite that number is small, so exactness is not optional.

    EXACTNESS IS NOT THE ONLY THING THAT MATTERS, AND IT WAS THE ONLY THING CHECKED. The exact
    binomial is exact under the null that each discordant pair is an independent coin flip.
    tau2's primary endpoint is preregistered `cluster_by="template_id"` precisely because its
    97 tasks are instantiations of ~18 banking scenarios and are NOT independent -- and this
    function took no cluster map at all, so the preregistered clustering had no effect on the
    test it governed. A pair of tasks from one template that both flip the same way is one
    piece of evidence counted twice.

    So: with a non-trivial cluster map the p is a CLUSTER-LEVEL sign-flip over per-cluster mean
    differences -- the same permutation unit the paired CI resamples, exact under
    cluster-exchangeability -- and `method` says so. With singleton clusters, or none, it is
    the exact binomial as before, because there nothing is being double-counted. The discordant
    counts are reported either way: they are the honest description of the data even when they
    are not the basis of the test.
    """
    keys = sorted(set(a) & set(b))
    b01 = sum(1 for k in keys if a[k] == 0 and b[k] == 1)
    b10 = sum(1 for k in keys if a[k] == 1 and b[k] == 0)
    n = b01 + b10
    delta = (
        (sum(a[k] for k in keys) - sum(b[k] for k in keys)) / len(keys) if keys else float("nan")
    )
    if n == 0:
        # n IS THE TASK COUNT, NOT THE DISCORDANT COUNT -- see the note on the final return.
        # A perfect tie (both arms identical on all 97 tau2 tasks) is a MEASUREMENT, and
        # reporting n=0 for it made it indistinguishable from "this contrast never ran".
        return Estimate(
            delta,
            float("nan"),
            float("nan"),
            1.0,
            len(keys),
            "mcnemar-exact",
            f"no discordant pairs over {len(keys)} tasks: the arms agreed everywhere",
        )
    k = min(b01, b10)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    p = min(1.0, 2 * tail)

    # THE INTERVAL IS ON THE SAME SCALE AS `delta`, WHICH IT WAS NOT.
    #
    # This used to return `_wilson(b10, n) - 0.5`: a Wilson interval for the DISCORDANT
    # PROPORTION pi = b10/(b01+b10), shifted to centre on zero. `delta` is a RISK DIFFERENCE,
    # (mean a - mean b) over all N tasks. The two are related by
    #
    #     delta = (b10 - b01)/N = (n/N) * (2*pi - 1)
    #
    # so the reported interval was larger than its own point estimate's scale by N/(2n) -- on
    # tau2 with 16 discordant pairs out of 97, a factor of three. The point could and did fall
    # outside its own bounds. Measured before the fix, on the case a paper most wants to print:
    #
    #     8 of 97 discordant, all favouring the treatment
    #     delta = +0.0825   CI = [+0.1756, +0.5000]   p = 0.0078
    #
    # A row asserting an effect of +0.08 whose 95%% interval starts at +0.18. Applying the exact
    # monotone transform above to the Wilson bounds gives a conditional CI for the risk
    # difference -- conditional on the discordant count, which is the same conditioning the
    # exact test itself uses -- so the point sits inside by construction.
    lo_pi, hi_pi = _wilson(b10, n)
    scale = n / len(keys)
    lo, hi = (2 * lo_pi - 1) * scale, (2 * hi_pi - 1) * scale
    method = "mcnemar-exact"
    note = f"b10={b10} b01={b01} discordant={n}; CI conditional on n, on the risk-difference scale"

    if clusters:
        groups: dict[str, list[float]] = {}
        for key in keys:
            groups.setdefault(clusters.get(key, key), []).append(float(a[key] - b[key]))
        if len(groups) < len(keys):
            # Non-trivial clustering: the exact binomial's independence assumption is false, and
            # so is the conditional Wilson interval's. Both the point and the interval move to
            # the cluster level with the p -- see cluster_bootstrap for why a pooled point beside
            # a cluster-level p is two estimands in one row.
            units = list(groups.values())
            p = sign_flip_p([sum(v) / len(v) for v in units], n_perm=n_perm, seed=seed)
            delta, lo, hi = cluster_bootstrap(units, n_boot=n_perm, seed=seed)
            method = "mcnemar-cluster-signflip"
            note = (
                f"b10={b10} b01={b01} discordant={n}; "
                f"p from a sign-flip over {len(groups)} clusters, not the exact binomial, "
                f"because {len(keys)} tasks are not {len(keys)} independent draws; "
                "point, CI and p all at cluster level"
            )

    # `n` IS THE NUMBER OF TASKS, ON EVERY PATH.
    #
    # This returned `n` -- the DISCORDANT-PAIR count -- while `paired_difference` returns
    # `len(diffs)`, the task count. One field, two meanings, decided by which test ran. On the
    # same 97-task tau2 primary with 16 discordant pairs:
    #
    #     mcnemar_exact     n = 16
    #     paired_difference n = 97
    #
    # Two things read it and both were wrong on the McNemar path:
    #
    #   * `report.floor_flag` computes the reportability floor as 2*sigma_J/sqrt(n). At
    #     sigma_J=0.5 that is 0.2500 from 16 and 0.1015 from 97 -- a floor 2.46x too high, so a
    #     judge-derived effect that clears the bar on every other suite is stamped
    #     `below_noise_floor` on the one endpoint that uses McNemar.
    #   * `report.killswitch` treats `est.n == 0` as "NOT RUN". Zero discordant pairs means the
    #     arms agreed on every task -- the strongest possible MATCHES verdict for a kill switch
    #     -- and it was printed as though the comparator had never been run.
    #
    # The discordant count is not lost: it is in `note`, where it describes the data rather than
    # standing in for the sample size.
    return Estimate(delta, lo, hi, p, len(keys), method, note)


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - m, c + m


# ----------------------------------------------------------------- multiplicity


def benjamini_hochberg(p_values: Mapping[str, float], q: float = 0.05) -> dict[str, bool]:
    """BH-FDR over the DECLARED secondary family.

    The design runs ~14 confirmatory tests. Without a declared family and a correction, a
    reviewer is entitled to read the abstract as a selection from a lottery, and they would
    be right: at ~2000 naive comparisons you expect ~100 spurious 'significant' results.
    """
    items = sorted(((k, v) for k, v in p_values.items() if not math.isnan(v)), key=lambda kv: kv[1])
    m = len(items)
    out = {k: False for k in p_values}
    if not m:
        return out
    crit = 0
    for i, (_, p) in enumerate(items, start=1):
        if p <= i / m * q:
            crit = i
    for k, _ in items[:crit]:
        out[k] = True
    return out


def noise_floor(sigma_j: float, n: int) -> float:
    """The smallest effect that may be reported: 2*sigma_J/sqrt(n).

    Published alongside every judge-derived number. An effect below this is reported with an
    explicit `below_noise_floor` flag rather than being silently claimed.
    """
    return 2 * sigma_j / math.sqrt(n) if n > 0 else float("inf")
