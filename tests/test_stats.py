"""Statistics, validated against cases with known answers.

The calibration tests matter most: a permutation p-value must be UNIFORM under the null and
a bootstrap interval must actually cover at its nominal rate. A CI that under-covers silently
turns a null result into a claim.
"""

import math
import random

import pytest

from pi_eval.stats.inference import (
    _nan_last,
    _unit_order_key,
    benjamini_hochberg,
    cluster_bootstrap,
    mcnemar_exact,
    noise_floor,
    paired_difference,
    sign_flip_p,
)

# ------------------------------------------------------------------ permutation


def test_sign_flip_detects_a_large_consistent_effect():
    diffs = [0.5] * 30
    assert sign_flip_p(diffs, n_perm=2000, seed=0) < 0.01


def test_sign_flip_is_null_on_symmetric_noise():
    rng = random.Random(3)
    diffs = [rng.gauss(0, 1) for _ in range(40)]
    assert sign_flip_p(diffs, n_perm=2000, seed=0) > 0.05


def test_sign_flip_p_is_uniform_under_the_null():
    """Calibration: ~5% of null datasets should fall below 0.05. This is the test that
    catches an off-by-one in the permutation counting."""
    rng = random.Random(11)
    ps = []
    for i in range(200):
        diffs = [rng.gauss(0, 1) for _ in range(25)]
        ps.append(sign_flip_p(diffs, n_perm=500, seed=i))
    below = sum(1 for p in ps if p < 0.05) / len(ps)
    assert below < 0.15, f"null rejection rate {below:.3f} is far above nominal"
    assert 0.35 < sum(1 for p in ps if p < 0.5) / len(ps) < 0.65


def test_permutation_p_is_never_exactly_zero():
    """A permutation test cannot make a statement of certainty; +1/+1 enforces that."""
    assert sign_flip_p([1.0] * 50, n_perm=100, seed=0) > 0


# ------------------------------------------------------------------ bootstrap


def test_bca_interval_brackets_the_point_estimate():
    rng = random.Random(5)
    units = [[rng.gauss(2.0, 1.0)] for _ in range(60)]
    point, lo, hi = cluster_bootstrap(units, n_boot=2000, seed=1)
    assert lo < point < hi
    assert lo > 1.0 and hi < 3.0


def test_bca_covers_at_roughly_the_nominal_rate():
    """Under-coverage is how a null becomes a claim, so the rate is asserted, not assumed."""
    rng = random.Random(7)
    covered = 0
    trials = 120
    for t in range(trials):
        units = [[rng.gauss(0.3, 1.0)] for _ in range(50)]
        _, lo, hi = cluster_bootstrap(units, n_boot=600, seed=t)
        if lo <= 0.3 <= hi:
            covered += 1
    assert covered / trials > 0.86, f"coverage {covered / trials:.3f} well below 95%"


def test_clustering_widens_the_interval_versus_ignoring_it():
    """The whole point of clustering: 40 correlated rollouts inside 10 templates carry less
    information than 40 independent ones, and the CI must say so."""
    rng = random.Random(9)
    templates = [[rng.gauss(0.5, 1.0)] * 4 for _ in range(10)]  # perfectly correlated within
    flat = [[v] for t in templates for v in t]
    _, clo, chi = cluster_bootstrap(templates, n_boot=3000, seed=2)
    _, flo, fhi = cluster_bootstrap(flat, n_boot=3000, seed=2)
    assert (chi - clo) > (fhi - flo), "clustered CI must be wider than the naive one"


def test_bca_endpoints_do_not_depend_on_the_order_the_units_arrive_in():
    """A PERMUTATION OF THE SAME UNITS MUST GIVE THE SAME INTERVAL.

    The seeded RNG draws unit INDICES, so the reps -- and therefore z0, the acceleration and
    both percentiles -- used to depend on the order the caller happened to build its list in.
    Measured on real data before this was fixed: FRAMES `answer_correct`, 824 singleton
    clusters, `n_boot=10000, seed=0`. Passing the units in run-id order rather than task order
    moved three of the four published endpoints by exactly one 1/824 step (`base8b`
    [0.1723, 0.2269] against [0.1711, 0.2257]). The POINT estimate never moved, because it is a
    mean over clusters -- which is why an order-dependent interval could sit in a record beside
    a stable point and look reproducible.

    Callers that reach this through `paired_difference` were always safe: it groups over
    `sorted(set(a) & set(b))`. The exposed callers are the SINGLE-ARM ones, which build their
    list from whatever order their rows arrived in.

    Asserted EXACTLY, on the endpoints themselves. A tolerance that swallowed a one-step
    difference would swallow precisely the defect this test exists to forbid.
    """
    rng = random.Random(23)
    binary = [[float(rng.random() < 0.2)] for _ in range(200)]
    ragged = [
        [rng.gauss(0.5, 1.0) for _ in range(rng.choice((1, 1, 1, 2, 4, 9)))] for _ in range(40)
    ]
    for name, units in (("binary singletons", binary), ("ragged clusters", ragged)):
        want = cluster_bootstrap(units, n_boot=2000, seed=0)
        for perm_seed in (1, 2, 3):
            shuffled = list(units)
            random.Random(perm_seed).shuffle(shuffled)
            got = cluster_bootstrap(shuffled, n_boot=2000, seed=0)
            assert got == want, f"{name}, shuffle seed {perm_seed}: {got} != {want}"


def _inner_shuffled(units, perm_seed):
    """The same units in the same outer order, each unit's own values permuted."""
    r = random.Random(perm_seed)
    out = []
    for u in units:
        v = list(u)
        r.shuffle(v)
        out.append(v)
    return out


def test_bca_endpoints_do_not_depend_on_the_order_within_a_cluster():
    """Same argument one level down. A cluster is a multiset of that cluster's values, so
    reordering them cannot be allowed to move the interval either.

    THIS DOCSTRING USED TO NAME THE WRONG MECHANISM. It said the only thing an inner
    permutation changes is summation rounding, "the sort of 1e-16 difference that flips a
    `rep < theta_hat` tie". MEASURED (2026-09-18), that channel is closed: `statistics.fmean`
    accumulates exactly, and over 30 gauss units of 7, 40 units of 9 spanning 1e-6 to 1e6, and
    the 18 real tau2 template sizes, NOT ONE unit's `fmean` differs by a bit under permutation
    (0/30, 0/40, 0/18). Feed the estimator canonical outer order but let `stat` see the
    caller's inner order and 0 of 8 permutations move an endpoint.

    THE OPERATIVE MECHANISM IS THE OUTER SORT KEY, which is why the fix had to reach inside a
    unit. `_unit_order_key` is a function of the sequence, not of the multiset, so if the inner
    values are not canonicalised FIRST then an inner permutation changes a unit's key, the
    outer sort puts the units in a different order, and the RNG's index draws read different
    units -- the very defect one level down. That is what
    `test_the_inner_sort_is_what_buys_the_inner_invariance` measures, and it is why "a mean
    does not depend on the order of what is averaged" is a TRUE statement that does not on its
    own clear this function.

    Asserted EXACTLY, on the endpoints themselves, and with a POSITIVE CONTROL: an invariance
    test that cannot detect a within-unit change is indistinguishable from a clean pass.
    """
    rng = random.Random(29)
    units = [[rng.gauss(0.0, 1.0) for _ in range(7)] for _ in range(30)]
    want = cluster_bootstrap(units, n_boot=2000, seed=0)

    # POSITIVE CONTROL, first, so a harness that permutes nothing reaching the estimator fails
    # here rather than passing the invariance check vacuously. One value, inside one unit.
    teeth = [list(u) for u in units]
    teeth[0][0] += 0.5
    assert cluster_bootstrap(teeth, n_boot=2000, seed=0) != want, (
        "positive control did not move the endpoints, so this test cannot detect a "
        "within-unit change and its pass below means nothing"
    )

    for perm_seed in (1, 2):
        inner = _inner_shuffled(units, perm_seed)
        assert any(list(a) != list(b) for a, b in zip(units, inner)), (
            f"shuffle seed {perm_seed} permuted nothing; the invariance check would be vacuous"
        )
        got = cluster_bootstrap(inner, n_boot=2000, seed=0)
        assert got == want, f"within-cluster shuffle {perm_seed}: {got} != {want}"


def test_the_inner_sort_is_what_buys_the_inner_invariance():
    """The inner sort is LOAD-BEARING, not belt-and-braces -- so nobody deletes it as redundant
    on the true-but-insufficient ground that a mean is order-invariant.

    `cluster_bootstrap` canonicalises the values inside each unit and only then sorts the units.
    Reverse that -- sort the outer list on the RAW unit -- and an inner permutation changes the
    key, hence the outer order, hence which unit each RNG index draw reads. Asserted here on
    the keys themselves rather than by replicating the estimator, because the key is where the
    dependence enters and a replica would drift from the function it stands in for.

    MEASURED end to end with only the inner sort removed: on synthetic shapes an inner
    permutation moved an endpoint on 5/5 permutations for 30x7 gauss (max 0.0170) and ragged
    40x(1..9) (max 0.0211); on the REAL tau2 and tau2_golden cluster values, 233 of 525
    permutations across 105 metrics moved an endpoint (e.g. tau2 `answer_n_words` 5/5, max
    0.408). Under the shipped code, 0 of those same 525 moved.
    """
    rng = random.Random(31)
    units = [[rng.gauss(0.0, 1.0) for _ in range(5)] for _ in range(12)]
    inner = _inner_shuffled(units, 7)
    assert any(list(a) != list(b) for a, b in zip(units, inner)), "nothing was permuted"

    raw_keys = [_unit_order_key(u) for u in units]
    raw_keys_permuted = [_unit_order_key(u) for u in inner]
    assert raw_keys != raw_keys_permuted, (
        "the outer sort key is already invariant to an inner permutation, which would mean "
        "this test no longer covers the mechanism the inner sort exists for"
    )
    # ... and with the inner sort first, it is invariant, which is the whole fix.
    canon = [_unit_order_key(sorted(u, key=_nan_last)) for u in units]
    canon_permuted = [_unit_order_key(sorted(u, key=_nan_last)) for u in inner]
    assert canon == canon_permuted


def test_paired_difference_end_to_end():
    a = {f"t{i}": 1.0 for i in range(30)}
    b = {f"t{i}": 0.5 for i in range(30)}
    est = paired_difference(a, b, n_boot=1000, n_perm=1000)
    assert est.point == pytest.approx(0.5)
    assert est.significant and est.n == 30
    assert "30 clusters" in est.note


def test_paired_difference_respects_the_cluster_map():
    a = {f"t{i}": 1.0 for i in range(20)}
    b = {f"t{i}": 0.0 for i in range(20)}
    clusters = {f"t{i}": f"tpl{i // 5}" for i in range(20)}
    est = paired_difference(a, b, clusters=clusters, n_boot=800, n_perm=800)
    assert "4 clusters over 20 tasks" in est.note


# ------------------------------------------------------------------ McNemar


def test_mcnemar_uses_only_discordant_pairs():
    """Tasks both arms get right say nothing about which arm is better.

    The TEST still uses only the discordant pairs -- that is what the p-value below checks. But
    `Estimate.n` is the TASK COUNT on every path, and this asserted it was the discordant count.
    CLAUDE.md rule 4: the belief about `n` was wrong, not the code. `paired_difference` returns
    len(diffs), so one field meant two things depending on which test ran, and two consumers
    read it as the sample size: `floor_flag` computes 2*sigma_J/sqrt(n) (a floor 2.46x too high
    from 16 discordant pairs instead of 97 tasks) and `killswitch` treats n == 0 as "NOT RUN".
    The discordant count is in `note`, where it describes the data instead of standing in for
    the sample size."""
    a = {f"t{i}": 1 for i in range(50)}
    b = dict(a)
    for i in range(10):
        b[f"t{i}"] = 0  # 10 discordant, all favouring a
    est = mcnemar_exact(a, b)
    assert est.n == 50, "n is the number of tasks compared"
    assert "discordant=10" in est.note, "and the discordant count is still reported"
    assert est.p_value == pytest.approx(2 * (1 / 2**10)), "computed from the 10 discordant pairs"
    assert "b10=10" in est.note


def test_mcnemar_is_one_when_arms_tie():
    """A perfect tie is a MEASUREMENT, not an absence of one.

    `n` was the discordant count, so this case reported n=0 -- and `report.killswitch` maps
    `est.n == 0` to "NOT RUN". Zero discordant pairs means the arms agreed on every task, which
    is the strongest possible MATCHES verdict for a kill switch, and it printed as though the
    comparator had never run."""
    a = {f"t{i}": i % 2 for i in range(40)}
    est = mcnemar_exact(a, dict(a))
    assert est.p_value == 1.0
    assert est.n == 40, "40 tasks were compared and agreed; that is not 'no data'"
    assert "no discordant pairs over 40 tasks" in est.note


def test_mcnemar_matches_the_binomial_by_hand():
    a = {"x": 1, "y": 1, "z": 0, "w": 0}
    b = {"x": 0, "y": 0, "z": 1, "w": 0}
    # b10=2, b01=1, n=3 -> 2*P(X<=1) = 2*(1+3)/8 = 1.0
    assert mcnemar_exact(a, b).p_value == pytest.approx(1.0)


# ------------------------------------------------------------------ multiplicity


def test_bh_rejects_the_obvious_and_spares_the_rest():
    ps = {"a": 0.001, "b": 0.004, "c": 0.30, "d": 0.70}
    out = benjamini_hochberg(ps, q=0.05)
    assert out["a"] and out["b"] and not out["c"] and not out["d"]


def test_bh_is_conservative_under_the_global_null():
    ps = {f"m{i}": (i + 1) / 20 for i in range(20)}
    assert sum(benjamini_hochberg(ps, q=0.05).values()) <= 1


def test_bh_handles_nan_without_rejecting_it():
    out = benjamini_hochberg({"a": 0.001, "b": float("nan")}, q=0.05)
    assert out["a"] and not out["b"]


def test_noise_floor_shrinks_with_n():
    """No effect below 2*sigma_J/sqrt(n) may be reported."""
    assert noise_floor(0.5, 100) == pytest.approx(0.1)
    assert noise_floor(0.5, 400) < noise_floor(0.5, 100)
    assert math.isinf(noise_floor(0.5, 0))


# --------------------------------------------------------------------------- one resolution
#
# `paired_difference` clustered its CI and did NOT cluster its p. One Estimate carried an
# interval at cluster resolution and a p-value at task resolution, and its own note printed
# both counts side by side with nothing reconciling them. `mcnemar_exact` took no cluster map
# at all, while tau2's primary endpoint is preregistered `cluster_by="template_id"` -- so the
# preregistered clustering had no effect on the test it governs.


def _clustered_null(icc, n_tmpl=25, m=4, seed=0):
    """Tasks within a template share an effect; the TRUE mean difference is exactly 0."""
    import random

    rng = random.Random(seed)
    a, b, clusters = {}, {}, {}
    for t in range(n_tmpl):
        shared = rng.gauss(0.0, icc**0.5)
        for j in range(m):
            k = f"t{t}_{j}"
            a[k] = shared + rng.gauss(0.0, (1 - icc) ** 0.5)
            b[k] = 0.0
            clusters[k] = f"T{t}"
    return a, b, clusters


def test_the_p_value_and_the_ci_describe_THE_SAME_QUANTITY():
    """The two numbers sat next to each other and answered different questions, and no test
    could see it because each was individually plausible.

    BEHAVIOURAL, not a source-string match. The first version of this guard pinned the exact
    expression `sign_flip_p([sum(v) / len(v) for v in units]`, which passed while the POINT and
    the CI were still the pooled task mean -- it checked that one of the two halves had been
    fixed, not that the halves agreed. Naming the quantity and asserting the point IS it catches
    both halves and survives refactoring.
    """
    import statistics

    from pi_eval.stats.inference import paired_difference

    # tau2's measured template sizes: one cluster is 28% of the suite, so pooling and averaging
    # over clusters are genuinely different numbers.
    sizes = [27, 12, 10, 8, 7, 6, 4, 4, 4, 3, 2, 2, 2, 2, 1, 1, 1, 1]
    a, b, clusters = {}, {}, {}
    t = 0
    for ci, n in enumerate(sizes):
        for _ in range(n):
            k = f"t{t}"
            # the big cluster favours the treatment; every other cluster is flat
            a[k], b[k] = (0.30 if ci == 0 else 0.0), 0.0
            clusters[k] = f"c{ci}"
            t += 1

    est = paired_difference(a, b, clusters=clusters, n_boot=500, n_perm=500, seed=0)
    per_cluster = [0.30] + [0.0] * 17
    assert est.point == pytest.approx(statistics.fmean(per_cluster)), (
        "the point must be the unweighted mean over CLUSTERS -- the quantity the p tests"
    )
    pooled = statistics.fmean([a[k] - b[k] for k in a])
    assert abs(est.point - pooled) > 0.005, "the fixture must actually separate the two estimands"
    assert est.ci_lo <= est.point <= est.ci_hi
    assert "unweighted mean over clusters" in est.note


def test_singleton_clusters_change_nothing():
    """Every suite but tau2 mints no template_id, so cluster_id == task_id and the clustered
    flip reduces to the per-task flip. A fix that moved those numbers would be a new bug."""
    from pi_eval.stats.inference import paired_difference, sign_flip_p

    a = {f"t{i}": (i % 7) - 3.0 for i in range(40)}
    b = {k: 0.0 for k in a}
    est = paired_difference(a, b, clusters={k: k for k in a}, n_boot=200, n_perm=2000, seed=0)
    assert est.p_value == pytest.approx(
        sign_flip_p([a[k] for k in sorted(a)], n_perm=2000, seed=0), abs=0.02
    )


def test_the_clustered_p_holds_its_nominal_rate_where_the_per_task_p_did_not():
    """MEASURED, under a true null. The per-task flip rejected at 15.5% (ICC=0.3), 23.0%
    (ICC=0.5) and 31.5% (ICC=0.8) against a nominal 5%. Tasks sharing a template share their
    sign, so the effective n is the number of TEMPLATES -- which is the same reason the CI
    resamples clusters."""
    from pi_eval.stats.inference import paired_difference, sign_flip_p

    trials, clustered, per_task = 120, 0, 0
    for s in range(trials):
        a, b, cl = _clustered_null(0.5, seed=s)
        est = paired_difference(a, b, clusters=cl, n_boot=120, n_perm=600, seed=s)
        clustered += est.p_value < 0.05
        per_task += sign_flip_p([a[k] - b[k] for k in sorted(a)], n_perm=600, seed=s) < 0.05

    assert clustered / trials < 0.12, f"clustered rejects {clustered / trials:.2f} of true nulls"
    assert per_task / trials > 0.15, (
        f"the per-task flip should be visibly anti-conservative here; got {per_task / trials:.2f}"
    )
    assert per_task > clustered


def test_mcnemar_honours_the_preregistered_clustering_and_says_which_test_it_ran():
    """tau2's 97 tasks are ~18 banking scenarios, and the endpoint is preregistered clustered at
    template_id. The exact binomial is exact under INDEPENDENT discordant pairs, which is
    precisely what clustering denies -- so with a real cluster map the p comes from a
    cluster-level permutation, and `method` names it rather than still saying 'exact'."""
    from pi_eval.stats.inference import mcnemar_exact

    a = {f"t{t}_{j}": (1 if t % 2 == 0 else 0) for t in range(10) for j in range(4)}
    b = {k: 0 for k in a}
    clusters = {k: k.split("_")[0] for k in a}

    plain = mcnemar_exact(a, b)
    assert plain.method == "mcnemar-exact"
    assert "discordant" in plain.note

    clustered = mcnemar_exact(a, b, clusters=clusters, n_perm=2000, seed=0)
    assert clustered.method == "mcnemar-cluster-signflip"
    assert "not the exact binomial" in clustered.note
    # The discordant counts are still reported: they describe the data honestly even when they
    # are not the basis of the test.
    assert "b10=" in clustered.note and "b01=" in clustered.note
    assert clustered.p_value >= plain.p_value, "clustering cannot make the test more permissive"


def test_mcnemar_with_singleton_clusters_is_still_the_exact_binomial():
    """Where no two tasks share a template there is nothing being double-counted, and the exact
    test is strictly better than a permutation approximation of it."""
    from pi_eval.stats.inference import mcnemar_exact

    a = {f"t{i}": int(i < 12) for i in range(30)}
    b = {k: 0 for k in a}
    est = mcnemar_exact(a, b, clusters={k: k for k in a})
    assert est.method == "mcnemar-exact"
    assert est.p_value == pytest.approx(mcnemar_exact(a, b).p_value)


def test_the_report_passes_its_cluster_map_to_mcnemar():
    """The call site had no `clusters=` at all, so tau2's preregistered clustering was inert."""
    import inspect

    from pi_eval import report

    src = inspect.getsource(report.estimate)
    i = src.index("mcnemar_exact(")
    assert "clusters=p.clusters" in src[i : i + 400]


# --------------------------------------------------- the McNemar interval and its point estimate


def _mc(N, b10, b01, clusters=None, n_perm=2000):
    from pi_eval.stats.inference import mcnemar_exact

    a, b, i = {}, {}, 0
    for _ in range(b10):
        a[f"t{i}"], b[f"t{i}"] = 1, 0
        i += 1
    for _ in range(b01):
        a[f"t{i}"], b[f"t{i}"] = 0, 1
        i += 1
    while i < N:
        a[f"t{i}"], b[f"t{i}"] = 1, 1
        i += 1
    return mcnemar_exact(a, b, clusters=clusters, n_perm=n_perm)


@pytest.mark.parametrize(
    "N,b10,b01",
    [(97, 12, 4), (97, 10, 2), (97, 8, 0), (97, 0, 8), (20, 6, 2), (97, 10, 10), (97, 1, 0)],
)
def test_the_point_estimate_lies_inside_its_own_confidence_interval(N, b10, b01):
    """It did not. The CI was Wilson on the DISCORDANT PROPORTION b10/(b01+b10) shifted by 0.5,
    while `delta` is a risk difference over all N tasks -- scales differing by N/(2n). On the
    case a paper most wants to print (8 of 97 discordant, all favouring the treatment) the
    shipped row read delta=+0.0825 with a 95% CI of [+0.1756, +0.5000]."""
    e = _mc(N, b10, b01)
    assert e.ci_lo <= e.point <= e.ci_hi, f"{e.point} not in [{e.ci_lo}, {e.ci_hi}]"


def test_the_interval_is_on_the_risk_difference_scale():
    """A sanity anchor with a hand-computable answer: 8 of 97 discordant, all one way, so the
    risk difference is exactly 8/97 and an interval for it cannot exceed that."""
    e = _mc(97, 8, 0)
    assert e.point == pytest.approx(8 / 97)
    assert e.ci_hi == pytest.approx(8 / 97), "pi_hat = 1 pins the upper bound at delta"
    assert 0 < e.ci_lo < e.point
    assert e.ci_hi - e.ci_lo < 0.2, "the old discordant-scale interval spanned 0.32 here"


def test_a_symmetric_result_is_centred_on_zero():
    e = _mc(97, 10, 10)
    assert e.point == pytest.approx(0.0)
    assert e.ci_lo == pytest.approx(-e.ci_hi, abs=1e-9)
    assert e.p_value == pytest.approx(1.0)


def test_clustering_moves_the_point_and_the_interval_with_the_p():
    """When the p goes to the cluster level the point and CI must go with it, or the row carries
    two estimands -- the same defect as in paired_difference."""
    sizes = [27, 12, 10, 8, 7, 6, 4, 4, 4, 3, 2, 2, 2, 2, 1, 1, 1, 1]  # tau2's measured templates
    cl, i = {}, 0
    for ci, n in enumerate(sizes):
        for _ in range(n):
            cl[f"t{i}"] = f"c{ci}"
            i += 1
    e = _mc(97, 8, 0, clusters=cl)
    assert e.method == "mcnemar-cluster-signflip"
    assert e.ci_lo <= e.point <= e.ci_hi
    # every discordant task falls in the 27-task template, so there is ONE piece of evidence
    assert e.p_value == pytest.approx(1.0), (
        "all discordance inside a single template is one observation, not eight"
    )
    assert _mc(97, 8, 0).p_value < 0.01, "the unclustered test calls the same data significant"


def test_estimate_n_means_the_same_thing_on_both_paths():
    """One field, one meaning. On the same 97-task tau2 primary with 16 discordant pairs,
    mcnemar_exact returned n=16 while paired_difference returned n=97 -- and `floor_flag`
    computes the reportability floor as 2*sigma_J/sqrt(n) from whichever it got. At
    sigma_J=0.5 that is 0.2500 from 16 against 0.1015 from 97: a floor 2.46x too high on the
    one endpoint that uses McNemar."""
    from pi_eval.report import noise_floor
    from pi_eval.stats.inference import mcnemar_exact, paired_difference

    a, b = {}, {}
    for i in range(97):
        a[f"t{i}"], b[f"t{i}"] = (1, 0) if i < 12 else (0, 1) if i < 16 else (1, 1)
    m = mcnemar_exact(a, b)
    p = paired_difference(
        {k: float(v) for k, v in a.items()},
        {k: float(v) for k, v in b.items()},
        n_boot=200,
        n_perm=200,
    )
    assert m.n == p.n == 97
    assert noise_floor(0.5, m.n) == pytest.approx(noise_floor(0.5, p.n))
    assert "discordant=16" in m.note


def test_the_frontier_band_resamples_clusters():
    """`frontier()` resampled tasks and took no cluster map, while the AUC point estimate
    reached through `report.frontier_auc` resamples clusters -- two intervals for one quantity
    over different units. Coverage of the nominal 95% AUC interval on tau2's measured template
    sizes, 250 datasets:

        ICC     tasks    clusters
        0.0     94.0%      92.4%
        0.5     52.0%      90.8%
        0.8     42.0%      91.2%

    This is the band that licenses "that dip is noise", printed under a figure."""
    import random

    from pi_eval.metrics.frontier import frontier

    sizes = [27, 12, 10, 8, 7, 6, 4, 4, 4, 3, 2, 2, 2, 2, 1, 1, 1, 1]
    grid = (1, 2, 4, 8, 16)
    rng = random.Random(5)
    per, cl = {}, {}
    t = 0
    for ci, n in enumerate(sizes):
        shared = rng.gauss(0, 0.7)  # one offset for the whole template: ICC ~ 0.5
        for _ in range(n):
            base = max(0.0, min(1.0, 0.5 + shared * 0.15 + rng.gauss(0, 0.7) * 0.15))
            per[f"t{t}"] = (list(grid), [base] * len(grid))
            cl[f"t{t}"] = f"c{ci}"
            t += 1

    by_task = frontier(per, grid, n_boot=400, seed=1)
    by_cluster = frontier(per, grid, n_boot=400, seed=1, clusters=cl)
    wide = (by_cluster.auc_hi - by_cluster.auc_lo) > (by_task.auc_hi - by_task.auc_lo)
    assert wide, "resampling 18 clusters cannot be tighter than resampling 97 tasks"


def test_singleton_clusters_leave_the_frontier_untouched():
    """Every suite but tau2 mints no template_id, so nothing there may move."""
    import random

    from pi_eval.metrics.frontier import frontier

    rng = random.Random(9)
    grid = (1, 2, 4, 8)
    per = {f"t{i}": (list(grid), [rng.uniform(0, 1)] * len(grid)) for i in range(40)}
    a = frontier(per, grid, n_boot=300, seed=2)
    b = frontier(per, grid, n_boot=300, seed=2, clusters={t: t for t in per})
    assert (a.auc, a.auc_lo, a.auc_hi) == (b.auc, b.auc_lo, b.auc_hi)


# ------------------------------------------------- the estimand, and the claim about clusters
#
# `cluster_bootstrap`'s docstring says its estimand is the unweighted mean over clusters, not the
# statistic over pooled values -- correct -- and USED TO ADD that the two coincide on singleton
# clusters, "which is every suite but tau2". That second half was false: musique and tau2_golden
# both cluster non-trivially. The assurance is why a published table reported POOLED means while
# describing itself as cluster estimates over as many clusters as distinct tasks.
#
# These two tests make both halves checkable instead of asserted.


def test_the_estimand_is_the_unweighted_mean_over_clusters_not_the_pooled_statistic():
    """One cluster of four, one of one. Pooling weights the big cluster 4x; the estimand does
    not. Pins the first docstring sentence numerically, on values chosen so the two answers
    cannot be confused: pooled 0.2, cluster mean 0.5."""
    import statistics

    units = [[0.0, 0.0, 0.0, 0.0], [1.0]]
    pooled = statistics.fmean([v for u in units for v in u])
    clustered = statistics.fmean([statistics.fmean(u) for u in units])
    assert (pooled, clustered) == (0.2, 0.5)

    point, _lo, _hi = cluster_bootstrap(units, n_boot=200, seed=0)
    assert point == pytest.approx(clustered), "the estimand is not the mean over clusters"
    assert point != pytest.approx(pooled), "the estimand collapsed to the pooled statistic"


def test_the_singleton_suites_are_exactly_those_emitting_no_template():
    """The claim the docstring now makes, checked against the data it is about.

    `report` clusters on `COALESCE(NULLIF(template_id, ''), task_id)`, so a suite that emits no
    template_id has one task per cluster BY CONSTRUCTION, and a suite that emits one does not.
    The docstring documents that correspondence rather than a list of suite names, so this test
    is what keeps it true as suites are added: it fails, naming the suite, either if a
    templating suite turns out to have singleton clusters or if a non-templating one does not.

    Measured 2026-09-17 over all ten suites: musique 589 tasks / 560 clusters, tau2 97 / 18 and
    tau2_golden 12 / 5 emit templates and are NOT singletons; drgym, strategyqa, synth,
    tau2_airline, tau2_retail, tau2_telecom and wiki2 emit none and are singletons.

    Skips rather than fails where the shared store is absent, which is every worktree: this is a
    property of the DATA, and a checkout with no data has no claim to check.
    """
    from pathlib import Path

    duckdb = pytest.importorskip("duckdb")
    store = Path(__file__).resolve().parents[1] / "scores" / "parquet" / "runs.parquet"
    if not store.is_file():
        pytest.skip(f"no shared runs store at {store}")

    rows = duckdb.sql(
        "SELECT suite_id, "
        "       count(DISTINCT task_id) AS tasks, "
        "       count(DISTINCT coalesce(nullif(template_id, ''), task_id)) AS clusters, "
        "       sum(CASE WHEN coalesce(template_id, '') <> '' THEN 1 ELSE 0 END) AS templated "
        f"FROM read_parquet('{store}') GROUP BY 1 ORDER BY 1"
    ).fetchall()
    assert rows, "the store holds no runs; this test needs the shared store"

    broken = []
    for suite, tasks, clusters, templated in rows:
        emits_template = templated > 0
        singleton = tasks == clusters
        if emits_template == singleton:  # the correspondence is an XOR; equality breaks it
            broken.append(
                f"{suite}: emits_template={emits_template} but tasks={tasks} clusters={clusters}"
                f" ({'singleton' if singleton else 'not singleton'})"
            )
    assert not broken, (
        "cluster_bootstrap's docstring says a suite's clusters are singletons exactly when it "
        "emits no template_id. That no longer holds:\n  " + "\n  ".join(broken)
    )
