"""Lane L1.1 tooling: does the diversity-gate flip survive un-pooling the seeds?

THE FAILURE THIS FILE EXISTS TO CATCH. `pinq_train.gate`'s `criteria["distinct3"]` pools every
seed of a task into one group before scoring trigram diversity (`within_task_distinct_n`,
grouped by `task_id` alone). A checkpoint run at N seeds that asks the SAME state-dependent
questions under every seed is therefore scored as if it had collapsed by a factor close to N,
purely because the evaluator ran it more than once. `test_the_pooling_artifact_is_exactly_the_
seed_count` below pins the mechanism with a fixture where the artifact is exact (duplicating a
fully-distinct 2-question set once must read exactly 0.5 pooled and exactly 1.0 once the seeds
are told apart) so a future change to the grouping logic cannot silently drift.
"""

from __future__ import annotations

import math

import pytest
from scripts.diversity_per_seed.diversity_variants import (
    NEAR_IDENTICAL_JACCARD,
    NEAR_IDENTICAL_MIN_SHARED,
    all_variants,
    classify_group,
    cross_seed_determinism,
    resolve_temperature,
    variant_a_pooled_by_task,
    variant_b_by_task_seed,
    variant_c_seed0_only,
    variant_d_per_seed_averaged,
)


def _row(task_id, seed, turn_idx, question):
    return {"task_id": task_id, "seed": seed, "turn_idx": turn_idx, "question": question}


def _two_question_task(task_id: str) -> tuple[str, str]:
    """Trigram-disjoint by construction (no shared token 3-gram between the two strings), so a
    single seed's own within-task distinct-3 is exactly 1.0."""
    return (f"who wrote the {task_id} novel", f"when did {task_id} die")


# --------------------------------------------------------------------------- the pooling artifact


def test_the_pooling_artifact_is_exactly_the_seed_count():
    """12 tasks, each asked identically under 2 seeds: pooled-by-task (a) must read exactly
    0.5 -- half of the single-seed 1.0 -- and grouping by (task, seed) (b) must recover 1.0."""
    rows = []
    for i in range(12):
        task = f"t{i}"
        q0, q1 = _two_question_task(task)
        for seed in (0, 1):
            rows.append(_row(task, seed, 0, q0))
            rows.append(_row(task, seed, 1, q1))

    a_value, a_n = variant_a_pooled_by_task(rows)
    assert a_value == pytest.approx(0.5)
    assert a_n == 12

    b_value, b_n = variant_b_by_task_seed(rows)
    assert b_value == pytest.approx(1.0)
    assert b_n == 24  # 12 tasks x 2 seeds, each an independent group

    c_value, c_n = variant_c_seed0_only(rows)
    assert c_value == pytest.approx(1.0)
    assert c_n == 12

    d_value, per_seed = variant_d_per_seed_averaged(rows)
    assert d_value == pytest.approx(1.0)
    assert per_seed == {0: pytest.approx(1.0), 1: pytest.approx(1.0)}


def test_three_seeds_pools_to_a_third_not_a_half():
    """The same mechanism at N=3 seeds divides pooled distinct-3 by 3, not 2 -- checking the
    denominator is the seed count and not a hardcoded halving."""
    rows = []
    for i in range(12):
        task = f"t{i}"
        q0, q1 = _two_question_task(task)
        for seed in (0, 1, 2):
            rows.append(_row(task, seed, 0, q0))
            rows.append(_row(task, seed, 1, q1))
    a_value, _ = variant_a_pooled_by_task(rows)
    assert a_value == pytest.approx(1 / 3)
    b_value, _ = variant_b_by_task_seed(rows)
    assert b_value == pytest.approx(1.0)


def test_a_single_seed_makes_every_variant_agree():
    """With one seed present, (a)/(b)/(c)/(d) are the same computation under four names -- the
    dev population today. A regression that makes them disagree on one seed is a bug in the
    grouping, not a property of seeds."""
    rows = []
    for i in range(8):
        task = f"t{i}"
        q0, q1 = _two_question_task(task)
        rows.append(_row(task, 0, 0, q0))
        rows.append(_row(task, 0, 1, q1))
    summary = all_variants(rows)
    assert summary["n_seeds"] == 1
    a = summary["a_pooled_by_task"]["value"]
    b = summary["b_by_task_seed"]["value"]
    c = summary["c_seed0_only"]["value"]
    d = summary["d_per_seed_averaged"]["value"]
    assert a == pytest.approx(1.0) and b == pytest.approx(a) and c == pytest.approx(a)
    assert d == pytest.approx(a)


def test_a_genuine_collapse_stays_low_under_every_variant():
    """The artifact this module exists to rule OUT is not the only way to fail the floor: a
    checkpoint asking the identical question at every turn of every task must read well below
    the 0.65 floor under every grouping, seeds or not -- otherwise `by_seed` would be
    laundering real mode collapse as a seed-count effect. Pooling K byte-identical copies of
    one text reads exactly 1/K (this text's own 8 trigrams stay the numerator while the
    denominator grows K-fold), not 0.0 -- (b)/(c)/(d) each pool 2 copies (0.5) where (a) pools
    4 (0.25), so this checks the ordering and the floor, not a hardcoded zero.
    """
    same = "what is the capital city of the country in question"
    rows = []
    for i in range(12):
        task = f"t{i}"
        for seed in (0, 1):
            rows.append(_row(task, seed, 0, same))
            rows.append(_row(task, seed, 1, same))
    summary = all_variants(rows)
    floor = 0.65
    assert summary["a_pooled_by_task"]["value"] == pytest.approx(0.25)
    assert summary["b_by_task_seed"]["value"] == pytest.approx(0.5)
    assert summary["c_seed0_only"]["value"] == pytest.approx(0.5)
    assert summary["d_per_seed_averaged"]["value"] == pytest.approx(0.5)
    for key in ("a_pooled_by_task", "b_by_task_seed", "c_seed0_only", "d_per_seed_averaged"):
        assert summary[key]["value"] < floor, (key, summary[key])


def test_a_task_with_one_question_contributes_nothing_not_zero():
    """Mirrors `within_task_distinct_n`'s own MIN_PER_TASK rule: a lone question cannot be
    more or less varied than itself, so it must not drag the mean toward 0."""
    rows = [_row("only_one", 0, 0, "who wrote it")]
    value, n = variant_a_pooled_by_task(rows)
    assert math.isnan(value)
    assert n == 0


# --------------------------------------------------------------------------- cross-seed determinism


def test_classify_group_byte_identical():
    assert classify_group({0: "who wrote it", 1: "who wrote it"}) == "identical"


def test_classify_group_near_identical_needs_both_jaccard_and_shared_count():
    """The repo's calibration (memory: pair-paraphrase-guard-calibration) is a ratio AND a
    floor together, because the ratio alone over-fires on short questions. A short pair that
    clears the 0.5 Jaccard ratio on fewer than 4 shared content tokens must NOT be called
    near-identical; a real paraphrase with enough shared content tokens must."""
    from pinq_train.export.distinctness import content_tokens, jaccard

    short_a, short_b = "where did she die", "when did she die"
    ta, tb = content_tokens(short_a), content_tokens(short_b)
    j, shared = jaccard(ta, tb), len(ta & tb)
    assert j >= NEAR_IDENTICAL_JACCARD and shared < NEAR_IDENTICAL_MIN_SHARED, (
        "fixture no longer probes the ratio-over-fires-on-short-questions edge; the guard's "
        f"own inputs moved: jaccard={j}, shared={shared}"
    )
    assert classify_group({0: short_a, 1: short_b}) == "different"

    paraphrase_a = "which country has the world's oldest navy"
    paraphrase_b = "which country possesses the world's oldest navy"
    pa, pb = content_tokens(paraphrase_a), content_tokens(paraphrase_b)
    assert jaccard(pa, pb) >= NEAR_IDENTICAL_JACCARD and len(pa & pb) >= NEAR_IDENTICAL_MIN_SHARED
    assert classify_group({0: paraphrase_a, 1: paraphrase_b}) == "near_identical"


def test_classify_group_different():
    assert classify_group({0: "who wrote the novel", 1: "what year did the war end"}) == "different"


def test_cross_seed_determinism_shares_are_nested():
    """identical is a SUBSET of near-identical-or-better by construction (a byte-identical pair
    always also clears the paraphrase guard), so share_near_identical_or_better must never be
    less than share_identical."""
    rows = [
        _row("t0", 0, 0, "who wrote it"),
        _row("t0", 1, 0, "who wrote it"),  # identical
        _row("t1", 0, 0, "which country has the world's oldest navy"),
        _row("t1", 1, 0, "which country possesses the world's oldest navy"),  # near
        _row("t2", 0, 0, "who wrote the novel"),
        _row("t2", 1, 0, "what year did the war end"),  # different
        _row("t3", 0, 0, "only one seed reached this turn"),  # no partner: excluded
    ]
    d = cross_seed_determinism(rows)
    assert d["n_task_turn_positions_with_ge2_seeds"] == 3
    assert d["share_identical"] == pytest.approx(1 / 3)
    assert d["share_near_identical_or_better"] >= d["share_identical"]


def test_cross_seed_determinism_empty_is_nan_not_zero():
    """No (task, turn_idx) position reached by >= 2 seeds is an absent measurement, not a
    measured zero -- an absent field read as zero is exactly the failure mode CONTRIBUTING.md warns
    this repo about."""
    rows = [_row("t0", 0, 0, "only one seed here")]
    d = cross_seed_determinism(rows)
    assert d["n_task_turn_positions_with_ge2_seeds"] == 0
    assert math.isnan(d["share_identical"])
    assert math.isnan(d["share_near_identical_or_better"])


# --------------------------------------------------------------------------- sampling temperature


def test_resolve_temperature_round_trips_through_the_real_hash():
    from pinq.ids import h

    sha = h("sampling", "t=0.0")
    assert resolve_temperature(sha) == 0.0


def test_resolve_temperature_returns_none_off_the_candidate_grid():
    assert resolve_temperature("not-a-real-sha") is None
