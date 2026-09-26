"""The published fork p-value counted 102 pairs as 102 independent observations.

Measured on the runs the paper reports (`scripts/report_forks.py --suite tau2_retail --split
test`): the 102 pairs are 34 fork points x 3 seeds, the 34 fork points are cut from 32 distinct
recorded dialogues (`foreign_trace_sha`), and those dialogues instantiate 25 distinct tau2 tasks.
Airline: 102 pairs, 34 points, 30 traces, 17 tasks. A sign test over 95 informative pairs states
its evidence at a resolution the data does not have; three seeds of one prefix are one
observation of that prefix, and two prefixes of one dialogue are not two dialogues.

These tests pin the three properties that make the recomputation a recomputation and not a
second opinion: the clustering unit is NAMED in the record, the counts nest, and the p-value
moves in the conservative direction when clusters are coarsened.

They also pin the two non-fork helpers this module grew for the airline confirmatory population:
a duplicate-cell selection rule that is outcome-independent by construction, and a minimum
detectable effect taken from the observed paired sd rather than from a rule of thumb.
"""

from __future__ import annotations

import math

from pi_eval.fork_report import (
    clustered_engagement,
    minimum_detectable_effect,
    select_one_per_cell,
)


def _run(arm, seed, trace, k, fu_total, prefix, reward=0.0, code="abc1234", task="1"):
    return {
        "run_id": f"{arm}-{seed}-{trace}-{k}",
        "arm_id": arm,
        "seed": seed,
        "task_id": task,
        "foreign_trace_sha": trace,
        "foreign_prefix_k": k,
        "n_user_turns": fu_total,
        "n_prefix_user_turns": prefix,
        "tau_reward": reward,
        "code_version": code,
        "status": "ok",
    }


def _pop(n_traces=8, points_per_trace=2, seeds=(0, 1, 2)):
    """Every pair favours the treatment by 1 follow-up: the maximally significant shape at the
    pair level, so any widening seen below comes from the clustering and nothing else."""
    rows = []
    for tr in range(n_traces):
        for j in range(points_per_trace):
            for s in seeds:
                rows.append(_run("t", s, f"sha{tr}", 2 + 2 * j, 8, 3, task=str(tr)))
                rows.append(_run("c", s, f"sha{tr}", 2 + 2 * j, 9, 3, task=str(tr)))
    return rows


# ------------------------------------------------------------------ clustering


def test_the_clustering_unit_is_named_in_the_record_at_every_level():
    rep = clustered_engagement(_pop(), treatment="t", control="c")
    for level in rep["levels"].values():
        assert level["unit"], "a level with no named unit is a number with no provenance"
        assert level["n_units"] >= 1
    assert rep["levels"]["pair"]["unit"] == "(foreign_trace_sha, foreign_prefix_k, seed)"
    assert rep["levels"]["trace"]["unit"] == "foreign_trace_sha"
    assert rep["levels"]["task"]["unit"] == "task_id"


def test_the_unit_counts_nest_from_pair_down_to_task():
    rep = clustered_engagement(_pop(n_traces=8, points_per_trace=2), treatment="t", control="c")
    n = {k: v["n_units"] for k, v in rep["levels"].items()}
    assert n["pair"] == 8 * 2 * 3 == 48
    assert n["fork_point"] == 16
    assert n["trace"] == 8
    assert n["task"] == 8
    assert n["pair"] >= n["fork_point"] >= n["trace"] >= n["task"]


def test_coarsening_the_cluster_widens_the_p_value_it_never_narrows_it():
    """The point of the recomputation. Same data, fewer independent units, weaker evidence."""
    rep = clustered_engagement(_pop(n_traces=8, points_per_trace=2), treatment="t", control="c")
    p = {k: v["sign_test_p"] for k, v in rep["levels"].items()}
    assert p["pair"] < p["fork_point"] < p["trace"]
    assert p["pair"] == 2.0 / 2**48
    assert p["trace"] == 2.0 / 2**8


def test_the_point_estimate_is_the_unweighted_mean_over_clusters_not_the_pooled_mean():
    """One big trace must not outvote seven small ones just for being big. `cluster_bootstrap`
    makes the same commitment; this keeps the sign test's estimand equal to the interval's."""
    rows = _pop(n_traces=2, points_per_trace=1, seeds=(0,))
    # give trace 0 nine extra pairs at a diff of 0, trace 1 keeps its single -1
    for j in range(1, 10):
        rows.append(_run("t", 0, "sha0", 2 + 2 * j, 9, 3, task="0"))
        rows.append(_run("c", 0, "sha0", 2 + 2 * j, 9, 3, task="0"))
    rep = clustered_engagement(rows, treatment="t", control="c")
    assert rep["levels"]["pair"]["diff_mean"] == -2 / 11  # pooled over 11 pairs
    # trace 0 mean = -1/10, trace 1 mean = -1  ->  unweighted mean over the two traces
    assert abs(rep["levels"]["trace"]["diff_mean"] - (-0.55)) < 1e-12


def test_the_bootstrap_and_the_permutation_share_the_cluster_unit_with_the_sign_test():
    rep = clustered_engagement(_pop(), treatment="t", control="c")
    tr = rep["levels"]["trace"]
    assert tr["estimate"]["n_clusters"] == tr["n_units"]
    assert "clusters" in tr["estimate"]["note"]
    assert tr["estimate"]["ci_hi"] < 0  # every pair favours the treatment


def test_a_trace_carrying_two_prefixes_is_one_unit_not_two():
    one = clustered_engagement(_pop(n_traces=4, points_per_trace=1), treatment="t", control="c")
    two = clustered_engagement(_pop(n_traces=4, points_per_trace=2), treatment="t", control="c")
    assert one["levels"]["trace"]["n_units"] == two["levels"]["trace"]["n_units"] == 4
    assert two["levels"]["pair"]["n_units"] == 2 * one["levels"]["pair"]["n_units"]
    assert two["levels"]["trace"]["sign_test_p"] == one["levels"]["trace"]["sign_test_p"]


# ------------------------------------------------------- duplicate-cell selection


def _cellrow(task, seed, arm, code, outcome, rid=None):
    return {
        "run_id": rid or f"{task}-{seed}-{arm}-{code}",
        "task_id": task,
        "seed": seed,
        "arm_id": arm,
        "code_version": code,
        "n_user_turns": outcome,
    }


def test_selection_is_not_outcome_dependent():
    """Swap the outcomes between the two runs of every duplicated cell; the kept run must not
    move. A rule that reads the outcome is a rule that can be tuned by it."""
    a = [_cellrow("1", 0, "x", "aaaa", 99), _cellrow("1", 0, "x", "bbbb", 1)]
    b = [_cellrow("1", 0, "x", "aaaa", 1), _cellrow("1", 0, "x", "bbbb", 99)]
    ka = select_one_per_cell(a, prefer_code_version="aaaa")
    kb = select_one_per_cell(b, prefer_code_version="aaaa")
    assert [r["run_id"] for r in ka["kept"]] == [r["run_id"] for r in kb["kept"]]


def test_selection_records_the_rule_and_the_cells_it_touched():
    rows = [
        _cellrow("1", 0, "x", "aaaa", 5),
        _cellrow("1", 0, "x", "bbbb", 6),
        _cellrow("2", 0, "x", "aaaa", 7),
    ]
    rec = select_one_per_cell(rows, prefer_code_version="aaaa")
    assert rec["rule"]
    assert "aaaa" in rec["rule"]
    assert rec["n_cells"] == 2
    assert rec["n_cells_touched"] == 1
    assert rec["n_rows_in"] == 3 and rec["n_rows_kept"] == 2
    assert rec["n_dropped"] == 1
    assert rec["cells_touched"] == [["1", 0, "x"]]


def test_a_cell_whose_only_run_is_the_non_preferred_version_is_kept_not_dropped():
    """Deduplication and restriction are different rules and give different populations.
    Restricting to one code version deletes whole cells and unbalances the arms; keeping one
    run per cell does not. Measured on the airline eligible population: restriction leaves
    58/60/57 per arm over 175 rows, deduplication leaves 66/66/66 over 198."""
    rows = [
        _cellrow("1", 0, "x", "aaaa", 5),
        _cellrow("1", 0, "x", "bbbb", 6),
        _cellrow("9", 0, "x", "bbbb", 7),  # bbbb is the SOLE occupant here
    ]
    rec = select_one_per_cell(rows, prefer_code_version="aaaa")
    assert rec["n_rows_kept"] == 2
    assert {r["run_id"] for r in rec["kept"]} == {"1-0-x-aaaa", "9-0-x-bbbb"}
    assert rec["n_cells_kept_on_a_fallback"] == 1
    assert rec["fallback_cells"] == [["9", 0, "x"]]


def test_selection_is_deterministic_when_the_preferred_version_is_absent_from_a_cell():
    rows = [
        _cellrow("1", 0, "x", "cccc", 5, rid="zzz"),
        _cellrow("1", 0, "x", "bbbb", 6, rid="aaa"),
    ]
    rec = select_one_per_cell(rows, prefer_code_version="aaaa")
    assert [r["run_id"] for r in rec["kept"]] == ["aaa"]  # lowest run_id, never the better score


# ----------------------------------------------------- minimum detectable effect


def _closed_form(sd, n):
    return (1.959963984540054 + 0.8416212335729143) * sd / math.sqrt(n)


def test_the_sd_is_the_SAMPLE_sd_with_an_n_minus_1_denominator():
    """Not a detail. An MDE is an estimate from a sample, so the variance must be the unbiased
    one; the population sd would understate the floor by sqrt(19/20) at n=20 -- 2.5% in the
    optimistic direction, on a number whose whole job is to say "this is not enough data".
    Written first as `abs(sd - 1.0) < 1e-9`, which asserted the POPULATION sd and failed. The
    test was wrong, not the code."""
    vals = {f"c{i}": (1.0 if i % 2 else -1.0) for i in range(20)}
    assert abs(minimum_detectable_effect(vals)["sd"] - math.sqrt(20 / 19)) < 1e-12


def test_mde_comes_from_the_observed_sd_and_not_from_a_rule_of_thumb():
    tight = {f"c{i}": (1.0 if i % 2 else -1.0) for i in range(20)}
    wide = {f"c{i}": (10.0 if i % 2 else -10.0) for i in range(20)}
    m_t = minimum_detectable_effect(tight)
    m_w = minimum_detectable_effect(wide)
    assert abs(m_w["mde"] / m_t["mde"] - 10.0) < 1e-9  # linear in the observed sd
    assert abs(m_t["mde"] - _closed_form(math.sqrt(20 / 19), 20)) < 1e-12


def test_mde_uses_the_cluster_count_and_not_the_row_count():
    """229 rows on 20 tasks is a 20-cluster experiment. The floor must be computed at 20, and
    computing it at the row count would understate it by sqrt(229/20) = 3.4x."""
    twenty = {f"c{i}": (1.0 if i % 2 else -1.0) for i in range(20)}
    eighty = {f"c{i}": (1.0 if i % 2 else -1.0) for i in range(80)}
    mt, me = minimum_detectable_effect(twenty), minimum_detectable_effect(eighty)
    assert mt["n_clusters"] == 20 and me["n_clusters"] == 80
    assert abs(mt["mde"] - _closed_form(math.sqrt(20 / 19), 20)) < 1e-12
    assert abs(me["mde"] - _closed_form(math.sqrt(80 / 79), 80)) < 1e-12
    assert mt["mde"] > me["mde"]  # fewer clusters, a higher floor


def test_mde_names_its_alpha_power_and_the_variance_it_came_from():
    rec = minimum_detectable_effect({f"c{i}": float(i % 3) for i in range(20)})
    for field in ("alpha", "power", "sd", "var", "n_clusters", "mde", "observed_mean", "method"):
        assert field in rec, field
    assert abs(rec["var"] - rec["sd"] ** 2) < 1e-12


def test_mde_refuses_a_single_cluster_rather_than_inventing_a_variance():
    rec = minimum_detectable_effect({"c0": 1.0})
    assert math.isnan(rec["mde"]) and rec["n_clusters"] == 1
