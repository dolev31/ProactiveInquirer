"""Tests for `scripts/length_matched/paired_contrast.py`'s join and McNemar/bootstrap logic.

These are pure-Python, GPU-free unit tests of the CONTRAST arithmetic only -- they do not touch
`score_checkpoint_paired.py`'s model-loading path (that needs a real checkpoint and a GPU, and
is exercised on the cluster, not here). Per CONTRIBUTING.md rule 2, each test is built to FAIL if the
join or the statistic is wrong, not merely to exercise the code: every expected number below is
computed by hand in the test body's comment, not copied from a first run of the implementation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "length_matched"))

from paired_contrast import join_by_pair_id, mcnemar_contrast, paired_bootstrap_ci  # noqa: E402


def test_join_by_pair_id_refuses_mismatched_ids():
    a = {"p1": {"ok": 1}, "p2": {"ok": 0}}
    b = {"p1": {"ok": 1}, "p3": {"ok": 1}}
    with pytest.raises(ValueError, match="pair_id sets differ"):
        join_by_pair_id(a, b)


def test_join_by_pair_id_joins_matching_ids():
    a = {"p1": {"ok": 1}, "p2": {"ok": 0}}
    b = {"p1": {"ok": 0}, "p2": {"ok": 1}}
    joined = join_by_pair_id(a, b)
    assert joined == {"p1": (1, 0), "p2": (0, 1)}


def test_mcnemar_all_concordant_gives_zero_diff_and_no_pvalue():
    # 4 pairs, A and B agree on every one (2 both-correct, 2 both-wrong): no discordant pairs,
    # diff must be exactly 0, and the p-value must be None (not a fabricated 1.0).
    joined = {"p1": (1, 1), "p2": (1, 1), "p3": (0, 0), "p4": (0, 0)}
    r = mcnemar_contrast(joined)
    assert r["n"] == 4
    assert r["acc_a"] == r["acc_b"] == 0.5
    assert r["diff_b_minus_a"] == 0.0
    assert r["n_discordant"] == 0
    assert r["mcnemar_exact_p_two_sided"] is None
    assert r["note"] is not None


def test_mcnemar_known_discordant_counts_and_pvalue():
    # 10 pairs: 3 both-correct, 2 both-wrong, 1 A-only-correct, 4 B-only-correct.
    # acc_a = (3+1)/10 = 0.4, acc_b = (3+4)/10 = 0.7, diff = +0.3.
    # n_discordant = 1 + 4 = 5; McNemar exact two-sided p = binomtest(4, 5, 0.5).pvalue.
    joined = {}
    for i in range(3):
        joined[f"bc{i}"] = (1, 1)
    for i in range(2):
        joined[f"bw{i}"] = (0, 0)
    joined["aonly0"] = (1, 0)
    for i in range(4):
        joined[f"bonly{i}"] = (0, 1)
    r = mcnemar_contrast(joined)
    assert r["n"] == 10
    assert r["acc_a"] == pytest.approx(0.4)
    assert r["acc_b"] == pytest.approx(0.7)
    assert r["diff_b_minus_a"] == pytest.approx(0.3)
    assert r["both_correct"] == 3
    assert r["both_wrong"] == 2
    assert r["a_only_correct"] == 1
    assert r["b_only_correct"] == 4
    assert r["n_discordant"] == 5
    from scipy.stats import binomtest

    expected_p = binomtest(4, 5, 0.5, alternative="two-sided").pvalue
    assert r["mcnemar_exact_p_two_sided"] == pytest.approx(expected_p)


def test_mcnemar_refuses_empty_join():
    with pytest.raises(ValueError, match="empty rung"):
        mcnemar_contrast({})


def test_paired_bootstrap_ci_brackets_a_known_zero_diff():
    # A and B are identical on every pair -> point diff is exactly 0 and every bootstrap
    # resample (drawing pairs, not arms, so both sides move together) must also read exactly 0.
    joined = {f"p{i}": (i % 2, i % 2) for i in range(50)}
    r = paired_bootstrap_ci(joined, n_boot=200, seed=1)
    assert r["point_diff_b_minus_a"] == 0.0
    assert r["ci_lo"] == 0.0
    assert r["ci_hi"] == 0.0


def test_paired_bootstrap_ci_point_matches_direct_computation():
    # 20 pairs, B correct 15/20, A correct 5/20 -> point diff = 0.75 - 0.25 = 0.50.
    joined = {}
    for i in range(5):
        joined[f"aonly{i}"] = (1, 0)
    for i in range(15):
        joined[f"bonly{i}"] = (0, 1)
    r = paired_bootstrap_ci(joined, n_boot=500, seed=2)
    assert r["point_diff_b_minus_a"] == pytest.approx(0.50)
    assert r["ci_lo"] <= r["point_diff_b_minus_a"] <= r["ci_hi"]
