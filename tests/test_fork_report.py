"""The paper's headline is a PAIRED-FORK quantity and it had no re-runnable report in the repo.

It was computed by scratchpad scripts: for each fork point (a reference trajectory prefix) and
seed, one `inquirer_prompted` continuation against one `self_ask` continuation, follow-ups =
`n_user_turns - n_prefix_user_turns`, the difference averaged per seed and pooled, a two-sided
sign test on the pairs, and `tau_reward` beside it. `ELIGIBLE` rightly excludes forks from the
main tables (they inherit a foreign prefix), so this protocol needs its own report, with its own
provenance: run ids, code versions, seeds, n pairs. Rule 1: a number without those is not a result.
"""

from __future__ import annotations

from pi_eval.fork_report import paired_engagement, sign_test_p


def _run(arm, seed, fork, fu_total, prefix, reward, code="abc1234"):
    return {
        "run_id": f"{arm}-{seed}-{fork}",
        "arm_id": arm,
        "seed": seed,
        "foreign_trace_sha": f"sha{fork}",
        "foreign_prefix_k": 2,
        "n_user_turns": fu_total,
        "n_prefix_user_turns": prefix,
        "tau_reward": reward,
        "code_version": code,
        "status": "ok",
    }


def test_sign_test_is_two_sided_and_exact():
    assert abs(sign_test_p(25, 6) - 0.000877) < 1e-4  # seed 0 of the retail headline
    assert sign_test_p(0, 0) != sign_test_p(0, 0)  # NaN when there are no informative pairs


def test_pairs_are_formed_per_fork_and_seed_and_unpaired_runs_are_counted():
    rows = [
        _run("inquirer_prompted", 0, "A", 9, 7, 1.0),  # 2 follow-ups
        _run("self_ask", 0, "A", 12, 7, 0.0),  # 5 follow-ups
        _run("inquirer_prompted", 0, "B", 10, 7, 0.0),  # 3
        _run("self_ask", 0, "B", 8, 7, 1.0),  # 1
        _run("inquirer_prompted", 1, "A", 8, 7, 1.0),  # unpaired: no self_ask at seed 1
    ]
    rep = paired_engagement(rows, treatment="inquirer_prompted", control="self_ask")
    assert rep["n_pairs"] == 2 and rep["n_unpaired"] == 1
    assert rep["per_seed"][0]["n_pairs"] == 2
    assert rep["per_seed"][0]["follow_ups"] == {"treatment": 2.5, "control": 3.0}
    assert rep["pooled"]["diff_mean"] == -0.5
    assert rep["pooled"]["fewer"] == 1 and rep["pooled"]["more"] == 1
    assert rep["pooled"]["reward"] == {"treatment": 0.5, "control": 0.5}
    assert rep["provenance"]["code_versions"] == {"abc1234": 4}
    assert rep["provenance"]["run_ids_sha"]  # a digest over the paired run ids, stable


def test_runs_without_a_foreign_prefix_are_not_forks_and_are_counted_not_paired():
    """The airline test set holds full-task runs beside the forks; two of them share the
    empty key (None, None, seed) and would be refused as duplicates. They are not part of
    the paired-fork protocol at all."""
    rows = [
        _run("inquirer_prompted", 0, "A", 9, 7, 1.0),
        _run("self_ask", 0, "A", 12, 7, 0.0),
        {
            **_run("inquirer_prompted", 0, "x", 9, 0, 1.0),
            "foreign_trace_sha": None,
            "foreign_prefix_k": None,
        },
        {
            **_run("inquirer_prompted", 0, "y", 8, 0, 1.0),
            "foreign_trace_sha": "",
            "foreign_prefix_k": None,
        },
    ]
    rep = paired_engagement(rows, treatment="inquirer_prompted", control="self_ask")
    assert rep["n_pairs"] == 1 and rep["n_not_forks"] == 2
