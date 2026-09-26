"""A paired tau2 cell is a FORK POINT, and a duplicate must be refused, not picked from.

These pin `pi_eval.fork_report._pair_forks` against the mistake three scripts in
`scripts/tau2_concordance/` made on 2026-09-22 and which this repository's own pairing never
made: keying on (task_id, seed, prompt_variant). Several fork points come from the SAME task at
different depths, so that key merges distinct fork points, and a `setdefault`-and-assign then
drops whichever run was walked first (213 ok airline units collapsed to 113 cells).
"""

from __future__ import annotations

import pytest

from pi_eval.fork_report import _pair_forks

T, C = "inquirer_trained", "inquirer_prompted"


def run(arm=T, task="45", seed=2, sha="aaa", k=8, run_id=None):
    return {
        "arm_id": arm,
        "task_id": task,
        "seed": seed,
        "foreign_trace_sha": sha,
        "foreign_prefix_k": k,
        "run_id": run_id or f"{arm[9:13]}-{task}-{seed}-{sha}-{k}",
        "status": "ok",
    }


def test_the_two_arms_of_one_fork_make_one_pair():
    pairs, unpaired, not_forks = _pair_forks([run(T), run(C)], treatment=T, control=C)
    assert len(pairs) == 1 and unpaired == 0 and not_forks == 0


def test_two_fork_depths_of_one_task_are_two_cells():
    """THE REGRESSION. Same task, seed and arm pair; different foreign_prefix_k."""
    runs = [
        run(T, k=8, sha="aaa"),
        run(C, k=8, sha="aaa"),
        run(T, k=14, sha="bbb"),
        run(C, k=14, sha="bbb"),
    ]
    pairs, _, _ = _pair_forks(runs, treatment=T, control=C)
    assert len(pairs) == 2, "two fork depths of one task are two pairs, not one"


def test_the_superseded_key_would_have_merged_them():
    """Pins WHY: (task, seed, variant) cannot tell two depths of one task apart."""
    a, b = run(T, k=8, sha="aaa"), run(T, k=14, sha="bbb")
    superseded = lambda r: (r["task_id"], r["seed"])  # noqa: E731
    correct = lambda r: (r["foreign_trace_sha"], r["foreign_prefix_k"], r["seed"])  # noqa: E731
    assert superseded(a) == superseded(b)
    assert correct(a) != correct(b)


def test_a_second_run_of_one_arm_is_refused_not_overwritten():
    """The difference between 213 units collapsing silently and the report stopping."""
    with pytest.raises(ValueError, match="two inquirer_trained runs at fork"):
        _pair_forks([run(T, run_id="first"), run(T, run_id="second")], treatment=T, control=C)


def test_a_run_with_no_foreign_prefix_is_excluded_not_collided():
    runs = [run(T), run(C), {**run(T, run_id="x"), "foreign_trace_sha": None}]
    pairs, _, not_forks = _pair_forks(runs, treatment=T, control=C)
    assert len(pairs) == 1 and not_forks == 1


def test_partitioning_by_prompt_variant_restores_uniqueness():
    """How this campaign uses it: (trace, k, seed) is not unique when a variant dimension exists."""
    base = [run(T), run(C)]
    stop = [run(T, run_id="t2"), run(C, run_id="c2")]
    with pytest.raises(ValueError):
        _pair_forks(base + stop, treatment=T, control=C)
    for part in (base, stop):
        pairs, _, _ = _pair_forks(part, treatment=T, control=C)
        assert len(pairs) == 1


def test_the_endpoint_cross_check_refuses_on_disagreement(tmp_path):
    """The two-route check must be able to FAIL, or it is decoration.

    `n_env_calls` and `len(outcome.json["env_calls"])` are the same quantity by two routes. A
    keying defect was found on 2026-09-22 only because two counts of one population disagreed and
    no check had failed, so this comparison is kept permanently -- and a check that cannot fail
    would reproduce the original problem.
    """
    import json
    import subprocess
    import sys
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts" / "tau2_concordance" / "fork_paired.py"

    def record(run_id, n_env_calls, from_outcome):
        return {
            "status": "ok",
            "arm_id": T,
            "task_id": "45",
            "seed": 0,
            "foreign_trace_sha": "aaa",
            "foreign_prefix_k": 8,
            "run_id": run_id,
            "key": ["tau2_airline", "45", T, 0, "", -1, -1, "aaa", 8, "tau2_base", ""],
            "n_env_calls": n_env_calls,
            "n_asks": 3,
            "native": {"tau_reward": 1.0},
            "_env_calls_from_outcome": from_outcome,
        }

    def pair(n_t, out_t, n_c, out_c):
        t_rec = record("r_t", n_t, out_t)
        c_rec = record("r_c", n_c, out_c)
        c_rec["arm_id"] = C
        c_rec["key"] = ["tau2_airline", "45", C, 0, "", -1, -1, "aaa", 8, "tau2_base", ""]
        return [t_rec, c_rec]

    agree = tmp_path / "agree.json"
    agree.write_text(json.dumps(pair(7, 7, 9, 9)))
    ok = subprocess.run(
        [sys.executable, str(script), "--records", str(agree)], capture_output=True, text=True
    )
    assert ok.returncode == 0, ok.stderr
    assert "cross-check: n_env_calls == len(outcome.env_calls) on 2 of 2" in ok.stdout

    disagree = tmp_path / "disagree.json"
    disagree.write_text(json.dumps(pair(7, 9, 9, 9)))
    bad = subprocess.run(
        [sys.executable, str(script), "--records", str(disagree)], capture_output=True, text=True
    )
    assert bad.returncode == 3, f"must refuse, got {bad.returncode}: {bad.stdout}{bad.stderr}"
    assert "REFUSING" in bad.stdout


def _run_script(tmp_path, records, name):
    import json
    import subprocess
    import sys
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts" / "tau2_concordance" / "fork_paired.py"
    f = tmp_path / name
    f.write_text(json.dumps(records))
    return subprocess.run(
        [sys.executable, str(script), "--records", str(f)], capture_output=True, text=True
    )


def _rec(arm, run_id, reward=1.0, calls=5):
    return {
        "status": "ok",
        "arm_id": arm,
        "task_id": "45",
        "seed": 0,
        "foreign_trace_sha": "aaa",
        "foreign_prefix_k": 8,
        "run_id": run_id,
        "key": ["tau2_airline", "45", arm, 0, "", -1, -1, "aaa", 8, "tau2_base", ""],
        "n_env_calls": calls,
        "n_asks": 3,
        "native": {"tau_reward": reward},
        "_env_calls_from_outcome": calls,
    }


def test_an_empty_store_is_refused_not_reported(tmp_path):
    """An empty store reads like a clean one to any check that only looks for violations."""
    r = _run_script(tmp_path, [], "empty.json")
    assert r.returncode == 3, f"expected refusal, got {r.returncode}: {r.stdout}"
    assert "no runs with status ok" in r.stdout


def test_a_variant_with_no_pairs_does_not_print_a_2x2_of_zeros(tmp_path):
    """`prompted_only = 0` from no pairs prints identically to the real finding. Refuse instead."""
    r = _run_script(tmp_path, [_rec(T, "only-one-arm")], "onearm.json")
    assert r.returncode == 3, f"expected refusal, got {r.returncode}: {r.stdout}"
    assert "2x2  REFUSED" in r.stdout
    assert "prompted_only=0" not in r.stdout, (
        "a zero cell must not be printed with no pairs behind it"
    )
    assert "0 complete pairs in any variant" in r.stdout


def test_a_real_pair_is_reported_so_the_refusals_are_not_blanket(tmp_path):
    """The refusals must not fire on a healthy input, or they are just an off switch."""
    r = _run_script(tmp_path, [_rec(T, "t", calls=4), _rec(C, "c", calls=9)], "pair.json")
    assert r.returncode == 0, f"expected success, got {r.returncode}: {r.stdout}"
    assert "prompted_only=0" in r.stdout
    assert "REFUS" not in r.stdout


def test_the_unanimous_bound_and_the_sign_test_are_the_same_condition():
    """For an n-0 split, "bound clears 0.5" and "two-sided p < 0.05" are ONE inequality: 2**-n < 0.025.

    So the bound adds no inferential content over the test for a unanimous split; its value is the
    only thing it contributes. Pinned because a future reader may otherwise treat two agreeing
    criteria as two pieces of evidence.
    """
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "scripts" / "tau2_concordance"))
    import fork_paired

    from pi_eval.fork_report import sign_test_p

    for n in range(1, 13):
        p_sig = sign_test_p(n, 0) < 0.05
        bound = fork_paired.cp_lower_bound(n, n)
        assert bound is not None
        assert p_sig == (bound > 0.5), f"n={n}: p<0.05 is {p_sig} but bound>0.5 is {bound > 0.5}"
        assert p_sig == (n >= 6), f"n={n}: threshold must be n>=6"


def test_the_bound_is_defined_when_a_counter_example_appears():
    """k < n is where the bound is MOST informative, so returning None there would hide it."""
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "scripts" / "tau2_concordance"))
    import fork_paired

    six_zero = fork_paired.cp_lower_bound(6, 6)
    seven_zero = fork_paired.cp_lower_bound(7, 7)
    six_one = fork_paired.cp_lower_bound(6, 7)
    five_one = fork_paired.cp_lower_bound(5, 6)

    assert round(six_zero, 4) == 0.5407
    assert round(seven_zero, 4) == 0.5904, "a 7th cluster agreeing must STRENGTHEN the magnitude"
    assert round(six_one, 4) == 0.4213, "a 7th disagreeing must drop it BELOW chance"
    assert round(five_one, 4) == 0.3588, "a flip among the six must drop it further"
    assert six_one < 0.5 < six_zero < seven_zero, "the two outcomes must be asymmetric about chance"
    assert fork_paired.cp_lower_bound(0, 5) == 0.0
    assert fork_paired.cp_lower_bound(3, 2) is None


def _typed_pair(tool_counts_t, tool_counts_c):
    t = _rec(T, "t", calls=sum(tool_counts_t.values()))
    c = _rec(C, "c", calls=sum(tool_counts_c.values()))
    t["_env_tool_counts"] = dict(tool_counts_t)
    c["_env_tool_counts"] = dict(tool_counts_c)
    return [t, c]


def test_typed_endpoints_split_lookups_from_actions(tmp_path):
    """n_env_calls sums lookups and actions, which move in opposite directions; report them apart."""
    r = _run_script(
        tmp_path,
        _typed_pair({"get_user_details": 2, "book_reservation": 1}, {"get_user_details": 6}),
        "typed.json",
    )
    assert r.returncode == 0, r.stdout
    assert "READ calls (agent lookups)" in r.stdout
    assert "WRITE calls (agent actions)" in r.stdout
    assert "every tool is mapped on 2 of 2 runs" in r.stdout


def test_an_unmapped_tool_refuses_instead_of_counting_as_zero(tmp_path):
    """A tool with no declared type must not silently drop out of READ and WRITE."""
    r = _run_script(
        tmp_path, _typed_pair({"not_a_real_tool": 3}, {"get_user_details": 3}), "unmapped.json"
    )
    assert r.returncode == 3, r.stdout
    assert "no declared type" in r.stdout


def test_tool_counts_that_do_not_sum_to_n_env_calls_refuse(tmp_path):
    recs = _typed_pair({"get_user_details": 4}, {"get_user_details": 4})
    recs[0]["n_env_calls"] = 9
    recs[0]["_env_calls_from_outcome"] = 9
    r = _run_script(tmp_path, recs, "badsum.json")
    assert r.returncode == 3, r.stdout
    assert "do not sum to n_env_calls" in r.stdout


def test_records_without_tool_counts_say_not_recorded_rather_than_zero(tmp_path):
    """Old extractions lack the field. The absence must read as absence, never as zero lookups."""
    r = _run_script(tmp_path, [_rec(T, "t", calls=4), _rec(C, "c", calls=9)], "untyped.json")
    assert r.returncode == 0, r.stdout
    assert "not recorded in these records" in r.stdout
    assert "READ calls (agent lookups)" not in r.stdout
