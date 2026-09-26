"""The campaign scorer: upstream's pass^k, and the two ways it silently becomes another metric.

The first is `k`. Upstream's own reader LOWERS k to the smallest task's trial count and warns,
so one truncated task turns the whole table into a different number wearing the same label. The
second is the clustering unit: the four trials of a task share its script, its database and its
gold action set, so a bootstrap over simulations halves the standard error for free.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "tau2_full_protocol_score", ROOT / "scripts" / "tau2_full_protocol" / "score.py"
)
assert _spec and _spec.loader
S = importlib.util.module_from_spec(_spec)
sys.modules["tau2_full_protocol_score"] = S
_spec.loader.exec_module(S)


def test_pass_hat_k_matches_upstreams_own_function() -> None:
    """Transcribed, so it is checked against the original rather than against a memory of it."""
    from tau2.metrics.agent_metrics import pass_hat_k as theirs

    for n in (1, 2, 3, 4, 5):
        for c in range(n + 1):
            for k in range(1, n + 1):
                assert S.pass_hat_k(n, c, k) == theirs(n, c, k)


def test_pass_hat_k_refuses_fewer_trials_than_k() -> None:
    with pytest.raises(ValueError):
        S.pass_hat_k(3, 3, 4)


def _rows(spec):
    """spec: {task_id: [reward, ...]} -> the row shape `pass_k_table` reads."""
    out = []
    for tid, rewards in spec.items():
        for i, r in enumerate(rewards):
            out.append(
                {
                    "task_id": tid,
                    "seed": i,
                    "status": "ok",
                    "n_user_turns": 3,
                    "n_asks": 0,
                    "n_env_calls": 0,
                    "n_messages": 0,
                    "termination_reason": "TerminationReason.USER_STOP",
                    "usd_billed": 0.0,
                    "usage_usd": 0.0,
                    "user_sim_usd": 0.0,
                    "stock_agent_usd": 0.0,
                    "native": {"upstream_reward": r},
                }
            )
    return out


def test_pass_k_is_averaged_over_tasks_and_matches_hand_arithmetic() -> None:
    rows = _rows({"a": [1.0, 1.0, 1.0, 1.0], "b": [1.0, 1.0, 0.0, 0.0]})
    t = S.pass_k_table(rows, "upstream_reward", kmax=4)
    # a: 4 of 4 successes. b: 2 of 4.
    assert t["pass^1"] == pytest.approx((1.0 + 0.5) / 2)
    # pass^2: a = C(4,2)/C(4,2) = 1; b = C(2,2)/C(4,2) = 1/6.
    assert t["pass^2"] == pytest.approx((1.0 + 1.0 / 6.0) / 2)
    # pass^4: a = 1; b = C(2,4) = 0.
    assert t["pass^4"] == pytest.approx(0.5)
    assert t["n_tasks_at_4"] == 2
    assert t["avg_reward"] == pytest.approx(6 / 8)


def test_a_task_with_too_few_trials_is_dropped_from_k_and_counted() -> None:
    """NOT scored at the trials it has, and NOT allowed to lower k for the whole table --
    which is what upstream's own reader does and why this function does not call it."""
    rows = _rows({"a": [1.0, 1.0, 1.0, 1.0], "short": [1.0, 1.0]})
    t = S.pass_k_table(rows, "upstream_reward", kmax=4)
    assert t["n_tasks_seen"] == 2
    assert t["n_tasks_at_2"] == 2
    assert t["n_tasks_at_4"] == 1, "the two-trial task cannot contribute to pass^4"
    assert t["pass^4"] == pytest.approx(1.0), "and the surviving task is not diluted by it"


def test_a_partial_reward_is_not_a_success() -> None:
    """Upstream's `is_successful` is reward within 1e-6 of 1.0. pass^k counts successes, so a
    0.5 must not be half a pass."""
    t = S.pass_k_table(_rows({"a": [0.5, 0.5, 0.5, 0.5]}), "upstream_reward", kmax=4)
    assert t["pass^1"] == 0.0
    assert t["avg_reward"] == pytest.approx(0.5)


def test_a_run_with_no_upstream_reward_contributes_no_trial() -> None:
    """Absent is not zero. A unit whose grade failed carries `upstream_error` and no reward,
    and counting it as a failure would report an ungradable transcript as a policy failure."""
    rows = _rows({"a": [1.0, 1.0, 1.0]})
    rows.append({**rows[0], "seed": 3, "native": {"upstream_error": "boom"}})
    t = S.pass_k_table(rows, "upstream_reward", kmax=4)
    assert t["n_tasks_at_3"] == 1
    assert t["n_tasks_at_4"] == 0


def test_the_paired_contrast_is_clustered_on_the_task() -> None:
    """One task, four trials, a constant difference. Clustered on the task there is ONE unit
    and the interval is degenerate; pooled over trials there would be four and the bootstrap
    would manufacture a spread."""
    a = S.per_task_mean(_rows({"t": [1.0, 1.0, 1.0, 1.0]}), lambda r: r["n_user_turns"])
    b = S.per_task_mean(_rows({"t": [1.0, 1.0, 1.0, 1.0]}), lambda r: r["n_user_turns"])
    out = S.paired_boot(a, b, n_boot=200, seed=0)
    assert out["n_tasks_paired"] == 1
    assert out["difference"] == pytest.approx(0.0)


def test_the_paired_contrast_uses_only_tasks_present_in_both_arms() -> None:
    a = {"x": 1.0, "y": 2.0, "z": 9.0}
    b = {"x": 0.0, "y": 0.0}
    out = S.paired_boot(a, b, n_boot=200, seed=0)
    assert out["n_tasks_paired"] == 2
    assert out["difference"] == pytest.approx(1.5)


def test_a_named_run_id_that_is_missing_raises(tmp_path) -> None:
    """A silently skipped run turns a population into whatever happened to be readable."""
    with pytest.raises(FileNotFoundError):
        S.load_runs(str(tmp_path), ["nope"])


def test_load_runs_reads_the_upstream_reward_out_of_outcome_json(tmp_path) -> None:
    d = tmp_path / "rid1"
    d.mkdir()
    (d / "manifest.json").write_text(
        json.dumps(
            {
                "suite_id": "tau2_retail",
                "task_id": "7",
                "arm_id": "tau2_stock",
                "seed": 2,
                "split": "test",
                "code_version": "abc",
                "upstream_pins": {"protocol": "tau2_upstream"},
            }
        )
    )
    (d / "status.json").write_text(
        json.dumps({"status": "ok", "n_user_turns": 5, "usd_billed": 0.5, "usage": {"usd": 0.7}})
    )
    (d / "outcome.json").write_text(
        json.dumps({"native": {"upstream_reward": 1.0, "tau_reward": 1.0}})
    )
    (row,) = S.load_runs(str(tmp_path), ["rid1"])
    assert row["native"]["upstream_reward"] == 1.0
    assert row["usage_usd"] == 0.7
    assert row["usd_billed"] == 0.5
    assert row["upstream_pins"]["protocol"] == "tau2_upstream"
