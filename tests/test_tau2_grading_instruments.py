"""The two tau2 instruments, pinned: a dead ask channel, and a positive control that scores 0.

WHY THESE TWO ASSERTIONS AND NOT THE OTHERS. Both scripts exist because a tau2 campaign can
run to `status: ok` on every unit while measuring nothing, and in both cases the assertion that
sees it is the one an ordinary reading of the output would drop:

  * `preflight_tau2_channel` -- on a retail fork unit the DRAFTER's agentic tool calls attach
    evidence and resolve required needs even while the Inquirer's own retrieval channel returns
    nothing, so assertions 2 and 3 PASS on a unit where 16 of 16 asks came back empty. Only
    assertion 1 (`retrieved_uids` non-empty on some ask) sees it. The first test below is
    exactly that configuration, because a preflight that reports "ok" there clears a sweep that
    cannot measure its own endpoint.

  * `verify_tau2_fork_grading` -- its fork verdict is only worth reading if the same code
    reproduces the FLAT grading that `pi verify tau2` reports 112/112. If the positive control
    scores 0 and the script goes on to print a fork verdict anyway, the instrument reports its
    own breakage as a finding about the grader. So the failure must be fatal AND must stop
    before the fork path runs at all.

Both scripts also distinguish "this instrument could not decide" from "the thing is broken",
and those branches are pinned here too: a task whose required gold nodes carry no
`gold_ev_uids` cannot satisfy assertion 3 under any policy, and zero graded fork points is a
fact about the instrument. Reporting either as a failure blocks a campaign for the wrong
reason; reporting either as a pass clears it without testing it.

The synthetic runs here carry no LLM, no environment and no gold root: `load_graphs` is
substituted, which is also what keeps this hermetic under the conftest env scrub.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from scripts import preflight_tau2_channel as pre
from scripts import verify_tau2_fork_grading as vfy

from pi_eval.gold import GoldGraph, GoldNode

SUITE = "tau2_retail"
TASK = "55"


# ------------------------------------------------------------------ synthetic run directories


def _write_run(
    root: Path,
    name: str,
    *,
    n_asks: int = 16,
    n_answered: int = 0,
    evidence_uids: tuple[str, ...] = (),
) -> Path:
    """A finished run directory, written exactly as a tau2 unit writes one.

    `n_answered` is the number of ask turns carrying a non-empty `retrieved_uids`: 0 is the
    measured shape of the killed campaign, and any positive value is a live channel.
    """
    d = root / name
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": name,
                "suite_id": SUITE,
                "task_id": TASK,
                "arm_id": "inquirer_prompted",
                "foreign_trace_sha": "deadbeef" * 8,
                "foreign_prefix_k": 4,
            }
        )
    )
    (d / "status.json").write_text(
        json.dumps({"status": "ok", "n_asks": n_asks, "stop_reason": "budget"})
    )
    turns = [{"action_kind": "draft", "retrieved_uids": []}]  # a non-ask turn is not an ask
    for i in range(n_asks):
        turns.append(
            {
                "action_kind": "ask",
                "retrieved_uids": ["kb:doc:1"] if i < n_answered else [],
            }
        )
    (d / "turns.jsonl").write_text("".join(json.dumps(t) + "\n" for t in turns))
    (d / "evidence.jsonl").write_text("".join(json.dumps({"uid": u}) + "\n" for u in evidence_uids))
    (d / "outcome.json").write_text(json.dumps({"env_calls": [{"name": "get_order_details"}]}))
    return d


def _graphs(*, required_uids: tuple[str, ...]) -> dict[str, GoldGraph]:
    """One required node, carrying `required_uids`. Empty means assertion 3 is undecidable."""
    return {
        TASK: GoldGraph(
            gold_suite=SUITE,
            gold_task_key=TASK,
            gold_nodes=(
                GoldNode(
                    gold_suite=SUITE,
                    gold_task_key=TASK,
                    gold_node_id="n1",
                    gold_text="the order the user means",
                    gold_partition="required",
                    gold_ev_uids=required_uids,
                ),
            ),
        )
    }


@pytest.fixture
def gold(monkeypatch):
    """Substitute `pi_eval.gold.load_graphs`, which `check_run` imports inside its body."""

    def _install(required_uids: tuple[str, ...]) -> None:
        monkeypatch.setattr(
            "pi_eval.gold.load_graphs", lambda suite, ver: _graphs(required_uids=required_uids)
        )

    return _install


# ------------------------------------------------------- 1. an ask that returned nothing is caught


def test_an_ask_that_returned_nothing_is_caught_though_evidence_and_needs_look_fine(tmp_path, gold):
    """The measured failure shape: 16 asks, 0 answered, and assertions 2 and 3 both passing.

    This is not a hypothetical. Over the fork campaign at `1d6d0f9` every ask on both domains
    came back empty while units wrote evidence and reached `status: ok`, and the retail arm's
    evidence came from the Drafter's own tool calls rather than from any question the Inquirer
    asked.
    """
    gold(("kb:doc:1",))
    d = _write_run(tmp_path, "dead", n_asks=16, n_answered=0, evidence_uids=("kb:doc:1",))
    row = pre.check_run(d)
    assert row["n_asks_recorded"] == 16
    assert row["n_asks_answered"] == 0
    assert row["a1_an_ask_was_answered"] is False
    # The two assertions that CANNOT see it, asserted true so that a future change which makes
    # them fail here does not make this test pass for the wrong reason.
    assert row["a2_evidence_non_empty"] is True
    assert row["a3_required_need_resolved"] is True
    assert row["a3_decidable"] is True
    assert row["ok"] is False


def test_a_live_channel_passes_so_the_check_is_not_a_constant(tmp_path, gold):
    """Non-vacuity: the same directory shape with ONE answered ask must pass all three."""
    gold(("kb:doc:1",))
    d = _write_run(tmp_path, "live", n_asks=16, n_answered=1, evidence_uids=("kb:doc:1",))
    row = pre.check_run(d)
    assert row["n_asks_answered"] == 1
    assert row["a1_an_ask_was_answered"] is True
    assert row["ok"] is True


def test_main_refuses_a_sweep_on_a_dead_channel(tmp_path, gold, capsys):
    gold(("kb:doc:1",))
    d = _write_run(tmp_path, "dead", n_asks=16, n_answered=0, evidence_uids=("kb:doc:1",))
    assert pre.main([str(d)]) == 1
    out = json.loads(capsys.readouterr().out)
    assert out["n_passed"] == 0
    assert out["verdict"].startswith("CHANNEL DEAD")


def test_a_task_that_cannot_decide_assertion_three_is_inconclusive_not_a_failure(
    tmp_path, gold, monkeypatch, capsys
):
    """A required node with no `gold_ev_uids` cannot be resolved by ANY policy.

    Reporting that as a dead channel would block a campaign for a property of the task, so the
    script must exit 2 and name tasks on which the question can be asked.
    """
    gold(())
    monkeypatch.setattr("pi_eval.gold.load_graphs", lambda suite, ver: _graphs(required_uids=()))
    d = _write_run(tmp_path, "undecidable", n_asks=16, n_answered=1, evidence_uids=("kb:doc:1",))
    assert pre.main([str(d)]) == 2
    out = json.loads(capsys.readouterr().out)
    assert out["verdict"].startswith("INCONCLUSIVE")
    assert out["units"][0]["unsuitable"] is True


# ------------------------------------------------- 2. a positive control that scores 0 is fatal


class _Suite:
    """Only `task_ids()` is reached once `grade_gold_on` is substituted."""

    domain = "retail"

    def task_ids(self):
        return ["1", "2", "3", "4"]


@pytest.fixture
def forkpoints(tmp_path) -> str:
    p = tmp_path / "forks.json"
    p.write_text(json.dumps({"fork_points": [{"task_id": TASK, "k": 4, "trace_sha": "ab" * 32}]}))
    return str(p)


@pytest.fixture
def wired(monkeypatch):
    """Substitute the two imports `main` performs, and count fork-path entries."""
    seen: dict[str, int] = {"prefix_loads": 0}

    monkeypatch.setattr("pi_run.worker.load_suite", lambda suite, split: _Suite())

    def _load_prefix(spec, root=None):
        seen["prefix_loads"] += 1
        return ["u1", "u2", "u3", "u4"]

    monkeypatch.setattr("pi_run.stages.tau2_runner.load_prefix", _load_prefix)
    return seen


def test_a_positive_control_that_scores_zero_is_fatal_and_never_reaches_the_fork_path(
    monkeypatch, wired, forkpoints, capsys
):
    """An instrument that cannot reproduce a known answer may not report a fork verdict."""
    monkeypatch.setattr(vfy, "grade_gold_on", lambda suite, tid, prefix: (0.0, 3, ""))
    code = vfy.main(["--suite", SUITE, "--positive-control", "3", "--forkpoints", forkpoints])
    out = json.loads(capsys.readouterr().out)
    assert code == 2
    assert out["verdict"].startswith("POSITIVE CONTROL FAILED")
    assert out["positive_control"]["ok"] is False
    assert out["positive_control"]["reproduced"] == 0
    # The fork path must not have run at all: no verdict about the grader, and no prefix read.
    assert out["fork"] == {}
    assert wired["prefix_loads"] == 0


def test_a_reproducing_positive_control_lets_the_fork_verdict_through(
    monkeypatch, wired, forkpoints, capsys
):
    """Non-vacuity for the test above: 1.0 everywhere is a clean pass on both halves."""
    monkeypatch.setattr(vfy, "grade_gold_on", lambda suite, tid, prefix: (1.0, 3, ""))
    code = vfy.main(["--suite", SUITE, "--positive-control", "3", "--forkpoints", forkpoints])
    out = json.loads(capsys.readouterr().out)
    assert code == 0
    assert out["positive_control"]["ok"] is True
    assert out["fork"]["reproduced"] == out["fork"]["graded"] == 1
    assert out["verdict"].startswith("FORK GRADING REPRODUCES")
    assert wired["prefix_loads"] == 1


def test_a_fork_point_that_does_not_reproduce_is_named_as_a_grader_failure(
    monkeypatch, wired, forkpoints, capsys
):
    """The flat path reproduces and the forked one does not: that, and only that, is a verdict
    about the fork grader."""
    monkeypatch.setattr(
        vfy, "grade_gold_on", lambda suite, tid, prefix: (0.0 if prefix else 1.0, 3, "")
    )
    code = vfy.main(["--suite", SUITE, "--positive-control", "3", "--forkpoints", forkpoints])
    out = json.loads(capsys.readouterr().out)
    assert code == 1
    assert out["positive_control"]["ok"] is True
    assert out["fork"]["graded"] == 1 and out["fork"]["reproduced"] == 0
    assert out["verdict"].startswith("FORK GRADING DOES NOT REPRODUCE")


def test_grading_nothing_is_inconclusive_rather_than_a_verdict(
    monkeypatch, wired, forkpoints, capsys
):
    """`graded == 0` is a fact about this instrument. The script's first draft printed a
    grader failure off a TypeError of its own making, which is the laundering this branch
    exists to prevent."""

    def _boom(suite, tid, prefix):
        if prefix:
            raise TypeError("set_state() takes 2 positional arguments but 3 were given")
        return 1.0, 3, ""

    monkeypatch.setattr(vfy, "grade_gold_on", _boom)
    vfy.main(["--suite", SUITE, "--positive-control", "3", "--forkpoints", forkpoints])
    out = json.loads(capsys.readouterr().out)
    assert out["fork"]["graded"] == 0
    assert out["verdict"].startswith("INCONCLUSIVE")
    assert "TypeError" in out["fork"]["errors"][0]["error"]
