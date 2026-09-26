"""A replay arm must replay the questions of the run it is PAIRED with.

`parallel_replay` exists as a determinism check. `pi report killswitch` states its rule
outright: "the delta is identically zero by construction, so a MATCH confirms purity ... A
NON-ZERO delta is the finding: it indicts the harness -- a Drafter that is not pure in
(view, subset_hash, seed), a retriever with hidden state, or a leaking cache -- and every
phi_LOO, prefix-ladder and stop-test number is void until it is explained."

That construction rests entirely on replaying the PAIRED run's questions. `recorded_questions`
did not:

  - it keyed on task_id ALONE, so a replay at seed 1 could be handed seed 0's questions,
    though the treatment asks different questions at each seed;
  - it took the first match in run-id sort order, which is an arbitrary hash ordering;
  - it filtered neither inadmissible (dev-) runs nor RUNG-2 BRANCH runs, which are deliberate
    off-policy forks of a single turn.

MEASURED on the first admissible musique sweep: 5 of 12 (task, seed) cells replayed a
question list that was not the treatment's, one of them recognisably a candidate-branch
question ("What region did Andy Bernard from The Office sail to?"). The kill switch then read
parallel_replay delta 0.0208 instead of 0 -- a manufactured indictment of the harness, which
is the most alarming verdict this project can produce.
"""

from __future__ import annotations

import json
from pathlib import Path

from pinq_expt.policies.controls import recorded_questions


def _run(
    root: Path,
    run_id: str,
    *,
    task: str,
    seed: int,
    arm: str = "inquirer_prompted",
    questions: list[str],
    dirty: bool = False,
    branch_of: str | None = None,
) -> None:
    d = root / run_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "suite_id": "musique",
                "arm_id": arm,
                "task_id": task,
                "seed": seed,
                "dirty": dirty,
                "branch_of_run_id": branch_of,
            }
        )
    )
    (d / "turns.jsonl").write_text(
        "\n".join(json.dumps({"action_kind": "ask", "question": q}) for q in questions)
    )


def test_the_seed_selects_the_paired_run(tmp_path: Path) -> None:
    """The defect that produced the false verdict."""
    _run(tmp_path, "aaa", task="t1", seed=0, questions=["q-seed0"])
    _run(tmp_path, "bbb", task="t1", seed=1, questions=["q-seed1"])
    assert recorded_questions(tmp_path, arm_id="inquirer_prompted", suite_id="musique", seed=1) == {
        "t1": ["q-seed1"]
    }
    assert recorded_questions(tmp_path, arm_id="inquirer_prompted", suite_id="musique", seed=0) == {
        "t1": ["q-seed0"]
    }


def test_a_branch_run_is_never_mined(tmp_path: Path) -> None:
    """A rung-2 branch is an off-policy fork of one turn, not the policy's own rollout.

    'aaa' sorts first, so without the filter the branch would win on run-id order alone.
    """
    _run(tmp_path, "aaa", task="t1", seed=0, questions=["branch-q"], branch_of="parent")
    _run(tmp_path, "zzz", task="t1", seed=0, questions=["real-q"])
    assert recorded_questions(tmp_path, arm_id="inquirer_prompted", suite_id="musique", seed=0) == {
        "t1": ["real-q"]
    }


def test_an_admissible_run_beats_a_dev_run(tmp_path: Path) -> None:
    """Same ordering trap: 'aaa' would otherwise win over the admissible 'zzz'."""
    _run(tmp_path, "aaa", task="t1", seed=0, questions=["dirty-q"], dirty=True)
    _run(tmp_path, "zzz", task="t1", seed=0, questions=["clean-q"])
    assert recorded_questions(tmp_path, arm_id="inquirer_prompted", suite_id="musique", seed=0) == {
        "t1": ["clean-q"]
    }


def test_a_dev_run_is_still_usable_when_it_is_all_there_is(tmp_path: Path) -> None:
    """A dirty sweep must still be able to seed its own replay arms."""
    _run(tmp_path, "aaa", task="t1", seed=0, questions=["dirty-q"], dirty=True)
    assert recorded_questions(tmp_path, arm_id="inquirer_prompted", suite_id="musique", seed=0) == {
        "t1": ["dirty-q"]
    }


def test_no_seed_filter_keeps_the_old_behaviour(tmp_path: Path) -> None:
    """`seed=None` must still return something: other callers do not pair by seed."""
    _run(tmp_path, "aaa", task="t1", seed=0, questions=["q0"])
    got = recorded_questions(tmp_path, arm_id="inquirer_prompted", suite_id="musique")
    assert got == {"t1": ["q0"]}


def test_a_task_with_no_run_at_that_seed_is_absent(tmp_path: Path) -> None:
    """Absent, not silently filled from another seed -- the arm refuses loudly instead."""
    _run(tmp_path, "aaa", task="t1", seed=0, questions=["q0"])
    assert (
        recorded_questions(tmp_path, arm_id="inquirer_prompted", suite_id="musique", seed=9) == {}
    )
