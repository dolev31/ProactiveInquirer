"""`pi train task-ids`: produce the train-split id file rung 0 requires.

`cmd_train_rung0` refuses without `--task-ids` ("rung 0 scores candidates on a FIXED task
set"), reads it as `suite_id/task_id` lines, and NO COMMAND PRODUCED IT. Rung 0 is on the
critical path for the headline claim, not just for training: `conf/prompts/rung0/` is
absent, so no arm runs an optimised prompt today.

The file must contain TRAIN-split ids only. Rung 0 selects a prompt by scoring candidates,
so any task it touches is burned for evaluation -- and unlike a pilot, nothing downstream
records that it was.
"""

from __future__ import annotations

import pytest


def test_the_writer_emits_suite_slash_task_lines(tmp_path) -> None:
    from pi_run.cmd_train import write_task_ids

    out = tmp_path / "ids.txt"
    n = write_task_ids(out, [("musique", "t1"), ("musique", "t2")])
    assert n == 2
    lines = [x for x in out.read_text().splitlines() if x and not x.startswith("#")]
    assert lines == ["musique/t1", "musique/t2"]


def test_the_file_round_trips_through_the_reader_rung0_uses(tmp_path) -> None:
    """The two halves must agree, or the file is written in a format nothing reads."""
    from pi_run.cmd_train import _read_task_keys, write_task_ids

    class _Gepa:
        class TaskKey:
            def __init__(self, *, suite_id, task_id):
                self.suite_id, self.task_id = suite_id, task_id

    out = tmp_path / "ids.txt"
    write_task_ids(out, [("musique", "a"), ("strategyqa", "b")])
    keys = _read_task_keys(_Gepa, out)
    assert [(k.suite_id, k.task_id) for k in keys] == [("musique", "a"), ("strategyqa", "b")]


def test_a_header_comment_is_written_and_ignored_by_the_reader(tmp_path) -> None:
    """Provenance in the file itself: which split, which command, how many."""
    from pi_run.cmd_train import _read_task_keys, write_task_ids

    class _Gepa:
        class TaskKey:
            def __init__(self, *, suite_id, task_id):
                self.suite_id, self.task_id = suite_id, task_id

    out = tmp_path / "ids.txt"
    write_task_ids(out, [("musique", "a")], split="train")
    assert out.read_text().startswith("#")
    assert "train" in out.read_text().splitlines()[0]
    assert len(_read_task_keys(_Gepa, out)) == 1


def test_writing_zero_ids_refuses(tmp_path) -> None:
    """An empty id file makes rung 0 score a candidate on nothing and report a winner."""
    from pi_run.cmd_train import EmptyTaskIds, write_task_ids

    with pytest.raises(EmptyTaskIds):
        write_task_ids(tmp_path / "ids.txt", [])


def test_the_ids_are_deduplicated_and_ordered(tmp_path) -> None:
    """A repeated id would weight one task twice in the candidate score."""
    from pi_run.cmd_train import write_task_ids

    out = tmp_path / "ids.txt"
    n = write_task_ids(out, [("m", "b"), ("m", "a"), ("m", "b")])
    assert n == 2
    lines = [x for x in out.read_text().splitlines() if not x.startswith("#")]
    assert lines == ["m/a", "m/b"]


def test_only_train_split_ids_are_selected() -> None:
    """The selection itself, against the real splitter."""
    from pi_run.cmd_train import select_train_ids
    from pinq.splitting import split_of

    ids = [f"t{i}" for i in range(200)]
    picked = select_train_ids("musique", ids, template_of=lambda t: None)
    assert picked, "no train ids selected from 200 candidates"
    assert all(split_of("musique", t, None) == "train" for t in picked)
    assert len(picked) < len(ids), "a split that keeps everything is not a split"


def test_the_limit_takes_a_prefix_not_a_sample() -> None:
    """Reproducibility: the same limit must give the same tasks."""
    from pi_run.cmd_train import select_train_ids

    ids = [f"t{i}" for i in range(200)]
    a = select_train_ids("musique", ids, template_of=lambda t: None, limit=5)
    b = select_train_ids("musique", ids, template_of=lambda t: None, limit=5)
    assert a == b and len(a) == 5
    assert a == select_train_ids("musique", ids, template_of=lambda t: None)[:5]


def test_selection_buckets_on_the_TEMPLATE_not_the_task_id() -> None:
    """The template is what stops a task leaking across the split, so it must be consulted.

    MEASURED on the real suite: musique task `4hop1__38130_8966_31714_79432` carries
    template `4hop1__31714_38130_79432_8966` -- the same hop set in canonical order. Tasks
    that share a hop set share a template, and `split_of` buckets on the template so they
    all land on one side. Selecting with `template_id=None` puts that task in TRAIN while
    the real bucketing calls it TEST: a train/test leak, produced by the very command meant
    to prevent one. (That mistake was made while verifying this, in the check rather than
    the code.)
    """
    from pi_run.cmd_train import select_train_ids
    from pinq.splitting import split_of

    task = "4hop1__38130_8966_31714_79432"
    template = "4hop1__31714_38130_79432_8966"
    assert split_of("musique", task, None) != split_of("musique", task, template), (
        "this task no longer distinguishes the two, pick another from the suite"
    )

    with_template = select_train_ids("musique", [task], template_of=lambda t: template)
    without = select_train_ids("musique", [task], template_of=lambda t: None)
    assert with_template != without, "the selector ignored the template it was handed"
