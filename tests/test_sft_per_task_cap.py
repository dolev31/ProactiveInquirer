"""No task may dominate the SFT set, and the diversity gate is why.

MEASURED, on the live export. `within_task_distinct_n` averages distinct-3 PER TASK, and
distinct-3 collapses as a task contributes more questions:

    questions/task   2-3    3-5    5-10   10-20   20-50   50+
    mean distinct-3  0.81   0.81   0.70   0.48    0.28    0.13

127 tasks contributed 50 or more questions each and scored 0.13. The whole set came to 0.638
against a floor of 0.65 -- a FAIL -- while the same rows capped at 20 per task score 0.666 and
at 10 per task 0.701. The over-represented tasks were not merely diluting the average; they
were the failure.

WHY A CAP AND NOT MORE DATA. Adding thin tasks raises the average too (365 new tasks at ~2
questions each would reach 0.688), but it does that WITHOUT making any repetitive task less
repetitive -- arithmetic, not repair. Both are worth doing; only the cap makes the pass a
property of the questions rather than of how many a task happened to get.

DETERMINISTIC, AND NOT BY VALUE. Keeping the highest-value examples would select exactly the
questions the reward already likes, which is the confound the gate exists to catch -- and
keeping the most DIVERSE subset would optimise the metric directly, which is worse. Selection
is by a stable hash of the example's identity, the same rule `MAX_PAIRS_PER_STATE` uses.

THE CAP IS OFF BY DEFAULT. It changes what a checkpoint imitates, so it belongs in the
manifest and on an explicit flag, never as a silent default.
"""

from __future__ import annotations

import json

from pinq.actions import ask_action_json
from pinq_train.export.dataset import export_sft


def _row(task, run, turn, q, *, value=0.9, done=False):
    return {
        "suite_id": "musique",
        "task_id": task,
        "template_id": None,
        "run_id": run,
        "turn_idx": turn,
        # One prompt per state. This was the constant "S", which made twenty distinct done
        # states byte-identical -- and under exact dedupe (one demonstration per (task,
        # prompt, target)) identical bytes ARE one row. The belief that state identity lives
        # in `run_id` was the fixture's, not the exporter's; a real state renders its own
        # evidence, so a real pair of distinct states never shares a prompt by accident.
        "state_text": f"S-{run}-{turn}",
        "action_json": ask_action_json(q),
        "value": value,
        "phi_tilde": value,
        "scorer_hash": "sh",
        "graph_version": "v1",
        "is_stop": False,
        "done_before": done,
        "coverage_before": 1.0 if done else 0.4,
    }


def _rows(task, n, *, done=False):
    """n distinct states of one task, each with its own question."""
    return [
        _row(task, f"r{task}-{i}", i, f"question number {i} about {task}?", done=done)
        for i in range(n)
    ]


def test_a_task_over_the_cap_is_trimmed():
    from pinq_train.split import split_of

    t = next(f"t{i}" for i in range(500) if split_of("musique", f"t{i}") == "train")
    ex, man = export_sft(_rows(t, 30), margin_threshold=0.0, max_examples_per_task=10)
    assert len(ex) == 10
    assert man.n_over_task_cap_dropped == 20
    assert man.max_examples_per_task == 10


def test_a_task_under_the_cap_is_untouched():
    from pinq_train.split import split_of

    t = next(f"t{i}" for i in range(500) if split_of("musique", f"t{i}") == "train")
    ex, man = export_sft(_rows(t, 4), margin_threshold=0.0, max_examples_per_task=10)
    assert len(ex) == 4 and man.n_over_task_cap_dropped == 0


def test_the_cap_is_off_by_default():
    """It changes what the checkpoint imitates, so it is never silent."""
    from pinq_train.split import split_of

    t = next(f"t{i}" for i in range(500) if split_of("musique", f"t{i}") == "train")
    ex, man = export_sft(_rows(t, 30), margin_threshold=0.0)
    assert len(ex) == 30 and man.max_examples_per_task is None
    assert man.n_over_task_cap_dropped == 0


def test_selection_is_deterministic_and_not_by_value():
    """Two exports of the same rows keep the same examples; and the kept set is not simply
    the top-value ones, which would select for the reward the gate is meant to audit."""
    from pinq_train.split import split_of

    t = next(f"t{i}" for i in range(500) if split_of("musique", f"t{i}") == "train")
    rows = [_row(t, f"r{i}", i, f"question number {i}?", value=i / 30.0) for i in range(30)]
    a, _ = export_sft(rows, margin_threshold=0.0, max_examples_per_task=10)
    b, _ = export_sft(list(reversed(rows)), margin_threshold=0.0, max_examples_per_task=10)
    ids = lambda ex: sorted((e.task_id, e.turn_idx) for e in ex)  # noqa: E731
    assert ids(a) == ids(b), "selection must not depend on input order"
    kept = {e.turn_idx for e in a}
    top10 = set(range(20, 30))
    assert kept != top10, "keeping the highest-value examples would select for the reward"


def test_the_cap_applies_per_task_not_globally():
    from pinq_train.split import split_of

    ts = [f"t{i}" for i in range(500) if split_of("musique", f"t{i}") == "train"][:2]
    ex, man = export_sft(
        _rows(ts[0], 20) + _rows(ts[1], 20), margin_threshold=0.0, max_examples_per_task=5
    )
    per = {}
    for e in ex:
        per[e.task_id] = per.get(e.task_id, 0) + 1
    assert per == {ts[0]: 5, ts[1]: 5} and man.n_over_task_cap_dropped == 30


def test_the_cap_keeps_the_ask_stop_mix():
    """Capping only the ASK rows would drive `stop_share` up toward the collapse ceiling by
    construction, so the cap is applied to each kind separately."""
    from pinq_train.split import split_of

    t = next(f"t{i}" for i in range(500) if split_of("musique", f"t{i}") == "train")
    rows = _rows(t, 20) + _rows(t + "x", 0)  # 20 ASK states
    rows += [_row(t, f"d{i}", 100 + i, f"done question {i}?", done=True) for i in range(20)]
    ex, _ = export_sft(rows, margin_threshold=0.0, max_examples_per_task=6)
    stops = sum(1 for e in ex if e.is_stop)
    assert stops == 6 and len(ex) - stops == 6, "6 of each kind, not 6 in total"


def test_the_cap_rides_into_the_manifest_json(tmp_path):
    """A dataset built under a cap is a different dataset; a reader must be able to tell."""
    from pinq_train.export.dataset import write_jsonl
    from pinq_train.split import split_of

    t = next(f"t{i}" for i in range(500) if split_of("musique", f"t{i}") == "train")
    ex, man = export_sft(_rows(t, 12), margin_threshold=0.0, max_examples_per_task=5)
    p = write_jsonl(tmp_path / "sft.jsonl", ex, man)
    side = json.loads((p.parent / "sft.manifest.json").read_text())
    assert side["max_examples_per_task"] == 5 and side["n_over_task_cap_dropped"] == 7
