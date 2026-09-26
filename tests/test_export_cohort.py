"""`pi train export --cohort {any,fixed,old} --turn0-any-cohort`.

THE CORPUS SPANS TWO LOOPS. In the OLD one the answer to a turn's question never saw the
documents fetched to answer it, so the history block of an evidence-bearing prompt carries a
stale per-turn answer. `conf/cohorts/fixed_loop.json` names the FIXED commits.

Until now that fact could only be used NEGATIVELY: `_cross_cohort` refuses to PAIR across the
two, so a pair whose sides straddle the fix is dropped. Nothing could select a cohort, so the
headline SFT file held 43,837 rows of which 33,653 were rendered by the broken loop.

WHY `--turn0-any-cohort` IS NOT A LOOPHOLE. At turn 0 nothing has been asked and no evidence
has been fetched, so there is no per-turn answer to be stale: the prompt an OLD run rendered at
turn 0 is byte-identical to the one the FIXED loop renders. Refusing those rows would throw
away 4,115 clean ASK targets and 4,721 clean pairs to no purpose. It is a FLAG and not the
default because it is a claim about the loop, and a claim goes on the manifest.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pinq_train.export.dataset import export_pairs, export_sft, write_jsonl
from pinq_train.split import split_of

# Two shas from `conf/cohorts/fixed_loop.json` and one that is deliberately not in it.
FIXED = "3ffa972472cdbc8510c431214beb9ca03a5208af"
FIXED2 = "a22015fb7acfe03214e9f91c26fad95df8a4ea59"
OLD = "0" * 40
MEMBERS = frozenset({FIXED, FIXED2})
COHORT_SHA = "deadbeef" * 8


def _train_task(suite="musique", start=0):
    for i in range(start, start + 500):
        t = f"t{i}"
        if split_of(suite, t) == "train":
            return t
    raise AssertionError("no train bucket found")


T1 = _train_task()
T2 = _train_task(start=800)


def _row(*, task, turn, run, code_version, value=1.0, action='{"action": "ASK", "q": "a"}'):
    """One candidate decision point. Turn > 0 rows carry an evidence block, which is exactly
    the prompt the OLD loop rendered wrongly; the turn-0 row does not, which is why it is
    admissible under `--turn0-any-cohort`."""
    state = "TASK\n" + ("EVIDENCE RETRIEVED SO FAR\n- d1: x\n" if turn else "")
    return {
        "suite_id": "musique",
        "task_id": task,
        "run_id": run,
        "turn_idx": turn,
        "state_text": state,
        "action_json": action,
        "value": value,
        "scorer_hash": "sh",
        "graph_version": "v1",
        "code_version": code_version,
        "pins_sha": "p1",
        "done_before": False,
    }


# ------------------------------------------------------------------ SFT


def test_a_fixed_cohort_export_keeps_the_fixed_row_and_refuses_the_old_one():
    rows = [
        _row(task=T1, turn=1, run="r_fixed", code_version=FIXED),
        _row(task=T1, turn=1, run="r_old", code_version=OLD),
    ]
    items, man = export_sft(
        rows,
        margin_threshold=-1.0,
        cohort=MEMBERS,
        cohort_file_sha=COHORT_SHA,
        cohort_mode="fixed",
    )
    assert [it.run_id for it in items] == ["r_fixed"]
    assert man.n_cohort_refused == 1
    assert man.cohort == "fixed" and man.cohort_file_sha == COHORT_SHA


def test_the_old_cohort_is_the_complement_of_the_fixed_one():
    """`old` is "not a member", not a second list. A row with no `code_version` at all is OLD
    under that reading -- it ran under no recorded code, which is not the fixed loop."""
    # Distinct actions: the exact-dedupe below collapses byte-identical (task, state, action)
    # rows, and three identical rows would leave one survivor whatever the cohort decided.
    rows = [
        _row(task=T1, turn=1, run="r_fixed", code_version=FIXED, action='{"q": "1"}'),
        _row(task=T1, turn=1, run="r_old", code_version=OLD, action='{"q": "2"}'),
        _row(task=T1, turn=1, run="r_blank", code_version="", action='{"q": "3"}'),
    ]
    items, man = export_sft(
        rows, margin_threshold=-1.0, cohort=MEMBERS, cohort_file_sha=COHORT_SHA, cohort_mode="old"
    )
    assert sorted(it.run_id for it in items) == ["r_blank", "r_old"]
    assert man.n_cohort_refused == 1


def test_turn0_any_cohort_keeps_an_old_turn_zero_row():
    """Nothing was asked yet, so there is no stale per-turn answer to carry: the turn-0 prompt
    is clean by construction whichever loop rendered it."""
    rows = [
        _row(task=T1, turn=0, run="r_old_t0", code_version=OLD),
        _row(task=T1, turn=1, run="r_old_t1", code_version=OLD),
        _row(task=T2, turn=1, run="r_fixed", code_version=FIXED),
    ]
    without, man_a = export_sft(
        rows,
        margin_threshold=-1.0,
        cohort=MEMBERS,
        cohort_file_sha=COHORT_SHA,
        cohort_mode="fixed",
    )
    with_t0, man_b = export_sft(
        rows,
        margin_threshold=-1.0,
        cohort=MEMBERS,
        cohort_file_sha=COHORT_SHA,
        cohort_mode="fixed",
        turn0_any_cohort=True,
    )
    assert [it.run_id for it in without] == ["r_fixed"] and man_a.n_cohort_refused == 2
    assert sorted(it.run_id for it in with_t0) == ["r_fixed", "r_old_t0"]
    assert man_b.n_cohort_refused == 1
    assert man_b.turn0_any_cohort is True and man_a.turn0_any_cohort is False


def _bytes(tmp_path, name, items, man):
    """The FILE, not the objects: `phi_tilde` is NaN on an exported row, and NaN never compares
    equal to itself, so two identical exports differ as dataclasses and not as artifacts."""
    path = write_jsonl(tmp_path / name, items, man)
    return path.read_bytes(), json.loads((path.parent / f"{path.stem}.manifest.json").read_text())


def test_any_is_byte_identical_to_the_export_that_had_no_cohort_flag(tmp_path):
    """The flag is new; the file produced without it must not be. `any` keeps every row and
    must not move a sample weight, a dedupe representative or the id-set hash."""
    rows = [
        _row(task=T1, turn=0, run="r_old_t0", code_version=OLD),
        _row(task=T1, turn=1, run="r_old_t1", code_version=OLD, action='{"q": "2"}'),
        _row(task=T2, turn=1, run="r_fixed", code_version=FIXED, action='{"q": "3"}'),
    ]
    before = export_sft(rows, margin_threshold=-1.0)
    after = export_sft(
        rows,
        margin_threshold=-1.0,
        cohort=MEMBERS,
        cohort_file_sha=COHORT_SHA,
        cohort_mode="any",
    )
    b0, m0 = _bytes(tmp_path / "before", "sft.jsonl", *before)
    b1, m1 = _bytes(tmp_path / "after", "sft.jsonl", *after)
    assert b0 == b1 and len(before[0]) == 3
    assert after[1].n_cohort_refused == 0
    assert before[1].train_id_set_hash == after[1].train_id_set_hash
    # The only manifest difference is the DECLARATION -- which cohort file was in hand.
    assert {k for k in m0 if m0[k] != m1[k]} == {"cohort_file_sha"}


# ------------------------------------------------------------------ pairs


def _pair_rows(*, task, turn, code_version):
    return [
        _row(
            task=task,
            turn=turn,
            run=f"c{i}",
            code_version=code_version,
            value=1.0 - i,
            action=json.dumps({"action": "ASK", "question": f"q{i}?"}),
        )
        for i in range(2)
    ]


def _with_parent(rows, parent):
    for r in rows:
        r["branch_of_run_id"] = parent
        r["branch_turn_idx"] = r["turn_idx"]
    return rows


def test_the_pairs_export_applies_the_same_predicate():
    rows = _with_parent(_pair_rows(task=T1, turn=1, code_version=FIXED), "p_fixed")
    rows += _with_parent(_pair_rows(task=T2, turn=1, code_version=OLD), "p_old")
    pairs, man = export_pairs(
        rows,
        margin_threshold=-1.0,
        cohort=MEMBERS,
        cohort_file_sha=COHORT_SHA,
        cohort_mode="fixed",
    )
    assert {p.task_id for p in pairs} == {T1}
    assert all(p.code_version == FIXED for p in pairs)
    assert man.n_cohort_refused == 2
    assert man.cohort == "fixed" and man.turn0_any_cohort is False


def test_turn0_any_cohort_keeps_an_old_turn_zero_pair():
    rows = _with_parent(_pair_rows(task=T1, turn=0, code_version=OLD), "p_old_t0")
    rows += _with_parent(_pair_rows(task=T2, turn=1, code_version=OLD), "p_old_t1")
    without, man_a = export_pairs(
        rows,
        margin_threshold=-1.0,
        cohort=MEMBERS,
        cohort_file_sha=COHORT_SHA,
        cohort_mode="fixed",
    )
    with_t0, man_b = export_pairs(
        rows,
        margin_threshold=-1.0,
        cohort=MEMBERS,
        cohort_file_sha=COHORT_SHA,
        cohort_mode="fixed",
        turn0_any_cohort=True,
    )
    assert without == [] and man_a.n_cohort_refused == 4
    assert {p.task_id for p in with_t0} == {T1}
    assert man_b.n_cohort_refused == 2


def test_any_is_byte_identical_on_the_pairs_export_too(tmp_path):
    rows = _with_parent(_pair_rows(task=T1, turn=1, code_version=FIXED), "p_fixed")
    rows += _with_parent(_pair_rows(task=T2, turn=1, code_version=OLD), "p_old")
    before = export_pairs(rows, margin_threshold=-1.0, cohort=MEMBERS, cohort_file_sha=COHORT_SHA)
    after = export_pairs(
        rows,
        margin_threshold=-1.0,
        cohort=MEMBERS,
        cohort_file_sha=COHORT_SHA,
        cohort_mode="any",
    )
    b0, m0 = _bytes(tmp_path / "before", "pairs.jsonl", *before)
    b1, m1 = _bytes(tmp_path / "after", "pairs.jsonl", *after)
    assert b0 == b1 and len(before[0]) == 2
    assert m0 == m1


# ------------------------------------------------------------------ refusals


def test_an_unknown_cohort_mode_is_refused():
    with pytest.raises(ValueError, match="cohort_mode"):
        export_sft([], margin_threshold=0.0, cohort=MEMBERS, cohort_mode="fixed_loop")


def test_a_cohort_selection_with_no_membership_set_is_refused():
    """An empty set would make `fixed` keep nothing and `old` keep everything, and the manifest
    would say a cohort decided. That is the failure `_cohort()` already refuses one layer up."""
    with pytest.raises(ValueError, match="no members"):
        export_sft([], margin_threshold=0.0, cohort=frozenset(), cohort_mode="fixed")
    with pytest.raises(ValueError, match="no members"):
        export_pairs([], margin_threshold=0.0, cohort=frozenset(), cohort_mode="old")


def test_turn0_any_cohort_under_any_is_refused():
    """Under `any` every row is kept already, so the flag decides nothing while the manifest
    records that it was asked for -- a knob that silently does not apply."""
    with pytest.raises(ValueError, match="turn0_any_cohort"):
        export_sft([], margin_threshold=0.0, cohort_mode="any", turn0_any_cohort=True)


# ------------------------------------------------------------------ the CLI, end to end


def _export(tmp_path, monkeypatch, *flags, expect=0):
    from tests.test_train_cli import _synth, _turns, _write_run

    from pi_run import cmd_train
    from pi_run.cli import build_parser

    monkeypatch.setenv("PI_GOLD_ROOT", str(tmp_path / "data" / "gold"))
    suite, tid, units = _synth(tmp_path)
    runs = tmp_path / "runs"
    _write_run(runs, "keep", task=tid, split="train", turns=_turns(units[:3]))
    args = build_parser().parse_args(
        [
            "train",
            "export",
            "--root",
            str(tmp_path),
            "--runs-root",
            str(runs),
            "--out",
            str(tmp_path / "rl"),
            "--tau",
            "0.0",
            *flags,
        ]
    )
    assert cmd_train.cmd_train_export(args) == expect
    if expect:
        return None
    return json.loads((tmp_path / "rl" / "sft.manifest.json").read_text())


def test_the_cli_refuses_turn0_any_cohort_without_a_cohort_selection(tmp_path, monkeypatch, capsys):
    """Refused BEFORE the export runs, not after: the exporter raises for the same reason, but
    a two-hour job that dies at the write is not a guard anybody benefits from."""
    _export(tmp_path, monkeypatch, "--turn0-any-cohort", expect=2)
    assert "only meaningful with --cohort fixed" in capsys.readouterr().err
    assert not (tmp_path / "rl" / "sft.manifest.json").exists()


def test_the_sft_manifest_names_the_cohort_file_that_was_in_hand(tmp_path, monkeypatch):
    """MEASURED DEFECT: `data/rl/sft.manifest.json` ships `cohort_file_sha: ""` while every
    pairs manifest beside it carries the sha -- the file was loaded inside the pairs branch, so
    an SFT export never saw one. An empty field reads as "no cohort definition existed", which
    is the fallback regime, not this one."""
    man = _export(tmp_path, monkeypatch)
    assert len(man["cohort_file_sha"]) == 64
    assert man["cohort"] == "any" and man["turn0_any_cohort"] is False
    assert man["n_cohort_refused"] == 0


def test_the_cohort_flags_reach_the_manifest(tmp_path, monkeypatch):
    man = _export(tmp_path, monkeypatch, "--cohort", "old", "--turn0-any-cohort")
    assert man["cohort"] == "old" and man["turn0_any_cohort"] is True


def test_pi_train_export_renders_the_new_flags_in_its_help():
    import argparse as _ap

    from pi_run.cli import build_parser

    parser = build_parser()
    train = next(
        a.choices["train"] for a in parser._actions if isinstance(a, _ap._SubParsersAction)
    )
    sub = next(a for a in train._actions if isinstance(a, _ap._SubParsersAction))
    text = sub.choices["export"].format_help()
    assert "--cohort " in text and "--turn0-any-cohort" in text


def test_the_scratch_predicate_and_the_exporter_agree_on_the_rule():
    """The counts in the report were produced by applying a predicate to the SHIPPED export
    files rather than by re-running a two-hour export. This pins that the predicate is the one
    the exporter applies, so those counts are about this code."""
    from pinq_train.export.dataset import cohort_keeps

    for turn in (0, 1):
        for cv in (FIXED, OLD, ""):
            row = {"turn_idx": turn, "code_version": cv}
            for t0 in (False, True):
                expect_fixed = (t0 and turn == 0) or cv in MEMBERS
                expect_old = (t0 and turn == 0) or cv not in MEMBERS
                assert cohort_keeps(row, mode="fixed", cohort=MEMBERS, turn0_any=t0) is expect_fixed
                assert cohort_keeps(row, mode="old", cohort=MEMBERS, turn0_any=t0) is expect_old
                assert cohort_keeps(row, mode="any", cohort=MEMBERS, turn0_any=False) is True


def test_the_cohort_file_of_record_still_names_its_members():
    """The predicate is only as good as the file. Four shas, and the sha256 the report quotes."""
    import hashlib

    blob = Path("conf/cohorts/fixed_loop.json").read_bytes()
    assert len(json.loads(blob)["members"]) == 4
    assert hashlib.sha256(blob).hexdigest().startswith("48891dc69d55")
