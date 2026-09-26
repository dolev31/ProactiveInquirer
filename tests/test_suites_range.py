"""`pi suites range` -- does a metric have room to move, and is it measured at all?

WHY THIS COMMAND EXISTS. A preregistered endpoint on a metric whose ceiling and floor are
0.02 apart cannot produce a finding whatever the policy does: the test is powered to detect
an effect the measurement cannot express. And a metric that is NaN on a third of the tasks is
not a smaller sample, it is a different population -- the tasks where it IS defined.

Both failures are silent in every table the repo renders: the row appears, the CI is tight,
and nothing says the span the arm is moving inside is 0.02 wide. So the check is a command,
read-only over `scores.parquet`, run against ONE `scorer_hash` -- because a metric's range is
a property of the scorer that produced it, and pooling two hashes averages two definitions.

THE RULE, from the plan: ceiling - floor >= 0.2 and non-NaN share >= 0.9, per (suite, metric).
The ceiling is `oracle_vreq` or `gold_evidence` (an arm handed the answer's evidence), the
floor is `drafter_only` (an arm that never asks), and the arm under test is
`inquirer_prompted`.

Every fixture here is built with pyarrow in `tmp_path`: the shared `scores/parquet` is
rewritten by `pi compact`, so a test that reads it asserts whatever last night's campaign
happened to hold.
"""

from __future__ import annotations

import argparse
import math

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

SCORER = "sc0"
OTHER_SCORER = "sc1"


def _run_row(rid: str, *, suite: str, task: str, arm: str, split: str = "test") -> dict:
    """One ELIGIBLE run. Every column `_ELIGIBLE_CORE` filters on is spelled out, because a
    default would let the predicate drift away from what the report actually applies. `split`
    defaults to 'test' because `_ELIGIBLE_CORE` ANDs in `_HELD_OUT` (`r.split = 'test'`): a
    fixture that never set the column could not tell a correct held-out filter from a broken
    one, since both look like "no rows dropped" over an all-test population."""
    return dict(
        run_id=rid,
        suite_id=suite,
        task_id=task,
        arm_id=arm,
        status="ok",
        gold_exposed=False,
        is_dev_run=False,
        dirty=False,
        pilot_flag=False,
        canary_hit=False,
        exploratory=False,
        firewall_ok=True,
        counterfactual_kind="none",
        split=split,
    )


def _build(tmp_path, cells, *, scorer=SCORER):
    """`cells` is (suite, arm, task, metric, value) or (suite, arm, task, metric, value,
    split); a value of None writes NO score row, which is how the scorer says a measurement
    was not taken. `split` defaults to 'test' when omitted -- every pre-existing cell in this
    file is a held-out row, and a bare 5-tuple should keep meaning exactly that."""
    runs: dict[str, dict] = {}
    scores: list[dict] = []
    for cell in cells:
        suite, arm, task, metric, value = cell[:5]
        split = cell[5] if len(cell) > 5 else "test"
        rid = f"{suite}-{arm}-{task}"
        runs.setdefault(rid, _run_row(rid, suite=suite, task=task, arm=arm, split=split))
        if value is None:
            continue
        scores.append(dict(run_id=rid, metric_name=metric, scorer_hash=scorer, value=float(value)))
    d = tmp_path / "parquet"
    d.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(list(runs.values())), d / "runs.parquet")
    pq.write_table(pa.Table.from_pylist(scores), d / "scores.parquet")
    return d


def _range(tmp_path, cells, **kw):
    from pi_run.cmd_suites import cmd_suites_range

    d = _build(tmp_path, cells, scorer=kw.pop("scorer", SCORER))
    a = argparse.Namespace(
        parquet=str(d),
        scorer_hash=kw.pop("scorer_hash", SCORER),
        metric=kw.pop("metric", None),
        root=None,
        json=True,
    )
    for k, v in kw.items():
        setattr(a, k, v)
    return a, cmd_suites_range(a)


def _rows(capsys, code):
    import json

    out = json.loads(capsys.readouterr().out)
    return {(r["suite"], r["metric"]): r for r in out["rows"]}, code


def _wide(suite="musique", metric="evidence_coverage", n=10):
    """A metric with room to move: floor 0.10, arm 0.45, ceiling 0.90 -> span 0.80.

    Plus one extra `drafter_only` row stamped split='train' at 0.95, on a task id ("ttrain")
    none of this file's arm/task filters touch. It is the population `_ELIGIBLE_CORE_ANY_SPLIT`
    admits and `_HELD_OUT` removes: averaged into the ten split='test' rows at 0.10 it would
    drag the floor to (10*0.10 + 0.95) / 11 ~= 0.177, which is why every test below that checks
    an exact floor or span is also a check that this row was excluded.
    """
    cells = []
    for i in range(n):
        cells.append((suite, "drafter_only", f"t{i}", metric, 0.10))
        cells.append((suite, "inquirer_prompted", f"t{i}", metric, 0.45))
        cells.append((suite, "oracle_vreq", f"t{i}", metric, 0.90))
    cells.append((suite, "drafter_only", "ttrain", metric, 0.95, "train"))
    return cells


# --------------------------------------------------------------------------------- the rule


def test_a_metric_with_room_to_move_passes_and_prints_all_four_numbers(tmp_path, capsys):
    a, code = _range(tmp_path, _wide())
    rows, code = _rows(capsys, code)
    r = rows[("musique", "evidence_coverage")]
    assert r["prompted"] == pytest.approx(0.45)
    assert r["ceiling"] == pytest.approx(0.90)
    assert r["ceiling_arm"] == "oracle_vreq"
    # _wide() also plants a split='train' drafter_only row at 0.95. If it leaked into the
    # floor mean this would read ~0.177, not 0.10 -- see _wide()'s docstring for the arithmetic.
    assert r["floor"] == pytest.approx(0.10), "a train-split row must not enter the floor mean"
    assert r["span"] == pytest.approx(0.80)
    assert r["nonnan_share"] == pytest.approx(1.0)
    assert r["verdict"] == "ok"
    assert code == 0


def test_a_metric_whose_ceiling_and_floor_nearly_touch_fails(tmp_path, capsys):
    """THE FAILURE THIS COMMAND EXISTS FOR. An endpoint here is powered to detect an effect
    the measurement cannot express, and every rendered table looks exactly the same."""
    cells = []
    for i in range(10):
        cells.append(("musique", "drafter_only", f"t{i}", "task_success", 0.80))
        cells.append(("musique", "inquirer_prompted", f"t{i}", "task_success", 0.85))
        cells.append(("musique", "oracle_vreq", f"t{i}", "task_success", 0.90))
    # One extra split='train' drafter_only row at 0.0. Counted, it drags the floor to
    # (10*0.80 + 0.0) / 11 ~= 0.727 and the span to ~0.173 -- the span assertion below is
    # also the check that it was not.
    cells.append(("musique", "drafter_only", "ttrain", "task_success", 0.0, "train"))
    a, code = _range(tmp_path, cells)
    rows, code = _rows(capsys, code)
    r = rows[("musique", "task_success")]
    assert r["span"] == pytest.approx(0.10), "a train-split row must not enter the floor mean"
    assert r["verdict"] == "no range"
    assert code == 1


def test_a_metric_missing_on_a_fifth_of_the_tasks_fails(tmp_path, capsys):
    """Absent is not zero, and it is not a smaller sample either: it is a different
    population. Two of ten prompted runs write no row at all."""
    cells = _wide()
    cells = [c for c in cells if not (c[1] == "inquirer_prompted" and c[2] in ("t0", "t1"))]
    cells += [("musique", "inquirer_prompted", t, "evidence_coverage", None) for t in ("t0", "t1")]
    a, code = _range(tmp_path, cells)
    rows, code = _rows(capsys, code)
    r = rows[("musique", "evidence_coverage")]
    assert r["nonnan_share"] == pytest.approx(0.8)
    assert r["verdict"] == "not measured"
    assert code == 1


def test_a_nan_value_counts_as_absent_not_as_zero(tmp_path, capsys):
    """A NaN written into the parquet and a row that was never written mean the same thing,
    and neither may average into the mean as a 0."""
    cells = _wide()
    cells = [c for c in cells if not (c[1] == "inquirer_prompted" and c[2] in ("t0", "t1"))]
    cells += [
        ("musique", "inquirer_prompted", t, "evidence_coverage", float("nan")) for t in ("t0", "t1")
    ]
    a, code = _range(tmp_path, cells)
    rows, code = _rows(capsys, code)
    r = rows[("musique", "evidence_coverage")]
    assert r["nonnan_share"] == pytest.approx(0.8)
    assert r["prompted"] == pytest.approx(0.45), "the NaN rows must not drag the mean toward 0"
    assert code == 1


# ------------------------------------------------------------------------ what it reads from


def test_it_reads_one_scorer_hash_and_not_the_others(tmp_path, capsys):
    """A metric's range is a property of the scorer that produced it. Pooling two hashes
    averages two definitions of the same name -- which is exactly what re-scoring creates."""
    a, code = _range(tmp_path, _wide(), scorer=OTHER_SCORER, scorer_hash=SCORER)
    rows, code = _rows(capsys, code)
    assert rows == {}, "rows under a different scorer_hash must not be read"


def test_gold_evidence_stands_in_for_the_ceiling_when_oracle_vreq_is_absent(tmp_path, capsys):
    cells = []
    for i in range(10):
        cells.append(("tau2", "drafter_only", f"t{i}", "evidence_coverage", 0.10))
        cells.append(("tau2", "inquirer_prompted", f"t{i}", "evidence_coverage", 0.40))
        cells.append(("tau2", "gold_evidence", f"t{i}", "evidence_coverage", 0.95))
    # One extra split='train' drafter_only row at 0.99. Counted, it drags the floor to
    # (10*0.10 + 0.99) / 11 ~= 0.181 and the span to ~0.769 -- the span assertion below is
    # also the check that it was not.
    cells.append(("tau2", "drafter_only", "ttrain", "evidence_coverage", 0.99, "train"))
    a, code = _range(tmp_path, cells)
    rows, code = _rows(capsys, code)
    r = rows[("tau2", "evidence_coverage")]
    assert r["ceiling_arm"] == "gold_evidence"
    assert r["span"] == pytest.approx(0.85), "a train-split row must not enter the floor mean"
    assert code == 0


def test_a_missing_ceiling_arm_is_reported_as_unmeasured_and_never_as_a_span(tmp_path, capsys):
    """No oracle arm ran, so the span is UNKNOWN. Reporting 0.0 there would invent a
    verdict; reporting the prompted-minus-floor difference would invent a ceiling."""
    cells = [c for c in _wide() if c[1] != "oracle_vreq"]
    a, code = _range(tmp_path, cells)
    rows, code = _rows(capsys, code)
    r = rows[("musique", "evidence_coverage")]
    assert r["ceiling"] is None or math.isnan(r["ceiling"])
    assert r["span"] is None or math.isnan(r["span"])
    assert r["verdict"] == "no ceiling arm"


def test_an_indexed_family_is_not_a_task_level_metric(tmp_path, capsys):
    """`frontier_q#3` is one point of a ladder, not a per-task score, and a mean over the
    points of a curve is not a number anyone may read."""
    cells = _wide()
    cells += [("musique", "inquirer_prompted", f"t{i}", "frontier_q#2", 0.5) for i in range(10)]
    a, code = _range(tmp_path, cells)
    rows, code = _rows(capsys, code)
    assert not [k for k in rows if k[1].startswith("frontier_q")]


def test_machine_artifacts_are_not_range_checked(tmp_path, capsys):
    """wall_ms and usd are recorded and never compared across arms, so a range rule over
    them would fail on day one and teach everyone to pass --metric."""
    cells = _wide()
    cells += [("musique", "inquirer_prompted", f"t{i}", "usd", 0.01) for i in range(10)]
    cells += [("musique", "drafter_only", f"t{i}", "usd", 0.009) for i in range(10)]
    cells += [("musique", "oracle_vreq", f"t{i}", "usd", 0.011) for i in range(10)]
    a, code = _range(tmp_path, cells)
    rows, code = _rows(capsys, code)
    assert ("musique", "usd") not in rows
    assert code == 0


def test_an_absent_parquet_is_an_error_and_not_an_empty_pass(tmp_path, capsys):
    from pi_run.cmd_suites import cmd_suites_range

    a = argparse.Namespace(
        parquet=str(tmp_path / "nothing"), scorer_hash=SCORER, metric=None, root=None, json=False
    )
    assert cmd_suites_range(a) == 2


def test_breadth_is_range_checked_like_any_other_structure_metric(tmp_path, capsys):
    """The metric this command was written alongside. It is a COUNT, so its span is in
    components rather than in probability -- which is why the units column exists."""
    cells = []
    for i in range(10):
        cells.append(("tau2_retail", "drafter_only", f"t{i}", "breadth_recall", 0.05))
        cells.append(("tau2_retail", "inquirer_prompted", f"t{i}", "breadth_recall", 0.30))
        cells.append(("tau2_retail", "oracle_vreq", f"t{i}", "breadth_recall", 0.70))
    # One extra split='train' drafter_only row at 0.99. Counted, it drags the floor to
    # (10*0.05 + 0.99) / 11 ~= 0.135 and the span to ~0.565 -- the span assertion below is
    # also the check that it was not.
    cells.append(("tau2_retail", "drafter_only", "ttrain", "breadth_recall", 0.99, "train"))
    a, code = _range(tmp_path, cells)
    rows, code = _rows(capsys, code)
    assert rows[("tau2_retail", "breadth_recall")]["span"] == pytest.approx(0.65), (
        "a train-split row must not enter the floor mean"
    )
    assert code == 0
