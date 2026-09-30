"""The ICLR figure/table generator, exercised on the REAL records.

The generator's whole job is to refuse. Its numbers are parsed out of promoted `.md` records
and printed appendix tables and then re-derived from the scored parquet stores, and a
disagreement must stop the build rather than reach a PDF. So every guard here is tested TWICE:
once that it passes on the record as it stands, and once that it FAILS on a deliberately
altered copy of that same record. A guard that cannot fail is indistinguishable from no guard,
and a passing reconciliation that could not have come out otherwise is not evidence.

Concretely, each of the four checks gets a paired positive and negative case:

  * `parse_level_table` finds the frontier table by header names, and refuses a document in
    which that table is absent or duplicated;
  * `reconcile` names the estimand that reproduced, and raises with BOTH numbers in the
    message when the record is moved off the store by more than the tolerance;
  * `fig2_frontier` refuses end to end when one coverage cell of the promoted record is edited,
    and writes no file at all in that case;
  * `collect_dev_rows` refuses when one printed cell of `paper/appendix_training.tex` is
    edited, naming the arm, and `table1_heldout` refuses when the published coverage lock moves.

It runs against the real artifacts because the thing most likely to break is a table losing a
column or a store losing a run-id list, which a fixture would hide. It therefore SKIPS, naming
the absent path, on a checkout that does not carry them.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
GEN = REPO / "scripts" / "paper_figures_iclr.py"


def _load():
    """`scripts/` is not an importable package, so load the file by path. The module must be in
    `sys.modules` BEFORE it executes: `@dataclass` under `from __future__ import annotations`
    resolves its string annotations through `sys.modules[cls.__module__]`."""
    spec = importlib.util.spec_from_file_location("paper_figures_iclr", GEN)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def gen():
    pytest.importorskip("duckdb")
    pytest.importorskip("matplotlib")
    if not GEN.exists():
        pytest.skip(f"generator absent: {GEN}")
    return _load()


def _need(*paths: Path) -> None:
    for p in paths:
        if not p.exists():
            pytest.skip(f"record absent on this checkout: {p}")


# --------------------------------------------------------------------------- markdown parsing


def test_parse_level_table_reads_the_promoted_record(gen):
    """The musique frontier table, by header names, with one cell checked to the digit."""
    _need(gen.FRONTIER_MUSIQUE)
    cells = gen.parse_level_table(gen.FRONTIER_MUSIQUE.read_text(), "evidence_coverage")
    assert len(cells) == 15, "three arms x five caps"
    assert {a for a, _ in cells} == {"teacher", "base8b", "trained"}
    c = cells[("trained", 4)]
    assert (c.n, c.mean_asks, c.mean_calls) == (400, 2.4950, 2.4950)
    assert (c.value, c.lo, c.hi) == (0.8379, 0.8032, 0.8668)


def test_parse_level_table_reads_the_other_two_records(gen):
    """The two later campaigns carry an extra `clusters` column and a different metric, so a
    parser keyed on column POSITION would read the wrong cells here and not fail."""
    _need(gen.FRONTIER_STRATEGYQA, gen.FRONTIER_FRAMES)
    s = gen.parse_level_table(gen.FRONTIER_STRATEGYQA.read_text(), "evidence_coverage")
    assert len(s) == 15
    assert s[("trained", 4)].value == 0.7853
    assert s[("trained", 4)].mean_calls == 1.5200
    f = gen.parse_level_table(gen.FRONTIER_FRAMES.read_text(), "answer_correct")
    assert len(f) == 20, "four pins x five caps"
    assert f[("drafter_only", 8)].value == 0.0959
    assert f[("drafter_only", 8)].mean_calls == 0.0


def test_parse_level_table_refuses_a_document_without_one(gen):
    with pytest.raises(gen.Mismatch, match="no level table"):
        gen.parse_level_table("| a | b |\n|---|---|\n| 1 | 2 |\n", "evidence_coverage")


def test_parse_level_table_refuses_two_level_tables(gen):
    """Two tables with the same header is an ambiguous record, not a choice for the script."""
    _need(gen.FRONTIER_MUSIQUE)
    text = gen.FRONTIER_MUSIQUE.read_text()
    with pytest.raises(gen.Mismatch, match="ambiguous"):
        gen.parse_level_table(text + "\n\n" + text, "evidence_coverage")


def test_parse_interval_refuses_a_cell_that_is_not_one(gen):
    assert gen.parse_interval("0.7807 [0.7452, 0.8164]") == (0.7807, 0.7452, 0.8164)
    with pytest.raises(gen.Mismatch):
        gen.parse_interval("231/400 (57.75%)")


def test_read_run_ids_handles_both_recorded_layouts(gen):
    """`FRONTIER.md`'s lists are `run_id, task_id, seed`; the two later campaigns' are
    `task_id, seed, run_id`. Reading a fixed column would silently return task ids."""
    a = REPO / "artifacts/frontier/run_ids.trained.cap4.txt"
    b = REPO / "artifacts/frontier_strategyqa/run_ids.trained.cap4.tsv"
    _need(a, b)
    for p, want in ((a, 400), (b, 400)):
        ids = gen.read_run_ids(p)
        assert len(ids) == want == len(set(ids))
        assert all(len(i) == 32 and all(ch in "0123456789abcdef" for ch in i) for i in ids)


# --------------------------------------------------------------------------- reconciliation


def test_reconcile_names_which_estimand_reproduced(gen):
    assert gen.reconcile("x", 0.8379, 0.836875, 0.837926) == "unweighted mean over clusters"
    assert gen.reconcile("x", 2.4950, 2.495000, 2.470430) == "pooled run mean"
    assert "coincide" in gen.reconcile("x", 0.7853, 0.785262, 0.785262)


def test_reconcile_raises_on_an_altered_value_and_names_both_numbers(gen):
    """THE NON-VACUITY CASE. The recorded value is moved by ten times the tolerance and the
    check must fail, with the record's number and both recomputations in the message."""
    with pytest.raises(gen.Mismatch) as exc:
        gen.reconcile("trained cap 4", 0.8379 + 10 * gen.TOL, 0.836875, 0.837926)
    msg = str(exc.value)
    assert "trained cap 4" in msg
    assert "0.836875" in msg and "0.837926" in msg
    assert "NOT DRAWING" in msg


def test_reconcile_tolerance_is_tight_enough_to_see_a_fourth_decimal(gen):
    """The records print four decimals, so the guard has to resolve a change in the last one.
    A tolerance that admitted 1e-3 would accept an edited record."""
    with pytest.raises(gen.Mismatch):
        gen.reconcile("x", 0.8379, 0.8390, 0.8390)


# --------------------------------------------------------------------------- fig 2 end to end


def _musique_only(gen, md: Path):
    """One panel, reading its levels from `md` and its rows from the real musique store.

    The spec is built EAGERLY, before `_panels` is monkeypatched, because a lambda that called
    `gen._panels()` after the patch would call itself.
    """
    real = gen._panels()[0]
    one = (
        gen.PanelSpec(
            key=real.key,
            title=real.title,
            md=md,
            metric=real.metric,
            ylabel=real.ylabel,
            arms=real.arms,
            units_note=real.units_note,
            store=real.store,
            ids=real.ids,
        ),
    )
    return lambda: one


def test_fig2_draws_and_records_the_estimand_per_cell(gen, tmp_path, monkeypatch):
    _need(gen.FRONTIER_MUSIQUE, REPO / "artifacts/frontier/scores_parquet.trained/runs.parquet")
    monkeypatch.setattr(gen, "_panels", _musique_only(gen, gen.FRONTIER_MUSIQUE))
    gen.fig2_frontier(tmp_path)
    prov = json.loads((tmp_path / "fig2_frontier.provenance.json").read_text())
    assert (tmp_path / "fig2_frontier.pdf").stat().st_size > 0
    assert (tmp_path / "fig2_frontier.png").stat().st_size > 0
    assert prov["inputs"], "a figure whose provenance names no input is a number without one"
    assert all(len(h) == 64 for h in prov["inputs"].values())
    assert prov["scorer_hash"] and prov["graph_version"] == ["v1"]
    cell = prov["values"]["musique"]["cells"]["trained.cap4"]
    # The record's coverage column is a cluster mean and its retrieval-calls column, on the same
    # row, is a pooled mean. Both are recorded, and this is the assertion that says so.
    assert cell["estimand_reproduced_y"] == "unweighted mean over clusters"
    assert cell["estimand_reproduced_x"] == "pooled run mean"
    assert cell["n"] == 400 and cell["n_clusters"] == 186


def test_fig2_refuses_an_edited_record_and_writes_nothing(gen, tmp_path, monkeypatch):
    """THE NON-VACUITY CASE for the whole pipeline: one coverage cell of the promoted record is
    edited by a thousandth and the build must stop with no file on disk."""
    _need(gen.FRONTIER_MUSIQUE, REPO / "artifacts/frontier/scores_parquet.trained/runs.parquet")
    text = gen.FRONTIER_MUSIQUE.read_text()
    edited = text.replace(
        "| trained | 4 | 400 | 2.4950 | 2.4950 | 0.8379 [0.8032, 0.8668] |",
        "| trained | 4 | 400 | 2.4950 | 2.4950 | 0.8479 [0.8032, 0.8668] |",
    )
    assert edited != text, "the row this test edits is no longer in the record"
    fake = tmp_path / "FRONTIER.md"
    fake.write_text(edited)
    monkeypatch.setattr(gen, "_panels", _musique_only(gen, fake))
    out = tmp_path / "figures"
    with pytest.raises(gen.Mismatch) as exc:
        gen.fig2_frontier(out)
    assert "0.8479" in str(exc.value) and "NOT DRAWING" in str(exc.value)
    assert not out.exists() or list(out.iterdir()) == []


def test_fig2_refuses_a_cell_whose_unit_count_moved(gen, tmp_path, monkeypatch):
    _need(gen.FRONTIER_MUSIQUE, REPO / "artifacts/frontier/scores_parquet.trained/runs.parquet")
    text = gen.FRONTIER_MUSIQUE.read_text()
    edited = text.replace(
        "| trained | 4 | 400 | 2.4950 |",
        "| trained | 4 | 399 | 2.4950 |",
    )
    assert edited != text
    fake = tmp_path / "FRONTIER.md"
    fake.write_text(edited)
    monkeypatch.setattr(gen, "_panels", _musique_only(gen, fake))
    with pytest.raises(gen.Mismatch, match="399"):
        gen.fig2_frontier(tmp_path / "figures")


# --------------------------------------------------------------------------- table 1


def test_table1_reproduces_the_published_coverage_lock(gen, tmp_path):
    _need(gen.TESTSPLIT_CONTRASTS, gen.TESTSPLIT_VERDICTS / "stacked-notdone.musique.json")
    gen.table1_heldout(tmp_path)
    body = (tmp_path / "table1_heldout.tex").read_text()
    prov = json.loads((tmp_path / "table1_heldout.provenance.json").read_text())
    assert body.lstrip().startswith("%") and r"\begin{tabular}" in body
    assert r"\begin{table}" not in body, "a bare tabular is pasted into the paper, not a float"
    symmetric = json.loads(gen.SYMMETRIC_RECORD.read_text())
    for suite, want in gen.TABLE1_COVERAGE_LOCK.items():
        cell = prov["values"][suite]["evidence_coverage"]
        # The printed delta is the symmetric record's exact float, and the helper reading it
        # supersedes is the published lock, so the chain verdict -> lock -> record is asserted,
        # not the agreement of two files with each other.
        assert (
            cell["matched"]["delta"]
            == symmetric[f"evidence_coverage::{suite}::matched"]["symmetric_delta"]
        )
        assert cell["superseded_helper_reading"]["delta"] == want
        assert cell["matched"]["delta"] != want
    assert r"$+0.117$ {\scriptsize $[+0.078, +0.155]$}" in body
    assert r"$+0.124$" not in body
    assert prov["scorer_hash"] == [
        "aa18b1fefd3b9b50d3d4922b86b00b4328d9a3a28b594a5ab1d0cc5b930e0ba7"
    ]


def test_table1_refuses_a_symmetric_record_for_a_different_cell(gen, tmp_path, monkeypatch):
    """THE NON-VACUITY CASE for the record chain: a record whose superseded value is not the
    verdict's lock is a different cell, and the table must not be written from it."""
    _need(gen.TESTSPLIT_VERDICTS / "stacked-notdone.musique.json", gen.SYMMETRIC_RECORD)
    doc = json.loads(gen.SYMMETRIC_RECORD.read_text())
    doc["evidence_coverage::musique::matched"]["published_value"] = 0.1238
    fake = tmp_path / "contrasts.symmetric.json"
    fake.write_text(json.dumps(doc))
    monkeypatch.setattr(gen, "SYMMETRIC_RECORD", fake)
    with pytest.raises(gen.Mismatch) as exc:
        gen.table1_heldout(tmp_path / "figures")
    assert "Not the same cell" in str(exc.value) and "NOT WRITING" in str(exc.value)


def test_table1_other_rows_follow_the_symmetric_record_when_it_carries_a_value(
    gen, tmp_path, monkeypatch
):
    """Null-with-reason cells fall back to the flagged helper reading; a cell that carries a
    symmetric value is used, and refused if it supersedes a different helper value.

    RE-ANCHORED 2026-09-19. The belief that changed: this used `dwr` on musique as its example of
    the FALLBACK path, because that cell's `symmetric_delta` was null with a reason. Commit
    7e1d805 recomputed dwr symmetrically and gave it a value, so the same cell became an example
    of the path immediately below and this assertion went false. The generator was right and the
    test's example went stale.

    The fallback branch now points at `max_depth_reached`, whose null is STRUCTURAL rather than
    pending: it has no committed script that reproduces it, so it has no symmetric form to gain.
    That is what makes this branch capable of failing for the right reason instead of expiring the
    next time a cell is recomputed. Two of the three branches now assert against the real record
    rather than a synthetic one."""
    _need(gen.TESTSPLIT_VERDICTS / "stacked-notdone.musique.json", gen.SYMMETRIC_RECORD)
    gen.table1_heldout(tmp_path / "a")
    prov = json.loads((tmp_path / "a" / "table1_heldout.provenance.json").read_text())

    # 1. A structurally null cell falls back to the flagged helper reading.
    depth = prov["values"]["musique"]["max_depth_reached"]
    assert depth["matched_rule"].startswith("helper")

    # 2. A cell carrying a symmetric value is used, asserted against the real record.
    dwr = prov["values"]["musique"]["dwr"]
    assert dwr["matched_rule"].startswith("symmetric")
    doc = json.loads(gen.SYMMETRIC_RECORD.read_text())
    cell = doc["dwr::musique::matched"]
    assert dwr["matched"]["delta"] == cell["symmetric_delta"]
    assert dwr["superseded_helper_reading"]["delta"] == cell["published_value"]
    helper_delta = cell["published_value"]

    # 3. A symmetric cell superseding some OTHER value is a different cell: refused. This branch
    # stays synthetic, because the real record holds no such cell and the point is that one must
    # not be accepted if it ever appears.
    fake = tmp_path / "sym.json"
    monkeypatch.setattr(gen, "SYMMETRIC_RECORD", fake)
    cell["published_value"] = helper_delta + 0.01
    fake.write_text(json.dumps(doc))
    with pytest.raises(gen.Mismatch, match="Not the same cell"):
        gen.table1_heldout(tmp_path / "c")


def test_table1_refuses_when_the_coverage_lock_moves(gen, tmp_path, monkeypatch):
    """THE NON-VACUITY CASE: move the published lock and the table must not be written."""
    _need(gen.TESTSPLIT_VERDICTS / "stacked-notdone.musique.json")
    monkeypatch.setitem(gen.TABLE1_COVERAGE_LOCK, "musique", 0.1238)
    out = tmp_path / "figures"
    with pytest.raises(gen.Mismatch) as exc:
        gen.table1_heldout(out)
    assert "0.1238" in str(exc.value) and "NOT WRITING" in str(exc.value)
    assert not out.exists() or list(out.iterdir()) == []


def test_every_tabular_declares_the_size_it_measured_at(gen, tmp_path):
    """The block's width is a property of the block. A document that pastes it cannot be
    expected to rediscover that it overflows \\linewidth at \\normalsize."""
    _need(gen.TESTSPLIT_CONTRASTS, gen.BCA_CELLS, gen.APPENDIX_TRAINING)
    gen.table1_heldout(tmp_path)
    gen.table2_mechanism(tmp_path)
    for name in ("table1_heldout", "table2_mechanism"):
        body = (tmp_path / f"{name}.tex").read_text()
        note = next(ln for ln in body.splitlines() if ln.startswith("% note: measured against"))
        assert "397.5pt" in note and r"\small" in note
        # The separators live in the column spec, not in \tabcolsep, because the asset is a
        # bare tabular and cannot give the document a \setlength.
        spec = next(ln for ln in body.splitlines() if ln.startswith(r"\begin{tabular}"))
        assert r"@{\hspace{5pt}}" in spec and spec.endswith(r"@{}}")


def test_table1_marks_the_out_of_order_row_as_lower_is_better(gen, tmp_path):
    """A rate whose good direction is down, printed beside four whose good direction is up, is
    read backwards unless the row says so."""
    _need(gen.TESTSPLIT_CONTRASTS)
    gen.table1_heldout(tmp_path)
    body = (tmp_path / "table1_heldout.tex").read_text()
    assert r"out-of-order rate$^{\dagger}$" in body
    assert r"$\dagger$ lower is better" in body


# --------------------------------------------------------------------------- table 2 / fig 3


def test_dev_rows_reproduce_the_printed_appendix(gen):
    _need(gen.BCA_CELLS, gen.APPENDIX_TRAINING)
    got = gen.collect_dev_rows()
    stacked = got["rows"]["stacked"]["suites"]
    assert stacked["musique"]["coverage_delta_matched"] == 0.16035353535353536
    assert stacked["musique"]["p_stop_given_done"] == 0.9375
    assert [round(x, 4) for x in stacked["musique"]["coverage_ci_post_fix"]] == [0.1225, 0.2088]
    # The interval ON DISK is the pre-fix one and is deliberately NOT what gets drawn.
    assert [round(x, 4) for x in stacked["musique"]["coverage_ci_pre_fix_on_disk"]] == [
        0.1181,
        0.2039,
    ]
    assert got["rows"]["prompted"]["suites"]["musique"]["p_ask_given_not_done"] == 0.964171974522293


def test_dev_rows_refuse_an_edited_appendix_and_name_the_arm(gen, tmp_path, monkeypatch):
    """THE NON-VACUITY CASE: change one printed stopping cell and the arm must be refused."""
    _need(gen.BCA_CELLS, gen.APPENDIX_TRAINING)
    text, _ = gen.read_tex_with_inputs(gen.APPENDIX_TRAINING)
    edited = text.replace("$0.9375$ & $0.8507$", "$0.9975$ & $0.8507$")
    assert edited != text, "the cell this test edits is no longer in the appendix"
    fake = tmp_path / "appendix_training.tex"
    fake.write_text(edited)
    monkeypatch.setattr(gen, "APPENDIX_TRAINING", fake)
    with pytest.raises(gen.Mismatch) as exc:
        gen.collect_dev_rows()
    msg = str(exc.value)
    assert "both kinds, from the question-pair arm" in msg and "MuSiQue" in msg
    assert "stop-when-done" in msg and "NOT DRAWN" in msg


def test_dev_rows_refuse_an_edited_coverage_cell(gen, tmp_path, monkeypatch):
    _need(gen.BCA_CELLS, gen.APPENDIX_TRAINING)
    text, _ = gen.read_tex_with_inputs(gen.APPENDIX_TRAINING)
    edited = text.replace("$+0.1604$ & $[+0.1225, +0.2088]$", "$+0.1604$ & $[+0.1325, +0.2088]$")
    assert edited != text
    fake = tmp_path / "appendix_training.tex"
    fake.write_text(edited)
    monkeypatch.setattr(gen, "APPENDIX_TRAINING", fake)
    with pytest.raises(gen.Mismatch, match="ci_lo vs appendix"):
        gen.collect_dev_rows()


def test_table2_carries_the_count_unstable_note(gen, tmp_path):
    """`paper/appendix_training.tex:132-137` records that the imitation reference's MuSiQue
    lower bound is a property of the resample count. The table keeps the verdict's value, so it
    has to keep the note with it."""
    _need(gen.BCA_CELLS, gen.APPENDIX_TRAINING)
    gen.table2_mechanism(tmp_path)
    body = (tmp_path / "table2_mechanism.tex").read_text()
    assert "% note:" in body and "COUNT-UNSTABLE" in body
    assert "$+0.049$ & $+0.080$" in body
    assert r"{\scriptsize $[+0.002, +0.097]$}" in body
    assert r"\begin{table}" not in body


def test_fig3_draws_one_point_per_arm_and_suite(gen, tmp_path):
    """RE-ANCHORED 2026-09-23. The belief that changed: the selected recipe's marker was the
    set-aside run's development verdict (stop-when-done 0.9375 on MuSiQue). The marker is now the
    recipe, training seeds 1 and 2 pooled (seed-0 audit C6.2), and 0.9375 is the lock it must
    reproduce first, recorded and not drawn."""
    _need(gen.BCA_CELLS, gen.APPENDIX_TRAINING, gen.FIG3_RECIPE_RECORD)
    gen.fig3_stopping(tmp_path)
    prov = json.loads((tmp_path / "fig3_stopping.provenance.json").read_text())
    assert (tmp_path / "fig3_stopping.pdf").stat().st_size > 0
    assert len(prov["values"]) == 7, "six trained arms and the prompted comparator"
    for row in prov["values"].values():
        assert set(row) == {"reader_name", "musique", "strategyqa"}
    drawn = prov["values"]["stacked"]
    got = {
        s: (
            f"{drawn[s]['x_ask_when_not_done']:.4f}",
            f"{drawn[s]['y_stop_when_done']:.4f}",
            f"{drawn[s]['coverage_delta_matched']:+.4f}",
        )
        for s in ("musique", "strategyqa")
    }
    assert got == {
        "musique": ("0.9436", "0.4272", "+0.1556"),
        "strategyqa": ("0.9091", "0.5454", "+0.0799"),
    }
    lock = prov["extra"]["seed0_lock"]
    assert lock["musique"]["y_stop_when_done"] == 0.9375
    assert f"{lock['strategyqa']['x_ask_when_not_done']:.4f}" == "0.8533"


def _tampered_dev_record(gen, tmp_path, edit):
    rec = json.loads(gen.FIG3_RECIPE_RECORD.read_text())
    edit(rec)
    bad = tmp_path / "dev_gate_by_seed.json"
    bad.write_text(json.dumps(rec))
    return bad


def test_fig3_refuses_when_seed0_no_longer_reproduces_the_published_marker(
    gen, tmp_path, monkeypatch
):
    """THE NON-VACUITY CASE for the lock: the record's reading of the set-aside run must equal the
    marker the published figure drew, or the store the recipe is read from is not that one."""
    _need(gen.BCA_CELLS, gen.APPENDIX_TRAINING, gen.FIG3_RECIPE_RECORD)

    def edit(rec):
        rec["cells"]["s0"]["musique"]["stop.p_stop_given_done"] = 0.9

    monkeypatch.setattr(gen, "FIG3_RECIPE_RECORD", _tampered_dev_record(gen, tmp_path, edit))
    with pytest.raises(gen.Mismatch) as exc:
        gen.fig3_stopping(tmp_path / "out")
    assert "seed-0" in str(exc.value) and "NOT WRITING" in str(exc.value)
    assert not (tmp_path / "out" / "fig3_stopping.pdf").exists()


def test_fig3_refuses_a_pooled_cell_that_is_not_the_pool_of_its_seeds(gen, tmp_path, monkeypatch):
    """The pooled marker is the decision-weighted pool of the two seeds' cells; a count that moved
    in the pooled cell alone must stop the build."""
    _need(gen.BCA_CELLS, gen.APPENDIX_TRAINING, gen.FIG3_RECIPE_RECORD)

    def edit(rec):
        rec["cells"]["s1s2"]["strategyqa"]["stop.n_done"] += 1

    monkeypatch.setattr(gen, "FIG3_RECIPE_RECORD", _tampered_dev_record(gen, tmp_path, edit))
    with pytest.raises(gen.Mismatch) as exc:
        gen.fig3_stopping(tmp_path / "out")
    assert "pool" in str(exc.value) and "NOT WRITING" in str(exc.value)


def test_fig3_refuses_a_marker_that_is_not_the_printed_one(gen, tmp_path, monkeypatch):
    """THE NON-VACUITY CASE for the printed lock: the caption prints the recipe's cells."""
    _need(gen.BCA_CELLS, gen.APPENDIX_TRAINING, gen.FIG3_RECIPE_RECORD)
    moved = {s: dict(v) for s, v in gen.FIG3_RECIPE_PRINTED.items()}
    moved["musique"]["p_ask_given_not_done"] = "0.9437"
    monkeypatch.setattr(gen, "FIG3_RECIPE_PRINTED", moved)
    with pytest.raises(gen.Mismatch) as exc:
        gen.fig3_stopping(tmp_path / "out")
    assert "printed" in str(exc.value) and "NOT WRITING" in str(exc.value)


# --------------------------------------------------------------------------- reader words


def test_no_identifier_reaches_a_rendered_label(gen):
    """House rule: no arm id, field name, file name or hash in a label or legend. Those live in
    the provenance record and in the `%` comments, which is where they belong."""
    banned = (
        "qwen",
        "gpt-oss",
        "inquirer_",
        "arm_id",
        "evidence_coverage",
        "facet_breadth",
        "precedence_violation",
        "max_depth_reached",
        "dwr",
        "stop_2x2",
        "p_stop_given_done",
        "drafter_only",
        "base8b",
        "teacher",
        "stacked",
        ".json",
        ".md",
    )
    labels = list(gen.ARM_LABEL.values())
    labels += [reader for _, reader, _ in gen.DEV_ARMS]
    labels += list(gen.TABLE2_LABEL.values())
    labels += [label for _, label, _ in gen.TABLE1_ROWS]
    labels += list(gen.TABLE1_TITLE.values()) + list(gen.SUITE_TITLE.values())
    labels += [p.title for p in gen._panels()] + [p.ylabel for p in gen._panels()]
    for label in labels:
        low = label.lower()
        assert not any(b in low for b in banned), f"identifier in a rendered label: {label!r}"


def test_emitted_tabulars_carry_no_identifier_in_a_printed_row(gen, tmp_path):
    """The `%` comment lines carry the provenance and may name anything. The PRINTED rows may
    not: they are what a reader sees."""
    _need(gen.TESTSPLIT_CONTRASTS, gen.BCA_CELLS, gen.APPENDIX_TRAINING)
    gen.table1_heldout(tmp_path)
    gen.table2_mechanism(tmp_path)
    banned = ("qwen", "gpt-oss", "inquirer_", "_rate", "_breadth", "stop_2x2", ".json")
    for name in ("table1_heldout", "table2_mechanism"):
        for line in (tmp_path / f"{name}.tex").read_text().splitlines():
            if line.lstrip().startswith("%"):
                continue
            low = line.lower()
            assert not any(b in low for b in banned), f"{name}: identifier in a printed row: {line}"


def test_example_refuses_a_gloss_that_no_longer_matches_the_gold_label(gen, tmp_path, monkeypatch):
    """THE NON-VACUITY CASE for the chain gloss: the figure prints readable labels in place of the
    benchmark's decomposition forms, so if a gold label changes the gloss describes a graph that no
    longer exists and nothing may be written."""
    _need(gen.MUSIQUE_GOLD, gen.EXAMPLE_STORE / "runs.parquet", gen.EXAMPLE_BY_SEED)
    broken = dict(gen.EXAMPLE_CHAIN_GLOSS)
    broken["s1"] = ("Gracie >> producer", "who produced it")
    monkeypatch.setattr(gen, "EXAMPLE_CHAIN_GLOSS", broken)
    with pytest.raises(gen.Mismatch) as exc:
        gen.examples_transcripts(tmp_path)
    assert "no longer stands for the graph" in str(exc.value)
    assert "NOT WRITING" in str(exc.value)
    assert not (tmp_path / "examples_transcripts.tex").exists()


def test_example_carries_no_canary_and_records_both_selection_counts(gen, tmp_path):
    """A canary nonce rides inside gold strings to catch gold reaching a served request. A
    published asset is another place it must never reach. And the rule selects FOR the phenomenon,
    so the count with the arms exchanged has to travel with it."""
    _need(gen.MUSIQUE_GOLD, gen.EXAMPLE_STORE / "runs.parquet", gen.EXAMPLE_BY_SEED)
    gen.examples_transcripts(tmp_path)
    body = (tmp_path / "examples_transcripts.tex").read_text()
    prov = json.loads((tmp_path / "examples_transcripts.provenance.json").read_text())
    assert "PINQCANARY" not in body
    assert "PINQCANARY" not in json.dumps(prov)
    assert set(prov["run_ids_from"]) == set(gen.EXAMPLE_ARMS)
    counts = prov["extra"]["selection_counts"]
    assert counts["trained_resolves_a_deep_need_the_prompted_does_not"] > 0
    assert counts["prompted_resolves_a_deep_need_the_trained_does_not"] > 0
    # Every arm must have actually run: absence would satisfy the rule without failing at it.
    assert all(prov["values"]["arms"][a]["run_id"] for a in gen.EXAMPLE_ARMS)


SEED0_EXAMPLE_TASK = "3hop1__738245_144587_56749"


def test_example_draws_the_recipe_at_both_seeds_after_seed0_reproduces(gen, tmp_path):
    """The trained panel is the recipe (training seeds 1 and 2), never the set-aside run. The rule
    names one trained run per task, so it is read with EACH training seed required to resolve the
    deep need; and before that reading counts, the same rule keyed to the set-aside run's model must
    reproduce the published figure's selection exactly (task, three run ids, 175/16/7)."""
    _need(gen.MUSIQUE_GOLD, gen.EXAMPLE_STORE / "runs.parquet", gen.EXAMPLE_BY_SEED)
    gen.examples_transcripts(tmp_path)
    body = (tmp_path / "examples_transcripts.tex").read_text()
    prov = json.loads((tmp_path / "examples_transcripts.provenance.json").read_text())
    sel = prov["values"]["selection"]
    assert sel["seed0_lock"] == {
        "task_id": SEED0_EXAMPLE_TASK,
        "run_ids": {
            "drafter_only": "8dac459d4191b93dac2ddfc81055eb84",
            "inquirer_prompted": "6e0d6f8f1b47c53f8cf6de8ce7acac80",
            "trained": "8cb53e1942a514274cd30d2a9246f006",
        },
        "n_paired_tasks": 175,
        "n_forward": 16,
        "n_reverse": 7,
    }
    assert (sel["n_paired_tasks"], sel["n_forward"], sel["n_reverse"]) == (175, 16, 3)
    assert {s: (c["n_forward"], c["n_reverse"]) for s, c in sel["per_seed"].items()} == {
        "s1": (18, 4),
        "s2": (17, 5),
    }
    assert sel["task_id"] == SEED0_EXAMPLE_TASK
    arms = prov["values"]["arms"]
    assert arms["trained_s1"]["run_id"] == "73ef10fc9df79eaad3d480a18ee1b13a"
    assert arms["trained_s2"]["run_id"] == "952981b5e20249aab52a72aa0938cae0"
    # The no-question and prompted panels are the published ones.
    assert arms["drafter_only"]["run_id"] == sel["seed0_lock"]["run_ids"]["drafter_only"]
    assert arms["inquirer_prompted"]["run_id"] == sel["seed0_lock"]["run_ids"]["inquirer_prompted"]
    # Seed 0 is locked, and drawn nowhere.
    assert sel["seed0_lock"]["run_ids"]["trained"] not in {a["run_id"] for a in arms.values()}
    low = body.lower()
    assert "seed 0" not in low and "resumed" not in low
    assert "training seed 1" in body and "training seed 2" in body
    # No recorded question is dropped from the drawing: each is a row, a counted repeat, or a
    # counted rewording, and the three add up to the store's own count of questions asked.
    for key, a in arms.items():
        assert a["n_questions_accounted"] == a["n_asks"], key


def test_example_refuses_when_the_seed0_selection_no_longer_reproduces(gen, tmp_path, monkeypatch):
    """THE NON-VACUITY CASE for the lock: the recipe's reading only counts if the same rule, on the
    same store, still selects what the published figure showed."""
    _need(gen.MUSIQUE_GOLD, gen.EXAMPLE_STORE / "runs.parquet", gen.EXAMPLE_BY_SEED)
    moved = dict(gen.EXAMPLE_PUBLISHED_SEED0, n_reverse=8)
    monkeypatch.setattr(gen, "EXAMPLE_PUBLISHED_SEED0", moved)
    with pytest.raises(gen.Mismatch) as exc:
        gen.examples_transcripts(tmp_path)
    assert "seed-0 selection" in str(exc.value) and "NOT WRITING" in str(exc.value)
    assert not (tmp_path / "examples_transcripts.tex").exists()


def test_example_refuses_when_the_audit_record_disagrees(gen, tmp_path, monkeypatch):
    """The recipe's selection is read twice, here and by scripts/seed0_audit/example_by_seed.py.
    One count moved in that record must stop the build: two readers that cannot disagree are one."""
    _need(gen.MUSIQUE_GOLD, gen.EXAMPLE_STORE / "runs.parquet", gen.EXAMPLE_BY_SEED)
    rec = json.loads(gen.EXAMPLE_BY_SEED.read_text())
    rec["by_seed"]["s1"]["n_forward"] = 19
    bad = tmp_path / "example_by_seed.json"
    bad.write_text(json.dumps(rec))
    monkeypatch.setattr(gen, "EXAMPLE_BY_SEED", bad)
    with pytest.raises(gen.Mismatch) as exc:
        gen.examples_transcripts(tmp_path / "out")
    assert "audit record" in str(exc.value) and "NOT WRITING" in str(exc.value)
    assert not (tmp_path / "out" / "examples_transcripts.tex").exists()


def test_example_refuses_a_chain_label_that_is_not_the_gold_answer(gen, tmp_path, monkeypatch):
    """Each arrow in the chain names the entity the need before it reveals. That is asserted
    against the gold graph's own answer for that need, not against a policy's questions: the
    recipe reaches the last need without ever naming the city, which is the point of the figure."""
    _need(gen.MUSIQUE_GOLD, gen.EXAMPLE_STORE / "runs.parquet", gen.EXAMPLE_BY_SEED)
    monkeypatch.setattr(gen, "EXAMPLE_CHAIN_REVEALS", ("Davis Guggenheim", "Chicago"))
    with pytest.raises(gen.Mismatch) as exc:
        gen.examples_transcripts(tmp_path)
    assert "gold answer" in str(exc.value) and "NOT WRITING" in str(exc.value)
    assert not (tmp_path / "examples_transcripts.tex").exists()


def test_tex_escape_emits_tex_quotes_not_curly_ones(gen):
    """Request 1a (main-text owner): the paper's T1 font drops U+201C, U+201D and U+2019 without
    an error, so a recorded answer's curly quotes vanished from the printed figure. The escape
    must emit TeX's own quote ligatures, and an en dash as TeX's."""
    got = gen._tex_escape("when “Meet Me in the birthplace of Gracie’s director”, ‘x’")
    assert got == "when ``Meet Me in the birthplace of Gracie's director'', `x'"
    assert gen._tex_escape("Summer 1903 – spring 1904.") == "Summer 1903 -- spring 1904."
    # A straight double-quoted title inside a question still becomes a single pair, because the
    # template wraps the whole question in double quotes.
    assert gen._tex_escape('titled "Gracie"?') == "titled `Gracie'?"


def test_example_body_carries_no_curly_quote(gen, tmp_path):
    """The end-to-end half of request 1a: the recorded answers DO carry curly quotes, so the
    emitted asset is where the escape is proven, not only the unit above."""
    _need(gen.MUSIQUE_GOLD, gen.EXAMPLE_STORE / "runs.parquet", gen.EXAMPLE_BY_SEED)
    gen.examples_transcripts(tmp_path)
    body = (tmp_path / "examples_transcripts.tex").read_text()
    prov = json.loads((tmp_path / "examples_transcripts.provenance.json").read_text())
    recorded = "".join(a["answer_text"] for a in prov["values"]["arms"].values())
    assert any(c in recorded for c in "“”’"), "the premise: a recorded curly quote"
    assert not any(c in body for c in "‘’“”–—")
    assert "``Meet Me''" in body


def test_no_absolute_home_path_reaches_an_output(gen, tmp_path):
    """`scripts/check_no_home_paths.sh` fails the build over a committed absolute home path, and
    these outputs are committed. The provenance names every input RELATIVE to the repository."""
    _need(gen.TESTSPLIT_CONTRASTS, gen.BCA_CELLS, gen.APPENDIX_TRAINING)
    gen.table1_heldout(tmp_path)
    gen.table2_mechanism(tmp_path)
    gen.fig3_stopping(tmp_path)
    for p in sorted(tmp_path.glob("*")):
        if p.suffix in (".pdf", ".png"):
            continue
        text = p.read_text()
        assert "/Users/" not in text and "/home/" not in text, p.name
        assert str(REPO) not in text, p.name


# --------------------------------------------------------------------- fig4, the chain ladder


def test_fig4_draws_the_three_constructions_and_records_each_scorer_hash(gen, tmp_path):
    _need(gen.CHAIN_LADDER_RECORD)
    gen.fig4_chain_ladder(tmp_path)
    prov = json.loads((tmp_path / "fig4_chain_ladder.provenance.json").read_text())
    assert (tmp_path / "fig4_chain_ladder.pdf").exists()
    shapes = [r["shape"] for r in prov["rows"]]
    assert shapes == ["2x4", "3x6", "4x8"]
    # Three constructions, three scorer hashes, never pooled. A single hash across the rows would
    # mean one store scored all three, which is exactly what the record says is not the case.
    assert len({r["scorer_hash"] for r in prov["rows"]}) == 3
    assert prov["suite_role"] == "calibration"
    deltas = [r["delta"] for r in prov["rows"]]
    assert deltas == sorted(deltas, reverse=True), "the figure claims a monotone decay"
    assert all(r["ci"][0] > 0 for r in prov["rows"]), "every interval must exclude zero"


def test_fig4_refuses_a_value_absent_from_the_record(gen, tmp_path, monkeypatch):
    """THE NON-VACUITY CASE: the lock is textual, so it has to fail on a transcription slip.

    A hand-copied figure's real failure mode is one digit wrong in one field, so that is what is
    injected here rather than a wholesale replacement.
    """
    _need(gen.CHAIN_LADDER_RECORD)
    bad = list(gen.CHAIN_LADDER)
    row = list(bad[1])
    row[3] = 0.0148  # published value is +0.0147
    bad[1] = tuple(row)
    monkeypatch.setattr(gen, "CHAIN_LADDER", tuple(bad))
    with pytest.raises(gen.Mismatch) as exc:
        gen.fig4_chain_ladder(tmp_path)
    assert "NOT WRITING" in str(exc.value) and "delta" in str(exc.value)


def test_fig4_refuses_a_non_monotone_advantage(gen, tmp_path, monkeypatch):
    """The figure's own claim is asserted over the table, not read off the drawing.

    An earlier version of this figure asserted that the deepest-need GAP widens with chain length.
    It does not (0.26, 0.24, 0.37) and the assertion caught it, which is why the shape claims are
    checked here rather than trusted.
    """
    _need(gen.CHAIN_LADDER_RECORD)
    # Swapping two rows keeps every value present in the record, so the textual lock still
    # passes and it is the shape assertion that has to fire.
    swapped = (gen.CHAIN_LADDER[1], gen.CHAIN_LADDER[0], gen.CHAIN_LADDER[2])
    monkeypatch.setattr(gen, "CHAIN_LADDER", swapped)
    with pytest.raises(gen.Mismatch) as exc:
        gen.fig4_chain_ladder(tmp_path)
    assert "monotone" in str(exc.value)


# ------------------------------------------------- table5, the completed-comparator rows


def test_table5_writes_every_row_at_the_completed_population(gen, tmp_path):
    _need(gen.COMPLETED_COVERAGE, gen.COMPLETED_PLAN / "symmetric_completed.json")
    gen.table5_completed(tmp_path)
    body = (tmp_path / "table5_completed.tex").read_text()
    prov = json.loads((tmp_path / "table5_completed.provenance.json").read_text())
    assert r"\begin{table}" not in body, "a bare tabular is pasted into the paper, not a float"
    assert prov["cells"]["evidence_coverage::musique"]["n"] == 200
    assert prov["cells"]["evidence_coverage::wiki2"]["n"] == 200
    # The direction is the finding and it is unflattering, so it is pinned: nine of the ten cells
    # off the control suite move DOWN, and the tenth is the adverse out-of-order cell.
    assert prov["cells_moved_down_off_the_control_suite"] == 9
    order = prov["cells"]["precedence_violation_rate::musique"]
    assert order["delta"] > order["published"], "the one adverse cell strengthens on completion"


def test_table5_refuses_when_the_control_suite_moves(gen, tmp_path, monkeypatch):
    """THE NON-VACUITY CASE, and it is the whole argument of the table.

    The implicit-reasoning comparator was never short, so its cells must reproduce. If they can
    move without the table refusing, then nothing distinguishes 'the population changed' from
    'the reader changed' and every other cell's movement is uninterpretable.
    """
    _need(gen.COMPLETED_COVERAGE)
    lock = dict(gen.COMPLETED_PUBLISHED_LOCK)
    lock[("evidence_coverage", "strategyqa")] = 0.0700
    monkeypatch.setattr(gen, "COMPLETED_PUBLISHED_LOCK", lock)
    with pytest.raises(gen.Mismatch) as exc:
        gen.table5_completed(tmp_path)
    assert "must not move" in str(exc.value) and "NOT WRITING" in str(exc.value)


def test_table5_refuses_a_record_that_locks_against_a_different_published_value(
    gen, tmp_path, monkeypatch
):
    """A completed cell is only interpretable against the value the paper actually prints."""
    _need(gen.COMPLETED_PLAN / "symmetric_completed.json")
    lock = dict(gen.COMPLETED_PUBLISHED_LOCK)
    lock[("dwr", "musique")] = 0.1376  # the asymmetric published value, not the symmetric one
    monkeypatch.setattr(gen, "COMPLETED_PUBLISHED_LOCK", lock)
    with pytest.raises(gen.Mismatch) as exc:
        gen.table5_completed(tmp_path)
    assert "Not the same cell" in str(exc.value)


# ------------------------------------------------- fig5, the cost-basis inversion


def test_fig5_reads_both_bases_from_one_population(gen, tmp_path):
    _need(gen.COMPLETED_COVERAGE)
    gen.fig5_cost_basis(tmp_path)
    prov = json.loads((tmp_path / "fig5_cost_basis.provenance.json").read_text())
    assert (tmp_path / "fig5_cost_basis.pdf").exists()
    for c in prov["cells"]:
        assert c["n"] == 200, "both bases must be read on the completed population"
        assert c["matched_ci"][0] > 0, "every equal-spend interval excludes zero"
        assert c["base_asks"] > c["trained_asks"], "the comparator outspends the trained arm"
    flips = [c for c in prov["cells"] if c["cap8"] < 0 < c["matched"]]
    assert len(flips) == 2, "the figure's claim is that two suites change sign"


def test_fig5_refuses_two_bases_on_two_populations(gen, tmp_path, monkeypatch):
    """THE NON-VACUITY CASE, and it is the figure's whole point.

    If the fixed-cap column could come from a different population than the equal-spend column,
    the drawn sign change would confound a basis difference with a population difference -- which
    is the exact error the completed-comparator work exists to remove.
    """
    _need(gen.COMPLETED_COVERAGE)
    doc = json.loads(gen.COMPLETED_COVERAGE.read_text())
    doc["complete"]["by_suite"]["musique"]["cap8_n_tasks"] = 176
    fake = tmp_path / "three.json"
    fake.write_text(json.dumps(doc))
    monkeypatch.setattr(gen, "COMPLETED_COVERAGE", fake)
    with pytest.raises(gen.Mismatch) as exc:
        gen.fig5_cost_basis(tmp_path / "figures")
    assert "ONE population" in str(exc.value) and "NOT WRITING" in str(exc.value)


# ------------------------------------------------- fig6, tokens per useful need


def test_fig6_recomputes_the_levels_and_lands_on_the_published_delta(gen, tmp_path):
    _need(gen.EFFICIENCY_RECORD, gen.EFFICIENCY_STORE / "runs.parquet")
    # The seed-0 route still lands on the published levels. The recomputed levels are the PAIRED
    # ones, which differ from the record's own unpaired levels in the third digit.
    s0 = {r["suite"]: r for r in gen._efficiency_rows()}
    assert round(s0["musique"]["trained"]) == 8365
    assert round(s0["musique"]["prompted"]) == 19402
    assert s0["musique"]["n_paired_tasks"] == 174
    # The figure DRAWS the recipe (seeds 1 and 2, 2026-09-23): seed 0 kept about a tenth of its
    # final training stage (artifacts/seed_identity_20260923/RESULT.md), and the drawn rows are the
    # per-seed record's s1s2 cells, which the generator only reaches after the seed-0 lock above.
    gen.fig6_efficiency(tmp_path)
    prov = json.loads((tmp_path / "fig6_efficiency.provenance.json").read_text())
    got = {r["suite"]: r for r in prov["rows"]}
    assert round(got["musique"]["trained"]) == 11610
    assert round(got["musique"]["prompted"]) == 19354
    assert got["musique"]["n_paired_tasks"] == 175
    for r in prov["rows"]:
        assert r["ci"][1] < 0, "every interval excludes zero on the favourable side"


def test_fig6_refuses_when_the_recomputation_misses_the_published_delta(gen, tmp_path, monkeypatch):
    """THE NON-VACUITY CASE: a reimplementation that does not reproduce the record is a bug."""
    _need(gen.EFFICIENCY_RECORD, gen.EFFICIENCY_STORE / "runs.parquet")
    deltas = dict(gen.EFFICIENCY_DELTAS)
    deltas["musique"] = (-11037 + 50, -13029, -9198)
    monkeypatch.setattr(gen, "EFFICIENCY_DELTAS", deltas)
    with pytest.raises(gen.Mismatch) as exc:
        gen.fig6_efficiency(tmp_path)
    assert "NOT WRITING" in str(exc.value)


# ------------------------------------------------- fig7, what each suite can express


def test_fig7_reproduces_the_gold_census_and_the_empty_region(gen, tmp_path):
    _need(gen.GOLD_GRAPHS / "musique" / "v1.jsonl")
    gen.fig7_suite_axes(tmp_path)
    prov = json.loads((tmp_path / "fig7_suite_axes.provenance.json").read_text())
    by = {r["suite"]: r for r in prov["rows"]}
    assert by["musique"]["n_tasks"] == 800 and by["musique"]["max_depth"] == 3
    assert by["wiki2"]["n_multi_facet"] == 2663 and by["wiki2"]["max_depth"] == 1
    # The anti-correlation is the finding: the only branching suite is the shallowest.
    assert by["musique"]["n_multi_facet"] == 0 and by["strategyqa"]["n_multi_facet"] == 0
    assert not [r for r in prov["rows"] if r["share_multi_facet"] > 0 and r["max_depth"] >= 2]


def test_fig7_refuses_a_census_that_does_not_reproduce(gen, tmp_path, monkeypatch):
    """THE NON-VACUITY CASE: the figure is a census, so a census that moved is a different claim.

    A first version of this read the node fields as `depth` and `facet_id` rather than
    `gold_depth` and `gold_facet_id` and got zeros back, which looked exactly like a finding.
    """
    _need(gen.GOLD_GRAPHS / "musique" / "v1.jsonl")
    lock = dict(gen.SUITE_CENSUS_LOCK)
    lock["musique"] = (800, 2660, 0, 2)  # published deepest chain is 3
    monkeypatch.setattr(gen, "SUITE_CENSUS_LOCK", lock)
    with pytest.raises(gen.Mismatch) as exc:
        gen.fig7_suite_axes(tmp_path)
    assert "recomputed census" in str(exc.value) and "NOT WRITING" in str(exc.value)


# ------------------------------------------------- fig8, the controls


def test_fig8_draws_three_states_and_names_the_margin(gen, tmp_path):
    _need(gen.CONTROLS_RECORD)
    gen.fig8_controls(tmp_path)
    prov = json.loads((tmp_path / "fig8_controls.provenance.json").read_text())
    assert prov["margin"] == 0.05
    rows = {r["variant"]: r for r in prov["rows"]}
    # Two clear the margin by their lower bound, one excludes zero while reaching into it, and the
    # single-model variant is an equivalence. An earlier version claimed THREE cleared the margin.
    assert sum(r["clears_margin_by_lower_bound"] for r in prov["rows"]) == 2
    blind = rows["blind to evidence"]
    assert blind["excludes_zero"] and not blind["clears_margin_by_lower_bound"]
    single = rows["one model, both roles"]
    assert single["ci"][0] < 0 < single["ci"][1]


def test_fig8_refuses_when_the_state_counts_change(gen, tmp_path, monkeypatch):
    """THE NON-VACUITY CASE for the three-state encoding the drawing relies on."""
    _need(gen.CONTROLS_RECORD)
    rows = list(gen.CONTROLS)
    r = list(rows[2])
    r[2] = 0.0600  # push the evidence-blind lower bound above the margin
    rows[2] = tuple(r)
    monkeypatch.setattr(gen, "CONTROLS", tuple(rows))
    with pytest.raises(gen.Mismatch) as exc:
        gen.fig8_controls(tmp_path)
    assert "does not contain" in str(exc.value) or "two decisive" in str(exc.value)


# ------------------------------------------------- fig9, the transfer tail


def test_fig9_keeps_the_airline_inversion(gen, tmp_path):
    """The figure's honesty hinges on this: what replicates is the RATE, not the magnitude."""
    _need(gen.TAU2_SOURCE, gen.TAU2_EFFECT_SOURCE)
    gen.fig9_transfer_tail(tmp_path)
    prov = json.loads((tmp_path / "fig9_transfer_tail.provenance.json").read_text())
    by = {(r["domain"], r["arm"]): r for r in prov["strata"]}
    for domain in ("retail", "airline"):
        q, c = by[(domain, "questioner")], by[(domain, "comparator")]
        assert q["over_n"] < c["over_n"], "the runaway RATE favours the questioner on both"
        assert q["inside_n"] + q["over_n"] == c["inside_n"] + c["over_n"] == 102
    # Retail: better in the tail. Airline: WORSE in the tail. Both must survive into the record.
    assert by[("retail", "questioner")]["over_mean"] < by[("retail", "comparator")]["over_mean"]
    assert by[("airline", "questioner")]["over_mean"] > by[("airline", "comparator")]["over_mean"]
    assert prov["retail_effect_trace_clustered"]["ci"] == [-6.28, -3.04]


def test_fig9_refuses_if_the_airline_inversion_disappears(gen, tmp_path, monkeypatch):
    """THE NON-VACUITY CASE: if the inversion can vanish silently, the figure overclaims.

    Airline's interval covers zero BECAUSE its few runaways are longer for the questioner. A
    version of this figure that lost that would present airline as a replication of the effect
    size rather than of the rate alone.
    """
    _need(gen.TAU2_SOURCE)
    rows = list(gen.TAU2_STRATA)
    i = next(k for k, r in enumerate(rows) if r[0] == "airline" and r[1] == "questioner")
    r = list(rows[i])
    r[4] = 6.02  # below the comparator's 8.02, i.e. pretend airline replicates retail
    rows[i] = tuple(r)
    monkeypatch.setattr(gen, "TAU2_STRATA", tuple(rows))
    with pytest.raises(gen.Mismatch) as exc:
        gen.fig9_transfer_tail(tmp_path)
    assert "NOT WRITING" in str(exc.value) or "INVERT" in str(exc.value)


# ------------------------------------------------- fig10, who ends the run


def test_fig10_draws_the_recipe_from_the_one_commit_record_at_the_printed_values(gen, tmp_path):
    """Who ends the run, for the reported questioner: the recipe (training seeds 1 and 2 averaged
    within the unit) and both prompted comparators at five ceilings, from one record at one commit.
    Every endpoint the main text prints is a lock, and the set-aside run's published three-suite
    sweep is still read and its ordering re-asserted, but not drawn."""
    _need(gen.FRONTIER_RECIPE, gen.FRONTIER_MUSIQUE, gen.FRONTIER_STRATEGYQA, gen.FRONTIER_FRAMES)
    gen.fig10_ceiling(tmp_path)
    prov = json.loads((tmp_path / "fig10_ceiling.provenance.json").read_text())
    assert (tmp_path / "fig10_ceiling.pdf").stat().st_size > 0
    cells = prov["values"]
    assert set(cells) == {"musique", "strategyqa"}
    for suite, arms in cells.items():
        assert set(arms) == {"recipe", "base", "teacher"}, suite
        for arm, by_cap in arms.items():
            assert sorted(int(c) for c in by_cap) == [4, 8, 12, 16, 24], (suite, arm)
            assert all(v["n"] == 400 for v in by_cap.values()), (suite, arm)
        for cap in ("4", "8", "12", "16", "24"):
            rec = arms["recipe"][cap]["ceiling_hit"]
            assert rec < arms["base"][cap]["ceiling_hit"], (suite, cap)
            assert rec < arms["teacher"][cap]["ceiling_hit"], (suite, cap)
    # The values the main text prints (section 5.2), to the printed precision.
    printed = {
        ("musique", "recipe", "4"): "0.3387",
        ("musique", "recipe", "24"): "0.0762",
        ("strategyqa", "recipe", "4"): "0.1850",
        ("strategyqa", "recipe", "24"): "0.0375",
    }
    for (suite, arm, cap), want in printed.items():
        assert f"{cells[suite][arm][cap]['ceiling_hit']:.4f}" == want, (suite, arm, cap)
    assert prov["extra"]["one_code_version"] == "a61c4f4be3e9"
    # The set-aside run is read and locked, never drawn.
    assert "s0" not in cells["musique"] and "s0" not in cells["strategyqa"]
    s0 = prov["extra"]["seed0_published_sweep_not_drawn"]
    assert set(s0) == {"MuSiQue", "StrategyQA", "FRAMES (no gold)"}


def test_fig10_refuses_when_the_ordering_fails_at_one_cap(gen, tmp_path, monkeypatch):
    """THE NON-VACUITY CASE: the panel asserts an ordering at EVERY cap, not on average."""
    _need(gen.FRONTIER_MUSIQUE)
    real = gen.parse_ceiling_hit

    def tampered(text, metric):
        cells = dict(real(text, metric))
        if ("trained", 8) in cells and ("base8b", 8) in cells:
            cells[("trained", 8)] = cells[("base8b", 8)] + 1.0
        return cells

    monkeypatch.setattr(gen, "parse_ceiling_hit", tampered)
    with pytest.raises(gen.Mismatch) as exc:
        gen.fig10_ceiling(tmp_path)
    assert "is not below" in str(exc.value) and "NOT WRITING" in str(exc.value)


def _tampered_frontier_recipe(gen, tmp_path, suite, cap, share):
    """A copy of the record with the recipe's ceiling share moved at one cell, kept internally
    consistent (both seeds and their average moved together) so only the guard under test fires."""
    rec = json.loads(gen.FRONTIER_RECIPE.read_text())
    for arm in ("recipe", "s1", "s2"):
        rec["levels"][suite][arm][str(cap)]["ceiling_hit"] = share
    bad = tmp_path / "frontier_recipe.json"
    bad.write_text(json.dumps(rec))
    return bad


def test_fig10_refuses_when_the_recipe_ordering_fails_at_one_cap(gen, tmp_path, monkeypatch):
    """THE NON-VACUITY CASE for the drawn curves: the recipe below BOTH prompted arms at EVERY cap."""
    _need(gen.FRONTIER_RECIPE, gen.FRONTIER_MUSIQUE, gen.FRONTIER_STRATEGYQA, gen.FRONTIER_FRAMES)
    bad = _tampered_frontier_recipe(gen, tmp_path, "musique", 8, 0.49)
    monkeypatch.setattr(gen, "FRONTIER_RECIPE", bad)
    with pytest.raises(gen.Mismatch) as exc:
        gen.fig10_ceiling(tmp_path / "out")
    assert "is not below" in str(exc.value) and "NOT WRITING" in str(exc.value)
    assert not (tmp_path / "out" / "fig10_ceiling.pdf").exists()


def test_fig10_refuses_a_drawn_value_that_is_not_the_printed_one(gen, tmp_path, monkeypatch):
    """THE NON-VACUITY CASE for the lock: section 5.2 prints 0.3387 at the smallest ceiling, so a
    record that moved there must not be drawn beside that sentence."""
    _need(gen.FRONTIER_RECIPE, gen.FRONTIER_MUSIQUE, gen.FRONTIER_STRATEGYQA, gen.FRONTIER_FRAMES)
    bad = _tampered_frontier_recipe(gen, tmp_path, "musique", 4, 0.35)
    monkeypatch.setattr(gen, "FRONTIER_RECIPE", bad)
    with pytest.raises(gen.Mismatch) as exc:
        gen.fig10_ceiling(tmp_path / "out")
    assert "printed" in str(exc.value) and "NOT WRITING" in str(exc.value)
    assert not (tmp_path / "out" / "fig10_ceiling.pdf").exists()


# ------------------------------------------------- fig2, the recipe's budget sweep at one commit


def test_fig2_recipe_draws_three_curves_and_the_seed0_point_at_one_commit(gen, tmp_path):
    _need(gen.FRONTIER_RECIPE)
    gen.fig2_frontier_recipe(tmp_path)
    prov = json.loads((tmp_path / "fig2_frontier_recipe.provenance.json").read_text())
    assert prov["extra"]["one_code_version"] == "a61c4f4be3e9"
    assert len(prov["extra"]["one_base_url_sha"]) == 1
    for suite in ("musique", "strategyqa"):
        cells = prov["values"][suite]
        for arm in ("recipe", "base", "teacher"):
            assert sorted(int(c) for c in cells[arm]) == [4, 8, 12, 16, 24]
            assert all(v["n"] == 400 for v in cells[arm].values())
        assert cells["s0_cap8"]["n"] == 400


def test_fig2_recipe_refuses_a_record_whose_lock_failed(gen, tmp_path, monkeypatch):
    """THE NON-VACUITY CASE: one published frontier cell that no longer reproduces stops the build."""
    _need(gen.FRONTIER_RECIPE)
    rec = json.loads(gen.FRONTIER_RECIPE.read_text())
    first = sorted(rec["lock"])[0]
    rec["lock"][first]["ok"] = False
    bad = tmp_path / "frontier_recipe.json"
    bad.write_text(json.dumps(rec))
    monkeypatch.setattr(gen, "FRONTIER_RECIPE", bad)
    with pytest.raises(gen.Mismatch) as exc:
        gen.fig2_frontier_recipe(tmp_path / "out")
    assert "NOT DRAWING" in str(exc.value)
    assert not (tmp_path / "out" / "fig2_frontier_recipe.pdf").exists()


def test_fig2_recipe_refuses_a_short_cell(gen, tmp_path, monkeypatch):
    _need(gen.FRONTIER_RECIPE)
    rec = json.loads(gen.FRONTIER_RECIPE.read_text())
    rec["levels"]["strategyqa"]["recipe"]["24"]["n"] = 399
    bad = tmp_path / "frontier_recipe.json"
    bad.write_text(json.dumps(rec))
    monkeypatch.setattr(gen, "FRONTIER_RECIPE", bad)
    with pytest.raises(gen.Mismatch):
        gen.fig2_frontier_recipe(tmp_path / "out")


# ------------------------------------------------- fig11, the two axes with the recipe's marker


def test_fig11_draws_the_recipe_beside_seed0_on_the_published_runs(gen, tmp_path):
    _need(gen.TWO_AXIS / "analysis_4x8.json", gen.TWO_AXIS_RECIPE / "4x8" / "analysis_rec.json")
    gen.fig11_axes_separable(tmp_path)
    prov = json.loads((tmp_path / "fig11_axes_separable.provenance.json").read_text())
    for shape in ("2x4", "3x6", "4x8"):
        cells = prov["values"][shape]
        assert {"inquirer_trained", "recipe", "inquirer_prompted"} <= set(cells)
        # seed 0 is still the published level; the recipe is a different point
        assert cells["inquirer_trained"] != cells["recipe"]


def test_fig11_refuses_a_farm_whose_prompted_runs_are_not_the_published_ones(
    gen, tmp_path, monkeypatch
):
    """THE NON-VACUITY CASE: the recipe's farm must hold the published comparator runs, byte for byte."""
    _need(gen.TWO_AXIS / "analysis_4x8.json", gen.TWO_AXIS_RECIPE / "4x8" / "analysis_rec.json")
    fake = tmp_path / "recipe"
    for shape in ("2x4", "3x6", "4x8"):
        rec = json.loads((gen.TWO_AXIS_RECIPE / shape / "analysis_rec.json").read_text())
        if shape == "3x6":
            rec["corner_levels"]["inquirer_prompted"]["evidence_coverage"] += 1e-6
        (fake / shape).mkdir(parents=True)
        (fake / shape / "analysis_rec.json").write_text(json.dumps(rec))
    monkeypatch.setattr(gen, "TWO_AXIS_RECIPE", fake)
    with pytest.raises(gen.Mismatch) as exc:
        gen.fig11_axes_separable(tmp_path / "out")
    assert "3x6/inquirer_prompted" in str(exc.value)


# ------------------------------------------------- fig8, the controls on the recipe's own weights


def test_fig8_recipe_draws_each_control_at_its_rule(gen, tmp_path):
    _need(gen.CONTROLS_RECIPE_DIR / "controls_symmetric_recipe.json")
    gen.fig8_controls_recipe(tmp_path)
    prov = json.loads((tmp_path / "fig8_controls_recipe.provenance.json").read_text())
    rows = prov["values"]
    assert len(rows) == 10, "random, 4 no-sequencing, 4 evidence-blind, single model"
    by = {(r["group"], r["suite"], r["seed"]): r for r in rows}
    assert by[("no sequencing", "musique", "s1")]["decided"]
    assert not by[("blind to evidence", "strategyqa", "s2")]["decided"]
    assert by[("one model, both roles", "both", "both")]["hi"] < gen.CONTROLS_MARGIN


def test_fig8_recipe_refuses_when_a_source_lock_is_not_locked(gen, tmp_path, monkeypatch):
    """THE NON-VACUITY CASE: a record whose published-cell lock failed must stop the figure."""
    _need(gen.CONTROLS_RECIPE_DIR / "controls_symmetric_recipe.json")
    fake = tmp_path / "rec"
    fake.mkdir()
    for f in (
        "depth1_matched_recipe.json",
        "controls_symmetric_recipe.json",
        "controls_recipe.json",
        "controls_by_seed.json",
    ):
        d = json.loads((gen.CONTROLS_RECIPE_DIR / f).read_text())
        if f == "depth1_matched_recipe.json":
            first = sorted(d["published_lock"])[0]
            d["published_lock"][first]["verdict"] = "MISMATCH"
        (fake / f).write_text(json.dumps(d))
    monkeypatch.setattr(gen, "CONTROLS_RECIPE_DIR", fake)
    with pytest.raises(gen.Mismatch) as exc:
        gen.fig8_controls_recipe(tmp_path / "out")
    assert "NOT DRAWING" in str(exc.value)


# ------------------------------------------------- fig4, the chain ladder on the recipe


def test_fig4_recipe_draws_both_questioners_on_the_same_corpora(gen, tmp_path):
    _need(gen.TWO_AXIS_STABILITY)
    gen.fig4_chain_ladder_recipe(tmp_path)
    prov = json.loads((tmp_path / "fig4_chain_ladder_recipe.provenance.json").read_text())
    cells = prov["values"]
    assert len(cells) == 12, "two questioners x two quantities x three shapes"
    assert cells["s0/facet_breadth_scorer/4x8"]["delta"] < 0
    assert cells["rec/facet_breadth_scorer/4x8"]["delta"] > 0


def test_fig4_recipe_refuses_a_stability_record_whose_lock_failed(gen, tmp_path, monkeypatch):
    """THE NON-VACUITY CASE: one contrast that no longer matches the published analysis stops it."""
    _need(gen.TWO_AXIS_STABILITY)
    rec = json.loads(gen.TWO_AXIS_STABILITY.read_text())
    rec["lock"][sorted(rec["lock"])[0]]["ok"] = False
    bad = tmp_path / "stability.json"
    bad.write_text(json.dumps(rec))
    monkeypatch.setattr(gen, "TWO_AXIS_STABILITY", bad)
    with pytest.raises(gen.Mismatch) as exc:
        gen.fig4_chain_ladder_recipe(tmp_path / "out")
    assert "NOT DRAWING" in str(exc.value)


# ------------------------------------------------- the two-regimes table (Q1 equal spend, Q2 own stop)


def test_table_regimes_prints_nine_decided_cells_and_marks_the_three_others(gen, tmp_path):
    _need(gen.REGIMES)
    gen.table_regimes(tmp_path)
    prov = json.loads((tmp_path / "table_regimes.provenance.json").read_text())
    assert prov["decided_cells"] == 9
    tex = (tmp_path / "table_regimes.tex").read_text()
    assert tex.count(r"$^{\circ}$") == 3


def test_table_regimes_refuses_a_cell_without_three_50k_bounds(gen, tmp_path, monkeypatch):
    """THE NON-VACUITY CASE: a Q2 cell that lost a verdict reading stops the build."""
    _need(gen.REGIMES)
    rec = json.loads(gen.REGIMES.read_text())
    rec["cells"]["teacher"]["q2"]["wiki2"]["b50"] = rec["cells"]["teacher"]["q2"]["wiki2"]["b50"][
        :2
    ]
    bad = tmp_path / "regimes.json"
    bad.write_text(json.dumps(rec))
    monkeypatch.setattr(gen, "REGIMES", bad)
    with pytest.raises(gen.Mismatch) as exc:
        gen.table_regimes(tmp_path / "out")
    assert "NOT WRITING" in str(exc.value)


def test_table_regimes_refuses_q1_that_disagrees_with_table1(gen, tmp_path, monkeypatch):
    _need(gen.REGIMES)
    rec = json.loads(gen.REGIMES.read_text())
    rec["cells"]["base"]["q1"]["musique"]["delta"] += 0.001
    bad = tmp_path / "regimes.json"
    bad.write_text(json.dumps(rec))
    monkeypatch.setattr(gen, "REGIMES", bad)
    with pytest.raises(gen.Mismatch):
        gen.table_regimes(tmp_path / "out")


# ------------------------------------------------- the tool-use figure (fig13)


# Figure 4 as redesigned on the user's round-2 comments (INBOX_FOR_PAPER_LANE section 14, item 2): two
# panels (task success in percentage points; the customer's follow-up turns per dialogue, in turns),
# and per row two comparators, the same model, prompted (final/transfer_rerun.json) and GPT-OSS-120B,
# prompted (final_120b/transfer_rerun_120b.json). Pooled mean_delta per row, read by hand from each
# record's pooled[*].guard.task_level (success) and pooled[*].secondary["follow-up user turns"]
# (follow-ups), to four decimals; the drawing must reproduce them from the records.
TAU2_EXPECTED = {
    "same": {
        "tau2_retail::tau2_base": (0.2033, -0.6600),
        "tau2_retail::tau2_stop": (0.2000, -0.7367),
        "tau2_airline::tau2_base": (0.0686, -0.0588),
        "tau2_airline::tau2_stop": (0.0637, -0.4412),
    },
    "teacher": {
        "tau2_retail::tau2_base": (0.1700, -1.6000),
        "tau2_retail::tau2_stop": (0.1267, -1.7500),
        "tau2_airline::tau2_base": (0.0000, 0.7549),
        "tau2_airline::tau2_stop": (0.0343, -0.0686),
    },
}


def _tau2_records(gen):
    return {"same": gen.TAU2_READER, "teacher": gen.TAU2_READER_120B}


def test_fig13_draws_both_comparators_from_their_own_records(gen, tmp_path):
    """Every drawn point and interval is its record's own pooled field (the reader's 10,000-resample
    printed interval at seed 0, over tasks), and lands on the value read by hand. Same model: every
    follow-up interval spans zero. GPT-OSS-120B: the retail follow-up intervals exclude zero, the
    airline ones span it. The provenance carries both records' sha256 and each record's source sha256,
    and reads nothing else (paper_levels.json drew the success levels, which are gone)."""
    import hashlib

    _need(gen.TAU2_READER, gen.TAU2_READER_120B)
    gen.fig13_tau2(tmp_path)
    prov = json.loads((tmp_path / "fig13_tau2.provenance.json").read_text())
    rows = prov["values"]["rows"]
    assert set(rows) == {k for k, _ in gen.TAU2_ROWS}  # the record is written with sorted keys
    assert all(rows[k]["label"] == label for k, label in gen.TAU2_ROWS)
    for who, path in _tau2_records(gen).items():
        rec = json.loads(path.read_text())
        meta = prov["values"]["records"][who]
        assert meta["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest(), who
        assert meta["source_sha256"] == rec["source"]["sha256"], who
        pooled = {f"{p['suite']}::{p['variant']}": p for p in rec["pooled"]}
        for key, (succ, fu) in TAU2_EXPECTED[who].items():
            f = pooled[key]["secondary"]["follow-up user turns"]
            src = {
                "success": pooled[key]["guard"]["task_level"],
                "follow_ups": f.get("task_level", f),
            }
            for name, want in (("success", succ), ("follow_ups", fu)):
                got, field = rows[key][who][name], src[name]
                pi = field["printed_interval"]
                assert (pi["n_boot"], pi["seed"]) == (10000, 0), (who, key, name)
                assert (got["point"], got["ci_lo"], got["ci_hi"]) == (
                    field["mean_delta"],
                    pi["lo"],
                    pi["hi"],
                ), (who, key, name)
                assert abs(got["point"] - want) < 5e-5, (who, key, name, got["point"])
    for key in rows:
        same, big = rows[key]["same"]["follow_ups"], rows[key]["teacher"]["follow_ups"]
        assert same["ci_lo"] < 0 < same["ci_hi"], key
        if key.startswith("tau2_retail"):
            assert big["ci_hi"] < 0, key
        else:
            assert big["ci_lo"] < 0 < big["ci_hi"], key
    assert set(prov["inputs"]) == {
        gen.rel(gen.TAU2_READER),
        gen.rel(gen.TAU2_READER_120B),
    }
    # the direction check is named, per record, in the provenance
    for who in ("same", "teacher"):
        direction = prov["values"]["records"][who]["direction"]
        assert "trained minus comparator" in direction["declared"], who
        assert direction["declared_by"] == "rules.primary", who


def _swap(d, a, b):
    d[a], d[b] = d[b], d[a]


# Each breakage leaves the drawn pooled numbers where they are, or moves only them, and must stop the
# build: the direction is read from the record's own fields, never assumed.
TAU2_BREAKAGES = {
    # the declared estimand says the other way round
    "declared_reversed": lambda r: r["rules"].__setitem__(
        "primary",
        r["rules"]["primary"].replace("trained minus comparator", "comparator minus trained"),
    ),
    # the treatment and control arms swapped
    "arms_swapped": lambda r: _swap(r["rules"], "treatment_arm", "control_arm"),
    # the record in the wrong slot (its comparator is the other one's)
    "wrong_comparator": lambda r: r["rules"].__setitem__(
        "comparator_model",
        "qwen3-8b-base"
        if r["rules"]["comparator_model"] != "qwen3-8b-base"
        else "openai/aws/gpt-oss-120b",
    ),
    # each seed's follow-up levels read the other way round
    "follow_up_levels_swapped": lambda r: [
        _swap(c["secondary"]["follow-up user turns"], "trained_level", "control_level")
        for c in r["cells"]
    ],
    # each seed's success counts read the other way round
    "success_counts_swapped": lambda r: [
        _swap(c["guard"]["pair_level_NOT_QUOTABLE"], "trained_only", "control_only")
        for c in r["cells"]
    ],
    # a pooled point that is not the pool of its seeds' task deltas
    "pooled_success_negated": lambda r: r["pooled"][0]["guard"]["task_level"].__setitem__(
        "mean_delta", -r["pooled"][0]["guard"]["task_level"]["mean_delta"]
    ),
    "pooled_follow_ups_negated": lambda r: r["pooled"][0]["secondary"][
        "follow-up user turns"
    ].__setitem__("mean_delta", -r["pooled"][0]["secondary"]["follow-up user turns"]["mean_delta"]),
}


@pytest.mark.parametrize("breakage", sorted(TAU2_BREAKAGES))
@pytest.mark.parametrize("who", ["same", "teacher"])
def test_fig13_refuses_a_record_it_cannot_read_as_trained_minus_comparator(
    gen, tmp_path, monkeypatch, who, breakage
):
    """THE NON-VACUITY CASE: a record whose own fields do not say trained minus comparator, or whose
    pooled point is not the pool of its seeds read that way, stops the build and writes nothing."""
    _need(gen.TAU2_READER, gen.TAU2_READER_120B)
    path = _tau2_records(gen)[who]
    rec = json.loads(path.read_text())
    TAU2_BREAKAGES[breakage](rec)
    bad = tmp_path / path.name
    bad.write_text(json.dumps(rec))
    monkeypatch.setattr(gen, "TAU2_READER" if who == "same" else "TAU2_READER_120B", bad)
    with pytest.raises(gen.Mismatch) as exc:
        gen.fig13_tau2(tmp_path / "out")
    assert "NOT DRAWING" in str(exc.value)
    assert not (tmp_path / "out").exists() or list((tmp_path / "out").iterdir()) == []


# ------------------------------------------------- the headline table (tab:heldout, G1)


def _headline_expected_undecided(gen) -> int:
    """Counted from the records, independently of the generator's own flags: a cell is decided
    only if all three 50k intervals exclude zero on one side."""

    def dec(bounds):
        return all(lo > 0 for lo, _ in bounds) or all(hi < 0 for _, hi in bounds)

    t1 = json.loads(gen.SEED_IDENTITY.read_text())["cells"]["s1s2"]
    reg = json.loads(gen.REGIMES.read_text())["cells"]
    ans = json.loads(gen.ANSWER_F1.read_text())["cells"]["s1s2"]
    n = 0
    for key, rows in t1.items():
        n += not dec([(r["ci_lo"], r["ci_hi"]) for r in rows if r["n_boot"] == 50000])
    for comp, q in (("teacher", "q1"), ("base", "q2"), ("teacher", "q2")):
        for c in reg[comp][q].values():
            n += not dec([(lo, hi) for _, lo, hi in c["b50"]])
    for c in ans.values():
        n += not dec([(r["ci_lo"], r["ci_hi"]) for r in c["readings"] if r["n_boot"] == 50000])
    return n


def test_table_app_headline_diffs_prints_both_blocks_and_marks_every_undecided_cell(gen, tmp_path):
    """REQUIREMENT MOVED (INBOX_FOR_PAPER_LANE section 15, the user's request of 2026-09-25): the
    differences, their intervals and the degree marks left Table 2 for the appendix copy
    (tab:app-headline-diffs). This test asked the same of table_headline until then."""
    _need(gen.SEED_IDENTITY, gen.REGIMES, gen.ANSWER_F1)
    gen.table_app_headline_diffs(tmp_path)
    tex = (tmp_path / "table_app_headline_diffs.tex").read_text()
    prov = json.loads((tmp_path / "table_app_headline_diffs.provenance.json").read_text())
    printed = [ln for ln in tex.splitlines() if not ln.lstrip().startswith("%")]
    body = "\n".join(printed)
    assert "MuSiQue & StrategyQA & 2WikiMultiHopQA" in body
    assert "equal spend" in body.lower() and "own stop" in body.lower()
    assert body.count(r"$^{\circ}$") == _headline_expected_undecided(gen) == 9
    assert prov["decided_cells"] == 27 - 9
    # the answer row is the pooled record's 10k reading, printed "point [lo, hi]" on one line, in
    # percentage points (the user's comment, 2026-09-24; was the 0.X scale)
    ans = json.loads(gen.ANSWER_F1.read_text())["cells"]["s1s2"]
    row = next(ln for ln in printed if ln.startswith("answer token F1 &"))
    for s in ("musique", "strategyqa", "wiki2"):
        c = ans[s]
        a, lo, hi = (_pp(gen, c[k]) for k in ("delta", "ci_lo", "ci_hi"))
        assert f"${a}$ {{\\scriptsize $[{lo}, {hi}]$}}" in row
    # the equal-spend coverage row is Table 1's pooled cell, in points
    t1 = json.loads(gen.SEED_IDENTITY.read_text())["cells"]["s1s2"]
    ten = next(r for r in t1["evidence_coverage::musique"] if r["n_boot"] == 10000)
    cov = next(ln for ln in printed if ln.startswith("required-evidence coverage &"))
    assert f"${_pp(gen, ten['delta'])}$" in cov
    # one line per row: no row carries a line break inside a cell
    assert r"\newline" not in body and r"\\[" not in body.replace(r"\\[1pt]", "")
    note = next(ln for ln in tex.splitlines() if ln.startswith("% note: measured against"))
    assert r"\footnotesize" in note


def test_table_headline_own_stop_block_reads_as_sec4_defines_it(gen, tmp_path):
    """The main-text owner, 2026-09-24 night: sec 4 now defines the own stop as "lets each policy decide
    for itself when to stop, with at most eight calls", and the main text no longer says "ceiling".

    REQUIREMENT CHANGED 2026-09-25 (INBOX_FOR_PAPER_LANE section 15): the two block headers asserted
    here moved with the differences to the appendix copy (table_app_headline_diffs), unchanged; Table 2
    opens two blocks (the compact layout the main-text owner chose), and its own-stop block label is
    "Own stop (at most eight calls)"."""
    _need(gen.SEED_IDENTITY, gen.REGIMES, gen.ANSWER_F1)
    gen.table_app_headline_diffs(tmp_path)
    tex = (tmp_path / "table_app_headline_diffs.tex").read_text()
    heads = re.findall(r"\\multicolumn\{4\}\{@\{\}l\}\{([^}]*)\} \\\\", tex)
    assert heads == [
        "Equal spend: both arms at the lower of their two question counts",
        "Own stop: each arm stops itself, with at most eight calls",
    ], heads
    assert "ceiling" not in tex.lower()
    gen.table_headline(tmp_path)
    main = (tmp_path / "table_headline.tex").read_text()
    labels = re.findall(r"\\multicolumn\{7\}\{@\{\}l\}\{\\textit\{([^}]*)\}\} \\\\", main)
    assert labels == [blk for blk, _rows in HEADLINE_LAYOUT], labels
    assert "Own stop (at most eight calls)" in labels
    assert "ceiling" not in main.lower()


# Table 2 at bc1035f (HEAD when the redesign was asked for, INBOX_FOR_PAPER_LANE section 15) carried two
# small sub-rows. The appendix copy of its differences prints every other line of it, and not these.
HEADLINE_SUBROWS_DROPPED = (
    r"{\scriptsize levels: trained / same} & {\scriptsize 89.5 / 78.3} & {\scriptsize 85.5 / 78.4} "
    r"& {\scriptsize 93.1 / 88.2} \\",
    r"{\scriptsize questions: trained / same / 120B} & {\scriptsize 4.2 / 5.9 / 5.3} & "
    r"{\scriptsize 3.0 / 6.0 / 6.1} & {\scriptsize 2.3 / 3.0 / 3.1} \\",
)
HEADLINE_DIFFS_FROM = "bc1035f"


def test_table_app_headline_diffs_is_bc1035f_table_2_without_its_two_sub_rows(gen, tmp_path):
    """REPLACES test_table_headline_keeps_two_sub_rows_and_every_other_row_of_e8aac18, whose
    requirement the user changed (INBOX_FOR_PAPER_LANE section 15): Table 2 prints levels now, and the
    difference-and-interval table it was moves to the appendix WITHOUT its two sub-rows, since the
    levels live in the main table. Every printed line of the appendix copy is byte-identical to Table
    2 as committed at bc1035f (git show), those two sub-rows removed; so every printed number, interval,
    degree mark and dagger is the string the paper printed. Both assets' provenance still carries both
    levels of every cell, equal to e8aac18's."""
    _need(gen.SEED_IDENTITY, gen.REGIMES, gen.ANSWER_F1)
    gen.table_app_headline_diffs(tmp_path)
    tex = (tmp_path / "table_app_headline_diffs.tex").read_text()
    printed = [ln for ln in tex.splitlines() if not ln.lstrip().startswith("%")]
    assert not [ln for ln in printed if ln.startswith(r"{\scriptsize")]
    was = _git_show(HEADLINE_DIFFS_FROM, "paper/iclr2027 2/figures/table_headline.tex")
    if was is None:
        pytest.skip("no git history on this checkout")
    old = [ln for ln in was.splitlines() if not ln.lstrip().startswith("%")]
    assert [ln for ln in old if ln.startswith(r"{\scriptsize")] == list(HEADLINE_SUBROWS_DROPPED)
    assert printed == [ln for ln in old if ln not in HEADLINE_SUBROWS_DROPPED]
    gen.table_headline(tmp_path)
    was_prov = _git_show("e8aac18", "paper/iclr2027 2/figures/table_headline.provenance.json")
    for name in ("table_headline", "table_app_headline_diffs"):
        cells = json.loads((tmp_path / f"{name}.provenance.json").read_text())["cells"]
        assert cells == json.loads(was_prov)["cells"], name
        assert all(len(c["levels"]) == 2 for c in cells.values())


# ------------------------------------------------- Table 2 redesigned: levels side by side (sec 15)

# Table 2's layout, written out here independently of the generator: the compact layout the main-text
# owner chose on 2026-09-25 from the two offered (INBOX_FOR_PAPER_LANE section 15). Two blocks, each
# opened by an italic label spanning the table, then its rows as (printed label, row id). A row that
# names GPT-OSS-120B compares against it: its Baseline cell is the 120B model's level. A row id is a
# provenance cell key without its suite, or, for the question counts, "<regime>::<comparator>::asks".
HEADLINE_LAYOUT = (
    (
        "Equal spend",
        (
            (r"Required-evidence coverage (\%)", "equal::same::evidence_coverage"),
            (r"Depth-weighted recall (\%)", "equal::same::dwr"),
            ("Deepest need resolved (depth)", "equal::same::max_depth_reached"),
            ("Breadth (lines)", "equal::same::facet_breadth_scorer"),
            (r"Out-of-order rate (\%) $\downarrow$", "equal::same::precedence_violation_rate"),
            (r"Coverage vs GPT-OSS-120B (\%)", "equal::teacher::evidence_coverage"),
        ),
    ),
    (
        "Own stop (at most eight calls)",
        (
            (r"Required-evidence coverage (\%)", "own::same::evidence_coverage"),
            (r"Answer token F1 (\%)", "own::same::answer_token_f1"),
            ("Questions per task", "own::same::asks"),
            (r"Coverage vs GPT-OSS-120B (\%)", "own::teacher::evidence_coverage"),
            ("Questions vs GPT-OSS-120B", "own::teacher::asks"),
        ),
    ),
)
HEADLINE_LEVEL_HEADER = (
    r"\toprule",
    r" & \multicolumn{2}{c}{MuSiQue} & \multicolumn{2}{c}{StrategyQA} & "
    r"\multicolumn{2}{c}{2WikiMultiHopQA} \\",
    r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(l){6-7}",
    r" & Baseline & Q\&D & Baseline & Q\&D & Baseline & Q\&D \\",
)
# The bold placements, derived by hand from table_headline.provenance.json at bc1035f (the caller's
# derivation, INBOX_FOR_PAPER_LANE section 15): per row id, per suite (MuSiQue, StrategyQA,
# 2WikiMultiHopQA), the bold cell, "qd" (Q&D), "base" (Baseline) or None. Bold marks the better level
# only where the cell is decided; lower is better on the out-of-order row; question counts never.
HEADLINE_BOLD = {
    "equal::same::evidence_coverage": ("qd", "qd", "qd"),
    "equal::same::dwr": ("qd", "qd", "qd"),
    "equal::same::max_depth_reached": ("qd", "qd", None),
    "equal::same::facet_breadth_scorer": ("qd", "qd", "qd"),
    "equal::same::precedence_violation_rate": ("base", None, None),
    "equal::teacher::evidence_coverage": ("qd", "qd", "base"),
    "own::same::evidence_coverage": ("qd", None, None),
    "own::same::answer_token_f1": (None, None, None),
    "own::same::asks": (None, None, None),
    "own::teacher::evidence_coverage": ("qd", "base", None),
    "own::teacher::asks": (None, None, None),
}
SUITES3 = ("musique", "strategyqa", "wiki2")


def _unbold(cell: str) -> tuple[str, bool]:
    """A printed cell without its bold wrapper, and whether it had one: \\textbf{x} for a text cell,
    $\\boldsymbol{x}$ for a math cell (whose sign and point \\mathbf would leave light)."""
    if cell.startswith(r"\textbf{") and cell.endswith("}"):
        return cell[len(r"\textbf{") : -1], True
    if cell.startswith(r"$\boldsymbol{") and cell.endswith("}$"):
        return "$" + cell[len(r"$\boldsymbol{") : -2] + "$", True
    return cell, False


def _tabular_columns(spec_line: str) -> str:
    """The column letters of a \\begin{tabular}{...} line: @{...} separators dropped, a fixed-width
    w{c}{...} column read as c."""
    spec = spec_line[len(r"\begin{tabular}{") : -1]
    spec = re.sub(r"@\{[^{}]*(\{[^{}]*\})?[^{}]*\}", "", spec)
    return re.sub(r"w\{([lcr])\}\{[^{}]*\}", r"\1", spec)


def _level_rows(tex: str) -> dict[tuple[str, str], list[str]]:
    """Table 2's data rows keyed (block label, row label), each its six printed cells."""
    rows: dict[tuple[str, str], list[str]] = {}
    block = None
    for ln in tex.splitlines():
        m = re.fullmatch(r"\\multicolumn\{7\}\{@\{\}l\}\{\\textit\{([^}]*)\}\} \\\\", ln)
        if m:
            block = m.group(1)
        elif block and ln.endswith(r" \\") and " & " in ln:
            label, *cells = ln[: -len(r" \\")].split(" & ")
            assert (block, label) not in rows, (block, label)
            rows[(block, label)] = cells
    return rows


def _expected_level(gen, prov: dict, row: str, suite: str) -> tuple[str, str]:
    """(Baseline, Q&D) as Table 2 must print them, from the provenance levels (Baseline = levels[1],
    Q&D = levels[0]): proportions in percent via the exact-decimal 3-place print with its point moved,
    depth and breadth at two decimals, question counts at one."""
    regime, comp, metric = row.split("::")
    if metric == "asks":
        asks = prov["asks_trained_same_teacher"][suite]
        base = asks["base" if comp == "same" else "teacher"]
        return gen._fmt(base, 1), gen._fmt(asks["recipe"], 1)
    qd, base = prov["cells"][f"{row}::{suite}"]["levels"]
    if metric in HEADLINE_POINTS_METRICS:
        return _pp(gen, base, False), _pp(gen, qd, False)
    assert metric in ("max_depth_reached", "facet_breadth_scorer"), metric
    return gen._fmt(base, 2), gen._fmt(qd, 2)


def test_table_headline_prints_levels_side_by_side_in_two_blocks(gen, tmp_path):
    """INBOX_FOR_PAPER_LANE section 15, in the compact layout the main-text owner chose (2026-09-25):
    six numeric columns (Baseline and Q&D under each suite), two header rows with a rule under each
    suite, two blocks each opened by a spanning italic label, the row labels in order, and the rows set
    at \\arraystretch 0.9 inside a group that ends with the table.

    REPLACES test_table_headline_prints_levels_side_by_side_in_four_blocks: the four-block layout was
    the other of the two offered, and was not chosen."""
    _need(gen.SEED_IDENTITY, gen.REGIMES, gen.ANSWER_F1)
    gen.table_headline(tmp_path)
    tex = (tmp_path / "table_headline.tex").read_text()
    printed = [ln for ln in tex.splitlines() if not ln.lstrip().startswith("%")]
    spec = next(ln for ln in printed if ln.startswith(r"\begin{tabular}{"))
    assert _tabular_columns(spec) == "l" + "c" * 6, spec
    i = printed.index(spec)
    # the stretch is scoped: opened in a group just before the tabular, closed just after it
    assert printed[:i] == [r"{\renewcommand{\arraystretch}{0.9}%"], printed[:i]
    assert printed[-1] == r"\end{tabular}}%", printed[-1]
    n = len(HEADLINE_LEVEL_HEADER)
    assert tuple(printed[i + 1 : i + 1 + n]) == HEADLINE_LEVEL_HEADER, printed[i + 1 : i + 1 + n]
    assert sum(ln.endswith(r"\\") for ln in printed[i + 1 : i + 1 + n]) == 2
    assert printed[i + 1 + n] == r"\midrule"
    labels = re.findall(r"\\multicolumn\{7\}\{@\{\}l\}\{\\textit\{([^}]*)\}\}", tex)
    assert labels == ["Equal spend", "Own stop (at most eight calls)"], labels
    rows = _level_rows(tex)
    assert [label for _blk, label in rows] == [
        r"Required-evidence coverage (\%)",
        r"Depth-weighted recall (\%)",
        "Deepest need resolved (depth)",
        "Breadth (lines)",
        r"Out-of-order rate (\%) $\downarrow$",
        r"Coverage vs GPT-OSS-120B (\%)",
        r"Required-evidence coverage (\%)",
        r"Answer token F1 (\%)",
        "Questions per task",
        r"Coverage vs GPT-OSS-120B (\%)",
        "Questions vs GPT-OSS-120B",
    ]
    want = [(blk, label) for blk, rs in HEADLINE_LAYOUT for label, _ in rs]
    assert list(rows) == want, list(rows)
    assert all(len(cells) == 6 for cells in rows.values())
    # every printed line is the header, a block label, a data row or a rule
    body = printed[i + 1 + n :]
    assert len([ln for ln in body if ln.endswith(r"\\")]) == 2 + len(want) == 13


def test_table_headline_prints_every_level_as_its_provenance_level(gen, tmp_path):
    """Every printed cell is the pair's level from the provenance, Baseline = levels[1] and Q&D =
    levels[0], at the precision the user asked for; the `printed` map carries each printed string and
    its bold flag, and the tex is that map. The first row reads 78.3 / 89.5 on MuSiQue, the level line
    Table 2 printed before as "89.5 / 78.3"."""
    _need(gen.SEED_IDENTITY, gen.REGIMES, gen.ANSWER_F1)
    gen.table_headline(tmp_path)
    tex = (tmp_path / "table_headline.tex").read_text()
    prov = json.loads((tmp_path / "table_headline.provenance.json").read_text())
    rows = _level_rows(tex)
    seen = set()
    for blk, rs in HEADLINE_LAYOUT:
        for label, row in rs:
            cells = rows[(blk, label)]
            for j, suite in enumerate(SUITES3):
                base, qd = (_unbold(c) for c in cells[2 * j : 2 * j + 2])
                assert (base[0], qd[0]) == _expected_level(gen, prov, row, suite), (row, suite)
                assert prov["printed"][f"{row}::{suite}"] == {
                    "Baseline": {"text": base[0], "bold": base[1]},
                    "Q&D": {"text": qd[0], "bold": qd[1]},
                }, (row, suite)
                seen.add(f"{row}::{suite}")
    assert set(prov["printed"]) == seen and len(seen) == 33
    first = rows[(HEADLINE_LAYOUT[0][0], HEADLINE_LAYOUT[0][1][0][0])]
    assert [_unbold(c)[0] for c in first] == ["78.3", "89.5", "78.4", "85.5", "88.2", "93.1"]
    # the question counts are the ones bc1035f printed as "trained / same / 120B"
    q_same = rows[(HEADLINE_LAYOUT[1][0], "Questions per task")]
    q_120b = rows[(HEADLINE_LAYOUT[1][0], "Questions vs GPT-OSS-120B")]
    assert q_same == ["5.9", "4.2", "6.0", "3.0", "3.0", "2.3"]
    assert q_120b == ["5.3", "4.2", "6.1", "3.0", "3.1", "2.3"]
    # A row that names GPT-OSS-120B is exactly a row against it, and its Baseline cell is the 120B
    # model's level, read here from the source records themselves: the equal-spend level from
    # teacher_by_seed.json (the 10k reading's level_b), the own-stop level from
    # teacher_ownstop_by_seed.json (level_teacher), the question count from regimes.json (teacher).
    tb = json.loads(gen.TEACHER_BY_SEED.read_text())["cells"]["s1s2"]
    tos = json.loads(gen.TEACHER_OWNSTOP.read_text())["cells"]["s1s2"]["evidence_coverage"]
    asks = json.loads(gen.REGIMES.read_text())["asks"]
    teacher = {
        "equal::teacher::evidence_coverage": lambda s: _pp(
            gen, next(r for r in tb[s] if r["n_boot"] == 10000)["level_b"], False
        ),
        "own::teacher::evidence_coverage": lambda s: _pp(gen, tos[s]["level_teacher"], False),
        "own::teacher::asks": lambda s: gen._fmt(asks[s]["teacher"], 1),
    }
    named = [(blk, label, row) for blk, rs in HEADLINE_LAYOUT for label, row in rs]
    assert {row for _b, label, row in named if "GPT-OSS-120B" in label} == set(teacher)
    assert {row for _b, _l, row in named if row.split("::")[1] == "teacher"} == set(teacher)
    for blk, label, row in named:
        if row in teacher:
            base = [_unbold(c)[0] for c in rows[(blk, label)][0::2]]
            assert base == [teacher[row](s) for s in SUITES3], (row, base)


def test_table_headline_bolds_the_better_level_only_where_decided(gen, tmp_path):
    """Bold lands on the better level of a pair only where the record decided the cell (all three
    50,000-resample intervals exclude zero); better is higher but on the out-of-order row, where it is
    lower; question counts are never bold. The placements are pinned literals (HEADLINE_BOLD), and each
    agrees with the provenance's own decided flag and levels."""
    _need(gen.SEED_IDENTITY, gen.REGIMES, gen.ANSWER_F1)
    gen.table_headline(tmp_path)
    tex = (tmp_path / "table_headline.tex").read_text()
    prov = json.loads((tmp_path / "table_headline.provenance.json").read_text())
    rows = _level_rows(tex)
    got = {}
    for blk, rs in HEADLINE_LAYOUT:
        for label, row in rs:
            cells = rows[(blk, label)]
            marks = []
            for j in range(3):
                base, qd = (_unbold(c)[1] for c in cells[2 * j : 2 * j + 2])
                assert not (base and qd)
                marks.append("base" if base else "qd" if qd else None)
            got[row] = tuple(marks)
    assert got == HEADLINE_BOLD
    # the pins against the record: bold exactly where decided, on the side the levels favour
    for row, marks in HEADLINE_BOLD.items():
        if row.endswith("::asks"):
            continue
        for suite, mark in zip(SUITES3, marks):
            c = prov["cells"][f"{row}::{suite}"]
            assert (mark is not None) is c["decided"], (row, suite)
            if c["decided"]:
                qd, base = c["levels"]
                lower = row.endswith("precedence_violation_rate")
                assert mark == ("qd" if (qd > base) != lower else "base"), (row, suite)
    # the three cells where the baseline wins, read off the page
    assert rows[(HEADLINE_LAYOUT[0][0], r"Out-of-order rate (\%) $\downarrow$")][:2] == [
        r"\textbf{17.1}",
        "22.8",
    ]
    assert rows[(HEADLINE_LAYOUT[0][0], r"Coverage vs GPT-OSS-120B (\%)")][4:] == [
        r"\textbf{92.4}",
        "88.9",
    ]
    assert rows[(HEADLINE_LAYOUT[1][0], r"Coverage vs GPT-OSS-120B (\%)")][2:4] == [
        r"\textbf{90.0}",
        "86.4",
    ]


def test_table_headline_prints_no_markers_and_no_intervals(gen, tmp_path):
    """The main table prints levels only: no degree marks, no daggers, no intervals, no signed
    differences and no small sub-rows (INBOX_FOR_PAPER_LANE section 15). Those live in
    table_app_headline_diffs."""
    _need(gen.SEED_IDENTITY, gen.REGIMES, gen.ANSWER_F1)
    gen.table_headline(tmp_path)
    tex = (tmp_path / "table_headline.tex").read_text()
    body = "\n".join(ln for ln in tex.splitlines() if not ln.lstrip().startswith("%"))
    for banned in (r"^{\circ}", r"\circ", r"\dagger", "[", r"\scriptsize", "$+", "$-", "levels:"):
        assert banned not in body, banned
    for cells in _level_rows(tex).values():
        for c in cells:
            assert re.fullmatch(r"\d+\.\d+", _unbold(c)[0]), c


def test_table_headline_refuses_an_answer_record_that_no_longer_reproduces_the_paper(
    gen, tmp_path, monkeypatch
):
    """THE NON-VACUITY CASE: the answer row is written only while its reader still reproduces the
    per-seed values section 5.5 prints."""
    _need(gen.ANSWER_F1)
    rec = json.loads(gen.ANSWER_F1.read_text())
    rec["lock"]["L1_printed_per_seed"]["s1::musique"]["printed"] = "+0.032"
    bad = tmp_path / "answer_f1_by_seed.json"
    bad.write_text(json.dumps(rec))
    monkeypatch.setattr(gen, "ANSWER_F1", bad)
    with pytest.raises(gen.Mismatch) as exc:
        gen.table_headline(tmp_path / "out")
    assert "NOT WRITING" in str(exc.value)


def test_table_headline_refuses_a_decided_flag_its_bounds_do_not_support(
    gen, tmp_path, monkeypatch
):
    _need(gen.ANSWER_F1)
    rec = json.loads(gen.ANSWER_F1.read_text())
    rec["cells"]["s1s2"]["wiki2"]["decided"] = True
    bad = tmp_path / "answer_f1_by_seed.json"
    bad.write_text(json.dumps(rec))
    monkeypatch.setattr(gen, "ANSWER_F1", bad)
    with pytest.raises(gen.Mismatch):
        gen.table_headline(tmp_path / "out")


def test_table_headline_refuses_an_answer_row_read_on_another_population(
    gen, tmp_path, monkeypatch
):
    """The answer row and the own-stop coverage row must stand on one population: the answer
    reader's coverage lock must equal the regimes table's own-stop cell."""
    _need(gen.ANSWER_F1, gen.REGIMES)
    rec = json.loads(gen.ANSWER_F1.read_text())
    rec["lock"]["L2_own_stop_population"]["s1s2::strategyqa"]["point"] += 0.001
    bad = tmp_path / "answer_f1_by_seed.json"
    bad.write_text(json.dumps(rec))
    monkeypatch.setattr(gen, "ANSWER_F1", bad)
    with pytest.raises(gen.Mismatch):
        gen.table_headline(tmp_path / "out")


def test_table_seeds_prints_every_headline_row_per_seed(gen, tmp_path):
    """tab:app-seeds carries the per-seed readings the headline table no longer prints, for every
    one of its rows, and its pooled column is the headline's point."""
    _need(gen.SEED_IDENTITY, gen.REGIMES, gen.ANSWER_F1)
    gen.table_seeds(tmp_path)
    gen.table_headline(tmp_path)
    tex = (tmp_path / "table_seeds.tex").read_text()
    body = "\n".join(ln for ln in tex.splitlines() if not ln.lstrip().startswith("%"))
    assert "seed 1 & seed 2 & seeds 1+2" in body
    assert ("seed 0" in body) is gen.SHOW_SEED0
    for label in (
        "required-evidence coverage",
        "breadth",
        "depth-weighted recall",
        "deepest need resolved",
        "out-of-order rate",
        "coverage vs GPT-OSS-120B",
        "questions asked",
        "answer token F1",
    ):
        assert label in body, label
    seeds = json.loads((tmp_path / "table_seeds.provenance.json").read_text())["cells"]
    head = json.loads((tmp_path / "table_headline.provenance.json").read_text())["cells"]
    shared = [k for k in head if k in seeds]
    assert len(shared) == 27
    for k in shared:
        assert seeds[k]["s1s2"] == head[k]["delta"], k


# ------------------------------------------------- tab:mechanism, main-text version (G2)

# "question pairs" since 2026-09-24 (was "question pairs, pooled export", 9dfb71d): the main text
# no longer defines "pooled export", so the main table must not print it. The other labels since the
# user's round-2 review (the coordinator's redirect of 2026-09-24 night, after INBOX_FOR_PAPER_LANE
# section 14, item 3): sec 5.5 reads the table as what each stage of the training teaches.
MECHANISM_MAIN_ROWS = (
    "same model, prompted",
    "imitation (first stage)",
    "question pairs",
    "stop contrasts alone",
    "final stage, seed 1",
    "final stage, seed 2",
)
# The table as printed at e8aac18, one MuSiQue/StrategyQA pair per cell, by its label there. The
# restructured table prints each half in its own column and no number may move: every new cell is
# one half of one of these strings, character for character.
MECHANISM_MAIN_AT_E8AAC18 = {
    "same model, prompted": ("---", "---", "15.3/14.6", "96.4/96.3", "6.29/5.59"),
    "imitation reference": ("$+4.9$", "$+8.0$", "95.1/90.7", "86.8/83.0", "3.05/1.34"),
    "question pairs": ("$+16.7$", "$+8.4$", "98.7/98.0", "86.0/83.4", "2.47/1.28"),
    "stop contrasts only": ("$-0.5$", "$-1.7$", "10.1/9.7", "99.6/98.1", "6.95/5.75"),
    "final stage (both kinds), seed 1": (
        "$+15.7$",
        "$+8.2$",
        "42.7/54.6",
        "94.2/91.2",
        "4.23/2.13",
    ),
    "final stage (both kinds), seed 2": (
        "$+15.4$",
        "$+7.8$",
        "42.7/54.5",
        "94.5/90.6",
        "4.24/2.14",
    ),
}
# The header: two column groups, MuSiQue and StrategyQA, each of four columns (coverage gain in
# points, stop when done and ask when not done in percent, questions asked), a rule under each
# group's name, and exactly two header rows (the 9-page trim): the group names, then the four short
# column names per group, whose meaning and units the caption gives.
MECHANISM_MAIN_HEADER = (
    r"\toprule",
    r" & \multicolumn{4}{c}{MuSiQue} & \multicolumn{4}{c}{StrategyQA} \\",
    r"\cmidrule(lr){2-5}\cmidrule(l){6-9}",
    r"supervision & gain & stop\,$|$\,done & ask\,$|$\,not done & asks & gain & stop\,$|$\,done & "
    r"ask\,$|$\,not done & asks \\",
    r"\midrule",
)


def test_table_mechanism_main_prints_one_number_per_cell_in_two_suite_groups(gen, tmp_path):
    """The user's round 2 (coordinator's redirect, 2026-09-24 night): no "/" pairs. Two column groups,
    MuSiQue then StrategyQA, each of four columns with a rule under its name; the six rows under their
    new labels; and every number the same printed string as the half of the e8aac18 pair it came from
    (the gains were already one per suite).

    Since 2026-09-25 (INBOX_FOR_PAPER_LANE section 15) some cells carry a bold wrapper; the number
    inside it is compared here, and where the bold lands is pinned by
    test_table_mechanism_main_bolds_the_largest_printed_value_per_column."""
    _need(gen.BCA_CELLS, gen.APPENDIX_TRAINING, gen.DEV_SEED_GATE)
    gen.table_mechanism_main(tmp_path)
    tex = (tmp_path / "table_mechanism_main.tex").read_text()
    printed = [ln for ln in tex.splitlines() if not ln.lstrip().startswith("%")]
    spec = printed[0]
    assert spec.startswith(r"\begin{tabular}{"), spec
    cols = re.sub(r"@\{[^{}]*(\{[^{}]*\})?[^{}]*\}", "", spec[len(r"\begin{tabular}{") : -1])
    assert cols == "l" + "c" * 8, cols
    n = len(MECHANISM_MAIN_HEADER)
    assert tuple(printed[1 : 1 + n]) == MECHANISM_MAIN_HEADER, printed[1 : 1 + n]
    # exactly two header rows: the group names, then the column names
    assert sum(ln.endswith(r"\\") for ln in printed[1 : 1 + n]) == 2
    body = [ln for ln in printed if ln.split(" & ")[0] in MECHANISM_MAIN_ROWS]
    assert [ln.split(" & ")[0] for ln in body] == list(MECHANISM_MAIN_ROWS)
    assert len([ln for ln in printed if ln.endswith(r"\\")]) == 2 + len(MECHANISM_MAIN_ROWS)
    assert not any("/" in ln for ln in printed[1 + n :]), [ln for ln in printed if "/" in ln]
    old_labels = list(MECHANISM_MAIN_AT_E8AAC18)
    for ln, old in zip(body, old_labels):
        assert ln.endswith(r" \\") and ln.count(r"\\") == 1, ln
        cells = ln[: -len(r" \\")].split(" & ")[1:]
        assert len(cells) == 8, ln
        g_m, g_s, stop, ask, asks = MECHANISM_MAIN_AT_E8AAC18[old]
        halves = [x.split("/") for x in (stop, ask, asks)]
        want = [g_m, *(h[0] for h in halves), g_s, *(h[1] for h in halves)]
        assert [_unbold(c)[0] for c in cells] == want, (old, cells, want)
    # the literal pairs are the ones e8aac18 printed (read from git when the history is here)
    was = _git_show("e8aac18", "paper/iclr2027 2/figures/table_mechanism_main.tex")
    if was is not None:
        rows = {
            ln[: -len(r" \\")].split(" & ")[0]: tuple(ln[: -len(r" \\")].split(" & ")[1:])
            for ln in was.splitlines()
            if ln.split(" & ")[0] in MECHANISM_MAIN_AT_E8AAC18
        }
        assert rows == MECHANISM_MAIN_AT_E8AAC18, rows


def _git_show(rev: str, path: str) -> str | None:
    """A file as committed at `rev`, or None when this checkout carries no git history."""
    import subprocess

    try:
        return subprocess.run(
            ["git", "show", f"{rev}:{path}"],
            cwd=REPO,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None


def test_table_mechanism_main_is_six_point_rows_read_from_the_appendix_rows(gen, tmp_path):
    """At most six rows, one line each, points only; every value is the appendix version's value
    for the same row.

    REQUIREMENT CHANGED 2026-09-25 (INBOX_FOR_PAPER_LANE section 15): this test banned bold ("no
    bold"); the user now asks for the largest value per column in bold. The ban on \\mathbf and
    \\textbf is dropped here, and the bold cells are pinned, exactly, by
    test_table_mechanism_main_bolds_the_largest_printed_value_per_column."""
    _need(gen.BCA_CELLS, gen.APPENDIX_TRAINING, gen.DEV_SEED_GATE)
    gen.table2_mechanism(tmp_path)
    gen.table_mechanism_main(tmp_path)
    tex = (tmp_path / "table_mechanism_main.tex").read_text()
    printed = [ln for ln in tex.splitlines() if not ln.lstrip().startswith("%")]
    rows = [ln for ln in printed if ln.split(" & ")[0] in MECHANISM_MAIN_ROWS]
    assert [r.split(" & ")[0] for r in rows] == list(MECHANISM_MAIN_ROWS)
    body = "\n".join(printed)
    for banned in (r"\newline", r"\scriptsize $[", r"\\[", "pooled export"):
        assert banned not in body, banned
    assert all(r.endswith(r"\\") and r.count(r"\\") == 1 for r in rows)
    full = json.loads((tmp_path / "table2_mechanism.provenance.json").read_text())["values"]
    main = json.loads((tmp_path / "table_mechanism_main.provenance.json").read_text())["values"]
    assert main
    for label, cells in main.items():
        assert cells == full[gen.MECHANISM_MAIN_SOURCE[label]], label
    # the recipe's seed-1 row, read off the page, its gain in percentage points (the user's
    # comment, 2026-09-24; was the 0.X scale)
    s1 = json.loads((tmp_path / "table_mechanism_main.provenance.json").read_text())["values"][
        "final stage, seed 1"
    ]
    assert f"${_pp(gen, s1['musique']['coverage_delta_matched'])}$" in rows[4]


@pytest.mark.parametrize("fn", ["table_mechanism_main", "table2_mechanism"])
def test_mechanism_header_names_the_gate_rule_not_equal_spend(gen, tmp_path, fn):
    """The development cells are the selection gate's one-way matching (the comparator read at the
    trained arm's length or its own end), not the paper's equal spend (both at the lower count): on the
    same held-out runs the two read +0.128 and +0.112. Neither header may call them equal spend.

    The REQUIREMENT CHANGED for the main-text table on the user's round-2 comments (INBOX_FOR_PAPER_LANE
    section 14, item 3): its header "gate-matched gain" read as unclear, and its caption already says the
    coverage gain is against the same model, prompted, at a matched spend, so the header is now "coverage
    gain". The appendix table keeps "gate-matched", where the rule is spelled out."""
    _need(gen.BCA_CELLS, gen.APPENDIX_TRAINING, gen.DEV_SEED_GATE)
    gen.table2_mechanism(tmp_path)
    gen.table_mechanism_main(tmp_path)
    name = "table_mechanism_main" if fn == "table_mechanism_main" else "table2_mechanism"
    tex = (tmp_path / f"{name}.tex").read_text()
    body = "\n".join(ln for ln in tex.splitlines() if not ln.lstrip().startswith("%"))
    assert "equal spend" not in body
    if fn == "table_mechanism_main":
        # "gain" in each suite's group (MECHANISM_MAIN_HEADER); the caption names it coverage gain
        assert "supervision & gain & " in body and " & asks & gain & " in body
        assert "gate-matched" not in body
    else:
        assert "gate-matched" in body


def test_table_mechanism_main_refuses_when_the_appendix_row_moves(gen, tmp_path, monkeypatch):
    """THE NON-VACUITY CASE: the slim table stands on the same locks as the appendix one."""
    _need(gen.BCA_CELLS, gen.APPENDIX_TRAINING)
    text, _ = gen.read_tex_with_inputs(gen.APPENDIX_TRAINING)
    edited = text.replace("$+0.1604$ & $[+0.1225, +0.2088]$", "$+0.1604$ & $[+0.1325, +0.2088]$")
    assert edited != text
    fake = tmp_path / "appendix_training.tex"
    fake.write_text(edited)
    monkeypatch.setattr(gen, "APPENDIX_TRAINING", fake)
    with pytest.raises(gen.Mismatch):
        gen.table_mechanism_main(tmp_path / "out")


# Table 3's bold cells (INBOX_FOR_PAPER_LANE section 15): per suite and column, the rows printing the
# largest value across all six rows, read off Table 3 as committed at bc1035f (the derivation below
# recomputes these from that file). The same model's "---" gain takes no part; asks is never bold.
MECHANISM_MAIN_BOLD = {
    ("musique", "gain"): ("question pairs",),
    ("musique", "stop"): ("question pairs",),
    ("musique", "ask"): ("stop contrasts alone",),
    ("strategyqa", "gain"): ("question pairs",),
    ("strategyqa", "stop"): ("question pairs",),
    ("strategyqa", "ask"): ("stop contrasts alone",),
}
# The eight cells of a row, in order: (suite, column), None for the never-bold asks.
MECHANISM_MAIN_CELLS = (
    ("musique", "gain"),
    ("musique", "stop"),
    ("musique", "ask"),
    None,
    ("strategyqa", "gain"),
    ("strategyqa", "stop"),
    ("strategyqa", "ask"),
    None,
)


def _mechanism_cells(tex: str) -> dict[str, list[str]]:
    return {
        ln[: -len(r" \\")].split(" & ")[0]: ln[: -len(r" \\")].split(" & ")[1:]
        for ln in tex.splitlines()
        if ln.split(" & ")[0] in MECHANISM_MAIN_ROWS
    }


def test_table_mechanism_main_bolds_the_largest_printed_value_per_column(gen, tmp_path):
    """Bold is the largest PRINTED value in each gain, stop | done and ask | not done column, per
    suite, across the six rows; ties would all be bold; the "---" gain and every asks cell are never
    bold. The placements are pinned (MECHANISM_MAIN_BOLD) and rederived here from Table 3 as bc1035f
    printed it; the numbers inside the bold are that table's strings, unchanged. The provenance names
    the rule and says it is not a tested difference."""
    from decimal import Decimal

    _need(gen.BCA_CELLS, gen.APPENDIX_TRAINING, gen.DEV_SEED_GATE)
    gen.table_mechanism_main(tmp_path)
    tex = (tmp_path / "table_mechanism_main.tex").read_text()
    rows = _mechanism_cells(tex)
    assert list(rows) == list(MECHANISM_MAIN_ROWS)
    got: dict[tuple[str, str], tuple[str, ...]] = {}
    for label, cells in rows.items():
        for where, c in zip(MECHANISM_MAIN_CELLS, cells):
            text, bold = _unbold(c)
            if bold:
                assert where is not None and text != "---", (label, c)
                got[where] = (*got.get(where, ()), label)
    assert got == MECHANISM_MAIN_BOLD
    # a math cell is bold whole, sign and point included; a text cell through \textbf
    assert rows["question pairs"][0] == r"$\boldsymbol{+16.7}$"
    assert rows["question pairs"][1] == r"\textbf{98.7}"
    assert rows["stop contrasts alone"][2] == r"\textbf{99.6}"
    was = _git_show(HEADLINE_DIFFS_FROM, "paper/iclr2027 2/figures/table_mechanism_main.tex")
    if was is None:
        pytest.skip("no git history on this checkout")
    old = _mechanism_cells(was)
    assert {k: [_unbold(c)[0] for c in v] for k, v in rows.items()} == old
    derived: dict[tuple[str, str], tuple[str, ...]] = {}
    for j, where in enumerate(MECHANISM_MAIN_CELLS):
        if where is None:
            continue
        vals = {k: v[j] for k, v in old.items() if v[j] != "---"}
        top = max(Decimal(s.strip("$")) for s in vals.values())
        derived[where] = tuple(k for k, s in vals.items() if Decimal(s.strip("$")) == top)
    assert derived == MECHANISM_MAIN_BOLD
    prov = json.loads((tmp_path / "table_mechanism_main.provenance.json").read_text())
    assert (
        "bold = largest printed value in the column, not a tested difference"
        in (prov["extra"]["rules"]["bold"])
    )
    assert prov["extra"]["bold_cells"] == {
        f"{s}::{col}": list(labels) for (s, col), labels in MECHANISM_MAIN_BOLD.items()
    }


def test_table_mechanism_main_bolds_every_tied_maximum(gen, tmp_path, monkeypatch):
    """THE TIE CASE, absent from the data: ties are decided on the printed strings, and every tied
    maximum is bold. Seed 2's MuSiQue stop | done is set to 0.987, which prints 98.7 as the question
    pairs' does, so both are bold, and the provenance names both."""
    _need(gen.BCA_CELLS, gen.APPENDIX_TRAINING, gen.DEV_SEED_GATE)
    real = gen._dev_seed_rows

    def tied(seed0):
        rows = []
        for label, suites in real(seed0):
            if label == "both kinds, from pairs, seed 2":
                suites = {**suites, "musique": {**suites["musique"], "p_stop_given_done": 0.987}}
            rows.append((label, suites))
        return rows

    monkeypatch.setattr(gen, "_dev_seed_rows", tied)
    gen.table_mechanism_main(tmp_path)
    rows = _mechanism_cells((tmp_path / "table_mechanism_main.tex").read_text())
    assert rows["question pairs"][1] == r"\textbf{98.7}"
    assert rows["final stage, seed 2"][1] == r"\textbf{98.7}"
    assert [lb for lb, v in rows.items() if _unbold(v[1])[1]] == [
        "question pairs",
        "final stage, seed 2",
    ]
    prov = json.loads((tmp_path / "table_mechanism_main.provenance.json").read_text())
    assert prov["extra"]["bold_cells"]["musique::stop"] == [
        "question pairs",
        "final stage, seed 2",
    ]


# ------------------------------------------------- reader terms in the main figures (G3, G4)

# The terms the main text uses for the arms (the main-text owner's G3 list), and the older names
# they replace, which must not be printed in a main figure any more.
ARM_TERMS = (
    "the trained questioner",
    "the same model, prompted",
    "GPT-OSS-120B, prompted",
    "Claude Opus 5, prompted",
)
RETIRED_TERMS = (
    "gpt-oss-120b",
    "same weights",
    "15x",
    "(120B)",
    "(8B)",
    "8B base",
    "8B recipe",
    "recipe",
    "seeds 1",
    "multi-hop",
    "implicit reasoning",
    "implicit-reasoning",
    # the tool-use prompt variants: the appendix tables name them base and stop
    "prompt A",
    "prompt B",
)


def _rendered(gen, tmp_path, fn, asset):
    fn(tmp_path)
    prov = json.loads((tmp_path / f"{asset}.provenance.json").read_text())
    text = prov["rendered_text"]
    assert text, f"{asset}: no rendered text recorded"
    for t in text:
        assert not any(r in t for r in RETIRED_TERMS), f"{asset}: retired term in {t!r}"
    return prov, " | ".join(text)


def test_fig2_recipe_names_the_arms_and_the_datasets_in_the_papers_terms(gen, tmp_path):
    _need(gen.FRONTIER_RECIPE)
    prov, text = _rendered(gen, tmp_path, gen.fig2_frontier_recipe, "fig2_frontier_recipe")
    for term in ARM_TERMS[:3] + ("MuSiQue", "StrategyQA"):
        assert term in text, term


def test_fig2_recipe_sets_the_frames_label_beside_the_frames_panel(gen, tmp_path):
    """FRAMES reads a different quantity; its y-label sits on the outer side of its own panel, not
    in the gap it shares with StrategyQA."""
    _need(gen.FRONTIER_RECIPE, gen.FRAMES_RECIPE_JSON)
    gen.fig2_frontier_recipe(tmp_path)
    geo = json.loads((tmp_path / "fig2_frontier_recipe.provenance.json").read_text())["extra"][
        "frames_ylabel_geometry"
    ]
    # in percent since the user's comment of 2026-09-24 (was "answer correctness", a fraction)
    assert geo["text"] == "answer correctness (%)"
    assert geo["label_x0"] >= geo["frames_axes_x1"]
    assert geo["label_x0"] > geo["strategyqa_axes_x1"]


def test_fig11_names_the_arms_in_the_papers_terms(gen, tmp_path):
    _need(gen.TWO_AXIS_RECIPE)
    _, text = _rendered(gen, tmp_path, gen.fig11_axes_separable, "fig11_axes_separable")
    for term in ARM_TERMS[:2]:
        assert term in text, term


def test_fig12_names_every_comparator_in_the_papers_terms(gen, tmp_path):
    _need(gen.COMPARATOR_DIR)
    _, text = _rendered(gen, tmp_path, gen.fig12_comparators, "fig12_comparators")
    for term in ARM_TERMS + ("MuSiQue", "StrategyQA"):
        assert term in text, term


def test_fig12_frames_panel_reads_answer_correctness_at_own_stop(gen, tmp_path):
    """FRAMES is gold-free: its panel reads answer correctness at each policy's own stop under the
    shared ceiling of 8, a different estimand from the coverage panels, and must say so. Every drawn
    cell equals the record's 10k contrast for the named comparator."""
    _need(gen.FRAMES_STRUCTURED)
    prov, text = _rendered(gen, tmp_path, gen.fig12_comparators, "fig12_comparators")
    assert "FRAMES" in text
    assert "answer correctness" in text and "own stop" in text
    rec = json.loads(gen.FRAMES_STRUCTURED.read_text())
    frames = [c for c in prov["values"] if c["suite"] == "frames"]
    assert len(frames) == 6
    for c in frames:
        want = rec["reading"]["contrasts"][c["contrast"]]
        assert (c["delta"], c["ci"][0], c["ci"][1]) == (want["delta"], want["ci_lo"], want["ci_hi"])
        assert c["verdict"] == ("DECIDED" if want["decided"] else "SPANS_ZERO")


@pytest.mark.parametrize("breakage", ["lock", "population"])
def test_fig12_refuses_a_frames_record_whose_checks_failed(gen, tmp_path, monkeypatch, breakage):
    """A FRAMES record whose lock or population check failed must not be drawn."""
    rec = {
        "lock": {"failures": ["levels.recipe.8: y differs"] if breakage == "lock" else []},
        "population_failures": ["par2_8b/cap8: 823 ok"] if breakage == "population" else [],
        "reading": {"contrasts": {}},
    }
    bad = tmp_path / "frames_structured.json"
    bad.write_text(json.dumps(rec))
    monkeypatch.setattr(gen, "FRAMES_STRUCTURED", bad)
    with pytest.raises(gen.Mismatch):
        gen._frames_comparator_cells()


def test_fig13_prints_no_levels_no_suptitle_and_no_questions_panel(gen, tmp_path, monkeypatch):
    """The REQUIREMENT CHANGED on the user's round-2 comments (INBOX_FOR_PAPER_LANE section 14, item 2;
    this replaces the test that required the level pairs "13 → 34" beside each point). The prose gives
    the levels, so the figure prints none and has no suptitle. The questions-per-dialogue panel is
    dropped (that result stays in the prose), and nothing is drawn per successful task: dividing by
    success double-counts the success gain, and per-dialogue turns is the declared measure. Each panel's
    x-axis names its quantity and unit, and the arms are named in the paper's terms."""
    _need(gen.TAU2_READER, gen.TAU2_READER_120B)
    fig, prov = _captured(gen, monkeypatch, gen.fig13_tau2, tmp_path)
    _no_retired(prov)
    text = " | ".join(prov["rendered_text"])
    assert "→" not in text and not re.search(r"\d+\s*(→|->|to)\s*\d+", text)
    assert "printed_levels" not in prov["extra"]
    assert fig._suptitle is None or not fig._suptitle.get_text().strip()
    assert not re.search(r"\bquestions\b", text), text
    assert "per success" not in text and "successful" not in text
    for term in ARM_TERMS[:3]:
        assert term in text, term
    # short page (at most 110bp): each panel's x-axis label names its quantity and unit, no titles
    left, right = (ax.get_xlabel() for ax in fig.axes)
    assert "task success" in left and "percentage points" in left, left
    assert "follow-up turns" in right and "per dialogue" in right, right


def test_fig8_recipe_names_the_datasets(gen, tmp_path):
    """The controls figure (appendix) names MuSiQue and StrategyQA, not the suite descriptions."""
    _need(gen.CONTROLS_RECIPE_DIR)
    _, text = _rendered(gen, tmp_path, gen.fig8_controls_recipe, "fig8_controls_recipe")
    assert "MuSiQue" in text and "StrategyQA" in text


# ------------------------------------------------- rounding (main-text owner, 2026-09-23)


def test_printed_values_round_half_up_on_the_exact_decimal(gen):
    """0.2175 is exactly 87/400 and prints +0.218; binary float formatting prints +0.217 because
    the nearest double is 0.21749999999999999. Every printed table value goes through one
    formatter that rounds the decimal the value reads as, half up."""
    assert gen._fmt3(0.2175) == "+0.218"
    assert gen._fmt3(0.0975) == "+0.098"
    assert gen._fmt3(-0.0094666) == "-0.009"
    assert gen._fmt3(-0.0005) == "-0.001"
    assert gen._fmt3(0.0) == "+0.000"
    assert gen._fmt(4.215, 2) == "4.22"
    assert gen._fmt(4.175, 2) == "4.18"
    assert gen._fmt(3.04875, 1) == "3.0"
    assert gen._fmt(0.1455026455026455, 3) == "0.146"
    assert gen._ci3((0.0005, 0.2175)) == r"{\scriptsize $[+0.001, +0.218]$}"


# ------------------------------------------------- percentage points (the user's comment, 2026-09-24)


def _shift_point(s: str) -> str:
    """A printed decimal string with its point moved two places right, done on the characters and
    independently of the generator: "+0.112" -> "+11.2", "-0.035" -> "-3.5", "0.13" -> "13"."""
    sign = s[:1] if s[:1] in "+-" else ""
    whole, _, frac = s[len(sign) :].partition(".")
    assert len(frac) >= 2, s
    lead = (whole + frac[:2]).lstrip("0") or "0"
    return sign + lead + ("." + frac[2:] if frac[2:] else "")


def _pp(gen, x: float, sign: bool = True) -> str:
    """What a percentage-point cell must print: the 3-decimal string the paper printed, moved."""
    return _shift_point(gen._fmt(x, 3, sign=sign))


def _assert_tick_labels_in_points(axis, name: str) -> list[str]:
    """Every tick label an axis draws inside its view reads 100 times its tick value, with no
    trailing zeros: a fraction tick ("0.8") or an unshifted one fails."""
    lo, hi = sorted(axis.get_view_interval())
    locs = [v for v in axis.get_majorticklocs() if lo - 1e-12 <= v <= hi + 1e-12]
    labels = list(axis.get_major_formatter().format_ticks(locs))
    assert labels, name
    for v, lab in zip(locs, labels):
        assert abs(float(lab.replace("−", "-")) - 100 * v) < 1e-9, (name, v, lab)
        assert not re.search(r"\.\d*0$", lab), (name, lab)
    return labels


def test_pts_is_the_printed_three_decimal_string_with_its_point_moved(gen):
    """A percentage-point value is the 3-decimal string the paper printed with its decimal point
    moved two places, never a fresh rounding of x*100, so every value stays digit-identical to the
    fraction printed before (the prose already reads composed coverage +0.152 as 15.2 and random
    questions on StrategyQA +0.633 as 63.3)."""
    assert gen._pts(0.112, sign=True) == "+11.2"
    assert gen._pts(-0.035, sign=True) == "-3.5"
    assert gen._pts(0.009, sign=True) == "+0.9"
    assert gen._pts(0.895) == "89.5"
    assert gen._pts(0.4272) == "42.7"  # a percent of a rate: stop-when-done
    assert gen._pts(0.6325267857142857, sign=True) == "+63.3"
    # composed coverage: its 4-dp display is the tie 0.1525, which re-rounded prints 15.3; the
    # value is 0.15246..., printed +0.152, so the point value is 15.2
    x = 0.15246465773809523
    assert gen._fmt(x, 4) == "0.1525" and gen._fmt(15.25, 1) == "15.3"
    assert gen._pts(x, sign=True) == "+15.2"
    # half up on the exact decimal: rounding x*100 in binary goes the other way on these
    assert gen._pts(0.1125, sign=True) == "+11.3" and f"{0.1125 * 100:+.1f}" == "+11.2"
    assert gen._pts(0.0045, sign=True) == "+0.5" and f"{0.0045 * 100:+.1f}" == "+0.4"
    # a two-decimal print moves the same way (Figure 3's success levels)
    assert gen._pts(0.13, places=2) == "13"
    # and over a sweep that includes every 4-dp tie, the helper is the character shift of _fmt
    xs = [i / 997 for i in range(-1000, 1001)] + [(k + 0.5) / 1000 for k in range(-200, 1000)]
    for x in xs:
        assert gen._pts(x, sign=True) == _shift_point(gen._fmt(x, 3, sign=True)), x
        assert gen._pts(x) == _shift_point(gen._fmt(x, 3)), x


def test_axis_ticks_in_points_move_the_same_point(gen):
    """A figure axis in points labels each tick with its value moved two places, trailing zeros
    dropped, a float-noise zero printed 0 and a minus sign typeset as matplotlib's."""
    assert gen._pts_tick(0.35) == "35"
    assert gen._pts_tick(1.0) == "100"
    assert gen._pts_tick(0.025) == "2.5"
    assert gen._pts_tick(-0.1) == "−10"
    assert gen._pts_tick(0.30000000000000004) == "30"
    assert gen._pts_tick(-2.7755575615628914e-17) == "0"
    assert gen._pts_tick(0.0) == "0"


# The rows of Table 2 whose quantity is a proportion, and so is printed in points. Breadth counts
# independent lines and deepest need resolved counts levels: those keep their own units.
HEADLINE_POINTS_METRICS = {
    "evidence_coverage",
    "dwr",
    "precedence_violation_rate",
    "answer_token_f1",
}


def test_table_app_headline_diffs_prints_proportions_in_points_and_counts_as_before(gen, tmp_path):
    """The differences in percentage points: coverage, depth-weighted recall, out-of-order and answer
    F1 cells and their intervals print in points with one decimal, each digit-identical to the
    3-decimal value printed before; breadth (lines) and deepest need resolved (levels) print exactly
    as before.

    REQUIREMENT MOVED 2026-09-25 (INBOX_FOR_PAPER_LANE section 15): asked of table_headline until
    then, which printed these differences with a level line ("89.5 / 78.3") and a question line. The
    differences now live in the appendix copy, without those two lines; the levels and the question
    counts are Table 2's cells (test_table_headline_prints_every_level_as_its_provenance_level)."""
    _need(gen.SEED_IDENTITY, gen.REGIMES, gen.ANSWER_F1)
    gen.table_app_headline_diffs(tmp_path)
    tex = (tmp_path / "table_app_headline_diffs.tex").read_text()
    prov = json.loads((tmp_path / "table_app_headline_diffs.provenance.json").read_text())
    printed = [ln for ln in tex.splitlines() if not ln.lstrip().startswith("%")]
    lines = [ln for ln in printed if " & " in ln]
    assert lines[0] == r" & MuSiQue & StrategyQA & 2WikiMultiHopQA \\"
    suites = ("musique", "strategyqa", "wiki2")
    want = []
    for _blk, key, label in gen.HEADLINE_ROWS:
        if key is None:
            continue
        pts = key.split("::")[2] in HEADLINE_POINTS_METRICS

        def num(x, sign=True):
            return _pp(gen, x, sign) if pts else gen._fmt(x, 3, sign=sign)

        row = []
        for s in suites:
            c = prov["cells"][f"{key}::{s}"]
            row.append(
                f"${num(c['delta'])}$ {{\\scriptsize $[{num(c['ci'][0])}, {num(c['ci'][1])}]$}}"
                + ("" if c["decided"] else r"$^{\circ}$")
            )
        want.append(label + " & " + " & ".join(row) + r" \\")
    assert lines[1:] == want
    body = "\n".join(printed)
    # the example the request gave, read off the page
    assert r"required-evidence coverage & $+11.2$ {\scriptsize $[+8.7, +14.3]$}" in body
    # the printed map is the page: one printed cell per provenance cell
    assert set(prov["printed"]) == set(prov["cells"]) and len(prov["printed"]) == 27
    for key, s_row in prov["printed"].items():
        assert s_row in body, key
    # no proportion is left on the 0.X scale: 3-decimal numbers remain only on the two count rows
    for ln in lines[1:]:
        if re.search(r"\d\.\d{3}", ln):
            assert ln.startswith(("breadth & ", "deepest need resolved & ")), ln


def test_table_mechanism_main_prints_gains_in_points_and_rates_in_percent(gen, tmp_path):
    """Table 3 in points: the gate-matched gains in percentage points and the stop-when-done and
    ask-when-not-done rates in percent, one decimal, each the 3-decimal value printed before with
    its point moved; the question counts unchanged. The appendix version is not converted."""
    _need(gen.BCA_CELLS, gen.APPENDIX_TRAINING, gen.DEV_SEED_GATE)
    gen.table2_mechanism(tmp_path)
    gen.table_mechanism_main(tmp_path)
    tex = (tmp_path / "table_mechanism_main.tex").read_text()
    values = json.loads((tmp_path / "table_mechanism_main.provenance.json").read_text())["values"]
    printed = [ln for ln in tex.splitlines() if not ln.lstrip().startswith("%")]
    rows = {ln.split(" & ")[0]: ln for ln in printed if ln.split(" & ")[0] in MECHANISM_MAIN_ROWS}
    assert list(rows) == list(MECHANISM_MAIN_ROWS)
    for label in MECHANISM_MAIN_ROWS:
        # one group per suite since the restructure (was "M/S" pairs in three shared columns)
        groups = []
        for suite in ("musique", "strategyqa"):
            c = values[label][suite]
            gain = (
                "---"
                if label == "same model, prompted"
                else f"${_pp(gen, c['coverage_delta_matched'])}$"
            )
            rates = [_pp(gen, c[k], False) for k in ("p_stop_given_done", "p_ask_given_not_done")]
            groups.append(" & ".join([gain, *rates, gen._fmt(c["mean_questions_asked"], 2)]))
        # the numbers inside any bold wrapper (bold since 2026-09-25, pinned in its own test)
        cells = rows[label][: -len(r" \\")].split(" & ")
        unbold = " & ".join([cells[0], *(_unbold(c)[0] for c in cells[1:])]) + r" \\"
        assert unbold == f"{label} & {groups[0]} & {groups[1]} \\\\", label
    # the request's example: stop-when-done 0.4272 prints 42.7
    seed1 = values["final stage, seed 1"]
    assert gen._fmt(seed1["musique"]["p_stop_given_done"], 4) == "0.4272"
    assert rows["final stage, seed 1"].split(" & ")[2] == "42.7"
    # the appendix table keeps the 0.X scale (the paper lane decides the appendix separately)
    app = (tmp_path / "table2_mechanism.tex").read_text()
    assert "0.427/0.546" in app and "42.7/54.6" not in app


def test_fig13_draws_success_in_points_and_follow_ups_in_turns(gen, tmp_path, monkeypatch):
    """Figure 4: the success difference in percentage points on its axis (drawn as the records'
    fractions, labelled moved two places); the follow-up difference in turns per dialogue, a count,
    labelled at its own value. Was test_fig13_draws_success_in_points_and_prints_levels_as_percents,
    whose level prints are gone and whose right panel (questions per dialogue) is replaced."""
    _need(gen.TAU2_READER, gen.TAU2_READER_120B)
    fig, prov = _captured(gen, monkeypatch, gen.fig13_tau2, tmp_path)
    labels = _assert_tick_labels_in_points(fig.axes[0].xaxis, "task success")
    assert "20" in labels
    assert "percentage points" in fig.axes[0].get_xlabel()
    q = fig.axes[1].xaxis
    lo, hi = sorted(q.get_view_interval())
    qlocs = [v for v in q.get_majorticklocs() if lo <= v <= hi]
    qlabels = list(q.get_major_formatter().format_ticks(qlocs))
    for v, lab in zip(qlocs, qlabels):
        assert abs(float(lab.replace("−", "-")) - v) < 1e-9, (v, lab)  # a count, not moved
    assert "−2" in qlabels
    assert "turns" in fig.axes[1].get_xlabel()
    assert not any(re.search(r"(?<![\d.])[−-]?0\.\d", t) for t in prov["rendered_text"])
    # every drawn point sits inside its panel's view, bars included
    for ax, metric in zip(fig.axes, ("success", "follow_ups")):
        xlo, xhi = sorted(ax.get_xlim())
        for row in prov["values"]["rows"].values():
            for who in ("same", "teacher"):
                c = row[who][metric]
                assert xlo < c["ci_lo"] and c["ci_hi"] < xhi, (metric, who, c)


def _data_marks(ax, marker):
    """The drawn data points of one marker shape in a panel: (hex color, x, y)."""
    from matplotlib.colors import to_hex

    return sorted(
        (to_hex(ln.get_color()), float(ln.get_xdata()[0]), float(ln.get_ydata()[0]))
        for ln in ax.lines
        if ln.get_marker() == marker and len(ln.get_xdata()) == 1
    )


def test_fig13_two_panels_two_comparators_in_their_colors_legend_on_top(gen, tmp_path, monkeypatch):
    """The user, round 2: two panels side by side, never a second y-axis; per row two markers, offset
    vertically, trained minus the same model, prompted in that model's orange and shape from Figures 2
    and 3 (ARM_STYLE base8b), and trained minus GPT-OSS-120B, prompted in its green and shape (ARM_STYLE
    teacher); the legend on top, above both panels; a zero line in each panel; each point at its
    provenance value."""
    from matplotlib.colors import to_hex

    _need(gen.TAU2_READER, gen.TAU2_READER_120B)
    fig, prov = _captured(gen, monkeypatch, gen.fig13_tau2, tmp_path)
    assert len(fig.axes) == 2
    left, right = fig.axes
    rows = prov["values"]["rows"]
    y = {key: -i for i, (key, _) in enumerate(gen.TAU2_ROWS)}
    style = {"same": gen.ARM_STYLE["base8b"], "teacher": gen.ARM_STYLE["teacher"]}
    offset = gen.TAU2_OFFSET  # a small offset, the same model above its row, the 120B model below
    assert offset["same"] == -offset["teacher"] and 0 < offset["same"] <= 0.25, offset
    for ax, metric in ((left, "success"), (right, "follow_ups")):
        for who, st in style.items():
            want = sorted(
                (to_hex(st["color"]), rows[k][who][metric]["point"], y[k] + offset[who]) for k in y
            )
            got = _data_marks(ax, st["marker"])
            assert len(got) == 4, (metric, who, got)
            for (gc, gx, gy), (wc, wx, wy) in zip(got, want):
                assert gc == wc and gx == wx and abs(gy - wy) < 1e-12, (metric, who)
        zero = [ln for ln in ax.lines if list(ln.get_xdata()) == [0, 0]]
        assert zero, metric
    assert gen.ARM_STYLE["base8b"]["marker"] != gen.ARM_STYLE["teacher"]["marker"]
    assert len(fig.legends) == 1
    legend = fig.legends[0]
    assert [t.get_text() for t in legend.get_texts()] == [
        gen.ARM_TERM["same"],
        gen.ARM_TERM["teacher"],
    ]
    # on the legend's row, before its entries: what each entry is subtracted from
    (lead,) = [t for t in fig.texts if t.get_text() == f"{gen.ARM_TERM['trained']} minus"]
    handles = legend.legend_handles
    assert [to_hex(h.get_color()) for h in handles] == [
        to_hex(style["same"]["color"]),
        to_hex(style["teacher"]["color"]),
    ]
    assert [h.get_marker() for h in handles] == [
        style["same"]["marker"],
        style["teacher"]["marker"],
    ]
    with gen._pdf_renderer(fig) as r:
        lb, rb = left.get_window_extent(r), right.get_window_extent(r)
        # side by side, on one row and one scale of rows (sharey), and no third axes
        assert lb.x1 < rb.x0 and abs(lb.y0 - rb.y0) < 1e-6 and abs(lb.y1 - rb.y1) < 1e-6
        assert left.get_ylim() == right.get_ylim()
        top = max(ax.get_tightbbox(r).y1 for ax in fig.axes)
        lb_, lg = lead.get_window_extent(r), legend.get_window_extent(r)
        assert lg.y0 >= top, (lg.y0, top)
        assert lb_.y0 >= top and lb_.x1 < lg.x0, (lb_, lg)
        entry = legend.get_texts()[0].get_window_extent(r)
        assert abs((lb_.y0 + lb_.y1) / 2 - (entry.y0 + entry.y1) / 2) < 0.25


def test_fig13_marks_fewer_turns_toward_negative_clear_of_everything(gen, tmp_path, monkeypatch):
    """The user, round 2: the follow-up panel marks its direction, "fewer turns" with an arrow toward
    negative. In that panel: the words, and under them an arrow whose head lies left of its tail and
    left of zero; both inside the panel, at least 1pt clear of every drawn mark (points, bars, caps,
    the zero line) and of every other text, tick label and the legend. Measured as the PDF sets it."""
    import matplotlib.text as mtext
    from matplotlib.path import Path as MplPath

    _need(gen.TAU2_READER, gen.TAU2_READER_120B)
    fig, _ = _captured(gen, monkeypatch, gen.fig13_tau2, tmp_path)
    right = fig.axes[1]
    (words,) = [t for t in right.texts if t.get_text() == "fewer turns"]
    arrows = [
        t
        for t in right.texts
        if isinstance(t, mtext.Annotation) and t.arrow_patch is not None and not t.get_text()
    ]
    assert len(arrows) == 1
    arrow = arrows[0]
    assert arrow.arrowprops["arrowstyle"] in ("->", "-|>")
    assert fig.axes[0].texts == [] or all(t.get_text() != "fewer turns" for t in fig.axes[0].texts)
    with gen._pdf_renderer(fig) as r:
        pt = fig.dpi / 72
        to_px = right.transData.transform
        head, tail = to_px(arrow.xy), to_px(arrow.xyann)
        zero_x = to_px((0.0, 0.0))[0]
        assert head[0] < tail[0] - 5 * pt and head[0] < zero_x and tail[0] < zero_x, (head, tail)
        assert abs(head[1] - tail[1]) < 1e-6  # level
        text_bb = mtext.Text.get_window_extent(words, r)
        arrow_bb = arrow.arrow_patch.get_window_extent(r)
        assert arrow_bb.y1 <= text_bb.y0 + 0.5 * pt  # under the words
        panel = right.get_window_extent(r)
        segments, boxes = _data_obstacles(right)
        for bb in (text_bb, arrow_bb):
            assert panel.x0 <= bb.x0 and bb.x1 <= panel.x1, bb
            assert panel.y0 <= bb.y0 and bb.y1 <= panel.y1, bb
            near = bb.padded(1.0 * pt)
            assert not any(near.overlaps(m) for m in boxes), "on a mark"
            assert not any(
                MplPath([a, b]).intersects_bbox(near, filled=False) for a, b in segments
            ), "on a bar or the zero line"
        others = [
            t
            for t in fig.findobj(mtext.Text)
            if t is not words and t is not arrow and t.get_visible() and t.get_text().strip()
        ]
        for t in others:
            ob = mtext.Text.get_window_extent(t, r)
            assert not text_bb.padded(pt).overlaps(ob), t.get_text()
            assert not arrow_bb.padded(pt).overlaps(ob), t.get_text()
        for lg in fig.legends:
            assert not text_bb.overlaps(lg.get_window_extent(r))


def test_fig2_recipe_reads_coverage_and_answer_correctness_in_percent(gen, tmp_path, monkeypatch):
    """Figure 2 (a): coverage in percent on both coverage panels, and FRAMES answer correctness in
    percent too, so the three panels read in one unit."""
    _need(gen.FRONTIER_RECIPE, gen.FRAMES_RECIPE_JSON)
    fig, prov = _captured(gen, monkeypatch, gen.fig2_frontier_recipe, tmp_path)
    for ax, name in zip(fig.axes, ("musique", "strategyqa", "frames")):
        _assert_tick_labels_in_points(ax.yaxis, name)
    # set on two lines, so it clears the panel titles and the page keeps its size
    assert fig.axes[0].get_ylabel().replace("\n", " ") == "required-evidence coverage (%)"
    assert fig.axes[2].get_ylabel() == "answer correctness (%)"
    assert not any(re.search(r"(?<![\d.])[−-]?0\.\d", t) for t in prov["rendered_text"])


# The page boxes (MediaBox, points) of the main-text figures as the main text was typeset with them
# at 1b1dd08. The paper places each at a fixed fraction of \linewidth, so a page that moves by a
# fraction of a point rescales the figure and moves the text around it.
MAIN_PAGES_AT_1B1DD08 = {
    "fig2_frontier_recipe": (392.134375, 191.230125),
    "fig12_comparators": (403.2925, 171.418125),
    "fig13_tau2": (403.1155177239, 162.742125),
}
# The user's round 2 (INBOX_FOR_PAPER_LANE section 14, and the coordinator's redirect after the user
# read the interim PDF): Figure 2's two parts, fig2_frontier_recipe (a) and fig12_comparators (b), sit
# in one float, each at width=\linewidth, "the same width, with matching font sizes"; Figure 4 is at
# \linewidth too. \textwidth is 5.5in = 397.5pt = 396.0bp, so a page 396.0bp wide prints 1:1 and its
# 8pt type prints at 8pt. The 9-page trim (main-text owner, 2026-09-24): each part of Figure 2 about
# 15% shorter than it printed under its 1b1dd08 box at \linewidth (193.1 -> 164bp, 168.3 -> 143bp),
# its fonts unchanged; Figure 4 at most 110bp (the redirect; it printed 159.9bp).
LINEWIDTH_BP = 396.0
TAU2_PAGE_MAX_BP = 110.0
MAIN_PAGES_ROUND2 = {
    "fig2_frontier_recipe": (LINEWIDTH_BP, 164.0),
    "fig12_comparators": (LINEWIDTH_BP, 143.0),
    "fig13_tau2": (LINEWIDTH_BP, TAU2_PAGE_MAX_BP),
}
# The user, 2026-09-25: Figure 2 at width=0.88\linewidth, with its type at 8pt and its part letters at
# 9pt AS PRINTED, and each part printing exactly as tall as its round-2 page printed at that width
# (164 x 0.88 = 144.32bp, 143 x 0.88 = 125.84bp). So each part is laid out on a page 0.88 x 396.0 =
# 348.48bp wide, which that placement prints 1:1. A REQUIREMENT CHANGE: the round-2 pages above are
# 396.0bp wide and, placed at 0.88\linewidth, printed 8pt type at 7.04pt and the 9pt letters at 7.92pt.
# Figure 4 stays at \linewidth on its round-2 page.
FIG2_PLACED_FRACTION = 0.88
MAIN_PAGES_ROUND3 = {
    "fig2_frontier_recipe": (348.48, 144.32),
    "fig12_comparators": (348.48, 125.84),
    "fig13_tau2": MAIN_PAGES_ROUND2["fig13_tau2"],
}
PAPER_TEX = REPO / "paper" / "iclr2027 2" / "iclr2027_conference.tex"
PAPER_FIGURES = PAPER_TEX.parent / "figures"


def _media_box(pdf: Path) -> tuple[float, float]:
    box = re.search(rb"/MediaBox\s*\[\s*0\s+0\s+([\d.]+)\s+([\d.]+)\s*\]", pdf.read_bytes())
    assert box, pdf
    return float(box.group(1)), float(box.group(2))


def test_main_figures_print_1to1_where_placed_and_trimmed(gen, tmp_path):
    """The user, 2026-09-25: Figure 2 at 0.88\\linewidth with its type at 8pt as printed. Each part of
    Figure 2 is written on a page 348.48bp wide (0.88 x 396.0, so placed 1:1) and exactly 0.88 of its
    round-2 height (144.32 and 125.84bp, the heights it printed at before this change); Figure 4 stays
    on its 396.0bp page (\\linewidth), at most 110bp. The 9-page trim still holds: each part of Figure 2
    prints 15% (+-1%) shorter than its 1b1dd08 box printed at the same placement. REPLACES
    test_main_figures_are_396bp_wide_and_trimmed, a REQUIREMENT CHANGE: that test pinned both parts of
    Figure 2 at 396.0 x 164 and 396.0 x 143bp, which at 0.88\\linewidth print 8pt type at 7.04pt."""
    _need(gen.FRONTIER_RECIPE, gen.FRAMES_RECIPE_JSON, gen.COMPARATOR_DIR, gen.TAU2_READER)
    for asset in ("fig2_frontier_recipe", "fig12_comparators"):
        w2, h2 = MAIN_PAGES_ROUND2[asset]
        assert MAIN_PAGES_ROUND3[asset] == (
            round(FIG2_PLACED_FRACTION * w2, 6),
            round(FIG2_PLACED_FRACTION * h2, 6),
        ), asset
    for fn, asset in (
        (gen.fig2_frontier_recipe, "fig2_frontier_recipe"),
        (gen.fig12_comparators, "fig12_comparators"),
        (gen.fig13_tau2, "fig13_tau2"),
    ):
        fn(tmp_path)
        w, h = _media_box(tmp_path / f"{asset}.pdf")
        assert (w, h) == MAIN_PAGES_ROUND3[asset], (asset, w, h)
        placed = 1.0 if asset == "fig13_tau2" else FIG2_PLACED_FRACTION
        old_w, old_h = MAIN_PAGES_AT_1B1DD08[asset]
        # its height under 1b1dd08, placed at the width it is placed at now
        printed_before = old_h * placed * LINEWIDTH_BP / old_w
        if asset == "fig13_tau2":
            assert h <= TAU2_PAGE_MAX_BP, h
        else:
            assert 0.84 <= h / printed_before <= 0.86, (asset, h, printed_before)
        box = json.loads((tmp_path / f"{asset}.provenance.json").read_text())["placement"]
        assert tuple(box["page_pt"]) == MAIN_PAGES_ROUND3[asset], asset
        # laid out at the width it prints at, not scaled into it
        assert box["natural_width_in"] == round(placed * LINEWIDTH_BP / 72, 4), asset
        cw, ch = box["content_pt"]
        assert cw <= w and ch <= h, asset


@pytest.mark.parametrize("asset", ["fig2_frontier_recipe", "fig12_comparators", "fig13_tau2"])
def test_main_figures_print_1to1_at_the_width_the_paper_places_them(gen, asset):
    """Type "8pt as printed" is a claim about the page AND the width the paper places it at, and nothing
    tied the two: the paper moved Figure 2 to 0.88\\linewidth (a440ed3) while both parts stayed on
    396.0bp pages, so their 8pt type printed at 7.04pt and every other test still passed. Here the
    figure's page, as the generator pins it and as the paper's figures directory carries it, times the
    width the paper's \\includegraphics gives it, is 1:1."""
    _need(PAPER_TEX)
    placed = re.findall(
        r"\\includegraphics\[width=([\d.]*)\\linewidth\]\{figures/" + re.escape(asset) + r"\.pdf\}",
        PAPER_TEX.read_text(),
    )
    assert len(placed) == 1, (asset, placed)
    target = float(placed[0] or 1.0) * LINEWIDTH_BP
    on_disk = PAPER_FIGURES / f"{asset}.pdf"
    _need(on_disk)
    for where, page_w in (
        ("MAIN_PAGE_PT", gen.MAIN_PAGE_PT[asset][0]),
        ("the paper's figures directory", _media_box(on_disk)[0]),
    ):
        scale = target / page_w
        assert abs(scale - 1.0) < 1e-9, (
            f"{asset} ({where}): a {page_w}bp page placed {target:.2f}bp wide prints 8pt type at "
            f"{8 * scale:.2f}pt"
        )


@pytest.mark.parametrize("fn", ["fig2_frontier_recipe", "fig12_comparators", "fig13_tau2"])
def test_main_figures_set_their_type_at_one_size(gen, tmp_path, monkeypatch, fn):
    """The user, round 2: Figure 2's parts "with matching font sizes", and Figure 4's fonts matching
    them. All three print 1:1 (the two page tests above), so every tick label, axis label, panel
    title, figure-level label and legend entry is set at the rcParams' 8pt, and the bold part letter
    of Figure 2 at 9pt (PANEL_LABEL_PT) in both parts; only notes set inside a panel (Figure 2's
    ceiling numbers, Figure 4's direction note) are smaller. Figure 2(b)'s two x-axis labels were 7pt
    until this change."""
    import matplotlib.text as mtext

    _need(gen.FRONTIER_RECIPE, gen.FRAMES_RECIPE_JSON, gen.COMPARATOR_DIR, gen.TAU2_READER_120B)
    fig, _ = _captured(gen, monkeypatch, getattr(gen, fn), tmp_path)
    inside = {id(t) for ax in fig.axes for t in ax.texts}
    sizes = {}
    for t in fig.findobj(mtext.Text):
        if not t.get_visible() or not t.get_text().strip() or id(t) in inside:
            continue
        if t.get_text() in ("(a)", "(b)"):
            assert t.get_fontsize() == gen.PANEL_LABEL_PT == 9, t.get_text()
            continue
        sizes.setdefault(t.get_fontsize(), []).append(t.get_text())
    assert set(sizes) == {8.0}, {k: v[:4] for k, v in sizes.items()}


@pytest.mark.parametrize("fn", ["fig2_frontier_recipe", "fig12_comparators"])
def test_figure2_part_letters_add_nothing_to_the_drawn_extent(gen, tmp_path, monkeypatch, fn):
    """The redirect keeps the bold (a) and (b) at the new width, on condition that they still add
    nothing to the drawn extent: the content box a tight save cuts, measured as the PDF sets it, is
    the same to a hundredth of a point with the letter drawn and with it hidden."""
    import matplotlib.text as mtext

    _need(gen.FRONTIER_RECIPE, gen.FRAMES_RECIPE_JSON, gen.COMPARATOR_DIR)
    fig, _ = _captured(gen, monkeypatch, getattr(gen, fn), tmp_path)
    (letter,) = [t for t in fig.findobj(mtext.Text) if t.get_text() in ("(a)", "(b)")]
    with gen._pdf_renderer(fig) as r:
        shown = _content_px(fig, r)
    letter.set_visible(False)
    with gen._pdf_renderer(fig) as r:
        hidden = _content_px(fig, r)
    assert all(abs(a - b) < 0.01 for a, b in zip(shown.extents, hidden.extents)), (
        shown.extents,
        hidden.extents,
    )


def test_figure2_parts_align_on_the_page(gen, tmp_path, monkeypatch):
    """The user, round 2: the stacked (a) and (b) of Figure 2 read as misaligned (they printed at
    scales 1.0099 and 0.9819). On pages of one width, each part's drawn content is the same width to
    a tenth of a point, so, centred, the bold (a) and (b) sit at the same distance from the page's
    left edge and the two parts end at the same right edge."""
    import matplotlib.text as mtext
    from matplotlib import rcParams

    _need(gen.FRONTIER_RECIPE, gen.FRAMES_RECIPE_JSON, gen.COMPARATOR_DIR)
    got = {}
    for fn, asset, letter in (
        ("fig2_frontier_recipe", "fig2_frontier_recipe", "(a)"),
        ("fig12_comparators", "fig12_comparators", "(b)"),
    ):
        fig, prov = _captured(gen, monkeypatch, getattr(gen, fn), tmp_path / asset)
        (t,) = [x for x in fig.findobj(mtext.Text) if x.get_text() == letter]
        pad = rcParams["savefig.pad_inches"] * 72
        with gen._pdf_renderer(fig) as r:
            content = fig.get_tightbbox(r)  # inches
            cw = content.width * 72 + 2 * pad
            page_w = gen.MAIN_PAGE_PT[asset][0]
            page_x0 = content.x0 * 72 - pad - (page_w - cw) / 2
            got[asset] = {
                "content_w": cw,
                "letter_x": t.get_window_extent(r).x0 - page_x0,
                "right": content.x1 * 72 + pad - page_x0,
                "recorded_w": prov["placement"]["content_pt"][0],
            }
    a, b = got["fig2_frontier_recipe"], got["fig12_comparators"]
    assert (
        abs(a["content_w"] - a["recorded_w"]) < 1e-3
        and abs(b["content_w"] - b["recorded_w"]) < 1e-3
    )
    assert abs(a["content_w"] - b["content_w"]) < 0.1, got
    assert abs(a["letter_x"] - b["letter_x"]) < 0.1, got
    assert abs(a["right"] - b["right"]) < 0.1, got


def _row_label_ink_gaps_pt(fig) -> list[float]:
    """The least white between the inks of each two vertically adjacent row labels of Figure 2(b), in
    points, each label drawn alone at 600 dpi. (A text's layout box spans the font's whole ascent and
    descent, so the boxes of two rows overlap well before their glyphs touch.)"""
    import numpy as np
    from matplotlib.backends.backend_agg import RendererAgg

    ndimage = pytest.importorskip("scipy.ndimage")
    dpi = 600
    fig.dpi = dpi
    fig.canvas.draw()
    w, h = (int(v) for v in fig.bbox.size)

    def ink(t):
        r = RendererAgg(w, h, dpi)
        t.draw(r)
        return np.asarray(r.buffer_rgba())[..., 3] > 0

    rows = [t for t in fig.axes[0].get_yticklabels() if t.get_visible() and t.get_text()]
    masks = [ink(t) for t in rows]
    assert len(masks) == 6 and all(m.any() for m in masks)
    return [
        float(ndimage.distance_transform_edt(~a)[b].min()) * 72 / dpi
        for a, b in zip(masks, masks[1:])
    ]


def _marker_edge_clearance_pt(fig) -> float:
    """The least distance, in points, from any marker's outer edge (half its size plus half its edge
    width) to the top or bottom of its panel; negative where the panel clips it."""
    fig.dpi = 72
    fig.canvas.draw()
    worst = float("inf")
    for ax in fig.axes:
        bb = ax.get_window_extent()
        for ln in ax.lines:
            if ln.get_marker() in (None, "None", "none", "", " "):
                continue
            half = ln.get_markersize() / 2 + ln.get_markeredgewidth() / 2
            for _x, y in ax.transData.transform(ln.get_xydata()):
                worst = min(worst, bb.y1 - (y + half), (y - half) - bb.y0)
    return worst


def test_fig12_rows_clear_each_other_and_the_panel_edges(gen, tmp_path, monkeypatch):
    """The user, 2026-09-25: Figure 2(b) on a page 17.16bp shorter, its type still 8pt. A fit that only
    shrank the panels set the six rows 7.0pt apart: the descenders of "prompted" touched the
    superscript of "PAR$^2$-RAG" in the row below (0.0pt of white at 600 dpi; 2.64pt at 396 x 143),
    and matplotlib's 5% margin set the outer markers' edges 0.62pt past the panels' top and bottom,
    where the panels clip them (0.15pt at 396 x 143). Here every two adjacent row labels have at least
    1pt of white between their inks, and every marker's edge is inside its panel. THE NON-VACUITY
    CASE: the round-2 spacing on the new page, with the markers set 1pt past the edges, fails both."""
    _need(gen.COMPARATOR_DIR, gen.FRAMES_STRUCTURED)
    fig, _ = _captured(gen, monkeypatch, gen.fig12_comparators, tmp_path / "now")
    gaps = _row_label_ink_gaps_pt(fig)
    assert min(gaps) >= 1.0, gaps
    assert _marker_edge_clearance_pt(fig) >= 0.0

    for name, value in (
        ("FIG12_GROUP_GAP", 0.6),
        ("FIG2_PARTS_TITLE_PAD_PT", 6.0),
        ("FIG2_PARTS_LABELPAD_PT", 4.0),
        ("FIG12_XLABEL_LINESPACING", 1.2),
        ("FIG12_MARKER_CLEAR_PT", -1.0),
    ):
        monkeypatch.setattr(gen, name, value)
    monkeypatch.setitem(gen.MAIN_PAGE_PT, "fig12_comparators", (gen.FIG2_PARTS_W_BP, 200.0))
    was, _ = _captured(gen, monkeypatch, gen.fig12_comparators, tmp_path / "was")
    assert min(_row_label_ink_gaps_pt(was)) < 1.0
    assert _marker_edge_clearance_pt(was) < 0.0


def test_a_figure_that_outgrows_its_pinned_page_is_refused(gen, tmp_path, monkeypatch):
    """THE NON-VACUITY CASE: content wider than the pinned page is refused, and nothing is written."""
    _need(gen.TAU2_READER, gen.TAU2_READER_120B)
    monkeypatch.setitem(gen.MAIN_PAGE_PT, "fig13_tau2", (380.0, TAU2_PAGE_MAX_BP))
    with pytest.raises(gen.Mismatch) as exc:
        gen.fig13_tau2(tmp_path / "out")
    assert "NOT DRAWING" in str(exc.value)
    assert not (tmp_path / "out").exists() or list((tmp_path / "out").iterdir()) == []


def test_fig12_reads_differences_in_percentage_points(gen, tmp_path, monkeypatch):
    """Figure 2 (b): every difference in percentage points, and the figure says so."""
    _need(gen.COMPARATOR_DIR)
    fig, prov = _captured(gen, monkeypatch, gen.fig12_comparators, tmp_path)
    for ax in fig.axes:
        _assert_tick_labels_in_points(ax.xaxis, ax.get_title(loc="left"))
    assert any("percentage points" in t for t in prov["rendered_text"])
    assert not any(re.search(r"(?<![\d.])[−-]?0\.\d", t) for t in prov["rendered_text"])


def test_table_headline_prints_levels_over_the_pairing_beside_its_deltas(gen, tmp_path):
    """Reviewer 2: absolute levels beside the deltas. Every cell carries both arms' levels from the
    record its delta comes from, over the same paired tasks, so the two levels difference to the
    delta.

    REQUIREMENT CHANGED 2026-09-25 (INBOX_FOR_PAPER_LANE section 15): Table 2 prints the levels as its
    cells (Baseline, then Q&D), no longer on a level line under the coverage row; the deltas are in the
    appendix copy. The first row's MuSiQue pair is read here from table1_by_seed.json directly."""
    _need(gen.SEED_IDENTITY, gen.REGIMES, gen.ANSWER_F1)
    gen.table_headline(tmp_path)
    prov = json.loads((tmp_path / "table_headline.provenance.json").read_text())
    for key, c in prov["cells"].items():
        a, b = c["levels"]
        assert abs((a - b) - c["delta"]) <= 1e-12, key
    tex = (tmp_path / "table_headline.tex").read_text()
    t1 = json.loads(gen.SEED_IDENTITY.read_text())["cells"]["s1s2"]["evidence_coverage::musique"]
    ten = next(r for r in t1 if r["n_boot"] == 10000)
    first = _level_rows(tex)[(HEADLINE_LAYOUT[0][0], r"Required-evidence coverage (\%)")]
    # the levels in percent (the user's comment, 2026-09-24; was the 0.X scale)
    assert [_unbold(c)[0] for c in first[:2]] == [
        _pp(gen, ten["level_b"], False),
        _pp(gen, ten["level_a"], False),
    ]
    assert "levels:" not in tex


@pytest.mark.parametrize("fn", ["table_headline", "table_app_headline_diffs"])
def test_table_headline_refuses_levels_that_do_not_difference_to_the_delta(
    gen, tmp_path, monkeypatch, fn
):
    """THE NON-VACUITY CASE: a level read over another task set than the delta is refused, by Table
    2 and by the appendix copy of its differences alike (both read _headline_cells), and nothing is
    written."""
    _need(gen.SEED_IDENTITY)
    rec = json.loads(gen.SEED_IDENTITY.read_text())
    for r in rec["cells"]["s1s2"]["evidence_coverage::wiki2"]:
        r["level_a"] += 0.01
    bad = tmp_path / "table1_by_seed.json"
    bad.write_text(json.dumps(rec))
    monkeypatch.setattr(gen, "SEED_IDENTITY", bad)
    with pytest.raises(gen.Mismatch) as exc:
        getattr(gen, fn)(tmp_path / "out")
    assert "NOT WRITING" in str(exc.value)
    assert not (tmp_path / "out").exists() or list((tmp_path / "out").iterdir()) == []


# ------------------------------------------------- the four complete runs of the final stage


def test_table_fourseeds_prints_every_run_and_pooled_cell_from_the_record(gen, tmp_path):
    """Coverage at equal spend for the four complete runs: each printed point and bound is the
    record's own value rounded half-up, the fresh seed-0 run is labeled as a retrained run, and the
    pooled column is the four-run reading."""
    _need(gen.FOUR_SEED)
    gen.table_fourseeds(tmp_path)
    tex = (tmp_path / "table_fourseeds.tex").read_text()
    body = "\n".join(ln for ln in tex.splitlines() if not ln.lstrip().startswith("%"))
    rec = json.loads(gen.FOUR_SEED.read_text())["table1"]
    assert "seed 0, retrained" in body
    for suite in ("musique", "strategyqa", "wiki2"):
        cells = rec["cells"][f"evidence_coverage::{suite}"]
        for seed in ("s1", "s2", "s3", "s0fresh"):
            c = cells[seed][0] if isinstance(cells[seed], list) else cells[seed]
            for x in (c["delta"], c["ci_lo"], c["ci_hi"]):
                assert gen._fmt(x, 3, sign=True) in body, (suite, seed, x)
        pooled = rec["pooled4"][f"evidence_coverage::{suite}"][0]
        assert gen._fmt(pooled["delta"], 3, sign=True) in body
    prov = json.loads((tmp_path / "table_fourseeds.provenance.json").read_text())
    assert prov["extra"]["record_sha256"].startswith("792dfac1286f779d")


def test_table_fourseeds_refuses_when_a_lock_moved(gen, tmp_path, monkeypatch):
    """The record's locks (seed 0 against the published cells, seeds 1 and 2 against their own
    record) must hold before any cell is drawn."""
    _need(gen.FOUR_SEED)
    rec = json.loads(gen.FOUR_SEED.read_text())
    k = next(iter(rec["table1"]["lock"]))
    rec["table1"]["lock"][k]["reproduced"] = rec["table1"]["lock"][k]["published"] + 1e-6
    bad = tmp_path / "four_seed_table.json"
    bad.write_text(json.dumps(rec))
    monkeypatch.setattr(gen, "FOUR_SEED", bad)
    with pytest.raises(gen.Mismatch):
        gen.table_fourseeds(tmp_path / "out")


def test_fig8_uses_the_tables_names_and_annotates_a_clipped_row(gen, tmp_path):
    """The controls figure names the variants as tab:app-controls-pooled and the prose do, and a
    row whose interval runs past the axis (random questions, about +0.5) is drawn at the edge with
    its value printed, so the axis resolves the other rows."""
    _need(gen.CONTROLS_RECIPE_DIR)
    prov, text = _rendered(gen, tmp_path, gen.fig8_controls_recipe, "fig8_controls_recipe")
    for name in ("depth one", "evidence blind", "single model"):
        assert name in text, name
    for old in ("no sequencing", "blind to evidence", "one model, both roles"):
        assert old not in text, old
    clipped = prov["extra"]["clipped_rows"]
    assert [c["group"] for c in clipped] == ["random questions"]
    assert prov["extra"]["xlim"][1] < 0.3
    assert gen._fmt(clipped[0]["delta"], 3, sign=True) in text


def test_fig12_colors_follow_panel_a(gen, tmp_path):
    """Figure 4 (a) colors the same model and GPT-OSS-120B by ARM_STYLE; (b) must use the same
    colors for the same models and give Claude Opus 5 a color neither panel uses for another model."""
    _need(gen.COMPARATOR_DIR)
    gen.fig12_comparators(tmp_path)
    colors = json.loads((tmp_path / "fig12_comparators.provenance.json").read_text())["extra"][
        "colors"
    ]
    assert colors["8B base"] == gen.ARM_STYLE["base8b"]["color"]
    assert colors["gpt-oss-120b"] == gen.ARM_STYLE["teacher"]["color"]
    used = {s["color"] for s in gen.ARM_STYLE.values()}
    assert colors["Claude Opus 5"] not in used


# ------------------------------------------------- reviewer pass 2026-09-24: legends and axes


def _captured(gen, monkeypatch, fn, out):
    """Build one figure and return the drawn matplotlib Figure with its provenance record, so a
    test reads the artists a reader sees (text positions, markers, legend order, plotted data)
    rather than the constants that were meant to produce them."""
    got: dict = {}
    real = gen._emit_figure

    def keep(fig, where, asset_id, prov):
        got["fig"] = fig
        written = real(fig, where, asset_id, prov)
        got["prov"] = json.loads((where / f"{asset_id}.provenance.json").read_text())
        return written

    monkeypatch.setattr(gen, "_emit_figure", keep)
    fn(out)
    fig = got["fig"]
    # `_emit_figure` closes the figure, which detaches its Agg canvas; give it one back so text
    # extents can be measured at the figure's own dpi.
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    FigureCanvasAgg(fig)
    fig.canvas.draw()
    return fig, got["prov"]


def _no_retired(prov):
    for t in prov["rendered_text"]:
        assert not any(r in t for r in RETIRED_TERMS), f"retired term in {t!r}"


def _data_obstacles(ax):
    """Every drawn data mark in a panel, in display pixels: line segments (curves and error bars)
    and marker boxes (points and error-bar caps), leaving out annotations."""
    from matplotlib.collections import LineCollection
    from matplotlib.transforms import Bbox

    segments, boxes = [], []
    px = ax.figure.dpi / 72
    for ln in ax.lines:
        xy = ax.transData.transform(ln.get_xydata())
        if ln.get_linestyle() not in ("None", "none", "", " ") and len(xy) > 1:
            segments += list(zip(xy[:-1], xy[1:]))
        if ln.get_marker() not in (None, "None", "none", "", " "):
            half = ln.get_markersize() * px / 2
            boxes += [Bbox.from_extents(x - half, y - half, x + half, y + half) for x, y in xy]
    for coll in ax.collections:
        if isinstance(coll, LineCollection):
            for seg in coll.get_segments():
                xy = ax.transData.transform(seg)
                segments += list(zip(xy[:-1], xy[1:]))
    return segments, boxes


def test_fig2_recipe_marks_every_ceiling_on_the_trained_questioners_points(
    gen, tmp_path, monkeypatch
):
    """Reviewer: the five points of a curve were never tied to their ceilings. Each of the trained
    questioner's points carries its ceiling (4, 8, 12, 16, 24), anchored at exactly the drawn
    point, inside its own panel (so the fixed-size figure cannot grow), clear of every other
    ceiling number and of every drawn mark (a number on another arm's point or bar would read as
    that arm's), with no leader crossing another leader or another number."""
    import matplotlib.text as mtext
    from matplotlib.path import Path as MplPath

    _need(gen.FRONTIER_RECIPE, gen.FRAMES_RECIPE_JSON)
    fig, prov = _captured(gen, monkeypatch, gen.fig2_frontier_recipe, tmp_path)
    r = fig.canvas.get_renderer()
    caps = [str(c) for c in gen.CAPS]
    pad = 0.5 * fig.dpi / 72
    for ax, suite in zip(fig.axes, ("musique", "strategyqa", "frames")):
        marks = [t for t in ax.texts if t.get_text() in caps]
        assert sorted(t.get_text() for t in marks) == sorted(caps), suite
        cells = prov["values"][suite]["recipe"]
        xkey = "mean_retrieval_calls" if suite == "frames" else "x_mean_retrieval_calls"
        panel = ax.get_window_extent(r)
        boxes = []
        for t in marks:
            c = cells[t.get_text()]
            assert tuple(t.xy) == (c[xkey], c["y"]), (suite, t.get_text())
            # the whole annotation (number and its leader line) inside the panel
            bb = t.get_window_extent(r)
            assert panel.x0 <= bb.x0 and bb.x1 <= panel.x1, (suite, t.get_text())
            assert panel.y0 <= bb.y0 and bb.y1 <= panel.y1, (suite, t.get_text())
            # the number alone, for the collision checks (an Annotation's extent includes its line)
            boxes.append(mtext.Text.get_window_extent(t, r))
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                assert not boxes[i].overlaps(boxes[j]), (
                    suite,
                    marks[i].get_text(),
                    marks[j].get_text(),
                )
        segments, marker_boxes = _data_obstacles(ax)
        # a reviewer read numbers within half a point of a bar as sitting on it: 3pt of clearance
        data_pad = 3.0 * fig.dpi / 72
        for t, bb in zip(marks, boxes):
            near = bb.padded(data_pad)
            assert not any(near.overlaps(m) for m in marker_boxes), (
                suite,
                t.get_text(),
                "on a mark",
            )
            assert not any(
                MplPath([a, b]).intersects_bbox(near, filled=False) for a, b in segments
            ), (suite, t.get_text(), "on a curve or bar")
        # each number's leader (its centre to its point) clears every OTHER number by 0.5pt, and
        # no two leaders cross
        leaders = [
            MplPath([bb.corners().mean(axis=0), ax.transData.transform(t.xy)])
            for t, bb in zip(marks, boxes)
        ]
        for t, line in zip(marks, leaders):
            for u, other, line2 in zip(marks, boxes, leaders):
                if u is not t:
                    assert not line.intersects_bbox(other.padded(pad), filled=False), (
                        suite,
                        f"the line of {t.get_text()} crosses {u.get_text()}",
                    )
                    assert not line.intersects_path(line2), (suite, t.get_text(), u.get_text())


def test_fig2_recipe_frames_panel_shows_only_its_own_axis(gen, tmp_path, monkeypatch):
    """Reviewer: the FRAMES panel must show only its own axis (answer correctness), not left-axis
    tick labels that belong to the coverage panels."""
    _need(gen.FRONTIER_RECIPE, gen.FRAMES_RECIPE_JSON)
    fig, _ = _captured(gen, monkeypatch, gen.fig2_frontier_recipe, tmp_path)
    ax = fig.axes[2]
    # in percent since the user's comment of 2026-09-24 (was "answer correctness", a fraction)
    assert ax.get_ylabel() == "answer correctness (%)"
    ticks = ax.yaxis.get_major_ticks()
    assert not any(t.label1.get_visible() or t.tick1line.get_visible() for t in ticks)
    assert any(t.label2.get_visible() for t in ticks)
    assert not ax.spines["left"].get_visible()


def _content_px(fig, r):
    """The figure's drawn extent (every artist, as a tight save cuts it), in display pixels (points,
    on the PDF renderer)."""
    from matplotlib.transforms import Bbox

    return Bbox(fig.get_tightbbox(r).get_points() * fig.dpi)


@pytest.mark.parametrize(
    ("fn", "asset", "want", "other"),
    [
        ("fig2_frontier_recipe", "fig2_frontier_recipe", "(a)", "(b)"),
        ("fig12_comparators", "fig12_comparators", "(b)", "(a)"),
    ],
)
def test_figure2_parts_carry_a_bold_panel_label_at_the_top_left(
    gen, tmp_path, monkeypatch, fn, asset, want, other
):
    """Reviewer 2026-09-24: Figure 2 is two PDFs stacked, and the caption's (a) and (b) had nothing
    on the page to point at. Each part carries its own letter in bold, at its top left (no artist
    left of it, none a row above it), clear of every other text, the legend and every panel, and
    only its own letter. Measured as the PDF sets it, which is what the page shows."""
    import matplotlib.text as mtext

    _need(gen.FRONTIER_RECIPE, gen.FRAMES_RECIPE_JSON, gen.COMPARATOR_DIR)
    fig, prov = _captured(gen, monkeypatch, getattr(gen, fn), tmp_path)
    assert want in prov["rendered_text"] and other not in prov["rendered_text"], asset
    labels = [t for t in fig.findobj(mtext.Text) if t.get_text() in ("(a)", "(b)")]
    assert [t.get_text() for t in labels] == [want], asset
    label = labels[0]
    assert label.get_fontweight() in ("bold", 700), asset
    with gen._pdf_renderer(fig) as r:
        bb = label.get_window_extent(r)
        content = _content_px(fig, r)
        # top left: nothing drawn left of it, and nothing a whole row (its own height) above it
        assert bb.x0 <= content.x0 + 0.05, (asset, bb.x0, content.x0)
        assert bb.y1 >= content.y1 - bb.height, (asset, bb.y1, content.y1)
        others = [
            t
            for t in fig.findobj(mtext.Text)
            if t is not label and t.get_visible() and t.get_text().strip()
        ]
        for t in others:
            assert not bb.overlaps(t.get_window_extent(r)), (asset, t.get_text())
        for lg in fig.legends:
            assert not bb.overlaps(lg.get_window_extent(r)), asset
        for ax in fig.axes:
            assert not bb.overlaps(ax.get_window_extent(r)), (asset, ax.get_title(loc="left"))


def test_fig2_recipe_legend_sits_at_the_top_of_panel_a(gen, tmp_path, monkeypatch):
    """Reviewer 2026-09-24: set below (a)'s panels, the legend sat between (a) and (b) of the
    stacked Figure 2 and read as keying both. It goes above (a)'s panels (their titles and ceiling
    numbers included), and nothing of (a) is drawn below its own axes' labels, so nothing sits
    between (a)'s panels and (b)."""
    _need(gen.FRONTIER_RECIPE, gen.FRAMES_RECIPE_JSON)
    fig, _ = _captured(gen, monkeypatch, gen.fig2_frontier_recipe, tmp_path)
    assert len(fig.legends) == 1
    with gen._pdf_renderer(fig) as r:
        legend = fig.legends[0].get_window_extent(r)
        panels = [ax.get_tightbbox(r) for ax in fig.axes]
        content = _content_px(fig, r)
    top, bottom = max(p.y1 for p in panels), min(p.y0 for p in panels)
    assert legend.y0 >= top, (legend.y0, top)
    assert content.y0 >= bottom - 0.05, (content.y0, bottom)


def test_fig13_names_the_prompt_variants_as_the_appendix_tables_do(gen, tmp_path):
    """tab:tau2-trained and its siblings name the two prompt variants base and stop."""
    _need(gen.TAU2_READER, gen.TAU2_READER_120B)
    _, text = _rendered(gen, tmp_path, gen.fig13_tau2, "fig13_tau2")
    for row in (
        "retail, base prompt",
        "retail, stop prompt",
        "airline, base prompt",
        "airline, stop prompt",
    ):
        assert row in text, row


def test_fig3_shapes_one_marker_per_group_and_names_the_prompted_model(gen, tmp_path, monkeypatch):
    """Reviewer: seven marker shapes times filled/open was hard to read. One shape per group of
    arms, color per arm, fill per suite: every point keeps an identity (shape, color) that is
    unique to its arm and that the legend carries, and the prompted arm is named in the paper's
    words."""
    import matplotlib.colors as mcolors

    _need(gen.BCA_CELLS, gen.APPENDIX_TRAINING, gen.FIG3_RECIPE_RECORD)
    fig, prov = _captured(gen, monkeypatch, gen.fig3_stopping, tmp_path)
    _no_retired(prov)
    assert "the same model, prompted" in prov["rendered_text"]
    ax = fig.axes[0]
    pts = [ln for ln in ax.lines if len(ln.get_xdata()) == 1]
    assert len(pts) == 14, "seven arms at two suites"
    ident = [(ln.get_marker(), mcolors.to_hex(ln.get_color())) for ln in pts]
    assert len({m for m, _ in ident}) == len(set(gen.FIG3_GROUP_MARKER.values())) <= 4
    assert len(set(ident)) == 7, "each arm keeps a (shape, color) of its own"
    arms_legend = next(lg for lg in fig.legends if len(lg.get_texts()) == 7)
    handles = {(h.get_marker(), mcolors.to_hex(h.get_color())) for h in arms_legend.legend_handles}
    assert handles == set(ident)
    # every drawn point is one of the recorded values, each at its own arm's identity
    for slug, row in prov["values"].items():
        want = (gen.FIG3_GROUP_MARKER[slug], mcolors.to_hex(gen.FIG3_STYLE[slug][0]))
        for suite in ("musique", "strategyqa"):
            xy = (row[suite]["x_ask_when_not_done"], row[suite]["y_stop_when_done"])
            hit = [ln for ln in pts if (ln.get_xdata()[0], ln.get_ydata()[0]) == xy]
            assert len(hit) == 1, (slug, suite)
            (ln,) = hit
            assert (ln.get_marker(), mcolors.to_hex(ln.get_color())) == want, (slug, suite)
            # the arm's color is what the eye sees: the fill of a filled marker, the ring of an
            # open one (an open marker ringed in black would read as the black arm)
            seen = ln.get_markerfacecolor() if suite == "musique" else ln.get_markeredgecolor()
            assert mcolors.to_hex(seen) == want[1], (slug, suite)
    # A point hidden under another is not recoverable from the drawing: two markers closer than
    # one marker width must be one arm's two suites, and the figure must say that both are there.
    px = {ln: ax.transData.transform((ln.get_xdata()[0], ln.get_ydata()[0])) for ln in pts}
    notes = [t for t in ax.texts if "both suites" in t.get_text()]
    for i, a in enumerate(pts):
        for b in pts[i + 1 :]:
            width = max(a.get_markersize(), b.get_markersize()) * fig.dpi / 72
            if ((px[a] - px[b]) ** 2).sum() ** 0.5 < width:
                assert mcolors.to_hex(a.get_color()) == mcolors.to_hex(b.get_color())
                anchors = {
                    (a.get_xdata()[0], a.get_ydata()[0]),
                    (b.get_xdata()[0], b.get_ydata()[0]),
                }
                assert any(tuple(t.xy) in anchors for t in notes), "coincident points unmarked"


def test_fig6_titles_name_what_is_plotted_not_a_claim(gen, tmp_path):
    """Reviewer: panel (b)'s title stated a finding ("every interval excludes zero"). A title says
    what is plotted; the claim belongs to the caption."""
    _need(gen.EFFICIENCY_RECORD, gen.EFFICIENCY_STORE / "runs.parquet")
    prov, _ = _rendered(gen, tmp_path, gen.fig6_efficiency, "fig6_efficiency")
    titles = [t for t in prov["rendered_text"] if t.startswith(("(a)", "(b)"))]
    assert len(titles) == 2
    for t in titles:
        assert not any(w in t for w in ("exclud", "zero", "decided", "significant")), t


def test_fig10_plots_percent_in_figure_2as_order_and_words(gen, tmp_path, monkeypatch):
    """Figure 2(a) orders and names the arms, and the x-axis is the ceiling. The share is drawn in
    percent, the unit its caption and the main-text sentence print (the paper lane kept percent for
    this figure rather than move both texts to fractions before the freeze). Each drawn curve is the
    record's share times 100."""
    _need(gen.FRONTIER_RECIPE, gen.FRONTIER_MUSIQUE, gen.FRONTIER_STRATEGYQA, gen.FRONTIER_FRAMES)
    _need(gen.FRAMES_RECIPE_JSON)
    fig, prov = _captured(gen, monkeypatch, gen.fig10_ceiling, tmp_path / "ceiling")
    _no_retired(prov)
    got = [t.get_text() for t in fig.legends[0].get_texts()]
    fig2, _ = _captured(gen, monkeypatch, gen.fig2_frontier_recipe, tmp_path / "frontier")
    want = [t.get_text() for t in fig2.legends[0].get_texts()]
    assert got == [w for w in want if w in got] and len(got) == 3
    assert got == [gen.ARM_TERM["trained"], gen.ARM_TERM["same"], gen.ARM_TERM["teacher"]]
    arm_of = {
        gen.ARM_TERM["trained"]: "recipe",
        gen.ARM_TERM["same"]: "base",
        gen.ARM_TERM["teacher"]: "teacher",
    }
    for ax, suite in zip(fig.axes, ("musique", "strategyqa")):
        assert "ceiling" in ax.get_xlabel()
        assert 50 < ax.get_ylim()[1] <= 100
        curves = {ln.get_label(): ln for ln in ax.lines if ln.get_label() in arm_of}
        assert set(curves) == set(arm_of)
        for label, ln in curves.items():
            cells = prov["values"][suite][arm_of[label]]
            assert list(ln.get_xdata()) == list(gen.CAPS)
            assert list(ln.get_ydata()) == [100 * cells[str(c)]["ceiling_hit"] for c in gen.CAPS]
    assert "%" in fig.axes[0].get_ylabel()


def test_fig8_recipe_names_the_trained_questioner_on_its_axis(gen, tmp_path):
    """The paper's name for the model the controls are read against is the trained questioner."""
    _need(gen.CONTROLS_RECIPE_DIR)
    _, text = _rendered(gen, tmp_path, gen.fig8_controls_recipe, "fig8_controls_recipe")
    assert "coverage the trained questioner recovers and the variant does not" in text
    assert "full protocol" not in text


def test_example_names_the_prompted_panel_as_the_same_model(gen, tmp_path):
    """The worked example's prompted panel is the same model, prompted, as everywhere else."""
    _need(gen.MUSIQUE_GOLD, gen.EXAMPLE_STORE / "runs.parquet", gen.EXAMPLE_BY_SEED)
    gen.examples_transcripts(tmp_path)
    body = "\n".join(
        ln
        for ln in (tmp_path / "examples_transcripts.tex").read_text().splitlines()
        if not ln.lstrip().startswith("%")
    )
    assert "The same model, prompted" in body
    assert "same weights" not in body.lower()


def test_table_fourseeds_fits_at_full_size_with_six_columns(gen, tmp_path):
    """The eight-column body measured 552pt against a 397.5pt text width and had to be shrunk; the
    between-run SD and the bootstrap SE move to the caption, leaving suite, four runs and the pooled
    reading (about 393pt at a 3pt column gap)."""
    _need(gen.FOUR_SEED)
    gen.table_fourseeds(tmp_path)
    tex = (tmp_path / "table_fourseeds.tex").read_text()
    body = [ln for ln in tex.splitlines() if not ln.lstrip().startswith("%")]
    spec = next(ln for ln in body if ln.startswith(r"\begin{tabular}"))
    assert "{@{}lccccc@{}}" in spec, spec
    assert not any("between-run SD" in ln or "bootstrap SE" in ln for ln in body)


# --------------------------------------------------------------------------- font embedding


def test_a_figure_pdf_embeds_its_fonts_as_truetype_not_type_3(gen):
    """Google Scholar's inclusion guidelines: "Avoid use of Type 3 fonts in PDF files, because
    they're often generated with missing or incorrect font size and character encoding information,
    which makes it difficult for our parser software to extract the bibliographic data." matplotlib
    writes Type 3 unless told otherwise, so the script's style must set TrueType (fonttype 42)."""
    import io

    plt = gen._mpl()
    fig, ax = plt.subplots(figsize=(2, 1))
    ax.set_title("(a) what each need costs")
    ax.plot([0, 1], [0, 1], label="the trained questioner")
    ax.legend()
    buf = io.BytesIO()
    fig.savefig(buf, format="pdf")
    plt.close(fig)
    pdf = buf.getvalue()
    assert b"/Type3" not in pdf
    assert b"/FontFile2" in pdf  # the TrueType program itself is embedded
    assert plt.rcParams["ps.fonttype"] == 42
