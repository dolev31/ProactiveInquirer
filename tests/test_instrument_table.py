"""T6_instrument: the judge's own diagnostics, rendered.

`pi_eval.score` writes `judge_diagnostics.json` -- position bias, win/tie/loss, order
inconsistency, the length-adjusted effect, Krippendorff alpha, parse failures and which
judges were disqualified -- and NOTHING READ IT. One grep hit in the whole repo, the write.

A judge-derived number whose instrument diagnostics are on disk but never rendered is a
number nobody can audit. Every judge-derived claim in the paper depends on this table
existing, because it is where "the judge was usable" stops being an assumption.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

DIAGS = {
    "n_judgments": 240,
    "n_runs_with_metrics": 60,
    "n_parse_failures": 7,
    "judges_disqualified": {"drgym/keypoint": "position bias 0.31 exceeds 0.10"},
    "unjudged": ["r1/keypoint: empty report", "r2/keypoint: empty report"],
    "diagnostics": {
        "drgym/keypoint": {
            "position_bias": 0.31,
            "usable": False,
            "why": "position bias 0.31 exceeds 0.10",
            "n_pairs": 40,
            "wins": 18,
            "ties": 4,
            "losses": 18,
            "length_adjusted": 0.02,
        },
        "musique/support": {
            "position_bias": 0.02,
            "usable": True,
            "why": "",
            "n_pairs": 60,
            "wins": 35,
            "ties": 10,
            "losses": 15,
            "length_adjusted": 0.18,
        },
        "krippendorff_alpha_nominal": float("nan"),
        "judge_families": ["gpt-oss-120b"],
    },
}


@pytest.fixture
def diag_dir(tmp_path: Path) -> Path:
    (tmp_path / "judge_diagnostics.json").write_text(json.dumps(DIAGS))
    return tmp_path


def _table(diag_dir: Path, sigma_j: dict[str, float] | None = None):
    from pi_eval.report import Agg, instrument_table

    class Stub(Agg):  # code_versions is a METHOD that queries duckdb; con is None here
        def code_versions(self):  # type: ignore[override]
            return []

    agg = Stub.__new__(Stub)
    for field, value in (
        ("parquet_dir", diag_dir),
        ("con", None),
        ("scorer_hash", "deadbeef"),
        ("graph_version", "v1"),
        ("sigma_j", sigma_j if sigma_j is not None else {}),
        ("allow_contaminated", False),
        ("excluded_pairs", frozenset()),
        ("excluded_run_ids", frozenset()),
        ("n_boot", 10),
        ("n_perm", 10),
        ("seed", 0),
        ("prereg_ok", True),
        ("prereg_note", ""),
    ):
        object.__setattr__(agg, field, value)
    return instrument_table(agg)


def test_one_row_per_suite_criterion(diag_dir: Path) -> None:
    t = _table(diag_dir)
    keys = {(r["suite"], r["criterion"]) for r in t.rows}
    assert keys == {("drgym", "keypoint"), ("musique", "support")}


def test_the_disqualified_judge_is_visible_with_its_reason(diag_dir: Path) -> None:
    t = _table(diag_dir)
    row = next(r for r in t.rows if r["suite"] == "drgym")
    assert row["usable"] is False
    assert "position bias" in row["why"]
    assert row["position_bias"] == pytest.approx(0.31)


def test_parse_failures_are_counted_not_hidden(diag_dir: Path) -> None:
    """Accuracy among parseable answers is not accuracy."""
    t = _table(diag_dir)
    assert any("7" in n and "parse" in n.lower() for n in t.notes), t.notes


def test_unjudged_runs_are_reported(diag_dir: Path) -> None:
    t = _table(diag_dir)
    assert any("unjudged" in n.lower() and "2" in n for n in t.notes), t.notes


def test_an_unmeasured_sigma_j_warns_that_nothing_is_reportable(diag_dir: Path) -> None:
    """Until sigma_J is measured, every judge-derived effect sits behind an infinite floor.

    That is correct conservative behaviour and it must be STATED, not discovered by a
    reader wondering why a column is empty.
    """
    t = _table(diag_dir, sigma_j={})
    assert any("sigma_j" in w.lower() for w in t.warnings), t.warnings
    assert any("not reportable" in w.lower() for w in t.warnings), t.warnings


def test_a_measured_sigma_j_is_shown_and_does_not_warn(diag_dir: Path) -> None:
    t = _table(diag_dir, sigma_j={"kpr_incremental": 0.08})
    assert not any("sigma_j" in w.lower() for w in t.warnings), t.warnings


def test_krippendorff_nan_is_explained_not_blank(diag_dir: Path) -> None:
    """It is structurally NaN under a single judge family, which is a design fact."""
    t = _table(diag_dir)
    assert any("krippendorff" in n.lower() for n in t.notes), t.notes
    assert any("famil" in n.lower() for n in t.notes), t.notes


def test_the_table_is_registered(diag_dir: Path) -> None:
    from pi_eval.report import ALL_TABLES

    assert "T6_instrument" in ALL_TABLES


def test_a_missing_diagnostics_file_is_an_empty_table_not_a_crash(tmp_path: Path) -> None:
    """Judging is skippable, so the file legitimately may not exist."""
    t = _table(tmp_path)
    assert t.rows == ()
    assert any("no judge" in w.lower() or "not run" in w.lower() for w in t.warnings), t.warnings
