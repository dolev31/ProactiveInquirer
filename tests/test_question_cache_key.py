"""One rule for locating a run's questions, used by every consumer.

`pi_eval.score.run_judges` keys the question cache on `corpus_dir` (falling back to
`corpus_hash` for runs written before that field existed). `scripts/measure_sigma_j.py`
keyed it on `corpus_hash` ALONE -- a different string over different inputs.

MEASURED on the distinct (suite, corpus_dir, corpus_hash) triples on disk:
`data/corpora/<suite>/<corpus_hash>/tasks.jsonl` exists for 0 of 8, while the
`corpus_dir` form exists for 5 of 8. So `measure_sigma_j` loaded ZERO questions for every
run and graded every item with `question=""`.

That is not a cosmetic divergence. sigma_J is the gate on whether ANY judge-derived metric
may be reported at all, it costs ~13,400 judge calls to measure, and it would have been
measured on question-less prompts -- a wrong number, expensively obtained, gating the
paper's judge-derived claims.
"""

from __future__ import annotations


def test_the_key_prefers_corpus_dir() -> None:
    from pi_eval.score import question_cache_key

    assert question_cache_key({"suite_id": "musique", "corpus_dir": "d", "corpus_hash": "h"}) == (
        "musique",
        "d",
    )


def test_the_key_falls_back_to_corpus_hash_for_pre_field_runs() -> None:
    """Those runs miss, as they always did -- but readably, and counted."""
    from pi_eval.score import question_cache_key

    assert question_cache_key({"suite_id": "s", "corpus_hash": "h"}) == ("s", "h")


def test_an_empty_corpus_dir_does_not_shadow_the_hash() -> None:
    """`corpus_dir` is "" on older manifests, not absent."""
    from pi_eval.score import question_cache_key

    assert question_cache_key({"suite_id": "s", "corpus_dir": "", "corpus_hash": "h"}) == ("s", "h")


def test_measure_sigma_j_uses_the_same_rule_as_the_scorer() -> None:
    """The divergence this file exists to close: one implementation, called by both.

    Checked in the source because the script imports `pi_eval` inside the function -- at
    module scope it would put a `scripts/` -> `pi_eval` edge under the firewall contract.
    Comment lines are stripped first: this repo has three past bugs where a test matched a
    comment and passed while the code said otherwise.
    """
    import pathlib

    src = pathlib.Path("scripts/measure_sigma_j.py").read_text()
    code = "\n".join(ln for ln in src.splitlines() if not ln.lstrip().startswith("#"))
    assert "question_cache_key(r)" in code, "the script re-derives the key again"
    assert 'r.get("corpus_hash")' not in code, "the corpus_hash-only form is back"


def test_the_scorer_still_uses_it() -> None:
    """A shared helper nothing calls is not a fix."""
    import pathlib
    import re

    src = pathlib.Path("src/pi_eval/score.py").read_text()
    code = "\n".join(ln for ln in src.splitlines() if not ln.lstrip().startswith("#"))
    assert re.search(r"key\s*=\s*question_cache_key\(", code), "run_judges stopped using it"


def test_the_corpus_hash_form_resolves_to_nothing_on_disk() -> None:
    """The measurement behind this file, re-run as an assertion.

    Skips when no compacted runs are present; it is a statement about this repo's data, not
    about the code, and it is what makes the fix worth its diff.
    """
    import pytest

    duckdb = pytest.importorskip("duckdb")
    from pathlib import Path

    parquet = Path("scores/parquet/runs.parquet")
    if not parquet.exists():
        pytest.skip("no compacted runs")
    rows = (
        duckdb.connect()
        .execute(
            "SELECT DISTINCT suite_id, corpus_dir, corpus_hash FROM read_parquet(?) "
            "WHERE status = 'ok'",
            [str(parquet)],
        )
        .fetchall()
    )
    root = Path("data/corpora")
    by_hash = sum(1 for su, _, ch in rows if ch and (root / su / ch / "tasks.jsonl").exists())
    by_dir = sum(1 for su, cd, _ in rows if cd and (root / su / cd / "tasks.jsonl").exists())
    assert by_hash == 0, "corpus_hash unexpectedly names a real corpus directory"
    assert by_dir > 0, "corpus_dir resolves nothing either -- the corpora tree moved"
