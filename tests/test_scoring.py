def test_a_drifted_parquet_is_refused_before_any_judging(tmp_path):
    """`_append` does `existing.cast(schema)` at the very END of `score()`, long after
    `run_judges` has dispatched and been billed for every judge call.

    The on-disk judgments.parquet carried the pre-rename 15-column schema (`judge_id`,
    `presentation_order`, `winner`, `tie`, `prompt_hash`) against the current 22, so the first
    judged `pi score` would have raised at write time -- after paying for the entire judging
    pass and writing nothing."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pytest

    from pi_eval.score import ParquetSchemaDrift, assert_appendable

    # the exact stale shape that was on disk
    stale = pa.table(
        {
            "judgment_id": pa.array([], pa.string()),
            "winner": pa.array([], pa.string()),
            "tie": pa.array([], pa.bool_()),
        }
    )
    pq.write_table(stale, tmp_path / "judgments.parquet")

    with pytest.raises(ParquetSchemaDrift) as exc:
        assert_appendable(tmp_path, ("judgments",))
    msg = str(exc.value)
    assert "would fail at write time" in msg
    assert "gone from code" in msg and "winner" in msg
    assert "missing from disk" in msg and "criterion" in msg


def test_a_matching_parquet_passes_the_preflight(tmp_path):
    import pyarrow.parquet as pq

    from pi_eval import schema as sch
    from pi_eval.score import assert_appendable

    pq.write_table(sch.empty_table("judgments"), tmp_path / "judgments.parquet")
    assert_appendable(tmp_path, ("judgments",))  # must not raise


def test_an_absent_parquet_is_not_an_error(tmp_path):
    """A first run has no files to append to."""
    from pi_eval.score import assert_appendable

    assert_appendable(tmp_path, ("matches", "scores", "judgments"))


def test_score_runs_the_preflight_before_judging():
    """Ordering is the whole point: after `run_judges` it is worthless."""
    import inspect

    from pi_eval import score

    src = inspect.getsource(score.score)
    code = "\n".join(ln for ln in src.splitlines() if not ln.lstrip().startswith("#"))
    assert "assert_appendable(" in code
    assert code.index("assert_appendable(") < code.index("run_judges("), (
        "the preflight must run BEFORE any judge call is dispatched"
    )
