def test_a_figure_is_refused_when_its_tables_are(tmp_path, monkeypatch):
    """FIGURES BYPASSED THE PROVENANCE GATE ENTIRELY. `pi render --all` refused every table with
    "input hash changed since it was scored" and then overwrote every figure PDF with numbers
    computed from exactly those changed inputs -- and `_write_figure` saved the PDF BEFORE
    computing provenance, so no check could have run even in principle.

    Reproduced end to end: append one row to scores.parquet without re-running `pi agg`, then
    `pi render --all` -> 7 tables refused, "tables": [], and F1_frontier.pdf rewritten. F1 is
    where the frontier's simultaneous band is published, so the artifact that survived the
    refusal is the one that licenses "that dip is noise"."""
    import json

    from pi_run.render import _inputs_moved

    class _Agg:
        parquet_dir = tmp_path / "pq"

    _Agg.parquet_dir.mkdir()
    (_Agg.parquet_dir / "scores.parquet").write_bytes(b"one")

    tables = tmp_path / "tables"
    (tables / "T1_primary").mkdir(parents=True)

    # a stored provenance that matches -> not stale
    from pi_eval.report import file_hashes
    from pinq.ids import canon, h

    files = file_hashes(_Agg.parquet_dir)
    good = {"inputs_hash": h("inputs", canon(sorted(files.items()))), "inputs": files}
    (tables / "T1_primary" / "provenance.json").write_text(json.dumps(good))
    assert _inputs_moved(_Agg, tables, ["T1_primary"]) == ""

    # now the input moves
    (_Agg.parquet_dir / "scores.parquet").write_bytes(b"two")
    why = _inputs_moved(_Agg, tables, ["T1_primary"])
    assert "input hash changed" in why and "scores.parquet" in why


def test_a_table_from_another_command_does_not_poison_the_figure_gate(tmp_path):
    """Globbing every directory under tables/ refused all three figures on the strength of
    K_killswitch, which `pi report killswitch` writes and `pi render --all` never touches:
    7 of 8 stored provenances matched and the eighth, from a different command, poisoned it."""
    import json

    from pi_eval.report import file_hashes
    from pi_run.render import _inputs_moved
    from pinq.ids import canon, h

    class _Agg:
        parquet_dir = tmp_path / "pq"

    _Agg.parquet_dir.mkdir()
    (_Agg.parquet_dir / "scores.parquet").write_bytes(b"one")
    files = file_hashes(_Agg.parquet_dir)
    tables = tmp_path / "tables"

    (tables / "T1_primary").mkdir(parents=True)
    (tables / "T1_primary" / "provenance.json").write_text(
        json.dumps({"inputs_hash": h("inputs", canon(sorted(files.items()))), "inputs": files})
    )
    (tables / "K_killswitch").mkdir(parents=True)
    (tables / "K_killswitch" / "provenance.json").write_text(
        json.dumps({"inputs_hash": "stale-from-another-command", "inputs": {}})
    )
    assert _inputs_moved(_Agg, tables, ["T1_primary"]) == "", "only the rendered tables count"
    assert _inputs_moved(_Agg, tables, ["T1_primary", "K_killswitch"]) != ""


def test_a_row_field_outside_the_columns_is_not_silently_dropped():
    """The primary endpoint's row carries a `notes` explaining a reduced n -- "excluded 6
    task(s) split across seeds (no consistent direction; coercing them would have scored each a
    loss)" -- and the .tex printed the shrunken n with the reason deleted. A number whose
    denominator changed for a stated reason, published without the reason."""
    from pi_eval.report import Table
    from pi_run.render import to_latex

    t = Table(
        table_id="T1_primary",
        title="Primary",
        columns=("suite", "metric", "n", "delta"),
        rows=[
            {
                "suite": "tau2",
                "metric": "tau_reward",
                "n": 44,
                "delta": 0.08,
                "notes": "excluded 6 tasks split across seeds (no consistent direction)",
                "_run_ids": ["r1", "r2"],
            }
        ],
        provenance={},
        notes=(),
        warnings=(),
    )
    tex = to_latex(t)
    assert "excluded 6 tasks" in tex, "the reason for the reduced n must survive"
    assert "_run_ids" not in tex and "r1" not in tex, "internal plumbing stays out"
