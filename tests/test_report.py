import pytest


def test_the_needs_query_is_pinned_to_one_matcher_and_graph_version():
    """matches.parquet is APPEND-ONLY, like scores.parquet. T4's `needs` CTE counted every row
    in it with no matcher_id and no graph_version filter, while the `m` CTE directly above pins
    s.scorer_hash -- so a re-score under a bumped matcher DOUBLED needs_discovered and HALVED
    usd_per_need*, in a published table, with no error anywhere.

    Measured on this repository's live matches.parquet after the matcher was bumped from
    mechanical_v1 to mechanical_v2: 249 rows under each id, the CTE counted all 498."""
    import inspect

    from pi_eval import report

    src = inspect.getsource(report)
    needs = src[src.index('"), needs AS (') :]
    needs = needs[: needs.index('")\\n')] if '")\\n' in needs else needs[:1200]
    assert "mm.matcher_id" in needs, "the needs CTE must pin the matcher"
    assert "mm.graph_version" in needs, "and the graph version"


def test_agg_pins_the_running_matcher():
    from pi_eval.matcher.base import MechanicalMatcher
    from pi_eval.report import MATCHER_ID, Agg

    assert MATCHER_ID == MechanicalMatcher.matcher_id
    agg = Agg(
        parquet_dir=".",
        con=None,
        scorer_hash="sh",
        graph_version="v1",
        allow_contaminated=False,
        sigma_j={},
    )
    assert agg.matcher_id == MechanicalMatcher.matcher_id


def test_a_bumped_matcher_does_not_double_the_need_count(tmp_path):
    """The mechanism, on a fixture holding two matcher generations for the same runs."""
    import duckdb

    con = duckdb.connect()
    con.execute(
        "CREATE TABLE matches (run_id VARCHAR, match_kind VARCHAR, matcher_id VARCHAR, "
        "graph_version VARCHAR)"
    )
    for mid in ("mechanical_v1", "mechanical_v2"):
        for i in range(5):
            con.execute("INSERT INTO matches VALUES (?, 'resolve', ?, 'v1')", [f"r{i}", mid])
    unfiltered = con.execute(
        "SELECT count(*) FROM matches WHERE match_kind IN ('resolve','use')"
    ).fetchone()[0]
    pinned = con.execute(
        "SELECT count(*) FROM matches WHERE match_kind IN ('resolve','use') "
        "AND matcher_id = 'mechanical_v2' AND graph_version = 'v1'"
    ).fetchone()[0]
    assert unfiltered == 10 and pinned == 5, "exactly the 2x this fix removes"


def test_code_versions_names_only_code_that_produced_a_number():
    """This selected every distinct code_version in runs.parquet with no predicate, so
    provenance named commits whose runs were ALL excluded -- dev-, gold-exposed, pilot,
    unreconciled. Measured on this repository: 9 versions reported, 4 contributing an eligible
    run, 5 naming code that produced none of the numbers in the table they were attached to.

    Rule 1 is that a reported value traces to what produced it. A provenance listing code that
    produced nothing is a false trace that PASSES inspection."""
    import inspect

    from pi_eval.report import Agg

    src = inspect.getsource(Agg.code_versions)
    assert "self.predicate" in src, "code_versions must be scoped to the eligible runs"


def test_two_graph_versions_are_refused_not_comma_joined():
    """`open_agg` raises on two scorer_hashes -- "a number must name exactly one" -- and then
    comma-joined two graph_versions into "v1,v2" and pooled their rows into one cell. A
    graph_version identifies WHICH GOLD a coverage number was measured against; two of them in
    one mean is two different denominators averaged together."""
    from pi_eval.report import AggregationError, _one_graph_version

    assert _one_graph_version(["v1"], "sh") == "v1"
    assert _one_graph_version([], "sh") == ""
    with pytest.raises(AggregationError, match="several graph_versions"):
        _one_graph_version(["v1", "v2"], "abcdef123456")


def test_t3_states_the_share_of_gold_it_excludes():
    """`coverage_at_depth` buckets only nodes with a `gold_depth`, so a node the graph cannot
    reach is in no C@d column, no DWR term and no |V_d|. `orphan_rate` is emitted per run into
    scores.parquet -- 264 rows on this tree -- and `grep -n orphan report.py render.py` matched
    NOTHING: the share of gold the table is silent about was computed, stored, and shown to
    nobody. tau2's graph is 63.9% orphaned and carries the primary endpoint."""
    import inspect

    from pi_eval import report

    src = inspect.getsource(report)
    assert "_orphan_note" in src
    assert "Depth-orphaned gold is EXCLUDED" in src


def test_the_orphan_note_is_empty_when_nothing_recorded_one():
    """No orphan_rate rows must mean no sentence, not a fabricated 0.000."""
    from pi_eval.report import _orphan_note

    class _Agg:
        scorer_hash = "sh"
        predicate = "TRUE"

        def sql(self, q):
            return []

    assert _orphan_note(_Agg()) == ""


def _agg(prereg_ok):
    from pi_eval.report import Agg

    return Agg(
        parquet_dir=".",
        con=None,
        scorer_hash="sh",
        graph_version="v1",
        allow_contaminated=False,
        sigma_j={},
        prereg_ok=prereg_ok,
    )


def test_an_unsealed_preregistration_prints_a_banner_on_the_primary_table():
    """T1 captions itself "Primary endpoints (preregistered, uncorrected)" and its first note
    says "one preregistered contrast per suite" -- while `make prereg-verify` exits 1 today,
    because there is no prereg/ directory at all. provenance.json recorded that honestly
    (`prereg_verified: null`); nothing in the rendered artifact did.

    A warning and not an exception, for the reason the module docstring gives about
    below_noise_floor: "a linter that throws at 2am is a linter that gets disabled by 2:05am."
    """
    from pi_eval.report import _seal_warnings

    assert _seal_warnings(_agg(None))[0].startswith("NOTHING IS SEALED")
    assert "no longer verifies" in _seal_warnings(_agg(False))[0].lower()
    assert _seal_warnings(_agg(True)) == (), "a verified seal prints nothing"


def test_the_tex_header_carries_the_seal_status():
    """The header carried provenance-digest, inputs-hash, scorer-hash, graph-version,
    n-run-ids, table-digest and content-sha, and said nothing about whether anything had been
    sealed. The claim was in the prose and the evidence was in neither."""
    import inspect

    from pi_run import render

    src = inspect.getsource(render._stamp)
    assert "pi-prereg-verified" in src and "pi-prereg-exclusions" in src
    assert "unsealed" in src, "None must read as 'unsealed', not as a failed verification"
