"""Lane L1.11 tooling (`scripts/answer_node_coverage/`).

Hermetic, same discipline as `tests/test_stopping_answer_test.py`: every fixture here is
synthetic. `answer_node.py` and `corpus_text.py` need no gold root or isolated store at all
(pure dataclass construction, or a tmp_path corpus file); `lib.py`'s DB-facing functions get an
in-memory duckdb connection with hand-built tables, never `artifacts/testsplit_qa` (untracked,
main-checkout-only, absent from a worktree).

CLAUDE.md rule 2: `test_coverage_gain_share_additivity_holds_when_a_group_is_empty` exists
because building `coverage_gain_share` against the real store found exactly this bug --
`shared_tasks = set(mc_answer) & set(mc_nonanswer)` silently dropped every task whose
depth-tied answer-node fallback leaves `nonanswer_group` empty (86 of 200 on StrategyQA), so
the reconstructed total disagreed with the direct, single-group computation by as much as
0.014 on StrategyQA before the fix. The fix (a zero-size group contributes a 0-weighted term,
never an excluded task) is asserted here on a minimal fixture, not just re-observed on the
real store.
"""

from __future__ import annotations

import json

import duckdb
import pytest
from scripts.answer_node_coverage import answer_node, corpus_text, lib

from pi_eval.gold import GoldGraph, GoldNode

# --------------------------------------------------------------------------- answer_node.py


def _node(node_id, text, *, partition="required", depth=None, ev_uids=()):
    return GoldNode(
        gold_suite="s",
        gold_task_key="t",
        gold_node_id=node_id,
        gold_text=text,
        gold_partition=partition,
        gold_depth=depth,
        gold_ev_uids=tuple(ev_uids),
    )


def _graph(nodes, *, answer, aliases=()):
    return GoldGraph(
        gold_suite="s",
        gold_task_key="t",
        gold_nodes=tuple(nodes),
        gold_answer=answer,
        gold_aliases=tuple(aliases),
    )


def test_evidence_hit_rule_picks_the_node_whose_evidence_contains_the_answer():
    # Mirrors the real musique example this rule was built against: node text is a
    # sub-question naming neither the answer, evidence text does.
    s1 = _node("s1", "Who manufactures the accessory?", depth=0, ev_uids=("u1",))
    s2 = _node("s2", "when did #1 become public", depth=1, ev_uids=("u2",))
    graph = _graph([s1, s2], answer="1980")
    uid_text = {"u1": "Some unrelated company history.", "u2": "It went public in 1980."}
    res = answer_node.answer_nodes_for_task(graph, uid_text)
    assert res.node_ids == ("s2",)
    assert res.rule == answer_node.RULE_EVIDENCE_HIT


def test_boolean_answer_never_uses_the_evidence_rule_even_on_a_coincidental_hit():
    # s1's evidence contains the literal word "yes" -- a real hit under the naive rule -- but
    # the boolean branch must not even try it, per the module's own reasoning (a coincidental
    # "yes" in prose is not evidence the question was settled there).
    s1 = _node("s1", "fact A", depth=0, ev_uids=("u1",))
    s2 = _node("s2", "fact B", depth=0, ev_uids=("u2",))
    graph = _graph([s1, s2], answer="yes", aliases=("true", "yes"))
    uid_text = {"u1": "yes, this happened in the record.", "u2": "unrelated text."}
    res = answer_node.answer_nodes_for_task(graph, uid_text)
    assert res.rule == answer_node.RULE_BOOLEAN_FALLBACK
    assert set(res.node_ids) == {"s1", "s2"}  # tied at depth 0: both, per "any covered"


def test_no_evidence_hit_falls_back_to_deepest_required_node():
    s1 = _node("s1", "prereq", depth=0, ev_uids=("u1",))
    s2 = _node("s2", "final hop", depth=1, ev_uids=("u2",))
    graph = _graph([s1, s2], answer="42")
    uid_text = {"u1": "no numbers here", "u2": "still no numbers here"}
    res = answer_node.answer_nodes_for_task(graph, uid_text)
    assert res.node_ids == ("s2",)
    assert res.rule == answer_node.RULE_NO_HIT_FALLBACK
    assert res.max_depth == 1


def test_orphaned_required_nodes_fall_back_to_all_required():
    s1 = _node("s1", "a", depth=None, ev_uids=("u1",))
    s2 = _node("s2", "b", depth=None, ev_uids=("u2",))
    graph = _graph([s1, s2], answer="42")
    res = answer_node.answer_nodes_for_task(graph, {"u1": "x", "u2": "y"})
    assert set(res.node_ids) == {"s1", "s2"}
    assert res.rule == answer_node.RULE_ALL_REQUIRED_FALLBACK


def test_no_required_nodes_returns_empty():
    s1 = _node("s1", "a", partition="optional", depth=0)
    graph = _graph([s1], answer="42")
    res = answer_node.answer_nodes_for_task(graph, {})
    assert res.is_empty
    assert res.rule == answer_node.RULE_NO_REQUIRED_NODES


def test_missing_evidence_uid_is_counted_not_raised():
    s1 = _node("s1", "a", depth=0, ev_uids=("u1", "u-missing"))
    graph = _graph([s1], answer="zzz")
    res = answer_node.answer_nodes_for_task(graph, {"u1": "no match here"})
    assert res.n_evidence_uids_unresolved == 1


def test_gold_uid_partition_is_disjoint_and_reunites_to_all_required_ev_uids():
    s1 = _node("s1", "a", depth=0, ev_uids=("u1", "u2"))
    s2 = _node("s2", "b", depth=1, ev_uids=("u3",))
    s3 = _node("s3", "c", partition="optional", depth=1, ev_uids=("u-optional",))
    graph = _graph([s1, s2, s3], answer="x")
    ans, non = answer_node.gold_uid_partition(graph, ["s2"])
    assert ans == frozenset({"u3"})
    assert non == frozenset({"u1", "u2"})
    assert not (ans & non)
    all_required_uids = frozenset(u for n in (s1, s2) for u in n.gold_ev_uids)
    assert ans | non == all_required_uids  # optional node's uid never enters either side


def test_rule_tally_counts_every_result_exactly_once():
    results = {
        "t1": answer_node.AnswerNodeResult(("a",), answer_node.RULE_EVIDENCE_HIT, None, 0),
        "t2": answer_node.AnswerNodeResult(("a",), answer_node.RULE_EVIDENCE_HIT, None, 0),
        "t3": answer_node.AnswerNodeResult(("a",), answer_node.RULE_BOOLEAN_FALLBACK, 0, 0),
    }
    assert answer_node.rule_tally(results) == {
        answer_node.RULE_EVIDENCE_HIT: 2,
        answer_node.RULE_BOOLEAN_FALLBACK: 1,
    }


# --------------------------------------------------------------------------- corpus_text.py


def test_uid_text_map_round_trips_through_unit_uid(tmp_path):
    suite_dir = tmp_path / "musique" / "abc123"
    suite_dir.mkdir(parents=True)
    rec = {
        "id": "task-1",
        "paragraphs": [
            {"idx": 0, "title": "T0", "text": "paragraph zero text"},
            {"idx": 1, "title": "T1", "text": "paragraph one text, the real evidence"},
        ],
    }
    (suite_dir / "tasks.jsonl").write_text(json.dumps(rec) + "\n")

    maps = corpus_text.uid_text_maps_for_suite(tmp_path, "musique", "abc123", ["task-1"])
    uid_text = maps["task-1"]
    assert len(uid_text) == 2
    assert set(uid_text.values()) == {
        "paragraph zero text",
        "paragraph one text, the real evidence",
    }

    # the SAME uid convention the gold builder used must resolve the SAME text back.
    from pi_eval.build.common import unit_uid

    want_uid = unit_uid(corpus_text.CORPUS_ID["musique"], "task-1", 1, rec["paragraphs"][1]["text"])
    assert uid_text[want_uid] == "paragraph one text, the real evidence"


def test_load_task_paragraphs_missing_file_returns_empty(tmp_path):
    assert corpus_text.load_task_paragraphs(tmp_path, "musique", "nope") == {}


def test_uid_text_maps_restricted_to_requested_task_ids(tmp_path):
    suite_dir = tmp_path / "wiki2" / "d"
    suite_dir.mkdir(parents=True)
    lines = [
        json.dumps({"id": "keep", "paragraphs": [{"idx": 0, "title": "a", "text": "x"}]}),
        json.dumps({"id": "drop", "paragraphs": [{"idx": 0, "title": "b", "text": "y"}]}),
    ]
    (suite_dir / "tasks.jsonl").write_text("\n".join(lines) + "\n")
    maps = corpus_text.uid_text_maps_for_suite(tmp_path, "wiki2", "d", ["keep"])
    assert set(maps) == {"keep"}


# --------------------------------------------------------------------------- lib.py: synthetic store

SCORER = "testscorer"


def _store():
    """An in-memory duckdb connection with `runs`, `turns`, `scores`, `matches`, `evidence` --
    the five tables `lib.py`'s functions read, built by hand rather than via `open_store`
    (which requires real parquet files on disk)."""
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE runs (run_id VARCHAR, suite_id VARCHAR, task_id VARCHAR, seed INTEGER, "
        "n_asks INTEGER, stop_reason VARCHAR, corpus_dir VARCHAR)"
    )
    con.execute("CREATE TABLE turns (run_id VARCHAR, turn_idx INTEGER, retrieved_uids VARCHAR[])")
    con.execute(
        "CREATE TABLE scores (run_id VARCHAR, metric_name VARCHAR, scorer_hash VARCHAR, "
        "value DOUBLE)"
    )
    con.execute("CREATE TABLE matches (run_id VARCHAR, node_id VARCHAR, match_kind VARCHAR)")
    con.execute("CREATE TABLE evidence (run_id VARCHAR, uid VARCHAR)")
    return con


def _run(run_id, suite, task, seed, n_asks, stop_reason):
    return {
        "run_id": run_id,
        "suite_id": suite,
        "task_id": task,
        "seed": seed,
        "n_asks": n_asks,
        "stop_reason": stop_reason,
    }


def test_covered_map_reads_resolve_and_use_and_none_for_no_answer_node():
    con = _store()
    con.execute(
        "INSERT INTO matches VALUES "
        "('r1','ans','resolve'), ('r1','other','none'), "
        "('r2','ans','ask'), "  # ask is below the RESOLVE bar -- not covered
        "('r3','ans','use')"
    )
    runs = [
        _run("r1", "s", "t1", 0, 1, "policy_stop"),
        _run("r2", "s", "t1", 1, 1, "policy_stop"),
        _run("r3", "s", "t2", 0, 1, "policy_stop"),
        _run("r4", "s", "t3", 0, 1, "policy_stop"),  # no answer node at all for t3
    ]
    answer_nodes = {
        ("s", "t1"): answer_node.AnswerNodeResult(("ans",), "x", None, 0),
        ("s", "t2"): answer_node.AnswerNodeResult(("ans",), "x", None, 0),
        ("s", "t3"): answer_node.AnswerNodeResult((), answer_node.RULE_NO_REQUIRED_NODES, None, 0),
    }
    covered = lib.covered_map(con, runs, answer_nodes)
    assert covered["r1"] is True
    assert covered["r2"] is False  # 'ask' does not count
    assert covered["r3"] is True  # 'use' counts too
    assert covered["r4"] is None  # no answer node to have covered


def test_p_covered_delta_point_estimate_and_level():
    # p_covered_delta takes no `con` -- it operates purely on run lists and the covered map.
    trained = [_run("t1", "s", "A", 0, 1, "policy_stop"), _run("t2", "s", "B", 0, 1, "policy_stop")]
    prompted = [
        _run("p1", "s", "A", 0, 1, "policy_stop"),
        _run("p2", "s", "B", 0, 1, "policy_stop"),
    ]
    covered = {"t1": True, "t2": False, "p1": False, "p2": False}
    out = lib.p_covered_delta(trained, prompted, covered, n_boot=50)
    assert out["level_trained"] == pytest.approx(0.5)
    assert out["level_prompted"] == pytest.approx(0.0)
    assert out["point"] == pytest.approx(0.5)
    assert out["n_tasks"] == 2


def test_p_stop_given_uncovered_delta_excludes_covered_runs_from_the_denominator():
    trained = [
        _run("t1", "s", "A", 0, 1, "policy_stop"),  # covered -- excluded
        _run("t2", "s", "B", 0, 1, "policy_stop"),  # uncovered, stopped
        _run("t3", "s", "C", 0, 1, "budget"),  # uncovered, forced (not a policy stop)
    ]
    prompted = [
        _run("p1", "s", "A", 0, 1, "policy_stop"),
        _run("p2", "s", "B", 0, 1, "budget"),
        _run("p3", "s", "C", 0, 1, "budget"),
    ]
    covered = {"t1": True, "t2": False, "t3": False, "p1": False, "p2": False, "p3": False}
    stop_reason = {r["run_id"]: r["stop_reason"] for r in [*trained, *prompted]}
    out = lib.p_stop_given_uncovered_delta(trained, prompted, covered, stop_reason, n_boot=50)
    assert out["n_uncovered_trained"] == 2  # t2, t3 (t1 excluded: covered)
    assert out["n_uncovered_prompted"] == 3  # p1, p2, p3 (none covered)
    assert out["level_trained"] == pytest.approx(1 / 2)  # only t2 stopped voluntarily
    assert out["level_prompted"] == pytest.approx(1 / 3)  # only p1


def test_retrieved_uids_by_run_unions_turns_and_evidence():
    con = _store()
    con.execute("INSERT INTO turns VALUES ('r1', 0, ['a','b']), ('r1', 1, ['b','c'])")
    con.execute("INSERT INTO evidence VALUES ('r1','c'), ('r1','d')")
    out = lib.retrieved_uids_by_run(con, ["r1"])
    assert out["r1"] == {"a", "b", "c", "d"}


def test_span_hit_counts_and_cross_check_agree_with_stored_evidence_coverage():
    con = _store()
    con.execute("INSERT INTO turns VALUES ('r1', 0, ['ua','un'])")  # both gold spans retrieved
    con.execute("INSERT INTO scores VALUES ('r1', 'evidence_coverage', ?, 1.0)", [SCORER])
    runs = [_run("r1", "s", "t1", 0, 1, "policy_stop")]
    gold_partition = {("s", "t1"): (frozenset({"ua"}), frozenset({"un"}))}
    hit_counts = lib.span_hit_counts(con, runs, gold_partition)
    assert hit_counts["r1"] == {
        "n_answer_hit": 1,
        "n_answer_gold": 1,
        "n_nonanswer_hit": 1,
        "n_nonanswer_gold": 1,
    }
    xcheck = lib.cross_check_evidence_coverage(con, runs, hit_counts, scorer_hash=SCORER)
    assert xcheck == {"n_checked": 1, "n_mismatched": 0, "examples": []}


def test_cross_check_catches_a_real_mismatch():
    con = _store()
    con.execute("INSERT INTO scores VALUES ('r1', 'evidence_coverage', ?, 0.9)", [SCORER])
    runs = [_run("r1", "s", "t1", 0, 1, "policy_stop")]
    hit_counts = {
        "r1": {"n_answer_hit": 1, "n_answer_gold": 1, "n_nonanswer_hit": 1, "n_nonanswer_gold": 1}
    }
    xcheck = lib.cross_check_evidence_coverage(con, runs, hit_counts, scorer_hash=SCORER)
    assert xcheck["n_mismatched"] == 1


# --------------------------------------------------------------------------- node_group_ladder


def test_node_group_ladder_matches_hand_computed_frontier():
    con = _store()
    con.execute("INSERT INTO turns VALUES ('r1', 0, ['ua']), ('r1', 1, ['un'])")
    runs = [_run("r1", "s", "t1", 0, 2, "policy_stop")]
    full_group = {("s", "t1"): frozenset({"ua", "un"})}
    ladder = lib.node_group_ladder(con, runs, full_group)
    assert ladder["r1"] == {0: 0.0, 1: pytest.approx(0.5), 2: pytest.approx(1.0)}


def test_node_group_ladder_absent_for_an_empty_group():
    con = _store()
    con.execute("INSERT INTO turns VALUES ('r1', 0, ['ua'])")
    runs = [_run("r1", "s", "t1", 0, 1, "policy_stop")]
    empty_group = {("s", "t1"): frozenset()}
    ladder = lib.node_group_ladder(con, runs, empty_group)
    assert "r1" not in ladder


def test_matched_cost_gain_for_group_matches_gate_matched_cost_on_the_full_group():
    """`matched_cost_gain_for_group`, handed a ladder reconstructed from `turns`, must reproduce
    `pinq_train.gate._matched_cost`'s own pooled delta computed from an EXPLICIT
    `scores.frontier_q#k` ladder describing the SAME retrieval schedule -- the two instruments
    read different tables and must agree on a case worked out by hand, not just by construction.

    Task A, gold group of 4 uids {g1..g4}: checkpoint retrieves 2 then 2 more (ladder
    0.0/0.5/1.0), baseline retrieves 1 per turn over 2 turns (0.0/0.25/0.5). Checkpoint's own
    k=2, baseline's rung at min(2,2)=2 is 0.5, so task A's delta is 1.0-0.5=0.5. Task B, a
    1-uid group, both arms retrieve nothing: delta 0.0-0.0=0.0. Pooled mean (0.5+0.0)/2=0.25.
    """
    from pinq_train.gate import _matched_cost

    con = duckdb.connect()
    con.execute("CREATE TABLE runs (run_id VARCHAR, n_asks INTEGER, stop_reason VARCHAR)")
    con.execute(
        "CREATE TABLE scores (run_id VARCHAR, metric_name VARCHAR, scorer_hash VARCHAR, value DOUBLE)"
    )
    con.execute("CREATE TABLE turns (run_id VARCHAR, turn_idx INTEGER, retrieved_uids VARCHAR[])")

    ck_runs = [
        _run("t-A-s0", "musique", "A", 0, 2, "policy_stop"),
        _run("t-B-s0", "musique", "B", 0, 1, "policy_stop"),
    ]
    ba_runs = [
        _run("p-A-s0", "musique", "A", 0, 2, "policy_stop"),
        _run("p-B-s0", "musique", "B", 0, 1, "policy_stop"),
    ]
    # (run_id, n_asks, per-turn retrieved uids, {k: frontier_q#k}, terminal evidence_coverage)
    fixture = {
        "t-A-s0": (2, [["g1", "g2"], ["g3", "g4"]], {0: 0.0, 1: 0.5, 2: 1.0}, 1.0),
        "p-A-s0": (2, [["g1"], ["g2"]], {0: 0.0, 1: 0.25, 2: 0.5}, 0.5),
        "t-B-s0": (1, [[]], {0: 0.0, 1: 0.0}, 0.0),
        "p-B-s0": (1, [[]], {0: 0.0, 1: 0.0}, 0.0),
    }
    run_rows, score_rows, turn_rows = [], [], []
    for rid, (n_asks, turns, ladder, cov) in fixture.items():
        run_rows.append((rid, n_asks, "policy_stop"))
        for k, q in ladder.items():
            score_rows.append((rid, f"frontier_q#{k}", SCORER, q))
        score_rows.append((rid, "evidence_coverage", SCORER, cov))
        for t, uids in enumerate(turns):
            turn_rows.append((rid, t, uids))
    con.executemany("INSERT INTO runs VALUES (?,?,?)", run_rows)
    con.executemany("INSERT INTO scores VALUES (?,?,?,?)", score_rows)
    con.executemany("INSERT INTO turns VALUES (?,?,?)", turn_rows)

    n_asks_by_run = {r["run_id"]: r["n_asks"] for r in [*ck_runs, *ba_runs]}
    full_group = {
        ("musique", "A"): frozenset({"g1", "g2", "g3", "g4"}),
        ("musique", "B"): frozenset({"g5"}),
    }
    ladders = lib.node_group_ladder(con, [*ck_runs, *ba_runs], full_group)
    assert ladders["t-A-s0"] == {0: 0.0, 1: pytest.approx(0.5), 2: pytest.approx(1.0)}
    assert ladders["p-A-s0"] == {0: 0.0, 1: pytest.approx(0.25), 2: pytest.approx(0.5)}
    by_task = lib.matched_cost_gain_for_group(ck_runs, ba_runs, ladders, n_asks_by_run)
    assert by_task[("musique", "A")] == pytest.approx(0.5)
    assert by_task[("musique", "B")] == pytest.approx(0.0)

    ck_keys = lib.stopping_lib._by_key(ck_runs)
    ba_keys = lib.stopping_lib._by_key(ba_runs)
    official = _matched_cost(
        con,
        ckpt=ck_runs,
        base=ba_runs,
        ck_keys=ck_keys,
        ba_keys=ba_keys,
        scorer_hash=SCORER,
        seed=0,
        n_resamples=50,
    )
    pooled_mean = sum(by_task.values()) / len(by_task)
    assert pooled_mean == pytest.approx(0.25)
    assert pooled_mean == pytest.approx(official["pooled"]["delta"])


def test_coverage_gain_share_additivity_holds_when_a_group_is_empty():
    """THE regression test for the bug found building this against the real store: task `B`'s
    entire required-evidence pool is the answer node (nonanswer_group empty, weight 0 by
    construction -- StrategyQA's depth-tied comparison shape). Before the fix,
    `set(mc_answer) & set(mc_nonanswer)` dropped B entirely from the pooled sum; after it, B
    contributes its full (weighted) share to the answer side and exactly zero to the
    non-answer side, and the two group totals must sum EXACTLY to the direct single-group
    computation on `answer | nonanswer`."""
    con = duckdb.connect()
    con.execute("CREATE TABLE turns (run_id VARCHAR, turn_idx INTEGER, retrieved_uids VARCHAR[])")
    # Task A: answer={a}, nonanswer={n}. Task B: answer={x}, nonanswer={} (empty).
    turn_rows = [
        ("t-A-s0", 0, ["a"]),
        ("t-A-s0", 1, ["n"]),
        ("p-A-s0", 0, ["a"]),
        ("t-B-s0", 0, ["x"]),
        ("p-B-s0", 0, []),
    ]
    con.executemany("INSERT INTO turns VALUES (?,?,?)", turn_rows)

    ck_runs = [
        _run("t-A-s0", "s", "A", 0, 2, "policy_stop"),
        _run("t-B-s0", "s", "B", 0, 1, "policy_stop"),
    ]
    ba_runs = [
        _run("p-A-s0", "s", "A", 0, 1, "policy_stop"),
        _run("p-B-s0", "s", "B", 0, 1, "policy_stop"),
    ]
    hit_counts = {}  # unused by the matched-cost path; span_hit_counts not exercised here
    gold_partition = {
        ("s", "A"): (frozenset({"a"}), frozenset({"n"})),
        ("s", "B"): (frozenset({"x"}), frozenset()),
    }
    out = lib.coverage_gain_share(con, ck_runs, ba_runs, hit_counts, gold_partition, n_boot=20)
    assert out["n_tasks_zero_nonanswer_group"] == 1
    assert out["n_tasks"] == 2
    total = (
        out["pooled_matched_cost_answer_weighted"] + out["pooled_matched_cost_nonanswer_weighted"]
    )
    assert total == pytest.approx(out["pooled_matched_cost_total_direct_check"], abs=1e-9)
    assert total == pytest.approx(out["pooled_matched_cost_total_reconstructed"], abs=1e-9)


def test_coverage_gain_share_share_is_between_zero_and_one_on_a_clean_split():
    """Non-vacuity check: when ALL of a task's gain is on the non-answer node and none on the
    answer node, the share must read 1.0, not some artefact of the weighting."""
    con = duckdb.connect()
    con.execute("CREATE TABLE turns (run_id VARCHAR, turn_idx INTEGER, retrieved_uids VARCHAR[])")
    con.executemany(
        "INSERT INTO turns VALUES (?,?,?)",
        [("t-A-s0", 0, ["n"]), ("p-A-s0", 0, [])],
    )
    ck_runs = [_run("t-A-s0", "s", "A", 0, 1, "policy_stop")]
    ba_runs = [_run("p-A-s0", "s", "A", 0, 1, "policy_stop")]
    gold_partition = {("s", "A"): (frozenset({"a"}), frozenset({"n"}))}
    out = lib.coverage_gain_share(con, ck_runs, ba_runs, {}, gold_partition, n_boot=20)
    assert out["share_nonanswer_of_gain"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- recall-by-coverage-group


def test_task_covered_groups_splits_on_any_seed_covering():
    covered = {"r1": True, "r2": False, "r3": False, "r4": None}
    runs = [
        _run("r1", "s", "A", 0, 1, "policy_stop"),
        _run("r2", "s", "A", 1, 1, "policy_stop"),  # A: one seed covered -> COVERED group
        _run("r3", "s", "B", 0, 1, "policy_stop"),  # B: never covered -> UNCOVERED group
        _run("r4", "s", "C", 0, 1, "policy_stop"),  # C: no answer node -> neither group
    ]
    cov_tasks, unc_tasks = lib._task_covered_groups(runs, covered)
    assert cov_tasks == {"A"}
    assert unc_tasks == {"B"}
