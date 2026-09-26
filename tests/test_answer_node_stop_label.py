"""`answer_node_covered_before`: the answer-bearing node's own evidence, before a decision.

WHY THE FIELD EXISTS. Lane L1.11 (`artifacts/answer_node_coverage_20260918/RESULT.md`) measured
that the trained policy's required-evidence coverage gain does not reach the node that names the
answer: conditional on that node being uncovered it stops anyway on 0.796 / 0.968 / 1.00 of
units (musique / strategyqa / wiki2) against the prompted arm's 0.255 / 0.390 / 0.571, and 57.9%
of its musique matched-cost coverage gain lands on non-answer nodes. The SFT STOP label is
derived from `done_before` -- POOLED required-evidence completeness -- so the corpus can only
ever say "stop when everything is in hand". This field is the per-node quantity a stop rule can
condition on instead.

WHAT "MIRRORING `done_before`'S CONVENTION EXACTLY" MEANS HERE, AND THE ONE CHOICE IT FORCED.
`done_before` is `_done(potential[i])`: required-evidence coverage at the prefix before decision
i, thresholded at 1.0. The per-node mirror is therefore EVIDENCE IN HAND, uid containment --
NOT the matcher's `match_kind in (resolve, use)`, which is the instrument L1.11 used for its own
terminal-state measurement. Two different questions: "did the policy hold this node's evidence
when it decided" (a property of the state, which is what an SFT label must be, and what every
candidate at one state shares by construction) against "did some question resolve this node"
(a property of the trajectory). The rule that PICKS the node is L1.11's, reused unchanged from
`pi_eval.answer_node`; only the coverage test is stated in `done_before`'s terms.

THE CONSEQUENCE, WHICH IS A RESULT AND NOT A DETAIL, and is pinned by
`test_done_before_true_forces_the_answer_node_covered`: an answer node is a REQUIRED node, so
its gold evidence uids are a subset of the required uid set, so `done_before is True` implies
`answer_node_covered_before is True`. Conditioning STOP on the answer node can therefore only
ADD stop labels, never remove one. Any lane reporting the label shift has to report it in that
direction.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
from tests.test_train_cli import _synth, _turns  # the row builder's own fixtures

from pi_run import cmd_train
from pi_run.cmd_train import SuiteCache, rows_from_run

FIELD = "answer_node_covered_before"
GOLDEN = Path(__file__).parent / "fixtures" / "decision_rows_before_answer_node.json"


def _write_run(root: Path, units, *, task: str, n_turns: int) -> Path:
    from pinq.types import Evidence

    d = root / "runs" / "cand"
    d.mkdir(parents=True, exist_ok=True)
    turns = _turns(units[:n_turns])
    turns[-1]["subset_hash_after"] = Evidence.of(tuple(units[:n_turns])).subset_hash
    d.joinpath("manifest.json").write_text(
        json.dumps(
            {
                "run_id": "cand",
                "suite_id": "synth",
                "task_id": task,
                "arm_id": "inquirer_prompted",
                "split": "train",
                "template_id": None,
                "branch_of_run_id": None,
                "branch_turn_idx": None,
                "budget_cap": 8,
                "max_turns": 16,
            }
        )
    )
    d.joinpath("turns.jsonl").write_text("\n".join(json.dumps(t) for t in turns))
    d.joinpath("ledger.jsonl").write_text(
        json.dumps({"currency": "retrieval_calls", "cumulative": float(n_turns)})
    )
    d.joinpath("status.json").write_text(
        json.dumps({"stop_reason": "policy_stop", "wall_ms": 5, "usage": {"tok_total": 50}})
    )
    d.joinpath("outcome.json").write_text(json.dumps({"answer": {"text": "some answer"}}))
    return d


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("PI_GOLD_ROOT", str(tmp_path / "data" / "gold"))
    suite, tid, units = _synth(tmp_path)
    return tmp_path, tid, units, cmd_train._train("reward").RewardWeights()


# --------------------------------------------------------------- the field, on real rows


def test_one_row_with_the_answer_node_covered_and_one_without(env):
    """The two cases the field exists to separate, on a run where `done_before` says neither.

    The synth fixture's gold answer is `V001 V011` -- two facet values, so no single node's
    evidence contains it and L1.11's RULE 1 finds nothing; the depth fallback fires and the
    answer-node set is the two depth-1 nodes. `_turns` retrieves the units in pool order and
    `units[0]` IS one of those two nodes' evidence, so the answer node goes from uncovered to
    covered across the first turn while pooled coverage only reaches 0.25 of 4 required spans.
    That is the point: the field must be able to disagree with `done_before`, or it is a second
    name for it.
    """
    root, tid, units, weights = env
    d = _write_run(root, units, task=tid, n_turns=2)
    rows = rows_from_run(d, SuiteCache(root), graph_version="v1", weights=weights)

    assert [r["turn_idx"] for r in rows] == [0, 1, 2]
    assert rows[0][FIELD] is False, "nothing retrieved yet: the answer node cannot be covered"
    assert rows[1][FIELD] is True, "turn 0 retrieved an answer node's only gold span"
    assert rows[2][FIELD] is True, "the recorded STOP decided on that same evidence"
    # ...and it is NOT a restatement of pooled completeness on any of the three.
    assert [r["done_before"] for r in rows] == [False, False, False]
    assert [r["coverage_before"] for r in rows] == [0.0, 0.25, 0.5]


def test_done_before_true_forces_the_answer_node_covered(env):
    """An answer node is a REQUIRED node, so `done_before` implies this field.

    Pinned as a test because it fixes the DIRECTION of the label shift for every consumer: a
    STOP rule keyed on the answer node is a superset of the `done_before` rule, so re-labelling
    can add STOP rows and can never turn one into an ASK. A lane that reported "n rows moved
    STOP -> ASK" without this invariant in front of it would be reporting an impossibility.
    """
    root, tid, units, weights = env
    graph = cmd_train._graphs("synth", "v1")[tid]
    by_uid = {u.uid: u for u in units}
    # Shallow required spans first, so the run passes through "answer node covered, pooled
    # coverage still incomplete" before it reaches completeness. `units` in pool order would
    # not: three of its first four are distractors and coverage never reaches 1.0.
    ordered = [
        by_uid[uid]
        for n in sorted(graph.required(), key=lambda n: (n.gold_depth or 0, n.gold_node_id))
        for uid in n.gold_ev_uids
    ]
    d = _write_run(root, ordered, task=tid, n_turns=len(ordered))
    rows = rows_from_run(d, SuiteCache(root), graph_version="v1", weights=weights)

    done = [r for r in rows if r["done_before"]]
    assert done, "fixture must reach pooled completeness or this proves nothing"
    assert all(r[FIELD] is True for r in done)
    # The converse must NOT hold, or the two fields are the same field.
    assert any(r[FIELD] is True and r["done_before"] is False for r in rows)


def test_the_field_is_absent_not_false_when_no_answer_node_can_be_identified(env, monkeypatch):
    """No identifiable answer node is UNKNOWN, and unknown is not `False`.

    `False` would assert "the answer evidence is missing", which is a claim about the state; the
    truth is a claim about the GRAPH -- it names no node whose evidence could be checked. A
    consumer reading `False` would label ASK there on a state where nothing is known, which is
    the same asymmetry `done_before is None` already encodes ("Unknown is not 'not done'").

    Built by doctoring the graph `rows_from_run` reads so its answer nodes carry no gold spans.
    The SCORER keeps the real graph -- `pi_run.serve.score._gold_uids` loads its own -- so
    `coverage_before` and `done_before` below are the undoctored quantities, which is what makes
    this a test of the answer-node branch alone.
    """
    root, tid, units, weights = env
    d = _write_run(root, units, task=tid, n_turns=2)
    real = cmd_train._graphs("synth", "v1")[tid]
    deepest = max(n.gold_depth for n in real.required() if n.gold_depth is not None)
    stripped = dataclasses.replace(
        real,
        gold_nodes=tuple(
            dataclasses.replace(n, gold_ev_uids=()) if n.gold_depth == deepest else n
            for n in real.gold_nodes
        ),
    )
    monkeypatch.setattr(cmd_train, "_graphs", lambda suite, gv: {tid: stripped})

    rows = rows_from_run(d, SuiteCache(root), graph_version="v1", weights=weights)
    assert rows, "the run still exports; only the new field is unknown"
    for r in rows:
        assert FIELD not in r, f"{FIELD} present as {r.get(FIELD)!r}; absent is the label"
        assert "done_before" in r, "the pooled field is unaffected by an unidentifiable node"


# --------------------------------------------------------------- nothing else moved


def test_every_other_field_is_byte_identical_to_the_pre_change_export(env):
    """Strip the new key and the rows must serialise to exactly what HEAD produced.

    The golden was captured by `tests/fixtures/make_decision_rows_golden.py` running the
    exporter BEFORE this field existed. Adding a gold-side computation to `rows_from_run` is
    exactly the kind of change that can perturb a neighbour -- a shared `seen` set, a graph
    cache keyed one field short -- without any test noticing, because every downstream number
    would still be a plausible float.
    """
    root, tid, units, weights = env
    d = _write_run(root, units, task=tid, n_turns=2)
    rows = rows_from_run(d, SuiteCache(root), graph_version="v1", weights=weights)

    got = [{k: v for k, v in r.items() if k != FIELD} for r in rows]
    want = json.loads(GOLDEN.read_text())
    assert json.dumps(got, sort_keys=True, indent=2) == json.dumps(want, sort_keys=True, indent=2)


def test_the_golden_would_fail_if_the_new_key_were_left_in(env):
    """The byte-identity check above is not vacuous: with the key kept, it must NOT match.

    Without this, a bug that dropped the field entirely would leave the golden test green and
    look like proof that nothing changed (`a-permutation-test-needs-a-non-vacuity-check`).
    """
    root, tid, units, weights = env
    d = _write_run(root, units, task=tid, n_turns=2)
    rows = rows_from_run(d, SuiteCache(root), graph_version="v1", weights=weights)
    want = json.loads(GOLDEN.read_text())
    assert json.dumps(rows, sort_keys=True) != json.dumps(want, sort_keys=True)


# --------------------------------------------------------------- the dataset variant


def _sft_row(run_id: str, value: float, question: str, *, task="t1", done, answer, is_stop=False):
    from pinq.actions import STOP_ACTION_JSON, ask_action_json

    r = {
        "suite_id": "musique",
        "task_id": task,
        "template_id": None,
        "run_id": run_id,
        "turn_idx": 1,
        "state_text": f"S-{task}",
        "action_json": STOP_ACTION_JSON if is_stop else ask_action_json(question),
        "value": value,
        "phi_tilde": value,
        "scorer_hash": "sh",
        "graph_version": "v1",
        "branch_of_run_id": "p",
        "branch_turn_idx": 1,
        "frontier_size": 1,
        "is_stop": is_stop,
        "done_before": done,
        "coverage_before": 1.0 if done else 0.5,
    }
    if answer is not None:  # ABSENT, not False, is how an unknown answer node reads
        r["answer_node_covered_before"] = answer
    return r


def _labels(rows, stop_label):
    from pinq_train.export.dataset import export_sft

    ex, man = export_sft(rows, margin_threshold=0.2, stop_label=stop_label)
    return {(e.suite_id, e.task_id): e for e in ex}, man


def test_the_variant_stops_where_only_the_answer_node_is_covered():
    """The state the whole lane is about: answer evidence in hand, pooled coverage incomplete.

    v1 teaches the best ASK here, because `done_before` is False. The variant teaches STOP,
    because the node that names the answer is already covered and every remaining question is
    a prerequisite the task no longer needs.
    """
    rows = [_sft_row("a", 0.9, "who?", done=False, answer=True)]
    v1, man1 = _labels(rows, "done_before")
    v2, man2 = _labels(rows, "answer_node_covered_before")

    assert v1[("musique", "t1")].is_stop is False
    assert v1[("musique", "t1")].label_rule == "ask_clears_floor"
    assert v2[("musique", "t1")].is_stop is True
    assert v2[("musique", "t1")].label_rule == "stop_answer_node_covered"
    # Both rows carry BOTH gold facts, so either file can be audited on its own.
    for e in (v1[("musique", "t1")], v2[("musique", "t1")]):
        assert e.done_before is False and e.answer_node_covered_before is True
    assert man1.sft_stop_label == "done_before"
    assert man2.sft_stop_label == "answer_node_covered_before"
    assert man2.n_stop_done_before_dedupe == 1, "the STOP counter counts STOP rows either way"


def test_the_variant_never_turns_a_done_state_into_an_ask():
    """The superset property, at the exporter rather than at the row builder.

    `done_before is True` can only arrive with `answer_node_covered_before is True` (the row
    builder's invariant), so a v1 STOP survives relabelling. The one case that does NOT is a
    done state whose answer node is UNKNOWN -- tested separately below, and the only way the
    shift can ever run STOP -> ASK.
    """
    rows = [_sft_row("a", 0.9, "who?", done=True, answer=True)]
    v1, _ = _labels(rows, "done_before")
    v2, _ = _labels(rows, "answer_node_covered_before")
    assert v1[("musique", "t1")].is_stop is True and v2[("musique", "t1")].is_stop is True
    assert v1[("musique", "t1")].label_rule == "stop_done"
    assert v2[("musique", "t1")].label_rule == "stop_answer_node_covered"


def test_an_unknown_answer_node_is_never_taught_as_a_stop_and_is_counted():
    """Unknown falls through to the ASK branch, exactly as an unknown `done_before` does.

    This is the ONLY direction in which the variant can move a row STOP -> ASK, so the count
    that reports it (`n_answer_node_unknown`) has to be on both manifests or a lane comparing
    them cannot tell a relabelled state from a state one file never had.
    """
    rows = [_sft_row("a", 0.9, "who?", done=True, answer=None)]
    v1, man1 = _labels(rows, "done_before")
    v2, man2 = _labels(rows, "answer_node_covered_before")

    assert v1[("musique", "t1")].is_stop is True, "v1 reads `done_before` and is unaffected"
    assert v2[("musique", "t1")].is_stop is False, "unknown is not covered, and not a STOP"
    assert v2[("musique", "t1")].label_rule == "ask_clears_floor"
    assert man1.n_answer_node_unknown == 1 and man2.n_answer_node_unknown == 1
    assert v2[("musique", "t1")].answer_node_covered_before is None


def test_the_default_label_reproduces_the_rule_of_record_row_for_row():
    """Carrying the new field must not perturb the v1 labels on a mixed population.

    Four states spanning every combination gold can produce. The `done_before` export must be
    exactly what it was before the field existed -- same targets, same counters -- or the
    contrast between the two files is confounded with a change to the control.
    """
    rows = [
        _sft_row("a", 0.9, "q1?", task="done_covered", done=True, answer=True),
        _sft_row("b", 0.9, "q2?", task="notdone_covered", done=False, answer=True),
        _sft_row("c", 0.9, "q3?", task="notdone_open", done=False, answer=False),
        _sft_row("d", 0.9, "q4?", task="notdone_unknown", done=False, answer=None),
    ]
    v1, man1 = _labels(rows, "done_before")
    v2, man2 = _labels(rows, "answer_node_covered_before")

    assert {t: e.is_stop for (_s, t), e in v1.items()} == {
        "done_covered": True,
        "notdone_covered": False,
        "notdone_open": False,
        "notdone_unknown": False,
    }
    assert {t: e.is_stop for (_s, t), e in v2.items()} == {
        "done_covered": True,
        "notdone_covered": True,  # the only row that moved
        "notdone_open": False,
        "notdone_unknown": False,
    }
    # The SUPERSET property, stated over the whole population rather than one row.
    stops1 = {k for k, e in v1.items() if e.is_stop}
    stops2 = {k for k, e in v2.items() if e.is_stop}
    assert stops1 <= stops2 and stops2 - stops1 == {("musique", "notdone_covered")}
    assert man1.n_stop_done_before_dedupe == 1 and man2.n_stop_done_before_dedupe == 2
    assert man1.n_answer_node_unknown == man2.n_answer_node_unknown == 1


def test_a_state_whose_candidates_disagree_about_the_answer_node_is_refused_on_both_labels():
    """Same treatment as `n_state_done_disagree`, and refused on BOTH files or the two
    populations differ and the label shift is not a label shift."""
    rows = [
        _sft_row("a", 0.9, "q?", done=False, answer=True),
        _sft_row("b", 0.1, "r?", done=False, answer=False),
    ]
    v1, man1 = _labels(rows, "done_before")
    v2, man2 = _labels(rows, "answer_node_covered_before")
    assert v1 == {} and v2 == {}
    assert man1.n_answer_node_state_disagree == 1 and man2.n_answer_node_state_disagree == 1


def test_an_unrecognised_stop_label_is_refused():
    from pinq_train.export.dataset import export_sft

    with pytest.raises(ValueError, match="expected one of"):
        export_sft([], margin_threshold=0.2, stop_label="answer_node")


def test_the_cli_choices_are_the_exporters_own():
    """`pi train export --sft-stop-label`'s choices, joined to the exporter's own tuple.

    `pi_run` may not import `pinq_train` (contract 4), so the names are spelled twice; the
    dynamic loader is what lets a test check that the two spellings agree. A choice added on
    one side only would be refused at the CLI or, worse, accepted and silently stamped.
    """
    from pi_run.cli import build_parser

    dataset = cmd_train._train("export.dataset")
    train = build_parser()._subparsers._group_actions[0].choices["train"]
    export = train._subparsers._group_actions[0].choices["export"]
    action = next(a for a in export._actions if a.dest == "sft_stop_label")
    assert tuple(action.choices) == tuple(dataset.SFT_STOP_LABELS)
    assert action.default == "done_before", "the rule of record stays the default"
