"""A5: was the STOP right? -- the only metric family in this project with no human validation.

`pinq` treats stopping as a decision on the same footing as asking, and `pi_eval` reports
`stop_overshoot`/`stop_undershoot` in question units, but nothing before this asked a PERSON
whether a stop was correct. This file locks down the four things that make A5 safe to ship:

  * it only ever judges a genuine DECISION (`stop_reason == "policy_stop"`), never a cap;
  * the item cannot leak how the run actually scored (`answer_correct`, `evidence_coverage`,
    `stop_reason`, the unresolved-need count all live in the key, never the bundle);
  * `missing` is required exactly when the verdict claims something was missed, both for a
    human record (`validate_records`) and a model reply (`annotate_llm.parse_reply`);
  * agreement is computed per CANDIDATE, not per SET, and the unit never reaches gold.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from pi_eval.annotate import (
    bundle_shape_errors,
    consensus,
    item_set_hash,
    merge_into_graphs,
    validate_records,
)
from pi_eval.annotate_llm import AnnotationParseError, parse_reply
from pi_eval.gold import GoldGraph, GoldNode
from pi_run.cmd_annotate import sample_a5_items

BUNDLE_ID = "synth-0-deadbeef"


# --------------------------------------------------------------------------- shared fixtures


def _node(nid: str, task_key: str = "t1", **kw) -> GoldNode:
    kw.setdefault("gold_partition", "required")
    kw.setdefault("gold_discoverability", "kb")
    kw.setdefault("gold_text", f"need {nid}")
    return GoldNode(gold_suite="synth", gold_task_key=task_key, gold_node_id=nid, **kw)


def _graph(task_key: str = "t1", nodes=(), **kw) -> GoldGraph:
    return GoldGraph(gold_suite="synth", gold_task_key=task_key, gold_nodes=tuple(nodes), **kw)


def _write_run(
    runs: Path,
    run_id: str,
    *,
    task: str,
    suite: str = "synth",
    arm: str = "inquirer_prompted",
    stop_reason: str = "policy_stop",
    answer_text: str = "an answer",
    retrieved_uids: tuple[str, ...] = ("u1",),
    question: str = "the question",
) -> Path:
    """The minimum `sample_a5_items` reads: manifest, one turn, a stop status, an outcome."""
    d = runs / run_id
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(
        json.dumps({"run_id": run_id, "suite_id": suite, "task_id": task, "arm_id": arm})
    )
    turns = [
        {
            "turn_idx": 0,
            "action_kind": "ask",
            "question": question,
            "response_text": "the reply",
            "retrieved_uids": list(retrieved_uids),
            "new_uids": list(retrieved_uids),
        }
    ]
    (d / "turns.jsonl").write_text("\n".join(json.dumps(t) for t in turns))
    (d / "status.json").write_text(json.dumps({"stop_reason": stop_reason}))
    (d / "outcome.json").write_text(
        json.dumps({"answer": {"text": answer_text, "cited_unit_ids": []}})
    )
    return d


def _item(task_type: str, iid: str, *, payload=None, task_key: str = "t1", **prov) -> dict:
    p = {"suite": "synth", "task_id": task_key, "task_key": task_key, "graph_version": "v1"}
    p.update(prov)
    return {
        "item_id": iid,
        "task_type": task_type,
        "context": {},
        "payload": payload or {},
        "provenance": p,
    }


def _bundle(items) -> dict:
    return {
        "manifest": {"bundle_id": BUNDLE_ID, "item_set_hash": item_set_hash(list(items))},
        "items": list(items),
    }


def _rec(iid: str, ann: str, task_type: str, response: dict, **kw) -> dict:
    r = {
        "record_id": f"{BUNDLE_ID}/{iid}/{ann}",
        "bundle_id": BUNDLE_ID,
        "item_id": iid,
        "task_type": task_type,
        "annotator_id": ann,
        "elapsed_ms": 1000,
        "ts": "2026-08-31T12:00:00Z",
        "response": response,
    }
    r.update(kw)
    return r


def _candidates(*node_ids: str) -> list[dict]:
    return [{"node_id": n, "text": f"need {n}"} for n in node_ids]


class _FakeCache:
    """Stands in for `SuiteCache`: only `.view(...).question` and `.units(...)` are read."""

    def __init__(self, question: str = "the task question"):
        self._question = question

    def view(self, suite_id, task_id):
        from types import SimpleNamespace

        return SimpleNamespace(question=self._question)

    def units(self, suite_id, task_id):
        return {}


def _from_rows(rows) -> dict[str, GoldGraph]:
    out = {}
    for r in rows:
        out[r["gold_task_key"]] = GoldGraph(
            gold_suite=r["gold_suite"],
            gold_task_key=r["gold_task_key"],
            gold_nodes=tuple(GoldNode(**_tup(n)) for n in r["gold_nodes"]),
            gold_edges=(),
            gold_graph_version=r["gold_graph_version"],
            gold_answer=r["gold_answer"],
            gold_canary=r["gold_canary"],
            gold_corpus_hash=r["gold_corpus_hash"],
        )
    return out


def _tup(d: dict) -> dict:
    return {k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items()}


# --------------------------------------------------------------------------- sampler: policy_stop only


def test_a5_item_excludes_budget_and_max_turns_runs(tmp_path):
    """`budget` and `max_turns` are CAPS, not choices -- there is no decision to judge where the
    loop was cut off rather than stopped on its own terms."""
    graph = _graph(nodes=(_node("n1", gold_ev_uids=("u1",)),))
    runs = tmp_path / "runs"
    _write_run(runs, "run_policy", task="t1", stop_reason="policy_stop")
    _write_run(runs, "run_budget", task="t1", stop_reason="budget")
    _write_run(runs, "run_maxturns", task="t1", stop_reason="max_turns")
    _write_run(runs, "run_unset", task="t1", stop_reason="")

    stats: dict[str, int] = {}
    items, key = sample_a5_items({"t1": graph}, [], runs, None, random.Random(0), n=10, stats=stats)
    run_ids = {it["provenance"]["run_id"] for it in items}
    assert run_ids == {"run_policy"}, run_ids
    assert stats.get("a5_not_policy_stop") == 3


def test_a5_item_excludes_dev_and_gold_exposed_runs(tmp_path):
    """Same posture as every other sampler: a dirty-tree `dev-` run and a gold-exposed arm are
    not admissible measurements, regardless of what they claim their stop_reason was."""
    from pi_run.cmd_train import GOLD_EXPOSED_ARMS

    graph = _graph(nodes=(_node("n1"),))
    runs = tmp_path / "runs"
    _write_run(runs, "keep0", task="t1", stop_reason="policy_stop")
    _write_run(runs, "dev-abc", task="t1", stop_reason="policy_stop")
    _write_run(
        runs, "ceiling0", task="t1", stop_reason="policy_stop", arm=next(iter(GOLD_EXPOSED_ARMS))
    )

    stats: dict[str, int] = {}
    items, key = sample_a5_items({"t1": graph}, [], runs, None, random.Random(0), n=10, stats=stats)
    run_ids = {it["provenance"]["run_id"] for it in items}
    assert run_ids == {"keep0"}
    assert stats.get("a5_dev_run") == 1
    assert stats.get("a5_gold_exposed") == 1


# --------------------------------------------------------------------------- blinding


def test_a5_item_hides_whether_the_answer_was_correct(tmp_path):
    """`answer_correct`, `evidence_coverage`, `stop_reason` and the unresolved-need count must
    reach the KEY and never the shipped item -- an annotator told the answer was wrong will
    find something missing, which is anchoring, not judgment."""
    graph = _graph(nodes=(_node("n1", gold_ev_uids=("u1",)),), gold_answer="Paris")
    runs = tmp_path / "runs"
    _write_run(
        runs,
        "run0",
        task="t1",
        stop_reason="policy_stop",
        answer_text="a wrong guess",
        retrieved_uids=("u9",),
    )

    items, key = sample_a5_items({"t1": graph}, [], runs, _FakeCache(), random.Random(0), n=5)
    assert len(items) == 1
    it = items[0]

    blob = json.dumps(it)
    for forbidden in ("answer_correct", "evidence_coverage", "stop_reason", "n_unresolved"):
        assert forbidden not in blob, f"{forbidden!r} leaked into the shipped A5 item"

    assert set(it["context"]) == {"question", "evidence", "history", "answer"}
    assert set(it["payload"]) == {"candidates"}
    assert it["context"]["answer"] == "a wrong guess"

    ke = key[it["item_id"]]
    assert ke["stop_reason"] == "policy_stop"
    assert ke["evidence_coverage"] == pytest.approx(0.0)  # u9 retrieved, u1 required: no overlap
    assert ke["answer_correct"] == pytest.approx(0.0)  # "a wrong guess" does not name "Paris"
    assert ke["n_unresolved"] == 1

    errs = bundle_shape_errors(_bundle(items))
    assert errs == [], errs


def test_a5_shape_is_the_consumer_contract(tmp_path):
    """A real sampled item must satisfy the same context/payload split every other task type
    is held to."""
    graph = _graph(nodes=(_node("n1", gold_ev_uids=("u1",)), _node("n2")))
    runs = tmp_path / "runs"
    _write_run(runs, "run0", task="t1", stop_reason="policy_stop", retrieved_uids=("u1",))
    items, key = sample_a5_items({"t1": graph}, [], runs, _FakeCache(), random.Random(0), n=5)
    assert items
    assert bundle_shape_errors(_bundle(items)) == []
    payload_ids = {c["node_id"] for c in items[0]["payload"]["candidates"]}
    assert payload_ids == {"n1", "n2"}, "no matches were supplied, so neither node is resolved"


def test_a5_candidates_are_only_the_unresolved_required_needs(tmp_path):
    """A candidate is what `matches.parquet` says is still unresolved for THIS run --
    `resolve`/`use` rank drops a node from the list, `ask` or no row at all does not, and a
    match for a DIFFERENT run must not resolve anything here."""
    graph = _graph(nodes=(_node("n1"), _node("n2"), _node("n3")))
    runs = tmp_path / "runs"
    _write_run(runs, "run0", task="t1", stop_reason="policy_stop")
    matches_rows = [
        {"run_id": "run0", "node_id": "n1", "match_kind": "resolve"},
        {"run_id": "run0", "node_id": "n2", "match_kind": "ask"},  # asked, but NOT resolved
        {"run_id": "other_run", "node_id": "n3", "match_kind": "use"},  # a different run
    ]
    items, key = sample_a5_items({"t1": graph}, matches_rows, runs, None, random.Random(0), n=5)
    assert len(items) == 1
    payload_ids = {c["node_id"] for c in items[0]["payload"]["candidates"]}
    assert payload_ids == {"n2", "n3"}, "only n1 was resolved on THIS run"
    assert key[items[0]["item_id"]]["n_unresolved"] == 2


# --------------------------------------------------------------------------- missing <-> verdict


def test_a5_missing_is_required_only_for_should_have_asked_more():
    item = _item("A5", "a5_run0", payload={"candidates": _candidates("n1", "n2")})
    bundle = _bundle([item])

    def errs(response: dict) -> list[str]:
        return validate_records(bundle, [_rec("a5_run0", "alice", "A5", response)])

    e = errs({"verdict": "should_have_asked_more", "missing": []})
    assert any("requires a non-empty missing" in x for x in e), e

    assert errs({"verdict": "should_have_asked_more", "missing": ["n1"]}) == []

    e = errs({"verdict": "stopping_was_right", "missing": ["n1"]})
    assert any("missing must be empty" in x for x in e), e

    assert errs({"verdict": "stopping_was_right", "missing": []}) == []
    assert errs({"verdict": "cant_tell", "missing": []}) == []

    e = errs({"verdict": "should_have_asked_more", "missing": ["not_a_candidate"]})
    assert any("are not candidates" in x for x in e), e

    # The LLM parser enforces the identical rule (see pi_eval.annotate_llm._parse_a5).
    with pytest.raises(AnnotationParseError):
        parse_reply(
            item,
            json.dumps({"verdict": "should_have_asked_more", "missing": [], "rationale": "why"}),
        )
    with pytest.raises(AnnotationParseError):
        parse_reply(
            item,
            json.dumps({"verdict": "stopping_was_right", "missing": ["n1"], "rationale": "why"}),
        )
    parsed = parse_reply(
        item,
        json.dumps({"verdict": "should_have_asked_more", "missing": ["n1"], "rationale": "why"}),
    )
    assert parsed.response == {"verdict": "should_have_asked_more", "missing": ["n1"]}
    assert parsed.rationale == "why"


def test_a5_llm_reply_missing_its_rationale_is_a_parse_error():
    """Same discipline as every other task type: a model verdict with no stated reason is never
    defaulted, it is a counted parse failure."""
    item = _item("A5", "a5_run0", payload={"candidates": _candidates("n1")})
    reply = json.dumps({"verdict": "stopping_was_right", "missing": []})
    with pytest.raises(AnnotationParseError):
        parse_reply(item, reply)


# --------------------------------------------------------------------------- per-candidate agreement


def test_a5_agreement_is_per_candidate_not_per_set():
    """Two annotators agreeing on 3 of 4 open needs must not score as total disagreement -- that
    is exactly what a set-equality comparison over `missing` would do."""
    item = _item("A5", "a5_run0", payload={"candidates": _candidates("n1", "n2", "n3", "n4")})
    bundle = _bundle([item])
    alice = _rec(
        "a5_run0",
        "alice",
        "A5",
        {"verdict": "should_have_asked_more", "missing": ["n1", "n2", "n3"]},
    )
    bob = _rec(
        "a5_run0", "bob", "A5", {"verdict": "should_have_asked_more", "missing": ["n1", "n2", "n4"]}
    )
    cons = consensus(bundle, [alice, bob], min_annotators=2)

    resolved = {u.unit_id: u.label for u in cons.by_kind("A5_missing")}
    assert resolved == {"a5_run0/n1": "still_needed", "a5_run0/n2": "still_needed"}

    disagreements = {d["unit_id"] for d in cons.disagreements if d["kind"] == "A5_missing"}
    assert disagreements == {"a5_run0/n3", "a5_run0/n4"}

    # Sets differ entirely ({n1,n2,n3} != {n1,n2,n4}), yet HALF the candidates resolved to a
    # real consensus -- proof the unit is the candidate, not the set.
    assert len(resolved) == 2
    assert len(disagreements) == 2

    # The overall A5 verdict is still its own separate unit (both agreed).
    verdict_units = cons.by_kind("A5")
    assert len(verdict_units) == 1
    assert verdict_units[0].label == "should_have_asked_more"


# --------------------------------------------------------------------------- never gold


def test_a5_never_writes_a_gold_field():
    """A5 is a judgment about a RUN, not about the task's graph. `merge_into_graphs` has no
    branch for `kind == 'A5'` / `'A5_missing'`, on purpose."""
    node = _node("n1")
    graph = _graph(
        nodes=(node,),
        gold_answer="X PINQCANARY_0123456789ABCDEF",
        gold_canary="PINQCANARY_0123456789ABCDEF",
        gold_corpus_hash="cafe1234",
    )
    item = _item("A5", "a5_run0", payload={"candidates": _candidates("n1")})
    bundle = _bundle([item])
    recs = [
        _rec("a5_run0", a, "A5", {"verdict": "should_have_asked_more", "missing": ["n1"]})
        for a in ("alice", "bob")
    ]
    cons = consensus(bundle, recs)
    assert cons.by_kind("A5_missing")[0].label == "still_needed", "the consensus DID form"

    rows = merge_into_graphs({"t1": graph}, cons, out_version="v1h")
    merged = _from_rows(rows)["t1"]
    merged_node = merged.gold_nodes[0]

    assert merged_node.gold_human_asked is None
    assert merged_node.gold_human_adjudicated is False
    assert merged_node.gold_partition == node.gold_partition
    assert merged_node.gold_discoverability == node.gold_discoverability
    assert merged.gold_answer == graph.gold_answer
    assert merged.gold_canary == graph.gold_canary


# --------------------------------------------------------------------------- placeholders


def test_a5_candidates_carry_no_placeholder(tmp_path):
    """MuSiQue node text carries a mechanical `#N` past depth 0 (see
    `pi_eval.build.musique_build`). A candidate nobody can read cannot be judged: the whole
    item is refused, not shipped with a placeholder in it."""
    s1 = GoldNode(
        gold_suite="musique",
        gold_task_key="t1",
        gold_node_id="s1",
        gold_text="The Collegian >> owned by",
        gold_partition="required",
        gold_discoverability="kb",
        gold_depth=0,
    )
    s2 = GoldNode(
        gold_suite="musique",
        gold_task_key="t1",
        gold_node_id="s2",
        gold_text="When was #1 founded?",
        gold_partition="required",
        gold_discoverability="kb",
        gold_depth=1,
    )
    graph = GoldGraph(gold_suite="musique", gold_task_key="t1", gold_nodes=(s1, s2))
    runs = tmp_path / "runs"
    _write_run(runs, "run0", task="t1", suite="musique", stop_reason="policy_stop")

    stats: dict[str, int] = {}
    refused, _key = sample_a5_items(
        {"t1": graph}, [], runs, None, random.Random(0), n=5, answers={}, stats=stats
    )
    assert refused == [], "s2's #1 cannot resolve with no subanswers -- the item must be refused"
    assert sum(v for k, v in stats.items() if "unresolved" in k) >= 1

    answers = {"t1": {"s1": "Houston Baptist University"}}
    items, _key2 = sample_a5_items(
        {"t1": graph}, [], runs, _FakeCache(), random.Random(1), n=5, answers=answers
    )
    assert len(items) == 1
    texts = {c["node_id"]: c["text"] for c in items[0]["payload"]["candidates"]}
    assert texts["s2"] == "When was Houston Baptist University founded?"
    assert "#1" not in json.dumps(items)
    assert bundle_shape_errors(_bundle(items)) == []


# --------------------------------------------------------------------------- CLI wiring


def test_export_wires_n_a5(tmp_path, monkeypatch):
    """`--n-a5` reaches `sample_a5_items`, and its output lands in the bundle (visibly) and the
    key (the blinded facts) exactly like every other sampler's."""
    from pi_run import cmd_annotate
    from pi_run.cli import build_parser

    monkeypatch.setenv("PI_CANARY_SALT", "test-salt")
    monkeypatch.setenv("PI_GOLD_ROOT", str(tmp_path / "data" / "gold"))

    from pi_eval.build.common import write_graphs

    write_graphs(
        tmp_path,
        "synth",
        "v1",
        [
            {
                "gold_suite": "synth",
                "gold_task_key": "t1",
                "gold_nodes": [],
                "gold_edges": [],
                "gold_facets": [],
                "gold_seed_node_ids": [],
                "gold_graph_version": "v1",
                "gold_answer": "answer",
                "gold_aliases": [],
            }
        ],
    )

    captured: dict[str, int] = {}

    def _stub(
        graphs, matches_rows, runs_root, cache, rng, *, n, answers=None, stats=None, blind=False
    ):
        # `blind` is captured too: the flag decides whether the item ships the gold frontier,
        # and a sampler that silently ignored it would build the sighted instrument under the
        # blinded bundle's name. See tests/test_a5_blind.py for what that list does to the
        # verdict.
        captured["n"] = n
        captured["blind"] = blind
        it = {
            "item_id": "stub_a5",
            "task_type": "A5",
            "context": {"question": "q", "evidence": [], "history": [], "answer": "a"},
            "payload": {"candidates": []},
            "provenance": {
                "suite": "synth",
                "task_id": "t1",
                "task_key": "t1",
                "graph_version": "v1",
            },
        }
        return [it], {
            "stub_a5": {
                "answer_correct": 1.0,
                "evidence_coverage": 1.0,
                "stop_reason": "policy_stop",
                "n_unresolved": 0,
            }
        }

    monkeypatch.setattr(cmd_annotate, "sample_a5_items", _stub)

    args = build_parser().parse_args(
        [
            "annotate",
            "export",
            "--suite",
            "synth",
            "--root",
            str(tmp_path),
            "--runs-root",
            str(tmp_path / "no-runs"),
            "--seed",
            "0",
            "--n-a1",
            "0",
            "--n-a2",
            "0",
            "--n-a3-node",
            "0",
            "--n-a3-edge",
            "0",
            "--n-a3-match",
            "0",
            "--n-a4",
            "0",
            "--n-a5",
            "3",
        ]
    )
    assert args.fn(args) == 0
    assert captured["n"] == 3
    assert captured["blind"] is False, "the default instrument is the sighted one"

    bundle_dir = tmp_path / "data" / "gold" / "human" / "synth" / "bundles"
    bundle_path = next(p for p in bundle_dir.glob("*.json") if not p.name.endswith(".key.json"))
    key_path = bundle_dir / (bundle_path.stem + ".key.json")
    bundle_text = bundle_path.read_text()
    key_text = key_path.read_text()

    bundle = json.loads(bundle_text)
    assert any(it["task_type"] == "A5" for it in bundle["items"])
    assert "answer_correct" not in bundle_text
    assert "answer_correct" in key_text


# --------------------------------------------------------------------------- review: saturation


def test_a5_straight_line_saturation_is_caught_by_review():
    """A rater who answers `stopping_was_right` to everything is the A1 tick-rate saturation
    failure in a new place. `_straight_line_runs` is fed by `_response_label`, which now reads
    A5's `verdict` -- this proves that wiring actually flags it, not just that the code path
    exists. `review_report` also surfaces the raw per-annotator verdict distribution alongside
    it, so a reviewer sees WHICH label saturated."""
    from pi_run.cmd_annotate import review_report

    items = [
        _item("A5", f"a5_run{i}", payload={"candidates": _candidates(f"n{i}")}) for i in range(6)
    ]
    bundle = _bundle(items)
    saturated = [
        _rec(
            f"a5_run{i}",
            "saturated_rater",
            "A5",
            {"verdict": "stopping_was_right", "missing": []},
            ts=f"2026-08-31T12:0{i}:00Z",
        )
        for i in range(6)
    ]
    # A second annotator on the same items, answering honestly (mixed verdicts) -- so a1's
    # tick-rate-style flag is genuinely about ONE rater, not an artifact of the bundle.
    honest = [
        _rec(
            f"a5_run{i}",
            "honest_rater",
            "A5",
            {
                "verdict": "should_have_asked_more" if i % 2 else "stopping_was_right",
                "missing": [f"n{i}"] if i % 2 else [],
            },
            ts=f"2026-08-31T12:0{i}:05Z",
        )
        for i in range(6)
    ]
    report = review_report(bundle, saturated + honest)

    runs = report["degenerate"]["straight_line_runs"]
    assert runs.get("saturated_rater", 0) >= 5, runs
    assert "honest_rater" not in runs, "a mixed rater must not be flagged"

    dist = report["a5_verdict_distribution"]
    assert dist["saturated_rater"] == {"stopping_was_right": 6}
    assert dist["honest_rater"] == {"should_have_asked_more": 3, "stopping_was_right": 3}
