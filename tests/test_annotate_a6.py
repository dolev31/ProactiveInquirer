"""A6: rank ALL k candidates from one forked state, not just two of them.

A2 shows an annotator two continuations of a state and asks which is better. `pi train
sample-candidates` forked that state into k of them, so the same annotation slot can carry
C(k,2) pairwise comparisons instead of one. This file locks down the four things that make
that generalisation safe:

  * an item exists only where there are >= 3 candidates -- a 2-candidate state IS an A2 item
    and must not be shipped as both;
  * the shipped candidate ids are item-local (`c0..c4`); the run ids they stand for live in
    the key, so an annotator cannot correlate candidates across items;
  * EQUAL TIERS ARE A TIE. A tier map is not a strict order, and nothing may turn one into
    one -- a manufactured preference goes straight into DPO training data;
  * agreement is computed per derived PAIR, never over the whole ranking, and the unit never
    reaches gold.
"""

from __future__ import annotations

import json
import random

import pytest

from pi_eval.annotate import (
    bundle_shape_errors,
    consensus,
    human_judgment_rows,
    item_set_hash,
    merge_into_graphs,
    validate_records,
)
from pi_eval.annotate_llm import AnnotationParseError, parse_reply
from pi_eval.gold import GoldGraph, GoldNode
from pi_run.cmd_annotate import sample_a6_items

BUNDLE_ID = "synth-0-deadbeef"


# --------------------------------------------------------------------------- fixtures


class _FakeCache:
    """Stands in for `SuiteCache`: only `.view(...).question` is read on this path."""

    def __init__(self, question: str = "the task question"):
        self._question = question

    def view(self, suite_id, task_id):
        from types import SimpleNamespace

        return SimpleNamespace(question=self._question)

    def units(self, suite_id, task_id):
        return {}


def _pair(
    chosen_run: str,
    rejected_run: str,
    *,
    chosen_q: str,
    rejected_q: str,
    margin: float = 0.2,
    turn_idx: int = 1,
    run_id: str = "parent0",
    task_id: str = "t1",
    pair_id: str | None = None,
    is_latent: bool = True,
) -> dict:
    return {
        "suite_id": "synth",
        "task_id": task_id,
        "run_id": run_id,
        "turn_idx": turn_idx,
        "state_text": "the rendered state the inquirer had in front of it",
        "chosen_json": json.dumps({"action": "ASK", "question": chosen_q}),
        "rejected_json": json.dumps({"action": "ASK", "question": rejected_q}),
        "chosen_run_id": chosen_run,
        "rejected_run_id": rejected_run,
        "margin": margin,
        "is_latent": is_latent,
        "pair_id": pair_id or f"{chosen_run}:{rejected_run}",
    }


def _star_state(n_candidates: int, *, run_id: str = "parent0", task_id: str = "t1") -> list[dict]:
    """One state whose winner beat every other candidate -- the shape `sample-candidates`
    actually produces (measured: a 5-candidate state carries 4 pairs, not 10)."""
    return [
        _pair(
            "cand0",
            f"cand{i}",
            chosen_q="which region did Andy sail to first?",
            rejected_q=f"what is fact number {i} about?",
            margin=0.1 * i,
            run_id=run_id,
            task_id=task_id,
        )
        for i in range(1, n_candidates)
    ]


def _item(task_type: str, iid: str, *, payload=None, **prov) -> dict:
    p = {"suite": "synth", "task_id": "t1", "task_key": "t1", "graph_version": "v1"}
    p.update(prov)
    return {
        "item_id": iid,
        "task_type": task_type,
        "context": {},
        "payload": payload or {},
        "provenance": p,
    }


def _a6_item(iid: str, *candidate_ids: str) -> dict:
    return _item(
        "A6",
        iid,
        payload={
            "candidates": [{"candidate_id": c, "question": f"question {c}"} for c in candidate_ids]
        },
        run_id="parent0",
        turn_idx=1,
    )


def _bundle(items) -> dict:
    return {
        "manifest": {"bundle_id": BUNDLE_ID, "item_set_hash": item_set_hash(list(items))},
        "items": list(items),
    }


def _rec(iid: str, ann: str, response: dict, task_type: str = "A6", **kw) -> dict:
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


def _node(nid: str, **kw) -> GoldNode:
    kw.setdefault("gold_partition", "required")
    kw.setdefault("gold_discoverability", "kb")
    kw.setdefault("gold_text", f"need {nid}")
    return GoldNode(gold_suite="synth", gold_task_key="t1", gold_node_id=nid, **kw)


def _tup(d: dict) -> dict:
    return {k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items()}


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


# --------------------------------------------------------------------------- the item


def test_a6_needs_three_or_more_candidates():
    """A 2-candidate state is already an A2 item. Shipping it as both spends two annotation
    slots on one question and puts the same comparison into the preference table twice."""
    rows = _star_state(2, run_id="two", task_id="t1") + _star_state(3, run_id="three", task_id="t2")
    stats: dict[str, int] = {}
    items, key = sample_a6_items(
        rows, _FakeCache(), random.Random(0), n=10, graph_version="v1", stats=stats
    )
    run_ids = {it["provenance"]["run_id"] for it in items}
    assert run_ids == {"three"}, run_ids
    assert stats.get("a6_too_few_candidates") == 1
    assert len(items[0]["payload"]["candidates"]) == 3


def test_a6_candidate_ids_are_item_local_and_run_ids_stay_in_the_key():
    """A run id in the shipped bundle would let an annotator correlate candidates across
    items. The `candidate_id -> run_id` map is exactly what the key exists to hold."""
    rows = _star_state(5)
    items, key = sample_a6_items(rows, _FakeCache(), random.Random(0), n=5, graph_version="v1")
    assert len(items) == 1
    it = items[0]

    ids = [c["candidate_id"] for c in it["payload"]["candidates"]]
    assert ids == ["c0", "c1", "c2", "c3", "c4"]
    for c in it["payload"]["candidates"]:
        assert set(c) == {"candidate_id", "question"}

    blob = json.dumps(it)
    for run_id in (f"cand{i}" for i in range(5)):
        assert run_id not in blob, f"{run_id!r} leaked into the shipped A6 item"
    for forbidden in ("margin", "auto_order", "auto_scores", "candidate_runs"):
        assert forbidden not in blob, f"{forbidden!r} leaked into the shipped A6 item"

    ke = key[it["item_id"]]
    assert set(ke["candidate_runs"]) == set(ids)
    assert set(ke["candidate_runs"].values()) == {f"cand{i}" for i in range(5)}
    assert sorted(ke["auto_order"]) == sorted(ids)
    # cand0 beat every other candidate, so the pipeline's own ordering puts it first.
    assert ke["candidate_runs"][ke["auto_order"][0]] == "cand0"
    assert bundle_shape_errors(_bundle(items)) == []


def test_a6_shuffles_and_the_shown_order_is_not_the_automatic_one():
    """`candidate_id` is assigned AFTER the shuffle, so c0 is not the pipeline's favourite --
    an annotator who noticed that would be reading the answer off the item's shape."""
    rows = _star_state(5)
    seen_first = set()
    for seed in range(12):
        _items, key = sample_a6_items(
            rows, _FakeCache(), random.Random(seed), n=5, graph_version="v1"
        )
        ke = next(iter(key.values()))
        seen_first.add(ke["candidate_runs"]["c0"])
    assert len(seen_first) > 1, f"c0 was always the same run: {seen_first}"


def test_a6_drops_candidates_whose_question_text_is_identical():
    """Measured on the live pairs.jsonl: 95 of the 158 eligible states carry two candidate run
    ids with byte-identical question text. Asking which of two identical questions is better
    manufactures a coin flip, and the pair it derives enters DPO as a preference between two
    copies of one question."""
    rows = [
        _pair("cand0", "cand1", chosen_q="A?", rejected_q="B?"),
        _pair("cand0", "cand2", chosen_q="A?", rejected_q="  b?  "),  # same as cand1 modulo case/ws
        _pair("cand0", "cand3", chosen_q="A?", rejected_q="C?"),
    ]
    stats: dict[str, int] = {}
    items, _key = sample_a6_items(
        rows, _FakeCache(), random.Random(0), n=5, graph_version="v1", stats=stats
    )
    assert len(items) == 1
    questions = sorted(c["question"] for c in items[0]["payload"]["candidates"])
    assert questions == ["A?", "B?", "C?"], questions
    assert stats.get("a6_duplicate_question") == 1


def test_a6_caps_the_candidate_set():
    """Measured on the live pairs.jsonl: states reach 17 candidates, which is C(17,2)=136 units
    on one screen. The cap keeps an item to at most C(5,2)=10 derived pairs."""
    from pi_run.cmd_annotate import A6_MAX_CANDIDATES

    rows = _star_state(9)
    items, key = sample_a6_items(rows, _FakeCache(), random.Random(0), n=5, graph_version="v1")
    assert len(items[0]["payload"]["candidates"]) == A6_MAX_CANDIDATES == 5
    assert len(key[items[0]["item_id"]]["candidate_runs"]) == 5


def test_a6_context_is_the_same_state_a2_would_show():
    """Both instruments must show the SAME state, built by the same replay path -- an A6 item
    that showed less than its A2 sibling would not be measuring the same judgment."""
    rows = _star_state(4)
    items, _key = sample_a6_items(rows, _FakeCache(), random.Random(0), n=5, graph_version="v1")
    ctx = items[0]["context"]
    assert set(ctx) == {"question", "evidence", "history", "draft", "state_text"}
    assert ctx["question"] == "the task question"
    assert ctx["state_text"] == "the rendered state the inquirer had in front of it"


# --------------------------------------------------------------------------- the tier judgment


def test_a6_tiers_must_cover_every_candidate_exactly_once():
    item = _a6_item("i0", "c0", "c1", "c2")
    bundle = _bundle([item])

    def errs(response: dict) -> list[str]:
        return validate_records(bundle, [_rec("i0", "alice", response)])

    assert errs({"tiers": {"c0": 1, "c1": 2, "c2": 2}}) == []
    assert errs({"tiers": {"c0": 1, "c1": 1, "c2": 3}}) == [], "gaps are allowed"

    e = errs({"tiers": {"c0": 1, "c1": 2}})
    assert any("no tier for" in x and "c2" in x for x in e), e

    e = errs({"tiers": {"c0": 1, "c1": 2, "c2": 3, "c9": 1}})
    assert any("c9" in x and "not a candidate" in x for x in e), e

    e = errs({"tiers": {"c0": 0, "c1": 1, "c2": 2}})
    assert any("positive integer" in x for x in e), e

    e = errs({"tiers": {"c0": 1.5, "c1": 2, "c2": 3}})
    assert any("positive integer" in x for x in e), e

    e = errs({"tiers": [1, 2, 3]})
    assert any("tiers must be an object" in x for x in e), e


def test_a6_equal_tiers_are_a_tie_not_a_forced_order():
    """Forcing a strict order on candidates a person considers equivalent manufactures a
    preference that was never there, and those false pairs go straight into DPO training."""
    item = _a6_item("i0", "c0", "c1", "c2")
    bundle = _bundle([item])
    recs = [_rec("i0", a, {"tiers": {"c0": 1, "c1": 2, "c2": 2}}) for a in ("alice", "bob")]
    cons = consensus(bundle, recs)
    labels = {u.unit_id: u.label for u in cons.by_kind("A6")}
    assert labels == {
        "i0/c0|c1": "a_better",
        "i0/c0|c2": "a_better",
        "i0/c1|c2": "tie",
    }

    rows = human_judgment_rows(
        bundle,
        [recs[0]],
        key={"items": {"i0": {"candidate_runs": {"c0": "r0", "c1": "r1", "c2": "r2"}}}},
    )
    by_pair = {(r["run_id_a"], r["run_id_b"]): r["pref_sign"] for r in rows}
    assert by_pair[("r1", "r2")] == 0, "an equal tier is a 0 sign, never a coin-flipped +/-1"
    assert by_pair[("r0", "r1")] == 1
    assert by_pair[("r0", "r2")] == 1


def test_a6_agreement_is_per_pair_not_per_ranking():
    """Two annotators who agree on 9 of 10 pairs must not read as total disagreement, which is
    exactly what comparing whole rankings for equality would give."""
    item = _a6_item("i0", "c0", "c1", "c2", "c3", "c4")
    bundle = _bundle([item])
    alice = _rec("i0", "alice", {"tiers": {"c0": 1, "c1": 2, "c2": 3, "c3": 4, "c4": 5}})
    # bob swaps the last two only: 9 of the 10 derived pairs are unchanged.
    bob = _rec("i0", "bob", {"tiers": {"c0": 1, "c1": 2, "c2": 3, "c3": 5, "c4": 4}})
    assert alice["response"] != bob["response"], "the rankings genuinely differ"

    cons = consensus(bundle, [alice, bob], min_annotators=2)
    resolved = cons.by_kind("A6")
    disagreed = [d for d in cons.disagreements if d["kind"] == "A6"]
    assert len(resolved) == 9, [u.unit_id for u in resolved]
    assert [d["unit_id"] for d in disagreed] == ["i0/c3|c4"]


def test_a6_derives_one_judgment_row_per_pair():
    """5 candidates => C(5,2) = 10 rows, each carrying the two RUN ids from the key -- the
    ~10x signal density that is the entire justification for the instrument."""
    item = _a6_item("i0", "c0", "c1", "c2", "c3", "c4")
    bundle = _bundle([item])
    key = {
        "items": {
            "i0": {"candidate_runs": {f"c{i}": f"run{i}" for i in range(5)}},
        }
    }
    rec = _rec("i0", "alice", {"tiers": {"c0": 1, "c1": 1, "c2": 2, "c3": 3, "c4": 3}})
    rows = human_judgment_rows(bundle, [rec], key=key)
    assert len(rows) == 10
    assert len({r["judgment_id"] for r in rows}) == 10, "each derived pair is its own judgment"
    assert {r["criterion"] for r in rows} == {"human_preference"}, "pools with A2's rows"
    assert {r["judge_family"] for r in rows} == {"human"}
    assert {r["judge_model"] for r in rows} == {"alice"}
    assert {(r["run_id_a"], r["run_id_b"]) for r in rows} == {
        (f"run{i}", f"run{j}") for i in range(5) for j in range(i + 1, 5)
    }
    signs = {(r["run_id_a"], r["run_id_b"]): r["pref_sign"] for r in rows}
    assert signs[("run0", "run1")] == 0 and signs[("run3", "run4")] == 0
    assert signs[("run0", "run2")] == 1
    assert signs[("run2", "run3")] == 1
    assert all(r["len_a_words"] > 0 and r["len_b_words"] > 0 for r in rows)

    from pi_eval import schema as sch

    assert set(rows[0]) == set(sch.JUDGMENTS.names)
    sch.to_table("judgments", rows)  # raises if a value does not fit the declared type

    # Stable: the same records import to the same judgment ids.
    assert {r["judgment_id"] for r in human_judgment_rows(bundle, [rec], key=key)} == {
        r["judgment_id"] for r in rows
    }


def test_a6_judgment_rows_follow_the_rater_kind():
    """A model's preference is not a person's. `judge_family` is the column judgments.parquet
    discriminates raters on, so an llm record must never land there labelled 'human'."""
    item = _a6_item("i0", "c0", "c1", "c2")
    bundle = _bundle([item])
    key = {"items": {"i0": {"candidate_runs": {"c0": "r0", "c1": "r1", "c2": "r2"}}}}
    rec = _rec(
        "i0",
        "llm:gpt-x",
        {"tiers": {"c0": 1, "c1": 2, "c2": 3}},
        annotator_kind="llm",
        model_pin="gpt-x@t1.0",
        rationale="because",
    )
    rows = human_judgment_rows(bundle, [rec], key=key)
    assert rows and {r["judge_family"] for r in rows} == {"llm"}


def test_a6_emits_nothing_without_the_key_that_names_the_runs():
    """Same rule A2 follows: no key entry, no judgment row. A run id guessed from the shown
    slot would be a fabricated provenance on a published preference."""
    item = _a6_item("i0", "c0", "c1", "c2")
    bundle = _bundle([item])
    rec = _rec("i0", "alice", {"tiers": {"c0": 1, "c1": 2, "c2": 3}})
    assert human_judgment_rows(bundle, [rec], key=None) == []
    assert human_judgment_rows(bundle, [rec], key={"items": {}}) == []


# --------------------------------------------------------------------------- never gold


def test_a6_never_writes_a_gold_field():
    """A6 is a preference between two RUNS, not a claim about the task's graph.
    `merge_into_graphs` has no branch for `kind == 'A6'`, exactly as A5 has none.

    The A6 item here is given a `gold_node_id` in its provenance ON PURPOSE, even though the
    sampler puts none there. Without it this test cannot fail: a wrong `elif u.kind == "A6"`
    branch in `merge_into_graphs` reads `provenance['gold_node_id']`, finds nothing, and writes
    nothing anyway -- so the test would pass over the very bug it claims to catch. With it, any
    such branch marks n1 adjudicated and this fails.
    """
    from pi_eval.annotate import Consensus

    node = _node("n1")
    graph = GoldGraph(
        gold_suite="synth",
        gold_task_key="t1",
        gold_nodes=(node,),
        gold_answer="X PINQCANARY_0123456789ABCDEF",
        gold_canary="PINQCANARY_0123456789ABCDEF",
        gold_corpus_hash="cafe1234",
    )
    item = _a6_item("i0", "c0", "c1", "c2")
    item["provenance"]["gold_node_id"] = "n1"
    bundle = _bundle([item])
    recs = [_rec("i0", a, {"tiers": {"c0": 1, "c1": 2, "c2": 3}}) for a in ("alice", "bob")]
    cons = consensus(bundle, recs)
    assert len(cons.by_kind("A6")) == 3, "the consensus DID form"
    assert all(u.provenance.get("gold_node_id") == "n1" for u in cons.by_kind("A6"))

    rows = merge_into_graphs({"t1": graph}, cons, out_version="v1h")
    # Byte-identical to the same merge over NO units at all: whatever field a wrong A6 branch
    # decided to touch, this notices it.
    assert rows == merge_into_graphs({"t1": graph}, Consensus(), out_version="v1h")

    merged = _from_rows(rows)["t1"]
    merged_node = merged.gold_nodes[0]
    assert merged_node.gold_human_asked is None
    assert merged_node.gold_human_adjudicated is False
    assert merged_node.gold_partition == node.gold_partition
    assert merged.gold_answer == graph.gold_answer
    assert merged.gold_canary == graph.gold_canary


# --------------------------------------------------------------------------- the model pass


def test_a6_llm_reply_with_a_missing_candidate_is_a_parse_error():
    """A ranking that leaves a candidate out has no pairwise reading for that candidate, and a
    defaulted tier would be a fabricated preference -- never a record, a counted failure."""
    item = _a6_item("i0", "c0", "c1", "c2")

    ok = parse_reply(item, json.dumps({"tiers": {"c0": 1, "c1": 2, "c2": 2}, "rationale": "why"}))
    assert ok.response == {"tiers": {"c0": 1, "c1": 2, "c2": 2}}
    assert ok.rationale == "why"

    with pytest.raises(AnnotationParseError):
        parse_reply(item, json.dumps({"tiers": {"c0": 1, "c1": 2}, "rationale": "why"}))
    with pytest.raises(AnnotationParseError):
        parse_reply(
            item, json.dumps({"tiers": {"c0": 1, "c1": 2, "c2": 3, "c7": 4}, "rationale": "why"})
        )
    with pytest.raises(AnnotationParseError):
        parse_reply(item, json.dumps({"tiers": {"c0": 0, "c1": 1, "c2": 2}, "rationale": "why"}))
    with pytest.raises(AnnotationParseError):
        parse_reply(item, json.dumps({"tiers": {"c0": 1, "c1": 2, "c2": 3}}))  # no rationale


def test_a6_prompt_states_the_tie_rule_and_names_every_candidate():
    from pi_eval.annotate_llm import build_prompt

    item = _a6_item("i0", "c0", "c1", "c2")
    item["context"] = {
        "question": "the task question",
        "evidence": [],
        "history": [],
        "draft": "",
        "state_text": "the state",
    }
    prompt = build_prompt(item)
    for cid in ("c0", "c1", "c2"):
        assert cid in prompt
    assert "tier" in prompt.lower()
    assert "rationale" in prompt


# --------------------------------------------------------------------------- review


def test_a6_review_reports_tier_spread_and_agreement_with_the_automatic_margin():
    """An annotator who puts every candidate in one tier is the A1 saturation failure again,
    and it is invisible in an alpha over pairs that are all 'tie'."""
    from pi_run.cmd_annotate import review_report

    items = [_a6_item(f"i{i}", "c0", "c1", "c2") for i in range(4)]
    bundle = _bundle(items)
    key = {
        "items": {
            f"i{i}": {
                "candidate_runs": {"c0": "r0", "c1": "r1", "c2": "r2"},
                "auto_pairs": {"c0|c1": "a_better", "c0|c2": "a_better"},
            }
            for i in range(4)
        }
    }
    flat = [
        _rec(f"i{i}", "flat_rater", {"tiers": {"c0": 1, "c1": 1, "c2": 1}}, ts=f"t{i}")
        for i in range(4)
    ]
    discriminating = [
        _rec(f"i{i}", "real_rater", {"tiers": {"c0": 1, "c1": 2, "c2": 3}}, ts=f"t{i}")
        for i in range(4)
    ]
    report = review_report(bundle, flat + discriminating, key=key)

    spread = report["a6_tier_spread"]
    assert spread["flat_rater"]["one_tier_rate"] == pytest.approx(1.0)
    assert spread["flat_rater"]["flag"] is True
    assert spread["real_rater"]["mean_distinct_tiers"] == pytest.approx(3.0)
    assert spread["real_rater"]["flag"] is False

    agree = report["a6_auto_agreement"]
    # real_rater matches the automatic ordering on both compared pairs of all four items.
    assert agree["overall"]["n"] == 8
    assert agree["overall"]["agreement"] == pytest.approx(1.0)
    # flat_rater called both a tie, which the automatic side has no tie label to match: those
    # are counted apart rather than scored as agreement or as disagreement.
    assert agree["n_human_tie_where_auto_strict"] == 8


# --------------------------------------------------------------------------- CLI wiring


def test_export_wires_n_a6(tmp_path, monkeypatch):
    """`--n-a6` reaches `sample_a6_items`, its items land in the bundle and its
    candidate->run map lands in the key and nowhere else."""
    from pi_eval.build.common import write_graphs
    from pi_run import cmd_annotate
    from pi_run.cli import build_parser

    monkeypatch.setenv("PI_CANARY_SALT", "test-salt")
    monkeypatch.setenv("PI_GOLD_ROOT", str(tmp_path / "data" / "gold"))
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

    def _stub(pairs_rows, cache, rng, *, n, runs_root=None, graph_version="", stats=None):
        captured["n"] = n
        it = {
            "item_id": "stub_a6",
            "task_type": "A6",
            "context": {
                "question": "q",
                "evidence": [],
                "history": [],
                "draft": "",
                "state_text": "s",
            },
            "payload": {
                "candidates": [
                    {"candidate_id": "c0", "question": "q0"},
                    {"candidate_id": "c1", "question": "q1"},
                    {"candidate_id": "c2", "question": "q2"},
                ]
            },
            "provenance": {
                "suite": "synth",
                "task_id": "t1",
                "task_key": "t1",
                "graph_version": "v1",
            },
        }
        return [it], {"stub_a6": {"candidate_runs": {"c0": "r0", "c1": "r1", "c2": "r2"}}}

    monkeypatch.setattr(cmd_annotate, "sample_a6_items", _stub)

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
            "0",
            "--n-a6",
            "7",
        ]
    )
    assert args.fn(args) == 0
    assert captured["n"] == 7

    bundle_dir = tmp_path / "data" / "gold" / "human" / "synth" / "bundles"
    bundle_path = next(p for p in bundle_dir.glob("*.json") if not p.name.endswith(".key.json"))
    key_path = bundle_dir / (bundle_path.stem + ".key.json")
    bundle_text = bundle_path.read_text()
    key_text = key_path.read_text()

    bundle = json.loads(bundle_text)
    assert any(it["task_type"] == "A6" for it in bundle["items"])
    assert bundle["manifest"]["counts"]["A6"] == 1
    assert "candidate_runs" not in bundle_text
    assert "candidate_runs" in key_text


def test_a6_provenance_run_id_is_the_state_not_a_candidate_tell():
    """The ONE run id an A6 item ships is the state's own, exactly as A2 already ships it.

    Measured on the live `data/rl/pairs.jsonl`: the forked-from run is itself one of the
    candidates in 114 of the 172 states, so that provenance id coincides with a candidate's run
    id. That is not an unblinding -- it says nothing about which `candidate_id` that run became
    -- and it is not a tell for the automatic pick either, since the parent is the pipeline's
    winner in 351 pairs and its loser in 151. What must never ship is the MAP, and the payload
    carries no run id at all.
    """
    rows = [
        _pair("cand0", f"cand{i}", chosen_q="A?", rejected_q=f"Q{i}?", run_id="cand0")
        for i in range(1, 4)
    ]
    items, key = sample_a6_items(rows, _FakeCache(), random.Random(0), n=5, graph_version="v1")
    it = items[0]
    assert it["provenance"]["run_id"] == "cand0"
    assert "cand0" not in json.dumps(it["payload"]), "the payload names no run, ever"
    assert "cand0" not in json.dumps(it["context"])
    assert "cand0" in set(key[it["item_id"]]["candidate_runs"].values())


def test_a6_sampling_is_deterministic_under_seed():
    """The contract every sampler in `cmd_annotate` is held to: one seed, one `item_set_hash`;
    two seeds, two."""
    rows = _star_state(9)

    def hashed(seed: int) -> str:
        items, _key = sample_a6_items(
            rows, _FakeCache(), random.Random(seed), n=5, graph_version="v1"
        )
        return item_set_hash(items)

    assert hashed(0) == hashed(0)
    assert hashed(0) != hashed(3), "a different seed must draw a different candidate set/order"
