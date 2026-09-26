"""An LLM may annotate. It may never become the human annotation.

`adr` is |V_found \\ V_human| / |V_annotated|, and V_human is what a PERSON would have thought
to ask. A model's prediction of that is a different quantity with the same shape: plausible,
well-formed, and not the measurement. Letting it reach `gold_human_asked` would produce the
exact artifact this repository exists to refuse -- a publishable number computed from a
measurement nobody took -- under the one metric whose purpose is to support "exceeds what the
user thought to ask".

So the bar is structural rather than procedural. `consensus` counts human raters only; the
rater classes it was built from ride along on the result; and `merge_into_graphs` refuses a
consensus that was not human-only. Three independent points, because a rule enforced in one
place is a rule one refactor away from nothing.

What an LLM pass IS for: triage before a human looks, the sanctioned `llm_elicited` recall
aid of the annotation plan (an LLM proposes needs a miner missed, and only a human pass can
confirm them), and measuring how well a model tracks the humans -- which is a finding in its
own right, reported and never fed to a gate.
"""

from __future__ import annotations

import math

import pytest

from pi_eval.annotate import (
    LLMAnnotationNotGold,
    consensus,
    gate_numbers,
    iaa_report,
    item_set_hash,
    llm_agreement,
    merge_into_graphs,
    validate_records,
)
from pi_eval.gold import GoldGraph, GoldNode

BUNDLE_ID = "musique-0-deadbeef"


def _node(nid):
    return GoldNode(
        gold_suite="musique",
        gold_task_key="t1",
        gold_node_id=nid,
        gold_text=f"need {nid}",
        gold_partition="required",
        gold_discoverability="kb",
        gold_graph_version="v1",
    )


def _graph(nodes):
    return GoldGraph(
        gold_suite="musique", gold_task_key="t1", gold_nodes=tuple(nodes), gold_graph_version="v1"
    )


def _item(task_type, iid, payload=None, **prov):
    p = {"suite": "musique", "task_id": "t1", "task_key": "t1", "graph_version": "v1"}
    p.update(prov)
    return {
        "item_id": iid,
        "task_type": task_type,
        "context": {},
        "payload": payload or {},
        "provenance": p,
    }


def _bundle(items, key_items=None):
    ish = item_set_hash(items)
    return (
        {
            "manifest": {"bundle_id": BUNDLE_ID, "item_set_hash": ish},
            "items": list(items),
        },
        {
            "manifest": {"bundle_id": BUNDLE_ID, "item_set_hash": ish},
            "items": dict(key_items or {}),
        },
    )


def _rec(iid, ann, task_type, response, *, kind=None, model=None):
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
    if kind is not None:
        r["annotator_kind"] = kind
    if model is not None:
        r["model_pin"] = model
    return r


def _llm(iid, task_type, response, who="llm:gpt-oss-120b"):
    """A well-formed model record, which now includes a rationale.

    The rationale is here because `validate_records` requires one on an `llm` record, not
    because anything in this file measures it: every test below asserts that a model's
    judgment does not become a human's, and a rationale changes none of that. It is carried
    so these fixtures keep meaning "a VALID model record" as the definition of valid grows.
    """
    r = _rec(iid, who, task_type, response, kind="llm", model="gpt-oss-120b@t0.0")
    r["rationale"] = "fixture rationale"
    return r


def _a1_bundle():
    items = [
        _item(
            "A1",
            "a1_t1",
            payload={"nodes": [{"node_id": n, "text": f"need {n}"} for n in ("n1", "n2")]},
        )
    ]
    return _bundle(items)


def _a1_resp(ticked):
    return {"ticked": list(ticked), "usefulness": {}}


# ------------------------------------------------------------------ the bar


def test_two_llm_raters_are_not_a_consensus():
    """Two models agreeing is two samples of one prediction, not two annotators."""
    bundle, key = _a1_bundle()
    recs = [
        _llm("a1_t1", "A1", _a1_resp(["n1"]), who="llm:model-a"),
        _llm("a1_t1", "A1", _a1_resp(["n1"]), who="llm:model-b"),
    ]
    cons = consensus(bundle, recs, key=key)
    assert cons.units == (), "no human rater judged anything here"
    assert cons.rater_kinds == ("human",)


def test_an_llm_does_not_complete_a_lone_human():
    """One person plus a model is one person. The unit must stay unresolved."""
    bundle, key = _a1_bundle()
    recs = [
        _rec("a1_t1", "alice", "A1", _a1_resp(["n1"])),
        _llm("a1_t1", "A1", _a1_resp(["n1"])),
    ]
    cons = consensus(bundle, recs, key=key)
    assert cons.units == ()
    assert any(d["reason"] == "insufficient_raters" for d in cons.disagreements)


def test_llm_records_never_reach_gold_human_asked():
    graph = _graph([_node("n1"), _node("n2")])
    bundle, key = _a1_bundle()
    recs = [
        _llm("a1_t1", "A1", _a1_resp(["n1"]), who="llm:model-a"),
        _llm("a1_t1", "A1", _a1_resp(["n1"]), who="llm:model-b"),
    ]
    rows = merge_into_graphs({"t1": graph}, consensus(bundle, recs, key=key), out_version="v1h")
    assert all(n["gold_human_asked"] is None for n in rows[0]["gold_nodes"])


def test_merge_refuses_a_consensus_that_included_a_model():
    """The belt to `consensus`'s braces: a caller that opts a model in cannot then write it to
    gold. Two humans and a model is still not three annotators for this purpose."""
    graph = _graph([_node("n1"), _node("n2")])
    bundle, key = _a1_bundle()
    recs = [
        _rec("a1_t1", "alice", "A1", _a1_resp(["n1"])),
        _rec("a1_t1", "bob", "A1", _a1_resp(["n1"])),
        _llm("a1_t1", "A1", _a1_resp(["n1"])),
    ]
    cons = consensus(bundle, recs, key=key, rater_kinds=("human", "llm"))
    assert cons.rater_kinds == ("human", "llm")
    with pytest.raises(LLMAnnotationNotGold):
        merge_into_graphs({"t1": graph}, cons, out_version="v1h")


def test_humans_still_resolve_with_a_model_in_the_file():
    """The model's presence must not cost the humans their consensus."""
    graph = _graph([_node("n1"), _node("n2")])
    bundle, key = _a1_bundle()
    recs = [
        _rec("a1_t1", "alice", "A1", _a1_resp(["n1"])),
        _rec("a1_t1", "bob", "A1", _a1_resp(["n1"])),
        _llm("a1_t1", "A1", _a1_resp(["n2"])),  # disagrees; must not matter
    ]
    cons = consensus(bundle, recs, key=key)
    rows = merge_into_graphs({"t1": graph}, cons, out_version="v1h")
    by_id = {n["gold_node_id"]: n for n in rows[0]["gold_nodes"]}
    assert by_id["n1"]["gold_human_asked"] is True
    assert by_id["n2"]["gold_human_asked"] is False


# ------------------------------------------------------------------ identity lock


def test_an_llm_id_without_the_kind_is_refused():
    """Two-way lock: the id and the kind must agree, so neither a forgotten field nor a
    borrowed name can smuggle a model in as a person."""
    bundle, _key = _a1_bundle()
    sneaky = _rec("a1_t1", "llm:model-a", "A1", _a1_resp([]))
    assert any("annotator_kind" in e for e in validate_records(bundle, [sneaky]))

    other = _rec("a1_t1", "alice", "A1", _a1_resp([]), kind="llm")
    assert any("annotator_kind" in e for e in validate_records(bundle, [other]))

    good_h = _rec("a1_t1", "alice", "A1", _a1_resp([]))
    good_l = _llm("a1_t1", "A1", _a1_resp([]))
    assert validate_records(bundle, [good_h]) == []
    assert validate_records(bundle, [good_l]) == []


def test_an_llm_record_must_name_its_model():
    """A model verdict with no pin is unattributable: two models, one annotator_id, and the
    agreement number silently mixes them."""
    bundle, _key = _a1_bundle()
    unpinned = _rec("a1_t1", "llm:model-a", "A1", _a1_resp([]), kind="llm")
    assert any("model_pin" in e for e in validate_records(bundle, [unpinned]))


# ------------------------------------------------------------------ gates untouched


def test_matcher_kappa_ignores_a_model_rater():
    """G-M2 is agreement between the matcher and a HUMAN. A model added there measures two
    machines agreeing, which is not what the gate is about."""
    items = [_item("A3_match", "m0"), _item("A3_match", "m1")]
    bundle, key = _bundle(
        items, {"m0": {"matcher_addresses": True}, "m1": {"matcher_addresses": True}}
    )
    humans = [
        _rec(i, w, "A3_match", {"addresses": True}) for i in ("m0", "m1") for w in ("alice", "bob")
    ]
    with_model = humans + [_llm(i, "A3_match", {"addresses": False}) for i in ("m0", "m1")]
    a = gate_numbers(consensus(bundle, humans, key=key), key)
    b = gate_numbers(consensus(bundle, with_model, key=key), key)
    assert a["matcher_kappa"] == pytest.approx(1.0)
    assert b["matcher_kappa"] == pytest.approx(a["matcher_kappa"])
    assert b["n_matcher_kappa"] == a["n_matcher_kappa"] == 2


def test_alpha_reports_humans_only():
    """The reported inter-annotator agreement is between annotators. A model in the file must
    not move it."""
    items = [_item("A3_edge", f"e{i}") for i in range(2)]
    bundle, key = _bundle(items)
    humans = [
        _rec(i, w, "A3_edge", {"holds": True}) for i in ("e0", "e1") for w in ("alice", "bob")
    ]
    noisy = humans + [_llm(i, "A3_edge", {"holds": False}) for i in ("e0", "e1")]
    assert (
        iaa_report(bundle, noisy, key=key)["A3_edge"]
        == iaa_report(bundle, humans, key=key)["A3_edge"]
    )


# ------------------------------------------------------------------ what it IS for


def test_llm_agreement_scores_the_model_against_the_humans():
    """The useful question: does the model track the people? Reported, never gated."""
    items = [_item("A3_edge", f"e{i}") for i in range(4)]
    bundle, key = _bundle(items)
    human_votes = {"e0": True, "e1": True, "e2": False, "e3": False}
    llm_votes = {"e0": True, "e1": True, "e2": False, "e3": True}  # wrong on one
    recs = [
        _rec(i, w, "A3_edge", {"holds": v})
        for i, v in human_votes.items()
        for w in ("alice", "bob")
    ]
    recs += [_llm(i, "A3_edge", {"holds": v}) for i, v in llm_votes.items()]

    rep = llm_agreement(bundle, recs, key=key)
    edge = rep["by_kind"]["A3_edge"]
    assert edge["n"] == 4
    assert edge["agreement"] == pytest.approx(0.75)
    assert rep["by_model"]["llm:gpt-oss-120b"]["n"] == 4


def test_llm_agreement_is_absent_where_no_human_resolved_the_unit():
    """A model answer on an item no two people reached is not evidence about the model."""
    items = [_item("A3_edge", "e0")]
    bundle, key = _bundle(items)
    recs = [
        _rec("e0", "alice", "A3_edge", {"holds": True}),  # one human only
        _llm("e0", "A3_edge", {"holds": True}),
    ]
    rep = llm_agreement(bundle, recs, key=key)
    assert rep["by_kind"] == {}
    assert math.isnan(rep["overall"]["agreement"])
    assert rep["overall"]["n"] == 0
