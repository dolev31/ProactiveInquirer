"""Human annotation records -> gold fields, and the one metric that turns on when they land.

ADR is the reason this pipeline exists. `gold_human_asked` is None on every node this
repository has ever built, so `anticipated_discovery_rate` correctly refuses to compute and
`pi_eval.score` writes no row. The flagship test here is the before/after: the identical
graph, scored with and without an imported annotation, must go from NOT COMPUTABLE to a real
number. A test that only checked the "after" half could pass against a function that always
returned something.
"""

from __future__ import annotations

import math

import pytest

from pi_eval.annotate import (
    consensus,
    human_judgment_rows,
    item_set_hash,
    merge_into_graphs,
    validate_records,
)
from pi_eval.gold import GoldEdge, GoldGraph, GoldNode
from pi_eval.matcher.base import MatchRecord
from pi_eval.metrics.human import anticipated_discovery_rate

BUNDLE_ID = "musique-0-deadbeef"


# ------------------------------------------------------------------ fixtures


def _node(nid, **kw):
    kw.setdefault("gold_partition", "required")
    kw.setdefault("gold_discoverability", "kb")
    return GoldNode(
        gold_suite="musique",
        gold_task_key="t1",
        gold_node_id=nid,
        gold_text=f"need {nid}",
        gold_graph_version="v1",
        **kw,
    )


def _graph(nodes=(), edges=()):
    return GoldGraph(
        gold_suite="musique",
        gold_task_key="t1",
        gold_nodes=tuple(nodes),
        gold_edges=tuple(edges),
        gold_graph_version="v1",
        gold_answer="Rome PINQCANARY_0123456789ABCDEF",
        gold_canary="PINQCANARY_0123456789ABCDEF",
        gold_corpus_hash="cafe1234",
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
    bundle = {
        "manifest": {
            "bundle_id": BUNDLE_ID,
            "tool_version": "pi_annotate/1",
            "suite": "musique",
            "graph_version": "v1",
            "item_set_hash": ish,
        },
        "items": list(items),
    }
    key = {
        "manifest": {"bundle_id": BUNDLE_ID, "item_set_hash": ish},
        "items": dict(key_items or {}),
    }
    return bundle, key


def _rec(iid, ann, task_type, response, **kw):
    r = {
        "record_id": f"{BUNDLE_ID}/{iid}/{ann}",
        "bundle_id": BUNDLE_ID,
        "item_id": iid,
        "task_type": task_type,
        "annotator_id": ann,
        "elapsed_ms": 1200,
        "ts": "2026-08-31T12:00:00Z",
        "tool_version": "pi_annotate/1",
        "response": response,
    }
    r.update(kw)
    return r


def _a1_bundle():
    """One checklist item over three needs, exactly the ADR universe shape."""
    items = [
        _item(
            "A1",
            "a1_t1",
            payload={"nodes": [{"node_id": n, "text": f"need {n}"} for n in ("n1", "n2", "n3")]},
        )
    ]
    return _bundle(items)


def _a1_records(ticked, useful=None):
    resp = {"ticked": list(ticked), "usefulness": dict(useful or {})}
    return [_rec("a1_t1", a, "A1", dict(resp)) for a in ("alice", "bob")]


def _match(nid):
    return MatchRecord("r1", "musique", "t1", nid, "resolve", 0, "m", "rule", 1.0, 1.0, "v1")


# ------------------------------------------------------------------ the flagship


def test_adr_emits_after_import():
    """Before: not computable, no row. After: a real number over a real denominator."""
    graph = _graph([_node("n1"), _node("n2"), _node("n3")])
    records = [_match("n1"), _match("n2"), _match("n3")]

    before = anticipated_discovery_rate(records, graph)
    assert before["n_annotated"] == 0
    assert math.isnan(before["adr"]), "the unannotated graph must not produce a number"

    bundle, key = _a1_bundle()
    cons = consensus(bundle, _a1_records(ticked=["n1"], useful={"n2": 4.0, "n3": 5.0}), key=key)
    rows = merge_into_graphs({"t1": graph}, cons, out_version="v1h")
    merged = _from_rows(rows)["t1"]

    after = anticipated_discovery_rate(records, merged)
    assert after["n_annotated"] == 3
    assert after["n_beyond_human"] == 2, "n2 and n3 were found and nobody asked for them"
    assert after["adr"] == pytest.approx(2 / 3)
    assert after["mean_usefulness_beyond_human"] == pytest.approx(4.5)


def _from_rows(rows):
    """Rows as `write_graphs` would serialise them, back into graphs. Round-tripping through
    the row form is the point: a field the serialiser drops is a field that never lands."""
    out = {}
    for r in rows:
        out[r["gold_task_key"]] = GoldGraph(
            gold_suite=r["gold_suite"],
            gold_task_key=r["gold_task_key"],
            gold_nodes=tuple(GoldNode(**_tup(n)) for n in r["gold_nodes"]),
            gold_edges=tuple(GoldEdge(**_tup(e)) for e in r["gold_edges"]),
            gold_graph_version=r["gold_graph_version"],
            gold_answer=r["gold_answer"],
            gold_canary=r["gold_canary"],
            gold_corpus_hash=r["gold_corpus_hash"],
        )
    return out


def _tup(d):
    return {k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items()}


# ------------------------------------------------------------------ write-back discipline


def test_unshown_nodes_keep_their_absent_annotation():
    """A node nobody was shown is not evidence of anything. False there would put it in
    ADR's numerator on the strength of a question never asked."""
    graph = _graph([_node("n1"), _node("n2"), _node("n3"), _node("n4")])
    bundle, key = _a1_bundle()  # shows n1..n3 only
    cons = consensus(bundle, _a1_records(ticked=["n1"]), key=key)
    merged = _from_rows(merge_into_graphs({"t1": graph}, cons, out_version="v1h"))["t1"]
    by_id = {n.gold_node_id: n for n in merged.gold_nodes}
    assert by_id["n1"].gold_human_asked is True
    assert by_id["n2"].gold_human_asked is False
    assert by_id["n4"].gold_human_asked is None


def test_v1h_preserves_canary_and_topology():
    """The human fields are the ONLY difference. Structural metrics must be unmoved, and the
    canary must survive: a re-serialised graph that lost its nonce is a disarmed firewall."""
    from pi_eval.score import graph_hash

    nodes = [_node("n1"), _node("n2"), _node("n3")]
    edges = [
        GoldEdge(
            gold_suite="musique",
            gold_task_key="t1",
            gold_src_node_id="n1",
            gold_dst_node_id="n2",
            gold_graph_version="v1",
        )
    ]
    graph = _graph(nodes, edges)
    bundle, key = _a1_bundle()
    cons = consensus(bundle, _a1_records(ticked=["n1"]), key=key)
    rows = merge_into_graphs({"t1": graph}, cons, out_version="v1h")
    merged = _from_rows(rows)["t1"]

    assert merged.gold_canary == graph.gold_canary
    assert merged.gold_answer == graph.gold_answer
    assert merged.gold_corpus_hash == graph.gold_corpus_hash
    assert merged.gold_graph_version == "v1h"
    assert all(n.gold_graph_version == "v1h" for n in merged.gold_nodes)
    assert all(e.gold_graph_version == "v1h" for e in merged.gold_edges)

    structural = lambda g: (  # noqa: E731
        sorted((n.gold_node_id, n.gold_depth, n.gold_partition) for n in g.gold_nodes),
        sorted((e.gold_src_node_id, e.gold_dst_node_id) for e in g.gold_edges),
    )
    assert structural(merged) == structural(graph)
    assert graph_hash({"musique": {"t1": merged}}) != graph_hash({"musique": {"t1": graph}})


def test_node_verdicts_do_not_rewrite_the_partition():
    """A human 'not a need' on a mined-required node is a PRECISION datum for G-M1. Editing
    the partition would move every coverage metric's universe mid-campaign instead."""
    graph = _graph([_node("n1", gold_partition="required")])
    items = [_item("A3_node", "an1", gold_node_id="n1")]
    bundle, key = _bundle(items)
    recs = [
        _rec("an1", a, "A3_node", {"verdict": "not_a_need", "discoverability": "kb"})
        for a in ("alice", "bob")
    ]
    merged = _from_rows(
        merge_into_graphs({"t1": graph}, consensus(bundle, recs, key=key), out_version="v1h")
    )["t1"]
    node = merged.gold_nodes[0]
    assert node.gold_partition == "required"
    assert node.gold_human_adjudicated is True


def test_disagreement_withheld_without_adjudication():
    graph = _graph([_node("n1"), _node("n2"), _node("n3")])
    bundle, key = _a1_bundle()
    recs = [
        _rec("a1_t1", "alice", "A1", {"ticked": ["n1"], "usefulness": {}}),
        _rec("a1_t1", "bob", "A1", {"ticked": ["n1", "n2"], "usefulness": {}}),
    ]
    cons = consensus(bundle, recs, key=key)
    merged = _from_rows(merge_into_graphs({"t1": graph}, cons, out_version="v1h"))["t1"]
    by_id = {n.gold_node_id: n for n in merged.gold_nodes}
    assert by_id["n1"].gold_human_asked is True, "they agreed on n1"
    assert by_id["n2"].gold_human_asked is None, "they disagreed on n2: nothing may be written"
    assert any(d["unit_id"].endswith("/n2") for d in cons.disagreements)


def test_single_rater_is_not_a_consensus():
    graph = _graph([_node("n1"), _node("n2"), _node("n3")])
    bundle, key = _a1_bundle()
    cons = consensus(bundle, [_rec("a1_t1", "alice", "A1", {"ticked": ["n1"]})], key=key)
    merged = _from_rows(merge_into_graphs({"t1": graph}, cons, out_version="v1h"))["t1"]
    assert all(n.gold_human_asked is None for n in merged.gold_nodes)


# ------------------------------------------------------------------ validation


def test_import_refuses_foreign_records():
    bundle, _key = _a1_bundle()
    ok = _rec("a1_t1", "alice", "A1", {"ticked": []})
    assert validate_records(bundle, [ok]) == []

    unknown = _rec("nope", "alice", "A1", {"ticked": []})
    assert any("unknown item" in e for e in validate_records(bundle, [unknown]))

    wrong_bundle = _rec("a1_t1", "alice", "A1", {"ticked": []}, bundle_id="other")
    assert any("bundle" in e for e in validate_records(bundle, [wrong_bundle]))

    wrong_type = _rec("a1_t1", "alice", "A3_edge", {"holds": True})
    assert any("task_type" in e for e in validate_records(bundle, [wrong_type]))

    dupe = [ok, _rec("a1_t1", "alice", "A1", {"ticked": ["n1"]})]
    assert any("duplicate" in e for e in validate_records(bundle, dupe))

    ghost = _rec("a1_t1", "alice", "A1", {"ticked": ["n99"]})
    assert any("n99" in e for e in validate_records(bundle, [ghost]))


def test_import_refuses_a_records_file_from_a_different_bundle_build():
    """Same bundle_id, different items: the hash is what catches a re-export nobody noticed."""
    bundle, _ = _a1_bundle()
    bundle["manifest"]["item_set_hash"] = "0" * 16
    errs = validate_records(bundle, [_rec("a1_t1", "alice", "A1", {"ticked": []})])
    assert any("item_set_hash" in e for e in errs)


# ------------------------------------------------------------------ A2 -> judgments


def _a2_bundle(order="ab"):
    items = [
        _item(
            "A2",
            "p0",
            payload={
                "option_a": {"question": "which region did Andy sail to"},
                "option_b": {"question": "what city was Gotham filmed in"},
            },
            run_id="parent",
            turn_idx=1,
            scorer_hash="sc0",
        )
    ]
    key_items = {
        "p0": {
            "order": order,
            "chosen_run_id": "c1",
            "rejected_run_id": "c2",
            "margin": 0.33,
            "pair_id": "pid0",
        }
    }
    return _bundle(items, key_items)


def test_a2_labels_are_canonicalised_by_the_key_not_by_the_shown_slot():
    """The bundle the annotators see must not say which candidate the pipeline preferred, or
    the judgment is contaminated by the thing it is meant to check. The mapping lives in the
    key file, which never ships."""
    picked_a = [_rec("p0", a, "A2", {"choice": "a"}) for a in ("alice", "bob")]

    ab_bundle, ab_key = _a2_bundle(order="ab")
    ba_bundle, ba_key = _a2_bundle(order="ba")
    assert ab_bundle["items"] == ba_bundle["items"], "the shipped item must not encode the order"

    ab = consensus(ab_bundle, picked_a, key=ab_key)
    ba = consensus(ba_bundle, picked_a, key=ba_key)
    assert [u.label for u in ab.units] == ["chosen"]
    assert [u.label for u in ba.units] == ["rejected"]


def test_human_judgment_rows_carry_the_candidate_runs_and_a_stable_id():
    bundle, key = _a2_bundle(order="ab")
    recs = [_rec("p0", a, "A2", {"choice": "a"}) for a in ("alice", "bob")]
    rows = human_judgment_rows(bundle, recs, key=key)
    assert len(rows) == 2
    r = rows[0]
    assert r["judge_family"] == "human" and r["judge_model"] in ("alice", "bob")
    assert r["run_id_a"] == "c1" and r["run_id_b"] == "c2"
    assert r["pref_sign"] == 1
    assert r["order"] == "ab"
    assert r["criterion"] == "human_preference"
    assert r["len_a_words"] == 6 and r["len_b_words"] == 6
    assert human_judgment_rows(bundle, recs, key=key)[0]["judgment_id"] == r["judgment_id"]
    assert len({x["judgment_id"] for x in rows}) == 2


def test_a_tie_is_a_zero_sign_and_both_bad_is_labelled():
    bundle, key = _a2_bundle()
    rows = human_judgment_rows(
        bundle,
        [
            _rec("p0", "alice", "A2", {"choice": "tie"}),
            _rec("p0", "bob", "A2", {"choice": "both_bad"}),
        ],
        key=key,
    )
    by_ann = {r["judge_model"]: r for r in rows}
    assert by_ann["alice"]["pref_sign"] == 0 and by_ann["alice"]["label"] == "tie"
    assert by_ann["bob"]["pref_sign"] == 0 and by_ann["bob"]["label"] == "both_bad"


def test_judgment_rows_fill_every_column_the_table_declares():
    from pi_eval import schema as sch

    bundle, key = _a2_bundle()
    rows = human_judgment_rows(bundle, [_rec("p0", "alice", "A2", {"choice": "b"})], key=key)
    assert set(rows[0]) == set(sch.JUDGMENTS.names)
    sch.to_table("judgments", rows)  # raises if a value does not fit the declared type


def test_ratings_on_anticipated_needs_survive_the_import():
    """The reference level for `adr_usefulness` comes from the needs the annotator DID tick.
    An import that carried ratings only for unticked needs would leave the beyond-human mean
    with nothing to be read against -- and would mean the tool asked for a rating on only one
    branch, which makes ticking the cheap answer and biases ADR toward zero."""
    graph = _graph([_node("n1"), _node("n2"), _node("n3")])
    records = [_match("n1"), _match("n2"), _match("n3")]
    bundle, key = _a1_bundle()
    cons = consensus(
        bundle,
        _a1_records(ticked=["n1"], useful={"n1": 2.0, "n2": 4.0, "n3": 5.0}),
        key=key,
    )
    merged = _from_rows(merge_into_graphs({"t1": graph}, cons, out_version="v1h"))["t1"]
    by_id = {n.gold_node_id: n for n in merged.gold_nodes}
    assert by_id["n1"].gold_human_asked is True
    assert by_id["n1"].gold_usefulness_rating == pytest.approx(2.0)

    out = anticipated_discovery_rate(records, merged)
    assert out["mean_usefulness_beyond_human"] == pytest.approx(4.5)
    assert out["mean_usefulness_anticipated"] == pytest.approx(2.0)
    assert out["n_rated"] == 2 and out["n_rated_anticipated"] == 1


def test_a_rating_on_a_shown_need_is_valid_whether_or_not_it_was_ticked():
    bundle, _key = _a1_bundle()
    rec = _rec("a1_t1", "alice", "A1", {"ticked": ["n1"], "usefulness": {"n1": 3, "n2": 4}})
    assert validate_records(bundle, [rec]) == []
