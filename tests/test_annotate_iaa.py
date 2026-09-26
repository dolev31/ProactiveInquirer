"""Agreement statistics over human annotation, against values computed by hand.

The matcher enters `matcher_kappa` AS ONE MORE RATER. That is not a convenience: G-M2 asks
whether the matcher agrees with a human on the same (ask, node) decisions, which is an
inter-rater question, and Krippendorff's alpha is the estimator this repository already uses
for it (paired.krippendorff_alpha_nominal, chosen over Cohen's kappa because it tolerates the
unbalanced rater sets that are the normal case here).
"""

from __future__ import annotations

import math

import pytest

from pi_eval.annotate import consensus, gate_numbers, iaa_report, item_set_hash


def _item(task_type: str, iid: str, **kw):
    payload = kw.pop("payload", {})
    prov = {"suite": "musique", "task_id": "t1", "task_key": "t1", "graph_version": "v1"}
    prov.update(kw)
    return {
        "item_id": iid,
        "task_type": task_type,
        "context": {},
        "payload": payload,
        "provenance": prov,
    }


def _bundle(items, key_items=None):
    return (
        {
            "manifest": {
                "bundle_id": "musique-0-deadbeef",
                "tool_version": "pi_annotate/1",
                "suite": "musique",
                "graph_version": "v1",
                "item_set_hash": item_set_hash(items),
            },
            "items": list(items),
        },
        {
            "manifest": {
                "bundle_id": "musique-0-deadbeef",
                "item_set_hash": item_set_hash(items),
            },
            "items": dict(key_items or {}),
        },
    )


def _rec(iid, ann, task_type, response, bundle_id="musique-0-deadbeef"):
    return {
        "record_id": f"{bundle_id}/{iid}/{ann}",
        "bundle_id": bundle_id,
        "item_id": iid,
        "task_type": task_type,
        "annotator_id": ann,
        "elapsed_ms": 1000,
        "ts": "2026-08-31T12:00:00Z",
        "tool_version": "pi_annotate/1",
        "response": response,
    }


# ------------------------------------------------------------------ alpha


def test_alpha_matches_hand_computed_value():
    """Four units, two raters, one disagreement.

    D_o = 2/8 = 0.25. Counts are x=5, y=3 over n=8, so
    D_e = 1 - (5*4 + 3*2)/(8*7) = 15/28, and alpha = 1 - (1/4)/(15/28) = 8/15.
    """
    items = [_item("A3_edge", f"e{i}") for i in range(4)]
    bundle, key = _bundle(items)
    holds, no = {"holds": True}, {"holds": False}
    records = [
        _rec("e0", "a", "A3_edge", holds),
        _rec("e0", "b", "A3_edge", holds),
        _rec("e1", "a", "A3_edge", holds),
        _rec("e1", "b", "A3_edge", holds),
        _rec("e2", "a", "A3_edge", no),
        _rec("e2", "b", "A3_edge", no),
        _rec("e3", "a", "A3_edge", holds),
        _rec("e3", "b", "A3_edge", no),
    ]
    rep = iaa_report(bundle, records, key=key)
    assert rep["A3_edge"]["alpha"] == pytest.approx(8 / 15)
    assert rep["A3_edge"]["n_units"] == 4
    assert rep["A3_edge"]["n_multi_rated"] == 4


def test_alpha_is_absent_not_zero_when_nobody_double_annotated():
    """One rater has nobody to agree with. A 0.0 there would read as total disagreement."""
    items = [_item("A3_edge", "e0")]
    bundle, key = _bundle(items)
    rep = iaa_report(bundle, [_rec("e0", "a", "A3_edge", {"holds": True})], key=key)
    assert math.isnan(rep["A3_edge"]["alpha"])
    assert rep["A3_edge"]["n_multi_rated"] == 0


# ------------------------------------------------------------------ matcher kappa


def _match_bundle(decisions: dict[str, bool]):
    items = [
        _item("A3_match", iid, run_id="r1", turn_idx=1, gold_node_id=f"n{iid}") for iid in decisions
    ]
    key_items = {iid: {"matcher_addresses": v} for iid, v in decisions.items()}
    return _bundle(items, key_items)


def test_matcher_kappa_treats_matcher_as_rater():
    """Perfect agreement is 1.0; one flip must move it off 1.0."""
    decisions = {"m0": True, "m1": True, "m2": False, "m3": False}
    bundle, key = _match_bundle(decisions)

    def recs(labels):
        out = []
        for iid, v in labels.items():
            for ann in ("a", "b"):
                out.append(_rec(iid, ann, "A3_match", {"addresses": v}))
        return out

    agree = gate_numbers(consensus(bundle, recs(decisions), key=key), key)
    assert agree["matcher_kappa"] == pytest.approx(1.0)
    assert agree["n_matcher_kappa"] == 4

    flipped = dict(decisions)
    flipped["m0"] = False
    disagree = gate_numbers(consensus(bundle, recs(flipped), key=key), key)
    assert disagree["matcher_kappa"] < 1.0


def test_matcher_kappa_is_absent_without_matcher_decisions():
    """A gate computed from a key that never recorded what the matcher said is not a gate."""
    bundle, key = _match_bundle({"m0": True})
    key["items"]["m0"] = {}
    recs = [_rec("m0", a, "A3_match", {"addresses": True}) for a in ("a", "b")]
    out = gate_numbers(consensus(bundle, recs, key=key), key)
    assert math.isnan(out["matcher_kappa"])
    assert out["n_matcher_kappa"] == 0


# ------------------------------------------------------------------ the other two gates


def test_edge_precision_excludes_unsure_and_counts_it():
    items = [_item("A3_edge", f"e{i}") for i in range(4)]
    bundle, key = _bundle(items)
    votes = {"e0": True, "e1": True, "e2": False, "e3": "unsure"}
    records = [
        _rec(iid, ann, "A3_edge", {"holds": v}) for iid, v in votes.items() for ann in ("a", "b")
    ]
    out = gate_numbers(consensus(bundle, records, key=key), key)
    assert out["edge_precision"] == pytest.approx(2 / 3)
    assert out["n_edge_precision"] == 3
    assert out["n_edge_unsure"] == 1


def test_node_recall_is_absent_until_missing_needs_are_adjudicated():
    """Two annotators' free text cannot be consensus'd mechanically. Computing a recall as
    if it could would put a number on a merge nobody performed."""
    items = [_item("A3_node", f"n{i}") for i in range(3)] + [_item("A3_missing", "miss0")]
    bundle, key = _bundle(items)
    records = [
        _rec(iid, ann, "A3_node", {"verdict": v, "discoverability": "kb"})
        for iid, v in (("n0", "required"), ("n1", "optional"), ("n2", "not_a_need"))
        for ann in ("a", "b")
    ]
    records += [
        _rec("miss0", "a", "A3_missing", {"missing_needs": ["the year of the flood"]}),
        _rec("miss0", "b", "A3_missing", {"missing_needs": ["what year it flooded"]}),
    ]
    cons = consensus(bundle, records, key=key)
    out = gate_numbers(cons, key)
    assert math.isnan(out["node_recall"])
    assert "adjudicat" in out["node_recall_absent_because"]
    assert len(cons.missing) == 2

    adjudicated = gate_numbers(cons, key, n_missing_adjudicated=1)
    assert adjudicated["node_recall"] == pytest.approx(2 / 3)


# ------------------------------------------------------------------ attention checks


def test_attention_check_items_never_reach_a_measurement():
    """An attention-check item's key entry carries `attention_check`; `_collect` must skip it
    so it forms no consensus unit, contributes no alpha, and enters no gate number -- the HARD
    CONSTRAINT that lets a planted attention check exist at all without contaminating a
    published number. Agreement is not what excludes it (both annotators here answer the
    attention items CORRECTLY and unanimously) -- only the key flag does."""
    real_items = [_item("A3_node", f"n{i}") for i in range(2)]
    attn_items = [_item("A3_node", f"attn{i}") for i in range(2)]
    items = real_items + attn_items
    key_items = {
        it["item_id"]: {"attention_check": {"expected": "not_a_need"}} for it in attn_items
    }
    bundle, key = _bundle(items, key_items)

    records = []
    for it in real_items:
        for ann in ("a", "b"):
            records.append(
                _rec(
                    it["item_id"], ann, "A3_node", {"verdict": "required", "discoverability": "kb"}
                )
            )
    for it in attn_items:
        for ann in ("a", "b"):
            records.append(
                _rec(
                    it["item_id"],
                    ann,
                    "A3_node",
                    {"verdict": "not_a_need", "discoverability": "kb"},
                )
            )

    cons = consensus(bundle, records, key=key)
    # Each real A3_node record also carries a discoverability sub-verdict, which resolves to
    # its OWN unit (kind "A3_node_disc") -- so "only the real items form a unit" is checked
    # kind-by-kind rather than as a raw count over the mixed `units` tuple.
    assert len(cons.by_kind("A3_node")) == 2, "only the 2 real items may resolve to a unit"
    assert {u.unit_id for u in cons.by_kind("A3_node")} == {it["item_id"] for it in real_items}
    assert len(cons.by_kind("A3_node_disc")) == 2

    rep = iaa_report(bundle, records, key=key)
    assert rep["A3_node"]["n_units"] == 2

    gates = gate_numbers(cons, key)
    assert gates["n_node_verdicts"] == 2
    assert gates["node_precision"] == pytest.approx(1.0)  # both real verdicts were "required"


def test_gates_feed_gold_validate():
    """The numbers must arrive in the shape `pi gold validate` already consumes.

    NaN is passed as None, never as a float: `gate_report` compares with `>=`, and
    `nan >= 0.75` is False — an unmeasured gate would print FAIL, which is a different claim
    from NOT RUN and the wrong one.
    """
    from pi_eval.mining.pipeline import gate_report

    items = [_item("A3_node", "n0"), _item("A3_edge", "e0")]
    bundle, key = _bundle(items)
    records = [_rec("e0", a, "A3_edge", {"holds": True}) for a in ("a", "b")]
    records += [
        _rec("n0", a, "A3_node", {"verdict": "required", "discoverability": "kb"})
        for a in ("a", "b")
    ]
    out = gate_numbers(consensus(bundle, records, key=key), key, n_missing_adjudicated=0)
    assert out["node_recall"] == pytest.approx(1.0)
    assert out["edge_precision"] == pytest.approx(1.0)

    gates = gate_report(
        [],
        node_recall=out["node_recall"],
        edge_precision=out["edge_precision"],
        matcher_kappa=None if math.isnan(out["matcher_kappa"]) else out["matcher_kappa"],
    )
    names = {g.name for g in gates}
    assert "G-M1 node recall vs human gold" in names
    assert "G-M1 edge precision vs human gold" in names
    assert "G-M2 matcher kappa" not in names, "an unmeasured gate must stay absent, not FAIL"
    assert all(g.passed for g in gates if g.name.startswith("G-M1"))
