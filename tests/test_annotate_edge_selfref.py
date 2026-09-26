"""An edge item whose DST names its own SRC cannot measure edge_precision.

MuSiQue's `#N` placeholders had to be rendered somehow for a human to read an edge item, and
both obvious renderings are degenerate in opposite directions. Substituting the sub-ANSWER
makes DST self-contained, so every true prerequisite reads as unnecessary (measured: a model
rejected 6 of 6). Substituting a bracketed REFERENCE puts SRC's text inside DST, so every
item reads "does <X> >> place-of-death require X?" and the answer is yes by construction.

The second failure is the dangerous one, because it does not look like a failure. It yields
edge_precision = 1.00, which clears the G-M1 edge gate at 0.80 and licenses a depth claim --
on a number that measures the exporter's string formatting and nothing about the graph.

So: self-referential edge units are excluded from the ratio and counted separately, and an
edge set with nothing left reports NaN, not 1.0. `gate_report` compares with `>=`, so the
distinction between "no measurement" and "a passing measurement" has to survive all the way
to the flag, and NaN is what carries it.
"""

from __future__ import annotations

import math

from pi_eval.annotate import bundle_shape_errors, consensus, gate_numbers


def _edge_item(iid: str, src: str, dst: str) -> dict:
    return {
        "item_id": iid,
        "task_type": "A3_edge",
        "context": {
            "question": "Where did Alice Walton's mother die?",
            "src_text": src,
            "dst_text": dst,
        },
        "payload": {},
        "provenance": {
            "suite": "musique",
            "task_id": "t1",
            "graph_version": "v1",
            "gold_src_node_id": "s1",
            "gold_dst_node_id": "s2",
        },
    }


def _rec(iid: str, ann: str, holds: object = True) -> dict:
    return {
        "record_id": f"b/{iid}/{ann}",
        "bundle_id": "b",
        "item_id": iid,
        "task_type": "A3_edge",
        "annotator_id": ann,
        "response": {"holds": holds},
    }


SELF_REF_DST = "⟨Alice Louise Walton >> mother⟩ >> place of death"
CLEAN_DST = "#1 >> place of death"


def _bundle(items: list[dict]) -> dict:
    return {"manifest": {"bundle_id": "b"}, "items": items}


def test_edge_precision_is_nan_when_every_item_names_its_own_src() -> None:
    """The whole exported musique edge set is this shape: 200 of 200."""
    items = [_edge_item("e1", "Alice Louise Walton >> mother", SELF_REF_DST)]
    bundle = _bundle(items)
    cons = consensus(bundle, [_rec("e1", "a"), _rec("e1", "b")], rater_kinds=("human",))

    g = gate_numbers(cons, None, bundle=bundle)

    assert math.isnan(g["edge_precision"]), (
        f"edge_precision came back {g['edge_precision']!r}; a self-referential item makes it "
        "1.0 by construction and that would clear the G-M1 edge gate on a rendering artifact"
    )
    assert g["n_edge_precision"] == 0
    assert g["n_edge_self_referential"] == 1


def test_edge_precision_counts_only_the_items_that_can_discriminate() -> None:
    items = [
        _edge_item("e1", "Alice Louise Walton >> mother", SELF_REF_DST),
        _edge_item("e2", "Crazy Alien >> director", CLEAN_DST),
        _edge_item("e3", "Edges of the Lord >> director", CLEAN_DST),
    ]
    bundle = _bundle(items)
    recs = [
        *[_rec("e1", a, True) for a in ("a", "b")],
        *[_rec("e2", a, True) for a in ("a", "b")],
        *[_rec("e3", a, False) for a in ("a", "b")],
    ]
    cons = consensus(bundle, recs, rater_kinds=("human",))

    g = gate_numbers(cons, None, bundle=bundle)

    # e1 excluded; of the two that survive one holds and one does not.
    assert g["n_edge_self_referential"] == 1
    assert g["n_edge_precision"] == 2
    assert g["edge_precision"] == 0.5


def test_bundle_shape_refuses_to_ship_a_self_referential_edge_item() -> None:
    """Catch it at export, so the degenerate item never reaches an annotator's screen."""
    errs = bundle_shape_errors(
        _bundle([_edge_item("e1", "Alice Louise Walton >> mother", SELF_REF_DST)])
    )
    assert any("self-referential" in e for e in errs), errs

    assert not [
        e
        for e in bundle_shape_errors(
            _bundle([_edge_item("e2", "Crazy Alien >> director", CLEAN_DST)])
        )
        if "self-referential" in e
    ]
