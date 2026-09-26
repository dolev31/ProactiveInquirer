"""Gate B: the numbers that decide whether the B labels are usable at all.

Four checks, and each one can kill the stage:

  foil catch       a rater who misses planted foils is not reading the state
  agreement        alpha per unit kind, across rater families, never within one
  base rates       if nothing is wasted and nothing is missed, there is nothing to learn
  coverage         how many items produced a decided verdict at all

The gate reports; it does not decide quietly. Every number carries its n, and a number that
cannot be computed is absent rather than zero.
"""

import math

import pytest

from pi_eval.build.convlog_gate import GateB, gate_b


def _bundle(n=10):
    items = []
    for i in range(n):
        items.append(
            {
                "task_type": "B1",
                "item_id": f"b1-{i}",
                "provenance": {"session": f"s{i % 3}"},
                "context": {
                    "task": "t",
                    "prior_turns": "p",
                    "tool_trace": "tt",
                    "asked_question": "q?",
                },
                "payload": {},
            }
        )
    return {"manifest": {"bundle_id": "pilot"}, "items": items}


def _key(bundle, foils=("b1-0", "b1-1")):
    return {
        "items": {
            i["item_id"]: (
                {
                    "session_id": "s",
                    "dp_index": 0,
                    "kind": "yield",
                    "observed_action": "ASK_USER",
                    "foil": True,
                    "expected": "answer_in_state",
                }
                if i["item_id"] in foils
                else {
                    "session_id": "s",
                    "dp_index": 0,
                    "kind": "yield",
                    "observed_action": "ASK_USER",
                }
            )
            for i in bundle["items"]
        }
    }


def _rec(item_id, annotator, verdict, tt="B1"):
    return {
        "item_id": item_id,
        "task_type": tt,
        "annotator_id": annotator,
        "annotator_kind": "llm" if annotator.startswith("llm:") else "human",
        "response": {"verdict": verdict},
    }


def test_foil_catch_is_measured_per_rater():
    b = _bundle()
    k = _key(b)
    recs = [
        _rec("b1-0", "llm:m1", "answer_in_state"),  # caught
        _rec("b1-1", "llm:m1", "necessary"),  # missed
        _rec("b1-0", "llm:m2", "answer_in_state"),
        _rec("b1-1", "llm:m2", "answer_in_state"),
    ]
    g = gate_b(b, k, recs)
    assert g.foil_catch["llm:m1"] == pytest.approx(0.5)
    assert g.foil_catch["llm:m2"] == pytest.approx(1.0)


def test_a_rater_shown_no_foil_gets_no_foil_score_rather_than_a_perfect_one():
    """A rater with no foils has not passed the check; it has not taken it. Recording 1.0
    would let a batch with no planted foils clear the gate."""
    b = _bundle()
    g = gate_b(b, _key(b), [_rec("b1-5", "llm:m3", "necessary")])
    assert "llm:m3" not in g.foil_catch


def test_base_rates_exclude_the_planted_foils():
    """A foil is an item built to have a known answer. Leaving it in the base rate measures
    the plant, not the corpus."""
    b = _bundle()
    k = _key(b)
    recs = [_rec("b1-0", "llm:m1", "answer_in_state")] + [
        _rec(f"b1-{i}", "llm:m1", "necessary") for i in range(2, 10)
    ]
    g = gate_b(b, k, recs)
    assert g.rates["wasted_ask_rate"].n == 8
    assert g.rates["wasted_ask_rate"].value == pytest.approx(0.0)


def test_agreement_is_computed_across_rater_families_not_within_one():
    """Two runs of one model agreeing tells you the model is deterministic, not that the
    instrument is legible. The A campaign's whole quarantine finding turned on this."""
    b = _bundle()
    k = _key(b, foils=())
    same_family = [_rec(f"b1-{i}", "llm:m1", "necessary") for i in range(10)] + [
        _rec(f"b1-{i}", "llm:m1-seed2", "necessary") for i in range(10)
    ]
    g = gate_b(b, k, same_family, families={"llm:m1": "f1", "llm:m1-seed2": "f1"})
    assert math.isnan(g.alpha.get("B1", float("nan")))


def test_agreement_is_reported_when_two_families_labelled_the_same_items():
    b = _bundle()
    k = _key(b, foils=())
    recs = []
    for i in range(10):
        recs.append(_rec(f"b1-{i}", "llm:m1", "necessary" if i % 2 else "answer_in_state"))
        recs.append(_rec(f"b1-{i}", "llm:m2", "necessary" if i % 2 else "answer_in_state"))
    g = gate_b(b, k, recs, families={"llm:m1": "f1", "llm:m2": "f2"})
    assert g.alpha["B1"] > 0.9


def test_the_verdict_names_every_criterion_that_failed():
    """A gate that returns one boolean makes a reader guess which half failed."""
    b = _bundle()
    k = _key(b)
    recs = [_rec("b1-0", "llm:m1", "necessary"), _rec("b1-1", "llm:m1", "necessary")]
    g = gate_b(b, k, recs)
    assert g.passed is False
    assert any("foil" in r for r in g.failures)


def test_an_empty_record_set_reports_absent_not_a_pass():
    g = gate_b(_bundle(), _key(_bundle()), [])
    assert g.passed is False
    assert math.isnan(g.rates["wasted_ask_rate"].value)
    assert isinstance(g, GateB)


def test_a_foil_is_scored_against_the_class_the_metric_uses_not_the_exact_label():
    """Measured on the pilot: 5 of gpt-oss-120b's 6 foil misses were `answer_inferable` where
    the plant expected `answer_in_state`, or `inferable_from_state` where it expected
    `stated_already`. Both members of each pair say the same thing -- the answer was already
    available -- and `metrics.convlog` ALREADY groups them into one numerator, because the rate
    is about the person's spent attention and both spend it.

    So a foil check that separates them is stricter than the metric it exists to validate, and
    it was reporting "did not read the state" for a rater that read it and picked the adjacent
    word. The strict count is still reported, so relaxing this cannot hide a real miss.
    """
    b = _bundle()
    k = _key(b)
    near = [_rec("b1-0", "llm:m1", "answer_inferable"), _rec("b1-1", "llm:m1", "default_existed")]
    g = gate_b(b, k, near)
    assert g.foil_catch["llm:m1"] == pytest.approx(1.0)
    assert g.foil_catch_exact["llm:m1"] == pytest.approx(0.0)


def test_a_foil_answered_with_the_opposite_class_is_still_a_miss():
    b = _bundle()
    g = gate_b(
        b, _key(b), [_rec("b1-0", "llm:m1", "necessary"), _rec("b1-1", "llm:m1", "cant_tell")]
    )
    assert g.foil_catch["llm:m1"] == pytest.approx(0.0)


def _rates(spec, kind="B1", verdict_map=None):
    """One record per rater per item, with each rater hitting `spec[rater]` of 20 items."""
    recs = []
    for ann, share in spec.items():
        for i in range(20):
            hit = i < round(share * 20)
            recs.append(_rec(f"b1-{i}", ann, "answer_in_state" if hit else "necessary", kind))
    return recs


def test_a_rater_whose_base_rate_is_an_outlier_is_named_not_silently_dropped():
    """The A campaign quarantined a rater that caught 99 of 99 foils, on a bias statistic and
    not on a hunch. Measured here: on B1 the four families report 73.1%, 67.2%, 62.7% and
    36.4% wasted. Three cluster; one does not.

    The rule is median absolute deviation, fixed in code, so it cannot be chosen after seeing
    which answer it produces. It REPORTS -- excluding a rater is a separate decision a person
    makes, because a rule that silently drops data is a rule nobody audits.
    """
    b = _bundle(20)
    k = _key(b, foils=())
    recs = _rates({"llm:a": 0.75, "llm:b": 0.65, "llm:c": 0.60, "llm:odd": 0.05})
    g = gate_b(b, k, recs)
    assert g.outliers["B1"] == ["llm:odd"]
    assert any("llm:odd" in f for f in g.failures)


def test_raters_that_agree_produce_no_outlier():
    b = _bundle(20)
    recs = _rates({"llm:a": 0.70, "llm:b": 0.65, "llm:c": 0.60})
    g = gate_b(b, _key(b, foils=()), recs)
    assert g.outliers.get("B1", []) == []


def test_fewer_than_three_raters_cannot_support_an_outlier_call():
    """With two raters there is no majority to be an outlier from; the median is the midpoint
    and both are equidistant from it."""
    b = _bundle(20)
    recs = _rates({"llm:a": 0.9, "llm:b": 0.1})
    g = gate_b(b, _key(b, foils=()), recs)
    assert g.outliers.get("B1", []) == []


def test_a_lone_dissenter_from_a_unanimous_majority_is_an_outlier():
    """The hole the B3 re-sample found. On that slice `should_have_asked` was returned 0% by
    both strong raters and 18% by the third. The median is 0 and so is the median absolute
    deviation, and a rule that divides by MAD reports nobody -- the one case where the
    disagreement is least ambiguous.

    When MAD is zero the raters are unanimous apart from the dissenters, and differing at all
    from a unanimous majority is what being an outlier means.
    """
    b = _bundle(20)
    recs = _rates({"llm:a": 0.0, "llm:b": 0.0, "llm:odd": 0.20})
    g = gate_b(b, _key(b, foils=()), recs)
    assert g.outliers["B1"] == ["llm:odd"]


def test_total_unanimity_produces_no_outlier():
    b = _bundle(20)
    recs = _rates({"llm:a": 0.5, "llm:b": 0.5, "llm:c": 0.5})
    g = gate_b(b, _key(b, foils=()), recs)
    assert g.outliers.get("B1", []) == []


def test_the_gate_reads_a_b1b_record_as_a_b1_unit():
    """`annotate._units` emits B1b under kind "B1" so the controlled pair lands in one column.
    The gate had its own label reader and did not know the variant, so a whole B1b pass scored
    zero on every criterion -- foil catch 0.000, coverage 0.000 -- which looks exactly like two
    raters failing and is in fact the reader dropping every row."""
    b = _bundle(4)
    for item in b["items"]:
        item["task_type"] = "B1b"
    k = _key(b, foils=("b1-0",))
    recs = [_rec(f"b1-{i}", "llm:a", "answer_in_state", "B1b") for i in range(4)]
    g = gate_b(b, k, recs)
    assert g.foil_catch["llm:a"] == pytest.approx(1.0)
    assert g.coverage["B1"] == pytest.approx(1.0)
    assert g.rates["wasted_ask_rate"].n == 3
