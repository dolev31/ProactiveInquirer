"""`_slot_bias` used to know only `{"A7": ...}` and return `{}` for every other task type.

All three call sites (`review_report`, `promote.check_quarantine`, `a7_artifacts.py`'s build
gate) pass `"A7"` literally, so an A6 (candidate-ranking) or A5 (single-item verdict) bundle
was never checked at all -- and `{}` is indistinguishable from "checked, clean", which is
exactly the failure mode `~/pi-corpus-backup/annotations-20260914`'s own README asserts away
("Slot bias does not apply: an A6 item has no two slots to read"). An A6 item has no A/B
slots, but it DOES print up to `A6_MAX_CANDIDATES` candidates in a fixed list order, and a
rater who reads position rather than content will over-place the first-LISTED one in its top
tier -- see `paper/sections/appendix_validity.tex`'s own measured pairwise analogue
(earlier-printed candidate won 4,627 of 8,579 separated pairs for the withdrawn rater,
p=3.3e-13) for a real instance of exactly this failure on this instrument.

This file locks down the corrected scope:
  * A6 gets a REAL statistic (top tier contains the first-listed candidate, against the null
    1/k for that item's own k), flagged at the same alpha A7 uses;
  * A5 (and A3_node, same reasoning: one item, one verdict, no second slot) get an explicit
    `{"applicable": False, ...}` rather than the silent `{}`;
  * A7's own branch is byte-identical to its pre-fix output -- pinned against the exact
    fixture in tests/test_annotate_a7.py::test_review_flags_a_rater_that_reads_the_slot_not_the_content.
"""

from __future__ import annotations

import pytest

from pi_run.cmd_annotate import _slot_bias

BUNDLE_ID = "synth-0-deadbeef"


# --------------------------------------------------------------------------- shared fixtures


def _bundle(items) -> dict:
    return {"manifest": {"bundle_id": BUNDLE_ID}, "items": list(items)}


# ------------------------------------------------------------------------------- A6 fixtures


def _a6_item(iid: str, *candidate_ids: str) -> dict:
    """`candidate_ids` in LIST (shown) order -- deliberately not required to be sorted, so a
    test can rotate which id is first-listed without changing the id spelling."""
    return {
        "item_id": iid,
        "task_type": "A6",
        "context": {},
        "payload": {
            "candidates": [{"candidate_id": c, "question": f"question {c}"} for c in candidate_ids]
        },
        "provenance": {
            "suite": "synth",
            "task_id": "t1",
            "task_key": "t1",
            "graph_version": "v1",
            "run_id": "parent0",
            "turn_idx": 1,
        },
    }


def _a6_rec(iid: str, ann: str, tiers: dict) -> dict:
    return {
        "record_id": f"{BUNDLE_ID}/{iid}/{ann}",
        "bundle_id": BUNDLE_ID,
        "item_id": iid,
        "task_type": "A6",
        "annotator_id": ann,
        "response": {"tiers": tiers},
    }


# ------------------------------------------------------------------------------- A7 fixtures
# Copied verbatim from tests/test_annotate_a7.py's `_a7_item`/`_rec` and the biased-rater
# fixture in `test_review_flags_a_rater_that_reads_the_slot_not_the_content`, so test (c)
# below pins the SAME inputs that test already exercises, not a lookalike.


def _a7_item(iid: str = "i1") -> dict:
    return {
        "item_id": iid,
        "task_type": "A7",
        "context": {
            "question": "who owned the paper that Smith founded?",
            "evidence": [],
            "history": [],
            "draft": "",
            "state_text": "S",
        },
        "payload": {
            "option_a": {"question": "who founded the paper?"},
            "option_b": {"question": "which press printed it?"},
        },
        "provenance": {
            "suite": "synth",
            "task_id": "t1",
            "task_key": "t1",
            "graph_version": "v1",
            "run_id": "parent0",
            "turn_idx": 1,
        },
    }


def _a7_rec(iid: str, ann: str = "llm:x", resp: dict | None = None) -> dict:
    return {
        "record_id": f"{BUNDLE_ID}/{iid}/{ann}",
        "bundle_id": BUNDLE_ID,
        "item_id": iid,
        "task_type": "A7",
        "annotator_id": ann,
        "annotator_kind": "llm",
        "model_pin": "m@t1.0",
        "response": resp,
        "rationale": "b reaches for the press, which the task never names",
    }


# ------------------------------------------------------------- (a) A6 gets a real statistic


def test_a6_slot_bias_flags_a_rater_that_always_tops_the_first_listed_candidate() -> None:
    """40 four-candidate items. `cA..cD` rotate through the first-listed slot (10 times each)
    so the statistic cannot pass by reading the label `c0`/`cA` instead of list position. The
    synthetic rater always puts whichever candidate is LISTED FIRST alone in tier 1, UNTIED
    (nobody else shares tier 1): that is 40/40 hits, and with an untied top tier the tie-aware
    null (the default, `expected`) and the naive 1/k null (`p_naive`) coincide at 0.25, so both
    readings must clear the same abort threshold A7 uses (`promote.SLOT_BIAS_ALPHA` /
    `a7_artifacts.ALPHA`, both 0.001). See the next test for a rater ties WOULD have separated.
    """
    ids_pool = ["cA", "cB", "cC", "cD"]
    items = []
    recs = []
    for i in range(40):
        rotated = ids_pool[i % 4 :] + ids_pool[: i % 4]
        items.append(_a6_item(f"i{i}", *rotated))
        first, rest = rotated[0], rotated[1:]
        tiers = {first: 1, **{c: 2 for c in rest}}
        recs.append(_a6_rec(f"i{i}", "llm:always_first", tiers))

    rep = _slot_bias(_bundle(items), recs, "A6")
    assert rep, "must not be the silent {} every non-A7 type used to return"
    stat = rep["top_tier"]
    assert stat["n"] == 40
    assert stat["n_first"] == 40
    assert stat["p_first"] == 1.0
    assert stat["expected"] == pytest.approx(0.25), "untied: tie-aware null equals 1/4 here too"
    assert stat["binomial_p"] < 0.001, "same abort threshold A7 uses, tie-aware (default) null"
    assert stat["p_naive"] < 0.001, "the naive 1/k null agrees when nobody ties the top"


def test_a6_slot_bias_does_not_flag_a_rater_who_reads_content() -> None:
    """Same rotation, but the rater ranks by the candidate_id itself (content-like, position-
    independent): cA always tier 1, cB tier 2, cC tier 3, cD tier 4, regardless of where each
    was LISTED. Since the first-listed slot cycles evenly through all four ids, this rater's
    'first-listed lands in top tier' rate is exactly 1/4 -- the null itself, tie-aware or
    naive, since every response is untied -- and must not be flagged either way.
    """
    ids_pool = ["cA", "cB", "cC", "cD"]
    rank = {"cA": 1, "cB": 2, "cC": 3, "cD": 4}
    items = []
    recs = []
    for i in range(40):
        rotated = ids_pool[i % 4 :] + ids_pool[: i % 4]
        items.append(_a6_item(f"i{i}", *rotated))
        tiers = {c: rank[c] for c in rotated}
        recs.append(_a6_rec(f"i{i}", "llm:reads_content", tiers))

    rep = _slot_bias(_bundle(items), recs, "A6")
    stat = rep["top_tier"]
    assert stat["n"] == 40
    assert stat["p_first"] == pytest.approx(0.25)
    assert stat["binomial_p"] > 0.05, "a rater at exactly the null must not be flagged"
    assert stat["p_naive"] > 0.05, "untied responses: the naive null agrees"


def test_a6_slot_bias_a_rater_exactly_at_the_tie_aware_null_does_not_flag() -> None:
    """The tie-aware null is the DEFAULT (`expected`/`binomial_p`): given a rater's own
    top-tier size t on an item, P(first-listed in the top tier) is t/k under no position
    effect, not 1/k. This rater ties TWO of four candidates at the top on every item (t=2,
    k=4, tie-aware null 0.5) by a purely CONTENT-based rule -- always `cA` and `cB`, regardless
    of where either was LISTED. As the first-listed slot rotates evenly through all four ids,
    this rater's observed rate lands exactly on 0.5, the tie-aware null, not on the naive
    null's 0.25. Under the default this must clear alpha; `p_naive` shows the naive 1/k null
    would have wrongly flagged the same rater, which is the whole reason the default changed.
    """
    ids_pool = ["cA", "cB", "cC", "cD"]
    top_tier_ids = {"cA", "cB"}
    items = []
    recs = []
    for i in range(40):
        rotated = ids_pool[i % 4 :] + ids_pool[: i % 4]
        items.append(_a6_item(f"i{i}", *rotated))
        tiers = {c: (1 if c in top_tier_ids else 2) for c in rotated}
        recs.append(_a6_rec(f"i{i}", "llm:ties_two_by_content", tiers))

    rep = _slot_bias(_bundle(items), recs, "A6")
    stat = rep["top_tier"]
    assert stat["n"] == 40
    assert stat["p_first"] == pytest.approx(0.5)
    assert stat["expected"] == pytest.approx(0.5), "tie-aware null: mean top-tier-size/k"
    assert stat["binomial_p"] > 0.05, "sitting exactly at the tie-aware (default) null: no flag"
    assert stat["p_naive"] < 0.001, "the naive 1/k null would have wrongly flagged this rater"


def test_a6_slot_bias_a_tie_for_top_counts_as_containing_the_first_listed() -> None:
    """Real A6 records tie at the top tier in the MAJORITY of responses (measured 2026-09-18
    on ~/pi-corpus-backup/annotations-20260914: 58.8% of 4,363 records, mean top-tier size
    2.05 of 3-5 shown) -- this is not an edge case, it is the common one, so 'top tier
    contains X' must mean membership in the tied-lowest set, not 'X is the unique top'. Two
    items: one where the first-listed candidate SHARES tier 1 with another (must count as a
    hit), one where another candidate has tier 1 ALONE and the first-listed sits at tier 2
    despite a tie existing elsewhere in that same response (must not count).
    """
    tied_with_first = _a6_item("i0", "c0", "c1", "c2")
    tied_without_first = _a6_item("i1", "c0", "c1", "c2")
    recs = [
        # c0 (first-listed) ties for tier 1 with c1; c2 is alone at tier 2.
        _a6_rec("i0", "llm:x", {"c0": 1, "c1": 1, "c2": 2}),
        # c1 and c2 tie for tier 1; c0 (first-listed) is alone at tier 2 -- not in the top.
        _a6_rec("i1", "llm:x", {"c0": 2, "c1": 1, "c2": 1}),
    ]
    rep = _slot_bias(_bundle([tied_with_first, tied_without_first]), recs, "A6")
    stat = rep["top_tier"]
    assert stat["n"] == 2
    assert stat["n_first"] == 1, "only i0's tie includes the first-listed candidate"


# ---------------------------------------------------- (b) A5 (and A3_node) are not silent {}


def test_a5_slot_bias_is_marked_not_applicable_not_silently_empty() -> None:
    """A5 shows one item and asks a yes/no verdict -- there is no second slot to read a
    position effect against. Before this fix, `_slot_bias` returned `{}` for A5 exactly as it
    did for A6, which is indistinguishable from 'nobody wired this up yet'. `{}` and
    'confirmed not applicable' must not look the same.
    """
    rep = _slot_bias({"items": []}, [], "A5")
    assert rep == {"applicable": False, "reason": "single-item instrument"}


def test_a3_node_slot_bias_is_also_marked_not_applicable() -> None:
    """Same reasoning as A5, not one of the three required sub-tests in the lane brief but the
    same fix: A3_node is the other single-item, single-verdict instrument the lane's own
    narrative names, and leaving it on the old `{}` path would keep the exact silent-empty
    defect for one more instrument while claiming the class was fixed.
    """
    rep = _slot_bias({"items": []}, [], "A3_node")
    assert rep == {"applicable": False, "reason": "single-item instrument"}


def test_unknown_task_type_still_returns_the_old_empty_dict() -> None:
    """Anything not named A7/A6/A5/A3_node keeps the pre-fix fallback -- this function does
    not silently invent a new contract for an instrument nobody asked it to cover."""
    rep = _slot_bias({"items": []}, [], "A1")
    assert rep == {}


# --------------------------------------------------- (c) A7's branch is byte-for-byte intact


def test_a7_slot_bias_output_is_byte_identical_to_its_pre_fix_pin() -> None:
    """Pins the EXACT dict `_slot_bias` returned, before this change, for the biased-rater
    fixture in tests/test_annotate_a7.py::test_review_flags_a_rater_that_reads_the_slot_not_the_content
    (40 items, rater always calls slot b `reaches_unstated`). Captured by running that
    fixture through the pre-fix `_slot_bias` directly:

        $ .venv/bin/python - <<'PY'
        # (see artifacts/slot_bias_scope_20260918/RESULT.md for the exact script and output)
        PY

    which printed:
        {"preference": {"binomial_p": 1.8189894035458565e-12, "n": 40, "n_first": 0,
                         "p_first": 0.0},
         "reaches":    {"binomial_p": 1.6543612251060553e-24, "n": 80, "n_first": 0,
                         "p_first": 0.0}}

    Adding the A6/A5/A3_node branches above must not move a single float in this one.
    """
    items = [_a7_item(f"i{i}") for i in range(40)]
    recs = [
        _a7_rec(
            f"i{i}",
            resp={
                "a_reaches": "stays_stated",
                "b_reaches": "reaches_unstated",
                "unstated_need_b": "whatever is in slot b",
                "preference": "b",
            },
        )
        for i in range(40)
    ]
    rep = _slot_bias(_bundle(items), recs, "A7")
    assert rep == {
        "preference": {
            "binomial_p": 1.8189894035458565e-12,
            "n": 40,
            "n_first": 0,
            "p_first": 0.0,
        },
        "reaches": {
            "binomial_p": 1.6543612251060553e-24,
            "n": 80,
            "n_first": 0,
            "p_first": 0.0,
        },
    }
