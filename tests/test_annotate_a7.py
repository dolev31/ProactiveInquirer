"""A7: the anticipation question, asked at the decision point where training happens.

A2 measures which candidate question is BETTER; its basis vocabulary (redundancy, form,
entity, specificity) contains no anticipation option, so the campaign could not say whether
anticipation ever separates real candidate pairs. A4 measured the mechanical `is_latent`
flag -- the thing that selects the trainset -- at 63.5% concordance with a careful read of
the decision point. A7 closes both holes with one instrument: for each of two same-state
candidates, does the question reach for something the TASK STATEMENT never names
(`reaches_unstated`) or not (`stays_stated`) -- a per-candidate judgment that doubles as a
decision-point latency relabel -- plus which candidate is the better ANTICIPATORY move, and
a basis vocabulary in which `anticipation` finally exists.

This file locks the design down:
  * per-candidate judgments and the derived preference are SEPARATE unit kinds, never
    pooled into one alpha (the A3_node/A5 rule);
  * units are canonical (chosen/rejected), derived through the key's shown-order coin flip;
    a record on an item whose key lacks `order` yields NO units (A2's rule);
  * `unstated_need_<side>` is required iff that side is judged `reaches_unstated` -- the
    named need is what makes a label auditable rather than a vibe (A5's `missing` rule);
  * nothing A7 produces may ever touch a gold field.
"""

from __future__ import annotations

import json
import random

import pytest

from pi_eval.annotate import (
    _LABELS,
    A7_BASIS_VALUES,
    TASK_TYPES,
    bundle_shape_errors,
    consensus,
    human_judgment_rows,
    merge_into_graphs,
    validate_records,
)
from pi_eval.annotate import _units as annotate_units
from pi_eval.annotate_llm import AnnotationParseError, build_prompt, parse_reply
from pi_run.cmd_annotate import sample_a7_items

BUNDLE_ID = "synth-0-deadbeef"


# --------------------------------------------------------------------------- fixtures


class _FakeCache:
    """Stands in for `SuiteCache`: only `.view(...).question` is read on this path."""

    def __init__(self, question: str = "who owned the paper that Smith founded?"):
        self._question = question

    def view(self, suite_id, task_id):
        from types import SimpleNamespace

        return SimpleNamespace(question=self._question)

    def units(self, suite_id, task_id):
        return {}


def _row(i: int, *, is_latent: bool = True, task_id: str = "t1") -> dict:
    return {
        "suite_id": "synth",
        "task_id": task_id,
        "run_id": "parent0",
        "turn_idx": 1,
        "state_text": "the rendered state the inquirer had in front of it",
        "chosen_json": json.dumps({"action": "ASK", "question": f"who founded variant {i}?"}),
        "rejected_json": json.dumps({"action": "ASK", "question": f"what year was it {i}?"}),
        "chosen_run_id": f"win{i}",
        "rejected_run_id": f"lose{i}",
        "margin": 0.1 + 0.01 * i,
        "is_latent": is_latent,
        "latent_depth": 1 if is_latent else 0,
        "pair_id": f"pair{i}",
    }


def _sample(rows, n, seed=0, **kw):
    stats: dict = {}
    items, key = sample_a7_items(
        _FakeCache(),
        rows,
        runs_root="/nonexistent",
        n=n,
        rng=random.Random(seed),
        stats=stats,
        **kw,
    )
    return items, key, stats


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


def _bundle(items):
    return {"manifest": {"bundle_id": BUNDLE_ID}, "items": items}


def _resp(**over):
    r = {
        "a_reaches": "stays_stated",
        "b_reaches": "reaches_unstated",
        "unstated_need_b": "the printing press, never named by the task",
        "preference": "b",
    }
    r.update(over)
    return r


def _rec(iid: str, ann: str = "llm:x", resp: dict | None = None) -> dict:
    return {
        "record_id": f"{BUNDLE_ID}/{iid}/{ann}",
        "bundle_id": BUNDLE_ID,
        "item_id": iid,
        "task_type": "A7",
        "annotator_id": ann,
        "annotator_kind": "llm",
        # An llm record must name its model: two models under one annotator_id would make
        # one agreement number out of two raters (test_an_llm_record_must_name_its_model).
        "model_pin": "m@t1.0",
        "response": resp if resp is not None else _resp(),
        "rationale": "b reaches for the press, which the task never names",
    }


# ------------------------------------------------------------------- registry & labels


def test_a7_is_registered_with_two_unpooled_vocabularies() -> None:
    assert "A7" in TASK_TYPES
    assert _LABELS["A7"] == ("reaches_unstated", "stays_stated", "cant_tell")
    assert _LABELS["A7_pref"] == ("chosen", "rejected", "tie", "both_bad")
    assert A7_BASIS_VALUES == ("anticipation", "hygiene", "no_difference")


# ------------------------------------------------------------------------ the sampler


def test_sampler_stratifies_by_the_latent_flag() -> None:
    """Both classes of the mechanical flag must be well represented: the relabel A7 exists
    to produce is only a correction if it is measured on both sides of the flag."""
    rows = [_row(i, is_latent=i < 6) for i in range(12)]
    items, key, _ = _sample(rows, n=6)
    lat = sum(1 for it in items if key[it["item_id"]]["is_latent"])
    assert lat == 3, f"expected a 3/3 split from a 6/6 pool, got {lat} latent of 6"


def test_shown_options_are_blind_and_the_key_holds_the_flip() -> None:
    rows = [_row(0)]
    items, key, _ = _sample(rows, n=1)
    it = items[0]
    k = key[it["item_id"]]
    assert set(it["payload"]) == {"option_a", "option_b"}
    assert set(it["payload"]["option_a"]) == {"question"}
    for secret in ("margin", "is_latent", "order", "chosen"):
        assert secret not in json.dumps(it), f"{secret} leaked into the shipped item"
    assert k["order"] in ("ab", "ba")
    assert {"chosen_run_id", "rejected_run_id", "margin", "pair_id", "is_latent"} <= set(k)
    shown_a = it["payload"]["option_a"]["question"]
    chosen_q = json.loads(rows[0]["chosen_json"])["question"]
    assert (shown_a == chosen_q) == (k["order"] == "ab")


def test_sampler_is_deterministic_under_seed() -> None:
    rows = [_row(i, is_latent=i % 2 == 0) for i in range(20)]
    a_items, a_key, _ = _sample(rows, n=8, seed=7)
    b_items, b_key, _ = _sample(rows, n=8, seed=7)
    assert [i["item_id"] for i in a_items] == [i["item_id"] for i in b_items]
    assert a_key == b_key


def test_provenance_is_the_state_not_a_candidate() -> None:
    items, _, _ = _sample([_row(0)], n=1)
    assert items[0]["provenance"]["run_id"] == "parent0"


# ------------------------------------------------------------------------- validation


def test_a_good_record_validates() -> None:
    errs = validate_records(_bundle([_a7_item()]), [_rec("i1")])
    assert errs == []


def test_unknown_reaches_and_preference_values_are_refused() -> None:
    bad1 = _rec("i1", resp=_resp(a_reaches="latent"))
    bad2 = _rec("i1", resp=_resp(preference="option_b"))
    for bad in (bad1, bad2):
        assert validate_records(_bundle([_a7_item()]), [bad]), "invalid label accepted"


def test_unstated_need_is_required_iff_reaches_unstated() -> None:
    """The named need is what makes the label auditable; symmetric with A5's `missing`."""
    missing = _rec(
        "i1", resp={"a_reaches": "stays_stated", "b_reaches": "reaches_unstated", "preference": "b"}
    )
    assert validate_records(_bundle([_a7_item()]), [missing]), (
        "reaches_unstated with no named need accepted"
    )
    spurious = _rec(
        "i1",
        resp=_resp(a_reaches="stays_stated", unstated_need_a="an invented need on a stated side"),
    )
    assert validate_records(_bundle([_a7_item()]), [spurious]), (
        "a need named for a stays_stated side accepted"
    )


def test_a7_basis_vocabulary_is_its_own() -> None:
    ok = dict(_rec("i1"), basis="anticipation")
    assert validate_records(_bundle([_a7_item()]), [ok]) == []
    wrong = dict(_rec("i1"), basis="non_redundancy")  # an A2 basis is not an A7 basis
    assert validate_records(_bundle([_a7_item()]), [wrong])


def test_bundle_shape_refuses_a_missing_option() -> None:
    it = _a7_item()
    del it["payload"]["option_b"]
    assert any("option_b" in e for e in bundle_shape_errors(_bundle([it])))


# ------------------------------------------------------------------------------ units


def test_units_are_canonical_through_the_key_order() -> None:
    """Shown slots mean nothing unblinded: labels must land on chosen/rejected."""
    key = {"items": {"i1": {"order": "ba", "chosen_run_id": "w", "rejected_run_id": "l"}}}
    units = annotate_units(_a7_item(), _rec("i1")["response"], key["items"]["i1"])
    by_id = {u[0]: (u[1], u[2]) for u in units}
    # order=ba: shown a IS the rejected side, shown b IS the chosen side
    assert by_id["i1/rejected"] == ("A7", "stays_stated")
    assert by_id["i1/chosen"] == ("A7", "reaches_unstated")
    assert by_id["i1/pref"] == ("A7_pref", "chosen")  # preference "b" = the chosen side


def test_no_key_order_means_no_units() -> None:
    assert annotate_units(_a7_item(), _rec("i1")["response"], {}) == []


def test_per_candidate_and_preference_are_separate_kinds() -> None:
    key_entry = {"order": "ab", "chosen_run_id": "w", "rejected_run_id": "l"}
    kinds = {u[1] for u in annotate_units(_a7_item(), _rec("i1")["response"], key_entry)}
    assert kinds == {"A7", "A7_pref"}


# ------------------------------------------------------------------- the wall to gold


def test_a7_never_writes_a_gold_field() -> None:
    """A7 is a preference between two runs, not a claim about the graph. The item is given
    a `gold_node_id` on purpose (the sampler puts none there): a wrong `elif` branch in
    `merge_into_graphs` would read it and mark n1 adjudicated, and this test would catch it.
    """
    from pi_eval.annotate import Consensus
    from pi_eval.gold import GoldGraph, GoldNode

    node = GoldNode(gold_suite="synth", gold_task_key="t1", gold_node_id="n1", gold_text="need n1")
    graph = GoldGraph(
        gold_suite="synth",
        gold_task_key="t1",
        gold_nodes=(node,),
        gold_edges=(),
        gold_answer="X PINQCANARY_0123456789ABCDEF",
        gold_canary="PINQCANARY_0123456789ABCDEF",
        gold_corpus_hash="cafe1234",
    )
    item = _a7_item("i1")
    item["provenance"]["gold_node_id"] = "n1"
    key = {"items": {"i1": {"order": "ab", "chosen_run_id": "w", "rejected_run_id": "l"}}}
    recs = [
        dict(_rec("i1", ann="human:x"), annotator_kind="human"),
        dict(_rec("i1", ann="human:y"), annotator_kind="human"),
    ]
    cons = consensus(_bundle([item]), recs, key=key)
    assert len(cons.by_kind("A7")) == 2, "the per-candidate consensus DID form"
    empty = Consensus(units=(), votes={}, kinds={}, provenance={})
    assert merge_into_graphs({"t1": graph}, cons, out_version="v1h") == merge_into_graphs(
        {"t1": graph}, empty, out_version="v1h"
    ), "an A7 consensus changed a merge that should be a no-op"


# ---------------------------------------------------------------------- the LLM path


def test_prompt_shows_both_options_and_the_task() -> None:
    p = build_prompt(_a7_item())
    assert "who founded the paper?" in p and "which press printed it?" in p
    assert "who owned the paper that Smith founded?" in p
    assert "margin" not in p.lower()


def test_parse_reply_roundtrips_and_requires_the_named_need() -> None:
    good = json.dumps(
        {
            "a_reaches": "stays_stated",
            "b_reaches": "reaches_unstated",
            "unstated_need_b": "the press",
            "preference": "b",
            "basis": "anticipation",
            "rationale": "b reaches for the press",
        }
    )
    parsed = parse_reply("A7", good)
    assert parsed.response["preference"] == "b"
    assert parsed.basis == "anticipation"
    bad = json.dumps(
        {
            "a_reaches": "stays_stated",
            "b_reaches": "reaches_unstated",
            "preference": "b",
            "rationale": "no need named",
        }
    )
    with pytest.raises(AnnotationParseError):
        parse_reply("A7", bad)


# --------------------------------------------------------------------- the judgments table


def test_a7_judgment_rows_pool_with_a2s_and_carry_the_basis() -> None:
    """One row per record with a real preference, run ids unblinded through the key's order
    exactly as A2's rows are; a tie names no winner and mints nothing."""
    key = {
        "items": {
            "i1": {"order": "ba", "chosen_run_id": "w", "rejected_run_id": "l", "pair_id": "p0"}
        }
    }
    rec = dict(_rec("i1", ann="human:x"), annotator_kind="human", basis="anticipation")
    rows = human_judgment_rows(_bundle([_a7_item()]), [rec], key=key)
    assert len(rows) == 1
    row = rows[0]
    # order "ba": shown A is the rejected side, and preference "b" names the chosen run.
    assert (row["run_id_a"], row["run_id_b"]) == ("l", "w")
    assert row["pref_sign"] == -1
    assert row["order"] == "ba"
    assert row["criterion"] == "human_preference"
    assert row["judge_family"] == "human"
    assert row["label"] == "anticipation"
    assert row["retest_group_id"] == "p0"

    tie = dict(rec, response=_resp(preference="tie"))
    assert human_judgment_rows(_bundle([_a7_item()]), [tie], key=key) == []


# ------------------------------------------------------------------------------- foils


def test_a7_foil_expected_answer_lives_only_in_the_key() -> None:
    """A planted foil's tell (one side IS the task question, verbatim, so that side's honest
    judgment is stays_stated) must be readable off the key and NEVER off the item -- the same
    hard constraint `sample_attention_items` lives under -- and a record on one must still
    import cleanly, or the plant would announce itself as a validation error."""
    from pi_run.cmd_annotate import sample_a7_foils

    cache = _FakeCache()
    rows = [_row(i) for i in range(4)]
    items, key = sample_a7_foils(cache, rows, n=2, rng=random.Random(0))
    assert items, "the fixture must actually produce a foil, or this proves nothing"
    for it in items:
        blob = json.dumps(it)
        assert "attention_check" not in blob
        assert "stays_stated" not in blob
        k = key[it["item_id"]]
        assert k["order"] in ("ab", "ba")
        expected = k["attention_check"]["expected"]
        (which,) = expected
        side = which.removesuffix("_reaches")
        assert expected[which] == "stays_stated"
        # The side the key pins really is the task question, shown verbatim.
        assert it["payload"][f"option_{side}"]["question"] == cache.view("synth", "t1").question
        assert "gold_node_id" not in it["provenance"]

    it = items[0]
    rec = {
        "record_id": f"{BUNDLE_ID}/{it['item_id']}/ann0",
        "bundle_id": BUNDLE_ID,
        "item_id": it["item_id"],
        "task_type": "A7",
        "annotator_id": "ann0",
        "response": {
            "a_reaches": "stays_stated",
            "b_reaches": "stays_stated",
            "preference": "tie",
        },
    }
    assert validate_records(_bundle([it]), [rec]) == []
    # The key entry carries an `order`, so units still form -- and `_collect` (not tested
    # here; see test_attention_checks_never_enter_a_measurement's pattern) drops them before
    # any measurement.
    assert len(annotate_units(it, rec["response"], key[it["item_id"]])) == 3


# ------------------------------------------------------------------------- CLI wiring


def test_export_wires_n_a7(tmp_path, monkeypatch):
    """`--n-a7` reaches `sample_a7_items`, its items land in the bundle and its unblinding
    side (order, run ids, margin, flag) lands in the key and nowhere else."""
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

    def _stub(cache, pairs_rows, runs_root=None, *, n, rng, stats=None, graph_version=""):
        captured["n"] = n
        it = _a7_item("stub_a7")
        return [it], {"stub_a7": {"order": "ab", "chosen_run_id": "w", "rejected_run_id": "l"}}

    monkeypatch.setattr(cmd_annotate, "sample_a7_items", _stub)

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
            "0",
            "--n-a7",
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
    assert any(it["task_type"] == "A7" for it in bundle["items"])
    assert bundle["manifest"]["counts"]["A7"] == 1
    assert "chosen_run_id" not in bundle_text
    assert "chosen_run_id" in key_text


# ------------------------------------------------------------------ the slot-bias gate


def test_review_flags_a_rater_that_reads_the_slot_not_the_content() -> None:
    """`order` is a per-item coin flip, so a rater reading CONTENT labels shown-slot A and
    shown-slot B at equal rates. gpt-5.6-terra labelled slot B `reaches_unstated` 466 times
    against slot A's 285 on a real 1,787-item pass (z = -6.60) while catching 99/99 foils
    and while claude-fable-5 sat at z = +0.13 on the identical prompt -- so this is a rater
    property, it is invisible in every canonical aggregate (the coin flip cancels it: chosen
    390 vs rejected 361, z = +1.06), and it cost $16.61 to discover by hand. Review computes
    it now, on the PER-CANDIDATE fields as well as the preference, because the per-candidate
    bias was the larger of the two.
    """
    from pi_run.cmd_annotate import _slot_bias

    items = [_a7_item(f"i{i}") for i in range(40)]
    # A rater who always calls the SECOND-shown option the reaching one.
    recs = [
        _rec(
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
    assert rep["reaches"]["n"] == 80
    assert rep["reaches"]["p_first"] == 0.0
    assert rep["reaches"]["binomial_p"] < 1e-6, "a 0/40 slot split must be flagged"
    assert rep["preference"]["p_first"] == 0.0

    balanced = [
        _rec(
            f"i{i}",
            resp=(
                {
                    "a_reaches": "reaches_unstated",
                    "unstated_need_a": "x",
                    "b_reaches": "stays_stated",
                    "preference": "a",
                }
                if i % 2
                else {
                    "a_reaches": "stays_stated",
                    "b_reaches": "reaches_unstated",
                    "unstated_need_b": "x",
                    "preference": "b",
                }
            ),
        )
        for i in range(40)
    ]
    ok = _slot_bias(_bundle(items), balanced, "A7")
    assert ok["reaches"]["binomial_p"] > 0.05, "a balanced rater must not be flagged"
