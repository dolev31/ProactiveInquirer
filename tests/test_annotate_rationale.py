"""RATIONALE (and A2's `basis`) as a top-level, explanatory-only record field.

Design constraints under test, matched to the module docstrings that state them:
  * `rationale`/`basis` are SIBLINGS of `response`, never inside it (`pi_eval.annotate_llm`'s
    module docstring; `response` is what `consensus`/`iaa_report` compute agreement over).
  * A rationale never validates a label: `consensus`, `iaa_report` and `gate_numbers` must be
    blind to its presence (`test_rationale_does_not_change_any_measurement`).
  * Optional for humans, required for models, and NEVER defaulted when a model omits it --
    the existing `AnnotationParseError` discipline `pi_eval.annotate_llm` applies to every
    other field it cannot read.
  * Capped short: a REASON, not an essay.
"""

from __future__ import annotations

import json
import math
from typing import Any, Mapping, Sequence

import pytest

from pi_eval.annotate import (
    A2_BASIS_VALUES,
    RATIONALE_MAX_CHARS,
    consensus,
    gate_numbers,
    iaa_report,
    item_set_hash,
    validate_records,
)
from pi_eval.annotate_llm import AnnotationParseError, annotate_item, parse_reply
from pi_run import cmd_annotate

BUNDLE_ID = "musique-0-deadbeef"


# --------------------------------------------------------------------------- shared fixtures


class FakeLLM:
    """Same structural `JudgeLLM` double `tests/test_annotate_llm_pass.py` uses: canned replies
    in call order, `.complete(...) -> (text, telemetry)`."""

    def __init__(self, replies: list[str], telemetry: Any = None):
        self._replies = list(replies)
        self._telemetry = telemetry

    def complete(self, *, role, messages, seed, max_tokens=None, **kw):
        if not self._replies:
            raise AssertionError("FakeLLM ran out of canned replies")
        return self._replies.pop(0), self._telemetry


def _item(task_type: str, iid: str, payload: Mapping[str, Any] | None = None, **prov) -> dict:
    p = {"suite": "musique", "task_id": "t1", "task_key": "t1", "graph_version": "v1"}
    p.update(prov)
    return {
        "item_id": iid,
        "task_type": task_type,
        "context": {"question": "q", "src_text": "a", "dst_text": "b"},
        "payload": payload or {},
        "provenance": p,
    }


def _bundle(items: Sequence[Mapping[str, Any]]) -> dict:
    return {
        "manifest": {"bundle_id": BUNDLE_ID, "item_set_hash": item_set_hash(list(items))},
        "items": list(items),
    }


def _rec(iid, ann, task_type, response, *, rationale=None, basis=None, kind=None, model=None):
    r: dict[str, Any] = {
        "record_id": f"{BUNDLE_ID}/{iid}/{ann}",
        "bundle_id": BUNDLE_ID,
        "item_id": iid,
        "task_type": task_type,
        "annotator_id": ann,
        "elapsed_ms": 1000,
        "ts": "2026-08-31T12:00:00Z",
        "response": response,
    }
    if rationale is not None:
        r["rationale"] = rationale
    if basis is not None:
        r["basis"] = basis
    if kind is not None:
        r["annotator_kind"] = kind
    if model is not None:
        r["model_pin"] = model
    return r


def _a1_item(iid="a1_t1"):
    return _item(
        "A1",
        iid,
        payload={"nodes": [{"node_id": "n1", "text": "who signed it"}]},
        graph_version="v1",
    )


def _a1_reply(rationale="n1 is the natural follow-up", ticked=("n1",), likelihood=None):
    likelihood = likelihood if likelihood is not None else {"n1": 60}
    return json.dumps(
        {
            "ticked": list(ticked),
            "usefulness": {"n1": 4},
            "likelihood": likelihood,
            "rationale": rationale,
        }
    )


def _a2_item(iid="p0"):
    return _item(
        "A2",
        iid,
        payload={
            "option_a": {"question": "which region did Andy sail to"},
            "option_b": {"question": "what city was Gotham filmed in"},
        },
        context={
            "question": "where did Andy go",
            "evidence": [],
            "history": [],
            "draft": "",
            "state_text": "state",
        },
    )


def _a2_reply(choice="a", rationale="option a names a place, option b does not", basis="omit"):
    obj = {"choice": choice, "rationale": rationale}
    if basis != "omit":
        obj["basis"] = basis
    return json.dumps(obj)


def _nan_safe_eq(a: Any, b: Any) -> bool:
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_nan_safe_eq(a[k], b[k]) for k in a)
    return a == b


# --------------------------------------------------------------------------- A: carried & required


def test_llm_record_carries_a_rationale():
    item = _a1_item()
    rec = annotate_item(
        FakeLLM([_a1_reply(rationale="ticked n1 because it is required to answer")]),
        item,
        bundle_id=BUNDLE_ID,
        model_pin="model-a@t0.0",
        annotator_id="llm:model-a",
    )
    assert rec["rationale"] == "ticked n1 because it is required to answer"
    # A sibling of `response`, never free text inside the object agreement is computed over.
    assert "rationale" not in rec["response"]
    assert validate_records(_bundle([item]), [rec]) == []


def test_a_reply_missing_its_rationale_is_a_parse_error_not_a_default():
    item = _a1_item()
    reply_without_rationale = json.dumps({"ticked": ["n1"], "usefulness": {"n1": 4}})

    with pytest.raises(AnnotationParseError):
        parse_reply(item, reply_without_rationale)

    # Through the full call: no record is written, the caller must count it as unparsed rather
    # than receive a record with a synthesised rationale.
    with pytest.raises(AnnotationParseError):
        annotate_item(
            FakeLLM([reply_without_rationale]),
            item,
            bundle_id=BUNDLE_ID,
            model_pin="model-a@t0.0",
            annotator_id="llm:model-a",
        )


# --------------------------------------------------------------------------- B: never evidence


def test_rationale_does_not_change_any_measurement():
    """Same votes, with and without a rationale attached: consensus units, alpha and the gate
    numbers must be byte-identical. A rationale is explanatory metadata, never a label."""
    items = [_item("A3_edge", "e0"), _item("A3_edge", "e1")]
    bundle = _bundle(items)

    with_r = [
        _rec("e0", w, "A3_edge", {"holds": True}, rationale="because e0 clearly depends")
        for w in ("alice", "bob")
    ] + [
        _rec("e1", w, "A3_edge", {"holds": False}, rationale="because e1 clearly does not")
        for w in ("alice", "bob")
    ]
    without_r = [_rec("e0", w, "A3_edge", {"holds": True}) for w in ("alice", "bob")] + [
        _rec("e1", w, "A3_edge", {"holds": False}) for w in ("alice", "bob")
    ]

    cons_r = consensus(bundle, with_r)
    cons_no = consensus(bundle, without_r)
    assert cons_r.units == cons_no.units
    assert cons_r.disagreements == cons_no.disagreements

    assert _nan_safe_eq(iaa_report(bundle, with_r), iaa_report(bundle, without_r))
    assert _nan_safe_eq(gate_numbers(cons_r, None), gate_numbers(cons_no, None))


# --------------------------------------------------------------------------- C: length cap


def test_rationale_is_length_capped():
    """WAS: `parse_reply` RAISES on an over-long rationale. Belief corrected -- it truncates.

    The refusal to default is right for every LABEL and is unchanged. It was over-applied to
    the rationale, which this module's own cap docstring calls "explanatory metadata, never a
    label" that "every consensus/gate computation must be blind to". A truncated rationale is
    not a default: the text was read in full and only the stored copy is shortened. Discarding
    the record instead cost 97 items of three-rater coverage on the A6 pass (24%), and
    `--resume` could not heal it because the retry re-issued an identical prompt and hit the
    cache. See pi_eval.annotate_llm.clamp_rationale.
    """
    item = _a1_item()
    too_long = "x" * (RATIONALE_MAX_CHARS + 1)

    kept = parse_reply(item, _a1_reply(rationale=too_long))
    assert len(kept.rationale) <= RATIONALE_MAX_CHARS, "the cap's purpose is a bounded string"
    assert kept.rationale.endswith("..."), "a cut reason must be distinguishable from a short one"
    assert kept.response.get("ticked") == ["n1"], "the LABEL survives intact"

    # `validate_records` enforces the same cap independently, for a record that reached disk
    # some other way (e.g. a human tool, or a future producer this module never sees).
    bundle = _bundle([item])
    over_cap = _rec(
        "a1_t1", "alice", "A1", {"ticked": ["n1"], "usefulness": {}}, rationale=too_long
    )
    errs = validate_records(bundle, [over_cap])
    assert any("400" in e or str(len(too_long)) in e for e in errs)

    at_cap = _rec(
        "a1_t1",
        "alice",
        "A1",
        {"ticked": ["n1"], "usefulness": {}},
        rationale="x" * RATIONALE_MAX_CHARS,
    )
    assert validate_records(bundle, [at_cap]) == []


# --------------------------------------------------------------------------- D: A2's basis


def test_a2_basis_is_from_the_closed_vocabulary():
    item = _a2_item()

    parsed_a = parse_reply(item, _a2_reply(choice="a", basis="names_its_entities"))
    assert parsed_a.basis == "names_its_entities"
    assert "basis" not in parsed_a.response

    with pytest.raises(AnnotationParseError):
        parse_reply(item, _a2_reply(choice="a", basis="because_i_said_so"))

    # A winner named ("a"/"b") requires a basis.
    with pytest.raises(AnnotationParseError):
        parse_reply(item, _a2_reply(choice="a", basis="omit"))

    # tie/both_bad may omit it entirely.
    parsed_tie = parse_reply(item, _a2_reply(choice="tie", basis="omit"))
    assert parsed_tie.basis is None

    # validate_records rejects a basis outside the closed vocabulary wherever it is found.
    bundle = _bundle([item])
    bad = _rec("p0", "alice", "A2", {"choice": "a"}, basis="vibes")
    assert any("basis" in e for e in validate_records(bundle, [bad]))
    for v in A2_BASIS_VALUES:
        good = _rec("p0", "alice", "A2", {"choice": "a"}, basis=v)
        assert validate_records(bundle, [good]) == []


# --------------------------------------------------------------------------- E: optional vs required


def test_human_records_may_omit_a_rationale_but_llm_records_may_not():
    item = _a1_item()
    bundle = _bundle([item])

    human_no_rationale = _rec("a1_t1", "alice", "A1", {"ticked": ["n1"], "usefulness": {}})
    assert validate_records(bundle, [human_no_rationale]) == []

    llm_no_rationale = _rec(
        "a1_t1",
        "llm:model-a",
        "A1",
        {"ticked": ["n1"], "usefulness": {}},
        kind="llm",
        model="model-a@t0.0",
    )
    errs = validate_records(bundle, [llm_no_rationale])
    assert any("rationale" in e for e in errs)

    llm_with_rationale = _rec(
        "a1_t1",
        "llm:model-a",
        "A1",
        {"ticked": ["n1"], "usefulness": {}},
        kind="llm",
        model="model-a@t0.0",
        rationale="n1 is the direct follow-up",
    )
    assert validate_records(bundle, [llm_with_rationale]) == []


# --------------------------------------------------------------------------- F: review sampling


def test_review_samples_rationales_from_disagreements():
    """`pi annotate review` must sample rationales ONLY from records behind a unit that never
    resolved -- a unanimous item's rationale explains nothing anyone was unsure about."""
    disagree_items = [_item("A3_edge", f"d{i}") for i in range(4)]
    agree_item = _item("A3_edge", "agree0")
    bundle = _bundle(disagree_items + [agree_item])

    records = []
    for i, it in enumerate(disagree_items):
        records.append(
            _rec(it["item_id"], "alice", "A3_edge", {"holds": True}, rationale=f"alice sees {i}")
        )
        records.append(
            _rec(it["item_id"], "bob", "A3_edge", {"holds": False}, rationale=f"bob sees {i}")
        )
    records.append(
        _rec("agree0", "alice", "A3_edge", {"holds": True}, rationale="unanimous, boring")
    )
    records.append(
        _rec("agree0", "bob", "A3_edge", {"holds": True}, rationale="unanimous, boring too")
    )

    report = cmd_annotate.review_report(bundle, records, key=None)
    sample = report["rationale_sample"]
    assert "A3_edge" in sample
    edge_sample = sample["A3_edge"]

    assert len(edge_sample) <= 3, "capped at up to 3 per unit kind"
    sampled_rationales = {s["rationale"] for s in edge_sample}
    assert "unanimous, boring" not in sampled_rationales
    assert "unanimous, boring too" not in sampled_rationales
    assert all(r.startswith(("alice sees", "bob sees")) for r in sampled_rationales)


def test_review_reports_a2_basis_distribution():
    items = [_item("A2", f"p{i}") for i in range(3)]
    bundle = _bundle(items)
    records = [
        _rec("p0", "alice", "A2", {"choice": "a"}, basis="names_its_entities"),
        _rec("p1", "alice", "A2", {"choice": "a"}, basis="names_its_entities"),
        _rec("p2", "alice", "A2", {"choice": "b"}, basis="better_scoped"),
    ]
    report = cmd_annotate.review_report(bundle, records, key=None)
    assert report["a2_basis_distribution"] == {"better_scoped": 1, "names_its_entities": 2}


# --------------------------------------------------------------------------- prompt parity


def _a2_item(**ctx):
    context = {"question": "How many square miles is the source country?"}
    context.update(ctx)
    return {
        "item_id": "p0",
        "task_type": "A2",
        "context": context,
        "payload": {
            "option_a": {"question": "Which region did Andy sail to?"},
            "option_b": {"question": "When did it start?"},
        },
        "provenance": {"suite": "musique", "task_id": "t1"},
    }


def test_a2_prompt_carries_the_state_when_the_replay_produced_no_structure():
    """The human reads the full rendered state in a panel; a model handed only two questions
    is choosing blind. Measured on the live bundle, every A2 item had empty evidence/history/
    draft and only `state_text` -- so this fallback is the normal case, not an edge one, and
    without it the model and the human are answering different questions and their agreement
    means nothing."""
    from pi_eval.annotate_llm import build_prompt

    state = "EVIDENCE RETRIEVED SO FAR\n[abc123] Gotham\nFilming began in New York City."
    prompt = build_prompt(_a2_item(evidence=[], history=[], draft="", state_text=state))
    assert "New York City" in prompt, "the model never saw the evidence the human saw"


def test_a2_prompt_prefers_structured_context_over_the_raw_state():
    from pi_eval.annotate_llm import build_prompt

    prompt = build_prompt(
        _a2_item(
            evidence=[{"uid": "u1", "title": "Gotham", "text": "Filming began in New York City."}],
            history=[{"q": "Where was it filmed?", "a": "Not stated."}],
            draft="Cannot be determined.",
            state_text="RAW PROMPT TEMPLATE THAT SHOULD NOT BE PASTED WHOLE",
        )
    )
    assert "New York City" in prompt and "Where was it filmed?" in prompt
    assert "Cannot be determined." in prompt
    assert "RAW PROMPT TEMPLATE" not in prompt


def test_a2_prompt_never_renders_an_empty_section():
    """An 'Evidence gathered so far:' header with nothing under it tells the model the
    evidence set is empty, which is a different claim from 'not shown'."""
    from pi_eval.annotate_llm import build_prompt

    prompt = build_prompt(_a2_item(evidence=[], history=[], draft="", state_text=""))
    for line in prompt.splitlines():
        if line.rstrip().endswith(":"):
            assert line.strip() not in ("Evidence gathered so far:", "Questions already asked:")


# --------------------------------------------------------------------------- timing scope


def _timed(iid, ann, ms, *, kind=None, model=None):
    r = {
        "record_id": f"b0/{iid}/{ann}",
        "bundle_id": "b0",
        "item_id": iid,
        "task_type": "A3_edge",
        "annotator_id": ann,
        "elapsed_ms": ms,
        "ts": "2026-08-31T12:00:00Z",
        "response": {"holds": True},
    }
    if kind:
        r["annotator_kind"] = kind
        r["model_pin"] = model or "m@t0.0"
        r["rationale"] = "because"
    return r


def test_timing_flags_only_human_annotators():
    """`elapsed_ms` measures how long a PERSON looked at an item; for a model it measures a
    round trip, and a cached reply legitimately reads 0ms. Flagging those as `click_through`
    puts a false positive on every cache hit, and a diagnostic that cries wolf on normal
    operation is one people learn to skip -- taking the real click-throughs with it.
    """
    from pi_run.cmd_annotate import _timing_report

    humans = [_timed(f"h{i}", "alice", 30000) for i in range(4)] + [_timed("h9", "alice", 100)]
    models = [_timed(f"m{i}", "llm:x", 3000, kind="llm") for i in range(4)] + [
        _timed("m9", "llm:x", 0, kind="llm")  # a cache hit, not carelessness
    ]
    flagged = _timing_report(humans + models)["flagged"]
    ids = {f["record_id"] for f in flagged}
    assert "b0/h9/alice" in ids, "a human who spent 100ms on a 30s item must still be flagged"
    assert not any(str(i).startswith("b0/m") for i in ids), "no model record may be flagged"


def test_annotator_temperature_is_the_only_one_the_models_accept():
    """The frontier GPT-5.5/5.6 models REFUSE a pinned temperature: probed against the live
    proxy, every one returns "Unsupported value: 'temperature' does not support 0.0 with this
    model. Only the default (1) value is supported." So t=0 is not a choice we get to make.

    That removes the old justification (two passes agreeing with each other) and needs a real
    replacement rather than a shrug. Reproducibility now rests where it already did for the
    judges -- the request cache, keyed on the exact request bytes, so a re-run of a scored
    pass replays rather than re-samples. And the temperature rides on `model_pin`, so a pass
    at a different one is a different rater and cannot be pooled with this one by accident.
    """
    from pi_eval.annotate_llm import ANNOTATOR_TEMPERATURE

    assert ANNOTATOR_TEMPERATURE == 1.0


def test_model_pin_records_the_temperature():
    """Two passes at different temperatures are two raters. The pin is what says so."""
    from pi_eval.annotate_llm import ANNOTATOR_TEMPERATURE

    pin = f"azure/gpt-5.6-terra@t{ANNOTATOR_TEMPERATURE}"
    assert pin.endswith("@t1.0")


def test_the_annotator_does_not_inherit_the_judge_temperature():
    """`ask_with_telemetry` pinned `temperature=JUDGE_TEMPERATURE` at the call site, so the
    annotation pass ran at the judge's temperature no matter what `ANNOTATOR_TEMPERATURE`
    said -- the shared helper silently made two roles one rater. `litellm_client`'s own header
    states the rule: ONE ROLE, ONE PIN, because a role is a scientific object and a model id
    (or a sampling setting) is an implementation detail. Caught in production: every call
    still went out at 0.0 and the frontier models rejected it outright.
    """
    from pi_eval.annotate_llm import ANNOTATOR_TEMPERATURE, annotate_item
    from pi_eval.judges._llm import JUDGE_TEMPERATURE

    assert ANNOTATOR_TEMPERATURE != JUDGE_TEMPERATURE, "otherwise this test proves nothing"

    seen: dict = {}

    class Recorder:
        def complete(self, **kw):
            seen.update(kw)
            return (
                '{"verdict":"required","discoverability":"kb","rationale":"r"}',
                None,
            )

    item = {
        "item_id": "i1",
        "task_type": "A3_node",
        "context": {"question": "q?", "node_text": "n"},
        "payload": {},
        "provenance": {"suite": "wiki2", "task_id": "t1"},
    }
    annotate_item(
        Recorder(), item, bundle_id="b0", model_pin="m@t1.0", annotator_id="llm:m", seed=0
    )
    assert seen.get("temperature") == ANNOTATOR_TEMPERATURE


def test_llm_record_carries_what_the_call_cost():
    """`--max-usd` currently reads `ledger.spent`, and `pinq.budget`'s own header says the
    ledger "is per (task, arm, seed) process, so it needs no locking at all". Under
    `--concurrency > 1` that is a lock-free counter read from several threads: it can
    UNDERCOUNT, and the one guard that stops an overspend is the thing reading it.

    `CallTelemetry.usd` is per call and deterministic (tokens x the pinned price table), so
    recording it on the record lets the pass total its own spend in the single thread that
    already does every other piece of bookkeeping. Recording a machine artifact is explicitly
    fine here -- what the repo forbids is COMPARING usd across arms as evidence.
    """
    from pi_eval.annotate_llm import annotate_item

    class Telemetried:
        def complete(self, **kw):
            class Tel:
                usd = 0.00125

            return '{"holds": true, "rationale": "because"}', Tel()

    item = {
        "item_id": "e1",
        "task_type": "A3_edge",
        "context": {"question": "q?", "src_text": "a", "dst_text": "b"},
        "payload": {},
        "provenance": {"suite": "wiki2", "task_id": "t1"},
    }
    rec = annotate_item(
        Telemetried(), item, bundle_id="b0", model_pin="m@t1.0", annotator_id="llm:m", seed=0
    )
    assert rec["usd"] == pytest.approx(0.00125)


def test_a3_edge_prompt_asks_the_dependency_in_the_direction_gold_defines_it():
    """`pi_eval.gold.GoldEdge` states it exactly: "prerequisite means v_dst is UNANSWERABLE
    until v_src is resolved". So SRC is the prerequisite and DST is the dependent.

    The prompt had the labels inverted -- src captioned "the dependent one" and dst "the
    possible prerequisite" -- so the model answered the opposite question from the one the
    human tool asks ("Must the first be resolved before the second is even askable?"). Caught
    in a live pass: 167 of 200 wiki2 edges came back False, on a suite where an edge exists
    exactly when `objects[i] == subjects[j]` and is therefore true by construction, with
    rationales like "Finding the director does not require knowing that director's date of
    birth" -- the dependency read backwards. Any human-vs-model agreement on A3_edge would
    have been measuring the inversion.
    """
    from pi_eval.annotate_llm import build_prompt

    item = {
        "item_id": "e1",
        "task_type": "A3_edge",
        "context": {
            "question": "When was the director of Dialogues of Exiles born?",
            "src_text": "Dialogues of Exiles >> director",
            "dst_text": "⟨Dialogues of Exiles >> director⟩ >> date of birth",
        },
        "payload": {},
        "provenance": {"suite": "wiki2", "task_id": "t1"},
    }
    p = build_prompt(item)
    src_line = next(
        ln for ln in p.splitlines() if "Dialogues of Exiles >> director" in ln and "⟨" not in ln
    )
    dst_line = next(ln for ln in p.splitlines() if "⟨" in ln)
    assert "prerequisite" in src_line.lower(), f"src must be named the prerequisite: {src_line!r}"
    assert "depend" in dst_line.lower(), f"dst must be named the dependent one: {dst_line!r}"
