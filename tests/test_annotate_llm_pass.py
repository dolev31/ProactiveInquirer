"""The automatic LLM annotation pass: `pi_eval.annotate_llm` (prompts, parsing, one record)
and `pi_run.cmd_annotate`'s `pi annotate llm` (the loop: resume-skip, spend cap, the CLI shell).

No network anywhere in this file. `FakeLLM` is the only `JudgeLLM` these tests ever construct,
matching the pattern `tests/test_annotate_llm.py` already locks down for the records
themselves: a model may annotate, it may never become the human annotation, and nothing here
tests otherwise -- that wall is `pi_eval.annotate`'s and is exercised by its own test file,
untouched.
"""

from __future__ import annotations

import json
import os
import time
from types import SimpleNamespace

import pytest

from pi_eval.annotate import consensus, item_set_hash, validate_records
from pi_eval.annotate_llm import (
    AnnotationParseError,
    annotate_item,
    build_prompt,
    parse_reply,
    prompt_sha,
)
from pi_run import cmd_annotate
from pi_run.cli import build_parser

BUNDLE_ID = "musique-0-deadbeef"


class FakeLLM:
    """Structurally a `pi_eval.judges._llm.JudgeLLM`: `.complete(role=, messages=, seed=,
    max_tokens=None, **kw) -> (text, telemetry)`. Returns canned replies in call order and
    records every prompt it was asked, which is what lets a test assert on what the model
    actually saw without a real client or a network call.
    """

    def __init__(self, replies: list[str]):
        self._replies = list(replies)
        self.prompts: list[str] = []
        self.roles: list[str] = []

    def complete(self, *, role, messages, seed, max_tokens=None, **kw):
        self.prompts.append(messages[-1]["content"])
        self.roles.append(role)
        if not self._replies:
            raise AssertionError("FakeLLM ran out of canned replies")
        return self._replies.pop(0), None


def _item(task_type, iid, payload=None, context=None, **prov):
    p = {"suite": "musique", "task_id": "t1", "task_key": "t1", "graph_version": "v1"}
    p.update(prov)
    return {
        "item_id": iid,
        "task_type": task_type,
        "context": context or {},
        "payload": payload or {},
        "provenance": p,
    }


def _bundle(items):
    return {
        "manifest": {"bundle_id": BUNDLE_ID, "item_set_hash": item_set_hash(items)},
        "items": list(items),
    }


def _a1_item(iid="a1_t1", question="what did the treaty establish?"):
    return _item(
        "A1",
        iid,
        payload={
            "nodes": [
                {"node_id": "n1", "text": "who signed the treaty"},
                {"node_id": "n2", "text": "when was the treaty ratified"},
            ]
        },
        context={"question": question},
    )


def _a1_reply(
    ticked=("n1",),
    usefulness=None,
    likelihood=None,
    rationale="n1 is directly needed to answer",
):
    usefulness = usefulness if usefulness is not None else {"n1": 5, "n2": 2}
    likelihood = likelihood if likelihood is not None else {"n1": 70, "n2": 20}
    return json.dumps(
        {
            "ticked": list(ticked),
            "usefulness": usefulness,
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
            "question": "where did Andy go after prison",
            "evidence": [{"uid": "u1", "title": "Shawshank", "text": "he sailed south"}],
            "history": [{"q": "who is Andy", "a": "a banker"}],
            "draft": "Andy went to a coastal town.",
            "state_text": "state",
        },
        run_id="parent",
        turn_idx=1,
    )


# --------------------------------------------------------------------------- Part 1: records


def test_records_are_valid_and_are_not_human():
    item = _a1_item()
    bundle = _bundle([item])

    rec_a = annotate_item(
        FakeLLM([_a1_reply()]),
        item,
        bundle_id=BUNDLE_ID,
        model_pin="model-a@t0.0",
        annotator_id="llm:model-a",
    )
    rec_b = annotate_item(
        FakeLLM([_a1_reply(ticked=("n2",))]),
        item,
        bundle_id=BUNDLE_ID,
        model_pin="model-b@t0.0",
        annotator_id="llm:model-b",
    )

    assert validate_records(bundle, [rec_a, rec_b]) == []
    for rec in (rec_a, rec_b):
        assert rec["annotator_kind"] == "llm"
        assert rec["annotator_id"].startswith("llm:")
        assert rec["model_pin"]
        assert rec["prompt_sha"] and rec["response_sha"]

    # Two model raters on the SAME item are two samples of one prediction, not two
    # annotators -- `consensus`'s default counts humans only, so this must resolve nothing.
    cons = consensus(bundle, [rec_a, rec_b])
    assert cons.units == ()
    assert cons.rater_kinds == ("human",)


def test_a1_prompt_shows_the_question_and_no_evidence():
    item = _a1_item(question="what caused the collapse of the bridge?")
    prompt = build_prompt(item)

    assert "what caused the collapse of the bridge?" in prompt
    assert "who signed the treaty" in prompt
    assert "when was the treaty ratified" in prompt

    lowered = prompt.lower()
    for forbidden in ("evidence", "partition", "discoverability"):
        assert forbidden not in lowered, f"A1 prompt leaked {forbidden!r}"


def test_a2_prompt_does_not_reveal_the_pipeline_preference():
    prompt = build_prompt(_a2_item())
    lowered = prompt.lower()
    assert "which region did andy sail to" in lowered
    assert "what city was gotham filmed in" in lowered
    for forbidden in ("chosen", "rejected", "margin"):
        assert forbidden not in lowered, f"A2 prompt leaked {forbidden!r}"


def test_an_unreadable_reply_is_counted_not_defaulted():
    item = _a1_item()

    # Directly: parse_reply refuses to guess.
    with pytest.raises(AnnotationParseError):
        parse_reply(item, "not json at all")

    # Through the pass loop: the item is counted unparsed and no record is written for it.
    result = cmd_annotate._run_llm_pass(
        FakeLLM(["not json at all"]),
        SimpleNamespace(spent={"usd": 0.0}),
        [item],
        bundle_id=BUNDLE_ID,
        model_pin="model-a@t0.0",
        annotator_id="llm:model-a",
        seed=0,
        max_usd=None,
        already=set(),
    )
    assert result["records"] == []
    assert result["n_unparsed"] == 1
    assert result["n_remaining"] == 0


def test_a1_rates_every_node_not_only_the_unticked():
    item = _a1_item()
    reply = _a1_reply(ticked=("n1",), usefulness={"n1": 4, "n2": 1})
    parsed = parse_reply(item, reply)
    assert parsed.response["usefulness"]["n1"] == 4.0, "a ticked node keeps its own rating too"
    assert parsed.response["usefulness"]["n2"] == 1.0


# ---------------------------------------------------------- A1 anti-saturation (prompt + likelihood)


def test_a1_prompt_warns_against_ticking_everything():
    """A soft calibration warning against ticking most of the list -- see the module docstring
    on why this is aimed at the SATURATION of the tick-rate distribution, not at any single
    answer. It must read as a self-check, never as a quota: no fixed number of ticks is named."""
    prompt = build_prompt(_a1_item())
    lowered = prompt.lower()

    assert "small, specific set" in lowered
    assert "re-read your selection" in lowered
    assert "ticking most" in lowered

    # A calibration warning, not a quota: no hard cap is imposed anywhere in the prompt.
    for quota in ("at most 1", "at most 2", "at most one", "at most two", "no more than"):
        assert quota not in lowered, f"A1 prompt smuggled in a quota: {quota!r}"


def test_a1_prompt_does_not_mention_depth_or_hops():
    """The anti-circularity guard. The model's own failure mode (measured live) is reasoning
    about the dependency chain -- 'is this needed to solve the task?' -- instead of predicting
    what a person would ask. Coaching the prompt toward depth/hops/decomposition would just
    rebuild `is_latent` with an LLM, at which point 'agreement with gold_depth' would be
    circular: it would measure how well the prompt described depth, not how well the model
    predicts a person. So none of that vocabulary may appear in the built prompt."""
    prompt = build_prompt(_a1_item())
    lowered = prompt.lower()
    for forbidden in ("depth", "hop", "decomposition", "placeholder", "#1"):
        assert forbidden not in lowered, f"A1 prompt leaked {forbidden!r}"


def test_a1_likelihood_is_parsed_and_is_top_level():
    item = _a1_item()
    rec = annotate_item(
        FakeLLM([_a1_reply(likelihood={"n1": 83, "n2": 14})]),
        item,
        bundle_id=BUNDLE_ID,
        model_pin="model-a@t0.0",
        annotator_id="llm:model-a",
    )
    assert rec["likelihood"] == {"n1": 83.0, "n2": 14.0}
    assert "likelihood" not in rec["response"]


def test_a1_response_shape_is_unchanged():
    """`response` still has exactly `ticked` and `usefulness` -- `likelihood` lives only as a
    top-level sibling, so `consensus`/`iaa_report` and every other reader of `response` are
    unaffected by this change."""
    item = _a1_item()
    parsed = parse_reply(item, _a1_reply())
    assert set(parsed.response) == {"ticked", "usefulness"}


def test_a1_likelihood_out_of_range_is_a_parse_error():
    item = _a1_item()
    with pytest.raises(AnnotationParseError):
        parse_reply(item, _a1_reply(likelihood={"n1": 101, "n2": 20}))
    with pytest.raises(AnnotationParseError):
        parse_reply(item, _a1_reply(likelihood={"n1": -1, "n2": 20}))


def test_a1_likelihood_with_unknown_node_id_is_a_parse_error():
    item = _a1_item()
    with pytest.raises(AnnotationParseError):
        parse_reply(item, _a1_reply(likelihood={"n1": 70, "n2": 20, "n99": 5}))


def test_a1_missing_likelihood_is_a_parse_error():
    item = _a1_item()
    reply_without_likelihood = json.dumps(
        {
            "ticked": ["n1"],
            "usefulness": {"n1": 5, "n2": 2},
            "rationale": "n1 is directly needed to answer",
        }
    )
    with pytest.raises(AnnotationParseError):
        parse_reply(item, reply_without_likelihood)


# --------------------------------------------------------------------------- Part 2: the pass


def test_resume_skips_only_the_same_prompt_sha():
    item1 = _a1_item("a1_t1", question="question one")
    item2 = _a1_item("a1_t2", question="question two")
    sha1 = prompt_sha(build_prompt(item1))

    already = {(item1["item_id"], sha1)}
    result = cmd_annotate._run_llm_pass(
        FakeLLM([_a1_reply()]),  # only item2 should ever be sent
        SimpleNamespace(spent={"usd": 0.0}),
        [item1, item2],
        bundle_id=BUNDLE_ID,
        model_pin="model-a@t0.0",
        annotator_id="llm:model-a",
        seed=0,
        max_usd=None,
        already=already,
    )
    assert result["n_resumed"] == 1
    assert [r["item_id"] for r in result["records"]] == [item2["item_id"]]

    # A revised item1 (different context -> different prompt_sha) must NOT be skipped, even
    # though its item_id is the same one recorded under `already`.
    item1_revised = _a1_item("a1_t1", question="a materially different question")
    assert prompt_sha(build_prompt(item1_revised)) != sha1
    result2 = cmd_annotate._run_llm_pass(
        FakeLLM([_a1_reply(), _a1_reply()]),
        SimpleNamespace(spent={"usd": 0.0}),
        [item1_revised, item2],
        bundle_id=BUNDLE_ID,
        model_pin="model-a@t0.0",
        annotator_id="llm:model-a",
        seed=0,
        max_usd=None,
        already=already,
    )
    assert result2["n_resumed"] == 0
    assert {r["item_id"] for r in result2["records"]} == {item1["item_id"], item2["item_id"]}


def test_spend_cap_stops_the_pass_cleanly():
    items = [_a1_item("a1_t1"), _a1_item("a1_t2", question="a different one")]
    over_cap_ledger = SimpleNamespace(spent={"usd": 10.0})

    result = cmd_annotate._run_llm_pass(
        FakeLLM([]),  # must never be called: the cap is already exceeded
        over_cap_ledger,
        items,
        bundle_id=BUNDLE_ID,
        model_pin="model-a@t0.0",
        annotator_id="llm:model-a",
        seed=0,
        max_usd=5.0,
        already=set(),
    )
    assert result["stopped_for_spend"] is True
    assert result["records"] == []
    assert result["n_remaining"] == len(items)


def test_llm_pass_needs_no_gold_root(tmp_path, monkeypatch, capsys):
    # The conftest scrubs PI_GOLD_ROOT before every test; assert that positively rather than
    # trusting ambient state, since a real .env on this machine DOES set it.
    assert "PI_GOLD_ROOT" not in os.environ

    item = _a1_item()
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(json.dumps(_bundle([item])))
    out_path = tmp_path / "records.jsonl"

    monkeypatch.setenv("PI_MODEL_ANNOTATOR", "fake/model")
    monkeypatch.setattr(
        cmd_annotate,
        "_annotator_client",
        lambda model, *, cache_root_path, env: (
            FakeLLM([_a1_reply()]),
            SimpleNamespace(spent={"usd": 0.0}),
        ),
    )

    args = build_parser().parse_args(
        ["annotate", "llm", "--bundle", str(bundle_path), "--out", str(out_path), "--seed", "0"]
    )
    assert args.fn(args) == 0
    capsys.readouterr()

    assert "PI_GOLD_ROOT" not in os.environ, "the pass must not have set it either"
    written = [json.loads(ln) for ln in out_path.read_text().splitlines() if ln.strip()]
    assert len(written) == 1
    assert written[0]["annotator_kind"] == "llm"
    assert written[0]["annotator_id"] == "llm:fake/model"


def test_llm_out_is_not_silently_overwritten(tmp_path, monkeypatch, capsys):
    """Running `pi annotate llm` a second time without `--resume` used to truncate `--out` and
    silently discard the first run's records. It must now refuse, name the file, and name both
    ways out (`--resume`, `--overwrite`) -- and it must refuse BEFORE spending anything, so the
    fake client below is never even asked for a reply."""
    item1 = _a1_item("a1_t1")
    item2 = _a1_item("a1_t2", question="a different one")
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(json.dumps(_bundle([item1, item2])))
    out_path = tmp_path / "records.jsonl"

    prior = {
        "item_id": "a1_t1",
        "task_type": "A1",
        "annotator_id": "llm:fake/model",
        "annotator_kind": "llm",
        "model_pin": "fake/model@t0.0",
        "prompt_sha": prompt_sha(build_prompt(item1)),
        "response": {"ticked": [], "usefulness": {}},
    }
    out_path.write_text(json.dumps(prior) + "\n")

    monkeypatch.setenv("PI_MODEL_ANNOTATOR", "fake/model")
    monkeypatch.setattr(
        cmd_annotate,
        "_annotator_client",
        lambda model, *, cache_root_path, env: (
            FakeLLM([]),  # must never be called: the refusal happens before any request
            SimpleNamespace(spent={"usd": 0.0}),
        ),
    )

    args = build_parser().parse_args(
        ["annotate", "llm", "--bundle", str(bundle_path), "--out", str(out_path), "--seed", "0"]
    )
    rc = args.fn(args)
    err = capsys.readouterr().err
    assert rc == 2
    assert str(out_path) in err
    assert "--resume" in err
    assert "--overwrite" in err
    # untouched: still exactly the one prior record
    still = [json.loads(ln) for ln in out_path.read_text().splitlines() if ln.strip()]
    assert still == [prior]

    # --resume is the documented way past the refusal, and must not lose the prior record.
    args_resume = build_parser().parse_args(
        [
            "annotate",
            "llm",
            "--bundle",
            str(bundle_path),
            "--out",
            str(out_path),
            "--seed",
            "0",
            "--resume",
        ]
    )
    monkeypatch.setattr(
        cmd_annotate,
        "_annotator_client",
        lambda model, *, cache_root_path, env: (
            FakeLLM([_a1_reply()]),  # only item2: item1 is skipped at its prior prompt_sha
            SimpleNamespace(spent={"usd": 0.0}),
        ),
    )
    assert args_resume.fn(args_resume) == 0
    after_resume = [json.loads(ln) for ln in out_path.read_text().splitlines() if ln.strip()]
    assert len(after_resume) == 2, "the prior record must survive a --resume pass"
    assert prior in after_resume

    # --overwrite is the documented way to intentionally start over.
    out_path.write_text(json.dumps(prior) + "\n")
    args_overwrite = build_parser().parse_args(
        [
            "annotate",
            "llm",
            "--bundle",
            str(bundle_path),
            "--out",
            str(out_path),
            "--seed",
            "0",
            "--overwrite",
        ]
    )
    monkeypatch.setattr(
        cmd_annotate,
        "_annotator_client",
        lambda model, *, cache_root_path, env: (
            FakeLLM([_a1_reply(), _a1_reply()]),
            SimpleNamespace(spent={"usd": 0.0}),
        ),
    )
    assert args_overwrite.fn(args_overwrite) == 0
    after_overwrite = [json.loads(ln) for ln in out_path.read_text().splitlines() if ln.strip()]
    assert len(after_overwrite) == 2
    assert all(r.get("annotator_id") == "llm:fake/model" for r in after_overwrite)
    assert prior not in after_overwrite, "--overwrite must start from scratch, not append"


def test_llm_records_are_written_incrementally(tmp_path, monkeypatch):
    """`cmd_annotate_llm` used to write `--out` only once, at the very end. A 700-item pass is
    ~3 hours; a crash midway lost everything, and it silently defeated `--resume`, which reads
    that same file to learn what is already done. This simulates a crash (the model client
    runs out of canned replies, an `AssertionError`) after 2 of 4 items and asserts the 2
    records already produced survived it on disk."""
    items = [_a1_item(f"a1_t{i}", question=f"question {i}") for i in range(4)]
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(json.dumps(_bundle(items)))
    out_path = tmp_path / "records.jsonl"

    monkeypatch.setenv("PI_MODEL_ANNOTATOR", "fake/model")
    monkeypatch.setattr(
        cmd_annotate,
        "_annotator_client",
        lambda model, *, cache_root_path, env: (
            FakeLLM([_a1_reply(), _a1_reply()]),  # only 2 replies for 4 items -> simulated crash
            SimpleNamespace(spent={"usd": 0.0}),
        ),
    )
    args = build_parser().parse_args(
        ["annotate", "llm", "--bundle", str(bundle_path), "--out", str(out_path), "--seed", "0"]
    )
    with pytest.raises(AssertionError):
        args.fn(args)

    written = [json.loads(ln) for ln in out_path.read_text().splitlines() if ln.strip()]
    assert len(written) == 2, "a crash mid-pass must leave every record produced so far on disk"
    assert {r["item_id"] for r in written} == {"a1_t0", "a1_t1"}


def test_resume_after_an_interrupted_pass(tmp_path, monkeypatch):
    """`--resume` reads the file left behind by an interrupted pass (see the test above) and
    completes only the items not already annotated at their current prompt_sha."""
    items = [_a1_item(f"a1_t{i}", question=f"question {i}") for i in range(4)]
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(json.dumps(_bundle(items)))
    out_path = tmp_path / "records.jsonl"

    monkeypatch.setenv("PI_MODEL_ANNOTATOR", "fake/model")
    monkeypatch.setattr(
        cmd_annotate,
        "_annotator_client",
        lambda model, *, cache_root_path, env: (
            FakeLLM([_a1_reply(), _a1_reply()]),
            SimpleNamespace(spent={"usd": 0.0}),
        ),
    )
    args = build_parser().parse_args(
        ["annotate", "llm", "--bundle", str(bundle_path), "--out", str(out_path), "--seed", "0"]
    )
    with pytest.raises(AssertionError):
        args.fn(args)
    interrupted = [json.loads(ln) for ln in out_path.read_text().splitlines() if ln.strip()]
    assert len(interrupted) == 2

    monkeypatch.setattr(
        cmd_annotate,
        "_annotator_client",
        lambda model, *, cache_root_path, env: (
            FakeLLM([_a1_reply(), _a1_reply()]),  # only the 2 remaining items must be asked
            SimpleNamespace(spent={"usd": 0.0}),
        ),
    )
    args_resume = build_parser().parse_args(
        [
            "annotate",
            "llm",
            "--bundle",
            str(bundle_path),
            "--out",
            str(out_path),
            "--seed",
            "0",
            "--resume",
        ]
    )
    assert args_resume.fn(args_resume) == 0

    final = [json.loads(ln) for ln in out_path.read_text().splitlines() if ln.strip()]
    assert {r["item_id"] for r in final} == {"a1_t0", "a1_t1", "a1_t2", "a1_t3"}
    assert len(final) == 4, "the 2 interrupted records must survive, not be duplicated"


# --------------------------------------------------------------------------- Part 3: --concurrency


class ContentFakeLLM:
    """A `JudgeLLM` keyed by PROMPT CONTENT, not call order -- unlike `FakeLLM`'s queue, which
    a thread pool's nondeterministic interleaving would hand out to the wrong item. `sleep_s`
    is what lets a test prove work actually overlapped instead of running one call at a time
    disguised behind a thread pool; `fail_markers` names prompt substrings whose reply is
    deliberately unparseable, so `AnnotationParseError` fires for a KNOWN subset rather than by
    chance; `raise_markers` maps a substring to an exception to raise instead of replying,
    simulating a real (non-parse) crash on one item deterministically, regardless of which
    worker happens to pick it up.
    """

    def __init__(self, *, reply, sleep_s=0.0, fail_markers=(), raise_markers=None):
        self._reply = reply
        self._sleep_s = sleep_s
        self._fail_markers = tuple(fail_markers)
        self._raise_markers = dict(raise_markers or {})

    def complete(self, *, role, messages, seed, max_tokens=None, **kw):
        prompt = messages[-1]["content"]
        if self._sleep_s:
            time.sleep(self._sleep_s)
        for marker, exc in self._raise_markers.items():
            if marker in prompt:
                raise exc
        if any(m in prompt for m in self._fail_markers):
            return "not json at all", None
        return self._reply, None


def _many_a1_items(n):
    return [_a1_item(f"a1_t{i}", question=f"question marker {i}") for i in range(n)]


def _run_llm_cli(
    tmp_path, monkeypatch, items, llm, *, concurrency=None, resume=False, name="records.jsonl"
):
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(json.dumps(_bundle(items)))
    out_path = tmp_path / name

    monkeypatch.setenv("PI_MODEL_ANNOTATOR", "fake/model")
    monkeypatch.setattr(
        cmd_annotate,
        "_annotator_client",
        lambda model, *, cache_root_path, env: (llm, SimpleNamespace(spent={"usd": 0.0})),
    )
    argv = ["annotate", "llm", "--bundle", str(bundle_path), "--out", str(out_path), "--seed", "0"]
    if concurrency is not None:
        argv += ["--concurrency", str(concurrency)]
    if resume:
        argv += ["--resume"]
    args = build_parser().parse_args(argv)
    return args, out_path


def test_concurrency_one_is_the_default_and_unchanged():
    """The flag exists, defaults to 1, and every pre-existing sequential test above (none of
    which pass --concurrency) is exercising exactly that default -- see the module docstring on
    why this file has no reason to duplicate them."""
    args = build_parser().parse_args(["annotate", "llm", "--bundle", "b.json", "--out", "o.jsonl"])
    assert args.concurrency == 1


def test_concurrent_pass_writes_one_json_object_per_line(tmp_path, monkeypatch):
    items = _many_a1_items(6)
    llm = ContentFakeLLM(reply=_a1_reply(), sleep_s=0.02)
    args, out_path = _run_llm_cli(tmp_path, monkeypatch, items, llm, concurrency=4)

    assert args.fn(args) == 0
    lines = [ln for ln in out_path.read_text().splitlines() if ln.strip()]
    assert len(lines) == 6
    parsed = [json.loads(ln) for ln in lines]  # raises if any line is half-written / interleaved
    assert {r["item_id"] for r in parsed} == {it["item_id"] for it in items}
    assert all(r["annotator_kind"] == "llm" for r in parsed)


def test_unparsed_count_is_accurate_under_concurrency(tmp_path, monkeypatch, capsys):
    items = _many_a1_items(8)
    failing = {"question marker 1", "question marker 3", "question marker 5"}
    llm = ContentFakeLLM(reply=_a1_reply(), sleep_s=0.01, fail_markers=failing)
    args, out_path = _run_llm_cli(tmp_path, monkeypatch, items, llm, concurrency=4)

    assert args.fn(args) == 0
    out = capsys.readouterr().out
    assert "unparsed: 3/8" in out

    lines = [json.loads(ln) for ln in out_path.read_text().splitlines() if ln.strip()]
    assert len(lines) == 5
    failed_ids = {f"a1_t{i}" for i in (1, 3, 5)}
    assert {r["item_id"] for r in lines} == {it["item_id"] for it in items} - failed_ids


def test_concurrent_output_is_order_stable(tmp_path, monkeypatch):
    """Two passes over the SAME items at DIFFERENT concurrencies, with enough sleep to make
    completion order genuinely depend on scheduling, must still produce the identical set of
    records in the identical order -- see `_run_llm_pass_concurrent`'s docstring on why the
    file is re-sorted by item_id rather than left in completion order."""
    items = _many_a1_items(7)

    llm_a = ContentFakeLLM(reply=_a1_reply(), sleep_s=0.03)
    args_a, out_a = _run_llm_cli(tmp_path, monkeypatch, items, llm_a, concurrency=2, name="a.jsonl")
    assert args_a.fn(args_a) == 0

    llm_b = ContentFakeLLM(reply=_a1_reply(), sleep_s=0.005)
    args_b, out_b = _run_llm_cli(tmp_path, monkeypatch, items, llm_b, concurrency=6, name="b.jsonl")
    assert args_b.fn(args_b) == 0

    ids_a = [json.loads(ln)["item_id"] for ln in out_a.read_text().splitlines() if ln.strip()]
    ids_b = [json.loads(ln)["item_id"] for ln in out_b.read_text().splitlines() if ln.strip()]
    assert ids_a == ids_b == sorted(it["item_id"] for it in items)


def test_concurrent_pass_is_resumable(tmp_path, monkeypatch):
    """Interrupt (a real, non-parse crash on one item -- see `ContentFakeLLM`) partway through
    a --concurrency 2 pass over 4 items, then --resume it, and end with the full set and no
    duplicates. The crash item is last in submission order (see `_run_llm_pass_concurrent`'s
    priming discipline), so items 0 and 1 are guaranteed complete and on disk before it fires."""
    items = _many_a1_items(4)
    crash_marker = "question marker 3"
    llm = ContentFakeLLM(
        reply=_a1_reply(),
        raise_markers={crash_marker: RuntimeError("simulated crash")},
    )
    args, out_path = _run_llm_cli(tmp_path, monkeypatch, items, llm, concurrency=2)

    with pytest.raises(RuntimeError, match="simulated crash"):
        args.fn(args)

    interrupted = [json.loads(ln) for ln in out_path.read_text().splitlines() if ln.strip()]
    interrupted_ids = {r["item_id"] for r in interrupted}
    assert len(interrupted) == len(interrupted_ids), "no duplicate lines even mid-crash"
    assert "a1_t3" not in interrupted_ids, "the crashing item must never be recorded"
    assert {"a1_t0", "a1_t1"} <= interrupted_ids, "items ahead of the crash must have landed"
    assert len(interrupted) < 4

    llm_resume = ContentFakeLLM(reply=_a1_reply())  # no crash marker this time
    args_resume, _ = _run_llm_cli(
        tmp_path, monkeypatch, items, llm_resume, concurrency=2, resume=True
    )
    assert args_resume.fn(args_resume) == 0

    final = [json.loads(ln) for ln in out_path.read_text().splitlines() if ln.strip()]
    final_ids = [r["item_id"] for r in final]
    assert len(final_ids) == len(set(final_ids)) == 4, "no duplicates after resume"
    assert set(final_ids) == {it["item_id"] for it in items}


def test_llm_out_empty_file_is_not_treated_as_a_conflict(tmp_path, monkeypatch, capsys):
    """An empty or missing `--out` has nothing to lose, so neither `--resume` nor
    `--overwrite` should be required."""
    item = _a1_item()
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(json.dumps(_bundle([item])))
    out_path = tmp_path / "records.jsonl"
    out_path.write_text("")  # exists, but empty

    monkeypatch.setenv("PI_MODEL_ANNOTATOR", "fake/model")
    monkeypatch.setattr(
        cmd_annotate,
        "_annotator_client",
        lambda model, *, cache_root_path, env: (
            FakeLLM([_a1_reply()]),
            SimpleNamespace(spent={"usd": 0.0}),
        ),
    )
    args = build_parser().parse_args(
        ["annotate", "llm", "--bundle", str(bundle_path), "--out", str(out_path), "--seed", "0"]
    )
    assert args.fn(args) == 0
