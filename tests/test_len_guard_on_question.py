"""The length guard must measure the QUESTION, not the rationale riding beside it.

The guard exists to block one confound: "longer question wins" (rung2_dpo.train names it).
But it was measured over the whole `action_json`, which includes the free-text `rationale`
-- 60-120 chars of sample-varying prose that says nothing the pair is about. Measured on the
6,896 branch runs on disk: the margin/length gate killed ~74% of same-state candidate pairs
while the identical-action drop killed ~12%, and kept pairs hug the cap (len_delta median 17,
max 40) -- i.e. two equally good questions with differently-worded rationales were routinely
discarded as if one were verbose.

So the delta is computed on the parsed question text (the same normalisation `_same_action`
already uses), in ONE shared function consumed by both the exporter and rung 2's load-time
re-check -- two implementations of the same guard would drift, and the load-time check
rejecting the exporter's own artifact is exactly the failure mode.
"""

from __future__ import annotations

import pytest

from pinq_train.export.dataset import export_pairs, question_len_delta
from pinq_train.rung2_dpo.train import assert_length_guard

LONG_RATIONALE = (
    "because the evidence chain requires resolving this entity before anything else can proceed"
)
SHORT_RATIONALE = "need it"


def _c(run_id, value, question, rationale, **kw):
    r = {
        "suite_id": "musique",
        "task_id": "t1",
        "template_id": None,
        "run_id": run_id,
        "turn_idx": 1,
        "state_text": "S",
        "action_json": f'{{"action":"ASK","question":"{question}","rationale":"{rationale}"}}',
        "value": value,
        "phi_tilde": value,
        "scorer_hash": "sh",
        "graph_version": "v1",
        "branch_of_run_id": "p",
        "branch_turn_idx": 1,
    }
    r.update(kw)
    return r


def test_rationale_wording_variance_does_not_kill_a_pair() -> None:
    """Same-length questions, rationales differing by far more than the cap: the pair is
    about the questions, and it survives."""
    a = _c("win", 0.9, "who owned The Collegian", LONG_RATIONALE)
    b = _c("lose", 0.1, "who founded the paper?!", SHORT_RATIONALE)
    assert abs(len(a["action_json"]) - len(b["action_json"])) > 40, (
        "fixture must exceed the old measure"
    )
    pairs, man = export_pairs([a, b], margin_threshold=0.05, len_delta_max=40)
    assert len(pairs) == 1, "a rationale-length difference is not a question-length difference"


def test_a_genuinely_verbose_question_is_still_blocked() -> None:
    """The confound the guard exists for is untouched: question deltas beyond the cap drop."""
    verbose = "who owned The Collegian " + "and also every predecessor publication " * 3
    a = _c("win", 0.9, verbose, SHORT_RATIONALE)
    b = _c("lose", 0.1, "who owned it", SHORT_RATIONALE)
    pairs, man = export_pairs([a, b], margin_threshold=0.05, len_delta_max=40)
    assert pairs == []


def test_exported_len_delta_field_is_the_question_delta() -> None:
    a = _c("win", 0.9, "who owned The Collegian in 1960", LONG_RATIONALE)
    b = _c("lose", 0.1, "who owned The Collegian", SHORT_RATIONALE)
    pairs, _ = export_pairs([a, b], margin_threshold=0.05, len_delta_max=40)
    assert pairs[0].len_delta == len("who owned The Collegian in 1960") - len(
        "who owned The Collegian"
    )


def test_rung2_load_guard_agrees_with_the_exporter() -> None:
    """The load-time re-check uses the same measure, so it accepts the exporter's own
    artifact -- and still rejects a question-length confound in a hand-edited file."""
    ok = {
        "chosen_json": f'{{"action":"ASK","question":"who owned The Collegian","rationale":"{LONG_RATIONALE}"}}',
        "rejected_json": '{"action":"ASK","question":"who founded the paper?!","rationale":"need it"}',
    }
    assert_length_guard(ok, len_delta_max=40)  # must not raise

    verbose = "who owned The Collegian " + "and also every predecessor publication " * 3
    bad = {
        "chosen_json": f'{{"action":"ASK","question":"{verbose}","rationale":"need it"}}',
        "rejected_json": '{"action":"ASK","question":"who owned it","rationale":"need it"}',
    }
    with pytest.raises(ValueError):
        assert_length_guard(bad, len_delta_max=40)


def test_unparseable_actions_fall_back_to_the_whole_string() -> None:
    """Conservative direction, same as `_same_action`: junk that cannot be parsed is
    measured whole, not waved through."""
    assert question_len_delta("not json at all", "x") == len("not json at all") - 1


def test_manifest_records_the_measure_and_counts_the_drop() -> None:
    """The rule change is part of the artifact's definition, so the manifest says which
    measure was in force -- and the margin/length gate is no longer a silent `continue`."""
    verbose = "who owned The Collegian " + "and also every predecessor publication " * 3
    a = _c("win", 0.9, verbose, SHORT_RATIONALE)
    b = _c("lose", 0.1, "who owned it", SHORT_RATIONALE)
    _, man = export_pairs([a, b], margin_threshold=0.05, len_delta_max=40)
    assert man.len_delta_on == "question"
    assert man.n_len_dropped == 1
    c = _c("w2", 0.5, "who owned The Collegian", SHORT_RATIONALE)
    d = _c("l2", 0.49, "who founded The Collegian", SHORT_RATIONALE)
    _, man2 = export_pairs([c, d], margin_threshold=0.05, len_delta_max=40)
    assert man2.n_below_margin_dropped == 1
