"""The pairs the rule cannot order, a rater can -- and the export threw them away.

WHAT WAS MEASURED, AND WHY IT MOTIVATES THIS. The A6 campaign asked three raters to rank every
candidate at a state, so it produced a verdict on EVERY C(k,2) pair, including the ones
`export_pairs` refuses. Classifying its 2,188 majority preferences by what the exporter did
with the same candidate pair (2026-09-07, `~/pi-corpus-backup/annotations-20260907/a6`):

    bucket                      n     rater unanimity      mechanical score agrees
    exported                  559     41.3% [37.3,45.5]    59.1% [55.0,63.1]
    refused: per-state cap    710     42.8% [39.2,46.5]    59.2% [55.5,62.7]
    refused: below margin     485     43.5% [39.2,48.0]    50.2% [44.4,55.9]  (198 ties)
    refused: length delta     434     46.3% [41.7,51.0]    47.9% [43.0,52.9]

Raters separate all four buckets with the same confidence. The cap- and margin-refused pairs
are indistinguishable from the exported ones on unanimity, and the margin bucket is where our
own score is at chance (50.2%) while raters are not -- 485 preferences that are pure rater
signal, discarded because a number we compute could not order them.

THE ONE BUCKET THAT STAYS REFUSED, AND WHY. On length-refused pairs the rater-preferred side
is the LONGER question 71.7% [67.2,75.7] of the time, and 79.6% when the three raters are
unanimous -- against 52.6% on exported pairs. That bucket's high agreement IS a length
preference. Rescuing it would train the policy to write longer questions and would reproduce
the rung-0 GEPA failure, where the search winner beat its control by +0.003 (p=0.96) once
length was controlled. So the length guard is never forgiven, and this file pins that.

WHAT THIS ADDS.
  * `_decide(a, b, ...) -> (bool, str)` returns WHICH key ordered the pair. `_prefers` becomes
    its first element, so every existing caller is unchanged. Until now the only record of the
    deciding key was the file-level `rank_rule`, and the per-key agreement table in the paper
    had to be RE-DERIVED by replaying the precedence over the carried fields -- a derivation
    that silently breaks the day the precedence changes.
  * `label_source` on every pair: "rule" for the ordering the exporter computed, "rater" for a
    pair only a rater majority could order. A consumer must be able to train on one, the other
    or both, and `DPOConfig.include_label_sources` puts that choice in `cfg.sha`.
  * Rescue happens ONLY under `--rank rater`. `pairs.jsonl` is the control the latent-pursuit
    and rater-agreement numbers are quoted on; adding rater-ordered rows to it would make both
    circular, for the same reason `_anticipates` is a flag and not the default.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from pinq.actions import STOP_ACTION_JSON, ask_action_json
from pinq_train.export.dataset import _decide, _prefers, export_pairs

GOLDEN = Path(__file__).parent / "fixtures" / "pairs_control_golden.jsonl"
LONG_Q = "who was the founder of the company that first manufactured this particular device?"


def _c(run_id, value, question, *, correct=None, ttc=None, reach=None, is_stop=False, done=False):
    r = {
        "suite_id": "musique",
        "task_id": "t1",
        "template_id": None,
        "run_id": run_id,
        "turn_idx": 1,
        "state_text": "S",
        "action_json": STOP_ACTION_JSON if is_stop else ask_action_json(question),
        "value": value,
        "phi_tilde": value,
        "scorer_hash": "sh",
        "graph_version": "v1",
        "branch_of_run_id": "p",
        "branch_turn_idx": 1,
        "pins_sha": "pins-A",
        "budget_cap": 24,
        "max_turns": 24,
        "is_stop": is_stop,
        "done_before": done,
        "coverage_before": 1.0 if done else 0.4,
    }
    if correct is not None:
        r["answer_correct"] = correct
    if ttc is not None:
        r["turns_to_complete"] = ttc
    if reach is not None:
        r["newly_reachable"] = reach
    return r


def _ex(rows, **kw):
    kw.setdefault("margin_threshold", 0.05)
    kw.setdefault("len_delta_max", 40)
    return export_pairs(rows, **kw)


def _by_runs(pairs):
    return {frozenset((p.chosen_run_id, p.rejected_run_id)): p for p in pairs}


# ----------------------------------------------------------------- decided_by


def test_decide_names_the_key_and_prefers_is_its_bool():
    """`_prefers` keeps its exact contract: `_decide(...)[0]`. Every caller is unchanged."""
    a = _c("a", 0.9, "who?", correct=1.0)
    b = _c("b", 0.1, "when?", correct=0.0)
    assert _decide(a, b) == (True, "outcome")
    assert _decide(b, a) == (False, "outcome")
    assert _prefers(a, b) is _decide(a, b)[0]
    assert _prefers(b, a) is _decide(b, a)[0]


def test_every_pair_records_which_key_decided_it():
    """Three states, three keys. Without this the per-key agreement table in the paper is a
    re-derivation that breaks silently the day the precedence changes."""
    rows = [
        _c("a", 0.9, "who?", correct=1.0, ttc=2),  # outcome beats b
        _c("b", 0.3, "when?", correct=0.0, ttc=1),
        _c("c", 0.1, "where?", ttc=9),  # no outcome: speed, then gain
    ]
    pairs = _by_runs(_ex(rows)[0])
    assert pairs[frozenset(("a", "b"))].decided_by == "outcome"
    assert pairs[frozenset(("b", "c"))].decided_by == "quickest"
    # a vs c: a has an outcome, c does not -> outcome is unknown on one side and cannot fire;
    # both carry ttc, so speed decides.
    assert pairs[frozenset(("a", "c"))].decided_by == "quickest"

    plain = _by_runs(_ex([_c("a", 0.9, "who?"), _c("b", 0.1, "when?")])[0])
    assert plain[frozenset(("a", "b"))].decided_by == "gain"


def test_the_anticipation_key_names_itself_only_when_it_is_on():
    rows = [_c("a", 0.9, "who?", reach=False), _c("b", 0.1, "when?", reach=True)]
    assert _ex(rows)[0][0].decided_by == "gain"
    p = _ex(rows, rank="anticipation")[0][0]
    assert p.chosen_run_id == "b" and p.decided_by == "anticipation"


def test_ask_stop_pairs_record_their_own_rule():
    """The STOP direction is decided by gold, never by a value comparison, and the pair should
    say which gold fact did it."""
    done = _by_runs(
        _ex([_c("a", 0.9, "who?", done=True), _c("s", 0.0, "", is_stop=True, done=True)])[0]
    )
    assert done[frozenset(("s", "a"))].decided_by == "stop_done"

    live = _by_runs(_ex([_c("a", 0.9, "who?"), _c("s", 0.0, "", is_stop=True)])[0])
    assert live[frozenset(("a", "s"))].decided_by == "ask_clears_floor"

    synth = [
        p
        for p in _ex([_c("a", 0.9, "who?", done=True), _c("b", 0.1, "when?", done=True)])[0]
        if p.pair_kind == "ask_stop_synth"
    ]
    assert len(synth) == 1 and synth[0].decided_by == "stop_done"


# ----------------------------------------------------------------- the rescue


def test_a_below_margin_pair_is_rescued_only_under_rank_rater():
    """0.50 vs 0.49 is inside the 0.05 floor: our score cannot order it and says so. A rater
    majority can, and on this bucket our score is at chance (50.2%) while raters are not."""
    rows = [_c("a", 0.50, "who?"), _c("b", 0.49, "when?")]

    ctl, man_ctl = _ex(rows)
    assert ctl == [] and man_ctl.n_below_margin_dropped == 1
    assert man_ctl.n_rater_rescued_below_margin == 0

    got, man = _ex(rows, rank="rater", rater_prefs={frozenset(("a", "b")): "b"})
    assert len(got) == 1
    p = got[0]
    assert p.chosen_run_id == "b" and p.rejected_run_id == "a"
    assert p.label_source == "rater" and p.decided_by == "rater"
    assert p.margin == pytest.approx(0.01), "the magnitude is measured, not felt"
    assert man.n_rater_rescued_below_margin == 1
    assert man.n_below_margin_dropped == 1, "the rule's own refusal is still counted"


def test_only_the_margin_is_forgiven_never_the_length():
    """THE ONE THE MEASUREMENT FORBIDS. On length-refused pairs raters prefer the longer
    question 71.7% of the time; that bucket's agreement is a length preference, not a
    judgment about which need to pursue."""
    rows = [_c("a", 0.50, "who?"), _c("b", 0.49, LONG_Q)]
    prefs = {frozenset(("a", "b")): "b"}
    got, man = _ex(rows, rank="rater", rater_prefs=prefs)
    assert got == [], "a rater verdict may not buy a pair past the length guard"
    assert man.n_rater_rescued_below_margin == 0
    assert man.n_len_dropped + man.n_below_margin_dropped >= 1


def test_a_pair_with_no_verdict_is_not_rescued():
    rows = [_c("a", 0.50, "who?"), _c("b", 0.49, "when?")]
    got, man = _ex(rows, rank="rater", rater_prefs={frozenset(("x", "y")): "x"})
    assert got == [] and man.n_rater_rescued_below_margin == 0


def test_outcome_still_outranks_the_rater_on_a_rescue():
    """A below-margin pair whose two episodes differ in whether they answered the task is a
    pair the rule CAN order. It is not "the rule cannot order this", so it is not rescued --
    and the rater does not get to overturn the outcome key by the back door."""
    rows = [_c("a", 0.50, "who?", correct=1.0), _c("b", 0.49, "when?", correct=0.0)]
    got, man = _ex(rows, rank="rater", rater_prefs={frozenset(("a", "b")): "b"})
    assert got == [] and man.n_rater_rescued_below_margin == 0


def test_a_pair_cut_by_the_per_state_cap_is_rescued_outside_it():
    """The cap bounds C(n,2) growth from an accidental n; it is not a quality judgment, and the
    cap-refused bucket is indistinguishable from the exported one on rater unanimity."""
    qs = [
        "who founded it?",
        "when did it open?",
        "where is it?",
        "why was it built?",
        "which company owns it?",
        "how tall is it?",
    ]
    rows = [_c(f"r{i}", 0.9 - 0.1 * i, q) for i, q in enumerate(qs)]
    ctl, man_ctl = _ex(rows)
    assert len(ctl) == 12 and man_ctl.n_over_cap_dropped == 3, "C(6,2)=15, cap 12"

    ids = [r["run_id"] for r in rows]
    prefs = {frozenset((x, y)): y for i, x in enumerate(ids) for y in ids[i + 1 :]}
    got, man = _ex(rows, rank="rater", rater_prefs=prefs)
    assert len(got) == 15, "the 12 rule pairs plus the 3 the cap cut"
    assert man.n_rater_rescued_over_cap == 3
    assert man.n_over_cap_dropped == 3
    assert sum(p.label_source == "rater" for p in got) == 3
    assert sum(p.label_source == "rule" for p in got) == 12


def test_rescued_pairs_are_deterministic():
    """Two exports of the same rows must be byte-identical, or a re-export changes the file for
    reasons no manifest records."""
    qs = ["who founded it?", "when did it open?", "where is it?", "why was it built?"]
    rows = [_c(f"r{i}", 0.5 - 0.001 * i, q) for i, q in enumerate(qs)]
    ids = [r["run_id"] for r in rows]
    prefs = {frozenset((x, y)): y for i, x in enumerate(ids) for y in ids[i + 1 :]}

    def dump(pairs):
        # NaN != NaN, so asdict() dicts never compare equal -- not even to themselves.
        # Compare the SERIALISED rows, which is what ships and how a consumer reads back.
        return [json.dumps(dataclasses.asdict(x), sort_keys=True, default=str) for x in pairs]

    assert dump(_ex(rows, rank="rater", rater_prefs=prefs)[0]) == dump(
        _ex(rows, rank="rater", rater_prefs=prefs)[0]
    )


# ----------------------------------------------------------------- the control


def test_the_control_export_is_unchanged_but_for_the_added_keys():
    """THE REGRESSION EVERYTHING ABOVE COULD CAUSE. `pairs.jsonl` is the artifact the
    latent-pursuit headline and the rater-agreement number are quoted on. The golden was
    generated at commit af3bc48, BEFORE any of these changes:

        .venv/bin/python tests/fixtures/make_pairs_control_golden.py
        sha256 4400ff9465ca824eb9b257aedf5c7985a33b68b2c940b42df0117983c281a95f

    Every control row must still match it once the ADDED keys are removed.

    WAS `..._but_for_the_two_new_keys`, popping `label_source` and `decided_by`. The BELIEF that
    changed, not the code: "two" was a fact about the day it was written, and `code_version` and
    `arm_id` were added under I.13 #5 so a row can name the loop that rendered its prompt and the
    policy that asked. The golden is NOT regenerated -- regenerating it would let the test rewrite
    the expectation it exists to check -- so the new keys are popped and asserted EMPTY, which is
    what an unstamped fixture row must serialise as.

    The test gets stricter, not laxer, in the same edit: `pair_id` is now asserted against the
    golden's byte for byte. That is the field the whole annotation corpus joins on (2,429 rater
    preferences are keyed by it), and popping keys without pinning it would have left the one
    regression that actually matters uncovered.
    """
    gen = GOLDEN.parent / "make_pairs_control_golden.py"
    src = gen.read_text()
    # `__name__` deliberately NOT "__main__": that is what keeps the generator from rewriting
    # the golden this test compares against.
    ns: dict = {"__file__": str(gen), "__name__": "make_golden"}
    assert '__name__ == "__main__"' in src, (
        "the generator must not rewrite the golden when this test execs it -- without the "
        "guard the test passes by regenerating its own expectation"
    )
    exec(
        compile(
            GOLDEN.parent.joinpath("make_pairs_control_golden.py").read_text(),
            "make_golden",
            "exec",
        ),
        ns,
    )
    pairs, man = export_pairs(ns["ROWS"], margin_threshold=0.05, len_delta_max=40)
    want = [x for x in GOLDEN.read_text().splitlines() if x.strip()]
    assert len(pairs) == len(want) == 6

    # PINNED BEFORE ANYTHING IS POPPED: the join key onto every annotation campaign.
    assert [p.pair_id for p in pairs] == [json.loads(w)["pair_id"] for w in want]

    for got_pair, want_row in zip(pairs, want, strict=True):
        got = dataclasses.asdict(got_pair)
        # Added by I.13 #5. Empty here because the fixture's rows name no code and no arm, and
        # "not recorded" must serialise as "" rather than as a guessed default.
        assert got.pop("code_version") == ""
        assert got.pop("arm_id") == ""
        assert got.pop("label_source") == "rule"
        assert got.pop("decided_by") in {
            "outcome",
            "quickest",
            "gain",
            "stop_done",
            "ask_clears_floor",
        }
        # Serialised, not as dicts: NaN != NaN would make every row unequal to itself.
        assert json.dumps(got, sort_keys=True, default=str) == want_row

    assert man.rank_rule == "outcome"
    assert man.n_rater_rescued_below_margin == 0 and man.n_rater_rescued_over_cap == 0


def test_the_manifest_names_the_preference_source():
    """Two artifacts can both be rank_rule="rater" and be different experiments: one ordered by
    A6 candidate rankings, one by A7's `reaches` axis. Without the source in the manifest they
    are indistinguishable after the fact, and rule 1 says a number that cannot name its inputs
    is not a result. The sha is of the preference SET, so a file renamed or re-sorted still
    identifies the same ordering."""
    rows = [_c("a", 0.50, "who?"), _c("b", 0.49, "when?")]
    prefs = {frozenset(("a", "b")): "b"}
    _, man = _ex(rows, rank="rater", rater_prefs=prefs)
    assert man.rater_prefs_sha, "an ordering with no named source is unprovenanced"
    assert len(man.rater_prefs_sha) == 16

    # a DIFFERENT preference set gives a different sha
    _, man2 = _ex(rows, rank="rater", rater_prefs={frozenset(("a", "b")): "a"})
    assert man2.rater_prefs_sha != man.rater_prefs_sha

    # and the control names nothing, because it ordered nothing this way
    _, ctl = _ex(rows)
    assert ctl.rater_prefs_sha == ""
