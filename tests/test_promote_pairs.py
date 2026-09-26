"""Re-attaching rater verdicts to a re-exported pairs file, without losing or inventing one.

WHAT THIS PROTECTS. `pi train export` overwrites `data/rl/pairs.jsonl` and carries no
annotation forward; three passes of A2 verdicts were lost that way. The verdicts survive
because the immutable records are the system of record -- but only if re-attaching them is
code with tests rather than a script in a transcript.

THE TWO DEFECTS THESE WERE WRITTEN FOR, both found by running the real corpus through the
first draft of `pi_run.promote`:

  * ITEM IDS ARE NOT UNIQUE ACROSS BUNDLES. The round-1 and round-2 bundles share them, so an
    items dict keyed on the id alone resolved every round-1 record to the round-2 item. Two
    consequences, both silent: the quarantine was looked up under the wrong bundle, so
    `claude-opus-5` and `gpt-5.6-sol` -- dropped from the shipped panel for slot bias -- had
    their votes counted back into the majority; and 310 pairs joined to the wrong state and
    lost their verdict. Keying on `(bundle_id, item_id)` and resolving each record through its
    OWN `bundle_id` fixed both, and the attach then reproduced all 2,588 shipped rows exactly.
  * AN UNDECIDED VERDICT IS NOT A DISAGREEMENT. The shipped file records
    `a7_agrees_with_label = false` on 410 rows whose majority is `split`, `tie` or `both_bad`.
    The raters did not decide, so agreement is UNKNOWN. Reading those as disagreements makes
    the rate 979/2588 = 37.8% instead of 979/1605 = 61.0%.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pi_run.promote import (
    AnnotationLoss,
    QuarantineDisagreement,
    QuarantineUnenforceable,
    a5_verdicts,
    a6_verdicts,
    a7_verdicts,
    annotation_loss,
    attach,
    check_quarantine,
    load_root,
    majority,
    pair_identity,
    promote,
    rater_of,
)

RATERS = ("gemini-3.1-pro", "claude-sonnet-5", "gpt-oss-120b")


def _bundle(root: Path, bundle_id: str, items: list[dict], keys: dict) -> None:
    d = root / "bundles"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{bundle_id}.json").write_text(json.dumps({"bundle_id": bundle_id, "items": items}))
    (d / f"{bundle_id}.key.json").write_text(json.dumps({"items": keys}))


def _records(root: Path, name: str, recs: list[dict]) -> None:
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    by: dict[str, list[dict]] = {}
    for r in recs:
        by.setdefault(rater_of(r["annotator_id"]), []).append(r)
    for rater, rs in by.items():
        (d / f"{rater}.jsonl").write_text("\n".join(json.dumps(r) for r in rs))


def _a7_item(item_id: str, *, run: str, turn: int, chosen: str, rejected: str, order="ab"):
    item = {
        "item_id": item_id,
        "task_type": "A7",
        "payload": {},
        "provenance": {"suite": "musique", "task_id": "t1", "run_id": run, "turn_idx": turn},
    }
    key = {
        "chosen_run_id": chosen,
        "rejected_run_id": rejected,
        "order": order,
        "pair_id": "OLD-" + item_id,
    }
    return item, key


def _a7_rec(
    item_id: str, bundle: str, rater: str, pref: str, *, a="stays_stated", b="reaches_unstated"
):
    return {
        "item_id": item_id,
        "bundle_id": bundle,
        "task_type": "A7",
        "annotator_id": f"llm:openai/x/{rater}",
        "response": {"preference": pref, "a_reaches": a, "b_reaches": b},
    }


def _pair_row(chosen: str, rejected: str, *, run="p", turn=1, **over):
    r = {
        "suite_id": "musique",
        "task_id": "t1",
        "run_id": run,
        "turn_idx": turn,
        "chosen_run_id": chosen,
        "rejected_run_id": rejected,
        "chosen_json": json.dumps({"action": "ASK", "question": "who founded it?"}),
        "rejected_json": json.dumps({"action": "ASK", "question": "when did it open?"}),
        "pair_id": "NEW-abc",
        "pair_kind": "ask_ask",
    }
    r.update(over)
    return r


@pytest.fixture
def root(tmp_path):
    """One bundle, three raters, one rated pair -- plus a SECOND bundle that reuses the same
    item id, which is the collision the real corpus has."""
    r = tmp_path / "ann"
    it, key = _a7_item("SHARED", run="p", turn=1, chosen="cand-a", rejected="cand-b")
    _bundle(r, "b1", [it], {"SHARED": key})
    # Same item id, different bundle, DIFFERENT pair. Resolving a b1 record to this item is
    # the bug.
    it2, key2 = _a7_item("SHARED", run="q", turn=2, chosen="cand-x", rejected="cand-y")
    _bundle(r, "b2", [it2], {"SHARED": key2})
    _records(r, "round1", [_a7_rec("SHARED", "b1", x, "a") for x in RATERS])
    _records(r, "round2", [_a7_rec("SHARED", "b2", x, "b") for x in RATERS])
    return r


@pytest.fixture
def quarantine(tmp_path):
    p = tmp_path / "q.json"
    p.write_text(json.dumps({"_README": ["ignored"], "b1": {}, "b2": {}}))
    return p


def test_a_shared_item_id_resolves_through_the_records_own_bundle(root, quarantine, tmp_path):
    """THE COLLISION. Both bundles have an item called SHARED and they are different pairs."""
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text(
        json.dumps(_pair_row("cand-a", "cand-b", run="p", turn=1))
        + "\n"
        + json.dumps(_pair_row("cand-x", "cand-y", run="q", turn=2))
        + "\n"
    )
    promote(pairs, root, quarantine)
    rows = [json.loads(x) for x in pairs.read_text().splitlines()]
    first = next(r for r in rows if r["chosen_run_id"] == "cand-a")
    second = next(r for r in rows if r["chosen_run_id"] == "cand-x")
    # b1 raters all said slot "a" and b1's order is "ab", so the chosen side won.
    assert first["a7_majority"] == "chosen" and first["a7_round"] == "r1"
    # b2 raters all said slot "b" -> the rejected side.
    assert second["a7_majority"] == "rejected" and second["a7_round"] == "r2"


def test_attach_by_identity_survives_a_pair_id_change(root, quarantine, tmp_path):
    """A re-export changed every `pair_id`: 0 of 27,266 matched a previously rated set."""
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text(json.dumps(_pair_row("cand-a", "cand-b", pair_id="TOTALLY-DIFFERENT")) + "\n")
    promote(pairs, root, quarantine)
    row = json.loads(pairs.read_text().splitlines()[0])
    assert row["a7_majority"] == "chosen" and row["a7_n_raters"] == 3


def test_a_flipped_direction_flips_the_agreement_not_the_verdict(root, quarantine, tmp_path):
    """`pairs.rater.jsonl` may order a pair the other way. The verdict is about the two
    candidates; agreement is about THIS row's direction."""
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text(json.dumps(_pair_row("cand-b", "cand-a")) + "\n")
    promote(pairs, root, quarantine)
    row = json.loads(pairs.read_text().splitlines()[0])
    assert row["a7_majority"] == "rejected", "the raters still preferred cand-a"
    assert row["a7_agrees_with_label"] is False


def test_an_undecided_majority_is_unknown_agreement_not_disagreement(root, quarantine, tmp_path):
    """410 shipped rows record `false` where the raters simply did not decide."""
    _records(
        root,
        "round1",
        [_a7_rec("SHARED", "b1", r, p) for r, p in zip(RATERS, ("a", "b", "tie"), strict=True)],
    )
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text(json.dumps(_pair_row("cand-a", "cand-b")) + "\n")
    promote(pairs, root, quarantine)
    row = json.loads(pairs.read_text().splitlines()[0])
    assert row["a7_majority"] == "split"
    assert row["a7_agrees_with_label"] is None


def test_a_quarantined_raters_vote_is_absent_from_the_row(root, tmp_path):
    q = tmp_path / "q.json"
    q.write_text(json.dumps({"b1": {"gpt-oss-120b": ["preference"]}, "b2": {}}))
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text(json.dumps(_pair_row("cand-a", "cand-b")) + "\n")
    promote(pairs, root, q)
    row = json.loads(pairs.read_text().splitlines()[0])
    assert "gpt-oss-120b" not in row["a7_votes"]
    assert row["a7_n_raters"] == 2 and "gpt-oss-120b" not in row["a7_rater_panel"]
    # quarantined on `preference` only -- its reach judgement is still carried
    assert "gpt-oss-120b" in row["a7_reaches"]


def test_a_biased_rater_that_is_not_quarantined_is_fatal(tmp_path):
    """The unsafe direction: position-reading on `preference`, the axis that decides a pair."""
    r = tmp_path / "ann"
    items, keys, recs = [], {}, []
    for i in range(14):
        it, key = _a7_item(f"i{i}", run="p", turn=1, chosen=f"c{i}", rejected=f"d{i}")
        items.append(it)
        keys[f"i{i}"] = key
        recs.append(_a7_rec(f"i{i}", "b1", "gemini-3.1-pro", "a"))  # always slot a
    _bundle(r, "b1", items, keys)
    _records(r, "round1", recs)
    q = tmp_path / "q.json"
    q.write_text(json.dumps({"b1": {}}))
    with pytest.raises(QuarantineDisagreement, match="preference"):
        check_quarantine(load_root(r), json.loads(q.read_text()))


def test_an_over_strict_quarantine_is_recorded_not_refused(root, tmp_path):
    """Discarding evidence is safe; admitting bad evidence is not. claude-opus-5 on the real
    round-1 bundle is exactly this case (p = 0.00144 against alpha = 0.001)."""
    q = tmp_path / "q.json"
    q.write_text(json.dumps({"b1": {"gemini-3.1-pro": ["preference"]}, "b2": {}}))
    rep = check_quarantine(load_root(root), json.loads(q.read_text()))
    assert rep["b1"]["gemini-3.1-pro"]["preference"]["conservative"] is True


def test_the_loss_guard_counts_and_aborts(root, quarantine, tmp_path):
    """A promote against a root missing a campaign must not silently thin the file."""
    pairs = tmp_path / "pairs.jsonl"
    row = _pair_row("cand-a", "cand-b")
    row["a6_winner_run"] = "cand-a"  # a verdict this root cannot reproduce
    pairs.write_text(json.dumps(row) + "\n")
    before = pairs.read_bytes()
    with pytest.raises(AnnotationLoss, match="a6_winner_run"):
        promote(pairs, root, quarantine)
    assert pairs.read_bytes() == before, "the target must be untouched when the guard fires"
    promote(pairs, root, quarantine, allow_loss=True)
    assert "a6_winner_run" not in json.loads(pairs.read_text().splitlines()[0])


def test_promote_is_idempotent(root, quarantine, tmp_path):
    """Re-running over unchanged inputs must be byte-identical, or "did anything change?" is
    not a question the artifact can answer. No timestamps anywhere."""
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text(json.dumps(_pair_row("cand-a", "cand-b")) + "\n")
    promote(pairs, root, quarantine)
    once = pairs.read_bytes()
    promote(pairs, root, quarantine)
    assert pairs.read_bytes() == once


def test_the_distinctness_label_rides_along(root, quarantine, tmp_path):
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text(json.dumps(_pair_row("cand-a", "cand-b")) + "\n")
    blocks = promote(pairs, root, quarantine)
    row = json.loads(pairs.read_text().splitlines()[0])
    assert row["q_distinctness"] in ("paraphrase", "related", "different")
    assert sum(blocks["promote"]["q_distinctness"].values()) == 1


def test_a_proposal_bundle_is_unjoinable_and_counted(root, quarantine, tmp_path):
    """The suggest campaign's items have no state provenance -- a proposal has no rollout --
    so they can never join a pair, and that must be a number rather than a silent skip."""
    d = root / "bundles"
    (d / "sug.json").write_text(
        json.dumps(
            {
                "bundle_id": "sug",
                "items": [
                    {
                        "item_id": "s1",
                        "task_type": "A7",
                        "payload": {},
                        "provenance": {"origin": "suggest_on_disagreement"},
                    }
                ],
            }
        )
    )
    (d / "sug.key.json").write_text(json.dumps({"items": {}}))
    assert load_root(root).n_unjoinable_items == 1


def test_majority_needs_a_strict_majority_and_two_raters_must_agree():
    assert majority({"a": "x", "b": "x", "c": "y"}) == ("x", False)
    assert majority({"a": "x", "b": "x", "c": "x"}) == ("x", True)
    assert majority({"a": "x", "b": "y"}) == ("split", False)
    assert majority({"a": "x", "b": "x"}) == ("x", True)
    assert majority({}) == ("split", False)


def test_the_rater_name_is_a_rule_not_a_table():
    assert rater_of("llm:openai/gcp/gemini-3.1-pro-preview") == "gemini-3.1-pro"
    assert rater_of("llm:openai/aws/claude-sonnet-5") == "claude-sonnet-5"
    assert rater_of("llm:azure/gpt-5.6-sol") == "gpt-5.6-sol"


def test_the_identity_is_unordered_in_the_candidates_only():
    a = pair_identity(_pair_row("x", "y"))
    assert a == pair_identity(_pair_row("y", "x"))
    assert a != pair_identity(_pair_row("x", "y", turn=2))
    assert a != pair_identity(_pair_row("x", "y", run="other"))


def test_annotation_loss_reports_a_missing_row_separately():
    old = [_pair_row("a", "b") | {"a7_majority": "chosen"}]
    assert annotation_loss(old, []) == {"row_absent": 1}
    assert annotation_loss(old, [_pair_row("a", "b")]) == {"a7_majority": 1}
    assert annotation_loss(old, old) == {}


def test_every_preferences_file_is_read_not_just_the_last(tmp_path, quarantine):
    """`load_root` ASSIGNED `ann.preferences` per file instead of extending it, so a root with
    two A6 campaigns kept only the last one read: 241 of 2,429 preferences survived, and the
    rescued pairs that the missing 2,188 had ordered arrived carrying no verdict to audit them
    with. Directory order decided which campaign counted, silently."""
    r = tmp_path / "ann"
    for bundle, item, runs in (("b1", "i1", ("ra", "rb")), ("b2", "i2", ("rc", "rd"))):
        it = {
            "item_id": item,
            "task_type": "A6",
            "payload": {},
            "provenance": {"suite": "musique", "task_id": "t1", "run_id": "p", "turn_idx": 1},
        }
        _bundle(r, bundle, [it], {item: {"candidate_runs": {"c0": runs[0], "c1": runs[1]}}})
    for d, item, runs in (("a6", "i1", ("ra", "rb")), ("a6b", "i2", ("rc", "rd"))):
        (r / d).mkdir(parents=True, exist_ok=True)
        (r / d / "preferences.jsonl").write_text(
            json.dumps(
                {
                    "item_id": item,
                    "winner": "c0",
                    "loser": "c1",
                    "winner_run": runs[0],
                    "loser_run": runs[1],
                    "n_votes": 3,
                    "raters": ["x", "y", "z"],
                    "unanimous": True,
                }
            )
            + "\n"
        )
    ann = load_root(r)
    assert len(ann.preferences) == 2, "both campaigns' preferences must survive"
    from pi_run.promote import a6_verdicts

    assert len(a6_verdicts(ann, {})) == 2


def test_one_rater_is_not_a_majority_and_not_unanimous():
    """AUDIT FINDING. `majority()` accepted a single vote: n=1, len=1, and 1*2 > 1, so one
    rater's opinion came back as a unanimous panel verdict. A7 never exposed it because every
    A7 row has at least two raters, but 21 A5 runs were judged by one -- and 30 shipped pairs
    carried the result flagged `unanimous`, indistinguishable from a 3-0 verdict.

    The docstring already said a majority of two is one person outvoting nobody. A majority of
    one is that with the pretence removed."""
    assert majority({"a": "x"}) == ("split", False)
    assert majority({"a": "x", "b": "x"}) == ("x", True)
    assert majority({"a": "x", "b": "y"}) == ("split", False)


def test_an_a5_verdict_carries_its_vote_count(root, quarantine, tmp_path):
    """A row that says `unanimous` without saying how many raters agreed cannot be filtered
    honestly: 3-0 and 2-0 are both unanimous and are not the same evidence."""
    r = root
    it = {
        "item_id": "a5x",
        "task_type": "A5",
        "payload": {"candidates": []},
        "provenance": {"suite": "musique", "task_id": "t1", "run_id": "cand-a"},
    }
    _bundle(r, "b3", [it], {"a5x": {}})
    _records(
        r,
        "round3",
        [
            {
                "item_id": "a5x",
                "bundle_id": "b3",
                "task_type": "A5",
                "annotator_id": f"llm:openai/x/{n}",
                "response": {"verdict": "stopping_was_right"},
            }
            for n in ("gemini-3.1-pro", "gpt-oss-120b")
        ],
    )
    pairs = tmp_path / "p.jsonl"
    pairs.write_text(json.dumps(_pair_row("cand-a", "cand-b")) + "\n")
    promote(pairs, r, quarantine)
    row = json.loads(pairs.read_text().splitlines()[0])
    assert row["a5_chosen_verdict"] == "stopping_was_right"
    assert row["a5_chosen_unanimous"] is True
    assert row["a5_chosen_n_votes"] == 2


def test_the_two_a5_instruments_are_never_pooled(root, quarantine, tmp_path):
    """AUDIT FINDING. `a5_verdicts` keyed on run id alone, so a run rated by BOTH the sighted
    and the blinded A5 had its votes merged into one verdict -- 34 runs on the live corpus.
    The two instruments deliberately show the rater different things (the sighted one prints
    the unresolved gold nodes and the final answer; the blind one prints neither), and their
    verdicts differ systematically because of it. Pooling them destroys the only comparison
    the blinded campaign exists to make.

    Partitioned on the ITEM's own `blind` flag, not on a list of bundle names, so a future
    campaign cannot be mis-filed by being named something new."""
    r = root
    sighted = {
        "item_id": "s1",
        "task_type": "A5",
        "payload": {"candidates": [{"node_id": "n1", "text": "the founder"}]},
        "provenance": {"suite": "musique", "task_id": "t1", "run_id": "cand-a"},
    }
    blind = {
        "item_id": "b1",
        "task_type": "A5",
        "payload": {"blind": True},
        "provenance": {"suite": "musique", "task_id": "t1", "run_id": "cand-a"},
    }
    _bundle(r, "bs", [sighted], {"s1": {}})
    _bundle(r, "bb", [blind], {"b1": {}})
    _records(
        r,
        "sighted",
        [
            {
                "item_id": "s1",
                "bundle_id": "bs",
                "task_type": "A5",
                "annotator_id": f"llm:openai/x/{n}",
                "response": {"verdict": "stopping_was_right"},
            }
            for n in ("gemini-3.1-pro", "gpt-oss-120b", "claude-sonnet-5")
        ],
    )
    _records(
        r,
        "zblind",
        [
            {
                "item_id": "b1",
                "bundle_id": "bb",
                "task_type": "A5",
                "annotator_id": f"llm:openai/x/{n}",
                "response": {"verdict": "should_have_asked_more", "missing": ["who founded it"]},
            }
            for n in ("gemini-3.1-pro", "gpt-oss-120b", "claude-sonnet-5")
        ],
    )
    pairs = tmp_path / "p.jsonl"
    pairs.write_text(json.dumps(_pair_row("cand-a", "cand-b")) + "\n")
    promote(pairs, r, quarantine)
    row = json.loads(pairs.read_text().splitlines()[0])
    assert row["a5_chosen_verdict"] == "stopping_was_right"
    assert row["a5_chosen_n_votes"] == 3
    assert row["a5blind_chosen_verdict"] == "should_have_asked_more"
    assert row["a5blind_chosen_n_votes"] == 3


def _a6_item(item_id: str, *, run: str, turn: int, runs: dict[str, str]):
    item = {
        "item_id": item_id,
        "task_type": "A6",
        "payload": {},
        "provenance": {"suite": "musique", "task_id": "t1", "run_id": run, "turn_idx": turn},
    }
    return item, {"candidate_runs": dict(runs)}


def _a6_rec(bundle: str, item_id: str, annotator: str, tiers: dict[str, int]):
    return {
        "annotator_id": annotator,
        "bundle_id": bundle,
        "item_id": item_id,
        "task_type": "A6",
        "response": {"tiers": dict(tiers)},
    }


def test_a6_verdicts_excludes_a_quarantined_raters_votes(tmp_path):
    """THE RATER AXIS TOOK NO QUARANTINE ARGUMENT AT ALL.

    `a7_verdicts(ann, q)` and `a5_verdicts(ann, q)` each test `_is_quarantined` per rater and
    per axis; `a6_verdicts(ann)` took no `q`, so the axis that ORDERS `pairs.rater.jsonl`
    could not be reached by a panel decision expressed as a quarantine entry. MEASURED
    2026-09-17: withdrawing `gpt-oss-120b` moved the reaches ordering (1,426 -> 1,818
    preferences, digest c15251f7ecb6cf5e -> 64e95b7439253f67) and left the rater ordering
    byte-identical -- 4,476 of 4,610 of its rows still carry the withdrawn rater. A parameter
    that does not exist cannot be bypassed and cannot be audited either.

    Three raters, `c0` beating `c1` two votes to one:
      * quarantine a rater in the MAJORITY -> 1-1, no strict majority, the verdict is gone;
      * quarantine the DISSENTER           -> 2-0, the verdict stands and is now unanimous;
      * quarantine on ANOTHER AXIS         -> the a6 verdict is untouched (per-axis, not
        per-rater-globally).
    """
    root = tmp_path / "ann"
    runs = {"c0": "RUN_C0", "c1": "RUN_C1"}
    item, key = _a6_item("i1", run="p", turn=1, runs=runs)
    _bundle(root, "b1", [item], {"i1": key})
    _records(
        root,
        "a6",
        [
            # lower tier == better, so c0 over c1 for the first two raters
            _a6_rec("b1", "i1", "llm:openai/aws/claude-sonnet-5", {"c0": 1, "c1": 2}),
            _a6_rec("b1", "i1", "llm:openai/gcp/gemini-3.1-pro-preview", {"c0": 1, "c1": 2}),
            _a6_rec("b1", "i1", "llm:openai/aws/gpt-oss-120b", {"c0": 2, "c1": 1}),
        ],
    )
    ann = load_root(root)
    ident = ("musique", "t1", "p", 1, frozenset({"RUN_C0", "RUN_C1"}))

    open_panel = a6_verdicts(ann, {})
    assert open_panel[ident]["a6_winner_run"] == "RUN_C0"
    assert open_panel[ident]["a6_n_votes"] == 2
    assert open_panel[ident]["a6_unanimous"] is False
    assert open_panel[ident]["a6_raters"] == [
        "claude-sonnet-5",
        "gemini-3.1-pro",
        "gpt-oss-120b",
    ]

    # A rater inside the winning majority: 2-1 becomes 1-1, which is not a strict majority.
    drop_majority = a6_verdicts(ann, {"b1": {"claude-sonnet-5": ["a6"]}})
    assert ident not in drop_majority, "a quarantined majority vote must not still decide"

    # The dissenter: 2-1 becomes 2-0 and the winner is unchanged.
    drop_dissenter = a6_verdicts(ann, {"b1": {"gpt-oss-120b": ["a6"]}})
    assert drop_dissenter[ident]["a6_winner_run"] == "RUN_C0"
    assert drop_dissenter[ident]["a6_n_votes"] == 2
    assert drop_dissenter[ident]["a6_unanimous"] is True
    assert drop_dissenter[ident]["a6_raters"] == ["claude-sonnet-5", "gemini-3.1-pro"]

    # PER AXIS. A quarantine on `preference` or `reaches` says nothing about `a6`.
    other_axis = a6_verdicts(ann, {"b1": {"gpt-oss-120b": ["preference", "reaches"]}})
    assert other_axis[ident] == open_panel[ident]


def test_a6_rederivation_reproduces_a_preaggregated_campaign(tmp_path):
    """The pairwise expansion `a6_verdicts` runs must agree with the one the campaigns ran.

    `a6_verdicts` derives the a6 axis from the raw TIER records so a quarantine can be applied
    per rater; the campaigns wrote the same tally into `preferences.jsonl` beforehand. Two
    implementations of one derivation drift, and the drift would be invisible -- both produce a
    plausible ordering. This pins the rule: LOWER TIER WINS, EQUAL TIER ABSTAINS, and the
    verdict needs a STRICT majority. sonnet ranks `c1` and `c2` in the SAME TIER, so it does
    not vote on that pair at all and the two raters who do decide it unanimously -- an
    abstention must shrink the panel for that pair, not count as a vote for either side.
    """
    root = tmp_path / "ann"
    runs = {"c0": "R0", "c1": "R1", "c2": "R2"}
    item, key = _a6_item("i1", run="p", turn=1, runs=runs)
    _bundle(root, "b1", [item], {"i1": key})
    _records(
        root,
        "a6",
        [
            _a6_rec("b1", "i1", "llm:x/claude-sonnet-5", {"c0": 1, "c1": 2, "c2": 2}),
            _a6_rec("b1", "i1", "llm:x/gemini-3.1-pro", {"c0": 1, "c1": 3, "c2": 2}),
            _a6_rec("b1", "i1", "llm:x/gpt-oss-120b", {"c0": 2, "c1": 3, "c2": 1}),
        ],
    )
    got = a6_verdicts(load_root(root), {})
    ident = lambda a, b: ("musique", "t1", "p", 1, frozenset({a, b}))  # noqa: E731

    # c0 vs c1: 3-0 for c0.
    assert got[ident("R0", "R1")]["a6_winner_run"] == "R0"
    assert (got[ident("R0", "R1")]["a6_n_votes"], got[ident("R0", "R1")]["a6_unanimous"]) == (
        3,
        True,
    )
    # c0 vs c2: sonnet and gemini rank c0 above c2, oss below it -> 2-1, not unanimous.
    assert got[ident("R0", "R2")]["a6_winner_run"] == "R0"
    assert (got[ident("R0", "R2")]["a6_n_votes"], got[ident("R0", "R2")]["a6_unanimous"]) == (
        2,
        False,
    )
    # c1 vs c2: sonnet puts them in ONE TIER and abstains; the remaining two agree on c2, so
    # the verdict is unanimous OVER THE RATERS WHO VOTED and names only those two.
    assert got[ident("R1", "R2")]["a6_winner_run"] == "R2"
    assert got[ident("R1", "R2")]["a6_n_votes"] == 2
    assert got[ident("R1", "R2")]["a6_unanimous"] is True
    assert got[ident("R1", "R2")]["a6_raters"] == ["gemini-3.1-pro", "gpt-oss-120b"], (
        "the abstaining rater must not appear on a pair it did not vote on"
    )


def test_a6_refuses_a_quarantine_it_cannot_enforce(tmp_path):
    """A parameter that is silently unenforceable is no better than the missing one.

    A root holding only a campaign's pre-aggregated `preferences.jsonl` records `raters`,
    `n_votes` and `unanimous` -- never who voted for what -- so a withdrawn rater cannot be
    subtracted from the tally. Dropping through would count its votes into the ordering while
    the manifest claimed the panel excluded them.
    """
    root = tmp_path / "ann"
    item, key = _a6_item("i1", run="p", turn=1, runs={"c0": "R0", "c1": "R1"})
    _bundle(root, "b1", [item], {"i1": key})
    (root / "a6").mkdir(parents=True, exist_ok=True)
    (root / "a6" / "preferences.jsonl").write_text(
        json.dumps(
            {
                "item_id": "i1",
                "winner": "c0",
                "loser": "c1",
                "winner_run": "R0",
                "loser_run": "R1",
                "n_votes": 2,
                "unanimous": False,
                "raters": ["gemini", "oss", "sonnet"],
            }
        )
        + "\n"
    )
    ann = load_root(root)
    assert len(a6_verdicts(ann, {})) == 1, "with no quarantine the aggregate still works"
    with pytest.raises(QuarantineUnenforceable, match="no raw A6 records"):
        a6_verdicts(ann, {"b1": {"gpt-oss-120b": ["a6"]}})


# --------------------------------------------------------------------------------------------
# THE NON-RECURSIVE GLOB IS A VALIDITY INVARIANT, NOT A STYLE CHOICE.
#
# `load_root` reads a campaign directory with `d.glob("*.jsonl")`. Panel 2 relies on exactly
# that non-recursion: its raw per-order records live under `panel2/raw/` and its own README
# says so -- "raw/ is a SUBDIRECTORY ON PURPOSE: `promote.load_root` does not recurse, so the
# raw records cannot enter a panel as `gpt-5.6-sol`". The A6 half added 2026-09-17 relies on it
# for the same reason.
#
# So an invariant the panel's validity rests on is currently expressed as the ABSENCE of a `**`
# in one glob pattern. Anyone who makes it recursive for a perfectly good reason -- reaching a
# nested campaign, say -- silently promotes every raw per-order record into a voter. Nothing
# raises. The counts simply move.
#
# WHAT SPECIFICALLY MOVES, and why it is worse than a plain duplicate: the orig-order raw
# records carry the ORIGINAL bundle_id (the `a7sub`/original bundle keeps it), so they DO join
# an item and DO vote, as `gpt-5.6-sol` -- one half of the very model whose merged, order-
# consistent record is already voting as `gpt-5.6-sol-mirrored`. `rater_of` keys them
# differently, so nothing dedups them. The panel gains a fourth vote that is correlated with
# one it already has, and the mirrored-order protocol -- whose entire purpose is to drop the
# position-driven half -- is bypassed by the half it was built to discard.
#
# The mirror-order raw records carry `<bid>-mirror`, which is not in `bundles/`, so they do not
# join and drop out silently. That asymmetry is what makes the failure hard to see by eye: half
# the raw records vanish and half become voters.


def _a6_item_3cand(item_id: str):
    """One A6 item, three candidates, with the candidate -> run map the key carries."""
    item = {
        "item_id": item_id,
        "task_type": "A6",
        "payload": {
            "candidates": [
                {"candidate_id": "c0", "question": "who founded it?"},
                {"candidate_id": "c1", "question": "when did it open?"},
                {"candidate_id": "c2", "question": "where is it?"},
            ]
        },
        "provenance": {"suite": "musique", "task_id": "t1", "run_id": "p", "turn_idx": 1},
    }
    key = {"candidate_runs": {"c0": "run-0", "c1": "run-1", "c2": "run-2"}}
    return item, key


def _a6_tier_rec(item_id: str, bundle: str, annotator: str, tiers: dict):
    return {
        "item_id": item_id,
        "bundle_id": bundle,
        "task_type": "A6",
        "annotator_id": annotator,
        "response": {"tiers": tiers},
    }


# The pair under test is (run-0, run-1). Votes are engineered so that the THREE voting raters
# give run-0 a 2-1 majority, and the ONE raw record hidden under raw/ would make it 2-2 -- no
# strict majority, so the verdict does not flip, it DISAPPEARS. A silent loss, not an error.
_TOP_LEVEL = [
    ("llm:openai/aws/claude-sonnet-5", {"c0": 1, "c1": 2, "c2": 3}),  # run-0
    ("llm:openai/gcp/gemini-3.1-pro-preview", {"c0": 2, "c1": 1, "c2": 3}),  # run-1
    ("llm:azure/gpt-5.6-sol-mirrored", {"c0": 1, "c1": 2, "c2": 3}),  # run-0
]
_RAW_HIDDEN = ("llm:azure/gpt-5.6-sol", {"c0": 2, "c1": 1, "c2": 3})  # run-1


@pytest.fixture
def raw_subdir_root(tmp_path):
    """A root shaped exactly like the panel-2 campaigns: merged records at a campaign's top
    level, the raw per-order records one directory deeper."""
    r = tmp_path / "ann"
    it, key = _a6_item_3cand("IT1")
    _bundle(r, "b1", [it], {"IT1": key})
    camp = r / "panel2a6"
    camp.mkdir(parents=True, exist_ok=True)
    for ann_id, tiers in _TOP_LEVEL:
        (camp / f"{rater_of(ann_id)}.jsonl").write_text(
            json.dumps(_a6_tier_rec("IT1", "b1", ann_id, tiers))
        )
    raw = camp / "raw"
    raw.mkdir()
    # Carries the ORIGINAL bundle_id, so it would join the item and vote if it were read.
    (raw / "r_sol_a6orig.b1.jsonl").write_text(
        json.dumps(_a6_tier_rec("IT1", "b1", _RAW_HIDDEN[0], _RAW_HIDDEN[1]))
    )
    return r


def _assert_raw_subdir_never_votes(root: Path) -> None:
    """The invariant, as assertions. Called twice on purpose: once against the real loader, and
    once under a deliberately recursive glob, where every one of these must fail."""
    ann = load_root(str(root))
    a6 = [x for x in ann.records if x.get("task_type") == "A6"]
    raters = sorted(rater_of(x["annotator_id"]) for x in a6)
    assert raters == ["claude-sonnet-5", "gemini-3.1-pro", "gpt-5.6-sol-mirrored"], raters
    assert "gpt-5.6-sol" not in raters, "a raw per-order record became a voter"
    v = a6_verdicts(ann, {})
    ident = ("musique", "t1", "p", 1, frozenset({"run-0", "run-1"}))
    assert ident in v, "the verdict the three voting raters carry has gone missing"
    assert v[ident]["a6_winner_run"] == "run-0"
    assert v[ident]["a6_raters"] == [
        "claude-sonnet-5",
        "gemini-3.1-pro",
        "gpt-5.6-sol-mirrored",
    ]
    assert ann.n_retest_duplicates == 0, "the raw record was read and then deduped away, which"
    " would hide the read rather than prevent it"


def test_load_root_does_not_recurse_into_a_raw_subdirectory(raw_subdir_root):
    """The guard. A record one directory below a campaign does not vote."""
    _assert_raw_subdir_never_votes(raw_subdir_root)


def test_the_raw_subdir_guard_fails_if_the_glob_is_made_recursive(raw_subdir_root, monkeypatch):
    """The teeth, so the guard's pass is not a zero that would look the same either way.

    Reading the same root with a recursive glob -- the one-character change a future edit makes
    for a good reason -- must break the guard above. If it does not, the guard is asserting
    something that is true regardless of the glob and protects nothing.
    """
    real_glob = Path.glob

    def recursive_glob(self, pattern):
        # Exactly the plausible edit: reach nested directories too.
        return (
            real_glob(self, "**/" + pattern) if pattern == "*.jsonl" else real_glob(self, pattern)
        )

    monkeypatch.setattr(Path, "glob", recursive_glob)

    # 1. The guard goes red.
    with pytest.raises(AssertionError):
        _assert_raw_subdir_never_votes(raw_subdir_root)

    # 2. And the specific damage, named: the raw record votes as `gpt-5.6-sol`, alongside the
    #    merged `gpt-5.6-sol-mirrored` built from that same model -- a fourth, correlated vote.
    ann = load_root(str(raw_subdir_root))
    raters = sorted(rater_of(x["annotator_id"]) for x in ann.records if x["task_type"] == "A6")
    assert raters == [
        "claude-sonnet-5",
        "gemini-3.1-pro",
        "gpt-5.6-sol",
        "gpt-5.6-sol-mirrored",
    ], raters

    # 3. And the count MOVES with nothing raising: 2-1 for run-0 becomes 2-2, no strict
    #    majority, so the verdict is gone rather than wrong.
    v = a6_verdicts(ann, {})
    ident = ("musique", "t1", "p", 1, frozenset({"run-0", "run-1"}))
    assert ident not in v, (
        "expected the verdict to VANISH under a recursive glob; if it survived, this test no "
        "longer demonstrates the silent-count-move failure it was written for"
    )


# --------------------------------------------------------------------------------------------
# A6 SLOT BIAS IN THE MANIFEST: RECORD ONLY, NEVER A GATE, NEVER A ROW.
#
# A7's slot-bias reading rides into `a7_campaign.slot_bias` via `check_quarantine`, which also
# COMPARES it against the quarantine file and can raise `QuarantineDisagreement`. A6 gets the
# analogous reading in `a6_campaign.slot_bias`, deliberately WITHOUT that comparison: A6
# quarantine is not applied here because doing so would move majorities the paper quotes. The
# two tests below check the two halves of that separately -- the numbers are wired through
# correctly, and wiring them through cannot touch the written file.


def test_a6_slot_bias_report_lands_in_the_manifest_per_rater(tmp_path):
    """`promote`'s `a6_campaign` block now carries a per-rater slot-bias reading, the A6
    analogue of `a7_campaign`'s. Field names are the manifest-facing ones, not `_slot_bias`'s
    internal ones: `p_first_top` is its `p_first`, `expected_tie_aware` is its `expected` (the
    tie-aware null is the default there), `p` is its `binomial_p`, `p_naive` passes through
    unchanged. Values are checked against calling `_slot_bias` directly on the same bundle and
    records, not hand-derived.
    """
    from pi_run.cmd_annotate import _slot_bias

    root = tmp_path / "ann"
    it, key = _a6_item_3cand("i1")
    _bundle(root, "b1", [it], {"i1": key})
    recs = [
        _a6_tier_rec("i1", "b1", "llm:openai/aws/claude-sonnet-5", {"c0": 1, "c1": 2, "c2": 2}),
        _a6_tier_rec(
            "i1", "b1", "llm:openai/gcp/gemini-3.1-pro-preview", {"c0": 2, "c1": 1, "c2": 1}
        ),
    ]
    _records(root, "a6", recs)
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text(json.dumps(_pair_row("cand-a", "cand-b")) + "\n")
    q = tmp_path / "q.json"
    q.write_text(json.dumps({"b1": {}}))

    blocks = promote(pairs, root, q)
    sb = blocks["a6_campaign"]["slot_bias"]
    assert set(sb) == {"b1"}
    assert set(sb["b1"]) == {"claude-sonnet-5", "gemini-3.1-pro"}

    items_for_slot_bias = [{"item_id": "i1", "task_type": "A6", "payload": it["payload"]}]
    for rater in ("claude-sonnet-5", "gemini-3.1-pro"):
        rater_recs = [r for r in recs if rater_of(r["annotator_id"]) == rater]
        want = _slot_bias({"items": items_for_slot_bias}, rater_recs, "A6")["top_tier"]
        assert sb["b1"][rater] == {
            "n": want["n"],
            "p_first_top": want["p_first"],
            "expected_tie_aware": want["expected"],
            "p": want["binomial_p"],
            "p_naive": want["p_naive"],
        }
        assert sb["b1"][rater]["n"] == 1, "sanity: real data reached the statistic, not {}"


def test_a6_slot_bias_report_does_not_change_the_promoted_pairs_file(tmp_path):
    """RECORD ONLY: `a6_campaign.slot_bias` must not move a majority, an agreement flag, or a
    byte of the written pairs file. Proven structurally, not by a before/after snapshot -- the
    row `attach` produces from `a7`/`a6`/`a5` verdicts alone, with no slot-bias reporting
    anywhere near it, must be byte-identical to what `promote` actually wrote.
    """
    root = tmp_path / "ann"
    it, key = _a6_item_3cand("i1")
    _bundle(root, "b1", [it], {"i1": key})
    recs = [
        _a6_tier_rec("i1", "b1", "llm:openai/aws/claude-sonnet-5", {"c0": 1, "c1": 2, "c2": 2}),
        _a6_tier_rec(
            "i1", "b1", "llm:openai/gcp/gemini-3.1-pro-preview", {"c0": 2, "c1": 1, "c2": 1}
        ),
    ]
    _records(root, "a6", recs)
    pairs = tmp_path / "pairs.jsonl"
    row = _pair_row("cand-a", "cand-b")
    pairs.write_text(json.dumps(row) + "\n")
    q = tmp_path / "q.json"
    q.write_text(json.dumps({"b1": {}}))

    promote(pairs, root, q)
    written = pairs.read_bytes()

    ann = load_root(root)
    a7 = a7_verdicts(ann, {})
    a6 = a6_verdicts(ann, {})
    a5 = a5_verdicts(ann, {})
    new, _counts = attach([row], a7, a6, a5)
    expected = "".join(json.dumps(r, sort_keys=True) + "\n" for r in new).encode()
    assert written == expected, "a6_slot_bias_report must have zero influence on the written row"
