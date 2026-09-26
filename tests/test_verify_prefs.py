"""`pi train verify-prefs`: does the preference file on disk still order the shipped pairs?

WHAT `rater_prefs_sha` IS. A 16-hex digest over the preference SET -- `{frozenset(winner, loser):
winner}` -- written onto a `rank_rule="rater"` manifest. Over the SET and not over the file, so a
renamed or re-sorted file still identifies the same ordering. `pairs.rater.jsonl` and
`pairs.reaches.jsonl` are both `rank_rule="rater"` and are DIFFERENT experiments (A6 candidate
rankings against A7's `reaches` axis); the digest is the only thing on the artifact that tells
them apart after the fact.

WHAT THIS COMMAND IS FOR. The digest proves an ordering came from a named preference set only
while somebody checks. `data/rl/a6_preferences.jsonl` has been rewritten by promotion campaigns
since the export ran -- one of which dropped 2,188 of 2,429 preferences -- and nothing in the
tree would have said so. This recomputes the digest from the file as it now stands and compares.

ONE IMPLEMENTATION. The digest is `dataset.rater_prefs_sha`, called by `export_pairs` and by this
command. Two implementations of one digest drift, and a verifier that disagrees with the exporter
it verifies is worse than no verifier: it fails on correct files and teaches people to ignore it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pinq.actions import ask_action_json
from pinq_train.export.dataset import export_pairs, rater_prefs_sha


def _cand(run_id: str, value: float, question: str) -> dict:
    return {
        "suite_id": "musique",
        "task_id": "t1",
        "template_id": None,
        "run_id": run_id,
        "turn_idx": 1,
        "state_text": "S",
        "action_json": ask_action_json(question),
        "value": value,
        "phi_tilde": value,
        "scorer_hash": "sh",
        "graph_version": "v1",
        "branch_of_run_id": "p",
        "branch_turn_idx": 1,
        "pins_sha": "pins-A",
        "budget_cap": 24,
        "max_turns": 24,
        "is_stop": False,
        "done_before": False,
        "coverage_before": 0.4,
    }


ROWS = [_cand("a", 0.50, "who founded it?"), _cand("b", 0.49, "when did it open?")]
PREFS = {frozenset(("a", "b")): "b"}


def _write_prefs(path: Path, prefs: dict) -> Path:
    lines = []
    for key, winner in prefs.items():
        loser = next(iter(key - {winner}))
        lines.append(json.dumps({"winner_run": winner, "loser_run": loser}))
    path.write_text("\n".join(lines) + "\n")
    return path


def _verify(tmp_path: Path, manifest: dict, prefs: dict) -> int:
    from pi_run.cli import build_parser

    m = tmp_path / "pairs.rater.manifest.json"
    m.write_text(json.dumps(manifest))
    p = _write_prefs(tmp_path / "prefs.jsonl", prefs)
    args = build_parser().parse_args(
        ["train", "verify-prefs", "--pairs-manifest", str(m), "--prefs", str(p)]
    )
    return args.fn(args)


def test_the_digest_is_the_exporters_own_function():
    """The verifier and the exporter must be the SAME function, not two that agree today."""
    _pairs, man = export_pairs(ROWS, margin_threshold=0.05, rank="rater", rater_prefs=PREFS)

    assert man.rater_prefs_sha == rater_prefs_sha(PREFS)
    assert len(man.rater_prefs_sha) == 16


def test_the_digest_is_over_the_set_not_over_the_file():
    """A renamed, re-sorted or re-serialised file identifies the same ordering. That is the
    property that makes the digest a statement about the EXPERIMENT rather than about bytes."""
    prefs = {frozenset(("a", "b")): "b", frozenset(("c", "d")): "c"}
    reordered = dict(reversed(list(prefs.items())))

    assert rater_prefs_sha(prefs) == rater_prefs_sha(reordered)
    assert rater_prefs_sha(prefs) != rater_prefs_sha({frozenset(("a", "b")): "a"})


def test_verify_prefs_exits_zero_when_the_file_is_the_one_the_export_used(tmp_path, capsys):
    _pairs, man = export_pairs(ROWS, margin_threshold=0.05, rank="rater", rater_prefs=PREFS)

    assert _verify(tmp_path, {"rater_prefs_sha": man.rater_prefs_sha}, PREFS) == 0
    assert "MATCH" in capsys.readouterr().out


def test_verify_prefs_exits_two_when_a_single_verdict_flipped(tmp_path, capsys):
    """One flipped verdict is the case that matters: a file that LOST rows is visible in a line
    count, and a file whose ordering changed is visible in nothing else at all."""
    _pairs, man = export_pairs(ROWS, margin_threshold=0.05, rank="rater", rater_prefs=PREFS)

    rc = _verify(tmp_path, {"rater_prefs_sha": man.rater_prefs_sha}, {frozenset(("a", "b")): "a"})

    assert rc == 2
    assert "MISMATCH" in capsys.readouterr().out


def test_verify_prefs_exits_two_when_a_verdict_is_missing(tmp_path):
    prefs = {frozenset(("a", "b")): "b", frozenset(("c", "d")): "c"}
    assert _verify(tmp_path, {"rater_prefs_sha": rater_prefs_sha(prefs)}, PREFS) == 2


def test_verify_prefs_refuses_a_manifest_that_names_no_preference_set(tmp_path, capsys):
    """A control export ordered nothing this way, so its `rater_prefs_sha` is "". Comparing
    against an empty digest would let every file "verify" against the control and report a
    match -- the failure mode a verifier exists to prevent."""
    rc = _verify(tmp_path, {"rank_rule": "outcome", "rater_prefs_sha": ""}, PREFS)

    assert rc == 2
    assert "names no preference set" in capsys.readouterr().out


def test_verify_prefs_reports_both_digests(tmp_path, capsys):
    """Rule 3: the command must print the measurement, not a verdict alone. A mismatch whose two
    sides are not shown cannot be acted on."""
    _pairs, man = export_pairs(ROWS, margin_threshold=0.05, rank="rater", rater_prefs=PREFS)
    _verify(tmp_path, {"rater_prefs_sha": man.rater_prefs_sha}, {frozenset(("a", "b")): "a"})

    out = capsys.readouterr().out
    assert man.rater_prefs_sha in out
    assert rater_prefs_sha({frozenset(("a", "b")): "a"}) in out


@pytest.mark.parametrize("n", [1, 3])
def test_verify_prefs_counts_the_verdicts_it_read(tmp_path, capsys, n):
    """A digest computed over an empty dict is still a digest. Printing the count is what
    separates "verified" from "read nothing and hashed the void"."""
    prefs = {frozenset((f"w{i}", f"l{i}")): f"w{i}" for i in range(n)}
    _verify(tmp_path, {"rater_prefs_sha": rater_prefs_sha(prefs)}, prefs)

    assert f"{n} verdict" in capsys.readouterr().out
