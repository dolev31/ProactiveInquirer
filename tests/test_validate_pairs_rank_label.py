"""The validation script must not call a rule-confirmation a measurement.

`scripts/validate_pairs.py` prints "of discordant pairs, the chosen side pursued the latent
need N% of the time" and its docstring called that GENUINE, on the strength of a sentence
asserting that `_prefers` contains no reference to a latent-need field. That sentence was a
claim about code the script cannot see, and `--rank anticipation` made it false: under that
ordering `newly_reachable` IS the key, and the headline is true by construction.

So the label is now derived from `ExportManifest.rank_rule` in the manifest beside the file.
These tests pin the three cases, because the difference between them is the difference
between a finding and a circularity.
"""

from __future__ import annotations

import json

from scripts.validate_pairs import main, rank_rule_of


def _export(tmp_path, name, manifest):
    p = tmp_path / name
    p.write_text(
        json.dumps(
            {
                "task_id": "t",
                "run_id": "r",
                "turn_idx": 1,
                "is_latent": True,
                "rejected_is_latent": False,
                "latent_depth": 1,
                "rejected_latent_depth": 0,
                "turns_to_complete": 1,
                "rejected_turns_to_complete": 2,
                "answer_correct": 1.0,
                "rejected_answer_correct": 0.0,
            }
        )
    )
    if manifest is not None:
        (tmp_path / name.replace(".jsonl", ".manifest.json")).write_text(json.dumps(manifest))
    return p


def test_the_control_export_is_labelled_genuine(tmp_path, capsys):
    p = _export(tmp_path, "pairs.jsonl", {"rank_rule": "outcome"})
    assert rank_rule_of(p) == "outcome"
    assert main(str(p)) == 0
    assert "genuine" in capsys.readouterr().out


def test_the_anticipation_export_is_labelled_tautological(tmp_path, capsys):
    p = _export(tmp_path, "pairs.anticipation.jsonl", {"rank_rule": "anticipation"})
    assert rank_rule_of(p) == "anticipation"
    main(str(p))
    out = capsys.readouterr().out
    assert "TAUTOLOGICAL" in out and "NOT EVIDENCE" in out


def test_a_manifest_without_the_field_predates_the_flag_and_is_the_control(tmp_path):
    """There was one ordering before `--rank`. Reading that off the writer is a fact, not an
    assumption -- and calling every pre-flag export "unknown" would make the label useless on
    exactly the artifacts already on disk."""
    assert rank_rule_of(_export(tmp_path, "pairs.jsonl", {"n_pairs": 3})) == "outcome"


def test_no_manifest_at_all_is_unknown(tmp_path, capsys):
    """A bare jsonl could have been filtered, concatenated or re-ranked by anything."""
    p = _export(tmp_path, "pairs.jsonl", None)
    assert rank_rule_of(p) == "unknown"
    main(str(p))
    assert "rank_rule UNKNOWN" in capsys.readouterr().out


def test_ask_stop_pairs_are_excluded_from_the_latent_check(tmp_path, capsys):
    """A STOP retrieves nothing, so it is `is_latent=False` by construction and every ask_stop
    pair would read as a discordant win for whichever side happened to be the question."""
    p = tmp_path / "pairs.jsonl"
    row = {
        "task_id": "t",
        "run_id": "r",
        "turn_idx": 1,
        "is_latent": True,
        "rejected_is_latent": False,
        "latent_depth": 1,
        "rejected_latent_depth": 0,
        "turns_to_complete": 1,
        "rejected_turns_to_complete": 2,
        "answer_correct": 1.0,
        "rejected_answer_correct": 0.0,
        "pair_kind": "ask_stop",
    }
    p.write_text(json.dumps(row) + "\n" + json.dumps({**row, "pair_kind": "ask_ask"}))
    (tmp_path / "pairs.manifest.json").write_text(json.dumps({"rank_rule": "outcome"}))
    main(str(p))
    out = capsys.readouterr().out
    assert "1 ask_stop pairs excluded" in out
    assert "discordant 1:" in out, "only the ask_ask pair may reach the check"
