"""A TRAINING ROW WITHOUT PROVENANCE IS NOT A TRAINING ROW.

CONTRIBUTING.md rule 1: every reported value traces to a `run_id`, a `scorer_hash` and a
`graph_version`. `rows_from_run` has always PRODUCED those keys -- and `Example` never
declared them, so `write_jsonl`'s `asdict()` dropped every one at write time. Measured on the
export that was on disk: `data/rl/sft.jsonl` carried 9 keys, none of them `scorer_hash`,
`graph_version` or `phi_tilde`, so a trained checkpoint could not name the instrument that
scored its own training data.

This is the same defect the `Example` docstring already warns about for the latent fields.
It is tested here rather than there because the failure is structural: any future field added
to the row dict and not to the dataclass disappears silently, and only a round-trip test
notices.
"""

from __future__ import annotations

import json
from dataclasses import fields

import pytest

from pinq_train.export.dataset import Example, PreferencePair, export_pairs, export_sft, write_jsonl

# Everything a downstream trainer or auditor needs and cannot reconstruct from the row alone.
REQUIRED_SFT_FIELDS = {
    "suite_id",
    "task_id",
    "run_id",
    "turn_idx",  # identity
    "scorer_hash",
    "graph_version",
    "matcher_id",  # CONTRIBUTING.md rule 1
    "reward_weights_sha",  # which reward ranked this row
    "split",
    "template_id",  # contamination auditing
    "phi_tilde",
    "rho",
    "value",  # reward, decomposed
    "branch_of_run_id",
    "branch_turn_idx",  # which parent this forked from
    "latent_depth",
    "is_latent",
    "newly_reachable",
    "frontier_size",
    "evidence_coverage",
    "answer_correct",
    "answer_hedged",  # episode outcome
    "turns_to_complete",  # the quickest-path signal
}

REQUIRED_PAIR_FIELDS = REQUIRED_SFT_FIELDS - {"value", "phi_tilde", "rho"} | {
    "margin",
    "chosen_run_id",
    "rejected_run_id",
    "pair_id",
}


def test_example_declares_every_required_field() -> None:
    have = {f.name for f in fields(Example)}
    missing = REQUIRED_SFT_FIELDS - have
    assert not missing, f"Example drops these at write time: {sorted(missing)}"


def test_preference_pair_declares_every_required_field() -> None:
    have = {f.name for f in fields(PreferencePair)}
    missing = REQUIRED_PAIR_FIELDS - have
    assert not missing, f"PreferencePair drops these at write time: {sorted(missing)}"


def _row(**over):
    r = {
        "suite_id": "musique",
        "task_id": "2hop__1_2",
        "run_id": "r0",
        "turn_idx": 0,
        "template_id": "2hop__1_2",
        "state_text": "S",
        "action_json": '{"action":"ASK"}',
        "value": 0.9,
        "phi_tilde": 0.9,
        "rho": 0.1,
        "scorer_hash": "sh",
        "graph_version": "v1",
        "matcher_id": "mechanical_v3",
        "reward_weights_sha": "rw",
        "split": "train",
        "branch_of_run_id": None,
        "branch_turn_idx": None,
        "latent_depth": 1,
        "is_latent": True,
        "newly_reachable": True,
        "frontier_size": 2,
        "evidence_coverage": 1.0,
        "answer_correct": 1.0,
        "answer_hedged": 0.0,
        "turns_to_complete": 1,
        "parent_uids": [],
    }
    r.update(over)
    return r


def test_provenance_survives_the_round_trip_to_disk(tmp_path) -> None:
    """The actual failure mode: declared on the row, dropped by asdict()."""
    ex, man = export_sft([_row()], margin_threshold=0.0)
    assert ex, "fixture row was refused"
    p = write_jsonl(tmp_path / "sft.jsonl", ex, man)
    on_disk = json.loads(p.read_text().splitlines()[0])
    missing = REQUIRED_SFT_FIELDS - set(on_disk)
    assert not missing, f"written row is missing {sorted(missing)}"
    assert on_disk["scorer_hash"] == "sh"
    assert on_disk["graph_version"] == "v1"
    assert on_disk["matcher_id"] == "mechanical_v3"


def test_pair_provenance_survives_the_round_trip_to_disk(tmp_path) -> None:
    rows = [
        _row(run_id="a", branch_of_run_id="p", branch_turn_idx=0, value=0.9, phi_tilde=0.9),
        _row(
            run_id="b",
            branch_of_run_id="p",
            branch_turn_idx=0,
            value=0.1,
            phi_tilde=0.1,
            action_json='{"action":"ASK","text":"other"}',
        ),
    ]
    pairs, man = export_pairs(rows, margin_threshold=0.0, len_delta_max=1000)
    assert pairs, "no pair produced from two same-state candidates"
    p = write_jsonl(tmp_path / "pairs.jsonl", pairs, man)
    on_disk = json.loads(p.read_text().splitlines()[0])
    missing = REQUIRED_PAIR_FIELDS - set(on_disk)
    assert not missing, f"written pair is missing {sorted(missing)}"


def test_a_row_missing_provenance_is_refused_not_silently_defaulted() -> None:
    """A default `scorer_hash=""` would be worse than the bug: it looks like provenance."""
    with pytest.raises((KeyError, ValueError)):
        export_sft([_row(scorer_hash=None)], margin_threshold=0.0)


def test_evidence_coverage_is_populated_not_left_nan() -> None:
    """It shipped as NaN on the first export: `_outcome_fields` initialised the key and
    nothing ever set it. A NaN here is indistinguishable from 'this suite has no gold', so
    a consumer filtering on complete evidence would silently drop every row."""
    import math

    ex, man = export_sft([_row(evidence_coverage=1.0)], margin_threshold=0.0)
    assert not math.isnan(ex[0].evidence_coverage)
    assert ex[0].evidence_coverage == 1.0
