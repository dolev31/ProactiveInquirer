"""`_load_provenance` must accept the arm model ids as arguments, not only its own constants.

WHY THIS TEST EXISTS. `scripts/stopping_answer_test/run.py` pins `MODEL_TRAINED` to
`qwen3-8b-dpo-stacked-notdone-both` and exits if a provenance file names anything else. Lane L6.1
serves `qwen3-8b-sft-headline-answernode` against `-refresh`, so the answer-node coverage endpoint
-- the one L1.11's diagnosis actually implicates -- could not be computed on those arms at all. A
reader that can only read one checkpoint pair is a reader that silently scopes a finding to that
pair.

The default must not move: every existing caller passes no models and must keep the old behaviour,
including the refusal when a file names an unexpected model.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "stopping_answer_test"))

import run as sat  # noqa: E402


def _write(d: Path, arm: str, model: str) -> None:
    (d / f"provenance.{arm}.json").write_text(
        json.dumps(
            {
                "arm": arm,
                "inquirer_model_id": model,
                "scorer_hash": "e82c7458",
                "graph_version": "v1",
                "model_pin_hash": "deadbeefcafe",
                "code_version": ["4280cdf7"],
            }
        )
    )


def test_overriding_the_models_lets_a_different_pair_be_read(tmp_path: Path) -> None:
    _write(tmp_path, "trained", "qwen3-8b-sft-headline-answernode")
    _write(tmp_path, "prompted", "qwen3-8b-sft-headline-refresh")
    prov = sat._load_provenance(
        tmp_path,
        model_trained="qwen3-8b-sft-headline-answernode",
        model_prompted="qwen3-8b-sft-headline-refresh",
    )
    assert prov["scorer_hash"] == "e82c7458"
    assert prov["graph_version"] == "v1"


def test_the_default_still_refuses_an_unexpected_model(tmp_path: Path) -> None:
    _write(tmp_path, "trained", "qwen3-8b-sft-headline-answernode")
    _write(tmp_path, "prompted", "qwen3-8b-base")
    with pytest.raises(SystemExit):
        sat._load_provenance(tmp_path)


def test_a_mismatch_against_an_explicit_override_still_refuses(tmp_path: Path) -> None:
    _write(tmp_path, "trained", "qwen3-8b-sft-headline-answernode")
    _write(tmp_path, "prompted", "qwen3-8b-base")
    with pytest.raises(SystemExit):
        sat._load_provenance(
            tmp_path, model_trained="something-else", model_prompted="qwen3-8b-base"
        )
