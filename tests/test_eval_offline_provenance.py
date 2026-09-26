"""Tier A gains a provenance block, and gains no new number.

TWO THINGS ARE BEING PROTECTED HERE AND THEY PULL IN OPPOSITE DIRECTIONS.

The first is that `pinq_train.eval_offline` now writes what CLAUDE.md rule 1 asks for: a dataset
identity, the scorer hashes and graph versions its rows were scored under, the adapter digest the
checkpoint registry knows the weights by, the base model, the code version, the STOP bytes the
2x2 argmaxes against, and the exact command line. Seven \\NOTREADY markers in the paper were
waiting on exactly that.

The second is that adding it moved no metric. `test_metrics_are_byte_identical_to_the_pre_change_code`
holds the three metric dicts computed by the code as it stood BEFORE this block was written,
captured on this fixture and pasted in. It is a golden test in the strict sense: the expected
values were not computed by the code under test. If a future edit to this module perturbs
`sft_nll`, `stop_confusion` or `pair_accuracy` by one float, the published offline cells stop
matching the files on disk and this is what says so.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from pinq.actions import STOP_ACTION_JSON, ask_action_json
from pinq_train.eval_offline import (
    PROVENANCE_SCHEMA,
    WEIGHTS_FILE,
    AdapterIdentityMismatch,
    Rendered,
    checkpoint_provenance,
    dataset_provenance,
    pair_accuracy,
    provenance,
    sft_nll,
    sha256_file,
    stop_confusion,
    weights_sha256,
)

REPO = Path(__file__).resolve().parents[1]
REF_ASK = ask_action_json("What else do I need to know?", "remaining need")


# --------------------------------------------------------------------------- the fixture
#
# A SCRIPTED MODEL, not a tokenizer and not weights. `render` and `logprob` are the only way the
# model enters any of the three functions (the module docstring's own design), so the right
# answer here is known by construction and the test needs no GPU, no download and no torch.


def render(state: str, action: str) -> Rendered:
    return Rendered(
        prompt_ids=tuple(ord(c) % 97 for c in state),
        completion_ids=tuple(ord(c) % 89 for c in action),
    )


def logprob(prompt_ids, completion_ids) -> float:
    return -(sum(completion_ids) % 1000) / 100.0 - 0.5 * len(completion_ids)


def _sft_rows() -> list[dict]:
    return [
        {
            "state_text": "s0",
            "action_json": ask_action_json("who wrote it?", "author"),
            "done_before": False,
            "split": "dev",
        },
        {"state_text": "s1", "action_json": STOP_ACTION_JSON, "done_before": True, "split": "dev"},
        {
            "state_text": "s2",
            "action_json": ask_action_json("when?", "date"),
            "done_before": True,
            "split": "dev",
        },
        {
            "state_text": "s3",
            "action_json": ask_action_json("where was the second author born?", "birthplace"),
            "done_before": False,
            "split": "dev",
        },
        {
            "state_text": "s4",
            "action_json": '{"action": "GARBAGE"}',
            "done_before": False,
            "split": "dev",
        },
        {
            "state_text": "s5",
            "action_json": ask_action_json("why?", "cause"),
            "done_before": None,
            "split": "dev",
        },
    ]


def _pair_rows() -> list[dict]:
    return [
        {
            "state_text": "s0",
            "chosen_json": ask_action_json("who wrote it?", "a"),
            "rejected_json": ask_action_json("who?", "b"),
            "pair_kind": "ask_ask",
            "decided_by": "outcome",
            "split": "dev",
        },
        {
            "state_text": "s1",
            "chosen_json": STOP_ACTION_JSON,
            "rejected_json": ask_action_json("more?", "c"),
            "pair_kind": "ask_stop_synth",
            "decided_by": "stop_done",
            "split": "dev",
        },
        {
            "state_text": "s2",
            "chosen_json": ask_action_json("when?", "d"),
            "rejected_json": ask_action_json("when exactly, to the day?", "e"),
            "pair_kind": "ask_ask",
            "decided_by": "gain",
            "split": "dev",
        },
        {
            "state_text": "s3",
            "chosen_json": ask_action_json("where?", "f"),
            "rejected_json": STOP_ACTION_JSON,
            "pair_kind": "ask_stop",
            "decided_by": "quickest",
            "split": "dev",
        },
        {
            "state_text": "s4",
            "chosen_json": ask_action_json("abc", "g"),
            "rejected_json": ask_action_json("xyz", "g"),
            "pair_kind": "ask_ask",
            "decided_by": "outcome",
            "split": "dev",
        },
    ]


# Captured 2026-09-18 from the module as it stood at commit 3ecff80, i.e. BEFORE the provenance
# block was added, by running the three functions over the fixture above. Pasted, not computed.
#
# THE STOP-CONFUSION ENTRIES WERE EXTENDED 2026-09-19 and no value in them was changed. Lane
# L6.1 made the 2x2's condition field selectable (`stop_confusion(condition_fields=...)`),
# because it read `row.get("done_before")` and nothing else and so every such number this
# programme has produced is keyed on POOLED completeness whatever the arm was called. That is an
# ADDITIVE change: two new keys, `condition_field` and `by_condition`, and every pre-existing key
# holds the byte-identical value it held at 3ecff80.
#
# The golden is updated rather than the assertion relaxed. A subset comparison would pass just
# as happily if a key were DROPPED, which is most of what this test is for. `LEGACY_STOP_KEYS`
# below keeps the pre-2026-09-19 numbers as their own literal, so the claim "nothing moved, two
# keys were added" is checked rather than asserted in this comment.
GOLDEN = {
    "sft_nll": {
        "n": 6.0,
        "nll_per_example": 32.688333333333325,
        "nll_per_token": 0.6148275862068965,
    },
    "stop_confusion": {
        "by_condition": {
            "done_before": {
                "n_done": 2.0,
                "n_not_done": 2.0,
                "n_skipped_no_label": 1.0,
                "n_skipped_no_reference": 0.0,
                "n_skipped_unparseable": 1.0,
                "p_ask_given_not_done": 1.0,
                "p_stop_given_done": 0.0,
            }
        },
        "condition_field": "done_before",
        "n_done": 2.0,
        "n_not_done": 2.0,
        "n_skipped_no_label": 1.0,
        "n_skipped_no_reference": 0.0,
        "n_skipped_unparseable": 1.0,
        "p_ask_given_not_done": 1.0,
        "p_stop_given_done": 0.0,
    },
    "stop_confusion_no_ref": {
        "by_condition": {
            "done_before": {
                "n_done": 1.0,
                "n_not_done": 2.0,
                "n_skipped_no_label": 1.0,
                "n_skipped_no_reference": 1.0,
                "n_skipped_unparseable": 1.0,
                "p_ask_given_not_done": 1.0,
                "p_stop_given_done": 0.0,
            }
        },
        "condition_field": "done_before",
        "n_done": 1.0,
        "n_not_done": 2.0,
        "n_skipped_no_label": 1.0,
        "n_skipped_no_reference": 1.0,
        "n_skipped_unparseable": 1.0,
        "p_ask_given_not_done": 1.0,
        "p_stop_given_done": 0.0,
    },
    "pair_accuracy": {
        "acc": 0.6,
        "acc_by_decided_by": {"gain": 1.0, "outcome": 0.5, "quickest": 0.0, "stop_done": 1.0},
        "acc_by_kind": {
            "ask_ask": 0.6666666666666666,
            "ask_stop": 0.0,
            "ask_stop_synth": 1.0,
        },
        "acc_by_len_sign": {
            "chosen_longer": {"acc": 0.0, "n": 1},
            "chosen_shorter": {"acc": 1.0, "n": 1},
            "equal": {"acc": 1.0, "n": 1},
        },
        "len_sign_gap": -1.0,
        "mean_margin": -0.4099999999999994,
        "n": 5,
    },
}


# The stop-2x2 numbers EXACTLY as they stood at 3ecff80, before `condition_fields` existed.
# Kept as their own literal so that "the 2026-09-19 change added two keys and moved nothing" is
# a checked claim rather than a sentence in a comment above a refreshed golden -- a golden that
# is simply re-captured after every change records what the code does, not what it should do.
LEGACY_STOP_KEYS = {
    "stop_confusion": {
        "n_done": 2.0,
        "n_not_done": 2.0,
        "n_skipped_no_label": 1.0,
        "n_skipped_no_reference": 0.0,
        "n_skipped_unparseable": 1.0,
        "p_ask_given_not_done": 1.0,
        "p_stop_given_done": 0.0,
    },
    "stop_confusion_no_ref": {
        "n_done": 1.0,
        "n_not_done": 2.0,
        "n_skipped_no_label": 1.0,
        "n_skipped_no_reference": 1.0,
        "n_skipped_unparseable": 1.0,
        "p_ask_given_not_done": 1.0,
        "p_stop_given_done": 0.0,
    },
}


def test_the_condition_field_change_added_keys_and_moved_no_number() -> None:
    """Every key the 2x2 carried before `condition_fields` existed still holds its old value.

    This is the justification for updating the golden above, made executable. Without it, the
    refresh is indistinguishable from a metric having silently moved and someone re-capturing
    the output -- which is exactly the failure the golden exists to catch.
    """
    got = {
        "stop_confusion": stop_confusion(
            _sft_rows(), render=render, logprob=logprob, reference_ask=REF_ASK
        ),
        "stop_confusion_no_ref": stop_confusion(_sft_rows(), render=render, logprob=logprob),
    }
    for name, legacy in LEGACY_STOP_KEYS.items():
        for key, value in legacy.items():
            assert got[name][key] == value, f"{name}.{key} moved: {got[name][key]} != {value}"
        added = set(got[name]) - set(legacy)
        assert added == {"condition_field", "by_condition"}, f"{name} gained {added}"


def test_metrics_are_byte_identical_to_the_pre_change_code() -> None:
    """The golden test. Expected values come from the old code, not from this one."""
    got = {
        "sft_nll": sft_nll(_sft_rows(), render=render, logprob=logprob),
        "stop_confusion": stop_confusion(
            _sft_rows(), render=render, logprob=logprob, reference_ask=REF_ASK
        ),
        "stop_confusion_no_ref": stop_confusion(_sft_rows(), render=render, logprob=logprob),
        "pair_accuracy": pair_accuracy(_pair_rows(), render=render, logprob=logprob),
    }
    # json.dumps, so the assertion is on the BYTES a verdict file would carry and not on a
    # float comparison that tolerates a difference the published table would not.
    assert json.dumps(got, sort_keys=True) == json.dumps(GOLDEN, sort_keys=True)


def test_the_golden_fixture_can_detect_a_changed_metric() -> None:
    """A golden test that cannot fail is indistinguishable from no test.

    Forces a known change -- one extra row -- and asserts the golden comparison rejects it. This
    is the non-vacuity check the permutation-test lesson asks for: an invariance assertion is
    evidence only once it has been shown to move.
    """
    rows = _sft_rows() + [
        {
            "state_text": "s6",
            "action_json": ask_action_json("and then?", "next"),
            "done_before": True,
            "split": "dev",
        }
    ]
    perturbed = sft_nll(rows, render=render, logprob=logprob)
    assert json.dumps(perturbed, sort_keys=True) != json.dumps(GOLDEN["sft_nll"], sort_keys=True)


# --------------------------------------------------------------------------- the dataset half


def _write_dev(tmp_path: Path, name: str, rows: list[dict], manifest: dict | None) -> Path:
    p = tmp_path / name
    p.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    if manifest is not None:
        p.with_suffix(".manifest.json").write_text(json.dumps(manifest, sort_keys=True))
    return p


def test_dataset_provenance_names_the_file_its_sha_and_its_row_count(tmp_path: Path) -> None:
    rows = [dict(r, scorer_hash="aa" * 32, graph_version="v1") for r in _sft_rows()]
    p = _write_dev(tmp_path, "sft.dev.jsonl", rows, {"split": "dev", "n_examples": len(rows)})
    got = dataset_provenance(p, rows)
    assert got["path"] == str(p)
    assert got["n_rows"] == len(rows)
    assert got["sha256"] == hashlib.sha256(p.read_bytes()).hexdigest()
    assert got["scorer_hash"] == ["aa" * 32]
    assert got["graph_version"] == ["v1"]
    # `sft.dev.jsonl` pairs with `sft.dev.manifest.json`, not with `sft.manifest.json`.
    assert got["manifest"]["path"].endswith("sft.dev.manifest.json")
    assert got["manifest"]["present"] is True
    assert got["manifest"]["split"] == "dev"


def test_a_file_mixing_two_scorers_reports_both(tmp_path: Path) -> None:
    """A SET, because one value would have to pick one and would pick it silently.

    This is the case that makes `scorer_hash` a list rather than a string: two scorer
    generations in one file is a real state of the world (`scorer_hash` is global over the gold
    set, so scoring a new suite re-stamps every run), and a reader comparing two checkpoints has
    to be able to see that their population was not one population.
    """
    rows = [dict(r, scorer_hash="aa" * 32, graph_version="v1") for r in _sft_rows()]
    rows[0] = dict(rows[0], scorer_hash="bb" * 32, graph_version="v2")
    p = _write_dev(tmp_path, "sft.dev.jsonl", rows, None)
    got = dataset_provenance(p, rows)
    assert got["scorer_hash"] == ["aa" * 32, "bb" * 32]
    assert got["graph_version"] == ["v1", "v2"]


def test_a_manifest_without_the_two_fields_says_so_rather_than_omitting_them(
    tmp_path: Path,
) -> None:
    """The dev manifests carry neither `scorer_hash` nor `graph_version` (measured 2026-09-18).

    Recorded as `false` rather than left out, so a reader can tell "the manifest does not carry
    it, the rows do" from "nobody looked".
    """
    rows = [dict(r, scorer_hash="aa" * 32, graph_version="v1") for r in _sft_rows()]
    p = _write_dev(tmp_path, "sft.dev.jsonl", rows, {"split": "dev"})
    got = dataset_provenance(p, rows)
    assert got["manifest"]["carries_scorer_hash"] is False
    assert got["manifest"]["carries_graph_version"] is False
    assert got["graph_version"] == ["v1"]


def test_a_sampled_file_has_no_manifest_and_records_its_absence(tmp_path: Path) -> None:
    rows = [dict(r, scorer_hash="aa" * 32) for r in _sft_rows()]
    p = _write_dev(tmp_path, "sft.dev.sample.jsonl", rows, None)
    assert dataset_provenance(p, rows)["manifest"]["present"] is False


# --------------------------------------------------------------------------- the adapter half


def _adapter(tmp_path: Path, payload: bytes, base: str = "Qwen/Qwen3-8B") -> Path:
    d = tmp_path / "ckpt"
    d.mkdir(exist_ok=True)
    (d / WEIGHTS_FILE).write_bytes(payload)
    (d / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": base}))
    return d


def _registry(tmp_path: Path, rows: dict) -> Path:
    p = tmp_path / "checkpoints.json"
    p.write_text(json.dumps({"_schema": "doc row, skipped", **rows}))
    return p


def test_the_adapter_digest_is_the_registrys_own_writers(tmp_path: Path) -> None:
    """`weights_sha256` and `scripts/hpc/register_checkpoint.py:sha256_file` must agree.

    The registry is written by that script and read by this one. Two implementations of one
    measure drift, and the drift is silent -- an output would name a row whose weights it never
    checked. The script is not importable as a package, so it is loaded by path here and the two
    are pinned equal on a fixture rather than assumed equal.
    """
    spec = importlib.util.spec_from_file_location(
        "_register_checkpoint_under_test", REPO / "scripts" / "hpc" / "register_checkpoint.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    d = _adapter(tmp_path, b"weights-bytes-for-the-test" * 1000)
    assert weights_sha256(d) == mod.sha256_file(d / WEIGHTS_FILE)
    assert weights_sha256(d) == sha256_file(d / WEIGHTS_FILE)


def test_a_checkpoint_with_no_adapter_records_a_null_digest_not_a_zero(tmp_path: Path) -> None:
    """A base-model reading is a legitimate Tier-A row and has no adapter to hash."""
    d = tmp_path / "full-model"
    d.mkdir()
    assert weights_sha256(d) is None
    got = checkpoint_provenance(d)
    assert got["adapter_sha"] is None
    assert got["weights_file"] is None


def test_a_registry_name_that_does_not_hash_to_these_weights_is_REFUSED(tmp_path: Path) -> None:
    """The refusal this whole argument exists for.

    A wrong name copies a wrong `base_model`, a wrong `rung` and a wrong `init_from` into the
    output, all of them plausible. Nothing downstream can catch that, so it is refused here.
    """
    d = _adapter(tmp_path, b"the real weights")
    reg = _registry(
        tmp_path,
        {"qwen3-8b-dpo-headline-notdone-only": {"adapter_sha": "de" * 32, "base_model": "X"}},
    )
    with pytest.raises(AdapterIdentityMismatch) as exc:
        checkpoint_provenance(
            d, registry_path=reg, registry_name="qwen3-8b-dpo-headline-notdone-only"
        )
    assert weights_sha256(d) in str(exc.value)


def test_an_unregistered_registry_name_is_REFUSED(tmp_path: Path) -> None:
    d = _adapter(tmp_path, b"the real weights")
    reg = _registry(tmp_path, {"some-other-row": {"adapter_sha": "de" * 32}})
    with pytest.raises(AdapterIdentityMismatch):
        checkpoint_provenance(d, registry_path=reg, registry_name="not-a-row")


def test_a_matching_registry_name_is_recorded_as_verified_with_its_row(tmp_path: Path) -> None:
    d = _adapter(tmp_path, b"the real weights")
    sha = weights_sha256(d)
    reg = _registry(
        tmp_path,
        {
            "qwen3-8b-dpo-headline-notdone-only": {
                "adapter_sha": sha,
                "base_model": "Qwen/Qwen3-8B",
                "rung": 2,
                "init_from": "qwen3-8b-sft-headline",
                "dataset_sha": "ab" * 32,
            }
        },
    )
    got = checkpoint_provenance(
        d, registry_path=reg, registry_name="qwen3-8b-dpo-headline-notdone-only"
    )
    assert got["adapter_sha_verified"] is True
    assert got["adapter_sha"] == sha
    assert got["base_model"] == "Qwen/Qwen3-8B"
    assert got["base_model_source"] == "registry"
    assert got["rung"] == 2
    assert got["init_from"] == "qwen3-8b-sft-headline"


def test_without_a_name_the_block_still_says_whether_the_weights_are_registered(
    tmp_path: Path,
) -> None:
    """Whether the weights are registered at all is answerable without being told the answer."""
    d = _adapter(tmp_path, b"the real weights")
    reg = _registry(tmp_path, {"a-row": {"adapter_sha": weights_sha256(d)}})
    got = checkpoint_provenance(d, registry_path=reg)
    assert got["registry_names_matching_weights"] == ["a-row"]
    assert got["adapter_sha_verified"] is None  # not checked, which is not "checked and false"
    assert got["base_model"] == "Qwen/Qwen3-8B"
    assert got["base_model_source"] == "adapter_config.json"


# --------------------------------------------------------------------------- the whole block


def test_provenance_carries_the_stop_bytes_the_2x2_argmaxed_against(tmp_path: Path) -> None:
    """A P(STOP|done) computed against different STOP bytes is a different number.

    `stop_confusion` argmaxes `STOP_ACTION_JSON` against an ASK. Change those bytes and every
    such cell moves, with nothing in an old file able to say which string it used.
    """
    rows = [dict(r, scorer_hash="aa" * 32, graph_version="v1") for r in _sft_rows()]
    sft = _write_dev(tmp_path, "sft.dev.jsonl", rows, {"split": "dev"})
    pairs = _write_dev(tmp_path, "pairs.dev.jsonl", _pair_rows(), {"split": "dev"})
    got = provenance(
        checkpoint=_adapter(tmp_path, b"w"),
        dev_sft=sft,
        sft_rows=rows,
        dev_pairs=pairs,
        pair_rows=_pair_rows(),
        argv=["pi", "train", "eval-offline", "--reference-ask", REF_ASK],
        repo_root=REPO,
        reference_ask=REF_ASK,
    )
    assert got["stop_action_json"] == STOP_ACTION_JSON
    assert got["schema"] == PROVENANCE_SCHEMA
    assert got["reference_ask"] == REF_ASK
    assert "--reference-ask" in got["cli"]


def test_the_block_is_json_serialisable_exactly_as_the_verdict_writes_it(tmp_path: Path) -> None:
    """`json.dumps(..., sort_keys=True)` is what the CLI writes; a NaN or a Path would break it
    only at the end of a multi-hour run, which is the worst possible time to find out."""
    rows = [dict(r, scorer_hash="aa" * 32, graph_version="v1") for r in _sft_rows()]
    sft = _write_dev(tmp_path, "sft.dev.jsonl", rows, {"split": "dev"})
    pairs = _write_dev(tmp_path, "pairs.dev.jsonl", _pair_rows(), {"split": "dev"})
    block = provenance(
        checkpoint=_adapter(tmp_path, b"w"),
        dev_sft=sft,
        sft_rows=rows,
        dev_pairs=pairs,
        pair_rows=_pair_rows(),
        argv=["pi", "train", "eval-offline"],
        repo_root=REPO,
    )
    text = json.dumps(block, indent=2, sort_keys=True, allow_nan=False)
    assert json.loads(text)["checkpoint"]["adapter_sha"] == block["checkpoint"]["adapter_sha"]


def test_code_version_is_the_commit_that_computed_the_number(tmp_path: Path) -> None:
    head = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    rows = [dict(r, scorer_hash="aa" * 32) for r in _sft_rows()]
    sft = _write_dev(tmp_path, "sft.dev.jsonl", rows, None)
    pairs = _write_dev(tmp_path, "pairs.dev.jsonl", _pair_rows(), None)
    block = provenance(
        checkpoint=_adapter(tmp_path, b"w"),
        dev_sft=sft,
        sft_rows=rows,
        dev_pairs=pairs,
        pair_rows=_pair_rows(),
        argv=["pi", "train", "eval-offline"],
        repo_root=REPO,
    )
    assert block["code_version"]["sha"] == head
    assert isinstance(block["code_version"]["dirty"], bool)


def test_a_missing_git_is_unknown_and_dirty_not_a_silent_blank(tmp_path: Path) -> None:
    """`git_info`'s convention, kept: an unnameable code version is the `dev-` prefix's case."""
    rows = [dict(r, scorer_hash="aa" * 32) for r in _sft_rows()]
    sft = _write_dev(tmp_path, "sft.dev.jsonl", rows, None)
    pairs = _write_dev(tmp_path, "pairs.dev.jsonl", _pair_rows(), None)
    block = provenance(
        checkpoint=_adapter(tmp_path, b"w"),
        dev_sft=sft,
        sft_rows=rows,
        dev_pairs=pairs,
        pair_rows=_pair_rows(),
        argv=["pi", "train", "eval-offline"],
        repo_root=tmp_path / "not-a-repo",
    )
    assert block["code_version"] == {"sha": "unknown", "dirty": True}


def test_the_cli_refuses_before_loading_weights_on_a_mismatched_name(tmp_path: Path) -> None:
    """The refusal reaches the command, and reaches it BEFORE the 105-minute part.

    Exercised through `cmd_train_eval_offline` with a registry name that does not match, on a
    machine with no GPU and no transformers install needed: the refusal must come back as exit 2
    with nothing written, from a path that never touches `from_pretrained`.
    """
    from pi_run import cmd_train

    rows = [dict(r, split="dev", scorer_hash="aa" * 32) for r in _sft_rows()]
    sft = _write_dev(tmp_path, "sft.dev.jsonl", rows, {"split": "dev"})
    pairs = _write_dev(
        tmp_path, "pairs.dev.jsonl", [dict(p, split="dev") for p in _pair_rows()], {"split": "dev"}
    )
    d = _adapter(tmp_path, b"the real weights")
    reg = _registry(tmp_path, {"qwen3-8b-sft-headline": {"adapter_sha": "de" * 32}})
    out = tmp_path / "verdict.json"

    args = cmd_train.argparse.Namespace(
        checkpoint=str(d),
        dev_sft=str(sft),
        dev_pairs=str(pairs),
        out=str(out),
        reference_ask=None,
        chat_template=True,
        enable_thinking=False,
        registry=str(reg),
        registry_name="qwen3-8b-sft-headline",
    )
    assert cmd_train.cmd_train_eval_offline(args) == 2
    assert not out.exists()


def test_the_real_dev_export_carries_one_scorer_and_one_graph_version() -> None:
    """Not a unit test: a reading of the file the paper's offline cells are computed on.

    Skipped where the export is absent (it is gitignored and lives only in the shared checkout),
    because a test that silently passes on a missing file would be the instrument error that
    announces itself as a finding.
    """
    root = Path(__file__).resolve().parents[1] / "data" / "rl" / "dev"
    sft = root / "sft.dev.jsonl"
    if not sft.is_file():
        pytest.skip(f"{sft} absent (data/rl/dev is gitignored); read it from the main checkout")
    rows = [json.loads(line) for line in sft.read_text().splitlines() if line.strip()]
    got = dataset_provenance(sft, rows)
    assert got["split"] == ["dev"]
    assert len(got["scorer_hash"]) == 1, got["scorer_hash"]
    assert len(got["graph_version"]) == 1, got["graph_version"]


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-q"]))
