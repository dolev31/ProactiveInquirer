"""`scripts/dev_samples.py`: the small dev slices the per-checkpoint Tier-A evaluator scores.

WHY A SAMPLE EXISTS AT ALL. `pi train eval-offline` has no `--limit`, and the full dev files
(6,975 SFT rows, 4,962 pairs) are one forward pass per row -- three per SFT row, two per pair --
on an 8B model. At rung 1's checkpoint cadence (one every 200 optimizer steps) a full-file
evaluation cannot finish between two checkpoints, and `save_total_limit=2` deletes the older one
while it is still being scored. So the monitoring slice is smaller ON PURPOSE, and its identity
has to be pinned, because a curve whose points were computed over different row sets is not a
curve.

WHAT THIS TEST PINS, and why each one is a way the sample could silently go wrong:

  * DETERMINISM, byte for byte. Two builds that differ by one row give two NLLs at the same
    step and no way to tell which moved -- the checkpoint or the denominator.
  * DEV ONLY. `_dev_rows_or_refuse` in `pi_run.cmd_train` refuses a non-dev row, but it refuses
    at evaluation time on the cluster, hours after the sample was built on a laptop. This
    refuses at build time. A train row would select the checkpoint on the training set; a test
    row would burn the held-out split before it is read once.
  * THE ask_ask ROWS ARE ALL THERE. `acc_by_kind['ask_ask']` is the only cell of the pair
    accuracy that is not decided by an ask-vs-stop length asymmetry, so it is the cell the
    curve is read off. Sampling it would put a sampling error on the y axis of the figure, on
    top of the checkpoint-to-checkpoint movement it exists to show. All 970 are kept.
  * THE STOP/ASK RATIO SURVIVES. `nll_per_token` pools STOP and ASK rows, and a STOP action is
    ~7 tokens against tens for an ASK, so a sample that drifted toward STOP would lower the
    pooled NLL without the checkpoint having changed.

It runs against the real dev files, and SKIPS naming the missing path when they are absent --
except the refusal tests, which build their own two-row fixtures, because a checkout with no
corpus must still be able to prove that a train row is refused.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
GEN = REPO / "scripts" / "dev_samples.py"
SFT = REPO / "data/rl/dev/sft.dev.jsonl"
PAIRS = REPO / "data/rl/dev/pairs.dev.jsonl"


def _load():
    """`scripts/` is not a package; load the file by path (as tests/test_figures_programme.py
    does, and for the same reason: the module must be in `sys.modules` before it executes)."""
    if not GEN.exists():
        pytest.fail(f"generator absent: {GEN}")
    spec = importlib.util.spec_from_file_location("dev_samples", GEN)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def gen():
    return _load()


def _rows(p: Path) -> list[dict]:
    return [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]


def _write(p: Path, rows: list[dict]) -> Path:
    p.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    return p


@pytest.fixture(scope="module")
def built(gen, tmp_path_factory):
    missing = [p for p in (SFT, PAIRS) if not p.exists()]
    if missing:
        pytest.skip("dev corpus absent: " + ", ".join(str(p) for p in missing))
    out = tmp_path_factory.mktemp("dev_samples")
    rc = gen.main(["--sft", str(SFT), "--pairs", str(PAIRS), "--out-dir", str(out)])
    assert rc == 0, "dev_samples refused the real dev files"
    return out


def test_sample_files_are_written(built):
    for name in ("sft.dev.sample.jsonl", "pairs.dev.sample.jsonl"):
        p = built / name
        assert p.exists() and p.stat().st_size > 0, f"missing or empty: {name}"


def test_the_build_is_deterministic_byte_for_byte(gen, built, tmp_path):
    """Same inputs, same seed, same bytes. Two builds that differ by one row put two
    different denominators under one curve."""
    again = tmp_path / "again"
    assert gen.main(["--sft", str(SFT), "--pairs", str(PAIRS), "--out-dir", str(again)]) == 0
    for name in ("sft.dev.sample.jsonl", "pairs.dev.sample.jsonl"):
        assert (again / name).read_bytes() == (built / name).read_bytes(), f"{name} is not stable"


def test_every_sampled_row_is_dev(built):
    """The mirror of `_dev_rows_or_refuse`, one build stage earlier."""
    for name in ("sft.dev.sample.jsonl", "pairs.dev.sample.jsonl"):
        for i, row in enumerate(_rows(built / name)):
            assert row.get("split") == "dev", f"{name} row {i}: split={row.get('split')!r}"


def test_every_ask_ask_pair_is_kept(built):
    """Complete, not sampled: `acc_by_kind['ask_ask']` is the cell the figure reads."""
    src = {r["pair_id"] for r in _rows(PAIRS) if r.get("pair_kind") == "ask_ask"}
    got = {
        r["pair_id"] for r in _rows(built / "pairs.dev.sample.jsonl") if r["pair_kind"] == "ask_ask"
    }
    assert got == src, f"{len(src - got)} ask_ask pairs dropped, {len(got - src)} invented"


def test_every_ask_stop_pair_is_kept(built):
    """115 real ask_stop pairs exist in all of dev; sampling them would leave single digits."""
    src = {r["pair_id"] for r in _rows(PAIRS) if r.get("pair_kind") == "ask_stop"}
    got = {
        r["pair_id"]
        for r in _rows(built / "pairs.dev.sample.jsonl")
        if r["pair_kind"] == "ask_stop"
    }
    assert got == src


def test_sampled_rows_are_a_subset_of_the_source(built):
    """No row is rewritten on the way through: the sample is a selection, not a transform."""
    src = {json.dumps(r, sort_keys=True) for r in _rows(SFT)}
    for row in _rows(built / "sft.dev.sample.jsonl"):
        assert json.dumps(row, sort_keys=True) in src, "a sampled SFT row is not a source row"


def test_the_stop_ask_ratio_is_preserved(gen, built):
    """Within one row of the exact proportional allocation. `nll_per_token` pools both."""
    src = _rows(SFT)
    got = _rows(built / "sft.dev.sample.jsonl")
    assert len(got) == gen.N_SFT
    for rule in {r["label_rule"] for r in src}:
        want = len([r for r in src if r["label_rule"] == rule]) / len(src) * len(got)
        have = len([r for r in got if r["label_rule"] == rule])
        assert abs(have - want) <= 1.0, f"{rule}: {have} sampled, {want:.1f} proportional"


def test_pair_sample_counts_are_the_declared_ones(gen, built):
    got = _rows(built / "pairs.dev.sample.jsonl")
    kinds = {k: len([r for r in got if r["pair_kind"] == k]) for k in {r["pair_kind"] for r in got}}
    assert kinds.get("ask_stop_synth") == gen.N_SYNTH
    assert len(got) == sum(kinds.values())


def test_a_non_dev_row_is_refused_not_dropped(gen, tmp_path):
    """The failure this exists to prevent is SILENT: a train row would be sampled, scored, and
    reported as a dev number. Build a two-row fixture so a corpus-free checkout proves it too."""
    sft = _write(
        tmp_path / "sft.jsonl",
        [
            {"split": "dev", "label_rule": "stop_done", "state_text": "s", "action_json": "{}"},
            {"split": "train", "label_rule": "stop_done", "state_text": "s", "action_json": "{}"},
        ],
    )
    pairs = _write(
        tmp_path / "pairs.jsonl", [{"split": "dev", "pair_kind": "ask_ask", "pair_id": "p1"}]
    )
    rc = gen.main(["--sft", str(sft), "--pairs", str(pairs), "--out-dir", str(tmp_path / "o")])
    assert rc == 2, "a train row was accepted into a dev sample"
    assert not (tmp_path / "o" / "sft.dev.sample.jsonl").exists(), "a refused build still wrote"


def test_a_non_dev_pair_is_refused_too(gen, tmp_path):
    sft = _write(
        tmp_path / "sft.jsonl",
        [{"split": "dev", "label_rule": "stop_done", "state_text": "s", "action_json": "{}"}],
    )
    pairs = _write(
        tmp_path / "pairs.jsonl",
        [
            {"split": "dev", "pair_kind": "ask_ask", "pair_id": "p1"},
            {"split": "test", "pair_kind": "ask_ask", "pair_id": "p2"},
        ],
    )
    rc = gen.main(["--sft", str(sft), "--pairs", str(pairs), "--out-dir", str(tmp_path / "o")])
    assert rc == 2, "a test row was accepted into a dev sample -- the held-out split, burned"
