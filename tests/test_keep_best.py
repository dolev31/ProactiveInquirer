"""The checkpoint evaluator must KEEP the checkpoints that dev says are best -- ALL of them.

Rung 1 saves a checkpoint every 200 steps for preemption and keeps only the last two
(`save_total_limit` 2, the home quota). `scripts/hpc/eval_checkpoints.sh` scores each one on
the dev sample and then the trainer deletes it -- so on 2026-09-14 the step-1,000 adapter
(dev NLL 0.314, the best so far) survived only because it happened to be one of the last two.
`scripts/hpc/keep_best.py` runs after every verdict and copies the winning adapter under
`<out>/best/<criterion>/`: by dev NLL per token (the SFT objective on held-out rows) and by
the balanced STOP 2x2 (the cell the plan's tier A reads). The untrained base (tag 0) is never
a candidate; `final` counts as the largest step.

WHY THE LAYOUT IS PER-STEP AND NOT ONE FLAT DIRECTORY. The first version of this script
OVERWROTE `<out>/best/<criterion>/` when a later checkpoint won, which makes the copy exactly
as perishable as the checkpoint it was made to outlive. MEASURED 2026-09-16 on the cluster:
`qwen3-8b-sft-sw05`'s by_stop2x2 winner at step 1200 was registered in conf/checkpoints.json
by its adapter sha (cd0a1884...) at 20:00; at 20:24 the final checkpoint won the criterion and
overwrote `best/by_stop2x2/`; checkpoint-1200 had already rotated out of the trainer's
retention, so the registered weights then existed NOWHERE and that row had to be deleted. A
row in conf/checkpoints.json naming weights that no longer exist is CLAUDE.md rule 1 failing
after the fact. So a winner is copied to `<out>/best/<criterion>/step-<tag>/`, one directory
per step ever selected, and a directory this script has written is never deleted, moved or
rewritten; `CURRENT` names the current winner and `SELECTION.json`'s `history` says which
steps were ever selected and at what metric.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "hpc" / "keep_best.py"


def _mod():
    if not SCRIPT.exists():
        pytest.fail(f"{SCRIPT} does not exist")
    spec = importlib.util.spec_from_file_location("keep_best", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _verdict(nll: float, p_stop: float | None, p_ask: float | None) -> dict:
    return {
        "sft_nll": {"nll_per_token": nll, "nll_per_example": nll * 10, "n": 1000},
        "stop_confusion": {"p_stop_given_done": p_stop, "p_ask_given_not_done": p_ask},
        "pair_accuracy": {"acc": 0.5},
    }


def test_select_best_excludes_the_base_and_picks_by_each_criterion():
    kb = _mod()
    verdicts = {
        "0": _verdict(4.77, 0.0, 1.0),  # the untrained base: never a candidate
        "400": _verdict(0.342, 1.0, 0.643),
        "600": _verdict(0.321, 1.0, 0.834),
        "800": _verdict(0.316, 1.0, 0.669),
        "1000": _verdict(0.314, 1.0, 0.700),
    }
    best = kb.select_best(verdicts)
    assert best["by_nll"] == "1000"
    assert best["by_stop2x2"] == "600"


def test_ties_go_to_the_later_step_and_final_is_the_largest_step():
    kb = _mod()
    verdicts = {
        "200": _verdict(0.30, 1.0, 0.7),
        "400": _verdict(0.30, 1.0, 0.7),
        "final": _verdict(0.30, 1.0, 0.7),
    }
    assert kb.select_best(verdicts) == {"by_nll": "final", "by_stop2x2": "final"}
    del verdicts["final"]
    assert kb.select_best(verdicts)["by_nll"] == "400"


def test_a_missing_stop_cell_drops_that_checkpoint_from_the_stop_criterion_only():
    kb = _mod()
    verdicts = {"200": _verdict(0.40, None, None), "400": _verdict(0.50, 0.9, 0.9)}
    best = kb.select_best(verdicts)
    assert best["by_nll"] == "200"
    assert best["by_stop2x2"] == "400"


def _adapter(dir_: Path, payload: bytes) -> None:
    dir_.mkdir(parents=True, exist_ok=True)
    (dir_ / "adapter_config.json").write_text('{"r": 32}')
    (dir_ / "adapter_model.safetensors").write_bytes(payload)


def _tree_sha(d: Path) -> str:
    """Every path under `d` and every byte in it -- so "untouched" is checkable, not asserted."""
    h = hashlib.sha256()
    for p in sorted(d.rglob("*")):
        h.update(str(p.relative_to(d)).encode() + b"\0")
        h.update(p.read_bytes() if p.is_file() else b"<dir>")
    return h.hexdigest()


def _verdicts_on_disk(eval_dir: Path, run: str, rows: dict[str, dict]) -> None:
    eval_dir.mkdir(parents=True, exist_ok=True)
    for tag, v in rows.items():
        (eval_dir / f"{run}.ckpt-{tag}.tierA.json").write_text(json.dumps(v))


def test_keep_copies_the_winner_under_its_own_step_directory(tmp_path: Path):
    kb = _mod()
    eval_dir, out, stage = tmp_path / "eval", tmp_path / "out", tmp_path / "stage"
    _verdicts_on_disk(
        eval_dir,
        "r",
        {"0": _verdict(4.7, 0.0, 1.0), "400": _verdict(0.34, 1.0, 0.64)},
    )
    _adapter(stage / "checkpoint-400", b"four hundred")
    kb.keep(run="r", eval_dir=eval_dir, keep_dir=out / "best", sources=[stage, out])
    kept = out / "best" / "by_nll" / "step-400"
    assert (kept / "adapter_model.safetensors").read_bytes() == b"four hundred"
    assert (kept / "TAG").read_text().strip() == "400"
    assert (out / "best" / "by_nll" / "CURRENT").read_text().strip() == "400"
    sel = json.loads((out / "best" / "SELECTION.json").read_text())
    assert sel["by_nll"]["tag"] == "400" and sel["by_stop2x2"]["tag"] == "400"


def test_a_later_winner_does_not_remove_or_alter_the_earlier_winners_directory(tmp_path: Path):
    """The sw05 incident, reproduced: registered weights must survive the next winner."""
    kb = _mod()
    eval_dir, out, stage = tmp_path / "eval", tmp_path / "out", tmp_path / "stage"
    _verdicts_on_disk(eval_dir, "r", {"1200": _verdict(0.33, 1.0, 0.86)})
    _adapter(stage / "checkpoint-1200", b"cd0a1884 the weights conf/checkpoints.json registered")
    kb.keep(run="r", eval_dir=eval_dir, keep_dir=out / "best", sources=[stage, out])
    registered = out / "best" / "by_stop2x2" / "step-1200"
    before = _tree_sha(registered)

    # 20:24: the final checkpoint wins the criterion, and save_total_limit has already taken
    # checkpoint-1200 away -- the copy under best/ is now the only one of those weights left.
    shutil.rmtree(stage / "checkpoint-1200")
    _verdicts_on_disk(eval_dir, "r", {"final": _verdict(0.30, 1.0, 0.93)})
    _adapter(out, b"final weights")
    kb.keep(run="r", eval_dir=eval_dir, keep_dir=out / "best", sources=[stage, out])

    assert registered.is_dir(), "the registered winner's directory was deleted"
    assert _tree_sha(registered) == before, "the registered winner's bytes were rewritten"
    assert (
        registered / "adapter_model.safetensors"
    ).read_bytes() == b"cd0a1884 the weights conf/checkpoints.json registered"
    new = out / "best" / "by_stop2x2" / "step-final"
    assert (new / "adapter_model.safetensors").read_bytes() == b"final weights"


def test_current_moves_to_the_new_winner_and_history_lists_both(tmp_path: Path):
    kb = _mod()
    eval_dir, out, stage = tmp_path / "eval", tmp_path / "out", tmp_path / "stage"
    _verdicts_on_disk(eval_dir, "r", {"1200": _verdict(0.33, 1.0, 0.86)})
    _adapter(stage / "checkpoint-1200", b"twelve hundred")
    kb.keep(run="r", eval_dir=eval_dir, keep_dir=out / "best", sources=[stage, out])
    assert (out / "best" / "by_stop2x2" / "CURRENT").read_text().strip() == "1200"

    _verdicts_on_disk(eval_dir, "r", {"final": _verdict(0.30, 1.0, 0.93)})
    _adapter(out, b"final weights")
    sel = kb.keep(run="r", eval_dir=eval_dir, keep_dir=out / "best", sources=[stage, out])

    assert (out / "best" / "by_stop2x2" / "CURRENT").read_text().strip() == "final"
    hist = json.loads((out / "best" / "SELECTION.json").read_text())["by_stop2x2"]["history"]
    assert [h["tag"] for h in hist] == ["1200", "final"]
    # the metric AT SELECTION TIME, so the history can be read without the verdict files
    assert hist[0]["metrics"]["stop2x2"] == pytest.approx(0.93)
    assert hist[1]["metrics"]["stop2x2"] == pytest.approx(0.965)
    assert sel["by_stop2x2"]["history"] == hist


def test_re_running_with_the_same_verdicts_changes_nothing_on_disk(tmp_path: Path):
    kb = _mod()
    eval_dir, out, stage = tmp_path / "eval", tmp_path / "out", tmp_path / "stage"
    _verdicts_on_disk(
        eval_dir, "r", {"400": _verdict(0.34, 1.0, 0.64), "600": _verdict(0.32, 1.0, 0.83)}
    )
    _adapter(stage / "checkpoint-400", b"four hundred")
    _adapter(stage / "checkpoint-600", b"six hundred")
    kb.keep(run="r", eval_dir=eval_dir, keep_dir=out / "best", sources=[stage, out])
    best = out / "best"
    before = {p.name: _tree_sha(p) for p in sorted(best.iterdir()) if p.is_dir()}
    sel_before = json.loads((best / "SELECTION.json").read_text())

    sel_after = kb.keep(run="r", eval_dir=eval_dir, keep_dir=best, sources=[stage, out])

    assert {p.name: _tree_sha(p) for p in sorted(best.iterdir()) if p.is_dir()} == before
    for crit in ("by_nll", "by_stop2x2"):
        assert sel_after[crit]["history"] == sel_before[crit]["history"]
        assert sel_after[crit]["kept_tag"] == sel_before[crit]["kept_tag"]
        assert sel_after[crit]["tag"] == sel_before[crit]["tag"]


def test_the_old_flat_layout_is_migrated_once_and_preserved(tmp_path: Path):
    """A `best/<crit>/` written by the overwriting version is evidence; it is not deleted."""
    kb = _mod()
    eval_dir, out, stage = tmp_path / "eval", tmp_path / "out", tmp_path / "stage"
    flat = out / "best" / "by_nll"
    _adapter(flat, b"the weights the old scheme left behind")
    (flat / "TAG").write_text("1600\n")
    flat_before = {p.name: p.read_bytes() for p in sorted(flat.iterdir()) if p.is_file()}

    _verdicts_on_disk(eval_dir, "r", {"1800": _verdict(0.31, 1.0, 0.80)})
    _adapter(stage / "checkpoint-1800", b"eighteen hundred")
    sel = kb.keep(run="r", eval_dir=eval_dir, keep_dir=out / "best", sources=[stage, out])

    migrated = flat / "step-1600"
    assert (
        migrated / "adapter_model.safetensors"
    ).read_bytes() == b"the weights the old scheme left behind"
    assert (migrated / "TAG").read_text().strip() == "1600"
    # every file that was there is still there, byte for byte (CURRENT is new, and is the
    # only thing this script may add at a criterion root)
    assert all((flat / n).read_bytes() == b for n, b in flat_before.items())
    assert sel["by_nll"]["migrated_flat_to"] == "step-1600"
    assert [h["tag"] for h in sel["by_nll"]["history"]] == ["1600", "1800"]
    assert (flat / "CURRENT").read_text().strip() == "1800"

    # once. a second pass neither re-migrates nor touches what it migrated
    sha = _tree_sha(migrated)
    sel2 = kb.keep(run="r", eval_dir=eval_dir, keep_dir=out / "best", sources=[stage, out])
    assert _tree_sha(migrated) == sha
    assert "migrated_flat_to" not in sel2["by_nll"]
    assert [h["tag"] for h in sel2["by_nll"]["history"]] == ["1600", "1800"]


def test_a_winner_that_is_nowhere_on_disk_leaves_the_kept_copy_alone_and_says_so(tmp_path: Path):
    kb = _mod()
    eval_dir, out, stage = tmp_path / "eval", tmp_path / "out", tmp_path / "stage"
    _verdicts_on_disk(eval_dir, "r", {"600": _verdict(0.32, 1.0, 0.83)})
    _adapter(stage / "checkpoint-600", b"six hundred")
    kb.keep(run="r", eval_dir=eval_dir, keep_dir=out / "best", sources=[stage, out])
    shutil.rmtree(stage / "checkpoint-600")

    # step 800 wins but was rotated away before it could be copied: the record says which tag
    # SHOULD be there, so the loss is visible rather than silent.
    _verdicts_on_disk(eval_dir, "r", {"800": _verdict(0.31, 1.0, 0.70)})
    sel = kb.keep(run="r", eval_dir=eval_dir, keep_dir=out / "best", sources=[stage, out])
    assert (
        out / "best" / "by_nll" / "step-600" / "adapter_model.safetensors"
    ).read_bytes() == b"six hundred"
    assert sel["by_nll"]["tag"] == "800" and sel["by_nll"]["kept_tag"] == "600"
    assert sel["by_nll"]["status"] == "unavailable"
    assert (out / "best" / "by_nll" / "CURRENT").read_text().strip() == "600"


def test_the_final_adapter_is_read_from_the_out_root(tmp_path: Path):
    kb = _mod()
    eval_dir, out = tmp_path / "eval", tmp_path / "out"
    _verdicts_on_disk(eval_dir, "r", {"final": _verdict(0.30, 1.0, 0.8)})
    _adapter(out, b"final weights")
    kb.keep(run="r", eval_dir=eval_dir, keep_dir=out / "best", sources=[out])
    assert (
        out / "best" / "by_nll" / "step-final" / "adapter_model.safetensors"
    ).read_bytes() == b"final weights"
