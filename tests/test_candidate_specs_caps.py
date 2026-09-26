"""A fork may be given a larger retrieval budget than its parent, and it must say so.

THE DEFECT THESE TESTS WERE WRITTEN FOR. `candidate_specs` copied `budget_cap` and
`max_turns` verbatim from the parent manifest, and nothing on the fork path -- no flag, no
env var, no grid field -- could override them. 93% of the candidates forked overnight ran at
cap 8, inherited from `conf/grids/latent_trainset.yaml`, and 53% of them stopped on the
budget rather than because the policy chose to. A candidate cut off at the cap has an answer
drafted from whatever it had gathered, and its `turns_to_complete` is -1, so the two keys the
preference label puts above evidence gain -- outcome and quickest -- are unavailable on most
pairs and the label falls back to per-turn gain: a retrieval-greedy signal.

WHY OVERRIDING THE CAP IS LEGITIMATE. The docstring said a candidate that differed in caps
"would not be a continuation of the same state". The policy is budget-blind by type
(`Inquirer.act(s)` takes no ledger; `FORBIDDEN_PLACEHOLDERS` keep the cap out of every
prompt), so the STATE at the fork turn is cap-invariant; only the episode-level labels
depend on the cap. Both values are in `semantic_hash`, so a re-fork at a larger cap is a
distinct run and cannot collide with the cap-8 candidates. The exporter's cross-cap guard
keeps the two cohorts from pairing.

The first test failed before the change with `TypeError: candidate_specs() got an unexpected
keyword argument 'budget_cap'`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pi_run.cmd_train import candidate_specs


def _parent(runs: Path, run_id: str, *, budget_cap: int = 8, max_turns: int = 16, n_turns: int = 3):
    d = runs / run_id
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "suite_id": "synth",
                "task_id": "t0",
                "arm_id": "inquirer_prompted",
                "seed": 0,
                "k": 3,
                "budget_cap": budget_cap,
                "max_turns": max_turns,
                "corpus_dir": str(runs.parent / "corpus"),
            }
        )
    )
    (runs.parent / "corpus").mkdir(exist_ok=True)
    (d / "turns.jsonl").write_text(
        "\n".join(json.dumps({"turn_idx": i, "action_kind": "ask"}) for i in range(n_turns))
    )
    return d


def _specs(runs: Path, **kw):
    return candidate_specs(
        runs_root=runs,
        cache_root=str(runs.parent / "cache"),
        parent_run_id="p1",
        turns=[1],
        n_candidates=2,
        code_version="deadbeef",
        dirty=False,
        temperature=1.0,
        **kw,
    )


def test_an_explicit_cap_flows_to_every_candidate(tmp_path):
    runs = tmp_path / "runs"
    _parent(runs, "p1")
    specs = _specs(runs, budget_cap=24, max_turns=24)
    assert [s.budget_cap for s in specs] == [24, 24]
    assert [s.max_turns for s in specs] == [24, 24]


def test_omitting_the_cap_still_inherits_the_parents(tmp_path):
    """The prior behaviour is the default, so every existing caller is unchanged."""
    runs = tmp_path / "runs"
    _parent(runs, "p1", budget_cap=8, max_turns=16)
    specs = _specs(runs)
    assert {s.budget_cap for s in specs} == {8}
    assert {s.max_turns for s in specs} == {16}


def test_a_max_turns_at_or_below_the_fork_turn_is_refused(tmp_path):
    """A fork at turn 1 with max_turns 1 would replay the prefix and never act. That is not a
    candidate, it is the parent's prefix wearing a new run id."""
    runs = tmp_path / "runs"
    _parent(runs, "p1")
    with pytest.raises(ValueError, match="max_turns"):
        _specs(runs, max_turns=1)


def test_the_two_caps_change_run_identity(tmp_path):
    """Both values are in semantic_hash, so a re-fork at a larger cap is a DISTINCT run: it
    cannot be skipped by resume-by-existence and cannot overwrite the cap-8 candidates."""
    from pi_run.manifest import build_manifest

    runs = tmp_path / "runs"
    _parent(runs, "p1")
    lo = _specs(runs)[0]
    hi = _specs(runs, budget_cap=24, max_turns=24)[0]
    assert lo.branch_seed == hi.branch_seed  # same candidate, two caps

    def ident(spec):
        return build_manifest(
            suite_id=spec.suite_id,
            task_id=spec.task_id,
            arm_id=spec.arm_id,
            policy_id="prompted",
            seed=spec.seed,
            corpus_hash="c",
            budget_cap=spec.budget_cap,
            max_turns=spec.max_turns,
            word_cap=30,
            code_version="deadbeef",
            dirty=False,
            branch_of_run_id=spec.branch_of_run_id,
            branch_turn_idx=spec.branch_turn_idx,
            branch_seed=spec.branch_seed,
        ).semantic_hash

    assert ident(lo) != ident(hi)


def _parent_without_caps(runs: Path, run_id: str, *, n_turns: int = 3):
    """Like `_parent`, but the manifest has no `max_turns`/`k` key at all -- the shape a real
    manifest has when the run predates those columns, or when a writer omits a null. This is
    the ONLY way to reach `unit_spec_defaults`'s fallback branch: `_parent` above always writes
    both keys, so every existing test in this file exercises the parent-inherits-its-own-cap
    path, never the UnitSpec-default path."""
    d = runs / run_id
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "suite_id": "synth",
                "task_id": "t0",
                "arm_id": "inquirer_prompted",
                "seed": 0,
                "corpus_dir": str(runs.parent / "corpus"),
            }
        )
    )
    (runs.parent / "corpus").mkdir(exist_ok=True)
    (d / "turns.jsonl").write_text(
        "\n".join(json.dumps({"turn_idx": i, "action_kind": "ask"}) for i in range(n_turns))
    )
    return d


def test_a_changed_unitspec_default_reaches_the_fallback_when_the_parent_has_no_cap(
    monkeypatch, tmp_path
):
    """`candidate_specs` used to retype `max_turns=16, k=5` as its own fallback literals; a
    change to either default on `UnitSpec` -- the one place these two numbers are meant to be
    declared -- could silently diverge from the copy here. Move both defaults on `UnitSpec`
    itself and confirm a parent manifest that omits `max_turns`/`k` (so the fallback, not the
    parent's own recorded cap, is what actually resolves) follows the change: this reads
    `dataclasses.fields(UnitSpec)` fresh on every call, it does not freeze a copy at import."""
    from pi_run.worker import UnitSpec

    monkeypatch.setattr(UnitSpec.__dataclass_fields__["max_turns"], "default", 41)
    monkeypatch.setattr(UnitSpec.__dataclass_fields__["k"], "default", 9)

    runs = tmp_path / "runs"
    _parent_without_caps(runs, "p1")
    specs = _specs(runs)
    assert {s.max_turns for s in specs} == {41}
    assert {s.k for s in specs} == {9}
