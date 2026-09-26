"""The fork launcher: which units it would run, and that `--dry-run` runs none of them.

WHY THE LAUNCHER IS A SCRIPT AND NOT `pi run`. A fork's unit is addressed by
(task, trace, cut point), not by (task, split, n). `pi run` enumerates a suite's task ids, and
two forks of one task are different units it has no way to name.

WHY `--dry-run` EXISTS AT ALL. Launching is the only irreversible thing here: a fork campaign is
34 points x 3 seeds x 2 arms of paid rollout, and a wrong `--min-k` or a stale fork-points file
is invisible until the bill arrives. The dry run prints the exact units and returns before the
first `run_tau2_unit` call, so the set can be checked against the recorded campaign first.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import run_tau2_forks as launcher  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures" / "run_manifests"
FORKS = (
    "fork_tau2_retail_test_inquirer_prompted_s0.json",
    "fork_tau2_retail_test_self_ask_s0.json",
    "fork_tau2_retail_test_inquirer_prompted_s2.json",
)


@pytest.fixture(autouse=True)
def _absolute_cache_root(monkeypatch, tmp_path):
    """A real launch refuses a relative cache root, and `cache_root` has no flag to pass.

    It comes from `$PI_CACHE_ROOT` and otherwise defaults to the relative "cache". These tests stub
    `_run_unit` and assert on the specs that reach it, so they are about wiring, not about the root
    guard -- that guard's own scope is pinned in tests/test_absolute_root_refusal_scope.py, both
    that it fires on a launch and that it does not fire on a plan. Setting it here keeps the two
    concerns from masking each other.
    """
    monkeypatch.setenv("PI_CACHE_ROOT", str(tmp_path / "cacheroot"))


@pytest.fixture
def runs_root(tmp_path):
    """A runs tree holding the three recorded retail forks and one non-fork."""
    runs = tmp_path / "runs"
    for name in FORKS + ("nonfork_tau2_retail.json",):
        d = json.loads((FIXTURES / name).read_text())
        rd = runs / d["run_id"]
        rd.mkdir(parents=True)
        (rd / "manifest.json").write_text(json.dumps(d))
        (rd / "status.json").write_text(json.dumps({"status": "ok"}))
    return runs


# ------------------------------------------------------------- reading the recorded points


ARMS = ("inquirer_prompted", "self_ask")


def test_the_recorded_points_come_off_the_fork_manifests(runs_root):
    pts = launcher.points_from_runs(runs_root, suite="tau2_retail", split="test", arms=ARMS)
    assert len(pts) == 1, "the three fixtures are one fork point at three (arm, seed) cells"
    p = pts[0]
    d = json.loads((FIXTURES / FORKS[0]).read_text())
    assert p["trace_sha"] == d["foreign_trace_sha"]
    assert p["k"] == d["foreign_prefix_k"]
    assert p["task_id"] == d["task_id"]


def test_a_run_of_our_own_is_not_a_fork_point(runs_root):
    """The non-fork fixture in the same tree must contribute nothing: it has no prefix, and a
    point built from it would launch a full-task rollout into the fork table."""
    pts = launcher.points_from_runs(runs_root, suite="tau2_retail", split="test", arms=ARMS)
    assert all(p["trace_sha"] for p in pts)


def test_the_split_and_the_suite_both_filter(runs_root):
    assert launcher.points_from_runs(runs_root, suite="tau2_retail", split="train", arms=ARMS) == []
    assert launcher.points_from_runs(runs_root, suite="tau2_airline", split="test", arms=ARMS) == []


def test_an_arm_outside_the_campaign_contributes_no_point(runs_root):
    """THE FILTER THAT DECIDES WHETHER THE DRY RUN AGREES WITH THE REPORT. `runs/`
    accumulates forks from every campaign that ever ran -- the real retail tree holds
    cuts under five arms -- so an unfiltered read returns 174 points where the recorded
    two-arm campaign ran 34. Measured: with `arms=` dropped, the live tree yields 174.
    """
    assert (
        launcher.points_from_runs(
            runs_root, suite="tau2_retail", split="test", arms=("reference_agent",)
        )
        == []
    )
    assert launcher.points_from_runs(runs_root, suite="tau2_retail", split="test", arms=()) == [], (
        "no arms named is no campaign, not every campaign"
    )


def test_the_points_are_ordered_deterministically(runs_root):
    """Two invocations must list the same campaign in the same order, or a `--dry-run` that
    was checked and a launch that follows it are different sets."""
    a = launcher.points_from_runs(runs_root, suite="tau2_retail", split="test", arms=ARMS)
    b = launcher.points_from_runs(runs_root, suite="tau2_retail", split="test", arms=ARMS)
    assert a == b


# --------------------------------------------------------------------------- the unit set


def test_the_units_are_points_x_arms_x_seeds():
    pts = [
        {"trace_sha": "a" * 64, "k": 4, "task_id": "1"},
        {"trace_sha": "b" * 64, "k": 9, "task_id": "2"},
    ]
    specs = launcher.build_specs(
        pts, suite="tau2_retail", arms=["inquirer_prompted", "self_ask"], seeds=[0, 1, 2]
    )
    assert len(specs) == 12
    assert {s.foreign_trace_sha for s in specs} == {"a" * 64, "b" * 64}
    assert {s.seed for s in specs} == {0, 1, 2}
    assert {s.arm_id for s in specs} == {"inquirer_prompted", "self_ask"}


def test_every_unit_carries_the_cut_it_was_built_from():
    pts = [{"trace_sha": "a" * 64, "k": 4, "task_id": "7"}]
    (spec,) = launcher.build_specs(pts, suite="tau2_retail", arms=["self_ask"], seeds=[0])
    assert spec.foreign_trace_sha == "a" * 64
    assert spec.foreign_prefix_k == 4
    assert spec.task_id == "7"


def test_two_cuts_of_one_trace_are_two_units():
    """They differ in nothing else, so without `foreign_prefix_k` on the spec they share a
    `UnitSpec.key` and `run_sweep` returns two aliases of one status."""
    pts = [
        {"trace_sha": "a" * 64, "k": 4, "task_id": "7"},
        {"trace_sha": "a" * 64, "k": 9, "task_id": "7"},
    ]
    specs = launcher.build_specs(pts, suite="tau2_retail", arms=["self_ask"], seeds=[0])
    assert specs[0].key != specs[1].key


def test_a_fork_is_marked_exploratory_by_construction():
    """It continues someone else's dialogue, so its follow-up count is measured from a
    different origin than a full run's. `ELIGIBLE` also refuses it on `foreign_trace_sha`;
    this makes the manifest say so on its own."""
    pts = [{"trace_sha": "a" * 64, "k": 4, "task_id": "7"}]
    (spec,) = launcher.build_specs(pts, suite="tau2_retail", arms=["self_ask"], seeds=[0])
    assert spec.exploratory is True


# ------------------------------------------------------------------------- the trained arm


def test_the_trained_arm_is_accepted_and_reaches_the_spec():
    """`--arm inquirer_trained` must survive to `UnitSpec.arm_id`, which is what
    `run_tau2_unit` looks up in the registry and what `build_manifest` stamps."""
    pts = [{"trace_sha": "a" * 64, "k": 4, "task_id": "7"}]
    (spec,) = launcher.build_specs(pts, suite="tau2_retail", arms=["inquirer_trained"], seeds=[0])
    assert spec.arm_id == "inquirer_trained"


def test_the_trained_arm_exists_in_the_registry_the_runner_consults():
    from pinq_expt import arms as arm_table

    arm = arm_table.get("inquirer_trained")
    assert arm.arm_id == "inquirer_trained"
    assert not arm.llm_free


def test_an_unknown_arm_is_refused_before_anything_is_launched():
    """A typo must cost a message, not a campaign. `run_tau2_unit` would raise on it too --
    once per unit, after the sweep had already started paying for the correct ones."""
    from pinq_expt.arms import UnknownArm

    with pytest.raises(UnknownArm):
        launcher.build_specs(
            [{"trace_sha": "a" * 64, "k": 4, "task_id": "7"}],
            suite="tau2_retail",
            arms=["inquirer_trainedd"],
            seeds=[0],
        )


def test_the_trained_arm_reaches_the_manifest_and_carries_the_served_name(monkeypatch):
    """THE PIN IS THE ONLY DIFFERENCE between inquirer_trained and inquirer_prompted -- same
    policy class, same prompt -- so if the served name did not reach `pins`, the two arms would
    hash IDENTICALLY, share a run_id and `--resume` would skip the second.

    Built the way `pi run` builds it: the model comes from PI_MODEL_INQUIRER and the
    adapter_sha from conf/checkpoints.json, neither of which this launcher touches. No network
    is reached -- `pin()` reads configuration and hashes it.
    """
    from pi_run.manifest import build_manifest, manifest_to_dict
    from pinq_adapters.llm.checkpoints import registry
    from pinq_adapters.llm.litellm_client import MeteredClient

    served = "qwen3-8b-sft-headline"
    assert served in registry(), "this test needs a registered deployment to pin against"
    monkeypatch.setenv("PI_MODEL_INQUIRER", served)

    from pinq.budget import BudgetLedger

    client = MeteredClient(BudgetLedger(cap=1), temperature=0.0)
    pin = client.pin("inquirer")
    assert pin.model_id == served
    assert pin.adapter_sha == registry()[served]["adapter_sha"]

    pts = [{"trace_sha": "a" * 64, "k": 4, "task_id": "7"}]
    (spec,) = launcher.build_specs(pts, suite="tau2_retail", arms=["inquirer_trained"], seeds=[0])
    m = build_manifest(
        suite_id=spec.suite_id,
        task_id=spec.task_id,
        arm_id=spec.arm_id,
        policy_id=spec.arm_id,
        seed=spec.seed,
        corpus_hash="c",
        budget_cap=spec.budget_cap,
        max_turns=spec.max_turns,
        word_cap=180,
        code_version="abc",
        dirty=False,
        pins={"inquirer": pin},
        foreign_trace_sha=spec.foreign_trace_sha,
        foreign_prefix_k=spec.foreign_prefix_k,
    )
    d = manifest_to_dict(m)
    assert d["arm_id"] == "inquirer_trained"
    assert d["pins"]["inquirer"]["model_id"] == served
    assert d["pins"]["inquirer"]["adapter_sha"] == registry()[served]["adapter_sha"]
    assert d["foreign_trace_sha"] == "a" * 64


def test_the_trained_pin_is_what_separates_the_two_arms(monkeypatch):
    """Stated as the failure it prevents: with an empty `pins` the two arms hash the same."""
    from pi_run.manifest import build_manifest

    def _m(arm, pins):
        return build_manifest(
            suite_id="tau2_retail",
            task_id="7",
            arm_id=arm,
            policy_id=arm,
            seed=0,
            corpus_hash="c",
            budget_cap=16,
            max_turns=16,
            word_cap=180,
            code_version="abc",
            dirty=False,
            pins=pins,
        )

    assert _m("inquirer_trained", {}).run_id != _m("inquirer_prompted", {}).run_id, (
        "arm_id is itself in SEMANTIC_FIELDS, so this holds even with no pins"
    )


# ------------------------------------------------------------------------------- --dry-run


def test_dry_run_launches_nothing(runs_root, monkeypatch, capsys):
    """The property the flag exists for. `run_tau2_unit` is replaced with a detonator."""

    def _boom(spec):  # pragma: no cover - reaching it is the failure
        raise AssertionError("--dry-run must not launch a unit")

    monkeypatch.setattr(launcher, "_run_unit", _boom)
    rc = launcher.main(
        [
            "--suite",
            "tau2_retail",
            "--split",
            "test",
            "--forkpoints-from-runs",
            str(runs_root),
            "--arm",
            "inquirer_prompted",
            "--arm",
            "self_ask",
            "--seeds",
            "0",
            "1",
            "2",
            "--dry-run",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "fork points: 1" in out
    assert "would launch 6 units" in out


def test_dry_run_prints_the_four_fields_of_every_unit(runs_root, capsys):
    """trace sha, prefix k, task id, seed -- the tuple that identifies a fork unit, and the
    tuple `fork_report` pairs on."""
    launcher.main(
        [
            "--suite",
            "tau2_retail",
            "--split",
            "test",
            "--forkpoints-from-runs",
            str(runs_root),
            "--arm",
            "self_ask",
            "--seeds",
            "0",
            "--dry-run",
        ]
    )
    out = capsys.readouterr().out
    d = json.loads((FIXTURES / FORKS[0]).read_text())
    assert d["foreign_trace_sha"][:16] in out
    assert f"k={d['foreign_prefix_k']}" in out
    assert f"task={d['task_id']}" in out
    assert "seed=0" in out


def test_dry_run_reports_the_source_runs_it_derived_the_points_from(runs_root, capsys):
    """Provenance: the points are a RECORD of a campaign that ran, so the dry run says which
    run directories it read them out of and digests their ids, which is directly comparable
    with `provenance.run_ids_sha` in docs/reports/forks_tau2_*_test.json."""
    launcher.main(
        [
            "--suite",
            "tau2_retail",
            "--split",
            "test",
            "--forkpoints-from-runs",
            str(runs_root),
            "--arm",
            "inquirer_prompted",
            "--arm",
            "self_ask",
            "--seeds",
            "0",
            "2",
            "--dry-run",
        ]
    )
    out = capsys.readouterr().out
    assert "source fork runs: 3" in out
    assert "run_ids_sha" in out


def test_a_sampling_flag_is_refused_when_replaying_a_recorded_campaign(runs_root):
    """The recorded points are not a candidate pool. Filtering or sampling them would silently
    launch a different campaign than the one the flags name, and the operator would compare it
    against the recorded numbers anyway."""
    with pytest.raises(SystemExit):
        launcher.main(
            [
                "--suite",
                "tau2_retail",
                "--forkpoints-from-runs",
                str(runs_root),
                "--n",
                "1",
                "--dry-run",
            ]
        )


def test_a_source_that_yields_no_points_is_an_error_not_an_empty_campaign(tmp_path, capsys):
    """An empty launch that exits 0 reads as 'done'."""
    (tmp_path / "runs").mkdir()
    rc = launcher.main(
        [
            "--suite",
            "tau2_retail",
            "--forkpoints-from-runs",
            str(tmp_path / "runs"),
            "--dry-run",
        ]
    )
    assert rc != 0


# --------------------------------------------------------------- the file-based path is kept


def test_a_forkpoints_file_still_works_and_still_samples(tmp_path, capsys):
    """convlog-work's own input format. `sample_round_robin` draws across tasks before depth,
    so a few long dialogues cannot dominate a small sample."""
    f = tmp_path / "pts.json"
    f.write_text(
        json.dumps(
            [
                {"trace_sha": "a" * 64, "k": 10, "task_id": "1"},
                {"trace_sha": "a" * 64, "k": 12, "task_id": "1"},
                {"trace_sha": "b" * 64, "k": 11, "task_id": "2"},
            ]
        )
    )
    rc = launcher.main(
        [
            "--suite",
            "tau2_retail",
            "--forkpoints",
            str(f),
            "--n",
            "2",
            "--arm",
            "self_ask",
            "--seeds",
            "0",
            "--dry-run",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "fork points: 2" in out
    assert "task=1" in out and "task=2" in out, "the sample must span tasks before depth"


def test_the_k_window_filters_the_file_path():
    pts = [
        {"trace_sha": "a" * 64, "k": 2, "task_id": "1"},
        {"trace_sha": "a" * 64, "k": 20, "task_id": "1"},
    ]
    assert [p["k"] for p in launcher.in_k_window(pts, min_k=8, max_k=40)] == [20]


def test_round_robin_spans_tasks_before_depth():
    pts = [
        {"task_id": "1", "k": 30, "trace_sha": "a"},
        {"task_id": "1", "k": 20, "trace_sha": "a"},
        {"task_id": "1", "k": 10, "trace_sha": "a"},
        {"task_id": "2", "k": 9, "trace_sha": "b"},
    ]
    assert [p["task_id"] for p in launcher.sample_round_robin(pts, 2)] == ["1", "2"]


# ----------------------------------------------------- dirty / code_version wiring (defect 1)
#
# WHY THIS SECTION EXISTS. `UnitSpec.dirty` defaults `True` and `build_specs()` never had a
# `dirty` parameter at all, so every launched spec was `dev-` (training-only, inadmissible as
# eval) even out of a clean tree -- artifacts/forks_dev/RESULT.md: "distinct spec.dirty values
# across all 68 = {True}". The normal sweep path (`pi_run.cli`, `pi_run.sweep.plan`) computes
# `git_info(root)` ONCE and forwards `code_version=gi.sha, dirty=gi.dirty`; this launcher must
# mirror that exactly.


def test_build_specs_stamps_dirty_false_when_asked():
    pts = [{"trace_sha": "a" * 64, "k": 4, "task_id": "7"}]
    specs = launcher.build_specs(
        pts, suite="tau2_retail", arms=["self_ask"], seeds=[0], dirty=False
    )
    assert all(s.dirty is False for s in specs)


def test_build_specs_stamps_dirty_true_when_asked():
    pts = [{"trace_sha": "a" * 64, "k": 4, "task_id": "7"}]
    specs = launcher.build_specs(pts, suite="tau2_retail", arms=["self_ask"], seeds=[0], dirty=True)
    assert all(s.dirty is True for s in specs)


def test_a_clean_specs_manifest_run_id_is_not_dev_prefixed():
    """The consequence that matters: `dev-` runs are excluded from every reported table
    (`pi_eval.report.ELIGIBLE`), so a clean-tree fork that still gets the prefix is silently
    unusable as an eval result."""
    from pi_run.manifest import build_manifest

    pts = [{"trace_sha": "a" * 64, "k": 4, "task_id": "7"}]
    (spec,) = launcher.build_specs(
        pts, suite="tau2_retail", arms=["self_ask"], seeds=[0], dirty=False, code_version="abc123"
    )
    m = build_manifest(
        suite_id=spec.suite_id,
        task_id=spec.task_id,
        arm_id=spec.arm_id,
        policy_id=spec.arm_id,
        seed=spec.seed,
        corpus_hash="c",
        budget_cap=spec.budget_cap,
        max_turns=spec.max_turns,
        word_cap=180,
        code_version=spec.code_version,
        dirty=spec.dirty,
        pins={},
        foreign_trace_sha=spec.foreign_trace_sha,
        foreign_prefix_k=spec.foreign_prefix_k,
    )
    assert not m.run_id.startswith("dev-")


def test_a_dirty_specs_manifest_run_id_is_dev_prefixed():
    """The contrast that proves the test above would actually catch the bug."""
    from pi_run.manifest import build_manifest

    pts = [{"trace_sha": "a" * 64, "k": 4, "task_id": "7"}]
    (spec,) = launcher.build_specs(
        pts, suite="tau2_retail", arms=["self_ask"], seeds=[0], dirty=True, code_version="abc123"
    )
    m = build_manifest(
        suite_id=spec.suite_id,
        task_id=spec.task_id,
        arm_id=spec.arm_id,
        policy_id=spec.arm_id,
        seed=spec.seed,
        corpus_hash="c",
        budget_cap=spec.budget_cap,
        max_turns=spec.max_turns,
        word_cap=180,
        code_version=spec.code_version,
        dirty=spec.dirty,
        pins={},
        foreign_trace_sha=spec.foreign_trace_sha,
        foreign_prefix_k=spec.foreign_prefix_k,
    )
    assert m.run_id.startswith("dev-")


def test_main_wires_a_clean_git_state_into_every_spec(monkeypatch, tmp_path):
    """THE ACTUAL DEFECT. Not `build_specs()` in isolation -- `main()` never called it with a
    `dirty=` at all. Stub `_git_state` clean and watch what reaches `_run_unit`."""
    from pi_run.manifest import GitInfo

    monkeypatch.setattr(
        launcher, "_git_state", lambda: GitInfo(sha="clean1234abcd", dirty=False, dirty_files=())
    )
    seen = []

    def _capture(spec):
        seen.append(spec)
        return {"status": "ok"}

    monkeypatch.setattr(launcher, "_run_unit", _capture)

    f = tmp_path / "pts.json"
    f.write_text(json.dumps([{"trace_sha": "a" * 64, "k": 10, "task_id": "1"}]))
    rc = launcher.main(
        [
            "--runs-root",
            str(tmp_path / "runsout"),
            "--suite",
            "tau2_retail",
            "--forkpoints",
            str(f),
            "--min-k",
            "0",
            "--max-k",
            "40",
            "--n",
            "1",
            "--arm",
            "self_ask",
            "--seeds",
            "0",
        ]
    )
    assert rc == 0
    assert len(seen) == 1
    assert seen[0].dirty is False
    assert seen[0].code_version == "clean1234abcd"


def test_main_wires_a_dirty_git_state_into_every_spec(monkeypatch, tmp_path):
    """The mirror image: a genuinely dirty tree must reach every spec as dirty=True too --
    this is legitimate (training-only), not something to hide."""
    from pi_run.manifest import GitInfo

    monkeypatch.setattr(
        launcher,
        "_git_state",
        lambda: GitInfo(sha="dirtyfeed1234", dirty=True, dirty_files=("scripts/x.py",)),
    )
    seen = []

    def _capture(spec):
        seen.append(spec)
        return {"status": "ok"}

    monkeypatch.setattr(launcher, "_run_unit", _capture)

    f = tmp_path / "pts.json"
    f.write_text(json.dumps([{"trace_sha": "a" * 64, "k": 10, "task_id": "1"}]))
    rc = launcher.main(
        [
            "--runs-root",
            str(tmp_path / "runsout"),
            "--suite",
            "tau2_retail",
            "--forkpoints",
            str(f),
            "--min-k",
            "0",
            "--max-k",
            "40",
            "--n",
            "1",
            "--arm",
            "self_ask",
            "--seeds",
            "0",
        ]
    )
    assert rc == 0
    assert len(seen) == 1
    assert seen[0].dirty is True
    assert seen[0].code_version == "dirtyfeed1234"


# ------------------------------------- the launcher refuses on its own wiring (coordinator's addition)


def test_the_launcher_refuses_when_specs_come_out_dirty_on_a_clean_tree(monkeypatch, tmp_path):
    """This is the exact defect coming back: a clean tree, but the specs disagree. If
    `build_specs()` (or `UnitSpec`'s own default) ever regresses to ignoring the `dirty` it was
    given, the launcher must refuse rather than silently stamping `dev-` again."""
    import dataclasses

    from pi_run.manifest import GitInfo

    monkeypatch.setattr(
        launcher, "_git_state", lambda: GitInfo(sha="clean1234abcd", dirty=False, dirty_files=())
    )
    real_build_specs = launcher.build_specs
    monkeypatch.setattr(
        launcher,
        "build_specs",
        lambda *a, **k: [dataclasses.replace(s, dirty=True) for s in real_build_specs(*a, **k)],
    )

    def _boom(spec):  # pragma: no cover - reaching it is the failure
        raise AssertionError("must not launch when the wiring self-check fails")

    monkeypatch.setattr(launcher, "_run_unit", _boom)

    f = tmp_path / "pts.json"
    f.write_text(json.dumps([{"trace_sha": "a" * 64, "k": 10, "task_id": "1"}]))
    rc = launcher.main(
        [
            "--suite",
            "tau2_retail",
            "--forkpoints",
            str(f),
            "--min-k",
            "0",
            "--max-k",
            "40",
            "--n",
            "1",
            "--arm",
            "self_ask",
            "--seeds",
            "0",
            "--dry-run",
        ]
    )
    assert rc != 0


def test_the_launcher_refuses_when_a_specs_code_version_disagrees_with_the_tree(
    monkeypatch, tmp_path
):
    """The other half of the same self-check: `code_version` must also agree with `gi.sha`."""
    import dataclasses

    from pi_run.manifest import GitInfo

    monkeypatch.setattr(
        launcher, "_git_state", lambda: GitInfo(sha="clean1234abcd", dirty=False, dirty_files=())
    )
    real_build_specs = launcher.build_specs
    monkeypatch.setattr(
        launcher,
        "build_specs",
        lambda *a, **k: [
            dataclasses.replace(s, code_version="stale-sha-999") for s in real_build_specs(*a, **k)
        ],
    )

    def _boom(spec):  # pragma: no cover - reaching it is the failure
        raise AssertionError("must not launch when the wiring self-check fails")

    monkeypatch.setattr(launcher, "_run_unit", _boom)

    f = tmp_path / "pts.json"
    f.write_text(json.dumps([{"trace_sha": "a" * 64, "k": 10, "task_id": "1"}]))
    rc = launcher.main(
        [
            "--suite",
            "tau2_retail",
            "--forkpoints",
            str(f),
            "--min-k",
            "0",
            "--max-k",
            "40",
            "--n",
            "1",
            "--arm",
            "self_ask",
            "--seeds",
            "0",
            "--dry-run",
        ]
    )
    assert rc != 0


def test_a_genuinely_dirty_tree_launches_normally_not_refused(monkeypatch, tmp_path, capsys):
    """The self-check's other half: a genuinely dirty tree is legitimate -- it just means
    training-only runs -- and must not be refused just because it is dirty."""
    from pi_run.manifest import GitInfo

    monkeypatch.setattr(
        launcher,
        "_git_state",
        lambda: GitInfo(sha="dirtyfeed1234", dirty=True, dirty_files=("scripts/x.py",)),
    )

    def _boom(spec):  # pragma: no cover - --dry-run must not reach this either
        raise AssertionError("--dry-run must not launch a unit")

    monkeypatch.setattr(launcher, "_run_unit", _boom)

    f = tmp_path / "pts.json"
    f.write_text(json.dumps([{"trace_sha": "a" * 64, "k": 10, "task_id": "1"}]))
    rc = launcher.main(
        [
            "--suite",
            "tau2_retail",
            "--forkpoints",
            str(f),
            "--min-k",
            "0",
            "--max-k",
            "40",
            "--n",
            "1",
            "--arm",
            "self_ask",
            "--seeds",
            "0",
            "--dry-run",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "would launch 1 units" in out


# ------------------------------------------------ pool_points(): the wrapped-vs-bare shape


def test_pool_points_accepts_a_bare_list_unchanged():
    """convlog-work's own input format: a plain JSON list. Must pass through untouched --
    this is the shape every pre-existing --forkpoints test in this file already uses."""
    pts = [{"trace_sha": "a" * 64, "k": 4, "task_id": "1"}]
    assert launcher.pool_points(pts) == pts


def test_pool_points_accepts_the_metadata_wrapped_shape_via_its_real_key():
    """conf/forks/tau2_retail_dev.*.json (both the 500-point pool and the 34-point selection)
    are dicts carrying provenance metadata, with the actual point list under `fork_points` --
    not `points`, not `pool`, not `data`. Before the fix, handing this shape to `in_k_window`
    raised `TypeError: string indices must be integers, not 'str'`, because it iterated the
    dict's own keys (strings) instead of the list under `fork_points`."""
    wrapped = {
        "suite": "tau2_retail",
        "n_points": 1,
        "fork_points": [{"trace_sha": "a" * 64, "k": 4, "task_id": "1"}],
    }
    assert launcher.pool_points(wrapped) == wrapped["fork_points"]


def test_pool_points_rejects_an_unrecognized_shape_by_naming_it():
    """Neither a list nor a dict with `fork_points` -- must fail loudly, naming the file, not
    resurface as the same opaque TypeError the fix replaces."""
    with pytest.raises(ValueError, match="whatever.json"):
        launcher.pool_points({"unexpected_key": []}, path="/tmp/whatever.json")


def test_pool_points_loads_the_real_committed_pool_file_without_crashing():
    """The 500-point pool, read-only, exactly as `--forkpoints` would load it."""
    real_file = ROOT / "conf" / "forks" / "tau2_retail_dev.pool.json"
    if not real_file.is_file():
        pytest.skip("committed pool file not present in this checkout")
    raw = json.loads(real_file.read_text())
    pts = launcher.pool_points(raw, path=str(real_file))
    assert len(pts) == 500


def test_pool_points_loads_the_real_committed_selection_file_without_crashing():
    """The 34-point dev selection, read-only -- the exact file and exact defect from
    artifacts/forks_dev/RESULT.md: `TypeError: string indices must be integers, not 'str'`."""
    real_file = ROOT / "conf" / "forks" / "tau2_retail_dev.selected34.json"
    if not real_file.is_file():
        pytest.skip("committed selection file not present in this checkout")
    raw = json.loads(real_file.read_text())
    pts = launcher.pool_points(raw, path=str(real_file))
    assert len(pts) == 34


# ------------------------------------- an explicit --forkpoints selection is not a pool to


def SELECTION_FIXTURE(tmp_path, n=3):
    """A same-shaped-as-the-real-file wrapped selection, small enough to reason about by
    hand: three points at k=2, k=3, k=20 across three tasks."""
    pts = [
        {"trace_sha": "a" * 64, "k": 2, "task_id": "1"},
        {"trace_sha": "b" * 64, "k": 3, "task_id": "2"},
        {"trace_sha": "c" * 64, "k": 20, "task_id": "3"},
    ][:n]
    f = tmp_path / "selection.json"
    f.write_text(json.dumps({"fork_points": pts, "n_points": len(pts)}))
    return f


def test_an_explicit_selection_file_with_no_k_flags_uses_every_point_not_the_pool_defaults(
    tmp_path, capsys
):
    """THE PROPERTY DEFECT 3 EXISTS FOR. When --forkpoints names a wrapped file, its points
    ARE the selection -- the old pool-path defaults (--min-k 8, --n 12) must not silently
    apply and truncate it. All three points here span k=2..20; the old --min-k 8 default
    alone would drop the k=2 and k=3 points before --n ever got a say."""
    f = SELECTION_FIXTURE(tmp_path)
    rc = launcher.main(
        [
            "--suite",
            "tau2_retail",
            "--forkpoints",
            str(f),
            "--arm",
            "self_ask",
            "--seeds",
            "0",
            "--dry-run",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "fork points: 3" in out
    assert "would launch 3 units" in out


def test_an_explicit_selection_file_narrowed_by_min_k_is_refused_not_silently_truncated(
    tmp_path,
):
    """min_k=8 against {2, 3, 20} would silently drop the two shallow points and launch a
    2-point campaign under a --dry-run banner that still says "fork points: 3" nowhere --
    that mismatch is exactly what must never happen again. Refuse instead."""
    f = SELECTION_FIXTURE(tmp_path)
    with pytest.raises(SystemExit):
        launcher.main(
            [
                "--suite",
                "tau2_retail",
                "--forkpoints",
                str(f),
                "--min-k",
                "8",
                "--arm",
                "self_ask",
                "--seeds",
                "0",
                "--dry-run",
            ]
        )


def test_an_explicit_selection_file_narrowed_by_n_is_refused_not_silently_sampled(tmp_path):
    """--n alone, with no k window, still drops a point from an explicit selection and must
    be refused the same way -- --min-k is not the only flag that can narrow it."""
    f = SELECTION_FIXTURE(tmp_path)
    with pytest.raises(SystemExit):
        launcher.main(
            [
                "--suite",
                "tau2_retail",
                "--forkpoints",
                str(f),
                "--n",
                "2",
                "--arm",
                "self_ask",
                "--seeds",
                "0",
                "--dry-run",
            ]
        )


def test_an_explicit_selection_files_refusal_names_the_drop_count_and_the_flag(tmp_path, capsys):
    """Not just a bare SystemExit -- the message must say how many points would be dropped
    and which flag caused it, so the operator can tell a real defect from a deliberate
    narrowing without re-deriving the count by hand. Fixture has 3 points {k=2, k=3, k=20};
    --min-k 8 drops exactly 2 of them (k=2 and k=3)."""
    f = SELECTION_FIXTURE(tmp_path)
    with pytest.raises(SystemExit):
        launcher.main(
            [
                "--suite",
                "tau2_retail",
                "--forkpoints",
                str(f),
                "--min-k",
                "8",
                "--arm",
                "self_ask",
                "--seeds",
                "0",
                "--dry-run",
            ]
        )
    err = capsys.readouterr().err
    assert "2" in err, "must name how many points would be dropped"
    assert "--min-k" in err, "must name the flag that caused the drop"


def test_an_explicit_selection_flag_combination_that_keeps_everything_still_works(tmp_path, capsys):
    """The operator must still be able to pass an explicit, wide flag combination and have it
    succeed -- the refusal is about SILENT narrowing, not about the flags being present at
    all. --min-k 0 --max-k 40 --n 3 keeps all three of {2, 3, 20}."""
    f = SELECTION_FIXTURE(tmp_path)
    rc = launcher.main(
        [
            "--suite",
            "tau2_retail",
            "--forkpoints",
            str(f),
            "--min-k",
            "0",
            "--max-k",
            "40",
            "--n",
            "3",
            "--arm",
            "self_ask",
            "--seeds",
            "0",
            "--dry-run",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "fork points: 3" in out


def test_a_bare_list_forkpoints_file_is_not_treated_as_an_explicit_selection(tmp_path, capsys):
    """The other half of the contrast: a BARE list (convlog-work's own format, not a
    metadata-wrapped selection) keeps the old pool defaults and old silent-sampling
    behaviour -- `test_a_forkpoints_file_still_works_and_still_samples` above already pins
    this, but pins it here too, explicitly contrasted with the wrapped-file tests just above,
    so a future reader sees both halves of the shape-detection together."""
    f = tmp_path / "bare.json"
    f.write_text(
        json.dumps(
            [
                {"trace_sha": "a" * 64, "k": 2, "task_id": "1"},
                {"trace_sha": "b" * 64, "k": 3, "task_id": "2"},
                {"trace_sha": "c" * 64, "k": 20, "task_id": "3"},
            ]
        )
    )
    # the old pool-path default --min-k 8 would drop k=2 and k=3 -- and must, for a bare
    # list, keep doing exactly that rather than refuse.
    rc = launcher.main(
        [
            "--suite",
            "tau2_retail",
            "--forkpoints",
            str(f),
            "--arm",
            "self_ask",
            "--seeds",
            "0",
            "--dry-run",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "fork points: 1" in out, "only k=20 survives the old default --min-k 8 window"


# --------------------------------------------------- the real committed 34-point selection


def test_the_real_committed_selection_file_dry_runs_to_34_points_204_units_with_no_k_flags(
    capsys,
):
    """THE MEASURED REGRESSION FROM artifacts/forks_dev/RESULT.md: the committed 34-point
    dev selection, with no k flags at all, must launch all 34 points x 2 arms x 3 seeds = 204
    units -- not the old pool defaults' 12 points x 2 x 3 = 72."""
    real_file = ROOT / "conf" / "forks" / "tau2_retail_dev.selected34.json"
    if not real_file.is_file():
        pytest.skip("committed selection file not present in this checkout")
    rc = launcher.main(
        [
            "--suite",
            "tau2_retail",
            "--forkpoints",
            str(real_file),
            "--arm",
            "inquirer_prompted",
            "--arm",
            "self_ask",
            "--seeds",
            "0",
            "1",
            "2",
            "--dry-run",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "fork points: 34" in out
    assert "would launch 204 units" in out


def test_the_real_committed_selection_file_refuses_min_k_8_rather_than_returning_12(capsys):
    """THE EXACT DEFECT MEASURED IN artifacts/forks_dev/RESULT.md: `--min-k 8` against this
    file used to silently return 12 points (the old --n default) with no warning that 22 of
    the file's 34 points never got a chance to be sampled at all. Empirically, `--min-k 8`
    alone (default --max-k 40) drops 13 of the 34 points (measured via in_k_window on the
    file's own k values: {2:5, 4:6, 6:2, 8:1, 11:2, 12:1, 13:2, 14:2, 15:1, 16:3, 18:1, 20:1,
    21:3, 25:1, 26:1, 28:1, 31:1} -- 13 of those sit below k=8). Must refuse, not truncate."""
    real_file = ROOT / "conf" / "forks" / "tau2_retail_dev.selected34.json"
    if not real_file.is_file():
        pytest.skip("committed selection file not present in this checkout")
    with pytest.raises(SystemExit):
        launcher.main(
            [
                "--suite",
                "tau2_retail",
                "--forkpoints",
                str(real_file),
                "--min-k",
                "8",
                "--arm",
                "inquirer_prompted",
                "--arm",
                "self_ask",
                "--seeds",
                "0",
                "1",
                "2",
                "--dry-run",
            ]
        )


# ------------------------------------------------------- retrieval k: build_specs() must not
# shadow UnitSpec's own default (coordinator-authorised fix, same defect class as dirty=)


def test_build_specs_defaults_k_to_unitspecs_own_default_when_no_k_is_passed():
    """THE DEFECT: build_specs() hardcoded its own local k=4, silently shadowing UnitSpec's
    own default of 5 -- every fork spec this launcher ever built carried a retrieval-k no
    other caller of UnitSpec uses, in a field nobody passed and no flag could set. Introspects
    UnitSpec's own field default rather than restating a literal, so this test tracks whatever
    UnitSpec's default legitimately is instead of re-encoding the bug it exists to catch."""
    import dataclasses

    from pi_run.worker import UnitSpec

    unitspec_default_k = next(f.default for f in dataclasses.fields(UnitSpec) if f.name == "k")

    pts = [{"trace_sha": "a" * 64, "k": 4, "task_id": "7"}]
    specs = launcher.build_specs(pts, suite="tau2_retail", arms=["self_ask"], seeds=[0])
    assert all(s.k == unitspec_default_k for s in specs)


def test_build_specs_stamps_k_when_asked():
    """The mirror image, same pattern as dirty's `test_build_specs_stamps_dirty_true_when_asked`:
    an explicit k must reach every spec unchanged."""
    pts = [{"trace_sha": "a" * 64, "k": 4, "task_id": "7"}]
    specs = launcher.build_specs(pts, suite="tau2_retail", arms=["self_ask"], seeds=[0], k=5)
    assert all(s.k == 5 for s in specs)


def test_the_204_recorded_runs_k_is_5_and_this_pin_must_fail_if_the_default_ever_changes():
    """THE PIN. The default test above is correct but insufficient alone: if UnitSpec's own
    class default for k ever changes, that test would silently track the new value and keep
    passing -- exactly the way build_specs()'s local k=4 tracked nothing and was never caught.
    Every one of the 204 manifests the recorded retail dev campaign wrote
    (artifacts/forks_dev/RESULT.md, run_ids_sha 1a060380806fff3e) carries k=5. This test
    hardcodes that 5 on purpose, so a future change to either build_specs() or UnitSpec's own
    default is caught here even if the two stay internally consistent with each other."""
    pts = [{"trace_sha": "a" * 64, "k": 4, "task_id": "7"}]
    (spec,) = launcher.build_specs(pts, suite="tau2_retail", arms=["self_ask"], seeds=[0])
    assert spec.k == 5


def test_main_wires_an_explicit_k_flag_into_every_spec(monkeypatch, tmp_path):
    """--k must reach every launched spec, the same way --max-turns/--budget-cap already do."""
    seen = []

    def _capture(spec):
        seen.append(spec)
        return {"status": "ok"}

    monkeypatch.setattr(launcher, "_run_unit", _capture)

    f = tmp_path / "pts.json"
    f.write_text(json.dumps([{"trace_sha": "a" * 64, "k": 10, "task_id": "1"}]))
    rc = launcher.main(
        [
            "--runs-root",
            str(tmp_path / "runsout"),
            "--suite",
            "tau2_retail",
            "--forkpoints",
            str(f),
            "--min-k",
            "0",
            "--max-k",
            "40",
            "--n",
            "1",
            "--arm",
            "self_ask",
            "--seeds",
            "0",
            "--k",
            "5",
        ]
    )
    assert rc == 0
    assert len(seen) == 1
    assert seen[0].k == 5


def test_main_with_no_k_flag_carries_unitspecs_own_default(monkeypatch, tmp_path):
    """No --k at all must not silently fall back to some launcher-local number: it must reach
    UnitSpec's own default, the same value every other caller of UnitSpec gets."""
    import dataclasses

    from pi_run.worker import UnitSpec

    unitspec_default_k = next(f.default for f in dataclasses.fields(UnitSpec) if f.name == "k")

    seen = []

    def _capture(spec):
        seen.append(spec)
        return {"status": "ok"}

    monkeypatch.setattr(launcher, "_run_unit", _capture)

    f = tmp_path / "pts.json"
    f.write_text(json.dumps([{"trace_sha": "a" * 64, "k": 10, "task_id": "1"}]))
    rc = launcher.main(
        [
            "--runs-root",
            str(tmp_path / "runsout"),
            "--suite",
            "tau2_retail",
            "--forkpoints",
            str(f),
            "--min-k",
            "0",
            "--max-k",
            "40",
            "--n",
            "1",
            "--arm",
            "self_ask",
            "--seeds",
            "0",
        ]
    )
    assert rc == 0
    assert len(seen) == 1
    assert seen[0].k == unitspec_default_k


def test_the_effective_retrieval_k_is_printed_once_so_it_is_visible_without_a_manifest(
    runs_root, capsys
):
    """Retrieval k is the field that hid: --dry-run's own per-unit line already prints
    `k=<foreign_prefix_k>`, a DIFFERENT field reusing the same letter, so a human checking a
    dry run against the recorded campaign had no way to see retrieval-k there at all. Printed
    with its own unambiguous label so it cannot be mistaken for foreign_prefix_k's `k=`."""
    launcher.main(
        [
            "--suite",
            "tau2_retail",
            "--split",
            "test",
            "--forkpoints-from-runs",
            str(runs_root),
            "--arm",
            "self_ask",
            "--seeds",
            "0",
            "--k",
            "5",
            "--dry-run",
        ]
    )
    out = capsys.readouterr().out
    assert "retrieval k=5" in out


def test_the_effective_retrieval_k_prints_unitspecs_default_when_no_flag_is_given(
    runs_root, capsys
):
    """The no-flag case must print the value it actually resolved to, not omit the line --
    an operator diffing a dry run against the recorded campaign must be able to see this field
    without opening a single manifest.json."""
    import dataclasses

    from pi_run.worker import UnitSpec

    unitspec_default_k = next(f.default for f in dataclasses.fields(UnitSpec) if f.name == "k")

    launcher.main(
        [
            "--suite",
            "tau2_retail",
            "--split",
            "test",
            "--forkpoints-from-runs",
            str(runs_root),
            "--arm",
            "self_ask",
            "--seeds",
            "0",
            "--dry-run",
        ]
    )
    out = capsys.readouterr().out
    assert f"retrieval k={unitspec_default_k}" in out
