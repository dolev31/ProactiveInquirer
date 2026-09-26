"""Every refusal in scripts/structured_baselines/read_contrasts.py must be able to fire.

One clean population passes; each test mutates ONE field and asserts the matching refusal. The
arm-identity check exists because `arm_id` alone does not name an arm here: `inquirer_prompted`
is run at three inquirer models in three stores, and a population pooled across pins once made a
finding look stable.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "structured_baselines"))

import read_contrasts as rc  # noqa: E402

SPEC = rc.ArmSpec("c_par2", Path("/nonexistent"), Path("/nonexistent"), "par2_rag", "qwen3-8b-base")


def _rows(n_tasks: int = 20) -> list[dict]:
    return [
        {
            "run_id": f"r{t}-{s}",
            "suite_id": "musique",
            "task_id": f"t{t}",
            "seed": s,
            "inquirer_model": "qwen3-8b-base",
            "code_version": "a61c4f4be3e9" + "0" * 28,
            "base_url_sha": "980ca894983a",
            "frozen_prompts": '{"drafter_draft": "e"}',
            "is_dev_run": False,
            "dirty": False,
            "canary_hit": False,
            "firewall_ok": True,
        }
        for t in range(n_tasks)
        for s in (0, 1)
    ]


def test_a_clean_population_passes() -> None:
    assert rc.population_failures(_rows(), SPEC) == []
    assert rc.contrast_failures(_rows(), _rows()) == []


def test_an_empty_arm_is_a_refusal() -> None:
    assert rc.population_failures([], SPEC)


def test_an_arm_pooled_across_inquirer_models_is_a_refusal() -> None:
    rows = copy.deepcopy(_rows())
    rows[3]["inquirer_model"] = "openai/aws/gpt-oss-120b"
    assert any("pooled across pins" in f for f in rc.population_failures(rows, SPEC))


def test_the_wrong_model_everywhere_is_a_refusal() -> None:
    rows = copy.deepcopy(_rows())
    for r in rows:
        r["inquirer_model"] = "openai/aws/claude-opus-5"
    assert any("expected only" in f for f in rc.population_failures(rows, SPEC))


def test_two_code_versions_in_one_arm_is_a_refusal() -> None:
    rows = copy.deepcopy(_rows())
    rows[0]["code_version"] = "deadbeef" * 5
    assert any("code_versions" in f for f in rc.population_failures(rows, SPEC))


def test_dev_dirty_canary_and_firewall_each_refuse() -> None:
    for field, value, word in (
        ("is_dev_run", True, "dev-"),
        ("dirty", True, "dirty"),
        ("canary_hit", True, "canary"),
        ("firewall_ok", False, "firewall"),
    ):
        rows = copy.deepcopy(_rows())
        rows[1][field] = value
        assert any(word in f for f in rc.population_failures(rows, SPEC)), field


def test_two_runs_at_one_cell_is_a_refusal() -> None:
    rows = copy.deepcopy(_rows())
    dup = copy.deepcopy(rows[0])
    dup["run_id"] = "r0-0-rerun"
    rows.append(dup)
    assert any("more than one run" in f for f in rc.population_failures(rows, SPEC))


def test_a_contrast_across_code_versions_base_urls_or_frozen_prompts_refuses() -> None:
    for field, value, word in (
        ("code_version", "f" * 40, "code_versions"),
        ("base_url_sha", "e6436030e8da", "base_url_shas"),
        ("frozen_prompts", '{"drafter_draft": "X"}', "frozen roles"),
    ):
        other = copy.deepcopy(_rows())
        for r in other:
            r[field] = value
        assert any(word in f for f in rc.contrast_failures(_rows(), other)), field


def test_the_instrument_check_needs_agreement_not_just_no_disagreement() -> None:
    """A store with zero comparable rows must not read as a passed instrument lock."""
    empty = {"agree": 0, "differ": 0, "emitted_here_absent_in_store": 0, "in_store_absent_here": 0}
    assert not rc.instrument_ok(empty)
    assert rc.instrument_ok({**empty, "agree": 5})
    assert not rc.instrument_ok({**empty, "agree": 5, "differ": 1})
    assert not rc.instrument_ok({**empty, "agree": 5, "in_store_absent_here": 1})


def test_parse_arm_keeps_a_model_id_that_contains_colons_and_slashes() -> None:
    spec = rc.parse_arm("d=/a/store:/a/runs:inquirer_prompted:openai/aws/claude-opus-5")
    assert (spec.label, spec.arm_id, spec.inquirer_model) == (
        "d",
        "inquirer_prompted",
        "openai/aws/claude-opus-5",
    )


def test_a_pool_needs_two_known_members(monkeypatch) -> None:
    """A pool of one seed, or of an unknown label, or reusing an arm's label, is refused (exit 3)."""
    monkeypatch.setenv("PI_GOLD_ROOT", "/nonexistent")
    arm = "s1=/x:/y:inquirer_trained:qwen3-8b-dpo-stacked-notdone-both-s1"
    for pool in ("rec=s1", "rec=s1,nosuch", "s1=s1,s1"):
        rc_ = rc.main(["--arm", arm, "--contrast", "s1,s1", "--pool", pool, "--out", "/tmp/x.json"])
        assert rc_ == 3, pool


def test_a_pool_on_the_b_side_is_a_refusal() -> None:
    """Pooling averages the A side's pairs within the task; a pooled comparator has no pairing."""
    import pytest

    specs = [rc.parse_arm(f"{k}=/x:/y:inquirer_trained:m{k}") for k in ("s1", "s2", "base")]
    with pytest.raises(rc.Refusal):
        rc.run(
            specs,
            [("base", "rec")],
            cohort=Path("/nonexistent"),
            graph_version="v1",
            tol=1e-9,
            pools={"rec": ["s1", "s2"]},
        )
