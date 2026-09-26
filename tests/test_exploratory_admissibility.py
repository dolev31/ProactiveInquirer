"""An exploratory run must be INADMISSIBLE, the way a pilot already is.

MEASURED ON THIS REPOSITORY, after the tier0 canary ran:

    exploratory column in runs.parquet ......... absent
    canary runs passing the ELIGIBLE predicate .. 60 of 66
    musique/inquirer_prompted cells ............ 3 runs at max_turns=1 (the canary)
                                                 14 runs at max_turns=16
                                                 -- in the SAME (suite, arm) cell, averaged

`conf/grids/tier0_canary.yaml` says `exploratory: true`, and that flag reached the console
banner and nothing else: not the manifest, not the parquet, not the eligibility predicate.

This is the pilot_flag failure exactly. `RunManifest.pilot_flag`'s own comment says a pilot
exists so its results can be LOOKED AT before the confirmatory analysis is fixed, which is
what makes them inadmissible afterwards. A canary is the same object: it exists to be looked
at. Sixty of them were one query away from a published mean.
"""

from __future__ import annotations

import dataclasses


def _manifest(**kw):
    from pinq.types import RunManifest

    base = dict(
        suite_id="musique",
        task_id="t",
        arm_id="inquirer_prompted",
        policy_id="p",
        seed=7,
        split="train",
        corpus_hash="c",
        budget_cap=1,
        max_turns=1,
        word_cap=30,
    )
    base.update(kw)
    return RunManifest(**base)


def test_the_manifest_carries_the_flag() -> None:
    assert _manifest().exploratory is False
    assert _manifest(exploratory=True).exploratory is True


def test_the_flag_is_inside_run_identity() -> None:
    """Same reason pilot_flag is: otherwise an exploratory and a confirmatory run of one
    cell share a run_id, a directory and a --resume sentinel, and the exploratory one is
    published as confirmatory."""
    a = _manifest(exploratory=False)
    b = _manifest(exploratory=True)
    assert a.semantic_hash != b.semantic_hash


def test_it_is_serialized_to_the_manifest_file() -> None:
    from pi_run.manifest import manifest_to_dict

    assert manifest_to_dict(_manifest(exploratory=True))["exploratory"] is True


def test_it_reaches_the_parquet_schema() -> None:
    from pi_eval.schema import RUNS

    assert "exploratory" in {f.name for f in RUNS}


def test_compaction_carries_it() -> None:
    from pi_run.compact import _run_row

    row = _run_row(
        {"run_id": "r", "suite_id": "s", "task_id": "t", "arm_id": "a", "exploratory": True},
        {"status": "ok"},
        {},
    )
    assert row["exploratory"] is True


def test_the_eligibility_predicate_excludes_it() -> None:
    """The whole point. Without this line the flag is decoration."""
    from pi_eval.report import ELIGIBLE

    assert "exploratory = FALSE" in ELIGIBLE.replace("r.", "")


def test_the_grid_flag_reaches_the_unit_spec() -> None:
    from pi_run.worker import UnitSpec

    assert "exploratory" in {f.name for f in dataclasses.fields(UnitSpec)}


def test_the_canary_grid_still_declares_itself_exploratory() -> None:
    import yaml

    assert yaml.safe_load(open("conf/grids/tier0_canary.yaml"))["exploratory"] is True


def test_a_confirmatory_grid_does_not() -> None:
    """The exemption must not quietly cover the grids that carry the claims."""
    import yaml

    for name in ("tier1_confirmatory", "tier2_confirmatory"):
        g = yaml.safe_load(open(f"conf/grids/{name}.yaml"))
        assert not g.get("exploratory"), name
