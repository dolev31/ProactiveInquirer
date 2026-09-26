"""Adding `foreign_*` to SEMANTIC_FIELDS must rename NO run that has already finished.

WHY THIS FILE EXISTS. `run_id` = h("runid", semantic_hash, counterfactual_kind, prefix_k)[:32],
and `semantic_hash` is `h("run", canon({k: fields[k] for k in SEMANTIC_FIELDS}))`. `canon` does
NOT drop a None-valued key -- `canon({"a": 1, "b": None})` is `{"a":1,"b":null}`, which is not
`{"a":1}` -- so merely NAMING a new field in SEMANTIC_FIELDS moves the hash of every run that
predates it. That already happened once: the comment in `ids.semantic_hash` records 736 of 736
runs moving when the branch triple was added, which is why the explicit pop loop is there.

So `foreign_trace_sha` and `foreign_prefix_k` join that pop loop rather than merely the field
list, and this file pins the consequence against seven manifests ALREADY ON DISK -- three
non-forks in main's manifest shape (keys absent), one non-fork in the branch's (keys present and
null), and three forks the paired-fork report cites -- by recomputing each one's run_id from its
own recorded fields and comparing with the id in the file, which is also the name of its run
directory. Both shapes are pinned because they are written by different code and must agree.

MEASURED OVER THE WHOLE CORPUS BEFORE THE FIELDS WERE ADDED, not just these seven. Over all
77,109 manifests under `runs/`:

    non-fork, reproduced by main's rule and by this one        49,408
    fork, reproduced by this rule only (main cannot see them)   5,096
    reproduced by NEITHER                                      22,605

    runs main reproduces that this rule does not:                   0
    fork runs this rule fails to reproduce:                         0

The 22,605 are exactly the `rehydrated_from` stubs -- run directories reconstructed from
`scores/parquet` after the `rm -rf`, which carry no `pins`, no `k` and no `upstream_pins` and so
cannot reproduce their own id under ANY rule. 0 of them reproduce under either; 0 manifests in
`manifest_to_dict` shape fail under both. That partition is why the two hashings can be
reproduced by ONE rule instead of a choice between them.

THE SHARP EDGE, pinned below. `None` is popped, `""` is NOT. A fork manifest whose
`foreign_trace_sha` were written as `""` rather than None would hash as neither a fork nor a
non-fork, so the writers must keep passing None. `manifest_to_dict` writes the field through
unchanged, and `test_an_empty_string_is_not_the_same_as_absent` is what would catch a writer
that starts coercing it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pinq.ids import SEMANTIC_FIELDS, canon, h, semantic_hash
from pinq.types import ModelPin, RunManifest

FIXTURES = Path(__file__).parent / "fixtures" / "run_manifests"

# Non-forks in MAIN'S manifest shape: `manifest_to_dict` never emitted the foreign_* keys, so
# they are ABSENT from the file rather than null. This is the shape the hash rule must not move.
NON_FORKS = (
    "nonfork_musique_dev.json",
    "nonfork_strategyqa_train.json",
    "nonfork_tau2_retail.json",
)
# A non-fork in the BRANCH'S shape: the keys are present and null. Kept as its own case because
# it is the one that exercises the pop directly -- for the three above there is nothing in the
# input to pop, so they would pass even if the pop loop were wrong about None.
#
# WHY IT IS A BANKING RUN AND NOT A SECOND RETAIL ONE, measured over all 77,109 manifests: of
# the `tau2` (banking) directories in full `manifest_to_dict` shape, 236 of 236 carry the keys
# and 0 do not -- every surviving full-shape banking run was written by the branch. There is no
# main-shape banking manifest on disk to use, so the tau2-family main-shape slot above is filled
# by `tau2_retail` (622 such runs) and the banking one appears here instead.
BRANCH_SHAPE_NON_FORK = "nonfork_tau2_banking_branchshape.json"
FORKS = (
    "fork_tau2_retail_test_inquirer_prompted_s0.json",
    "fork_tau2_retail_test_self_ask_s0.json",
    "fork_tau2_retail_test_inquirer_prompted_s2.json",
)
ALL = NON_FORKS + (BRANCH_SHAPE_NON_FORK,) + FORKS


def _manifest(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def _run_id_from_fields(d: dict) -> str:
    """The run_id `RunManifest.run_id` would produce, computed from the recorded fields alone.

    Driven off the stored dict rather than a reconstructed dataclass on purpose: the dict is
    what `pi compact`, `report_forks.py` and every debugging human read, and a rule that holds
    for the dataclass but not for the file it wrote would be a rule about the wrong artifact.
    `test_the_dataclass_agrees_with_the_recorded_fields` closes that gap from the other side.
    """
    sem = semantic_hash(d)
    core = h("runid", sem, d.get("counterfactual_kind", "none"), str(d.get("prefix_k")))
    return ("dev-" if d.get("dirty") else "") + core[:32]


def _rebuild(d: dict) -> RunManifest:
    """The dataclass the recorded fields describe. Only the semantic fields are restored."""
    return RunManifest(
        suite_id=d["suite_id"],
        task_id=d["task_id"],
        arm_id=d["arm_id"],
        policy_id=d.get("policy_id", ""),
        seed=d["seed"],
        split=d["split"],
        corpus_hash=d["corpus_hash"],
        budget_cap=d["budget_cap"],
        max_turns=d["max_turns"],
        word_cap=d.get("word_cap", 0),
        pins={
            r: ModelPin(
                role=p["role"],
                model_id=p["model_id"],
                provider=p["provider"],
                base_url_sha=p.get("base_url_sha", ""),
                adapter_sha=p.get("adapter_sha"),
                system_prompt_sha=p.get("system_prompt_sha", ""),
                sampling_sha=p.get("sampling_sha", ""),
                price_table_version=p.get("price_table_version", ""),
            )
            for r, p in (d.get("pins") or {}).items()
        },
        prompt_hashes=dict(d.get("prompt_hashes") or {}),
        upstream_pins=dict(d.get("upstream_pins") or {}),
        pilot_flag=d.get("pilot_flag", False),
        exploratory=d.get("exploratory", False),
        counterfactual_kind=d.get("counterfactual_kind", "none"),
        prefix_of_run_id=d.get("prefix_of_run_id"),
        prefix_k=d.get("prefix_k"),
        branch_of_run_id=d.get("branch_of_run_id"),
        branch_turn_idx=d.get("branch_turn_idx"),
        branch_seed=d.get("branch_seed"),
        foreign_trace_sha=d.get("foreign_trace_sha"),
        foreign_prefix_k=d.get("foreign_prefix_k"),
        prompt_variant_id=d.get("prompt_variant_id", "v1"),
        k=d.get("k", 0),
        questions_hash=d.get("questions_hash", ""),
        code_version=d.get("code_version", ""),
        dirty=d.get("dirty", False),
    )


# --------------------------------------------------------------- the seven recomputations


@pytest.mark.parametrize("name", ALL)
def test_the_recorded_run_id_is_reproduced(name: str) -> None:
    """THE PIN. Seven ids already on disk, each recomputed from its own fields.

    A failure here means the ported hash rule renamed a finished run -- the thing
    `conf/checkpoints.json::_why_it_is_empty` and
    `test_checkpoint_registry.py::test_no_registry_row_RENAMES_A_RUN_THAT_HAS_ALREADY_FINISHED`
    both exist to forbid, arriving through a different field.
    """
    d = _manifest(name)
    assert _run_id_from_fields(d) == d["run_id"]


@pytest.mark.parametrize("name", ALL)
def test_the_recorded_semantic_hash_is_reproduced(name: str) -> None:
    """One level below the run_id, so a failure names the hash rather than the truncation."""
    d = _manifest(name)
    assert semantic_hash(d) == d["semantic_hash"]


@pytest.mark.parametrize("name", ALL)
def test_the_dataclass_agrees_with_the_recorded_fields(name: str) -> None:
    """`RunManifest` rebuilt from the file reproduces the file's own ids.

    This is what catches a field added to `SEMANTIC_FIELDS` and NOT to the dict
    `RunManifest.semantic_hash` builds -- the failure mode `ids.semantic_hash`'s docstring
    records for `k` and `questions_hash`, where the field was named and silently did not count.
    """
    d = _manifest(name)
    m = _rebuild(d)
    assert m.model_pin_hash == d["model_pin_hash"]
    assert m.semantic_hash == d["semantic_hash"]
    assert m.run_id == d["run_id"]


# ------------------------------------------------------------------ why one rule suffices


@pytest.mark.parametrize("name", NON_FORKS)
def test_a_non_fork_hashes_exactly_as_it_did_before_the_fields_existed(name: str) -> None:
    """The load-bearing half: naming the fields must not move a run that has none of them.

    Computed the hard way -- the fields deleted from the input entirely, which is literally
    what these manifests looked like to the code that wrote them, since `manifest_to_dict` did
    not emit the keys at all.
    """
    d = _manifest(name)
    assert "foreign_trace_sha" not in d, f"{name} is not a main-shape non-fork fixture"
    assert _run_id_from_fields(d) == d["run_id"]
    assert (
        _run_id_from_fields(dict(d, foreign_trace_sha=None, foreign_prefix_k=None)) == d["run_id"]
    ), "a None-valued foreign field must be popped, not hashed as null"


def test_a_non_fork_written_with_the_keys_present_and_null_hashes_the_same_way() -> None:
    """The branch's own shape. `canon` would serialise a present null and move the hash, so
    this is the case the pop loop is actually FOR -- and the one the three fixtures above
    cannot see, because they have no key to pop."""
    d = _manifest(BRANCH_SHAPE_NON_FORK)
    assert "foreign_trace_sha" in d and d["foreign_trace_sha"] is None
    assert d["foreign_prefix_k"] is None
    assert _run_id_from_fields(d) == d["run_id"]
    stripped = {k: v for k, v in d.items() if k not in ("foreign_trace_sha", "foreign_prefix_k")}
    assert _run_id_from_fields(stripped) == d["run_id"], (
        "present-and-null must hash identically to absent, or the two writers disagree"
    )


@pytest.mark.parametrize("name", FORKS)
def test_a_fork_stripped_of_its_foreign_fields_collapses_onto_a_non_fork(name: str) -> None:
    """The other half: the fields DO count when present, which is the whole reason to add them.

    Two forks of one task differ in nothing else -- same suite, arm, seed, task, corpus,
    prompts, code -- so without these they share a run_id, share a run directory, and `--resume`
    keeps only the first.
    """
    d = _manifest(name)
    assert d["foreign_trace_sha"], f"{name} is not a fork fixture"
    stripped = {k: v for k, v in d.items() if k not in ("foreign_trace_sha", "foreign_prefix_k")}
    assert _run_id_from_fields(stripped) != d["run_id"]


def test_two_cuts_of_one_trace_are_two_runs() -> None:
    """The failure the fields prevent, stated directly: same task, same trace, different k."""
    d = _manifest(FORKS[0])
    at_k = _run_id_from_fields(d)
    other = _run_id_from_fields(dict(d, foreign_prefix_k=int(d["foreign_prefix_k"]) - 1))
    assert at_k != other
    assert at_k == d["run_id"]


def test_two_traces_cut_at_one_k_are_two_runs() -> None:
    """And the same for the trace, which is the field a k-only identity would drop."""
    d = _manifest(FORKS[0])
    other = _run_id_from_fields(dict(d, foreign_trace_sha="0" * 64))
    assert other != d["run_id"]


# -------------------------------------------------------------------------- the sharp edge


def test_canon_does_not_drop_none_so_the_pop_must_be_explicit() -> None:
    """The reason the pop loop exists at all. Stated here so a future reader who assumes
    `canon` is None-eliding is corrected by a test rather than by a renamed corpus."""
    assert canon({"a": 1}) == '{"a":1}'
    assert canon({"a": 1, "b": None}) == '{"a":1,"b":null}'
    assert canon({"a": 1}) != canon({"a": 1, "b": None})


def test_an_empty_string_is_not_the_same_as_absent() -> None:
    """`""` is NOT popped, and must therefore never be written where None is meant.

    A writer that coerced `foreign_trace_sha` to `""` for a non-fork would give every such run
    a third identity -- neither the historical one nor a fork's -- and rename the entire
    non-fork corpus. `manifest_to_dict` passes the field through unchanged; this is the test
    that fails if that ever becomes `str(... or "")`.
    """
    d = _manifest(NON_FORKS[0])
    assert _run_id_from_fields(dict(d, foreign_trace_sha=None)) == d["run_id"]
    assert _run_id_from_fields(dict(d, foreign_trace_sha="")) != d["run_id"]


def test_the_fields_are_named_in_semantic_fields() -> None:
    """Directly, because every test above would also pass if the fields were simply ignored."""
    assert "foreign_trace_sha" in SEMANTIC_FIELDS
    assert "foreign_prefix_k" in SEMANTIC_FIELDS


def test_the_fork_fixtures_are_the_runs_the_paired_report_cites() -> None:
    """Provenance: these are rows of `docs/reports/forks_tau2_retail_test.json`, not samples
    invented for a test. Their ids are directory names under `runs/`."""
    for name in FORKS:
        d = _manifest(name)
        assert d["suite_id"] == "tau2_retail"
        assert d["split"] == "test"
        assert d["arm_id"] in ("inquirer_prompted", "self_ask")
        assert d["seed"] in (0, 1, 2)
        assert not d["dirty"], "a dev- run is excluded from every reported table"
