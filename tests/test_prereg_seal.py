"""The seal edits: write-once fields that cannot be fixed after Gate 4.

Every number quoted here was MEASURED on this repo, not assumed. The commands are in the
commit message; the findings are:

  * gold node depth, per suite -- drgym 0/16156 at depth>=2, tau2 0/5790, wiki2 0/31120,
    against musique 664/2660 and strategyqa 679/6720. A POOLED `cad_ge2` endpoint therefore
    draws on three of six suites, one of which is `synth`.
  * pooled `evidence_coverage` rows by suite -- musique 176, synth 69, strategyqa 5,
    wiki2 2. `synth` is the CALIBRATION suite: closed-form metrics, zero tokens. It was
    contributing 28% of the rows to a confirmatory endpoint.
  * grid seeds against the sealed tuple -- sealed (0,1); tier2_confirmatory (tau2) and
    tier1_trained run (0,1,2); tier1_pilot, tier1_oracle and tier3_drgym (drgym) run (0,).
    Two of the three PRIMARY suites contradicted the seal.
"""

from __future__ import annotations

import pytest

from pi_eval import prereg

# --------------------------------------------------------------------- pooling is declared


def test_every_pooled_endpoint_names_the_suites_it_pools() -> None:
    """ "Pooled" with no list is a claim whose denominator changes when a suite is added."""
    for e in prereg.all_endpoints():
        if e.suite_id == prereg.POOLED:
            assert e.pool_suites, f"{e.key} pools silently"


def test_the_calibration_suite_is_never_pooled_into_a_confirmatory_endpoint() -> None:
    """synth has closed-form metrics and zero tokens; it checks the pipeline, not the thesis.

    Measured: it was 69 of 252 rows (27.4%) of pooled `evidence_coverage`.
    """
    for e in prereg.all_endpoints():
        if e.suite_id == prereg.POOLED and e.claim == "confirmatory":
            assert "synth" not in e.pool_suites, f"{e.key} pools the calibration suite"


def test_a_pooled_endpoint_never_names_a_suite_that_cannot_compute_it() -> None:
    for e in prereg.all_endpoints():
        if e.suite_id != prereg.POOLED:
            continue
        for suite in e.pool_suites:
            assert not prereg.not_computable(suite, e.metric), (
                f"{e.key} pools {suite}, which structurally cannot produce {e.metric}"
            )


def test_not_computable_records_the_measured_reason() -> None:
    """A bare exclusion list is unauditable; each entry states what was counted."""
    assert prereg.not_computable("drgym", "cad_ge2")
    assert prereg.not_computable("tau2", "cad_ge2")
    assert prereg.not_computable("wiki2", "cad_ge2")
    assert not prereg.not_computable("musique", "cad_ge2")
    for (suite, metric), reason in prereg.NOT_COMPUTABLE.items():
        assert len(reason) > 30, f"{suite}:{metric} has no stated reason"


# ------------------------------------------------------------------------------- seeds


def test_seeds_are_declared_per_suite() -> None:
    """A single global tuple cannot be true of a design whose suites differ in cost."""
    for e in prereg.PRIMARY:
        assert prereg.seeds_for(e.suite_id), f"no seeds declared for {e.suite_id}"


def test_the_sealed_seeds_match_the_grids_that_will_run() -> None:
    """The drift this catches was live: two of three primaries disagreed with the seal."""
    from pathlib import Path

    import yaml

    by_suite: dict[str, set[int]] = {}
    for p in sorted(Path("conf/grids").glob("*.yaml")):
        g = yaml.safe_load(p.read_text()) or {}
        # `suites` is PLURAL in every grid file. Reading `suite` collected nothing and made
        # this test pass without testing -- caught by the assert below, which is the only
        # reason it is here.
        if g.get("exploratory"):
            continue  # an exploratory grid is not preregistered and declares no seeds
        seeds = g.get("seeds") or ()
        # NOTE: the `exploratory` GRID flag and an exploratory SUITE are different facts, and
        # conflating them is what this loop used to do. The grid flag means "these runs are
        # data generation, not measurement" (`latent_trainset.yaml`) and excludes them from
        # `report.ELIGIBLE`. A suite's `endpoint_status` means "the seal does not name it, so
        # it may not carry a p-value". `frames` is the second without being the first: its
        # runs are measurements and must reach a table, and the seal names no seeds for it
        # because it was chosen long after the seal was written. The filter for THIS test is
        # the second fact, and it is applied per suite below rather than per grid here.
        for suite in g.get("suites") or ():
            by_suite.setdefault(str(suite), set()).update(int(x) for x in seeds)

    assert {"musique", "tau2", "drgym"} <= set(by_suite), (
        f"the three PRIMARY suites must appear in the grids; found {sorted(by_suite)}"
    )
    from pi_run.suites import REGISTRY

    checked: set[str] = set()
    for suite, grid_seeds in sorted(by_suite.items()):
        spec = REGISTRY.get(suite)
        if spec is not None and spec.endpoint_status != "confirmatory":
            # Not named in the preregistration, so it declares no seeds and can carry no
            # p-value. Skipping it here is not a loosening: the assert below still requires
            # every CONFIRMATORY suite to have sealed seeds, and the anti-vacuity assert above
            # still requires the three primaries to be present.
            continue
        declared = set(prereg.seeds_for(suite))
        # STRICTER THAN BEFORE. A grid with no `seeds:` key yields an empty set, and
        # `set() <= set()` passed a confirmatory suite whose seeds had vanished from the seal.
        assert declared, (
            f"{suite}: declared confirmatory in the registry, but the seal declares no seeds "
            f"for it -- either the seal is stale or the suite may not carry a p-value"
        )
        assert grid_seeds <= declared, (
            f"{suite}: grids run seeds {sorted(grid_seeds)} but the seal declares "
            f"{sorted(declared)} -- a run at an undeclared seed is not preregistered"
        )
        checked.add(suite)

    # ANTI-VACUITY, DERIVED. CHANGED DELIBERATELY, 2026-09-15, T20: this named
    # {"musique", "tau2", "drgym"} -- the PROMPTED era's three primaries -- and the registry
    # now calls tau2 and drgym exploratory, so the loop skips them on purpose and the typed
    # set asserted that a correct skip had not happened. The property the assert is actually
    # for is that the loop compared SOMETHING and skipped nothing it should have checked, and
    # that follows from the registry rather than from three names.
    confirmatory = {
        s
        for s in by_suite
        if (spec := REGISTRY.get(s)) is not None and spec.endpoint_status == "confirmatory"
    }
    assert checked, "the seed comparison ran on no suite at all"
    assert confirmatory <= checked, (
        f"every suite the registry calls confirmatory must reach the seed comparison; "
        f"confirmatory {sorted(confirmatory)}, checked {sorted(checked)}"
    )


# ------------------------------------------------------------------- claim rule and family


def test_the_claim_rule_is_sealed_and_says_per_suite() -> None:
    s1 = prereg.default_stage1()
    assert s1.claim_rule, "the multiplicity rule must be sealed BEFORE the numbers exist"
    assert "per-suite" in s1.claim_rule.lower()


def test_the_fdr_family_size_is_fixed_not_derived_from_what_ran() -> None:
    """An unrun endpoint is a non-rejection, not a smaller family."""
    s1 = prereg.default_stage1()
    assert s1.fdr_family_size == len(prereg.SECONDARY)
    assert s1.fdr_family_size > 0


def test_a_qualitative_rule_is_sealed() -> None:
    """Case selection after seeing the numbers is how a qualitative section becomes cherry-picking."""
    assert prereg.default_stage1().qualitative_rule


# --------------------------------------------------------------------- burned pilot ids


def test_burned_pilot_ids_are_produced_not_defaulted_to_empty() -> None:
    """`{}` is a formal declaration that no pilot was ever run on these tasks."""
    import scripts.seal_prereg as sp

    assert hasattr(sp, "_load_burned_pilots"), "nothing produces burned_pilot_ids"


def test_sealing_refuses_when_a_pilot_ran_and_the_list_is_empty() -> None:
    import scripts.seal_prereg as sp

    with pytest.raises(Exception) as exc:
        sp._check_burned_pilots({"musique": ()}, pilot_ran={"musique"})
    assert "pilot" in str(exc.value).lower()


# ------------------------------------------------------- the declaration must actually bind


def test_report_carries_pool_suites_onto_the_contrast() -> None:
    """A declared pool that report.py ignores is decorative."""
    from pi_eval.report import Contrast

    e = next(x for x in prereg.all_endpoints() if x.metric == "cad_ge2")
    c = Contrast.of(e)
    assert c.pool_suites == e.pool_suites


def test_the_pooled_query_filters_to_the_declared_suites() -> None:
    """The SQL must name them; otherwise `synth` is still in the denominator."""
    from pi_eval.report import Agg, metric_query

    agg = Agg.__new__(Agg)
    for field, value in (
        ("scorer_hash", "deadbeef"),
        ("allow_contaminated", False),
        ("excluded_pairs", frozenset()),
        ("excluded_run_ids", frozenset()),
    ):
        object.__setattr__(agg, field, value)
    sql = metric_query(agg, "cad_ge2", suite=prereg.POOLED, pool_suites=("musique", "strategyqa"))
    assert "musique" in sql and "strategyqa" in sql
    assert "synth" not in sql
    # and an unpooled query is unchanged
    assert "IN (" not in metric_query(agg, "cad_ge2", suite="musique")


def test_pull_threads_pool_suites_into_the_sql_it_runs() -> None:
    """The end-to-end binding: a POOLED contrast's query must carry its declared pool.

    Checking `metric_query` alone proves only that the parameter exists. This asserts the
    filter reaches the SQL that `pull` actually executes, which is where `synth` was
    entering the denominator.
    """
    from pi_eval.report import Agg, pull

    captured: list[str] = []

    class Spy(Agg):  # Agg uses slots, so `sql` cannot be patched onto an instance
        def sql(self, query: str):  # type: ignore[override]
            captured.append(query)
            return []

    agg = Spy.__new__(Spy)
    for field, value in (
        ("scorer_hash", "deadbeef"),
        ("allow_contaminated", False),
        ("excluded_pairs", frozenset()),
        ("excluded_run_ids", frozenset()),
    ):
        object.__setattr__(agg, field, value)

    pull(agg, "rnr_resolve", suite=prereg.POOLED, pool_suites=("musique", "tau2"))
    assert captured, "pull ran no query"
    assert "musique" in captured[0] and "tau2" in captured[0]
    assert "synth" not in captured[0]


def test_the_refusal_fires_when_burned_is_empty_BECAUSE_that_is_the_failure_mode() -> None:
    """The call site, not the helper.

    The first version computed `pilot_ran` as `pilot_suites & set(burned)`, which is empty
    exactly when `burned` is empty -- so the guard could never fire in the one case it
    exists for.

    CHANGED DELIBERATELY, 2026-09-15, T20, and the change COSTS COVERAGE, which is recorded
    here rather than quietly dropped. The second argument was `executed` (any compacted run on
    the pilot's own tasks) and is now `piloted` (`pilot_flag = TRUE` only), because the old
    signal false-positived on real data: 3,830 runs from fourteen other grid labels sit on the
    pilot slice's 120 musique ids and none of them is a pilot, so the seal refused over a
    pilot that never happened.

    The failure mode this test's title names -- "a pilot grid ran and `pilot_flag` did not
    survive compaction" -- is therefore NO LONGER DETECTABLE HERE: both sides now read the
    same flag, so the refusal below is a coherence check between two reads of it. What the
    seal rests on instead is `_check_trained_arm_not_on_test`, which asks whether a trained
    arm has already been run on the confirmatory population; see
    tests/test_prereg.py::test_a_trained_arm_run_on_a_confirmatory_test_id_is_found.
    """
    import scripts.seal_prereg as sp

    slices = {"musique": {"t200", "t201"}}
    ran = sp._pilots_that_ran(pilot_slices=slices, piloted={"musique": {"t200", "t999"}})
    assert ran == {"musique"}, "a pilot whose OWN tasks ran must be checked even if burned is empty"

    with pytest.raises(sp.BurnedPilotsMissing):
        sp._check_burned_pilots({}, ran)

    # The false positive the old signal produced, now the case that must NOT fire: other
    # grids' runs on the pilot's own tasks, and runs elsewhere in the suite.
    assert sp._pilots_that_ran(pilot_slices=slices, piloted={"musique": {"t0", "t1"}}) == set()
    assert sp._pilots_that_ran(pilot_slices=slices, piloted={}) == set()


# ------------------------------------------------------------------- the ancestor endpoints


def test_the_additional_ancestors_are_declared() -> None:
    """self_ask and ircot both run in tier1_confirmatory at n=200 x 2 seeds on musique, so
    these endpoints cost ZERO extra rollouts. Without them the abstract's ancestor claim has
    no preregistered test and no rendered interval anywhere."""
    keys = {e.key for e in prereg.all_endpoints()}
    for comparator in ("self_ask", "ircot"):
        assert f"musique:answer_token_f1:inquirer_prompted-vs-{comparator}" in keys


def test_every_musique_quality_contrast_has_a_recall_companion() -> None:
    """MEASURED, and the reason this is preregistered rather than merely reported:

    against ircot, `answer_token_recall` is TIED at 0.500 while `answer_token_f1` leads
    0.433 to 0.359 -- the entire F1 gap is precision, i.e. concision. Declaring F1 alone
    would let "we beat the ancestor" mean "we were shorter". Recall is the half padding
    cannot manufacture, so it is what separates discovery from brevity.
    """
    f1 = {
        e.contrast
        for e in prereg.all_endpoints()
        if e.suite_id == "musique" and e.metric == "answer_token_f1"
    }
    rec = {
        e.contrast
        for e in prereg.all_endpoints()
        if e.suite_id == "musique" and e.metric == "answer_token_recall"
    }
    assert f1, "no musique f1 contrast found"
    assert f1 <= rec, f"f1 contrasts without a recall companion: {sorted(f1 - rec)}"


def test_the_fdr_family_grew_with_the_new_endpoints() -> None:
    """The family size is FIXED, so adding endpoints must move it -- a stale literal here
    would silently give every secondary test more power than it earned."""
    assert prereg.default_stage1().fdr_family_size == len(prereg.SECONDARY)


# ------------------------------------------------------------------------- S5, re-pointed


def test_parallel_replay_is_not_sealed_as_a_sequencing_test() -> None:
    """Its delta is exactly zero BEFORE any data exists, so it can only ever "match".

    `ParallelReplayInquirer` emits the recorded run's question strings without looking at
    what the previous one returned. The same strings retrieve the same uids, and `draft()`
    and the Answerer are pure in the evidence subset hash, so the answer text is identical
    too -- proven LLM-free on synth by
    tests/test_prereg.py::test_parallel_replay_is_input_identical_to_the_arm_it_replays,
    and observed here on 2 of 2 paired musique cells (identical subset_hash AND identical
    answer_evidence_hash).

    A kill switch that fires with certainty is not evidence. "Drop the graph framing" must
    not hang off it.
    """
    text = prereg.KILL_SWITCHES["parallel_replay"]
    assert "drop the graph framing" not in text.lower(), (
        "the sequencing conclusion cannot rest on an arm whose delta is identically zero"
    )
    assert "determinism" in text.lower() or "purity" in text.lower()


def test_the_sequencing_claim_has_an_arm_that_can_actually_vary_it() -> None:
    """`inquirer_depth1` enumerates its questions from x alone, so none saw an answer AND
    the strings genuinely differ from the treatment's."""
    assert "inquirer_depth1" in prereg.KILL_SWITCHES
    assert "inquirer_depth1" in prereg.KILL_SWITCH_MARGINS


def test_every_kill_switch_has_a_margin() -> None:
    assert set(prereg.KILL_SWITCHES) == set(prereg.KILL_SWITCH_MARGINS)


# --------------------------------------------------- P1 relabel and the P3 contingency


def test_the_tau2_primary_is_labelled_directional_not_confirmatory() -> None:
    """The grid and the seal disagreed, and the grid was the honest one.

    conf/grids/tier2_confirmatory.yaml already says "preregistered as a DIRECTIONAL
    REPLICATION WITH A CI, not a powered confirmatory test": MDE 22.1pp at 80% power,
    simulated through the test that actually runs. At the 12.6pp the document previously
    claimed, real power is 42%. The seal said `confirmatory` anyway.

    This changes the LABEL, not the arithmetic: `confirmatory_count`, `is_confirmatory` and
    `classify` all key off membership in the primary/secondary tuples, so tau2 still counts
    for multiplicity and is still tested. Only the strength of conclusion is now honest.
    """
    tau = next(e for e in prereg.PRIMARY if e.suite_id == "tau2")
    assert tau.claim == "directional"
    assert "22.1" in tau.rationale, "the MDE belongs in the sealed rationale, not only a grid"
    assert prereg.confirmatory_count(prereg.default_stage1()) == 20, "multiplicity unchanged"
    assert prereg.is_confirmatory(prereg.default_stage1(), "tau2", "tau_reward")


def test_the_drgym_primary_carries_a_written_contingency() -> None:
    """P3 is sealed as primary and DRGYM_API_KEY is absent. What happens if it never
    arrives has to be decided now, not after seeing whether the other two suites worked."""
    dr = next(e for e in prereg.PRIMARY if e.suite_id == "drgym")
    assert dr.contingency, "a primary that cannot run today needs a precommitted outcome"
    assert "not run" in dr.contingency.lower()


def test_a_contingency_is_optional_for_endpoints_that_can_run() -> None:
    mus = next(e for e in prereg.PRIMARY if e.suite_id == "musique")
    assert mus.contingency == ""


# ------------------------------------------------------------------ grid provenance


def test_the_grid_hash_is_computed() -> None:
    from pi_run.grids import load

    g = load("conf/grids/tier1_pilot.yaml")
    assert len(g.source_sha256) == 64


def test_the_grid_hash_reaches_the_manifest() -> None:
    """It was computed and PRINTED (cli.py) and entered no record at all, so a run could
    not be traced back to the grid text that produced it."""
    from pinq.types import RunManifest

    assert "grid_sha256" in {f for f in RunManifest.__dataclass_fields__}
    assert "grid_name" in {f for f in RunManifest.__dataclass_fields__}


def test_the_grid_hash_is_NOT_in_semantic_hash() -> None:
    """Provenance, not identity.

    Two grids can legitimately produce the same run. Putting the file hash into run identity
    would mean a reworded YAML comment changes every run_id and evicts a whole sweep -- the
    same failure the per-call cache key and RunManifest.semantic_hash are kept apart to
    avoid.
    """
    import dataclasses

    from pinq.types import RunManifest

    base = RunManifest(
        suite_id="s",
        task_id="t",
        arm_id="a",
        policy_id="p",
        seed=0,
        split="test",
        corpus_hash="c",
        budget_cap=1,
        max_turns=1,
        word_cap=30,
    )
    other = dataclasses.replace(base, grid_sha256="f" * 64, grid_name="other")
    assert base.semantic_hash == other.semantic_hash  # a property, not a method


def test_the_grid_hash_reaches_the_parquet() -> None:
    from pi_eval.schema import RUNS

    names = {f.name for f in RUNS}
    assert {"grid_sha256", "grid_name"} <= names


def test_manifest_to_dict_serializes_EVERY_manifest_field() -> None:
    """The gap that let grid_sha256 be plumbed end to end and still reach no file.

    `manifest_to_dict` is an explicit field list, so a field added to `RunManifest` is
    silently dropped at the JSON boundary: the dataclass has it, the schema has it, compact
    reads it -- and `manifest.json` never contained it, so the column is empty forever.
    Verified by running a real LLM-free sweep, which recorded `grid_name: None`.

    This asserts the general property rather than the one field, so the next field added
    cannot repeat it.
    """
    import dataclasses

    from pi_run.manifest import manifest_to_dict
    from pinq.types import RunManifest

    m = RunManifest(
        suite_id="s",
        task_id="t",
        arm_id="a",
        policy_id="p",
        seed=0,
        split="test",
        corpus_hash="c",
        budget_cap=1,
        max_turns=1,
        word_cap=30,
    )
    declared = {f.name for f in dataclasses.fields(RunManifest)}
    serialized = set(manifest_to_dict(m))
    assert declared <= serialized, f"dropped at the JSON boundary: {sorted(declared - serialized)}"


# ------------------------------------------------------------------ sealed grid identity


def test_stage1_can_seal_the_grid_hashes() -> None:
    """`Grid.source_sha256` was computed, printed, and entered no SEALED record.

    Recording it in the run manifest (done separately) tells you which grid a run came
    from. Sealing it tells you whether the grid CHANGED after the preregistration
    committed to it -- which is the question a reader of a frozen document actually has.
    """
    s1 = prereg.default_stage1(grid_hashes={"tier1_confirmatory": "a" * 64})
    assert s1.grid_hashes == {"tier1_confirmatory": "a" * 64}


def test_grid_hashes_default_to_empty_not_missing() -> None:
    assert prereg.default_stage1().grid_hashes == {}


def test_a_changed_grid_is_detected_against_the_seal() -> None:
    """The whole point of sealing bytes."""
    sealed = {"tier1_confirmatory": "a" * 64, "tier2_confirmatory": "b" * 64}
    now = {"tier1_confirmatory": "a" * 64, "tier2_confirmatory": "c" * 64}
    assert prereg.grid_drift(sealed, now) == ("tier2_confirmatory",)


def test_a_removed_grid_is_drift_too() -> None:
    assert prereg.grid_drift({"g": "a" * 64}, {}) == ("g",)


def test_a_NEW_grid_is_not_drift_against_what_was_sealed() -> None:
    """A grid added after sealing commits to nothing; it just is not preregistered."""
    assert prereg.grid_drift({"g": "a" * 64}, {"g": "a" * 64, "new": "b" * 64}) == ()


def test_no_drift_when_nothing_changed() -> None:
    assert prereg.grid_drift({"g": "a" * 64}, {"g": "a" * 64}) == ()


def test_the_seal_script_produces_real_grid_hashes() -> None:
    """Read from the grid loader, so the sealed value is the same one runs record."""
    import scripts.seal_prereg as sp

    from pi_run.grids import load

    hashes = sp._load_grid_hashes()
    assert "tier1_confirmatory" in hashes
    assert hashes["tier1_confirmatory"] == load("conf/grids/tier1_confirmatory.yaml").source_sha256
