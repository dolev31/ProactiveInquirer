"""Preregistration: write-once, hash-checked, and the classifier that gates p-values."""

import dataclasses
import json
import pathlib
import tempfile

import pytest

from pi_eval.prereg import (
    PreregStage2,
    classify,
    confirmatory_count,
    default_stage1,
    excluded,
    is_confirmatory,
    seal,
    verify,
)


def test_the_confirmatory_count_is_the_multiplicity_denominator():
    """THE INVARIANT, not the number. This test used to hard-code the prompted era's 3 / 17 /
    20, which made it a second declaration of the family size: `default_stage1()` said one
    thing, this file said another, and the two could only agree by being edited together.
    Worse, it made the test fail for the RIGHT change -- the trained programme seals its own
    stage file with its own endpoints, and a count typed into a test is not evidence about
    any of them.

    What actually has to hold, and what the BH correction and the abstract both rest on:

      * `fdr_family_size` is the size of the declared secondary family, FIXED, never
        `len(what ran)` -- an endpoint that did not run is a non-rejection, not a smaller
        family, and deriving the size from completed cells silently raises the power of every
        test that survived;
      * `confirmatory_count` is primaries plus that family and nothing else;
      * CALIBRATION is deliberately outside the count. It carries no confirmatory claim, so
        counting it would inflate the multiplicity denominator with a test nobody may cite.

    The numbers are printed rather than asserted, so a change to the programme shows up in
    the run log as a fact instead of as a red test nobody can interpret.

    CHANGED DELIBERATELY, 2026-09-15, T20: was `test_the_confirmatory_count_is_twenty`.
    """
    s1 = default_stage1()
    print(
        f"primary={len(s1.primary)} secondary={len(s1.secondary_family)} "
        f"calibration={len(s1.calibration)} fdr_family_size={s1.fdr_family_size} "
        f"confirmatory={confirmatory_count(s1)}"
    )
    assert s1.fdr_family_size == len(s1.secondary_family), (
        "the sealed family size must be the declared family, not a number typed beside it"
    )
    assert confirmatory_count(s1) == len(s1.primary) + s1.fdr_family_size
    assert len(s1.calibration) >= 1
    assert confirmatory_count(s1) == len(s1.primary) + len(s1.secondary_family), (
        "a calibration endpoint must not enter the multiplicity denominator"
    )
    assert s1.primary and s1.secondary, "an empty confirmatory programme is not a programme"


def test_every_primary_endpoint_names_a_rationale_and_a_test():
    s1 = default_stage1()
    for e in s1.primary:
        assert e.test and e.rationale.strip()
        assert e.contrast[1] == "self_inquire", "primaries are paired against the closest ancestor"
    tau = next(e for e in s1.primary if e.suite_id == "tau2")
    assert tau.test == "mcnemar_exact" and tau.cluster_by == "template_id"


def test_every_kill_switch_has_a_written_consequence():
    for arm, rule in default_stage1().kill_switches.items():
        assert rule.strip(), arm
    assert "STOP" in default_stage1().kill_switches["verbosity"]


def test_classification_decides_whether_a_p_value_is_allowed():
    """The metric names here are the names a SCORER EMITS, not prose labels for the contrast.

    This test used to say `token_f1`, which nothing has ever written into scores.parquet --
    `pi_eval.score` emits `answer_token_f1`. The preregistration therefore committed to an
    endpoint the paper's own table could not query, and no test could see it. See
    test_every_preregistered_metric_resolves_to_an_emitter, which now makes that impossible.
    """
    s1 = default_stage1()
    assert classify(s1, "musique", "answer_token_f1") == "primary"
    assert classify(s1, "musique", "rnr_resolve") == "secondary"
    assert classify(s1, "musique", "phi_loo") == "exploratory"
    assert is_confirmatory(s1, "musique", "answer_token_f1")
    assert not is_confirmatory(s1, "musique", "phi_loo")
    # `classify` keys on (suite, metric) ALONE, and the synth calibration endpoint shares its
    # metric with the pooled secondary family, so this pair genuinely cannot be told apart
    # here: it resolves to the stronger label. The calibration/confirmatory distinction lives
    # on the ENDPOINT (`Endpoint.claim`), which is what report.py reads and what
    # `confirmatory_count` excludes -- see the two tests at the bottom of this file.
    assert classify(s1, "synth", "evidence_coverage") == "secondary"


def test_a_primary_metric_is_only_primary_for_its_own_suite():
    """answer_token_f1 is the musique endpoint; it is not automatically primary on tau2."""
    s1 = default_stage1()
    assert classify(s1, "tau2", "answer_token_f1") == "exploratory"


def test_burned_pilot_ids_are_excluded():
    """Data used to choose a threshold cannot also be used to test it."""
    s1 = default_stage1(
        burned_pilot_ids={"musique": ["p1", "p2"]}, excluded_task_ids={"musique": ["leak1"]}
    )
    assert excluded(s1, "musique", "p1")
    assert excluded(s1, "musique", "leak1")
    assert not excluded(s1, "musique", "ok")


def test_seal_then_verify_round_trips(tmp_path):
    seal(tmp_path, "stage1", default_stage1())
    r = verify(tmp_path)
    assert r.ok and r.checked == 1 and not r.mismatches


def test_seal_refuses_to_overwrite(tmp_path):
    """Write-once by design: a silently rewritten prereg would invalidate the whole defence."""
    seal(tmp_path, "stage1", default_stage1())
    with pytest.raises(FileExistsError, match="already sealed"):
        seal(tmp_path, "stage1", default_stage1())


def test_verify_detects_a_tampered_file(tmp_path):
    seal(tmp_path, "stage1", default_stage1())
    p = tmp_path / "stage1.json"
    d = json.loads(p.read_text())
    d["alpha"] = 0.20  # the classic post-hoc loosening
    p.write_text(json.dumps(d, indent=2, sort_keys=True) + "\n")
    r = verify(tmp_path)
    assert not r.ok and "stage1.json" in r.mismatches


def test_verify_detects_a_deleted_file(tmp_path):
    seal(tmp_path, "stage1", default_stage1())
    (tmp_path / "stage1.json").unlink()
    r = verify(tmp_path)
    assert not r.ok and "stage1.json" in r.missing


def test_verify_fails_without_a_manifest(tmp_path):
    assert not verify(tmp_path).ok


def test_two_stages_seal_independently(tmp_path):
    """Stage 2 holds constants that cannot honestly exist on day one."""
    seal(tmp_path, "stage1", default_stage1())
    seal(
        tmp_path,
        "stage2",
        PreregStage2(
            sigma_j=0.62,
            tau=1.24,
            dwr_weights={"0": 0.1, "1": 0.3, "2": 0.6},
            theta_nli=0.85,
            discoverability_threshold=0.60,
            word_cap=120,
            budget_grid=(1, 2, 4, 8, 16),
        ),
    )
    r = verify(tmp_path)
    assert r.ok and r.checked == 2


def test_tau_is_two_sigma_j_by_construction(tmp_path):
    """No effect below 2*sigma_J/sqrt(n) is reportable, so tau is derived, not chosen."""
    s2 = PreregStage2(
        sigma_j=0.62,
        tau=2 * 0.62,
        dwr_weights={},
        theta_nli=0.85,
        discoverability_threshold=0.60,
        word_cap=120,
        budget_grid=(1, 4, 16),
    )
    assert s2.tau == pytest.approx(2 * s2.sigma_j)


# --------------------------------------------------------------- the two-declarations bug
#
# `prereg.py` and `report.py` used to hold INDEPENDENT endpoint lists that named the same
# endpoints with DIFFERENT metric strings, cross-checked by nothing. The sealed preregistration
# could therefore commit to `kpr_incremental` while the paper's primary table queried
# `keypoint_recall`, and both files looked correct in isolation. These three tests are what
# make that unrepresentable: prereg is the single source of truth, report reads it, and every
# name in it has to resolve to something a scorer actually writes.


def test_every_preregistered_metric_resolves_to_an_emitter():
    """A preregistered endpoint naming a metric no scorer emits reads as a committed test that
    was quietly never run. Every name is checked against the live metric register."""
    from pi_eval import report as rp
    from pi_eval.prereg import all_endpoints, endpoint_metrics
    from pi_eval.score import BY_NAME, family_of

    # cad_ge2 and frontier_auc are DERIVED pulls: they are computed in report.py from the
    # `cad#d`/`cad_n#d` and `frontier_*#k` row families rather than emitted under their own
    # name, so their emitter is the pull, and it is named here rather than assumed.
    derived = {"cad_ge2": rp._pull_cad_ge2, "frontier_auc": rp._pull_frontier_auc}
    for metric in endpoint_metrics():
        assert metric in BY_NAME or family_of(metric) in BY_NAME or metric in derived, (
            f"preregistered metric {metric!r} is emitted by nothing: it is neither in "
            f"pi_eval.score.METRICS nor a declared derived pull in pi_eval.report"
        )
    assert "kpr_incremental" in endpoint_metrics()
    assert BY_NAME["kpr_incremental"].judge_derived, "the P3 endpoint is behind the noise floor"
    for e in all_endpoints():
        assert e.test in ("mcnemar_exact", "sign_flip_permutation"), e.key


def test_report_declares_no_endpoint_of_its_own():
    """report.py's confirmatory tables ARE prereg's endpoints, key for key and in order."""
    from pi_eval import report as rp
    from pi_eval.prereg import default_stage1

    s1 = default_stage1()
    assert [c.key for c in rp.PREREG_PRIMARY] == [e.key for e in (*s1.calibration, *s1.primary)]
    assert [c.key for c in rp.SECONDARY] == [e.key for e in s1.secondary]
    assert rp.SECONDARY_FAMILY_Q == s1.fdr_q
    assert rp.POOLED == "*"


def test_the_calibration_contrast_carries_no_confirmatory_claim():
    from pi_eval import report as rp

    cal = [c for c in rp.PREREG_PRIMARY if c.claim == "calibration"]
    assert len(cal) == 1 and cal[0].suite_id == "synth"
    assert cal[0].treatment == rp.SYNTH_TREATMENT and cal[0].comparator == rp.SYNTH_COMPARATOR


def test_the_kill_switches_have_exactly_one_source_of_truth():
    """`pi report killswitch` printed FIVE rows while `default_stage1` sealed FOUR.

    The missing one was `random_q`, whose written consequence is that the Inquirer is not
    selecting and the thesis does not survive. Sealing stage 1 in that state would have
    committed to a document that omits it, while the report went on showing a decision the
    sealed document had never made -- which is exactly the substitution preregistration exists
    to prevent. Nothing was sealed when this was found, so it was fixable; after
    `git tag prereg-stage1` it would not have been.
    """
    from pi_eval.prereg import KILL_SWITCHES, default_stage1
    from pi_eval.report import KILLSWITCH_RULES

    assert dict(KILLSWITCH_RULES) == dict(KILL_SWITCHES)
    assert dict(default_stage1().kill_switches) == dict(KILL_SWITCHES)
    assert set(KILL_SWITCHES) == {
        "verbosity",
        "self_inquire",
        "inquirer_noevidence",
        "parallel_replay",
        "random_q",
        # The SEQUENCING switch, moved here from parallel_replay, whose delta is
        # identically zero by construction and so could only ever "match".
        "inquirer_depth1",
        # ADDED DELIBERATELY, 2026-09-15, T20. `compute_matched` is a declared SECONDARY
        # comparator -- the pooled evidence_coverage endpoint against "the same token budget
        # spent without inquiry" -- and it had no kill switch, i.e. a preregistered test whose
        # outcome had no preregistered consequence. It is the same shape as `verbosity` and
        # its arm-table `kind` was corrected from "control" to "killswitch" to match.
        "compute_matched",
    }


def test_every_kill_switch_names_a_real_arm_and_a_consequence():
    """A comparator that is not in the arm table can never be run, so its "consequence" is a
    sentence nobody will ever have to act on. And a consequence that says "investigate" is not
    a consequence: the point is to write down the decision before the data can influence it."""
    from pi_eval.prereg import KILL_SWITCHES
    from pinq_expt import arms as arm_table

    known = set(arm_table.arm_ids())
    for arm, consequence in KILL_SWITCHES.items():
        assert arm in known, f"{arm} is preregistered as a kill switch but is not an arm"
        assert arm_table.get(arm).kind == "killswitch", arm
        assert len(consequence) > 40
        assert not any(
            w in consequence.lower() for w in ("investigate", "look into", "consider whether")
        ), f"{arm}: a consequence must be a decision, not a plan to decide"

    # And the arm table must not carry a kill switch the preregistration never committed to.
    # `pare_plus_verbosity` is the one allowed exception: Grid E is exploratory, spends none
    # of the multiplicity budget and carries no confirmatory test, so its kill switch has
    # nothing to be preregistered against. Any OTHER unlisted kill-switch arm is a comparator
    # that could be run and reported with no decision written down in advance.
    declared = {a for a in known if arm_table.get(a).kind == "killswitch"}
    assert declared - set(KILL_SWITCHES) == {"pare_plus_verbosity"}, (
        f"kill-switch arms with no preregistered consequence: "
        f"{sorted(declared - set(KILL_SWITCHES) - {'pare_plus_verbosity'})}"
    )


def test_every_preregistered_endpoint_resolves_to_something_that_produces_rows():
    """A preregistered endpoint naming a metric nothing produces is not an endpoint: the test
    is declared, the multiplicity budget is spent on it, and the table row is empty. That is
    why the tau2 primary read `task_success` for as long as nothing emitted `tau_reward`.

    TWO legitimate producers, and the distinction is real rather than a loophole. Most
    endpoints name a metric `pi score` writes per run. `cad_ge2` and `frontier_auc` are
    DERIVED at report time -- one is |V_d|-weighted across depth rows, the other integrates a
    curve over cumulative spend -- so neither can be a per-run scalar. Both must have a puller,
    and this asserts that rather than exempting them.
    """
    from pi_eval import report
    from pi_eval.prereg import PRIMARY, SECONDARY
    from pi_eval.score import BY_NAME, family_of

    for e in list(PRIMARY) + list(SECONDARY):
        if family_of(e.metric) in BY_NAME:
            continue
        puller = getattr(report, f"_pull_{e.metric}", None)
        assert callable(puller), (
            f"{e.suite_id}/{e.metric} is emitted by no scorer and derived by no report puller"
        )


def test_tau2s_two_confirmatory_contrasts_use_one_metric():
    """P1 and S2 are the same ladder on the same suite. Measuring one against DB-hash equality
    and the other against need coverage lets them disagree with nobody able to say what that
    would mean -- an agent can resolve every need and change nothing."""
    from pi_eval.prereg import PRIMARY, SECONDARY

    tau2 = [e for e in list(PRIMARY) + list(SECONDARY) if e.suite_id == "tau2"]
    binary = [e for e in tau2 if e.metric in ("tau_reward", "task_success")]
    assert {e.metric for e in binary} == {"tau_reward"}, (
        f"tau2's binary contrasts disagree on the metric: {[e.metric for e in binary]}"
    )
    assert all(e.cluster_by == "template_id" for e in tau2 if e.cluster_by), (
        "tau2's 97 tasks derive from ~18 templates; resampling tasks understates the SE"
    )


def test_tau_reward_is_the_metric_pi_verify_tau2_validates():
    """The endpoint and the validation must name the same thing. `pi verify tau2 --replay-gold`
    reproduces the gold DB hash on 97/97, and that is only evidence for the endpoint if the
    endpoint is the DB-hash reward."""
    from pi_eval.prereg import PRIMARY
    from pinq_adapters.tau2.actuator import TAU_REWARD

    assert TAU_REWARD == "tau_reward"
    assert next(e for e in PRIMARY if e.suite_id == "tau2").metric == TAU_REWARD


# ---------------------------------------------------------------- the seal must actually seal


def test_sealing_a_later_stage_cannot_re_bless_an_edited_earlier_one(tmp_path):
    """`seal()` refuses to overwrite a stage FILE, but `_rebuild_manifest` re-hashed everything
    on disk -- so an edited stage1.json was silently re-blessed by the next ordinary seal.
    Reproduced on the normal workflow (seal stage 1, run the pilot, seal stage 2):

        after sealing stage1      ok=True
        after editing stage1.json ok=False  mismatches=['stage1.json']
        after sealing stage2      ok=True                                <- evidence cleared

    with the tampered text still on disk. A silently edited preregistration is the one failure
    mode `verify` names as invalidating the whole multiplicity defence."""
    import dataclasses

    from pi_eval.prereg import PreregTampered, seal, verify

    @dataclasses.dataclass
    class _P:
        note: str

    seal(tmp_path, "stage1", _P("the preregistered plan"))
    assert verify(tmp_path).ok

    (tmp_path / "stage1.json").write_text(json.dumps({"note": "TAMPERED"}) + "\n")
    assert not verify(tmp_path).ok

    with pytest.raises(PreregTampered, match="no longer matches its sealed digest"):
        seal(tmp_path, "stage2", _P("sigma_j"))
    assert not verify(tmp_path).ok, "the tamper must still be visible after the refusal"


def test_the_ordinary_two_stage_workflow_is_untouched(tmp_path):
    import dataclasses

    from pi_eval.prereg import seal, verify

    @dataclasses.dataclass
    class _P:
        note: str

    seal(tmp_path, "stage1", _P("a"))
    seal(tmp_path, "stage2", _P("b"))
    r = verify(tmp_path)
    assert r.ok and r.checked == 2


# ------------------------------------------------------- the exclusion list must reach the seal


def test_the_sealer_reads_the_exclusion_list_the_builder_actually_writes():
    """Two independent mismatches, both hidden by `except Exception: pass`: the glob
    (`data/gold/*/excluded.json`) matched neither the builder's directory depth nor its
    filename (`<gold>/graphs/<suite>/excluded_<split>.json`), and `sorted(json.loads(...))`
    over the builder's DICT would have frozen its KEYS as the task-id list.

    Measured: `_load_exclusions()` returned {} while 24 leaked musique ids sat on disk. Sealing
    stage 1 would have frozen a preregistration formally declaring no contamination control."""
    import importlib.util
    import pathlib

    gold = pathlib.Path("data/gold/graphs")
    if not list(gold.glob("*/excluded_*.json")):
        pytest.skip("gold not built")

    spec = importlib.util.spec_from_file_location("_sp", "scripts/seal_prereg.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    got = mod._load_exclusions()
    assert got, "the whole point: this returned {} while exclusion files existed"
    for suite, ids in got.items():
        assert all(isinstance(i, str) for i in ids)
        assert not any(i in {"excluded_task_ids", "exclusion_list", "split"} for i in ids), (
            "these are the DICT KEYS the old shape would have frozen as task ids"
        )
        on_disk = set()
        for f in gold.glob(f"{suite}/excluded_*.json"):
            on_disk |= set(json.loads(f.read_text())["excluded_task_ids"])
        assert set(ids) == on_disk


def test_a_malformed_exclusion_file_is_loud(tmp_path, monkeypatch):
    """An exclusion list that cannot be read is the one case where proceeding quietly produces
    exactly the document the loader exists to prevent."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("_sp2", "scripts/seal_prereg.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    d = tmp_path / "data" / "gold" / "graphs" / "musique"
    d.mkdir(parents=True)
    (d / "excluded_train.json").write_text('{"excluded_task_ids": "not-a-list"}')
    monkeypatch.setattr(mod, "__file__", str(tmp_path / "scripts" / "seal_prereg.py"))
    with pytest.raises(TypeError, match="expected list"):
        mod._load_exclusions()


def _sealer():
    import importlib.util

    spec = importlib.util.spec_from_file_location("_sp3", "scripts/seal_prereg.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_stage2_freezes_the_weights_the_scorer_actually_uses(tmp_path, monkeypatch):
    """These were literals in the sealer and had drifted: it froze
    {"0":0.1,"1":0.2,"2":0.3,"3":0.4} while pi_eval.score.DWR_WEIGHTS is
    {0:0.5,1:1.0,2:1.5,3:2.0,4:2.0,5:2.0} -- different values and two fewer depths. The
    preregistration would have frozen a weighting no reported `dwr` was ever computed with."""
    import sys

    from pi_eval.score import DWR_WEIGHTS

    mod = _sealer()
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["x", "seal-stage2", "--sigma-j", "0.42"])
    assert mod.main() == 0

    sealed = json.loads((tmp_path / "stage2.json").read_text())["dwr_weights"]
    assert sealed == {str(k): float(v) for k, v in sorted(DWR_WEIGHTS.items())}


def test_verify_catches_the_code_moving_away_from_the_sealed_document():
    """Digests catch a changed PREREG. Nothing caught changed CODE: PreregStage2 is defined,
    sealed and read by nothing in src/, so an edit to DWR_WEIGHTS after sealing diverged from
    the preregistration with no signal anywhere."""
    from pi_eval.prereg import check_live_constants, live_constants

    assert check_live_constants(live_constants()) == []
    drift = check_live_constants({"dwr_weights": {"0": 0.1, "1": 0.2, "2": 0.3, "3": 0.4}})
    assert drift and "dwr_weights" in drift[0]


def _isolate_burned_pilots(mod, monkeypatch):
    """Stop a seal test from depending on how many runs happen to be on disk.

    `monkeypatch.setattr(mod, "ROOT", tmp_path)` redirects the seal's OUTPUT, but
    `_pilot_slices()` and `_executed_tasks()` read conf/grids and scores/parquet -- the LIVE
    repo. So these tests' outcome moved with the corpus: once exploratory sweeps had touched 73
    of musique's 120 pilot-slice tasks, `_pilots_that_ran` returned {"musique"}, the
    burned-pilot guard refused before the seal was reached, and two tests about seal IDEMPOTENCY
    started failing for a reason that has nothing to do with idempotency.

    The guard is not disabled anywhere else; it gets its own tests below, which it previously
    had none of.

    The trained-arm scan is isolated for exactly the same reason and on exactly the same
    terms: it reads the LIVE `scores/parquet/runs.parquet`, so leaving it live would make a
    test about seal IDEMPOTENCY fail whenever that file is absent or a trained sweep lands.
    It has its own tests below.
    """
    monkeypatch.setattr(mod, "_pilots_that_ran", lambda **_: set())
    monkeypatch.setattr(mod, "_trained_runs_on_confirmatory_tasks", lambda **_: [])


def test_a_second_seal_is_a_clean_refusal_not_a_traceback(tmp_path, monkeypatch, capsys):
    """Re-running a seal is an ordinary operator mistake and the refusal is the designed
    behaviour, so it should read as one."""
    import sys

    mod = _sealer()
    _isolate_burned_pilots(mod, monkeypatch)
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["x", "seal-stage1"])
    assert mod.main() == 0
    assert mod.main() == 2
    assert "refused" in capsys.readouterr().err


def test_verify_exits_nonzero_on_a_tampered_stage(tmp_path, monkeypatch):
    import sys

    mod = _sealer()
    _isolate_burned_pilots(mod, monkeypatch)
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["x", "seal-stage1"])
    assert mod.main() == 0
    monkeypatch.setattr(sys, "argv", ["x", "verify"])
    assert mod.main() == 0
    (tmp_path / "stage1.json").write_text(json.dumps({"note": "TAMPERED"}) + "\n")
    assert mod.main() == 1


# ------------------------------------------------- the sealed document now has a reader


def test_the_sealed_exclusions_reach_the_eligibility_predicate(tmp_path):
    """`prereg.excluded()` had zero callers and nothing in src/ opened stage1.json, so the
    exclusion list was written, digested, frozen -- and had no effect on any analysis.

    For `excluded_task_ids` that was belt-and-braces (the builder drops a contaminated task
    from the corpus; verified 0 of 24 present in the built musique corpus). For
    `burned_pilot_ids` it was a real hole: ELIGIBLE filters `pilot_flag = FALSE`, dropping the
    pilot RUN, while the rule is about the TASK -- a tau chosen by looking at task T
    contaminates T's confirmatory run in every arm."""
    from pi_eval.prereg import excluded_pairs
    from pi_eval.report import ELIGIBLE, Agg

    (tmp_path / "stage1.json").write_text(
        json.dumps(
            {
                "excluded_task_ids": {"musique": ["2hop__1_2"]},
                "burned_pilot_ids": {"tau2": ["task_007"]},
            }
        )
    )
    pairs, note = excluded_pairs(tmp_path)
    assert ("tau2", "task_007") in pairs, "a burned pilot TASK must be excluded"
    assert ("musique", "2hop__1_2") in pairs
    assert "1 burned pilot" in note

    agg = Agg(
        parquet_dir=tmp_path,
        con=None,
        scorer_hash="sh",
        graph_version="v1",
        allow_contaminated=False,
        sigma_j={},
        excluded_pairs=pairs,
        prereg_note=note,
    )
    pred = agg.predicate
    assert pred.startswith(ELIGIBLE)
    assert "NOT IN" in pred and "'task_007'" in pred, pred


def test_no_seal_leaves_the_predicate_exactly_as_it_was(tmp_path):
    """Nothing sealed is the repository's current state, and it must not change any number."""
    from pi_eval.prereg import excluded_pairs
    from pi_eval.report import ELIGIBLE, Agg

    pairs, note = excluded_pairs(tmp_path)
    assert pairs == frozenset() and "no sealed stage 1" in note
    agg = Agg(
        parquet_dir=tmp_path,
        con=None,
        scorer_hash="sh",
        graph_version="v1",
        allow_contaminated=False,
        sigma_j={},
    )
    assert agg.predicate == ELIGIBLE


def test_provenance_records_the_predicate_that_actually_ran():
    """It recorded the module constant ELIGIBLE unconditionally, so an `--allow-contaminated`
    aggregation -- which runs CONTAMINATED -- produced a provenance naming a filter it had not
    applied. A provenance that names the wrong filter is worse than one that names none,
    because it is checkable and passes."""
    from pi_eval.report import CONTAMINATED, ELIGIBLE, provenance

    p = provenance(
        table_id="t",
        parquet_dir=".",
        run_ids=["r1"],
        scorer_hash="sh",
        graph_version="v1",
        code_versions=["c"],
        query="q",
        tool="pytest",
        eligibility_predicate=CONTAMINATED,
        prereg_ok=False,
        prereg_note="2 pairs",
    )
    assert p["eligibility_predicate"] == CONTAMINATED != ELIGIBLE
    assert p["prereg_verified"] is False
    assert p["prereg_exclusions"] == "2 pairs"


def test_verify_sees_a_file_in_the_sealed_directory_that_no_digest_covers(tmp_path):
    """`measure_sigma_j.py --write` writes prereg/sigma_j.json directly rather than through
    seal(), so the published sigma_J -- the constant deciding what may be claimed at all --
    sat inside the sealed directory with no digest, and `verify` reported ok=True because it
    only ever walked the manifest."""
    import dataclasses

    from pi_eval.prereg import seal, verify

    @dataclasses.dataclass
    class _P:
        note: str

    seal(tmp_path, "stage1", _P("a"))
    assert verify(tmp_path).unsealed == []

    (tmp_path / "sigma_j.json").write_text(json.dumps({"quality": 0.42}))
    r = verify(tmp_path)
    assert r.ok, "digest integrity is unaffected; the file is uncovered, not corrupt"
    assert r.unsealed == ["sigma_j.json"]

    seal(tmp_path, "stage2", _P("b"))
    assert verify(tmp_path).unsealed == [], "the next seal brings it under the manifest"


# ------------------------------------------- the kill switch accepts the null, so it needs a margin


def test_every_kill_switch_has_an_equivalence_margin():
    """A kill switch fires when the treatment MATCHES a comparator, with consequences up to
    "STOP; no reframe survives". That is ACCEPTING THE NULL, and a CI covering zero is not
    evidence of equivalence -- it is equally consistent with "no effect" and with "an effect
    this pilot cannot resolve"."""
    from pi_eval.prereg import KILL_SWITCH_MARGINS, KILL_SWITCHES

    assert set(KILL_SWITCH_MARGINS) == set(KILL_SWITCHES), "every switch needs a margin"
    assert all(0.0 < m < 0.5 for m in KILL_SWITCH_MARGINS.values())
    assert KILL_SWITCH_MARGINS["verbosity"] < KILL_SWITCH_MARGINS["self_inquire"], (
        "verbosity's consequence is the unconditional STOP, so its margin must be the tightest"
    )


def test_a_wide_interval_is_inconclusive_and_never_a_stop():
    """Measured at the pilot's real n=120 through paired_difference: a TRUE +2pt effect
    produced a zero-covering CI 77% of the time at sd=0.20. Under the old rule every one of
    those said "apply the rule"."""
    import math

    from pi_eval.prereg import KILL_SWITCH_MARGINS

    def verdict(arm, n, lo, hi):
        m = float(KILL_SWITCH_MARGINS.get(arm, 0.0))
        if n == 0:
            return "NOT RUN"
        if math.isnan(lo) or math.isnan(hi):
            return "INDETERMINATE"
        if lo > 0:
            return "SEPARATED"
        if hi < 0:
            return "INVERTED"
        if m > 0 and lo >= -m and hi <= m:
            return "EQUIVALENT"
        return "INCONCLUSIVE" if m > 0 else "MATCHES"

    assert verdict("verbosity", 120, -0.09, 0.11) == "INCONCLUSIVE"
    assert verdict("verbosity", 120, -0.010, 0.020) == "EQUIVALENT"
    assert verdict("verbosity", 120, 0.02, 0.09) == "SEPARATED"


def test_the_report_implements_the_three_way_verdict():
    import inspect

    from pi_eval import report

    src = inspect.getsource(report.killswitch_table)
    assert "EQUIVALENT" in src and "INCONCLUSIVE" in src
    assert "KILL_SWITCH_MARGINS" in src
    assert "do NOT stop" in src, "the inconclusive branch must say it is not a licence to stop"


# --------------------------------- a comparator that cannot differ is not a test


def test_parallel_replay_is_input_identical_to_the_arm_it_replays():
    """PROVEN, not argued. ParallelReplayInquirer emits the treatment's own recorded question
    strings, so the retrieval is the same retrieval; `draft()` and the Answerer are pure in the
    evidence subset hash, so the answer is the same answer.

    Consequence: preregistered secondary S5 (inquirer_prompted vs parallel_replay on
    evidence_coverage) has a delta of identically zero and a p of 1.0 before any data exists,
    and the kill-switch row that says "drop the graph framing" is decided by arithmetic."""
    import pathlib
    import tempfile

    from pi_eval.build.synth_build import build
    from pinq.budget import BudgetLedger
    from pinq.loop import run_loop
    from pinq_adapters.synth.suite import SynthSuite
    from pinq_expt.fakes import ChainInquirer, EchoDrafter, FrozenAnswerer
    from pinq_expt.policies.controls import ParallelReplayInquirer

    root = pathlib.Path(tempfile.mkdtemp())
    corpus, _, _ = build(n_tasks=2, n_facets=3, depth=3, seed=7, root=root)
    suite = SynthSuite(corpus.parent)
    tid = next(iter(suite.task_ids()))

    def run(pol):
        return run_loop(
            view=suite.view(tid),
            inquirer=pol,
            retriever=suite.retriever(tid),
            drafter=EchoDrafter(),
            answerer=FrozenAnswerer(),
            ledger=BudgetLedger(cap=64),
            max_turns=16,
            k=5,
            seed=0,
        )

    seq = run(ChainInquirer())
    asked = [t.action.text for t in seq.turns if getattr(t.action, "text", None)]
    assert asked, "the fixture must actually ask something"
    rep = run(ParallelReplayInquirer(questions={tid: asked}))
    assert set(rep.evidence.uids) == set(seq.evidence.uids)
    assert rep.outcome.answer.text == seq.outcome.answer.text


def test_the_kill_switch_reports_a_degenerate_comparator_as_such():
    """A verdict of "drop the graph framing" must not be reachable by arithmetic. The
    equivalence margin added just before this made it WORSE, upgrading a hedged
    "MATCHES -> apply the rule" to a confident "EQUIVALENT -> apply the rule"."""
    import inspect

    from pi_eval import report

    # CODE ONLY. The comment above the branch explains the equivalence reading it replaced, so
    # a check that cannot tell an explanation from a statement matches the wrong line -- which
    # is exactly what my first version of this assertion did.
    code = "\n".join(
        ln
        for ln in inspect.getsource(report.killswitch_table).splitlines()
        if not ln.lstrip().startswith("#")
    )
    assert "DEGENERATE" in code and "cannot differ" in code
    # and it must be decided BEFORE the equivalence branch, or the margin swallows it
    assert code.index("degenerate") < code.index("EQUIVALENT (")


# --------------------------------------------------------------- the burned-pilot guard
#
# IT HAD NO TESTS AT ALL, and it is currently the thing refusing to seal this repo: 73 of
# musique's 120 pilot-slice tasks have runs, none of them flagged `pilot_flag=True`. That is
# the guard working -- those tasks HAVE been executed and their results looked at, so sealing
# would falsely declare that no pilot ever touched them -- but nothing pinned the behaviour.


def test_the_guard_refuses_when_a_pilot_slice_has_runs_but_no_burned_ids():
    mod = _sealer()
    with pytest.raises(mod.BurnedPilotsMissing):
        mod._check_burned_pilots({}, {"musique"})


def test_the_guard_is_satisfied_once_the_burned_ids_are_recorded():
    mod = _sealer()
    mod._check_burned_pilots({"musique": ["2hop__1_2"]}, {"musique"})  # must not raise


def test_the_guard_says_nothing_when_no_pilot_slice_was_touched():
    """A pilot grid that was DEFINED but never run must not block a seal."""
    mod = _sealer()
    mod._check_burned_pilots({}, set())  # must not raise


def test_pilots_that_ran_keys_on_TASK_IDS_not_on_the_suite_having_runs():
    """The distinction the docstring is built on: musique carries runs from sweeps that were
    not the pilot, so 'the suite has any runs' would block a seal over a pilot that never
    happened. Overlap with the pilot's OWN tasks is the signal.

    CHANGED DELIBERATELY, 2026-09-15, T20: the second argument was `executed` (any compacted
    run) and is now `piloted` (`pilot_flag = TRUE` only). Task-id overlap was necessary and
    not sufficient -- 3,830 runs from latent_trainset, growth_parents_* and dev_select_musique
    sit on the pilot slice's own 120 musique ids, and every one of them said "a pilot ran".
    """
    mod = _sealer()
    slices = {"musique": {"a", "b"}}
    assert mod._pilots_that_ran(pilot_slices=slices, piloted={"musique": {"z"}}) == set()
    assert mod._pilots_that_ran(pilot_slices=slices, piloted={"musique": {"b"}}) == {"musique"}


def test_an_empty_pilot_slice_never_fires():
    mod = _sealer()
    assert (
        mod._pilots_that_ran(pilot_slices={"musique": set()}, piloted={"musique": {"a"}}) == set()
    )


# ----------------------------------------------------------- the kill switches and their home
#
# Five blockers the stage-1 draft found. Each one is a way for the sealed document and the
# running code to say different things while every command keeps printing.


def test_compute_matched_is_a_kill_switch_with_a_written_consequence():
    """IT WAS MISSING. `compute_matched` is a DECLARED SECONDARY comparator -- the pooled
    evidence_coverage endpoint `(inquirer_prompted, compute_matched)` -- and it is the arm whose
    match means the gain was the token budget rather than the questions. A comparator with an
    endpoint and no kill switch is a test that can be run and then have no consequence, which
    is the one thing a preregistration exists to prevent."""
    from pi_eval.prereg import KILL_SWITCH_MARGINS, KILL_SWITCHES

    assert "compute_matched" in KILL_SWITCHES
    assert KILL_SWITCHES["compute_matched"].strip()
    assert KILL_SWITCH_MARGINS["compute_matched"] == 0.05


def test_every_kill_switch_has_a_margin_and_every_margin_a_switch():
    """Two mappings keyed by hand are two lists that drift. The pair that drifted before was
    `report.KILLSWITCH_RULES` against this module's four."""
    from pi_eval.prereg import KILL_SWITCH_MARGINS, KILL_SWITCHES

    assert set(KILL_SWITCHES) == set(KILL_SWITCH_MARGINS)
    assert KILL_SWITCH_MARGINS["verbosity"] == 0.03, "the unconditional STOP keeps the tight one"
    assert all(0.0 < m <= 0.05 for m in KILL_SWITCH_MARGINS.values())


def test_the_margins_are_sealed_and_not_just_a_module_constant():
    """A SEALED CONSTANT NOTHING COMPARES AGAINST IS A DECORATION, and a module constant no
    sealed document carries is worse: the margins decide whether a kill switch FIRES, and an
    edit to them after sealing would change every verdict with no signal anywhere. They now
    ride in the stage-1 payload, and `check_live_constants` compares the two."""
    from pi_eval.prereg import KILL_SWITCH_MARGINS, check_live_constants, default_stage1

    s1 = default_stage1()
    assert s1.kill_switch_margins == dict(KILL_SWITCH_MARGINS)

    sealed = dataclasses.asdict(s1)
    assert check_live_constants(sealed) == [], "a freshly built stage 1 cannot already drift"

    sealed["kill_switch_margins"] = dict(sealed["kill_switch_margins"], verbosity=0.20)
    drift = check_live_constants(sealed)
    assert len(drift) == 1 and "kill_switch_margins" in drift[0]


def test_check_live_constants_reads_either_stage():
    """It was written for stage 2 and typed to it. Stage 1 now carries comparable constants
    too, and a checker that can only read one stage leaves the other unguarded."""
    from pi_eval.prereg import check_live_constants, live_constants

    live = live_constants()
    assert "kill_switch_margins" in live and "ci_resamples" in live
    assert "dwr_weights" in live, "the stage-2 fields must not have been dropped"
    assert check_live_constants({}) == [], "a payload carrying none of them cannot drift"


# --------------------------------------------------------------- one resample count, not two


def test_the_resample_count_has_exactly_one_source():
    """THE DRAFT'S BLOCKER, IN ONE LINE. The draft's claim_rule sealed 'task-clustered BCa 95%
    intervals at 1,000 resamples, seed 0' while `report.Agg` defaulted to n_boot=10,000 and
    `pi report --n-boot` defaulted to 10,000 -- so the sealed number and the rendered number
    were different numbers, and nothing compared them. The sealed value is the source."""
    from pi_eval.prereg import CI_RESAMPLES, default_stage1
    from pi_eval.report import Agg

    assert CI_RESAMPLES == 1000
    assert str(CI_RESAMPLES) in default_stage1().claim_rule.replace(",", "")
    assert dataclasses.fields(Agg)
    defaults = {f.name: f.default for f in dataclasses.fields(Agg)}
    assert defaults["n_boot"] == CI_RESAMPLES, "report must READ the sealed constant"


def test_the_cli_carries_no_resample_literal_of_its_own():
    """`pi report --n-boot` is where a rendered table's resample count actually comes from,
    and it carried its own literal 10,000 beside the document's 1,000.

    It cannot simply be set to `CI_RESAMPLES` here: `build_parser` runs on the ROLLOUT path,
    and `pi_run.cli`'s contract is that no gold-side import reaches that address space --
    which is the same firewall the whole repository is built on. So the flag defaults to None
    and the SCORING-side `_open_agg` resolves it, and what this pins is that no literal
    survives on either side. EVERY subparser offering the flag is checked, because the option
    is registered in a helper several of them call."""
    import inspect

    from pi_eval.prereg import CI_RESAMPLES
    from pi_run import cli
    from pi_run.cli import build_parser

    found = [
        action.default
        for action in _walk_actions(build_parser())
        if "--n-boot" in getattr(action, "option_strings", ())
    ]
    assert found, "no --n-boot option found on any subparser"
    assert all(d is None for d in found), found
    src = inspect.getsource(cli._open_agg)
    assert "CI_RESAMPLES" in src and "10_000" not in src
    assert CI_RESAMPLES == 1000


def _walk_actions(parser):
    import argparse

    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for sub in action.choices.values():
                yield from _walk_actions(sub)
        else:
            yield action


# --------------------------------------------------------- the fork-pair test, named and wired


def test_sign_test_exact_is_in_the_vocabulary_and_resolves_to_the_function_that_runs_it():
    """The fork-pair primary is a COUNT of follow-up turns over pairs keyed on
    (foreign_trace_sha, foreign_prefix_k, seed). There is no cluster to permute and no binary
    to McNemar, and `fork_report.sign_test_p` is what actually computes its p-value -- but the
    test vocabulary had no name for it, so a declaration would have had to invent a spelling
    report.py could not resolve. That is exactly how the two endpoint declarations were able
    to disagree in the first place."""
    from pi_eval import fork_report
    from pi_eval.prereg import MCNEMAR_EXACT, SIGN_FLIP_PERMUTATION, SIGN_TEST_EXACT, test_function

    assert SIGN_TEST_EXACT == "sign_test_exact"
    assert test_function(SIGN_TEST_EXACT) is fork_report.sign_test_p
    for name in (MCNEMAR_EXACT, SIGN_FLIP_PERMUTATION):
        assert callable(test_function(name)), name
    with pytest.raises(KeyError):
        test_function("a_test_nobody_wrote")


def test_the_sign_test_is_the_exact_two_sided_one():
    """Pinned against a hand-computable case so the name cannot drift onto another function:
    3 fewer and 0 more is 2 * (1/8) = 0.25."""
    from pi_eval.prereg import SIGN_TEST_EXACT, test_function

    assert test_function(SIGN_TEST_EXACT)(3, 0) == pytest.approx(0.25)


# ------------------------------------------------- exclusions that key on the RUN, not the cell


ORPHAN = "4ab7e54494ad9e183a5b0aed9793a7ed"


def test_the_n6_orphans_are_excluded_by_run_id():
    """19 duplicate run directories from the N6 sweep: its agent restarted after a
    mid-campaign commit, so the same (suite, task, arm, seed) cells were re-minted at
    code_version 3295544 and re-spent. The fc1def5 runs are the ones of record."""
    from pi_eval.prereg import EXCLUDED_RUN_IDS, default_stage1

    assert len(EXCLUDED_RUN_IDS) == 19
    assert ORPHAN in EXCLUDED_RUN_IDS
    assert "sweep restart" in EXCLUDED_RUN_IDS[ORPHAN]
    assert default_stage1().excluded_run_ids == dict(EXCLUDED_RUN_IDS)


def test_the_exclusion_keys_on_run_id_and_not_on_the_cell():
    """THE WHOLE POINT. The kept run shares (suite, task, seed) with the orphan, so a
    (suite, task) exclusion would delete the run of record along with the duplicate and leave
    the cell empty -- silently, since a missing cell is not an error anywhere."""
    from pi_eval.prereg import excluded_runs
    from pi_eval.report import ELIGIBLE, Agg

    root = pathlib.Path(tempfile.mkdtemp())
    (root / "stage1.json").write_text(
        json.dumps({"excluded_run_ids": {ORPHAN: "duplicate cell from a sweep restart"}})
    )
    ids, note = excluded_runs(root)
    assert ids == frozenset({ORPHAN})
    assert "1 run" in note

    agg = Agg(
        parquet_dir=root,
        con=None,
        scorer_hash="sh",
        graph_version="v1",
        allow_contaminated=False,
        sigma_j={},
        excluded_run_ids=ids,
        prereg_note=note,
    )
    pred = agg.predicate
    assert pred.startswith(ELIGIBLE)
    assert f"'{ORPHAN}'" in pred and "r.run_id NOT IN" in pred
    assert "task_id" not in pred.split("r.run_id NOT IN")[1], (
        "the run exclusion must not reach for the cell the kept run also occupies"
    )


def test_no_seal_leaves_the_run_exclusion_empty(tmp_path):
    from pi_eval.prereg import excluded_runs

    ids, note = excluded_runs(tmp_path)
    assert ids == frozenset() and "no sealed stage 1" in note


# ------------------------------------------------ the two guards that decide whether to seal
#
# The drafted stage 1 could not be sealed: `make prereg-draft` exits 2 on
# BurnedPilotsMissing for musique. Measured, the pilot NEVER RAN. `conf/grids/tier1_pilot.yaml`
# takes musique at offset 200, 120 tasks, and declares no split, so its slice straddles the
# wall (73 train / 29 test / 18 dev) -- and those task ids carry 3,830 runs from OTHER grids
# (latent_trainset, growth_parents_*, dev_select_musique), none of them `pilot_flag = TRUE`.
# "The suite's tasks have runs" is not "a pilot ran here".


def test_a_pilot_grid_whose_tasks_carry_other_grids_runs_does_not_block_the_seal():
    """THE FALSE POSITIVE, in one assertion. 3,830 runs from three other grids sit on the
    pilot slice's musique ids. Under the old definition every one of them said "a pilot ran",
    and the seal refused over a pilot that never happened."""
    mod = _sealer()
    slices = {"musique": {"a", "b", "c"}}
    assert mod._pilots_that_ran(pilot_slices=slices, piloted={"musique": set()}) == set()
    assert mod._pilots_that_ran(pilot_slices=slices, piloted={}) == set()


def test_a_real_pilot_run_on_the_slice_still_says_the_pilot_ran():
    """The guard must keep its true positive: a `pilot_flag = TRUE` run on the pilot grid's
    OWN tasks is a pilot, and sealing then must not declare that none ever happened."""
    mod = _sealer()
    slices = {"musique": {"a", "b"}}
    assert mod._pilots_that_ran(pilot_slices=slices, piloted={"musique": {"b"}}) == {"musique"}
    with pytest.raises(mod.BurnedPilotsMissing, match="burned_pilot_ids is empty"):
        mod._check_burned_pilots({}, {"musique"})


def test_the_piloted_task_query_reads_pilot_flag_and_nothing_else():
    """The signal is the FLAG, not the presence of rows. Pinned on the source because the
    query needs a parquet nobody should have to build to assert a WHERE clause."""
    import inspect

    src = inspect.getsource(_sealer()._piloted_tasks)
    assert "pilot_flag = TRUE" in src
    assert "SELECT DISTINCT suite_id, task_id" in src


# ----------------------------------------- the assertion the seal actually rests on
#
# A trained-arm run that ALREADY EXISTS on a confirmatory test id means the checkpoint has
# been evaluated there before the document that authorises the comparison was written. The
# burned-pilot guard cannot see that: those runs carry no pilot_flag and sit on tasks no pilot
# grid selects.


def _runs_parquet(tmp_path, rows):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    d = tmp_path / "parquet"
    d.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), d / "runs.parquet")
    return d / "runs.parquet"


def _row(run_id, *, suite, task, arm, template=None):
    return dict(run_id=run_id, suite_id=suite, task_id=task, arm_id=arm, template_id=template or "")


def _test_id(suite="musique"):
    """A task id this repo's own splitter calls `test`, found rather than assumed."""
    from pinq.splitting import split_of

    for i in range(10_000):
        t = f"synthetic_task_{i}"
        if split_of(suite, t) == "test":
            return t
    raise AssertionError("no test-bucket id found")


def _train_id(suite="musique"):
    from pinq.splitting import split_of

    for i in range(10_000):
        t = f"synthetic_task_{i}"
        if split_of(suite, t) == "train":
            return t
    raise AssertionError("no train-bucket id found")


def test_a_trained_arm_run_on_a_confirmatory_test_id_is_found(tmp_path):
    mod = _sealer()
    t = _test_id()
    p = _runs_parquet(tmp_path, [_row("r1", suite="musique", task=t, arm="inquirer_trained")])
    bad = mod._trained_runs_on_confirmatory_tasks(runs_parquet=p)
    assert [b["run_id"] for b in bad] == ["r1"]
    assert bad[0]["recomputed_split"] == "test"
    with pytest.raises(mod.TrainedArmAlreadyOnTest, match="r1"):
        mod._check_trained_arm_not_on_test(bad)


def test_a_trained_arm_run_on_a_train_id_is_not_a_violation(tmp_path):
    """Training on the train split is the design. The check is about the CONFIRMATORY
    population, and firing on correct behaviour is how a check gets disabled."""
    mod = _sealer()
    p = _runs_parquet(
        tmp_path, [_row("r1", suite="musique", task=_train_id(), arm="inquirer_trained")]
    )
    assert mod._trained_runs_on_confirmatory_tasks(runs_parquet=p) == []


def test_a_prompted_arm_run_on_a_test_id_is_not_a_violation(tmp_path):
    """`inquirer_prompted` is the comparator and runs the confirmatory tasks on purpose; a
    policy that never trained cannot have trained on them."""
    mod = _sealer()
    p = _runs_parquet(
        tmp_path, [_row("r1", suite="musique", task=_test_id(), arm="inquirer_prompted")]
    )
    assert mod._trained_runs_on_confirmatory_tasks(runs_parquet=p) == []


def test_the_split_is_recomputed_with_the_TEMPLATE_argument(tmp_path):
    """THE FALSE ALARM THE DRAFT'S FIRST PASS PRODUCED. musique hashes its split on the
    TEMPLATE where one exists, so `split_of(suite, task_id)` buckets a different thing from
    `split_of(suite, task_id, template_id)` -- and on a FIXED-split suite the two-argument
    call raises outright. The template must reach the call."""
    import inspect

    from pinq.splitting import split_of

    mod = _sealer()
    src = inspect.getsource(mod._trained_runs_on_confirmatory_tasks)
    assert "template_id" in src

    # a (task, template) pair the two callings disagree about
    task, template = None, None
    for i in range(10_000):
        t, tpl = f"task_{i}", f"template_{i}"
        if split_of("musique", t) != split_of("musique", t, tpl):
            task, template = t, tpl
            break
    assert task, "no disagreeing pair found; the fixture cannot make the point"

    p = _runs_parquet(
        tmp_path,
        [_row("r1", suite="musique", task=task, template=template, arm="inquirer_trained")],
    )
    found = mod._trained_runs_on_confirmatory_tasks(runs_parquet=p)
    want = split_of("musique", task, template) == "test"
    assert bool(found) is want, "the verdict must follow the TEMPLATE-aware split"


def test_an_absent_parquet_is_not_a_silent_pass(tmp_path):
    """Nothing to read is not evidence of nothing to find, and the seal must say which."""
    mod = _sealer()
    with pytest.raises(FileNotFoundError):
        mod._trained_runs_on_confirmatory_tasks(runs_parquet=tmp_path / "nope.parquet")


def test_a_clean_scan_raises_nothing():
    mod = _sealer()
    mod._check_trained_arm_not_on_test([])  # must not raise


def test_a_missing_corpus_skips_the_suite_instead_of_killing_the_sealer():
    """`_resolve_corpus` raises SystemExit, which derives from BaseException, so the
    `except Exception` that meant "a suite whose corpus is absent cannot have been piloted"
    did not catch it: a checkout without data/ crashed the sealer rather than skipping."""
    import inspect

    src = inspect.getsource(_sealer()._pilot_slices)
    assert "except (Exception, SystemExit)" in src


def test_train_split_violations_passes_the_template_as_the_template():
    """THE SAME BUG, in the check that guards the trained arm's own table. It called
    `split_of(suite, template_id or task_id)`, which gives the right BUCKET (split_key hashes
    on `template_id or task_id`) and is not the same call: a FIXED_SPLITS suite resolves by
    looking the TASK id up in its table, so a tau2_airline trained run carrying a template
    raised UnknownFixedSplitTask instead of being checked."""
    import inspect

    from pi_eval.report import train_split_violations

    src = inspect.getsource(train_split_violations)
    assert 'str(r.get("template_id") or "") or None' in src
    assert 'key = str(r.get("template_id")' not in src


def test_a_fixed_split_suite_with_a_template_is_checked_not_crashed():
    """tau2_airline has a FIXED split. The old call handed it a template id as a task id,
    which is not in the table, and `split_of` raises there rather than answering."""
    from pinq.splitting import UnknownFixedSplitTask, split_of

    with pytest.raises(UnknownFixedSplitTask):
        split_of("tau2_airline", "some_template_id")
    assert split_of("tau2_airline", "2", "some_template_id") == "test"
