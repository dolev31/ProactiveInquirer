"""The judge as an instrument: bias diagnostics, noise floor, agreement, length adjustment."""

import json
import math

import pytest

from pi_eval.judges.paired import (
    DISQUALIFY_POSITION_BIAS,
    judge_is_usable,
    krippendorff_alpha_nominal,
    length_adjusted_effect,
    position_bias,
    resolve_pair,
    sigma_j,
    win_tie_loss,
)
from pi_eval.judges.types import CITATION_SCORE, Judgment


def _j(order, sign, task="t", mag=1.0, la=100, lb=100, retest=None):
    return Judgment(
        judgment_id=f"{task}-{order}",
        run_id_a="A",
        run_id_b="B",
        suite_id="s",
        task_id=task,
        criterion="paired_pref",
        order=order,
        judge_family="fam",
        judge_model="m",
        judge_prompt_sha="p",
        pref_sign=sign,
        magnitude=mag,
        len_a_words=la,
        len_b_words=lb,
        retest_group_id=retest,
    )


def test_order_consistent_pair_keeps_its_sign():
    o = resolve_pair(_j("ab", 1), _j("ba", 1))
    assert o.sign == 1 and o.order_consistent


def test_order_inconsistent_pair_becomes_a_tie():
    """Such a pair says the judge is order-sensitive, not which answer is better. Counting it
    either way would import position bias straight into the effect estimate."""
    o = resolve_pair(_j("ab", 1), _j("ba", -1))
    assert o.sign == 0 and not o.order_consistent and o.magnitude == 0.0


def test_position_bias_detects_a_judge_that_always_picks_first():
    js = [_j("ab", 1), _j("ba", -1), _j("ab", 1), _j("ba", -1)]
    assert position_bias(js) == pytest.approx(1.0)
    ok, why = judge_is_usable(js)
    assert not ok and "position bias" in why


def test_an_unbiased_judge_is_usable():
    js = [_j("ab", 1), _j("ba", 1), _j("ab", -1), _j("ba", -1)]
    assert position_bias(js) == pytest.approx(0.5)
    assert judge_is_usable(js)[0]


def test_disqualification_threshold_is_explicit():
    assert DISQUALIFY_POSITION_BIAS == 0.55


def test_ties_are_excluded_from_the_bias_diagnostic():
    assert math.isnan(position_bias([_j("ab", 0), _j("ba", 0)]))


def test_sigma_j_is_estimated_from_test_retest():
    """Identical items judged twice: the spread IS the instrument's noise."""
    js = [
        _j("ab", 1, task="t1", mag=3.0, retest="g1"),
        _j("ab", 1, task="t1", mag=5.0, retest="g1"),
        _j("ab", 1, task="t2", mag=2.0, retest="g2"),
        _j("ab", 1, task="t2", mag=4.0, retest="g2"),
    ]
    # stdev([3,5]) == stdev([2,4]) == sqrt(2)
    assert sigma_j(js) == pytest.approx(math.sqrt(2))


def test_sigma_j_is_nan_without_retest_groups():
    assert math.isnan(sigma_j([_j("ab", 1)]))


def test_length_adjustment_removes_a_pure_length_artifact():
    """Construct a judge that ONLY ever prefers the longer answer. The raw effect is +1.0 and
    the adjusted effect must collapse toward 0 — this is the control that decides whether
    there is a paper at all."""
    outs = []
    for i in range(20):
        longer_a = i % 2 == 0
        la, lb = (200, 100) if longer_a else (100, 200)
        outs.append(
            resolve_pair(
                _j("ab", 1 if longer_a else -1, task=f"t{i}", la=la, lb=lb),
                _j("ba", 1 if longer_a else -1, task=f"t{i}", la=la, lb=lb),
            )
        )
    res = length_adjusted_effect(outs)
    assert res["raw"] == pytest.approx(0.0, abs=1e-9)
    assert res["slope"] > 0, "the judge must be detected as length-preferring"


def test_length_adjustment_leaves_a_genuine_effect_intact():
    """LENGTHS MUST ACTUALLY VARY, or this tests nothing.

    Every pair here used to be la=100, lb=100, so the regressor had no variance and the slope
    was undefined -- and the assertion passed only because `sxx == 0` returned a slope of 0.0,
    making `adjusted == raw` by arithmetic rather than by measurement. CLAUDE.md rule 4: the
    belief ("a genuine effect survives adjustment for length") is right, the fixture did not
    exercise it. Lengths now differ across pairs while A wins regardless, which is the case the
    test names: an effect that has nothing to do with length."""
    outs = [
        resolve_pair(
            _j("ab", 1, task=f"t{i}", la=100 + 10 * i, lb=100),
            _j("ba", 1, task=f"t{i}", la=100 + 10 * i, lb=100),
        )
        for i in range(20)
    ]
    res = length_adjusted_effect(outs)
    assert res["identifiable"] == 1.0, "the fixture must have length variance to adjust for"
    assert res["raw"] == pytest.approx(1.0)
    assert res["adjusted"] == pytest.approx(1.0, abs=1e-9)
    assert res["slope"] == pytest.approx(0.0, abs=1e-9), "A wins at every length"


def test_win_tie_loss_counts_order_inconsistency_separately():
    outs = [
        resolve_pair(_j("ab", 1, task="a"), _j("ba", 1, task="a")),
        resolve_pair(_j("ab", 1, task="b"), _j("ba", -1, task="b")),
        resolve_pair(_j("ab", -1, task="c"), _j("ba", -1, task="c")),
    ]
    w = win_tie_loss(outs)
    # `unjudged` is a fourth category, not a fourth way to be a tie: a pair the judge never
    # graded is reported separately from one it graded as equal. Still asserted exactly.
    assert w == {"win": 1, "tie": 1, "loss": 1, "order_inconsistent": 1, "unjudged": 0}


def test_krippendorff_alpha_is_one_on_perfect_agreement():
    r = {f"u{i}": {"j1": "Supported", "j2": "Supported"} for i in range(6)}
    r["u6"] = {"j1": "Omitted", "j2": "Omitted"}
    assert krippendorff_alpha_nominal(r) == pytest.approx(1.0)


def test_krippendorff_alpha_is_near_zero_on_chance_agreement():
    r = {
        "u0": {"j1": "Supported", "j2": "Omitted"},
        "u1": {"j1": "Omitted", "j2": "Supported"},
        "u2": {"j1": "Supported", "j2": "Omitted"},
        "u3": {"j1": "Omitted", "j2": "Supported"},
    }
    assert krippendorff_alpha_nominal(r) < 0.1


def test_citation_scores_match_the_upstream_rubric():
    """Reimplemented, not vendored: the upstream eval repo carries no licence."""
    assert CITATION_SCORE["full_support"] == 1.0
    assert CITATION_SCORE["partial_support"] == 0.5
    assert CITATION_SCORE["no_support"] == 0.0


# ------------------------------------------------------------ an unjudged pair is not a verdict


def _meta(**kw):
    d = dict(
        criterion="quality",
        suite_id="s",
        task_id="t",
        run_id_a="ra",
        run_id_b="rb",
        judge_family="f",
        judge_model="m",
        judge_prompt_sha="sha",
    )
    d.update(kw)
    return d


def _mkj(**kw):
    from pi_eval.judges.types import Judgment

    d = dict(
        judgment_id="x",
        run_id_a="ra",
        run_id_b="rb",
        suite_id="s",
        task_id="t",
        criterion="quality",
        order="ab",
        judge_family="f",
        judge_model="m",
        judge_prompt_sha="sha",
    )
    d.update(kw)
    return Judgment(**d)


def _pair(sa, sb, la=100, lb=100):
    from pi_eval.judges.paired import paired_from_absolute, resolve_pair

    a = [_mkj(score=sa)] if sa is not None else []
    b = [_mkj(score=sb)] if sb is not None else []
    kw = dict(_meta(), len_a_words=la, len_b_words=lb)
    return resolve_pair(
        paired_from_absolute(a=a, b=b, order="ab", **kw),
        paired_from_absolute(a=a, b=b, order="ba", **kw),
    )


def test_a_grading_the_judge_failed_to_produce_is_not_a_tie():
    """`mean_score` returns NaN so that "0.0 would read as a verdict", and the caller turned it
    straight back into one: pref_sign 0, the same value a judge that read both answers and found
    them equal emits. Every parse failure then entered the effect estimate as evidence of no
    difference, diluting it 1:1 with the instrument's failure rate."""
    from pi_eval.judges.paired import length_adjusted_effect, win_tie_loss

    wins = [_pair(8.0, 5.0, la=100 + 10 * i) for i in range(10)]
    failed = [_pair(None, 5.0, la=100 + 10 * i) for i in range(10)]
    counts = win_tie_loss(wins + failed)
    assert counts["tie"] == 0, "a pair the judge never graded is not a measured tie"
    assert counts["unjudged"] == 10 and counts["win"] == 10

    # Lengths vary so the adjustment is identifiable at all -- with a constant regressor
    # `adjusted` is now correctly NaN, and this assertion would be vacuous.
    eff = length_adjusted_effect(wins + failed)
    assert eff["identifiable"] == 1.0
    assert eff["adjusted"] == pytest.approx(1.0, abs=1e-9), (
        "counted as ties these ten failures halve the measured effect"
    )


def test_a_real_tie_is_still_a_tie():
    """The distinction only matters if the other side of it still works."""
    from pi_eval.judges.paired import win_tie_loss

    counts = win_tie_loss([_pair(5.0, 5.0) for _ in range(4)])
    assert counts["tie"] == 4 and counts["unjudged"] == 0


# --------------------------------------------------- an all-tie criterion is a result, not a fault


def test_a_criterion_on_which_everything_ties_keeps_its_judge():
    """position_bias is NaN whenever nothing is decisive, and that was mapped to "unusable" --
    so the judge was disqualified exactly when the answer was "no effect", and harness.py then
    stripped every judge-derived metric for that suite/criterion including kpr_incremental, a
    primary. A null was deleted rather than reported."""
    from pi_eval.judges.paired import judge_is_usable

    ok, why = judge_is_usable([_mkj(pref_sign=0) for _ in range(20)])
    assert ok, why
    assert "not estimable" in why


def test_a_judge_that_parsed_nothing_is_still_disqualified():
    """The other way to have no decisive judgment, and this one IS an instrument failure."""
    from pi_eval.judges.paired import judge_is_usable

    ok, why = judge_is_usable([_mkj(pref_sign=None) for _ in range(20)])
    assert not ok and "parsed" in why


def test_position_bias_still_disqualifies():
    from pi_eval.judges.paired import judge_is_usable

    ok, why = judge_is_usable([_mkj(pref_sign=1, order="ab") for _ in range(20)])
    assert not ok and "position bias" in why


# ------------------------------------------------------------- Krippendorff on unbalanced designs


def test_alpha_normalises_per_unit_not_over_the_pooled_pair_count():
    """D_o divides each unit's disagreement by (m_u - 1) and the total by the number of RATINGS.
    Dividing pooled disagreement by the pooled pair count over-weights units with more raters,
    and the function's own docstring says the unbalanced case is the normal one."""
    from pi_eval.judges.paired import krippendorff_alpha_nominal

    unbalanced = {
        "u1": {"f1": "A", "f2": "B", "f3": "A"},
        "u2": {"f1": "A", "f2": "A"},
        "u3": {"f1": "B", "f2": "B"},
        "u4": {"f1": "A", "f2": "A"},
        "u5": {"f1": "B", "f2": "A"},
    }
    assert krippendorff_alpha_nominal(unbalanced) == pytest.approx(0.2857, abs=1e-4)


def test_alpha_is_unchanged_on_a_balanced_design():
    """The two normalisations coincide when every unit has the same number of raters, so no
    balanced result moves."""
    from pi_eval.judges.paired import krippendorff_alpha_nominal

    balanced = {"u1": {"a": "A", "b": "B"}, "u2": {"a": "A", "b": "A"}, "u3": {"a": "B", "b": "B"}}
    assert krippendorff_alpha_nominal(balanced) == pytest.approx(0.4444, abs=1e-4)


def test_perfect_agreement_is_alpha_one():
    from pi_eval.judges.paired import krippendorff_alpha_nominal

    assert krippendorff_alpha_nominal(
        {"u1": {"a": "A", "b": "A", "c": "A"}, "u2": {"a": "B", "b": "B"}}
    ) == pytest.approx(1.0)


# ------------------------------------------ the position-bias gate could not fire, and said 0.500


def test_position_bias_is_reported_as_not_measured_when_it_could_not_have_moved():
    """All three graders (kpr, citation, quality) grade ONE report in isolation, and `order`
    reaches only the judgment_id hash and a stamped field -- never the prompt, never the
    request. So grade("ab") == grade("ba"), every ab/ba pair carries the same pref_sign, and
    exactly one of each pair counts as "first": position_bias is 0.5 EXACTLY, by construction,
    against a 0.55 threshold. Measured over 200 pairs before this change: 0.5, gate unfireable.

    The double grading still earns its keep as a tripwire for a future pairwise judge -- but it
    costs a second billed call per run per criterion, and it must not be reported as a passed
    check."""
    import random

    from pi_eval.judges.paired import judge_is_usable, position_bias

    rng = random.Random(0)
    js = []
    for i in range(50):
        sa, sb = rng.uniform(0, 10), rng.uniform(0, 10)
        for o in ("ab", "ba"):
            js.append(_mk_pair_judgment(sa, sb, order=o, task_id=f"t{i}"))
    assert position_bias(js) == pytest.approx(0.5), "0.5 by construction, not by measurement"
    ok, why = judge_is_usable(js)
    assert ok and "NOT MEASURED" in why


def test_an_order_sensitive_judge_still_gets_a_real_number():
    import random

    from pi_eval.judges.paired import judge_is_usable

    rng = random.Random(1)
    js = []
    for i in range(50):
        sa, sb = rng.uniform(0, 10), rng.uniform(0, 10)
        js.append(_mk_pair_judgment(sa, sb, order="ab", task_id=f"t{i}"))
        flip = i % 3 == 0
        js.append(
            _mk_pair_judgment(sb if flip else sa, sa if flip else sb, order="ba", task_id=f"t{i}")
        )
    ok, why = judge_is_usable(js)
    assert ok and "NOT MEASURED" not in why


def _mk_pair_judgment(sa, sb, *, order, task_id):
    from pi_eval.judges.paired import paired_from_absolute

    return paired_from_absolute(
        a=[_mkj(score=sa)],
        b=[_mkj(score=sb)],
        order=order,
        criterion="quality",
        suite_id="s",
        task_id=task_id,
        run_id_a="ra",
        run_id_b="rb",
        judge_family="f",
        judge_model="m",
        judge_prompt_sha="sha",
    )


# ------------------------------------------------------- sigma_J pools variances, not SDs


def test_sigma_j_pools_variances_rather_than_averaging_standard_deviations():
    """E[s] = sigma*sqrt(2/pi) ~= 0.798*sigma for the two-replicate design this estimator
    documents, so the mean of per-group SDs is biased ~20% LOW. Measured against a known
    sigma=1.0 over 400 groups x 200 datasets: mean-of-SDs 0.7958, pooled 1.0012.

    sigma_J is the gate on what may be CLAIMED -- no effect below 2*sigma_J/sqrt(n) is
    reportable -- so the bias is toward reporting more."""
    import random

    from pi_eval.judges.paired import sigma_j

    rng = random.Random(0)
    js = []
    for i in range(600):
        for _ in range(2):
            js.append(_mkj(retest_group_id=f"g{i}", magnitude=rng.gauss(0, 1.0)))
    assert sigma_j(js) == pytest.approx(1.0, abs=0.06)


def test_sigma_j_agrees_with_the_script_that_produces_the_published_constant():
    """scripts/measure_sigma_j.py computes SD(differences)/sqrt(2). Over two-replicate groups
    the pooled estimator is algebraically identical, and two estimators of one FROZEN prereg
    constant that disagreed is how the published value comes to depend on which function
    produced it."""
    import math
    import random
    import statistics

    from pi_eval.judges.paired import sigma_j

    rng = random.Random(3)
    js, diffs = [], []
    for i in range(500):
        x1, x2 = rng.gauss(0, 1.3), rng.gauss(0, 1.3)
        js += [
            _mkj(retest_group_id=f"g{i}", magnitude=x1),
            _mkj(retest_group_id=f"g{i}", magnitude=x2),
        ]
        diffs.append(x1 - x2)
    # Agree to O(1/n), not exactly: `stdev` centres on the sample mean and divides by n-1.
    # A tolerance that still catches the 20% bias this replaced (which would be ~0.26 here).
    assert sigma_j(js) == pytest.approx(statistics.stdev(diffs) / math.sqrt(2), abs=0.01)


# ------------------------------------------------------------ claim_id is a judgment identity


def test_duplicate_claim_ids_are_refused():
    """`grade` hashes claim_id into judgment_id and harness.py dedups on judgment_id, so two
    claims sharing a claim_id collide and the SECOND IS SILENTLY DROPPED -- a shorter
    denominator for citation_support, invisible because nothing downstream sees the count the
    judge emitted. Verified: two different claims with claim_id=1 both hash to
    4b3e925c746bf504..."""
    from pi_eval.judges.citation import JudgeParseError, parse_claims

    dup = json.dumps(
        {
            "claims": [
                {"claim_id": 1, "claim": "Paris is in France", "sources": ["u1"]},
                {"claim_id": 1, "claim": "The Nile is in Egypt", "sources": ["u2"]},
            ]
        }
    )
    with pytest.raises(JudgeParseError, match="claim_id is not unique"):
        parse_claims(dup)


def test_unique_claim_ids_still_parse():
    from pi_eval.judges.citation import parse_claims

    ok = json.dumps(
        {
            "claims": [
                {"claim_id": 1, "claim": "Paris is in France", "sources": ["u1"]},
                {"claim_id": 2, "claim": "The Nile is in Egypt", "sources": ["u2"]},
            ]
        }
    )
    assert len(parse_claims(ok)) == 2


def test_a_length_adjustment_with_no_variance_is_not_a_measured_null():
    """`slope = sxy / sxx if sxx else 0.0` made `adjusted == raw` -- the STRONGEST reading of
    the paper's claim ("the effect survives adjustment for length") produced by having
    performed no adjustment at all, and indistinguishable in the table from a genuine finding
    that length explained nothing.

    It is the EXPECTED case, not a corner: the frozen Answerer applies the same word cap to
    every arm by construction, precisely so length cannot be what a judge rewards. Every pair
    having the same delta-log-length is what success looks like."""
    import math

    from pi_eval.judges.paired import PairedOutcome, length_adjusted_effect

    same = [PairedOutcome(f"t{i}", 1 if i % 3 else -1, 1.0, True, 0.0) for i in range(30)]
    r = length_adjusted_effect(same)
    assert math.isnan(r["adjusted"]) and math.isnan(r["slope"])
    assert r["identifiable"] == 0.0
    assert not math.isnan(r["raw"]), "the raw effect is still measured"

    varied = [
        PairedOutcome(f"t{i}", 1 if i % 3 else -1, 1.0, True, (i - 15) * 0.1) for i in range(30)
    ]
    v = length_adjusted_effect(varied)
    assert not math.isnan(v["adjusted"]) and v["identifiable"] == 1.0


# -------------------------------------------- a judge metric now carries the n it was computed over


def _kp_run(n_gold=40):
    import dataclasses

    from pi_eval.judges.harness import RunInput

    class _T:
        suite_id = "drgym"
        task_id = "t1"
        docs = {}
        question = "q"
        key_points = [(f"kp{i}", f"point {i}") for i in range(n_gold)]

    run = RunInput.__new__(RunInput)
    object.__setattr__(run, "task", _T())
    for f in dataclasses.fields(RunInput):
        if f.name != "task":
            try:
                object.__setattr__(run, f.name, "")
            except Exception:
                pass
    return run


def _kp(kp, label):
    from pi_eval.judges.types import Judgment

    return Judgment(
        judgment_id=f"j{kp}",
        run_id_a="r",
        run_id_b=None,
        suite_id="drgym",
        task_id="t1",
        criterion="keypoint",
        order="ab",
        judge_family="f",
        judge_model="m",
        judge_prompt_sha="sha",
        label=label,
        key_point_id=kp,
    )


def test_a_judge_metric_carries_the_n_it_was_computed_over():
    """Judge rows were emitted with no `n=`, and `_emit` defaults n to 1 -- so
    kpr_incremental, the drgym PRIMARY endpoint, reached scores.parquet claiming a sample size
    of one. 40 gold key points with verdicts parsed for 2 gave keypoint_recall = 1.0 at n=1,
    and nothing downstream could see that 38 were never judged.

    `report.floor_flag` divides by sqrt(n): at sigma_J=0.5 the floor was 2*0.5/sqrt(1) = 1.000,
    above every possible value of a [0,1] metric."""
    from pi_eval.judges.harness import _metrics_for

    vals, ns = _metrics_for(
        _kp_run(40), {"keypoint": (_kp("kp0", "Supported"), _kp("kp1", "Supported"))}
    )
    assert vals["kpr_incremental"] == pytest.approx(1.0)
    for name in vals:
        n, note = ns[name]
        assert n > 1, f"{name} still reports n={n}"
        assert "38 of 40" in note, name


def test_the_n_is_the_denominator_the_value_actually_used_not_the_gold_universe():
    """`kpr_incremental.score` computes raw over JUDGMENT ROWS and incremental over JUDGED
    POINTS; nothing divides by n_gold. Using the gold universe as n would claim a precision the
    measurement does not have -- n=40 where two points were judged. n_gold goes in the note."""
    from pi_eval.judges.harness import _metrics_for

    _, ns = _metrics_for(
        _kp_run(40), {"keypoint": (_kp("kp0", "Supported"), _kp("kp1", "Supported"))}
    )
    assert ns["kpr_incremental"][0] == 2, "judged points, not the 40-point universe"
    assert ns["keypoint_recall"][0] == 2, "judgment rows"
    assert "40" in ns["kpr_incremental"][1], "the universe is recorded in the note"


def test_no_shrinkage_note_when_everything_was_judged():
    from pi_eval.judges.harness import _metrics_for

    js = tuple(_kp(f"kp{i}", "Supported") for i in range(3))
    _, ns = _metrics_for(_kp_run(3), {"keypoint": js})
    assert ns["kpr_incremental"] == (3, "")


def test_the_scorer_passes_the_n_through():
    import inspect

    from pi_eval import score

    src = inspect.getsource(score)
    assert "judged.metric_n" in src and "n=n," in src


def test_the_fidelity_instrument_has_a_production_caller():
    """`pi_eval.judges.fidelity` is fully built -- ReferenceKPR, load_kpr_reference, kpr_delta,
    KPRDelta -- and had NO caller: `grep -rn fidelity src/ tests/ scripts/` matched only
    docstrings.

    That matters because docs/DATA.md commits IN WRITING to publishing the number: the DRGym
    eval harness has no licence file, so its judge prompts are reimplemented here rather than
    vendored, and the justification given for that is "a published fidelity delta
    (pi_eval.judges.fidelity)". judges/{kpr,citation,quality}.py each repeat the promise. A
    documented obligation that no code path can discharge is the same defect class as a sealed
    constant nothing reads."""
    import inspect

    from pi_run import cli

    assert hasattr(cli, "cmd_verify_judge_fidelity")
    src = inspect.getsource(cli.cmd_verify_judge_fidelity)
    assert "load_kpr_reference" in src and "kpr_delta" in src


def test_it_refuses_rather_than_reporting_a_delta_of_zero(tmp_path):
    """It cannot invent the reference data -- that is upstream's own judge output on the same
    reports -- so an absent file must say what is needed, not report perfect agreement."""
    import argparse

    from pi_run.cli import cmd_verify_judge_fidelity

    a = argparse.Namespace(
        reference=str(tmp_path / "nope.json"),
        parquet_dir=str(tmp_path),
        system="s",
        judge_model="m",
        source_url="u",
        commit="c",
        ours_model="",
    )
    assert cmd_verify_judge_fidelity(a) == 1
