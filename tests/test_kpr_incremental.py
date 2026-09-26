"""kpr_incremental: the DeepResearchGym primary endpoint (P3).

THE PROPERTY THIS FILE EXISTS FOR. Raw key-point recall divides by the number of JUDGMENTS,
so a report that restates one key point twenty times and a report that covers twenty key
points once each produce the same 20/22. On a suite whose entire question is "did the system
discover the things it did not know to look for", that is the one distinction that has to
survive, and it is why the endpoint is the incremental form.
"""

import math

import pytest

from pi_eval.judges.types import Judgment
from pi_eval.metrics import kpr_incremental as kpri


def _j(kp, label, run="r1", task="t1", criterion="keypoint", n=[0]):
    n[0] += 1
    return Judgment(
        judgment_id=f"j{n[0]}",
        run_id_a=run,
        run_id_b=None,
        suite_id="drgym",
        task_id=task,
        criterion=criterion,
        order="ab",
        judge_family="fam",
        judge_model="m",
        judge_prompt_sha="p",
        label=label,
        score={"Supported": 1.0}.get(label, 0.0),
        key_point_id=kp,
    )


def test_restating_one_point_must_not_outscore_covering_many():
    """The whole reason the endpoint is incremental. Both reports collect 22 verdicts and 20
    of them are Supported, so RAW cannot tell them apart -- and incremental scores the broad
    report 2.7x the repetitive one."""
    gold = [f"kp{i}" for i in range(1, 23)]
    repetitive = [_j("kp1", "Supported") for _ in range(20)] + [
        _j("kp2", "Omitted"),
        _j("kp3", "Omitted"),
    ]
    broad = [_j(f"kp{i}", "Supported") for i in range(1, 21)] + [
        _j("kp21", "Omitted"),
        _j("kp22", "Omitted"),
    ]

    a, b = kpri.score(repetitive, gold), kpri.score(broad, gold)
    assert a.raw == pytest.approx(20 / 22) and b.raw == pytest.approx(20 / 22)
    assert a.incremental == pytest.approx(1 / 3)
    assert b.incremental == pytest.approx(20 / 22)
    assert b.incremental > a.incremental * 2.7


def test_the_two_forms_coincide_when_every_point_is_judged_once():
    """Which is the normal path: judges.kpr.grade makes exactly one call per key point. The
    gap between them IS the repetition diagnostic, so it must be exactly 0 here."""
    js = [_j("kp1", "Supported"), _j("kp2", "Omitted"), _j("kp3", "Supported")]
    r = kpri.score(js, ["kp1", "kp2", "kp3"])
    assert r.raw == pytest.approx(2 / 3) == pytest.approx(r.incremental)
    assert r.repetition_gap == pytest.approx(0.0)


def test_one_supported_verdict_credits_the_point_however_it_was_reached():
    """Credit each key point ONCE however many claims cover it -- including when the other
    verdicts on that point disagree. A point the report supports somewhere is covered."""
    js = [_j("kp1", "Omitted"), _j("kp1", "Supported"), _j("kp1", "Contradicted")]
    r = kpri.score(js, ["kp1"])
    assert r.incremental == pytest.approx(1.0)
    assert r.raw == pytest.approx(1 / 3)
    assert r.n_points_contradicted == 0, "a supported point is not also a contradicted one"


def test_a_contradiction_scores_zero_and_stays_countable():
    js = [_j("kp1", "Contradicted"), _j("kp2", "Omitted")]
    r = kpri.score(js, ["kp1", "kp2"])
    assert r.incremental == 0.0 and r.raw == 0.0
    assert r.n_points_contradicted == 1
    assert r.contradiction_rate == pytest.approx(0.5)


def test_an_unjudged_point_is_excluded_from_the_denominator_and_counted():
    """Scoring an unparseable verdict as unsupported charges the system for the judge's
    stutter. It leaves the denominator and shows up as n_unjudged instead."""
    r = kpri.score([_j("kp1", "Supported")], ["kp1", "kp2", "kp3"])
    assert r.incremental == pytest.approx(1.0), "1 of 1 judged, not 1 of 3 gold"
    assert r.n_gold == 3 and r.n_points_judged == 1 and r.n_unjudged == 2


def test_nothing_judged_is_nan_and_never_zero():
    """0.0 reads as 'the system supported nothing'; NaN reads as 'nothing was judged'."""
    r = kpri.score([], ["kp1"])
    assert math.isnan(r.raw) and math.isnan(r.incremental)
    assert math.isnan(r.contradiction_rate) and math.isnan(r.repetition_gap)
    assert r.as_dict()["kpr_incremental"] != r.as_dict()["kpr_incremental"]  # NaN


def test_a_verdict_naming_a_key_point_the_gold_does_not_have_is_a_join_bug_not_credit():
    r = kpri.score([_j("kp1", "Supported"), _j("kp999", "Supported")], ["kp1", "kp2"])
    assert r.n_points_judged == 1 and r.n_points_supported == 1
    assert r.incremental == pytest.approx(1.0)


def test_a_judgment_with_no_key_point_id_is_dropped_not_pooled():
    """Pooling them under a shared empty key would merge unrelated points into one unit and
    let a single Supported verdict credit all of them."""
    js = [_j("kp1", "Omitted"), _j("", "Supported"), _j(None, "Supported")]
    assert kpri.score(js).n_points_judged == 1
    assert kpri.score(js).incremental == 0.0


def test_only_key_point_judgments_are_counted():
    """A citation-support judgment carries a key_point_id too (`claim7`). Pooling the two
    criteria would average a citation label into key-point recall."""
    js = [_j("kp1", "Supported"), _j("claim7", "full_support", criterion="citation_support")]
    assert kpri.score(js).n_judgments == 1


def test_by_run_keys_on_run_id_a_and_the_task_s_gold_list():
    js = [
        _j("kp1", "Supported", run="A", task="900001"),
        _j("kp2", "Omitted", run="A", task="900001"),
        _j("kp1", "Supported", run="B", task="900001"),
        _j("kp2", "Supported", run="B", task="900001"),
    ]
    out = kpri.by_run(js, {"900001": ["kp1", "kp2"]})
    assert sorted(out) == ["A", "B"]
    assert out["A"].incremental == pytest.approx(0.5)
    assert out["B"].incremental == pytest.approx(1.0)


def test_the_metric_names_it_contributes_are_the_declared_ones():
    """as_dict() is what score() emits, so its keys must be in the metric register or the
    rows would fail the frozen contract on write."""
    from pi_eval.score import BY_NAME

    r = kpri.score([_j("kp1", "Supported")], ["kp1"])
    for name in r.as_dict():
        assert name in BY_NAME, name
        assert BY_NAME[name].judge_derived, f"{name} must sit behind the sigma_J noise floor"
