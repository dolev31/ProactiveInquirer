"""The three reimplemented DeepResearchGym judges, as instruments.

NO NETWORK, NO KEY, NO MODEL. Every judge here is driven by a scripted client, because what
is being tested is the PARSER and the WIRING, and both must be exercised against replies a
real model will eventually produce — including the malformed ones.

THE PROPERTY THIS FILE EXISTS FOR: a malformed verdict RAISES. Upstream uses OpenAI
structured outputs, so it never had to parse; we do, and the tempting `.get("label",
"Supported")` would convert every provider hiccup into a passing grade and bias key-point
recall upward by however often the judge stutters.
"""

import math

import pytest

from pi_eval.judges import citation, fidelity, kpr, quality
from pi_eval.judges._llm import JUDGE_TEMPERATURE, JudgeParseError
from pi_eval.judges.paired import (
    both_orders,
    judge_is_usable,
    krippendorff_alpha_nominal,
    mean_score,
    paired_from_absolute,
    position_bias,
    ratings_by_family,
    resolve_pair,
    sigma_j,
)
from pi_eval.judges.types import CITATION_SCORE, Judgment

REPORT = (
    "Oslo removed on-street parking and NO2 fell "
    "(https://example.org/oslo-car-free). Retail revenue effects differ by sector "
    "(https://example.org/retail-pedestrianisation)."
)
QUESTION = "should city centres ban private cars"
DOCS = {
    "https://example.org/oslo-car-free": "Oslo measured a fall in NO2 after removing parking.",
    "https://example.org/retail-pedestrianisation": "Revenue effects differ by sector.",
}


class ScriptedLLM:
    """Returns canned replies in order and records exactly how it was called."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def complete(self, *, role, messages, seed, max_tokens=None, **kw):
        self.calls.append({"role": role, "messages": messages, "seed": seed, **kw})
        if not self.replies:
            raise AssertionError("the judge made more calls than the script provides")
        return self.replies.pop(0), None


class ConstantLLM:
    def __init__(self, reply):
        self.reply = reply
        self.n = 0

    def complete(self, *, role, messages, seed, max_tokens=None, **kw):
        self.n += 1
        return self.reply, None


# ------------------------------------------------------------------ key-point recall


@pytest.mark.parametrize(
    "label,score", [("Supported", 1.0), ("Omitted", 0.0), ("Contradicted", 0.0)]
)
def test_the_three_labels_map_to_the_documented_scores(label, score):
    assert kpr.KEYPOINT_SCORE[label] == score


def test_a_contradiction_is_not_partial_credit_but_keeps_its_label():
    """Recall counts Supported only, so a contradiction scores 0 like an omission. The label
    survives because 'contradicted 8% of the key points' and 'missed them' are different
    findings about a system."""
    llm = ScriptedLLM('{"label": "Contradicted", "justification": "says the opposite"}')
    (j,) = kpr.grade(
        llm,
        key_points=[("kp1", "NO2 fell")],
        report=REPORT,
        suite_id="drgym",
        task_id="900001",
        run_id="r1",
        judge_model="m",
    )
    assert j.label == "Contradicted" and j.score == 0.0
    assert j.criterion == "keypoint" and j.key_point_id == "kp1" and j.run_id_b is None


@pytest.mark.parametrize(
    "reply",
    [
        '{"justification": "no label at all"}',
        '{"label": "supported"}',  # case matters: it is not the label we asked for
        '{"label": "Partially Supported"}',
        '{"label": 1}',
        "Supported",
        '["Supported"]',
        "",
        "   ",
    ],
)
def test_a_malformed_verdict_raises_and_never_becomes_a_pass(reply):
    with pytest.raises(JudgeParseError):
        kpr.parse(reply)


def test_a_fenced_json_block_is_the_only_repair_performed():
    label, _ = kpr.parse('```json\n{"label": "Omitted", "justification": "x"}\n```')
    assert label == "Omitted"
    with pytest.raises(JudgeParseError):
        kpr.parse('Here is my verdict: {"label": "Omitted"}')


def test_grade_propagates_a_parse_error_rather_than_scoring_the_item():
    llm = ScriptedLLM("not json")
    with pytest.raises(JudgeParseError):
        kpr.grade(
            llm,
            key_points=[("kp1", "x")],
            report=REPORT,
            suite_id="s",
            task_id="t",
            run_id="r",
            judge_model="m",
        )


def test_temperature_is_pinned_at_the_call_site():
    """Not left to the client's constructor default, so a run cannot silently sample its
    judge because a default changed two packages away."""
    llm = ScriptedLLM('{"label": "Supported"}')
    kpr.grade(
        llm,
        key_points=[("kp1", "x")],
        report=REPORT,
        suite_id="s",
        task_id="t",
        run_id="r",
        judge_model="m",
    )
    assert llm.calls[0]["temperature"] == JUDGE_TEMPERATURE == 0.0
    assert llm.calls[0]["actor"] == "judge"


def test_one_call_per_key_point():
    """Grading the list in one call makes the verdicts non-exchangeable and loses all of them
    to one malformed reply."""
    llm = ConstantLLM('{"label": "Supported"}')
    kpr.grade(
        llm,
        key_points=[("kp1", "a"), ("kp2", "b"), ("kp3", "c")],
        report=REPORT,
        suite_id="s",
        task_id="t",
        run_id="r",
        judge_model="m",
    )
    assert llm.n == 3


def test_recall_is_the_supported_share_and_nan_on_nothing_judged():
    llm = ScriptedLLM('{"label": "Supported"}', '{"label": "Omitted"}', '{"label": "Contradicted"}')
    js = kpr.grade(
        llm,
        key_points=[("kp1", "a"), ("kp2", "b"), ("kp3", "c")],
        report=REPORT,
        suite_id="s",
        task_id="t",
        run_id="r",
        judge_model="m",
    )
    assert kpr.recall(js) == pytest.approx(1 / 3)
    assert kpr.label_counts(js) == {"Supported": 1, "Omitted": 1, "Contradicted": 1}
    assert math.isnan(kpr.recall([]))


def test_the_prompt_is_pinned_and_carries_the_rubric():
    assert len(kpr.PROMPT_SHA) == 64
    for label in kpr.LABELS:
        assert label in kpr.TEMPLATE


# ------------------------------------------------------------------ citation support


def test_citation_labels_map_to_the_upstream_scores():
    assert [CITATION_SCORE[x] for x in citation.LABELS] == [1.0, 0.5, 0.0]


def test_claim_extraction_then_support_check():
    llm = ScriptedLLM(
        '{"claims": [{"claim_id": 1, "claim": "NO2 fell in Oslo.", '
        '"sources": ["https://example.org/oslo-car-free"]}]}',
        '{"support": "partial_support", "justification": "timing unclear"}',
    )
    js, diag = citation.grade(
        llm,
        report=REPORT,
        docs=DOCS,
        suite_id="drgym",
        task_id="900001",
        run_id="r1",
        judge_model="m",
    )
    assert len(js) == 1 and js[0].label == "partial_support" and js[0].score == 0.5
    assert js[0].criterion == "citation_support" and js[0].key_point_id == "claim1"
    assert diag == {"n_claims": 1, "n_checked": 1, "n_unresolved": 0, "unresolved_claim_ids": []}
    assert citation.mean_support(js) == 0.5


def test_a_claim_whose_sources_have_no_text_is_counted_not_scored_zero():
    """A divergence from upstream, named on purpose: upstream crawls at judging time and a
    failed crawl scores no_support, which conflates 'the source does not support this' with
    'we could not fetch the source'."""
    llm = ScriptedLLM(
        '{"claims": [{"claim_id": 7, "claim": "X.", "sources": ["https://unfetched.example/a"]}]}'
    )
    js, diag = citation.grade(
        llm,
        report=REPORT,
        docs=DOCS,
        suite_id="s",
        task_id="t",
        run_id="r",
        judge_model="m",
    )
    assert js == () and diag["n_unresolved"] == 1 and diag["unresolved_claim_ids"] == [7]
    assert math.isnan(citation.mean_support(js))


@pytest.mark.parametrize(
    "reply",
    [
        '{"claims": "not a list"}',
        '{"nope": []}',
        '{"claims": [{"claim_id": 1, "claim": "x"}]}',  # no sources
        '{"claims": [{"claim_id": 1, "claim": "x", "sources": []}]}',
        '{"claims": [{"claim_id": "1", "claim": "x", "sources": ["u"]}]}',
        '{"claims": [{"claim_id": 1, "claim": "", "sources": ["u"]}]}',
        '{"claims": [{"claim_id": 1, "claim": "x", "sources": [3]}]}',
    ],
)
def test_a_malformed_extraction_raises(reply):
    with pytest.raises(JudgeParseError):
        citation.parse_claims(reply)


@pytest.mark.parametrize("reply", ['{"support": "full"}', '{"label": "full_support"}', "{}"])
def test_a_malformed_support_verdict_raises(reply):
    with pytest.raises(JudgeParseError):
        citation.parse_support(reply)


def test_the_documents_offered_to_the_checker_are_the_retrieved_ones():
    from pinq.types import EvidenceUnit

    units = [
        EvidenceUnit.make(
            corpus_id="c", doc_id="d1", span="0:3", title="https://a.example/x", text="abc"
        ),
        EvidenceUnit.make(corpus_id="c", doc_id="d2", span="0:3", title="not a url", text="xyz"),
    ]
    assert citation.docs_from_units(units) == {"https://a.example/x": "abc"}


# ------------------------------------------------------------------ report quality


def test_there_are_exactly_six_criteria_with_the_published_names():
    assert [c.name for c in quality.CRITERIA] == [
        "Clarity",
        "Depth",
        "Balance",
        "Breadth",
        "Support",
        "Insightfulness",
    ]
    assert quality.CRITERION_KEYS == (
        "clarity",
        "depth",
        "balance",
        "breadth",
        "support",
        "insightfulness",
    )


def test_every_criterion_key_is_a_declared_judgment_criterion():
    from typing import get_args

    from pi_eval.judges.types import Criterion

    assert set(quality.CRITERION_KEYS) <= set(get_args(Criterion))


def test_a_rating_is_an_integer_in_range_and_anything_else_raises():
    assert quality.parse('{"rating": 0, "justification": "empty"}')[0] == 0
    assert quality.parse('{"rating": 10}')[0] == 10
    for bad in (
        '{"rating": 11}',
        '{"rating": -1}',
        '{"rating": "7"}',
        '{"rating": 7.5}',
        '{"rating": 7.0}',
        '{"rating": true}',
        '{"justification": "forgot"}',
    ):
        with pytest.raises(JudgeParseError):
            quality.parse(bad)


def test_grading_produces_one_judgment_per_criterion_with_raw_scores():
    llm = ConstantLLM('{"rating": 6, "justification": "ok"}')
    js = quality.grade(
        llm,
        question=QUESTION,
        report=REPORT,
        suite_id="drgym",
        task_id="900001",
        run_id="r1",
        judge_model="m",
    )
    assert len(js) == 6 and llm.n == 6
    assert {j.criterion for j in js} == set(quality.CRITERION_KEYS)
    assert all(j.score == 6.0 for j in js), "scores are raw 0-10, never rescaled"
    assert quality.by_criterion(js) == {k: 6.0 for k in quality.CRITERION_KEYS}


def test_a_quality_judgment_carries_no_label():
    """The criterion name in `label` would make every quality judgment agree with every
    other one and any nominal agreement statistic read 1.0."""
    llm = ConstantLLM('{"rating": 3}')
    js = quality.grade(
        llm,
        question=QUESTION,
        report=REPORT,
        suite_id="s",
        task_id="t",
        run_id="r",
        judge_model="m",
    )
    assert all(j.label is None for j in js)


def test_the_support_rubric_states_the_hard_zero_the_emitter_defends():
    support = next(c for c in quality.CRITERIA if c.key == "support")
    assert "the rating is 0" in support.description


# ------------------------------------------------------------------ paired wiring


def _abs(score, task="t", criterion="keypoint", run="A"):
    return Judgment(
        judgment_id="x",
        run_id_a=run,
        run_id_b=None,
        suite_id="drgym",
        task_id=task,
        criterion=criterion,
        order="ab",
        judge_family="fam",
        judge_model="m",
        judge_prompt_sha="p",
        score=score,
    )


META = dict(
    criterion="keypoint",
    suite_id="drgym",
    task_id="900001",
    run_id_a="A",
    run_id_b="B",
    judge_family="fam",
    judge_model="m",
    judge_prompt_sha="p",
)


def test_absolute_gradings_fold_into_a_signed_paired_judgment():
    j = paired_from_absolute(a=[_abs(1.0), _abs(1.0)], b=[_abs(0.0), _abs(1.0)], order="ab", **META)
    assert j.pref_sign == 1 and j.magnitude == pytest.approx(0.5)
    assert j.run_id_b == "B"


def test_the_dead_band_keeps_judge_noise_from_manufacturing_wins():
    j = paired_from_absolute(
        a=[_abs(6.2)], b=[_abs(6.0)], order="ab", tol=0.5, **{**META, "criterion": "clarity"}
    )
    assert j.pref_sign == 0


def test_an_order_invariant_grader_survives_both_orders_and_shows_no_position_bias():
    """Grading each report in isolation SHOULD be order-invariant, so 0.5 is the honest
    expectation. Running both orders is what measures that instead of assuming it."""
    js = []
    for i in range(8):
        a_better = i % 2 == 0
        ab, ba = both_orders(
            lambda order, a_better=a_better: (
                [_abs(1.0 if a_better else 0.0)],
                [_abs(0.0 if a_better else 1.0)],
            ),
            **{**META, "task_id": f"t{i}"},
        )
        assert resolve_pair(ab, ba).order_consistent
        js += [ab, ba]
    assert position_bias(js) == pytest.approx(0.5)
    assert judge_is_usable(js)[0]


def test_a_grader_that_leaks_order_is_caught_and_disqualified():
    """A judge handed both reports in one call, or a reused conversation, makes the verdict
    depend on presentation. That is a real bug and this is where it surfaces."""
    js = []
    for i in range(8):
        ab, ba = both_orders(
            lambda order: (
                [_abs(1.0 if order == "ab" else 0.0)],
                [_abs(0.0 if order == "ab" else 1.0)],
            ),
            **{**META, "task_id": f"t{i}"},
        )
        outcome = resolve_pair(ab, ba)
        assert outcome.sign == 0 and not outcome.order_consistent
        js += [ab, ba]
    assert position_bias(js) == pytest.approx(1.0)
    ok, why = judge_is_usable(js)
    assert not ok and "position bias" in why


def test_sigma_j_applies_to_the_new_judges_through_retest_groups():
    llm = ConstantLLM('{"rating": 6}')
    a = quality.grade(
        llm,
        question=QUESTION,
        report=REPORT,
        suite_id="drgym",
        task_id="900001",
        run_id="r1",
        judge_model="m",
        retest_group_id="g1",
    )
    llm2 = ConstantLLM('{"rating": 8}')
    b = quality.grade(
        llm2,
        question=QUESTION,
        report=REPORT,
        suite_id="drgym",
        task_id="900001",
        run_id="r1",
        judge_model="m",
        retest_group_id="g1",
    )
    # A retest group is ONE item judged twice, so the group id is per criterion: pooling six
    # different criteria into one group would measure the spread between criteria and call it
    # judge noise.
    paired = [
        paired_from_absolute(
            a=[x],
            b=[_abs(0.0)],
            order="ab",
            retest_group_id=f"g1-{x.criterion}",
            **{**META, "criterion": x.criterion},
        )
        for x in list(a) + list(b)
    ]
    assert sigma_j(paired) == pytest.approx(math.sqrt(2))


def test_key_point_labels_feed_krippendorff_alpha_per_unit():
    ours = kpr.grade(
        ScriptedLLM('{"label": "Supported"}', '{"label": "Omitted"}'),
        key_points=[("kp1", "a"), ("kp2", "b")],
        report=REPORT,
        suite_id="drgym",
        task_id="900001",
        run_id="r1",
        judge_model="m",
        judge_family="famA",
    )
    theirs = kpr.grade(
        ScriptedLLM('{"label": "Supported"}', '{"label": "Omitted"}'),
        key_points=[("kp1", "a"), ("kp2", "b")],
        report=REPORT,
        suite_id="drgym",
        task_id="900001",
        run_id="r1",
        judge_model="m2",
        judge_family="famB",
    )
    ratings = ratings_by_family(list(ours) + list(theirs))
    assert set(ratings) == {"drgym/900001/keypoint/kp1", "drgym/900001/keypoint/kp2"}
    assert krippendorff_alpha_nominal(ratings) == pytest.approx(1.0)


def test_mean_score_is_nan_on_an_empty_set():
    assert math.isnan(mean_score([]))


# ------------------------------------------------------------------ fidelity


@pytest.fixture
def reference(request):
    return fidelity.load_kpr_reference(
        request.config.rootpath
        / "tests"
        / "fixtures"
        / "drgym"
        / "reference"
        / "evaluation_results_kpr_fixture.json",
        system="fixture-system",
        judge_model="gpt-4.1-mini",
        source_url="https://example.invalid/results",
        commit="d4d2433309d9da9be637c365618e7e01f8f9205a",
    )


def test_a_reference_without_provenance_is_refused(request):
    with pytest.raises(fidelity.MissingProvenance):
        fidelity.load_kpr_reference(
            request.config.rootpath
            / "tests"
            / "fixtures"
            / "drgym"
            / "reference"
            / "evaluation_results_kpr_fixture.json",
            system="fixture-system",
            judge_model="",
            source_url="",
            commit="",
        )


def test_the_reference_loader_skips_null_records_and_hashes_the_file(reference):
    assert len(reference.sha256) == 64
    assert ("900099", "1") not in reference.labels
    assert reference.labels[("900001", "2")] == "Omitted"


def _ours(pairs, model="ours-model"):
    return [
        Judgment(
            judgment_id=f"j{i}",
            run_id_a="r1",
            run_id_b=None,
            suite_id="drgym",
            task_id=task,
            criterion="keypoint",
            order="ab",
            judge_family="fam",
            judge_model=model,
            judge_prompt_sha="p",
            label=label,
            score=kpr.KEYPOINT_SCORE[label],
            key_point_id=kp,
        )
        for i, (task, kp, label) in enumerate(pairs)
    ]


def test_the_delta_is_computed_over_shared_units_only(reference):
    """Comparing our mean over our subset against their mean over theirs would confound the
    instrument with the sample, which is the exact error this table exists to rule out."""
    ours = _ours(
        [
            ("900001", "kp1", "Supported"),
            ("900001", "kp2", "Supported"),  # they said Omitted
            ("900001", "kp3", "Supported"),
            ("900003", "kp1", "Supported"),
            ("900003", "kp2", "Contradicted"),
            ("900007", "kp1", "Supported"),  # not in the reference at all
        ]
    )
    d = fidelity.kpr_delta(ours, reference)
    assert d.n_units == 5 and d.n_ours_only == 1 and d.n_ref_only == 0
    assert d.ours_recall == pytest.approx(4 / 5)
    assert d.ref_recall == pytest.approx(3 / 5)
    assert d.delta_recall == pytest.approx(0.2)
    assert d.agreement == pytest.approx(4 / 5)
    assert d.confusion[("Supported", "Omitted")] == 1
    table = fidelity.render_kpr_delta(d)
    assert "d4d243330" in table and d.ref_sha256[:12] in table and "+0.200" in table


def test_a_delta_over_no_shared_unit_is_refused(reference):
    with pytest.raises(fidelity.NoOverlap):
        fidelity.kpr_delta(_ours([("999999", "kp1", "Supported")]), reference)


def test_node_ids_and_upstream_point_numbers_join_by_prefix_strip():
    assert fidelity.normalize_key_point_id("kp12") == "12"
    assert fidelity.normalize_key_point_id("12") == "12"


def test_an_aggregate_reference_needs_provenance_and_a_shared_criterion(request):
    base = request.config.rootpath / "tests" / "fixtures" / "drgym" / "reference"
    ref = fidelity.load_aggregate_reference(base / "quality_aggregate_fixture.json")
    rows, missing = fidelity.aggregate_delta({"clarity": 7.0, "depth": 5.0}, ref)
    assert [r["criterion"] for r in rows] == ["clarity", "depth"]
    assert rows[0]["delta"] == pytest.approx(1.0)
    assert missing == ["support"], "a criterion we could not compare is never silently dropped"
    assert "fixture-judge" in fidelity.render_aggregate_delta(rows, ref)

    with pytest.raises(fidelity.MissingProvenance):
        fidelity.load_aggregate_reference(base / "quality_aggregate_no_provenance.json")
    with pytest.raises(fidelity.NoOverlap):
        fidelity.aggregate_delta({"breadth": 1.0}, ref)
