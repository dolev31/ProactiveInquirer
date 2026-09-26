"""The N1 checkpoint-selection rule (scripts/select_checkpoint.py), promoted from a session
scratchpad because it decides which checkpoint the paper reports and must be reviewable and
reproducible from a clean checkout, not something that lives only in a scratch dir that can
vanish.

Two kinds of case here. First, the tool's own --self-test, run as a real pytest case rather
than something a reviewer has to remember to invoke by hand. Second, one test per REFUSAL the
tool implements, each against small static verdict fixtures under
tests/fixtures/select_checkpoint/ (built with the tool's own `_synth_verdict` helper, then
mutated in exactly the one way that must be refused) rather than the self-test's in-memory
ones, so each refusal is demonstrated against a file a reviewer can open and diff against a
working fixture.

The failure this file exists to catch is not a crash: it is the rule silently reporting a
number it should have refused to report -- a pre-f348feb stop_2x2 whose done axis is
identically 0, a matched-cost delta compared across two checkpoints scored by different
scorers, or a coverage_rule shape nobody wrote a reader for -- because each of those still
produces a plausible-looking PASS/FAIL with no symptom at all.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from scripts.select_checkpoint import Refused, evaluate, main, set_repo

REPO = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "select_checkpoint"

# The label the tool itself prints for criterion (a'); see select_checkpoint.py:self_test,
# which builds it the same way ("(′".replace("(", "(a")) rather than hard-coding it twice.
A_PRIME = "(a′"


def _gate(tmp_path: Path, case: str) -> Path:
    """Copy one fixture pair into its own gate dir under tmp_path.

    evaluate() reads real files off disk (that is the point: a reviewer can open the same
    fixture and see the same refusal), so each case gets an isolated directory rather than
    sharing tests/fixtures/select_checkpoint itself.
    """
    gate = tmp_path / "gate"
    gate.mkdir()
    for src in (FIXTURES / case).glob("*.json"):
        shutil.copy(src, gate / src.name)
    return gate


def test_self_test_passes():
    """The tool's own self-test: 7 synthetic candidates x 2 suites against 6 invariants, plus
    the pre-f348feb refusal exercised a second way (in-memory, not from a fixture file). See
    select_checkpoint.py:self_test for what each of those checks; this only asserts the exit
    code so a broken invariant fails CI instead of only a manual `--self-test` run.
    """
    assert main(["--repo", str(REPO), "--self-test"]) == 0


def test_refuses_pre_f348feb_stop_2x2(tmp_path):
    """A stop_2x2 block written before commit f348feb reads the done axis off
    `scores.metric_name='stop_undershoot'`, which is identically 0 on every checkpoint: every
    such block has n_not_done=0 and p_ask_given_not_done=None. docs/TRAINING.md 5.3.1 forbids
    quoting it, and _check_live_2x2 must refuse it -- here, by the "missing per-state cells"
    signal, since this fixture (like a real pre-fix record) never had per-decision-point cells
    at all, not just a degenerate done axis.
    """
    gate = _gate(tmp_path, "pre_f348feb")
    set_repo(REPO)
    with pytest.raises(Refused, match="per-state cells") as exc:
        evaluate(
            [gate],
            candidates=["cand"],
            reference="ref",
            base_model="qwen3-8b-base",
            suites=["musique"],
            p13=Path("unused"),
            gold_root=Path("unused"),
        )
    assert "f348feb" in str(exc.value)


def test_refuses_scorer_hash_mismatch(tmp_path):
    """ "A scorer mismatch is a refusal for that criterion, not a comparison" (module
    docstring). The candidate and reference fixtures here clear every RAW threshold on (a')
    and (b) -- the rise and the stop floor are computed and passed on their own -- but their
    selection.scorer_hash differ, so scorer_match is False and both criteria must be forced to
    fail rather than compared.

    Criterion (c) is deliberately NOT forced to fail here, and that is asserted too: its own
    scorer check (_same_scorer) matches candidate against reference by the matched-cost
    contrast's source/sha256/how, and `how` is the identical literal string
    "verdict `coverage_rule` block" for both sides whenever both hit the in-verdict
    coverage_rule path, regardless of selection.scorer_hash. So (c) passes here on its own
    merits, unaffected by the very mismatch that must sink (a') and (b) -- this is what makes
    the guard specific to the criteria the module docstring says it protects, rather than a
    blanket "reject the whole row" check that would pass this test for the wrong reason.
    """
    gate = _gate(tmp_path, "scorer_mismatch")
    set_repo(REPO)
    res = evaluate(
        [gate],
        candidates=["cand"],
        reference="ref",
        base_model="qwen3-8b-base",
        suites=["musique"],
        p13=Path("unused"),
        gold_root=Path("unused"),
    )
    row = res["rows"][0]
    assert row["candidate"] == "cand"
    assert row["suite"] == "musique"

    # The raw numbers alone clear both thresholds on (a') ...
    assert row["a"]["rise"] >= row["a"]["rise_min"]
    assert row["a"]["stop_passed"] is True
    # ... and (b) rises too ...
    assert row["b"]["rise"] > 0
    # ... but the scorer guard still forces both to fail, because scorer_match is False.
    assert row["a"]["scorer_match"] is False
    assert row["a"]["rise_passed"] is False
    assert row["a"]["passed"] is False
    assert row["b"]["scorer_match"] is False
    assert row["b"]["passed"] is False

    # (c) is untouched by this refusal.
    assert row["c"]["scorer_match"] is True
    assert row["c"]["passed"] is True

    got = {c["candidate"]: c for c in res["candidates"]}
    assert got["cand"]["verdict"] == "FAIL"
    assert f"{A_PRIME}) musique" in got["cand"]["failed"]
    assert "(b) musique" in got["cand"]["failed"]


def test_refuses_unknown_coverage_rule_shape(tmp_path):
    """A `coverage_rule` block that is a Mapping but matches none of the four shapes
    _inverdict_matched knows how to read (a contrasts list, a [comparator][metric] block, a
    flat {delta/value} block, or a by_suite block) must be refused outright rather than
    silently skipped or read as zero -- a matched-cost number pulled from the wrong key would
    be a wrong published number nothing downstream could catch, since it would still look like
    a plausible float.
    """
    gate = _gate(tmp_path, "bad_coverage_rule")
    set_repo(REPO)
    with pytest.raises(Refused, match="any accepted shape"):
        evaluate(
            [gate],
            candidates=["cand"],
            reference="ref",
            base_model="qwen3-8b-base",
            suites=["musique"],
            p13=Path("unused"),
            gold_root=Path("unused"),
        )


def test_refuses_coverage_value_predating_matched_cost_rule(tmp_path):
    """A verdict written before the matched-cost rule existed has `coverage_rule: null` and no
    top-level `matched_cost` key at all -- confirmed against every real verdict of that shape
    under artifacts/gate/ (34 of them; none carry a top-level `matched_cost` key or a
    `cap8_value` sibling either). In that shape, `criteria.evidence_coverage.value` is the
    fixed-budget (cap8) delta, not a matched-cost reading -- the same field that, in a verdict
    written under the rule, holds the matched-cost delta instead, with the fixed-budget figure
    moved to a sibling `cap8_value` field. One pair of real numbers, two field names, depending
    entirely on `coverage_rule`, which this fixture's cand.musique.json sets to `null` while
    still carrying a realistic fixed-budget delta (-0.05, the same value the sibling
    bad_coverage_rule fixture uses) at `criteria.evidence_coverage.value`.

    _inverdict_matched never reads that field under any of its four shapes (see its docstring
    for the full audit), so a null `coverage_rule` returns None from it rather than a match; ref
    is untouched here (still a valid coverage_rule.contrasts block) so this isolates the
    refusal to cand's shape alone, the same pattern bad_coverage_rule uses. With no top-level
    `matched_cost` key, no `gate/matched_cost.json` sidecar in this fixture, and the fake
    `p13`/`gold_root` every refusal test here uses, matched_cost() has no source left and
    _run_matched_cost refuses on the fake --gold-root -- proving the refusal is real and not
    merely assumed, per the current, unmodified code (no fix was needed: this test was
    confirmed to fail -- i.e. accept the fixed-budget number as a matched-cost delta, landing
    it in criterion (c)'s `delta` -- only when a hypothetical fallback reading
    `criteria.evidence_coverage.value` as a last resort was deliberately injected for that one
    check; that fallback is not, and must never be, part of this module).
    """
    gate = _gate(tmp_path, "pre_matched_cost_rule")
    set_repo(REPO)
    with pytest.raises(Refused, match="gold-root"):
        evaluate(
            [gate],
            candidates=["cand"],
            reference="ref",
            base_model="qwen3-8b-base",
            suites=["musique"],
            p13=Path("unused"),
            gold_root=Path("unused"),
        )


def test_refuses_cap8_report_disagreeing_with_recomputed_delta(tmp_path):
    """`d_reported.cap8_coverage_delta` is report-only (`"gated": False`, never a `passed`
    decision -- criterion (d) in the module docstring) but it IS formatted straight into the
    printed report table (render(), the cap8_coverage_delta column), so a wrong number there
    still reaches a human, who may quote it. This fixture's cand.musique.json is a post-rule
    (new-shape) verdict -- `coverage_rule: "matched_cost"` plus a real top-level `matched_cost`
    block, resolvable by _inverdict_matched shape 1 with no gold_root/p13 fallback ever reached
    -- whose two claims about the fixed-budget (cap8) coverage delta deliberately disagree:
    `criteria.evidence_coverage.cap8_value` is -0.05, but the independently-resolvable
    `matched_cost.contrasts` entry for comparator="cap8" has `delta: 0.05`. Both numbers are
    real floats sourced from the same file by two different routes (one a value some earlier
    process wrote into the criteria block, the other a contrast row `matched_cost()` reads
    completely separately); nothing but this refusal would ever notice they are not the same
    quantity.

    Before _reported_cap8_delta existed, this fixture did NOT raise: `evaluate()` read
    `criteria.evidence_coverage.value` unconditionally (0.06, the MATCHED-COST delta, wearing
    the fixed-budget name for this post-rule verdict, the exact mislabelling
    _inverdict_matched's docstring flagged but did not fix) and never compared it against
    `cap8_coverage_delta_recomputed` (`cap8["delta"]`, 0.05) sitting right beside it in the same
    `d_reported` block -- so the wrong number (0.06) would have reached row["d_reported"]
    ["cap8_coverage_delta"]["value"] and, from there, the printed table, silently. That
    was confirmed by running this exact test against the code before this fix: it did not
    raise, the call returned normally, and the returned row carried 0.06 under the
    cap8_coverage_delta label -- none of -0.05, 0.05, or their disagreement anywhere in sight.

    `_reported_cap8_delta` now reads the correct sibling for this shape (`cap8_value`, not
    `value`) and then refuses because that correctly-selected number (-0.05) still disagrees
    with the recomputed one (0.05) by far more than TOL -- refusing rather than warning, the
    same choice `_agree_outside_stop` makes for a re-run 2x2 that disagrees with the verdict of
    record, because a warning is a column a busy reader can skip past and a refusal is not.
    """
    gate = _gate(tmp_path, "cap8_report_disagreement")
    set_repo(REPO)
    with pytest.raises(Refused, match="disagrees with"):
        evaluate(
            [gate],
            candidates=["cand"],
            reference="ref",
            base_model="qwen3-8b-base",
            suites=["musique"],
            p13=Path("unused"),
            gold_root=Path("unused"),
        )
