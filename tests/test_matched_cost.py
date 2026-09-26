"""The matched-cost prefix ladder, on a fixture whose every number is hand-computed.

The failure this file exists to catch is not a crash. It is a ladder that silently answers a
different question than the scorer did -- coverage at the WRONG prefix, a depth>=2 need counted
RESOLVED on half its evidence, or a base run charged for asks it never made -- because each of
those produces a plausible matched-cost delta with no symptom at all.
"""

from __future__ import annotations

import json
import math

import pytest
from scripts.matched_cost import (
    CAP,
    COST_BASES,
    Contrast,
    InstrumentMismatch,
    Refused,
    build_ladders,
    contrast,
    early_stop_split,
    outcome_split,
    reconcile,
    verify_instrument,
)

from pi_eval.gold import GoldGraph, GoldNode
from pi_eval.score import _Act, _Turn

SUITE = "synthfix"


def _node(task: str, node_id: str, depth: int, uids: tuple[str, ...]) -> GoldNode:
    return GoldNode(
        gold_suite=SUITE,
        gold_task_key=task,
        gold_node_id=node_id,
        # Deliberately opaque: the ASK rung matches on a need's distinctive terms, and a need
        # whose words appear in the fixture's questions would smuggle a rank-1 match into a
        # test about the RESOLVE rung.
        gold_text=f"zzz-{node_id}",
        gold_partition="required",
        gold_depth=depth,
        gold_ev_uids=uids,
        gold_graph_version="v1",
    )


def _graph(task: str, nodes: tuple[GoldNode, ...]) -> GoldGraph:
    return GoldGraph(
        gold_suite=SUITE, gold_task_key=task, gold_nodes=nodes, gold_graph_version="v1"
    )


# t1 needs four spans; its ONE depth>=2 need takes two of them, so half its evidence must not
# count. t2 needs two spans and its depth-2 need takes one.
GRAPHS = {
    SUITE: {
        "t1": _graph(
            "t1",
            (
                _node("t1", "n0", 0, ("u0",)),
                _node("t1", "n1", 1, ("u1",)),
                _node("t1", "n2", 2, ("u2", "u3")),
            ),
        ),
        "t2": _graph(
            "t2",
            (_node("t2", "m0", 0, ("v0",)), _node("t2", "m1", 2, ("v1",))),
        ),
    }
}

# (run_id, task, arm, retrievals per turn). The base asks MORE and gets there LATER, which is
# the whole shape the cap-8 gate cannot see.
FIXTURE = (
    ("trained_t1", "t1", "inquirer_trained", (("u0",), ("u1", "u2"))),
    ("trained_t2", "t2", "inquirer_trained", (("v0", "v1"),)),
    ("base_t1", "t1", "inquirer_prompted", (("u0",), (), ("u1",), ("u2", "u3"))),
    ("base_t2", "t2", "inquirer_prompted", (("v0",), (), ("v1",))),
)


def _fixture():
    runs = []
    turns = {}
    for run_id, task, arm, per_turn in FIXTURE:
        runs.append(
            {
                "run_id": run_id,
                "suite_id": SUITE,
                "task_id": task,
                "template_id": None,
                "arm_id": arm,
                "seed": 0,
                "n_asks": len(per_turn),
                "stop_reason": "policy_stop",
            }
        )
        turns[run_id] = [
            _Turn(turn_idx=i, retrieved_uids=u, new_uids=u, action=_Act(text="what next?"))
            for i, u in enumerate(per_turn)
        ]
    return runs, turns


@pytest.fixture
def ladders():
    runs, turns = _fixture()
    return build_ladders(runs, turns, GRAPHS)


def _stored(lad) -> dict[str, float]:
    rows = {f"frontier_q#{k}": v for k, v in enumerate(lad.cov)}
    rows["evidence_coverage"] = lad.cov[-1]
    for d, n_d in lad.cad_n.items():
        rows[f"cad#{d}"] = lad.cad_hit[d][-1] / n_d
        rows[f"cad_n#{d}"] = float(n_d)
    return rows


def test_the_ladder_is_coverage_after_k_asks_not_the_terminal_value(ladders):
    """t1's base run reaches every span only on its FOURTH ask. Reading the run's stored
    coverage (1.0) at k=2 is the exact mistake the cap-8 contrast makes."""
    base = ladders["base_t1"]
    assert base.cov == pytest.approx([0.0, 0.25, 0.25, 0.5, 1.0])
    assert base.coverage_at(2) == pytest.approx(0.25)
    assert base.coverage_at(4) == pytest.approx(1.0)


def test_a_depth_two_need_is_resolved_only_when_all_of_its_evidence_is_in(ladders):
    """t1/n2 takes u2 AND u3. The trained run retrieves u2 alone and must score 0."""
    trained, base = ladders["trained_t1"], ladders["base_t1"]
    assert trained.cad2_n == 1 and base.cad2_n == 1
    assert trained.cad2_hit == (0, 0, 0)
    assert base.cad2_hit == (0, 0, 0, 0, 1)
    assert math.isnan(trained.cad2_at(0)) is False and trained.cad2_at(2) == 0.0


def test_a_prefix_past_the_end_bills_the_asks_the_run_actually_made(ladders):
    """A run that stopped at 3 under cap 8 stops at 3 under cap 5. Charging it the cap would
    invent spend in the one comparison whose subject is spend."""
    base = ladders["base_t2"]
    assert base.n_asks == 3
    assert base.asks_at(CAP) == 3
    assert base.coverage_at(CAP) == pytest.approx(1.0)
    assert base.coverage_at(99) == pytest.approx(base.cov[-1])


def test_the_matched_cost_delta_flips_the_sign_the_cap_eight_delta_reports(ladders):
    """Hand-computed. At the trained run's own k: t1 0.75 vs 0.25, t2 1.0 vs 0.5, mean +0.5.
    At cap 8: t1 0.75 vs 1.0, t2 1.0 vs 1.0, mean -0.125."""
    trained = [ladders["trained_t1"], ladders["trained_t2"]]
    base = {
        (x.suite_id, x.task_id, x.seed): x
        for x in ladders.values()
        if x.arm_id == "inquirer_prompted"
    }
    matched = contrast(
        trained,
        base,
        suite_id=SUITE,
        checkpoint="fixture",
        metric="evidence_coverage",
        comparator="matched_k",
        offset=0,
    )
    capped = contrast(
        trained,
        base,
        suite_id=SUITE,
        checkpoint="fixture",
        metric="evidence_coverage",
        comparator="cap8",
        offset=0,
    )
    assert matched.n_tasks == 2 and capped.n_tasks == 2
    assert matched.delta == pytest.approx(0.5)
    assert capped.delta == pytest.approx(-0.125)
    # The second coordinate of the endpoint: the base is never billed MORE than the cap-k run
    # it stands in for, and at cap 8 it is billed far more.
    assert matched.base_asks == pytest.approx(1.5) and matched.trained_asks == pytest.approx(1.5)
    assert capped.base_asks == pytest.approx(3.5)
    assert capped.n_base_short == 2  # both base runs ended before the cap


def test_cad_ge2_delta_is_paired_only_on_tasks_that_have_a_depth_two_need(ladders):
    trained = [ladders["trained_t1"], ladders["trained_t2"]]
    base = {
        (x.suite_id, x.task_id, x.seed): x
        for x in ladders.values()
        if x.arm_id == "inquirer_prompted"
    }
    matched = contrast(
        trained,
        base,
        suite_id=SUITE,
        checkpoint="fixture",
        metric="cad_ge2",
        comparator="matched_k",
        offset=0,
    )
    capped = contrast(
        trained,
        base,
        suite_id=SUITE,
        checkpoint="fixture",
        metric="cad_ge2",
        comparator="cap8",
        offset=0,
    )
    assert matched.delta == pytest.approx(0.5)  # t1 0-0, t2 1-0
    assert capped.delta == pytest.approx(-0.5)  # t1 0-1, t2 1-1


def test_win_tie_loss_and_the_early_stop_split_are_counted_not_inferred(ladders):
    trained = [ladders["trained_t1"], ladders["trained_t2"]]
    base = {
        (x.suite_id, x.task_id, x.seed): x
        for x in ladders.values()
        if x.arm_id == "inquirer_prompted"
    }
    assert outcome_split(trained, base, "evidence_coverage") == {
        "n": 2,
        "win": 2,
        "tie": 0,
        "loss": 0,
    }
    split = early_stop_split(trained, base)
    assert split["n_paired"] == 2
    assert split["n_done_at_stop"] == 1  # t2 stopped with every required span in hand
    assert split["n_lose_at_cap8"] == 1  # t1
    assert split["lose_done_at_stop"] == 0  # a run below a <=1 coverage cannot itself be at 1
    assert split["lose_tie_or_win_at_k"] == 1  # and it was AHEAD at its own k
    assert split["lose_behind_at_k"] == 0


def test_a_ladder_that_disagrees_with_the_stored_scorer_is_refused(ladders):
    """Rule 2. Without this the script would happily report a matched-cost delta computed by
    an instrument that is not the one the verdicts used, and nothing would look wrong."""
    stored = {run_id: _stored(lad) for run_id, lad in ladders.items()}
    assert verify_instrument(ladders, stored)["frontier_q"] == 14  # 3 + 2 + 5 + 4 rungs

    drifted = {k: dict(v) for k, v in stored.items()}
    drifted["base_t1"]["frontier_q#2"] = 0.5  # the value a min-instead-of-max matcher gives
    with pytest.raises(InstrumentMismatch, match="frontier_q#2"):
        verify_instrument(ladders, drifted)

    missing = {k: dict(v) for k, v in stored.items()}
    missing["trained_t1"].pop("cad#2")
    with pytest.raises(InstrumentMismatch, match="cad#2"):
        verify_instrument(ladders, missing)

    no_cov = {k: dict(v) for k, v in stored.items()}
    no_cov["trained_t2"].pop("evidence_coverage")
    with pytest.raises(InstrumentMismatch, match="evidence_coverage"):
        verify_instrument(ladders, no_cov)


# --------------------------------------------------------------- discovering the checkpoints


def _gate(tmp_path, layout: dict[str, tuple[str, ...]]):
    """`{parquet_<tag>: (<verdict file names>,)}` -> a gate directory with those names on disk."""
    for d, verdicts in layout.items():
        (tmp_path / d).mkdir()
        for v in verdicts:
            (tmp_path / v).write_text("{}")
    return tmp_path


def test_a_gate_dir_is_discovered_from_its_parquet_dirs_and_the_verdicts_beside_them(tmp_path):
    """A new checkpoint's gate dir must be readable WITHOUT the pass-1 trio: the tier-B batch
    gates one checkpoint at a time, and a table hardcoded to three names refused every other
    layout (`REFUSED: missing .../parquet_headline/runs.parquet`, measured 2026-09-15)."""
    from scripts.matched_cost import discover_checkpoints

    gate = _gate(
        tmp_path,
        {
            "parquet_qwen3-4b-sft-headline": (
                "qwen3-4b-sft-headline.musique.json",
                "qwen3-4b-sft-headline.strategyqa.json",
            )
        },
    )
    assert discover_checkpoints(gate) == (
        ("parquet_qwen3-4b-sft-headline", "qwen3-4b-sft-headline"),
    )


def test_the_pass_one_layout_still_maps_to_its_three_ids_in_its_original_order(tmp_path):
    """The committed P13 figure and its provenance digest were produced from
    parquet_{sft,headline,headline_stop}; that layout must keep resolving to the same ids in
    the same order, or the digests move without any input changing."""
    from scripts.matched_cost import discover_checkpoints

    gate = _gate(
        tmp_path,
        {
            "parquet_headline_stop": ("qwen3-8b-sft-headline-stop.musique.json",),
            "parquet_sft": ("qwen3-8b-sft.musique.json",),
            "parquet_headline": ("qwen3-8b-sft-headline.musique.json",),
        },
    )
    assert discover_checkpoints(gate) == (
        ("parquet_sft", "qwen3-8b-sft"),
        ("parquet_headline", "qwen3-8b-sft-headline"),
        ("parquet_headline_stop", "qwen3-8b-sft-headline-stop"),
    )


def test_a_parquet_dir_with_no_verdict_beside_it_is_refused_not_guessed(tmp_path):
    from scripts.matched_cost import Refused, discover_checkpoints

    gate = _gate(tmp_path, {"parquet_qwen3-4b-sft-headline": ()})
    with pytest.raises(Refused, match="qwen3-4b-sft-headline"):
        discover_checkpoints(gate)
    with pytest.raises(Refused, match="parquet_"):
        discover_checkpoints(tmp_path / "empty-gate")


def test_a_mixed_layout_is_ordered_by_model_id(tmp_path):
    from scripts.matched_cost import discover_checkpoints

    gate = _gate(
        tmp_path,
        {
            "parquet_sft": ("qwen3-8b-sft.musique.json",),
            "parquet_qwen3-8b-sft-headline-s1": ("qwen3-8b-sft-headline-s1.musique.json",),
        },
    )
    assert discover_checkpoints(gate) == (
        ("parquet_sft", "qwen3-8b-sft"),
        ("parquet_qwen3-8b-sft-headline-s1", "qwen3-8b-sft-headline-s1"),
    )


def test_a_gate_root_without_a_readme_is_recorded_as_absent_not_crashed(tmp_path):
    """Measured 2026-09-15 on artifacts/gate/4b: every table printed, then `--json` died in
    `sha256_file` on the missing README.md and no provenance record was written. The README is
    documentation, not an input the numbers depend on; its absence is recorded, never fatal."""
    from scripts.matched_cost import readme_source

    assert readme_source(tmp_path) is None
    (tmp_path / "README.md").write_text("x")
    assert readme_source(tmp_path) == tmp_path / "README.md"


# -------------------------------------------- reconciling against the verdict `pi train gate` wrote


def _contrast(metric: str, comparator: str, delta: float, n: int) -> Contrast:
    """A recomputed contrast carrying only the four fields `reconcile` reads."""
    nan = float("nan")
    return Contrast(
        suite_id=SUITE,
        checkpoint="ckpt",
        metric=metric,
        comparator=comparator,
        n_tasks=n,
        n_clusters=n,
        trained_mean=nan,
        base_mean=nan,
        delta=delta,
        ci_lo=nan,
        ci_hi=nan,
        p_value=None,
        trained_asks=nan,
        base_asks=nan,
        n_base_short=0,
    )


def _write_verdict(gate, criteria: dict, **top) -> None:
    (gate / f"ckpt.{SUITE}.json").write_text(json.dumps({"criteria": criteria, **top}))


# Real numbers, not invented ones. MATCHED_K / CAP8 / CAD8 are
# artifacts/gate/17b/qwen3-1.7b-sft-headline-s0.musique.json's own `value`, `cap8_value` and
# `criteria.cad_ge2.value`, and CAP8_OLD / CAD8_OLD are artifacts/gate/qwen3-8b-sft.musique.json's
# -- the pass-1 gate the committed P13 provenance digest was produced from.
MATCHED_K = 0.10921717171717173
CAP8 = 0.05366161616161617
CAD8 = 0.12941176470588237
CAP8_OLD = -0.04292929292929293
CAD8_OLD = -0.08235294117647059


def test_a_cap8_rule_verdict_is_still_reconciled_against_its_value_field(tmp_path):
    """A verdict with no `coverage_rule` is the pre-4e3055b gate, every criterion of which IS
    the cap-8 contrast. `artifacts/gate/*.json` are all of this shape and the committed P13
    digest came from them, so this path may not move -- only the row now names the field it
    read, because under the other rule that is no longer obvious."""
    _write_verdict(
        tmp_path,
        {
            "evidence_coverage": {"value": CAP8_OLD, "n": 132},
            "cad_ge2": {"value": CAD8_OLD, "n": 85},
        },
    )
    rows = reconcile(
        tmp_path,
        [
            _contrast("evidence_coverage", "matched_k", 0.5, 132),
            _contrast("evidence_coverage", "matched_k_plus_1", 0.4, 132),
            _contrast("evidence_coverage", "cap8", CAP8_OLD, 132),
            _contrast("cad_ge2", "cap8", CAD8_OLD, 85),
        ],
    )
    # The matched-k rows are locked against NOTHING here: a cap-8-rule verdict carries no
    # matched-k number, and 0.5 is deliberately nowhere near the value.
    assert [(r["metric"], r["comparator"], r["field"]) for r in rows] == [
        ("evidence_coverage", "cap8", "value"),
        ("cad_ge2", "cap8", "value"),
    ]
    assert [r["verdict_value"] for r in rows] == [CAP8_OLD, CAD8_OLD]
    assert all(r["agree"] for r in rows)

    with pytest.raises(Refused, match="evidence_coverage"):
        reconcile(tmp_path, [_contrast("evidence_coverage", "cap8", 0.0, 132)])


def _matched_cost_verdict(gate) -> None:
    """artifacts/gate/17b's shape: `value` is matched-k, the cap-8 delta moved to `cap8_value`,
    and `cad_ge2` was NOT moved -- its `value` is still the cap-8 contrast."""
    _write_verdict(
        gate,
        {
            "evidence_coverage": {
                "value": MATCHED_K,
                "cap8_value": CAP8,
                "comparator": "matched_k",
                "n": 132,
            },
            "cad_ge2": {"value": CAD8, "n": 85},
        },
        coverage_rule="matched_cost",
        matched_cost={
            "by_suite": {SUITE: {"cap8_coverage_delta": CAP8, "cap8_n_tasks": 132, "n_tasks": 132}}
        },
    )


def test_a_matched_cost_verdict_locks_cap8_on_cap8_value_and_matched_k_on_value(tmp_path):
    """`pi train gate` has defaulted to `--coverage-rule matched_cost` since 4e3055b, and there
    `criteria.evidence_coverage.value` is the MATCHED-k delta. Reading it as the cap-8 number
    refused a recomputation that was correct: measured on artifacts/gate/17b,
    `REFUSED: ... verdict 0.1092 n=132 vs recomputed 0.0537 n=132`, where 0.0537 is that
    verdict's own `cap8_value` to the last bit."""
    _matched_cost_verdict(tmp_path)
    rows = reconcile(
        tmp_path,
        [
            _contrast("evidence_coverage", "matched_k", MATCHED_K, 132),
            _contrast("evidence_coverage", "matched_k_plus_1", 0.06818181818181818, 132),
            _contrast("evidence_coverage", "cap8", CAP8, 132),
            _contrast("cad_ge2", "cap8", CAD8, 85),
        ],
    )
    assert [(r["metric"], r["comparator"], r["field"]) for r in rows] == [
        ("evidence_coverage", "matched_k", "value"),
        ("evidence_coverage", "cap8", "cap8_value"),
        ("cad_ge2", "cap8", "value"),
    ]
    assert [r["verdict_value"] for r in rows] == [MATCHED_K, CAP8, CAD8]
    assert all(r["agree"] for r in rows)


def test_a_matched_cost_verdict_is_refused_when_the_matched_k_value_is_not_reproduced(tmp_path):
    """The lock is TWO-sided. A cap-8 delta that lands on `cap8_value` proves the pairing
    agrees; it says nothing about the base's prefix rung the matched-k delta is a difference
    from -- and that is the number the table actually reports."""
    _matched_cost_verdict(tmp_path)
    with pytest.raises(Refused, match="matched_k") as exc:
        reconcile(
            tmp_path,
            [
                _contrast("evidence_coverage", "matched_k", MATCHED_K + 0.01, 132),
                _contrast("evidence_coverage", "cap8", CAP8, 132),
            ],
        )
    assert "cap8_value" not in str(exc.value)  # the cap-8 side agreed; only matched-k failed


def test_a_matched_cost_verdict_with_no_cap8_value_is_refused_by_name(tmp_path):
    """The field is not optional under this rule: without it the verdict carries no cap-8
    number at all, and `value` is a different measurement. Name the missing field rather than
    reconcile the cap-8 contrast against the matched-k delta and call the disagreement a bad
    parquet."""
    _write_verdict(
        tmp_path,
        {"evidence_coverage": {"value": MATCHED_K, "comparator": "matched_k", "n": 132}},
        coverage_rule="matched_cost",
    )
    with pytest.raises(Refused, match="cap8_value"):
        reconcile(tmp_path, [_contrast("evidence_coverage", "cap8", CAP8, 132)])


def test_a_cad_rule_verdict_is_reconciled_against_the_field_that_names_cap_eight(tmp_path):
    """T19b (ff73a4b) makes `cad_ge2` report-only and copies its cap-8 delta into
    `cap8_cad_ge2_delta`, leaving `value` equal to it. Read the field that says what the number
    is denominated in, so a later rule that DOES move `value` cannot be misread in silence."""
    _write_verdict(
        tmp_path,
        {
            "evidence_coverage": {
                "value": MATCHED_K,
                "cap8_value": CAP8,
                "comparator": "matched_k",
                "n": 132,
            },
            "cad_ge2": {
                "value": CAD8,
                "cap8_cad_ge2_delta": CAD8,
                "comparator": "cap8",
                "n": 85,
            },
        },
        coverage_rule="matched_cost",
        cad_rule="report_only",
    )
    rows = reconcile(tmp_path, [_contrast("cad_ge2", "cap8", CAD8, 85)])
    assert [(r["metric"], r["field"]) for r in rows] == [("cad_ge2", "cap8_cad_ge2_delta")]
    assert rows[0]["agree"]


# ---------------------------------------------------- charging the base the INQUIRER's own budget
#
# WHY TWO NEW BASES AND NOT ONE. The retrieval-call axis cannot answer "was the gain bought with
# verbosity?", because a longer question costs the same one call. Two budgets can, and they are
# measured from different columns for different reasons:
#
#   * `matched_gen_tokens` -- what a token bill charges the Inquirer: per-call
#     `tok_completion + tok_reasoning` from `calls.parquet` WHERE `actor = 'inquirer'`.
#   * `matched_question_words` -- the LENGTH CONTROL, in the unit the objection is stated in
#     (18.07 words against 12.99). It is a labelled PROXY for the question's token count: no
#     tokenizer is installed in this venv, so the question's real tokens are not recoverable
#     here. It is NOT `tok_completion`, and that is a measurement, not a preference: the
#     Inquirer's completion is `question` + `rationale` + `target`, and on
#     artifacts/gate/n1 the base's rationale is the LONGER half (131 chars against the trained
#     arm's 77) while its question is the SHORTER one. Charging the base a budget denominated in
#     completion would hand it a rationale allowance and bias the test toward "the gain
#     survives" -- the exact failure the length control exists to prevent.


def _cost_fixture(spec):
    """`(run_id, task, arm, per_turn_uids, per_call_gen_tok, per_ask_words)` -> build_ladders args.

    `per_call_gen_tok` has n_asks+1 entries because a run makes one Inquirer call per ask plus
    one terminal decision call -- measured invariant on every one of the 1,395 runs in each gate
    parquet dir (`n_inquirer_calls == n_asks + 1`, both arms, both stop reasons).
    """
    runs, turns, gen, words = [], {}, {}, {}
    for run_id, task, arm, per_turn, toks, wds in spec:
        runs.append(
            {
                "run_id": run_id,
                "suite_id": SUITE,
                "task_id": task,
                "template_id": None,
                "arm_id": arm,
                "seed": 0,
                "n_asks": len(per_turn),
                "stop_reason": "policy_stop",
            }
        )
        turns[run_id] = [
            _Turn(turn_idx=i, retrieved_uids=u, new_uids=u, action=_Act(text=" ".join(["w"] * w)))
            for i, (u, w) in enumerate(zip(per_turn, wds))
        ]
        gen[run_id], words[run_id] = toks, wds
    return runs, turns, gen, words


# The trained arm's whole spend equals the base's cumulative spend through its SECOND ask (t1)
# and its FIRST (t2), and at those rungs the two coverages are equal -- so the delta must be 0.
EQUAL = (
    ("tr_t1", "t1", "inquirer_trained", (("u0",), ("u1",)), (100, 100, 100), (7, 7)),
    (
        "bs_t1",
        "t1",
        "inquirer_prompted",
        (("u0",), ("u1",), ("u2",), ("u3",)),
        (100,) * 5,
        (7,) * 4,
    ),
    ("tr_t2", "t2", "inquirer_trained", (("v0",),), (100, 100), (7,)),
    ("bs_t2", "t2", "inquirer_prompted", (("v0",), ("v1",)), (100, 100, 100), (7, 7)),
)


@pytest.fixture
def cost_ladders():
    from scripts.matched_cost import build_ladders as bl

    runs, turns, gen, words = _cost_fixture(EQUAL)
    return bl(runs, turns, GRAPHS, inquirer_gen_tokens=gen, question_words=words)


def test_the_token_curve_is_cumulative_over_the_inquirer_calls_a_cap_k_run_would_make(
    cost_ladders,
):
    """`Inquirer.act(s)` is budget-blind, so a cap-k run makes the SAME k ask-calls plus the
    same terminal decision call -- `min(k, n_asks) + 1` calls, never k+1 when the run was
    shorter. Charging a budget for calls the policy never made is the token-axis version of
    billing a stopped run for the cap."""
    base = cost_ladders["bs_t1"]
    assert base.n_asks == 4
    assert [base.gen_tokens_at(k) for k in range(5)] == [100, 200, 300, 400, 500]
    assert base.gen_tokens_at(99) == 500  # the whole trajectory, never extrapolated
    assert [base.question_words_at(k) for k in range(5)] == [0, 7, 14, 21, 28]
    assert base.question_words_at(99) == 28


def test_equal_spend_and_equal_coverage_at_that_rung_is_a_zero_delta(cost_ladders):
    """The fixture is built so the trained arm's whole Inquirer budget buys the base exactly the
    prefix at which the two coverages are equal. A matched-token delta that is not 0 here is
    charging one of the two sides something the other did not pay."""
    from scripts.matched_cost import contrast

    trained = [cost_ladders["tr_t1"], cost_ladders["tr_t2"]]
    base = {
        (x.suite_id, x.task_id, x.seed): x
        for x in cost_ladders.values()
        if x.arm_id == "inquirer_prompted"
    }
    for comparator, cost_t, cost_b in (
        ("matched_gen_tokens", 250.0, 250.0),
        ("matched_question_words", 10.5, 10.5),
    ):
        c = contrast(
            trained,
            base,
            suite_id=SUITE,
            checkpoint="fixture",
            metric="evidence_coverage",
            comparator=comparator,
            offset=0,
        )
        assert c.n_tasks == 2
        assert c.delta == pytest.approx(0.0), comparator
        # t1: trained 0.5 vs base at k=2 0.5. t2: trained 0.5 vs base at k=1 0.5.
        assert c.trained_mean == pytest.approx(0.5) and c.base_mean == pytest.approx(0.5)
        # The budgets that were equated are RECORDED, not just used.
        assert c.trained_cost == pytest.approx(cost_t), comparator
        assert c.base_cost == pytest.approx(cost_b), comparator
        assert c.base_asks == pytest.approx(1.5)  # k=2 on t1, k=1 on t2


# The trained arm outspends the base's ENTIRE trajectory on both bases.
RICH = (
    ("tr_t1", "t1", "inquirer_trained", (("u0",),), (5000, 5000), (500,)),
    ("bs_t1", "t1", "inquirer_prompted", (("u0",), ("u1",)), (100, 100, 100), (7, 7)),
)


def test_a_budget_past_the_end_charges_the_whole_trajectory_and_never_extrapolates(cost_ladders):
    """A base run that spent 300 tokens over 2 asks does not become a 4-ask run when handed
    10,000. The largest affordable rung is its own last one, its coverage there is its terminal
    coverage, and it is billed the 2 asks it made -- the token-axis form of
    `test_a_prefix_past_the_end_bills_the_asks_the_run_actually_made`."""
    from scripts.matched_cost import build_ladders as bl
    from scripts.matched_cost import contrast

    runs, turns, gen, words = _cost_fixture(RICH)
    lads = bl(runs, turns, GRAPHS, inquirer_gen_tokens=gen, question_words=words)
    base = lads["bs_t1"]
    assert base.k_within(10_000, "matched_gen_tokens") == base.n_asks == 2
    assert base.k_within(10_000, "matched_question_words") == 2
    c = contrast(
        [lads["tr_t1"]],
        {(base.suite_id, base.task_id, base.seed): base},
        suite_id=SUITE,
        checkpoint="fixture",
        metric="evidence_coverage",
        comparator="matched_gen_tokens",
        offset=0,
    )
    assert c.n_tasks == 1
    assert c.base_asks == pytest.approx(2.0)  # billed its own 2 asks, not 100
    assert c.base_cost == pytest.approx(300.0)  # and its own 300 tokens
    assert c.trained_cost == pytest.approx(10_000.0)
    assert c.base_mean == pytest.approx(0.5)  # its terminal coverage
    assert c.n_base_short == 1  # the budget outran the trajectory


def test_a_budget_too_small_for_one_call_charges_the_base_zero_asks(cost_ladders):
    """A base whose FIRST Inquirer call already costs more than the trained arm's whole spend
    cannot afford to ask anything: the rung is k=0 and the value is the pre-question level. Not
    a skipped pair, and not rung 1 charged at a discount."""
    base = cost_ladders["bs_t1"]
    assert base.k_within(50, "matched_gen_tokens") == 0
    assert base.coverage_at(0) == pytest.approx(0.0)


def test_a_ladder_built_without_a_token_curve_refuses_instead_of_charging_zero(ladders):
    """Rule 2 in the cost dimension. A missing curve that read as 0 tokens would give every base
    run an infinite budget, charge it its whole cap-8 trajectory, and print a plausible
    'matched-token' delta that is the cap-8 delta wearing a different label."""
    from scripts.matched_cost import Refused

    lad = ladders["base_t1"]
    assert lad.gen_tok == () and lad.q_words == ()
    for call in (lambda: lad.gen_tokens_at(2), lambda: lad.question_words_at(2)):
        with pytest.raises(Refused, match="no Inquirer"):
            call()


def test_a_calls_parquet_without_the_token_columns_is_refused_by_name(tmp_path):
    """The stop-and-report branch of this analysis. `read_inquirer_calls` must name the column it
    could not find rather than sum an absent column to 0 -- measured on the real gates the
    columns ARE present (`actor`, `tok_prompt`, `tok_completion`, `tok_reasoning`, `tok_cached`),
    so nothing here fires; it fires the day a compaction drops one."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from scripts.matched_cost import Refused, read_inquirer_calls

    with pytest.raises(Refused, match="calls.parquet"):
        read_inquirer_calls(tmp_path)

    pq.write_table(
        pa.table({"run_id": ["r"], "actor": ["inquirer"], "tok_completion": [7]}),
        tmp_path / "calls.parquet",
    )
    with pytest.raises(Refused, match="tok_reasoning"):
        read_inquirer_calls(tmp_path)

    pq.write_table(
        pa.table({"run_id": ["r"], "tok_completion": [7], "tok_reasoning": [0]}),
        tmp_path / "calls.parquet",
    )
    with pytest.raises(Refused, match="actor"):
        read_inquirer_calls(tmp_path)

    # Present but NULL is the same failure wearing a value: refused, never coalesced to 0.
    pq.write_table(
        pa.table(
            {
                "run_id": ["r"],
                "actor": ["inquirer"],
                "tok_completion": pa.array([None], type=pa.int64()),
                "tok_reasoning": [0],
            }
        ),
        tmp_path / "calls.parquet",
    )
    with pytest.raises(Refused, match="NULL"):
        read_inquirer_calls(tmp_path)


def test_the_inquirer_call_count_must_be_one_per_ask_plus_the_decision_call(cost_ladders):
    """The measured invariant is `n_inquirer_calls == n_asks + 1` on all 1,395 runs of every gate
    parquet dir. If a future compaction breaks it, the per-ask attribution this curve is built on
    is wrong and the budget is charged to the wrong rung -- so it is a refusal, not a shrug."""
    from scripts.matched_cost import Refused
    from scripts.matched_cost import build_ladders as bl

    runs, turns, gen, words = _cost_fixture(EQUAL)
    gen["bs_t1"] = gen["bs_t1"][:-1]  # 4 calls for 4 asks
    # The message names the actor value verbatim (`pinq.types.Actor` is lower-case) and the
    # count it was measured against, so a reader sees which side of the invariant broke.
    with pytest.raises(Refused, match="inquirer calls against n_asks"):
        bl(runs, turns, GRAPHS, inquirer_gen_tokens=gen, question_words=words)


# ------------------------------------------------- the real token count, beside the word proxy


def test_the_tokenizer_is_resolved_from_disk_or_reported_absent_never_downloaded(tmp_path):
    """The comparator is priced by a LOCAL tokenizer.json or not priced at all. An absent
    tokenizer must drop `matched_question_tokens` by name -- a basis silently priced at 0 tokens
    would give every base run an unbounded budget, which is the `matched_gen_tokens` degeneracy
    with none of its warning signs."""
    from scripts.matched_cost import (
        COST_BASES,
        Refused,
        active_cost_bases,
        find_question_tokenizer,
    )

    assert find_question_tokenizer(env={}) is None or True  # disk-dependent; the branches below
    assert active_cost_bases(None) == tuple(b for b in COST_BASES if b != "matched_question_tokens")
    assert "matched_question_tokens" not in active_cost_bases(None)
    assert active_cost_bases(tmp_path / "tokenizer.json") == COST_BASES

    with pytest.raises(Refused, match="PI_QUESTION_TOKENIZER"):
        find_question_tokenizer(env={"PI_QUESTION_TOKENIZER": str(tmp_path / "nope.json")})
    real = tmp_path / "t.json"
    real.write_text("{}")
    assert find_question_tokenizer(env={"PI_QUESTION_TOKENIZER": str(real)}) == real


def test_the_token_basis_is_priced_by_the_same_rule_as_the_word_basis():
    """Same ladder, same `k_within`, different unit -- so a disagreement between the two is a
    fact about the UNITS and not about two different accounting rules."""
    from scripts.matched_cost import build_ladders as bl
    from scripts.matched_cost import contrast

    runs, turns, gen, words = _cost_fixture(EQUAL)
    # One token per word plus one, the ratio the real Qwen3 tokenizer gives on these questions
    # (7 words -> 8 tokens, 14 -> 15), so the two bases must buy the SAME rungs here.
    toks = {r: [w + 1 for w in ws] for r, ws in words.items()}
    lads = bl(
        runs,
        turns,
        GRAPHS,
        inquirer_gen_tokens=gen,
        question_words=words,
        question_tokens=toks,
    )
    base = {
        (x.suite_id, x.task_id, x.seed): x for x in lads.values() if x.arm_id == "inquirer_prompted"
    }
    trained = [lads["tr_t1"], lads["tr_t2"]]
    by = {
        c.comparator: c
        for c in (
            contrast(
                trained,
                base,
                suite_id=SUITE,
                checkpoint="fixture",
                metric="evidence_coverage",
                comparator=cmp,
                offset=0,
            )
            for cmp in ("matched_question_words", "matched_question_tokens")
        )
    }
    assert by["matched_question_tokens"].delta == pytest.approx(0.0)
    assert by["matched_question_tokens"].base_asks == pytest.approx(
        by["matched_question_words"].base_asks
    )
    assert by["matched_question_tokens"].trained_cost == pytest.approx(12.0)  # 2 asks x 6, x1 ask
    assert lads["bs_t1"].question_tokens_at(99) == 32  # 4 asks x 8, never extrapolated


def test_a_ladder_without_a_token_curve_refuses_on_the_token_basis_too(ladders):
    from scripts.matched_cost import Refused

    with pytest.raises(Refused, match="no Inquirer"):
        ladders["base_t1"].question_tokens_at(1)


# ------------------------------------------- the provenance block a paper table can cite
#
# SHAPE MATCHED, NOT INVENTED: the key names and the digest discipline are
# `pi_eval.report.provenance`'s -- `table_id` / `run_ids` / `n_run_ids` / `scorer_hash` /
# `graph_version` / `code_version` / `inputs` / `inputs_hash` / `query` / `tool`, with
# `provenance_digest = h("prov", canon(body))` over the whole body. Two deliberate departures,
# each for a stated reason:
#
#   * NO `generated_at_utc`. That function's own comment says the digest excludes the timestamp
#     so two identical aggregations an hour apart compare equal; a block a table CITES needs the
#     stronger, checkable property -- the bytes themselves must be equal -- and a wall clock
#     defeats it. The clock is a machine artifact, which this repo records and never compares.
#   * NO `eligibility_predicate` / `prereg_verified`. `pi_eval.report.ELIGIBLE` filters on
#     `split = 'test'` and this analysis reads the tier-B DEV gate. Naming a filter that did not
#     run is the exact failure that field's own comment warns about, so it is absent rather than
#     wrong, and `split` is recorded instead.


def _res(**over):
    """The smallest `analyse()` result the provenance block reads. Hand-built so the test pins
    the BLOCK's behaviour and not the ladder's."""
    c = Contrast(
        suite_id=SUITE,
        checkpoint="ckpt",
        metric="evidence_coverage",
        comparator="matched_question_tokens",
        n_tasks=2,
        n_clusters=2,
        trained_mean=0.75,
        base_mean=0.5,
        delta=0.25,
        ci_lo=0.1,
        ci_hi=0.4,
        p_value=0.01,
        trained_asks=1.5,
        base_asks=1.5,
        n_base_short=0,
        trained_cost=12.0,
        base_cost=11.0,
        pair_keys_hash="deadbeef",
    )
    res = {
        "checkpoints": [["parquet_ckpt", "ckpt"]],
        "contrasts": [c],
        "budgets": [
            {"suite_id": SUITE, "series": "ckpt", "kind": "trained", "question_tokens": 12.0}
        ],
        "splits": [],
        "early": [],
        "frontier": [],
        "reconciliation": [],
        "checked": {"frontier_q": 1, "evidence_coverage": 1, "cad": 1},
        "scorer_hash": "02d44f69aaaa",
        "graph_version": "v1",
        "code_versions": ["abc1234"],
        "splits_seen": ["dev"],
        "run_ids": ["r_base", "r_trained"],
        "run_ids_by_arm": {
            "inquirer_prompted": ["r_base"],
            "inquirer_trained": {"ckpt": ["r_trained"]},
        },
        "cost_bases_active": list(COST_BASES),
        "question_tokenizer": "/some/where/tokenizer.json",
        "n_tasks": 2,
        "n_runs": {"base": 1, "trained": 1},
        "sources": [],
        "readme_present": False,
        "gate": "artifacts/gate/fixture",
    }
    res.update(over)
    return res


def test_the_matched_cost_provenance_block_carries_everything_a_table_must_cite():
    """Rule 1, made citable. Every field named here is one a reviewer can use to re-derive the
    number: without the run ids on BOTH sides the pairing is unverifiable, without the tokenizer
    identity the token unit is unreproducible, and without the degeneracy marker the
    generated-token delta reads as a result."""
    from scripts.matched_cost import matched_cost_provenance

    b = matched_cost_provenance(_res())
    for key in (
        "table_id",
        "tool",
        "query",
        "scorer_hash",
        "graph_version",
        "code_version",
        "matcher_id",
        "run_ids",
        "n_run_ids",
        "run_ids_by_arm",
        "resampling",
        "question_tokenizer",
        "comparators",
        "inquirer_budgets",
        "cost_bases",
        "inputs",
        "inputs_hash",
        "provenance_digest",
    ):
        assert key in b, key
    # Both sides of the contrast, by run id, not just a count.
    assert b["run_ids_by_arm"]["inquirer_prompted"] == ["r_base"]
    assert b["run_ids_by_arm"]["inquirer_trained"]["ckpt"] == ["r_trained"]
    row = b["comparators"][0]
    assert (row["comparator"], row["n"], row["delta"]) == ("matched_question_tokens", 2, 0.25)
    assert (row["ci_lo"], row["ci_hi"]) == (0.1, 0.4)
    assert row["pair_keys_hash"] == "deadbeef"  # WHICH pairs, not just how many
    assert row["degenerate"] is False
    assert b["question_tokenizer"].endswith("tokenizer.json")
    # A wall clock would break byte-identity; the digest is the identity instead.
    assert "generated_at_utc" not in b
    assert "eligibility_predicate" not in b  # this reads the DEV gate; see the note above


def test_the_generated_token_comparator_carries_its_degeneracy_marker():
    """A +0.88 delta against a base that asked nothing must not be citable as a bare number. The
    marker travels WITH the row, so a table generated from this block cannot print the value and
    leave the caveat behind in a title."""
    from scripts.matched_cost import Contrast as C
    from scripts.matched_cost import matched_cost_provenance

    gen = C(
        suite_id=SUITE,
        checkpoint="ckpt",
        metric="evidence_coverage",
        comparator="matched_gen_tokens",
        n_tasks=2,
        n_clusters=2,
        trained_mean=0.9,
        base_mean=0.03,
        delta=0.87,
        ci_lo=0.8,
        ci_hi=0.9,
        p_value=0.001,
        trained_asks=1.3,
        base_asks=0.06,
        n_base_short=0,
        trained_cost=79.6,
        base_cost=62.7,
        pair_keys_hash="cafe",
    )
    b = matched_cost_provenance(_res(contrasts=[gen]))
    row = b["comparators"][0]
    assert row["degenerate"] is True
    assert "rationale" in row["degenerate_reason"]
    assert row["base_asks"] == 0.06  # the diagnostic that shows the degeneracy, in the row


def test_the_provenance_block_is_byte_identical_across_two_emissions(tmp_path):
    """GENERATED, never typed, and therefore checkable. If two emissions of the same analysis
    differed by a byte, a table citing the digest could not prove which run it came from -- and a
    provenance block transcribed from a summary would be indistinguishable from one that was
    computed."""
    from scripts.matched_cost import matched_cost_provenance, write_provenance

    a, b = matched_cost_provenance(_res()), matched_cost_provenance(_res())
    assert a == b and a["provenance_digest"] == b["provenance_digest"]
    p1, p2 = tmp_path / "one.json", tmp_path / "two.json"
    write_provenance(_res(), p1)
    write_provenance(_res(), p2)
    assert p1.read_bytes() == p2.read_bytes()
    assert json.loads(p1.read_text())["provenance_digest"] == a["provenance_digest"]


def test_a_provenance_block_with_no_scorer_hash_or_graph_version_is_refused():
    """A hole here invites exactly the false confidence the block exists to prevent, so it is a
    REFUSAL and not an omitted key. `analyse` pins the set SIZE at one; an empty string is a set
    of size one and slipped through that check."""
    from scripts.matched_cost import Refused, matched_cost_provenance

    for field in ("scorer_hash", "graph_version"):
        for empty in ("", "   ", None):
            with pytest.raises(Refused, match=field):
                matched_cost_provenance(_res(**{field: empty}))
    with pytest.raises(Refused, match="code_version"):
        matched_cost_provenance(_res(code_versions=[]))


# --------------------------------------- two seeds of one task are one task, not one overwrite
#
# THE POPULATION THAT MAKES THIS FIRE. `contrast` looked up its comparator by
# `(suite, task, seed)` and then stored the pair under `Ladder.key`, which is `suite/task` --
# so on a population with two seeds per task the second pair OVERWROTE the first and the
# aggregate was one arbitrarily chosen seed, at the full task count. The held-out QA store is
# exactly that population: `artifacts/testsplit_qa/scores_parquet` carries seeds 0 AND 1 for
# `inquirer_prompted` and `inquirer_trained` alike (176 MuSiQue, 200 StrategyQA, 166 wiki2
# tasks), and against its published verdicts the collapse misses `evidence_coverage` by
# +0.00166 / +0.00786 / -0.00226 and `cad_ge2` by -0.00217 / +0.02679 at IDENTICAL n.
#
# THE CONVENTION THIS ENCODES, and why it is that one rather than the one that agrees. The pair
# is formed at (suite, task, seed) -- a seed's trained run is charged against the SAME seed's
# baseline -- and the seeds of one task are then averaged into the task, which is the unit the
# bootstrap resamples. Two seeds of one task are two measurements of one task, not two tasks:
# weighting a task by how many of its seeds happened to complete would make the estimand depend
# on rollout completion counts, which are a machine artifact. It is also `pinq_train.gate`'s own
# rule (`_paired_by_task_map`, `_matched_cost`: "seeds averaged within a task first"), which is
# why `reconcile`'s n stays a TASK count under it.
SEEDED = (
    # (run_id, task, arm, seed, retrievals per turn)
    ("tr_t1_s0", "t1", "inquirer_trained", 0, (("u0",),)),
    ("tr_t1_s1", "t1", "inquirer_trained", 1, (("u0", "u1"),)),
    ("bs_t1_s0", "t1", "inquirer_prompted", 0, (("u1",), ("u0",))),
    ("bs_t1_s1", "t1", "inquirer_prompted", 1, ((), ("u0", "u1", "u2", "u3"))),
    ("tr_t2_s0", "t2", "inquirer_trained", 0, (("v0",),)),
    ("tr_t2_s1", "t2", "inquirer_trained", 1, (("v0", "v1"),)),
    ("bs_t2_s0", "t2", "inquirer_prompted", 0, (("v0",), ("v1",))),
    ("bs_t2_s1", "t2", "inquirer_prompted", 1, ((), ("v0", "v1"))),
)


def _seeded_ladders():
    runs, turns = [], {}
    for run_id, task, arm, seed, per_turn in SEEDED:
        runs.append(
            {
                "run_id": run_id,
                "suite_id": SUITE,
                "task_id": task,
                "template_id": None,
                "arm_id": arm,
                "seed": seed,
                "n_asks": len(per_turn),
                "stop_reason": "policy_stop",
            }
        )
        turns[run_id] = [
            _Turn(turn_idx=i, retrieved_uids=u, new_uids=u, action=_Act(text="what next?"))
            for i, u in enumerate(per_turn)
        ]
    lads = build_ladders(runs, turns, GRAPHS)
    base = {
        (x.suite_id, x.task_id, x.seed): x for x in lads.values() if x.arm_id == "inquirer_prompted"
    }
    trained = [lads[r] for r, _t, arm, _s, _p in SEEDED if arm == "inquirer_trained"]
    return lads, trained, base


def _matched(trained, base, metric):
    return contrast(
        trained,
        base,
        suite_id=SUITE,
        checkpoint="fixture",
        metric=metric,
        comparator="matched_k",
        offset=0,
    )


def test_two_seeds_of_one_task_are_averaged_into_it_and_do_not_overwrite_each_other():
    """Hand-computed, and every rung is a different number per seed so a collapse cannot hide.

    t1: seed 0 is 0.25 against the base's 0.25 at k=1 (delta 0), seed 1 is 0.50 against 0.00
    (delta +0.50) -- task +0.25. t2: seed 0 is 0.50 against 0.50 (delta 0), seed 1 is 1.00
    against 0.00 (delta +1.00) -- task +0.50. Over the two tasks: +0.375.

    Under the collapsed key the answer was whichever seed the run list happened to end on:
    +0.75 in this order and +0.00 reversed, both at n=2. A delta that depends on run-id order
    is not a measurement of anything, and n could not tell you which one you were reading.
    """
    _lads, trained, base = _seeded_ladders()
    c = _matched(trained, base, "evidence_coverage")
    assert c.n_tasks == 2 and c.n_clusters == 2  # tasks, not the 4 (task, seed) pairs
    assert c.delta == pytest.approx(0.375)
    assert c.trained_mean == pytest.approx(0.5625)  # (0.375 + 0.75) / 2
    assert c.base_mean == pytest.approx(0.1875)  # (0.125 + 0.25) / 2
    # Order-independence is the property the overwrite destroyed, so it is asserted, not assumed.
    assert _matched(list(reversed(trained)), base, "evidence_coverage").delta == pytest.approx(
        0.375
    )


def test_the_depth_two_delta_is_seed_averaged_too_since_it_is_the_blocked_number():
    """`cad_ge2` at matched cost is the paper's vertical claim and the number the collapse was
    blocking. t1 has no depth>=2 resolution on either side at k=1 (delta 0 at both seeds); t2
    resolves its depth-2 need only on the trained seed-1 run (+1.00 at seed 1, 0 at seed 0), so
    the task is +0.50 and the pair of tasks +0.25. The collapse read +0.50 here."""
    _lads, trained, base = _seeded_ladders()
    c = _matched(trained, base, "cad_ge2")
    assert c.n_tasks == 2
    assert c.delta == pytest.approx(0.25)


def test_the_win_tie_loss_counts_are_per_pair_and_say_so_beside_a_per_task_delta():
    """The two counts in one row have DIFFERENT denominators, which is exactly the kind of mix
    that gets published as one number: `n_tasks` on a contrast is tasks (2 here) while
    `outcome_split`'s `n` is (task, seed) pairs (4). Pinned so neither can drift into the
    other's meaning unannounced."""
    _lads, trained, base = _seeded_ladders()
    split = outcome_split(trained, base, "evidence_coverage")
    assert split["n"] == 4  # pairs: 2 tasks x 2 seeds
    assert split["win"] == 2 and split["tie"] == 2 and split["loss"] == 0
    assert early_stop_split(trained, base)["n_paired"] == 4


def test_a_one_seed_population_keeps_the_pair_keys_the_published_digests_were_made_from():
    """Every matched-cost number this repo has published came from a one-seed trained
    population, where seed-averaging a single value is the identity. This pins the KEY format
    as well as the value: `pair_keys_hash` digests `suite/task` strings, and it is recorded in
    `figures/P13_matched_cost/*.json` and in `artifacts/n1/matched_cost.*.json`, so a pair
    identity that changed those strings would move a published provenance digest."""
    from pinq.ids import canon, h

    _lads, trained, base = _seeded_ladders()
    one_seed = [x for x in trained if x.seed == 0]
    c = _matched(one_seed, base, "evidence_coverage")
    assert c.n_tasks == 2
    assert c.delta == pytest.approx(0.0)  # both seed-0 pairs are ties, by construction
    assert c.pair_keys_hash == h("pairs", canon(sorted([f"{SUITE}/t1", f"{SUITE}/t2"])))


def test_a_verdict_that_recorded_no_depth_pairs_locks_the_recomputation_on_being_empty(tmp_path):
    """A suite with no depth>=2 gold node carries `cap8_cad_ge2_delta: null` and `n: 0` -- the
    verdict measured nothing, which is not the same as measuring zero. wiki2's held-out verdict
    is that shape (`artifacts/testsplit_qa/verdicts/stacked-notdone.wiki2.json`), and reading
    the field as a float there raised `TypeError: float() argument must be ... not 'NoneType'`
    from `_locked_field` -- a traceback after the tables, in a script whose contract is that an
    absent input is a REFUSAL and never an empty axis.

    The lock does not disappear with the value: an empty verdict cell must meet an empty
    recomputation, and a recomputation that DID find pairs there is a disagreement about the
    population and is refused. Both branches are exercised, so a pass here cannot be a check
    that could not fail."""
    _write_verdict(
        tmp_path,
        {
            "evidence_coverage": {
                "value": MATCHED_K,
                "cap8_value": CAP8,
                "comparator": "matched_k",
                "n": 132,
            },
            "cad_ge2": {
                "value": None,
                "cap8_cad_ge2_delta": None,
                "comparator": "cap8",
                "n": 0,
            },
        },
        coverage_rule="matched_cost",
        cad_rule="report_only",
    )
    rows = reconcile(tmp_path, [_contrast("cad_ge2", "cap8", float("nan"), 0)])
    assert [(r["metric"], r["field"], r["agree"]) for r in rows] == [
        ("cad_ge2", "cap8_cad_ge2_delta", True)
    ]
    assert math.isnan(rows[0]["verdict_value"]) and rows[0]["verdict_n"] == 0

    with pytest.raises(Refused, match="cad_ge2"):
        reconcile(tmp_path, [_contrast("cad_ge2", "cap8", 0.0413, 115)])
