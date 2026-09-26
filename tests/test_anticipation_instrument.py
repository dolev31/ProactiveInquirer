"""The anticipation instrument: how many times did the USER have to speak?

WHY THIS IS A VALID ANTICIPATION MEASURE AND NOT A COUNT OF OUR OWN ASKS

`pi_run.stages.tau2_runner` runs the inner policy with `allow_user_target=False`
(`pinq.loop.run_loop:63`), which closes the user channel by construction and charges a
user-targeted ASK as a wasted turn. So an ASK NEVER becomes a message in the tau2
transcript. Every `role == "user"` message is therefore the user simulator speaking of its
own accord -- stating the task, or coming back because the answer did not resolve the need.

That is what makes this endpoint fair: an arm cannot inflate it by asking more, so
"the inquirer reduces how much the user has to say" is a claim the transcript can refute.
The opening task statement is always present, which is why the follow-up count -- not the
raw count -- is the quantity with a meaningful zero.
"""

from __future__ import annotations

import math

from pi_eval.metrics.environment import is_measured, user_followups, user_turns


def test_user_turns_counts_only_user_role() -> None:
    run = {"n_user_turns": 3, "n_messages": 11}
    assert user_turns(run) == 3.0


def test_absent_user_turns_is_nan_not_zero() -> None:
    """A musique run has no user simulator. Zero would say "the user never spoke"."""
    assert math.isnan(user_turns({"suite_id": "musique"}))
    assert not is_measured(user_turns({}))


def test_parquet_null_is_nan_not_zero() -> None:
    """A nullable int32 column reads back as None for every non-tau2 row."""
    assert math.isnan(user_turns({"n_user_turns": None}))


def test_followups_exclude_the_opening_task_statement() -> None:
    """The user always speaks once to state the task; that turn is not a failure to anticipate."""
    assert user_followups({"n_user_turns": 1}) == 0.0
    assert user_followups({"n_user_turns": 4}) == 3.0


def test_followups_absent_when_turns_absent() -> None:
    assert math.isnan(user_followups({}))


def test_zero_user_turns_is_degenerate_not_negative() -> None:
    """A transcript with no user message at all is malformed; it must not yield -1 follow-ups."""
    assert user_followups({"n_user_turns": 0}) == 0.0


def test_compaction_leaves_user_turns_null_on_a_flat_run() -> None:
    """A musique run has no user simulator; stamping 0 would be a fabricated best score."""
    from pi_run.compact import _run_row

    row = _run_row(
        {"run_id": "r1", "suite_id": "musique", "task_id": "t", "arm_id": "a"},
        {"status": "ok", "n_turns": 3},
        {},
    )
    assert row["n_user_turns"] is None, "flat suites must be NULL, never 0"


def test_compaction_carries_the_tau2_count() -> None:
    from pi_run.compact import _run_row

    row = _run_row(
        {"run_id": "r2", "suite_id": "tau2", "task_id": "t", "arm_id": "a"},
        {"status": "ok", "n_user_turns": 4, "n_messages": 15},
        {},
    )
    assert row["n_user_turns"] == 4


def test_runner_counts_user_role_only() -> None:
    """The count must exclude assistant and tool messages, which track agent verbosity."""
    from pi_run.stages.tau2_runner import count_user_turns

    class M:
        def __init__(self, role: str) -> None:
            self.role = role

    msgs = [M("user"), M("assistant"), M("tool"), M("assistant"), M("user"), M("tool")]
    assert count_user_turns(msgs) == 2
    assert count_user_turns(()) == 0


def _score(run_extra: dict, answer_text: str | None = None) -> dict[str, float]:
    """One scored run with a trivial gold graph; returns {metric_name: value}."""
    from pi_eval.gold import GoldGraph, GoldNode
    from pi_eval.score import AnswerRecord, score_run

    graph = GoldGraph(
        gold_suite="s",
        gold_task_key="t",
        gold_nodes=(
            GoldNode(
                gold_suite="s",
                gold_task_key="t",
                gold_node_id="n1",
                gold_text="n1",
                gold_partition="required",
                gold_discoverability="kb",
                gold_ev_uids=("u1",),
            ),
        ),
    )
    run = {"run_id": "r", "suite_id": "s", "task_id": "t", "arm_id": "a", **run_extra}
    rows = score_run(
        run,
        graph=graph,
        turns=[],
        evidence=[{"uid": "u1"}],
        env_calls=[],
        ledger=[],
        records=[],
        answer=None if answer_text is None else AnswerRecord(text=answer_text, cited_uids=()),
    )
    return {r["metric_name"]: r["value"] for r in rows}


def test_anticipation_metrics_absent_on_a_suite_with_no_user() -> None:
    by = _score({})
    assert "n_user_turns" not in by
    assert "n_user_followups" not in by


def test_anticipation_metrics_emitted_when_measured() -> None:
    by = _score({"n_user_turns": 4})
    assert by["n_user_turns"] == 4.0
    assert by["n_user_followups"] == 3.0


def test_answer_length_is_emitted() -> None:
    """The length confound on the musique primary had no instrument at all."""
    by = _score({}, answer_text="the treaty was signed in nineteen nineteen at versailles")
    assert by["answer_n_words"] == 9.0


def test_answer_length_shares_support_with_token_f1() -> None:
    """A covariate that exists on runs the primary does not is unpairable."""
    assert "answer_n_words" not in _score({}), "no answer -> no length, not a 0-word answer"


def test_anticipation_metrics_are_lower_is_better() -> None:
    """A policy that anticipates well makes the user speak LESS."""
    from pi_eval.score import METRICS

    by_name = {m.name: m for m in METRICS}
    for name in ("n_user_turns", "n_user_followups"):
        assert by_name[name].higher_is_better is False, name
        assert by_name[name].judge_derived is False, f"{name} must not sit behind sigma_J"


def test_pre_field_turn_compacts_but_stays_unexportable() -> None:
    """Two guards, two jobs: schema = structural, render_state = semantic.

    A turn recorded before `draft_text` existed must not kill the whole compaction -- 273
    run dirs predate the field -- but it must still be refused at export. Backfilling ""
    is what lets both hold: `cmd_train.render_state` refuses exactly
    `draft_sha and not draft_text`, so the empty string is the SIGNAL it keys on.

    The semantic half is covered by
    `tests/test_train_cli.py::test_a_turn_that_recorded_only_a_draft_sha_is_refused`; this
    test pins the structural half and the fact that "" does not defeat it.
    """
    from pi_run.compact import _migrate

    row = _migrate("turns", {"turn_idx": 0, "draft_sha": "abc123def456"})
    assert row["draft_text"] == "", "structural fill, so compaction survives"
    assert not row["draft_text"], "must stay FALSY, or render_state stops refusing"

    # A current sweep's value is never overwritten by the backfill.
    kept = _migrate("turns", {"turn_idx": 0, "draft_sha": "d", "draft_text": "real draft"})
    assert kept["draft_text"] == "real draft"


# ---------------------------------------------------------------------------
# The length confound on the musique primary, and the endpoint that survives it.
#
# MEASURED, not assumed. On 164 scored musique runs, Spearman(answer_n_words,
# answer_token_f1) is -0.677 pooled and -0.587 averaged WITHIN arm -- -0.934 for
# `inquirer_prompted` alone (n=14). Holding the policy fixed, a longer answer scores worse,
# and the arms span 10.4 to 17.5 mean words. An F1 difference of that size is not
# separable from a length difference of that size.
#
# `token_recall` is the companion that is immune to padding: appending words can never
# lower it. If the proactive arm leads on RECALL, the claim is about finding the answer.
# If it leads only on F1, the claim is about brevity -- and that must be said out loud.
# ---------------------------------------------------------------------------


def test_recall_is_immune_to_padding() -> None:
    """The whole point: appending words cannot lower recall."""
    from pi_eval.metrics.quality import token_recall

    gold = "Versailles"
    assert token_recall("Versailles", gold) == 1.0
    padded = "the treaty was signed at Versailles in nineteen nineteen after long talks"
    assert token_recall(padded, gold) == 1.0


def test_f1_IS_destroyed_by_the_same_padding() -> None:
    """The contrast that makes the confound concrete."""
    from pi_eval.metrics.quality import token_f1

    gold = "Versailles"
    padded = "the treaty was signed at Versailles in nineteen nineteen after long talks"
    assert token_f1("Versailles", gold) == 1.0
    assert token_f1(padded, gold) < 0.25


def test_precision_is_what_length_destroys() -> None:
    from pi_eval.metrics.quality import token_precision

    gold = "Versailles"
    assert token_precision("Versailles", gold) == 1.0
    assert token_precision("Versailles France", gold) == 0.5


def test_recall_and_precision_reconstruct_f1() -> None:
    """If they did not, they would be measuring something other than the primary."""
    from pi_eval.metrics.quality import token_f1, token_precision, token_recall

    for pred, gold in (
        ("paris is the capital", "paris"),
        ("the quick brown fox", "quick fox jumped"),
        ("versailles", "versailles"),
    ):
        p, r = token_precision(pred, gold), token_recall(pred, gold)
        expect = 0.0 if p + r == 0 else 2 * p * r / (p + r)
        assert abs(token_f1(pred, gold) - expect) < 1e-12, (pred, gold)


def test_empty_prediction_scores_zero_not_one() -> None:
    """A policy that says nothing must not be credited with perfect precision."""
    from pi_eval.metrics.quality import token_precision, token_recall

    assert token_recall("", "versailles") == 0.0
    assert token_precision("", "versailles") == 0.0


def test_decomposition_uses_the_SAME_alias_the_primary_scored() -> None:
    """Independent maxima can pick different aliases and then decompose nothing.

    The claim is "recall is the length-robust half of THIS primary". If recall came from a
    different gold string than the F1 being reported, that sentence is false.
    """
    from pi_eval.metrics.quality import best_over_aliases, decompose_over_aliases

    pred = "paris france"
    gold, aliases = "paris", ("france the country", "paris france")
    f1, prec, rec = decompose_over_aliases(pred, gold, aliases)
    assert f1 == best_over_aliases(pred, gold, aliases), "must equal the emitted primary"
    assert abs(f1 - (2 * prec * rec / (prec + rec) if prec + rec else 0.0)) < 1e-12


def test_decomposition_holds_on_every_musique_shaped_case() -> None:
    from pi_eval.metrics.quality import decompose_over_aliases

    cases = [
        ("versailles", "versailles", ()),
        ("the treaty was signed at versailles", "versailles", ()),
        ("", "versailles", ()),
        ("nothing in common", "versailles", ("palace of versailles",)),
    ]
    for pred, gold, al in cases:
        f1, p, r = decompose_over_aliases(pred, gold, al)
        expect = 0.0 if p + r == 0 else 2 * p * r / (p + r)
        assert abs(f1 - expect) < 1e-12, (pred, gold, al, f1, p, r)


def test_an_empty_answer_can_never_score_a_perfect_primary() -> None:
    """DEFENSIVE, and not reachable in today's gold.

    `token_f1("", "")` is 1.0 -- correct for a symmetric measure. It would become a perfect
    primary for an answer that said NOTHING if any gold alias were the empty string.
    Measured: 0 of 16,768 gold records carry an empty alias, so this cannot fire on the
    current gold. It is guarded anyway because the failure is silent and the cost is a line.
    """
    from pi_eval.metrics.quality import decompose_over_aliases

    f1, prec, rec = decompose_over_aliases("", "versailles", ("",))
    assert f1 == 0.0, "an empty answer scoring 1.0 is a fabricated perfect score"
    assert (prec, rec) == (0.0, 0.0)


# ---------------------------------------------------------------------------
# The malformed-answer confound.
#
# MEASURED on 164 musique runs: the fraction of "answers" that are actually a retrieval
# tool call serialized as JSON -- e.g. {"query": "...", "top_n": 5, "source": "corpus"} --
# is strongly arm-dependent, and highest in exactly the arms that lose:
#
#     random_q 40.0%   drafter_only 28.6%   verbosity 14.3%   self_ask 7.1%
#     every inquirer variant, ircot, checklist, query_expansion, compute_matched: 0.0%
#
# So part of any headline gap could be "produces prose" vs "emits a tool call" rather than
# proactivity. Restricting to well-formed answers, drafter_only still scores 0.000 recall on
# 10 answers and random_q 0.000 on 6, so on THIS data the gap survives -- but that is a
# measurement that has to be repeatable at n=200, not an ad-hoc script run once.
# ---------------------------------------------------------------------------


def test_a_serialized_tool_call_is_not_a_wellformed_answer() -> None:
    from pi_eval.metrics.quality import is_toolcall_answer

    assert is_toolcall_answer('{"query": "Gotham filming location", "top_n": 5}')
    assert is_toolcall_answer('  [{"name": "retrieve", "arguments": {}}]  ')


def test_prose_is_wellformed_even_when_it_mentions_braces() -> None:
    from pi_eval.metrics.quality import is_toolcall_answer

    assert not is_toolcall_answer("The film was shot in New York.")
    assert not is_toolcall_answer("The set notation {a, b} appears in the article.")
    assert not is_toolcall_answer("")


def test_a_bare_json_scalar_is_still_prose() -> None:
    """`json.loads("1960")` succeeds. A year is an ANSWER, not a tool call."""
    from pi_eval.metrics.quality import is_toolcall_answer

    assert not is_toolcall_answer("1960")
    assert not is_toolcall_answer('"Versailles"')


def test_wellformed_is_emitted_as_a_metric() -> None:
    by = _score({}, answer_text='{"query": "x", "top_n": 5}')
    assert by["answer_wellformed"] == 0.0
    assert _score({}, answer_text="Versailles")["answer_wellformed"] == 1.0
