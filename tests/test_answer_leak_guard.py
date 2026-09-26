"""A QUESTION CONTAINING THE GOLD ANSWER IS NOT A QUESTION. It is the answer key, recalled.

WHY THIS EXISTS BEFORE ANY FRONTIER MODEL IS USED TO GENERATE CANDIDATES. Measured over 25
musique tasks, asking each model for its turn-0 question with no evidence retrieved:

    aws/gpt-oss-120b              0/25   0%
    azure/gpt-5.6-terra           3/25  12%   e.g. "Boris Yeltsin death where did he die
                                               Moscow Central Clinical Hospital" (answer: Moscow)
    aws/claude-opus-5             1/21   5%
    gcp/gemini-3.1-pro-preview    0/5    0%   (20 rate-limited)

A stronger model has read the benchmark. It does not infer the chain from evidence; it recalls
it and writes the answer into the query.

THE REWARD CANNOT SEE THIS. Naming the answer retrieves the gold passage perfectly, so a leaked
candidate scores MAXIMUM evidence gain and WINS its preference pair. Contamination therefore
concentrates in exactly the candidates the pipeline selects as best, and the pair teaches the
small model to emit answer-shaped strings it has no ability to recall.

GOLD-SIDE BY NECESSITY: the comparison is against `gold_answer`, which a rollout worker cannot
read (PI_GOLD_ROOT is unset there -- that raise is the firewall). So the flag is computed where
the row is built and enforced where the row is exported.

Catches verbatim recall only. A paraphrase -- "where did Russia's first president die?" -- is
invisible to it, so this bounds the damage rather than eliminating it, and that is why it is a
guard on a frontier generator and not a licence to use one unexamined.
"""

from __future__ import annotations

from pi_eval.metrics.quality import leaks_answer


def test_a_question_naming_the_gold_answer_is_flagged() -> None:
    assert leaks_answer(
        "Boris Yeltsin death where did he die Moscow Central Clinical Hospital", "Moscow"
    )
    assert leaks_answer("Trajan mother Marcia Roman Empire greatest extent", "Marcia")


def test_an_honest_question_is_not() -> None:
    assert not leaks_answer("Where did the Xeer system develop?", "Hassan Gouled Aptidon")
    assert not leaks_answer("Who was the first president of Djibouti?", "Hassan Gouled Aptidon")


def test_the_canary_nonce_is_stripped_before_comparison() -> None:
    """`gold_answer` carries a canary; comparing against the raw field never matches, which is
    how the first version of this measurement reported 0% leakage for every model."""
    assert leaks_answer("the answer is Moscow", "Moscow PINQCANARY_DEADBEEFDEADBEEF")


def test_matching_is_on_token_boundaries() -> None:
    """Substring matching credits gold '18' against 'Founded in 1985', and short numeric golds
    are common."""
    assert not leaks_answer("Founded in 1985", "18")
    assert not leaks_answer("He lived in Austrian lands", "Austria")


def test_an_alias_counts_as_a_leak() -> None:
    assert leaks_answer("Born in Bombay, when?", "Mumbai", ("Bombay",))


def test_a_short_gold_answer_is_not_matched_at_all() -> None:
    """Below a few characters the false-positive rate swamps the signal: a two-letter gold
    would flag half the corpus."""
    assert not leaks_answer("what is it in NY?", "NY")


def test_empty_inputs_are_not_leaks() -> None:
    assert not leaks_answer("", "Moscow")
    assert not leaks_answer("anything", "")


# ------------------------------------------------------------------ enforcement


def _c(run_id, value, action, leak=False):
    return {
        "suite_id": "musique",
        "task_id": "t1",
        "template_id": None,
        "run_id": run_id,
        "turn_idx": 1,
        "state_text": "S",
        "action_json": action,
        "value": value,
        "phi_tilde": value,
        "scorer_hash": "sh",
        "graph_version": "v1",
        "branch_of_run_id": "p",
        "branch_turn_idx": 1,
        "pins_sha": "pins",
        "leaks_gold_answer": leak,
    }


def test_a_leaked_candidate_never_reaches_a_pair() -> None:
    """It would WIN, because naming the answer retrieves the gold passage perfectly."""
    from pinq_train.export.dataset import export_pairs

    pairs, man = export_pairs(
        [
            _c("leaky", 0.99, '{"action":"ASK","question":"who died in Moscow"}', leak=True),
            _c("honest", 0.20, '{"action":"ASK","question":"where did he die?"}'),
        ],
        margin_threshold=0.0,
        len_delta_max=1000,
    )
    assert pairs == [], "a candidate carrying the gold answer reached the training data"
    assert man.n_answer_leak_dropped == 1


def test_an_honest_pair_is_untouched() -> None:
    from pinq_train.export.dataset import export_pairs

    pairs, man = export_pairs(
        [
            _c("a", 0.9, '{"action":"ASK","question":"where did he die?"}'),
            _c("b", 0.1, '{"action":"ASK","question":"when did he die?"}'),
        ],
        margin_threshold=0.0,
        len_delta_max=1000,
    )
    assert len(pairs) == 1 and man.n_answer_leak_dropped == 0


def test_a_leaked_candidate_is_not_an_SFT_target_either() -> None:
    """SFT imitates the best candidate at a state. Teaching the policy to emit a string it
    could only produce by having memorised the benchmark is worse than teaching it nothing."""
    from pinq_train.export.dataset import export_sft

    ex, man = export_sft(
        [_c("leaky", 0.99, '{"action":"ASK","question":"who died in Moscow"}', leak=True)],
        margin_threshold=0.0,
    )
    assert ex == [] or all(not e.action_json.count("Moscow") for e in ex)
    assert man.n_answer_leak_dropped == 1


# ------------------------------------------------------------------ precision


def test_a_short_answer_is_measured_AFTER_normalisation() -> None:
    """`"U.S."` is four raw characters and two normalised ones. Measuring the raw form let it
    through the floor, and it then flagged every question containing "which U.S. state ..." --
    where the answer is context the question needs, not an answer being leaked.

    Measured over 6,788 recorded questions: normalising the length check cut flags from 113 to
    85 without losing either known true positive.
    """
    assert not leaks_answer("Which U.S. state has German as its largest ancestry group?", "U.S.")


def test_a_four_digit_year_is_below_the_floor() -> None:
    """gold "2003" against "What date did Bush declare the war in Iraq (the 2003 invasion)?".

    That is WORLD KNOWLEDGE being used to disambiguate, not benchmark recall -- the model knows
    the Iraq War was 2003 the way anyone does. It is categorically unlike gpt-5.6-terra naming
    "Hassan Gouled Aptidon", which can only come from having the answer chain. The floor is what
    separates the two, and it is set from that distinction rather than from a tuning sweep.
    """
    assert not leaks_answer(
        "What date did President George W. Bush declare the war in Iraq (the 2003 invasion)?",
        "2003",
    )


def test_the_frontier_leaks_still_fire() -> None:
    """The cases the guard exists for. Both are six characters, so the floor does not reach
    them -- verified rather than assumed, because a floor raised far enough to silence the
    false positives would eventually silence these too."""
    assert leaks_answer(
        "Boris Yeltsin death where did he die Moscow Central Clinical Hospital", "Moscow"
    )
    assert leaks_answer("Trajan mother Marcia Roman Empire greatest extent", "Marcia")
