"""musique_x2 on the VIEW side and in the registry: eval-only, reachable, and read as one suite.

The suite is composed from held-out MuSiQue test tasks, so it may never contribute a training
row: `split_of` must answer `test` for every id and `assert_trainable` must refuse it by name.
Its tasks come in pairs (both question orders of one composition), and the two orders are one
cluster, so `template_id` names the pair.
"""

from __future__ import annotations

import json

import pytest

from pinq_adapters.musique_x2.suite import INSTRUCTIONS, MusiqueX2Suite

PAIR = "x2_0123456789ab"


@pytest.fixture()
def corpus(tmp_path):
    rows = [
        {
            "id": f"{PAIR}_{order}",
            "question": "Answer both questions. (1) Where is A? (2) Who is B?",
            "paragraphs": [
                {"idx": 0, "title": "A", "text": "A is in Lyon."},
                {"idx": 1, "title": "B", "text": "B is Jane."},
            ],
        }
        for order in ("ab", "ba")
    ]
    (tmp_path / "tasks.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return tmp_path


def test_constants(corpus):
    suite = MusiqueX2Suite(corpus)
    assert suite.suite_id == "musique_x2"
    assert suite.corpus_id == "musique_x2_v1"
    assert suite.word_cap == 60
    v = suite.view(f"{PAIR}_ab")
    assert v.word_cap == 60 and v.suite_id == "musique_x2" and v.instructions == INSTRUCTIONS
    # The final answer must state both answers; the instructions are the only channel that says
    # so to the Drafter and the Answerer (both render `view.instructions`).
    assert "both" in INSTRUCTIONS and "(1)" in INSTRUCTIONS and "(2)" in INSTRUCTIONS


def test_both_orders_of_a_pair_are_one_template(corpus):
    suite = MusiqueX2Suite(corpus)
    assert suite.template_id(f"{PAIR}_ab") == suite.template_id(f"{PAIR}_ba") == PAIR
    assert suite.template_id("not_a_composed_id") is None


def test_units_mint_the_composed_corpus_uid(corpus):
    from pi_eval.build.common import unit_uid

    suite = MusiqueX2Suite(corpus)
    uids = {u.uid for u in suite.units(f"{PAIR}_ab")}
    assert unit_uid("musique_x2_v1", f"{PAIR}_ab", 0, "A is in Lyon.") in uids


def test_reachable_from_the_runtime(corpus):
    from pi_run.worker import CORPUS_BACKED, SELF_SOURCED, load_suite

    assert "musique_x2" in CORPUS_BACKED and "musique_x2" not in SELF_SOURCED
    assert isinstance(load_suite("musique_x2", str(corpus)), MusiqueX2Suite)


def test_eval_only_and_refused_by_the_exporter():
    from pinq.splitting import EVAL_ONLY_SUITES, split_of
    from pinq_train.split import SplitViolation, assert_trainable

    assert "musique_x2" in EVAL_ONLY_SUITES
    assert split_of("musique_x2", f"{PAIR}_ab", PAIR) == "test"
    with pytest.raises(SplitViolation, match="eval-only"):
        assert_trainable("musique_x2", f"{PAIR}_ab", PAIR)


def test_registry_declares_the_role():
    from pi_run import suites

    spec = suites.REGISTRY["musique_x2"]
    assert spec.source == "corpus"
    assert spec.status == "live"
    assert spec.mines_training_data is False
    assert spec.carries_endpoints is True
    assert spec.axis == ("vertical", "horizontal")
    assert spec.endpoint_status == "exploratory"
    assert spec.driven_by == "loop"
    assert spec.eval_only
    assert "musique_x2" in suites.transfer_suites()
