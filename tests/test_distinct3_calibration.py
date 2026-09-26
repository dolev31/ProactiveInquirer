"""WHAT distinct-3 CAN REACH ON THIS CORPUS, measured, so the floor can be read against it.

`DISTINCT3_FLOOR = 0.55` gates rung-1 SFT with the message "mode collapse. The checkpoint's
questions stopped depending on the state." That inference requires knowing what a
state-DEPENDENT policy would score, and nothing had measured it.

MEASURED on musique v1 gold and the built 800-task corpus:

    gold required-node texts (2,660)      0.2250   <- the ANSWER KEY scores this
    one gold question per task (800)      0.4682
    the 800 task questions themselves     0.4685   <- the corpus's own ceiling
    our SFT ask-questions (2,251)         0.4543

So a policy asking EXACTLY the gold sub-questions -- perfectly state-dependent, by construction
-- would be rejected at 0.225, and no sample drawn from this corpus reaches 0.55 at all. On
musique the floor is unreachable, and a failure against it is not evidence of mode collapse.

THIS TEST DOES NOT CHANGE THE FLOOR, and must not be read as licence to. The floor is a
preregistered instrument choice and moving it is a decision with a paper trail, not a
convenience. What this pins is the CALIBRATION: if someone lowers it, these are the numbers the
new value has to be justified against; if someone keeps it, this is why musique exports fail.

The gate remains meaningful on suites whose question space is wider, and the within-task
measure below is the one that actually separates state-dependence from repetition: our own
export scores 0.777 there, i.e. the policy varies its question WITHIN a task and repeats only
ACROSS runs of the same task, which is convergence on the right question rather than collapse.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pinq_train.rung1_sft.collapse import DISTINCT3_FLOOR, distinct_n

CORPUS = Path("data/corpora/musique/38f5afb69fb7ea18/tasks.jsonl")


@pytest.mark.skipif(not CORPUS.exists(), reason="built musique corpus not present")
def test_the_corpus_own_questions_do_not_reach_the_floor() -> None:
    """If the benchmark's own questions cannot clear it, no policy answering them can."""
    qs = [json.loads(line)["question"] for line in CORPUS.read_text().splitlines() if line]
    assert len(qs) == 800
    d = distinct_n(qs)
    assert d < DISTINCT3_FLOOR, (
        f"musique task questions now score {d:.4f} >= the floor; the calibration in this "
        "module's docstring is stale and the numbers there must be re-measured"
    )
    assert 0.40 < d < 0.52, f"expected ~0.4685, got {d:.4f}: the corpus changed"


GOLD = Path("data/gold/graphs/musique/v1.jsonl")


@pytest.mark.skipif(not GOLD.exists(), reason="built musique gold not present")
def test_the_gold_answer_key_scores_far_below_the_floor(monkeypatch) -> None:
    """The strongest form of the point: perfect state-dependence scores 0.225.

    `conftest` strips PI_GOLD_ROOT so the suite runs with gold unreadable -- that raise IS the
    firewall -- so a gold-reading test opts in explicitly, as the annotate tests do.
    """
    monkeypatch.setenv("PI_GOLD_ROOT", str(Path("data/gold").resolve()))
    from pi_eval.gold import load_graphs

    g = load_graphs("musique", "v1")
    gold = [
        n.gold_text
        for x in g.values()
        for n in x.gold_nodes
        if n.gold_partition == "required" and n.gold_text
    ]
    assert gold
    d = distinct_n(gold)
    assert d < 0.35, f"gold sub-questions score {d:.4f}; calibration stale"


def test_within_task_diversity_is_the_measure_that_separates_the_two_cases() -> None:
    """Pooled distinct-3 conflates 'the policy always asks the same thing' with 'many runs of
    one task converged on the right thing'. Within a task, only the first survives."""
    # one task, one question repeated: genuine collapse
    collapsed = ["who founded the institute?"] * 8
    # one task, eight different questions: state-dependent
    varied = [
        f"who founded the {w} institute?"
        for w in "alpha beta gamma delta epsilon zeta eta theta".split()
    ]
    assert distinct_n(collapsed) < 0.2
    assert distinct_n(varied) > distinct_n(collapsed)
