"""musique_x2, VIEW SIDE: two held-out MuSiQue test tasks composed into one task.

Reads only data/corpora/musique_x2/<hash>/tasks.jsonl: {id, question, paragraphs[{idx, title,
text}]}. The question states both constituent questions; the pool is the union of both
constituents' 20-paragraph pools with identical text once. Which constituent a paragraph came
from, the constituents' ids, their decompositions and answers are all gold, built by
pi_eval.build.compose_build into data/gold/. Nothing here knows they exist.

EVAL-ONLY (`pinq.splitting.EVAL_ONLY_SUITES`): every constituent is a MuSiQue TEST task, so a
training row drawn from this suite would put held-out content into a training file.

Framing: composed natural tasks, never "a natural benchmark".
"""

from __future__ import annotations

from typing import ClassVar

from ..paragraphs import ParagraphSuite

# Adapted MINIMALLY from pinq_adapters.musique.suite.INSTRUCTIONS: the pool is two pools, and
# the final answer must state both answers. The Drafter and the frozen Answerer both render
# `view.instructions`, so this is the one channel that tells them there are two questions.
INSTRUCTIONS = (
    "This task states two independent questions. Answer both using a closed pool of about 40 "
    "paragraphs. You may issue retrieval queries against that pool before answering; several "
    "paragraphs are distractors, and each answer usually requires composing facts from more "
    "than one of them. State both answers, labelled (1) and (2)."
)

_ORDERS = ("_ab", "_ba")


def pair_of(task_id: str) -> str | None:
    """The pair a composed task id belongs to: `<pair_id>_ab` / `<pair_id>_ba` -> `<pair_id>`.

    A module function so a reader can cluster on the pair from the public id alone, with the
    same rule the adapter stamps as `template_id`. None for an id of any other shape.
    """
    for suffix in _ORDERS:
        if task_id.endswith(suffix) and task_id.startswith("x2_"):
            return task_id[: -len(suffix)]
    return None


class MusiqueX2Suite(ParagraphSuite):
    suite_id: ClassVar[str] = "musique_x2"
    suite_version: ClassVar[str] = "v1"
    # A NEW corpus id, not musique's: every gold uid is re-minted against it by the builder,
    # which fails the build unless each one is a uid `units()` below emits.
    corpus_id: ClassVar[str] = "musique_x2_v1"
    instructions: ClassVar[str] = INSTRUCTIONS
    # Two short answers plus their labels; frozen across arms like musique's 30.
    word_cap: ClassVar[int] = 60

    def template_id(self, task_id: str) -> str | None:
        """The PAIR: both question orders of one composition are one cluster.

        The id is `<pair_id>_ab` or `<pair_id>_ba` by construction (compose_build), so the pair
        is recoverable from the public id alone. None for an id of any other shape.
        """
        return pair_of(task_id)
