"""MuSiQue-Ans suite, VIEW SIDE.

Reads only data/corpora/musique/<hash>/tasks.jsonl: {id, question, paragraphs[{idx,title,
text}]} — 20 paragraphs for all but 21 of the 19,938 train tasks (those ship 16-19), of
which 2-4 support the answer. The composition recipe that produced
the question — MuSiQue's `question_decomposition`, its `#N` prerequisite placeholders, the
supporting-paragraph indices and the answer itself — is gold, lives in data/gold/, and is
built by pi_eval.build.musique_build. Nothing here knows it exists.

Licence: CC BY 4.0 (Trivedi et al., TACL 2022).
"""

from __future__ import annotations

from typing import ClassVar

from ..paragraphs import ParagraphSuite

INSTRUCTIONS = (
    "Answer the question using a closed pool of 20 paragraphs. You may issue retrieval "
    "queries against that pool before answering; several paragraphs are distractors, and "
    "the answer usually requires composing facts from more than one of them."
)


class MusiqueSuite(ParagraphSuite):
    suite_id: ClassVar[str] = "musique"
    suite_version: ClassVar[str] = "v1"
    corpus_id: ClassVar[str] = "musique_ans_v1p0"
    instructions: ClassVar[str] = INSTRUCTIONS
    # MuSiQue answers are short spans; the cap is frozen across arms so answer length can
    # never be the thing a judge is rewarding.
    word_cap: ClassVar[int] = 30

    def template_id(self, task_id: str) -> str | None:
        """The COMPOSITION, independent of the order the hops were written in.

        A MuSiQue id is `<n>hop<variant>__<hop1>_<hop2>_...`, where the hop ids name the
        single-hop questions the multi-hop question was composed from. The same set of hops
        composed in a different order is the same question in different words -- an actual
        paraphrase, not a topical neighbour. Measured on the built 800-task corpus:

            43 groups of tasks sharing a hop SET but not a hop ORDER (86 tasks)
            26 of those 43 straddled the train/dev/test wall

        including this pair, which the split put on opposite sides:

            test  3hop2__30654_75297_89329
                  "Who started the Bethel branch of the religion founded by the black
                   community in the city that used to be the US capital?"
            dev   3hop2__75297_30654_89329
                  "Who started the Bethel branch of the religion the black community
                   founded in the former US capital?"

        `pinq.splitting.split_key` exists to bind near-duplicates to one side and hashes
        `template_id or task_id`, but NO adapter that can contribute a training row implemented
        `template_id` -- only tau2, which short-circuits to eval-only before the hash is
        reached. So layer 1 of the five-layer contamination guarantee in pinq_train/split.py
        was inert on every trainable suite, and layers 2-5 are all "is this row train?" checks
        that cannot see a near-duplicate. tests/test_train_export.py hand-supplied
        `template_id="tpl7"` and so tested the hash rather than whether any suite produces one.

        Returning the SORTED hop ids makes every permutation hash to one bucket. It also makes
        these tasks one cluster for the CI, which is independently correct: they are not two
        independent draws.

        None for an id that does not parse, so a corpus revision with a different id scheme
        degrades to per-task splitting rather than silently grouping everything.
        """
        head, sep, rest = task_id.partition("__")
        if not sep or not rest:
            return None
        hops = rest.split("_")
        if len(hops) < 2:
            return None
        return f"{head}__" + "_".join(sorted(hops))
