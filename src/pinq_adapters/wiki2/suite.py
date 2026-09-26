"""2WikiMultihopQA suite, VIEW SIDE.

Reads only data/corpora/wiki2/<hash>/tasks.jsonl: {id, question, paragraphs[...]}, ten
paragraphs per task. The Wikidata evidence triples, the supporting_facts sentence ids, the
reasoning `type` and the answer are gold.

Licence: Apache-2.0 (Ho et al., COLING 2020).
"""

from __future__ import annotations

from typing import ClassVar

from ..paragraphs import ParagraphSuite

INSTRUCTIONS = (
    "Answer the question using a closed pool of paragraphs. You may issue retrieval queries "
    "against that pool before answering. Some questions compare two entities and some chain "
    "through an intermediate one; several paragraphs are distractors."
)


class Wiki2Suite(ParagraphSuite):
    suite_id: ClassVar[str] = "wiki2"
    suite_version: ClassVar[str] = "v1"
    corpus_id: ClassVar[str] = "wiki2_v1"
    instructions: ClassVar[str] = INSTRUCTIONS
    word_cap: ClassVar[int] = 30
