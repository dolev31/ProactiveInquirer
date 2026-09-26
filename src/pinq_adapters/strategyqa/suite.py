"""StrategyQA suite, VIEW SIDE.

Reads only data/corpora/strategyqa/<hash>/tasks.jsonl: {id, question, paragraphs[...]}. The
question is a yes/no question whose reasoning steps are NOT stated in it — that implicitness
is the entire point of the dataset, and it is why the public record carries the question and
a paragraph pool and nothing else. The strategy (decomposition), the annotator `facts`, the
evidence paragraph ids and the boolean answer are all gold.

Licence: MIT (Geva et al., TACL 2021).
"""

from __future__ import annotations

from typing import ClassVar

from ..paragraphs import ParagraphSuite

INSTRUCTIONS = (
    "Answer the yes/no question using a closed pool of paragraphs. You may issue retrieval "
    "queries against that pool before answering. The steps needed to reach the answer are "
    "not stated in the question; you must decide for yourself what to look up."
)


class StrategyQASuite(ParagraphSuite):
    suite_id: ClassVar[str] = "strategyqa"
    suite_version: ClassVar[str] = "v1"
    corpus_id: ClassVar[str] = "strategyqa_v1"
    instructions: ClassVar[str] = INSTRUCTIONS
    # A yes/no answer plus a one-clause justification. Small on purpose: with a large cap a
    # verbose arm can hedge both ways and be scored correct by an LLM judge.
    word_cap: ClassVar[int] = 20
