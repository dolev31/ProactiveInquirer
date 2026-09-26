"""Lane L1.11: does the trained arm's coverage gain reach the ANSWER-BEARING node, or does it
land on prerequisites while the answer node stays uncovered.

See `answer_node.py` for the node-selection rule, `corpus_text.py` for resolving a node's gold
evidence uids back to corpus text (needed because a node's own `gold_text` is the sub-question,
not the passage -- see that module's docstring), and `lib.py` for the DB/statistics glue.
"""
