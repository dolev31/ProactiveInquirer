"""Lane L6.1: condition the SFT STOP label on the ANSWER-BEARING node, and read what it buys.

L1.11 (`artifacts/answer_node_coverage_20260918/RESULT.md`) measured that the trained policy's
required-evidence coverage gain lands on prerequisite nodes while the node that names the answer
stays open, and that it stops anyway on 80-100% of such states. The SFT STOP target is derived
from `done_before` -- POOLED completeness -- so nothing in the corpus was ever keyed on that
node. `export_variants.py` builds the two datasets that differ in exactly that and nothing else;
`label_shift.py` is the accounting that says how far apart they are.
"""
