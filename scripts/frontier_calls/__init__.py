"""Lane L1.4: the budget frontiers on the calls axis, not the cap axis.

A cap is an allowance, not a spend: a policy that stops itself (`Inquirer.act(s: State)`
takes no budget, so nothing about it is cap-aware) realizes fewer calls than the cap it ran
under, and comparing two arms at the same CAP compares unequal SPENDS whenever their
ceiling-hit rates differ. `compute.py` re-reads the three frontier snapshots and reports both
axes side by side; `interp.py` holds the pure matched-calls comparison logic; `plot.py` draws
the calls-axis figure beside the cap-axis one.
"""
