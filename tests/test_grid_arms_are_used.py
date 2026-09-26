"""A kill-switch arm on a suite that cannot compute the kill-switch metric.

`verbosity` is in tier3_drgym. Its ONLY declared job is the verbosity kill switch, and kill
switches are decided on `report.KILLSWITCH_METRIC` = evidence_coverage, which
`prereg.NOT_COMPUTABLE` records as impossible on drgym: 0 of 16,156 gold nodes carry
gold_ev_uids, so |E n gold| has no denominator. No drgym endpoint names it either -- the
three are kpr_incremental vs self_inquire, vs drafter_only, and pooled rnr_resolve vs
drafter_only.

So on drgym it can only ever appear as a descriptive row in T2_arm_ladder. That is not
nothing -- `arm_ladder_table` iterates whatever arms are present, so every arm in a grid is
read SOMEWHERE -- which is why this test does NOT claim "unused". It claims the narrower and
checkable thing: an arm whose declared purpose is a kill switch is running on a suite where
that switch cannot be decided, at a measured $0.0305/unit x 100 tasks = $3.05.

The earlier version of this file asserted "every grid arm is read by some endpoint" and was
wrong: it flagged `checklist` and `query_expansion` on tier1_confirmatory, which are
legitimate descriptive ladder arms.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from pi_eval.prereg import KILL_SWITCHES, POOLED, all_endpoints, not_computable
from pi_eval.report import KILLSWITCH_METRIC


def _endpoint_arms(suite: str) -> set[str]:
    out: set[str] = set()
    for e in all_endpoints():
        if e.suite_id == suite or (e.suite_id == POOLED and suite in (e.pool_suites or ())):
            out.update(e.contrast)
    return out


def test_drgym_cannot_decide_a_kill_switch() -> None:
    """The premise. If evidence_coverage becomes computable there, revisit the arm list."""
    assert not_computable("drgym", KILLSWITCH_METRIC)


def test_no_grid_runs_a_killswitch_only_arm_where_the_switch_cannot_be_decided() -> None:
    for path in sorted(Path("conf/grids").glob("*.yaml")):
        g = yaml.safe_load(path.read_text()) or {}
        if g.get("exploratory"):
            continue  # a canary runs every arm on purpose: that IS its job
        for suite in g.get("suites") or ():
            if not not_computable(str(suite), KILLSWITCH_METRIC):
                continue  # the switch is decidable here; the arm earns its cost
            endpoint_arms = _endpoint_arms(str(suite))
            for arm in g.get("arms") or ():
                if arm in KILL_SWITCHES and arm not in endpoint_arms:
                    raise AssertionError(
                        f"{path.name} runs kill switch {arm!r} on {suite}, where "
                        f"{KILLSWITCH_METRIC} is not computable and no endpoint names it. "
                        "It can only appear as a descriptive ladder row."
                    )
