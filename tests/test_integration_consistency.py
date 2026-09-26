"""Cross-cutting consistency between parts that are individually correct.

Each module below is tested in isolation elsewhere. These are the seams: a preregistered arm
that nobody implemented, a grid that names a nonexistent arm, or a primary endpoint whose
metric no scorer emits, would all pass every unit test and then fail silently at the point
where a table is generated — which is the worst possible moment to find out.
"""

import pytest

from pi_eval.prereg import default_stage1
from pi_run.grids import load_all
from pinq_expt import arms as arm_mod


def _arm_ids() -> set[str]:
    if hasattr(arm_mod, "arm_ids"):
        return set(arm_mod.arm_ids())
    return set(getattr(arm_mod, "ARMS", {}))


def test_every_preregistered_arm_is_implemented():
    """A preregistered arm that does not exist is a promise the paper cannot keep."""
    missing = set(default_stage1().arms) - _arm_ids()
    assert missing == set(), f"preregistered but unimplemented: {sorted(missing)}"


def test_every_grid_names_only_real_arms():
    """A typo in a grid would otherwise surface as a silently smaller sweep.

    THIS USED TO EXEMPT `pare_*`, on the theory that PARE arms were composed at the adapter
    boundary rather than in the shared table. They were not composed anywhere: `pare_transfer`
    named three ids that existed only as string constants in `pinq_adapters.pare.suite`, the
    grid could not be run, and the exemption is what stopped anyone finding out. The three
    arms now live in `pinq_expt.arms.ARMS` like every other arm, so no id gets a pass here.
    """
    known = _arm_ids()
    for g in load_all():
        unknown = set(g.arms) - known
        assert unknown == set(), f"grid {g.name} names unknown arms: {sorted(unknown)}"


def test_the_pare_transfer_grid_moves_exactly_one_slot():
    """Grid E's whole claim rests on the executor being held fixed.

    If the three PARE arms differed in their drafter or their answerer, a win could be the
    actuator rather than the inquiry stage, and the transfer question would have no answer.
    The kill switch is the one deliberate exception: it moves the DRAFTER, because "the
    effect is length" is exactly the hypothesis it exists to test.
    """
    grid = next(g for g in load_all() if g.name == "pare_transfer")
    arms = {a: arm_mod.get(a) for a in grid.arms}
    assert set(arms) == {"pare_baseline", "pare_plus_inquirer_prompted", "pare_plus_verbosity"}

    held_fixed = {
        a: (arms[a].drafter, arms[a].answerer) for a in arms if a != "pare_plus_verbosity"
    }
    assert len(set(held_fixed.values())) == 1, held_fixed

    answerers = {arm.answerer for arm in arms.values()}
    assert len(answerers) == 1, "the frozen answerer is shared by every PARE arm"
    assert {arm.kind for arm in arms.values()} == {"control", "treatment", "killswitch"}
    # Budget parity: the transfer arms carry the same cap as every other confirmatory arm,
    # so "PARE improved" can never reduce to "PARE was given more retrieval calls".
    assert {arm.budget_cap for arm in arms.values()} == {arm_mod.DEFAULT_CAP}


def test_every_kill_switch_arm_exists_and_is_in_the_pilot_grid():
    s1 = default_stage1()
    known = _arm_ids()
    pilot = next(g for g in load_all() if g.name == "tier1_pilot")
    for arm in s1.kill_switches:
        assert arm in known, f"kill switch {arm} is not implemented"
        assert arm in pilot.arms, f"kill switch {arm} is not in the pilot: it would fire too late"


def test_primary_contrasts_reference_implemented_arms():
    known = _arm_ids()
    for e in default_stage1().primary:
        for arm in e.contrast:
            assert arm in known, f"primary endpoint {e.suite_id} references missing arm {arm}"


def test_confirmatory_suites_all_have_a_grid():
    """An endpoint with no grid is an endpoint that will never produce a number."""
    grid_suites = {s for g in load_all() for s in g.suites}
    for e in default_stage1().primary:
        assert e.suite_id in grid_suites, f"no grid runs {e.suite_id}"


def test_oracle_arms_never_appear_in_a_confirmatory_grid():
    """gold_evidence / oracle_vreq read gold. The aggregator's exclusion is the LAST line of
    defence; keeping them out of the grid is the first."""
    gold_arms = {"gold_evidence", "oracle_vreq"}
    for g in load_all():
        if g.exploratory or g.pilot:
            continue
        assert not (gold_arms & set(g.arms)), f"{g.name} mixes oracle arms into a scored grid"


@pytest.mark.parametrize(
    "arm",
    sorted({"verbosity", "self_inquire", "inquirer_noevidence", "parallel_replay", "random_q"}),
)
def test_each_control_that_can_kill_the_paper_is_implemented(arm):
    assert arm in _arm_ids()
