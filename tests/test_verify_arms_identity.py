"""`pi verify arms` must not call an input-identical arm a dead code path.

FOUND BY THE CANARY (66 units, $0.0609, 22 arms on musique at seed 7). Four cells failed
and three were false positives:

  musique/parallel_replay:             0 of 12 calls reached the provider (all cache hits)
  musique/pare_plus_inquirer_prompted: 0 of 15 calls reached the provider (all cache hits)

Both are input-identical to `inquirer_prompted` BY CONSTRUCTION -- verified on task
3hop1__194721_295697_126906 at seed 7, where all three emit the byte-identical question
'Who is the performer of the song or work titled "The Galaxy Kings"?'. `parallel_replay`
replays a recorded question list; `pare_plus_inquirer_prompted` is the same PromptedInquirer
with the same template. Identical requests are identical cache keys, so 100% cache hits are
CORRECT.

The check's remedy ("Use an unused --seeds value") is also wrong for them: no seed helps,
because the collision is with another arm in the SAME sweep, not with a warm cache from an
earlier one.

For these arms the live-call check is not merely noise -- it is backwards. The property
worth asserting is that the requests ARE duplicates, which is a determinism check on the
harness. That is the same reasoning that re-pointed kill switch S5 off parallel_replay.
"""

from __future__ import annotations


def test_the_duplicating_arms_declare_it() -> None:
    from pinq_expt.arms import ARMS

    for arm_id in ("parallel_replay", "pare_plus_inquirer_prompted"):
        assert ARMS[arm_id].input_identical_by_design, arm_id


def test_an_ordinary_arm_does_not_declare_it() -> None:
    """Otherwise the exemption would silently cover the arms it must not."""
    from pinq_expt.arms import ARMS

    for arm_id in ("inquirer_prompted", "drafter_only", "self_ask", "random_q"):
        assert not ARMS[arm_id].input_identical_by_design, arm_id


def test_random_q_is_NOT_exempt() -> None:
    """random_q draws OTHER tasks' questions, so its requests are genuinely new and a
    100%-cache-hit cell there is a real defect."""
    from pinq_expt.arms import ARMS

    assert not ARMS["random_q"].input_identical_by_design


def test_the_live_call_check_skips_only_the_declared_arms() -> None:
    from pi_run.cli import _live_call_failure

    ordinary = _live_call_failure(arm_id="drafter_only", exempt=False, n_live=0, n_all=12)
    assert ordinary and "reached the provider" in ordinary

    exempt = _live_call_failure(arm_id="parallel_replay", exempt=True, n_live=0, n_all=12)
    assert exempt is None, "an input-identical arm must not be reported as a dead path"


def test_a_live_call_is_fine_for_both() -> None:
    from pi_run.cli import _live_call_failure

    assert _live_call_failure(arm_id="drafter_only", exempt=False, n_live=3, n_all=12) is None
    assert _live_call_failure(arm_id="parallel_replay", exempt=True, n_live=3, n_all=12) is None


def test_zero_total_calls_is_reported_even_for_an_exempt_arm() -> None:
    """0 of 0 means the compaction lost the calls table, not that the arm duplicated."""
    from pi_run.cli import _live_call_failure

    msg = _live_call_failure(arm_id="parallel_replay", exempt=True, n_live=0, n_all=0)
    assert msg and ("no calls" in msg.lower() or "0 of 0" in msg)


def test_the_fake_arms_are_not_in_the_musique_canary() -> None:
    """`ChainInquirer` follows the SYNTHETIC suite's dependency chain by construction, so
    fake_chain and fake_depth1 emit 0 asks on musique -- measured, 3 runs each. They are
    synth stand-ins; canarying them on musique proves nothing and reports a false defect."""
    import yaml

    g = yaml.safe_load(open("conf/grids/tier0_canary.yaml"))
    assert "musique" in g["suites"]
    for arm in ("fake_chain", "fake_depth1"):
        assert arm not in g["arms"], f"{arm} cannot ask on musique"


# --------------------------------------------------------- a suite mismatch is not a defect


def test_the_synth_stand_ins_declare_their_home_suite() -> None:
    """`expects_asks` is a property of (arm, SUITE), not of the arm alone.

    ChainInquirer follows the synthetic suite's dependency chain BY CONSTRUCTION, so
    fake_chain cannot ask on musique -- measured on the canary: 3 runs, n_asks 0.0,
    n_turns 0.0, tok_total 0. Reporting that as "expected to ask but did not" describes a
    suite mismatch as a dead policy.
    """
    from pinq_expt.arms import ARMS

    assert ARMS["fake_chain"].home_suites == ("synth",)
    assert ARMS["fake_depth1"].home_suites == ("synth",)


def test_a_general_arm_declares_no_home_suite() -> None:
    """Empty means "meaningful anywhere", which is the default and must stay the default."""
    from pinq_expt.arms import ARMS

    for arm_id in ("inquirer_prompted", "drafter_only", "self_ask", "parallel_replay"):
        assert ARMS[arm_id].home_suites == (), arm_id


def test_an_off_suite_cell_is_skipped_not_failed() -> None:
    from pi_run.cli import _asks_failure

    off = _asks_failure(arm_id="fake_chain", home=("synth",), suite="musique", mean_asks=0.0)
    assert off is None, "an off-suite cell must not be reported as a defect"


def test_an_on_suite_zero_ask_cell_still_fails() -> None:
    """The exemption must not swallow the real finding it was derived from."""
    from pi_run.cli import _asks_failure

    on = _asks_failure(arm_id="fake_chain", home=("synth",), suite="synth", mean_asks=0.0)
    assert on and "expected to ask" in on

    general = _asks_failure(arm_id="self_ask", home=(), suite="musique", mean_asks=0.0)
    assert general and "expected to ask" in general


def test_a_nonzero_ask_cell_passes_everywhere() -> None:
    from pi_run.cli import _asks_failure

    assert _asks_failure(arm_id="self_ask", home=(), suite="musique", mean_asks=1.0) is None
    assert _asks_failure(arm_id="fake_chain", home=("synth",), suite="synth", mean_asks=2.0) is None
