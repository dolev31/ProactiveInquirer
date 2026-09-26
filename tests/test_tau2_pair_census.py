"""The beliefs `scripts/tau2_trained/pair_census.py` encodes, each with a way to be wrong.

Every test here fails if the corresponding line of `artifacts/tau2_trained_20260918/RESULT.md`
is wrong about the code, not merely if the code changes shape.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

_spec = importlib.util.spec_from_file_location(
    "tau2_pair_census", ROOT / "scripts" / "tau2_trained" / "pair_census.py"
)
assert _spec is not None and _spec.loader is not None
pc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pc)


def _rec(**kw):
    base = {
        "run_id": "r",
        "arm_id": "inquirer_prompted",
        "seed": 0,
        "task_id": "1",
        "split": "train",
        "trace": "T",
        "k": 4,
        "model_pin_hash": "pin",
        "budget_cap": 16,
        "code_version": "c",
        "dirty": False,
        "prompt_hashes": {
            "inquirer_prompted": "tmpl",
            "fragment_user_channel_placebo": "placebo",
        },
        "view_question": "the customer said this",
        "action_kind": "ask",
        "target": "kb",
        "question": "what is the order id",
        "subset_hash_before": "",
        "db_reward": 0.0,
        "tau_reward": 0.0,
        "followups": 3,
        "n_turns": 4,
        "n_asks": 4,
        "usd_billed": 0.0,
        "usage_usd": 0.4,
        "user_sim_usd": 0.001,
    }
    base.update(kw)
    return base


# --------------------------------------------------------------------------- the ladder


def test_db_reward_outranks_fewer_followups() -> None:
    """The declared primary key wins even when the loser looks cheaper on every other key."""
    win = _rec(run_id="win", db_reward=1.0, followups=9, n_turns=9)
    lose = _rec(run_id="lose", db_reward=0.0, followups=1, n_turns=1)
    assert pc.decide(win, lose) == ("win", "db_reward")
    assert pc.decide(lose, win) == ("win", "db_reward")


def test_followups_break_a_reward_tie_and_turns_break_a_followup_tie() -> None:
    a = _rec(run_id="a", db_reward=1.0, followups=1, n_turns=7)
    b = _rec(run_id="b", db_reward=1.0, followups=4, n_turns=2)
    assert pc.decide(a, b) == ("a", "followups")

    c = _rec(run_id="c", db_reward=1.0, followups=2, n_turns=2)
    d = _rec(run_id="d", db_reward=1.0, followups=2, n_turns=6)
    assert pc.decide(c, d) == ("c", "quickest")


def test_a_pair_that_ties_on_every_key_is_undecided_not_a_coin_flip() -> None:
    a = _rec(run_id="a")
    b = _rec(run_id="b")
    assert pc.decide(a, b) == ("", "")


def test_the_ladder_reads_db_reward_and_not_the_gated_tau_reward_copy() -> None:
    """`tau_reward` is written only when the task's reward_basis names DB.

    Ordering on it would drop every task graded on another basis, silently, as a tie.
    """
    a = _rec(run_id="a", db_reward=1.0, tau_reward=None)
    b = _rec(run_id="b", db_reward=0.0, tau_reward=None)
    assert pc.decide(a, b) == ("a", "db_reward")


def test_a_missing_reward_is_undecidable_rather_than_zero() -> None:
    a = _rec(run_id="a", db_reward=None)
    b = _rec(run_id="b", db_reward=0.0)
    assert pc.decide(a, b) == ("", "")


# --------------------------------------------------------------- what "same state" means


def test_the_two_user_channel_arms_do_not_share_a_state() -> None:
    """THE BELIEF THIS ENCODES: `target='user'` vs `target='kb'` cannot be a same-state pair.

    `may_ask_user` swaps a prompt FRAGMENT, so the two arms are shown different prompts. If
    `state_key` ever grouped them together, an export would pair two different prompts and
    `rung2_dpo.train.assert_same_state` would be the only thing left to catch it.
    """
    kb = _rec(arm_id="inquirer_prompted")
    user = _rec(
        arm_id="inquirer_may_ask_user",
        target="user",
        prompt_hashes={"inquirer_prompted": "tmpl", "fragment_user_channel": "real"},
    )
    assert pc.state_key(kb, strict=False) == pc.state_key(user, strict=False)
    assert pc.state_key(kb, strict=True) != pc.state_key(user, strict=True)


def test_strict_also_separates_a_different_question_pin_or_cap() -> None:
    a = _rec()
    for field, value in (
        ("view_question", "a different opening"),
        ("model_pin_hash", "other"),
        ("budget_cap", 48),
    ):
        b = _rec(**{field: value})
        assert pc.state_key(a, strict=True) != pc.state_key(b, strict=True), field
        assert pc.state_key(a, strict=False) == pc.state_key(b, strict=False), field


def test_the_fork_key_uses_the_manifest_cut_depth_not_the_trace_alone() -> None:
    """Two cuts of one dialogue are two states, and a census keyed on the trace alone
    would pool them. `prefix_k` is NULL on every tau2 fork row, which is how this gets lost."""
    a = _rec(k=4)
    b = _rec(k=18)
    assert pc.state_key(a, strict=False) != pc.state_key(b, strict=False)


# --------------------------------------------------------------------------- the guards


def test_identical_and_over_length_pairs_are_counted_separately_not_pooled() -> None:
    same = [_rec(run_id="a"), _rec(run_id="b")]
    y = pc.pair_yield(same, strict=True)
    assert y["candidate_pairs"] == 1 and y["identical_question"] == 1
    assert y.get("kept", 0) == 0

    long_q = "x" * (pc.LEN_DELTA_MAX + 80)
    over = [
        _rec(run_id="a", question="short one", db_reward=1.0),
        _rec(run_id="b", question=long_q, db_reward=0.0),
    ]
    y2 = pc.pair_yield(over, strict=True)
    assert y2["candidate_pairs"] == 1 and y2["len_guard_dropped"] == 1
    assert y2.get("kept", 0) == 0


def test_a_decided_pair_is_kept_and_attributed_to_its_key() -> None:
    rows = [
        _rec(run_id="a", question="what is the order id", db_reward=1.0),
        _rec(run_id="b", question="what is the order no", db_reward=0.0),
    ]
    y = pc.pair_yield(rows, strict=True)
    assert y["kept"] == 1
    assert y["decided_by"] == {"db_reward": 1}
    assert y["task_ids_in_kept_pairs"] == 1


def test_the_vacuity_check_can_tell_a_live_guard_from_a_dead_one() -> None:
    """A vacuity report that always says "vacuous" is itself vacuous.

    Forced the other way: a first turn whose evidence is NOT empty must be counted as a state
    where the guard could have fired.
    """
    from pinq.ids import subset_hash

    empty = subset_hash(())
    dead = pc.guard_block([_rec(subset_hash_before=empty) for _ in range(3)])
    assert dead["guard_is_vacuous_on"] == 3 and dead["guard_could_have_fired_on"] == 0

    live = pc.guard_block(
        [_rec(subset_hash_before=empty), _rec(subset_hash_before=subset_hash(("call:x",)))]
    )
    assert live["guard_is_vacuous_on"] == 1 and live["guard_could_have_fired_on"] == 1


def test_a_template_absent_from_the_tree_is_reported_as_absent(tmp_path: Path) -> None:
    prompts = tmp_path / "src" / "pinq" / "prompts"
    prompts.mkdir(parents=True)
    (prompts / "inquirer_prompted.txt").write_text("the template that is in the tree")
    in_tree = pc._sha("the template that is in the tree")

    rows = [
        _rec(run_id="a", prompt_hashes={"inquirer_prompted": in_tree}),
        _rec(run_id="b", prompt_hashes={"inquirer_prompted": "0" * 64}),
        _rec(run_id="c", prompt_hashes={"inquirer_prompted": "0" * 64}),
    ]
    block = pc.template_block(rows, tmp_path)
    assert block["runs_on_a_template_absent_from_the_tree"] == 2
    present = [r for r in block["templates_pinned"] if r["in_working_tree"]]
    assert len(present) == 1 and present[0]["runs"] == 1


def test_the_cost_block_prefers_usage_usd_over_the_short_usd_billed_field() -> None:
    rows = [_rec(usd_billed=0.0, usage_usd=1.0), _rec(usd_billed=1.0, usage_usd=2.0)]
    c = pc.cost_block(rows)
    assert c["usd_billed_is_zero_on"] == 1
    assert c["total_usd_billed"] == 1.0
    assert c["total_usage_usd"] == 3.0
    assert c["usage_over_billed"] == 3.0


def test_the_census_report_is_json_serialisable() -> None:
    rows = [
        _rec(run_id="a", question="q one", db_reward=1.0),
        _rec(run_id="b", question="q two", db_reward=0.0),
    ]
    payload = {
        "user_channel": pc.channel_block(rows),
        "state_guard": pc.guard_block(rows),
        "cost": pc.cost_block(rows),
        "pairs": {"train": {"strict": pc.pair_yield(rows, strict=True)}},
    }
    assert json.loads(json.dumps(payload, sort_keys=True))


@pytest.mark.parametrize("strict", [True, False])
def test_a_single_run_at_a_state_yields_no_pair(strict: bool) -> None:
    y = pc.pair_yield([_rec()], strict=strict)
    assert y["states"] == 1
    assert y["states_with_two_or_more"] == 0
    assert y.get("candidate_pairs", 0) == 0
