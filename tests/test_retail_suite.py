"""The tau2 RETAIL suite, view side.

WHAT IT SHARES WITH BANKING AND WHAT IT CANNOT
    Both are driven through the Orchestrator, and for the same reason: `view()` REFUSES,
    because the only task text available is the customer's roleplay script. Handing that to
    the agent inverts the role -- observed live on banking, the agent answered AS the customer
    -- and hands over every user-private fact for free, which collapses the partition that
    bounds what any autonomous inquirer could reach.

    What retail cannot share is the retriever. Banking has `search_documents` over 698
    knowledge documents; retail has 15 typed, key-addressed tools over a relational DB and no
    free-text search at all. Evidence is therefore reconstructed from what tool calls ASKED
    FOR (`retail_units.uids_from_env_calls`), not from a retrieval channel.

WHY THE CORPUS HASH IS TESTED HERE AND NOT ONLY IN THE BUILDER
    Gold is written against the DB this suite reads. If the two disagree the mismatch is
    invisible in every downstream number, because an unmatched uid reads as "the policy
    retrieved nothing relevant" rather than as an error.
"""

from __future__ import annotations

import pytest

from pinq_adapters.tau2._probe import available
from pinq_adapters.tau2.retail_suite import RETAIL_DOMAIN, Tau2RetailSuite

tau2_only = pytest.mark.skipif(not available()[0], reason=available()[1])


def test_suite_and_corpus_ids_match_the_gold_builder() -> None:
    """Registered as its OWN suite so `tau2` can stay in EVAL_ONLY_SUITES while this one is
    trainable. Compared against the builder's constants rather than string literals."""
    from pi_eval.build.retail_build import CORPUS_ID, SUITE

    assert Tau2RetailSuite.suite_id == SUITE == "tau2_retail"
    assert Tau2RetailSuite.corpus_id == CORPUS_ID == "tau2_retail"
    assert RETAIL_DOMAIN == "retail"


@tau2_only
def test_task_ids_are_the_full_retail_set() -> None:
    s = Tau2RetailSuite()
    ids = s.task_ids()
    assert len(ids) == 114, f"expected 114 retail tasks, got {len(ids)}"
    assert len(set(ids)) == len(ids)


@tau2_only
def test_corpus_hash_agrees_with_gold() -> None:
    from pi_eval.build.retail_build import corpus_hash_of

    s = Tau2RetailSuite()
    assert s.corpus_hash == corpus_hash_of(s.db)


@tau2_only
def test_view_refuses_because_the_only_text_is_a_roleplay_script() -> None:
    """Same refusal banking makes, for the same two reasons. A loud failure beats a wrong
    number nobody can see is wrong."""
    from pinq_adapters.tau2.suite import Tau2NeedsOrchestrator

    s = Tau2RetailSuite()
    with pytest.raises(Tau2NeedsOrchestrator):
        s.view(s.task_ids()[0])


@tau2_only
def test_the_debug_escape_hatch_exists_and_is_marked() -> None:
    """`allow_user_script=True` is for debugging only; runs made that way must be excluded
    from reported tables exactly as gold-exposed runs are."""
    s = Tau2RetailSuite(allow_user_script=True)
    v = s.view(s.task_ids()[0])
    assert v.question


@tau2_only
def test_tool_schemas_exclude_nothing_retail_does_not_advertise() -> None:
    s = Tau2RetailSuite()
    schemas = s.tool_schemas(s.task_ids()[0])
    # FLAT, matching Tau2Suite. `tau2_runner` reads `t["name"]` at the top level; the nested
    # OpenAI form killed a live dialogue with KeyError: 'name' after the customer had already
    # spoken. This test asserts the SHAPE the runner consumes, not merely that names exist.
    assert all(set(t) >= {"name", "description", "parameters"} for t in schemas)
    names = [t["name"] for t in schemas]
    assert "get_order_details" in names
    assert names == sorted(names), "advertised order must be stable between runs"


@tau2_only
def test_environment_is_memoized_per_task() -> None:
    """Rebuilding the env per turn would reset the DB mid-episode, and 176 of 550 required
    retail calls MUTATE it."""
    s = Tau2RetailSuite()
    tid = s.task_ids()[0]
    assert s.environment(tid) is s.environment(tid)


@tau2_only
def test_two_tasks_do_not_share_one_environment() -> None:
    """The other half of the same concern: a mutation from task A must not be visible to
    task B."""
    s = Tau2RetailSuite()
    a, b = s.task_ids()[0], s.task_ids()[1]
    assert s.environment(a) is not s.environment(b)


def test_template_id_groups_nothing_by_default() -> None:
    """Retail task ids are bare integers with no composition to recover, so there is no
    near-duplicate structure to bind. Returning None degrades to per-task splitting, which is
    correct here -- inventing a grouping would silently cluster unrelated tasks."""
    assert Tau2RetailSuite.template_id_for("0") is None


@tau2_only
def test_both_tau2_suites_advertise_tools_in_the_SAME_shape() -> None:
    """The contract `tau2_runner` actually consumes, asserted across both domains.

    The runner builds its allowed-tool set with `{t["name"] for t in suite.tool_schemas(tid)}`.
    Retail first returned the nested OpenAI form, so a live dialogue reached the customer's
    opening message -- "I'd like to exchange a couple of items from my recent order #W2378156"
    -- and then died on `KeyError: 'name'`, having already spent real money.

    Testing the two suites against EACH OTHER rather than against a literal is what keeps them
    from drifting apart again: whichever shape the runner wants, both must speak it.
    """
    from pinq_adapters.tau2.suite import Tau2Suite

    retail = Tau2RetailSuite()
    banking = Tau2Suite()
    r = retail.tool_schemas(retail.task_ids()[0])
    b = banking.tool_schemas(banking.task_ids()[0])
    assert r and b
    assert {frozenset(x) for x in r} == {frozenset(x) for x in b}, (
        "the two tau2 suites advertise tools with different keys; the runner reads one shape"
    )
