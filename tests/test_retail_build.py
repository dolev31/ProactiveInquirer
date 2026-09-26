"""Gold graphs for tau2 retail. GOLD SIDE.

WHY RETAIL NEEDS ITS OWN BUILDER RATHER THAN A DOMAIN FLAG ON tau2_build
    `tau2_build` keys entirely off `Task.required_documents`, the benchmark's own answer key.
    MEASURED: retail carries that field on 0 of its 114 tasks (banking: 97 of 97). Retail's
    gold is an ACTION SEQUENCE -- typed tool calls with arguments -- and its corpus is a
    relational DB, not a document set.

WHAT MAKES A NODE
    Every argument value the gold actions require is a fact the agent had to obtain. Where it
    could have come from is measurable, and the three answers are three different node kinds:

        in `user_scenario`  -> the customer holds it; discoverability="user_private"; a ROOT
        in `db.json`        -> retrievable; discoverability="kb"; gold_ev_uids = the records
        neither             -> unresolved, COUNTED not dropped (3.5% of values)

WHERE DEPTH COMES FROM
    To read a DB record you need its key, and that key is itself a value the agent had to
    obtain. So the prerequisite edge is `key(record) -> value-inside-record`, which is a real
    data dependency rather than a re-labelling of action order. Measured over all 112 tasks
    with actions: 1,015 nodes, 70.4% with non-empty gold_ev_uids, 72.3% of tasks reaching
    depth >= 1, depths {0:603, 1:205, 2:153, 3:54}.

    Those two numbers are the kill conditions from the build scope. A builder that emits nodes
    without evidence uids reproduces the DRGym failure exactly -- 0 edges and empty uids on all
    16,156 nodes, PoolUnavailable to the exporter -- so `build_graphs` refuses a graph with no
    required node rather than writing one.
"""

from __future__ import annotations

import pytest

from pi_eval.build.retail_build import (
    CORPUS_ID,
    MIN_VALUE_LEN,
    build_graphs,
    record_doc_id,
    record_text,
    record_uid,
)
from pinq_adapters.tau2._probe import available as _tau2_available

DB = {
    "users": {
        "user_77": {"user_id": "user_77", "email": "yusuf@example.com", "zip": "19122"},
    },
    "orders": {
        "#W100": {"order_id": "#W100", "user_id": "user_77", "item_ids": ["item_5150"]},
    },
    "products": {
        "prod_9": {"product_id": "prod_9", "name": "Desk Lamp", "item_ids": ["item_5150"]},
    },
}


def _task(tid="0", actions=(), scenario=None):
    return {
        "id": tid,
        "user_scenario": scenario or {"persona": "", "instructions": ""},
        "evaluation_criteria": {"actions": list(actions)},
    }


def _act(name, **kw):
    return {"name": name, "arguments": dict(kw)}


# ------------------------------------------------------------------ the uid convention


def test_record_uid_is_derived_from_table_and_id_only() -> None:
    """The uid must come from what a tool call ASKED FOR, never from what it returned.

    A tau2 tool result is a customer record and the runner stores only a `result_digest`. Since
    retail's tools are key-addressed -- 458 of 550 required calls name a record in their
    arguments -- the identity is available from the request, so no payload is ever persisted
    to build one.
    """
    assert record_doc_id("orders", "#W100") == "orders:#W100"
    a = record_uid("orders", "#W100", DB["orders"]["#W100"])
    b = record_uid("orders", "#W100", DB["orders"]["#W100"])
    assert a == b and len(a) == 64


def test_record_text_is_canonical_so_key_order_cannot_move_the_uid() -> None:
    """Two serialisations of one record must not mint two uids: the corpus hash and every
    memoisation key downstream would fork on dict ordering."""
    x = {"order_id": "#W100", "user_id": "user_77"}
    y = {"user_id": "user_77", "order_id": "#W100"}
    assert record_text(x) == record_text(y)


# ------------------------------------------------------------------ node kinds


def test_a_value_the_customer_told_us_is_a_user_private_root() -> None:
    t = _task(
        actions=[_act("find_user_id_by_name_zip", zip="19122")],
        scenario={"instructions": "my zip is 19122"},
    )
    g = build_graphs(DB, [t])[0]
    n = next(n for n in g.gold_nodes if "19122" in n.gold_text)
    assert n.gold_discoverability == "user_private"
    assert n.gold_depth == 0, "the customer holding it is what makes it a root"


def test_a_value_carries_exactly_the_ONE_record_that_provided_it() -> None:
    """Evidence is the record the agent READ to obtain the value, not every record the value
    appears in.

    Containment was the first rule and it was unusable: `gold_ev_uids` is an AND, so listing
    every record containing a value demands the policy retrieve all of them. A required value
    sits in a median of 3 records but a MEAN of 68.5, and one appears in all 1,500 -- which put
    37,136 required uids across the graph-bearing tasks and drove the ORACLE ceiling to 1.2%.
    (That note said "111 tasks"; the population is 112 of 114, asserted below.)
    """
    t = _task(
        actions=[
            _act("get_order_details", order_id="#W100"),
            _act("get_item_details", item_id="item_5150"),
        ]
    )
    g = build_graphs(DB, [t])[0]
    n = next(n for n in g.gold_nodes if "item_5150" in n.gold_text)
    assert n.gold_discoverability == "kb"
    assert len(n.gold_ev_uids) == 1, f"evidence must be ONE record, got {len(n.gold_ev_uids)}"
    assert n.gold_ev_uids[0] == record_uid("orders", "#W100", DB["orders"]["#W100"])


def test_a_first_action_key_has_no_provider_and_is_not_kb() -> None:
    """Nothing was read before action 0, so its arguments cannot have been retrieved. The
    first lookup key comes from the customer -- which is exactly the population Option B
    exists to reach -- or, with no scenario text to confirm it, is honestly 'unknown'."""
    g = build_graphs(DB, [_task(actions=[_act("get_order_details", order_id="#W100")])])[0]
    n = next(n for n in g.gold_nodes if "#W100" in n.gold_text)
    assert n.gold_discoverability in {"user_private", "unknown"}
    assert n.gold_ev_uids == ()


def test_an_unresolved_value_is_kept_and_labelled_not_dropped() -> None:
    """3.5% of real values resolve to neither source. Dropping them would silently shrink
    every denominator; they are emitted with discoverability='unknown'."""
    t = _task(actions=[_act("get_order_details", order_id="#NOPE999")])
    g = build_graphs(DB, [t])[0]
    n = next(n for n in g.gold_nodes if "#NOPE999" in n.gold_text)
    assert n.gold_discoverability == "unknown"
    assert n.gold_ev_uids == ()


def test_short_values_are_skipped() -> None:
    """A two-character argument matches half the database by containment."""
    t = _task(actions=[_act("get_order_details", order_id="#W100", reason="no")])
    g = build_graphs(DB, [t])[0]
    assert all("no" != n.gold_text for n in g.gold_nodes)
    assert MIN_VALUE_LEN >= 3


# ------------------------------------------------------------------ depth


def test_a_value_reachable_only_through_an_earlier_key_gets_a_prerequisite_edge() -> None:
    """item_5150 lives inside order #W100. You cannot name it before you have read that
    order, so #W100 is its prerequisite and it sits at depth 1."""
    t = _task(
        actions=[
            _act("get_order_details", order_id="#W100"),
            _act("get_item_details", item_id="item_5150"),
        ]
    )
    g = build_graphs(DB, [t])[0]
    edges = [
        (e.gold_src_node_id, e.gold_dst_node_id)
        for e in g.gold_edges
        if e.gold_edge_kind == "prerequisite"
    ]
    assert edges, "no prerequisite edge: retail would have no latent population"
    item = next(n for n in g.gold_nodes if "item_5150" in n.gold_text)
    assert item.gold_depth == 1


def test_order_alone_does_not_create_an_edge() -> None:
    """The edge must be a DATA dependency, not a restatement of action order. Two independent
    lookups in sequence are both roots."""
    t = _task(
        actions=[
            _act("get_order_details", order_id="#W100"),
            _act("get_product_details", product_id="prod_9"),
        ]
    )
    g = build_graphs(DB, [t])[0]
    depths = {n.gold_text: n.gold_depth for n in g.gold_nodes}
    assert depths.get("prod_9") == 0, f"prod_9 is independently addressable: {depths}"


# ------------------------------------------------------------------ refusals


def test_a_task_with_no_actions_yields_no_graph_rather_than_an_empty_one() -> None:
    """2 of 114 retail tasks carry no actions. An empty graph would be scored as a task the
    policy failed rather than one there was nothing to measure."""
    assert build_graphs(DB, [_task(actions=[])]) == []


@pytest.mark.skipif(not _tau2_available()[0], reason=_tau2_available()[1])
def test_the_two_tasks_with_no_gold_are_named_and_are_upstreams_choice() -> None:
    """ASSERTED, NOT ASSUMED, because the count is a denominator.

    Retail ships 114 tasks and 112 gold graphs, and WHICH two are missing decides whether
    this is a build bug. Both are upstream's own `"actions": []`, and both are tasks whose
    correct outcome is that the DB does NOT change:

      * 24 -- the customer asks to cancel a grill, then regrets it and keeps it, and then asks
        which two t-shirts an order contains and what they are made of. The gold is an
        NL_ASSERTION ("polyester and cotton") over a communicated fact.
      * 57 -- the customer asks when an order arrives and wants to cancel one item from it if
        it has not shipped; it has, so nothing is cancelled. No assertions either.

    Both carry `reward_basis: [DB, NL_ASSERTION]`, so their DB half is "unchanged". This
    builder's nodes are the argument VALUES the gold actions required, so a task with no
    actions has no nodes -- there is nothing to be discovered, and a graph asserting that is
    not an easier task, it is a different one. A DELIBERATE EXCLUSION, and the same two tasks
    `pi verify tau2 --replay-gold --suite tau2_retail` reports as skipped for the same reason.
    """
    import json as _json

    from pinq_adapters.tau2._probe import domain_data_dir

    d = domain_data_dir("retail")
    tasks = _json.loads((d / "tasks.json").read_text())
    db = _json.loads((d / "db.json").read_text())
    assert len(tasks) == 114, f"upstream shipped {len(tasks)} retail tasks, not 114"

    graphs = build_graphs(db, tasks)
    built = {str(g.gold_task_key) for g in graphs}
    missing = sorted({str(t["id"]) for t in tasks} - built, key=int)
    assert len(graphs) == 112
    assert missing == ["24", "57"]
    for tid in missing:
        rec = next(t for t in tasks if str(t["id"]) == tid)
        assert (rec["evaluation_criteria"].get("actions") or []) == [], (
            f"task {tid} HAS gold actions and still produced no graph -- that would be a "
            "build bug rather than an upstream exclusion"
        )


def test_every_emitted_graph_has_at_least_one_required_node() -> None:
    for g in build_graphs(DB, [_task(actions=[_act("get_order_details", order_id="#W100")])]):
        assert any(n.gold_partition == "required" for n in g.gold_nodes)


def test_suite_and_corpus_ids_are_distinct_from_banking() -> None:
    """`tau2` stays in EVAL_ONLY_SUITES; this must be a DIFFERENT suite_id or registering it
    as trainable would make the eval suite trainable too."""
    g = build_graphs(DB, [_task(actions=[_act("get_order_details", order_id="#W100")])])[0]
    assert g.gold_suite == "tau2_retail"
    assert CORPUS_ID == "tau2_retail" and "banking" not in CORPUS_ID


# ------------------------------------------------------------------ the driver


def test_corpus_hash_is_order_independent() -> None:
    """A DB re-serialised with its tables or rows in another order is the SAME corpus. If it
    hashed differently, every run's identity would fork on a dict ordering nobody chose."""
    from pi_eval.build.retail_build import corpus_hash_of

    flipped = {k: DB[k] for k in reversed(list(DB))}
    assert corpus_hash_of(DB) == corpus_hash_of(flipped)


def test_build_writes_gold_but_no_corpus_tree(tmp_path) -> None:
    """SELF-SOURCED, exactly like banking. `tau2_build` writes no corpus for the stated
    reason: minting a second copy of upstream's bytes gives them two corpus_hashes, and the
    adapter's hash and gold's then disagree -- which is invisible downstream, because a uid
    mismatch reads as "the policy retrieved nothing relevant" rather than as an error.
    """
    from pi_eval.build.retail_build import build

    res = build(
        root=tmp_path, db=DB, tasks=[_task(actions=[_act("get_order_details", order_id="#W100")])]
    )
    assert (tmp_path / "data" / "gold" / "graphs" / "tau2_retail" / "v1.jsonl").exists()
    assert not (tmp_path / "data" / "corpora" / "tau2_retail").exists()
    assert res.n_tasks == 1
    assert res.corpus_hash


def test_written_graphs_round_trip_through_the_firewalled_reader(tmp_path, monkeypatch) -> None:
    """The builder and `load_graphs` must agree. A graph that cannot be read back is a graph
    `score()` skips as "no gold for task" -- silently, and as a zero rather than an error.

    Goes through `load_graphs`, which refuses without PI_GOLD_ROOT, rather than parsing the
    file directly: that raise IS the firewall, and a test that bypassed it would not be
    testing the path the scorer takes.
    """
    from pi_eval.build.retail_build import build
    from pi_eval.gold import load_graphs

    build(
        root=tmp_path,
        db=DB,
        tasks=[
            _task(
                "0",
                [
                    _act("get_order_details", order_id="#W100"),
                    _act("get_item_details", item_id="item_5150"),
                ],
            ),
        ],
    )
    monkeypatch.setenv("PI_GOLD_ROOT", str(tmp_path / "data" / "gold"))
    back = load_graphs("tau2_retail", "v1")
    assert list(back) == ["0"]
    g = back["0"]
    assert g.gold_suite == "tau2_retail"
    assert any(n.gold_ev_uids for n in g.gold_nodes)
    assert any(e.gold_edge_kind == "prerequisite" for e in g.gold_edges)
    assert any((n.gold_depth or 0) >= 1 for n in g.gold_nodes)


def test_the_canary_is_minted_and_registered_even_though_it_has_no_carrier(tmp_path) -> None:
    """Layer 4 is ARMED on retail and DETECTS NOTHING, and both halves are deliberate.

    `write_graphs` mints a nonce per graph, so the registry has something to scan for. But the
    carrier is `gold_answer`, and retail has none (the benchmark's reward is over an executed
    action sequence), so the nonce never rides inside a string a leak would carry. Same gap as
    tau2 and drgym. This test pins the half that IS true, so that a future change which starts
    emitting a gold_answer is forced to decide about the carrier rather than acquire one by
    accident.
    """
    import json

    from pi_eval.build.retail_build import build

    build(
        root=tmp_path, db=DB, tasks=[_task(actions=[_act("get_order_details", order_id="#W100")])]
    )
    rows = [
        json.loads(x)
        for x in (tmp_path / "data" / "gold" / "graphs" / "tau2_retail" / "v1.jsonl")
        .read_text()
        .splitlines()
    ]
    assert all(r["gold_canary"] for r in rows), "no nonce: layer 4 has nothing to scan for"
    assert all(r["gold_corpus_hash"] for r in rows), "gold that cannot name its corpus"
    assert all(r["gold_answer"] == "" for r in rows), (
        "retail grew a gold_answer; decide about the layer-4 carrier before shipping it"
    )


def test_user_held_reads_the_authored_partition_not_the_roleplay_text() -> None:
    """Retail states which facts the customer holds; banking has to infer it. Using the
    authored field makes the label bench_author rather than a threshold choice.

    `persona` and `task_instructions` are excluded on purpose -- they are stage direction, and
    matching against them manufactures user-held values out of roleplay.
    """
    from pi_eval.build.retail_build import _user_held

    t = {
        "user_scenario": {
            "persona": "You are ZIPPY9999 and impatient",
            "instructions": {
                "known_info": "You are in zip code 19122.",
                "reason_for_call": "About order #W100.",
                "task_instructions": "You are detail-oriented.",
                "unknown_info": "You do not remember your email.",
            },
        }
    }
    held = _user_held(t)
    assert "19122" in held
    assert "w100" in held
    assert "zippy9999" not in held, "persona is roleplay direction, not a fact the user holds"
    assert "detailoriented" not in held


def test_user_held_survives_a_scenario_with_no_structured_instructions() -> None:
    """Not every tau2 domain ships the dict form; degrade to the raw text rather than
    silently returning nothing, which would class every value as 'unknown'."""
    from pi_eval.build.retail_build import _user_held

    assert "19122" in _user_held({"user_scenario": {"instructions": "zip 19122"}})
    assert _user_held({}) == ""
