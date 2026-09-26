"""`pi annotate export`: samplers, the canary firewall, and the CLI that wires them together.

Fixtures build a REAL synth gold tree (`pi_eval.build.synth_build.build`) rather than faking
GoldGraph rows by hand, so the canary minted into `gold_answer` is the one the registry
actually knows about -- a hand-rolled nonce would never be able to prove the firewall test.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pi_eval.annotate import bundle_shape_errors
from pi_eval.build.synth_build import build as synth_build
from pi_run import cmd_annotate
from pi_run.cli import build_parser
from pi_run.cmd_annotate import (
    a1_node_projection,
    sample_a1_items,
    sample_a2_items,
    sample_a3_items,
    sample_a4_items,
)

pytest.importorskip("pyarrow")


# --------------------------------------------------------------------------- fixtures


def _turns(units, *, questions=None):
    from pinq.types import Evidence

    rows = []
    held: list = []
    for i, u in enumerate(units):
        before = Evidence.of(tuple(held)).subset_hash
        held.append(u)
        q = questions[i] if questions else f"Retrieve record {i}."
        rows.append(
            {
                "turn_idx": i,
                "action_kind": "ask",
                "question": q,
                "rationale": f"because {i}",
                "response_text": f"answer {i}",
                "retrieved_uids": [u.uid],
                "new_uids": [u.uid],
                "subset_hash_before": before,
            }
        )
    return rows


def _write_run(runs: Path, run_id: str, *, task: str, split: str, turns: list, arm: str):
    d = runs / run_id
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "suite_id": "synth",
                "task_id": task,
                "arm_id": arm,
                "split": split,
                "template_id": None,
            }
        )
    )
    (d / "turns.jsonl").write_text("\n".join(json.dumps(t) for t in turns))
    (d / "ledger.jsonl").write_text(
        json.dumps({"currency": "retrieval_calls", "cumulative": float(len(turns))})
    )
    (d / "status.json").write_text(
        json.dumps({"stop_reason": "policy_stop", "wall_ms": 5, "usage": {"tok_total": 50}})
    )
    return d


def _setup(tmp_path: Path, monkeypatch, *, n_tasks: int = 2):
    """A real synth gold tree, one trainable run, one dev- run, one gold-exposed run."""
    from pi_run.cmd_train import SuiteCache

    monkeypatch.setenv("PI_CANARY_SALT", "test-salt")
    monkeypatch.setenv("PI_GOLD_ROOT", str(tmp_path / "data" / "gold"))
    synth_build(n_tasks=n_tasks, n_facets=1, depth=2, root=tmp_path)

    cache = SuiteCache(tmp_path)
    suite = cache.suite("synth")
    tid = str(suite.task_ids()[0])
    units = list(suite.retriever(tid)._units)

    runs = tmp_path / "runs"
    keep = _write_run(
        runs, "keep0", task=tid, split="train", turns=_turns(units[:2]), arm="inquirer_prompted"
    )
    _write_run(
        runs,
        "dev-abc123",
        task=tid,
        split="train",
        turns=_turns(units[:1]),
        arm="inquirer_prompted",
    )
    _write_run(
        runs, "ceiling0", task=tid, split="train", turns=_turns(units[:1]), arm="gold_evidence"
    )
    return cache, suite, tid, units, runs, keep


def _write_pairs(tmp_path: Path, *, tid: str, run_id: str, margin_threshold: float = 0.1):
    out = tmp_path / "data" / "rl" / "pairs.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "suite_id": "synth",
            "task_id": tid,
            "run_id": run_id,
            "turn_idx": 1,
            "state_text": "fallback state text for pair0",
            "chosen_json": json.dumps(
                {"action": "ASK", "question": "what is the value at step 1?", "rationale": "r"}
            ),
            "rejected_json": json.dumps(
                {"action": "ASK", "question": "what is the distractor about?", "rationale": "r"}
            ),
            "margin": 0.31,
            "len_delta": 2,
            "latent_depth": 1,
            "is_latent": True,
            "newly_reachable": True,
            "chosen_run_id": "cand_chosen_0",
            "rejected_run_id": "cand_rejected_0",
            "pair_id": "pair0",
        }
    ]
    out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (out.parent / "pairs.manifest.json").write_text(
        json.dumps({"margin_threshold": margin_threshold})
    )
    return out


def _write_matches(tmp_path: Path, *, run_id: str, node_id: str):
    import pyarrow.parquet as pq

    from pi_eval import schema as sch

    out_dir = tmp_path / "scores" / "parquet"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "run_id": run_id,
            "suite_id": "synth",
            "task_id": "s0",
            "node_id": node_id,
            "match_kind": "resolve",
            "matched_turn_idx": 0,
            "matcher_id": "mechanical_v3",
            "matcher_family": "rule",
            "matcher_score": 1.0,
            "threshold": 1.0,
            "graph_version": "v1",
            "cited_uids": [],
        },
        {
            "run_id": run_id,
            "suite_id": "synth",
            "task_id": "s0",
            "node_id": "n0_0_1",
            "match_kind": "none",
            "matched_turn_idx": None,
            "matcher_id": "mechanical_v3",
            "matcher_family": "rule",
            "matcher_score": 0.0,
            "threshold": 1.0,
            "graph_version": "v1",
            "cited_uids": [],
        },
    ]
    pq.write_table(sch.to_table("matches", rows), out_dir / "matches.parquet")
    return out_dir


# --------------------------------------------------------------------------- A1 blinding


def test_a1_item_hides_partition_discoverability_and_answer():
    import random

    from pi_eval.gold import GoldGraph, GoldNode

    node = GoldNode(
        gold_suite="synth",
        gold_task_key="s0",
        gold_node_id="n0",
        gold_text="the value for facet 0 at step 0 is V000",
        gold_partition="required",
        gold_discoverability="kb",
        gold_depth=0,
        gold_canary="PINQCANARY_DEADBEEFDEADBEEF",
    )
    graph = GoldGraph(
        gold_suite="synth",
        gold_task_key="s0",
        gold_nodes=(node,),
        gold_answer="THE_SECRET_ANSWER PINQCANARY_DEADBEEFDEADBEEF",
        gold_canary="PINQCANARY_DEADBEEFDEADBEEF",
    )
    items = sample_a1_items({"s0": graph}, random.Random(0), n_tasks=1)
    assert len(items) == 1
    blob = json.dumps(items[0])
    for forbidden in (
        "gold_partition",
        "gold_discoverability",
        "gold_depth",
        "gold_answer",
        "gold_canary",
        "required",
        "kb",
        "THE_SECRET_ANSWER",
        "PINQCANARY_DEADBEEFDEADBEEF",
    ):
        assert forbidden not in blob, f"{forbidden!r} leaked into an A1 item"
    assert a1_node_projection(node) == {"node_id": "n0", "text": node.gold_text}


def test_bundle_deterministic_under_seed():
    import random

    from pi_eval.gold import GoldGraph, GoldNode

    graphs = {}
    for t in range(4):
        nodes = tuple(
            GoldNode(
                gold_suite="synth",
                gold_task_key=f"s{t}",
                gold_node_id=f"n{t}_{i}",
                gold_text=f"need {t} {i}",
                gold_partition="required",
                gold_discoverability="kb",
            )
            for i in range(3)
        )
        graphs[f"s{t}"] = GoldGraph(gold_suite="synth", gold_task_key=f"s{t}", gold_nodes=nodes)

    from pi_eval.annotate import item_set_hash

    a = sample_a1_items(graphs, random.Random(7), n_tasks=2)
    b = sample_a1_items(graphs, random.Random(7), n_tasks=2)
    c = sample_a1_items(graphs, random.Random(8), n_tasks=2)
    assert item_set_hash(a) == item_set_hash(b)
    assert item_set_hash(a) != item_set_hash(c)


# --------------------------------------------------------------------------- CLI: export


def test_export_refuses_without_gold_root(tmp_path, monkeypatch):
    monkeypatch.delenv("PI_GOLD_ROOT", raising=False)
    args = build_parser().parse_args(
        ["annotate", "export", "--suite", "synth", "--root", str(tmp_path)]
    )
    assert args.fn(args) == 2
    assert not (tmp_path / "data" / "gold" / "human").exists()


def test_export_excludes_dev_and_gold_exposed_runs(tmp_path, monkeypatch):
    cache, suite, tid, units, runs, keep = _setup(tmp_path, monkeypatch)
    args = build_parser().parse_args(
        [
            "annotate",
            "export",
            "--suite",
            "synth",
            "--root",
            str(tmp_path),
            "--runs-root",
            str(runs),
            "--seed",
            "0",
            "--n-a1",
            "1",
            "--n-a2",
            "0",
            "--n-a3-node",
            "0",
            "--n-a3-edge",
            "0",
            "--n-a3-match",
            "0",
            "--n-a4",
            "5",
        ]
    )
    assert args.fn(args) == 0
    bundles = list((tmp_path / "data" / "gold" / "human" / "synth" / "bundles").glob("*.json"))
    bundle_path = next(p for p in bundles if not p.name.endswith(".key.json"))
    bundle = json.loads(bundle_path.read_text())
    refused = bundle["manifest"]["refused"]
    assert refused.get("dev_run", 0) >= 1
    assert sum(v for k, v in refused.items() if "state_mismatch" in k or "gold_exposed" in k) >= 1
    run_ids = {it["provenance"].get("run_id") for it in bundle["items"] if it["task_type"] == "A4"}
    assert "dev-abc123" not in run_ids
    assert "ceiling0" not in run_ids


def test_bundle_carries_no_canary(tmp_path, monkeypatch):
    cache, suite, tid, units, runs, keep = _setup(tmp_path, monkeypatch)
    args = build_parser().parse_args(
        [
            "annotate",
            "export",
            "--suite",
            "synth",
            "--root",
            str(tmp_path),
            "--runs-root",
            str(runs),
            "--seed",
            "0",
            "--n-a1",
            "2",
            "--n-a2",
            "0",
            "--n-a3-node",
            "0",
            "--n-a3-edge",
            "0",
            "--n-a3-match",
            "0",
            "--n-a4",
            "0",
        ]
    )
    assert args.fn(args) == 0
    bundle_dir = tmp_path / "data" / "gold" / "human" / "synth" / "bundles"
    before = set(bundle_dir.glob("*.json"))
    assert before

    from pi_eval.canary import load as load_canaries

    real_canary = sorted(load_canaries(tmp_path))[0]

    def _poisoned(*a, **k):
        return [
            {
                "item_id": "poison",
                "task_type": "A1",
                "context": {"question": f"hint: {real_canary}"},
                "payload": {"nodes": []},
                "provenance": {"suite": "synth", "task_id": "s0", "task_key": "s0"},
            }
        ]

    monkeypatch.setattr(cmd_annotate, "sample_a1_items", _poisoned)
    args2 = build_parser().parse_args(
        [
            "annotate",
            "export",
            "--suite",
            "synth",
            "--root",
            str(tmp_path),
            "--runs-root",
            str(runs),
            "--seed",
            "1",
            "--n-a1",
            "2",
            "--n-a2",
            "0",
            "--n-a3-node",
            "0",
            "--n-a3-edge",
            "0",
            "--n-a3-match",
            "0",
            "--n-a4",
            "0",
        ]
    )
    assert args2.fn(args2) == 2
    after = set(bundle_dir.glob("*.json"))
    assert after == before, "a poisoned export must write NO new file"


def test_key_file_is_not_the_bundle(tmp_path, monkeypatch):
    cache, suite, tid, units, runs, keep = _setup(tmp_path, monkeypatch)
    node_id = "n0_0_0"
    _write_pairs(tmp_path, tid=tid, run_id=keep.name)
    _write_matches(tmp_path, run_id=keep.name, node_id=node_id)

    args = build_parser().parse_args(
        [
            "annotate",
            "export",
            "--suite",
            "synth",
            "--root",
            str(tmp_path),
            "--runs-root",
            str(runs),
            "--parquet",
            str(tmp_path / "scores" / "parquet"),
            "--pairs",
            str(tmp_path / "data" / "rl" / "pairs.jsonl"),
            "--seed",
            "0",
            "--n-a1",
            "0",
            "--n-a2",
            "1",
            "--n-a3-node",
            "0",
            "--n-a3-edge",
            "0",
            "--n-a3-match",
            "2",
            "--n-a4",
            "0",
        ]
    )
    assert args.fn(args) == 0
    bundle_dir = tmp_path / "data" / "gold" / "human" / "synth" / "bundles"
    bundle_path = next(p for p in bundle_dir.glob("*.json") if not p.name.endswith(".key.json"))
    key_path = bundle_dir / (bundle_path.stem + ".key.json")
    bundle_text = bundle_path.read_text()
    key_text = key_path.read_text()

    for secret in ("order", "matcher_addresses", "chosen_run_id"):
        assert secret not in bundle_text, f"{secret!r} leaked into the shipped bundle"
        assert secret in key_text, f"{secret!r} missing from the key that must carry it"


# --------------------------------------------------------------------------- context/payload shape


def test_bundle_shape_is_the_consumer_contract(tmp_path, monkeypatch):
    """A real export, over every task type it can produce, must satisfy the annotator-tool
    contract: nothing before this checked that."""
    cache, suite, tid, units, runs, keep = _setup(tmp_path, monkeypatch)
    node_id = "n0_0_0"
    _write_pairs(tmp_path, tid=tid, run_id=keep.name)
    _write_matches(tmp_path, run_id=keep.name, node_id=node_id)

    args = build_parser().parse_args(
        [
            "annotate",
            "export",
            "--suite",
            "synth",
            "--root",
            str(tmp_path),
            "--runs-root",
            str(runs),
            "--parquet",
            str(tmp_path / "scores" / "parquet"),
            "--pairs",
            str(tmp_path / "data" / "rl" / "pairs.jsonl"),
            "--seed",
            "0",
            "--n-a1",
            "2",
            "--n-a2",
            "1",
            "--n-a3-node",
            "2",
            "--n-a3-edge",
            "1",
            "--n-a3-match",
            "2",
            "--n-a4",
            "2",
        ]
    )
    assert args.fn(args) == 0
    bundle_dir = tmp_path / "data" / "gold" / "human" / "synth" / "bundles"
    bundle_path = next(p for p in bundle_dir.glob("*.json") if not p.name.endswith(".key.json"))
    bundle = json.loads(bundle_path.read_text())
    assert bundle["items"], "the fixture must actually produce items for the shape to be proven"

    errs = bundle_shape_errors(bundle)
    assert errs == [], f"a real export violated the consumer contract: {errs}"

    seen_types = {it["task_type"] for it in bundle["items"]}
    assert seen_types >= {"A1", "A3_node", "A3_edge", "A3_missing"}, seen_types


def test_a1_context_carries_only_the_question(tmp_path, monkeypatch):
    """A1 must not carry evidence: the whole point is what a user would ask having read
    nothing. `bundle_shape_errors` is the load-bearing check -- a real A1 item passes it, and
    an item with even one more key (however innocuous-looking) is flagged."""
    import random

    cache, suite, tid, units, runs, keep = _setup(tmp_path, monkeypatch)
    from pi_eval.gold import GoldGraph, GoldNode

    node = GoldNode(
        gold_suite="synth",
        gold_task_key=tid,
        gold_node_id="n0",
        gold_text="the value for facet 0 at step 0 is V000",
        gold_partition="required",
        gold_discoverability="kb",
    )
    graph = GoldGraph(gold_suite="synth", gold_task_key=tid, gold_nodes=(node,))
    items = sample_a1_items({tid: graph}, random.Random(0), n_tasks=1, cache=cache)
    assert set(items[0]["context"]) == {"question"}
    assert items[0]["context"]["question"], "the real task question must reach A1's context"

    def _bundle(items):
        from pi_eval.annotate import item_set_hash

        return {"manifest": {"item_set_hash": item_set_hash(items)}, "items": items}

    assert bundle_shape_errors(_bundle(items)) == []

    poisoned = [dict(items[0], context={"question": "x", "evidence": [{"uid": "u", "text": "t"}]})]
    errs = bundle_shape_errors(_bundle(poisoned))
    assert any("unexpected key" in e and "evidence" in e for e in errs)


def test_a2_ships_structured_context_not_only_the_prompt(tmp_path, monkeypatch):
    """A2 must not force an annotator to read a raw prompt template to compare two questions:
    the evidence units, prior Q/A pairs and draft the Inquirer actually saw must be shipped
    structured, replayed from the same objects `render_state` renders from -- not parsed back
    out of its output. `state_text` stays too, as the escape hatch."""
    cache, suite, tid, units, runs, keep = _setup(tmp_path, monkeypatch)
    _write_pairs(tmp_path, tid=tid, run_id=keep.name)
    pairs_rows = cmd_annotate._load_jsonl(tmp_path / "data" / "rl" / "pairs.jsonl")

    import random

    items, key_entries = sample_a2_items(
        pairs_rows,
        cache,
        random.Random(0),
        n=1,
        runs_root=runs,
        margin_threshold=0.1,
        graph_version="v1",
    )
    assert len(items) == 1
    context = items[0]["context"]
    assert set(context) == {"question", "evidence", "history", "draft", "state_text"}
    assert context["question"], "the task question must be shown, not just the raw prompt"
    assert context["state_text"], "the escape hatch must still be there"
    # turn_idx=1 on the fixture's pairs row: exactly one prior turn was replayed.
    assert len(context["history"]) == 1
    assert context["history"][0]["q"]
    assert isinstance(context["evidence"], list) and len(context["evidence"]) >= 1
    for uid_entry in context["evidence"]:
        assert set(uid_entry) == {"uid", "title", "text"}
    assert isinstance(context["draft"], str)

    bundle = {
        "manifest": {"item_set_hash": ""},
        "items": items,
    }
    from pi_eval.annotate import item_set_hash

    bundle["manifest"]["item_set_hash"] = item_set_hash(items)
    assert bundle_shape_errors(bundle) == []


def test_a3_match_item_names_exactly_one_question(tmp_path, monkeypatch):
    """A near-miss (`match_kind == 'none'`) row has no single matched turn, so it must not
    ship as one item listing every question asked in the run -- `matcher_kappa` is computed
    over one (ask, node) judgment at a time. Fan-out is capped per run, and what the cap drops
    is counted."""
    from pi_eval.gold import GoldGraph, GoldNode

    cache, suite, tid, units, runs, keep = _setup(tmp_path, monkeypatch, n_tasks=1)
    # A run with more asked questions than the fan-out cap, so at least one is dropped.
    many_turns = _turns(units[:1] * 5, questions=[f"q{i}?" for i in range(5)])
    _write_run(runs, "chatty0", task=tid, split="train", turns=many_turns, arm="inquirer_prompted")

    node = GoldNode(
        gold_suite="synth",
        gold_task_key="s0",
        gold_node_id="near_miss_node",
        gold_text="a need nothing asked addressed",
        gold_partition="required",
        gold_discoverability="kb",
    )
    graphs = {"s0": GoldGraph(gold_suite="synth", gold_task_key="s0", gold_nodes=(node,))}
    matches_rows = [
        {
            "run_id": "chatty0",
            "suite_id": "synth",
            "task_id": "s0",
            "node_id": "near_miss_node",
            "match_kind": "none",
            "matched_turn_idx": None,
            "graph_version": "v1",
        }
    ]

    import random

    items, key_entries = sample_a3_items(
        graphs,
        matches_rows,
        cache,
        random.Random(0),
        n_node=0,
        n_edge=0,
        n_match=1,
        runs_root=runs,
        match_fanout_cap=3,
    )
    match_items = [it for it in items if it["task_type"] == "A3_match"]
    assert match_items, "the near-miss row must produce at least one item"
    assert len(match_items) <= 3, "the fan-out cap must be enforced"
    for it in match_items:
        ctx = it["context"]
        assert isinstance(ctx["asked_question"], str) and ctx["asked_question"]
        assert ctx["node_text"] == node.gold_text
    # every item names a DIFFERENT question -- one candidate question per item, not a list.
    asked = [it["context"]["asked_question"] for it in match_items]
    assert len(set(asked)) == len(asked)
    ids = {it["item_id"] for it in match_items}
    assert len(ids) == len(match_items), "fanned-out items must not collide on item_id"


def test_export_refuses_a_malformed_item(tmp_path, monkeypatch):
    """The exporter must refuse to write ANY item whose shape violates the contract -- same
    posture as the canary refusal."""
    cache, suite, tid, units, runs, keep = _setup(tmp_path, monkeypatch)
    bundle_dir = tmp_path / "data" / "gold" / "human" / "synth" / "bundles"
    before = set(bundle_dir.glob("*.json")) if bundle_dir.exists() else set()

    def _bad_a1(*a, **k):
        return [
            {
                "item_id": "bad",
                "task_type": "A1",
                "context": {"question": "q", "evidence": [{"uid": "u", "text": "t"}]},
                "payload": {"nodes": []},
                "provenance": {"suite": "synth", "task_id": "s0", "task_key": "s0"},
            }
        ]

    monkeypatch.setattr(cmd_annotate, "sample_a1_items", _bad_a1)
    args = build_parser().parse_args(
        [
            "annotate",
            "export",
            "--suite",
            "synth",
            "--root",
            str(tmp_path),
            "--runs-root",
            str(runs),
            "--seed",
            "0",
            "--n-a1",
            "1",
            "--n-a2",
            "0",
            "--n-a3-node",
            "0",
            "--n-a3-edge",
            "0",
            "--n-a3-match",
            "0",
            "--n-a4",
            "0",
        ]
    )
    assert args.fn(args) == 2
    after = set(bundle_dir.glob("*.json")) if bundle_dir.exists() else set()
    assert after == before, "a malformed export must write NO new file"


# --------------------------------------------------------------------------- A4: latent/stated mix


def _a4_row(run_id, turn_idx, *, is_latent, latent_depth=0, question=None):
    q = question or f"q-{run_id}-{turn_idx}"
    return {
        "run_id": run_id,
        "turn_idx": turn_idx,
        "task_id": "s0",
        "suite_id": "synth",
        "graph_version": "v1",
        "action_json": json.dumps({"action": "ASK", "question": q, "rationale": "r"}),
        "is_latent": is_latent,
        "latent_depth": latent_depth,
    }


def test_a4_sample_has_both_latent_and_stated():
    """Measured: `sample_a4_items` used to sort by (is_latent, latent_depth, ...) reverse=True
    and take a prefix -- deliberately latent-first, so an 8-item pilot came back 8/8
    mechanically latent. An agreement statistic computed against a constant measures nothing;
    the latent/stated CONTRAST is the whole thing A4 exists to validate."""
    import random

    latent_rows = [
        _a4_row(f"latent{i}", 0, is_latent=True, latent_depth=i % 3 + 1) for i in range(6)
    ]
    stated_rows = [_a4_row(f"stated{i}", 0, is_latent=False) for i in range(6)]
    rows = latent_rows + stated_rows

    items, key = sample_a4_items(rows, {}, random.Random(0), n=6)
    assert len(items) == 6
    labels = [key[it["item_id"]]["mechanical_is_latent"] for it in items]
    assert True in labels and False in labels, (
        "a sample that is all one label measures agreement with a constant, not a contrast"
    )
    assert sum(labels) >= 2 and (len(labels) - sum(labels)) >= 2


def test_a4_sample_falls_back_gracefully_when_one_side_is_short():
    """Only 2 latent decision points exist; the quota (n=6 -> 3/3) cannot be filled from that
    side, so the shortfall must be backfilled from stated rows rather than shipping a short
    bundle, and the shortfall must be counted in stats."""
    import random

    latent_rows = [_a4_row(f"latent{i}", 0, is_latent=True, latent_depth=1) for i in range(2)]
    stated_rows = [_a4_row(f"stated{i}", 0, is_latent=False) for i in range(10)]
    rows = latent_rows + stated_rows
    stats: dict[str, int] = {}

    items, key = sample_a4_items(rows, {}, random.Random(0), n=6, stats=stats)
    assert len(items) == 6, "the quota must still be filled from the other side"
    labels = [key[it["item_id"]]["mechanical_is_latent"] for it in items]
    assert sum(labels) == 2, "every available latent row was used"
    assert any("shortfall" in k for k in stats), "the shortfall must be counted in stats"


# --------------------------------------------------------------------------- A3: a different suite than A1


class _FakeCache:
    """Stands in for `SuiteCache` so `--a3-suite` can be exercised end-to-end without a real
    corpus on disk for the second suite -- only `.view(suite_id, task_id).question` and
    `.units(suite_id, task_id)` are read by the A1/A3 samplers under test."""

    def __init__(self, root):
        self.root = root

    def view(self, suite_id, task_id):
        from types import SimpleNamespace

        return SimpleNamespace(question=f"Q[{suite_id}/{task_id}]")

    def units(self, suite_id, task_id):
        return {}


def _write_suite_gold(tmp_path: Path, suite: str, nodes_by_task: dict) -> None:
    from pi_eval.build.common import write_graphs

    rows = []
    for tk, nodes in nodes_by_task.items():
        rows.append(
            {
                "gold_suite": suite,
                "gold_task_key": tk,
                "gold_nodes": [
                    {
                        "gold_suite": suite,
                        "gold_task_key": tk,
                        "gold_node_id": n["id"],
                        "gold_text": n["text"],
                        "gold_partition": n.get("partition", "required"),
                        "gold_discoverability": n.get("discoverability", "kb"),
                    }
                    for n in nodes
                ],
                "gold_edges": [],
                "gold_facets": [],
                "gold_seed_node_ids": [],
                "gold_graph_version": "v1",
                "gold_answer": "answer",
                "gold_aliases": [],
            }
        )
    write_graphs(tmp_path, suite, "v1", rows)


def test_a3_can_be_drawn_from_a_different_suite_than_a1(tmp_path, monkeypatch):
    """Measured: every one of musique's 2,660 nodes is `gold_partition == 'required'` -- there
    are no `dropped` nodes, so A3_node's planted-foil check can never fire and node precision
    can only ever be 1.0. `--a3-suite` lets the A3 family (A3_node/A3_edge/A3_match/
    A3_missing) draw from a suite that DOES carry both partitions (e.g. wiki2), while A1/A2/A4
    stay on `--suite`. Per-item `provenance.suite` must name the suite the item actually came
    from, not the manifest's primary suite."""
    monkeypatch.setenv("PI_CANARY_SALT", "test-salt")
    monkeypatch.setenv("PI_GOLD_ROOT", str(tmp_path / "data" / "gold"))
    monkeypatch.setattr(cmd_annotate, "SuiteCache", _FakeCache)

    _write_suite_gold(
        tmp_path,
        "musique",
        {"m1": [{"id": "n1", "text": "musique need", "partition": "required"}]},
    )
    _write_suite_gold(
        tmp_path,
        "wiki2",
        {
            "w1": [
                {"id": "n1", "text": "wiki2 required need", "partition": "required"},
                {"id": "n2", "text": "wiki2 dropped foil", "partition": "dropped"},
            ]
        },
    )

    args = build_parser().parse_args(
        [
            "annotate",
            "export",
            "--suite",
            "musique",
            "--a3-suite",
            "wiki2",
            "--root",
            str(tmp_path),
            "--runs-root",
            str(tmp_path / "no-runs"),
            "--seed",
            "0",
            "--n-a1",
            "1",
            "--n-a2",
            "0",
            "--n-a3-node",
            "1",
            "--n-a3-edge",
            "0",
            "--n-a3-match",
            "0",
            "--n-a4",
            "0",
        ]
    )
    assert args.fn(args) == 0

    bundle_dir = tmp_path / "data" / "gold" / "human" / "musique" / "bundles"
    bundle_path = next(p for p in bundle_dir.glob("*.json") if not p.name.endswith(".key.json"))
    bundle = json.loads(bundle_path.read_text())

    assert bundle["manifest"]["suite"] == "musique"
    a1 = [it for it in bundle["items"] if it["task_type"] == "A1"]
    a3 = [it for it in bundle["items"] if it["task_type"] == "A3_node"]
    assert a1 and a3, "the fixture must produce both an A1 and an A3_node item"
    assert {it["provenance"]["suite"] for it in a1} == {"musique"}
    assert {it["provenance"]["suite"] for it in a3} == {"wiki2"}


# --------------------------------------------------------------------------- A3_node: unknown-discoverability


def _disc_graphs(n_kb: int, n_unknown: int) -> dict:
    from pi_eval.gold import GoldGraph, GoldNode

    graphs = {}
    for i in range(n_kb):
        nid = f"kb{i}"
        node = GoldNode(
            gold_suite="musique",
            gold_task_key=nid,
            gold_node_id="n0",
            gold_text=f"kb need {i}",
            gold_partition="required",
            gold_discoverability="kb",
        )
        graphs[nid] = GoldGraph(gold_suite="musique", gold_task_key=nid, gold_nodes=(node,))
    for i in range(n_unknown):
        nid = f"unk{i}"
        node = GoldNode(
            gold_suite="musique",
            gold_task_key=nid,
            gold_node_id="n0",
            gold_text=f"unknown need {i}",
            gold_partition="required",
            gold_discoverability="unknown",
        )
        graphs[nid] = GoldGraph(gold_suite="musique", gold_task_key=nid, gold_nodes=(node,))
    return graphs


def test_a3_node_oversamples_unknown_discoverability():
    """`gold_discoverability == 'unknown'` nodes directly soften `ceiling_private_share`, which
    this project's own framing calls the single most important number, and a human verdict on
    them is immediately useful -- wiki2 has 1,022 of them, strategyqa 1,247. Aim for roughly a
    third of A3_node items to be unknown-discoverability where supply allows.

    The pool is deliberately lopsided (200 kb vs. 6 unknown): a plain uniform draw of 9 from
    206 nodes would put ~0.26 unknown nodes in expectation, so seeing 3 (the 1/3 target, and
    exactly what supply allows) is not something chance produces -- only stratification does.
    """
    import random

    graphs = _disc_graphs(n_kb=200, n_unknown=6)
    items, key = sample_a3_items(graphs, [], None, random.Random(0), n_node=9, n_edge=0, n_match=0)
    node_items = [it for it in items if it["task_type"] == "A3_node"]
    assert len(node_items) == 9
    n_unknown_shown = sum(1 for it in node_items if "unknown need" in it["context"]["node_text"])
    assert n_unknown_shown == 3, f"expected the 1/3 target (3/9), got {n_unknown_shown}/9"


def test_a3_node_unknown_shortfall_is_backfilled_and_counted():
    """Only 1 unknown-discoverability node exists; the target for n_node=9 (~3) cannot be
    filled from that side, so the shortfall must be backfilled from the kb side rather than
    shipping fewer than 9 items, and the shortfall must be counted in stats."""
    import random

    graphs = _disc_graphs(n_kb=20, n_unknown=1)
    stats: dict[str, int] = {}
    items, key = sample_a3_items(
        graphs, [], None, random.Random(0), n_node=9, n_edge=0, n_match=0, stats=stats
    )
    node_items = [it for it in items if it["task_type"] == "A3_node"]
    assert len(node_items) == 9, "the quota must still be filled from the kb side"
    n_unknown_shown = sum(1 for it in node_items if "unknown need" in it["context"]["node_text"])
    assert n_unknown_shown == 1, "the one available unknown node was used"
    assert any("unknown_discoverability" in k and "shortfall" in k for k in stats), stats


# --------------------------------------------------------------------------- planted attention checks


def test_attention_check_is_marked_only_in_the_key():
    """Attention-check items pair a task's question with a need mined for an UNRELATED task, so
    the honest verdict is always 'not a need' -- they exist to catch an annotator, human or
    model, who is not reading. The HARD CONSTRAINT: the expected answer lives in the KEY only,
    never in the shipped item, or a careful annotator could read the tell off the item itself."""
    import random

    from pi_eval.gold import GoldGraph, GoldNode

    graphs = {}
    for t in range(3):
        node = GoldNode(
            gold_suite="synth",
            gold_task_key=f"s{t}",
            gold_node_id=f"n{t}",
            gold_text=f"need for task {t}",
            gold_partition="required",
            gold_discoverability="kb",
        )
        graphs[f"s{t}"] = GoldGraph(gold_suite="synth", gold_task_key=f"s{t}", gold_nodes=(node,))

    items, key_entries = cmd_annotate.sample_attention_items(graphs, random.Random(0), n=2)
    assert items, "the fixture must actually produce attention-check items"
    for it in items:
        assert "attention_check" not in json.dumps(it), "the shipped item must not carry the flag"
        assert it["item_id"] in key_entries
        assert key_entries[it["item_id"]]["attention_check"]["expected"] == "not_a_need"
        # the mismatch itself: the shown node_text must not be the task's own need -- that IS
        # what makes "not a need" the honest, correct answer to what is on screen.
        task_key = it["provenance"]["task_key"]
        own_node_text = graphs[task_key].gold_nodes[0].gold_text
        assert it["context"]["node_text"] != own_node_text


def test_export_wires_n_attention_and_ships_no_flag(tmp_path, monkeypatch):
    cache, suite, tid, units, runs, keep = _setup(tmp_path, monkeypatch, n_tasks=3)
    args = build_parser().parse_args(
        [
            "annotate",
            "export",
            "--suite",
            "synth",
            "--root",
            str(tmp_path),
            "--runs-root",
            str(runs),
            "--seed",
            "0",
            "--n-a1",
            "0",
            "--n-a2",
            "0",
            "--n-a3-node",
            "0",
            "--n-a3-edge",
            "0",
            "--n-a3-match",
            "0",
            "--n-a4",
            "0",
            "--n-attention",
            "2",
        ]
    )
    assert args.fn(args) == 0
    bundle_dir = tmp_path / "data" / "gold" / "human" / "synth" / "bundles"
    bundle_path = next(p for p in bundle_dir.glob("*.json") if not p.name.endswith(".key.json"))
    key_path = bundle_dir / (bundle_path.stem + ".key.json")
    bundle_text = bundle_path.read_text()
    key_text = key_path.read_text()
    assert "attention_check" not in bundle_text
    assert "attention_check" in key_text
    bundle = json.loads(bundle_text)
    assert any(it["task_type"] == "A3_node" for it in bundle["items"])


def test_a6_is_sampled_last_so_it_does_not_shift_the_other_samplers(tmp_path, monkeypatch):
    """Adding A6 to an export must not change WHICH A1/A3/A4 items a seed draws.

    Every sampler shares one `random.Random`, so a new one called in the middle of the chain
    silently re-rolls every draw after it: the same `--seed` would ship a different bundle, and
    a campaign comparing an old export against a new one would be comparing two populations
    while believing it held the seed fixed. `sample_a6_items` is called after all of them for
    exactly this reason, and this is the check that keeps it there.
    """
    # Four tasks, so the A3 node/edge pools are big enough that the draw genuinely depends on
    # the rng state -- with a pool the size of the quota, `rng.sample` returns everything and a
    # shifted stream would be invisible.
    cache, suite, tid, units, runs, keep = _setup(tmp_path, monkeypatch, n_tasks=4)
    node_id = "n0_0_0"
    _write_matches(tmp_path, run_id=keep.name, node_id=node_id)

    # Three candidates on ONE state: the minimum an A6 item is built from.
    pairs = tmp_path / "data" / "rl" / "pairs.jsonl"
    pairs.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "suite_id": "synth",
            "task_id": tid,
            "run_id": keep.name,
            "turn_idx": 1,
            "state_text": "fallback state text",
            "chosen_json": json.dumps({"action": "ASK", "question": "what is the value here?"}),
            "rejected_json": json.dumps({"action": "ASK", "question": f"distractor {i}?"}),
            "margin": 0.1 * i,
            "is_latent": True,
            "chosen_run_id": "cand_win",
            "rejected_run_id": f"cand_lose_{i}",
            "pair_id": f"pair{i}",
        }
        for i in range(2)
    ]
    pairs.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    def _export(n_a6: int, out: Path) -> dict:
        args = build_parser().parse_args(
            [
                "annotate",
                "export",
                "--suite",
                "synth",
                "--root",
                str(tmp_path),
                "--runs-root",
                str(runs),
                "--parquet",
                str(tmp_path / "scores" / "parquet"),
                "--pairs",
                str(pairs),
                "--out",
                str(out),
                "--seed",
                "0",
                "--n-a1",
                "1",
                "--n-a2",
                "1",
                "--n-a3-node",
                "3",
                "--n-a3-edge",
                "2",
                "--n-a3-match",
                "1",
                "--n-a4",
                "1",
                "--n-a5",
                "1",
                "--n-a6",
                str(n_a6),
            ]
        )
        assert args.fn(args) == 0
        path = next(p for p in out.glob("*.json") if not p.name.endswith(".key.json"))
        return json.loads(path.read_text())

    without = _export(0, tmp_path / "out_none")
    with_a6 = _export(5, tmp_path / "out_a6")

    assert [it["item_id"] for it in with_a6["items"] if it["task_type"] == "A6"], (
        "the fixture must actually produce an A6 item, or this proves nothing"
    )
    assert without["manifest"]["counts"]["A6"] == 0

    def non_a6(bundle):
        return [it["item_id"] for it in bundle["items"] if it["task_type"] != "A6"]

    assert non_a6(with_a6) == non_a6(without)


def test_a7_is_sampled_last_so_it_does_not_shift_the_other_samplers(tmp_path, monkeypatch):
    """The same seed-stability contract `test_a6_is_sampled_last...` pins, one sampler later:
    `sample_a7_items` (and its foils) run strictly after A6, so turning A7 on must leave every
    A1..A6 item a seed draws exactly where it was."""
    cache, suite, tid, units, runs, keep = _setup(tmp_path, monkeypatch, n_tasks=4)
    node_id = "n0_0_0"
    _write_matches(tmp_path, run_id=keep.name, node_id=node_id)

    # Three candidates on ONE state, as the A6 mirror builds them, so an A6 item exists in
    # both exports and would move if A7 were sampled before it.
    pairs = tmp_path / "data" / "rl" / "pairs.jsonl"
    pairs.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "suite_id": "synth",
            "task_id": tid,
            "run_id": keep.name,
            "turn_idx": 1,
            "state_text": "fallback state text",
            "chosen_json": json.dumps({"action": "ASK", "question": "what is the value here?"}),
            "rejected_json": json.dumps({"action": "ASK", "question": f"distractor {i}?"}),
            "margin": 0.1 * i,
            "is_latent": True,
            "chosen_run_id": "cand_win",
            "rejected_run_id": f"cand_lose_{i}",
            "pair_id": f"pair{i}",
        }
        for i in range(2)
    ]
    pairs.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    def _export(n_a7: int, out: Path) -> dict:
        args = build_parser().parse_args(
            [
                "annotate",
                "export",
                "--suite",
                "synth",
                "--root",
                str(tmp_path),
                "--runs-root",
                str(runs),
                "--parquet",
                str(tmp_path / "scores" / "parquet"),
                "--pairs",
                str(pairs),
                "--out",
                str(out),
                "--seed",
                "0",
                "--n-a1",
                "1",
                "--n-a2",
                "1",
                "--n-a3-node",
                "3",
                "--n-a3-edge",
                "2",
                "--n-a3-match",
                "1",
                "--n-a4",
                "1",
                "--n-a5",
                "1",
                "--n-a6",
                "1",
                "--n-a7",
                str(n_a7),
            ]
        )
        assert args.fn(args) == 0
        path = next(p for p in out.glob("*.json") if not p.name.endswith(".key.json"))
        return json.loads(path.read_text())

    without = _export(0, tmp_path / "out_none")
    with_a7 = _export(2, tmp_path / "out_a7")

    assert [it["item_id"] for it in with_a7["items"] if it["task_type"] == "A7"], (
        "the fixture must actually produce an A7 item, or this proves nothing"
    )
    assert without["manifest"]["counts"]["A7"] == 0

    def non_a7(bundle):
        return [it["item_id"] for it in bundle["items"] if it["task_type"] != "A7"]

    assert non_a7(with_a7) == non_a7(without)
