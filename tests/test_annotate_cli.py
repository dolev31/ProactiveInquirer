"""`pi annotate import` and `pi annotate review`: the CLI shell around `pi_eval.annotate`.

Fixtures build gold through the REAL `write_graphs`, so a real canary nonce is minted and
registered exactly the way `tests/test_canary.py` does it -- these tests exercise the CLI, not
a second, hand-rolled notion of what a gold row looks like.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pi_eval.annotate import item_set_hash
from pi_eval.build.common import write_graphs
from pi_eval.gold import GoldNode
from pi_run import cmd_annotate
from pi_run.cli import build_parser

BUNDLE_ID = "musique-0-deadbeef"


# --------------------------------------------------------------------------- fixtures


def _node(nid, **kw):
    kw.setdefault("gold_partition", "required")
    kw.setdefault("gold_discoverability", "kb")
    return GoldNode(
        gold_suite="musique",
        gold_task_key="t1",
        gold_node_id=nid,
        gold_text=f"need {nid}",
        gold_graph_version="v1",
        **kw,
    )


def _write_gold(tmp_path: Path, monkeypatch, nodes) -> None:
    monkeypatch.setenv("PI_CANARY_SALT", "test-salt")
    monkeypatch.setenv("PI_GOLD_ROOT", str(tmp_path / "data" / "gold"))
    row = {
        "gold_suite": "musique",
        "gold_task_key": "t1",
        "gold_nodes": [
            {
                "gold_suite": n.gold_suite,
                "gold_task_key": n.gold_task_key,
                "gold_node_id": n.gold_node_id,
                "gold_text": n.gold_text,
                "gold_partition": n.gold_partition,
                "gold_discoverability": n.gold_discoverability,
            }
            for n in nodes
        ],
        "gold_edges": [],
        "gold_facets": [],
        "gold_seed_node_ids": [],
        "gold_graph_version": "v1",
        "gold_answer": "Rome",
        "gold_aliases": [],
    }
    write_graphs(tmp_path, "musique", "v1", [row])


def _item(task_type, iid, payload=None, **prov):
    p = {"suite": "musique", "task_id": "t1", "task_key": "t1", "graph_version": "v1"}
    p.update(prov)
    return {
        "item_id": iid,
        "task_type": task_type,
        "context": {},
        "payload": payload or {},
        "provenance": p,
    }


def _bundle(items, key_items=None):
    ish = item_set_hash(items)
    bundle = {
        "manifest": {
            "bundle_id": BUNDLE_ID,
            "tool_version": "pi_annotate/1",
            "suite": "musique",
            "graph_version": "v1",
            "item_set_hash": ish,
        },
        "items": list(items),
    }
    key = {
        "manifest": {"bundle_id": BUNDLE_ID, "item_set_hash": ish},
        "items": dict(key_items or {}),
    }
    return bundle, key


def _rec(iid, ann, task_type, response, **kw):
    r = {
        "record_id": f"{BUNDLE_ID}/{iid}/{ann}",
        "bundle_id": BUNDLE_ID,
        "item_id": iid,
        "task_type": task_type,
        "annotator_id": ann,
        "elapsed_ms": 1200,
        "ts": "2026-08-31T12:00:00Z",
        "tool_version": "pi_annotate/1",
        "response": response,
    }
    r.update(kw)
    return r


def _a1_bundle():
    items = [
        _item(
            "A1",
            "a1_t1",
            payload={"nodes": [{"node_id": n, "text": f"need {n}"} for n in ("n1", "n2", "n3")]},
        )
    ]
    return _bundle(items)


def _write(tmp_path: Path, name: str, obj) -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(obj))
    return p


def _write_records(tmp_path: Path, name: str, records) -> Path:
    p = tmp_path / name
    p.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return p


# --------------------------------------------------------------------------- import


def test_import_writes_a_new_graph_version_and_leaves_v1_untouched(tmp_path, monkeypatch):
    _write_gold(tmp_path, monkeypatch, [_node("n1"), _node("n2"), _node("n3")])
    v1_path = tmp_path / "data" / "gold" / "graphs" / "musique" / "v1.jsonl"
    v1_before = v1_path.read_bytes()

    bundle, key = _a1_bundle()
    bundle_p = _write(tmp_path, "bundle.json", bundle)
    key_p = _write(tmp_path, "key.json", key)
    records_p = _write_records(
        tmp_path,
        "records.jsonl",
        [
            _rec("a1_t1", "alice", "A1", {"ticked": ["n1"], "usefulness": {}}),
            _rec("a1_t1", "bob", "A1", {"ticked": ["n1"], "usefulness": {}}),
        ],
    )

    args = build_parser().parse_args(
        [
            "annotate",
            "import",
            "--suite",
            "musique",
            "--bundle",
            str(bundle_p),
            "--key",
            str(key_p),
            "--records",
            str(records_p),
            "--root",
            str(tmp_path),
            "--out-version",
            "v1h",
        ]
    )
    assert args.fn(args) == 0

    assert v1_path.read_bytes() == v1_before, "v1.jsonl must be byte-identical after the import"
    v1h_path = tmp_path / "data" / "gold" / "graphs" / "musique" / "v1h.jsonl"
    assert v1h_path.exists()
    row = json.loads(v1h_path.read_text().splitlines()[0])
    by_id = {n["gold_node_id"]: n for n in row["gold_nodes"]}
    assert by_id["n1"]["gold_human_asked"] is True
    assert by_id["n2"]["gold_human_asked"] is False


def test_import_refuses_invalid_records_and_writes_nothing(tmp_path, monkeypatch):
    _write_gold(tmp_path, monkeypatch, [_node("n1"), _node("n2"), _node("n3")])
    bundle, key = _a1_bundle()
    bundle_p = _write(tmp_path, "bundle.json", bundle)
    key_p = _write(tmp_path, "key.json", key)
    records_p = _write_records(
        tmp_path, "records.jsonl", [_rec("nope", "alice", "A1", {"ticked": []})]
    )

    v1_path = tmp_path / "data" / "gold" / "graphs" / "musique" / "v1.jsonl"
    v1_before = v1_path.read_bytes()

    args = build_parser().parse_args(
        [
            "annotate",
            "import",
            "--suite",
            "musique",
            "--bundle",
            str(bundle_p),
            "--key",
            str(key_p),
            "--records",
            str(records_p),
            "--root",
            str(tmp_path),
        ]
    )
    assert args.fn(args) == 1
    assert v1_path.read_bytes() == v1_before
    v1h_path = tmp_path / "data" / "gold" / "graphs" / "musique" / "v1h.jsonl"
    assert not v1h_path.exists()
    human_dir = tmp_path / "data" / "gold" / "human" / "musique"
    assert not human_dir.exists()


def test_gate_line_omits_an_unmeasured_gate(tmp_path, monkeypatch, capsys):
    """No A3_match items in this campaign: the printed line must carry no --matcher-kappa."""
    _write_gold(tmp_path, monkeypatch, [_node("n1"), _node("n2", gold_partition="optional")])
    items = [
        _item("A3_node", "n0", gold_node_id="n1"),
        _item("A3_node", "n1i", gold_node_id="n2"),
    ]
    bundle, key = _bundle(items)
    bundle_p = _write(tmp_path, "bundle.json", bundle)
    key_p = _write(tmp_path, "key.json", key)
    records_p = _write_records(
        tmp_path,
        "records.jsonl",
        [
            _rec(iid, ann, "A3_node", {"verdict": "required", "discoverability": "kb"})
            for iid in ("n0", "n1i")
            for ann in ("a", "b")
        ],
    )

    args = build_parser().parse_args(
        [
            "annotate",
            "import",
            "--suite",
            "musique",
            "--bundle",
            str(bundle_p),
            "--key",
            str(key_p),
            "--records",
            str(records_p),
            "--root",
            str(tmp_path),
            "--n-missing-adjudicated",
            "0",
        ]
    )
    assert args.fn(args) == 0
    out = capsys.readouterr().out
    line = next(ln for ln in out.splitlines() if ln.startswith("pi gold validate"))
    assert "--matcher-kappa" not in line
    assert "--node-recall" in line
    assert "--edge-precision" not in line, "no A3_edge items either"

    gates = json.loads(
        (tmp_path / "data" / "gold" / "human" / "musique" / "gates.json").read_text()
    )
    assert gates["bundle_id"] == BUNDLE_ID


def test_import_dry_run_writes_nothing(tmp_path, monkeypatch):
    _write_gold(tmp_path, monkeypatch, [_node("n1")])
    bundle, key = _a1_bundle()
    bundle_p = _write(tmp_path, "bundle.json", bundle)
    key_p = _write(tmp_path, "key.json", key)
    records_p = _write_records(
        tmp_path, "records.jsonl", [_rec("a1_t1", a, "A1", {"ticked": ["n1"]}) for a in ("a", "b")]
    )
    args = build_parser().parse_args(
        [
            "annotate",
            "import",
            "--suite",
            "musique",
            "--bundle",
            str(bundle_p),
            "--key",
            str(key_p),
            "--records",
            str(records_p),
            "--root",
            str(tmp_path),
            "--dry-run",
        ]
    )
    assert args.fn(args) == 0
    assert not (tmp_path / "data" / "gold" / "graphs" / "musique" / "v1h.jsonl").exists()
    assert not (tmp_path / "data" / "gold" / "human").exists()


# --------------------------------------------------------------------------- review


def _a2_bundle(order="ab"):
    items = [
        _item(
            "A2",
            "p0",
            payload={
                "option_a": {"question": "which region did Andy sail to"},
                "option_b": {"question": "what city was Gotham filmed in"},
            },
            run_id="parent",
            turn_idx=1,
        )
    ]
    key_items = {
        "p0": {
            "order": order,
            "chosen_run_id": "c1",
            "rejected_run_id": "c2",
            "margin": 0.33,
            "pair_id": "pid0",
        }
    }
    return _bundle(items, key_items)


def test_review_flags_position_bias(tmp_path, capsys):
    bundle, key = _a2_bundle(order="ab")
    bundle_p = _write(tmp_path, "bundle.json", bundle)
    key_p = _write(tmp_path, "key.json", key)
    always_first = _write_records(
        tmp_path,
        "always_first.jsonl",
        [_rec("p0", f"ann{i}", "A2", {"choice": "a"}) for i in range(12)],
    )

    args = build_parser().parse_args(
        [
            "annotate",
            "review",
            "--bundle",
            str(bundle_p),
            "--key",
            str(key_p),
            "--records",
            str(always_first),
            "--json",
        ]
    )
    assert args.fn(args) == 0
    report = json.loads(capsys.readouterr().out)
    bias = report["a2_position_bias"]
    assert bias["n"] == 12
    assert bias["p_first"] == pytest.approx(1.0)
    assert bias["binomial_p"] < 0.01

    balanced = _write_records(
        tmp_path,
        "balanced.jsonl",
        [_rec("p0", f"ann{i}", "A2", {"choice": "a" if i % 2 == 0 else "b"}) for i in range(12)],
    )
    args2 = build_parser().parse_args(
        [
            "annotate",
            "review",
            "--bundle",
            str(bundle_p),
            "--key",
            str(key_p),
            "--records",
            str(balanced),
            "--json",
        ]
    )
    assert args2.fn(args2) == 0
    report2 = json.loads(capsys.readouterr().out)
    bias2 = report2["a2_position_bias"]
    assert bias2["p_first"] == pytest.approx(0.5)
    assert bias2["binomial_p"] > 0.5


def test_review_writes_nothing(tmp_path):
    bundle, key = _a2_bundle()
    bundle_p = _write(tmp_path, "bundle.json", bundle)
    key_p = _write(tmp_path, "key.json", key)
    records_p = _write_records(
        tmp_path, "records.jsonl", [_rec("p0", "alice", "A2", {"choice": "a"})]
    )
    before = sorted(p.name for p in tmp_path.iterdir())

    args = build_parser().parse_args(
        [
            "annotate",
            "review",
            "--bundle",
            str(bundle_p),
            "--key",
            str(key_p),
            "--records",
            str(records_p),
        ]
    )
    assert args.fn(args) == 0
    after = sorted(p.name for p in tmp_path.iterdir())
    assert before == after, "review is read-only and must write no file"


# --------------------------------------------------------------------------- review honesty


def _rev_bundle_and_records(labels_a, labels_b):
    """N A3_edge items, two annotators, agreeing exactly where the two lists agree."""
    items = [
        {
            "item_id": f"e{i}",
            "task_type": "A3_edge",
            "context": {},
            "payload": {},
            "provenance": {"suite": "musique", "task_id": "t", "task_key": "t"},
        }
        for i in range(len(labels_a))
    ]
    bundle = {
        "manifest": {"bundle_id": "b0", "item_set_hash": item_set_hash(items)},
        "items": items,
    }
    records = []
    for who, labels in (("alice", labels_a), ("bob", labels_b)):
        for i, v in enumerate(labels):
            records.append(
                {
                    "record_id": f"b0/e{i}/{who}",
                    "bundle_id": "b0",
                    "item_id": f"e{i}",
                    "task_type": "A3_edge",
                    "annotator_id": who,
                    "elapsed_ms": 1000,
                    "ts": "2026-08-31T12:00:00Z",
                    "response": {"holds": v},
                }
            )
    return bundle, records


def test_agreement_vs_consensus_is_not_one_by_construction():
    """With TWO annotators -- the pilot's configuration -- a majority computed INCLUDING the
    annotator being scored can only ever be 1.0: either they agree and both match it, or they
    tie and the unit is dropped. A statistic that cannot take another value is not a
    measurement, and this one reads like a quality score. The comparison must leave the
    scored annotator out.
    """
    bundle, records = _rev_bundle_and_records([True, True, True, True], [True, True, True, False])
    rep = cmd_annotate.review_report(bundle, records, key=None)
    per = rep["per_annotator"]
    assert per["alice"]["n"] == 4, "every unit both of them judged is comparable"
    assert per["alice"]["agreement"] == pytest.approx(0.75)
    assert per["bob"]["agreement"] == pytest.approx(0.75)


def test_agreement_vs_consensus_uses_the_others_at_three_raters():
    """At n>=3 leave-one-out is a real consensus: the odd one out must score below the two
    who agree, which the include-yourself form cannot express."""
    items = [
        {
            "item_id": f"e{i}",
            "task_type": "A3_edge",
            "context": {},
            "payload": {},
            "provenance": {"suite": "musique", "task_id": "t", "task_key": "t"},
        }
        for i in range(4)
    ]
    bundle = {
        "manifest": {"bundle_id": "b0", "item_set_hash": item_set_hash(items)},
        "items": items,
    }
    votes = {"a": [True] * 4, "b": [True] * 4, "c": [False, False, True, True]}
    records = [
        {
            "record_id": f"b0/e{i}/{who}",
            "bundle_id": "b0",
            "item_id": f"e{i}",
            "task_type": "A3_edge",
            "annotator_id": who,
            "elapsed_ms": 1000,
            "ts": "2026-08-31T12:00:00Z",
            "response": {"holds": v},
        }
        for who, vs in votes.items()
        for i, v in enumerate(vs)
    ]
    per = cmd_annotate.review_report(bundle, records, key=None)["per_annotator"]
    assert per["a"]["agreement"] == pytest.approx(1.0)
    assert per["c"]["agreement"] == pytest.approx(0.5)


def test_review_refuses_an_empty_or_missing_records_file(tmp_path, capsys):
    """An empty report reads as 'nothing to flag'. A records file that is not there has
    nothing to say about annotation quality, and must not be rendered as a clean one."""
    bundle_path = tmp_path / "b.json"
    bundle, _ = _rev_bundle_and_records([True], [True])
    bundle_path.write_text(json.dumps(bundle))

    args = build_parser().parse_args(
        [
            "annotate",
            "review",
            "--bundle",
            str(bundle_path),
            "--records",
            str(tmp_path / "nope.jsonl"),
        ]
    )
    assert args.fn(args) == 2
    assert "no records" in capsys.readouterr().err.lower()


def test_import_refuses_an_empty_records_file_and_writes_no_graph_version(
    tmp_path, monkeypatch, capsys
):
    """Importing nothing must not mint a graph version. A `v1h` with zero annotations is a
    new `scorer_hash` over gold identical to `v1` -- a fresh provenance identity asserting a
    measurement nobody took -- and every later run scored against it would carry that claim."""
    _write_gold(tmp_path, monkeypatch, [_node("n1"), _node("n2"), _node("n3")])
    bundle, key = _a1_bundle()
    bundle_p = _write(tmp_path, "bundle.json", bundle)
    key_p = _write(tmp_path, "key.json", key)

    args = build_parser().parse_args(
        [
            "annotate",
            "import",
            "--suite",
            "musique",
            "--bundle",
            str(bundle_p),
            "--key",
            str(key_p),
            "--records",
            str(tmp_path / "never-written.jsonl"),
            "--root",
            str(tmp_path),
        ]
    )
    assert args.fn(args) == 2
    assert "no records" in capsys.readouterr().err.lower()
    assert not (tmp_path / "data" / "gold" / "graphs" / "musique" / "v1h.jsonl").exists()


def test_import_refuses_when_gold_is_read_from_one_tree_and_written_to_another(
    tmp_path, monkeypatch, capsys
):
    """`load_graphs` resolves against PI_GOLD_ROOT; `write_graphs` resolves against --root.
    Point them at different trees and the import reads one graph and overwrites a DIFFERENT
    repository's gold -- silently, because both paths exist and neither call can see the
    other's. The out-version must land in the tree its base version came from."""
    _write_gold(tmp_path, monkeypatch, [_node("n1"), _node("n2"), _node("n3")])
    elsewhere = tmp_path / "elsewhere" / "gold"
    (elsewhere / "graphs" / "musique").mkdir(parents=True)
    (elsewhere / "graphs" / "musique" / "v1.jsonl").write_text(
        (tmp_path / "data" / "gold" / "graphs" / "musique" / "v1.jsonl").read_text()
    )
    monkeypatch.setenv("PI_GOLD_ROOT", str(elsewhere))

    bundle, key = _a1_bundle()
    args = build_parser().parse_args(
        [
            "annotate",
            "import",
            "--suite",
            "musique",
            "--bundle",
            str(_write(tmp_path, "bundle.json", bundle)),
            "--key",
            str(_write(tmp_path, "key.json", key)),
            "--records",
            str(
                _write_records(
                    tmp_path,
                    "records.jsonl",
                    [_rec("a1_t1", w, "A1", {"ticked": ["n1"]}) for w in ("alice", "bob")],
                )
            ),
            "--root",
            str(tmp_path),
        ]
    )
    assert args.fn(args) == 2
    err = capsys.readouterr().err.lower()
    assert "gold" in err and ("same tree" in err or "differ" in err)
    assert not (tmp_path / "data" / "gold" / "graphs" / "musique" / "v1h.jsonl").exists()
    assert not (elsewhere / "graphs" / "musique" / "v1h.jsonl").exists()


def test_validate_line_states_the_n_each_gate_rests_on():
    """A gate value carries no power information, and these are pasted straight into a
    command that prints PASS or FAIL. `matcher_kappa = -0.25` computed over two units is
    nearer to not-measured than to failed, and nothing in a bare `--matcher-kappa -0.2500`
    says which. The counts ride along so the reader cannot miss them."""
    line = cmd_annotate._validate_line(
        "musique",
        {
            "node_recall": 0.7273,
            "n_node_confirmed": 8,
            "n_missing_adjudicated": 3,
            "edge_precision": 0.5,
            "n_edge_precision": 4,
            "matcher_kappa": -0.25,
            "n_matcher_kappa": 2,
        },
    )
    assert "--node-recall 0.7273" in line
    assert "--matcher-kappa -0.2500" in line
    assert "n=2" in line, "the matcher kappa's n must be visible beside the number"
    assert "n=4" in line and "n=11" in line


def test_review_reports_attention_check_pass_rate_per_annotator():
    items = [_item("A3_node", "attn0"), _item("A3_node", "attn1")]
    bundle, key = _bundle(items, {"attn0": {"attention_check": {"expected": "not_a_need"}}})
    key["items"]["attn1"] = {"attention_check": {"expected": "not_a_need"}}
    records = [
        _rec("attn0", "alice", "A3_node", {"verdict": "not_a_need", "discoverability": "kb"}),
        _rec("attn1", "alice", "A3_node", {"verdict": "required", "discoverability": "kb"}),
        _rec("attn0", "bob", "A3_node", {"verdict": "not_a_need", "discoverability": "kb"}),
        _rec("attn1", "bob", "A3_node", {"verdict": "not_a_need", "discoverability": "kb"}),
    ]
    report = cmd_annotate.review_report(bundle, records, key=key)
    ac = report["attention_checks"]
    assert ac["alice"]["n"] == 2 and ac["alice"]["pass_rate"] == pytest.approx(0.5)
    assert ac["bob"]["n"] == 2 and ac["bob"]["pass_rate"] == pytest.approx(1.0)


def test_validate_line_still_omits_an_unmeasured_gate():
    line = cmd_annotate._validate_line(
        "musique", {"node_recall": float("nan"), "edge_precision": 1.0, "n_edge_precision": 3}
    )
    assert "--node-recall" not in line
    assert "--edge-precision 1.0000" in line
