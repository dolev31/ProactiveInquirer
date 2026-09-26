"""MuSiQue gold node text carries a mechanical `#N` decomposition placeholder wherever the
step is not depth-0 (see `pi_eval.build.musique_build._graph`). Measured on
`data/gold/graphs/musique/v1.jsonl` over the ADR universe: 1196/1196 depth-0 nodes are
readable, 1464/1464 depth>=1 nodes carry an unresolved `#N` and are NOT -- a perfect,
mechanical split that makes `gold_human_asked` track legibility instead of anticipation. This
file locks down the fix: `#N` is resolved to the referenced sub-question's ANSWER at
item-export time, never in `gold_text` itself, and an item that cannot be fully resolved is
refused rather than shipped with a placeholder.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pi_eval.annotate import bundle_shape_errors, item_set_hash
from pi_eval.build.musique_build import (
    UnresolvedPlaceholder,
    load_subanswers,
    reference_placeholders,
    resolve_placeholders,
)
from pi_eval.gold import GoldEdge, GoldGraph, GoldNode
from pi_run import cmd_annotate
from pi_run.cli import build_parser
from pi_run.cmd_annotate import a1_node_projection, sample_a1_items, sample_a3_items

MUSIQUE = "musique"


# --------------------------------------------------------------------------- resolve_placeholders


def test_resolver_substitutes_the_referenced_subanswer():
    assert (
        resolve_placeholders("When was #1 founded?", {"s1": "Houston Baptist University"})
        == "When was Houston Baptist University founded?"
    )


def test_resolver_handles_multiple_placeholders_in_one_string():
    answers = {"s1": "Houston Baptist University", "s2": "Texas"}
    assert resolve_placeholders("#1 >> #2", answers) == "Houston Baptist University >> Texas"


def test_resolver_does_not_let_1_match_the_prefix_of_10():
    answers = {"s1": "ONE", "s10": "TEN"}
    assert resolve_placeholders("who is #10?", answers) == "who is TEN?"
    assert resolve_placeholders("who is #1?", answers) == "who is ONE?"


def test_resolver_is_a_noop_on_text_with_no_placeholder():
    assert resolve_placeholders("Green >> performer", {}) == "Green >> performer"


def test_resolver_raises_when_the_answer_is_missing():
    with pytest.raises(UnresolvedPlaceholder):
        resolve_placeholders("Who published #1?", {})


def test_reference_resolver_substitutes_a_bracketed_reference_not_the_answer():
    ref = reference_placeholders(
        "Which portion of the Nile runs from Ethiopia to #1 ?",
        {"s1": "University of Khartoum >> country"},
    )
    assert "Sudan" not in ref  # the ANSWER to s1 must never appear
    assert "University of Khartoum" in ref  # the REFERENT (s1's own text) must appear
    assert "#1" not in ref


def test_reference_resolver_raises_when_the_referent_is_missing():
    with pytest.raises(UnresolvedPlaceholder):
        reference_placeholders("Who published #1?", {})


# --------------------------------------------------------------------------- load_subanswers


def _write_raw_musique(raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    train_rows = [
        {
            "id": "2hop__1_2",
            "question_decomposition": [
                {
                    "id": 1,
                    "question": "The Collegian >> owned by",
                    "answer": "Houston Baptist University",
                },
                {"id": 2, "question": "When was #1 founded?", "answer": "1963"},
            ],
        },
        # A task the caller does not ask for -- must not appear in the result and must not
        # slow the loader down by being parsed.
        {
            "id": "2hop__unwanted",
            "question_decomposition": [{"id": 9, "question": "irrelevant", "answer": "x"}],
        },
    ]
    (raw_dir / "musique_ans_v1.0_train.jsonl").write_text(
        "\n".join(json.dumps(r) for r in train_rows) + "\n"
    )
    dev_rows = [
        {
            "id": "2hop__3_4",
            "question_decomposition": [
                {"id": 3, "question": "Green >> performer", "answer": "Steve Hillage"},
                {"id": 4, "question": "#1 >> spouse", "answer": "Miquette Giraudy"},
            ],
        }
    ]
    (raw_dir / "musique_ans_v1.0_dev.jsonl").write_text(
        "\n".join(json.dumps(r) for r in dev_rows) + "\n"
    )


def test_load_subanswers_indexes_by_task_id_and_streams_both_splits(tmp_path):
    raw_dir = tmp_path / "raw" / "musique"
    _write_raw_musique(raw_dir)

    out = load_subanswers(raw_dir, ["2hop__1_2", "2hop__3_4"])

    assert out == {
        "2hop__1_2": {"s1": "Houston Baptist University", "s2": "1963"},
        "2hop__3_4": {"s1": "Steve Hillage", "s2": "Miquette Giraudy"},
    }
    assert "2hop__unwanted" not in out, "only requested task ids may be indexed"


def test_load_subanswers_returns_empty_for_no_requested_tasks(tmp_path):
    raw_dir = tmp_path / "raw" / "musique"
    _write_raw_musique(raw_dir)
    assert load_subanswers(raw_dir, []) == {}


# --------------------------------------------------------------------------- sampler-level resolution


def _musique_graph(task_key: str) -> GoldGraph:
    """One task, two nodes: s1 stated (no placeholder), s2 latent (`#1`) -- exactly the shape
    that produced the perfect readable/latent split this fix closes."""
    n1 = GoldNode(
        gold_suite=MUSIQUE,
        gold_task_key=task_key,
        gold_node_id="s1",
        gold_text="The Collegian >> owned by",
        gold_partition="required",
        gold_discoverability="kb",
        gold_depth=0,
    )
    n2 = GoldNode(
        gold_suite=MUSIQUE,
        gold_task_key=task_key,
        gold_node_id="s2",
        gold_text="When was #1 founded?",
        gold_partition="required",
        gold_discoverability="kb",
        gold_depth=1,
    )
    return GoldGraph(gold_suite=MUSIQUE, gold_task_key=task_key, gold_nodes=(n1, n2))


def test_a1_node_projection_resolves_the_placeholder():
    node = _musique_graph("t1").gold_nodes[1]
    answers = {"s1": "Houston Baptist University"}
    projected = a1_node_projection(node, answers)
    assert projected["text"] == "When was Houston Baptist University founded?"
    assert "#1" not in projected["text"]


def test_node_ids_are_unchanged_by_resolution():
    node = _musique_graph("t1").gold_nodes[1]
    projected = a1_node_projection(node, {"s1": "Houston Baptist University"})
    # the annotation must still map back to node s2, regardless of what its TEXT resolved to
    assert projected["node_id"] == "s2" == node.gold_node_id


def test_an_unresolvable_item_is_refused_not_shipped():
    import random

    graphs = {"t1": _musique_graph("t1")}
    stats: dict[str, int] = {}
    # No answers supplied at all -> s2's `#1` cannot resolve -> the whole A1 item for t1 must
    # be refused, not shipped with a placeholder in it.
    items = sample_a1_items(graphs, random.Random(0), n_tasks=1, answers={}, stats=stats)
    assert items == []
    assert sum(v for k, v in stats.items() if "unresolved" in k) >= 1


def test_a1_item_ships_when_the_answer_is_present():
    import random

    graphs = {"t1": _musique_graph("t1")}
    answers = {"t1": {"s1": "Houston Baptist University"}}
    items = sample_a1_items(graphs, random.Random(0), n_tasks=1, answers=answers)
    assert len(items) == 1
    texts = {n["node_id"]: n["text"] for n in items[0]["payload"]["nodes"]}
    assert texts["s1"] == "The Collegian >> owned by"
    assert texts["s2"] == "When was Houston Baptist University founded?"


def test_a3_node_context_resolves_and_a3_edge_context_resolves():
    import random

    graphs = {"t1": _musique_graph("t1")}
    answers = {"t1": {"s1": "Houston Baptist University"}}
    items, _key = sample_a3_items(
        graphs, [], None, random.Random(0), n_node=2, n_edge=0, n_match=0, answers=answers
    )
    node_items = [it for it in items if it["task_type"] == "A3_node"]
    assert node_items, "the fixture must produce an A3_node item to prove the resolution"
    texts = {it["provenance"]["gold_node_id"]: it["context"]["node_text"] for it in node_items}
    assert texts["s2"] == "When was Houston Baptist University founded?"
    assert "#1" not in json.dumps(items)


def test_a1_still_resolves_to_the_answer():
    """A1 asks what a person would have thought to ask, over what is LEGIBLE from the task
    statement alone -- the answer is illegible to them (they had no way to reach it), so
    substituting it is what makes 'I would not have asked this' an honest verdict. This must
    stay the answer-substitution path, unlike A3_edge's dst_text below."""
    import random

    graphs = {"t1": _musique_graph("t1")}
    answers = {"t1": {"s1": "Houston Baptist University"}}
    items = sample_a1_items(graphs, random.Random(0), n_tasks=1, answers=answers)
    texts = {n["node_id"]: n["text"] for n in items[0]["payload"]["nodes"]}
    assert texts["s2"] == "When was Houston Baptist University founded?"
    assert "#1" not in texts["s2"]


# --------------------------------------------------------------------------- A3_edge: reference, not answer


def _musique_graph_with_edge(task_key: str) -> GoldGraph:
    """The measured live-bundle bug: dst's placeholder resolves to src's ANSWER, which makes
    dst independently answerable and the honest annotator verdict becomes 'no edge' -- the
    wrong instrument for judging a dependency. s1 is 'University of Khartoum >> country'
    (answer 'Sudan'); s2 references it via '#1'."""
    n1 = GoldNode(
        gold_suite=MUSIQUE,
        gold_task_key=task_key,
        gold_node_id="s1",
        gold_text="University of Khartoum >> country",
        gold_partition="required",
        gold_discoverability="kb",
        gold_depth=0,
    )
    n2 = GoldNode(
        gold_suite=MUSIQUE,
        gold_task_key=task_key,
        gold_node_id="s2",
        gold_text="Which portion of the Nile runs from Ethiopia to #1 ?",
        gold_partition="required",
        gold_discoverability="kb",
        gold_depth=1,
    )
    edge = GoldEdge(
        gold_suite=MUSIQUE,
        gold_task_key=task_key,
        gold_src_node_id="s1",
        gold_dst_node_id="s2",
        gold_edge_kind="prerequisite",
    )
    return GoldGraph(
        gold_suite=MUSIQUE, gold_task_key=task_key, gold_nodes=(n1, n2), gold_edges=(edge,)
    )


def test_edge_items_show_the_referent_as_a_reference_not_an_answer():
    import random

    graphs = {"t1": _musique_graph_with_edge("t1")}
    answers = {"t1": {"s1": "Sudan"}}
    items, _key = sample_a3_items(
        graphs, [], None, random.Random(0), n_node=0, n_edge=1, n_match=0, answers=answers
    )
    edge_items = [it for it in items if it["task_type"] == "A3_edge"]
    assert edge_items, "the fixture must produce an A3_edge item to prove the fix"
    ctx = edge_items[0]["context"]
    assert ctx["src_text"] == "University of Khartoum >> country"
    assert "Sudan" not in ctx["dst_text"], "the resolved sub-ANSWER must not appear in dst_text"
    assert "University of Khartoum" in ctx["dst_text"], (
        "a REFERENCE to the source need must appear instead"
    )
    assert "#1" not in ctx["dst_text"]
    import re

    assert re.search(r"#\d+", json.dumps(items)) is None


# --------------------------------------------------------------------------- bundle_shape_errors gate


def _bundle(items):
    return {"manifest": {"item_set_hash": item_set_hash(items)}, "items": items}


def test_bundle_shape_errors_flags_a_leaked_placeholder_in_context():
    item = {
        "item_id": "x1",
        "task_type": "A3_node",
        "context": {
            "question": "when was it founded?",
            "node_text": "When was #1 founded?",
            "evidence": [],
        },
        "payload": {},
        "provenance": {"suite": MUSIQUE, "task_id": "t1", "task_key": "t1"},
    }
    errs = bundle_shape_errors(_bundle([item]))
    assert any("#" in e and "placeholder" in e.lower() for e in errs), errs


def test_bundle_shape_errors_flags_a_leaked_placeholder_in_payload():
    item = {
        "item_id": "x2",
        "task_type": "A1",
        "context": {"question": "q"},
        "payload": {"nodes": [{"node_id": "s2", "text": "When was #1 founded?"}]},
        "provenance": {"suite": MUSIQUE, "task_id": "t1", "task_key": "t1"},
    }
    errs = bundle_shape_errors(_bundle([item]))
    assert any("placeholder" in e.lower() for e in errs), errs


def test_bundle_shape_errors_is_clean_for_resolved_text():
    item = {
        "item_id": "x3",
        "task_type": "A1",
        "context": {"question": "q"},
        "payload": {
            "nodes": [{"node_id": "s2", "text": "When was Houston Baptist University founded?"}]
        },
        "provenance": {"suite": MUSIQUE, "task_id": "t1", "task_key": "t1"},
    }
    assert bundle_shape_errors(_bundle([item])) == []


def test_no_shipped_item_carries_a_placeholder_export_refuses(tmp_path, monkeypatch):
    """The end-to-end gate: if a leaked placeholder ever reached `export`'s bundle (e.g. a
    sampler regression bypassing resolution), `pi annotate export` must refuse and write
    nothing -- the same posture as the canary firewall."""
    from tests.test_annotate_export import _setup

    cache, suite, tid, units, runs, keep = _setup(tmp_path, monkeypatch)
    bundle_dir = tmp_path / "data" / "gold" / "human" / "synth" / "bundles"
    before = set(bundle_dir.glob("*.json")) if bundle_dir.exists() else set()

    def _leaky_a1(*a, **k):
        return [
            {
                "item_id": "leaky",
                "task_type": "A1",
                "context": {"question": "q"},
                "payload": {"nodes": [{"node_id": "s2", "text": "When was #1 founded?"}]},
                "provenance": {"suite": "synth", "task_id": "s0", "task_key": "s0"},
            }
        ]

    monkeypatch.setattr(cmd_annotate, "sample_a1_items", _leaky_a1)
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
    assert after == before, "a bundle with a leaked placeholder must write NO new file"


# --------------------------------------------------------------------------- relation-triple edges


def test_relation_triple_dst_hides_the_subject_behind_a_reference():
    """wiki2 node text arrives ALREADY resolved, so `reference_placeholders` has no `#N` to
    rewrite and the dependency is still invisible:

        src: "Anumodhanam >> director"
        dst: "I. V. Sasi >> date of death"      <- "I. V. Sasi" IS the director

    `wiki2_build` creates a prerequisite edge exactly when `objects[i] == subjects[j]`
    (wiki2_build.py:251), so dst's subject IS src's answer BY CONSTRUCTION -- which is what
    makes replacing it a lookup rather than a guess. Without this the annotator reads a dst
    that is answerable on its own and correctly says "no edge", which is the wrong verdict
    about the dependency being judged.
    """
    from pi_run.cmd_annotate import mask_edge_subject

    out = mask_edge_subject("Anumodhanam >> director", "I. V. Sasi >> date of death")
    assert out == "⟨Anumodhanam >> director⟩ >> date of death"
    assert "I. V. Sasi" not in out


def test_mask_edge_subject_leaves_a_non_triple_dst_alone():
    """A dst that is a real question carries its dependency in prose; rewriting its first
    words would corrupt it."""
    from pi_run.cmd_annotate import mask_edge_subject

    dst = "When was the university founded?"
    assert mask_edge_subject("Anumodhanam >> director", dst) == dst
    assert mask_edge_subject("a plain need", "another plain need") == "another plain need"


def test_masking_is_idempotent():
    from pi_run.cmd_annotate import mask_edge_subject

    once = mask_edge_subject("A >> rel", "B >> other")
    assert mask_edge_subject("A >> rel", once) == once


def test_a3_match_reads_the_run_bearing_graphs_not_the_a3_suite_graphs():
    """`--a3-suite` moves the graph-validation instruments to a suite with real partition
    variety. A3_match must NOT follow them: it judges whether an ASKED QUESTION addresses a
    need, so it needs RUNS, and runs belong to `--suite`.

    Measured: matches.parquet holds 3,148 musique rows over 466 run directories against 12
    for wiki2, so routing A3_match to wiki2 produced ZERO items -- starving matcher kappa,
    the one preregistered gate human annotation can still open (G-M1 is blocked upstream on
    a miner that has admitted nothing).

    Behavioural rather than a source check: match items must be built from `match_graphs`,
    so withholding those while `graphs` still holds the node must yield no match item.
    """
    import random

    from pi_run.cmd_annotate import sample_a3_items

    rows = [
        {
            "run_id": "r1",
            "suite_id": "musique",
            "task_id": "t1",
            "node_id": "s1",
            "match_kind": "resolve",
        }
    ]
    stats: dict[str, int] = {}
    items, _ = sample_a3_items(
        {},
        rows,
        None,
        random.Random(0),
        n_node=0,
        n_edge=0,
        n_match=4,
        stats=stats,
        match_graphs={},
    )
    assert [i for i in items if i["task_type"] == "A3_match"] == []
    assert stats.get("a3_match_unresolvable"), "the shortfall must be COUNTED, not hidden"


def test_match_rows_are_filtered_to_the_suite_before_sampling():
    """`matches.parquet` holds every suite's rows together — measured, musique is 3,148 of
    21,994 (14.3%), drgym 16,704. Sampling stratified over the whole table spends six draws
    in seven on rows whose graphs were never loaded, and each one is counted as
    `a3_match_unresolvable`: asking for 10 match items yielded 2. The shortfall is honest but
    the sample is not — it is a draw from the wrong population, and the instrument it starves
    feeds matcher kappa.
    """
    import random

    from pi_run.cmd_annotate import sample_a3_items

    rows = [
        {
            "run_id": f"r{i}",
            "suite_id": s,
            "task_id": "t1",
            "node_id": "s1",
            "match_kind": "resolve",
        }
        for s in ("drgym", "musique")
        for i in range(20)
    ]
    stats: dict[str, int] = {}
    sample_a3_items(
        {},
        rows,
        None,
        random.Random(0),
        n_node=0,
        n_edge=0,
        n_match=6,
        stats=stats,
        match_graphs={},
        match_suite="musique",
    )
    assert stats.get("a3_match_off_suite_skipped", 0) == 0, (
        "off-suite rows must be excluded from the POOL, not drawn and then counted as failures"
    )
    assert stats.get("a3_match_unresolvable", 0) <= 6


def test_bundle_shape_errors_flags_a_duplicated_item_id():
    """Two items with the same content-addressed id ARE the same item: `item_id` hashes
    (task_type, provenance, payload), so a collision means identical content, and shipping it
    twice asks one annotator one question twice while spending two slots.

    Measured on a real export: 90 items, 89 distinct — two A2 items were byte-identical.
    Nothing caught it, because `item_set_hash` is computed over a SET of ids and silently
    dedupes; the collision only surfaced downstream when two annotation records claimed the
    same `record_id`.
    """
    from pi_eval.annotate import bundle_shape_errors, item_set_hash

    item = {
        "item_id": "dup0",
        "task_type": "A3_edge",
        "context": {"question": "q?", "src_text": "a", "dst_text": "b"},
        "payload": {},
        "provenance": {"suite": "wiki2", "task_id": "t1"},
    }
    items = [item, dict(item)]
    bundle = {
        "manifest": {"bundle_id": "b0", "item_set_hash": item_set_hash(items)},
        "items": items,
    }
    errs = bundle_shape_errors(bundle)
    assert any("dup" in e.lower() or "duplicate" in e.lower() for e in errs), errs


def test_placeholder_gate_ignores_corpus_text_that_really_contains_a_hash_number():
    """The `#N` gate polices GOLD-derived text, where `#1` is always a MuSiQue decomposition
    reference. It must not police corpus prose, where `#1` is just a number.

    Caught in production: a seed-7 export died after ten minutes because an A5 evidence
    paragraph described a radio station as "Connecticut's #1 Rock Station". Refusing one item
    over a false positive is the safe direction; refusing the whole export is not, and the
    text was never gold to begin with.
    """
    from pi_eval.annotate import bundle_shape_errors, item_set_hash

    ev = {
        "uid": "u1",
        "title": "WPLR",
        "text": 'WPLR (99.1 FM, also known as "Connecticut\'s #1 Rock Station") is a classic rock station.',
    }
    item = {
        "item_id": "a5x",
        "task_type": "A5",
        "context": {
            "question": "Who owns the #1 rock station in New Haven?",
            "evidence": [ev],
            "history": [{"q": "which station?", "a": "the #1 one"}],
            "answer": "Connoisseur Media",
        },
        "payload": {"candidates": [{"node_id": "s1", "text": "the owner of the station"}]},
        "provenance": {"suite": "musique", "task_id": "t1", "run_id": "r1"},
    }
    bundle = {
        "manifest": {"bundle_id": "b0", "item_set_hash": item_set_hash([item])},
        "items": [item],
    }
    assert bundle_shape_errors(bundle) == [], "corpus prose is not a decomposition placeholder"


def test_placeholder_gate_still_catches_gold_text():
    """The case it exists for must keep firing: a CANDIDATE is gold node text."""
    from pi_eval.annotate import bundle_shape_errors, item_set_hash

    item = {
        "item_id": "a5y",
        "task_type": "A5",
        "context": {"question": "q?", "evidence": [], "history": [], "answer": "a"},
        "payload": {"candidates": [{"node_id": "s2", "text": "When was #1 founded?"}]},
        "provenance": {"suite": "musique", "task_id": "t1", "run_id": "r1"},
    }
    bundle = {
        "manifest": {"bundle_id": "b0", "item_set_hash": item_set_hash([item])},
        "items": [item],
    }
    assert any("#N" in e or "placeholder" in e for e in bundle_shape_errors(bundle))


def test_a3_match_refuses_dev_runs_and_records_which_matcher_ruled():
    """Two things the A3_match sampler was missing, both measured on a real 2,347-item export.

    DEV RUNS: 127 of 211 A3_match items came from `dev-` runs, against 0 for A4 and A5. A
    `dev-` prefix means the tree was dirty when the run was made; `report.ELIGIBLE` excludes
    those (`run_id LIKE 'dev-%'`, report.py:1852) and D9 makes that mechanical. Sixty percent
    of the evidence for matcher kappa -- the only preregistered gate human annotation can
    still open -- was drawn from runs nothing else in the repository counts.

    WHICH MATCHER: `matches.parquet` is keyed on `(run_id, node_id, matcher_id,
    graph_version)` because more than one matcher exists. It currently holds THREE --
    mechanical_v1/v2/v3 -- disagreeing on 475 (run, node) pairs. G-M2 asks whether "the
    matcher" agrees with a human, and without `matcher_id` on the item there is no way to say
    which one a kappa validated.
    """
    import random

    from pi_run.cmd_annotate import sample_a3_items

    rows = [
        {
            "run_id": "dev-abc",
            "suite_id": "musique",
            "task_id": "t1",
            "node_id": "s1",
            "match_kind": "resolve",
            "matcher_id": "mechanical_v3",
        },
        {
            "run_id": "clean1",
            "suite_id": "musique",
            "task_id": "t1",
            "node_id": "s1",
            "match_kind": "resolve",
            "matcher_id": "mechanical_v3",
        },
    ]
    stats: dict[str, int] = {}
    items, key = sample_a3_items(
        {},
        rows,
        None,
        random.Random(0),
        n_node=0,
        n_edge=0,
        n_match=4,
        stats=stats,
        match_graphs={},
        match_suite="musique",
    )
    assert stats.get("a3_match_dev_run", 0) >= 1, (
        "a dev- run must be refused and COUNTED, as A4/A5 already do"
    )
    for entry in key.values():
        if "matcher_addresses" in entry:
            assert entry.get("matcher_id"), "the key must name which matcher ruled"
