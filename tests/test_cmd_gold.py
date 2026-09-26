"""`pi data` and `pi gold`: the two commands the docs have always claimed existed.

WHAT THESE TESTS ARE FOR. Every corpus in this repository used to be produced by typing a
Python one-liner into a shell, while `docs/REPRODUCE.md` documented `pi data fetch` and
`pi gold stats`. A documented command that does not exist is worse than an undocumented one:
the reader concludes the pipeline is reproducible and finds out otherwise only by trying. So
what is asserted here is not "the functions work" — it is that the COMMAND SURFACE the
documentation names is real, idempotent, and refuses rather than guesses.

THE FIXTURE POOL IS SYNTHETIC AND SAYS SO. `build_fixture_pool` writes run directories in the
recorded-rollout format so `pi_eval.mining.from_runs` is exercised against the same bytes the
worker writes. Its `run_id`s are prefixed `fixture-` and its `code_version` is `"fixture"`, so
a fixture run can never be mistaken for a recorded one. It exists because an ADMISSIBLE pool
needs >= 2 model families, and this repository's recorded synth runs have exactly one
(`llm_free`) — which is why the real pool is refused, and why the refusal is tested too.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pi_run.cmd_data import build_suite, inventory
from pi_run.cmd_gold import EXACT_ENTAIL_PIN, gold_stats, mine_suite, read_mined, write_mined

FIX = Path(__file__).parent / "fixtures"

# The question is written so that two of the three needs are ENTITY-COVERED by it and the
# third is not: that makes the seed set, the depth assignment and the orphan rate all
# predictable, rather than a property of whatever the miner happened to do.
QUESTION = "Where was Acme incorporated, Delaware or Nevada, and who was its first CEO?"
NEEDS = (
    ("Acme was incorporated in Delaware", ("u_delaware",)),  # seed: every token is in x
    ("Nevada rejected the Acme filing", ("u_nevada",)),  # seed
    ("the first CEO later founded Zenith", ("u_zenith",)),  # not in x: orphan, no edges
)

# Two model families x two policy forms. Both minima in `pool_is_admissible` are exactly 2,
# so this is the smallest pool that may promote a candidate at all.
CELLS = (
    ("anthropic/claude-sonnet-4-5", "inquirer_prompted"),
    ("anthropic/claude-sonnet-4-5", "checklist"),
    ("gemini/gemini-2.5-pro", "inquirer_prompted"),
    ("gemini/gemini-2.5-pro", "checklist"),
)


def _write_run(
    runs_root: Path,
    *,
    run_id: str,
    task_id: str,
    policy_id: str,
    model: str | None,
    needs=NEEDS,
) -> str:
    d = runs_root / run_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "suite_id": "synth",
                "task_id": task_id,
                "arm_id": policy_id,
                "policy_id": policy_id,
                "seed": 0,
                "code_version": "fixture",
            }
        )
    )
    (d / "status.json").write_text(json.dumps({"run_id": run_id, "status": "ok"}))
    (d / "turns.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "turn_idx": i,
                    "action_kind": "ask",
                    "question": text,
                    "retrieved_uids": list(uids),
                }
            )
            for i, (text, uids) in enumerate(needs)
        )
    )
    # `actor` is not decoration. `RecordedRun.model_family` is derived from the INQUIRER's
    # calls, because the generator of a need is whatever produced the ASK -- so a fixture run
    # that carries asks has to say which role made the call, or it reads as LLM-free and the
    # diversity gate rightly refuses it.
    (d / "calls.jsonl").write_text(
        ""
        if model is None
        else json.dumps({"model": model, "provider": "fixture", "actor": "inquirer"})
    )
    return run_id


def build_fixture_pool(root: Path) -> tuple[Path, Path, Path]:
    """A runs root, a corpus and a scores parquet that together make an admissible pool.

    Returns (runs_root, corpus_dir, parquet_dir). Everything is written under `root`, so a
    caller that points `--root` here can never touch data/.
    """
    import pandas as pd

    runs_root = root / "runs"
    rows: list[dict] = []
    for fam, form in CELLS:
        for seed in (0, 1):
            rid = f"fixture-{fam.split('/')[0]}-{form}-{seed}"
            _write_run(runs_root, run_id=rid, task_id="t1", policy_id=form, model=fam)
            rows.append({"run_id": rid, "metric_name": "task_success", "value": 1.0})
    # The negative control. A pool with no failures cannot compute a lift, and a need that
    # appears just as often in failures is not diagnostic of anything.
    for i, (fam, form) in enumerate(CELLS[:2]):
        rid = f"fixture-fail-{i}"
        _write_run(runs_root, run_id=rid, task_id="t1", policy_id=form, model=fam)
        rows.append({"run_id": rid, "metric_name": "task_success", "value": 0.0})

    corpus = root / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    (corpus / "tasks.jsonl").write_text(
        json.dumps(
            {
                "id": "t1",
                "question": QUESTION,
                "docs": [
                    {"doc_id": "d0", "text": "Acme was incorporated in Delaware in 1912."},
                    {"doc_id": "d1", "text": "Nevada rejected the Acme filing that year."},
                ],
            }
        )
    )

    parquet = root / "parquet"
    parquet.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(parquet / "scores.parquet")
    return runs_root, corpus, parquet


# --------------------------------------------------------------------------- pi gold mine


@pytest.fixture
def pool(tmp_path):
    return build_fixture_pool(tmp_path)


def test_mine_emits_a_graph_that_carries_its_own_diagnostics(tmp_path, pool):
    runs_root, corpus, parquet = pool
    report = mine_suite(
        root=tmp_path,
        suite="synth",
        runs_root=runs_root,
        parquet_dir=parquet,
        corpus_dir=corpus,
        metric="task_success",
        theta=0.85,
    )
    assert report["n_tasks"] == 1
    assert report["n_admissible"] == 1, report["refusals"]
    g = report["graphs"][0]
    assert g["admissible"] and g["reason"] == "ok"

    d = g["diagnostics"]
    # A graph without these is not an instrument, it is an assertion.
    for key in (
        "n_candidates",
        "n_promoted",
        "promotion_rate",
        "orphan_rate",
        "theta_sweep",
        "seed_basis",
        "partition_elasticity",
        "user_private_share",
    ):
        assert key in d, key
    assert d["n_promoted"] == len(NEEDS), d
    # Two of three needs are entity-covered by the question; the third is not, and with no
    # edge to reach it, it has no depth at all rather than being given one.
    assert d["seed_basis"]["n_seeds"] == 2
    assert d["seed_basis"]["entity_coverage"] == 1.0
    assert abs(d["orphan_rate"] - 1 / 3) < 1e-9, d["orphan_rate"]


def test_the_pin_records_that_no_nli_model_ran(tmp_path, pool):
    """String identity is a WEAKER canonicalizer than the pinned NLI model the design calls
    for. Recording it in the graph version is what stops a graph mined this way from being
    read as one mined under a real entailment model."""
    runs_root, corpus, parquet = pool
    report = mine_suite(
        root=tmp_path,
        suite="synth",
        runs_root=runs_root,
        parquet_dir=parquet,
        corpus_dir=corpus,
        metric="task_success",
        theta=0.85,
    )
    assert report["nli_pin"] == EXACT_ENTAIL_PIN
    assert report["graphs"][0]["nli_pin"] == EXACT_ENTAIL_PIN
    assert report["graphs"][0]["graph_version"].startswith("synth/v1.0+")


def test_order_alone_mints_no_edge_without_a_probe(tmp_path, pool):
    """Every trace here resolves need 1 before need 3 in every trace, which is exactly the
    co-occurrence an order-based miner would read as a prerequisite. No probe policy is
    wired, so no intervention can be run, so no edge may be minted."""
    runs_root, corpus, parquet = pool
    report = mine_suite(
        root=tmp_path,
        suite="synth",
        runs_root=runs_root,
        parquet_dir=parquet,
        corpus_dir=corpus,
        metric="task_success",
        theta=0.85,
    )
    assert report["graphs"][0]["edges"] == []


def test_a_single_family_pool_is_refused_with_its_reason(tmp_path):
    """The real synth pool in this repository is exactly this shape: every arm is LLM-free,
    so every successful trace carries the family `llm_free` and one generator's habits would
    be indistinguishable from a real need."""
    import pandas as pd

    runs_root = tmp_path / "runs"
    rows = []
    for seed in (0, 1, 2):
        rid = f"fixture-llmfree-{seed}"
        _write_run(runs_root, run_id=rid, task_id="t1", policy_id="chain", model=None)
        rows.append({"run_id": rid, "metric_name": "task_success", "value": 1.0})
    parquet = tmp_path / "parquet"
    parquet.mkdir()
    pd.DataFrame(rows).to_parquet(parquet / "scores.parquet")

    report = mine_suite(
        root=tmp_path,
        suite="synth",
        runs_root=runs_root,
        parquet_dir=parquet,
        corpus_dir=None,
        metric="task_success",
        theta=0.85,
    )
    assert report["n_admissible"] == 0
    assert report["n_refused"] == 1
    # Emitted, not dropped: a systematically-skipped stratum has to stay visible.
    assert len(report["graphs"]) == 1
    reason = report["refusals"][0]["reason"]
    assert "cells" in reason or "families" in reason, reason


def test_an_unscored_run_is_excluded_not_defaulted_to_failure(tmp_path, pool):
    """`pool.failures` is the negative control behind every lift statistic. Filling it with
    runs that were never scored would bias every lift toward 'everything is diagnostic'."""
    import pandas as pd

    runs_root, corpus, parquet = pool
    _write_run(
        runs_root,
        run_id="fixture-unscored",
        task_id="t1",
        policy_id="checklist",
        model="anthropic/claude-sonnet-4-5",
    )
    report = mine_suite(
        root=tmp_path,
        suite="synth",
        runs_root=runs_root,
        parquet_dir=parquet,
        corpus_dir=corpus,
        metric="task_success",
        theta=0.85,
    )
    pool_row = report["graphs"][0]["pool"]
    assert pool_row["n_unscored"] == 1
    assert pool_row["n_success"] + pool_row["n_failure"] + 1 == pool_row["n_runs"]
    assert pd  # the import is what writes the parquet the fixture reads


def test_mine_round_trips_through_the_artifact(tmp_path, pool):
    runs_root, corpus, parquet = pool
    report = mine_suite(
        root=tmp_path,
        suite="synth",
        runs_root=runs_root,
        parquet_dir=parquet,
        corpus_dir=corpus,
        metric="task_success",
        theta=0.85,
    )
    graphs_path, diag_path = write_mined(tmp_path, report)
    assert graphs_path.exists() and diag_path.exists()
    back = read_mined(tmp_path, "synth")
    assert [g["task_key"] for g in back] == [g["task_key"] for g in report["graphs"]]
    # The diagnostics file must NOT carry the graphs: it is the thing a reviewer reads first.
    assert "graphs" not in json.loads(diag_path.read_text())


def test_mine_refuses_a_runs_root_with_nothing_in_it(tmp_path):
    with pytest.raises(SystemExit) as exc:
        mine_suite(
            root=tmp_path,
            suite="synth",
            runs_root=tmp_path / "empty",
            parquet_dir=tmp_path,
            corpus_dir=None,
            metric="task_success",
            theta=0.85,
        )
    assert "pi run" in str(exc.value)


# --------------------------------------------------------------------------- pi gold stats


def test_gold_stats_recomputes_depth_rather_than_trusting_the_record(tmp_path):
    """A stamped depth that disagrees with the edges it was stamped from is exactly the bug
    that would otherwise surface first in a C@d table."""
    res = build_suite(
        "strategyqa", root=tmp_path, raw_dir=FIX / "strategyqa", limit=4, offline=True
    )
    st = gold_stats(tmp_path, "strategyqa")
    assert st["present"] and st["n_graphs"] == res.n_tasks == 4
    assert st["n_nodes"] > 0
    assert sum(st["depth_histogram"].values()) == st["n_nodes"]
    assert st["provenance"] == {"human_composed": st["n_nodes"]}
    assert 0.0 <= st["orphan_rate"] <= 1.0


def test_gold_stats_on_a_suite_that_was_never_built_says_so(tmp_path):
    st = gold_stats(tmp_path, "wiki2")
    assert st["present"] is False
    assert st["n_graphs"] == 0


# --------------------------------------------------------------------------- pi data


def test_build_is_idempotent_in_the_corpus_hash(tmp_path):
    """A content-addressed corpus directory is what stops a second build from shadowing the
    first. Two builds from the same raw input must land on the same path."""
    kw = dict(root=tmp_path, raw_dir=FIX / "strategyqa", offline=True)
    a = build_suite("strategyqa", limit=4, **kw)
    b = build_suite("strategyqa", limit=4, **kw)
    assert a.corpus_hash == b.corpus_hash
    assert a.corpus == b.corpus
    assert inventory(tmp_path, "strategyqa").corpora == ((a.corpus_hash, 4),)


def test_a_limited_build_changes_the_corpus_hash(tmp_path):
    """...and a build over a different task set must NOT: two corpora are two frozen
    artifacts and collapsing them would make corpus_hash a lie inside every run identity."""
    kw = dict(root=tmp_path, raw_dir=FIX / "strategyqa", offline=True)
    a = build_suite("strategyqa", limit=4, **kw)
    b = build_suite("strategyqa", limit=2, **kw)
    assert a.corpus_hash != b.corpus_hash
    assert {h for h, _ in inventory(tmp_path, "strategyqa").corpora} == {
        a.corpus_hash,
        b.corpus_hash,
    }


def test_an_unknown_split_is_refused_by_name(tmp_path):
    with pytest.raises(SystemExit) as exc:
        build_suite("musique", root=tmp_path, split="test", offline=True)
    assert "no split" in str(exc.value)


def test_offline_build_of_a_missing_raw_input_names_the_fetch(tmp_path):
    """Offline-first: nothing in CI may depend on an upstream URL being up, and the error a
    developer sees must be the command that fixes it."""
    with pytest.raises(FileNotFoundError) as exc:
        build_suite("wiki2", root=tmp_path, offline=True)
    assert "curl" in str(exc.value)


def test_inventory_counts_only_real_raw_files(tmp_path):
    """Two separate facts, which `raw_verified` used to conflate into one.

    This asserted `raw_verified == 1` for a file whose sidecar reads "deadbeef" -- which is not
    the sha256 of "{}" and never was. The assertion held because the field counted files that
    HAD a sidecar, not files whose digest matched, so a truncated download reported as verified
    for as long as its sidecar survived. Per CLAUDE.md rule 4 the test encoded a wrong belief;
    it now asserts the two things separately."""
    import hashlib

    raw = tmp_path / "data" / "raw" / "musique"
    raw.mkdir(parents=True)
    (raw / "a.jsonl").write_text("{}")
    (raw / "a.jsonl.sha256").write_text("deadbeef  a.jsonl\n")

    inv = inventory(tmp_path, "musique")
    assert inv.raw_files == 1  # the sidecar is not an input
    assert inv.raw_with_digest == 1  # it HAS a digest
    assert inv.raw_verified == 0  # and nothing checked it

    checked = inventory(tmp_path, "musique", verify_raw=True)
    assert checked.raw_verified == 0 and checked.raw_mismatched == ("a.jsonl",)

    (raw / "a.jsonl.sha256").write_text(hashlib.sha256(b"{}").hexdigest() + "  a.jsonl\n")
    assert inventory(tmp_path, "musique", verify_raw=True).raw_verified == 1


# --------------------------------------------------------------------------- gold questions
#
# `tier1_oracle` carries the two CEILING arms and both were seeded from
# `recorded_questions(arm_id="inquirer_prompted")` -- the treatment's own questions. That made
# gold_evidence input-identical to parallel_replay, oracle_vreq a second copy of it, and the
# pre-spend headroom gate a tautology: it compared the treatment against itself and could only
# ever report no headroom. A ceiling seeded from the arm it bounds is not a ceiling.


def _graph(task_key, nodes, edges=()):
    from pi_eval.gold import GoldEdge, GoldGraph, GoldNode

    return GoldGraph(
        gold_suite="musique",
        gold_task_key=task_key,
        gold_nodes=tuple(
            GoldNode(gold_suite="musique", gold_task_key=task_key, **n) for n in nodes
        ),
        gold_edges=tuple(
            GoldEdge(gold_suite="musique", gold_task_key=task_key, **e) for e in edges
        ),
    )


def test_prerequisites_are_asked_in_dependency_order():
    """`prerequisite` means the destination is UNANSWERABLE until the source is resolved, so an
    oracle asking them out of order would be asking a question it could not yet have phrased --
    which is precisely the difference `parallel_replay` exists to demonstrate."""
    from pi_run.cmd_gold import questions_for

    g = _graph(
        "t",
        [
            {"gold_node_id": "s2", "gold_text": "second", "gold_partition": "required"},
            {"gold_node_id": "s1", "gold_text": "first", "gold_partition": "required"},
        ],
        [{"gold_src_node_id": "s1", "gold_dst_node_id": "s2", "gold_edge_kind": "prerequisite"}],
    )
    assert questions_for(g, mode="oracle_vreq") == ["first", "second"]


def test_musique_placeholders_are_resolved_from_the_prior_hop():
    """MuSiQue writes later hops as "When was #1 founded?", where #1 is hop 1's ANSWER. Left
    unresolved that string retrieves nothing useful, and a ceiling that scores BELOW the arm it
    bounds is measuring a bug rather than headroom."""
    from pi_run.cmd_gold import questions_for

    g = _graph(
        "t",
        [
            {
                "gold_node_id": "s1",
                "gold_text": "The Collegian >> owned by",
                "gold_aliases": ("Houston Baptist University",),
                "gold_partition": "required",
            },
            {
                "gold_node_id": "s2",
                "gold_text": "When was #1 founded?",
                "gold_aliases": ("1960",),
                "gold_partition": "required",
            },
        ],
        [{"gold_src_node_id": "s1", "gold_dst_node_id": "s2", "gold_edge_kind": "prerequisite"}],
    )
    qs = questions_for(g, mode="oracle_vreq")
    assert qs == ["The Collegian >> owned by", "When was Houston Baptist University founded?"]
    assert not any("#" in q for q in qs)


def test_the_two_ceiling_modes_are_different_ceilings():
    """They bound different things. oracle_vreq bounds question SELECTION and SEQUENCING with
    ordinary retrieval; gold_evidence appends the answer to force retrieval onto the gold
    paragraph and so bounds the ANSWERER. Emitting the same list for both is the bug this
    command was written to remove."""
    from pi_run.cmd_gold import questions_for

    g = _graph(
        "t",
        [
            {
                "gold_node_id": "s1",
                "gold_text": "who makes it",
                "gold_aliases": ("Nike",),
                "gold_partition": "required",
            }
        ],
    )
    vreq = questions_for(g, mode="oracle_vreq")
    gold = questions_for(g, mode="gold_evidence")
    assert vreq == ["who makes it"]
    assert gold == ["who makes it Nike"]
    assert vreq != gold


def test_only_required_nodes_are_asked():
    """A ceiling that also asks the optional and dropped nodes is not the same ceiling: it
    spends retrieval calls the required partition never needed, and the arm's budget is the
    same 64 as the other ceiling's."""
    from pi_run.cmd_gold import questions_for

    g = _graph(
        "t",
        [
            {"gold_node_id": "a", "gold_text": "req", "gold_partition": "required"},
            {"gold_node_id": "b", "gold_text": "opt", "gold_partition": "optional"},
            {"gold_node_id": "c", "gold_text": "drop", "gold_partition": "dropped"},
        ],
    )
    assert questions_for(g, mode="oracle_vreq") == ["req"]


def test_a_cycle_degrades_to_id_order_rather_than_dropping_a_node():
    """There should be no prerequisite cycles. If one ever appears, silently emitting a shorter
    question list would make the ceiling quietly weaker on exactly the tasks with the most
    structure."""
    from pi_run.cmd_gold import questions_for

    g = _graph(
        "t",
        [
            {"gold_node_id": "a", "gold_text": "A", "gold_partition": "required"},
            {"gold_node_id": "b", "gold_text": "B", "gold_partition": "required"},
        ],
        [
            {"gold_src_node_id": "a", "gold_dst_node_id": "b", "gold_edge_kind": "prerequisite"},
            {"gold_src_node_id": "b", "gold_dst_node_id": "a", "gold_edge_kind": "prerequisite"},
        ],
    )
    assert sorted(questions_for(g, mode="oracle_vreq")) == ["A", "B"]


def test_a_relevance_edge_does_not_constrain_the_order():
    """Only `prerequisite` means unanswerable-until. Ordering on relevance edges too would
    invent a dependency the graph never asserted."""
    from pi_run.cmd_gold import questions_for

    g = _graph(
        "t",
        [
            {"gold_node_id": "b", "gold_text": "B", "gold_partition": "required"},
            {"gold_node_id": "a", "gold_text": "A", "gold_partition": "required"},
        ],
        [{"gold_src_node_id": "a", "gold_dst_node_id": "b", "gold_edge_kind": "relevance"}],
    )
    # No prerequisite constraint -> deterministic id order, not the relevance edge's direction.
    assert questions_for(g, mode="oracle_vreq") == ["A", "B"]


def test_the_gold_commands_offer_every_suite_that_has_gold():
    """`--suite` choices came from `SUITES`, which is what the fetch/build pipeline DRIVES.
    tau2 is self-sourced so it is not in that list -- but it HAS a gold graph (5,790 nodes) and
    it carries the primary endpoint, and `pi gold stats --suite tau2` answered
    `invalid choice: 'tau2'`. `pi gold questions` read the same list, so the two ceiling arms
    and the pre-spend headroom gate could not be seeded on tau2 either."""
    from pi_run.cmd_data import SUITES
    from pi_run.cmd_gold import GOLD_SUITES

    assert "tau2" in GOLD_SUITES and "tau2" not in SUITES
    assert set(SUITES) <= set(GOLD_SUITES)


def test_tau2_gold_stats_are_reachable():
    """The numbers existed all along; nothing could ask for them."""
    import pathlib

    from pi_run.cmd_gold import gold_stats

    if not pathlib.Path("data/gold/graphs/tau2").is_dir():
        pytest.skip("tau2 gold not built")
    d = gold_stats(pathlib.Path("."), "tau2")
    assert d["n_nodes"] > 0
    assert 0.0 <= d["orphan_rate"] <= 1.0
    assert "unreachable" in d["depth_histogram"]
