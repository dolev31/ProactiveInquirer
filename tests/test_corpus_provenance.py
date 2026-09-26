"""The two strings both called "the corpus hash", and what went wrong between them.

`RunManifest.corpus_hash` is the ADAPTER's hash: `pinq.ids.corpus_hash`, order-independent
over `(doc_id, title, sha256(text))` triples, 64 hex.
The corpus DIRECTORY is named for `sha256(tasks.jsonl payload)[:16]`.

Both are legitimate and neither is the other. What was not legitimate was using one where the
other was meant:

  * `pi_eval.score.load_questions` opened `data/corpora/<suite>/<corpus_hash>/tasks.jsonl` -- a
    path that has never existed -- caught the absence and returned {}. So EVERY judge input
    carried `question=""`, on the path that produces kpr_incremental, a PRIMARY endpoint.
    Nobody hit it because judging had never run: PI_JUDGE_CLIENT had no implementation. Two
    holes, and each hid the other.

  * The gold tree is not content-addressed -- one file per (suite, graph_version), last build
    wins -- so a rebuild silently repoints every graph at a new corpus while finished runs
    still name the old one. Task ids match, graphs load, nothing errors, and every gold_ev_uid
    refers to spans of a corpus the run never saw.
"""

from __future__ import annotations

import json
from pathlib import Path

from pi_eval.build.synth_build import build
from pi_eval.score import load_questions


def test_the_two_hashes_are_genuinely_different_strings(tmp_path):
    """If they ever coincide this whole file is unnecessary -- and the bug it describes would
    have been impossible. They do not coincide."""
    from pinq_adapters.synth.suite import SynthSuite

    corpus, _gold, chash = build(n_tasks=6, n_facets=2, depth=2, root=tmp_path)
    adapter_hash = SynthSuite(corpus.parent).corpus_hash

    assert corpus.parent.name == chash
    assert adapter_hash != chash
    assert len(chash) == 16 and len(adapter_hash) == 64


def test_load_questions_finds_the_corpus_by_directory_and_not_by_hash(tmp_path):
    """The regression. Passing the adapter hash returns {}; passing the directory returns the
    questions -- and {} is what every run got."""
    from pinq_adapters.synth.suite import SynthSuite

    corpus, _gold, chash = build(n_tasks=6, n_facets=2, depth=2, root=tmp_path)
    root = tmp_path / "data" / "corpora"
    adapter_hash = SynthSuite(corpus.parent).corpus_hash

    assert load_questions(root, "synth", chash), "the directory must resolve"
    assert len(load_questions(root, "synth", chash)) == 6
    assert load_questions(root, "synth", adapter_hash) == {}, (
        "the adapter hash is not a directory name; this returning {} silently is the bug"
    )


def test_a_run_records_the_directory_its_corpus_lived_in(tmp_path):
    """Without this the scorer cannot find the public corpus at all: the manifest carried only
    a hash that names no path."""
    from pi_run.worker import UnitSpec, run_unit

    corpus, _gold, chash = build(n_tasks=4, n_facets=2, depth=2, root=tmp_path)
    res = run_unit(
        UnitSpec(
            suite_id="synth",
            corpus_dir=str(corpus.parent),
            task_id="s0",
            arm_id="fake_chain",
            seed=0,
            runs_root=str(tmp_path / "runs"),
            cache_root=str(tmp_path / "cache"),
            code_version="t",
            dirty=False,
        )
    )
    manifest = json.loads((tmp_path / "runs" / res["run_id"] / "manifest.json").read_text())
    assert manifest["corpus_dir"] == chash
    assert manifest["corpus_hash"] != manifest["corpus_dir"]
    # A pointer, not an identity: it must not move run_id.
    assert "corpus_dir" not in json.dumps(
        {k: v for k, v in manifest.items() if k == "semantic_hash"}
    )


def test_corpus_dir_is_not_inside_semantic_hash():
    """`corpus_hash` is the identity of the corpus content. The directory is where it happened
    to be written; putting it in run identity would make two byte-identical corpora at two
    paths produce different run ids."""
    from pinq.types import RunManifest

    kw = dict(
        suite_id="synth",
        task_id="t",
        arm_id="a",
        policy_id="p",
        seed=0,
        split="test",
        corpus_hash="c",
        budget_cap=16,
        max_turns=16,
        word_cap=120,
    )
    a = RunManifest(**kw, corpus_dir="aaaa")
    b = RunManifest(**kw, corpus_dir="bbbb")
    assert a.semantic_hash == b.semantic_hash
    assert a.run_id == b.run_id


def test_gold_names_the_corpus_it_was_built_from(tmp_path):
    from pi_eval.build.common import read_graphs

    _corpus, gold, chash = build(n_tasks=5, n_facets=2, depth=2, root=tmp_path)
    graphs = read_graphs(gold)
    assert graphs and all(g.gold_corpus_hash == chash for g in graphs)


def test_scoring_refuses_gold_that_describes_a_different_corpus(tmp_path, monkeypatch):
    """The failure this prevents is not an error. It is `evidence_coverage` and the whole RNR
    ladder collapsing toward zero because every gold_ev_uid names a span of a corpus the run
    never read -- indistinguishable, in a table, from a policy that found nothing."""
    from pi_run.compact import compact
    from pi_run.worker import UnitSpec, run_unit

    # Two corpora, two golds. Build the second LAST so data/gold/graphs/synth/v1.jsonl -- which
    # is not content-addressed -- describes corpus B while the runs were rolled against A.
    corpus_a, _ga, hash_a = build(n_tasks=8, n_facets=2, depth=2, root=tmp_path)
    runs_root = tmp_path / "runs"
    for tid in ("s0", "s1", "s2"):
        run_unit(
            UnitSpec(
                suite_id="synth",
                corpus_dir=str(corpus_a.parent),
                task_id=tid,
                arm_id="fake_chain",
                seed=0,
                runs_root=str(runs_root),
                cache_root=str(tmp_path / "cache"),
                code_version="t",
                dirty=False,
            )
        )
    _corpus_b, _gb, hash_b = build(n_tasks=5, n_facets=2, depth=2, root=tmp_path)
    assert hash_a != hash_b

    compact(runs_root, tmp_path / "parquet", include_dev=True)
    monkeypatch.setenv("PI_GOLD_ROOT", str(tmp_path / "data" / "gold"))

    from pi_eval.score import score

    res = score(
        parquet_dir=tmp_path / "parquet",
        runs_root=runs_root,
        corpora_root=tmp_path / "data" / "corpora",
    )
    assert res.n_runs_scored == 0
    assert res.runs_skipped_corpus_mismatch, "a stale gold tree must be impossible to miss"
    key = next(iter(res.runs_skipped_corpus_mismatch))
    assert hash_b[:12] in key and hash_a[:12] in key


def test_matching_gold_scores_normally(tmp_path, monkeypatch):
    """The guard must not fire on the ordinary case, or it would be disabled within a week."""
    from pi_run.compact import compact
    from pi_run.worker import UnitSpec, run_unit

    corpus, _gold, _chash = build(n_tasks=6, n_facets=2, depth=2, root=tmp_path)
    runs_root = tmp_path / "runs"
    for tid in ("s0", "s1"):
        run_unit(
            UnitSpec(
                suite_id="synth",
                corpus_dir=str(corpus.parent),
                task_id=tid,
                arm_id="fake_chain",
                seed=0,
                runs_root=str(runs_root),
                cache_root=str(tmp_path / "cache"),
                code_version="t",
                dirty=False,
            )
        )
    compact(runs_root, tmp_path / "parquet", include_dev=True)
    monkeypatch.setenv("PI_GOLD_ROOT", str(tmp_path / "data" / "gold"))

    from pi_eval.score import score

    res = score(
        parquet_dir=tmp_path / "parquet",
        runs_root=runs_root,
        corpora_root=tmp_path / "data" / "corpora",
    )
    assert res.n_runs_scored == 2
    assert res.runs_skipped_corpus_mismatch == {}


def test_an_unknown_hash_on_either_side_is_allowed(tmp_path):
    """Empty means unknown, not wrong: gold written before this field existed, and self-sourced
    suites whose corpus is an upstream checkout, must still score."""
    src = Path("src/pi_eval/score.py").read_text()
    assert "if want and got and want != got:" in src, (
        "the guard must require BOTH sides to be known before it refuses"
    )
