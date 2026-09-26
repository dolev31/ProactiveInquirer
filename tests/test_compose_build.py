"""musique_x2: two held-out MuSiQue tasks composed into one, built on the gold side.

Every structural property the declaration (artifacts/composed_pairs_20260923/DECLARATION.md)
fixes before any data is asserted here on a hand-built source set, so each test names the shape
it checks rather than a golden file:

  * two tasks per pair (both question orders), neutral ids, one shared pair id;
  * the pool is the union of both pools, identical text once, the same pool for both orders;
  * every gold evidence uid is REMAPPED to the composed corpus and is a uid the adapter emits --
    a missing remap must fail the build loudly, because coverage would otherwise read ~0;
  * depths unchanged per node, exactly two facets, no edge across constituents;
  * pairing refusals (shared title, an answer inside the other task, two shallow constituents)
    and the exclusions (training ids, duplicate template, duplicate answer, not one facet) are
    counted by reason;
  * the composed gold carries its own canaries, the corpus carries none, and no source canary
    survives into the composed gold.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from pi_eval.build import compose_build as cb
from pi_eval.build.common import read_graphs, unit_uid
from pi_eval.build.musique_build import build as musique_build

CANARY_RE = re.compile(r"PINQCANARY_[0-9A-F]{16}")
# One paragraph TEXT in every candidate pool, under a different title each time: identical text
# is deduplicated in the union, and a shared text under distinct titles does not refuse a pair.
TWIN = "This exact paragraph text appears in every candidate pool."


def _para(title: str, text: str, supporting: bool = False) -> dict:
    return {"title": title, "paragraph_text": text, "is_supporting": supporting}


def _rec(tid: str, question: str, hops: list[tuple[str, str, int]], paras: list[dict], answer: str):
    """An upstream-shaped MuSiQue record. `hops` = (sub-question, sub-answer, support idx)."""
    return {
        "id": tid,
        "question": question,
        "answer": answer,
        "answer_aliases": [],
        "answerable": True,
        "question_decomposition": [
            {"id": 100 + i, "question": q, "answer": a, "paragraph_support_idx": idx}
            for i, (q, a, idx) in enumerate(hops)
        ],
        "paragraphs": [{"idx": i, **p} for i, p in enumerate(paras)],
    }


def _distractors(prefix: str, n: int) -> list[dict]:
    return [
        _para(f"{prefix} distractor {i}", f"{prefix} filler sentence number {i}.") for i in range(n)
    ]


def _source_records() -> list[dict]:
    """Seven tasks. Hand-counted expectations, per task:

    3hop1__11_12_13   chain s1->s2->s3, depth 2, one facet            answer "Lyon"
    3hop1__21_22_23   chain, depth 2, one facet                       answer "Oslo"
    2hop__31_32       depth 1, one facet                              answer "Quito"
    2hop__41_42       depth 1, one facet                              answer "Hanoi"
    3hop1__13_12_11   SAME hop set as the first -> duplicate template  answer "Lyon2"
    2hop__51_52       answer "Oslo" -> duplicate answer of 3hop1__21_22_23; the deeper
                      task is kept (dedup keeps the first in (-max depth, id) order)
    3hop2__61_62_63   s2 and s3 both reference only #1 -> TWO facets
    """
    recs = []
    recs.append(
        _rec(
            "3hop1__11_12_13",
            "Where was the founder of the company that made Widget born?",
            [
                ("Widget >> manufacturer", "Acme", 0),
                ("#1 >> founded by", "Jane Roe", 1),
                ("#2 >> place of birth", "Lyon", 2),
            ],
            [
                _para("Widget", "Widget is made by Acme.", True),
                _para("Acme", "Acme was founded by Jane Roe.", True),
                _para("Jane Roe", "Jane Roe was born in Lyon.", True),
                _para("A twin", TWIN),
                *_distractors("A", 3),
            ],
            "Lyon",
        )
    )
    recs.append(
        _rec(
            "3hop1__21_22_23",
            "What is the capital of the country where the Gizmo river rises?",
            [
                ("Gizmo river >> source", "Mount Kel", 0),
                ("#1 >> country", "Norway", 1),
                ("#2 >> capital", "Oslo", 2),
            ],
            [
                _para("Gizmo river", "The Gizmo river rises at Mount Kel.", True),
                _para("Mount Kel", "Mount Kel is in Norway.", True),
                _para("Norway", "The capital of Norway is Oslo.", True),
                _para("B twin", TWIN),
                *_distractors("B", 3),
            ],
            "Oslo",
        )
    )
    recs.append(
        _rec(
            "2hop__31_32",
            "What city is the seat of the team Pat Doe plays for?",
            [("Pat Doe >> member of sports team", "Condors", 0), ("#1 >> located in", "Quito", 1)],
            [
                _para("Pat Doe", "Pat Doe plays for the Condors.", True),
                _para("Condors", "The Condors are based in Quito.", True),
                _para("C twin", TWIN),
                *_distractors("C", 3),
            ],
            "Quito",
        )
    )
    recs.append(
        _rec(
            "2hop__41_42",
            "Where is the headquarters of the label that released Blue Song?",
            [("Blue Song >> record label", "Lotus", 0), ("#1 >> headquarters", "Hanoi", 1)],
            [
                _para("Blue Song", "Blue Song was released by Lotus.", True),
                _para("Lotus", "Lotus is headquartered in Hanoi.", True),
                _para("D twin", TWIN),
                *_distractors("D", 3),
            ],
            "Hanoi",
        )
    )
    recs.append(
        _rec(
            "3hop1__13_12_11",
            "A paraphrase that shares the first task's hop set?",
            [
                ("Thing >> maker", "Maker", 0),
                ("#1 >> founder", "Founder", 1),
                ("#2 >> birthplace", "Lyon2", 2),
            ],
            [
                _para("Thing", "Thing is made by Maker.", True),
                _para("Maker", "Maker was founded by Founder.", True),
                _para("Founder", "Founder was born in Lyon2.", True),
            ],
            "Lyon2",
        )
    )
    recs.append(
        _rec(
            "2hop__51_52",
            "Which city hosts the museum that holds Red Vase?",
            [("Red Vase >> collection", "Nordmus", 0), ("#1 >> city", "Oslo", 1)],
            [
                _para("Red Vase", "Red Vase is held by Nordmus.", True),
                _para("Nordmus", "Nordmus is in Oslo.", True),
            ],
            "Oslo",
        )
    )
    recs.append(
        _rec(
            "3hop2__61_62_63",
            "Two strands hang off one seed?",
            [
                ("Seed thing >> owner", "Owner", 0),
                ("#1 >> spouse", "Spouse", 1),
                ("#1 >> employer", "Employer", 2),
            ],
            [
                _para("Seed thing", "Seed thing is owned by Owner.", True),
                _para("Owner spouse", "Owner is married to Spouse.", True),
                _para("Owner job", "Owner works for Employer.", True),
            ],
            "Employer",
        )
    )
    return recs


def _source_build(root: Path, records: list[dict]) -> tuple[Path, Path]:
    raw = root / "raw"
    raw.mkdir(parents=True)
    (raw / "musique_ans_v1.0_dev.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records) + "\n"
    )
    res = musique_build(split="dev", raw_dir=raw, root=root, allow_download=False, verify=False)
    return res.corpus, res.gold


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    """The source MuSiQue build, through the REAL musique builder, so the uid convention and the
    source canaries are exactly what data/ holds."""
    return _source_build(tmp_path_factory.mktemp("musique_src"), _source_records())


def _build(source, out: Path, **kw):
    corpus, gold = source
    kw.setdefault("is_test", lambda tid: True)
    kw.setdefault("train_ids", frozenset())
    return cb.build(musique_corpus=corpus, musique_gold=gold, root=out, seed=7, **kw)


@pytest.fixture(scope="module")
def built(source, tmp_path_factory):
    out = tmp_path_factory.mktemp("x2")
    res = _build(source, out)
    public = [json.loads(x) for x in res.corpus.read_text().splitlines() if x.strip()]
    graphs = {g.gold_task_key: g for g in read_graphs(res.gold)}
    side = [json.loads(x) for x in res.sidecar.read_text().splitlines() if x.strip()]
    return res, public, graphs, side


# ------------------------------------------------------------------ selection and pairing


def test_exclusions_are_counted_by_reason(built):
    res, *_ = built
    ex = res.manifest["exclusions"]
    assert ex["duplicate_template"] == 1  # 3hop1__13_12_11 shares a hop set with 3hop1__11_12_13
    assert ex["duplicate_answer"] == 1  # 2hop__51_52 answers "Oslo" like 3hop1__21_22_23
    assert ex["constituent_facets_not_1"] == 1  # 3hop2__61_62_63 has two facets
    assert res.manifest["n_candidates"] == 4


def test_pairs_obey_every_pairing_rule(built):
    """Four candidates: two deep (depth 2) and two shallow (depth 1). Two shallow tasks may not
    pair, so the only perfect matching pairs each deep task with a shallow one, and every pair
    must carry a constituent of depth >= 2."""
    res, _, _, side = built
    pairs = {tuple(sorted((r["a_id"], r["b_id"]))) for r in side}
    assert len(pairs) == res.manifest["n_pairs"] == 2
    for r in side:
        assert max(r["max_depth"].values()) >= 2
    assert ("2hop__31_32", "2hop__41_42") not in pairs


def test_shared_title_and_answer_leak_refuse_a_pair(tmp_path):
    """Put the second deep task's answer into a shallow task's question and give the other
    shallow task a title from the first deep task's pool: neither shallow task may then pair
    with that deep task."""
    recs = _source_records()
    for r in recs:
        if r["id"] == "2hop__31_32":
            r["question"] = "Which Oslo team does Pat Doe play for?"
        if r["id"] == "2hop__41_42":
            r["paragraphs"][0]["title"] = "Widget"
    corpus, gold = _source_build(tmp_path / "src", recs)
    res = cb.build(
        musique_corpus=corpus,
        musique_gold=gold,
        root=tmp_path / "out",
        seed=7,
        is_test=lambda t: True,
        train_ids=frozenset(),
    )
    side = [json.loads(x) for x in res.sidecar.read_text().splitlines() if x.strip()]
    pairs = {tuple(sorted((r["a_id"], r["b_id"]))) for r in side}
    assert ("2hop__31_32", "3hop1__21_22_23") not in pairs
    assert ("2hop__41_42", "3hop1__11_12_13") not in pairs
    # A census over EVERY candidate pair, independent of the order the matcher tried them in.
    assert res.manifest["pair_rule_census"]["answer_in_other_task"] >= 1
    assert res.manifest["pair_rule_census"]["shared_title"] >= 1
    assert res.manifest["n_pairs"] == 2


def test_training_ids_are_excluded_and_counted(source, tmp_path):
    res = _build(source, tmp_path, train_ids=frozenset({"musique/2hop__31_32"}))
    assert res.manifest["exclusions"]["in_training_ids"] == 1
    side = [json.loads(x) for x in res.sidecar.read_text().splitlines() if x.strip()]
    assert all("2hop__31_32" not in (r["a_id"], r["b_id"]) for r in side)


def test_a_train_id_file_whose_set_hash_is_not_the_pinned_one_is_refused(tmp_path):
    p = tmp_path / "train_ids.txt"
    p.write_text("musique/2hop__1_2\nwiki2/x\n")
    ids = cb.read_train_ids(p, expect_set_hash=None)
    assert ids == frozenset({"musique/2hop__1_2", "wiki2/x"})
    with pytest.raises(cb.ComposeError, match="set hash"):
        cb.read_train_ids(p, expect_set_hash="d6750e34" + "0" * 56)
    good = cb.id_set_hash(ids)
    assert cb.read_train_ids(p, expect_set_hash=good) == ids


def test_source_gold_built_from_another_corpus_is_refused(source, tmp_path):
    """Gold whose uids were minted against another corpus build would make every remap miss."""
    corpus, gold = source
    other = tmp_path / "0000000000000000"
    other.mkdir()
    (other / "tasks.jsonl").write_text(corpus.read_text())
    with pytest.raises(cb.ComposeError, match="describes corpus"):
        cb.build(
            musique_corpus=other / "tasks.jsonl",
            musique_gold=gold,
            root=tmp_path / "out",
            seed=7,
            is_test=lambda t: True,
            train_ids=frozenset(),
        )


def test_the_id_set_hash_is_the_exporters():
    from pinq_train.split import id_set_hash

    ids = ["musique/b", "wiki2/a", "musique/b", "synth/c"]
    assert cb.id_set_hash(ids) == id_set_hash(ids)


def test_non_test_tasks_are_excluded(source, tmp_path):
    res = _build(source, tmp_path, is_test=lambda t: t != "3hop1__21_22_23")
    assert res.manifest["exclusions"]["not_test_split"] == 1


def test_default_split_is_the_harness_split(source):
    """The builder's default candidate filter is the one `pi run --split test` applies."""
    from pi_run.cli import _select_task_ids
    from pinq_adapters.musique.suite import MusiqueSuite

    corpus, _ = source
    suite = MusiqueSuite(corpus.parent)
    harness = set(_select_task_ids(suite, "musique", n=10**6, split="test", offset=0))
    mine = {t for t in suite.task_ids() if cb.harness_is_test(suite)(t)}
    assert mine == harness


# ------------------------------------------------------------------ the composed tasks


def test_two_orders_per_pair_with_neutral_ids_and_the_declared_question(built, source):
    res, public, graphs, side = built
    assert len(public) == 2 * res.manifest["n_pairs"] == res.n_tasks
    by_pair: dict[str, list[dict]] = {}
    for r in side:
        by_pair.setdefault(r["pair_id"], []).append(r)
    for pid, rows in by_pair.items():
        assert sorted(r["order"] for r in rows) == ["ab", "ba"]
        assert {r["task_id"] for r in rows} == {f"{pid}_ab", f"{pid}_ba"}
        assert re.fullmatch(r"x2_[0-9a-f]{12}", pid)
    src = {
        json.loads(x)["id"]: json.loads(x)["question"]
        for x in source[0].read_text().splitlines()
        if x.strip()
    }
    q = {r["id"]: r["question"] for r in public}
    for r in side:
        want = cb.QUESTION_TEMPLATE.format(first=src[r["first_id"]], second=src[r["second_id"]])
        assert q[r["task_id"]] == want
        assert r["first_id"] == (r["a_id"] if r["order"] == "ab" else r["b_id"])
    assert res.manifest["question_template"] == cb.QUESTION_TEMPLATE


def test_pool_is_the_deduplicated_union_and_identical_across_orders(built, source):
    res, public, _, side = built
    corpus, _ = source
    src = {json.loads(x)["id"]: json.loads(x) for x in corpus.read_text().splitlines() if x}
    by_id = {r["id"]: r for r in public}
    for r in side:
        a, b = src[r["a_id"]], src[r["b_id"]]
        texts = [p["text"] for p in a["paragraphs"]] + [p["text"] for p in b["paragraphs"]]
        pool = by_id[r["task_id"]]["paragraphs"]
        assert sorted(p["text"] for p in pool) == sorted(set(texts))
        assert [p["idx"] for p in pool] == list(range(len(pool)))
        twin = by_id[r["pair_id"] + ("_ba" if r["order"] == "ab" else "_ab")]["paragraphs"]
        assert [(p["title"], p["text"]) for p in pool] == [(p["title"], p["text"]) for p in twin]
    assert res.manifest["n_paragraphs_deduplicated"] == res.manifest["n_pairs"]  # the TWIN text


def test_public_records_carry_only_public_keys_and_no_canary(built):
    res, public, _, _ = built
    for r in public:
        assert set(r) == {"id", "question", "paragraphs"}
        for p in r["paragraphs"]:
            assert set(p) == {"idx", "title", "text"}
    assert not CANARY_RE.search(res.corpus.read_text())


def test_every_gold_uid_is_remapped_into_the_adapters_units(built):
    from pinq_adapters.musique_x2.suite import MusiqueX2Suite

    res, public, graphs, _ = built
    suite = MusiqueX2Suite(res.corpus.parent)
    assert set(suite.task_ids()) == set(graphs)
    n = 0
    for tid, g in graphs.items():
        units = {u.uid for u in suite.units(tid)}
        for node in g.gold_nodes:
            assert node.gold_ev_uids, node.gold_node_id
            for uid in node.gold_ev_uids:
                assert uid in units, (tid, node.gold_node_id)
                n += 1
    assert n == res.manifest["uid_remap"]["n_gold_uids"] > 0
    assert res.manifest["uid_remap"]["n_mapped"] == n
    assert res.manifest["uid_remap"]["n_in_adapter_units"] == n


def test_a_gold_uid_with_no_pool_paragraph_fails_the_build(source, tmp_path):
    """A remap that silently dropped a uid would read as coverage ~0, not as an error."""
    corpus, gold = source
    lines = []
    for x in gold.read_text().splitlines():
        if not x.strip():
            continue
        g = json.loads(x)
        if g["gold_task_key"] == "3hop1__11_12_13":
            g["gold_nodes"][1]["gold_ev_uids"] = [
                unit_uid("musique_ans_v1p0", "3hop1__11_12_13", 1, "not the paragraph text")
            ]
        lines.append(json.dumps(g))
    bad = tmp_path / "v1.jsonl"
    bad.write_text("\n".join(lines))
    with pytest.raises(cb.UidRemapError):
        cb.build(
            musique_corpus=corpus,
            musique_gold=bad,
            root=tmp_path / "out",
            seed=7,
            is_test=lambda t: True,
            train_ids=frozenset(),
        )


def test_the_adapter_check_fails_when_a_remapped_uid_is_not_emitted(source, tmp_path, monkeypatch):
    """The second half of the check: a uid that maps but that the adapter never emits (a
    corpus_id drift between builder and adapter) must also refuse the build."""
    monkeypatch.setattr(cb, "CORPUS_ID", "musique_x2_DRIFTED")
    with pytest.raises(cb.UidRemapError, match="adapter"):
        _build(source, tmp_path)


def test_depths_are_unchanged_facets_are_two_and_no_edge_crosses(built, source):
    _, gold = source
    src = {g.gold_task_key: g for g in read_graphs(gold)}
    _, _, graphs, side = built
    for r in side:
        g = graphs[r["task_id"]]
        assert len(g.gold_facets) == 2
        owners = []
        for f in g.gold_facets:
            prefixes = {n.split("_", 1)[0] for n in f.gold_node_ids}
            assert len(prefixes) == 1
            owners.append(prefixes.pop())
        assert sorted(owners) == ["a", "b"]
        for e in g.gold_edges:
            assert e.gold_src_node_id[:2] == e.gold_dst_node_id[:2]
        for prefix, cid in (("a_", r["a_id"]), ("b_", r["b_id"])):
            want = {n.gold_node_id: n.gold_depth for n in src[cid].gold_nodes}
            got = {
                n.gold_node_id[len(prefix) :]: n.gold_depth
                for n in g.gold_nodes
                if n.gold_node_id.startswith(prefix)
            }
            assert got == want
        assert all(n.gold_suite == "musique_x2" for n in g.gold_nodes)
        assert g.gold_corpus_hash == Path(built[0].corpus).parent.name


def test_composed_gold_carries_its_own_canaries_and_no_source_canary(built, source):
    res, _, graphs, _ = built
    _, gold = source
    source_canaries = set(CANARY_RE.findall(gold.read_text()))
    assert source_canaries
    text = res.gold.read_text()
    assert not (set(CANARY_RE.findall(text)) & source_canaries)
    registry = set((res.root / "data" / "canaries" / "canaries.txt").read_text().split())
    for g in graphs.values():
        assert CANARY_RE.fullmatch(g.gold_canary)
        assert g.gold_canary in g.gold_answer
        assert g.gold_canary in registry
        assert CANARY_RE.findall(g.gold_answer) == [g.gold_canary]


def test_sidecar_names_constituents_answers_and_depths(built):
    _, _, graphs, side = built
    for r in side:
        assert set(r) >= {
            "task_id",
            "pair_id",
            "order",
            "a_id",
            "b_id",
            "first_id",
            "second_id",
            "answers",
            "max_depth",
            "n_nodes",
            "n_gold_uids",
        }
        assert set(r["answers"]) == {"a", "b"}
        assert not any(CANARY_RE.search(v) for v in r["answers"].values())
        assert r["n_nodes"]["a"] + r["n_nodes"]["b"] == len(graphs[r["task_id"]].gold_nodes)


def test_builder_and_adapter_agree_on_the_uid_convention():
    from pinq_adapters.musique_x2.suite import MusiqueX2Suite

    assert cb.SUITE == MusiqueX2Suite.suite_id == "musique_x2"
    assert cb.CORPUS_ID == MusiqueX2Suite.corpus_id
    assert cb.CORPUS_ID != "musique_ans_v1p0"


def test_the_build_is_deterministic_under_a_salt(source, tmp_path, monkeypatch):
    monkeypatch.setenv("PI_CANARY_SALT", "fixed-salt-for-test")
    r1 = _build(source, tmp_path / "one")
    r2 = _build(source, tmp_path / "two")
    assert r1.corpus_hash == r2.corpus_hash
    assert r1.gold.read_bytes() == r2.gold.read_bytes()
    assert r1.sidecar.read_bytes() == r2.sidecar.read_bytes()
