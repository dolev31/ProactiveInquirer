"""musique_x2: two held-out MuSiQue TEST tasks composed into one task, corpus and gold.

WHAT A TASK IS. Two MuSiQue test tasks A and B, both questions stated
("Answer both questions. (1) <Q_first> (2) <Q_second>"), the union of both paragraph pools
(identical text once), and the union of both gold need-graphs with every node id prefixed per
constituent (`a_`, `b_`) and no edge between them -- so the composed graph has exactly two
facets, one per constituent, and every node keeps the depth it had alone. Covering the second
question's chain is horizontal proactivity (a need the task states but the first chain never
leads to); each chain's depth is vertical. Both question orders are separate tasks
(`<pair_id>_ab`, `<pair_id>_ba`) sharing one pool and one pair id. Declared before any data in
artifacts/composed_pairs_20260923/DECLARATION.md.

WHO MAY BE A CONSTITUENT, in the order the filters apply (each exclusion counted by reason):
  not_test_split            `split_of("musique", id, template)` is not "test" -- the SAME call
                            `pi run --split test` makes (`harness_is_test`)
  no_gold_graph             no graph for the id in the source gold
  in_training_ids           in the checkpoint's training id set (the rung-2 union,
                            train_id_set_hash d6750e34..., data/rl/notdone_v2/train_ids.*.txt)
  constituent_facets_not_1  its graph has other than one facet, so the composed graph could not
                            have exactly two
  duplicate_template        its hop set (MusiqueSuite.template_id) is another candidate's
  duplicate_answer          its normalized answer is another candidate's
Deduplication keeps the first task in (-max depth, id) order: the pairing rule needs deep
constituents, and the order uses no outcome of any run.

HOW TASKS ARE PAIRED. Disjoint pairs, deterministic under `seed`: candidates are shuffled once;
each shallow task (max depth < 2) takes the first compatible deep task in that order, then the
remaining deep tasks pair among themselves the same way. Shallow-with-shallow is never legal,
so taking shallow tasks first is what lets all of them be used. A pair is refused when:
  shared_title          the two pools share a paragraph title
  answer_in_other_task  either answer (or alias) appears, as a whole word, in the other task's
                        question or pool
  no_deep_constituent   neither constituent has gold depth >= 2
  shared_template       the two share a hop set (impossible after dedup; checked anyway)
`pair_rule_census` counts each rule over EVERY candidate pair, independent of matching order.

THE UID REMAP, AND WHY IT FAILS LOUDLY. EvidenceUnit.uid is h(corpus_id, doc_id, span) and the
composed corpus has its own corpus_id and its own doc ids (`<task>:<idx>`), so every gold uid
must be re-minted. A uid that failed to map would not raise anywhere downstream: it would make
its node unresolvable and read as the policy retrieving nothing, i.e. coverage ~0 for a plumbing
reason. So the build raises `UidRemapError` unless 100% of gold uids map to a pool paragraph
AND every remapped uid is one the ADAPTER (`MusiqueX2Suite.units`) actually emits.

CANARIES. The composed gold gets its own nonces through `write_graphs`, registered under the
output root; the constituents' musique nonces are stripped (`GoldGraph.answer`) and never copied.
The corpus side carries none, and the build refuses a corpus payload containing one.

OUTPUT, to an isolated root (never the shared data/):
  <root>/data/corpora/musique_x2/<hash>/tasks.jsonl       public
  <root>/data/gold/graphs/musique_x2/v1.jsonl             gold
  <root>/data/gold/graphs/musique_x2/constituents.jsonl   gold-side sidecar (answers)
  <root>/data/gold/graphs/musique_x2/build_manifest.json  counts, exclusions, input digests
  <root>/data/canaries/canaries.txt                       the composed gold's nonces

    python -m pi_eval.build.compose_build \\
        --musique-corpus <main>/data/corpora/musique/38f5afb69fb7ea18/tasks.jsonl \\
        --musique-gold <main>/data/gold/graphs/musique/v1.jsonl \\
        --train-ids <main>/data/rl/notdone_v2/train_ids.d6750e34239f398f.txt \\
        --root /private/tmp/musique_x2_root
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from pi_eval.build.common import (
    finalize_graph,
    read_graphs,
    sha256_file,
    unit_uid,
    write_corpus,
    write_graphs,
)
from pi_eval.build.musique_build import CORPUS_ID as SRC_CORPUS_ID
from pi_eval.build.musique_build import SUITE as SRC_SUITE
from pi_eval.canary import PREFIX as CANARY_PREFIX
from pi_eval.gold import GoldGraph

SUITE = "musique_x2"
CORPUS_ID = "musique_x2_v1"
GRAPH_VERSION = "v1"
QUESTION_TEMPLATE = "Answer both questions. (1) {first} (2) {second}"
ANSWER_TEMPLATE = "(1) {first} (2) {second}"
DEFAULT_SEED = 20260923
DEFAULT_ROOT = Path("/private/tmp/musique_x2_root")
# The recipe's training id set: rung 2's row carries the UNION over its lineage
# (conf/checkpoints.json `qwen3-8b-dpo-stacked-notdone-both.train_id_set_hash`).
TRAIN_ID_SET_HASH = "d6750e34239f398f0bb912bd74729ab22b11d8f8b7e762b5d8a7035038a22f7c"
MIN_DEEP = 2
PREFIX_A, PREFIX_B = "a_", "b_"

EXCLUSION_REASONS = (
    "not_test_split",
    "no_gold_graph",
    "in_training_ids",
    "constituent_facets_not_1",
    "duplicate_template",
    "duplicate_answer",
)
PAIR_RULES = ("shared_title", "answer_in_other_task", "no_deep_constituent", "shared_template")


class ComposeError(RuntimeError):
    """The inputs cannot produce the declared suite."""


class UidRemapError(ComposeError):
    """A gold evidence uid did not map to the composed corpus, or maps to a uid the adapter
    never emits. Either would read as coverage ~0, not as an error, if it were not raised."""


# ------------------------------------------------------------------ the training id set


def id_set_hash(ids: Iterable[str]) -> str:
    """`pinq_train.split.id_set_hash`, restated: nothing may import pinq_train (contract 4).
    tests/test_compose_build.py pins the two against each other."""
    return hashlib.sha256("\n".join(sorted(set(ids))).encode()).hexdigest()


def read_train_ids(path: Path, *, expect_set_hash: str | None) -> frozenset[str]:
    """`suite/task_id` lines. Refused unless their set hash is the pinned one, so a stale or
    wrong export cannot quietly stand in for the checkpoint's own training set."""
    ids = frozenset(ln.strip() for ln in Path(path).read_text().splitlines() if ln.strip())
    got = id_set_hash(ids)
    if expect_set_hash is not None and got != expect_set_hash:
        raise ComposeError(
            f"train id file {path} has set hash {got[:16]}..., expected {expect_set_hash[:16]}... "
            "-- it is not the checkpoint's training id set"
        )
    return ids


def harness_is_test(suite: Any) -> Callable[[str], bool]:
    """The split `pi run --split test` applies: `split_of` with the adapter's template id
    (pi_run.cli._select_task_ids -> pi_run.manifest.template_id_of)."""
    from pinq.splitting import split_of

    def is_test(tid: str) -> bool:
        tpl = suite.template_id(tid)
        return split_of(SRC_SUITE, tid, str(tpl) if tpl else None) == "test"

    return is_test


# ------------------------------------------------------------------ constituents


@dataclass(frozen=True)
class Constituent:
    tid: str
    question: str
    paragraphs: tuple[Mapping[str, Any], ...]
    graph: GoldGraph
    answer: str
    answer_strings: tuple[str, ...]
    template: str
    max_depth: int

    @property
    def n_facets(self) -> int:
        return len(self.graph.gold_facets)

    @property
    def n_gold_uids(self) -> int:
        return len({u for n in self.graph.gold_nodes for u in n.gold_ev_uids})


def _norm(s: str) -> str:
    return " ".join(str(s).lower().split())


def _mentions(needle: str, hay: str) -> bool:
    """Whole-word, case-insensitive: "Oslo" is in "Nordmus is in Oslo." and not in "Oslon"."""
    n = _norm(needle)
    if not n:
        return False
    return re.search(rf"(?<!\w){re.escape(n)}(?!\w)", _norm(hay)) is not None


def _constituent(rec: Mapping[str, Any], graph: GoldGraph, template: str) -> Constituent:
    answer = graph.answer  # canary stripped; the source nonce never enters the composed gold
    strings = tuple(s for s in (answer, *graph.gold_aliases) if str(s).strip())
    depths = [n.gold_depth for n in graph.gold_nodes if n.gold_depth is not None]
    return Constituent(
        tid=str(rec["id"]),
        question=str(rec["question"]),
        paragraphs=tuple(rec["paragraphs"]),
        graph=graph,
        answer=answer,
        answer_strings=strings,
        template=template,
        max_depth=max(depths) if depths else -1,
    )


def select(
    records: Sequence[Mapping[str, Any]],
    graphs: Mapping[str, GoldGraph],
    *,
    is_test: Callable[[str], bool],
    train_ids: frozenset[str],
    template_of: Callable[[str], str | None],
) -> tuple[list[Constituent], dict[str, int], dict[str, list[str]]]:
    """The candidate constituents, and every exclusion by reason (with ids)."""
    counts = dict.fromkeys(EXCLUSION_REASONS, 0)
    who: dict[str, list[str]] = {r: [] for r in EXCLUSION_REASONS}

    def drop(reason: str, tid: str) -> None:
        counts[reason] += 1
        who[reason].append(tid)

    kept: list[Constituent] = []
    for rec in sorted(records, key=lambda r: str(r["id"])):
        tid = str(rec["id"])
        if not is_test(tid):
            drop("not_test_split", tid)
            continue
        g = graphs.get(tid)
        if g is None:
            drop("no_gold_graph", tid)
            continue
        if f"{SRC_SUITE}/{tid}" in train_ids:
            drop("in_training_ids", tid)
            continue
        c = _constituent(rec, g, str(template_of(tid) or tid))
        if c.n_facets != 1:
            drop("constituent_facets_not_1", tid)
            continue
        kept.append(c)

    out: list[Constituent] = []
    seen_tpl: set[str] = set()
    seen_ans: set[str] = set()
    for c in sorted(kept, key=lambda c: (-c.max_depth, c.tid)):
        if c.template in seen_tpl:
            drop("duplicate_template", c.tid)
            continue
        if _norm(c.answer) in seen_ans:
            drop("duplicate_answer", c.tid)
            continue
        seen_tpl.add(c.template)
        seen_ans.add(_norm(c.answer))
        out.append(c)
    return sorted(out, key=lambda c: c.tid), counts, who


# ------------------------------------------------------------------ pairing


def _pool_text(c: Constituent) -> str:
    return "\n".join(f"{p['title']}\n{p['text']}" for p in c.paragraphs)


def pair_violations(a: Constituent, b: Constituent) -> list[str]:
    """Every pairing rule the two violate, in PAIR_RULES order. Empty means they may pair."""
    out: list[str] = []
    if {_norm(p["title"]) for p in a.paragraphs} & {_norm(p["title"]) for p in b.paragraphs}:
        out.append("shared_title")
    if any(
        _mentions(s, x.question) or _mentions(s, _pool_text(x))
        for s_owner, x in ((a, b), (b, a))
        for s in s_owner.answer_strings
    ):
        out.append("answer_in_other_task")
    if max(a.max_depth, b.max_depth) < MIN_DEEP:
        out.append("no_deep_constituent")
    if a.template == b.template:
        out.append("shared_template")
    return out


def match_pairs(
    cands: Sequence[Constituent], *, seed: int
) -> list[tuple[Constituent, Constituent]]:
    """Disjoint pairs, deterministic under `seed`; each pair returned (A, B) with A.tid < B.tid."""
    order = sorted(cands, key=lambda c: c.tid)
    random.Random(seed).shuffle(order)
    deep = [c for c in order if c.max_depth >= MIN_DEEP]
    shallow = [c for c in order if c.max_depth < MIN_DEEP]
    used: set[str] = set()
    pairs: list[tuple[Constituent, Constituent]] = []

    def take(x: Constituent, pool: Sequence[Constituent]) -> None:
        for y in pool:
            if y.tid in used or y.tid == x.tid:
                continue
            if not pair_violations(x, y):
                pairs.append(tuple(sorted((x, y), key=lambda c: c.tid)))  # type: ignore[arg-type]
                used.update({x.tid, y.tid})
                return

    for s in shallow:
        take(s, deep)
    for d in deep:
        if d.tid not in used:
            take(d, deep)
    return pairs


def pair_rule_census(cands: Sequence[Constituent]) -> dict[str, int]:
    out = dict.fromkeys(PAIR_RULES, 0)
    out["legal"] = 0
    for i, a in enumerate(cands):
        for b in cands[i + 1 :]:
            v = pair_violations(a, b)
            for r in v:
                out[r] += 1
            out["legal"] += not v
    return out


def pair_id_of(a: Constituent, b: Constituent) -> str:
    x, y = sorted((a.tid, b.tid))
    return "x2_" + hashlib.sha256(f"{x}|{y}".encode()).hexdigest()[:12]


# ------------------------------------------------------------------ composition


def union_pool(
    a: Constituent, b: Constituent, pid: str
) -> tuple[list[tuple[str, str]], dict[tuple[str, int], int], int]:
    """(pool as (title, text), (constituent id, source idx) -> composed idx, n deduplicated).

    Identical TEXT appears once, under the title of its first occurrence (A before B). The pool
    order is a shuffle seeded by the pair id, so it is the same for both question orders and
    does not reveal which paragraph came from which constituent.
    """
    items: list[tuple[str, str]] = []
    at: dict[str, int] = {}
    where: dict[tuple[str, int], int] = {}
    dup = 0
    for c in (a, b):
        for p in c.paragraphs:
            text = str(p["text"])
            if text in at:
                dup += 1
            else:
                at[text] = len(items)
                items.append((str(p["title"]), text))
            where[(c.tid, int(p["idx"]))] = at[text]
    order = list(range(len(items)))
    random.Random(int(hashlib.sha256(pid.encode()).hexdigest(), 16)).shuffle(order)
    new_of = {old: new for new, old in enumerate(order)}
    return [items[old] for old in order], {k: new_of[v] for k, v in where.items()}, dup


def remap_uids(c: Constituent, tid: str, where: Mapping[tuple[str, int], int]) -> dict[str, str]:
    """Source gold uid -> composed uid, for every paragraph of the constituent's pool."""
    out: dict[str, str] = {}
    for p in c.paragraphs:
        idx, text = int(p["idx"]), str(p["text"])
        out[unit_uid(SRC_CORPUS_ID, c.tid, idx, text)] = unit_uid(
            CORPUS_ID, tid, where[(c.tid, idx)], text
        )
    return out


def compose_graph(
    a: Constituent,
    b: Constituent,
    tid: str,
    where: Mapping[tuple[str, int], int],
    answer: str,
) -> tuple[dict, int]:
    """The composed gold row (finalize_graph's output) and the number of gold uids remapped.

    Raises UidRemapError for any source uid with no pool paragraph behind it.
    """
    nodes, edges, seeds = [], [], []
    n_uids = 0
    for prefix, c in ((PREFIX_A, a), (PREFIX_B, b)):
        table = remap_uids(c, tid, where)
        for n in c.graph.gold_nodes:
            missing = [u for u in n.gold_ev_uids if u not in table]
            if missing:
                raise UidRemapError(
                    f"{c.tid} node {n.gold_node_id}: {len(missing)} gold uid(s) match no "
                    f"paragraph of its pool (first {missing[0][:16]}...); the composed node "
                    "would be unresolvable and read as coverage 0"
                )
            n_uids += len(n.gold_ev_uids)
            nodes.append(
                replace(
                    n,
                    gold_suite=SUITE,
                    gold_task_key=tid,
                    gold_node_id=prefix + n.gold_node_id,
                    gold_ev_uids=tuple(table[u] for u in n.gold_ev_uids),
                    gold_depth=None,
                    gold_facet_id=None,
                    gold_graph_version=GRAPH_VERSION,
                )
            )
        for e in c.graph.gold_edges:
            edges.append(
                replace(
                    e,
                    gold_suite=SUITE,
                    gold_task_key=tid,
                    gold_src_node_id=prefix + e.gold_src_node_id,
                    gold_dst_node_id=prefix + e.gold_dst_node_id,
                    gold_graph_version=GRAPH_VERSION,
                )
            )
        seeds += [prefix + s for s in c.graph.gold_seed_node_ids]
    row = finalize_graph(
        suite=SUITE,
        task_key=tid,
        nodes=nodes,
        edges=edges,
        seed_ids=seeds,
        answer=answer,
        aliases=(),
        version=GRAPH_VERSION,
    )
    return row, n_uids


def structure_problems(row: Mapping[str, Any], a: Constituent, b: Constituent) -> list[str]:
    """The declared structure of a composed graph, checked on the finalized row."""
    out: list[str] = []
    for prefix, c in ((PREFIX_A, a), (PREFIX_B, b)):
        want = {n.gold_node_id: n.gold_depth for n in c.graph.gold_nodes}
        got = {
            n["gold_node_id"][len(prefix) :]: n["gold_depth"]
            for n in row["gold_nodes"]
            if n["gold_node_id"].startswith(prefix)
        }
        if got != want:
            out.append(f"depths changed for constituent {c.tid}")
    facets = row["gold_facets"]
    owners = []
    for f in facets:
        pre = {n.split("_", 1)[0] for n in f["gold_node_ids"]}
        owners.append(pre.pop() if len(pre) == 1 else "mixed")
    if len(facets) != 2 or sorted(owners) != ["a", "b"]:
        out.append(f"facets {len(facets)} owned by {sorted(owners)}, expected one each of a, b")
    for e in row["gold_edges"]:
        if e["gold_src_node_id"][:2] != e["gold_dst_node_id"][:2]:
            out.append(
                f"edge crosses constituents: {e['gold_src_node_id']}->{e['gold_dst_node_id']}"
            )
    return out


# ------------------------------------------------------------------ build


@dataclass(frozen=True)
class ComposeResult:
    root: Path
    corpus: Path
    gold: Path
    sidecar: Path
    manifest_path: Path
    corpus_hash: str
    n_tasks: int
    manifest: dict


def _display(path: Path) -> str:
    """A path as recorded: from `data/` on when it has one, so a record copied into the repo
    names no home directory."""
    s = str(path)
    i = s.find("/data/")
    return s[i + 1 :] if i >= 0 else s


def build(
    *,
    musique_corpus: Path,
    musique_gold: Path,
    root: Path = DEFAULT_ROOT,
    seed: int = DEFAULT_SEED,
    is_test: Callable[[str], bool] | None = None,
    train_ids: frozenset[str] | None = None,
    train_ids_path: Path | None = None,
    expect_train_id_set_hash: str | None = TRAIN_ID_SET_HASH,
) -> ComposeResult:
    from pinq_adapters.musique.suite import MusiqueSuite
    from pinq_adapters.musique_x2.suite import MusiqueX2Suite

    musique_corpus, musique_gold, root = Path(musique_corpus), Path(musique_gold), Path(root)
    if train_ids is None:
        if train_ids_path is None:
            raise ComposeError("no training id set given: refusing to build without the check")
        train_ids = read_train_ids(train_ids_path, expect_set_hash=expect_train_id_set_hash)

    records = [json.loads(x) for x in musique_corpus.read_text().splitlines() if x.strip()]
    graphs = {g.gold_task_key: g for g in read_graphs(musique_gold)}
    stale = sorted(
        {
            g.gold_corpus_hash
            for g in graphs.values()
            if g.gold_corpus_hash and g.gold_corpus_hash != musique_corpus.parent.name
        }
    )
    if stale:
        raise ComposeError(
            f"source gold describes corpus {stale}, not {musique_corpus.parent.name}: the uids "
            "would not match the pools"
        )
    src_suite = MusiqueSuite(musique_corpus.parent)
    cands, excl, excluded_ids = select(
        records,
        graphs,
        is_test=is_test or harness_is_test(src_suite),
        train_ids=frozenset(train_ids),
        template_of=src_suite.template_id,
    )
    census = pair_rule_census(cands)
    pairs = match_pairs(cands, seed=seed)
    paired = {c.tid for p in pairs for c in p}

    public: list[dict] = []
    rows: list[dict] = []
    side: list[dict] = []
    n_dedup = n_uids = 0
    composed_excluded: list[str] = []
    for a, b in pairs:
        pid = pair_id_of(a, b)
        pool, where, dup = union_pool(a, b, pid)
        pair_rows, pair_public, pair_side, pair_uids = [], [], [], 0
        problems: list[str] = []
        for order, first, second in (("ab", a, b), ("ba", b, a)):
            tid = f"{pid}_{order}"
            answer = ANSWER_TEMPLATE.format(first=first.answer, second=second.answer)
            row, k = compose_graph(a, b, tid, where, answer)
            problems += structure_problems(row, a, b)
            pair_uids += k
            pair_rows.append(row)
            pair_public.append(
                {
                    "id": tid,
                    "question": QUESTION_TEMPLATE.format(
                        first=first.question, second=second.question
                    ),
                    "paragraphs": [
                        {"idx": i, "title": t, "text": x} for i, (t, x) in enumerate(pool)
                    ],
                }
            )
            pair_side.append(
                {
                    "task_id": tid,
                    "pair_id": pid,
                    "order": order,
                    "a_id": a.tid,
                    "b_id": b.tid,
                    "first_id": first.tid,
                    "second_id": second.tid,
                    "answers": {"a": a.answer, "b": b.answer},
                    "max_depth": {"a": a.max_depth, "b": b.max_depth},
                    "n_nodes": {"a": len(a.graph.gold_nodes), "b": len(b.graph.gold_nodes)},
                    "n_gold_uids": {"a": a.n_gold_uids, "b": b.n_gold_uids},
                    "depth_hist": {
                        "a": {str(k): v for k, v in a.graph.depth_histogram().items()},
                        "b": {str(k): v for k, v in b.graph.depth_histogram().items()},
                    },
                    "n_paragraphs": len(pool),
                }
            )
        if problems:
            # Exclude and count, never repair: a pair whose union is not two clean facets is
            # not the task the declaration describes.
            composed_excluded.append(pid)
            continue
        n_dedup += dup
        n_uids += pair_uids
        rows += pair_rows
        public += pair_public
        side += pair_side

    payload = "\n".join(json.dumps(r, sort_keys=True, ensure_ascii=False) for r in public)
    if CANARY_PREFIX in payload:
        raise ComposeError("a canary nonce reached the composed CORPUS; the corpus must carry none")
    corpus, chash = write_corpus(root, SUITE, public)

    # THE ADAPTER HALF OF THE REMAP CHECK: every remapped uid must be one the view side emits.
    adapter = MusiqueX2Suite(corpus.parent)
    n_in_units = 0
    for row in rows:
        units = {u.uid for u in adapter.units(row["gold_task_key"])}
        for n in row["gold_nodes"]:
            for u in n["gold_ev_uids"]:
                if u not in units:
                    raise UidRemapError(
                        f"{row['gold_task_key']} node {n['gold_node_id']}: remapped uid "
                        f"{u[:16]}... is not emitted by the adapter's units() -- builder and "
                        f"adapter disagree on the uid convention (corpus_id {CORPUS_ID!r} vs "
                        f"{adapter.corpus_id!r})"
                    )
                n_in_units += 1
    if n_in_units != n_uids:
        raise UidRemapError(f"remapped {n_uids} gold uids but the adapter check saw {n_in_units}")

    gold = write_graphs(root, SUITE, GRAPH_VERSION, rows, corpus_hash=chash)
    gdir = gold.parent
    sidecar = gdir / "constituents.jsonl"
    sidecar.write_text("\n".join(json.dumps(r, sort_keys=True) for r in side) + "\n")

    manifest = {
        "suite": SUITE,
        "corpus_id": CORPUS_ID,
        "graph_version": GRAPH_VERSION,
        "corpus_hash": chash,
        "seed": seed,
        "question_template": QUESTION_TEMPLATE,
        "answer_template": ANSWER_TEMPLATE,
        "inputs": {
            "musique_corpus": _display(musique_corpus),
            "musique_corpus_sha256": sha256_file(musique_corpus),
            "musique_gold": _display(musique_gold),
            "musique_gold_sha256": sha256_file(musique_gold),
            "train_ids": _display(train_ids_path) if train_ids_path else None,
            "train_ids_sha256": sha256_file(train_ids_path) if train_ids_path else None,
            "train_id_set_hash": id_set_hash(train_ids),
            "n_train_ids": len(train_ids),
            "n_train_ids_musique": sum(1 for t in train_ids if t.startswith(f"{SRC_SUITE}/")),
        },
        "n_source_tasks": len(records),
        "exclusions": excl,
        "excluded_ids": excluded_ids,
        "n_candidates": len(cands),
        "candidates_by_max_depth": dict(sorted(Counter(str(c.max_depth) for c in cands).items())),
        "pairing_rule": (
            "candidates shuffled once under `seed`; each shallow task (max depth < 2) takes the "
            "first compatible deep task, then remaining deep tasks pair the same way"
        ),
        "pair_rule_census": census,
        "n_pairs": len(pairs) - len(composed_excluded),
        "pairs_by_depth_class": dict(
            sorted(
                Counter(
                    "deep+deep" if min(a.max_depth, b.max_depth) >= MIN_DEEP else "deep+shallow"
                    for a, b in pairs
                    if pair_id_of(a, b) not in composed_excluded
                ).items()
            )
        ),
        "unpaired_candidates": sorted(c.tid for c in cands if c.tid not in paired),
        "pairs_excluded_composed_structure": composed_excluded,
        "n_tasks": len(public),
        "n_paragraphs_deduplicated": n_dedup,
        "uid_remap": {
            "n_gold_uids": n_uids,
            "n_mapped": n_uids,
            "n_in_adapter_units": n_in_units,
        },
    }
    manifest_path = gdir / "build_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return ComposeResult(
        root=root,
        corpus=corpus,
        gold=gold,
        sidecar=sidecar,
        manifest_path=manifest_path,
        corpus_hash=chash,
        n_tasks=len(public),
        manifest=manifest,
    )


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--musique-corpus", type=Path, required=True)
    ap.add_argument("--musique-gold", type=Path, required=True)
    ap.add_argument("--train-ids", type=Path, required=True)
    ap.add_argument("--train-id-set-hash", default=TRAIN_ID_SET_HASH)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    a = ap.parse_args(argv)
    try:
        res = build(
            musique_corpus=a.musique_corpus,
            musique_gold=a.musique_gold,
            root=a.root,
            seed=a.seed,
            train_ids_path=a.train_ids,
            expect_train_id_set_hash=a.train_id_set_hash,
        )
    except ComposeError as exc:
        print(f"compose_build: REFUSING: {exc}", file=sys.stderr)
        return 2
    m = res.manifest
    print(
        json.dumps(
            {
                k: m[k]
                for k in (
                    "corpus_hash",
                    "n_source_tasks",
                    "exclusions",
                    "n_candidates",
                    "candidates_by_max_depth",
                    "pair_rule_census",
                    "n_pairs",
                    "pairs_by_depth_class",
                    "unpaired_candidates",
                    "pairs_excluded_composed_structure",
                    "n_tasks",
                    "n_paragraphs_deduplicated",
                    "uid_remap",
                )
            },
            indent=1,
            sort_keys=True,
        )
    )
    print(f"corpus  {res.corpus}\ngold    {res.gold}\nsidecar {res.sidecar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
