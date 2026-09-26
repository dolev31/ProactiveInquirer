"""2WikiMultihopQA -> a PUBLIC paragraph corpus and a GOLD need graph.

READ THIS FIRST: THIS IS AN ENTITY GRAPH, NOT A SUB-QUESTION GRAPH.

MuSiQue and StrategyQA annotate what a human would ASK. 2Wiki annotates what a human would
LOOK UP: `evidences` is a list of Wikidata triples

    ["Polish-Russian War", "director", "Xawery Zulawski"]
    ["Xawery Zulawski", "mother", "Malgorzata Braunek"]

so a node here is "the value of relation R for entity E", not "the question a policy would
utter". The two are close enough to score against — a policy that asks "who directed
Polish-Russian War?" has manifestly discovered the first triple — but they are not the same
object, and the gap is where an LLM matcher on this suite will disagree with the mechanical
one. Any RNR_ask number from this suite therefore reads as an approximation, while
RNR_resolve (which is uid set inclusion and knows nothing about phrasing) is exact.

EDGES COME FROM OBJECT -> SUBJECT LINKAGE, GATED ON `type`
    compositional / inference / bridge_comparison   chain: triple i's OBJECT is triple j's
        SUBJECT, so j cannot be looked up until i is known. This is a genuine prerequisite
        and it is mechanical — no model, no annotation, just string identity under the same
        title normalisation used to locate paragraphs (2Wiki writes "Charles Saunders" as an
        object and "Charles Saunders (director)" as a subject, so raw equality would miss
        roughly a third of bridge_comparison chains).
    comparison   FLAT, by construction. "Which film came out first, A or B?" needs both
        publication dates and neither depends on the other. Every node is a depth-0 seed and
        the task has no facets, which is the correct shape rather than a missing one.

Linkage is restricted to i < j, the order the annotators wrote the reasoning in. That makes
cycles impossible; a cycle would leave Depth undefined for every node in it.

WHAT IS PUBLIC AND WHAT IS NOT
    public   _id, question, ten paragraphs {idx, title, text}
    gold     answer, `evidences`, `supporting_facts`, `type`

`type` is gold and not a hint: telling a policy in advance that its question is a comparison
would hand it the shape of the graph it is supposed to discover.

SENTENCE IDS ARE ROLLED UP TO PARAGRAPHS. `supporting_facts` is [title, sent_id], but the
retriever returns paragraphs, and a gold uid the retriever can never emit is a node that can
never be resolved. Granularity is a property of the retrieval interface, so gold follows it
rather than the other way round.

Licence: Apache-2.0 (Ho et al., COLING 2020). Source: HF `xanhho/2WikiMultihopQA`, whose
parquet stores context/supporting_facts/evidences as JSON strings; `materialize()` converts
a split back to the authors' original JSON shape and everything downstream reads that.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from pi_eval.build.common import (
    BuildResult,
    finalize_graph,
    require_raw,
    unit_uid,
    write_corpus,
    write_graphs,
)
from pi_eval.gold import GoldEdge, GoldNode

SUITE = "wiki2"
CORPUS_ID = "wiki2_v1"
GRAPH_VERSION = "v1"

_HF = "https://huggingface.co/datasets/xanhho/2WikiMultihopQA/resolve/main/"
SHA256 = {
    "train.parquet": "a0f464f1604edf40ad990b605c61c03d56911c106d09ad32becf6afcf3201d52",
    "dev.parquet": "c0d8b60b9026b728fb07ad74c5252a0f188f6942e8ba5c02df4dfa369502ea8d",
}
_JSON_COLUMNS = ("context", "supporting_facts", "evidences")

# The reasoning types whose evidence triples form a chain. Upstream writes
# "bridge_comparison"; the paper writes "bridge-comparison".
CHAIN_TYPES = frozenset({"compositional", "inference", "bridge_comparison"})

_PARENS = re.compile(r"\s*\([^)]*\)")
_NONWORD = re.compile(r"[^a-z0-9]+")


def _key(title: str) -> str:
    """Normalise an entity string for matching: lowercase, drop a trailing disambiguator.

    "Charles Saunders (director)" and "Charles Saunders" are the same entity written twice,
    and 2Wiki uses both forms inside a single example.
    """
    return _NONWORD.sub(" ", _PARENS.sub("", title).lower()).strip()


# ------------------------------------------------------------------ raw loading


def materialize(
    split: str,
    raw_dir: Path,
    *,
    allow_download: bool = True,
    verify: bool = True,
    limit: int | None = None,
):
    """Return the split's records, converting parquet -> the authors' JSON shape once.

    pyarrow is needed only for that one conversion, and only on a machine that is doing the
    download. The cached `<split>.jsonl` and the committed fixture are plain JSON lines, so
    an offline build — and all of CI — needs nothing beyond the standard library.
    """
    for name in (f"{split}.jsonl", f"{split}.json"):
        path = raw_dir / name
        if path.exists():
            text = path.read_text()
            if name.endswith(".jsonl"):
                return [json.loads(x) for x in text.splitlines() if x.strip()]
            return json.loads(text)

    parquet = require_raw(
        raw_dir / f"{split}.parquet",
        _HF + f"{split}.parquet",
        expect_sha256=SHA256.get(f"{split}.parquet") if verify else None,
        allow_download=allow_download,
    )
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - exercised only on a cold download
        raise ImportError(
            "reading the upstream parquet needs pyarrow (pip install '.[analysis]'). "
            f"Alternatively drop a converted {split}.jsonl into {raw_dir}."
        ) from exc

    rows: list[dict] = []
    for batch in pq.ParquetFile(parquet).iter_batches(batch_size=2048):
        for r in batch.to_pylist():
            rows.append({k: (json.loads(v) if k in _JSON_COLUMNS else v) for k, v in r.items()})
        if limit is not None and len(rows) >= limit:
            # A TRUNCATED CONVERSION IS NEVER CACHED. Writing <split>.jsonl here would make
            # every subsequent full build silently read the first 25 rows and report a
            # complete corpus, which is the kind of bug that only surfaces in a results table.
            return rows[:limit]
    out = raw_dir / f"{split}.jsonl"
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    return rows


def _rows(value):
    """Upstream ships context/supporting_facts/evidences as lists of lists; some HF exports
    ship them column-wise as {"title": [...], "content": [...]}. Accept both."""
    if isinstance(value, dict):
        cols = list(value.values())
        return [list(t) for t in zip(*cols)]
    return [list(v) for v in value]


# ------------------------------------------------------------------ build


def build(
    *,
    split: str = "dev",
    raw_dir: Path | None = None,
    root: Path | None = None,
    limit: int | None = None,
    allow_download: bool = True,
    verify: bool = True,
) -> BuildResult:
    root = root or Path.cwd()
    raw_dir = raw_dir or (root / "data" / "raw" / SUITE)
    records = materialize(split, raw_dir, allow_download=allow_download, verify=verify, limit=limit)

    public: list[dict] = []
    graphs: list[dict] = []
    for rec in records:
        tid = rec.get("_id") or rec["id"]
        context = _rows(rec["context"])
        paragraphs = [
            {"idx": i, "title": title, "text": " ".join(s.strip() for s in sents)}
            for i, (title, sents) in enumerate(context)
        ]
        public.append({"id": tid, "question": rec["question"], "paragraphs": paragraphs})
        graphs.append(_graph(tid, rec, paragraphs))
        if limit is not None and len(public) >= limit:
            break

    corpus, chash = write_corpus(root, SUITE, public)
    gold = write_graphs(root, SUITE, GRAPH_VERSION, graphs, corpus_hash=chash)
    return BuildResult(corpus, gold, chash, len(public), 0)


def _locate(entity: str, paragraphs: list[dict], preferred: set[str]) -> int | None:
    """Index of the paragraph describing `entity`, or None.

    Tried in decreasing strictness, and paragraphs named by `supporting_facts` win ties: the
    annotators already told us which ten-of-ten paragraphs carry the answer, so a substring
    coincidence with a distractor must never beat one of them. Roughly 3.5% of dev triples
    match nothing at all; those nodes carry no evidence and are marked UNTESTABLE rather
    than being attached to a guess.
    """
    ek = _key(entity)
    order = sorted(
        range(len(paragraphs)),
        key=lambda i: (paragraphs[i]["title"] not in preferred, i),
    )
    for exact in (True, False):
        for i in order:
            tk = _key(paragraphs[i]["title"])
            if (tk == ek) if exact else (ek and (ek in tk or tk in ek)):
                return i
    return None


def _graph(tid: str, rec: dict, paragraphs: list[dict]) -> dict:
    triples = [t for t in _rows(rec.get("evidences", [])) if len(t) >= 3]
    preferred = {row[0] for row in _rows(rec.get("supporting_facts", []))}
    qtype = str(rec.get("type", "")).replace("-", "_").lower()

    nodes: list[GoldNode] = []
    subjects: list[str] = []
    objects: list[str] = []
    for i, (subj, rel, obj) in enumerate((t[0], t[1], t[2]) for t in triples):
        nid = f"e{i + 1}"
        subjects.append(_key(subj))
        objects.append(_key(obj))
        idx = _locate(subj, paragraphs, preferred)
        ev = () if idx is None else (unit_uid(CORPUS_ID, tid, idx, paragraphs[idx]["text"]),)
        nodes.append(
            GoldNode(
                gold_suite=SUITE,
                gold_task_key=tid,
                gold_node_id=nid,
                # Written in MuSiQue's "entity >> relation" form so one matcher prompt and
                # one set of calibration thresholds serve both suites.
                gold_text=f"{subj} >> {rel}",
                gold_aliases=(obj,) if obj else (),
                gold_kind="fact",
                gold_provenance=("bench_author",),
                gold_provenance_primary="bench_author",
                gold_partition="required" if ev else "optional",
                gold_discoverability="kb" if ev else "unknown",
                gold_ablation_verdict="NECESSARY" if ev else "UNTESTABLE",
                gold_ev_uids=ev,
                gold_confidence=1.0 if ev else 0.0,
                gold_graph_version=GRAPH_VERSION,
            )
        )

    edges: list[GoldEdge] = []
    if qtype in CHAIN_TYPES:
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                if objects[i] and objects[i] == subjects[j]:
                    edges.append(
                        GoldEdge(
                            gold_suite=SUITE,
                            gold_task_key=tid,
                            gold_src_node_id=nodes[i].gold_node_id,
                            gold_dst_node_id=nodes[j].gold_node_id,
                            gold_edge_kind="prerequisite",
                            gold_verified="mechanical",
                            gold_provenance="mechanical",
                            gold_confidence=1.0,
                            gold_graph_version=GRAPH_VERSION,
                        )
                    )

    has_parent = {e.gold_dst_node_id for e in edges}
    seeds = [n.gold_node_id for n in nodes if n.gold_node_id not in has_parent]
    return finalize_graph(
        suite=SUITE,
        task_key=tid,
        nodes=nodes,
        edges=edges,
        seed_ids=seeds,
        answer=rec.get("answer", ""),
        aliases=(),
        version=GRAPH_VERSION,
    )
