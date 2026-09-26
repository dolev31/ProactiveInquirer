"""MuSiQue-Ans -> a PUBLIC paragraph corpus and a GOLD need graph.

MuSiQue is the one real suite whose dependency structure is not inferred, mined or elicited:
it was CONSTRUCTED by composing single-hop questions, and the composition recipe ships with
every example. `question_decomposition` is an ordered list of

    {id, question, answer, paragraph_support_idx}

and a sub-question that reads "When was #1 founded?" contains a literal, human-authored
statement that this step is UNANSWERABLE until step 1 has been answered. That is a
prerequisite edge with no annotation, no model and no judgement in it — we recover it with
one regex, which is why every edge here is gold_verified="mechanical" and confidence 1.0.

WHY THAT MATTERS. Every other real suite gives us edges by inference (2Wiki), by heuristic
(StrategyQA) or by mining (tau2). MuSiQue is the calibration set: a Depth or C@d bug shows
up here as an obviously wrong number against a graph a human wrote down on purpose, exactly
as it does on synth — with the difference that these are real questions over real text.

WHAT IS PUBLIC AND WHAT IS NOT
    public   id, question, the paragraph pool {idx, title, text} — 20 paragraphs for
             19,917 of the 19,938 train tasks, 16-19 for the remaining 21
    gold     answer, answer_aliases, the decomposition, the #N edges, is_supporting and
             paragraph_support_idx

`is_supporting` in particular must never reach data/corpora/: it labels the 2-4 paragraphs
that matter out of 20, so a record carrying it would let a policy skip discovery entirely
and would silently turn every retrieval metric into a measure of nothing.

Licence: CC BY 4.0. Source: HF `dgslibisey/MuSiQue` (musique-ANS v1.0, train 19,938 /
validation 2,417), which mirrors the authors' release byte-for-byte.
"""

from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path
from typing import Iterable, Mapping

from pi_eval.build.common import (
    BuildResult,
    finalize_graph,
    require_raw,
    unit_uid,
    write_corpus,
    write_graphs,
)
from pi_eval.gold import GoldEdge, GoldNode

SUITE = "musique"
CORPUS_ID = "musique_ans_v1p0"
GRAPH_VERSION = "v1"

_HF = "https://huggingface.co/datasets/dgslibisey/MuSiQue/resolve/main/"
FILES = {
    "train": "musique_ans_v1.0_train.jsonl",
    "dev": "musique_ans_v1.0_dev.jsonl",
}
# Hard pins: digests of the files this builder was written and verified against.
SHA256 = {
    "musique_ans_v1.0_train.jsonl": (
        "83a75b1e11e4e9bb8f8308e72ac40ca617ae4431b3a0d955b61cab259248490a"
    ),
    "musique_ans_v1.0_dev.jsonl": (
        "15fa63794d18a94ce12411aca6e2327e65b6e83b0b1490efab3f1962e48abf3b"
    ),
    "dev_test_singlehop_questions_v1.0.json": (
        "013be00ab914799891e5ae40de6cdc1baf03aaffd1f3ce4bf0de796020069613"
    ),
    "musique_v1.0.zip": "98f839bf2fd5319f5c688aed77901a6d5c30b3b9f9f691ab9a8ecafb045ee0cd",
}

EXCLUSION_NAME = "dev_test_singlehop_questions_v1.0.json"
# The authors publish the exclusion list only inside the full data zip, so that is what we
# fetch and unpack when the bare json is not already cached.
ZIP_NAME = "musique_v1.0.zip"
ZIP_URL = (
    "https://drive.usercontent.google.com/download"
    "?id=1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h&export=download&confirm=t"
)

REF_RE = re.compile(r"#(\d+)")


# ------------------------------------------------------------------ leakage exclusion list


def _norm_q(s: str) -> str:
    """Fold a single-hop question to a comparable key.

    THE FOLD THAT MATTERS is `[SEP]` -> `>>`. The seed datasets render a relation query as
    "Hurghada [SEP] capital of"; MuSiQue renders the identical question as
    "Hurghada >> capital of". Without folding them together the exclusion list matches
    NOTHING — measured: 0 of 19,938 train records — and the leakage guard becomes a silent
    no-op that still looks like it ran, which is the worst possible failure mode for a
    contamination control.

    With the fold it excludes 24 train records. The sanity check on the fold itself is the
    dev split: all 2,417 dev records match, which is exactly what a list *of* the dev/test
    single-hop questions must do.
    """
    s = s.lower().replace("[sep]", ">>")
    s = re.sub(r"[^a-z0-9>]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def load_exclusions(path: Path) -> frozenset[str]:
    """{seed_dataset: [{id, question}, ...]} -> normalized question keys.

    The `id` fields are SEED-dataset ids (SQuAD hashes, NQ integers, T-REx uuids) and share
    no namespace with MuSiQue's own `question_decomposition[*].id`, so matching by id would
    quietly match nothing. Text is the only join key that exists.
    """
    raw = json.loads(path.read_text())
    groups = raw.values() if isinstance(raw, dict) else [raw]
    out: set[str] = set()
    for group in groups:
        for item in group:
            q = item.get("question") if isinstance(item, dict) else item
            if isinstance(q, str) and q.strip():
                out.add(_norm_q(q))
    return frozenset(out)


def ensure_exclusions(raw_dir: Path, *, allow_download: bool = True, verify: bool = True) -> Path:
    """Materialise the exclusion list, unpacking it from the full data zip if need be.

    `verify=False` exists for the committed test fixture, which is a hand-written stand-in
    for the real list and therefore cannot carry the real list's digest.
    """
    want = SHA256[EXCLUSION_NAME] if verify else None
    bare = raw_dir / EXCLUSION_NAME
    if bare.exists():
        return require_raw(bare, ZIP_URL, expect_sha256=want, allow_download=False)
    archive = require_raw(
        raw_dir / ZIP_NAME,
        ZIP_URL,
        expect_sha256=SHA256[ZIP_NAME] if verify else None,
        allow_download=allow_download,
    )
    with zipfile.ZipFile(archive) as z:
        bare.write_bytes(z.read(f"data/{EXCLUSION_NAME}"))
    return require_raw(bare, ZIP_URL, expect_sha256=want, allow_download=False)


# ------------------------------------------------------------------ build


def _stratified_order(task_ids, limit: int) -> list[int]:
    """Round-robin indices across hop classes, so the sample is balanced at every n.

    Extracted so the stratification is testable without a build: it is the only thing standing
    between a head-N sample and a depth axis that does not exist.

    THE RETURNED ORDER IS THE ROUND-ROBIN ORDER, NOT A SORTED ONE, and that is the whole
    function. It used to end `return sorted(order)`, justified as "indices are returned in file
    order so corpus and gold stay aligned row-for-row" -- which is false: `public` and `graphs`
    are both indexed by this same list, so they stay aligned under ANY permutation. What the
    sort actually did was undo the stratification in the WRITTEN FILE.

    That mattered because every consumer takes a PREFIX. `TaskSuite.task_ids()` returns file
    order and `pi run --n N` slices `[:N]`. Measured on the 800-task build with the sort in
    place: the first 40 ids were 100% 2hop and the first 200 were 135 2hop + 65 3hop2 with no
    4hop at all -- so `tier1_pilot` (n=120) and `tier1_confirmatory` (n=200) would both have
    run an effectively 2hop sample, and `coverage_at_depth_ge2` would have been unmeasurable
    again, on a corpus built specifically to make it measurable. The retrieval probe failed at
    n=40 (20% against a 30% floor) for exactly this reason and was the thing that surfaced it.
    """
    by_hops: dict[str, list[int]] = {}
    for i, tid in enumerate(task_ids):
        by_hops.setdefault(str(tid).split("__")[0], []).append(i)
    order: list[int] = []
    pools = [by_hops[k] for k in sorted(by_hops)]
    while len(order) < limit and any(pools):
        for pool in pools:
            if pool and len(order) < limit:
                order.append(pool.pop(0))
    return order


def build(
    *,
    split: str = "train",
    raw_dir: Path | None = None,
    root: Path | None = None,
    limit: int | None = None,
    stratify_by_hops: bool = True,
    allow_download: bool = True,
    verify: bool = True,
) -> BuildResult:
    """Split one MuSiQue-Ans split into data/corpora/musique/ and data/gold/graphs/musique/.

    `raw_dir` points at a cache of the upstream files (data/raw/musique by default, or a
    test fixture directory). Nothing in CI may depend on the network, so a fixture directory
    plus allow_download=False is a complete, offline build.
    """
    root = root or Path.cwd()
    raw_dir = raw_dir or (root / "data" / "raw" / SUITE)
    name = FILES[split]
    src = require_raw(
        raw_dir / name,
        _HF + name,
        expect_sha256=SHA256[name] if verify else None,
        allow_download=allow_download,
    )

    blocked: frozenset[str] = frozenset()
    if split == "train":
        # TRAIN ONLY, and not by accident: the list enumerates the single-hop questions used
        # in dev/test, so applying it to dev deletes the dev set entirely. It is a guard on
        # what a model may be trained on, not a filter on what may be evaluated.
        blocked = load_exclusions(
            ensure_exclusions(raw_dir, allow_download=allow_download, verify=verify)
        )

    public: list[dict] = []
    graphs: list[dict] = []
    excluded: list[str] = []

    # Streamed, not read_text(): the train split is 241 MB and a builder that needs half a
    # gigabyte of RSS to emit a 50-task fixture is a builder nobody runs.
    with src.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            if not rec.get("answerable", True):
                continue  # musique-FULL ships unanswerable items; they have no gold to compose
            decomp = rec["question_decomposition"]
            if blocked and any(_norm_q(s["question"]) in blocked for s in decomp):
                excluded.append(rec["id"])
                continue

            tid = rec["id"]
            paragraphs = sorted(rec["paragraphs"], key=lambda p: p["idx"])
            public.append(
                {
                    "id": tid,
                    "question": rec["question"],
                    "paragraphs": [
                        {"idx": p["idx"], "title": p["title"], "text": p["paragraph_text"]}
                        for p in paragraphs
                    ],
                }
            )
            graphs.append(_graph(tid, rec, {p["idx"]: p for p in paragraphs}))
            if limit is not None and len(public) >= limit and stratify_by_hops is False:
                break

    if limit is not None and stratify_by_hops:
        # MuSiQue's files are HOP-SORTED: the first 13,900 train records and the first 1,179
        # dev records are all 2hop. Head-N truncation therefore yields an all-2hop sample, whose
        # decomposition graphs are depth<=1 -- so `coverage_at_depth_ge2`, a preregistered
        # secondary endpoint and the axis the paper's central claim rests on, becomes
        # STRUCTURALLY UNMEASURABLE at any n. Verified: a 40-task head-N build has depth
        # histogram {0: 40, 1: 40} and not one depth-2 node.
        #
        # So select round-robin across hop classes instead, which keeps the sample balanced at
        # every n and makes the deep strata reachable.
        keep = _stratified_order([r["id"] for r in public], limit)
        public = [public[i] for i in keep]
        graphs = [graphs[i] for i in keep]

    corpus, chash = write_corpus(root, SUITE, public)
    gold = write_graphs(root, SUITE, GRAPH_VERSION, graphs, corpus_hash=chash)
    # The audit trail for the contamination control. It lives on the GOLD side because the
    # exclusion is a statement about dev/test content.
    (gold.parent / f"excluded_{split}.json").write_text(
        json.dumps(
            {
                "split": split,
                "exclusion_list": EXCLUSION_NAME if blocked else None,
                "n_blocked_questions": len(blocked),
                "n_excluded_tasks": len(excluded),
                "excluded_task_ids": excluded,
            },
            indent=1,
            sort_keys=True,
        )
    )
    return BuildResult(corpus, gold, chash, len(public), len(excluded))


def _graph(tid: str, rec: dict, by_idx: dict[int, dict]) -> dict:
    """One task's need graph, read straight off the human-authored decomposition."""
    decomp = rec["question_decomposition"]
    nodes: list[GoldNode] = []
    edges: list[GoldEdge] = []
    seeds: list[str] = []

    for i, step in enumerate(decomp):
        # Node ids are 1-INDEXED so that node "s3" is literally what "#3" refers to. The
        # step's own `id` field is the seed dataset's question id (e.g. 460946), not its
        # position, so it must not be used here.
        nid = f"s{i + 1}"
        ev: tuple[str, ...] = ()
        idx = step.get("paragraph_support_idx")
        if idx is not None and idx in by_idx:
            ev = (unit_uid(CORPUS_ID, tid, idx, by_idx[idx]["paragraph_text"]),)

        nodes.append(
            GoldNode(
                gold_suite=SUITE,
                gold_task_key=tid,
                gold_node_id=nid,
                gold_text=step["question"],
                # The hop's answer is what a matcher must find in the evidence; the
                # sub-question is what it must recognise in an Ask.
                gold_aliases=(step["answer"],) if step.get("answer") else (),
                gold_kind="fact",
                gold_provenance=("human_composed",),
                gold_provenance_primary="human_composed",
                gold_partition="required",
                gold_discoverability="kb",
                # NECESSARY BY CONSTRUCTION, not by an ablation we ran: the question was
                # composed from these steps, so dropping one makes it unanswerable. The
                # ablation fields stay at zero precisely so no table can read this verdict
                # as a measured effect size.
                gold_ablation_verdict="NECESSARY",
                gold_ev_uids=ev,
                gold_confidence=1.0,
                gold_graph_version=GRAPH_VERSION,
            )
        )

        refs = sorted({int(m.group(1)) for m in REF_RE.finditer(step["question"])})
        valid = [n for n in refs if 1 <= n <= len(decomp) and n != i + 1]
        for n in valid:
            edges.append(
                GoldEdge(
                    gold_suite=SUITE,
                    gold_task_key=tid,
                    gold_src_node_id=f"s{n}",
                    gold_dst_node_id=nid,
                    gold_edge_kind="prerequisite",
                    gold_verified="mechanical",
                    gold_provenance="mechanical",
                    gold_confidence=1.0,
                    gold_graph_version=GRAPH_VERSION,
                )
            )
        if not valid:
            # No placeholder: everything this step needs is stated in x. That IS depth 0.
            seeds.append(nid)

    return finalize_graph(
        suite=SUITE,
        task_key=tid,
        nodes=nodes,
        edges=edges,
        seed_ids=seeds,
        answer=rec.get("answer", ""),
        aliases=tuple(rec.get("answer_aliases", ())),
        version=GRAPH_VERSION,
    )


# ------------------------------------------------------------------ annotation-time `#N` resolution
#
# `_graph` above leaves `#N` in `gold_text` ON PURPOSE -- that text is scored (matchers,
# ablations) and a scored string must stay byte-identical to what the dataset authors wrote,
# not to some paraphrase this repository invented. But the same text is also what an A1/A3
# annotator is SHOWN, and "#1" names nothing a person who has not read the graph can resolve.
# Measured on data/gold/graphs/musique/v1.jsonl: every depth>=1 node (1464/1464) carries an
# unresolved placeholder and every depth-0 node (1196/1196) does not -- a perfect, mechanical
# split, so an annotator's ability to even READ a need was silently standing in for whether
# they would have anticipated it.
#
# The fix lives here, not in `_graph`: it runs at item-EXPORT time, over a COPY of the text,
# and never touches a byte of `gold_text` or `data/gold/`.


class UnresolvedPlaceholder(LookupError):
    """A `#N` in gold node text names a sub-question this task's decomposition has no answer
    for (the raw record wasn't loaded, or that step's `answer` was empty). Raised rather than
    silently left in place -- the caller's only correct response is to refuse the item, which
    is exactly the defect ("Who published #1?" shown verbatim) this module exists to close."""


def resolve_placeholders(text: str, answers: Mapping[str, str]) -> str:
    """Replace every `#N` in `text` with `answers["sN"]` -- the ANSWER to sub-question N, not
    a description of it.

    WHY THE ANSWER AND NOT A DESCRIPTION. A description ("When was [the institution that owned
    The Collegian] founded?") reads as a paraphrase of the composed task, so an annotator ticks
    it as something they'd have asked and a genuinely LATENT need gets recorded as anticipated
    -- the wrong answer, in the direction that flatters the system. The answer names an entity
    ("When was Houston Baptist University founded?") the annotator had no way to reach from the
    task statement alone, so "I would not have thought to ask this" becomes an honest judgment
    about anticipation instead of an artifact of what happened to be legible.

    `node "sN"` IS what `#N` refers to by construction (`_graph` above assigns node ids
    1-indexed for exactly this reason), so the substitution needs no fuzzy matching -- only a
    lookup. `#10` cannot be mistaken for a prefix of `#1`: `REF_RE`'s `\\d+` is greedy and
    matches the full run of digits.
    """

    def _sub(m: re.Match[str]) -> str:
        node_id = f"s{m.group(1)}"
        try:
            return answers[node_id]
        except KeyError:
            raise UnresolvedPlaceholder(node_id) from None

    return REF_RE.sub(_sub, text)


def reference_placeholders(text: str, node_texts: Mapping[str, str]) -> str:
    """Replace every `#N` in `text` with a bracketed REFERENCE to node `sN` itself -- its own
    `gold_text` (already resolved of any placeholders it carries in turn), never `sN`'s answer.

    WHY A REFERENCE HERE AND THE ANSWER EVERYWHERE ELSE. `resolve_placeholders` above is right
    for A1/A3_node/A3_missing/A3_match: those ask "would a person have thought to ask this",
    and printing the answer is what makes "I would not have" an honest verdict about a need
    they had no way to reach. A3_edge asks a DIFFERENT question -- "does dst depend on src" --
    and for that judgment the answer is the wrong thing to print, not merely unhelpful.
    Measured: a model shown `resolve_placeholders`'d edges rejected 6 of 6 true prerequisite
    edges, because substituting src's ANSWER into dst's text (e.g. "...to Sudan?" for a `#1`
    that names "University of Khartoum >> country", whose answer is Sudan) makes dst
    independently answerable from what is on screen -- the honest response to THAT text is "no
    edge", which is the wrong verdict about the dependency being judged. Referencing the
    source NEED instead ("...to ⟨University of Khartoum >> country⟩?") keeps the dependency
    visible: dst still cannot be answered without resolving src, which is exactly the relation
    A3_edge exists to confirm or deny.
    """

    def _sub(m: re.Match[str]) -> str:
        node_id = f"s{m.group(1)}"
        try:
            referent = node_texts[node_id]
        except KeyError:
            raise UnresolvedPlaceholder(node_id) from None
        return f"⟨{referent}⟩"

    return REF_RE.sub(_sub, text)


# The `id` field is always the first key MuSiQue writes on every line (verified against both
# shipped splits) so a cheap prefix match on the raw line -- no JSON parsing -- skips the
# ~21,500 of ~22,355 lines this loader does not need, before paying for `json.loads` on a
# record whose `paragraphs` field alone can run to hundreds of KB.
_ID_RE = re.compile(r'^\{"id":\s*"([^"]*)"')


def load_subanswers(raw_dir: Path, task_keys: Iterable[str]) -> dict[str, dict[str, str]]:
    """`{task_key: {"s1": answer, "s2": answer, ...}}`, streamed off both raw MuSiQue splits.

    STREAMED, NOT `read_text()`'d: the train split alone is 241 MB, and a resolver that needs
    that much RSS to caption ~800 exported tasks is not one anyone will run before a deadline.
    Only `task_keys` are indexed -- the caller passes exactly the task ids in the graphs it
    loaded -- so the returned dict, unlike the file it was read from, is small enough to hold
    for the life of one `export` run (see its docstring: read once, per run).

    A step whose `answer` is empty is OMITTED from that task's map rather than mapped to `""`:
    an empty string is not a fact an annotator could not have reached, it is nothing at all, and
    `resolve_placeholders` must raise on it exactly as it would on a missing key.
    """
    wanted = {str(t) for t in task_keys}
    out: dict[str, dict[str, str]] = {}
    if not wanted:
        return out
    for name in (FILES["train"], FILES["dev"]):
        path = raw_dir / name
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                m = _ID_RE.match(line)
                if m is None or m.group(1) not in wanted:
                    continue
                rec = json.loads(line)
                out[rec["id"]] = {
                    f"s{i + 1}": str(step["answer"])
                    for i, step in enumerate(rec["question_decomposition"])
                    if step.get("answer")
                }
    return out
