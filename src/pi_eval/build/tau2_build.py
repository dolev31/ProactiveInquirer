"""Gold graphs for tau2 / tau-Knowledge. GOLD SIDE — nothing in pinq_adapters may import this.

THREE SOURCES OF GOLD, EACH WITH A DIFFERENT EPISTEMIC STATUS, KEPT SEPARATE
  1. `required_documents` -> fact nodes, `bench_author`. The benchmark's own answer key.
  2. doc -> tool edges     -> tool_unlock nodes, `mechanical`. Extracted by regex and
     cross-referenced against the live tool list; no annotator, no LLM, binary-verifiable.
  3. the user-private partition -> `mechanical`. Whether a need can be found in the KB at
     all, decided by normalized string containment and nothing else.

WHY (3) IS THE NUMBER THAT MATTERS
    The user-private share is the CEILING on any Inquirer: a fact that exists only in the
    customer's head cannot be discovered by reading, only by asking the user — which the
    confirmatory arms forbid. Reporting a discovery rate without it would credit the policy
    for failing at something structurally impossible.

A CORRECTION TO THE UPSTREAM FIELD DESCRIPTION
    `Task.required_documents` is documented upstream as "document titles". It is not: all
    959 references across the 97 tasks resolve as document **IDs** and none as titles
    (verified against v1.0.1). We resolve by id and keep a title fallback, so a future
    upstream switch degrades to a miss we can count rather than a silent empty graph.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from pi_eval.build.common import BuildResult, as_dict, write_graphs
from pi_eval.gold import GoldEdge, GoldFacet, GoldGraph, GoldNode, compute_depths
from pinq.ids import corpus_hash as _corpus_hash
from pinq.ids import evidence_uid, h

SUITE = "tau2"
CORPUS_ID = "tau2_banking_knowledge"

# RECONCILED WITH `pi_eval.score.DEFAULT_GRAPH_VERSION` and `pi score --graph-version`.
# This string is BOTH a path component (data/gold/graphs/<suite>/<version>.jsonl) and the
# version `score()` asks EVERY suite for in one call. A suite-scoped value like "tau2/v1"
# therefore wrote graphs/tau2/tau2/v1.jsonl and then matched nothing: `score()` skipped every
# tau2 run as "no graph", and the primary endpoint read as ABSENT rather than as broken.
# The suite is already the directory; the version must be the version alone.
# tests/test_adapters_tau2.py::test_graph_version_is_what_pi_score_asks_for pins it.
GRAPH_VERSION = "v1"

# The partition instrument, pinned. Every knob that could move the user-private share is
# in this string, so the number is reproducible and its arbitrariness is visible rather
# than buried: a reader can re-run with a different n and see the elasticity.
PARTITION_PIN = "user_private/v2:overlap>=0.60,stopwords=en-lite,casefold,alnum,sentence"

# Deliberately tiny. A large stoplist is a tuning knob that silently moves the headline
# number; this one only removes tokens that carry no topical content at all.
_STOP = frozenset(
    """a an and are as at be been but by can do does for from had has have if in into is it
    its of on or that the their them then there these they this to was were will with you
    your not no""".split()
)

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_HEADER = re.compile(r"^#{2,}\s*(.+?)\s*$", re.M)
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


# ------------------------------------------------------------------------- normalization


def norm_tokens(text: str) -> tuple[str, ...]:
    """Casefold, strip non-alphanumerics, drop stopwords. The only normalization applied."""
    return tuple(
        t for t in _NON_ALNUM.sub(" ", (text or "").lower()).split() if t and t not in _STOP
    )


def ngrams(tokens: Sequence[str], n: int) -> frozenset[str]:
    """Contiguous content-token n-grams, joined by a space so containment is a set test."""
    if len(tokens) < n:
        return frozenset({" ".join(tokens)}) if tokens else frozenset()
    return frozenset(" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def segment_instructions(text: str) -> tuple[tuple[str, str], ...]:
    """Split user_scenario.instructions into (block_label, block_text).

    Segments on '##' headers where they exist and on blank lines where they do not. The
    brief specified '## Phase' blocks, but only 4 of the 97 tasks use a Phase header (48
    use '## Conversation Flow', and 13 have no header at all), so a Phase-only parser would
    return nothing for 93 tasks and silently report a user-private share of zero for them.
    """
    text = (text or "").strip()
    if not text:
        return ()
    heads = list(_HEADER.finditer(text))
    if heads:
        out = []
        if heads[0].start() > 0:
            pre = text[: heads[0].start()].strip()
            if pre:
                out.append(("preamble", pre))
        for i, m in enumerate(heads):
            end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
            body = text[m.end() : end].strip()
            if body:
                out.append((m.group(1), body))
        return tuple(out)
    return tuple(
        (f"para{i}", p.strip()) for i, p in enumerate(re.split(r"\n\s*\n", text)) if p.strip()
    )


def sentences(block: str) -> tuple[str, ...]:
    return tuple(s.strip() for s in _SENT_SPLIT.split(block or "") if len(s.strip()) > 2)


# ------------------------------------------------------------------------- the partition


@dataclass(frozen=True, slots=True)
class FactCandidate:
    """One sentence of the user's scenario, with the verdict on where it can be found."""

    block: str
    text: str
    discoverability: str  # "kb" | "user_private"
    matched_doc_id: str | None = None
    score: float = 0.0


def partition_facts(
    instructions: str,
    required_docs: Sequence[Mapping[str, str]],
    *,
    mode: str = "overlap",
    threshold: float = 0.6,
    n: int = 4,
) -> tuple[FactCandidate, ...]:
    """Mark each scenario sentence kb-discoverable or user-private. NO LLM anywhere.

    TWO INSTRUMENTS, AND WHY THE DEFAULT IS `overlap`. Both curves below are from
    `partition_elasticity` over all 97 v1.0.1 tasks; re-run it to reproduce them.
      * `ngram`: the sentence shares a contiguous content n-gram with a required document.
        Stronger evidence per hit, but it is a cliff — the private share runs
        0.617 / 0.911 / 0.982 / 0.996 for n = 2 / 3 / 4 / 5. A knob that swings the headline
        ceiling by 38 points cannot be a silent default, and at n=4 it is degenerate: 57 of
        97 tasks come back at exactly 1.00.
      * `overlap` (default): the fraction of the sentence's content tokens that appear
        anywhere in a required document. On the same tasks it degrades smoothly —
        0.651 / 0.811 / 0.899 / 0.934 / 0.952 for thresholds 0.5 / 0.6 / 0.7 / 0.8 / 0.9 —
        so the operating point sits on a slope rather than on a cliff edge. (These five
        numbers replace an earlier set — 0.425 / 0.591 / 0.734 / 0.845 / 0.921 — that no
        longer reproduces against this corpus and this stoplist.)

    Both are exact normalized string containment. Neither is the truth: this is a
    measurement instrument, so `partition_elasticity()` exists to publish the curve and
    the scalar is never reported without it.
    """
    prepared: list[tuple[str, frozenset[str]]] = []
    for d in required_docs:
        toks = norm_tokens(d.get("content", ""))
        prepared.append((d.get("id", ""), ngrams(toks, n) if mode == "ngram" else frozenset(toks)))

    out: list[FactCandidate] = []
    for label, block in segment_instructions(instructions):
        for sent in sentences(block):
            toks = norm_tokens(sent)
            best_doc, best = None, 0.0
            if toks:
                probe = ngrams(toks, n) if mode == "ngram" else None
                for doc_id, ref in prepared:
                    if mode == "ngram":
                        score = 1.0 if (probe and probe & ref) else 0.0
                    else:
                        score = sum(1 for t in toks if t in ref) / len(toks)
                    if score > best:
                        best_doc, best = doc_id, score
            hit = best >= threshold and best > 0.0
            out.append(
                FactCandidate(
                    block=label,
                    text=sent,
                    discoverability="kb" if hit else "user_private",
                    matched_doc_id=best_doc if hit else None,
                    score=best,
                )
            )
    return tuple(out)


def partition_elasticity(
    tasks: Sequence[Mapping[str, Any]],
    documents: Sequence[Mapping[str, str]],
    *,
    mode: str = "overlap",
    grid: Sequence[float] = (0.5, 0.6, 0.7, 0.8, 0.9),
) -> tuple[tuple[float, float], ...]:
    """(threshold, user_private_share) over a grid. The curve behind the ceiling number.

    Exists because the plan's own rule is that a scalar never appears without its curve,
    and because gate G-M4 asks whether a conclusion survives the granularity knob.
    """
    out = []
    ngram_mode = mode == "ngram"
    for param in grid:
        kb = private = 0
        for task in tasks:
            req, _ = resolve_required(task.get("required_documents") or [], documents)
            facts = partition_facts(
                (task.get("user_scenario") or {}).get("instructions", ""),
                req,
                mode=mode,
                # THE KNOB IS `n`, NOT THE THRESHOLD, IN NGRAM MODE. The ngram instrument
                # scores 1.0 or 0.0, so feeding the grid value (2, 3, 4, 5) in as a threshold
                # made `best >= threshold` false for EVERY sentence and returned a flat curve
                # of 1.00 at every n — a "measurement" that could not move and that reported
                # the ceiling as 100% user-private for the whole suite. The threshold in this
                # mode is 1.0 by construction: one shared contiguous n-gram is a hit.
                threshold=1.0 if ngram_mode else float(param),
                n=int(param) if ngram_mode else 4,
            )
            for f in facts:
                if f.discoverability == "kb":
                    kb += 1
                else:
                    private += 1
        total = kb + private
        out.append((float(param), (private / total) if total else 0.0))
    return tuple(out)


# ------------------------------------------------------------------------- graph assembly


def _node_id(task_id: str, kind: str, key: str) -> str:
    return h("gold", SUITE, task_id, kind, key)[:32]


def resolve_required(
    required: Iterable[str], documents: Sequence[Mapping[str, str]]
) -> tuple[tuple[Mapping[str, str], ...], tuple[str, ...]]:
    """(resolved documents, unresolved references). Unresolved are counted, never dropped
    silently — a rising miss count is how an upstream id/title switch announces itself."""
    by_id = {d.get("id", ""): d for d in documents}
    by_title = {d.get("title", ""): d for d in documents}
    found, missing = [], []
    for ref in required:
        doc = by_id.get(ref) or by_title.get(ref)
        if doc is None:
            missing.append(ref)
        else:
            found.append(doc)
    return tuple(found), tuple(missing)


def build_task_graph(
    task: Mapping[str, Any],
    documents: Sequence[Mapping[str, str]],
    tool_edges: Sequence[Any],
    *,
    graph_version: str = GRAPH_VERSION,
    partition_mode: str = "overlap",
    partition_threshold: float = 0.6,
    ngram_n: int = 4,
) -> GoldGraph:
    """One task's gold graph: doc facts at depth 0, the tools they unlock at depth 1."""
    task_id = task.get("id", "")
    required_refs = list(task.get("required_documents") or [])
    req_docs, _missing = resolve_required(required_refs, documents)
    req_ids = {d.get("id", "") for d in req_docs}

    nodes: list[GoldNode] = []
    edges: list[GoldEdge] = []
    seed_ids: list[str] = []

    # --- (1) bench-author fact nodes, one per required document -------------------------
    doc_node: dict[str, str] = {}
    for d in req_docs:
        doc_id = d.get("id", "")
        nid = _node_id(task_id, "fact", doc_id)
        doc_node[doc_id] = nid
        seed_ids.append(nid)  # reachable by a single KB search: depth 0 by construction
        content = d.get("content", "")
        nodes.append(
            GoldNode(
                gold_suite=SUITE,
                gold_task_key=task_id,
                gold_node_id=nid,
                gold_text=d.get("title", "") or doc_id,
                gold_kind="fact",
                gold_provenance=("bench_author",),
                gold_provenance_primary="bench_author",
                gold_partition="required",
                gold_discoverability="kb",
                # the benchmark author asserted necessity; we did not measure it here
                gold_ablation_verdict="UNTESTABLE",
                # same span convention the retriever emits, so gold and rollout agree on
                # uids by construction rather than by convention
                gold_ev_uids=(evidence_uid(CORPUS_ID, doc_id, f"0:{len(content)}"),),
                gold_confidence=1.0,
                gold_extractor_pin="required_documents/v1",
                gold_graph_version=graph_version,
            )
        )

    # --- (2) mechanical tool_unlock nodes + the prerequisite edge ------------------------
    # Only tools unlocked by a document THIS task requires: a tool named in some unrelated
    # document is not this task's prerequisite, and counting it would inflate the
    # denominator of the unlock-and-invoke endpoint.
    for e in tool_edges:
        if e.doc_id not in req_ids:
            continue
        nid = _node_id(task_id, "tool_unlock", e.tool_name)
        if not any(n.gold_node_id == nid for n in nodes):
            nodes.append(
                GoldNode(
                    gold_suite=SUITE,
                    gold_task_key=task_id,
                    gold_node_id=nid,
                    gold_text=f"the tool {e.tool_name} exists and can be unlocked",
                    # The tool name, MACHINE-READABLE. `discoverable_tool_unlock_tau2` joins
                    # gold nodes against `env_calls.kwargs_json`, and splitting the third word
                    # out of an English sentence is a join key that a reworded docstring
                    # silently breaks.
                    gold_aliases=(e.tool_name,),
                    gold_kind="tool_unlock",
                    gold_provenance=("mechanical",),
                    gold_provenance_primary="mechanical",
                    gold_partition="required",
                    gold_discoverability="kb",
                    # binary-verifiable: either the call succeeded or it did not
                    gold_ablation_verdict="NECESSARY",
                    gold_ablation_delta=1.0,
                    gold_confidence=1.0,
                    gold_extractor_pin="discoverable/v1",
                    gold_graph_version=graph_version,
                )
            )
        edges.append(
            GoldEdge(
                gold_suite=SUITE,
                gold_task_key=task_id,
                gold_src_node_id=doc_node[e.doc_id],
                gold_dst_node_id=nid,
                gold_edge_kind="prerequisite",
                # the strongest verification tier in the schema, and it is earned: the
                # suffix cannot be guessed, so reading the document is strictly necessary
                gold_verified="mechanical",
                gold_provenance="mechanical",
                gold_confidence=1.0,
                gold_graph_version=graph_version,
            )
        )

    # --- (3) the user-private partition --------------------------------------------------
    facts = partition_facts(
        (task.get("user_scenario") or {}).get("instructions", ""),
        req_docs,
        mode=partition_mode,
        threshold=partition_threshold,
        n=ngram_n,
    )
    for i, fact in enumerate(facts):
        nid = _node_id(task_id, "user_fact", f"{i}:{fact.text[:80]}")
        nodes.append(
            GoldNode(
                gold_suite=SUITE,
                gold_task_key=task_id,
                gold_node_id=nid,
                gold_text=fact.text,
                gold_kind="fact",
                gold_provenance=("mechanical",),
                gold_provenance_primary="mechanical",
                # optional, never required: these are mined from the user's own script and
                # carry no author assertion of necessity, so they must not enter a primary
                # endpoint's denominator
                gold_partition="optional",
                gold_discoverability=fact.discoverability,
                gold_ablation_verdict="UNTESTABLE",
                gold_facet_id=None,
                gold_confidence=0.5,
                gold_extractor_pin=PARTITION_PIN,
                gold_graph_version=graph_version,
            )
        )
        if fact.discoverability == "kb" and fact.matched_doc_id in doc_node:
            seed_ids.append(nid)

    # --- depth + facets, both DERIVED --------------------------------------------------
    depths = compute_depths([n.gold_node_id for n in nodes], edges, seed_ids)
    nodes = [
        GoldNode(
            **{
                **{f: getattr(n, f) for f in n.__dataclass_fields__},
                "gold_depth": depths.get(n.gold_node_id),
            }
        )
        for n in nodes
    ]
    facets = _facets(task_id, nodes, edges)
    by_facet = {n: f.gold_facet_id for f in facets for n in f.gold_node_ids}
    nodes = [
        GoldNode(
            **{
                **{f: getattr(n, f) for f in n.__dataclass_fields__},
                "gold_facet_id": by_facet.get(n.gold_node_id),
            }
        )
        for n in nodes
    ]

    return GoldGraph(
        gold_suite=SUITE,
        gold_task_key=task_id,
        gold_nodes=tuple(nodes),
        gold_edges=tuple(edges),
        gold_facets=tuple(facets),
        gold_seed_node_ids=tuple(seed_ids),
        gold_graph_version=graph_version,
    )


def _facets(task_id: str, nodes: Sequence[GoldNode], edges: Sequence[GoldEdge]) -> list[GoldFacet]:
    """Weakly-connected components AFTER deleting the depth-0 frontier.

    Same convention as the synthetic suite, so horizontal breadth means the same thing in
    both places and the two are comparable.
    """
    keep = {n.gold_node_id for n in nodes if n.gold_depth not in (0, None)}
    adj: dict[str, set[str]] = {n: set() for n in keep}
    for e in edges:
        a, b = e.gold_src_node_id, e.gold_dst_node_id
        if a in keep and b in keep:
            adj[a].add(b)
            adj[b].add(a)
    seen: set[str] = set()
    out: list[GoldFacet] = []
    for start in sorted(keep):
        if start in seen:
            continue
        comp, stack = [], [start]
        seen.add(start)
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nxt in adj.get(cur, ()):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        out.append(
            GoldFacet(
                gold_suite=SUITE,
                gold_task_key=task_id,
                gold_facet_id=_node_id(task_id, "facet", sorted(comp)[0]),
                gold_node_ids=tuple(sorted(comp)),
            )
        )
    return out


def partition_summary(graphs: Iterable[GoldGraph]) -> dict[str, float]:
    """The ceiling number, aggregated. Reported next to every discovery rate."""
    kb = private = 0
    for g in graphs:
        for n in g.gold_nodes:
            if n.gold_provenance_primary != "mechanical" or n.gold_kind != "fact":
                continue
            if n.gold_discoverability == "kb":
                kb += 1
            elif n.gold_discoverability == "user_private":
                private += 1
    total = kb + private
    return {
        "n_kb": float(kb),
        "n_user_private": float(private),
        "user_private_share": (private / total) if total else 0.0,
    }


# ------------------------------------------------------------------------- the build driver


def graph_row(g: GoldGraph) -> dict:
    """One GoldGraph -> the JSON line `pi_eval.gold.load_graphs` reads back.

    `common.finalize_graph` is deliberately NOT used here. It re-derives depth and facets
    from scratch, and `build_task_graph` has already derived both under tau2's own seed rule
    (a required document is reachable by a single KB search, so it IS the depth-0 frontier).
    Running both would compute the same two fields by two different routes, which is exactly
    the situation where they drift and no test notices.
    """
    return {
        "gold_suite": g.gold_suite,
        "gold_task_key": g.gold_task_key,
        "gold_nodes": [as_dict(n) for n in g.gold_nodes],
        "gold_edges": [as_dict(e) for e in g.gold_edges],
        "gold_facets": [as_dict(f) for f in g.gold_facets],
        "gold_seed_node_ids": list(g.gold_seed_node_ids),
        "gold_graph_version": g.gold_graph_version,
        # tau2 has no free-text answer: the reward is a hash over an EXECUTED action
        # sequence. Leaving this "" is what keeps `score_run` from emitting
        # answer_token_f1 / answer_exact_match rows for a suite where they mean nothing.
        "gold_answer": "",
        "gold_aliases": [],
    }


def corpus_hash_of(documents: Iterable[Mapping[str, str]]) -> str:
    """The SAME hash `pinq_adapters.tau2.Tau2Suite` mints, recomputed on the gold side.

    Duplicated deliberately rather than imported: pi_eval importing the adapter to learn the
    identity of the corpus it is annotating would make the gold artifact depend on the
    rollout package being installed, and the import-linter contract forbids the reverse. The
    two are held equal by
    tests/test_adapters_tau2.py::test_gold_and_adapter_agree_on_the_corpus_hash.
    """
    return _corpus_hash(
        (
            d.get("id", ""),
            d.get("title", ""),
            hashlib.sha256((d.get("content", "") or "").encode()).hexdigest(),
        )
        for d in documents
    )


def build(
    *,
    root: Path | None = None,
    documents: Sequence[Mapping[str, str]] | None = None,
    tasks: Sequence[Mapping[str, Any]] | None = None,
    tool_edges: Sequence[Any] | None = None,
    graph_version: str = GRAPH_VERSION,
    partition_mode: str = "overlap",
    partition_threshold: float = 0.6,
    ngram_n: int = 4,
    strict_counts: bool = True,
) -> BuildResult:
    """Write data/gold/graphs/tau2/<version>.jsonl, one line per task.

    NO CORPUS IS WRITTEN, and that is not an omission. tau2 is self-sourced: the adapter reads
    upstream's own 698 documents through TAU2_DATA_DIR, so `BuildResult.corpus` names that
    directory rather than a tree this repo produced. Minting a second copy under data/corpora/
    would give the same bytes two different corpus_hashes and make the adapter's hash and
    gold's disagree — which is invisible in every downstream number, because a uid mismatch
    reads as "the policy retrieved nothing relevant".

    `documents` / `tasks` / `tool_edges` are arguments so the whole driver is testable against
    the committed fixture with tau2 absent. When they are omitted they come from the live
    adapter, which is the only path that touches upstream.
    """
    root = Path(root or Path.cwd())
    corpus_dir = root
    if documents is None or tasks is None or tool_edges is None:
        # Imported HERE, never at module scope: tau2 pulls litellm and a domain registry, and
        # `pytest -m "not integration"` has to stay runnable on a machine without the extra.
        from pinq_adapters.tau2._probe import documents_dir
        from pinq_adapters.tau2.suite import Tau2Suite

        suite = Tau2Suite(strict_counts=strict_counts)
        ids = suite.task_ids()
        if documents is None:
            documents = list(suite.documents)
        if tasks is None:
            tasks = [suite.task_record(t) for t in ids]
        if tool_edges is None:
            tool_edges = list(suite.tool_edges(suite.environment(ids[0])))
        corpus_dir = documents_dir()

    graphs = [
        build_task_graph(
            t,
            documents,
            tool_edges,
            graph_version=graph_version,
            partition_mode=partition_mode,
            partition_threshold=partition_threshold,
            ngram_n=ngram_n,
        )
        for t in tasks
    ]
    gold = write_graphs(
        root,
        SUITE,
        graph_version,
        [graph_row(g) for g in graphs],
        # The SAME hash Tau2Suite computes adapter-side, so a gold graph and the rollout
        # that produced its runs can be proven to describe one corpus.
        corpus_hash=corpus_hash_of(documents),
    )
    return BuildResult(
        corpus=corpus_dir,
        gold=gold,
        corpus_hash=corpus_hash_of(documents),
        n_tasks=len(graphs),
        # Nothing is excluded: every one of the 97 tasks gets a graph. A task with no
        # required_documents still gets one — with no fact nodes and no edges — because a
        # missing row and a genuinely empty graph must not look the same to `score()`.
        n_excluded=0,
    )
