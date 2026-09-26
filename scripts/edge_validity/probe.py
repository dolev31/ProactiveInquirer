"""Edge validity: does a gold prerequisite edge u -> v encode reachability? (lane L1)

OPERATOR-SIDE, NEVER A RUN. The calls made here are an instrument applied to the gold graphs, not
a policy rollout: no `run_id`, no `RunManifest`, nothing written under `runs/`, no `pi run`. They
are gold-derived (context A/B/C carry gold sub-answers), so every prompt is canary-scanned before
it is sent (`pi_eval.canary.scan_text` through `assert_canary_clean`), which is firewall layer 4.
`PI_GOLD_ROOT` is set in this analysis process only.

The declared criterion is `artifacts/edge_validity_20260923/CRITERION.md`, committed in c2bdf77
before any probe output existed. This module implements it; it does not tune it. The class rule,
verbatim from that file:

    Classes (primary formulator; fixed now):
    - VALIDATED: floor(v) = 0 AND hit_A(v) <= 0.2 AND G(v) >= 0.4 AND (where testable) closed-book Qwen3-8B base
      does NOT answer u.
    - REFUTED: hit_A(v) >= 0.6, OR closed-book Qwen3-8B base answers u.
    - UNTESTABLE: v has no gold evidence, or v's gold evidence overlaps u's, or neither of the above holds.
    Secondary sweep (reported, not used to decide): hit_A threshold in {0, 0.2} x G threshold in {0.2, 0.4, 0.6}.

and the instrument it rests on, also from that file:

    - hit_X(v): the per-query hit rate in context X (hits / 5), taken as the MAX over the two prompts.
    - G(v) = hit_B(v) - max(hit_A(v), hit_C(v)).

Implementation decisions the criterion leaves open, each stated where it is made:

- Rates are `fractions.Fraction`s with denominator 5 and the thresholds are exact fractions, so
  the boundaries are exact. In floats 0.6 - 0.2 = 0.39999999999999997 < 0.4 and an edge with
  hit_B = 0.6, hit_A = 0.2 would silently miss VALIDATED.
- A reply that parses to fewer than 5 queries scores its missing queries as misses (the
  denominator is always 5); more than 5 keeps the first 5; an unparseable reply is 0 of 5 and is
  counted in the summary.
- The evidence conditions of UNTESTABLE are checked first: on an edge whose v has no evidence, or
  whose v evidence overlaps u's, the hit instrument does not measure the edge, so closed-book
  alone does not move it to REFUTED (the reason is still recorded).
- Context C's random node is drawn from every OTHER task of the suite's v1 graphs (not only the
  held-out 200), at u's depth, same relation for 2Wiki when one exists (flagged when not), and
  never a node whose answer/title equals (casefolded) any answer/title of THIS task's own nodes,
  so the null can never hand the formulator u's or v's own answer by coincidence.
- The chance control draws 5 other held-out tasks' questions per edge (seeded by suite, task, u,
  v) and reports the per-query rate, the same unit as hit_X.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from pi_eval.gold import GoldGraph, GoldNode

# --------------------------------------------------------------------------------- constants

SUITES = ("musique", "strategyqa", "wiki2")
OUT_DIR = Path("artifacts/edge_validity_20260923")
COHORT_DIR = Path("artifacts/completed_cohort_20260922")
PUBLISHED_CB = Path("artifacts/precedence_mechanism_20260918/probe_cache.json")
GRAPH_VERSION = "v1"

PRIMARY = "openai/aws/claude-opus-5"
SECONDARY = "qwen3-8b-base"
CB_MODEL = "qwen3-8b-base"  # the closed-book model the class rule reads
CB_UPPER = "openai/aws/claude-opus-5"  # reported, not used by the class rule

K = 5  # BM25 top-k, the harness's
RANK_DEPTH = 20  # best rank is read in the top 20
N_QUERIES = 5
N_CHANCE = 5

THR_A = Fraction("0.2")
THR_G = Fraction("0.4")
THR_REFUTE = Fraction("0.6")
SWEEP_A = (Fraction(0), Fraction("0.2"))
SWEEP_G = (Fraction("0.2"), Fraction("0.4"), Fraction("0.6"))

CONTEXTS = ("A", "B", "C")

_HEADER = (
    "You are helping to answer a question by searching a small collection of Wikipedia "
    "paragraphs with a keyword search engine.\n\n"
    "Question: {question}\n\n"
    "Already found so far:\n{known}\n\n"
)
_FOOTER = (
    " Each query should be a short keyword search string.\n\n"
    'Respond with JSON only, exactly in this form: {{"queries": ["query 1", "query 2", '
    '"query 3", "query 4", "query 5"]}}'
)
PROMPTS = {
    "P1": _HEADER
    + "Write the 5 search queries that look up what is needed next to answer the question, "
    "given what is already known." + _FOOTER,
    "P2": _HEADER
    + "Write 5 search queries that would retrieve evidence answering the question directly, "
    "skipping intermediate steps if possible." + _FOOTER,
}
NOTHING_YET = "(nothing yet)"

_REL = re.compile(r"^(?P<entity>.+?)\s*>>\s*(?P<relation>.+)$")
_FENCE = re.compile(r"^```(?:json)?\s*(?P<body>.*?)\s*```$", re.DOTALL)
_BRACES = re.compile(r"\{.*\}", re.DOTALL)
_CANARY_SHAPE = re.compile(r"PINQCANARY_[0-9A-F]{16}")


def prompt_sha256() -> dict[str, str]:
    return {k: hashlib.sha256(v.encode()).hexdigest() for k, v in sorted(PROMPTS.items())}


# --------------------------------------------------------------------------------- pure: graph


def prerequisite_edges(graph: GoldGraph) -> list[tuple[str, str]]:
    """Every (u, v) with edge kind `prerequisite`, sorted. Relevance edges are not edges here."""
    return sorted(
        (e.gold_src_node_id, e.gold_dst_node_id)
        for e in graph.gold_edges
        if e.gold_edge_kind == "prerequisite"
    )


def _depth_key(node: GoldNode) -> tuple[int, str]:
    return (node.gold_depth if node.gold_depth is not None else 10**6, node.gold_node_id)


def ancestors(graph: GoldGraph, node_id: str) -> list[str]:
    """All nodes with a prerequisite path to `node_id`, never `node_id` itself, ordered by
    (depth, node id) so the rendered context is deterministic."""
    parents: dict[str, list[str]] = defaultdict(list)
    for u, v in prerequisite_edges(graph):
        parents[v].append(u)
    seen: set[str] = set()
    stack = list(parents.get(node_id, ()))
    while stack:
        p = stack.pop()
        if p in seen or p == node_id:
            continue
        seen.add(p)
        stack.extend(parents.get(p, ()))
    nodes = {n.gold_node_id: n for n in graph.gold_nodes}
    return sorted(seen, key=lambda i: _depth_key(nodes[i]))


def relation_of(text: str) -> str | None:
    """The relation of a "subject >> relation" node; None for a node phrased as a question."""
    m = _REL.match(text)
    return m.group("relation").strip() if m else None


def node_fact(suite: str, node: GoldNode, titles: dict[str, str]) -> tuple[str, ...]:
    """What resolving `node` adds to the context. MuSiQue / 2Wiki: its gold answer
    (`gold_aliases[0]`). StrategyQA (no per-node answers): the titles of its gold evidence
    paragraphs, deduplicated in order. Empty when the node has neither."""
    if suite == "strategyqa":
        out: list[str] = []
        for uid in node.gold_ev_uids:
            t = titles.get(uid)
            if t and t not in out:
                out.append(t)
        return tuple(out)
    return (node.gold_aliases[0],) if node.gold_aliases else ()


def _dedupe(xs) -> tuple[str, ...]:
    out: list[str] = []
    for x in xs:
        if x not in out:
            out.append(x)
    return tuple(out)


def context_facts(
    suite: str,
    graph: GoldGraph,
    u: str,
    variant: str,
    *,
    titles: dict[str, str],
    random_fact: tuple[str, ...] | None = None,
) -> tuple[str, ...]:
    """The facts listed under "already found" in context A, B or C for parent `u`.

    A: every ancestor of u's fact (u and every descendant withheld). B: A plus u's own fact.
    C: A plus a random other task's fact (`pick_random_node`). Deduplicated in order, so a B
    whose u fact is already in A is textually identical to A (counted in the summary)."""
    nodes = {n.gold_node_id: n for n in graph.gold_nodes}
    facts = [f for a in ancestors(graph, u) for f in node_fact(suite, nodes[a], titles)]
    if variant == "A":
        return _dedupe(facts)
    if variant == "B":
        return _dedupe([*facts, *node_fact(suite, nodes[u], titles)])
    if variant == "C":
        if random_fact is None:
            raise ValueError("context C needs the random node's fact")
        return _dedupe([*facts, *random_fact])
    raise ValueError(f"unknown context {variant!r}")


def render_known(facts: tuple[str, ...]) -> str:
    return "\n".join(f"- {f}" for f in facts) if facts else NOTHING_YET


def render_prompt(prompt_id: str, question: str, facts: tuple[str, ...]) -> str:
    return PROMPTS[prompt_id].format(question=question, known=render_known(facts))


# --------------------------------------------------------------------------------- pure: context C


@dataclass(frozen=True, slots=True)
class Candidate:
    task_id: str
    node_id: str
    depth: int
    relation: str | None
    fact: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Pick:
    task_id: str
    node_id: str
    depth: int
    relation: str | None
    fact: tuple[str, ...]
    type_matched: bool | None  # None: the suite has no types


def seeded_rng(*parts: str) -> random.Random:
    digest = hashlib.sha256("|".join(parts).encode()).hexdigest()
    return random.Random(int(digest[:16], 16))


def _norm(s: str) -> str:
    return s.strip().casefold()


def build_candidates(
    suite: str, graphs: dict[str, GoldGraph], *, titles_by_task: dict[str, dict[str, str]]
) -> list[Candidate]:
    """Every node of every task with a depth and a non-empty fact, sorted by (task, node)."""
    out: list[Candidate] = []
    for tid in sorted(graphs):
        titles = titles_by_task.get(tid, {})
        for n in sorted(graphs[tid].gold_nodes, key=lambda x: x.gold_node_id):
            if n.gold_depth is None:
                continue
            fact = node_fact(suite, n, titles)
            if not fact:
                continue
            rel = relation_of(n.gold_text) if suite == "wiki2" else None
            out.append(Candidate(tid, n.gold_node_id, n.gold_depth, rel, fact))
    return out


def pick_random_node(
    suite: str,
    task_id: str,
    u: str,
    *,
    depth: int,
    relation: str | None,
    exclude_facts: set[str],
    candidates: list[Candidate],
) -> Pick:
    """A random node at `depth` from ANOTHER task, seeded by (suite, task, u). Type-matched on
    `relation` when any candidate has it (2Wiki); else any same-depth node, flagged."""
    rng = seeded_rng(suite, task_id, u, "C")
    pool = sorted(
        (
            c
            for c in candidates
            if c.task_id != task_id
            and c.depth == depth
            and not any(_norm(f) in exclude_facts for f in c.fact)
        ),
        key=lambda c: (c.task_id, c.node_id),
    )
    if not pool:
        raise ValueError(f"no context-C candidate for {suite}/{task_id}/{u} at depth {depth}")
    matched: bool | None = None
    if relation is not None:
        typed = [c for c in pool if c.relation and _norm(c.relation) == _norm(relation)]
        matched = bool(typed)
        if typed:
            pool = typed
    c = rng.choice(pool)
    return Pick(c.task_id, c.node_id, c.depth, c.relation, c.fact, matched)


# --------------------------------------------------------------------------------- pure: scoring


def parse_queries(text: str) -> tuple[list[str], bool]:
    """`{"queries": [...]}` -> (first 5 non-empty strings, parsed?). Tolerates a code fence and
    surrounding prose; anything else is ([], False)."""
    s = (text or "").strip()
    m = _FENCE.match(s)
    if m:
        s = m.group("body")
    for attempt in (s, *(x.group(0) for x in [_BRACES.search(s)] if x)):
        try:
            d = json.loads(attempt)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(d, dict) and isinstance(d.get("queries"), list):
            qs = [q.strip() for q in d["queries"] if isinstance(q, str) and q.strip()]
            return qs[:N_QUERIES], True
    return [], False


def query_hits(retriever, queries: list[str], v_uids: tuple[str, ...]) -> list[dict]:
    """Per query: hit (every v uid in the top 5), hit_any (some v uid in the top 5), best_rank
    (1-based rank of v's best uid in the top 20, None if absent)."""
    want = set(v_uids)
    out = []
    for q in queries:
        uids = [x.uid for x in retriever.search(q, RANK_DEPTH)]
        top = set(uids[:K])
        ranks = [i + 1 for i, x in enumerate(uids) if x in want]
        out.append(
            {
                "hit": bool(want) and want <= top,
                "hit_any": bool(want & top),
                "best_rank": min(ranks) if ranks else None,
            }
        )
    return out


def hit_rate(hits: list[bool]) -> Fraction:
    return Fraction(sum(1 for h in hits[:N_QUERIES] if h), N_QUERIES)


def gap(hit_b: Fraction | None, hit_a: Fraction | None, hit_c: Fraction | None):
    """G = hit_B - max(hit_A, hit_C); None when any input is undefined."""
    if hit_b is None or hit_a is None or hit_c is None:
        return None
    return hit_b - max(hit_a, hit_c)


def assign_class(
    *,
    v_has_evidence: bool,
    overlap: bool,
    floor: int | None,
    hit_a: Fraction | None,
    g: Fraction | None,
    cb_answers: bool | None,
    thr_a: Fraction = THR_A,
    thr_g: Fraction = THR_G,
    thr_refute: Fraction = THR_REFUTE,
) -> tuple[str, list[str]]:
    """The CRITERION.md class rule (module docstring). `cb_answers` None = not testable."""
    blocked = []
    if not v_has_evidence:
        blocked.append("v_has_no_gold_evidence")
    if overlap:
        blocked.append("v_evidence_overlaps_u")
    if blocked:
        if cb_answers:
            blocked.append("closed_book_8b_answers_u_not_applied")
        return "UNTESTABLE", blocked
    refute = []
    if hit_a is not None and hit_a >= thr_refute:
        refute.append(f"hit_A>={thr_refute}")
    if cb_answers:
        refute.append("closed_book_8b_answers_u")
    if refute:
        return "REFUTED", refute
    fails = []
    if floor is None or floor != 0:
        fails.append("floor!=0")
    if hit_a is None or hit_a > thr_a:
        fails.append(f"hit_A>{thr_a}")
    if g is None or g < thr_g:
        fails.append(f"G<{thr_g}")
    if fails:
        return "UNTESTABLE", ["neither", *fails]
    ok = ["floor=0", f"hit_A<={thr_a}", f"G>={thr_g}"]
    ok.append("closed_book_not_testable" if cb_answers is None else "closed_book_8b_misses_u")
    return "VALIDATED", ok


# --------------------------------------------------------------------------------- population (I/O)


@dataclass
class EdgeRec:
    suite: str
    task_id: str
    u: str
    v: str
    child_depth: int | None
    u_depth: int | None
    v_uids: tuple[str, ...]
    u_uids: tuple[str, ...]


@dataclass
class SuitePop:
    suite: str
    corpus_dir: str
    corpus_hash: str
    run_ids: list[str]
    task_ids: list[str]
    graphs: dict[str, GoldGraph]
    suite_obj: object
    questions: dict[str, str]
    titles: dict[str, dict[str, str]]
    edges: list[EdgeRec]
    candidates: list[Candidate]
    graph_sha256: str

    def retriever(self, tid: str):
        return self.suite_obj.retriever(tid)


def _sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cohort_rows(suite: str) -> tuple[list[str], list[tuple]]:
    import duckdb

    run_ids = []
    for arm in ("prompted", "trained"):
        p = COHORT_DIR / "cohort" / f"run_ids.{arm}.{suite}.txt"
        run_ids += [x.strip() for x in p.read_text().splitlines() if x.strip()]
    con = duckdb.connect()
    rows = con.execute(
        "select run_id, task_id, split, corpus_dir, corpus_hash, suite_id from "
        f"'{COHORT_DIR / 'scores_parquet' / 'runs.parquet'}' where run_id in (select unnest(?))",
        [run_ids],
    ).fetchall()
    return run_ids, rows


def load_population(suite: str) -> SuitePop:
    from pi_eval.gold import gold_root, load_graphs
    from pi_run.worker import load_suite

    run_ids, rows = cohort_rows(suite)
    if len(rows) != len(run_ids):
        raise SystemExit(f"{suite}: {len(run_ids)} run ids, {len(rows)} found in runs.parquet")
    splits = {r[2] for r in rows}
    cdirs = {r[3] for r in rows}
    chashes = {r[4] for r in rows}
    sids = {r[5] for r in rows}
    if splits != {"test"} or len(cdirs) != 1 or len(chashes) != 1 or sids != {suite}:
        raise SystemExit(f"{suite}: splits {splits} corpus_dirs {cdirs} suites {sids}")
    corpus_dir = next(iter(cdirs))
    corpus_hash = next(iter(chashes))
    suite_obj = load_suite(suite, str(Path("data/corpora") / suite / corpus_dir))
    if suite_obj.corpus_hash != corpus_hash:
        raise SystemExit(f"{suite}: pool corpus_hash {suite_obj.corpus_hash} != runs {corpus_hash}")
    task_ids = sorted({r[1] for r in rows})
    graphs = load_graphs(suite, GRAPH_VERSION)
    graph_path = gold_root() / "graphs" / suite / f"{GRAPH_VERSION}.jsonl"
    questions = {t: suite_obj.view(t).question for t in task_ids}
    titles: dict[str, dict[str, str]] = {}
    for t in task_ids:
        titles[t] = {x.uid: x.title for x in suite_obj.units(t)}
    edges: list[EdgeRec] = []
    for t in task_ids:
        g = graphs[t]
        nodes = {n.gold_node_id: n for n in g.gold_nodes}
        pool = set(titles[t])
        for n in g.gold_nodes:
            missing = [x for x in n.gold_ev_uids if x not in pool]
            if missing:
                raise SystemExit(f"{suite}/{t}/{n.gold_node_id}: gold uids not in pool {missing}")
        for u, v in prerequisite_edges(g):
            edges.append(
                EdgeRec(
                    suite,
                    t,
                    u,
                    v,
                    nodes[v].gold_depth,
                    nodes[u].gold_depth,
                    tuple(nodes[v].gold_ev_uids),
                    tuple(nodes[u].gold_ev_uids),
                )
            )
    titles_by_task = dict(titles)
    if suite == "strategyqa":  # C draws titles from other tasks' pools too
        for t in graphs:
            if t not in titles_by_task and t in suite_obj._idx:
                titles_by_task[t] = {x.uid: x.title for x in suite_obj.units(t)}
    candidates = build_candidates(suite, graphs, titles_by_task=titles_by_task)
    return SuitePop(
        suite=suite,
        corpus_dir=corpus_dir,
        corpus_hash=corpus_hash,
        run_ids=run_ids,
        task_ids=task_ids,
        graphs=graphs,
        suite_obj=suite_obj,
        questions=questions,
        titles=titles,
        edges=edges,
        candidates=candidates,
        graph_sha256=_sha_file(graph_path),
    )


@dataclass
class ParentCtx:
    suite: str
    task_id: str
    u: str
    facts: dict[str, tuple[str, ...]]
    pick: Pick


def parent_contexts(pop: SuitePop) -> dict[tuple[str, str], ParentCtx]:
    out: dict[tuple[str, str], ParentCtx] = {}
    for e in pop.edges:
        key = (e.task_id, e.u)
        if key in out:
            continue
        g = pop.graphs[e.task_id]
        nodes = {n.gold_node_id: n for n in g.gold_nodes}
        titles = pop.titles[e.task_id]
        own = {_norm(f) for n in g.gold_nodes for f in node_fact(pop.suite, n, titles)}
        u_node = nodes[e.u]
        rel = relation_of(u_node.gold_text) if pop.suite == "wiki2" else None
        pick = pick_random_node(
            pop.suite,
            e.task_id,
            e.u,
            depth=u_node.gold_depth,
            relation=rel,
            exclude_facts=own,
            candidates=pop.candidates,
        )
        facts = {
            "A": context_facts(pop.suite, g, e.u, "A", titles=titles),
            "B": context_facts(pop.suite, g, e.u, "B", titles=titles),
            "C": context_facts(pop.suite, g, e.u, "C", titles=titles, random_fact=pick.fact),
        }
        out[key] = ParentCtx(pop.suite, e.task_id, e.u, facts, pick)
    return out


# --------------------------------------------------------------------------------- requests + cache


# MEASURED 2026-09-23: Opus 5 on this gateway spends hidden (redacted) thinking INSIDE
# max_tokens -- the reply carries `thinking_blocks` with a signature and no text, and usage
# reports it as text tokens with reasoning_tokens=0. At max_tokens=512 the first pass returned
# finish_reason=length on 502 of 5,206 formulator replies (most with EMPTY content; concentrated
# in context C, whose random fact the model deliberates over), and at max_tokens=64 on 232 of
# 431 closed-book replies. The 512-token pass stays in cache/ as a record and is not read. The
# CONTENT target is unchanged (a 5-query JSON is ~60 tokens; a closed-book answer is short);
# these are floors on the BUDGET so thinking cannot eat the reply, the same fix
# `scripts/precedence_mechanism/probe.py::MIN_TOKEN_BUDGET` applies to gpt-oss-120b.
FORMULATOR_MAX_TOKENS = 4096
CB_MAX_TOKENS = 64  # the published closed-book budget, kept for the 8B (the lock)
CB_MIN_TOKEN_BUDGET = {"openai/aws/claude-opus-5": 2048}


def formulator_body(model: str, prompt_text: str) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt_text}],
        "temperature": 0,
        "max_tokens": FORMULATOR_MAX_TOKENS,
    }
    if model == "qwen3-8b-base":
        body["chat_template_kwargs"] = {"enable_thinking": False}
    return body


def closed_book_body(model: str, question: str) -> dict:
    """The published closed-book request (`scripts/precedence_mechanism/probe.py::call_model`:
    same template, temperature 0.0, max_tokens 64, seed 0), plus thinking off for Qwen3."""
    from scripts.precedence_mechanism.probe import PROMPT_TEMPLATE

    body = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT_TEMPLATE.format(question=question)}],
        "temperature": 0.0,
        "max_tokens": max(CB_MAX_TOKENS, CB_MIN_TOKEN_BUDGET.get(model, 0)),
        "seed": 0,
    }
    if model == "qwen3-8b-base":
        body["chat_template_kwargs"] = {"enable_thinking": False}
    return body


def request_key(body: dict) -> str:
    raw = json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def cache_path(out_dir: Path, key: str) -> Path:
    return out_dir / "cache" / key[:2] / f"{key}.json"


def read_cache(out_dir: Path, key: str) -> dict | None:
    p = cache_path(out_dir, key)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def response_text(entry: dict) -> str:
    return (entry["response"]["choices"][0]["message"].get("content") or "").strip()


def _finish_reason(entry: dict) -> str | None:
    return entry["response"]["choices"][0].get("finish_reason")


def reply_status(entry: dict) -> str:
    """ "ok" (parses to a query list), "truncated" (cut by the token budget and unparseable:
    NOT a 0-of-5 result, it is a missing measurement), or "unparseable" (a complete reply that
    is not the requested JSON: scored 0 of 5 and counted)."""
    _, ok = parse_queries(response_text(entry))
    if ok:
        return "ok"
    if entry["response"]["choices"][0].get("finish_reason") == "length":
        return "truncated"
    return "unparseable"


def load_canaries() -> frozenset[str]:
    from pi_eval.canary import load

    c = load(Path.cwd())
    if not c:
        raise SystemExit("canary registry empty or missing: refusing to send gold-derived text")
    return frozenset(c)


def canary_check(bodies: dict[str, dict], canaries: frozenset[str]) -> int:
    """Scan every message of every request; raise on the first hit. Returns texts scanned."""
    from scripts.precedence_mechanism.probe import assert_canary_clean

    n = 0
    for key, body in bodies.items():
        for m in body["messages"]:
            if _CANARY_SHAPE.search(m["content"]):
                raise RuntimeError(f"{key}: canary-shaped token in prompt")
            assert_canary_clean(m["content"], canaries, where=key)
            n += 1
    return n


def _post(client, url: str, headers: dict, body: dict, attempts: int = 6) -> dict:
    last: Exception | None = None
    for i in range(attempts):
        try:
            r = client.post(url, json=body, headers=headers)
            r.raise_for_status()
            data = r.json()
            data["choices"][0]["message"]
            return data
        except Exception as exc:  # noqa: BLE001 - bounded retries, then raised
            last = exc
            time.sleep(min(60, 2 ** (i + 1)))
    assert last is not None
    raise last


def run_requests(
    bodies: dict[str, dict],
    *,
    out_dir: Path,
    base_url: str,
    concurrency: int,
    label: str,
) -> dict:
    """Send every uncached request (after the canary scan), caching each response on disk
    keyed by sha256 of the request bytes. One real completion first, then the batch."""
    import httpx

    canaries = load_canaries()
    n_scanned = canary_check(bodies, canaries)
    todo = [k for k in sorted(bodies) if not cache_path(out_dir, k).exists()]
    stats = {
        "label": label,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "n_requests": len(bodies),
        "n_cache_hits": len(bodies) - len(todo),
        "n_calls": 0,
        "n_errors": 0,
        "canary_texts_scanned": n_scanned,
        "canary_registry_size": len(canaries),
        "models_returned": Counter(),
        "errors": [],
    }
    print(f"[{label}] {len(bodies)} requests, {len(todo)} to call, canary-clean", flush=True)
    api_key = os.environ["LITELLM_API_KEY"]
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    url = f"{base_url.rstrip('/')}/chat/completions"
    lock = threading.Lock()

    def one(client, key: str) -> None:
        body = bodies[key]
        try:
            data = _post(client, url, headers, body)
        except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
            with lock:
                stats["n_errors"] += 1
                stats["errors"].append({"key": key, "error": repr(exc)[:300]})
            return
        p = cache_path(out_dir, key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps({"key": key, "request": body, "response": data}))
        tmp.replace(p)
        with lock:
            stats["n_calls"] += 1
            stats["models_returned"][data.get("model")] += 1

    timeout = httpx.Timeout(180.0, connect=15.0)
    with httpx.Client(timeout=timeout) as client:
        if todo:
            one(client, todo[0])  # one real completion before the batch
            if stats["n_calls"] != 1:
                print(f"[{label}] first completion failed: {stats['errors']}", flush=True)
                return _finish(stats, out_dir)
            print(f"[{label}] first completion ok: {dict(stats['models_returned'])}", flush=True)
        done = 1
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futs = [ex.submit(one, client, k) for k in todo[1:]]
            for _ in as_completed(futs):
                done += 1
                if done % 200 == 0:
                    print(
                        f"[{label}] {done}/{len(todo)} calls={stats['n_calls']} "
                        f"errors={stats['n_errors']}",
                        flush=True,
                    )
    return _finish(stats, out_dir)


def _finish(stats: dict, out_dir: Path) -> dict:
    stats["models_returned"] = dict(stats["models_returned"])
    stats["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    with (out_dir / "call_log.jsonl").open("a") as f:
        f.write(json.dumps(stats) + "\n")
    print(json.dumps({k: v for k, v in stats.items() if k != "errors"}), flush=True)
    return stats


# --------------------------------------------------------------------------------- request sets


def formulator_requests(pops: list[SuitePop], model: str):
    """{key: body} and, per (suite, task, u), {(ctx, prompt): key}."""
    bodies: dict[str, dict] = {}
    index: dict[tuple[str, str, str], dict[tuple[str, str], str]] = {}
    ctxs: dict[tuple[str, str, str], ParentCtx] = {}
    for pop in pops:
        for (tid, u), pc in parent_contexts(pop).items():
            ctxs[(pop.suite, tid, u)] = pc
            q = pop.questions[tid]
            slot = index.setdefault((pop.suite, tid, u), {})
            for c in CONTEXTS:
                for pid in PROMPTS:
                    body = formulator_body(model, render_prompt(pid, q, pc.facts[c]))
                    k = request_key(body)
                    bodies[k] = body
                    slot[(c, pid)] = k
    return bodies, index, ctxs


def closed_book_parents(
    pops: list[SuitePop],
) -> list[tuple[str, str, str, str, GoldNode, GoldGraph]]:
    """(suite, task, u, question, u node, graph) for every distinct parent of a held-out edge in
    musique and wiki2. MuSiQue: `closed_book_question` (resolve_placeholders + to_question);
    2Wiki: `to_question` on "subject >> relation"."""
    from scripts.precedence_mechanism.probe import closed_book_question, to_question

    out = []
    for pop in pops:
        if pop.suite not in ("musique", "wiki2"):
            continue
        seen = set()
        for e in pop.edges:
            if (e.task_id, e.u) in seen:
                continue
            seen.add((e.task_id, e.u))
            g = pop.graphs[e.task_id]
            node = {n.gold_node_id: n for n in g.gold_nodes}[e.u]
            if pop.suite == "musique":
                q = closed_book_question(g, node)
                if q is None:
                    raise SystemExit(f"unresolved placeholder in {e.task_id}/{e.u}")
            else:
                if re.search(r"#\d", node.gold_text):
                    raise SystemExit(f"wiki2 node text carries #N: {e.task_id}/{e.u}")
                q = to_question(node.gold_text)
            out.append((pop.suite, e.task_id, e.u, q, node, g))
    return out


def lock_parents(musique_graphs: dict[str, GoldGraph]):
    """The 364 parents of the published closed-book cache, question rebuilt by the same path."""
    from scripts.precedence_mechanism.probe import closed_book_question

    pub = json.loads(PUBLISHED_CB.read_text())
    out = []
    for key in sorted(pub):
        row = pub[key]
        g = musique_graphs[row["task_id"]]
        node = {n.gold_node_id: n for n in g.gold_nodes}[row["node_id"]]
        out.append((key, row, closed_book_question(g, node), node))
    return out


# --------------------------------------------------------------------------------- assemble


def _f(x: Fraction | None) -> float | None:
    return None if x is None else float(x)


def _mean(xs) -> float | None:
    xs = [x for x in xs if x is not None]
    return (sum(xs) / len(xs)) if xs else None


def _mean_frac(xs) -> float | None:
    xs = [x for x in xs if x is not None]
    return float(sum(xs, Fraction(0)) / len(xs)) if xs else None


def assemble(pops: list[SuitePop], out_dir: Path) -> dict:
    from scripts.precedence_mechanism.probe import score_answer

    probe_rows: list[dict] = []
    per_edge: dict[tuple, dict] = {}
    missing = Counter()
    parse_fail = Counter()
    truncated = Counter()
    returned_models = Counter()
    formulators = [PRIMARY, SECONDARY]
    req = {m: formulator_requests(pops, m) for m in formulators}
    diag = defaultdict(Counter)

    # closed book
    cb_rows = []
    cb_by = {}
    for suite, tid, u, q, node, _g in closed_book_parents(pops):
        for model in (CB_MODEL, CB_UPPER):
            body = closed_book_body(model, q)
            k = request_key(body)
            ent = read_cache(out_dir, k)
            if ent is None:
                missing[f"closed_book:{model}"] += 1
                ans, score, ret = None, None, None
            elif not response_text(ent) and _finish_reason(ent) == "length":
                missing[f"closed_book:{model}"] += 1
                truncated[f"closed_book:{model}"] += 1
                ans, score, ret = None, None, ent["response"].get("model")
            else:
                ans = response_text(ent)
                score = score_answer(ans, node)
                ret = ent["response"].get("model")
                returned_models[ret] += 1
                if _finish_reason(ent) == "length":  # scored as the published 8B protocol does
                    truncated[f"closed_book_scored_anyway:{model}"] += 1
            cb_rows.append(
                {
                    "suite": suite,
                    "task_id": tid,
                    "u": u,
                    "model": model,
                    "model_returned": ret,
                    "question": q,
                    "gold": node.gold_aliases[0] if node.gold_aliases else None,
                    "answer": ans,
                    "score": score,
                    "request_sha256": k,
                }
            )
            cb_by[(suite, tid, u, model)] = score

    for pop in pops:
        pop_tids = pop.task_ids
        for e in pop.edges:
            r = pop.retriever(e.task_id)
            q = pop.questions[e.task_id]
            base = {
                "suite": e.suite,
                "task_id": e.task_id,
                "u": e.u,
                "v": e.v,
                "child_depth": e.child_depth,
            }
            has_ev = bool(e.v_uids)
            overlap = bool(set(e.v_uids) & set(e.u_uids))
            # floor: the bare task question
            fh = query_hits(r, [q], e.v_uids)
            floor = int(fh[0]["hit"]) if has_ev else None
            probe_rows.append(
                {
                    **base,
                    "formulator": None,
                    "context": "floor",
                    "prompt": None,
                    "queries": [q],
                    "per_query": fh if has_ev else None,
                    "hits": floor,
                    "hit_rate": floor,
                }
            )
            # chance: 5 other held-out tasks' questions against THIS pool
            rng = seeded_rng(e.suite, e.task_id, e.u, e.v, "chance")
            others = rng.sample([t for t in pop_tids if t != e.task_id], N_CHANCE)
            cq = [pop.questions[t] for t in others]
            ch = query_hits(r, cq, e.v_uids)
            chance = hit_rate([x["hit"] for x in ch]) if has_ev else None
            probe_rows.append(
                {
                    **base,
                    "formulator": None,
                    "context": "chance",
                    "prompt": None,
                    "chance_task_ids": others,
                    "queries": cq,
                    "per_query": ch if has_ev else None,
                    "hits": sum(x["hit"] for x in ch) if has_ev else None,
                    "hit_rate": _f(chance),
                }
            )
            rec = {"floor": floor, "chance": chance, "has_ev": has_ev, "overlap": overlap}
            for model in formulators:
                bodies, index, ctxs = req[model]
                slot = index[(e.suite, e.task_id, e.u)]
                pc = ctxs[(e.suite, e.task_id, e.u)]
                rates = {}
                for c in CONTEXTS:
                    for pid in PROMPTS:
                        k = slot[(c, pid)]
                        ent = read_cache(out_dir, k)
                        if ent is None:
                            missing[model] += 1
                            rates[(c, pid)] = None
                            continue
                        returned_models[ent["response"].get("model")] += 1
                        status = reply_status(ent)
                        if status == "truncated":  # a missing measurement, never a 0 of 5
                            missing[model] += 1
                            truncated[model] += 1
                            rates[(c, pid)] = None
                            continue
                        qs, ok = parse_queries(response_text(ent))
                        if not ok:
                            parse_fail[model] += 1
                        ph = query_hits(r, qs, e.v_uids) if has_ev else None
                        rate = hit_rate([x["hit"] for x in ph]) if has_ev else None
                        rates[(c, pid)] = rate
                        row = {
                            **base,
                            "formulator": model,
                            "context": c,
                            "prompt": pid,
                            "request_sha256": k,
                            "parse_ok": ok,
                            "n_queries": len(qs),
                            "queries": qs,
                            "per_query": ph,
                            "hits": (sum(x["hit"] for x in ph) if has_ev else None),
                            "hit_rate": _f(rate),
                            "hit_any": (any(x["hit_any"] for x in ph) if has_ev else None),
                            "best_rank": (
                                min((x["best_rank"] for x in ph if x["best_rank"]), default=None)
                                if has_ev
                                else None
                            ),
                        }
                        if c == "C":
                            row["c_node"] = {
                                "task_id": pc.pick.task_id,
                                "node_id": pc.pick.node_id,
                                "depth": pc.pick.depth,
                                "relation": pc.pick.relation,
                                "type_matched": pc.pick.type_matched,
                            }
                        probe_rows.append(row)
                hx = {}
                for c in CONTEXTS:
                    vals = [rates[(c, p)] for p in PROMPTS]
                    hx[c] = None if any(x is None for x in vals) else max(vals)
                rec[model] = {
                    "hit": hx,
                    "per_prompt": {f"{c}_{p}": rates[(c, p)] for c in CONTEXTS for p in PROMPTS},
                    "G": gap(hx["B"], hx["A"], hx["C"]),
                    "b_equals_a": pc.facts["B"] == pc.facts["A"],
                    "c_type_matched": pc.pick.type_matched,
                }
            if e.suite in ("musique", "wiki2"):
                s8 = cb_by.get((e.suite, e.task_id, e.u, CB_MODEL))
                so = cb_by.get((e.suite, e.task_id, e.u, CB_UPPER))
                rec["cb8"] = None if s8 is None else bool(s8 >= 1.0)
                rec["cb8_score"] = s8
                rec["cb_opus"] = so
                rec["cb_testable"] = True
            else:
                rec["cb8"], rec["cb8_score"], rec["cb_opus"], rec["cb_testable"] = (
                    None,
                    None,
                    None,
                    False,
                )
            # diagnostics: does the question or context A already state u's fact (answer, or
            # evidence title for StrategyQA), or v's? Substring, casefolded.
            g = pop.graphs[e.task_id]
            nodes = {n.gold_node_id: n for n in g.gold_nodes}
            pc = req[PRIMARY][2][(e.suite, e.task_id, e.u)]
            a_text = (q + "\n" + "\n".join(pc.facts["A"])).casefold()
            b_text = (q + "\n" + "\n".join(pc.facts["B"])).casefold()
            uf = node_fact(e.suite, nodes[e.u], pop.titles[e.task_id])
            vf = node_fact(e.suite, nodes[e.v], pop.titles[e.task_id])
            d = diag[e.suite]
            d["edges"] += 1
            d["u_fact_already_in_question_or_A"] += bool(uf) and all(
                f.casefold() in a_text for f in uf
            )
            d["v_fact_in_question_or_A"] += any(f.casefold() in a_text for f in vf)
            d["v_fact_in_question_or_B"] += any(f.casefold() in b_text for f in vf)
            d["b_equals_a"] += pc.facts["B"] == pc.facts["A"]
            per_edge[(e.suite, e.task_id, e.u, e.v)] = (e, rec)

    # classes (primary) + sweep
    class_rows = []
    sweep = defaultdict(Counter)
    for (suite, tid, u, v), (e, rec) in per_edge.items():
        p = rec[PRIMARY]
        hx = p["hit"]
        cb_answers = rec["cb8"] if rec["cb_testable"] else None
        if rec["cb_testable"] and rec["cb8"] is None:
            raise SystemExit(f"closed-book 8B missing for {suite}/{tid}/{u}")
        cls, reasons = assign_class(
            v_has_evidence=rec["has_ev"],
            overlap=rec["overlap"],
            floor=rec["floor"],
            hit_a=hx["A"],
            g=p["G"],
            cb_answers=cb_answers,
        )
        rec["class"] = cls
        class_rows.append(
            {
                "suite": suite,
                "task_id": tid,
                "u": u,
                "v": v,
                "child_depth": e.child_depth,
                "floor": rec["floor"],
                "hit_A": _f(hx["A"]),
                "hit_B": _f(hx["B"]),
                "hit_C": _f(hx["C"]),
                "G": _f(p["G"]),
                "closed_book_8b": rec["cb8_score"],
                "class": cls,
                "reasons": reasons,
            }
        )
        for ta in SWEEP_A:
            for tg in SWEEP_G:
                c2, _ = assign_class(
                    v_has_evidence=rec["has_ev"],
                    overlap=rec["overlap"],
                    floor=rec["floor"],
                    hit_a=hx["A"],
                    g=p["G"],
                    cb_answers=cb_answers,
                    thr_a=ta,
                    thr_g=tg,
                )
                sweep[(suite, f"hitA<={ta},G>={tg}")][c2] += 1

    # summary by suite x child depth
    by_cell: dict[tuple, list] = defaultdict(list)
    for key, (e, rec) in per_edge.items():
        by_cell[(e.suite, e.child_depth)].append((e, rec))
        by_cell[(e.suite, "all")].append((e, rec))
    table = []
    for (suite, d), items in sorted(by_cell.items(), key=lambda x: (x[0][0], str(x[0][1]))):
        testable = [(e, r) for e, r in items if r["has_ev"] and not r["overlap"]]
        row = {
            "suite": suite,
            "child_depth": d,
            "n_edges": len(items),
            "n_v_no_evidence": sum(1 for _, r in items if not r["has_ev"]),
            "n_overlap": sum(1 for _, r in items if r["has_ev"] and r["overlap"]),
            "n_testable": len(testable),
            "means_over": "testable edges (v has evidence, no overlap with u)",
            "floor": _mean([r["floor"] for _, r in testable]),
            "chance": _mean_frac([r["chance"] for _, r in testable]),
            "class_counts": dict(Counter(r["class"] for _, r in items)),
        }
        for model, tag in ((PRIMARY, "opus"), (SECONDARY, "qwen8b")):
            row[tag] = {
                **{
                    f"hit_{c}": _mean_frac([r[model]["hit"][c] for _, r in testable])
                    for c in CONTEXTS
                },
                "G": _mean_frac([r[model]["G"] for _, r in testable]),
                **{
                    f"hit_{cp}": _mean_frac([r[model]["per_prompt"][cp] for _, r in testable])
                    for cp in (f"{c}_{p}" for c in CONTEXTS for p in PROMPTS)
                },
                "n_b_equals_a": sum(1 for _, r in items if r[model]["b_equals_a"]),
            }
        cbt = [r for _, r in items if r["cb_testable"]]
        row["closed_book_8b_rate_over_edges"] = _mean([r["cb8_score"] for r in cbt])
        row["closed_book_opus_rate_over_edges"] = _mean([r["cb_opus"] for r in cbt])
        row["n_c_type_matched"] = sum(1 for _, r in items if r[PRIMARY]["c_type_matched"] is True)
        row["n_c_type_unmatched"] = sum(
            1 for _, r in items if r[PRIMARY]["c_type_matched"] is False
        )
        table.append(row)

    cb_summary = {}
    for suite in ("musique", "wiki2"):
        for model in (CB_MODEL, CB_UPPER):
            xs = [x["score"] for x in cb_rows if x["suite"] == suite and x["model"] == model]
            got = [x for x in xs if x is not None]
            cb_summary[f"{suite}:{model}"] = {
                "n_parents": len(xs),
                "n_scored": len(got),
                "n_answers": int(sum(got)),
                "rate": (sum(got) / len(got)) if got else None,
            }

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out_dir / "probe_rows.jsonl", probe_rows)
    _write_jsonl(out_dir / "closed_book.jsonl", cb_rows)
    _write_jsonl(out_dir / "edge_classes.jsonl", class_rows)
    calls = [json.loads(x) for x in (out_dir / "call_log.jsonl").read_text().splitlines() if x]
    summary = {
        "criterion": "artifacts/edge_validity_20260923/CRITERION.md (committed c2bdf77)",
        "graph_version": GRAPH_VERSION,
        "retriever": f"pinq_adapters BM25 via pi_run.worker.load_suite, k={K}, best rank in top {RANK_DEPTH}",
        "primary_formulator": PRIMARY,
        "secondary_formulator": SECONDARY,
        "closed_book_model": CB_MODEL,
        "closed_book_upper": CB_UPPER,
        "prompt_sha256": prompt_sha256(),
        "closed_book_prompt_sha256": _cb_prompt_sha(),
        "thresholds": {
            "hit_A_max": str(THR_A),
            "G_min": str(THR_G),
            "refute_hit_A": str(THR_REFUTE),
        },
        "models_returned_in_cache": dict(returned_models),
        "missing_responses": dict(missing),
        "parse_failures": dict(parse_fail),
        "truncated_replies": dict(truncated),
        "call_log": [{k: v for k, v in c.items() if k != "errors"} for c in calls],
        "inputs": {
            p.suite: {
                "graph_file_sha256": p.graph_sha256,
                "task_ids_sha256": hashlib.sha256("\n".join(p.task_ids).encode()).hexdigest(),
                "run_ids_sha256": hashlib.sha256("\n".join(sorted(p.run_ids)).encode()).hexdigest(),
                "n_run_ids": len(p.run_ids),
                "n_tasks": len(p.task_ids),
                "corpus_dir": f"data/corpora/{p.suite}/{p.corpus_dir}",
                "corpus_hash": p.corpus_hash,
                "n_edges": len(p.edges),
                "edges_by_child_depth": dict(
                    sorted(Counter(e.child_depth for e in p.edges).items())
                ),
            }
            for p in pops
        },
        "by_suite_depth": table,
        "closed_book": cb_summary,
        "secondary_sweep": {f"{s}|{k}": dict(v) for (s, k), v in sorted(sweep.items())},
        "diagnostics": {k: dict(v) for k, v in diag.items()},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=False) + "\n")
    return summary


def _cb_prompt_sha() -> str:
    from scripts.precedence_mechanism.probe import PROMPT_TEMPLATE

    return hashlib.sha256(PROMPT_TEMPLATE.encode()).hexdigest()


def _write_jsonl(p: Path, rows: list[dict]) -> None:
    with p.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------------- CLI


def _pops(args) -> list[SuitePop]:
    return [load_population(s) for s in args.suites]


def cmd_population(args) -> int:
    for p in _pops(args):
        depth = dict(sorted(Counter(e.child_depth for e in p.edges).items()))
        print(
            f"{p.suite}: run_ids={len(p.run_ids)} tasks={len(p.task_ids)} edges={len(p.edges)} "
            f"by_child_depth={depth} v_no_evidence={sum(1 for e in p.edges if not e.v_uids)} "
            f"overlap={sum(1 for e in p.edges if e.v_uids and set(e.v_uids) & set(e.u_uids))} "
            f"distinct_parents={len({(e.task_id, e.u) for e in p.edges})} "
            f"corpus=data/corpora/{p.suite}/{p.corpus_dir} graph_sha256={p.graph_sha256[:16]}"
        )
    return 0


def cmd_formulate(args) -> int:
    pops = _pops(args)
    bodies, _, ctxs = formulator_requests(pops, args.model)
    tm = Counter(pc.pick.type_matched for pc in ctxs.values())
    print(f"distinct parents={len(ctxs)} distinct requests={len(bodies)} C type_matched={dict(tm)}")
    if args.limit:
        bodies = dict(sorted(bodies.items())[: args.limit])
    stats = run_requests(
        bodies,
        out_dir=args.out_dir,
        base_url=args.base_url,
        concurrency=args.concurrency,
        label=f"formulate:{args.model}",
    )
    return 1 if stats["n_errors"] else 0


def cmd_closedbook(args) -> int:
    pops = _pops(args)
    bodies = {}
    for _s, _t, _u, q, _n, _g in closed_book_parents(pops):
        b = closed_book_body(args.model, q)
        bodies[request_key(b)] = b
    print(f"closed-book parents={len(closed_book_parents(pops))} distinct requests={len(bodies)}")
    if args.lock:
        from pi_eval.gold import load_graphs

        for _k, _row, q, _n in lock_parents(load_graphs("musique", GRAPH_VERSION)):
            b = closed_book_body(args.model, q)
            bodies[request_key(b)] = b
        print(f"with the published lock parents: distinct requests={len(bodies)}")
    stats = run_requests(
        bodies,
        out_dir=args.out_dir,
        base_url=args.base_url,
        concurrency=args.concurrency,
        label=f"closedbook:{args.model}",
    )
    return 1 if stats["n_errors"] else 0


def cmd_lock(args) -> int:
    """Our 8B closed-book answers vs the published cache, on every published parent, and on
    the ones inside this lane's population."""
    from scripts.precedence_mechanism.probe import score_answer

    from pi_eval.gold import load_graphs

    pops = _pops(args)
    ours = {(t, u) for s, t, u, *_ in closed_book_parents(pops) if s == "musique"}
    rows = []
    for key, pub, q, node in lock_parents(load_graphs("musique", GRAPH_VERSION)):
        ent = read_cache(args.out_dir, request_key(closed_book_body(CB_MODEL, q)))
        ans = response_text(ent) if ent else None
        sc = score_answer(ans, node) if ans is not None else None
        rows.append(
            {
                "key": key,
                "in_population": (pub["task_id"], pub["node_id"]) in ours,
                "question_identical": q == pub["question"],
                "published_answer": pub["qwen3_8b_base"]["answer"],
                "published_score": pub["qwen3_8b_base"]["score"],
                "answer": ans,
                "score": sc,
                "answer_identical": ans == pub["qwen3_8b_base"]["answer"],
                "score_agrees": sc == pub["qwen3_8b_base"]["score"],
            }
        )
    out = {}
    for name, sel in (
        ("all_published", rows),
        ("in_population", [r for r in rows if r["in_population"]]),
    ):
        got = [r for r in sel if r["score"] is not None]
        out[name] = {
            "n": len(sel),
            "n_scored": len(got),
            "question_identical": sum(r["question_identical"] for r in sel),
            "answer_identical": sum(r["answer_identical"] for r in got),
            "score_agrees": sum(r["score_agrees"] for r in got),
            "published_answers": int(sum(r["published_score"] for r in sel)),
            "our_answers": int(sum(r["score"] for r in got)),
            "published_rate": sum(r["published_score"] for r in sel) / len(sel) if sel else None,
            "our_rate": sum(r["score"] for r in got) / len(got) if got else None,
        }
    (args.out_dir / "closed_book_lock.json").write_text(
        json.dumps({"summary": out, "rows": rows}, indent=1, ensure_ascii=False) + "\n"
    )
    print(json.dumps(out, indent=1))
    return 0


def cmd_assemble(args) -> int:
    s = assemble(_pops(args), args.out_dir)
    print(json.dumps({k: s[k] for k in ("missing_responses", "parse_failures")}, indent=1))
    # the secondary formulator runs after the primary outputs are written (brief step 7), so
    # only a missing PRIMARY or closed-book response fails the assembly
    return 1 if any(k != SECONDARY for k in s["missing_responses"]) else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m scripts.edge_validity.probe")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--suites", nargs="+", default=list(SUITES))
    ap.add_argument("--base-url", default="http://127.0.0.1:4030/v1")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("population")
    f = sub.add_parser("formulate")
    f.add_argument("--model", required=True)
    f.add_argument("--concurrency", type=int, default=8)
    f.add_argument("--limit", type=int, default=0)
    c = sub.add_parser("closedbook")
    c.add_argument("--model", required=True)
    c.add_argument("--concurrency", type=int, default=8)
    c.add_argument("--lock", action="store_true")
    sub.add_parser("lock")
    sub.add_parser("assemble")
    args = ap.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    return {
        "population": cmd_population,
        "formulate": cmd_formulate,
        "closedbook": cmd_closedbook,
        "lock": cmd_lock,
        "assemble": cmd_assemble,
    }[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
