"""Gold graphs for tau2 RETAIL. GOLD SIDE — nothing in pinq_adapters may import this.

WHY THIS IS NOT A DOMAIN FLAG ON `tau2_build`
    `tau2_build` keys entirely off `Task.required_documents`, the benchmark's own answer key.
    MEASURED against tau2-bench-data v1.0.1: retail carries that field on **0 of its 114
    tasks**, where banking_knowledge carries it on 97 of 97. Retail has no document set at all
    — its corpus is a relational DB (500 users, 1,000 orders, 50 products) and its gold is an
    ACTION SEQUENCE of typed tool calls. Sharing a builder would mean sharing a code path that
    reads a field one of the two domains does not have.

THE THREE NODE KINDS, AND WHY THE THIRD IS KEPT
    Every argument value the gold actions require is a fact the agent had to obtain. Where it
    could have come from is measurable, and the answer decides the node:

        an earlier action read a record supplying it -> "kb", gold_ev_uids = that ONE record
        `known_info` / `reason_for_call` names it     -> "user_private", a ROOT
        neither                                      -> "unknown", ev_uids empty

    Measured over the 1,015 nodes the 112 tasks produce: 492 kb (48.5%), 219 user_private
    (21.6%), 304 unknown (30.0%). The last bucket is emitted and labelled rather than dropped
    — dropping it would shrink every denominator by an amount no reader could see. It is large
    because the honest answer often is "we cannot say": the value is in the DB, but no action
    in the gold sequence reads a record that supplies it.

    PROVIDER BEATS USER-HELD when both apply. A value the customer stated that is ALSO in a
    record the agent read is not user-private, because user-private is the ceiling on what no
    amount of retrieval could reach. Classing it the other way would overstate that ceiling.

WHERE DEPTH COMES FROM, AND WHAT IT IS NOT
    To read a DB record you need its key, and that key is itself a value the agent had to
    obtain. The prerequisite edge is therefore `key(record) -> value-inside-record`: a real
    data dependency. It is deliberately NOT "action k depends on action k-1", which would make
    depth a relabelling of sequence position and would put two independent lookups in a chain.

    Measured over the 112 tasks that carry actions: 1,015 nodes, 423 prerequisite edges,
    72.3% of tasks reaching depth >= 1, depths {0:592, 1:278, 2:122, 3:23} — 423 latent nodes
    (41.7%), 145 of them at depth >= 2, 0 orphans. For scale, musique is 55% latent.

    THE ORACLE CEILING IS 100%, AND THAT IS THE TEST THAT MATTERS. Replaying each task's gold
    action sequence through the view-side `uids_for_call` reaches 170 of 170 required evidence
    uids, on 82 of 82 tasks that have any. A gold graph its own answer key cannot satisfy is
    not a hard benchmark, it is a broken instrument — and the first version of this builder
    scored 1.2% there, which is how the containment bug was caught rather than shipped.

LAYER 4 OF THE FIREWALL IS ARMED BUT HAS NO CARRIER HERE, AND THAT IS INHERITED
    `write_graphs` mints a nonce per graph and registers it, so `pi verify firewall` has 112
    retail nonces to scan for — confirmed on disk. But the CARRIER is `gold_answer`, chosen
    because it is the one gold string that never legitimately reaches a model, and retail has
    no free-text answer: like banking, `gold_answer` is "" because the benchmark's reward is
    over an executed action sequence. So the nonce sits in a sibling field that no leak would
    ever carry along, and layer 4 detects nothing on this suite.

    This is the SAME pre-existing gap tau2 and drgym already have, not a new one, and it is
    recorded here rather than left to be discovered. Layers 1-3 (the import contract, the type
    wall, the process split) are unaffected and still stop leakage through code; only leakage
    through a STRING goes unseen. Closing it needs a gold string that never legitimately
    reaches a model, and retail's `gold_text` values are matcher inputs, so there is no free
    carrier to appropriate — a real design decision, deliberately not made in passing here.

THE UID COMES FROM THE REQUEST, NEVER THE PAYLOAD
    A tau2 tool result is a customer record; the runner stores a `result_digest` and never the
    content. Retail's tools are key-addressed — 458 of 550 required calls (83.3%) name a record
    directly in their arguments — so a retrieved record's identity is available from what was
    ASKED FOR. `record_uid` is built from (table, record_id) and the record's canonical length
    only, so nothing that reaches an artifact was read out of a result.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from pi_eval.build.common import BuildResult, as_dict, write_graphs
from pi_eval.gold import GoldEdge, GoldGraph, GoldNode, compute_depths
from pinq.ids import corpus_hash as _corpus_hash
from pinq.ids import evidence_uid

SUITE = "tau2_retail"
CORPUS_ID = "tau2_retail"
GRAPH_VERSION = "v1"

# A two-character argument matches half the database by containment; "no" as a `reason`
# argument would resolve against every record holding the word. Three is the shortest length
# at which retail's own identifiers (#W100, item ids, zips) survive and noise does not.
MIN_VALUE_LEN = 3

# Argument names that carry a LIST of record ids rather than a scalar. Flattened rather than
# skipped: `item_ids` is where the depth lives — those are the values you can only name after
# reading the order that contains them.
_LIST_ARGS = ("item_ids", "new_item_ids")

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def norm(s: Any) -> str:
    """The one normalisation, shared by every containment test in this module.

    Two normalisations would be free to disagree about whether a value is user-held or
    DB-resident, which is the difference between a root and a latent node.
    """
    return _NON_ALNUM.sub("", str(s).lower())


def record_text(record: Mapping[str, Any]) -> str:
    """Canonical JSON for a DB record. SORTED KEYS, because the uid depends on the length.

    Two serialisations of one record must not mint two uids: the corpus hash and every
    memoisation key downstream would fork on dict ordering.
    """
    return json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def record_doc_id(table: str, record_id: str) -> str:
    """`<table>:<record_id>` — the convention the adapter must mirror exactly.

    `EvidenceUnit.make()` on the view side mints the uid from the same three parts. A
    disagreement here is invisible in every downstream number, which is why the paragraph
    suites pin theirs with a test and this one does too.
    """
    return f"{table}:{record_id}"


def record_uid(table: str, record_id: str, record: Mapping[str, Any]) -> str:
    """The evidence uid for one DB record. Whole-record span, mirroring `common.unit_uid`."""
    text = record_text(record)
    return evidence_uid(CORPUS_ID, record_doc_id(table, record_id), f"0:{len(text)}")


def corpus_records(db: Mapping[str, Mapping[str, Any]]) -> list[dict]:
    """The DB as evidence units. One record, one unit — the grain a tool call returns.

    Record-level rather than field-level for the same reason the paragraph suites are
    paragraph-level: a gold uid the environment can never emit is a node that can never be
    resolved, and no retail tool returns a single field.
    """
    out: list[dict] = []
    for table in sorted(db):
        for rid in sorted(db[table]):
            rec = db[table][rid]
            text = record_text(rec)
            out.append(
                {
                    "uid": record_uid(table, rid, rec),
                    "corpus_id": CORPUS_ID,
                    "doc_id": record_doc_id(table, rid),
                    "span": f"0:{len(text)}",
                    "title": f"{table}/{rid}",
                    "n_chars": len(text),
                }
            )
    return out


def _index(db: Mapping[str, Mapping[str, Any]]) -> tuple[dict[str, set[str]], dict[str, str]]:
    """(doc_id -> the normalised values inside it, doc_id -> that record's OWN key).

    BY RECORD, not by value. The question this answers is "if the agent read this record, what
    did it learn?", which is what makes evidence one record rather than every record a value
    appears in. The second map is what makes a prerequisite edge a data dependency: a record's
    key is the value you must already hold to read it.
    """
    doc_values: dict[str, set[str]] = {}
    doc_key: dict[str, str] = {}
    for table in sorted(db):
        for rid in sorted(db[table]):
            doc = record_doc_id(table, rid)
            doc_key[doc] = norm(rid)
            seen: set[str] = set()
            _scalars(db[table][rid], seen)
            doc_values[doc] = seen
    return doc_values, doc_key


def _docs_for_call(index: Any, tool_name: str, arguments: Mapping[str, Any]) -> tuple[str, ...]:
    """doc_ids a call reads, via the VIEW-side mapping. One definition, imported.

    `pi_eval` may import `pinq_adapters` (the contract forbids only the reverse, and
    `tau2_build` already does it), so the gold side uses the same argument->record mapping the
    runner will use to write `retrieved_uids`. Two copies would be free to disagree about what
    a call read, and gold that disagrees with the runner is gold nothing can ever satisfy.
    """
    from pinq_adapters.tau2.retail_units import uids_for_call

    by_uid = {u: d for d, u in index.uids.items()}
    return tuple(by_uid[u] for u in uids_for_call(index, tool_name, arguments) if u in by_uid)


def _scalars(obj: Any, out: set[str]) -> None:
    if isinstance(obj, Mapping):
        for v in obj.values():
            _scalars(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _scalars(v, out)
    else:
        n = norm(obj)
        if len(n) >= MIN_VALUE_LEN:
            out.add(n)


def _required_values(actions: Sequence[Mapping[str, Any]]) -> dict[str, tuple[int, str]]:
    """normalised value -> (index of the action that FIRST needs it, its raw form).

    First appearance, not every appearance: the same order id argued to three tools is one
    fact obtained once, and counting it three times would inflate every per-task denominator.
    """
    first: dict[str, tuple[int, str]] = {}
    for k, a in enumerate(actions):
        for name, v in (a.get("arguments") or {}).items():
            values = v if (name in _LIST_ARGS and isinstance(v, list)) else [v]
            for item in values:
                if isinstance(item, (Mapping, list, tuple)):
                    continue
                n = norm(item)
                if len(n) >= MIN_VALUE_LEN:
                    first.setdefault(n, (k, str(item)))
    return first


def _user_held(task: Mapping[str, Any]) -> str:
    """The text carrying facts the CUSTOMER can supply, normalised for containment.

    RETAIL SHIPS AN AUTHORED PARTITION AND IT IS BETTER THAN INFERRING ONE. Every one of the
    114 tasks carries `user_scenario.instructions.known_info` ("You are Yusuf Rossi in zip code
    19122.") and `unknown_info` ("You do not remember your email address."). `tau2_build` has
    to infer the equivalent for banking with an overlap threshold and a pinned instrument
    (PARTITION_PIN); here the benchmark states it, so the provenance is bench_author.

    `reason_for_call` joins it because that IS the customer's opening statement -- the facts in
    it were supplied by the user by definition. `persona` and `task_instructions` are
    deliberately EXCLUDED: they are roleplay direction, not facts, and matching against them
    manufactures user-held values out of stage directions. Measured over 773 required values:
    the whole scenario blob matches 264 (34.2%), known_info alone 213 (27.6%), and
    known_info + reason_for_call 260 (33.6%) -- so excluding the roleplay text costs 4 values
    and removes a class of false positive.

    `unknown_info` is present on 89 of 114 tasks and is recorded here only as prose; it names
    what the customer CANNOT supply, which bounds the ceiling from the other side but does not
    name values, so it is not parsed into labels.
    """
    ins = (task.get("user_scenario") or {}).get("instructions") or {}
    if not isinstance(ins, Mapping):
        return norm(str(ins))
    return norm(f"{ins.get('known_info') or ''} {ins.get('reason_for_call') or ''}")


def _later_user_id(actions: Sequence[Mapping[str, Any]], after: int) -> str | None:
    """The first `user_id` argument in an action AFTER index `after`. Arguments only."""
    for a in actions[after + 1 :]:
        v = (a.get("arguments") or {}).get("user_id")
        if isinstance(v, str) and v:
            return v
    return None


def build_graphs(
    db: Mapping[str, Mapping[str, Any]], tasks: Iterable[Mapping[str, Any]]
) -> list[GoldGraph]:
    """One GoldGraph per task that has required actions.

    A task with no actions yields NO graph rather than an empty one. 2 of retail's 114 tasks
    are in that state, and an empty graph is scored as a task the policy failed rather than
    one there was never anything to measure — the distinction `score()` cannot make for itself.
    """
    from pinq_adapters.tau2.retail_units import RetailIndex

    doc_values, doc_key = _index(db)
    index = RetailIndex.from_db(db)
    out: list[GoldGraph] = []

    for task in tasks:
        actions = (task.get("evaluation_criteria") or {}).get("actions") or []
        if not actions:
            continue
        task_key = str(task.get("id"))
        held = _user_held(task)
        required = _required_values(actions)
        if not required:
            continue

        # WHICH RECORD PROVIDED EACH VALUE. Evidence for "the agent obtained V" is the ONE
        # record it read to get V, not every record V happens to appear in.
        #
        # THE FIRST VERSION USED CONTAINMENT AND IT WAS UNUSABLE. `gold_ev_uids` is an AND --
        # the matcher requires `ev <= retrieved` -- so listing every record containing a value
        # demanded the policy retrieve all of them. Measured: a required value sits in a median
        # of 3 records but a MEAN of 68.5, and one appears in all 1,500. That put 37,136
        # required uids on the graph-bearing tasks (for ~9 nodes each), and an oracle replaying
        # the gold action sequence exactly reached 1.2% of them. A ceiling that low is not a
        # hard task, it is a broken instrument.
        #
        # The note used to say "111 tasks". The graph-bearing population is 112 of 114, MEASURED
        # on this checkout and asserted in `test_retail_build`; the containment build that
        # produced 37,136 no longer exists, so the total is left as it was recorded rather than
        # re-derived against a denominator nobody can re-measure.
        #
        # The provider is the earliest record named by an EARLIER action that contains V. That
        # is what the agent actually read, it is one record, and it makes the oracle ceiling
        # 100% by construction -- which is the property a gold graph has to have.
        read_before: list[tuple[int, str]] = []  # (action index, doc_id), in order
        for k, a in enumerate(actions):
            name = str(a.get("name") or "")
            docs = _docs_for_call(index, name, a.get("arguments") or {})
            if not docs and name.startswith("find_user_id_by"):
                # THE 75 `find_*` CALLS, RESOLVED FROM ARGUMENTS ALONE. They return a user_id
                # rather than taking one, so the record they read is not in their own
                # arguments -- and reading their RESULT is exactly what the no-payload rule
                # forbids. But whatever they found, a later call keys on: the next `user_id`
                # argument in the sequence names it.
                #
                # This is INFERENCE, not measurement, and it is labelled as such here because
                # it is the one place in this builder that is. Without it 301 nodes -- values
                # that demonstrably ARE in the DB -- were classed "unknown" purely because the
                # honest reading of their provenance was unavailable, which understated the
                # retrievable population by a third.
                nxt = _later_user_id(actions, k)
                if nxt is not None:
                    doc = record_doc_id("users", nxt)
                    if doc in index.uids:
                        docs = (doc,)
            for doc in docs:
                read_before.append((k, doc))

        provider: dict[str, str] = {}
        kinds: dict[str, str] = {}
        for v, (k, _raw) in required.items():
            src = next(
                (doc for j, doc in read_before if j < k and v in doc_values.get(doc, ())),
                None,
            )
            if src is not None:
                provider[v] = src
                kinds[v] = "kb"
            elif v in held:
                kinds[v] = "user_private"
            else:
                kinds[v] = "unknown"

        # PREREQUISITE = the key of the record that provided this value, when that key is
        # itself a value this task required. A real data dependency: you could not have read
        # the providing record without already holding its key.
        edges: list[GoldEdge] = []
        for v, doc in provider.items():
            key = doc_key.get(doc, "")
            if key and key != v and key in required and required[key][0] < required[v][0]:
                edges.append(
                    GoldEdge(
                        gold_suite=SUITE,
                        gold_task_key=task_key,
                        gold_src_node_id=key,
                        gold_dst_node_id=v,
                        gold_edge_kind="prerequisite",
                        gold_verified="mechanical",
                        gold_provenance="mechanical",
                        gold_confidence=1.0,
                        gold_graph_version=GRAPH_VERSION,
                    )
                )

        gated = {e.gold_dst_node_id for e in edges}
        seeds = [v for v in required if v not in gated]
        depths = compute_depths(list(required), edges, seeds)

        nodes: list[GoldNode] = []
        for v, (_k, raw) in sorted(required.items()):
            doc = provider.get(v)
            uids = (index.uids[doc],) if doc else ()
            nodes.append(
                GoldNode(
                    gold_suite=SUITE,
                    gold_task_key=task_key,
                    gold_node_id=v,
                    gold_text=raw,
                    gold_aliases=(raw,),
                    gold_kind="fact",
                    gold_provenance=("bench_author",),
                    gold_provenance_primary="bench_author",
                    # REQUIRED means the benchmark's own answer key names it. That is the
                    # strongest partition this repo has and it is what `graph.required()`
                    # counts, so it must not be applied to anything weaker.
                    gold_partition="required",
                    gold_discoverability=kinds[v],
                    gold_depth=depths.get(v),
                    gold_depth_basis="prereq_only",
                    gold_ev_uids=uids,
                    gold_confidence=1.0,
                    gold_ablation_verdict="NECESSARY",
                    gold_graph_version=GRAPH_VERSION,
                )
            )

        if not any(n.gold_partition == "required" for n in nodes):
            continue
        out.append(
            GoldGraph(
                gold_suite=SUITE,
                gold_task_key=task_key,
                gold_nodes=tuple(nodes),
                gold_edges=tuple(edges),
                gold_seed_node_ids=tuple(n.gold_node_id for n in nodes if (n.gold_depth or 0) == 0),
                gold_graph_version=GRAPH_VERSION,
            )
        )
    return out


def corpus_hash_of(db: Mapping[str, Mapping[str, Any]]) -> str:
    """Order-independent over (doc_id, title, sha256(record_text)).

    The adapter must compute the identical value from the same DB; a disagreement is invisible
    in every downstream number because a uid mismatch reads as "the policy retrieved nothing
    relevant" rather than as an error.
    """
    import hashlib

    return _corpus_hash(
        (
            record_doc_id(table, rid),
            f"{table}/{rid}",
            hashlib.sha256(record_text(db[table][rid]).encode()).hexdigest(),
        )
        for table in db
        for rid in db[table]
    )


def _row(g: GoldGraph) -> dict:
    """One graph -> one JSON row. `as_dict` is per-dataclass, so the nested nodes, edges and
    facets have to be converted explicitly or `json.dumps` meets a GoldNode.

    `gold_answer` stays EMPTY, as it does for banking: retail has no free-text answer -- the
    benchmark's reward is over an executed action sequence -- and leaving it set is what would
    make `score_run` emit answer_token_f1 and answer_exact_match rows for a suite where they
    mean nothing.
    """
    return {
        "gold_suite": g.gold_suite,
        "gold_task_key": g.gold_task_key,
        "gold_nodes": [as_dict(n) for n in g.gold_nodes],
        "gold_edges": [as_dict(e) for e in g.gold_edges],
        "gold_facets": [as_dict(f) for f in g.gold_facets],
        "gold_seed_node_ids": list(g.gold_seed_node_ids),
        "gold_graph_version": g.gold_graph_version,
        "gold_answer": "",
        "gold_aliases": [],
    }


def build(
    *,
    root: Path | None = None,
    db: Mapping[str, Mapping[str, Any]] | None = None,
    tasks: Sequence[Mapping[str, Any]] | None = None,
    graph_version: str = GRAPH_VERSION,
) -> BuildResult:
    """Write data/gold/graphs/tau2_retail/<version>.jsonl, one line per task.

    NO CORPUS TREE IS WRITTEN, for the reason `tau2_build` states: retail is self-sourced from
    an upstream checkout, so the adapter reads the same `db.json` this did. A second copy under
    data/corpora/ would give one set of bytes two corpus_hashes.

    `db` / `tasks` are arguments so the driver is testable with no upstream checkout present.
    """
    root = Path(root or Path.cwd())
    if db is None or tasks is None:
        from pinq_adapters.tau2._probe import domain_data_dir

        d = domain_data_dir("retail")
        db = json.loads((d / "db.json").read_text())
        tasks = json.loads((d / "tasks.json").read_text())
    graphs = build_graphs(db, tasks)
    chash = corpus_hash_of(db)
    gold = write_graphs(root, SUITE, graph_version, (_row(g) for g in graphs), corpus_hash=chash)
    return BuildResult(corpus=root, gold=gold, corpus_hash=chash, n_tasks=len(graphs))
