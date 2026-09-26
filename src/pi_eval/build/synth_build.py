"""Generate the synthetic suite: the graph is CONSTRUCTED FIRST and the corpus derived from it.

Because G is known by construction, RNR, C@d and phi have closed-form values, so metric code
can be debugged at edit-test speed with zero LLM calls and zero dollars. This is the only
suite where a metric bug is unambiguous rather than a plausible null.

The generator lives in pi_eval (gold side). It emits two physically separate artifacts:
  data/corpora/synth/<corpus_hash>/tasks.jsonl   PUBLIC  — all the adapter may read
  data/gold/graphs/synth/<version>.jsonl         GOLD    — only pi_eval may read
Nothing crosses, which is the same rule every real suite follows.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from pi_eval.canary import mint, register
from pi_eval.gold import GoldEdge, GoldFacet, GoldGraph, GoldNode, compute_depths
from pinq.ids import evidence_uid

CORPUS_ID = "synth_v1"


def _token(task: int, facet: int, depth: int) -> str:
    """An unguessable handle. A retriever can only surface depth d+1 once depth d has been
    read, which is how the synthetic suite reproduces tau2's discoverable-tool mechanic."""
    return "K" + hashlib.sha256(f"{task}:{facet}:{depth}".encode()).hexdigest()[:8].upper()


def build(
    n_tasks: int = 24,
    n_facets: int = 3,
    depth: int = 3,
    n_distractors: int = 6,
    seed: int = 7,
    root: Path | None = None,
) -> tuple[Path, Path, str]:
    rng = random.Random(seed)
    root = root or Path.cwd()
    public: list[dict] = []
    graphs: list[dict] = []
    canaries: set[str] = set()

    for t in range(n_tasks):
        docs: list[dict] = []
        nodes: list[GoldNode] = []
        edges: list[GoldEdge] = []
        facets: list[GoldFacet] = []
        seed_ids: list[str] = []
        answers: list[str] = []

        for f in range(n_facets):
            facet_nodes: list[str] = []
            for d in range(depth):
                tok = _token(t, f, d)
                nxt = _token(t, f, d + 1) if d + 1 < depth else ""
                node_id = f"n{t}_{f}_{d}"
                value = f"V{t}{f}{d}"
                # The doc is FOUND by its own token and REVEALS the next one.
                body = f"Record {tok}. The value for facet {f} step {d} is {value}." + (
                    f" To continue, consult record {nxt}." if nxt else " This is the final step."
                )
                docs.append(
                    {
                        "doc_id": f"{t}:{f}:{d}",
                        "title": f"Record {tok}",
                        "text": body,
                        "key_token": tok,
                    }
                )
                nodes.append(
                    GoldNode(
                        gold_suite="synth",
                        gold_task_key=f"s{t}",
                        gold_node_id=node_id,
                        gold_text=f"the value for facet {f} at step {d} is {value}",
                        gold_kind="fact",
                        gold_provenance=("mechanical",),
                        gold_provenance_primary="mechanical",
                        gold_partition="required",
                        gold_discoverability="kb",
                        gold_ablation_verdict="NECESSARY",
                        gold_ablation_delta=1.0,
                        # The objective link from a need to the evidence that resolves it.
                        # Computed with the same pure function the adapter uses, so gold and
                        # rollout agree on uids by construction rather than by convention.
                        gold_ev_uids=(evidence_uid(CORPUS_ID, f"{t}:{f}:{d}", f"0:{len(body)}"),),
                        # A facet is a component AFTER deleting the depth-0 frontier,
                        # so seed nodes belong to no facet. Keeping this consistent with
                        # GoldFacet.gold_node_ids is what makes breadth and depth separable.
                        gold_facet_id=(f"f{t}_{f}" if d > 0 else None),
                        gold_graph_version="v1",
                    )
                )
                facet_nodes.append(node_id)
                if d == 0:
                    seed_ids.append(node_id)  # depth-0: its token is stated in the question
                else:
                    edges.append(
                        GoldEdge(
                            gold_suite="synth",
                            gold_task_key=f"s{t}",
                            gold_src_node_id=f"n{t}_{f}_{d - 1}",
                            gold_dst_node_id=node_id,
                            gold_edge_kind="prerequisite",
                            gold_verified="mechanical",
                            gold_provenance="mechanical",
                            gold_confidence=1.0,
                            gold_graph_version="v1",
                        )
                    )
            facets.append(
                GoldFacet(
                    gold_suite="synth",
                    gold_task_key=f"s{t}",
                    gold_facet_id=f"f{t}_{f}",
                    gold_node_ids=tuple(facet_nodes[1:]),  # facets exclude the depth-0 frontier
                )
            )
            answers.append(f"V{t}{f}{depth - 1}")

        for j in range(n_distractors):
            tok = "D" + hashlib.sha256(f"{t}:dist:{j}".encode()).hexdigest()[:8].upper()
            docs.append(
                {
                    "doc_id": f"{t}:dist:{j}",
                    "title": f"Note {tok}",
                    "text": f"Unrelated note {tok} about topic {rng.randint(100, 999)}.",
                    "key_token": tok,
                }
            )
        rng.shuffle(docs)

        opening = ", ".join(_token(t, f, 0) for f in range(n_facets))
        public.append(
            {
                "id": f"s{t}",
                "question": f"Starting from records {opening}, report the final value of every facet.",
                "docs": docs,
            }
        )

        # Depth is DERIVED by BFS from the seed frontier, never annotated by hand.
        depths = compute_depths([n.gold_node_id for n in nodes], edges, seed_ids)
        # Layer 4 of the firewall: a nonce appearing nowhere but here. If it ever
        # shows up in a serialized request, gold reached a model and the run is void.
        canary = mint("synth", f"s{t}")
        canaries.add(canary)
        graphs.append(
            {
                "gold_canary": canary,
                "gold_suite": "synth",
                "gold_task_key": f"s{t}",
                "gold_nodes": [_as_dict(n, depths) for n in nodes],
                "gold_edges": [_as_dict(e, None) for e in edges],
                "gold_facets": [_as_dict(f, None) for f in facets],
                "gold_seed_node_ids": seed_ids,
                "gold_graph_version": "v1",
                "gold_answer": " ".join(answers),
                "gold_aliases": [],
            }
        )

    payload = "\n".join(json.dumps(r, sort_keys=True) for r in public)
    corpus_hash = hashlib.sha256(payload.encode()).hexdigest()[:16]
    cdir = root / "data" / "corpora" / "synth" / corpus_hash
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / "tasks.jsonl").write_text(payload + "\n")

    # THROUGH THE CHOKE POINT, not around it. write_graphs is where the canary and the
    # corpus hash are stamped, and a builder that writes its own jsonl is a suite that
    # silently opts out of both. Synth mints its own canaries above, which write_graphs
    # leaves alone; what it gains here is `gold_corpus_hash`, without which gold from a
    # 5-task build could score runs rolled against a 12-task corpus -- the exact pairing
    # `make smoke` produces on any machine that has built synth twice.
    from pi_eval.build.common import write_graphs

    gpath = write_graphs(root, "synth", "v1", graphs, corpus_hash=corpus_hash)
    register(canaries, root)

    return cdir / "tasks.jsonl", gpath, corpus_hash


def _as_dict(obj, depths) -> dict:
    from dataclasses import fields

    out = {}
    for f in fields(obj):
        v = getattr(obj, f.name)
        out[f.name] = list(v) if isinstance(v, tuple) else v
    if depths is not None and "gold_depth" in out:
        out["gold_depth"] = depths.get(out["gold_node_id"])
    return out


def load_graphs_for_test(path: Path) -> list[GoldGraph]:
    from pi_eval.gold import GoldGraph as _G

    out = []
    for line in path.read_text().splitlines():
        d = json.loads(line)
        out.append(
            _G(
                gold_suite=d["gold_suite"],
                gold_task_key=d["gold_task_key"],
                gold_nodes=tuple(GoldNode(**_t(n)) for n in d["gold_nodes"]),
                gold_edges=tuple(GoldEdge(**_t(e)) for e in d["gold_edges"]),
                gold_facets=tuple(GoldFacet(**_t(f)) for f in d["gold_facets"]),
                gold_seed_node_ids=tuple(d["gold_seed_node_ids"]),
                gold_graph_version=d["gold_graph_version"],
                gold_answer=d["gold_answer"],
            )
        )
    return out


def _t(d: dict) -> dict:
    return {k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items()}
