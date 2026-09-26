"""DeepResearchGym -> a PUBLIC query list and a GOLD key-point graph.

WHAT THE GOLD IS. The DRGym authors publish, per Researchy query, an aggregated list of key
points a good report should cover, committed at `key_point/<id>_aggregated.json`:

    {"question": str,
     "key_points": [{"point_number": int,
                     "point_content": str,
                     "original_point_number": [int, ...]}]}

Each entry becomes one required GoldNode with gold_provenance_primary="bench_author". These
are the benchmark authors' assertion of what the answer must contain -- which is a strong
claim about relevance and NO claim about causality, so gold_ablation_verdict stays
UNTESTABLE. We ran no ablation here; writing NECESSARY would put an unmeasured verdict in the
same column as measured ones.

THIS BUILDER EMITS NO EDGES, ON PURPOSE.
`original_point_number` is MERGE PROVENANCE: the aggregation step collapsed several
per-document key points into one, and the list records which pre-merge indices were folded
in. It is many-to-one and it points BACKWARD into a pipeline stage, not from one need to
another. Shipping it as a prerequisite DAG would be a fabrication with three visible
consequences: Depth would become a function of how aggressively the aggregator merged,
every "deep" node would be one that happened to absorb many duplicates, and the vertical-
proactivity claim would rest on an artefact of deduplication. So the DRGym graph is FLAT:
every node is a depth-0 seed, there are no edges and no facets, and that is the correct
shape rather than a missing one. The merge lists are preserved as an audit sidecar
(`merge_provenance.json`, gold side) so the decision is inspectable, never as edges.
tests/test_adapters_drgym.py asserts the edge set is empty.

gold_ev_uids IS EMPTY, TOO. The corpus is a hosted open-web index; there is no released
document set in which a key point could be located, so there is no uid a retriever is
guaranteed to be able to return. Resolution on this suite is judged (pi_eval.judges.kpr),
not matched by uid, and an invented uid would make RNR_resolve read 0.0 forever for a reason
no plot would show.

SOME FILES ARE EMPTY UPSTREAM. Counted at the pinned commit on 2026-08-24: the query list
has 1000 unique ids, `key_point/` has exactly 1000 `_aggregated.json` files, and the two id
sets match 1:1 -- but 24 of those files carry `key_points: []` (121_aggregated.json among
them), so a full build yields 976 tasks. A task with no key points has no recall denominator,
so it is filtered, COUNTED, and its id is written to the gold-side report. A task filtered
without a count is a silently shrinking denominator.

PROVENANCE. Files are fetched from raw.githubusercontent.com at a PINNED COMMIT SHA, never
from a branch. The query list carries a hard sha256 (we downloaded it and recorded the
digest). The 1000 key-point files carry per-file sha256 SIDECARS on first materialisation --
trust on first use -- because inventing 1000 constants for files we had not fetched is
exactly the failure mode pi_eval.build.common.require_raw warns about, and one wrong pin
teaches you to pass the flag that disables checking.

LICENCE. That repository has NO LICENCE FILE (GitHub's licence API returns 404, verified
2026-08-24). Nothing from it is redistributed here: this builder downloads to data/raw/ on
the user's machine, and the judge prompts are reimplemented from scratch in pi_eval.judges.
"""

from __future__ import annotations

import json
from pathlib import Path

from pi_eval.build.common import (
    BuildResult,
    finalize_graph,
    require_raw,
    write_corpus,
    write_graphs,
)
from pi_eval.gold import GoldNode

SUITE = "drgym"
GRAPH_VERSION = "v1"

BENCH_REPO = "cxcscmu/deepresearch_benchmarking"
# Pinned, not `main`: the key points ARE the gold, and a branch that moves under a warm cache
# changes every denominator in the table without changing a line of code.
BENCH_COMMIT = "d4d2433309d9da9be637c365618e7e01f8f9205a"
_RAW = f"https://raw.githubusercontent.com/{BENCH_REPO}/{BENCH_COMMIT}/"

QUERIES = "queries/researchy_queries_sample_doc_click.jsonl"
QUERIES_100 = "queries/researchy_queries_sample_doc_click_100.jsonl"
KEY_POINT_DIR = "key_point"

# Hard pin: the digest of the file this builder was written and verified against.
SHA256 = {
    QUERIES: "39c97a79da4be948e53a90cdf73c30c664388b39df1f9f5947a123ed6d7a1568",
}


def key_point_url(query_id: str) -> str:
    return f"{_RAW}{KEY_POINT_DIR}/{query_id}_aggregated.json"


def load_queries(path: Path, limit: int | None = None) -> list[dict[str, str]]:
    """`{"id": "879779", "query": "..."}` per line. 1000 lines at the pinned commit."""
    out: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            out.append({"id": str(rec["id"]), "query": str(rec["query"])})
            if limit is not None and len(out) >= limit:
                break
    return out


def _norm(s: str) -> str:
    return " ".join(s.lower().split())


def build(
    *,
    raw_dir: Path | None = None,
    root: Path | None = None,
    limit: int | None = None,
    allow_download: bool = True,
    verify: bool = True,
    queries_file: str = QUERIES,
) -> BuildResult:
    """Split the benchmark into data/corpora/drgym/ (queries) and data/gold/graphs/drgym/.

    Offline-first, like every builder here: a fixture directory plus allow_download=False is
    a complete build with no network.
    """
    root = root or Path.cwd()
    raw_dir = raw_dir or (root / "data" / "raw" / SUITE)
    # The upstream subpath is preserved under raw_dir (queries/..., key_point/...), so a
    # data/raw/drgym checkout is a mirror of the pinned tree rather than a flattened copy.
    qpath = require_raw(
        raw_dir / queries_file,
        _RAW + queries_file,
        expect_sha256=SHA256.get(queries_file) if verify else None,
        allow_download=allow_download,
    )
    queries = load_queries(qpath, limit)

    public: list[dict] = []
    graphs: list[dict] = []
    empty: list[str] = []  # committed upstream with key_points: []
    missing: list[str] = []  # no aggregated file at all
    question_mismatch: list[str] = []
    merges: dict[str, dict[str, list[int]]] = {}

    kp_dir = raw_dir / KEY_POINT_DIR
    for q in queries:
        qid = q["id"]
        dest = kp_dir / f"{qid}_aggregated.json"
        if not dest.exists() and not allow_download:
            missing.append(qid)
            continue
        # No hard pin: TOFU sidecars, one per file. See the module docstring.
        path = require_raw(dest, key_point_url(qid), allow_download=allow_download)

        text = path.read_text(encoding="utf-8").strip()
        if not text:
            empty.append(qid)
            continue
        # A malformed file is NOT filtered: an unparseable input is corruption, and silently
        # dropping it would shrink the denominator for a reason nobody would ever see.
        data = json.loads(text)
        points = data.get("key_points") or []
        if not points:
            empty.append(qid)
            continue

        if data.get("question") and _norm(str(data["question"])) != _norm(q["query"]):
            question_mismatch.append(qid)

        public.append({"id": qid, "question": q["query"]})
        graphs.append(_graph(qid, points))
        merge = {
            f"kp{p['point_number']}": list(p.get("original_point_number") or []) for p in points
        }
        if any(merge.values()):
            merges[qid] = merge

    corpus, chash = write_corpus(root, SUITE, public)
    gold = write_graphs(root, SUITE, GRAPH_VERSION, graphs, corpus_hash=chash)

    (gold.parent / "keypoint_filter_report.json").write_text(
        json.dumps(
            {
                "bench_repo": BENCH_REPO,
                "bench_commit": BENCH_COMMIT,
                "queries_file": queries_file,
                "n_queries": len(queries),
                "n_tasks": len(public),
                "n_filtered_empty_key_points": len(empty),
                "filtered_task_ids": empty,
                "n_missing_key_point_files": len(missing),
                "missing_task_ids": missing,
                "n_question_text_mismatch": len(question_mismatch),
                "question_text_mismatch_ids": question_mismatch,
            },
            indent=1,
            sort_keys=True,
        )
    )
    # Preserved for audit, NEVER emitted as edges. See the module docstring.
    (gold.parent / "merge_provenance.json").write_text(
        json.dumps(
            {
                "what": "original_point_number: which pre-merge key points were folded into "
                "each aggregated point. Merge provenance, not a dependency DAG; it is "
                "deliberately not converted into GoldEdges.",
                "bench_commit": BENCH_COMMIT,
                "by_task": merges,
            },
            indent=1,
            sort_keys=True,
        )
    )
    return BuildResult(corpus, gold, chash, len(public), len(empty) + len(missing))


def _graph(qid: str, points: list[dict]) -> dict:
    """One task's key points as a FLAT graph: all seeds, no edges, no facets."""
    nodes = [
        GoldNode(
            gold_suite=SUITE,
            gold_task_key=qid,
            # kp<point_number> so a fidelity comparison against upstream's own per-point
            # labels is a prefix strip rather than a lookup table.
            gold_node_id=f"kp{p['point_number']}",
            gold_text=str(p["point_content"]),
            gold_kind="fact",
            gold_provenance=("bench_author",),
            gold_provenance_primary="bench_author",
            gold_partition="required",
            gold_discoverability="kb",
            # The benchmark says this content is required. We ran no ablation, so the causal
            # verdict stays UNTESTABLE and the ablation fields stay at zero.
            gold_ablation_verdict="UNTESTABLE",
            # Confidence that this IS the published key point, not that the point is true.
            gold_confidence=1.0,
            gold_graph_version=GRAPH_VERSION,
        )
        for p in points
    ]
    return finalize_graph(
        suite=SUITE,
        task_key=qid,
        nodes=nodes,
        edges=[],  # merge provenance is not dependency
        seed_ids=[n.gold_node_id for n in nodes],
        answer="",  # the deliverable is a report; there is no short gold answer
        version=GRAPH_VERSION,
    )
