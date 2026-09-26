"""Shared machinery for the corpus/gold split that every real suite performs.

Four things every builder has to do IDENTICALLY, so they live here once rather than being
re-derived (and re-diverged) three times:

  * materialise a raw download under data/raw/<suite>/ and VERIFY a sha256, so an upstream
    file that is silently re-uploaded becomes a hard error instead of a drifting benchmark
    whose old numbers can no longer be reproduced;

  * mint evidence uids with the SAME (doc_id, span) convention the adapter will use. Gold
    says "need v is resolved by uid U"; the rollout says "the retriever returned uid U".
    If those two conventions drift by one character, every node silently becomes
    unresolvable and RNR reads 0.0 for a reason no plot will ever show. `unit_uid` is the
    single definition, and each suite's tests assert the adapter reproduces it;

  * derive depth by BFS from the seed frontier — never annotate it — and cut facets as the
    weakly-connected components that remain AFTER deleting depth-0 nodes, exactly as
    synth_build does. Keeping seed nodes out of every facet is what makes horizontal breadth
    and vertical depth separable rather than two views of the same number;

  * write the two artifacts to two physically separate trees. data/corpora/ is all an
    adapter may read; data/gold/ is read only by pi_eval. Nothing crosses.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Iterable, Sequence

from pi_eval.gold import GoldEdge, GoldFacet, GoldGraph, GoldNode, compute_depths
from pinq.ids import evidence_uid

_UA = "proactive-inquirer/0.1 (dataset builder)"


@dataclass(frozen=True, slots=True)
class BuildResult:
    """What every builder returns: the two paths, the content hash, and the two counts a
    reviewer will ask for first — how many tasks survived, and how many a contamination
    control removed."""

    corpus: Path
    gold: Path
    corpus_hash: str
    n_tasks: int
    n_excluded: int = 0


# ------------------------------------------------------------------ raw cache + integrity


def sha256_file(path: Path) -> str:
    d = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            d.update(chunk)
    return d.hexdigest()


def download(url: str, dest: Path, *, timeout: int = 900) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=timeout) as r, tmp.open("wb") as out:
        while chunk := r.read(1 << 20):
            out.write(chunk)
    tmp.replace(dest)  # atomic: a killed download never leaves a valid-looking cache entry
    return dest


def require_raw(
    dest: Path, url: str, *, expect_sha256: str | None = None, allow_download: bool = True
) -> Path:
    """Return a verified raw file, fetching it only if it is not already cached.

    INTEGRITY IS TWO-TIERED, deliberately. `expect_sha256` is a HARD pin: the digest we
    ourselves downloaded and recorded in this module's constants, and a mismatch raises.
    When no hard pin is supplied the first materialisation writes a `<file>.sha256` sidecar
    and every later load must match it — trust on first use, which still turns "upstream
    quietly replaced the file under my warm cache" into an exception.

    What we do NOT do is invent a plausible-looking constant for a file we never fetched.
    A wrong pin fails every honest download, and the first thing it teaches you is to pass
    the flag that disables checking.
    """
    if not dest.exists():
        if not allow_download:
            raise FileNotFoundError(
                f"missing raw input {dest}. Fetch it once with:\n  curl -L -o {dest} {url}\n"
                "Builders are offline-first: nothing in CI may depend on that URL being up."
            )
        download(url, dest)

    got = sha256_file(dest)
    pin = dest.with_name(dest.name + ".sha256")
    want = expect_sha256 or (pin.read_text().split()[0] if pin.exists() else None)
    if want and got != want:
        raise ValueError(
            f"sha256 mismatch for {dest}\n  expected {want}\n  got      {got}\n"
            "The upstream file changed, or the cache is corrupt. Delete it and refetch; if "
            "upstream genuinely republished, the suite_version must be bumped with it."
        )
    if not pin.exists():
        pin.write_text(f"{got}  {dest.name}\n")
    return dest


# ------------------------------------------------------------------ the uid convention


def doc_id(task_id: str, idx: int) -> str:
    """Documents are namespaced BY TASK because every suite here ships a per-task pool.

    Two tasks may include the same Wikipedia paragraph at different indices; scoping the
    doc_id to the task keeps a leave-one-out over task A from colliding with task B's memo.
    """
    return f"{task_id}:{idx}"


def unit_uid(corpus_id: str, task_id: str, idx: int, text: str) -> str:
    """THE convention, in one place: span is the whole paragraph, `0:len(text)`.

    The adapter mints the identical uid through EvidenceUnit.make(); this function exists so
    the gold side never has to re-guess it. The suite tests assert the two agree, because a
    disagreement here is invisible in every downstream number.
    """
    return evidence_uid(corpus_id, doc_id(task_id, idx), f"0:{len(text)}")


# ------------------------------------------------------------------ graph assembly


def _components(node_ids: Sequence[str], edges: Sequence[GoldEdge]) -> list[list[str]]:
    """Weakly-connected components by union-find; edge direction is ignored on purpose.

    A facet is "a strand of the question you could pursue independently", which is an
    undirected notion: two needs joined by a prerequisite belong to the same strand whichever
    way the arrow points.
    """
    parent = {n: n for n in node_ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for e in edges:
        a, b = e.gold_src_node_id, e.gold_dst_node_id
        if a in parent and b in parent:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

    groups: dict[str, list[str]] = {}
    for n in node_ids:
        groups.setdefault(find(n), []).append(n)
    return [sorted(v) for _, v in sorted(groups.items(), key=lambda kv: min(kv[1]))]


def finalize_graph(
    *,
    suite: str,
    task_key: str,
    nodes: Sequence[GoldNode],
    edges: Sequence[GoldEdge],
    seed_ids: Sequence[str],
    answer: str = "",
    aliases: Sequence[str] = (),
    version: str = "v1",
) -> dict:
    """Derive depth and facets, stamp them onto the nodes, and return the JSON row.

    Every builder ends here so that the derived fields cannot be computed three subtly
    different ways. Depth comes from `compute_depths` (BFS over prerequisites from the seed
    frontier); a node unreachable from any seed keeps depth None and is counted in
    orphan_rate rather than being quietly assigned a number.
    """
    node_ids = [n.gold_node_id for n in nodes]
    depths = compute_depths(node_ids, list(edges), list(seed_ids))

    inner = [n for n in node_ids if depths.get(n) != 0]
    comps = _components(inner, edges)  # _components ignores edges touching a deleted seed
    facet_of: dict[str, str | None] = {n: None for n in node_ids}
    facets: list[GoldFacet] = []
    for i, comp in enumerate(comps):
        fid = f"{task_key}_f{i}"
        for n in comp:
            facet_of[n] = fid
        facets.append(
            GoldFacet(
                gold_suite=suite,
                gold_task_key=task_key,
                gold_facet_id=fid,
                gold_node_ids=tuple(comp),
            )
        )

    stamped = [
        replace(
            n,
            gold_depth=depths[n.gold_node_id],
            gold_facet_id=facet_of[n.gold_node_id],
            gold_graph_version=version,
        )
        for n in nodes
    ]
    return {
        "gold_suite": suite,
        "gold_task_key": task_key,
        "gold_nodes": [as_dict(n) for n in stamped],
        "gold_edges": [as_dict(replace(e, gold_graph_version=version)) for e in edges],
        "gold_facets": [as_dict(f) for f in facets],
        "gold_seed_node_ids": list(seed_ids),
        "gold_graph_version": version,
        "gold_answer": answer,
        "gold_aliases": list(aliases),
    }


def as_dict(obj) -> dict:
    """Dataclass -> JSON row. Tuples become lists; read_graphs turns them back."""
    out: dict = {}
    for f in fields(obj):
        v = getattr(obj, f.name)
        out[f.name] = list(v) if isinstance(v, tuple) else v
    return out


# ------------------------------------------------------------------ the two output trees


def write_corpus(root: Path, suite: str, records: Iterable[dict]) -> tuple[Path, str]:
    """PUBLIC artifact. Content-addressed: the directory name IS the hash of the payload, so
    two runs that produce different corpora can never overwrite each other's provenance."""
    payload = "\n".join(json.dumps(r, sort_keys=True, ensure_ascii=False) for r in records)
    chash = hashlib.sha256(payload.encode()).hexdigest()[:16]
    cdir = root / "data" / "corpora" / suite / chash
    cdir.mkdir(parents=True, exist_ok=True)
    path = cdir / "tasks.jsonl"
    path.write_text(payload + "\n")
    return path, chash


def write_graphs(
    root: Path,
    suite: str,
    version: str,
    graphs: Iterable[dict],
    *,
    corpus_hash: str = "",
) -> Path:
    """GOLD artifact. Separate tree, separate permissions, read only by pi_eval.

    ALSO MINTS THE CANARY, and does it here rather than in each builder for one reason: this
    is the single choke point every real suite passes through, so layer 4 of the firewall
    becomes a property of "gold was written" instead of "the builder remembered to". It had
    been implemented only in `synth_build`, which meant `pi verify firewall` scanned the cache
    against a registry containing nonces for the ONE suite that carries no real corpus text --
    the layer was vacuous on musique, strategyqa, wiki2, drgym and tau2, i.e. on every suite
    where a leak could actually happen, while reporting "clean".

    Idempotent: a row that already carries a `gold_canary` keeps it, so a builder that mints
    its own (synth) is untouched and a re-serialized graph does not get a second nonce.
    Deterministic under `PI_CANARY_SALT`, so a rebuild reproduces the same registry.

    ALSO STAMPS `gold_corpus_hash`. The corpora tree is content-addressed and the gold tree is
    not -- one file per (suite, graph_version), last build wins -- so nothing otherwise
    connects a graph to the corpus it was derived from, and gold built from corpus A can score
    runs rolled against corpus B. Same task ids, different evidence uids, every match silently
    missing. `pi score` refuses that pairing; it can only do so if the hash is written here.
    """
    from pi_eval.canary import mint, register

    rows = []
    minted: set[str] = set()
    for g in graphs:
        row = dict(g)
        if corpus_hash and not row.get("gold_corpus_hash"):
            row["gold_corpus_hash"] = corpus_hash
        canary = str(row.get("gold_canary") or "")
        if not canary:
            canary = mint(suite, str(row.get("gold_task_key", "")))
            row["gold_canary"] = canary
        minted.add(canary)

        # AND INTO A STRING SOMETHING COULD ACTUALLY LEAK. A nonce sitting in its own sibling
        # field is unreachable: layer 4 scans serialized requests for the token, and no code
        # path that leaks gold would ever carry `gold_canary` along with the text it leaked.
        # The registry was armed and the detector could not fire.
        #
        # `gold_answer` is the carrier, chosen because it is the most dangerous field to leak
        # (it is the answer key) and because it is the only gold string that NEVER legitimately
        # reaches a model: the judges are handed `gold_text` key points, `pi gold questions`
        # emits `gold_text` and `gold_aliases`, and `gold_answer` is read at exactly two places
        # in `pi_eval.score`, both of which compare it locally and strip the nonce first. So a
        # canary in a cached request means the answer key reached a model, and there is no
        # benign explanation -- which is what makes the check worth having.
        answer = str(row.get("gold_answer") or "")
        if answer and canary not in answer:
            row["gold_answer"] = f"{answer} {canary}"
        rows.append(row)

    gdir = root / "data" / "gold" / "graphs" / suite
    gdir.mkdir(parents=True, exist_ok=True)
    path = gdir / f"{version}.jsonl"
    path.write_text("\n".join(json.dumps(g, sort_keys=True, ensure_ascii=False) for g in rows))
    # Append-only, and registered AFTER the graphs are on disk: a canary in the registry with
    # no gold behind it is harmless, whereas gold on disk with no canary registered is a hole.
    if minted:
        register(minted, root)
    return path


def read_graphs(path: Path) -> list[GoldGraph]:
    """Path-addressed loader for tests and builders. The env-guarded reader lives in
    pi_eval.gold.load_graphs and is what production scoring uses."""
    out: list[GoldGraph] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        out.append(
            GoldGraph(
                gold_suite=d["gold_suite"],
                gold_task_key=d["gold_task_key"],
                gold_nodes=tuple(GoldNode(**_t(n)) for n in d["gold_nodes"]),
                gold_edges=tuple(GoldEdge(**_t(e)) for e in d["gold_edges"]),
                gold_facets=tuple(GoldFacet(**_t(f)) for f in d["gold_facets"]),
                gold_seed_node_ids=tuple(d["gold_seed_node_ids"]),
                gold_graph_version=d["gold_graph_version"],
                gold_answer=d["gold_answer"],
                gold_aliases=tuple(d.get("gold_aliases", [])),
                gold_canary=str(d.get("gold_canary", "")),
                gold_corpus_hash=str(d.get("gold_corpus_hash", "")),
            )
        )
    return out


def _t(d: dict) -> dict:
    return {k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items()}
