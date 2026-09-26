"""`pi data` — fetch, build and inventory the corpora and gold graphs.

WHY THIS COMMAND EXISTS AT ALL. Until now every corpus in this repository was produced by
typing a Python one-liner into a shell (`pi_eval.build.musique_build.build(...)`), while
docs/REPRODUCE.md documented `pi data fetch --suite musique --verify-sha`. A documented
command that does not exist is worse than an undocumented one: the reader concludes the
pipeline is reproducible and has no way to find out otherwise until they try. This module
makes the documented command the real one.

THREE PROPERTIES, EACH LOAD-BEARING

  * IDEMPOTENT. `fetch` re-verifies a cached file instead of re-downloading it, and `build`
    writes to a CONTENT-ADDRESSED corpus directory, so building twice from the same raw
    input produces the same `corpus_hash` and the same path rather than a second corpus that
    silently shadows the first.

  * IT SAYS WHAT IT VERIFIED. Every fetched file is printed with its digest and with whether
    that digest was checked against a HARD PIN (a constant recorded in the builder) or against
    a trust-on-first-use sidecar. Those are different strengths of guarantee and collapsing
    them into a single "ok" would overstate the weaker one.

  * THERE IS NO --no-verify. A flag that skips the checksum is a flag that gets passed the
    first time an honest download fails, and after that the pin protects nothing. If upstream
    genuinely republished a file, the fix is to bump the pin and the suite version together.

tau2 is deliberately ABSENT from the registry: it is self-sourced from a repo checkout
(`TAU2_DATA_DIR`) rather than from anything this command could fetch, and its builder is
owned elsewhere. `pi data status` says so rather than pretending the suite does not exist.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

# Suites this command can drive end to end, in reporting order.
SUITES = ("synth", "musique", "strategyqa", "wiki2", "drgym")

# Suites that exist but that `pi data` does not own, with the reason. Printed by `status` so
# a missing row is never mistaken for a missing suite.
NOT_OWNED = {
    "tau2": "self-sourced from a tau2-bench checkout (TAU2_DATA_DIR); no builder driver here",
    "pare": "self-sourced from a pare checkout (PARE_BENCHMARK_SPLITS_DIR); scenarios, no corpus",
}

GRAPH_VERSION = "v1"

# How many built corpora `status` lists before it summarises the rest.
MAX_CORPORA_SHOWN = 1


@dataclass(frozen=True, slots=True)
class Fetched:
    """One raw input that is now on disk and has been checked."""

    path: Path
    sha256: str
    pinned: bool  # True: checked against a constant in the builder. False: TOFU sidecar.

    @property
    def size(self) -> int:
        return self.path.stat().st_size if self.path.exists() else 0


def _fetched(path: Path, *, pinned: bool) -> Fetched:
    from pi_eval.build.common import sha256_file

    return Fetched(path, sha256_file(path), pinned)


# --------------------------------------------------------------------------- per-suite fetch
#
# Each function reuses its builder's OWN url and digest constants. Re-declaring them here
# would create a second source of truth for what "the pinned file" means, and the two would
# drift the first time an upstream url moved.


def _fetch_synth(raw_dir: Path, *, split: str, offline: bool) -> list[Fetched]:
    return []  # generated from a seed; there is nothing to download and nothing to pin


def _fetch_musique(raw_dir: Path, *, split: str, offline: bool) -> list[Fetched]:
    from pi_eval.build import musique_build as m
    from pi_eval.build.common import require_raw

    name = m.FILES[split]
    split_file = require_raw(
        raw_dir / name,
        m._HF + name,
        expect_sha256=m.SHA256[name],
        allow_download=not offline,
    )
    # The authors publish the single-hop exclusion list only inside the full data zip, so
    # this pulls (and verifies) the zip when the bare json is not already cached.
    exclusions = m.ensure_exclusions(raw_dir, allow_download=not offline, verify=True)
    return [_fetched(split_file, pinned=True), _fetched(exclusions, pinned=True)]


def _fetch_strategyqa(raw_dir: Path, *, split: str, offline: bool) -> list[Fetched]:
    from pi_eval.build import strategyqa_build as s
    from pi_eval.build.common import require_raw

    archive = require_raw(
        raw_dir / s.ZIP_NAME,
        s.ZIP_URL,
        expect_sha256=s.SHA256[s.ZIP_NAME],
        allow_download=not offline,
    )
    return [_fetched(archive, pinned=True)]


def _fetch_wiki2(raw_dir: Path, *, split: str, offline: bool) -> list[Fetched]:
    from pi_eval.build import wiki2_build as w
    from pi_eval.build.common import require_raw

    name = f"{split}.parquet"
    parquet = require_raw(
        raw_dir / name,
        w._HF + name,
        expect_sha256=w.SHA256[name],
        allow_download=not offline,
    )
    return [_fetched(parquet, pinned=True)]


def _fetch_drgym(raw_dir: Path, *, split: str, offline: bool) -> list[Fetched]:
    """Only the query list is pinned and fetched here.

    The 1000 `key_point/<qid>_aggregated.json` files ARE the gold, and there is one HTTP
    request per query id, so they are pulled by `build` (which knows how many the caller
    asked for) rather than unconditionally by `fetch`. They carry trust-on-first-use
    sidecars, not hard pins, and `build` reports that.
    """
    from pi_eval.build import drgym_build as d
    from pi_eval.build.common import require_raw

    queries = require_raw(
        raw_dir / d.QUERIES,
        d._RAW + d.QUERIES,
        expect_sha256=d.SHA256[d.QUERIES],
        allow_download=not offline,
    )
    return [_fetched(queries, pinned=True)]


FETCHERS: dict[str, Callable[..., list[Fetched]]] = {
    "synth": _fetch_synth,
    "musique": _fetch_musique,
    "strategyqa": _fetch_strategyqa,
    "wiki2": _fetch_wiki2,
    "drgym": _fetch_drgym,
}

# Splits each suite accepts, first entry the default. `pi data build --suite X` with no
# --split therefore always names a real file rather than guessing.
SPLITS: dict[str, tuple[str, ...]] = {
    "synth": ("test",),
    "musique": ("train", "dev"),
    "strategyqa": ("train",),
    "wiki2": ("dev", "train"),
    "drgym": ("full",),
}


def default_split(suite: str) -> str:
    return SPLITS[suite][0]


# --------------------------------------------------------------------------- per-suite build


@dataclass(frozen=True, slots=True)
class Built:
    suite: str
    corpus: Path
    gold: Path
    corpus_hash: str
    n_tasks: int
    n_excluded: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "suite": self.suite,
            "corpus": str(self.corpus),
            "gold": str(self.gold),
            "corpus_hash": self.corpus_hash,
            "n_tasks": self.n_tasks,
            "n_excluded": self.n_excluded,
        }


def build_suite(
    suite: str,
    *,
    root: Path,
    split: str | None = None,
    limit: int | None = None,
    offline: bool = False,
    raw_dir: Path | None = None,
) -> Built:
    """Drive one suite's builder and normalise its return.

    `synth_build.build` predates `BuildResult` and returns a bare tuple; every other builder
    returns `BuildResult`. Normalising here rather than changing five builders keeps this
    change out of files other work is touching.

    `raw_dir` overrides where the pinned inputs are read from and defaults to
    `<root>/data/raw/<suite>`. It exists so a build can be driven from a mirror or from a
    committed test fixture WITHOUT also redirecting where the corpus and gold are written —
    those two paths are the reason `root` cannot serve both purposes.
    """
    split = split or default_split(suite)
    if split not in SPLITS[suite]:
        raise SystemExit(f"suite {suite!r} has no split {split!r}; known: {list(SPLITS[suite])}")

    if suite == "synth":
        from pi_eval.build.synth_build import build as build_synth

        n_tasks = limit if limit is not None else 24
        corpus, gold, chash = build_synth(n_tasks=n_tasks, root=root)
        return Built("synth", corpus, gold, chash, n_tasks, 0)

    if suite == "musique":
        from pi_eval.build.musique_build import build as build_musique

        res = build_musique(
            split=split,
            root=root,
            raw_dir=raw_dir,
            limit=limit,
            allow_download=not offline,
            verify=True,
        )
    elif suite == "strategyqa":
        from pi_eval.build.strategyqa_build import build as build_sqa

        res = build_sqa(
            root=root, raw_dir=raw_dir, limit=limit, allow_download=not offline, verify=True
        )
    elif suite == "wiki2":
        from pi_eval.build.wiki2_build import build as build_wiki2

        res = build_wiki2(
            split=split,
            root=root,
            raw_dir=raw_dir,
            limit=limit,
            allow_download=not offline,
            verify=True,
        )
    elif suite == "drgym":
        from pi_eval.build.drgym_build import build as build_drgym

        res = build_drgym(
            root=root, raw_dir=raw_dir, limit=limit, allow_download=not offline, verify=True
        )
    else:  # pragma: no cover - SUITES is the only caller-facing list
        raise SystemExit(f"unknown suite {suite!r}; known: {list(SUITES)}")

    return Built(suite, res.corpus, res.gold, res.corpus_hash, res.n_tasks, res.n_excluded)


# --------------------------------------------------------------------------- inventory


@dataclass(frozen=True, slots=True)
class Inventory:
    suite: str
    raw_files: int
    raw_bytes: int
    raw_with_digest: int  # files carrying a .sha256 sidecar -- what `raw_verified` used to mean
    raw_verified: int  # files whose digest was RECOMPUTED and matched (0 unless --verify)
    corpora: tuple[tuple[str, int], ...]  # (corpus_hash, n_tasks)
    gold_version: str
    gold_graphs: int
    gold_nodes: int
    raw_mismatched: tuple[str, ...] = ()
    corpus_mismatched: tuple[str, ...] = ()  # dirs whose bytes no longer hash to their name

    @property
    def ok(self) -> bool:
        return not self.raw_mismatched and not self.corpus_mismatched

    @property
    def has_raw(self) -> bool:
        return self.raw_files > 0

    @property
    def has_corpus(self) -> bool:
        return bool(self.corpora)

    @property
    def has_gold(self) -> bool:
        return self.gold_graphs > 0


def _count_lines(path: Path) -> int:
    return sum(1 for line in path.read_text().splitlines() if line.strip())


def _sha256(path: Path) -> str:
    d = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            d.update(chunk)
    return d.hexdigest()


def corpus_hash_of(tasks: Path) -> str:
    """Recompute the content hash a corpus directory is NAMED for.

    `write_corpus` writes `payload + "\n"` and names the directory sha256(payload)[:16], so
    this is the exact inverse. Cheap enough to run unconditionally (the corpora total ~90 MB)
    and worth it: corpus_hash is inside semantic_hash, so a corpus whose bytes no longer match
    its directory name silently changes what every run_id built from it means.
    """
    payload = tasks.read_text()
    if payload.endswith("\n"):
        payload = payload[:-1]
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def inventory(
    root: Path, suite: str, *, graph_version: str = GRAPH_VERSION, verify_raw: bool = False
) -> Inventory:
    """What is on disk for one suite. Reads only; never fetches and never builds.

    `raw_verified` USED TO COUNT SIDECARS, NOT DIGESTS: `sum(1 for p in files if
    p.with_name(p.name + ".sha256").exists())` -- present, therefore "verified". A raw file
    truncated mid-download or edited by hand reported as verified for as long as its sidecar
    survived, which is always, because nothing deletes it. Nothing here checked the corpus
    directory's own content hash either, though the directory NAME is that hash and it is
    inside semantic_hash.

    Raw verification is opt-in because raw is 1.5 GB; the corpus check always runs.
    """
    raw_dir = root / "data" / "raw" / suite
    files = [p for p in raw_dir.rglob("*") if p.is_file() and p.suffix != ".sha256"]
    with_digest = [p for p in files if p.with_name(p.name + ".sha256").exists()]
    verified = 0
    raw_bad: list[str] = []
    if verify_raw:
        for f in with_digest:
            want = f.with_name(f.name + ".sha256").read_text().split()[0]
            if _sha256(f) == want:
                verified += 1
            else:
                raw_bad.append(f.name)

    corpora: list[tuple[str, int]] = []
    corpus_bad: list[str] = []
    base = root / "data" / "corpora" / suite
    if base.is_dir():
        for d in sorted(base.iterdir()):
            tasks = d / "tasks.jsonl"
            if tasks.exists():
                corpora.append((d.name, _count_lines(tasks)))
                if corpus_hash_of(tasks) != d.name:
                    corpus_bad.append(d.name)

    gold_path = root / "data" / "gold" / "graphs" / suite / f"{graph_version}.jsonl"
    n_graphs = n_nodes = 0
    if gold_path.exists():
        for line in gold_path.read_text().splitlines():
            if not line.strip():
                continue
            n_graphs += 1
            n_nodes += len(json.loads(line).get("gold_nodes", ()))

    return Inventory(
        suite=suite,
        raw_files=len(files),
        raw_bytes=sum(p.stat().st_size for p in files),
        raw_with_digest=len(with_digest),
        raw_verified=verified,
        raw_mismatched=tuple(raw_bad),
        corpus_mismatched=tuple(corpus_bad),
        corpora=tuple(corpora),
        gold_version=graph_version,
        gold_graphs=n_graphs,
        gold_nodes=n_nodes,
    )


def _inventory_dict(inv: Inventory) -> dict[str, Any]:
    return {
        "suite": inv.suite,
        "raw_files": inv.raw_files,
        "raw_bytes": inv.raw_bytes,
        "raw_with_digest": inv.raw_with_digest,
        "raw_verified": inv.raw_verified,
        "raw_mismatched": list(inv.raw_mismatched),
        "corpus_mismatched": list(inv.corpus_mismatched),
        "ok": inv.ok,
        "corpora": [{"corpus_hash": h, "n_tasks": n} for h, n in inv.corpora],
        "gold_version": inv.gold_version,
        "gold_graphs": inv.gold_graphs,
        "gold_nodes": inv.gold_nodes,
    }


def _size(n: int) -> str:
    x = float(n)
    for unit in ("B", "K", "M", "G", "T"):
        if x < 1024 or unit == "T":
            return f"{x:.0f}{unit}"
        x /= 1024
    return f"{x:.0f}T"  # pragma: no cover


# --------------------------------------------------------------------------- handlers


def cmd_data_fetch(a: argparse.Namespace) -> int:
    from pi_run.manifest import repo_root

    root = Path(a.root).resolve() if a.root else repo_root()
    suites = [a.suite] if a.suite else list(SUITES)
    rc = 0
    for suite in suites:
        if suite not in FETCHERS:
            print(f"{suite}: not fetchable by this command ({NOT_OWNED.get(suite, 'unknown')})")
            rc = 1
            continue
        raw_dir = root / "data" / "raw" / suite
        split = a.split or default_split(suite)
        try:
            got = FETCHERS[suite](raw_dir, split=split, offline=a.offline)
        except Exception as exc:  # noqa: BLE001 - the message IS the deliverable here
            print(f"{suite}: FAILED {type(exc).__name__}: {exc}")
            rc = 1
            continue
        if not got:
            print(f"{suite}: nothing to fetch (generated from a seed)")
            continue
        for f in got:
            kind = "hard pin" if f.pinned else "sidecar (trust-on-first-use)"
            print(f"{suite}: verified {f.path.relative_to(root)}  {_size(f.size)}")
            print(f"    sha256 {f.sha256}  [{kind}]")
    return rc


def cmd_data_build(a: argparse.Namespace) -> int:
    from pi_run.manifest import repo_root

    root = Path(a.root).resolve() if a.root else repo_root()
    suites = [a.suite] if a.suite else list(SUITES)
    rc = 0
    out: list[dict[str, Any]] = []
    for suite in suites:
        try:
            # `--raw-dir` was registered on the parser, documented in --help, accepted by
            # build_suite -- and never passed. A user pointing at a different pinned-input
            # directory got a build from the default one, with no error and a corpus_hash
            # printed as though it came from their inputs.
            built = build_suite(
                suite,
                root=root,
                split=a.split,
                limit=a.limit,
                offline=a.offline,
                raw_dir=Path(a.raw_dir).resolve() if getattr(a, "raw_dir", None) else None,
            )
        except Exception as exc:  # noqa: BLE001 - an actionable message, not a traceback
            print(f"{suite}: FAILED {type(exc).__name__}: {exc}")
            rc = 1
            continue
        print(
            f"{suite}: corpus_hash={built.corpus_hash} n_tasks={built.n_tasks} "
            f"n_excluded={built.n_excluded}"
        )
        print(f"    corpus {built.corpus.relative_to(root)}")
        print(f"    gold   {built.gold.relative_to(root)}")
        out.append(built.as_dict())
    if a.json:
        print(json.dumps(out, indent=2, sort_keys=True))
    return rc


def cmd_data_status(a: argparse.Namespace) -> int:
    from pi_run.manifest import repo_root

    root = Path(a.root).resolve() if a.root else repo_root()
    rows = [
        inventory(root, s, graph_version=a.graph_version, verify_raw=getattr(a, "verify", False))
        for s in SUITES
    ]
    if a.json:
        # The table caps the corpus column; this does not. A suite with several built corpora
        # is a real situation and the full list has to be reachable without reading the tree.
        print(json.dumps([_inventory_dict(i) for i in rows], indent=2, sort_keys=True))
        return 0

    # LEFT-aligned, and the corpus column is capped. A suite with three built corpora used to
    # overflow a right-aligned column and shove every later column out of line, so the table
    # became unreadable exactly when it had the most to say.
    head = (f"{'suite':<11} {'raw':<16} {'corpus':<32} {'gold':<22}").rstrip()
    print(head)
    print("-" * len(head))
    for inv in rows:
        raw = (
            f"{inv.raw_files} files {_size(inv.raw_bytes)}"
            if inv.has_raw
            else ("generated" if inv.suite == "synth" else "-")
        )
        shown = [f"{h[:12]} n={n}" for h, n in inv.corpora[:MAX_CORPORA_SHOWN]]
        extra = len(inv.corpora) - len(shown)
        corpus = (", ".join(shown) + (f", +{extra} more" if extra > 0 else "")) or "-"
        gold = f"{inv.gold_version} g={inv.gold_graphs} v={inv.gold_nodes}" if inv.has_gold else "-"
        print(f"{inv.suite:<11} {raw:<16} {corpus:<32} {gold:<22}".rstrip())
    print("-" * len(head))
    for suite, why in sorted(NOT_OWNED.items()):
        print(f"{suite:<11} not driven by `pi data`: {why}")
    # SAY WHICH CHECK ACTUALLY RAN. This line used to read "verified sidecars: musique=7/7"
    # from a count of files that merely HAD a .sha256 -- present, therefore verified. A raw file
    # truncated mid-download reported 7/7 for as long as its sidecar survived, which is always.
    if getattr(a, "verify", False):
        # THE DENOMINATOR IS EVERY RAW FILE, NOT JUST THE ONES CARRYING A DIGEST.
        #
        # This printed `raw_verified/raw_with_digest`, so wiki2 -- 5 raw inputs of which 2 have
        # a sidecar, in a 1 GB corpus -- reported "wiki2=2/2", a reassuring 100% produced by
        # re-basing the denominator on the files that happened to be checkable. The 3 with no
        # recorded digest at all are the ones a reader most needs to know about, and they
        # vanished from the line rather than being counted as unverifiable. That is the same
        # defect this whole check was written to remove: a number that looks like coverage and
        # is a property of what was measurable.
        print(
            "\nraw digests RECOMPUTED (denominator = every raw file): "
            + ", ".join(f"{i.suite}={i.raw_verified}/{i.raw_files}" for i in rows if i.has_raw)
        )
        nodig = [(i.suite, i.raw_files - i.raw_with_digest) for i in rows if i.has_raw]
        nodig = [(s, n) for s, n in nodig if n]
        if nodig:
            print(
                "  UNVERIFIABLE (no .sha256 recorded): "
                + ", ".join(f"{s}={n}" for s, n in nodig)
                + " -- these cannot be checked at all; re-fetch to record a digest."
            )
    else:
        print(
            "\nraw counts exclude .sha256 sidecars; sidecars PRESENT (not checked -- use "
            "--verify): "
            + ", ".join(f"{i.suite}={i.raw_with_digest}/{i.raw_files}" for i in rows if i.has_raw)
        )
    # The corpus directory NAME is the sha256 of its payload and is inside semantic_hash, so
    # this one is always recomputed: bytes that no longer match the name silently change what
    # every run_id built from that corpus means.
    bad = [(i.suite, d) for i in rows for d in i.corpus_mismatched]
    raw_bad = [(i.suite, f) for i in rows for f in i.raw_mismatched]
    if bad or raw_bad:
        for suite, d in bad:
            print(f"  CORPUS HASH MISMATCH  {suite}/{d}: contents do not hash to the directory")
        for suite, f in raw_bad:
            print(f"  RAW DIGEST MISMATCH   {suite}/{f}")
        return 1
    print("corpus content hashes: all match their directory names")
    return 0


# --------------------------------------------------------------------------- registration


def register(sub: argparse._SubParsersAction) -> None:
    """Attach `pi data` to the top-level dispatch table."""
    p = sub.add_parser("data", help="fetch, build and inventory corpora + gold")
    s = p.add_subparsers(dest="sub", required=True)

    f = s.add_parser("fetch", help="download and verify each suite's pinned raw inputs")
    f.add_argument("--suite", choices=sorted(SUITES))
    f.add_argument("--split")
    f.add_argument("--root")
    f.add_argument(
        "--verify-sha",
        action="store_true",
        help="explicit form of the default: every fetch verifies. There is no way to skip it.",
    )
    f.add_argument("--offline", action="store_true", help="fail rather than download")
    f.set_defaults(fn=cmd_data_fetch)

    b = s.add_parser("build", help="raw -> data/corpora/<suite>/<hash>/ + data/gold/graphs/")
    b.add_argument("--suite", choices=sorted(SUITES))
    b.add_argument("--split")
    b.add_argument("--limit", type=int)
    b.add_argument("--root")
    b.add_argument(
        "--raw-dir", help="read pinned inputs from here (default <root>/data/raw/<suite>)"
    )
    b.add_argument("--offline", action="store_true", help="fail rather than download")
    b.add_argument("--json", action="store_true")
    b.set_defaults(fn=cmd_data_build)

    st = s.add_parser("status", help="what is on disk, per suite, with counts and hashes")
    st.add_argument("--root")
    st.add_argument("--graph-version", default=GRAPH_VERSION)
    st.add_argument("--json", action="store_true", help="every corpus, uncapped")
    st.add_argument(
        "--verify",
        action="store_true",
        help="recompute raw file digests (1.5 GB; corpus hashes are always checked)",
    )
    st.set_defaults(fn=cmd_data_status)
