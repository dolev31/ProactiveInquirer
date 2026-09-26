#!/usr/bin/env python3
"""Copy every isolated score store under given roots to a backup root, verify the copy
file-for-file, and write a committed census beside the (uncommitted) backup.

This is the tooling for lane L0.10b (`artifacts/evidence_backup_20260919/RESULT.md`): those
stores are untracked by design (`.gitignore`: "content-addressed and regenerable") and several
exist in exactly one place on one machine. Two are already confirmed gone as of 2026-09-19 and
this tool cannot bring them back: `artifacts/granite_family_20260918/scores_parquet_dev` (the
dev-gate checkpoint-selection store cited in that campaign's own `RESULT.md`, independently
reconfirmed missing by `artifacts/decomposition_test_20260918/RESULT.md`) and the `random_q`
all-three-suites farm cited by `paper/results.tex` (`artifacts/random_q_all_suites_20260918`'s
own store, built under a private scratch root that no longer exists on this machine). Neither is
backed up here because neither survives; see `artifacts/evidence_backup_20260919/RESULT.md` for
what those two gaps do and do not affect. What this tool can do is stop the same thing happening
to every store that still exists.

COPY ONLY. Never move, never hard-link, never delete -- `shutil.copytree` with `copy2` leaves
the source exactly as it was, and this module has no code path that removes a file anywhere.
Nothing here runs `pi compact` or `pi score`; a census is read from tables a prior scoring pass
already wrote.

Usage:
    .venv/bin/python -m scripts.evidence_backup.backup \\
        --artifacts-root artifacts \\
        --extra-root /path/to/a/farm/outside/the/repo \\
        --backup-name evidence-20260919 \\
        --out artifacts/evidence_backup_20260919/backup_run.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from scripts.evidence_backup.census import compute_census
from scripts.evidence_backup.discover import find_score_stores
from scripts.evidence_backup.manifest import diff_manifests, sha256_manifest

REPO_ROOT = Path(__file__).resolve().parents[2]


def default_backup_root() -> Path:
    """`$PI_CORPUS_BACKUP` if set, else `~/pi-corpus-backup` resolved at run time via
    `Path.home()` -- never a literal path in source, so this file carries no machine-specific
    string for `check_no_home_paths.sh` to catch."""
    env = os.environ.get("PI_CORPUS_BACKUP")
    return Path(env) if env else Path.home() / "pi-corpus-backup"


def _safe_display(path: Path, home: Path | None = None) -> str:
    """`str(path)` with a leading `$HOME` in place of the real home directory.

    Every string this module writes ends up either committed (`*.census.json`, and the `--out`
    summary if pointed under a committed directory) or copy-pasted into a committed `RESULT.md`.
    `scripts/check_no_home_paths.sh` greps committed text for a literal `/Users/<name>/`, and a
    store backed up by this tool is not always under the repo -- lane L0.10b's own extra roots
    (`~/pinq-ablation`, `/private/tmp/...`) prove a store can live directly under `$HOME`.
    Substituting the prefix once here means every caller gets a safe string for free.
    """
    home = Path(home) if home is not None else Path.home()
    text, home_text = str(path), str(home)
    if text == home_text:
        return "$HOME"
    prefix = home_text.rstrip("/") + "/"
    return "$HOME/" + text[len(prefix) :] if text.startswith(prefix) else text


def store_size_bytes(store: Path) -> int:
    return sum(p.stat().st_size for p in Path(store).rglob("*") if p.is_file())


def copy_store(src: Path, dst: Path) -> None:
    """Copy `src` to `dst`, preserving structure and metadata (mtime, mode) the way `cp -a`
    would. `dirs_exist_ok=True` makes a rerun top up an existing backup rather than fail; nothing
    in this function ever deletes a file, in `src` or in a `dst` that already has extra content
    from an earlier run.
    """
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst, dirs_exist_ok=True, copy_function=shutil.copy2)


def verify_copy(src: Path, dst: Path) -> dict:
    """sha256-manifest both sides and diff them. `ok` is True only when the backup is missing
    nothing, has nothing extra, and matches every shared file byte-for-byte.

    Returns the two raw manifests too (`source_manifest`, `backup_manifest`), not just the diff:
    `backup_one` persists both beside the backup (lane L0.10b's own requirement -- a future
    reader must be able to prove a restored store matches without needing this machine again),
    and computing them here means neither side is ever hashed twice.
    """
    source_manifest = sha256_manifest(src)
    backup_manifest = sha256_manifest(dst)
    diff = diff_manifests(source_manifest, backup_manifest)
    ok = not any(diff.values())
    return {
        "ok": ok,
        "n_files": len(source_manifest),
        "source_manifest": source_manifest,
        "backup_manifest": backup_manifest,
        **diff,
    }


def census_path_for(rel: Path, census_root: Path) -> Path:
    """`<census_root>/<rel.parent>/<rel.name>.census.json` -- mirrors the store's position under
    its root so many stores that share a top-level name (`artifacts/gate/...` holds dozens) get
    distinct, collision-free census files instead of one file trying to describe all of them."""
    return census_root / rel.parent / f"{rel.name}.census.json"


def manifest_path_for(rel: Path, backup_root: Path, kind: str) -> Path:
    """`<backup_root>/<rel.parent>/<rel.name>.<kind>.sha256.json` -- a sibling of the copied
    store's own directory (`<rel.name>/`), never a file placed *inside* it. Putting a manifest
    inside the store directory it describes would make every later `verify_copy` against the
    live source see it as `unexpected_in_backup` forever after (the source never has it), which
    would turn today's clean verification into a permanent false failure. `kind` is `"source"`
    or `"backup"`.
    """
    return backup_root / rel.parent / f"{rel.name}.{kind}.sha256.json"


def backup_one(
    store: Path,
    *,
    rel: Path,
    backup_root: Path,
    census_root: Path,
) -> dict:
    """Copy one store to `backup_root/rel`, verify it, write its census to
    `census_path_for(rel, census_root)`, and return a summary row. Raises `EvidenceMismatch` if
    the post-copy verification does not come back clean -- a silent partial backup is worse than
    a loud failed one.
    """
    dst = backup_root / rel
    size_bytes = store_size_bytes(store)
    copy_store(store, dst)
    verification = verify_copy(store, dst)
    if not verification["ok"]:
        raise EvidenceMismatch(str(store), verification)

    manifest_paths = {}
    for kind in ("source", "backup"):
        p = manifest_path_for(rel, backup_root, kind)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(verification[f"{kind}_manifest"], indent=2, sort_keys=True) + "\n")
        manifest_paths[kind] = p

    census = compute_census(store)
    census["store_source"] = _safe_display(store)
    census["backup_relative_path"] = rel.as_posix()
    census["size_bytes"] = size_bytes
    census["source_manifest_path"] = _safe_display(manifest_paths["source"])
    census["backup_manifest_path"] = _safe_display(manifest_paths["backup"])
    out_path = census_path_for(rel, census_root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(census, indent=2, sort_keys=True) + "\n")

    return {
        "store": _safe_display(store),
        "backup_path": _safe_display(dst),
        "census_path": _safe_display(out_path),
        "source_manifest_path": _safe_display(manifest_paths["source"]),
        "backup_manifest_path": _safe_display(manifest_paths["backup"]),
        "size_bytes": size_bytes,
        "verification": verification,
    }


class EvidenceMismatch(RuntimeError):
    def __init__(self, store: str, verification: dict) -> None:
        super().__init__(f"backup of {store!r} did not verify: {verification!r}")
        self.store = store
        self.verification = verification


def _rows_for_root(root: Path, *, strip_parent: bool) -> list[tuple[Path, Path]]:
    """`(store, rel)` pairs for every store under `root`. `strip_parent=False` (the
    `--artifacts-root` case) computes `rel` directly against `root`, e.g.
    `artifacts/banking/scores_parquet` -> `banking/scores_parquet`. `strip_parent=True` (the
    `--extra-root` case) computes it against `root`'s own parent instead, so an external farm's
    own directory name becomes the leading path component, e.g. `/x/pinq-ablation/c1_score/...`
    -> `pinq-ablation/c1_score/...` -- traceable back to the farm it came from without ever
    hardcoding that farm's name in source.
    """
    root = Path(root)
    base = root.parent if strip_parent else root
    return [(store, store.relative_to(base)) for store in find_score_stores(root)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--artifacts-root", type=Path, default=REPO_ROOT / "artifacts")
    ap.add_argument(
        "--extra-root",
        type=Path,
        action="append",
        default=[],
        help="a farm/scratch directory outside --artifacts-root to scan and back up too; repeatable",
    )
    ap.add_argument("--backup-root", type=Path, default=default_backup_root())
    ap.add_argument("--backup-name", default="evidence-20260919")
    ap.add_argument(
        "--census-root", type=Path, default=REPO_ROOT / "artifacts" / "evidence_backup_20260919"
    )
    ap.add_argument("--out", type=Path, help="write a JSON summary of this run here")
    args = ap.parse_args(argv)

    rows = _rows_for_root(args.artifacts_root, strip_parent=False)
    for root in args.extra_root:
        rows += _rows_for_root(root, strip_parent=True)
    rows.sort(key=lambda sr: sr[1])

    backup_root = args.backup_root / args.backup_name
    results = []
    failures = []
    for store, rel in rows:
        try:
            results.append(
                backup_one(store, rel=rel, backup_root=backup_root, census_root=args.census_root)
            )
        except EvidenceMismatch as exc:
            failures.append({"store": _safe_display(store), "verification": exc.verification})

    summary = {
        "backup_root": _safe_display(backup_root),
        "n_stores": len(rows),
        "n_verified_ok": len(results),
        "n_failed": len(failures),
        "total_bytes": sum(r["size_bytes"] for r in results),
        "results": results,
        "failures": failures,
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(
        f"stores: {summary['n_stores']}  verified_ok: {summary['n_verified_ok']}  "
        f"failed: {summary['n_failed']}  total_bytes: {summary['total_bytes']}  "
        f"backup_root: {summary['backup_root']}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
