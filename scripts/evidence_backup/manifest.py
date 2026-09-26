"""Content hashing for backup verification and for the committed census.

Used twice, deliberately by the same function both times: once to compare a freshly copied
backup against its source file-for-file (`evidence-20260919`'s own verification step), and once
inside a census, so a store restored from backup later can be re-hashed and checked against the
committed census without needing the original copy to still exist anywhere.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

_CHUNK_BYTES = 1 << 20  # 1 MiB


def sha256_file(path: Path) -> str:
    """sha256 of one file's bytes, streamed so a 93 MB parquet file is never read whole into
    memory just to be hashed."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_manifest(directory: Path, pattern: str = "*") -> dict[str, str]:
    """`{relative posix path: sha256}` for every file under `directory` matching `pattern`
    (recursive). Paths are relative to `directory` and posix-separated so the manifest compares
    equal across machines regardless of the absolute prefix either copy lives under -- the whole
    point of a manifest that has to agree between a source and a backup root that are never the
    same path.
    """
    directory = Path(directory)
    return {
        p.relative_to(directory).as_posix(): sha256_file(p)
        for p in sorted(directory.rglob(pattern))
        if p.is_file()
    }


def diff_manifests(source: dict[str, str], backup: dict[str, str]) -> dict[str, list[str]]:
    """Where `source` and `backup` (two `sha256_manifest` results) disagree: files the backup is
    missing, files it has that the source does not, and files present in both whose hash
    differs. All three empty is the only passing result."""
    source_keys = set(source)
    backup_keys = set(backup)
    return {
        "missing_in_backup": sorted(source_keys - backup_keys),
        "unexpected_in_backup": sorted(backup_keys - source_keys),
        "sha_mismatch": sorted(k for k in source_keys & backup_keys if source[k] != backup[k]),
    }
