"""Find isolated score stores by the property that defines one, not by naming convention.

A store is any directory that directly holds at least one `*.parquet` file. That is the
definition this module uses -- not `scores_parquet*` or `*farm*` in the directory name, both of
which are common but neither of which is reliable: a `farm` directory is, at least once in this
repo (`artifacts/gate_wiki2_stacked_20260918/farm`), a curated symlink farm of *inputs* to `pi
compact`, holding zero real bytes, while the store that compaction wrote is named
`parquet_qwen3-8b-dpo-stacked-notdone-both` and matches neither pattern. Walking for the
`*.parquet` files themselves and taking their parent directories finds every real store
regardless of what its campaign happened to name it, and finds none of the symlink farms that
merely feed one.
"""

from __future__ import annotations

import os
from pathlib import Path


def find_score_stores(root: Path) -> list[Path]:
    """Every directory under `root` (inclusive of `root` itself) that directly contains at
    least one `*.parquet` file.

    A directory qualifies if it holds a matching file *directly*; a parent directory two levels
    up does not also count merely because a store is nested somewhere beneath it. Symlinked
    `*.parquet` files count as files (a store built from `cp -a`'d or hard-linked inputs should
    not be invisible to this scan), but this walk never follows a directory symlink
    (`os.walk(..., followlinks=False)`), so a symlink *farm* of raw run directories (no
    `*.parquet` directly inside it) is correctly never reported as a store.

    Also never descends past a directory that itself contains a `.git` entry (file or dir) other
    than `root`: a farm directory can contain a full second checkout of this repo (a real example
    from lane L0.10b: a scratch tree's own `wt/` was a git worktree with its own `artifacts/`),
    and that nested checkout's contents belong to a different backup decision, not to whatever
    farm happened to be scanned to reach it.

    Returns paths sorted for deterministic output, so two runs over an unchanged tree produce
    the same order.
    """
    root = Path(root)
    if not root.is_dir():
        return []
    stores: set[Path] = set()
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        current = Path(dirpath)
        if any(name.endswith(".parquet") for name in filenames):
            stores.add(current)
        if current != root and (current / ".git").exists():
            dirnames[:] = []
    return sorted(stores)
