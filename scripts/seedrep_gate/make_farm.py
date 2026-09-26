"""Build the isolated symlink farm for the seed-replicate gate, and PROVE the links resolve.

Reuses `scripts/decomposition_test/build_farm.build_farm`, which already carries the fix for
the latent relative-root bug: a relative `source_runs_root` produces symlinks whose target text
is relative, resolved against the SYMLINK's own directory, so every link dangles and
`pi compact` reports `run_dirs: 0` -- a silent zero, not an error. This driver additionally
refuses to continue unless every link it just made resolves to a directory holding a
`manifest.json`, so the zero cannot reach `pi compact` at all.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "decomposition_test"),
)
from pathlib import Path  # noqa: E402

from build_farm import build_farm, read_run_id_list  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", required=True, help="ABSOLUTE source runs root")
    ap.add_argument("--farm", required=True, help="farm directory; runs land in <farm>/runs")
    ap.add_argument("--ids", action="append", required=True, help="run-id list file")
    args = ap.parse_args()

    if not os.path.isabs(args.runs_root):
        print(f"REFUSING: --runs-root must be absolute, got {args.runs_root!r}", file=sys.stderr)
        return 2

    farm_runs = Path(args.farm) / "runs"
    all_ids: list[str] = []
    for path in args.ids:
        ids = read_run_id_list(Path(path))
        print(f"{path}: {len(ids)} ids")
        all_ids.extend(ids)
    dupes = len(all_ids) - len(set(all_ids))
    print(f"total ids: {len(all_ids)} ({len(set(all_ids))} distinct, {dupes} duplicate)")

    res = build_farm(farm_runs, all_ids, source_runs_root=Path(args.runs_root))
    print(
        f"linked={len(res['linked'])} already_present={len(res['already_present'])} "
        f"missing={len(res['missing'])}"
    )
    if res["missing"]:
        print(f"MISSING sample: {res['missing'][:10]}", file=sys.stderr)
        return 3

    # PROVE the links resolve. A dangling symlink is the failure mode this whole file exists
    # for, and it presents downstream as `run_dirs: 0` with exit status 0.
    entries = sorted(os.listdir(farm_runs))
    resolved = 0
    broken: list[str] = []
    for name in entries:
        link = farm_runs / name
        target = os.path.realpath(link)
        if os.path.isdir(target) and os.path.isfile(os.path.join(target, "manifest.json")):
            resolved += 1
        else:
            broken.append(f"{name} -> {os.readlink(link)} (realpath {target})")
    print(f"farm entries: {len(entries)}")
    print(f"resolve to a dir containing manifest.json: {resolved}")
    print(f"BROKEN: {len(broken)}")
    for b in broken[:10]:
        print(f"  {b}", file=sys.stderr)
    if broken or resolved != len(set(all_ids)):
        return 4
    print(f"sample link text: {os.readlink(farm_runs / entries[0])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
