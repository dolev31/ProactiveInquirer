"""Name this campaign's population: every run id, keyed by cell and domain.

WHY A COLLECTOR AND NOT A QUERY. `arm_id` alone does not identify a cell -- two cells share
`inquirer_prompted` and differ only in the questioner's pin -- and `runs/` is shared with every
other campaign this repository has ever run, so a selection has to be by (grid_name,
code_version, suite, arm, model_pin_hash, protocol) and then written down. What is written down
is the run-id list, because that is the only selector that cannot go stale.

IT READS MANIFESTS, NOT THE SHARD LOGS. A shard's JSONL records what that shard believed it
did; the manifests are what is on disk. When a shard died mid-unit the two disagree, and only
one of them is the population.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

sys.path.insert(0, str(Path(__file__).resolve().parent))
from campaign import CELLS, DOMAINS  # noqa: E402


def cell_of(man: dict[str, Any]) -> str:
    """Which table column this run belongs to, from its own manifest.

    The inquirer's `model_id` is what separates the two `inquirer_prompted` cells, and it is
    read off `pins` rather than off an argument: a run that recorded a different model than
    the campaign intended must land in no cell at all rather than in the intended one.
    """
    arm = str(man.get("arm_id") or "")
    inq = str(((man.get("pins") or {}).get("inquirer") or {}).get("model_id") or "")
    for c in CELLS:
        if c["arm_id"] == arm and c["inquirer"] == inq:
            return c["cell_id"]
    return ""


def collect(
    runs_root: str, *, grid_name: str, code_version: str, protocol: str
) -> dict[str, dict[str, list[str]]]:
    suites = {v: k for k, v in DOMAINS.items()}
    out: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for m in sorted(Path(runs_root).glob("*/manifest.json")):
        try:
            man = json.loads(m.read_text())
        except (OSError, ValueError):
            continue
        if man.get("grid_name") != grid_name or man.get("code_version") != code_version:
            continue
        if str((man.get("upstream_pins") or {}).get("protocol") or "") != protocol:
            continue
        domain = suites.get(str(man.get("suite_id") or ""))
        if domain is None:
            continue
        cid = cell_of(man)
        if not cid:
            continue
        if not (m.parent / "status.json").is_file():
            continue  # a unit that is still running is not part of a population yet
        out[cid][domain].append(m.parent.name)
    return {c: {d: sorted(v) for d, v in sorted(dd.items())} for c, dd in sorted(out.items())}


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-root", required=True)
    p.add_argument("--grid-name", default="tau2_full_protocol_20260918")
    p.add_argument("--code-version", required=True)
    p.add_argument("--protocol", default="tau2_upstream")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    index = collect(
        args.runs_root,
        grid_name=args.grid_name,
        code_version=args.code_version,
        protocol=args.protocol,
    )
    Path(args.out).write_text(json.dumps(index, indent=2, sort_keys=True))
    total = 0
    for cid, per_domain in index.items():
        for domain, ids in per_domain.items():
            total += len(ids)
            digest = hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()[:16]
            print(f"{cid:22s} {domain:8s} {len(ids):5d} runs  run_ids_sha {digest}")
    print(f"total {total} runs -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
