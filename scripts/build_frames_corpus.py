#!/usr/bin/env python3
"""Build the FRAMES corpus: the FRAMES task file + every linked Wikipedia article.

    python scripts/build_frames_corpus.py --root .            # full build, ~2,476 articles
    python scripts/build_frames_corpus.py --limit 20          # a slice, to see the shape
    python scripts/build_frames_corpus.py --offline           # re-emit from the page cache

Writes three trees:

    data/raw/frames/test.tsv                 the pinned dataset commit, sha256-verified
    data/raw/frames/pages/<key>.json         one file per article: revid, digest, paragraphs
    data/corpora/frames/<hash>/tasks.jsonl   PUBLIC: {id, question, paragraphs}
    data/corpora/frames/<hash>/manifest.json provenance: revids, per-page sha256, failures
    data/gold/graphs/frames/v1.jsonl         GOLD: the answer, node-free, canary-stamped

RESUMABLE AND IDEMPOTENT. An article already in the page cache whose stored digest matches
its own payload is never refetched, so a killed build resumes where it stopped and a second
run is a no-op that re-emits the same corpus hash. A page that fails is NAMED in the manifest
with its reason and its task is kept with the pages that did arrive -- dropping it would make
the denominator a function of Wikipedia's availability on the day of the build.

The interesting decisions (rendered HTML rather than plain-text extracts; the revision that
was live at 2024-09-01 rather than today's; the per-page digest) are argued in the module
docstring of pi_eval.build.frames_build, which is where the work happens. This file is the
command line around it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", default=".", help="repo root; writes under <root>/data/")
    ap.add_argument("--limit", type=int, default=None, help="build only the first N tasks")
    ap.add_argument(
        "--workers", type=int, default=4, help="concurrent article fetches; 8 drew HTTP 429"
    )
    ap.add_argument(
        "--as-of",
        default=None,
        help="pin each article to the revision live at this ISO timestamp "
        "(default: the FRAMES release date; see frames_build.REVISION_AS_OF)",
    )
    ap.add_argument(
        "--offline",
        action="store_true",
        help="never touch the network: re-emit from the page cache and name what is missing",
    )
    ap.add_argument("--json", action="store_true", help="print the manifest instead of a summary")
    a = ap.parse_args(argv)

    from pi_eval.build.frames_build import REVISION_AS_OF, build

    root = Path(a.root).resolve()
    t0 = time.time()
    last = [0.0]

    def progress(done: int, total: int, title: str) -> None:
        now = time.time()
        if now - last[0] < 5.0 and done != total:
            return
        last[0] = now
        rate = done / max(1e-9, now - t0)
        eta = (total - done) / rate if rate else 0.0
        print(
            f"  {done}/{total} pages  {rate:.1f}/s  eta {eta / 60:.1f}m  {title[:50]}",
            file=sys.stderr,
            flush=True,
        )

    res = build(
        root=root,
        allow_download=not a.offline,
        as_of=a.as_of or REVISION_AS_OF,
        workers=a.workers,
        limit=a.limit,
        progress=progress,
    )
    wall = time.time() - t0

    man = json.loads((res.corpus.parent / "manifest.json").read_text())
    if a.json:
        print(json.dumps(man, indent=1, sort_keys=True, ensure_ascii=False))
        return 0

    corpus_bytes = res.corpus.stat().st_size
    raw_bytes = sum(
        p.stat().st_size for p in (root / "data" / "raw" / "frames").rglob("*") if p.is_file()
    )
    per_task = sorted(
        len(json.loads(line)["paragraphs"]) for line in res.corpus.read_text().splitlines() if line
    )
    mid = per_task[len(per_task) // 2] if per_task else 0

    print(f"corpus     {res.corpus.parent}")
    print(f"gold       {res.gold}")
    print(f"tasks      {man['n_tasks']}")
    print(
        f"articles   wanted {man['n_titles_wanted']}  fetched {man['n_pages_fetched']}  "
        f"failed {len(man['failures'])}"
    )
    print(
        f"paragraphs {man['n_paragraphs']}  per task min {per_task[0] if per_task else 0} "
        f"median {mid} max {per_task[-1] if per_task else 0}"
    )
    print(f"bytes      corpus {corpus_bytes:,}  raw cache {raw_bytes:,}")
    print(f"wall       {wall / 60:.1f} min")
    if man["failures"]:
        print(f"FAILURES ({len(man['failures'])}), named rather than dropped:")
        for f in man["failures"]:
            print(f"  - {f['title']}: {f['reason']}")
    if man["unresolvable_urls"]:
        print(f"UNRESOLVABLE URLS ({len(man['unresolvable_urls'])}):")
        for u in man["unresolvable_urls"]:
            print(f"  - {u['task_id']}: {u['url']} ({u['reason']})")
    if man.get("pages_after_pin_date"):
        print(
            f"pages taken from their oldest revision because none existed at the pin date "
            f"({len(man['pages_after_pin_date'])}): {', '.join(man['pages_after_pin_date'])}"
        )
    if man["tasks_with_missing_pages"]:
        print(
            f"tasks with an incomplete pool ({len(man['tasks_with_missing_pages'])}): "
            f"{', '.join(man['tasks_with_missing_pages'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
