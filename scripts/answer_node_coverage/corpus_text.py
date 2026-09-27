"""Resolve a gold node's `gold_ev_uids` back to corpus TEXT.

`EvidenceUnit.uid` is `h("ev", corpus_id, doc_id, span)` (`pinq.ids.evidence_uid`) -- a hash,
not a reversible encoding (CONTRIBUTING.md: "Hashing any model-produced string into it makes every
memoization key nondeterministic", which is exactly why it is opaque). So the only way back to
text is the same forward computation the builder used: `pi_eval.build.common.unit_uid` over
every paragraph of the task's public corpus record, keeping whichever paragraph's uid matches.

Reads ONLY `data/corpora/<suite>/<corpus_dir>/tasks.jsonl` -- the PUBLIC corpus tree, the one
`pi_eval.gold`'s own module docstring says "an adapter may read". Nothing here touches
`data/gold/`. This module does not need `PI_GOLD_ROOT`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from pi_eval.build.common import unit_uid

# One constant per suite, read verbatim from each builder (`CORPUS_ID = ...`) rather than
# re-derived: a mismatch here silently resolves zero uids, which is exactly the failure mode
# CONTRIBUTING.md's corpus_hash paragraph warns about (looks like an absence, is a plumbing bug).
CORPUS_ID = {
    "musique": "musique_ans_v1p0",
    "strategyqa": "strategyqa_v1",
    "wiki2": "wiki2_v1",
}


def load_task_paragraphs(corpora_root: Path, suite: str, corpus_dir: str) -> dict[str, list[dict]]:
    """`task_id -> [{"idx": int, "title": str, "text": str}, ...]`, from the suite's public
    corpus file. Mirrors `pi_eval.score.load_questions`'s path convention exactly (same
    directory, same file), so a run scored against a given `corpus_dir` and this resolver
    always read the identical bytes.
    """
    p = Path(corpora_root) / suite / corpus_dir / "tasks.jsonl"
    out: dict[str, list[dict]] = {}
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if "id" in rec and "paragraphs" in rec:
            out[str(rec["id"])] = list(rec["paragraphs"])
    return out


def uid_text_map_for_task(suite: str, task_id: str, paragraphs: Sequence[dict]) -> dict[str, str]:
    """`uid -> text` for one task's paragraph pool, by brute-force reproducing `unit_uid` over
    every paragraph and keeping the ones that match something -- there is no other direction
    to compute a hash in.
    """
    corpus_id = CORPUS_ID[suite]
    out: dict[str, str] = {}
    for p in paragraphs:
        text = str(p["text"])
        uid = unit_uid(corpus_id, task_id, int(p["idx"]), text)
        out[uid] = text
    return out


def uid_text_maps_for_suite(
    corpora_root: Path, suite: str, corpus_dir: str, task_ids: Sequence[str]
) -> dict[str, dict[str, str]]:
    """`task_id -> {uid: text}`, restricted to `task_ids` (the population actually scored --
    no reason to materialise text for the ~12k musique tasks this lane never touches)."""
    paragraphs_by_task = load_task_paragraphs(corpora_root, suite, corpus_dir)
    wanted = set(task_ids)
    return {
        tid: uid_text_map_for_task(suite, tid, paragraphs)
        for tid, paragraphs in paragraphs_by_task.items()
        if tid in wanted
    }
