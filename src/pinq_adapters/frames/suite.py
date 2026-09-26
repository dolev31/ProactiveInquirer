"""FRAMES suite, VIEW SIDE.

Reads only data/corpora/frames/<hash>/{tasks.jsonl,manifest.json}. The gold answer that
upstream ships in the same TSV row as the question lives in data/gold/ and is read only by
pi_eval; nothing here knows it exists.

Licence: Apache-2.0 (Krishna et al., NAACL 2025; arXiv:2409.12941).

WHY THIS ONE ADAPTER VERIFIES A HASH AND THE OTHER SEVEN DO NOT.

Every other corpus in this repo is bytes we downloaded once from a frozen release and pinned
with a sha256 sidecar. FRAMES ships URLs, not text: the pool is assembled from Wikipedia,
which is live and mutable, by a builder that can be re-run. `corpus_hash` and the
content-addressed directory name pin the corpus as a WHOLE and catch a rebuild -- they cannot
catch a byte edited inside a directory that keeps its name, because the name is not
recomputed at load. For a corpus derived from a third party that edits continuously, "the
paragraphs I am scoring are the paragraphs that were fetched" has to be checkable, so the
per-page digest the builder recorded is re-computed here and a mismatch is a refusal.

The cost of the check is one sha256 over the corpus at construction, which is the same order
as the JSON parse that precedes it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import ClassVar

from ..paragraphs import ParagraphSuite

CORPUS_ID = "frames_v1"

INSTRUCTIONS = (
    "Answer the question using a closed pool of paragraphs drawn from the Wikipedia articles "
    "that together answer it. You may issue retrieval queries against that pool before "
    "answering. The pool is large and mostly irrelevant to any single step; the answer "
    "requires composing several facts in sequence, and some of them are rows of a table."
)


class CorpusTampered(RuntimeError):
    """A corpus paragraph does not match the digest the builder recorded for its page.

    Not a warning. A single edited paragraph changes what the retriever can surface and
    therefore every number computed from the run, while leaving the corpus directory name --
    the only other integrity signal -- untouched.
    """


class FramesSuite(ParagraphSuite):
    suite_id: ClassVar[str] = "frames"
    suite_version: ClassVar[str] = "v1"
    corpus_id: ClassVar[str] = CORPUS_ID
    instructions: ClassVar[str] = INSTRUCTIONS
    # 50, not musique's 30. MEASURED over the 824 gold answers: 97.3% are <= 30 words and
    # 99.5% are <= 50, with a 206-word maximum. `answer_correct` is a token-SUBSEQUENCE test,
    # so a gold answer longer than the cap can never be contained in an answer and scores a
    # structural zero -- 22 tasks at 30, 4 at 50. The cap is frozen across arms either way,
    # which is the property that keeps answer length from being what a judge rewards; the
    # four remaining tasks are named in docs/SUITES.md rather than quietly absorbed.
    word_cap: ClassVar[int] = 50

    def __post_init__(self) -> None:
        super().__post_init__()
        self._verify_pages(self.root)

    def _verify_pages(self, root: Path) -> None:
        """Re-compute each page's digest from the paragraphs actually loaded.

        Pages are grouped by title within a task, which is unambiguous because the builder
        dedupes an article out of its own pool: a title appears at most once per task, and its
        paragraphs are contiguous and in fetch order. A manifest that names no pages (an older
        build) verifies nothing and says so by omission rather than by passing silently -- see
        the `not pages` branch.
        """
        man_path = root / "manifest.json"
        if not man_path.is_file():
            raise CorpusTampered(
                f"no manifest.json beside {root / 'tasks.jsonl'}. The frames corpus carries a "
                "per-page sha256 because its source is live and mutable; a corpus directory "
                "without one cannot be verified and is refused rather than trusted."
            )
        pages = json.loads(man_path.read_text()).get("pages") or {}
        if not pages:
            raise CorpusTampered(f"{man_path} records no page digests, so nothing can be verified")

        for row in self._idx.values():
            by_title: dict[str, list[str]] = {}
            for p in row["paragraphs"]:
                by_title.setdefault(p["title"], []).append(p["text"])
            for title, texts in by_title.items():
                want = str((pages.get(title) or {}).get("sha256") or "")
                if not want:
                    raise CorpusTampered(
                        f"task {row['id']} cites page {title!r}, which the manifest does not "
                        "record a sha256 for"
                    )
                got = hashlib.sha256("\n".join(texts).encode("utf-8")).hexdigest()
                if got != want:
                    raise CorpusTampered(
                        f"page {title!r} in task {row['id']}: sha256 {got[:12]} does not match "
                        f"the manifest's {want[:12]}. The corpus has been edited since it was "
                        "built, or the manifest belongs to a different build."
                    )
