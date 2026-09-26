"""The report emitter: `<id>.q` and `<id>.a`, plain text, with literal URLs in the body.

THE FILESYSTEM *IS* THE INTERFACE. DeepResearchGym's evaluation harness takes one directory
per system under test, containing `<query_id>.q` (the question) and `<query_id>.a` (the
report) as plain text. Reproducing that layout exactly is what lets our judges and the
upstream judges be pointed at the same bytes, which is the only reason a fidelity delta is
meaningful at all.

WHY A ZERO-URL REPORT IS REFUSED RATHER THAN WRITTEN. Two of the three judges key off
literal URLs in the report text:

  * the citation judge first extracts (claim, source URL) pairs from the report; with no
    URLs it extracts nothing, and "no claims" scores identically to "no supported claims";
  * the Support rubric says explicitly that if no section provides source URLs the score is
    zero -- a hard zero out of 10 on one of six criteria.

So a report with no URLs does not score badly, it scores UNINTERPRETABLY: the number
measures the emitter, not the system. Every such report we have ever seen came from a
formatting bug (markdown link syntax stripped, citations rendered as `[1]` with the URL
table dropped), so this raises at emit time where the bug is one frame away, instead of
surfacing as a suspiciously low Support mean three stages downstream.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

from pinq.types import EvidenceUnit

# Deliberately permissive on the scheme and strict on the terminator: markdown wraps URLs in
# ()/[] and prose ends them with sentence punctuation, and a URL that keeps a trailing ")"
# will not match the URL the retriever handed us.
URL_RE = re.compile(r"https?://[^\s<>\[\]()\"'`]+")
_TRAILING = ".,;:!?'\""


def extract_urls(text: str) -> tuple[str, ...]:
    """Every literal URL in the text, in order, deduplicated, trailing punctuation stripped."""
    out: list[str] = []
    seen: set[str] = set()
    for m in URL_RE.finditer(text):
        u = m.group(0).rstrip(_TRAILING)
        if u and u not in seen:
            seen.add(u)
            out.append(u)
    return tuple(out)


class ReportWithoutCitations(ValueError):
    """A report containing no URL. See the module docstring: this is always a bug."""


def sources_block(units: Sequence[EvidenceUnit], *, header: str = "Sources") -> str:
    """A references section built from the URLs the retriever actually returned.

    `EvidenceUnit.title` carries the document URL for this suite (FineWeb records have no
    title; see drgym.client.SearchDoc.as_unit), so this cannot cite a document that was
    never retrieved -- which is the failure mode a free-text bibliography invites.
    """
    urls = [u.title for u in units if u.title.startswith(("http://", "https://"))]
    seen: list[str] = []
    for u in urls:
        if u not in seen:
            seen.append(u)
    if not seen:
        return ""
    lines = "\n".join(f"[{i + 1}] {u}" for i, u in enumerate(seen))
    return f"{header}:\n{lines}\n"


def emit(out_dir: Path, task_id: str, question: str, report: str) -> tuple[Path, Path]:
    """Write `<id>.q` and `<id>.a`. Raises ReportWithoutCitations on a URL-free report."""
    urls = extract_urls(report)
    if not urls:
        raise ReportWithoutCitations(
            f"report for task {task_id} contains no literal URL. The citation judge extracts "
            "(claim, URL) pairs from the report text and the Support rubric hard-zeros a "
            "report with no source URLs, so emitting this file would produce a number that "
            "measures the emitter rather than the system. Include the source URLs inline "
            "(drgym.report.sources_block builds a references section from the retrieved "
            "units)."
        )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    q = out_dir / f"{task_id}.q"
    a = out_dir / f"{task_id}.a"
    q.write_text(question.strip() + "\n", encoding="utf-8")
    a.write_text(report.strip() + "\n", encoding="utf-8")
    return q, a


def read_system_dir(system_dir: Path) -> dict[str, tuple[str, str]]:
    """`<id>.q`/`<id>.a` back into {task_id: (question, report)}.

    Keyed off `*.q` exactly as the upstream harness is, so a system that wrote an answer with
    no question is skipped identically here and there rather than scored against a blank.
    """
    out: dict[str, tuple[str, str]] = {}
    for q in sorted(Path(system_dir).glob("*.q")):
        a = q.with_suffix(".a")
        if not a.exists():
            continue
        out[q.stem] = (q.read_text(encoding="utf-8").strip(), a.read_text(encoding="utf-8").strip())
    return out
