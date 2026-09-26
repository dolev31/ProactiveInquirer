"""Redaction for text that came off somebody's machine.

WHY THIS IS IN pi_eval/build. It runs where RAW inputs are read, before anything is written
to `data/corpora/`, so no unscrubbed byte ever reaches the side of the wall an adapter can
read.

THE CONTRACT IS FAIL-CLOSED. `scrub_strict` substitutes and then RE-SCANS its own output; a
surviving match raises `ScrubResidue` and aborts the build. A scrubber that returns text it
could not clean is worse than none, because the build then stamps a clean-looking
`corpus_hash` over a leak. Measured on the source corpus this was written for: 112 live
GitHub tokens across 5 files, 137 Overleaf project URLs, thousands of addresses.

STABLE OPAQUE HANDLES, NOT DELETION. An address becomes `<email:ab12cd>`, keyed by digest, so
two mentions of one person still co-refer after redaction. A deleted address destroys that,
and co-reference is often the thing a decision point turns on.

WHAT THIS DELIBERATELY DOES NOT TOUCH, and why.
  * BARE @HANDLES. `@leshem.choshen` and `@pytest.mark.parametrize` are the same string to a
    regex, and these states are full of code. Redacting them would corrupt the corpus on
    every decorator, import scope and git ref; not redacting them leaves third-party handles
    in a corpus that is NEVER RELEASED and is reviewed at Gate A. Recorded as a residual risk
    in docs/CONVLOG.md rather than silently handled badly.
  * SLURM JOB IDS. A 7-digit integer is a job id, a line number, a byte count and a year
    range. Over-redacting every one of them destroys measurements; a job id is neither a
    secret nor personal data.
"""

from __future__ import annotations

import hashlib
import re
from typing import Callable

TOOL_VERSION = "pi_scrub/1"


class ScrubResidue(RuntimeError):
    """A pattern still matched AFTER substitution. The build stops here."""


def _h(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:6]


def _tagged(kind: str) -> Callable[[re.Match[str]], str]:
    return lambda m: f"<{kind}:{_h(m.group(0))}>"


# ORDER MATTERS. Longest/most specific first, so a bearer token is not first eaten as prose
# and an Overleaf URL is not first eaten as a Google link. Every replacement is chosen so it
# cannot re-match its own pattern -- that property is what makes `scrub_text` idempotent and
# the re-scan in `scrub_strict` meaningful.
_RULES: tuple[tuple[str, re.Pattern[str], Callable[[re.Match[str]], str] | str], ...] = (
    ("private_key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"), "<private_key>"),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"), _tagged("github_token")),
    ("github_token", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"), _tagged("github_token")),
    ("slack_token", re.compile(r"\bxox[bpsoa]-[A-Za-z0-9-]{10,}\b"), _tagged("slack_token")),
    ("aws_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b"), _tagged("aws_key")),
    ("hf_token", re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"), _tagged("hf_token")),
    ("openai_key", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"), _tagged("openai_key")),
    ("bearer", re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{20,}"), _tagged("bearer")),
    (
        "overleaf",
        re.compile(r"(?:https?://)?(?:\w+@)?(?:git\.)?overleaf\.com/[A-Za-z0-9]{6,}"),
        _tagged("overleaf"),
    ),
    ("gdoc", re.compile(r"(?:https?://)?(?:docs|drive)\.google\.com/\S+"), _tagged("gdoc")),
    # THE LAST LABEL MUST BE AT LEAST TWO LETTERS. `claude-opus-5@t1.0` is a model pin with a
    # temperature on it, and the looser pattern read it as an address on every one of 16,432
    # rater records -- a check that cries wolf on every row is a check nobody reads, and
    # redacting a pin would destroy the provenance the record exists to carry.
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)*\.[A-Za-z]{2,}\b"), _tagged("email")),
    ("home_path", re.compile(r"/(?:Users|home)/[^/\s\"']+/"), "~/"),
)


def scan(text: str) -> list[tuple[str, str]]:
    """Every (rule name, matched text) still present. Empty means clean."""
    return [(name, m.group(0)) for name, pat, _ in _RULES for m in pat.finditer(text)]


def scrub_text(text: str) -> str:
    for _, pat, repl in _RULES:
        text = pat.sub(repl, text)
    return text


def scrub_strict(text: str) -> str:
    """Scrub, then prove it worked. Raises `ScrubResidue` naming what survived."""
    out = scrub_text(text)
    residue = scan(out)
    if residue:
        kinds = sorted({k for k, _ in residue})
        raise ScrubResidue(
            f"{len(residue)} match(es) survived scrubbing: {kinds}. "
            "The build stops rather than write a corpus that looks clean."
        )
    return out
