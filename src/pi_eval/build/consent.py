"""The gate on ingesting somebody else's conversations.

WHY IT IS CODE AND NOT A NOTE. A consent given in a chat message is a consent nobody can check
six months later, and `convlog` is built from a collaborator's own working sessions. So the
digest of the consent record lives in docs/DATA.md, the build takes it as an argument, and
there is no `--no-verify` path past it. The flag is not a formality: typing it is the moment
someone asserts the consent exists.

WHY THE ERROR NEVER PRINTS THE EXPECTED DIGEST. An error message that names the value it wanted
turns the gate into a prompt. It says where the recorded digest lives, not what it is.

ONLY SOURCES THAT NEED IT CARRY IT. `recorded=None` means the suite is public data and the gate
is a no-op -- a flag every suite had to pass would be passed by reflex within a week.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

# Suites whose source is somebody's private conversations. A suite absent from this set is
# public data and carries no gate at all.
GATED_SUITES = frozenset({"convlog"})

CONSENT_DIR = Path(__file__).resolve().parents[3] / "docs" / "consent"


class ConsentMissing(RuntimeError):
    """No consent digest, or the wrong one. The build stops."""


class ConsentRecordMissing(RuntimeError):
    """The suite is gated and there is no committed record to check against."""


def recorded_for(suite_id: str, *, consent_dir: Path | None = None) -> str | None:
    """The digest of the suite's committed consent record, or None if it needs no consent.

    DERIVED, NEVER TYPED IN. An earlier draft pasted the digest into a constant here, which
    means two places can disagree and the one that is wrong is the one nobody re-runs. Reading
    the file makes the record itself the single source: edit the record and the digest moves,
    the operator's `--consent-sha` stops matching, and the build stops. That is the tamper
    evidence, and it is the whole reason the value is a digest rather than a boolean.
    """
    if suite_id not in GATED_SUITES:
        return None
    path = (consent_dir or CONSENT_DIR) / f"{suite_id}.md"
    if not path.exists():
        raise ConsentRecordMissing(
            f"{suite_id} is a consent-gated suite and {path} does not exist; "
            "the build cannot proceed without a committed consent record"
        )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_consent(supplied: str | None, *, recorded: str | None) -> str | None:
    if recorded is None:
        return None
    if not supplied:
        raise ConsentMissing(
            "this suite is built from private conversations and needs --consent-sha; "
            "the digest of the consent record is in docs/DATA.md"
        )
    if supplied.strip().lower() != recorded.strip().lower():
        raise ConsentMissing(
            "--consent-sha does not match the digest recorded in docs/DATA.md for this suite"
        )
    return recorded
