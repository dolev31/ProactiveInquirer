"""Ingesting somebody else's conversations requires their consent, and the code has to be the
place that knows it.

A consent recorded only in a chat message is a consent nobody can audit later. So the digest of
the consent record is written into docs/DATA.md, the build takes it as a flag, and there is no
path that builds without one.
"""

import pytest

from pi_eval.build.consent import ConsentMissing, require_consent

RECORDED = "a" * 64


def test_a_build_without_a_consent_digest_is_refused():
    with pytest.raises(ConsentMissing) as e:
        require_consent(None, recorded=RECORDED)
    assert "--consent-sha" in str(e.value)


def test_a_wrong_digest_is_refused_and_the_message_does_not_leak_the_right_one():
    with pytest.raises(ConsentMissing) as e:
        require_consent("b" * 64, recorded=RECORDED)
    assert RECORDED not in str(e.value)


def test_the_recorded_digest_passes():
    assert require_consent(RECORDED, recorded=RECORDED) == RECORDED


def test_an_unrecorded_suite_needs_no_consent():
    """Only sources that are somebody's private conversations carry this gate. MuSiQue does
    not, and making every suite pay for it would train people to pass a flag by reflex."""
    assert require_consent(None, recorded=None) is None


def test_the_recorded_digest_is_read_from_the_committed_record():
    """Not typed into a constant. Two places that can disagree about a digest disagree
    eventually, and the one that is wrong is the one nobody re-runs."""
    import hashlib

    from pi_eval.build.consent import CONSENT_DIR, recorded_for

    path = CONSENT_DIR / "convlog.md"
    assert path.exists(), "the convlog consent record must be committed"
    assert recorded_for("convlog") == hashlib.sha256(path.read_bytes()).hexdigest()


def test_editing_the_record_changes_the_digest_and_stops_the_build(tmp_path):
    """The tamper evidence. This is why the gate takes a digest rather than a boolean."""
    from pi_eval.build.consent import recorded_for

    (tmp_path / "convlog.md").write_text("original consent text")
    before = recorded_for("convlog", consent_dir=tmp_path)
    (tmp_path / "convlog.md").write_text("original consent text, quietly amended")
    after = recorded_for("convlog", consent_dir=tmp_path)
    assert before != after
    with pytest.raises(ConsentMissing):
        require_consent(before, recorded=after)


def test_a_gated_suite_with_no_committed_record_refuses_loudly(tmp_path):
    """Absent is not consented. Returning None here would turn a missing record into a
    public-data suite and silently open the gate."""
    from pi_eval.build.consent import ConsentRecordMissing, recorded_for

    with pytest.raises(ConsentRecordMissing):
        recorded_for("convlog", consent_dir=tmp_path)


def test_a_public_suite_is_not_gated():
    from pi_eval.build.consent import recorded_for

    assert recorded_for("musique") is None
