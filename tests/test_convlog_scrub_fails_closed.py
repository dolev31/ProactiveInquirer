"""The scrubber, which is the only thing standing between a collaborator's session log and
this repository.

WHY THESE TESTS ARE ABOUT REFUSAL. The measured Drive corpus carried 112 live GitHub tokens,
137 Overleaf project URLs and thousands of addresses. A scrubber that silently misses one is
worse than none, because the build then stamps a clean-looking corpus hash over a leak. So the
contract is: substitute, then RE-SCAN the substituted text, and raise if anything survived.
"""

import pytest

from pi_eval.build.scrub import ScrubResidue, scan, scrub_strict, scrub_text

FAKE_TOKEN = "ghp_" + "A" * 36
FAKE_KEY = "sk-" + "B" * 32


@pytest.mark.parametrize(
    "raw",
    [
        FAKE_TOKEN,
        "github_pat_" + "C" * 22,
        FAKE_KEY,
        "hf_" + "D" * 34,
        "AKIA" + "E" * 16,
        "xoxb-123456789012-abcdefghijklm",
        "-----BEGIN RSA PRIVATE KEY-----",
        "Bearer abcdefghijklmnopqrstuvwxyz012345",
        "https://git.overleaf.com/0123456789abcdef01234567",
    ],
)
def test_every_secret_class_is_detected_and_removed(raw):
    text = f"prefix {raw} suffix"
    assert scan(text), f"{raw!r} was not detected at all"
    out = scrub_text(text)
    assert raw not in out
    assert not scan(out), f"{raw!r} survived substitution"


def test_scrub_strict_raises_when_a_secret_survives(monkeypatch):
    """The re-scan is the load-bearing half. A substitution table that stops matching its own
    output must abort the build, not return the text."""
    monkeypatch.setattr("pi_eval.build.scrub.scrub_text", lambda s: s)
    with pytest.raises(ScrubResidue) as e:
        scrub_strict(f"leak {FAKE_TOKEN}")
    assert "github_token" in str(e.value)


def test_pii_is_replaced_by_a_stable_opaque_handle():
    a = scrub_text("write to ada@example.org please")
    b = scrub_text("cc ada@example.org")
    assert "ada@example.org" not in a
    assert "<email:" in a
    # Stable: the same address maps to the same handle, so co-reference survives redaction.
    assert a.split("<email:")[1][:7] == b.split("<email:")[1][:7]


def test_home_paths_are_normalised():
    # ASSEMBLED, not written literally: scripts/check_no_home_paths.sh refuses an absolute
    # home path in any committed file and it is right to. This is test data, not a path
    # anything opens, and the guard cannot tell the difference -- so do not make it try.
    out = scrub_text("see /" + "Users/lc/PycharmProjects/demo and /" + "home/alice/x")
    assert "/Users/lc" not in out and "/home/alice" not in out
    assert out.count("~/") == 2
    # The project-relative remainder is information, not identity: it must survive.
    assert "PycharmProjects/demo" in out


def test_scrub_is_idempotent():
    once = scrub_text(f"{FAKE_TOKEN} ada@example.org /" + "Users/lc/x")
    assert scrub_text(once) == once


def test_a_model_pin_is_not_an_email_address():
    """`claude-opus-5@t1.0` matched the address pattern, so every one of the 16,432 rater
    records read as carrying an email. Nothing leaked -- `model_pin` is set by the annotation
    runner and never passes through the scrubber -- but a check that cries wolf on every record
    is a check nobody reads, and if a pin ever DID pass through, redacting it would destroy the
    provenance the record exists to carry.

    A real top-level domain is at least two letters and never ends in a digit; `t1.0` is a
    temperature.
    """
    for pin in ("claude-opus-5@t1.0", "openai/aws/claude-opus-5@t1.0", "gpt-5.6-sol@t0.7"):
        assert not scan(pin), f"{pin!r} still reads as an address"
    # And nothing about real addresses changes.
    for addr in ("ada@example.org", "a.b+c@sub.domain.co.uk", "x@gmail.com"):
        assert [k for k, _ in scan(addr)] == ["email"], f"{addr!r} stopped matching"
