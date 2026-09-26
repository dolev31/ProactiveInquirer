"""Defect two: skips that hide a runnable test.

`tests/_skip_gates.py` is the enumeration; this file is what keeps it honest and what proves
the `PI_STRICT_ENV_SKIPS` hook in `conftest.py` actually does what its docstring claims,
rather than merely reading like it should. Without the second half of this file, the
strict-mode hook would have zero coverage in the default run -- `PI_STRICT_ENV_SKIPS` is
unset here, so the "convert skip to failure" branch never executes, and a future edit could
silently break it with nothing here to notice. `pytester` (pytest's own plugin-testing
fixture) runs a real, nested pytest process against a throwaway test file, so this is the
mechanism itself under test, not a description of it.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from tests import _skip_gates

# The `pytester` fixture used below is enabled in tests/conftest.py (the root conftest is the
# documented, always-supported place for `pytest_plugins`; a plain test module is not).

_REPO_ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------- the registry stays honest


def test_every_registered_reason_substring_occurs_verbatim_in_its_named_file():
    """The one thing code CAN check without an environment: the string still exists.

    If a gate's message text changes (or the gate is removed entirely), this fails --
    forcing the registry to be updated in the same change rather than silently drifting into
    describing a skip that no longer says what the registry claims it says.
    """
    for gate in _skip_gates.REGISTRY:
        path = _REPO_ROOT / gate.file
        assert path.is_file(), f"{gate.gate_id}: {gate.file} does not exist"
        text = path.read_text()
        assert gate.reason_substring in text, (
            f"{gate.gate_id}: {gate.reason_substring!r} no longer appears in {gate.file}; "
            "either the skip reason changed (update the registry) or the gate is gone "
            "(remove the registry entry, the way test_eval_offline.py's dead skip was removed)"
        )


def test_no_duplicate_gate_ids():
    ids = [gate.gate_id for gate in _skip_gates.REGISTRY]
    assert len(ids) == len(set(ids)), ids


def test_classify_round_trips_every_registered_reason_to_its_own_gate():
    for gate in _skip_gates.REGISTRY:
        found = _skip_gates.classify(gate.reason_substring)
        assert found is not None and found.gate_id == gate.gate_id


def test_classify_of_an_unregistered_reason_is_none_not_a_guess():
    assert _skip_gates.classify("some completely unrelated reason") is None


def test_the_paid_live_simulator_gate_is_never_in_the_registry():
    """PI_USERBENCH_LIVE (see tests/_skip_gates.py's module docstring) spends real money on a
    real user simulator; the correct behaviour on every machine is to skip by default,
    forever. If this ever starts passing for the wrong reason -- someone adds the gate here
    thinking it is just another "unconfigured" case -- PI_STRICT_ENV_SKIPS=all would start
    demanding a paid call run to keep a test suite green, which is exactly backwards."""
    for gate in _skip_gates.REGISTRY:
        assert "PI_USERBENCH_LIVE" not in gate.reason_substring
        assert "paid" not in gate.reason_substring.lower()


@pytest.mark.parametrize("gate_id", sorted(_skip_gates.known_ids()))
def test_every_gate_names_a_here_ci_and_cluster_verdict(gate_id):
    gate = _skip_gates.by_id(gate_id)
    for verdict in (gate.here, gate.ci, gate.cluster):
        assert verdict.note.strip(), f"{gate_id}: empty verdict note"


# ---------------------------------------------------------- the two claims code can re-check


def test_the_torch_verdict_matches_this_interpreters_actual_torch_importability():
    """Not "should be true" -- checked, in the same process that would skip on it."""
    gate = _skip_gates.by_id("torch")
    importable = importlib.util.find_spec("torch") is not None
    assert importable == bool(gate.here.satisfiable), (
        f"registry says here.satisfiable={gate.here.satisfiable} but torch import spec "
        f"found={importable} in {importlib.util.find_spec('torch')!r}; this interpreter is "
        "not the one the registry's note was measured against -- update the note"
    )


def test_the_trl_verdict_matches_this_interpreters_actual_trl_importability():
    gate = _skip_gates.by_id("trl")
    importable = importlib.util.find_spec("trl") is not None
    assert importable == bool(gate.here.satisfiable), (
        f"registry says here.satisfiable={gate.here.satisfiable} but trl import spec "
        f"found={importable}; this interpreter is not the one the registry's note was "
        "measured against -- update the note"
    )


# --------------------------------------------------------- the mechanism itself, end to end


_FIXTURE_GATE_REASON = "the loss is torch; the gate venv has none"


def _write_fixture(pytester: pytest.Pytester) -> None:
    """A throwaway conftest.py + test file inside pytester's isolated tmp dir, exercising the
    real registry's `classify()` against both skip mechanisms actually used by the real
    gates: an in-body `pytest.skip()`/`importorskip()` (reported at the "call" phase) and a
    `@pytest.mark.skipif(...)` decorator (reported at "setup"). `sys.path` still has this
    repo's `tests` package importable, so this is the real hook and the real registry, not a
    reimplementation of them."""
    pytester.makeconftest(
        f"""
        import sys
        sys.path.insert(0, {str(_REPO_ROOT)!r})
        from tests.conftest import pytest_configure, pytest_runtest_makereport  # noqa: F401
        """
    )
    pytester.makepyfile(
        test_fixture=f"""
        import os
        import pytest

        def test_plain_pass():
            assert True

        def test_gated_skip_in_body():
            pytest.skip({_FIXTURE_GATE_REASON!r})

        def test_unrelated_skip_in_body():
            pytest.skip("some other reason entirely")

        @pytest.mark.skipif(not os.environ.get("NOPE"), reason={_FIXTURE_GATE_REASON!r})
        def test_gated_skip_via_decorator():
            assert True
        """
    )


def test_unset_strict_env_leaves_every_skip_alone(pytester: pytest.Pytester, monkeypatch):
    monkeypatch.delenv("PI_STRICT_ENV_SKIPS", raising=False)
    _write_fixture(pytester)
    result = pytester.runpytest_subprocess("-q")
    # test_plain_pass passes; the other three all skip on their own terms (two unconditional
    # in-body skips, plus the decorator whose NOPE-unset condition is unconditionally true in
    # this fixture) -- none of that depends on PI_STRICT_ENV_SKIPS, which is unset here.
    result.assert_outcomes(passed=1, skipped=3)
    assert result.ret == 0


def test_strict_env_naming_the_gate_fails_only_the_matching_skips(
    pytester: pytest.Pytester, monkeypatch
):
    """THE FIX, caught failing without it: before this hook existed, no environment variable
    could turn `test_gated_skip_in_body` red, which is precisely defect two -- a skip that
    hides a runnable test looks identical to a pass in the summary line. With the gate named,
    it must fail; the unrelated skip must not."""
    monkeypatch.setenv("PI_STRICT_ENV_SKIPS", "torch")
    _write_fixture(pytester)
    result = pytester.runpytest_subprocess("-q")
    # setup-phase skipif -> reported as an error, not a failure; call-phase skip -> a failure.
    # Both are non-zero-exit and both name the gate; see conftest.py's docstring.
    result.assert_outcomes(passed=1, skipped=1, failed=1, errors=1)
    assert result.ret != 0
    result.stdout.fnmatch_lines(["*PI_STRICT_ENV_SKIPS includes 'torch'*"])


def test_strict_env_all_behaves_like_naming_every_known_gate(
    pytester: pytest.Pytester, monkeypatch
):
    monkeypatch.setenv("PI_STRICT_ENV_SKIPS", "all")
    _write_fixture(pytester)
    result = pytester.runpytest_subprocess("-q")
    result.assert_outcomes(passed=1, skipped=1, failed=1, errors=1)
    assert result.ret != 0


def test_an_unknown_gate_id_is_a_loud_configuration_error_not_a_silent_noop(
    pytester: pytest.Pytester, monkeypatch
):
    """A typo (`torhc` for `torch`) must not quietly behave like unset. It must refuse to
    collect anything at all -- exit code 4 (pytest's USAGE_ERROR) -- so the mistake is caught
    at the moment it is made, not inferred later from a suite that stayed green for the wrong
    reason."""
    monkeypatch.setenv("PI_STRICT_ENV_SKIPS", "torhc")
    _write_fixture(pytester)
    result = pytester.runpytest_subprocess("-q")
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines(["*unknown gate id*torhc*"]) if result.stderr.lines else (
        result.stdout.fnmatch_lines(["*unknown gate id*torhc*"])
    )
