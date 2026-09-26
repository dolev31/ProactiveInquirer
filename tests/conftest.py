"""Test hermeticity.

At least two dependencies call `load_dotenv()` as an import side effect -- `litellm` and
`tau2` -- so merely having a populated `.env` on the machine injects `PI_MODEL_*`,
`LITELLM_*` and provider keys into `os.environ` the moment any test touches them. That made
the suite order-dependent (a tau2 test running earlier changed a later runtime test) and,
worse, made it behave differently for a developer with credentials than for CI without them.
A suite that passes on your laptop and fails in CI teaches people to distrust the suite.

A one-shot scrub at collection cannot hold, because the offending import happens lazily
inside whichever test needs it. So the scrub is AUTOUSE and runs before every test. A test
that wants a pin sets it explicitly with monkeypatch, which is the only way it should ever
arrive.
"""

from __future__ import annotations

import os

import pytest
from tests import _skip_gates

# Enables the `pytester` fixture used by tests/test_skip_gate_inventory.py to run a real,
# nested pytest process against a throwaway fixture -- the documented, always-supported place
# for this declaration is the root conftest.py, not a plain test module.
pytest_plugins = ["pytester"]

_PREFIXES = ("PI_MODEL_", "LITELLM_", "DRGYM_")
_EXACT = frozenset(
    {
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "ANTHROPIC_API_KEY",
        "GROQ_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
        "TOGETHER_API_KEY",
        "FIREWORKS_API_KEY",
        "OPENROUTER_API_KEY",
        "OPENROUTER_BASE_URL",
        "AZURE_OPENAI_API_KEY",
        "HF_TOKEN",
        "PI_SPEND_CAP_USD",
        "PI_MAX_CONCURRENCY",
        "PI_CACHE_ROOT",
        "PI_RUNS_ROOT",
        # Load-bearing rather than incidental: a rollout worker MUST see this unset, and a stray
        # value would make the firewall test pass for the wrong reason.
        "PI_GOLD_ROOT",
    }
)


def _leaked() -> list[str]:
    return [k for k in os.environ if k.startswith(_PREFIXES) or k in _EXACT]


@pytest.fixture(autouse=True)
def _fast_retry_ladder(monkeypatch):
    """No test may pay the provider retry ladder.

    MeteredClient retries 8 times from 4.0s -- ~148s, correct against a real per-minute rate
    limit and exactly wrong when there is no provider to reach. A test of the OFFLINE path
    (an LLM arm with no key, which must still write its manifest) otherwise costs 148 seconds
    per unit, and a slow suite is a suite people stop running.
    """
    monkeypatch.setenv("PI_LLM_MAX_ATTEMPTS", "1")
    monkeypatch.setenv("PI_LLM_RETRY_WAIT_S", "0.01")


@pytest.fixture(autouse=True)
def _hermetic_env():
    """Scrub before the test, and restore afterwards so an interactive session is unharmed."""
    saved = {k: os.environ[k] for k in _leaked()}
    for k in saved:
        os.environ.pop(k, None)
    try:
        yield
    finally:
        os.environ.update(saved)


# ---------------------------------------------------------------------- PI_STRICT_ENV_SKIPS
#
# Defect two's mechanism. See tests/_skip_gates.py for the registry and for why one gate
# (PI_USERBENCH_LIVE, a spend gate) can never be named here.
#
# A skip that is "merely unconfigured" on a machine that is SUPPOSED to have the gate
# satisfied -- a training-venv CI job missing torch, a cluster job that forgot to export
# USERBENCH_DIR -- is a silent pass in the one-line summary. This variable turns a CHOSEN set
# of registered gates from skip into failure, so that job finds out immediately instead of
# reporting green while quietly covering less than it thinks.
#
# OFF BY DEFAULT: unset, this changes nothing, for anyone, ever -- this laptop, CI, a cluster
# job that does not set it all behave exactly as before. A job that IS supposed to satisfy
# some subset sets this once:
#
#   PI_STRICT_ENV_SKIPS=torch,trl .venv-train/bin/python -m pytest tests/
#   PI_STRICT_ENV_SKIPS=all PI_VLLM_URL=http://127.0.0.1:8000 PI_HARMONY_TOKENIZER=openai/gpt-oss-20b \
#       PI_SMOKE_MODEL=Qwen/Qwen3-0.6B pytest -m integration
#
# An unmatched skip (anything not in tests/_skip_gates.REGISTRY, including PI_USERBENCH_LIVE)
# is left exactly as it was -- this mechanism only ever escalates a gate it was explicitly
# told to expect, never guesses. An unknown gate id is a hard, immediate pytest.UsageError at
# configure time, not a silent no-op: the entire point is that a typo here must not quietly
# do nothing.
#
# MEASURED (this task, 2026-09-17, in an isolated /tmp scratch conftest before this one was
# written): unset -> unchanged (exit 0, both a matching and a non-matching skip stay skips).
# PI_STRICT_ENV_SKIPS=<matching id> or "all" -> the matching skip becomes a failure (exit 1);
# an unrelated registered-but-not-chosen or unregistered skip is untouched. An unknown id ->
# pytest.UsageError, exit 4, before collection. Both skip mechanisms actually used by the real
# gates were exercised: a `pytest.importorskip`/manual `pytest.skip()` called from inside a
# test body (reported at the "call" phase, shows as FAILED) and a `@pytest.mark.skipif(...)`
# decorator (evaluated at the "setup" phase, shows as ERROR -- pytest's own terminology for
# any non-pass outcome during setup, not a defect of this hook); both are loud and both make
# the run exit non-zero, which is the only property this mechanism needs.

_STRICT_ENV_VAR = "PI_STRICT_ENV_SKIPS"
_strict_gate_ids: frozenset[str] = frozenset()


def pytest_configure(config: pytest.Config) -> None:
    global _strict_gate_ids
    raw = os.environ.get(_STRICT_ENV_VAR, "").strip()
    if not raw:
        return
    known = _skip_gates.known_ids()
    if raw.lower() == "all":
        _strict_gate_ids = frozenset(known)
        return
    chosen = frozenset(part.strip() for part in raw.split(",") if part.strip())
    unknown = chosen - known
    if unknown:
        raise pytest.UsageError(
            f"{_STRICT_ENV_VAR} names unknown gate id(s) {sorted(unknown)}; known ids are "
            f"{sorted(known)} (see tests/_skip_gates.py)."
        )
    _strict_gate_ids = chosen


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo):
    outcome = yield
    if not _strict_gate_ids:
        return
    report = outcome.get_result()
    if not report.skipped or call.excinfo is None:
        return
    exc = call.excinfo.value
    reason = getattr(exc, "msg", None) or str(exc)
    gate = _skip_gates.classify(reason or "")
    if gate is None or gate.gate_id not in _strict_gate_ids:
        return
    report.outcome = "failed"
    report.longrepr = (
        f"{_STRICT_ENV_VAR} includes {gate.gate_id!r}: this test skipped ({reason!r}) on a "
        f"gate this run declared should be satisfied. Either the environment is not actually "
        f"configured the way {_STRICT_ENV_VAR} claims, or this gate no longer belongs in the "
        f"chosen set (see tests/_skip_gates.py)."
    )


# ---------------------------------------------------------------------------------------------
# Tests that read files this repository does not ship: a canary list built by `pi data build`,
# recorded runs and experiment outputs from the original machine, a cluster job script, and the
# ML stack the skip-gate registry records for that machine. Each runs wherever its file or
# module exists, and otherwise skips with the missing piece named instead of failing.
_NEEDS_FILE = {
    "tests/test_firewall.py::test_the_tripwire_arms_from_a_foreign_working_directory": (
        "data/canaries/canaries.txt"
    ),
    "tests/test_tau2_user_sim_silent_stop.py::test_the_two_crashed_runs_recorded_this_exact_error": (
        "runs/f7843949e387af86db2bf60b5e2624ec/status.json"
    ),
    "tests/test_length_audit_kill_rule_fires.py::test_the_kill_rule_can_actually_fire": (
        "artifacts/a14_complete_20260919/rung2-8b-headline-control.tierA.json"
    ),
    "tests/test_litellm_tau2_thinking.py::test_the_injob_generated_config_still_disables_thinking": (
        "scripts/hpc/tau2_cluster_campaign.sh"
    ),
}
_NEEDS_MODULE = {
    "tests/test_skip_gate_inventory.py::test_the_torch_verdict_matches_this_interpreters_actual_torch_importability": "torch",
    "tests/test_skip_gate_inventory.py::test_the_trl_verdict_matches_this_interpreters_actual_trl_importability": "trl",
}


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for item in items:
        need = _NEEDS_FILE.get(item.nodeid)
        if need and not (root / need).exists():
            item.add_marker(
                pytest.mark.skip(reason=f"needs {need}, which this repository does not ship")
            )
        mod = _NEEDS_MODULE.get(item.nodeid)
        if mod and importlib.util.find_spec(mod) is None:
            item.add_marker(
                pytest.mark.skip(reason=f"needs {mod} installed (pip install -e '.[train]')")
            )
