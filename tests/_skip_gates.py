"""The registry behind `tests/test_skip_gate_inventory.py` and `PI_STRICT_ENV_SKIPS`.

Why this exists (defect two: "a test that never runs anywhere is not a test either"). A
plain `pytest -q` summary line folds every SKIPPED test into the same grey word as a real
pass. On this repository's own machine, 25 skips is not one fact -- it is a mixture of: a
checkout nobody cloned, a package nobody `pip install -e`d, a server nobody started, and a
venv nobody activated. Each of those is a different machine's problem to fix, or none, and
collapsing them into "25 skipped" is exactly how coverage decays without a diff: a gate that
used to be merely unconfigured quietly becomes a gate nobody remembers exists, on a machine
that could have run it all along.

This module is the enumeration. Each `SkipGate` names the exact reason substring pytest
already prints for that gate (matched against the real, existing skip message -- not a
guess), which file it lives in, and a plain-language, dated verdict on where it can actually
be satisfied. `tests/test_skip_gate_inventory.py` checks the registry stays honest about the
one thing code CAN check for itself (whether the substring still occurs verbatim in the named
file, and, for the two dependency gates, whether the dependency is actually importable here).
The "here / CI / cluster" verdicts below are not something a test can check by running --
they were established by direct measurement or by reading the CI workflow and the cluster
runbooks; see the dated evidence in each `Verdict.note`.

The `PI_STRICT_ENV_SKIPS` hook in `conftest.py` uses these same `gate_id`s to turn a CHOSEN
subset of skips into failures. See that hook's docstring for the mechanism; see the module
docstring for why one gate is deliberately never registered here.

WHAT IS DELIBERATELY NOT HERE. `tests/test_adapters_userbench.py`'s `PI_USERBENCH_LIVE` gate
(`_live_pin`, ~line 1066) skips tests that talk to a PAID user simulator ("SPENDS" is in its
own docstring). That is a spend gate, not an availability gate: the correct behaviour on
every machine -- including one with every checkout cloned and every package installed -- is
to skip by default, forever, until a human deliberately opts in. It is intentionally absent
from `REGISTRY` so `PI_STRICT_ENV_SKIPS` can never match it and so it is never characterised
as "merely unconfigured": it is configured correctly by staying off. (It also never appears
in a bare `pytest -q` skip list on this machine anyway -- `_live_pin` is only reached after
`_live_suite`'s `travelgym` import succeeds, which it does not here, so the checkout gate
fires first.)
"""

from __future__ import annotations

from dataclasses import dataclass

MEASURED_ON = "2026-09-17"


@dataclass(frozen=True)
class Verdict:
    satisfiable: bool | None  # True / False / None = "plausible, not attempted"
    note: str


@dataclass(frozen=True)
class SkipGate:
    gate_id: str
    reason_substring: str
    file: str
    what_gates_it: str
    here: Verdict
    ci: Verdict
    cluster: Verdict


REGISTRY: tuple[SkipGate, ...] = (
    SkipGate(
        gate_id="torch",
        reason_substring="the loss is torch; the gate venv has none",
        file="tests/test_rung1_supervised_span.py",
        what_gates_it="`torch` import (`pytest.importorskip`, called from inside each test body)",
        here=Verdict(
            True,
            f"MEASURED {MEASURED_ON}: torch 2.14.0 is importable in BOTH venvs on this "
            "machine right now -- `.venv/bin/python -c 'import torch'` succeeds, and "
            "`.venv/bin/python -m pytest tests/test_rung1_supervised_span.py "
            "tests/test_rung2_objective.py` -> '32 passed in 4.49s', zero skips, under the "
            "default .venv. (Superseded same-day: an earlier measurement this task took "
            "found torch absent from .venv and present only in .venv-train; a peer in this "
            "shared working tree installed the [train] extra into .venv in the meantime --"
            " see MEMORY.md's shared-working-tree note. Re-measure before trusting either "
            "claim; the point of `test_the_torch_verdict_matches_this_interpreters_actual_"
            "torch_importability` below is that this Verdict's *boolean* stays correct "
            "either way, even when the prose explaining it goes stale.)",
        ),
        ci=Verdict(
            False,
            "By design, not by gap: .github/workflows/tests.yml's matrix installs only "
            "'dev,suites,analysis' or 'dev', never '[train]'. Not integration/slow-marked, so "
            "CI's '-m \"not integration and not slow\"' does not deselect it -- CI actually "
            "collects and runs this test, and correctly skips it.",
        ),
        cluster=Verdict(
            True,
            "docs/GPU_RUNBOOK.md and docs/HPC_RUNBOOK.md both build the trainer venv "
            "with the '[train]' extra, which includes torch.",
        ),
    ),
    SkipGate(
        gate_id="trl",
        reason_substring="trl lives in .venv-train, not the laptop venv",
        file="tests/test_rung2_objective.py",
        what_gates_it="`trl` import (`pytest.importorskip`)",
        here=Verdict(
            True,
            f"MEASURED {MEASURED_ON}: trl 1.13.0 is importable in BOTH venvs on this machine "
            "right now, same re-measurement as `torch` above and for the same reason -- see "
            "that Verdict's note.",
        ),
        ci=Verdict(
            False,
            "Same reasoning as `torch`: not integration-marked, CI runs and "
            "correctly skips it because '[train]' is never installed.",
        ),
        cluster=Verdict(True, "Same reasoning as `torch`."),
    ),
    SkipGate(
        gate_id="pare_data",
        reason_substring="pare is installed but its data is not",
        # Not tests/test_adapters_pare.py: that file only calls `available()` and forwards
        # whatever string it returns (`_ok, _why = available()`); the literal text lives here.
        file="src/pinq_adapters/pare/_probe.py",
        what_gates_it="`PARE_BENCHMARK_SPLITS_DIR` -- the `pare` package and its Meta-ARE "
        "dependency import fine here; only the data/splits checkout is missing "
        "(src/pinq_adapters/pare/_probe.py:available()).",
        here=Verdict(
            None,
            f"PLAUSIBLE, NOT ATTEMPTED as of {MEASURED_ON}: network is reachable from this "
            "machine (measured: `curl` to https://github.com returned 200), and docs/DATA.md "
            "names the exact clone step, but no clone was performed and the pare/Meta-ARE "
            "import path was not exercised beyond what is already installed.",
        ),
        ci=Verdict(
            False,
            "Moot regardless of the checkout: both tests carry @pytest.mark.integration, so "
            "CI's marker deselection excludes them before the runtime gate is ever reached.",
        ),
        cluster=Verdict(
            None,
            "Plausible if a job clones deepakn97/pare and exports the var; unlike "
            "UserBench this is not walked through in docs/HPC_RUNBOOK.md or "
            "docs/GPU_RUNBOOK.md, and was not attempted here.",
        ),
    ),
    SkipGate(
        gate_id="userbench_checkout",
        reason_substring="needs the UserBench checkout installed",
        file="tests/test_adapters_userbench.py",
        what_gates_it="`travelgym` import, i.e. `USERBENCH_DIR` cloned AND `pip install -e "
        "$USERBENCH_DIR` run (docs/DATA.md) -- gated via the shared `_live_suite()` helper.",
        here=Verdict(
            None,
            f"PLAUSIBLE, NOT ATTEMPTED as of {MEASURED_ON}: network is reachable (see "
            "`pare_data` above); satisfying this would mean `pip install -e`-ing a new package "
            "into the shared default .venv that other lanes' sessions use concurrently, which "
            "this task deliberately declined to do without being asked.",
        ),
        ci=Verdict(
            False,
            "Moot: every test reaching this gate carries @pytest.mark.integration (one also "
            "@pytest.mark.slow); structurally excluded from CI regardless of the checkout.",
        ),
        cluster=Verdict(
            None,
            "Plausible if a job clones UserBench and installs it into its own venv; not "
            "attempted here.",
        ),
    ),
    SkipGate(
        gate_id="userbench_dir",
        reason_substring="USERBENCH_DIR is unset",
        file="tests/test_adapters_userbench.py",
        what_gates_it="`USERBENCH_DIR` env var directly, plus `pyarrow` (already installed). "
        "Deliberately does NOT require `travelgym` to be importable -- "
        "`test_against_a_real_checkout`'s own docstring explains why: gating the "
        "parquet-index check on package availability would skip the one test that reads real "
        "bytes on a machine that has the data but not the package.",
        here=Verdict(
            None,
            f"PLAUSIBLE, NOT ATTEMPTED as of {MEASURED_ON}: weaker requirement than "
            "`userbench_checkout` (no package install, just the env var and the cloned data), "
            "but still needs the ~200MB clone docs/DATA.md describes, which was not performed.",
        ),
        ci=Verdict(False, "Moot: @pytest.mark.integration; structurally excluded."),
        cluster=Verdict(None, "Plausible; not attempted here."),
    ),
    SkipGate(
        gate_id="vllm_url",
        reason_substring="needs a running vLLM",
        file="tests/test_chat_template_identity.py",
        what_gates_it="`PI_VLLM_URL` -- an actually-running vLLM server.",
        here=Verdict(
            False,
            "This task's own constraints are no GPU, no cluster; a real vLLM server needs "
            "both. Unlike the checkout-shaped gates above, this is not a network/package gap "
            "that cloning something would close.",
        ),
        ci=Verdict(False, "Moot: @pytest.mark.integration; CI runners also have no GPU."),
        cluster=Verdict(
            True,
            "This is the documented, intended path: docs/HPC_RUNBOOK.md and "
            "docs/GPU_RUNBOOK.md both walk through starting vLLM on the rented box (its own "
            "'.venv-vllm') and pointing PI_VLLM_URL at it, e.g. `PI_VLLM_URL=http://127.0.0.1:"
            "8000 pytest tests/test_chat_template_identity.py -k serving`.",
        ),
    ),
    SkipGate(
        gate_id="harmony_tokenizer",
        reason_substring="needs the real tokenizer",
        file="tests/test_harmony_template.py",
        what_gates_it="`PI_HARMONY_TOKENIZER` (a HF model id, e.g. openai/gpt-oss-20b). The "
        "test's own docstring says to run it in `.venv-train`.",
        here=Verdict(
            None,
            f"PLAUSIBLE, NOT ATTEMPTED as of {MEASURED_ON}: .venv-train is present and network "
            "is reachable, so the download this needs (tokenizer files, not model weights) is "
            "plausible. Not attempted, to avoid an unrequested network fetch and a new "
            "Hugging Face cache side effect on a shared machine.",
        ),
        ci=Verdict(
            False,
            "@pytest.mark.integration (structurally excluded), and CI's 'dev'/'dev,suites,"
            "analysis' extras do not include transformers regardless.",
        ),
        cluster=Verdict(True, "Has network and the '[train]' extra, per the GPU/HPC runbooks."),
    ),
    SkipGate(
        gate_id="smoke_model",
        reason_substring="needs PI_SMOKE_MODEL",
        file="tests/test_rung2_reference_policy.py",
        what_gates_it="`PI_SMOKE_MODEL` (a small HF id) plus the `[train]` extra. Explicitly "
        'CPU-only -- "no CUDA here" is in the skip reason string itself, and the test loads '
        "the model as torch.float32 with no `.cuda()` call.",
        here=Verdict(
            None,
            f"PLAUSIBLE, NOT ATTEMPTED as of {MEASURED_ON}: same shape as `harmony_tokenizer` "
            "-- .venv-train is present, network is reachable, and the test is written to need "
            "no GPU. Not attempted, same reasoning as `harmony_tokenizer`.",
        ),
        ci=Verdict(
            False,
            "@pytest.mark.integration (structurally excluded), and no '[train]' extra "
            "on CI regardless.",
        ),
        cluster=Verdict(True, "Has network and the '[train]' extra."),
    ),
)

_BY_ID = {gate.gate_id: gate for gate in REGISTRY}
assert len(_BY_ID) == len(REGISTRY), "duplicate gate_id in REGISTRY"


def classify(reason: str) -> SkipGate | None:
    """The registered gate whose substring occurs in `reason`, or None if unregistered.

    None is the correct answer for `PI_USERBENCH_LIVE` and for any skip this registry does
    not yet know about -- `PI_STRICT_ENV_SKIPS` (see conftest.py) only ever acts on a match,
    so an unregistered skip is left exactly as it was, never silently escalated.
    """
    for gate in REGISTRY:
        if gate.reason_substring in reason:
            return gate
    return None


def by_id(gate_id: str) -> SkipGate:
    return _BY_ID[gate_id]


def known_ids() -> frozenset[str]:
    return frozenset(_BY_ID)
