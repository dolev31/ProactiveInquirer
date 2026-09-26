"""The dialogue driver must carry the SPEC's suite id, not a literal "tau2".

Both tau2 domains are dialogue benchmarks and both need the Orchestrator: the only task text
upstream ships is the customer's script, and the reward is over a transcript. So both route to
`run_tau2_unit`. But the driver stamped `suite_id="tau2"` into the manifest and the status dict
regardless of what was asked for, which would have written every retail run into the eval-only
suite's namespace -- and `assert_trainable` refuses `tau2`, so those rows would have been
silently refused by the exporter as eval-only while sitting in retail run directories.

Banking must be BYTE-IDENTICAL after this change, which is what the equivalence test pins.
"""

from __future__ import annotations

from pi_run.stages.tau2_runner import DIALOGUE_SUITES


def test_both_tau2_domains_route_to_the_dialogue_driver() -> None:
    assert "tau2" in DIALOGUE_SUITES
    assert "tau2_retail" in DIALOGUE_SUITES


def test_a_paragraph_suite_does_not() -> None:
    """musique and friends run flat; routing them here would demand an Orchestrator they have
    no user simulator for."""
    for s in ("musique", "strategyqa", "wiki2", "synth", "drgym"):
        assert s not in DIALOGUE_SUITES


def test_the_driver_no_longer_pins_a_literal_suite_id() -> None:
    """The specific regression: a retail run stamped `suite_id="tau2"` lands in the eval-only
    namespace, and `assert_trainable("tau2", ...)` refuses it -- so the rows would vanish from
    the export with no error anyone could see."""
    import inspect

    from pi_run.stages import tau2_runner

    # Comments legitimately quote the old code to explain the fix, so strip them before
    # grepping -- otherwise the test fails on its own explanation.
    src = "\n".join(
        line
        for line in inspect.getsource(tau2_runner).splitlines()
        if not line.lstrip().startswith("#")
    )
    assert 'suite_id="tau2"' not in src, "the driver still pins a literal suite id"
    assert '"suite_id": "tau2"' not in src, "the status dict still pins a literal suite id"
    assert 'load_suite("tau2"' not in src, "the driver still loads a literal suite"


def test_routing_is_by_membership_not_equality() -> None:
    """`worker.run_unit` used `spec.suite_id == "tau2"`, which sends retail down the flat
    path -- where `view()` raises Tau2NeedsOrchestrator and every unit fails."""
    import inspect

    from pi_run import worker

    src = inspect.getsource(worker.run_unit)
    assert 'spec.suite_id == "tau2"' not in src
    assert "DIALOGUE_SUITES" in src


def test_the_driver_no_longer_pins_a_literal_domain_either() -> None:
    """THE SAME CLASS OF BUG THIS FILE EXISTS TO CATCH, ONE FIELD OVER.

    The suite id was fixed and the DOMAIN was not, so both suites routed to one driver that
    then built `Orchestrator(domain=DOMAIN)` and graded through
    `build_environment(DOMAIN, ...)` -- banking, on every retail run. A driver serving two
    domains may not name one; it asks the suite it was handed.
    """
    import inspect

    from pi_run.stages import tau2_runner

    src = "\n".join(
        line
        for line in inspect.getsource(tau2_runner).splitlines()
        if not line.lstrip().startswith("#")
    )
    # The domain NAME, not the word "banking": the module docstring legitimately explains a
    # role-inversion that was observed on a banking task, and a test that fails on its own
    # explanation is a test people delete.
    assert "banking_knowledge" not in src, "the driver still names one of the two domains"
    assert "import DOMAIN" not in src, "the driver still imports banking's domain constant"
    assert "domain=DOMAIN" not in src
    assert "suite.domain" in src, "the domain must come off the suite"
