"""`load_suite` must know tau2_retail, and must keep refusing what it does not know.

An unregistered suite RAISES rather than falling back, because a sweep that quietly ran the
wrong corpus is worse than one that did not run. Adding retail must not soften that.
"""

from __future__ import annotations

import pytest

from pi_run.worker import load_suite
from pinq_adapters.tau2._probe import available

tau2_only = pytest.mark.skipif(not available()[0], reason=available()[1])


@tau2_only
def test_load_suite_builds_the_retail_adapter() -> None:
    from pinq_adapters.tau2.retail_suite import Tau2RetailSuite

    s = load_suite("tau2_retail", "")
    assert isinstance(s, Tau2RetailSuite)
    assert s.suite_id == "tau2_retail"
    assert len(s.task_ids()) == 114


@tau2_only
def test_retail_and_banking_are_different_adapters() -> None:
    """Same upstream checkout, different domains. If one fell back to the other the corpus
    would be wrong and every uid would silently miss."""
    from pinq_adapters.tau2.retail_suite import Tau2RetailSuite

    assert isinstance(load_suite("tau2_retail", ""), Tau2RetailSuite)
    assert not isinstance(load_suite("tau2", ""), Tau2RetailSuite)


def test_an_unknown_suite_still_raises() -> None:
    # WAS "tau2_airline", which is now a real suite -- so this test passed by naming something
    # that had not been built yet, and would have gone green forever once it was. The name
    # below is not a domain tau2 has.
    with pytest.raises(Exception):
        load_suite("tau2_not_a_domain", "")


def test_cli_split_selection_agrees_with_the_manifest_stamp() -> None:
    """`pi run --split` and the manifest must bucket a task the SAME way.

    `pi_run.cli._template_of` did `str(fn(task_id)) or None`, and `str(None)` is the
    TRUTHY string "None", so the `or None` never fired. Every task of a suite whose
    `template_id()` returns None collapsed onto the single split key "<suite>|None" and
    therefore into ONE bucket -- while `pi_run.manifest.template_id_of`, which stamps the
    run, did `str(v) if v else None` and bucketed per task.

    Selection and stamping disagreeing is exactly the failure `pinq.splitting`'s own
    docstring was written about. Concretely it made
    `pi run --suite tau2_retail --split test` select 0 of 114 tasks and exit ok=true.
    """
    from pi_run.cli import _template_of
    from pi_run.manifest import template_id_of

    class NoTemplates:
        def template_id(self, tid: str) -> str | None:
            return None

    s = NoTemplates()
    assert template_id_of(s, "0") is None
    assert _template_of(s, "0") is None, "str(None) == 'None' is truthy; `or None` cannot fire"


@tau2_only
def test_retail_split_test_is_not_empty() -> None:
    """The bug's user-visible face: a sweep that selects nothing and reports success."""
    from pi_run.cli import _select_task_ids

    s = load_suite("tau2_retail", "")
    picked = _select_task_ids(s, "tau2_retail", n=3, split="test", offset=0)
    assert len(picked) == 3, f"--split test selected {len(picked)} of 114 retail tasks"
