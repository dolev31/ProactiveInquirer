"""A tool-backed retriever with no model must REFUSE, and both runners must give it one.

WHY THIS EXISTS, MEASURED. `ToolBackedRetriever.search` returned `()` on its first line when
`self._llm is None`. It FAILED OPEN. So a runner that constructed the retriever before the client
existed produced runs where every Ask on retail, airline and telecom resolved to nothing -- and
the run still succeeded, reconciled, and carried a real graded reward. The only symptom was
`retrieved_uids` empty on every turn, which reads as "the policy asked bad questions" rather than
"the channel was never opened".

MEASURED CONSEQUENCE: 1,228 of 1,228 asks across 87 fork units returned zero uids, and a reported
"the trained policy does not stop out of domain" finding was produced against it and withdrawn.
The trained arm looked like it never stopped because a policy whose questions are never answered
can never resolve a need and so can never satisfy a stopping condition.

THE WARNING EXISTED AND WAS DELETED WITH THE CODE IT GUARDED. At `922060c`, `828a720` and
`401b8fe` -- every revision behind the published tau2 campaign -- `tau2_runner` had a factory:

    try:    return suite.retriever(tid, llm=llm, seed=seed, recorder=recorder)
    except TypeError: return suite.retriever(tid)

carrying this docstring: "THE TOOL-BACKED RETRIEVER IS INERT WITHOUT ONE. `ToolBackedRetriever.
search` returns `()` when `llm is None` ... so a runner that builds it without a client produces
runs where every Ask on airline and telecom resolves to nothing. The run succeeds, the reward is
real, and the only symptom is `retrieved_uids` empty on every turn, which reads as 'the policy
asked bad questions' rather than as 'the channel was never opened'."

Someone had already been bitten by exactly this, wrote both the fix and the warning, and a later
refactor took both. That is the argument for a test over a comment: a comment cannot fail.

THE REFUSAL KEYS ON THE ARM'S DECLARATION, NOT ON THE MODEL BEING ABSENT, because model-free arms
are legitimate and must stay runnable: `fake_chain`, `fake_depth1`, `fake_drafter_only`.
REPLAYED AGAINST REAL HISTORY before landing: 93 recorded runs use a declared model-free arm
(synth 78, musique 6, drgym 6, strategyqa 3) and NONE is on a tau2 suite, so this refuses no work
we depend on.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"


class _StubEnv:
    """The constructor asks the ENVIRONMENT which tools are read-only rather than declaring it
    here, so a fixture has to answer that. Reads only; nothing in these tests mutates."""

    @staticmethod
    def _is_mutating_tool(name: str) -> bool:
        return False


def _retriever_calls(rel: str) -> list[tuple[int, set[str]]]:
    """(lineno, keyword names) for every `<x>.retriever(...)` call in a module."""
    tree = ast.parse((SRC / rel).read_text())
    out = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "retriever"
        ):
            out.append((node.lineno, {k.arg for k in node.keywords if k.arg}))
    return out


@pytest.mark.parametrize("rel", ["pi_run/stages/tau2_runner.py", "pi_run/worker.py"])
def test_the_runner_hands_the_retriever_a_model(rel: str):
    """Both paths. `worker.py` is the flat path and `tau2_runner.py` the fork path, and BOTH built
    the retriever before the client existed, so both produced blind runs.

    Accepts EITHER a direct `suite.retriever(..., llm=...)` or construction through
    `_wire_retriever`, whose own contract is asserted below. What it refuses is a bare
    `suite.retriever(tid)` on a rollout path, which is the shape that caused this.
    """
    src = (SRC / rel).read_text()
    bare = [ln for ln, kw in _retriever_calls(rel) if "llm" not in kw]
    assert "_wire_retriever(" in src, (
        f"{rel} does not construct its retriever through _wire_retriever; if that is deliberate "
        "it must pass llm= directly at every call site"
    )
    # A bare `.retriever(` is allowed ONLY inside the factory itself, which resolves the kwargs.
    limit = 1 if rel.endswith("tau2_runner.py") else 0
    assert len(bare) <= limit, f"{rel}: unwired `.retriever(` call(s) at line(s) {bare}"


def test_the_wiring_factory_passes_the_model_when_the_suite_accepts_it():
    """The factory's contract, decided by SIGNATURE rather than by catching TypeError.

    CHECKED, and this is why the published-era `except TypeError` fallback was NOT restored
    verbatim: banking's `Tau2Suite.retriever(tid)` accepts neither a client nor a declaration and
    needs none -- it retrieves by BM25 text search rather than asking a model to pick a tool --
    while retail, airline and telecom accept both. An `except TypeError` would also swallow a
    TypeError raised from any depth inside a suite's own construction and hand back an UNWIRED
    retriever, which is the original defect wearing a handler.
    """
    from pi_run.stages.tau2_runner import _wire_retriever

    seen = {}

    class _Suite:
        def retriever(self, tid, *, llm=None, llm_free=False, seed=0, recorder=None):
            seen.update(tid=tid, llm=llm, llm_free=llm_free, seed=seed)
            return "wired"

    class _Banking:
        def retriever(self, tid):  # accepts neither; must still construct
            seen.update(tid=tid, llm="NOT-OFFERED")
            return "bm25"

    sentinel = object()
    assert _wire_retriever(_Suite(), "t1", llm=sentinel, llm_free=False, seed=3, recorder=None)
    assert seen["llm"] is sentinel and seen["seed"] == 3
    assert _wire_retriever(_Banking(), "t2", llm=sentinel, llm_free=False, seed=3, recorder=None)
    assert seen["llm"] == "NOT-OFFERED"


def test_a_tool_backed_retriever_with_no_model_refuses():
    """FAIL CLOSED. The defect was that this returned `()`."""
    from pinq_adapters.tau2.tool_retriever import RetrieverNotWired, ToolBackedRetriever

    r = ToolBackedRetriever(
        _StubEnv(),
        corpus_id="c",
        corpus_hash="h",
        tool_schemas=({"name": "get_order_details", "description": "read"},),
        llm=None,
    )
    with pytest.raises(RetrieverNotWired):
        r.search("where is my order", 5)


def test_a_declared_model_free_arm_still_gets_an_empty_channel():
    """The exemption the refusal must preserve, keyed on the DECLARATION and not on `llm is
    None` -- otherwise `fake_drafter_only`, which declares no home suite and so may run on tau2,
    would start raising where it used to be measurable."""
    from pinq_adapters.tau2.tool_retriever import ToolBackedRetriever

    r = ToolBackedRetriever(
        _StubEnv(),
        corpus_id="c",
        corpus_hash="h",
        tool_schemas=({"name": "get_order_details", "description": "read"},),
        llm=None,
        llm_free=True,
    )
    assert r.search("where is my order", 5) == ()


def test_the_refusal_would_not_have_refused_any_recorded_tau2_run():
    """NON-VACUITY IN THE OTHER DIRECTION, and the check that makes the refusal safe to land: a
    guard is only proven by fixtures to FIRE; only history proves it does not refuse work we
    depend on. Declared model-free arms have 93 recorded runs and none is on tau2."""
    import json

    runs = Path(__file__).resolve().parent.parent / "runs"
    if not runs.is_dir():
        pytest.skip("no runs/ tree in this checkout")
    free = {"fake_chain", "fake_depth1", "fake_drafter_only"}
    offenders = []
    for m in runs.glob("*/manifest.json"):
        try:
            d = json.loads(m.read_text())
        except (OSError, ValueError):
            continue
        if d.get("arm_id") in free and str(d.get("suite_id", "")).startswith("tau2"):
            offenders.append(d.get("run_id"))
    assert not offenders, (
        "a declared model-free arm HAS run on a tau2 suite; the exemption above is therefore "
        f"load-bearing and must not be narrowed: {offenders[:5]}"
    )
