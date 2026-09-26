"""Every prompt template a live module RENDERS must exist on disk.

WHY THIS EXISTS, MEASURED. `src/pinq/prompts/retriever_select.txt` -- tau2's entire retrieval
channel for domains with no text search -- went missing between `922060c` (present, renders,
sha a403cd38a6e27fcd) and `cebd54f`/`1d6d0f9` (absent), while
`pinq_adapters/tau2/tool_retriever.py` kept calling `promptlib.render(SELECT_TEMPLATE, ...)` and
`promptlib.sha(SELECT_TEMPLATE)`. No commit deleted it: `git log --all --diff-filter=D` finds
nothing and both parents of the HEAD merge were already missing it. Every tau2 run made in that
window had a retrieval channel that raised `MissingPrompt` on use, so no Inquirer question could
ever return evidence, no need could ever be resolved, and no policy could ever satisfy a stopping
condition. 87 rollout units and one reported "the trained policy does not stop out of domain"
finding were produced against it and had to be withdrawn.

NOTHING CAUGHT IT ACROSS AT LEAST FOUR REVISIONS -- not pytest, not ruff, not `lint-imports`, not
`pi suites audit`. And `pi verify tau2` PASSED 112/112 at both revisions throughout, because gold
replay executes recorded gold actions against the task's own initial state and never invokes the
policy or its retrieval channel: a dead channel and a live one produce identical replays. A check
that structurally cannot fail on the question produces a false clearance, which is worse than no
check, because "we verified the environment" is what stops anyone looking further.

THE REFERENCES ARE DISCOVERED, NOT LISTED. A hand-maintained list of expected templates is how
this class of gap survives: the list and the code drift, and the drift is invisible. This walks
the AST of every module under `src/` and collects the first argument of every
`promptlib.render/sha/load/path/tokens/placeholders/hashes` call, resolving a module-level
`NAME = "literal"` constant when the call passes a Name rather than a string. A dynamic name it
cannot resolve is reported as unresolved rather than silently skipped.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
# promptlib functions whose FIRST positional argument is a template name.
_BY_NAME = ("render", "sha", "load", "path", "tokens", "placeholders")


def _module_string_consts(tree: ast.Module) -> dict[str, str]:
    """Module-level `NAME = "literal"`, which is how `tool_retriever` spells SELECT_TEMPLATE."""
    out: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            if isinstance(node.value.value, str):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        out[t.id] = node.value.value
    return out


def _referenced(path: Path) -> tuple[set[str], set[str]]:
    """(resolved template names, unresolved call descriptions) for one module."""
    tree = ast.parse(path.read_text())
    consts = _module_string_consts(tree)
    found: set[str] = set()
    unresolved: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        fn = node.func.attr
        base = node.func.value
        # only `promptlib.<fn>(...)`, never an arbitrary `.render(` on some other object
        if not (isinstance(base, ast.Name) and base.id == "promptlib"):
            continue
        if fn == "hashes":  # *names_, every positional argument is a template name
            args = list(node.args)
        elif fn in _BY_NAME:
            args = node.args[:1]
        else:
            continue
        for a in args:
            if isinstance(a, ast.Constant) and isinstance(a.value, str):
                found.add(a.value)
            elif isinstance(a, ast.Name) and a.id in consts:
                found.add(consts[a.id])
            elif isinstance(a, ast.Starred):
                continue  # a splat of a collection; covered by the collection's own literals
            else:
                unresolved.add(f"{path.name}:{node.lineno} promptlib.{fn}(<dynamic>)")
    return found, unresolved


def _all_references() -> tuple[dict[str, set[str]], set[str]]:
    refs: dict[str, set[str]] = {}
    unresolved: set[str] = set()
    for p in sorted(SRC.rglob("*.py")):
        if "__pycache__" in p.parts or ".egg-info" in str(p):
            continue
        names, unres = _referenced(p)
        for n in names:
            refs.setdefault(n, set()).add(str(p.relative_to(SRC)))
        unresolved |= unres
    return refs, unresolved


def test_the_scanner_finds_the_reference_that_was_missed():
    """NON-VACUITY. If this scanner cannot see `tool_retriever`'s own reference -- a module-level
    constant, not a literal at the call site -- it would have passed while the file was missing,
    which is the whole failure it exists to prevent."""
    refs, _ = _all_references()
    assert "retriever_select" in refs, sorted(refs)
    assert any("tool_retriever" in m for m in refs["retriever_select"]), refs["retriever_select"]


def test_every_rendered_template_exists_on_disk():
    from pinq import promptlib

    refs, _ = _all_references()
    on_disk = set(promptlib.names())
    missing = {n: sorted(m) for n, m in sorted(refs.items()) if n not in on_disk}
    assert not missing, (
        "prompt templates rendered by live modules but ABSENT from "
        f"{promptlib.PROMPT_DIR}: {missing}. A module that renders a template which does not "
        "exist raises MissingPrompt on every invocation of that path."
    )


def test_every_rendered_template_actually_renders_its_own_sha():
    """Existence is not enough: `sha` reads the file, so a template present but unreadable, or
    one whose overlay shadows it with nothing, still breaks the caller."""
    from pinq import promptlib

    refs, _ = _all_references()
    broken = {}
    for name in sorted(refs):
        try:
            promptlib.sha(name)
        except Exception as exc:  # noqa: BLE001 - the point is that NOTHING may raise here
            broken[name] = f"{type(exc).__name__}: {exc}"
    assert not broken, broken


def _arm_table_templates() -> set[str]:
    """Template names the ARM TABLE declares, resolved at runtime.

    The dynamic call sites the AST scanner cannot resolve -- `components.py` rendering an arm's
    inquirer slot, `ancestors.py` rendering an ancestor baseline's -- take their name from
    `Arm.prompts`, which IS enumerable. So the two routes together cover the dynamic sites'
    actual values instead of exempting them.
    """
    from pinq_expt import arms

    return {v for a in arms.ARMS.values() for v in (a.prompts or {}).values()}


def test_every_arm_declared_template_exists_and_renders():
    """The dynamic half of the gate. An arm that names a template the tree does not hold is the
    same defect as a module rendering a missing one, and it reaches a sweep the same way: at the
    first unit that builds that arm."""
    from pinq import promptlib

    on_disk = set(promptlib.names())
    declared = _arm_table_templates()
    assert declared, "the arm table declared no prompts; the enumeration is broken, not clean"
    missing = sorted(declared - on_disk)
    assert not missing, f"arm-table templates absent from {promptlib.PROMPT_DIR}: {missing}"
    broken = {}
    for name in sorted(declared):
        try:
            promptlib.sha(name)
        except Exception as exc:  # noqa: BLE001
            broken[name] = f"{type(exc).__name__}: {exc}"
    assert not broken, broken


def test_dynamic_call_sites_are_covered_by_the_arm_table():
    """UNRESOLVED IS ALLOWED, UNACCOUNTED IS NOT.

    An earlier draft of this file asserted the unresolved set was EMPTY. That encoded a wrong
    belief rather than a defect in the code: `Arm.prompts` maps a slot to a template name and the
    render happens through that mapping, so a dynamic first argument is the design and demanding
    zero would demand its removal. What must hold is that the dynamic route has a runtime
    enumeration behind it -- which `_arm_table_templates` is -- so no template can be rendered
    from a name that nothing checks. The sites are LISTED here so a future one is a visible
    decision.
    """
    _, unresolved = _all_references()
    modules = {u.split(":")[0] for u in unresolved}
    # Every module with a dynamic promptlib call must be one whose names come from the arm table.
    # Each of these takes its template name from a RUNTIME ENUMERATION that this file also
    # checks: the arm table (`Arm.prompts`) for the policy/ancestor/control slots, and the
    # two-valued user-channel selector in `policies/base.py::_user_channel_fragment`.
    accounted = {
        "components.py",
        "ancestors.py",
        "render.py",
        "controls.py",
        "search.py",
        "base.py",
        "prompted.py",
        # par2_rag's two templates: TRACK 3, 2026-09-22. `Par2RagInquirer` renders
        # `self.prompt_name` (stage 1) exactly like prompted.py's policies, and additionally
        # `self.stage2_prompt_name` (stage 2) -- the second one is why the arm table now
        # declares an "inquirer_stage2" slot (see arms.SLOTS): both values are enumerable via
        # `_arm_table_templates()`, so this module is covered the same way ancestors.py and
        # prompted.py are, not exempted from coverage.
        "structured_baselines.py",
    }
    assert modules <= accounted, (
        "a module calls promptlib with a name this gate cannot resolve statically AND is not "
        f"known to take its names from the arm table: {sorted(modules - accounted)}. Either make "
        "the name a module-level constant, or add the module here with its enumeration."
    )
    assert _arm_table_templates(), "the dynamic route has no runtime enumeration behind it"


def test_both_user_channel_fragments_exist_and_render():
    """The other runtime enumeration. `PromptedInquirer` renders `{{user_channel}}` from one of
    exactly two fragment names depending on whether the arm may address the user, and NEITHER is
    in the arm table -- so without this the placebo could go missing and only the confirmatory
    arms would break, which is the half nobody would notice first."""
    from pinq import promptlib

    on_disk = set(promptlib.names())
    for name in ("fragment_user_channel", "fragment_user_channel_placebo"):
        assert name in on_disk, f"{name} absent from {promptlib.PROMPT_DIR}"
        promptlib.sha(name)


@pytest.mark.parametrize("name", ["retriever_select"])
def test_retriever_select_is_present_and_its_hash_is_pinned(name: str):
    """THE SPECIFIC LOSS, pinned so a recurrence fails loudly rather than silently.

    The sha is the one this file renders at `922060c`, where the published tau2 transfer
    campaign ran. If it moves, the tau2 retrieval channel's prompt changed and every tau2
    run_id moves with it -- which is correct and must be deliberate, not incidental.
    """
    from pinq import promptlib

    assert name in set(promptlib.names())
    assert promptlib.sha(name).startswith("a403cd38a6e27fcd"), promptlib.sha(name)
