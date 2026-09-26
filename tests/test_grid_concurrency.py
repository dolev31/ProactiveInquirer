"""A grid can pin its own concurrency, because some hosts ask for a ceiling.

DeepResearchGym's maintainer, granting the API key on 2026-08-26: the key is not rate
limited, but fineweb is the busiest endpoint (it requires no licence), "it may timeout, so
I suggest you do not issue many parallel calls (e.g., max 16), and implement some retry
logic."

That ceiling is a property of the SUITE's host, not of the machine running the sweep, so it
belongs in the grid rather than in PI_MAX_CONCURRENCY -- which is global, is read from the
environment, and would silently carry a laptop's setting into a request to someone else's
server. Retries already exist (DrGymClient: max_attempts=3, backoff=0.5, timeout=60.0).
"""

from __future__ import annotations


def test_a_grid_may_declare_concurrency() -> None:
    from pi_run.grids import load

    assert load("conf/grids/tier3_drgym.yaml").concurrency == 8


def test_a_grid_that_declares_none_leaves_it_to_the_runner() -> None:
    from pi_run.grids import load

    assert load("conf/grids/tier1_confirmatory.yaml").concurrency is None


def test_the_drgym_ceiling_is_under_what_its_host_asked_for() -> None:
    """16 is the number in the email; 8 is what we run. If someone raises it, this fails."""
    from pi_run.grids import load

    assert load("conf/grids/tier3_drgym.yaml").concurrency <= 16


def test_the_grid_value_reaches_the_runner() -> None:
    """A pin the dispatcher ignores is decoration.

    2026-09-18: this grepped `cmd_run`'s source for "grid.concurrency". The assignment moved
    into `grid_child_namespace` (every grid override now lives in one function, so the
    conflict check can diff against it), which made the grep false while the behaviour it
    cares about was unchanged -- the test encoded WHERE the read happens, not THAT it
    happens. Asserted on the dispatched namespace instead, which relocation cannot falsify.
    """
    from pi_run.cli import build_parser, grid_child_namespace
    from pi_run.grids import load

    grid = load("conf/grids/tier3_drgym.yaml")
    a = build_parser().parse_args(["run", "--sweep", "conf/grids/tier3_drgym.yaml"])
    sub = grid_child_namespace(a, grid, grid.suites[0])

    assert sub.concurrency == grid.concurrency == 8, "the grid's ceiling never reached the child"


def test_a_grid_that_declares_no_concurrency_leaves_the_operators_value_alone() -> None:
    """The other half of the pin: `concurrency: None` means "leave it to the runner", so an
    explicit --concurrency must SURVIVE rather than be overwritten with None -- and therefore
    must not be refused as a conflict either."""
    from pi_run.cli import build_parser, grid_child_namespace, grid_discarded_flags
    from pi_run.grids import load

    grid = load("conf/grids/tier1_confirmatory.yaml")
    assert grid.concurrency is None
    a = build_parser().parse_args(
        ["run", "--sweep", "conf/grids/tier1_confirmatory.yaml", "--concurrency", "32"]
    )
    sub = grid_child_namespace(a, grid, grid.suites[0])

    assert sub.concurrency == 32
    assert grid_discarded_flags(a, sub) == ()


def test_an_explicit_flag_still_conflicts_with_the_grid() -> None:
    """Same rule as seeds/n/max-turns: a contradicting flag is refused, not silently dropped."""
    from pi_run.cli import build_parser, grid_flag_conflicts

    a = build_parser().parse_args(
        ["run", "--sweep", "conf/grids/tier3_drgym.yaml", "--concurrency", "32"]
    )
    clashes = grid_flag_conflicts(a, seeds=(0,), n_tasks=1, max_turns=16, k=5, concurrency=8)
    assert any("concurrency" in c for c in clashes), clashes


def test_matching_concurrency_is_not_a_conflict() -> None:
    from pi_run.cli import build_parser, grid_flag_conflicts

    a = build_parser().parse_args(
        ["run", "--sweep", "conf/grids/tier3_drgym.yaml", "--concurrency", "8"]
    )
    assert grid_flag_conflicts(a, seeds=(0,), n_tasks=1, max_turns=16, k=5, concurrency=8) == ()


def test_every_grid_field_is_actually_read_by_the_loader() -> None:
    """The unknown-key guard checks one direction only, and this is the other.

    `load()` refuses a yaml key that is not a Grid field. But its `return Grid(...)` is a
    SECOND allowlist, so a field that IS declared and is NOT constructed passes the guard
    and is silently defaulted -- which is exactly what happened to `concurrency`: the yaml
    said 8, the dataclass had the field, the guard was satisfied, and the loaded value was
    None.

    Same shape as the manifest_to_dict bug (a field added to RunManifest that
    manifest_to_dict never serialized) and the SEMANTIC_FIELDS one. Asserted generally so
    the next field cannot repeat it.
    """
    import ast
    import inspect
    import textwrap
    from dataclasses import fields

    from pi_run.grids import Grid, load

    tree = ast.parse(textwrap.dedent(inspect.getsource(load)))
    call = next(
        n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "Grid"
    )
    constructed = {kw.arg for kw in call.keywords}
    declared = {f.name for f in fields(Grid)}
    missing = sorted(declared - constructed)
    assert not missing, f"declared on Grid but never read from the yaml: {missing}"
