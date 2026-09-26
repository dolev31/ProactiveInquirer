"""PARE adapter: the Actuator contract, the split loader, and the leakage firewall.

Everything except the `integration` block runs with PARE (and Meta-ARE) ABSENT. The fake
apps below are the point: they let the Actuator's per-app state hashing, its read/write
classification and its oracle folding be tested without a simulation engine.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pi_eval.gold import GoldNode
from pinq.protocols import Actuator, Retriever, TaskSuite
from pinq.view import LeakageError, make_view
from pinq_adapters.pare import (
    APPS,
    GRID_E_ARMS,
    N_SCENARIOS,
    AppStateRetriever,
    PareActuator,
    PareSuite,
    PareSuiteUnavailable,
    app_of,
    available,
    load_scenario_ids,
)

FIX = Path(__file__).parent / "fixtures" / "pare"


class FakeApp:
    """A minimal StatefulApp stand-in: named, stateful, with one read and one write tool."""

    def __init__(self, name: str, items: list[str] | None = None) -> None:
        self.name = name
        self.items = list(items or [])

    def get_state(self) -> dict:
        return {"name": self.name, "items": list(self.items)}

    def list_items(self) -> str:
        return json.dumps(self.items)

    list_items.read_only = True  # type: ignore[attr-defined]

    def add_item(self, value: str) -> str:
        self.items.append(value)
        return f"added {value}"

    def boom(self) -> str:
        raise RuntimeError("tool exploded")


@pytest.fixture
def apps():
    return [FakeApp("contacts", ["ada"]), FakeApp("calendar")]


@pytest.fixture
def suite():
    return PareSuite(splits_root=FIX / "splits", split="tiny", strict_counts=False)


# ------------------------------------------------------------------------- the split loader


def test_split_loader_skips_comments_and_blank_lines():
    ids = load_scenario_ids(FIX / "splits", "tiny")
    assert ids == (
        "add_missing_group_contacts",
        "project_task_filtering_calendar",
        "airport_return_ride_from_notes",
    )


def test_strict_count_rejects_a_truncated_benchmark_split():
    """143 is asserted, not assumed: a truncated split would silently shrink Grid E.

    Targets the `full` split specifically, because that is the only split the guard
    applies to — the 50-scenario `ablation` split is legitimately shorter.
    """
    with pytest.raises(ValueError, match=str(N_SCENARIOS)):
        PareSuite(splits_root=FIX / "splits", split="full")


def test_a_non_benchmark_split_is_not_count_checked():
    """The complement of the guard: an ablation-sized split must still load."""
    s = PareSuite(splits_root=FIX / "splits", split="tiny")
    assert len(s.task_ids()) == 3


def test_app_labels_are_derived_and_total():
    assert app_of("project_task_filtering_calendar") == "calendar"
    assert app_of("add_missing_group_contacts") == "contacts"
    assert app_of("something_unrecognisable") == "unknown"
    assert len(APPS) == 9


def test_grid_e_arm_ids_are_the_three_declared_arms():
    assert GRID_E_ARMS == (
        "pare_baseline",
        "pare_plus_inquirer_prompted",
        "pare_plus_verbosity",
    )


# ------------------------------------------------------------------------- protocol shape


def test_suite_satisfies_the_task_suite_protocol(suite):
    assert isinstance(suite, TaskSuite)
    assert suite.task_ids()[0] == "add_missing_group_contacts"
    assert suite.corpus_hash


def test_actuator_satisfies_the_actuator_protocol(apps):
    assert isinstance(PareActuator(apps), Actuator)


def test_retriever_satisfies_the_retriever_protocol(apps):
    r = AppStateRetriever(apps, corpus_id="pare_apps_v1", corpus_hash="abc")
    assert isinstance(r, Retriever)


def test_live_scenario_access_raises_without_pare(suite):
    ok, _ = available()
    if ok:
        pytest.skip("pare is installed; the unavailable path cannot be exercised")
    with pytest.raises(PareSuiteUnavailable):
        suite.scenario("add_missing_group_contacts")


# ------------------------------------------------------------------------- the Actuator


def test_execute_logs_reads_and_writes_distinctly(apps):
    act = PareActuator(apps, scenario_id="s1")
    calls = act.execute(
        [
            {"app": "contacts", "name": "list_items"},
            {"app": "contacts", "name": "add_item", "args": {"value": "grace"}},
        ],
        turn_idx=3,
    )
    assert [c.mutating for c in calls] == [False, True]
    assert [c.tool_name for c in calls] == ["contacts.list_items", "contacts.add_item"]
    assert all(c.ok and c.turn_idx == 3 for c in calls)
    assert [c.seq for c in calls] == [1, 2]
    assert apps[0].items == ["ada", "grace"]


def test_unmarked_tools_are_assumed_mutating(apps):
    """The safe direction: a write mislabelled as a read is the failure nobody notices."""
    act = PareActuator(apps)
    (call,) = act.execute(
        [{"app": "calendar", "name": "add_item", "args": {"value": "x"}}], turn_idx=0
    )
    assert call.mutating is True


def test_a_failing_tool_is_logged_not_raised(apps):
    act = PareActuator(apps)
    (call,) = act.execute([{"app": "contacts", "name": "boom"}], turn_idx=0)
    assert call.ok is False and call.mutating is True
    assert call.result_digest  # a digest, never the raw payload


def test_unknown_app_is_logged_as_a_failed_call(apps):
    act = PareActuator(apps)
    (call,) = act.execute([{"app": "nope", "name": "list_items"}], turn_idx=0)
    assert call.ok is False


def test_final_hashes_are_per_app_and_move_only_for_the_mutated_app(apps):
    act = PareActuator(apps)
    before = dict(act.final_hashes())
    assert set(before) == {"contacts", "calendar"}

    act.execute([{"app": "contacts", "name": "add_item", "args": {"value": "grace"}}], turn_idx=0)
    after = dict(act.final_hashes())
    assert after["contacts"] != before["contacts"], "a write must move that app's hash"
    assert after["calendar"] == before["calendar"], "an untouched app's hash must not move"


def test_a_read_does_not_move_any_hash(apps):
    act = PareActuator(apps)
    before = dict(act.final_hashes())
    act.execute([{"app": "contacts", "name": "list_items"}], turn_idx=0)
    assert dict(act.final_hashes()) == before


def test_final_hashes_are_deterministic_across_instances(apps):
    a = PareActuator([FakeApp("contacts", ["ada"])]).final_hashes()
    b = PareActuator([FakeApp("contacts", ["ada"])]).final_hashes()
    assert a == b, "the hash must be a pure function of state, or scoring is not replayable"


def test_native_is_empty_until_the_oracle_speaks(apps):
    assert PareActuator(apps).native() == {}


def test_attach_validation_folds_the_oracle_verdict(apps):
    class Result:
        success = True
        proposal_count = 3
        acceptance_count = 2
        read_only_actions = 5
        write_actions = 1
        number_of_turns = 7

    act = PareActuator(apps)
    act.attach_validation(Result())
    n = act.native()
    assert n["oracle_success"] == 1.0
    assert n["proposal_count"] == 3.0 and n["acceptance_count"] == 2.0
    assert n["write_actions"] == 1.0 and n["number_of_turns"] == 7.0


def test_attach_validation_tolerates_a_failed_or_absent_result(apps):
    act = PareActuator(apps)
    act.attach_validation(None)
    assert act.native() == {}

    class Failed:
        success = False

    act.attach_validation(Failed())
    assert act.native()["oracle_success"] == 0.0


# ------------------------------------------------------------------------- the retriever


def test_app_state_retrieval_is_logged_as_non_mutating(apps):
    r = AppStateRetriever(apps, corpus_id="pare_apps_v1", corpus_hash="abc")
    units = r.search("ada contacts", k=2)
    assert units and units[0].doc_id == "contacts"
    assert all(not c.mutating for c in r.env_calls)


def test_retrieval_always_returns_something(apps):
    """A query matching nothing still yields the top app: an empty observation would make
    the Inquirer's first turn structurally uninformative on every scenario."""
    r = AppStateRetriever(apps, corpus_id="pare_apps_v1", corpus_hash="abc")
    assert len(r.search("zzzz", k=3)) >= 1


# ------------------------------------------------------------------------- leakage firewall


def _view_kwargs(**over):
    kw = dict(
        task_id="add_missing_group_contacts",
        suite_id="pare",
        question="Observe the user's activity.",
        instructions="You are a proactive assistant.",
        corpus_id="pare_apps_v1",
        corpus_hash="deadbeef",
        word_cap=180,
    )
    kw.update(over)
    return kw


@pytest.mark.firewall
@pytest.mark.parametrize(
    "leaky", ["oracle_events", "expected_actions", "validation_result", "events_flow"]
)
def test_pare_gold_bearing_fields_cannot_reach_a_view(leaky):
    with pytest.raises(LeakageError):
        make_view(**_view_kwargs(**{leaky: "leaked"}))


@pytest.mark.firewall
@pytest.mark.parametrize("gold_field", sorted(GoldNode.__dataclass_fields__))
def test_every_gold_node_field_raises(gold_field):
    with pytest.raises(LeakageError):
        make_view(**_view_kwargs(**{gold_field: "leaked"}))


@pytest.mark.firewall
def test_the_suite_view_carries_no_oracle(suite):
    v = suite.view("project_task_filtering_calendar")
    assert v.task_id == "project_task_filtering_calendar"
    assert v.suite_id == "pare"
    assert "calendar" in v.question


# ------------------------------------------------------------------------- offline guarantee


def test_available_returns_a_reason_either_way():
    ok, why = available()
    assert isinstance(ok, bool) and isinstance(why, str) and why


def test_no_pare_import_at_module_scope():
    """OFFLINE-FIRST. PARE drags in Meta-ARE, so a module-scope import would make the
    default test run impossible without a simulation engine installed."""
    import ast

    for path in (Path(__file__).parents[1] / "src" / "pinq_adapters" / "pare").rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in tree.body:  # module scope ONLY; function-body imports are the pattern
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                assert not name.startswith(("pare", "are.")), f"{path.name}: {name} at module scope"


def test_adapter_never_imports_pi_eval():
    import ast

    for path in (Path(__file__).parents[1] / "src" / "pinq_adapters" / "pare").rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                assert not name.startswith("pi_eval"), f"{path.name} imports {name}"


# ------------------------------------------------------------------------------ integration

_ok, _why = available()
integration = pytest.mark.skipif(not _ok, reason=_why)


@pytest.mark.integration
@integration
def test_real_split_has_143_scenarios():
    assert len(load_scenario_ids()) == N_SCENARIOS


@pytest.mark.integration
@integration
def test_real_suite_builds_and_views_every_scenario():
    s = PareSuite()
    ids = s.task_ids()
    assert len(ids) == N_SCENARIOS
    for tid in ids:
        assert s.view(tid).task_id == tid


def test_pares_oracle_verdict_has_no_caller_and_that_is_recorded():
    """A PARE run writes no `oracle_success` row, because `attach_validation` is never called.
    Downstream that is indistinguishable from a scenario the policy failed.

    NOT A BUG REPORT LEFT LYING AROUND -- a pinned decision. Grid E is cut: exploratory, no
    confirmatory test, oracle reproduction never demonstrated. This test asserts the CURRENT
    state so it cannot be quietly forgotten, and it FAILS the moment someone wires a caller,
    which is exactly when the decision should be revisited.
    """
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    callers = []
    for py in sorted((root / "src").rglob("*.py")):
        if py.name == "actuator.py" and py.parent.name == "pare":
            continue  # the definition itself
        if "attach_validation(" in py.read_text():
            callers.append(str(py.relative_to(root)))

    assert not callers, (
        f"PARE's oracle is now wired from {callers}. Grid E was cut partly because this "
        "path did not exist -- revisit conf/grids/pare_transfer.yaml and the plan's cut list, "
        "then delete this test."
    )

    from pinq_adapters.pare.actuator import PareActuator

    assert "nothing calls it" in (PareActuator.native.__doc__ or "").lower()
