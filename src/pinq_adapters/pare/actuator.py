"""PARE's FSM apps behind the standard Actuator protocol.

THE POINT OF THIS ADAPTER
    tau2 is scored by comparing a database hash. PARE is scored by its own oracle. If the
    two were wired differently, "does the Inquirer transfer to an action environment?"
    would be answered by two incomparable pipelines. Putting PARE behind the SAME Actuator
    protocol means its app state is scored from the stored `env_calls` log exactly like
    tau2's DB hash — one scoring path, one parquet schema, one join key.

WHY final_hashes() HASHES PER APP RATHER THAN ONCE GLOBALLY
    A scenario touches several of the nine apps and only mutates some of them. One global
    hash would report "different" whenever any app moved, losing which one. Per-app hashes
    let a diff name the app that diverged, which is the difference between a usable failure
    analysis and a binary that says only "wrong".

WHY THE ASYMMETRY IS PRESERVED
    PARE's defining property is that the user agent sees only the current screen's tools
    while the proactive agent gets flat API access. This Actuator is the PROACTIVE side, so
    it deliberately does not filter by current screen. Filtering here would quietly hand
    the proactive agent the user's handicap and destroy the very property that makes PARE
    worth running.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from pinq.ids import canon, h
from pinq.types import EnvCall

# PARE marks read-only tools; anything unmarked is treated as a write for the same reason
# tau2's probe does — a write mislabelled as a read is the failure nobody notices.
_READ_ONLY_ATTR = "read_only"


class PareActuator:
    """Satisfies pinq.protocols.Actuator over a PARE scenario's app set."""

    def __init__(self, apps: Sequence[Any] = (), *, scenario_id: str = "") -> None:
        self._apps = list(apps)
        self.scenario_id = scenario_id
        self._calls: list[EnvCall] = []
        self._seq = 0
        self._native: dict[str, float] = {}

    # ------------------------------------------------------------------ Actuator protocol

    def execute(self, plan: Sequence[Mapping[str, Any]], *, turn_idx: int) -> Sequence[EnvCall]:
        """Run a tool plan. Each entry is {app, name|tool_name, args|kwargs}."""
        out: list[EnvCall] = []
        for step in plan:
            name = str(step.get("name") or step.get("tool_name") or "")
            if not name:
                continue
            out.append(
                self._call(
                    app_name=str(step.get("app") or ""),
                    name=name,
                    kwargs=dict(step.get("args") or step.get("kwargs") or {}),
                    turn_idx=turn_idx,
                )
            )
        self._calls.extend(out)
        return tuple(out)

    def final_hashes(self) -> Mapping[str, str]:
        """One stable hash per app's serialized FSM state. The PARE analogue of a DB hash."""
        out: dict[str, str] = {}
        for app in self._apps:
            label = getattr(app, "name", None) or type(app).__name__
            try:
                state = app.get_state()
                blob = (
                    state
                    if isinstance(state, str)
                    else json.dumps(state, sort_keys=True, default=str)
                )
            except Exception as exc:
                blob = f"ERROR: {exc}"
            out[str(label)] = h("pare_app", blob)
        return out

    def native(self) -> Mapping[str, float]:
        """PARE's own oracle verdict and intervention-timing counters.

        EMPTY UNLESS `attach_validation` WAS CALLED, AND NOTHING CALLS IT. The verdict only
        exists once the scenario has finished, while this Actuator is driven mid-rollout, so
        the two are deliberately separate -- and the second half was never wired. A PARE run
        therefore writes no `oracle_success` row, which downstream is indistinguishable from a
        scenario the policy simply failed.

        This is not fixed because GRID E (PARE) IS CUT: it is exploratory, carries no
        confirmatory test, and its oracle reproduction was never demonstrated. Un-cutting it
        means writing the driver that runs a scenario to completion and calls
        `attach_validation` -- the same shape as `pi_run.stages.tau2_runner`, which exists.
        `tests/test_adapters_pare.py` pins this state so it cannot be forgotten, and fails the
        moment a caller appears.
        """
        return dict(self._native)

    # ------------------------------------------------------------------ PARE specifics

    @property
    def env_calls(self) -> tuple[EnvCall, ...]:
        return tuple(self._calls)

    def attach_validation(self, result: Any) -> None:
        """Fold a PAREScenarioValidationResult into native().

        Separate from execute() because the oracle verdict only exists once the scenario
        has finished, while the Actuator is driven mid-rollout.
        """
        if result is None:
            return
        success = getattr(result, "success", None)
        if success is not None:
            self._native["oracle_success"] = 1.0 if success else 0.0
        for field in (
            "proposal_count",
            "acceptance_count",
            "read_only_actions",
            "write_actions",
            "number_of_turns",
        ):
            val = getattr(result, field, None)
            if val is not None:
                try:
                    self._native[field] = float(val)
                except (TypeError, ValueError):
                    continue

    def _app(self, app_name: str) -> Any | None:
        for app in self._apps:
            if str(getattr(app, "name", None) or type(app).__name__) == app_name:
                return app
        return self._apps[0] if (self._apps and not app_name) else None

    @staticmethod
    def _is_mutating(app: Any, name: str) -> bool:
        tool = getattr(app, name, None)
        flag = getattr(tool, _READ_ONLY_ATTR, None)
        return True if flag is None else not bool(flag)

    def _call(self, *, app_name: str, name: str, kwargs: dict, turn_idx: int) -> EnvCall:
        app = self._app(app_name)
        ok = True
        if app is None:
            ok, result, mutating = False, f"ERROR: no app {app_name!r}", False
        else:
            mutating = self._is_mutating(app, name)
            try:
                fn = getattr(app, name)
                raw = fn(**kwargs)
                result = raw if isinstance(raw, str) else canon(raw)
            except Exception as exc:
                ok, result = False, f"ERROR: {exc}"

        self._seq += 1
        return EnvCall(
            seq=self._seq,
            turn_idx=turn_idx,
            requestor="assistant",
            tool_name=f"{app_name}.{name}" if app_name else name,
            kwargs_json=canon(kwargs),
            ok=ok,
            result_digest=h("res", result)[:16],
            mutating=mutating,
        )
