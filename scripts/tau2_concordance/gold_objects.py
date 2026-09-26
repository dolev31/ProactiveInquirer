"""Gold-object uids of a tau2 task, for READING the transfer rerun. Never imported by a rollout.

WHAT A GOLD OBJECT IS. A pre-dialogue DB record that one of the task's gold actions
(`evaluation_criteria.actions` in tasks.json) names, as the HARNESS's own `uids_for_call`
resolves the action's arguments against the index the harness builds from `db.json`
(`pinq_adapters.tau2.{retail,airline}_units`). The same files the harness reads: TAU2_DATA_DIR's
`tau2/domains/<domain>/{db.json,tasks.json}`, the layout `pinq_adapters.tau2._probe.domain_data_dir`
uses. Never PI_GOLD_ROOT, and never a uid parsed from model text.

HOW A CREATED RECORD IS EXCLUDED. The index is built from `db.json`, the state before the
dialogue, and `uids_for_call` returns a uid only for an id that index holds. A gold WRITE that
creates a record (`book_reservation`, a new order) names an id no pre-dialogue table holds -- or
names none at all -- so it contributes nothing. A WRITE that MODIFIES an existing record (cancel
this order, update that reservation) names a pre-dialogue id and counts, because the record
existed before the dialogue and an ask could have fetched it. `conf/tau2/tool_types.json` supplies
which gold actions are WRITEs, and the per-task summary reports how many there were.

WHY THIS FILE EXISTS APART FROM THE READER. tasks.json carries the answer key. Rollout workers
must never import this; `tests/test_transfer_rerun.py` checks that no module under `src/` names
it and that importing the rollout worker does not load it.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from pinq_adapters.tau2 import airline_units, retail_units

#: suite -> (domain directory name, the harness's units module, its index class)
DOMAINS: Mapping[str, tuple[str, Any, Any]] = {
    "tau2_retail": ("retail", retail_units, retail_units.RetailIndex),
    "tau2_airline": ("airline", airline_units, airline_units.AirlineIndex),
}
TOOL_TYPES_PATH = Path(__file__).resolve().parents[2] / "conf" / "tau2" / "tool_types.json"


class GoldObjects:
    """Per task: the pre-dialogue record uids its gold actions name."""

    def __init__(self, data_root: str | Path) -> None:
        self.root = Path(data_root)
        self.types = json.loads(TOOL_TYPES_PATH.read_text())
        self._tasks: dict[str, dict[str, Mapping[str, Any]]] = {}
        self._index: dict[str, Any] = {}
        self.files: dict[str, dict[str, str]] = {}
        for suite, (domain, _, index_cls) in DOMAINS.items():
            d = self.root / "tau2" / "domains" / domain
            db_raw, tasks_raw = (d / "db.json").read_bytes(), (d / "tasks.json").read_bytes()
            self._index[suite] = index_cls.from_db(json.loads(db_raw))
            self._tasks[suite] = {str(t["id"]): t for t in json.loads(tasks_raw)}
            self.files[suite] = {
                "db.json": hashlib.sha256(db_raw).hexdigest(),
                "tasks.json": hashlib.sha256(tasks_raw).hexdigest(),
            }

    @classmethod
    def from_arg_or_env(cls, data_root: str | None) -> "GoldObjects | None":
        root = data_root or os.environ.get("TAU2_DATA_DIR")
        return cls(root) if root else None

    def has_task(self, suite: str, task_id: str) -> bool:
        return str(task_id) in self._tasks.get(suite, {})

    def _actions(self, suite: str, task_id: str) -> list[Mapping[str, Any]]:
        crit = self._tasks[suite][str(task_id)].get("evaluation_criteria") or {}
        return list(crit.get("actions") or [])

    def uids(self, suite: str, task_id: str) -> frozenset[str]:
        """The task's gold objects: pre-dialogue record uids its gold actions name."""
        _, mod, _ = DOMAINS[suite]
        index = self._index[suite]
        return frozenset(
            u
            for a in self._actions(suite, task_id)
            for u in mod.uids_for_call(index, str(a.get("name")), a.get("arguments") or {})
        )

    def summary(self, suite: str, task_id: str) -> dict[str, int]:
        types = self.types.get(DOMAINS[suite][0], {})
        acts = self._actions(suite, task_id)
        return {
            "n_gold_actions": len(acts),
            "n_gold_write_actions": sum(1 for a in acts if types.get(a.get("name")) == "WRITE"),
            "n_gold_objects": len(self.uids(suite, task_id)),
        }
