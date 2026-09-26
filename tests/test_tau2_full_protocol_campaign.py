"""The campaign planner: what gets run, in what order, and in which shard.

Every assertion here defends a way a stopped or resharded campaign silently reports a biased
population rather than a smaller one.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "tau2_full_protocol_campaign", ROOT / "scripts" / "tau2_full_protocol" / "campaign.py"
)
assert _spec and _spec.loader
C = importlib.util.module_from_spec(_spec)
sys.modules["tau2_full_protocol_campaign"] = C
_spec.loader.exec_module(C)


def _data_dir() -> str:
    import os

    d = os.environ.get("TAU2_DATA_DIR", "")
    if not d or not (Path(d) / "tau2" / "domains" / "retail" / "tasks.json").is_file():
        pytest.skip("TAU2_DATA_DIR is not set to an upstream checkout")
    return d


def test_every_cell_names_a_distinct_column() -> None:
    """`arm_id` is NOT the cell id and must not be treated as one: two cells share
    `inquirer_prompted` and differ only in the questioner's model, which is the contrast."""
    ids = [c["cell_id"] for c in C.CELLS]
    assert len(ids) == len(set(ids))
    prompted = [c for c in C.CELLS if c["arm_id"] == "inquirer_prompted"]
    assert len(prompted) == 2, "the prompted arm appears twice, at two questioner pins"
    assert len({c["inquirer"] for c in prompted}) == 2


def test_the_stock_cell_pins_no_questioner() -> None:
    assert C.CELLS_BY_ID["stock"]["inquirer"] == ""
    assert C.CELLS_BY_ID["stock"]["arm_id"] == "tau2_stock"


def test_the_task_order_is_a_seeded_shuffle_and_not_the_files_order() -> None:
    """A prefix of `tasks.json` is a biased subsample -- upstream groups related scenarios --
    so a campaign that stops early must have been running a random order all along."""
    import json

    d = _data_dir()
    raw = [
        str(t["id"]) for t in json.loads((Path(d) / "tau2/domains/retail/tasks.json").read_text())
    ]
    got = C.task_order("retail", d)
    assert sorted(got) == sorted(raw), "no task may be added or dropped by the shuffle"
    assert got != raw, "the order must actually be shuffled"
    assert got == C.task_order("retail", d), "and it must be reproducible from the seed alone"
    assert (
        C.task_order("airline", d) != C.task_order("retail", d)[: len(C.task_order("airline", d))]
    )


def test_a_stopped_campaign_holds_whole_tasks() -> None:
    """The unit order is (task, trial, cell). `pass^k` aggregates the k trials of one task, so
    a prefix that held three of four trials of every task would report nothing at k=4."""
    d = _data_dir()
    units = C.plan(["retail"], [c["cell_id"] for c in C.CELLS], 4, d)
    first = units[: 4 * len(C.CELLS)]
    assert len({u["task_id"] for u in first}) == 1, "one task finishes before the next starts"
    assert sorted({u["seed"] for u in first}) == [0, 1, 2, 3]
    assert {u["cell_id"] for u in first} == set(C.CELLS_BY_ID)


def test_the_plan_is_exactly_tasks_times_trials_times_cells() -> None:
    d = _data_dir()
    units = C.plan(["retail", "airline"], [c["cell_id"] for c in C.CELLS], 4, d)
    assert len(units) == (114 + 50) * 4 * len(C.CELLS)
    assert len({(u["domain"], u["task_id"], u["seed"], u["cell_id"]) for u in units}) == len(units)


def test_shards_partition_by_task_and_are_stable_across_processes() -> None:
    """A task's trials and cells must land together, or a dead shard leaves every task
    incomplete instead of leaving some tasks missing. `hash()` is salted per process, so the
    partition has to come from a digest."""
    d = _data_dir()
    units = C.plan(["retail"], [c["cell_id"] for c in C.CELLS], 4, d)
    by_task: dict[str, set[int]] = {}
    for u in units:
        by_task.setdefault(u["task_id"], set()).add(C.shard_of(u, 8))
    assert all(len(v) == 1 for v in by_task.values()), "a task was split across shards"
    assert len({s for v in by_task.values() for s in v}) > 1, "one shard took everything"
    # Stable: the same unit, asked twice, in this process and in any other.
    assert C.shard_of(units[0], 8) == C.shard_of(dict(units[0]), 8)


def test_every_unit_is_run_by_exactly_one_shard() -> None:
    d = _data_dir()
    units = C.plan(["retail", "airline"], [c["cell_id"] for c in C.CELLS], 4, d)
    for n in (1, 8, 16):
        covered = sum(1 for u in units if 0 <= C.shard_of(u, n) < n)
        assert covered == len(units)


def test_the_bridge_cell_differs_from_its_twin_only_in_the_simulator() -> None:
    """It exists to join this campaign to the recorded fork campaign, so anything else that
    moved with it would make the join measure two changes at once."""
    assert C.BRIDGE_CELL in C.CELLS_BY_ID
    assert C.BRIDGE_USERSIM == "openai/aws/gpt-oss-120b"


def test_a_campaign_without_a_code_version_is_refused() -> None:
    """Runs land in a shared root holding 120k directories; a campaign that cannot name one
    commit cannot be selected back out of it."""
    with pytest.raises(SystemExit):
        C.main(["--runs-root", "/tmp/r", "--cache-root", "/tmp/c", "--data-dir", "/tmp/d"])
