"""tau2_airline uses UPSTREAM's split, not a hash of the task id.

WHY A FIXED TABLE AND NOT `bucket()`. The public airline traces we fork prefixes from were
produced on upstream's train tasks. If our split disagreed with upstream's, a prefix taken from
a "train" trace could land on a task this repo calls `test`, and the contamination would be
invisible: `assert_trainable` would pass, because it asks OUR `split_of`.

WHY IT LIVES IN `split_of` AND NOT IN THE MINING QUERY. All five layers that decide what is
trainable route through `split_of` -- the manifest stamp, `assert_trainable` at both export
entry points, the eligibility report, and the grids. Restricting `task_ids()` for mining
instead would leave the manifest's stamped `split` disagreeing with what was mined, which is
exactly the two-definitions bug `pinq.splitting`'s docstring records.

An id absent from the table RAISES. Falling back to the hash would silently give a task a
split that upstream never assigned it, and the fallback would be indistinguishable from a typo
in the table.

WHY RETAIL IS NOT IN THE TABLE, THOUGH UPSTREAM SHIPS A FILE FOR IT. Measured, not assumed --
see `test_the_fork_point_tasks_of_both_domains_are_test` below, which is the decision record.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from pinq.splitting import EVAL_ONLY_SUITES, FIXED_SPLITS, UnknownFixedSplitTask, split_of

# Committed expectations, read off upstream's split_tasks.json for the airline domain.
# These four are spot checks; the whole-table equality is the integration test below.
TRAIN_SAMPLES = ("0", "1", "3")
TEST_SAMPLES = ("2", "6", "8")

# The distinct `task_id`s of the paired-fork runs on disk: arms inquirer_prompted/self_ask,
# split `test`, in scores/parquet/runs.parquet. 20 for airline, 25 for retail -- the grid the
# paper's headline (docs/reports/forks_tau2_*_test.json, 102 pairs each) is computed over.
AIRLINE_FORK_TASKS = (
    "2", "6", "8", "13", "16", "18", "19", "22", "24", "25",
    "26", "29", "30", "31", "32", "35", "37", "44", "45", "48",
)  # fmt: skip
RETAIL_FORK_TASKS = (
    "15", "18", "19", "22", "37", "40", "51", "55", "60", "61", "64", "65", "67",
    "71", "76", "84", "88", "90", "91", "95", "97", "98", "104", "105", "108",
)  # fmt: skip


@pytest.mark.parametrize("tid", TRAIN_SAMPLES)
def test_upstream_train_tasks_are_train(tid: str) -> None:
    assert split_of("tau2_airline", tid) == "train"


@pytest.mark.parametrize("tid", TEST_SAMPLES)
def test_upstream_test_tasks_are_test(tid: str) -> None:
    assert split_of("tau2_airline", tid) == "test"


def test_airline_has_no_dev_split() -> None:
    """Upstream ships train/test only. Inventing a dev split would put tasks in a bucket
    upstream never assigned, and dev is what the ladder tunes on."""
    assert set(FIXED_SPLITS["tau2_airline"].values()) == {"train", "test"}


def test_unknown_task_id_raises_rather_than_hashing() -> None:
    with pytest.raises(UnknownFixedSplitTask):
        split_of("tau2_airline", "no-such-task")


def test_fixed_split_does_not_change_other_suites() -> None:
    """The table is consulted only for suites that appear in it. Every existing suite must
    keep the split it already has, or previously exported rows silently reclassify."""
    assert set(FIXED_SPLITS) == {"tau2_airline"}
    assert "tau2_airline" not in EVAL_ONLY_SUITES  # eval-only is checked first and would win


def test_the_fork_point_tasks_of_both_domains_are_test() -> None:
    """THE DECISION RECORD for why airline is in the table and retail is not.

    The fork benchmark's evaluation grid is the task ids above. Every one of them must be
    `test`, or the headline is computed over tasks the policy may train on.

    AIRLINE NEEDED THE TABLE. Measured on this checkout, the hash splitter put 14 of the 20
    airline fork-point tasks in `train` and 2 in `dev`, leaving 4 in `test`:

        train  6 8 16 18 19 22 25 26 29 30 32 35 37 45
        dev    24 44
        test   2 13 31 48

    Upstream's split_tasks.json puts all 20 in `test`, and the runs on disk are stamped
    `split=test`. Without the table the grid is contaminated by construction.

    RETAIL MUST NOT HAVE IT, AND THE SAME MEASUREMENT IS WHY. Upstream ships a retail
    split_tasks.json too (74 train / 40 test), but 13 of the 25 retail fork-point tasks are in
    its TRAIN half:

        104 105 15 19 22 37 67 76 84 88 91 95 98

    The hash splitter puts all 25 in `test`, which is how the 204 retail fork runs on disk came
    to be stamped `split=test`. Adopting upstream's file for retail would move 13 of them into
    `train` -- reclassifying rows that already exist and contaminating the grid it was meant to
    protect. The direction of the defect is opposite in the two domains, so the remedy is too.
    """
    for tid in AIRLINE_FORK_TASKS:
        assert split_of("tau2_airline", tid) == "test", tid
    for tid in RETAIL_FORK_TASKS:
        assert split_of("tau2_retail", tid) == "test", tid


# ------------------------------------------------------------------------------ integration

_DATA = os.environ.get("TAU2_DATA_DIR", "")
_SPLIT_FILE = Path(_DATA) / "tau2" / "domains" / "airline" / "split_tasks.json" if _DATA else None
_ok = _SPLIT_FILE is not None and _SPLIT_FILE.is_file()
integration = pytest.mark.skipif(
    not _ok, reason="TAU2_DATA_DIR unset or airline/split_tasks.json missing"
)


@integration
def test_table_is_exactly_upstreams_file() -> None:
    """The literal must equal the file it was copied from, or it is a stale transcription."""
    assert _SPLIT_FILE is not None
    upstream = json.loads(_SPLIT_FILE.read_text())
    expected = {tid: "train" for tid in upstream["train"]}
    expected.update({tid: "test" for tid in upstream["test"]})
    assert FIXED_SPLITS["tau2_airline"] == expected
    assert set(expected) == set(upstream["base"])


@integration
def test_the_airline_fork_tasks_are_exactly_upstreams_test_half() -> None:
    """The grid is not a sample of upstream's test split; it IS upstream's test split. If a
    future campaign forks a task outside it, this fails rather than quietly widening."""
    assert _SPLIT_FILE is not None
    upstream = json.loads(_SPLIT_FILE.read_text())
    assert set(AIRLINE_FORK_TASKS) == set(map(str, upstream["test"]))
