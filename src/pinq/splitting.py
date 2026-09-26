"""The train/dev/test split. ONE definition, because two were worse than either.

There were two: `pi_run.manifest.split_of` hashed `h("split", suite, key)` while
`pinq_train.split.bucket` hashed `sha256(f"{suite}|{key}")`. A run was STAMPED with the first
and RE-CHECKED against the second, so a task could be `train` for the runner and `test` for
the exporter. The exporter took the intersection, so nothing ever leaked -- but it silently
discarded 66 of 109 rows, and a disagreement that only ever loses data is still a disagreement
about which tasks are trainable.

This lives in `pinq` because it is stdlib-only and both sides may import it: `pi_run` stamps
the manifest with it, `pinq_train` refuses on it. Neither may import the other's copy.

Changing the hash RECLASSIFIES which tasks are trainable, so it is a thing to settle before
anything is trained and before the preregistration is sealed -- which is where this repo is.
"""

from __future__ import annotations

from typing import Literal

from .ids import h

Split = Literal["train", "dev", "test"]

TRAIN_MAX, DEV_MAX = 59, 74  # 0-59 train, 60-74 dev, 75-99 test

# Suites that may never contribute a training example. tau2 has 97 tasks -- far too few to
# split -- and "transfers to an action-consequential environment it never trained on" is a much
# stronger claim than a within-suite gain. PARE is the transfer experiment for the same reason.
#
# userbench is the third, and its reason is the strongest of the three: it is the only
# benchmark in this project that its authors, not we, designed, which is the entire answer to
# "you measured yourself with your own ruler". Upstream ships 2,651 train scenarios against
# 471 test ones, so training on it is not merely possible, it is easy and it is one config
# line away -- and a policy trained on travelgym scenarios would still produce a number that
# LOOKS like an external validation. There is no artefact anywhere downstream that would show
# the difference, which is why the refusal is here, by name, before any hash is taken.
#
# frames is the fourth, and it is the only one that was never ours to train on in the first
# place: FRAMES ships a single `test` split of 824 questions and no train split exists
# upstream. Declaring it here rather than relying on that absence is the point -- "there is no
# train split" is a property of today's dataset card, while this is a property of the code.
#
# drgym is the fifth, by the user's decision on 2026-09-15: DeepResearchGym is an evaluation
# benchmark, and a training row drawn from it contaminates the suite that carries the paper's
# horizontal endpoint. It is the addition that changes the least and is therefore the one most
# worth writing down. MEASURED the same day, before the line was edited:
#
#   * no shipped training file holds a drgym row -- `grep -c drgym` is 0 on each of
#     data/rl/{sft,pairs,a6_preferences,reaches_preferences,latency_relabel}.jsonl, and
#     data/rl/sft.manifest.json names suites musique, strategyqa, synth, wiki2;
#   * `pi_run.suites.REGISTRY["drgym"]` already declared `mines_training_data=False`, so the
#     audit was clean before and after and no dataset moves;
#   * runs.parquet nonetheless carries 201 drgym runs over 57 tasks stamped `split=train`
#     (plus 51 over 17 at `dev`) from campaigns that predate the decision. All 57 of those
#     task ids hash into the train bucket, so the stamp was this function's own answer.
#
# Those three together are the reason this is a line of code and not a note in a document. The
# refusal used to hold because nobody had run the export; from here it holds because
# `split_of` says `test` and `assert_trainable` refuses by name. The stale `train` stamps stay
# on disk and stay honest -- `split` is not in `pinq.ids.SEMANTIC_FIELDS`, so no run_id moves
# and nothing already recorded becomes unreproducible; what changes is that they can no longer
# be mined. See docs/SUITES.md for the decision record.
#
# musique_x2 is the sixth (2026-09-23): each of its tasks joins two held-out MuSiQue TEST tasks
# (pi_eval.build.compose_build), so a training row from it would carry test content into a
# training file. Its reason is a property of how it was built, not a decision about the dataset.
EVAL_ONLY_SUITES: frozenset[str] = frozenset(
    {"tau2", "pare", "userbench", "frames", "drgym", "musique_x2"}
)


class UnknownFixedSplitTask(KeyError):
    """A task id in a fixed-split suite that the table does not assign.

    Raised rather than falling back to `bucket()`, because a fallback would hand the task a
    split upstream never gave it and would be indistinguishable from a typo in the table.
    """


# Suites whose split is UPSTREAM's, not a hash of ours.
#
# tau2-bench ships `domains/airline/split_tasks.json` with 30 train and 20 test ids, and the
# public trajectories we fork prefixes from were produced against that split. Hashing the id
# ourselves scatters those 30 across our own train/dev/test, so a prefix drawn from an upstream
# "train" trace can land on a task we call `test` -- and nothing downstream would notice,
# because `assert_trainable` asks THIS function. The contamination would be real and invisible
# at once. MEASURED on the 20 airline tasks the fork benchmark is evaluated over: the hash put
# 14 of them in `train` and 2 in `dev`, leaving 4 in `test`.
#
# It belongs here and not in the mining query for the reason this module exists: all five
# layers that decide trainability route through `split_of`, so a restriction applied anywhere
# else leaves the manifest's stamped `split` disagreeing with what was actually mined.
#
# No dev bucket: upstream ships train/test only, and inventing one would put tasks in a
# bucket upstream never assigned. Transcribed from the file; `tests/test_airline_fixed_split.py`
# asserts the table still equals it.
#
# RETAIL IS DELIBERATELY ABSENT, though upstream ships a `retail/split_tasks.json` (74 train /
# 40 test). The defect points the other way there: MEASURED, the hash already places all 25
# retail fork-point tasks in `test`, which is how the 204 retail fork runs on disk came to be
# stamped `split=test`, while upstream's file puts 13 of those 25 in its TRAIN half. Adopting
# it for retail would reclassify rows that already exist and contaminate the grid the table is
# meant to protect. Same reasoning, opposite conclusion, because the measurement differs.
# fmt: off
FIXED_SPLITS: dict[str, dict[str, Split]] = {
    "tau2_airline": {
        **dict.fromkeys((
            "0", "1", "3", "4", "5", "7", "9", "10", "11", "12", "14", "15", "17", "20",
            "21", "23", "27", "28", "33", "34", "36", "38", "39", "40", "41", "42", "43",
            "46", "47", "49",
        ), "train"),
        **dict.fromkeys((
            "2", "6", "8", "13", "16", "18", "19", "22", "24", "25", "26", "29", "30",
            "31", "32", "35", "37", "44", "45", "48",
        ), "test"),
    },
}
# fmt: on


def split_key(suite_id: str, task_id: str, template_id: str | None = None) -> str:
    """Hash on `template_id` where a suite has one.

    Two instantiations of one template are near-duplicates, and letting them straddle the
    boundary is the standard way a 'clean' split leaks.
    """
    return f"{suite_id}|{template_id or task_id}"


def bucket(suite_id: str, task_id: str, template_id: str | None = None) -> int:
    """Deterministic 0..99. Domain-separated, matching every other id in the project."""
    return int(h("split", suite_id, template_id or task_id)[:8], 16) % 100


def split_of(suite_id: str, task_id: str, template_id: str | None = None) -> Split:
    if suite_id in EVAL_ONLY_SUITES:
        return "test"
    fixed = FIXED_SPLITS.get(suite_id)
    if fixed is not None:
        # Eval-only is checked FIRST, so a suite can never be both; `pi suites audit` asserts
        # the same thing from the registry side.
        try:
            return fixed[task_id]
        except KeyError as exc:
            raise UnknownFixedSplitTask(
                f"{suite_id} has a fixed split and does not assign task {task_id!r}"
            ) from exc
    b = bucket(suite_id, task_id, template_id)
    return "train" if b <= TRAIN_MAX else "dev" if b <= DEV_MAX else "test"
