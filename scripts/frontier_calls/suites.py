"""Where the three frontier snapshots live and how their arms are selected.

Musique stores each arm in its OWN directory (the campaign copied three separate isolated
stores); strategyqa and frames store all arms/pins together in ONE directory, distinguished by
`model_pin_hash` -- `arm_id` is explicitly not a selector on either of those two (both
`teacher` and `base8b` carry `arm_id='inquirer_prompted'`; see each FRONTIER*.md's "arm_id IS
NOT A SELECTOR" section). The `model_pin_hash` values below are transcribed from the three
promoted files and cross-checked against `SELECT DISTINCT model_pin_hash` on each store as
part of `compute.py`'s own sanity checks -- they are not trusted blind.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ArmSpec:
    runs_dir: str  # directory holding runs.parquet + scores.parquet, relative to repo root
    model_pin_hash: str | None  # None: the directory already holds only this arm's rows


@dataclass(frozen=True, slots=True)
class SuiteSpec:
    name: str
    metric: str  # the y-axis metric_name in scores.parquet
    n_per_cell: int  # published population size per (arm, cap) cell
    arms: dict[str, ArmSpec]
    caps: tuple[int, ...] = (4, 8, 12, 16, 24)


MUSIQUE = SuiteSpec(
    name="musique",
    metric="evidence_coverage",
    n_per_cell=400,
    arms={
        "trained": ArmSpec("artifacts/frontier/scores_parquet.trained", None),
        "teacher": ArmSpec("artifacts/frontier/scores_parquet.teacher", None),
        "base8b": ArmSpec("artifacts/frontier/scores_parquet.base8b", None),
    },
)

STRATEGYQA = SuiteSpec(
    name="strategyqa",
    metric="evidence_coverage",
    n_per_cell=400,
    arms={
        "trained": ArmSpec(
            "artifacts/frontier_strategyqa/scores_parquet.strategyqa",
            "6bc87062cbbdd37802120ffc97d2bcd63a924797b456a99115644dc7aae4f856",
        ),
        "teacher": ArmSpec(
            "artifacts/frontier_strategyqa/scores_parquet.strategyqa",
            "5a1322e03e1b7588c214529a36ba718ca96aab243ac9c82c83944ece3c096ebf",
        ),
        "base8b": ArmSpec(
            "artifacts/frontier_strategyqa/scores_parquet.strategyqa",
            "5d6fce016f3dbfc5af8d6c99459459ea0234b5413b2048102245467b3c204856",
        ),
    },
)

FRAMES = SuiteSpec(
    name="frames",
    metric="answer_correct",
    n_per_cell=824,
    arms={
        "trained": ArmSpec(
            "artifacts/frames_frontier/scores_parquet.frames",
            "6bc87062cbbdd37802120ffc97d2bcd63a924797b456a99115644dc7aae4f856",
        ),
        "teacher": ArmSpec(
            "artifacts/frames_frontier/scores_parquet.frames",
            "5a1322e03e1b7588c214529a36ba718ca96aab243ac9c82c83944ece3c096ebf",
        ),
        "base8b": ArmSpec(
            "artifacts/frames_frontier/scores_parquet.frames",
            "e9af49a4b5ff0a0b27a57f7c4feca06566cbfbec3cc0f810920f5f414862fcfe",
        ),
        "drafter_only": ArmSpec(
            "artifacts/frames_frontier/scores_parquet.frames",
            "9793b4e9d3e8fafdfccac34e6d900982cda9b87a0a5431e14ab52bc789d62b4e",
        ),
    },
)

ALL_SUITES = (MUSIQUE, STRATEGYQA, FRAMES)

# The two comparators the paper's dominance claim is about. `drafter_only` (FRAMES-only) is
# reported in the per-cell table but not put through the matched-calls dominance test: it is
# cap-invariant by construction (zero asks at every cap), so "the calls axis" is a single point
# for it and there is nothing to locate a trained point against.
COMPARATORS = ("base8b", "teacher")
