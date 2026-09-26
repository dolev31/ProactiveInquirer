"""Is UserBench actually here, and which 255 scenarios are the test split?

WHY A PROBE RATHER THAN A MODULE-SCOPE IMPORT
    Same reason as tau2's: `travelgym` imports gymnasium, numpy and the OpenAI client at
    module scope, and it is not on PyPI at all -- it is a package inside a git checkout. If
    any module the default test run touches imported it eagerly, `pytest -m "not integration"`
    would stop being runnable on a machine that has never cloned UserBench, which is every
    machine except the one that runs this suite. Every `travelgym` import in this package
    therefore lives inside a function body, and this probe is the only thing that reports the
    outcome.

WHY `USERBENCH_DIR` AND NOT A BUILT CORPUS
    UserBench ships its 3,122 scenarios as ~200 MB of JSON inside the repository, under an
    Apache-2.0 licence that permits redistribution and a size that makes it a bad idea. So
    this suite is SELF-SOURCED exactly as tau2 is: point `USERBENCH_DIR` at a checkout and
    the adapter reads it in place. It never fabricates a corpus when the variable is unset --
    a missing corpus reads downstream as "the policy found nothing", which is a wrong number,
    not an error.

WHAT IS READ FROM WHERE
    <root>/data/<env>_multiturn/test.parquet   -- upstream's own 85/15 split index
    <root>/travelgym/data/travelgym_data_NN.json -- the scenarios themselves

    THE SPLIT INDEX IS GOLD-BEARING AND ONLY ONE OF ITS COLUMNS IS NOT. Every parquet row
    carries `reward_model.ground_truth`, which is the literal answer key ("['C16', 'A17']").
    `split_index` reads `reward_model.id` and nothing else, and `_GOLD_PARQUET_FIELDS` names
    what it is refusing to touch so that a later "just grab the whole row" is a visible
    change rather than an invisible one.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

# The commit this adapter was written and measured against. Recorded here as well as in
# docs/DATA.md because a number in a paper has to name the bytes it came from, and "main"
# is not a name for bytes.
PINNED_COMMIT = "80506d2ab484cab843e60a2401ff3e0290d05b87"

# The three headline environments of the UserBench paper, and upstream's own default for
# `eval.py --envs`. Two aspects with two preferences each, three with three, four with four.
ENVS: tuple[str, ...] = ("travel22", "travel33", "travel44")

# EVAL-ONLY, so there is exactly one split this adapter will ever read. Naming it as a
# constant rather than a parameter is the point: a `split=` argument is an invitation to
# pass "train", and `pinq.splitting.EVAL_ONLY_SUITES` exists to make that impossible.
SPLIT = "test"

# Asserted, never assumed. A silent upstream data change would otherwise move every
# denominator in the paper without moving a line of code. Measured at PINNED_COMMIT.
N_SCENARIOS = 3122  # across all eight data files
N_TRAIN, N_TEST = 2651, 471  # upstream's 85/15 split, all eight environments
N_TEST_BY_ENV = {"travel22": 101, "travel33": 87, "travel44": 67}  # the three headline envs
N_TEST_TASKS = 255  # == sum(N_TEST_BY_ENV.values())

# The eight data files, by the tag that appears in both the filename and the env name:
# travelgym_data_22.json <-> travel22. Listed so a missing file is reported by name.
DATA_TAGS: tuple[str, ...] = ("22", "33", "44", "2222", "233", "334", "444", "333")

# Columns of the split index that this module must never read. `ground_truth` is the answer
# key; `prompt` is upstream's system prompt, which is public, but is not this adapter's
# business either -- our instructions are our own and are hashed into the run identity.
_GOLD_PARQUET_FIELDS = ("ground_truth",)

# Keys of a scenario record that are PUBLIC. Everything else in that dict -- `scenario`, and
# every per-aspect block with its `preferences`, `best_id`, `correct_ids` and `options` -- is
# the answer key. See `scenario_public` for the measurement that says why `scenario` is on
# the wrong side of this line despite reading like a task description.
PUBLIC_SCENARIO_KEYS = ("initial_description",)


class UserBenchUnavailable(RuntimeError):
    """Raised when a UserBench operation is attempted without a usable checkout.

    Separate from a plain KeyError or FileNotFoundError on purpose: the fix is never in the
    code, it is a clone and an environment variable, and the message says so.
    """


def env_tag(env: str) -> str:
    """`travel22` -> `22`, the tag that names the data file holding that env's scenarios."""
    return env[len("travel") :] if env.startswith("travel") else env


def userbench_root(root: str | os.PathLike[str] | None = None) -> Path:
    """The UserBench checkout, from `USERBENCH_DIR` unless one is passed explicitly.

    Raises with the remedy rather than returning a path that does not exist. tau2's
    `domain_data_dir` learned this the hard way: a returned-but-absent path turns a broken
    install into an empty corpus, and an empty corpus scores as a policy that retrieved
    nothing rather than as a machine that was never set up.
    """
    if root is not None:
        d = Path(root)
    else:
        raw = os.environ.get("USERBENCH_DIR", "").strip()
        if not raw:
            raise UserBenchUnavailable(
                "USERBENCH_DIR is unset. UserBench is not on PyPI and its 3,122 scenarios "
                "ship inside the repository, so there is nothing to fall back to. Run:\n"
                "  git clone https://github.com/SalesforceAIResearch/UserBench.git\n"
                f"  git -C UserBench checkout {PINNED_COMMIT}\n"
                "  export USERBENCH_DIR=<path to that checkout>\n"
                "See docs/DATA.md."
            )
        d = Path(raw)
    if not d.is_dir():
        raise UserBenchUnavailable(f"USERBENCH_DIR points at {d}, which is not a directory")
    if not (d / "travelgym" / "data").is_dir():
        raise UserBenchUnavailable(
            f"{d} does not look like a UserBench checkout: {d / 'travelgym' / 'data'} is "
            "missing. Point USERBENCH_DIR at the repository ROOT, not at the travelgym "
            "package inside it."
        )
    return d


def available(root: str | os.PathLike[str] | None = None) -> tuple[bool, str]:
    """(usable, reason). The reason is shown in skip messages, so it must be specific.

    Three separate things can be missing and they have three different fixes: the checkout,
    the `travelgym` package on sys.path, and pyarrow for the split index. Collapsing them
    into one "userbench unavailable" would send a reader to fix the wrong one.
    """
    try:
        d = userbench_root(root)
    except UserBenchUnavailable as exc:
        return False, str(exc)

    missing = [t for t in DATA_TAGS if not data_file(d, t).is_file()]
    if missing:
        return False, f"{d} is missing travelgym/data/travelgym_data_{{{','.join(missing)}}}.json"

    try:
        import pyarrow.parquet  # noqa: F401
    except ImportError as exc:
        return False, (
            f"the UserBench split index is parquet and pyarrow is missing ({exc}); "
            "install with: uv pip install -e '.[analysis]'"
        )

    try:
        import travelgym  # noqa: F401
    except ImportError as exc:
        return False, (
            f"the UserBench checkout is present but `travelgym` is not importable ({exc}); "
            f"install it from the checkout: uv pip install -e {d}"
        )
    return True, "userbench available"


def data_file(root: Path, tag: str) -> Path:
    return root / "travelgym" / "data" / f"travelgym_data_{tag}.json"


def split_index_path(root: Path, env: str, split: str = SPLIT) -> Path:
    """`<root>/data/<env>_multiturn/<split>.parquet`.

    NOT the `_multiturn_onechoice` sibling. The two directories hold the SAME task ids (all
    three headline envs verified at PINNED_COMMIT); they differ only in which prompt upstream
    paired them with, and `one_choice_per_aspect` is a config field we set ourselves. Reading
    the plain one keeps the choice in one place instead of two.
    """
    return root / "data" / f"{env}_multiturn" / f"{split}.parquet"


def split_index(
    root: Path, envs: tuple[str, ...] = ENVS, split: str = SPLIT
) -> tuple[tuple[str, str], ...]:
    """[(scenario_key, env)] for upstream's own split, in file order.

    ONLY `reward_model.id` IS READ. `reward_model.ground_truth` sits in the same struct and
    is the answer key; `extra_info.tools_kwargs` repeats the id. Reading the column rather
    than the row is the whole discipline here -- see `_GOLD_PARQUET_FIELDS`.
    """
    import pyarrow.parquet as pq

    out: list[tuple[str, str]] = []
    for env in envs:
        path = split_index_path(root, env, split)
        if not path.is_file():
            raise UserBenchUnavailable(
                f"no split index at {path}. Check USERBENCH_DIR and that the checkout is at "
                f"{PINNED_COMMIT}."
            )
        column = pq.read_table(path, columns=["reward_model"]).column("reward_model").to_pylist()
        for cell in column:
            key = str((cell or {}).get("id", "")).strip()
            if not key:
                raise UserBenchUnavailable(f"{path}: a row carries no reward_model.id")
            out.append((key, env))
    return tuple(out)


def load_scenarios(root: Path, tag: str) -> dict[str, Any]:
    """One data file, whole. ~33 MB of JSON; measured at 0.5 s for all eight together.

    Returned raw and GOLD-BEARING. Nothing agent-side may touch the result of this function:
    it is fed to `scenario_public` and to the env, and to nothing else.
    """
    path = data_file(root, tag)
    if not path.is_file():
        raise UserBenchUnavailable(f"no UserBench data file at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def scenario_public(record: dict[str, Any]) -> dict[str, str]:
    """The projection: a scenario record down to the fields an agent may be shown.

    THE ONE FIELD THAT LOOKS PUBLIC AND IS NOT is `scenario`. It reads like a task
    description and it is where the user's every preference is written out in prose --
    "I want to ensure that my flight ... includes carry-on baggage allowance, as I always
    keep my travel essentials with me". Measured over the 1,462 preferences in the 255 test
    tasks at PINNED_COMMIT, by content-word (>=4 letters) overlap:

        `scenario`             covers 52.7% of a preference's content words on average,
                               and makes 334 of 1,462 preferences >=80% recoverable;
        `initial_description`  covers  6.0%,
                               and makes    0 of 1,462 preferences >=80% recoverable.

    An agent shown `scenario` does not have to elicit anything, so `elicitation_ratio` --
    the number this suite exists to report -- becomes a measurement of nothing. It is also
    what `TravelEnv` puts in `observation["task_description"]` on every single step, which
    is why there is an observation wall in `session.py` as well as this projection.

    `initial_description` is the honest surface: it is what the env hands back as
    `observation["feedback"]` at reset, and it is the ONLY field upstream's own baseline
    agent is fed (eval.py threads `observation["feedback"]` into the message list and
    nothing else). Using it keeps our runs comparable to the published ones.
    """
    missing = [k for k in PUBLIC_SCENARIO_KEYS if not str(record.get(k, "")).strip()]
    if missing:
        raise UserBenchUnavailable(
            f"scenario record has no {missing}; the upstream data shape has changed and the "
            "public/gold partition must be re-derived before anything is run"
        )
    return {k: str(record[k]).strip() for k in PUBLIC_SCENARIO_KEYS}
