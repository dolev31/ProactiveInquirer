"""The train/dev/test split, and the five layers that keep it honest.

Contamination here is not recoverable after the fact: if a task the policy trained on shows
up in a confirmatory table, every number in that table is void and there is no way to tell
from the output that it happened. So the split is enforced five independent ways, and each
one alone would be enough to catch the common case:

  1. the bucket function itself, hashed on `template_id` where one exists so near-duplicate
     variants of the same template cannot straddle the boundary;
  2. the exporter refuses any row outside `train`;
  3. the exported manifest records a set-hash of the ids it contains, AND the set itself is
     written beside it as `train_ids.<hash16>.txt` (`export.dataset.write_jsonl`), so a later
     run can prove which ids a checkpoint saw. UNTIL 2026-09-15 THIS LINE CLAIMED A PROOF THAT
     COULD NOT BE PERFORMED: the hash had no preimage anywhere on disk -- the exporter built
     the id list, hashed it and dropped it -- nothing stamped the hash onto a run, and
     `pi_eval.report.train_split_violations` selected the column and ignored it. What makes it
     true now is the whole chain: the sidecar, `conf/checkpoints.json` carrying the hash per
     deployment, `pi_run.manifest.build_manifest` stamping it on any run whose Inquirer is a
     registered checkpoint, and that check loading the id file and flagging a trained-arm row
     whose task is IN the set -- whatever the split says today, which is the case layer 1
     structurally cannot catch;
  4. the scorer refuses to score anything that is NOT `train` and exits non-zero
     (`pi_run.serve.score.assert_scorable`: `split_assert` is fixed at "train", and a
     run in any other split raises `SplitRefused`). This line used to say the scorer
     refuses a `train` task, which is the opposite protection -- the layer keeps the
     TRAINER off held-out data, it does not keep it off its own training data;
  5. poison canary ids that no exporter may ever emit -- if one appears in a training log,
     the run is burned.

`tau2` -- the BANKING domain -- contributes ZERO training tasks by construction (see
`EVAL_ONLY_SUITES`), so it remains a genuine zero-shot transfer target rather than a suite the
policy has quietly seen. This used to read "tau2 contributes zero training tasks", which was
true of the only tau2 suite that existed when it was written and is now misleading: `tau2` is
one of three tau2 suite ids. `tau2_retail` and `tau2_airline` are separate ids precisely so
banking can stay untouched while a tool domain is trainable, and they are NOT eval-only.
Airline's split is upstream's own `split_tasks.json` rather than our hash
(`pinq.splitting.FIXED_SPLITS`); retail's is the hash. Under decision D2 neither is trained on
in this paper, but that is a decision about this paper, not a property of the code -- and only
the property of the code protects banking.
"""

from __future__ import annotations

import hashlib
from typing import Iterable, Literal

from pinq.splitting import EVAL_ONLY_SUITES
from pinq.splitting import bucket as _bucket
from pinq.splitting import split_of as _split_of

Split = Literal["train", "dev", "test"]

# Suites that may never contribute a training example. tau2 is eval-only because 97 tasks is
# far too few to split, and because "transfers to an action-consequential environment it was
# never trained on" is a much stronger claim than a within-suite gain.
#
# RE-EXPORTED, NOT REDEFINED. This module already delegates `bucket` and `split_of` to
# `pinq.splitting` for the reason that module's docstring gives -- two definitions of one
# split silently discarded 66 of 109 rows. The eval-only set was the one piece of the same
# policy still written out twice, so adding a suite meant editing two literals and hoping.
# `pinq` is stdlib-only and importable from everywhere, so there is no firewall reason for a
# second copy.
__all__ = [
    "EVAL_ONLY_SUITES",
    "Split",
    "bucket",
    "split_of",
    "id_set_hash",
    "assert_trainable",
    "assert_evaluable",
]

TRAIN_MAX, DEV_MAX = 59, 74  # 0-59 train, 60-74 dev, 75-99 test


def bucket(suite_id: str, task_id: str, template_id: str | None = None) -> int:
    """Delegates to pinq.splitting. See there for why there is only one of these."""
    return _bucket(suite_id, task_id, template_id)


def split_of(suite_id: str, task_id: str, template_id: str | None = None) -> Split:
    """Delegates to pinq.splitting."""
    return _split_of(suite_id, task_id, template_id)


def id_set_hash(ids: Iterable[str]) -> str:
    """Set-hash of exported ids. Order-independent, so it proves membership rather than
    recording an accident of iteration order."""
    return hashlib.sha256("\n".join(sorted(set(ids))).encode()).hexdigest()


class SplitViolation(RuntimeError):
    """Raised the moment a non-train row reaches the exporter."""


def assert_trainable(suite_id: str, task_id: str, template_id: str | None = None) -> None:
    if suite_id in EVAL_ONLY_SUITES:
        raise SplitViolation(
            f"{suite_id} is eval-only: exporting it would destroy the zero-shot transfer "
            f"claim that is the point of including it."
        )
    s = split_of(suite_id, task_id, template_id)
    if s != "train":
        raise SplitViolation(f"{suite_id}/{task_id} is in '{s}', not 'train'")


def assert_evaluable(
    suite_id: str, task_id: str, template_id: str | None = None, *, split: str
) -> None:
    """Layer 2, for a HELD-OUT export. `assert_trainable` above is untouched.

    A SEPARATE FUNCTION, not a parameter on the one above. `assert_trainable` is enumerated in
    this module's docstring as one of the five layers that keep the split honest, and adding a
    `split=` to it would mean the layer that refuses held-out data could be told to permit it --
    the precise shape of the defect `pi_run.serve.score.assert_scorable` was fixed for.

    THE SPLIT MUST BE NAMED, not merely "not train". `split_of` maps every eval-only suite to
    `test` by name, so a test-split export that only checked "is it test?" would ADMIT tau2,
    pare and userbench -- the three benchmarks whose whole value is that the policy never saw
    them. Hence the by-name refusal first, in every split, exactly as `assert_trainable` does it.
    """
    if suite_id in EVAL_ONLY_SUITES:
        raise SplitViolation(
            f"{suite_id} is eval-only: it is scored through the transfer benchmark's own "
            f"evaluator, never mined into a dataset -- in any split."
        )
    s = split_of(suite_id, task_id, template_id)
    if s != split:
        raise SplitViolation(f"{suite_id}/{task_id} is in '{s}', not '{split}'")
