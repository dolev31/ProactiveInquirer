"""Which checkpoint a preempted rung picks up from. Pure, stdlib-only, no GPU.

WHY THIS IS ITS OWN MODULE. Both rungs need it and neither may import the other; putting it in
`rung1_sft` would make rung 2 import rung 1 for a directory listing. It touches no library from
the `[train]` extra either, so the gate venv -- which has no torch, trl, peft or transformers --
exercises the whole of it.

WHAT A CHECKPOINT IS, HERE. `transformers.Trainer` writes `<output_dir>/checkpoint-<global_step>/`
and puts `trainer_state.json` in it LAST, after the weights and the optimiser state. That file is
therefore the completeness marker: a directory without it is a job that was killed mid-write, and
handing it to `resume_from_checkpoint` raises after the base model has loaded -- on a rented or
reserved card, the expensive place to find out. So it is skipped and the previous one is used.

THE STEP IS PARSED AS AN INTEGER, NOT SORTED AS A STRING. `sorted()` over the names puts
`checkpoint-1000` before `checkpoint-200`, so a lexicographic pick resumes from the older
checkpoint and replays 800 steps while reporting that it resumed. Nothing downstream could see
that: the loss curve of a run that replayed 800 steps looks like the loss curve of a run.
"""

from __future__ import annotations

from pathlib import Path

PREFIX = "checkpoint-"
# Written last by `Trainer._save_checkpoint`, which is what makes it the completeness marker.
STATE_FILE = "trainer_state.json"


def latest_checkpoint(out_dir: str) -> str | None:
    """The `checkpoint-N` under `out_dir` with the highest N that carries a `trainer_state.json`.

    `None` -- not an empty string and not a raise -- when there is nothing to resume from: that
    is the value `Trainer.train(resume_from_checkpoint=...)` wants for "start fresh", so the
    first submission of a job and a resubmission go down the same call.
    """
    root = Path(out_dir)
    if not root.is_dir():
        return None
    best: tuple[int, Path] | None = None
    for child in root.iterdir():
        if not child.is_dir() or not child.name.startswith(PREFIX):
            continue
        step = child.name[len(PREFIX) :]
        if not step.isdigit() or not (child / STATE_FILE).is_file():
            continue
        n = int(step)
        if best is None or n > best[0]:
            best = (n, child)
    return str(best[1]) if best else None
