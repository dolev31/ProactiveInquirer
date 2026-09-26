"""Regenerate `pairs_control_golden.jsonl` -- the byte-level pin on the CONTROL export.

`pairs.jsonl` (rank_rule="outcome") is the artifact the latent-pursuit headline and the
rater-agreement number are quoted on. Any change to `export_pairs` must leave it alone, so this
writes a small fixture export that `tests/test_pairs_rater_rescue.py` compares against field by
field. Fixture rows rather than the live corpus, so the golden is reproducible by anyone from
this file alone and does not move when the corpus grows.

The task ids are chosen to bucket to the TRAIN split (`pinq.splitting.split_of`): "t3" is dev
and "t5" is test, and rows on them are refused by `assert_trainable` -- which is the firewall
working, and cost an hour of debugging when this fixture first used "t3".

    .venv/bin/python tests/fixtures/make_pairs_control_golden.py

The write is guarded by `__main__` on purpose: the test execs this module to reuse ROWS, and an
unguarded write would regenerate the expectation the test is checking against, so the test
would pass by rewriting its own answer.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")

from pinq.actions import STOP_ACTION_JSON, ask_action_json  # noqa: E402
from pinq_train.export.dataset import export_pairs  # noqa: E402

GOLDEN = Path(__file__).parent / "pairs_control_golden.jsonl"


def row(
    run_id: str,
    value: float,
    question: str,
    *,
    turn: int = 1,
    done: bool = False,
    is_stop: bool = False,
    correct: float | None = None,
    ttc: int | None = None,
    reach: bool | None = None,
    task: str = "t1",
    parent: str = "p",
) -> dict:
    r = {
        "suite_id": "musique",
        "task_id": task,
        "template_id": None,
        "run_id": run_id,
        "turn_idx": turn,
        "state_text": f"S{turn}",
        "action_json": STOP_ACTION_JSON if is_stop else ask_action_json(question),
        "value": value,
        "phi_tilde": value,
        "scorer_hash": "sh",
        "graph_version": "v1",
        "branch_of_run_id": parent,
        "branch_turn_idx": turn,
        "pins_sha": "pins-A",
        "budget_cap": 24,
        "max_turns": 24,
        "is_stop": is_stop,
        "done_before": done,
        "coverage_before": 1.0 if done else 0.4,
        "frontier_size": 0 if done else 2,
        "matcher_id": "m1",
        "reward_weights_sha": "rw1",
        "split": "train",
    }
    if correct is not None:
        r["answer_correct"] = correct
    if ttc is not None:
        r["turns_to_complete"] = ttc
    if reach is not None:
        r["newly_reachable"] = reach
        r["is_latent"] = reach
        r["latent_depth"] = 1 if reach else 0
    return r


# Three states, chosen so the golden covers all three pair kinds and three deciding keys.
ROWS = [
    row("a", 0.90, "who founded it?", correct=1.0, ttc=2, reach=True),
    row("b", 0.30, "when was it founded?", correct=0.0, ttc=4, reach=False),
    row("c", 0.10, "where is it?", ttc=4, reach=False),
    row("d", 0.50, "who wrote it?", turn=2, done=True, parent="p2", task="t2"),
    row("e", 0.20, "what year was it published?", turn=2, done=True, parent="p2", task="t2"),
    row("f", 0.60, "which country issued it?", turn=3, done=True, parent="p4", task="t4"),
    row("g", 0.00, "", turn=3, done=True, is_stop=True, parent="p4", task="t4"),
]


if __name__ == "__main__":
    pairs, man = export_pairs(ROWS, margin_threshold=0.05, len_delta_max=40)
    lines = [json.dumps(dataclasses.asdict(p), sort_keys=True, default=str) for p in pairs]
    out = "\n".join(lines) + "\n"
    GOLDEN.write_text(out)
    print(f"golden rows: {len(lines)}")
    print("sha256:", hashlib.sha256(out.encode()).hexdigest())
    print("kinds:", sorted({p.pair_kind for p in pairs}))
    print("manifest n_pairs:", man.n_pairs, "rank_rule:", man.rank_rule)
