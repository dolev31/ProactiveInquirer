"""fig:example's selection rule applied to the recipe's training seeds, after it reproduces seed 0's.

WHY. The example figure (App example) shows "the trained questioner" as seed 0, the resumed run.
The rule that chose its task was fixed before any transcript was read
(`scripts/paper_figures_iclr.py::select_example_task`): among held-out MuSiQue tasks where the
no-question, prompted and trained arms all have an ok run at rollout seed 0, keep the tasks where
the trained arm resolves a required need at depth two or more that the prompted arm does not, and
take the lexicographically first task id.

WHAT. That rule, replicated with one change the recipe forces: the published store held one
trained model, so `select_example_task` keys the trained arm by `arm_id` alone, while the
seed-replicate store (`artifacts/seedrep_gate_20260919/scores_parquet`, the same comparator and
no-question runs as `artifacts/testsplit_qa`, plus seeds 0, 1 and 2) holds three. The trained arm is
therefore selected by `arm_id` AND the inquirer model read from `calls.parquet`, as every gate does.

LOCK. Seed 0 in this store must reproduce the published selection exactly: task id, the three run
ids, 175 paired tasks, 16 forward, 7 reverse
(`paper/iclr2027 2/figures/examples_transcripts.provenance.json`). Only then are seeds 1 and 2 read,
each alone (the rule names one trained run per task, so the two seeds are two applications of it),
and their intersection reported.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import duckdb

S0 = "qwen3-8b-dpo-stacked-notdone-both"
MODELS = {"s0": S0, "s1": f"{S0}-s1", "s2": f"{S0}-s2"}
CANARY = re.compile(r"\s*PINQCANARY_[0-9A-Fa-f]+")  # scripts/paper_figures_iclr.py:1879
PUBLISHED = {
    "task_id_run": {
        "drafter_only": "8dac459d4191b93dac2ddfc81055eb84",
        "inquirer_prompted": "6e0d6f8f1b47c53f8cf6de8ce7acac80",
        "inquirer_trained": "8cb53e1942a514274cd30d2a9246f006",
    },
    "n_paired": 175,
    "n_forward": 16,
    "n_reverse": 7,
}


def depths_by_node(gold: Path) -> dict[tuple[str, str], int]:
    out: dict[tuple[str, str], int] = {}
    for line in gold.read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            for n in d["gold_nodes"]:
                out[(n["gold_task_key"], n["gold_node_id"])] = n["gold_depth"]
    return out


def select(con, store: Path, model: str, depths) -> dict:
    runs = con.execute(
        f"""
        WITH m AS (SELECT DISTINCT run_id, model FROM read_parquet('{store}/calls.parquet')
                   WHERE actor = 'inquirer')
        SELECT r.arm_id, r.task_id, r.run_id, m.model
        FROM read_parquet('{store}/runs.parquet') r LEFT JOIN m USING (run_id)
        WHERE r.suite_id = 'musique' AND r.split = 'test' AND r.seed = 0 AND r.status = 'ok'
          AND r.arm_id IN ('drafter_only', 'inquirer_prompted', 'inquirer_trained')"""
    ).fetchall()
    have: dict[str, dict[str, str]] = {}
    for arm, task, run, m in runs:
        if arm == "inquirer_trained" and m != model:
            continue
        if arm == "inquirer_prompted" and m != "qwen3-8b-base":
            continue
        if arm in have.get(task, {}):
            raise SystemExit(f"two {arm} runs at {task} seed 0: refusing to pick one")
        have.setdefault(task, {})[arm] = run
    arms = ("drafter_only", "inquirer_prompted", "inquirer_trained")
    paired = sorted(t for t, d in have.items() if set(arms) <= set(d))
    resolved: dict[str, dict[str, set]] = {}
    for run, node in con.execute(
        f"SELECT run_id, node_id FROM read_parquet('{store}/matches.parquet') "
        "WHERE match_kind = 'resolve'"
    ).fetchall():
        resolved.setdefault(run, set()).add(node)

    def deep(task: str, arm: str) -> set:
        run = have[task][arm]
        return {n for n in resolved.get(run, set()) if depths.get((task, n), 0) >= 2}

    fwd = [t for t in paired if deep(t, "inquirer_trained") - deep(t, "inquirer_prompted")]
    rev = [t for t in paired if deep(t, "inquirer_prompted") - deep(t, "inquirer_trained")]
    first = fwd[0] if fwd else None
    return {
        "n_paired": len(paired),
        "n_forward": len(fwd),
        "n_reverse": len(rev),
        "forward": fwd,
        "reverse": rev,
        "task_id": first,
        "run_ids": have[first] if first else None,
    }


def transcript(con, store: Path, run: str, task: str, depths) -> dict:
    res: dict[int, list[str]] = {}
    for node, turn in con.execute(
        f"SELECT node_id, matched_turn_idx FROM read_parquet('{store}/matches.parquet') "
        "WHERE run_id = ? AND match_kind = 'resolve'",
        [run],
    ).fetchall():
        res.setdefault(int(turn), []).append(node)
    turns = []
    for turn, kind, q in con.execute(
        f"SELECT turn_idx, action_kind, question FROM read_parquet('{store}/turns.parquet') "
        "WHERE run_id = ? ORDER BY turn_idx",
        [run],
    ).fetchall():
        if q:
            got = sorted(res.get(int(turn), []))
            turns.append(
                {
                    "turn_idx": int(turn),
                    "question": CANARY.sub("", q).strip(),
                    "resolved": got,
                    "depths": [depths.get((task, n)) for n in got],
                }
            )
    scores = dict(
        con.execute(
            f"SELECT metric_name, value FROM read_parquet('{store}/scores.parquet') "
            "WHERE run_id = ? AND metric_name IN ('task_success','answer_correct','answer_token_f1')",
            [run],
        ).fetchall()
    )
    n_asks, stop = con.execute(
        f"SELECT n_asks, stop_reason FROM read_parquet('{store}/runs.parquet') WHERE run_id = ?",
        [run],
    ).fetchone()
    return {"run_id": run, "n_asks": int(n_asks), "stop_reason": stop, "turns": turns, **scores}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    store = a.data_root / "artifacts/seedrep_gate_20260919/scores_parquet"
    depths = depths_by_node(a.data_root / "data/gold/graphs/musique/v1.jsonl")
    con = duckdb.connect()
    res: dict = {"store": "artifacts/seedrep_gate_20260919/scores_parquet", "by_seed": {}}
    for tag, model in MODELS.items():
        res["by_seed"][tag] = select(con, store, model, depths)
    s0 = res["by_seed"]["s0"]
    lock = {
        "run_ids": s0["run_ids"] == PUBLISHED["task_id_run"],
        "n_paired": s0["n_paired"] == PUBLISHED["n_paired"],
        "n_forward": s0["n_forward"] == PUBLISHED["n_forward"],
        "n_reverse": s0["n_reverse"] == PUBLISHED["n_reverse"],
    }
    res["lock"] = lock
    print("LOCK s0:", lock, s0["task_id"], s0["n_paired"], s0["n_forward"], s0["n_reverse"])
    if not all(lock.values()):
        raise SystemExit("LOCK FAILED")
    both = sorted(set(res["by_seed"]["s1"]["forward"]) & set(res["by_seed"]["s2"]["forward"]))
    res["both_seeds_forward"] = both
    res["both_seeds_reverse"] = sorted(
        set(res["by_seed"]["s1"]["reverse"]) & set(res["by_seed"]["s2"]["reverse"])
    )
    for tag in ("s1", "s2"):
        r = res["by_seed"][tag]
        print(
            f"{tag}: paired {r['n_paired']} forward {r['n_forward']} reverse {r['n_reverse']} "
            f"first {r['task_id']}"
        )
        r["transcripts"] = {
            arm: transcript(con, store, run, r["task_id"], depths)
            for arm, run in r["run_ids"].items()
        }
    print(f"both seeds forward: {len(both)} first {both[0] if both else None}")
    print(f"both seeds reverse: {len(res['both_seeds_reverse'])}")
    # The recipe reading of the rule: the first task where EACH training seed resolves a deep need
    # the prompted arm does not. Every arm's transcript on that task, both seeds included.
    if both:
        task = both[0]
        rows = con.execute(
            f"""
            WITH m AS (SELECT DISTINCT run_id, model FROM read_parquet('{store}/calls.parquet')
                       WHERE actor = 'inquirer')
            SELECT r.arm_id, m.model, r.run_id FROM read_parquet('{store}/runs.parquet') r
            LEFT JOIN m USING (run_id)
            WHERE r.task_id = ? AND r.seed = 0 AND r.status = 'ok' AND r.split = 'test'""",
            [task],
        ).fetchall()
        pick = {
            "no_question": [r for a_, m, r in rows if a_ == "drafter_only"],
            "prompted": [
                r for a_, m, r in rows if a_ == "inquirer_prompted" and m == "qwen3-8b-base"
            ],
            "s1": [r for a_, m, r in rows if a_ == "inquirer_trained" and m == MODELS["s1"]],
            "s2": [r for a_, m, r in rows if a_ == "inquirer_trained" and m == MODELS["s2"]],
        }
        if any(len(v) != 1 for v in pick.values()):
            raise SystemExit(f"example task {task}: not exactly one run per arm {pick}")
        tr = {k: transcript(con, store, v[0], task, depths) for k, v in pick.items()}
        for k, t in tr.items():
            outcome = json.loads((a.data_root / "runs" / t["run_id"] / "outcome.json").read_text())
            t["answer_text"] = CANARY.sub("", outcome["answer"]["text"]).strip()
        res["recipe_example"] = {"task_id": task, "transcripts": tr}
        print(f"recipe example task {task}: runs { ({k: v[0] for k, v in pick.items()}) }")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1, sort_keys=True, default=str) + "\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
