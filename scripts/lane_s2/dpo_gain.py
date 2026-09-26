"""Offline DEV ranking: does preference training move the ask/ask ordering, and does that grow with scale?

WHAT THIS IS. An OFFLINE reading of RANKING ability on the DEVELOPMENT pair file -- never held-out
coverage, never a rollout. Each checkpoint's per-pair records come from the length-matched paired
runner (the one behind the published paired files), which reproduces the published 32B imitation
ladder to within ONE PAIR on identical weights: rung1-32b-headline-s1/checkpoint-1600 reads 469/970
here against 470/970 in artifacts/rung32b_tierA_20260920/RESULT.md. So no difference of a pair or two
is signal, and by the rules fixed with the paper lane (2026-09-23 00:08 IDT), no gain under ~0.03
(about two binomial SE at n = 970) is described as a difference.

THE CONTRAST IS A GAIN, NOT A LEVEL. Each preference-trained arm is scored against ITS OWN
INITIALISATION on the same pairs, per pair: d = ok(arm) - ok(init) in {-1, 0, +1}. Comparing a 32B
level to an 8B level would mix scale with everything else that differs; comparing the DPO GAIN at
32B to the DPO gain at 8B, each against its own init, holds the recipe fixed.

THE UNIT IS THE TASK. 970 pairs come from 58 tasks, up to 63 pairs per task, so a pair count is not an
n. Every gain is reported at pair level and at task level side by side; only the task level is
quotable. Task level = a sign count over task-mean gains with its n-0 floor printed, plus a bootstrap
over tasks of the mean pair-level gain.

REFUSALS, each exit 3, because a silent drop reads as a result: a file scored on a different pair
source; a per-pair id absent from the pairs file; an arm and its init scored on different pair sets; a
missing stratum or per-pair block.

stdlib only, so it runs where the data lives.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from math import comb
from pathlib import Path

STRATA = ("unmatched_ask_ask_all", "tol0", "tol2", "tol5", "tol10", "tol20", "tol30")
EXPECTED_SOURCE_SHA = "d3d80faa2e281d0f56dee4988eb61561d486d1d6df146be476f35bc18c522a0e"
NOISE_FLOOR = (
    "one pair on identical weights (469 vs 470 of 970, rung1-32b-headline-s1/checkpoint-1600)"
)
MIN_DESCRIBED_GAIN = 0.03


class Refuse(Exception):
    pass


def sign_p(pos: int, neg: int) -> float:
    n = pos + neg
    if n == 0:
        return 1.0
    k = min(pos, neg)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)


def load_arm(path: str) -> dict:
    d = json.loads(Path(path).read_text())
    src = d.get("stratum_manifest_source_sha256")
    if src != EXPECTED_SOURCE_SHA:
        raise Refuse(
            f"{path}: scored on pair source {src!r}, expected {EXPECTED_SOURCE_SHA[:16]}..."
        )
    out = {
        "label": d.get("label"),
        "checkpoint": d.get("checkpoint"),
        "adapter_sha": (d.get("checkpoint_provenance") or {}).get("adapter_sha"),
        "registry_name": (d.get("checkpoint_provenance") or {}).get("registry_name"),
        "strata": {},
    }
    for s in STRATA:
        block = (d.get("results") or {}).get(s)
        if not block or "paired" not in block or "per_pair" not in block["paired"]:
            raise Refuse(f"{path}: stratum {s} has no per-pair block")
        out["strata"][s] = {pid: bool(r["ok"]) for pid, r in block["paired"]["per_pair"].items()}
    return out


def gain(arm: dict, init: dict, stratum: str, task_of: dict, n_boot: int, seed: int) -> dict:
    a, b = arm["strata"][stratum], init["strata"][stratum]
    if set(a) != set(b):
        raise Refuse(
            f"{stratum}: arm and init scored on different pair sets "
            f"({len(set(a) ^ set(b))} ids differ)"
        )
    missing = [p for p in a if p not in task_of]
    if missing:
        raise Refuse(
            f"{stratum}: {len(missing)} pair ids absent from the pairs file, e.g. {missing[0]}"
        )
    d = {p: int(a[p]) - int(b[p]) for p in a}
    up = sum(1 for v in d.values() if v > 0)
    down = sum(1 for v in d.values() if v < 0)
    by_task: dict = defaultdict(list)
    for p, v in d.items():
        by_task[task_of[p]].append(v)
    means = [statistics.fmean(v) for v in by_task.values()]
    t_up, t_down = sum(x > 0 for x in means), sum(x < 0 for x in means)
    rng = random.Random(seed)
    tasks = list(by_task)
    boots = []
    for _ in range(n_boot):
        pick = [by_task[rng.choice(tasks)] for _ in tasks]
        flat = [v for grp in pick for v in grp]
        boots.append(statistics.fmean(flat))
    boots.sort()
    return {
        "n_pairs": len(d),
        "gain_pair_mean": statistics.fmean(d.values()),
        "pairs_improved": up,
        "pairs_worsened": down,
        "pair_level_mcnemar_p_NOT_QUOTABLE": sign_p(up, down),
        "n_tasks": len(means),
        "tasks_improved": t_up,
        "tasks_worsened": t_down,
        "tasks_tied": len(means) - t_up - t_down,
        "task_sign_p": sign_p(t_up, t_down),
        "task_sign_floor": sign_p(t_up + t_down, 0),
        "task_bootstrap_ci95": [boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]],
        "n_boot": n_boot,
        "seed": seed,
    }


def level(arm: dict, stratum: str) -> dict:
    v = arm["strata"][stratum]
    return {"n": len(v), "acc": sum(v.values()) / len(v) if v else None}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--pairs", required=True, help="ask_ask_all.jsonl carrying pair_id, suite_id, task_id"
    )
    ap.add_argument("--spec", required=True, help="JSON list of {label, path, role, init}")
    ap.add_argument("--out")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    try:
        task_of = {}
        for line in Path(a.pairs).read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("split") != "dev":
                    raise Refuse(f"pair {r.get('pair_id')} has split {r.get('split')!r}, not dev")
                task_of[r["pair_id"]] = f"{r['suite_id']}/{r['task_id']}"
        spec = json.loads(Path(a.spec).read_text())
        arms = {s["label"]: {**s, **load_arm(s["path"])} for s in spec}
        report: dict = {
            "what": "OFFLINE ranking on the DEV pair file; not held-out coverage",
            "pairs_source_sha256": EXPECTED_SOURCE_SHA,
            "noise_floor": NOISE_FLOOR,
            "min_described_gain": MIN_DESCRIBED_GAIN,
            "unit": "task (suite_id/task_id)",
            "arms": {},
        }
        print(
            f"OFFLINE DEV ranking -- {len(task_of)} pairs over {len(set(task_of.values()))} tasks"
        )
        print(
            f"noise floor: {NOISE_FLOOR}; no gain under {MIN_DESCRIBED_GAIN} is called a difference\n"
        )
        for lab, arm in arms.items():
            row = {
                k: arm.get(k)
                for k in ("role", "init", "path", "checkpoint", "adapter_sha", "registry_name")
            }
            row["levels"] = {s: level(arm, s) for s in STRATA}
            u = row["levels"]["unmatched_ask_ask_all"]
            print(
                f"[{arm.get('role')}] {lab}  unmatched ask_ask {u['acc']:.4f} (n={u['n']})  "
                f"adapter_sha {str(arm.get('adapter_sha'))[:16]}  registry={arm.get('registry_name')}"
            )
            if arm.get("init"):
                if arm["init"] not in arms:
                    raise Refuse(f"{lab}: its init {arm['init']!r} is not in the spec")
                row["gain_vs_init"] = {
                    s: gain(arm, arms[arm["init"]], s, task_of, a.n_boot, a.seed) for s in STRATA
                }
                g = row["gain_vs_init"]["unmatched_ask_ask_all"]
                lo, hi = g["task_bootstrap_ci95"]
                tag = (
                    ""
                    if abs(g["gain_pair_mean"]) >= MIN_DESCRIBED_GAIN
                    else "  [below the described-difference threshold]"
                )
                print(
                    f"      GAIN vs {arm['init']}: {g['gain_pair_mean']:+.4f}  "
                    f"pairs {g['pairs_improved']} up / {g['pairs_worsened']} down (n={g['n_pairs']})"
                )
                print(
                    f"      TASK level (quotable): {g['tasks_improved']} up / {g['tasks_worsened']} down / "
                    f"{g['tasks_tied']} tied of {g['n_tasks']}  sign p={g['task_sign_p']:.4f} "
                    f"(floor {g['task_sign_floor']:.4f})  bootstrap-over-tasks 95% [{lo:+.4f}, {hi:+.4f}]{tag}"
                )
            report["arms"][lab] = row
    except Refuse as e:
        print(f"REFUSING: {e}")
        return 3
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=2))
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
