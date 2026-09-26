"""Does the rung-0 winner's gain come from what the prompt says, or how long it is?

Scores three prompts on rung 0's OWN held-out slice, through the SAME SeamEvaluator, the same
reward weights and the same config, so the numbers are comparable with the search's own.

  base    the shipped template          -- rung 0 measured 0.3326 here
  winner  the rung-0 winner (522 tok)   -- rung 0 measured 0.6217 here
  lenctl  base CONTENT at winner LENGTH -- the answer

base and winner are re-scored rather than quoted: if this harness cannot reproduce the
search's own two numbers, its verdict on the third is worthless.
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")
from pi_run.cmd_train import _read_task_keys
from pinq import promptlib
from pinq_train import rung0_gepa as gepa
from pinq_train.client import SeamClient
from pinq_train.reward import DEFAULT_WEIGHTS

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
TASK_IDS = Path(sys.argv[2])

# READ, NOT RETYPED: `budget_cap`/`max_turns` below used to duplicate `Rung0Config`'s own
# field defaults by coincidence of value. A change to either default on `Rung0Config` now
# reaches this length-control harness automatically instead of silently diverging from it --
# and the harness's whole point is comparing prompts under the SAME cap the search used.
_rung0_defaults = {f.name: f.default for f in dataclasses.fields(gepa.Rung0Config)}

cfg = gepa.Rung0Config(
    suites=("musique", "wiki2"),
    n_tasks=12,
    n_val_tasks=6,
    n_candidates=6,
    n_generations=4,
    seed=0,
    budget_cap=_rung0_defaults["budget_cap"],
    max_turns=_rung0_defaults["max_turns"],
    arm_id="inquirer_prompted",
    base_prompt="inquirer_prompted",
)
keys = _read_task_keys(gepa, TASK_IDS)
n_suites = len(cfg.suites)
LO = int(os.environ.get("PI_VAL_LO", cfg.n_tasks * n_suites))
HI = int(os.environ.get("PI_VAL_HI", (cfg.n_tasks + cfg.n_val_tasks) * n_suites))
val = keys[LO:HI]
print(f"held-out slice: keys[{LO}:{HI}] = {len(val)} tasks")
for k in val:
    print("   ", k)

client = SeamClient(base_url="http://127.0.0.1:8077", rollout_url="http://127.0.0.1:8078")
ev = gepa.SeamEvaluator(client=client, cfg=cfg, overlay_root=OUT / "overlays")

# the weights the search used; refuse to proceed if they differ
want = json.load(open("artifacts/rung0/rung0.manifest.json"))["weights_sha"]
got = DEFAULT_WEIGHTS.sha
print(
    f"\nweights_sha: search={want[:16]}  here={got[:16]}  {'MATCH' if got == want else 'MISMATCH'}"
)
if got != want:
    raise SystemExit(
        "reward weights differ from the search's -- the numbers would not be comparable"
    )

ARMS = {
    "base": promptlib.load("inquirer_prompted"),
    "lenctl": promptlib.load("inquirer_lenctl"),
    "winner": Path("conf/prompts/rung0/inquirer_prompted.txt").read_text(),
}
for n, t in ARMS.items():
    print(f"  {n:<8} {promptlib.count_tokens(t):>4} tokens")

# RESUME. Written after every task, so a network drop costs at most one rollout -- and even
# that is usually a cache hit, because the seam caches by request bytes.
STATE = OUT / "progress.json"
done: dict[str, dict[str, float]] = {}
if STATE.exists():
    try:
        done = json.load(open(STATE))
        print(f"\nRESUMING: {sum(len(v) for v in done.values())} (candidate, task) results on disk")
    except Exception as exc:
        print(f"  progress.json unreadable ({exc}); starting clean")
        done = {}

results: dict[str, dict] = {}
for name, text in ARMS.items():
    cand = gepa.Candidate(text=text, generation=0, note=f"lenctl-experiment:{name}")
    per: dict[str, float] = dict(done.get(name, {}))
    print(f"\n=== {name} ({cand.cid[:12]}) ===  {len(per)}/{len(val)} already done", flush=True)
    for i, key in enumerate(val, 1):
        if str(key) in per:
            print(
                f"  [{i}/{len(val)}] {key}  reward={per[str(key)]:+.4f}  (cached from a prior run)",
                flush=True,
            )
            continue
        t0 = time.time()
        try:
            r = ev(cand, key)
        except Exception as exc:  # a network drop must not lose the finished tasks
            print(
                f"  [{i}/{len(val)}] {key}  ABORT {type(exc).__name__}: {str(exc)[:90]}", flush=True
            )
            done[name] = per
            json.dump(done, open(STATE, "w"), indent=1)
            raise
        per[str(key)] = r.reward
        print(
            f"  [{i}/{len(val)}] {key}  reward={r.reward:+.4f}  "
            f"{time.time() - t0:.0f}s{'  ERR ' + str(r.error)[:60] if r.error else ''}",
            flush=True,
        )
        done[name] = per
        json.dump(done, open(STATE, "w"), indent=1)
    ok = [v for v in per.values() if v != float("-inf")]
    results[name] = {
        "cid": cand.cid,
        "per_task": per,
        "mean": (sum(ok) / len(ok)) if ok else None,
        "n_ok": len(ok),
        "n": len(per),
        "tokens": promptlib.count_tokens(text),
    }
    print(f"  MEAN {results[name]['mean']}  ({len(ok)}/{len(per)} scored)")
    # `_config` sits BESIDE the per-arm entries on disk, not inside the in-memory `results`
    # that the closing print loop iterates -- so it cannot be mistaken for a fourth arm there
    # (which would KeyError on `r["tokens"]`), and a later change to `Rung0Config`'s own
    # defaults cannot silently invalidate what this file already claims about `winner`.
    to_disk = {**results, "_config": {"budget_cap": cfg.budget_cap, "max_turns": cfg.max_turns}}
    json.dump(to_disk, open(OUT / "lenctl.json", "w"), indent=1)

print("\n================ RESULT ================")
for n, r in results.items():
    print(f"  {n:<8} {r['tokens']:>4} tok   mean {r['mean']}   ({r['n_ok']}/{r['n']})")
