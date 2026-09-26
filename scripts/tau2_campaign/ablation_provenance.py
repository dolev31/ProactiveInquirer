"""The instruction ablation, with the population stated and the run-id sets digested.

THE POPULATION, exactly: every run directory under runs/ whose manifest carries a non-empty
`foreign_trace_sha` (a fork), whose `suite_id` is the suite named, whose `arm_id` is
`inquirer_prompted`, whose Inquirer pin `model_id` is `openai/aws/claude-sonnet-5`, and whose
`status.json` says ok. The ARM and the QUESTIONER'S MODEL are therefore held fixed and the only
thing that varies is which Inquirer template the run rendered, read from `prompt_hashes`.

WHY THE MODEL MUST BE HELD FIXED: pooled over pins the same contrast reads 0.873 vs 0.126 on
retail, which is a mix of the template effect and a model effect, and reports the mix.

The denominator is ASKS, not runs: every `ask` turn in `turns.jsonl`, with the numerator the
ones carrying a non-empty `retrieved_uids`.
"""

import collections
import hashlib
import json
import sys
from pathlib import Path

rows = json.loads(Path(sys.argv[1]).read_text())
G = {"inquirer_prompted:814658ea": "pre-3052546", "inquirer_prompted:3006f5bf": "post-3052546"}

agg = collections.defaultdict(lambda: {"runs": 0, "asks": 0, "ans": 0, "ids": []})
for r in rows:
    if not r["foreign_trace_sha"] or r["status"] != "ok" or r["arm_id"] != "inquirer_prompted":
        continue
    pin = str(((r.get("pins") or {}).get("inquirer") or {}).get("model_id", ""))
    if pin != "openai/aws/claude-sonnet-5":
        continue
    tpl = ",".join(f"{k}:{v[:8]}" for k, v in sorted(r["inq_templates"].items()))
    if tpl not in G:
        continue
    d = agg[(r["suite_id"], G[tpl])]
    d["runs"] += 1
    d["asks"] += r["asks"]
    d["ans"] += r["answered"]
    d["ids"].append(str(r["run_id"]))

print(
    f"{'suite':<14}{'template':<14}{'runs':>6}{'asks':>8}{'answered':>10}{'rate':>8}  run_ids_sha"
)
for k in sorted(agg):
    d = agg[k]
    rate = d["ans"] / d["asks"] if d["asks"] else float("nan")
    dg = hashlib.sha256("\n".join(sorted(d["ids"])).encode()).hexdigest()[:16]
    print(f"{k[0]:<14}{k[1]:<14}{d['runs']:>6}{d['asks']:>8}{d['ans']:>10}{rate:>8.4f}  {dg}")
for suite in ("tau2_retail", "tau2_airline"):
    a, b = agg.get((suite, "pre-3052546")), agg.get((suite, "post-3052546"))
    if a and b:
        print(f"{suite}: delta = {b['ans'] / b['asks'] - a['ans'] / a['asks']:+.4f}")
