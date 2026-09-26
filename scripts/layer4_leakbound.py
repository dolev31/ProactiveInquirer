"""An upper bound on VERBATIM gold-text echo on the five suites layer 4 cannot cover.

WHAT THIS CAN AND CANNOT SEE. There are two directions a gold string can travel. Into a PROMPT:
a cache record holds `request_sha` and never the request text (verified: record keys are
actor/http_status/model/provider/request_sha/response_sha/text/tok_*/usd), so that direction is
not recoverable for ANY suite and no instrument here bounds it. Out of a RESPONSE: `turns.jsonl`
persists `draft_text` and `response_text` in the run directory itself. This scans the second.

SPECIFICITY IS A PRECONDITION, NOT A DETAIL. A needle must be secret and long enough that a
verbatim match cannot arise by chance or by sanctioned reading. Needles that appear in the public
domain text the agent legitimately reads are EXCLUDED, because a match on them is not evidence of
a leak, and needles below a length threshold are excluded, because 'BOS' and '21.' match ordinary
text. Both exclusions are counted and reported: a bound whose needle set is empty is not a bound.

NON-VACUITY. A content detector that finds nothing is indistinguishable from one that cannot
fire. So for each suite the identical scan is re-run over the same text with one real in-scope
needle planted in it, and the bound is reported only where that plant is caught.
"""

import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

MINLEN = int(sys.argv[1]) if len(sys.argv) > 1 else 40
SUITES = ["drgym", "tau2", "tau2_airline", "tau2_retail", "tau2_telecom"]
PUB = Path("data/upstream/tau2-bench-data/tau2/domains")
DOM = {
    "tau2_airline": "airline",
    "tau2_retail": "retail",
    "tau2_telecom": "telecom",
    "tau2": "banking_knowledge",
}


def public_blob(suite):
    """Text the agent legitimately reads. For drgym the corpus holds only {id, question}: its
    documents live in a content-addressed search cache and these runs recorded no evidence at
    all, so there is no sanctioned document text to exclude against."""
    if suite == "drgym":
        p = Path("data/corpora/drgym/18fd59f66a9e24ec/tasks.jsonl")
        return p.read_text(errors="ignore") if p.exists() else ""
    d = PUB / DOM[suite]
    blob = []
    if d.is_dir():
        for f in sorted(d.rglob("*")):
            if f.is_file() and f.suffix in (".json", ".md", ".txt", ".toml", ".yaml"):
                try:
                    blob.append(f.read_text(errors="ignore"))
                except OSError:
                    pass
    return "\n".join(blob)


def scan(text, needles):
    return sorted({t for t in needles if t in text})


rows = pq.read_table(
    "scores/parquet/runs.parquet", columns=["run_id", "suite_id", "arm_id", "task_id"]
).to_pylist()

print(f"needle length threshold: {MINLEN} characters\n")
print(
    f"{'suite':<14} {'needles':>8} {'public':>7} {'short':>6} {'IN SCOPE':>9} "
    f"{'runs':>6} {'turns':>7} {'chars':>10} {'HITS':>5} {'plant':>6}"
)
summary = []
for s in SUITES:
    gold = {}
    for line in (Path("data/gold/graphs") / s / "v1.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        g = json.loads(line)
        gold[g["gold_task_key"]] = sorted(
            {
                str(n.get("gold_text") or "").strip()
                for n in g.get("gold_nodes", [])
                if str(n.get("gold_text") or "").strip()
            }
        )
    blob = public_blob(s)
    allned = sorted({t for v in gold.values() for t in v})
    n_pub = sum(1 for t in allned if t in blob)
    n_short = sum(1 for t in allned if len(t) < MINLEN)
    scope = {}
    for k, v in gold.items():
        scope[k] = [t for t in v if len(t) >= MINLEN and t not in blob]
    n_scope = len({t for v in scope.values() for t in v})

    sel = [r for r in rows if r["suite_id"] == s]
    n_runs = n_turns = n_chars = n_hits = 0
    planted = None
    for r in sel:
        d = Path("runs") / r["run_id"]
        p = d / "turns.jsonl"
        if not p.exists():
            continue
        try:
            lines = [x for x in p.read_text(errors="ignore").splitlines() if x.strip()]
        except OSError:
            continue
        nd = scope.get(r["task_id"], [])
        texts = []
        for x in lines:
            try:
                t = json.loads(x)
            except ValueError:
                continue
            texts.append(str(t.get("draft_text") or ""))
            texts.append(str(t.get("response_text") or ""))
        body = "\n".join(texts)
        n_runs += 1
        n_turns += len(lines)
        n_chars += len(body)
        if nd:
            n_hits += len(scan(body, nd))
            if planted is None:
                # THE SAME SCAN, SAME TEXT, ONE REAL IN-SCOPE NEEDLE PLANTED.
                planted = len(scan(body + "\n" + nd[0], nd)) > 0
    print(
        f"{s:<14} {len(allned):>8} {n_pub:>7} {n_short:>6} {n_scope:>9} "
        f"{n_runs:>6} {n_turns:>7} {n_chars:>10} {n_hits:>5} "
        f"{('FIRES' if planted else 'INERT') if planted is not None else 'n/a':>6}"
    )
    summary.append((s, n_scope, n_runs, n_chars, n_hits, planted))

print()
for s, n_scope, n_runs, n_chars, n_hits, planted in summary:
    if n_scope == 0:
        print(
            f"  {s}: NO BOUND. Every needle is public or below {MINLEN} chars, so nothing secret and specific remains to detect."
        )
    elif planted is not True:
        print(f"  {s}: NO BOUND. The plant did not fire, so a zero here is not a measurement.")
    else:
        print(
            f"  {s}: bound available. {n_scope} in-scope needles, {n_chars:,} chars of recorded "
            f"response text over {n_runs} runs, {n_hits} verbatim hits, plant FIRES."
        )
