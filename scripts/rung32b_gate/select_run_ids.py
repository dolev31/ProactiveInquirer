"""Pick THIS lane's 1,395 run ids out of a shared runs/ that peers also write to.

runs/ grew by 136 dirs in the first three minutes of this campaign and the FIRST of them was
a peer's (`grid_name=tier1_trained_qa_base`, `code_version=3ae099d0`). Suite+seed is not a
selector here; neither is recency. Three predicates together are:

  grid_name          one of this lane's four grids
  code_version       993a38a  -- the pinned worktree this campaign ran from
  inquirer model_id  qwen3-32b-base (comparator) or qwen3-32b-sft-headline (checkpoint)

The candidate set is the snapshot diff (dirs that did not exist before launch), which is
complete because every unit here is a NEW model pin and therefore a new run_id -- nothing
could have resumed. The count is ASSERTED against the plan, not eyeballed.
"""

import json
import sys
from pathlib import Path

MAIN = Path.home() / "PycharmProjects/ProactiveInquirer"
A = MAIN / "artifacts/rung32b_gate_20260919"
CODE = "993a38a59d31efe47a2fe50f88c008cd204329c0"
GRIDS = {
    "dev_baseline_musique",
    "dev_baseline_strategyqa",
    "dev_select_musique",
    "dev_select_strategyqa",
}
MODELS = {"qwen3-32b-base", "qwen3-32b-sft-headline"}
EXPECT = {
    ("dev_baseline_musique", "qwen3-32b-base"): 264,
    ("dev_baseline_strategyqa", "qwen3-32b-base"): 666,
    ("dev_select_musique", "qwen3-32b-sft-headline"): 132,
    ("dev_select_strategyqa", "qwen3-32b-sft-headline"): 333,
}

before = set((A / "runs_snapshot_before.txt").read_text().split())
new = [d for d in (x.name for x in (MAIN / "runs").iterdir()) if d not in before]
print(f"candidate new run dirs: {len(new)}")

rows, other, bad = [], {}, []
for rid in new:
    mp = MAIN / "runs" / rid / "manifest.json"
    if not mp.exists():
        continue
    try:
        m = json.loads(mp.read_text())
    except Exception as e:
        bad.append((rid, repr(e)))
        continue
    g = m.get("grid_name")
    # THE KEY IS `pins`, NOT `model_pins`. The first draft of this file guessed
    # `model_pins` and every one of my own runs read model_id=None -- a plausible
    # ZERO ("0 of 1395 mine") rather than an error. Verified against a real manifest
    # before use: manifest["pins"]["inquirer"]["model_id"].
    inq = (m.get("pins") or {}).get("inquirer") or {}
    mid = inq.get("model_id")
    cv = m.get("code_version")
    if g in GRIDS and cv == CODE and mid in MODELS:
        st = (
            json.loads((MAIN / "runs" / rid / "status.json").read_text())
            if (MAIN / "runs" / rid / "status.json").exists()
            else {}
        )
        pins = m.get("pins") or {}
        rows.append(
            (
                rid,
                g,
                mid,
                m.get("suite_id"),
                m.get("seed"),
                m.get("arm_id"),
                bool(m.get("dirty")),
                m.get("split"),
                st.get("status"),
                float(st.get("usd_billed") or 0.0),
                inq.get("base_url_sha"),
                inq.get("adapter_sha"),
                (pins.get("drafter") or {}).get("base_url_sha"),
            )
        )
    else:
        other[(g, cv[:7] if cv else None, mid)] = other.get((g, cv[:7] if cv else None, mid), 0) + 1

print(f"unreadable manifests: {len(bad)}  {bad[:3]}")
print("\nNOT MINE (grid, code, inquirer) -> count:")
for k, v in sorted(other.items(), key=lambda kv: -kv[1])[:12]:
    print("   ", k, v)

cnt, billed, dirty_n, nonok = {}, 0.0, 0, 0
for r in rows:
    cnt[(r[1], r[2])] = cnt.get((r[1], r[2]), 0) + 1
    billed += r[9]
    dirty_n += r[6]
    nonok += r[8] != "ok"
print("\nMINE (grid, inquirer) -> count [expected]:")
ok = True
for k, exp in EXPECT.items():
    got = cnt.get(k, 0)
    flag = "OK" if got == exp else "MISMATCH"
    ok &= got == exp
    print(f"    {k[0]:26s} {k[1]:24s} {got:5d} [{exp}] {flag}")
print(f"\ntotal mine      : {len(rows)} (expected 1395)")
print(f"dirty runs      : {dirty_n} (must be 0)")
print(f"non-ok status   : {nonok} (must be 0)")
print(f"usd_billed sum  : ${billed:.4f}")

# base_url_sha IS INSIDE RUN IDENTITY. If the two arms (or the Inquirer and the
# frozen Drafter) reached different proxies, the contrast is a base-url contrast
# too. Asserted across every role of every run, not assumed from one manifest.
urls = {r[10] for r in rows} | {r[12] for r in rows}
print(f"distinct base_url_sha across arms AND roles: {len(urls)} (must be 1) {sorted(urls)}")
ad = {(r[2], r[11]) for r in rows}
print("inquirer (model_id, adapter_sha):")
for a_ in sorted(ad, key=str):
    print("   ", a_)
ok &= len(urls) == 1

(A / "run_ids.txt").write_text("".join(r[0] + "\n" for r in sorted(rows)))
with (A / "run_ids.tsv").open("w") as f:
    f.write(
        "run_id\tgrid_name\tinquirer_model\tsuite\tseed\tarm\tdirty\tsplit\tstatus\tusd_billed\n"
    )
    for r in sorted(rows):
        f.write("\t".join(str(x) for x in r) + "\n")
print(f"\nwrote {A / 'run_ids.txt'} and run_ids.tsv")
sys.exit(0 if (ok and len(rows) == 1395 and dirty_n == 0 and nonok == 0) else 1)
