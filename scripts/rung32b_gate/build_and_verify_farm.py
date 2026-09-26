"""Build the isolated symlink farm, compact into a PRIVATE parquet, and prove the rows survived.

WHY ISOLATED. `pi score` into the shared scores/parquet re-stamps scorer_hash over the whole
gold set, and the shared store mixes code_versions under one grid_name. This lane's parquet
holds exactly its own 1,395 runs -- 930 comparator + 465 checkpoint -- which is also the
"one parquet per checkpoint" rule artifacts/gate/README.md states: `pi train gate` has no
--checkpoint-model-id, so two checkpoints in one store are silently pooled into one arm.

THE SILENT ZERO THIS GUARDS. build_farm.build_farm() resolves source_runs_root, because a
RELATIVE root makes every symlink point at <farm>/runs/runs/<id> and `pi compact` then reports
run_dirs: 0 -- not an error. The root passed here is absolute, and the links are PROVEN to
resolve before compaction rather than after.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(
    0, str(Path.home() / "PycharmProjects/ProactiveInquirer/scripts/decomposition_test")
)
from build_farm import build_farm, read_run_id_list  # noqa: E402

MAIN = Path.home() / "PycharmProjects/ProactiveInquirer"
A = MAIN / "artifacts/rung32b_gate_20260919"
FARM = A / "farm"
RUNS = FARM / "runs"
PARQ = FARM / "scores_parquet"

ids = read_run_id_list(A / "run_ids.txt")
print(f"run ids from run_ids.txt: {len(ids)}")
res = build_farm(RUNS, ids, source_runs_root=MAIN / "runs")  # ABSOLUTE, not Path("runs")
print(
    f"linked={len(res['linked'])} already={len(res['already_present'])} missing={len(res['missing'])}"
)
assert not res["missing"], f"MISSING run dirs: {res['missing'][:5]}"

# PROVE THE LINKS RESOLVE. A dangling symlink is the failure mode that reads as run_dirs: 0.
dangling = [p.name for p in RUNS.iterdir() if not (p / "manifest.json").exists()]
print(
    f"farm entries: {len(list(RUNS.iterdir()))}  dangling (no manifest.json through the link): {len(dangling)}"
)
assert not dangling, f"DANGLING: {dangling[:5]}"
sample = next(RUNS.iterdir())
print(
    f"sample link: {sample} -> {os.readlink(sample)}  (absolute: {os.path.isabs(os.readlink(sample))})"
)
assert os.path.isabs(os.readlink(sample)), "relative symlink target -- the silent-zero bug"

# ------------------------------------------------------------------ compact
cmd = [
    str(MAIN / ".venv/bin/pi"),
    "compact",
    "--runs-root",
    str(RUNS),
    "--out",
    str(PARQ),
    "--exclude-dev",
]
print("\n$ " + " ".join(cmd))
r = subprocess.run(cmd, capture_output=True, text=True)
print(r.stdout[-4000:])
print(r.stderr[-2000:], file=sys.stderr)
assert r.returncode == 0, f"pi compact exit {r.returncode}"


# ------------------------------------------------------------------ the row-survival check
# Compaction has dropped turn/evidence/call rows silently before (21,001 runs read as
# n_turns 0 with turns.jsonl intact). So the parquet is compared against the RUN DIRS, per run.
def jsonl_count(p: Path) -> int:
    if not p.exists():
        return 0
    return sum(1 for line in p.open() if line.strip())


def disk_counts(run_ids):
    out = {}
    for rid in run_ids:
        d = RUNS / rid
        out[rid] = (
            jsonl_count(d / "turns.jsonl"),
            jsonl_count(d / "evidence.jsonl"),
            jsonl_count(d / "calls.jsonl"),
        )
    return out


def parquet_counts(run_ids):
    import duckdb

    con = duckdb.connect()
    got = {rid: [0, 0, 0] for rid in run_ids}
    for i, tbl in enumerate(("turns", "evidence", "calls")):
        f = PARQ / f"{tbl}.parquet"
        if not f.exists():
            print(f"  NOTE: {f.name} absent")
            continue
        for rid, n in con.execute(f"SELECT run_id, count(*) FROM '{f}' GROUP BY 1").fetchall():
            if rid in got:
                got[rid][i] = n
    return {k: tuple(v) for k, v in got.items()}


def compare(disk, parq, label):
    bad = [(r, disk[r], parq[r]) for r in disk if disk[r] != parq[r]]
    print(f"  {label}: {len(disk)} runs, {len(bad)} disagreements")
    for b in bad[:5]:
        print("     ", b)
    return bad


print("\n=== ROW SURVIVAL: run dirs vs the private parquet ===")
disk = disk_counts(ids)
parq = parquet_counts(ids)
bad = compare(disk, parq, "as compacted")
tot = tuple(sum(x[i] for x in disk.values()) for i in range(3))
print(f"  totals on disk (turns, evidence, calls): {tot}")
print(f"  runs with 0 turns on disk: {sum(1 for v in disk.values() if v[0] == 0)}")

# NON-VACUITY. A check that cannot fail is not a check: force ONE disagreement and require
# that the comparator reports exactly it.
print("\n=== NON-VACUITY: force one disagreement ===")
victim = sorted(disk)[0]
poisoned = dict(disk)
poisoned[victim] = (disk[victim][0] + 1, disk[victim][1], disk[victim][2])
bad2 = compare(poisoned, parq, "with one count perturbed")
assert len(bad2) == len(bad) + 1 and any(b[0] == victim for b in bad2), (
    "the row-survival check did NOT notice a forced disagreement -- it is vacuous"
)
print(f"  the check caught the forced disagreement on {victim}: NON-VACUOUS")

json.dump(
    {
        "n_run_ids": len(ids),
        "linked": len(res["linked"]),
        "missing": res["missing"],
        "dangling": dangling,
        "disagreements": len(bad),
        "disk_totals_turns_evidence_calls": tot,
        "nonvacuity_forced_caught": True,
    },
    (A / "farm_verification.json").open("w"),
    indent=2,
)
print(f"\nwrote {A / 'farm_verification.json'}")
sys.exit(1 if bad else 0)
