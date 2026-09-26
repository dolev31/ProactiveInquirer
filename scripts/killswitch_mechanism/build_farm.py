# Repo root and scratch dir come from the environment: a committed absolute home path is
# how a research repo stops being reproducible for anyone but its author.

import json
import os
import pathlib

PI_REPO = os.environ.get("PI_REPO", os.getcwd()).rstrip("/")


ROOT = pathlib.Path(PI_REPO)
SC = pathlib.Path("/private/tmp/mech-ks-idl")
ctl = json.load(open(SC / "controls_disk.json"))
sel = [
    r for r in ctl if r["cv"].startswith("107ef22") and r["split"] == "test" and r["status"] == "ok"
]
ids_ctl = sorted({r["run_id"] for r in sel})
(SC / "ids.controls.txt").write_text("\n".join(ids_ctl) + "\n")
ks = [
    ln.strip()
    for ln in open(ROOT / "artifacts/killswitch/run_ids.killswitch.txt")
    if ln.strip() and not ln.startswith("#")
]
trt = [
    ln.strip()
    for ln in open(ROOT / "artifacts/killswitch/run_ids.treatment.txt")
    if ln.strip() and not ln.startswith("#")
]
prior = [ln.strip() for ln in open("/private/tmp/ks-scratch/ids.ref_prompted.txt") if ln.strip()]
dro = [ln.strip() for ln in open("/private/tmp/ks-scratch/ids.ref_drafter_only.txt") if ln.strip()]
allids = sorted(set(ids_ctl) | set(ks) | set(trt) | set(prior) | set(dro))
print(
    "controls",
    len(ids_ctl),
    "ks",
    len(ks),
    "treatment",
    len(trt),
    "prompted",
    len(prior),
    "drafter_only",
    len(dro),
    "TOTAL distinct",
    len(allids),
)
farm = SC / "farm"
farm.mkdir(exist_ok=True)
miss = []
for rid in allids:
    src = ROOT / "runs" / rid
    if not src.exists():
        miss.append(rid)
        continue
    ln = farm / rid
    if not ln.exists():
        os.symlink(src, ln)
print("linked", len(os.listdir(farm)), "missing", len(miss))
(SC / "ids.all.txt").write_text("\n".join(allids) + "\n")
