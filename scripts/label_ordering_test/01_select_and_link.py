"""Step 1 of 4: the run selection, as a symlink runs-root.

Run from the repo root. `RUNS_ROOT` and `WORK` come from the environment so this file carries no
absolute path; `WORK` should be a PRIVATE scratch directory (a shared one has destroyed an
export here before).

WHY A SYMLINK ROOT AND NOT `--exclude-dev`: `pi compact` has no run filter, and compacting all
141k runs to reach 4,800 of them would also merge every unrelated arm into the store the gate
then reads. The selection IS the artefact, so it is written out as four run-id lists beside it.
"""

import json
import os
import pathlib

REPO = pathlib.Path(os.environ.get("PI_REPO", "."))
RUNS = pathlib.Path(os.environ.get("PI_RUNS_ROOT", REPO / "runs"))
WORK = pathlib.Path(os.environ["PI_WORK"])

CODE_VERSION = "107ef2221a55229ac7ba60a54ebfc9849971bde0"
GRID = "tier1_trained_qa_base"
SUITES = {"musique", "strategyqa", "wiki2"}
MODELS = {
    "qwen3-8b-base",
    "qwen3-8b-dpo-headline-control",
    "qwen3-8b-dpo-headline-rater",
    "qwen3-8b-dpo-headline-reaches",
}

dest = WORK / "runs_iso"
dest.mkdir(parents=True, exist_ok=True)
lists: dict[str, list[str]] = {}
for p in RUNS.glob("*/manifest.json"):
    try:
        m = json.loads(p.read_text())
    except Exception:
        continue
    pin = (m.get("pins") or {}).get("inquirer") or {}
    model = pin.get("model_id")
    if model not in MODELS:
        continue
    if m.get("code_version") != CODE_VERSION or m.get("grid_name") != GRID:
        continue
    if m.get("split") != "test" or m.get("budget_cap") != 8:
        continue
    if m.get("suite_id") not in SUITES or m.get("dirty") or m.get("gold_exposed"):
        continue
    sp = p.parent / "status.json"
    if not sp.exists() or json.loads(sp.read_text()).get("status") != "ok":
        continue
    rid = m["run_id"]
    link = dest / rid
    if not link.exists():
        os.symlink(p.parent.resolve(), link)
    lists.setdefault(model, []).append(rid)

out = WORK / "run_ids"
out.mkdir(exist_ok=True)
for model, ids in sorted(lists.items()):
    (out / f"{model}.txt").write_text("\n".join(sorted(ids)) + "\n")
    print(f"{model:32s} {len(ids)}")
print("total", sum(len(v) for v in lists.values()))
