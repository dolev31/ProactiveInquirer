"""Dump a tau2 runs root to a JSON array of run records, for `fork_paired.py`.

Pure IO, no analysis: it exists so the analysis runs against the repo's own tested pairing
(`pi_eval.fork_report`) on a machine that has it, while the runs live on a cluster whose pinned
worktree is an older commit. Copying the analysis to the data would mean a hand copy of the
pairing, which is the failure this directory already made once.

`status.json` is already the record `_pair_forks` expects -- it carries `foreign_trace_sha`,
`foreign_prefix_k`, `seed`, `arm_id`, `task_id`, `status` and the endpoints. The one field added
here is `_env_calls_from_outcome`: `len(outcome.json["env_calls"])`, a SECOND ROUTE to
`status.json["n_env_calls"]`. `fork_paired.py` refuses to report if the two disagree. Keeping that
comparison permanently rather than as a one-off is the check that catches this class of defect: a
keying bug was found on 2026-09-22 only because two counts of one population disagreed, and no
check had failed.

`_manifest` is the run's own `manifest.json`, because the provenance a reader must check before it
reads -- the questioner's model pin, each role's `base_url_sha`, `adapter_sha`, `budget_cap` -- is
on the manifest and not on the status. `transfer_rerun.py` refuses a record without it.
`_env_requestor_counts` counts the recorded env calls by `requestor` (which role made the call).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _read_json(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _ask_uids(path: Path) -> list[list[str]] | None:
    if not path.is_file():
        return None
    out = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("action_kind") == "ask":
            out.append([str(u) for u in row.get("retrieved_uids") or []])
    return out


def _ask_questions(path: Path) -> list[str] | None:
    """Per ask, its question text, in TURN ORDER (turn_idx, then file order): the `question` of
    each `action_kind == "ask"` row of turns.jsonl. None when turns.jsonl is absent or an ask row
    carries no question -- never a guessed "". (RULES amendment 8's exploratory repeat rates read
    the order: a repeat is of an EARLIER ask.)"""
    if not path.is_file():
        return None
    rows = []
    for pos, line in enumerate(path.read_text().splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("action_kind") != "ask":
            continue
        if row.get("question") is None:
            return None
        ti = row.get("turn_idx")
        rows.append((ti if isinstance(ti, int) else pos, pos, str(row["question"])))
    return [q for _, _, q in sorted(rows)]


def _inquirer_calls(path: Path) -> list[list] | None:
    """The questioner role's LLM calls from calls.jsonl, in order: [model, tok_completion,
    http_status] per `actor == "inquirer"` row (tok_completion is the provider's completion_tokens,
    reasoning included). None when calls.jsonl is absent -- never []."""
    if not path.is_file():
        return None
    out = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("actor") == "inquirer":
            out.append([row.get("model"), row.get("tok_completion"), row.get("http_status")])
    return out


def status_files(runs_root: str | Path) -> list[Path]:
    """Every status.json under `runs_root`, FOLLOWING SYMLINKED DIRECTORIES.

    An isolated store is a symlink farm: one symlink per run directory. Python 3.12's
    `Path.rglob` does not descend a symlinked directory, so `rglob("status.json")` over a farm
    returned [] and a farm read as an EMPTY STORE (measured 2026-09-23, 3.12.9; see
    tests/test_transfer_rerun.py::test_extract_reads_a_symlink_farm).
    """
    found = []
    for dirpath, _dirs, files in os.walk(runs_root, followlinks=True):
        if "status.json" in files:
            found.append(Path(dirpath) / "status.json")
    return sorted(found)


def extract(runs_root: str | Path) -> list[dict]:
    runs_root = Path(runs_root)
    out = []
    for st in status_files(runs_root):
        if ".preflight_" in str(st):
            continue
        s = _read_json(st)
        if not isinstance(s, dict):
            continue
        ec = None
        tools: dict[str, int] = {}
        requestors: dict[str, int] = {}
        # ORDERED (tool_name, ok): a reader recounts the gate's charge from index
        # n_prefix_env_calls on, so order and the ok flag are data, not detail. `ok` is copied as
        # recorded -- a missing flag stays None, never a default.
        seq: list[list] = []
        v = None
        try:
            v = json.loads((st.parent / "outcome.json").read_text()).get("env_calls")
            ec = len(v) if isinstance(v, list) else (v if isinstance(v, (int, float)) else None)
            # Raw per-tool counts, unclassified. Classification into READ / WRITE / GENERIC is
            # ANALYSIS and lives in fork_paired.py against tau2's own declared tool types -- not
            # the record's `mutating` field, which is True on every tau2 call because the
            # actuator's probe falls back to True when the environment cannot answer.
            for c in v if isinstance(v, list) else []:
                name = str((c or {}).get("tool_name") or "")
                tools[name] = tools.get(name, 0) + 1
                who = str((c or {}).get("requestor") or "")
                requestors[who] = requestors.get(who, 0) + 1
                seq.append([name, (c or {}).get("ok")])
        except Exception:
            pass
        s["_env_calls_from_outcome"] = ec
        s["_env_tool_counts"] = tools
        s["_env_requestor_counts"] = requestors
        s["_env_call_seq"] = seq if isinstance(v, list) else None
        s["_manifest"] = _read_json(st.parent / "manifest.json")
        # PER ASK, the record uids the harness recorded it retrieving (turns.jsonl `retrieved_uids`
        # on `action_kind == "ask"` rows), in order. None when turns.jsonl is absent -- never [].
        s["_ask_retrieved_uids"] = _ask_uids(st.parent / "turns.jsonl")
        s["_ask_questions"] = _ask_questions(st.parent / "turns.jsonl")
        # RULES amendment 11: the questioner's calls, for the share at its token cap.
        s["_inquirer_calls"] = _inquirer_calls(st.parent / "calls.jsonl")
        # WHERE THE RUN SAT, relative to the root: under the launcher's layout this is
        # `<suite>/<pin>/<run_id>`, and a reader checks the manifest's pin against that sub-root.
        s["_relpath"] = st.parent.relative_to(runs_root).as_posix()
        out.append(s)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs-root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    out = extract(a.runs_root)
    Path(a.out).write_text(json.dumps(out))
    print(f"wrote {a.out}: {len(out)} records from {a.runs_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
