"""Read-only access to run directories under a shared `runs/` root.

WHY THIS EXISTS SEPARATELY FROM `scripts/report_forks.py`'s `load_runs`. That loader reads
exactly the six fields `pi_eval.fork_report` needs for the paired-fork contrast. This lane
needs the full per-turn ask text (`turns.jsonl`) and the executed tool-call log
(`outcome.json["env_calls"]`) as well, for the ask-target and outcome classifiers, so it
carries its own loader rather than widening that one's contract for a use case it was never
meant to serve (`tau2-driver-is-a-copy-of-run-unit` is exactly the failure mode to avoid: THIS
module is the one and only reader, and every population in `populations.py` calls into it
rather than re-opening manifests itself).

Every function here is read-only: it never writes into `runs_root`, and callers are expected
to point it at the shared checkout's `runs/` by absolute path (see `populations.py`).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


@dataclass(frozen=True, slots=True)
class RunRecord:
    """One run directory, lazily exposing the files this lane reads.

    `manifest` and `status` are always loaded (cheap, needed for every filter). `turns` and
    `outcome` are loaded on first access and cached, because a taxonomy pass over ~1,000 runs
    that only needs `status.json` for most of them should not pay for every `turns.jsonl` too.
    """

    run_id: str
    path: Path
    manifest: Mapping[str, Any]
    status: Mapping[str, Any]
    _turns: list[dict] | None = field(default=None, compare=False)
    _outcome: dict | None = field(default=None, compare=False)

    @property
    def turns(self) -> list[dict]:
        if self._turns is None:
            object.__setattr__(self, "_turns", _read_jsonl(self.path / "turns.jsonl"))
        return self._turns  # type: ignore[return-value]

    @property
    def outcome(self) -> dict:
        if self._outcome is None:
            object.__setattr__(self, "_outcome", _read_json(self.path / "outcome.json"))
        return self._outcome  # type: ignore[return-value]

    # -- convenience accessors onto the fields this lane reads over and over --

    @property
    def suite_id(self) -> str:
        return str(self.manifest.get("suite_id") or "")

    @property
    def arm_id(self) -> str:
        return str(self.manifest.get("arm_id") or "")

    @property
    def task_id(self) -> str:
        return str(self.manifest.get("task_id") or "")

    @property
    def seed(self) -> Any:
        return self.manifest.get("seed")

    @property
    def code_version(self) -> str:
        return str(self.manifest.get("code_version") or "")

    @property
    def inquirer_model(self) -> str:
        return str(((self.manifest.get("pins") or {}).get("inquirer") or {}).get("model_id") or "")

    @property
    def prompt_variant_id(self) -> str:
        return str(self.manifest.get("prompt_variant_id") or "")

    @property
    def is_fork(self) -> bool:
        return bool(self.manifest.get("foreign_trace_sha"))

    @property
    def run_status(self) -> str | None:
        return self.status.get("status")

    @property
    def ok(self) -> bool:
        return self.run_status in (None, "ok")


def scan_manifests(
    runs_root: str | os.PathLike[str],
    *,
    suite_ids: Iterable[str] | None = None,
    code_version_prefixes: Iterable[str] | None = None,
) -> Iterator[tuple[Path, dict]]:
    """Yield (run_dir, manifest_dict) for every run under `runs_root` matching the filters.

    A full `os.scandir` walk, not `Path.glob("*/manifest.json")`: `runs/` holds on the order of
    10^5 entries in this repository and `scandir` avoids `glob`'s extra `stat` per match. Both
    filters are cheap pre-checks against the parsed manifest so a caller can narrow a ~170k-dir
    root to one campaign without loading `status.json`/`turns.jsonl` for the other 99%.
    """
    suite_set = set(suite_ids) if suite_ids is not None else None
    prefixes = tuple(code_version_prefixes) if code_version_prefixes is not None else None
    root = Path(runs_root)
    with os.scandir(root) as it:
        for entry in it:
            if not entry.is_dir(follow_symlinks=False):
                continue
            mpath = Path(entry.path) / "manifest.json"
            try:
                with open(mpath, "r") as fh:
                    m = json.load(fh)
            except (FileNotFoundError, NotADirectoryError, json.JSONDecodeError):
                continue
            if suite_set is not None and m.get("suite_id") not in suite_set:
                continue
            if prefixes is not None:
                cv = str(m.get("code_version") or "")
                if not any(cv.startswith(p) for p in prefixes):
                    continue
            yield Path(entry.path), m


def load_run(run_dir: Path, manifest: dict | None = None) -> RunRecord:
    m = manifest if manifest is not None else _read_json(run_dir / "manifest.json")
    s = _read_json(run_dir / "status.json")
    return RunRecord(
        run_id=str(m.get("run_id") or run_dir.name), path=run_dir, manifest=m, status=s
    )


def load_runs_by_id(runs_root: str | os.PathLike[str], run_ids: Sequence[str]) -> list[RunRecord]:
    """Load specific run ids directly by directory name (no scan). For populations that
    already carry a curated run-id list, e.g. `artifacts/banking/run_ids.*.txt`."""
    root = Path(runs_root)
    out = []
    for rid in run_ids:
        d = root / rid
        if (d / "manifest.json").is_file():
            out.append(load_run(d))
    return out


def read_run_id_list(path: str | os.PathLike[str]) -> list[str]:
    """One run id per non-comment, non-blank line; the first field of a tab-separated row."""
    out = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.append(line.split("\t")[0])
    return out
