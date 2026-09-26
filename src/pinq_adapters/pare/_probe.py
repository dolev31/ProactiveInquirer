"""Is PARE importable here, and which scenarios are in the benchmark split?

WHY THE PROBE MATTERS MORE FOR PARE THAN FOR tau2
    PARE sits on top of Meta-ARE (`are.simulation`), so importing `pare.scenario_runner`
    drags in a simulation engine, an app registry and a notification system. That is a hard
    dependency the default offline test run must never touch, so every PARE import in this
    package lives inside a function body.
"""

from __future__ import annotations

from pathlib import Path

# 143 scenarios across 9 FSM apps. Asserted, never assumed.
N_SCENARIOS = 143
SPLIT = "full"

APPS = (
    "apartment",
    "cab",
    "calendar",
    "contacts",
    "email",
    "messaging",
    "note",
    "reminder",
    "shopping",
)


def available() -> tuple[bool, str]:
    """(importable, reason). The reason is surfaced in skip messages, so be specific."""
    try:
        import pare  # noqa: F401
    except ImportError as exc:
        return False, f"pare not installed ({exc}); install with: uv pip install -e '.[pare]'"
    try:
        import are.simulation  # noqa: F401
    except ImportError as exc:
        return False, f"pare installed but Meta-ARE (are.simulation) is missing ({exc})"
    try:
        d = splits_dir()
    except Exception as exc:
        return False, f"pare installed but its splits dir cannot be resolved ({exc})"
    if not (d / f"{SPLIT}.txt").is_file():
        # Same trap as tau2: the wheel ships code only, and `data/` lives at the repo root
        # outside the package, so a pip install leaves the splits path pointing at nothing.
        return False, (
            f"pare is installed but its data is not: {d / f'{SPLIT}.txt'} does not exist. "
            "The wheel ships no data/ directory. Clone deepakn97/pare and export "
            "PARE_BENCHMARK_SPLITS_DIR=<checkout>/data/splits."
        )
    return True, "pare available"


def splits_dir() -> Path:
    from pare.benchmark.scenario_loader import get_splits_dir

    return Path(get_splits_dir())


def load_scenario_ids(root: Path | None = None, split: str = SPLIT) -> tuple[str, ...]:
    """Scenario ids from `<splits>/<split>.txt`, one per line.

    Parsed here rather than through PARE's loader so the id list — and therefore the task
    list, the manifest and the run ids — can be produced with PARE absent. Blank lines and
    '#' comments are skipped so the file stays human-editable.
    """
    d = Path(root) if root is not None else splits_dir()
    path = d / f"{split}.txt"
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(line)
    return tuple(out)


def app_of(scenario_id: str) -> str:
    """Best-effort app label from the scenario id, for stratified reporting.

    A label only: PARE scenarios routinely span several apps, so this is never used to
    decide what a scenario may touch — only to group rows in a table.
    """
    for app in APPS:
        if scenario_id.endswith(f"_{app}") or scenario_id.startswith(f"{app}_"):
            return app
    for app in APPS:
        if app in scenario_id:
            return app
    return "unknown"
