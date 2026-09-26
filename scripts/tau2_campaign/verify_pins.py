"""Refuse a launch whose RECORDED pins do not match what was REQUESTED.

WHY THIS EXISTS. The first phase-5 campaign shipped 408/408 units with zero attrition and wrong
frozen roles: `phase5_shard.sh` wrote `export PI_MODEL_DRAFTER=${PI_MODEL_DRAFTER:-claude-sonnet-5}`
AFTER sourcing `.env`, which had already set the variable, so the default never applied. Every
manifest recorded `pins.drafter.model_id == "openai/aws/gpt-oss-120b"`, and nothing caught it
before the bill arrived because nothing read a unit's own manifest back and compared it to what
the shard asked for. `load_env.sh` now protects the `PI_MODEL_*` precedence chain itself (see
`test_model_role_pin_precedence.py`), but that fixes ONE mechanism. This module is the
mechanism-independent backstop: it reads what a completed unit actually recorded and refuses to
let a campaign proceed if that does not match the request, regardless of why it drifted.

WHERE EACH ROLE LIVES, MEASURED ON A REAL MANIFEST (runs/a1e7ac0217f92b975849ba5779dc2b0b).
`inquirer`, `drafter`, `answerer` are built by `LitellmClient.pin(role)` / `collect_pins` and
recorded under `pins.<role>.model_id`. `user_sim` is tau2's own simulator, upstream of pinq's
client entirely, and is recorded as a raw string under `upstream_pins.user_sim` -- NOT under
`pins`. A comparator that looked for `pins.user_sim` would find nothing there and could read
that as "no mismatch", which is a false negative of the same shape as an empty search that goes
uninterrogated. `judge` is not checked here: no tau2 shard freezes it, and it is not present in
either `pins` or `upstream_pins` on a rollout manifest.

WHAT COUNTS AS A MISMATCH. Only roles present in the REQUESTED mapping are checked -- a role a
launcher chose not to freeze must not manufacture a finding against whatever the manifest
happens to record for it (non-vacuity: see `test_a_role_not_requested_is_not_checked`). A role
that IS requested but absent from the manifest (missing key, empty pins dict, schema drift) is
a mismatch, not a pass -- "nothing to compare" is not "nothing wrong".

CLI, for use as a preflight or a post-hoc audit:

    python scripts/tau2_campaign/verify_pins.py MANIFEST.json \\
        --inquirer qwen3-8b-dpo-stacked-notdone-both \\
        --drafter openai/aws/claude-sonnet-5 \\
        --answerer openai/aws/claude-sonnet-5 \\
        --user-sim openai/aws/gpt-oss-120b

Exits 0 and prints "pins ok" if every requested role matches; exits 1 and prints each mismatch
otherwise. This is what `phase5_launch.sh`'s preflight probe runs against the manifest a single
real unit produces before the full launch is allowed to proceed.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# inquirer/drafter/answerer come from LitellmClient.pin(role), under manifest["pins"][role].
# user_sim comes from tau2's own simulator and is recorded under manifest["upstream_pins"],
# never under "pins" -- see the module docstring. judge is never frozen by a tau2 shard and is
# checked nowhere.
PINNED_ROLES = ("inquirer", "drafter", "answerer")
UPSTREAM_ROLES = ("user_sim",)
ALL_ROLES = PINNED_ROLES + UPSTREAM_ROLES


@dataclass(frozen=True)
class PinMismatch:
    role: str
    requested: str
    recorded: str | None  # None means the role was absent from the manifest entirely


def _recorded(manifest: dict[str, Any], role: str) -> str | None:
    if role in PINNED_ROLES:
        pins = manifest.get("pins") or {}
        entry = pins.get(role)
        if not isinstance(entry, dict):
            return None
        model_id = entry.get("model_id")
        return model_id if isinstance(model_id, str) else None
    if role in UPSTREAM_ROLES:
        upstream = manifest.get("upstream_pins") or {}
        value = upstream.get(role)
        return value if isinstance(value, str) else None
    raise ValueError(f"unknown role {role!r}; expected one of {ALL_ROLES}")


def check_pins(manifest: dict[str, Any], requested: dict[str, str]) -> list[PinMismatch]:
    """Compare a manifest's RECORDED pins to a REQUESTED role->model_id mapping.

    Only roles present as keys in `requested` are checked (see module docstring on
    non-vacuity). A requested role absent from the manifest is a mismatch with
    `recorded=None`, never a silent pass.
    """
    unknown = set(requested) - set(ALL_ROLES)
    if unknown:
        raise ValueError(
            f"requested unknown role(s) {sorted(unknown)}; expected one of {ALL_ROLES}"
        )
    mismatches: list[PinMismatch] = []
    for role, want in requested.items():
        got = _recorded(manifest, role)
        if got != want:
            mismatches.append(PinMismatch(role=role, requested=want, recorded=got))
    return mismatches


def _main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("manifest", type=Path, help="path to a run's manifest.json")
    p.add_argument("--inquirer")
    p.add_argument("--drafter")
    p.add_argument("--answerer")
    p.add_argument("--user-sim")
    args = p.parse_args(argv)

    requested = {
        "inquirer": args.inquirer,
        "drafter": args.drafter,
        "answerer": args.answerer,
        "user_sim": args.user_sim,
    }
    requested = {k: v for k, v in requested.items() if v is not None}
    if not requested:
        print(
            "verify_pins: no role requested (pass at least one of "
            "--inquirer/--drafter/--answerer/--user-sim); refusing to certify nothing",
            file=sys.stderr,
        )
        return 2

    manifest = json.loads(args.manifest.read_text())
    mismatches = check_pins(manifest, requested)
    if mismatches:
        print(
            f"PIN MISMATCH in {args.manifest} (run_id={manifest.get('run_id', '?')}):",
            file=sys.stderr,
        )
        for m in mismatches:
            print(f"  {m.role}: requested={m.requested!r} recorded={m.recorded!r}", file=sys.stderr)
        return 1
    print(f"pins ok: {args.manifest} matches {requested}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
