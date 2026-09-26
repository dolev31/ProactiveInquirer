"""Enumerate, shard and run one tau2-bench campaign under the BENCHMARK's own protocol.

WHY A SCRIPT AND NOT `pi run`. A grid names (suite, arm, seed); a leaderboard cell names
(domain, arm, ROLE PINS, user simulator, protocol). Two cells here share `arm_id` and differ
only in which model the Inquirer is -- `inquirer_prompted` at the 8B base and at the frontier
questioner -- and `pi run` has no way to name that. The specs are built here and handed to the
same `run_tau2_unit` every other tau2 run goes through, so a unit written by this script is
byte-identical in shape to any other: same seven files, same status dict, same resume sentinel.

TASK ORDER IS A SEEDED SHUFFLE, AND THAT IS THE DESIGN. A campaign of this size may be stopped
before it finishes, and a prefix of `tasks.json` is a biased subset of the benchmark (upstream
groups related scenarios together). Shuffled with a fixed seed, ANY prefix is a uniform random
subsample of the domain, so a truncated campaign still reports an unbiased estimate over the
tasks it reached -- and it reports them with all four trials and all five arms, because the
unit order is (task, trial, arm) rather than (arm, task, trial).

SHARDS ARE BY TASK, NOT BY UNIT. A task's four trials and five arms land in the same shard, so
a shard that dies leaves whole tasks missing rather than half of every task's cells -- and
`pass^k` needs all k trials of a task or the task contributes nothing.

RESUME IS BY EXISTENCE. `run_tau2_unit` returns `status: resumed` when a completed `status.json`
already sits in the run directory, so re-running this script costs nothing for the units that
finished and re-rolls only the ones that did not.

NOTHING HERE READS GOLD. `PI_GOLD_ROOT` is unset in every worker by `assert_firewall`, which is
layer 3 of the firewall working rather than a bug.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))


# --------------------------------------------------------------------------- the cells

# ONE ROW PER TABLE COLUMN. `arm_id` alone does not identify a cell: two rows here share
# `inquirer_prompted` and differ only in the Inquirer's model, which is the whole point of the
# prompted-versus-trained contrast. `cell_id` is what the artifact's run-id lists are keyed on
# and what `--cell` selects; it never reaches a manifest, because the manifest already carries
# the pins that make the cell distinct (`model_pin_hash`) plus `upstream_pins`.
#
# THE DRAFTER AND THE ANSWERER ARE THE SAME MODEL IN EVERY ROW. That is what makes the table a
# comparison of questioners rather than of answer models, and it is the pin the published fork
# campaign used.
CELLS: tuple[dict[str, Any], ...] = (
    {
        "cell_id": "stock",
        "arm_id": "tau2_stock",
        "inquirer": "",  # upstream's agent has no questioner
        "label": "stock tau2 agent (upstream LLMAgent)",
    },
    {
        "cell_id": "prompted_8b_base",
        "arm_id": "inquirer_prompted",
        "inquirer": "qwen3-8b-base",
        "label": "Inquirer, prompted, 8B base",
    },
    {
        "cell_id": "trained_8b_dpo",
        "arm_id": "inquirer_trained",
        "inquirer": "qwen3-8b-dpo-stacked-notdone-both",
        "label": "Inquirer, trained, 8B DPO stacked",
    },
    {
        "cell_id": "may_ask_user_8b_dpo",
        "arm_id": "inquirer_may_ask_user",
        "inquirer": "qwen3-8b-dpo-stacked-notdone-both",
        "label": "Inquirer, trained, user channel open",
    },
    {
        "cell_id": "prompted_frontier",
        "arm_id": "inquirer_prompted",
        "inquirer": "openai/aws/claude-sonnet-5",
        "label": "Inquirer, prompted, frontier questioner (the published one)",
    },
)

CELLS_BY_ID = {c["cell_id"]: c for c in CELLS}

# The bridge cell: one arm re-run with the OLD user simulator, so this campaign and the
# recorded fork campaign share a customer on at least one column. Everything else about it is
# identical to `prompted_frontier`, so the simulator is the only thing that varies.
BRIDGE_CELL = "prompted_frontier"
BRIDGE_USERSIM = "openai/aws/gpt-oss-120b"

DOMAINS = {"retail": "tau2_retail", "airline": "tau2_airline"}

# The seed the task shuffle is drawn with. Fixed, printed, and recorded in the plan file: a
# shuffle whose seed is not written down makes a truncated campaign's population unrecoverable.
SHUFFLE_SEED = 20260918


def task_order(domain: str, data_dir: str) -> list[str]:
    """Every task id of a domain, in the campaign's seeded random order."""
    path = Path(data_dir) / "tau2" / "domains" / domain_dir(domain) / "tasks.json"
    ids = [str(t["id"]) for t in json.loads(path.read_text())]
    rng = random.Random(f"{SHUFFLE_SEED}:{domain}")
    rng.shuffle(ids)
    return ids


def domain_dir(domain: str) -> str:
    return "banking_knowledge" if domain == "banking" else domain


def plan(
    domains: Sequence[str],
    cells: Sequence[str],
    trials: int,
    data_dir: str,
) -> list[dict[str, Any]]:
    """Every unit of the campaign, in the order it should be run.

    (domain, task, trial, cell): a task finishes all its trials and all its cells before the
    next task starts, so a stopped campaign holds COMPLETE tasks -- which is the only shape
    `pass^k` can read, since k trials of a task are its unit of aggregation.
    """
    out: list[dict[str, Any]] = []
    for domain in domains:
        for tid in task_order(domain, data_dir):
            for seed in range(trials):
                for cid in cells:
                    out.append({"domain": domain, "task_id": tid, "seed": seed, "cell_id": cid})
    return out


def shard_of(unit: dict[str, Any], n_shards: int) -> int:
    """By (domain, task): all of a task's cells and trials share a shard.

    `hash()` is salted per process and would give two shards of one campaign different
    partitions, so this uses a stable digest.
    """
    import hashlib

    key = f"{unit['domain']}/{unit['task_id']}".encode()
    return int(hashlib.sha256(key).hexdigest()[:8], 16) % max(1, int(n_shards))


# --------------------------------------------------------------------------- running


def run_one(unit: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Set this cell's role pins, build the spec, run it.

    THE PINS ARE SET PER UNIT, IN THIS PROCESS. `MeteredClient` resolves a role's model from
    the environment at call time, so a shard that runs more than one cell has to move them
    between units. Safe because a shard is single-threaded by construction: two threads would
    race on `os.environ` and silently run one cell's dialogue on another cell's model.
    """
    from pi_run.stages.tau2_runner import UPSTREAM_PROTOCOL, run_tau2_unit
    from pi_run.worker import UnitSpec

    cell = CELLS_BY_ID[unit["cell_id"]]
    os.environ["PI_MODEL_DRAFTER"] = args.answer_model
    os.environ["PI_MODEL_ANSWERER"] = args.answer_model
    os.environ["PI_MODEL_USERSIM"] = (
        BRIDGE_USERSIM if args.bridge and unit["cell_id"] == BRIDGE_CELL else args.user_sim
    )
    if cell["inquirer"]:
        os.environ["PI_MODEL_INQUIRER"] = cell["inquirer"]
    else:
        # The stock arm has no questioner. Left set rather than unset would be a pin the run
        # did not use, sitting in the environment where a later unit could read it.
        os.environ.pop("PI_MODEL_INQUIRER", None)

    spec = UnitSpec(
        suite_id=DOMAINS[unit["domain"]],
        corpus_dir="",
        task_id=unit["task_id"],
        arm_id=cell["arm_id"],
        seed=int(unit["seed"]),
        runs_root=args.runs_root,
        cache_root=args.cache_root,
        max_turns=args.max_turns,
        k=args.k,
        budget_cap=args.budget_cap,
        resume=True,
        code_version=args.code_version,
        dirty=args.dirty,
        grid_name=args.grid_name,
        timeout_s=args.timeout_s,
        protocol=UPSTREAM_PROTOCOL,
    )
    status = run_tau2_unit(spec)
    status["cell_id"] = unit["cell_id"]
    status["domain"] = unit["domain"]
    return status


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--domain", action="append", default=None, choices=sorted(DOMAINS))
    p.add_argument("--cell", action="append", default=None, choices=sorted(CELLS_BY_ID))
    p.add_argument("--trials", type=int, default=4, help="tau2's pass^k trials per task")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--n-shards", type=int, default=1)
    p.add_argument("--runs-root", required=True)
    p.add_argument("--cache-root", required=True)
    p.add_argument("--data-dir", default=os.environ.get("TAU2_DATA_DIR", ""))
    p.add_argument("--answer-model", default="openai/aws/claude-sonnet-5")
    # THE `openai/` PREFIX IS LOAD-BEARING AND IT IS NOT PART OF THE MODEL'S NAME. tau2 hands
    # the user simulator's pin STRAIGHT to `litellm.completion`, which needs a provider prefix
    # to route at all -- measured: a bare `Azure/gpt-4.1` dies on "LLM Provider NOT provided"
    # in the first user turn of every unit, five of five. `openai/` selects the
    # OpenAI-compatible transport, is stripped client-side before the request is built, and
    # leaves `Azure/gpt-4.1` on the wire, which is the name the gateway's allow-list carries
    # (lower-case `azure/gpt-4.1` is refused 403 by the team policy; the case matters).
    p.add_argument("--user-sim", default="openai/Azure/gpt-4.1")
    p.add_argument(
        "--bridge",
        action="store_true",
        help=f"run {BRIDGE_CELL!r} against {BRIDGE_USERSIM} instead of --user-sim",
    )
    p.add_argument("--budget-cap", type=int, default=16)
    p.add_argument("--max-turns", type=int, default=16)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--timeout-s", type=int, default=2400)
    p.add_argument("--grid-name", default="tau2_full_protocol_20260918")
    p.add_argument("--code-version", default="")
    p.add_argument("--dirty", action="store_true")
    p.add_argument("--limit", type=int, default=0, help="stop after N units; 0 is no limit")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--out", default="", help="JSONL of every status this shard collected")
    args = p.parse_args(argv)

    if not args.data_dir:
        raise SystemExit("set TAU2_DATA_DIR or pass --data-dir")
    if not args.code_version:
        raise SystemExit(
            "--code-version is required: a campaign whose runs cannot name one commit cannot "
            "be selected out of a shared runs root"
        )
    # THE TREE MUST MATCH THE COMMIT THE CAMPAIGN NAMES. A dirty tree stamps every run `dev-`,
    # and a `dev-` run is training-only by decision -- so a whole confirmatory campaign would
    # be inadmissible as an eval arm and nothing but this check would say so. It also catches
    # the case that makes it invisible: a PEER'S uncommitted file in a shared checkout.
    if not args.dirty:
        from pi_run.manifest import git_info

        info = git_info(str(ROOT))
        if info.sha != args.code_version:
            raise SystemExit(
                f"--code-version {args.code_version[:12]} but the tree is at {info.sha[:12]}"
            )
        if info.dirty:
            raise SystemExit(
                "the tree is dirty, so every run would be stamped dev- (training-only): "
                + ", ".join(info.dirty_files[:8])
            )

    domains = args.domain or list(DOMAINS)
    cells = args.cell or [c["cell_id"] for c in CELLS]
    units = [
        u
        for u in plan(domains, cells, args.trials, args.data_dir)
        if shard_of(u, args.n_shards) == args.shard
    ]
    if args.limit:
        units = units[: args.limit]

    print(
        f"shard {args.shard}/{args.n_shards}: {len(units)} units  domains={domains} "
        f"cells={cells} trials={args.trials} user_sim={args.user_sim} "
        f"answerer={args.answer_model} cap={args.budget_cap} code={args.code_version[:12]}",
        flush=True,
    )
    if args.dry_run:
        for u in units[:10]:
            print("  ", u, flush=True)
        print(f"   ... {len(units)} total; launched nothing", flush=True)
        return 0

    out = open(args.out, "a", buffering=1) if args.out else None
    t0 = time.perf_counter()
    counts: dict[str, int] = {}
    for i, unit in enumerate(units, 1):
        t1 = time.perf_counter()
        try:
            status = run_one(unit, args)
        except Exception as exc:  # noqa: BLE001 - one unit's failure is a row, not the shard's
            status = {
                **unit,
                "status": "driver_error",
                "error": f"{type(exc).__name__}: {exc}",
                "run_id": "",
            }
        counts[str(status.get("status"))] = counts.get(str(status.get("status")), 0) + 1
        if out is not None:
            out.write(json.dumps(status, sort_keys=True, default=str) + "\n")
        print(
            f"[{i}/{len(units)}] {unit['domain']}/{unit['task_id']} s{unit['seed']} "
            f"{unit['cell_id']:20s} {str(status.get('status')):9s} "
            f"{time.perf_counter() - t1:6.1f}s  {status.get('run_id', '')[:12]}  "
            f"usd={status.get('usd_billed', 0)}",
            flush=True,
        )
    print(
        f"[done] shard={args.shard} wall_s={time.perf_counter() - t0:.1f} "
        f"counts={json.dumps(counts, sort_keys=True)}",
        flush=True,
    )
    if out is not None:
        out.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
