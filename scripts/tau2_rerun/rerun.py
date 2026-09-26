"""Plan, cost and drive the local rerun of the withdrawn tau2 transfer campaign.

WHAT IS REPRODUCED, AND FROM WHERE. The withdrawn campaign (artifacts/tau2_armsensitive_20260922,
retail 408/408 at ecdbb88a, airline 408/408 at 0276ce81) ran `scripts/run_tau2_forks.py` once per
(arm config, seed) shard from `scripts/tau2_campaign/phase5_shard.sh`:

    run_tau2_forks.py --suite <suite> --forkpoints conf/forks/<suite>_test.recovered34.json
                      --arm <inquirer_prompted|inquirer_trained> --seeds <s>
                      --prompt-variant <tau2_base|tau2_stop> --runs-root <root>

with max_turns 16, budget_cap 16 and retrieval k 5 (the script's and UnitSpec's defaults), the
drafter and answerer pinned to openai/aws/claude-sonnet-5 and the user simulator to
openai/aws/gpt-oss-120b. The rerun builds its UnitSpecs with THAT script's own `build_specs` and runs
each through THAT script's own `_run_unit` (= `run_tau2_unit`), on the same 34 fork points (the file
is pinned by sha256 below and refused if it changed), the same two prompt variants (pinned by their
recorded digests), the same three seeds and the same frozen roles. Three things change, by design:
the questioner's thinking is off (the proxy config, owned by another lane), the 16-call cap is
enforced (owned by another lane), and the arms are the prompted base plus the genuine trained seeds
s1/s2.

LAYOUT. `<root>/<suite>/<pin>/<run_id>/`. s1 and s2 both run as arm `inquirer_trained`, exactly as
the seed-0 pin did, and a status.json record carries no model pin, so in one flat root they would
share every (trace, k, seed) key and `pi_eval.fork_report._pair_forks` would refuse. `runs_root` is
not in run identity (`pinq.ids.SEMANTIC_FIELDS`), so the sub-roots move no run_id.

BOTH SUITES RUN AT ONCE, BEHIND ONE TUNNEL AND ONE PROXY. `stage2` opens the tunnel, starts the
proxy, runs the thinking probe and the gateway route proof ONCE, then starts one `run` process per
suite. Each suite has its own process, pool, spend cap and abort, so a cap or an abort in one never
touches the other's in-flight units; the tunnel and proxy stay up until every suite has returned.
Both suites therefore share one LITELLM_BASE_URL, i.e. one base_url_sha per role.

WHY A PROCESS POOL AND NOT SHARD PROCESSES. The unit function, the spec builder and the flags are
unchanged; only the grouping of units into processes differs, and grouping is not in run identity.
A pool lets the spend cap stop SCHEDULING while in-flight units finish and record their spend -- a
shard can only be stopped by killing it mid-unit, which spends money and writes no status.json.

THE PILOT (`--pilot N`) is the same design on the first N fork points of a declared order, seed 0
only: 6N units per suite. Its units carry `pilot=True`, which is `pilot_flag` in run identity and
is excluded by `pi_eval.report.ELIGIBLE`, and it runs in its own root, which the launcher refuses
to resume as a campaign root.

SHARDED ON HPC (`run --shard K/N`). The slots of a suite's plan -- (trace, k, task, seed, variant),
each carrying all three pins -- are sorted by that key and dealt round-robin, so shards are
disjoint, cover the plan, and never split a slot's arms. Every shard writes into the SAME
`<root>/<suite>/<pin>/`; its own copies of the probe and serve records it ran under, JOB.json and
RUN_SUMMARY.json go to `<root>/<suite>/_jobs/<LSB_JOBID[.LSB_JOBINDEX]>/`. `run` refuses a probe
record that is not PASS for its own proxy, or a serve record without both s1/s2 sha256s. The LSF
wrapper (in-job vLLM, proxy, chaining, staging) is another lane's.

    python scripts/tau2_rerun/rerun.py plan  --suite tau2_retail --concurrency 8 [--pilot 2] [--shard 3/6]
    python scripts/tau2_rerun/rerun.py stats --records retail_final.json airline_final.json --out F

`stage2` and `run` are called by `launch_local.sh` or a HPC job; no test invokes them for real.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import hashlib
import importlib.util
import json
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

# ------------------------------------------------------------------------------ the design
# (model pin, arm_id). The arm_id is the withdrawn campaign's own label for each side: the
# comparator ran as `inquirer_prompted` on `qwen3-8b-base`, the treatment as `inquirer_trained`
# (same class, same prompt -- pinq_expt/arms.py) on the resumed seed-0 pin. s1/s2 are UNREGISTERED
# in conf/checkpoints.json on purpose: registering a name re-labels finished runs that pinned it.
# Their weight provenance is the serve record (`--serve-record`), required at launch.
PINS: tuple[tuple[str, str], ...] = (
    ("qwen3-8b-base", "inquirer_prompted"),
    ("qwen3-8b-dpo-stacked-notdone-both-s1", "inquirer_trained"),
    ("qwen3-8b-dpo-stacked-notdone-both-s2", "inquirer_trained"),
)
TRAINED = tuple(p for p, arm in PINS if arm == "inquirer_trained")
FROZEN_ROLES: dict[str, str] = {
    "PI_MODEL_DRAFTER": "openai/aws/claude-sonnet-5",
    "PI_MODEL_ANSWERER": "openai/aws/claude-sonnet-5",
    "PI_MODEL_USERSIM": "openai/aws/gpt-oss-120b",
}
VARIANTS: tuple[str, ...] = ("tau2_base", "tau2_stop")
SEEDS: tuple[int, ...] = (0, 1, 2)
PILOT_SEEDS: tuple[int, ...] = (0,)
MAX_TURNS = 16
BUDGET_CAP = 16
PRICE_TABLE_REL = "scripts/price_tables/2026-09.json"
PROXY_CONFIG_REL = "conf/serving/litellm.tau2.yaml"
PROXY_UP_REL = "scripts/tau2_campaign/proxy_up.sh"
PROBE_REL = "scripts/tau2_concordance/thinking_probe.py"
COSTS_JSON = HERE / "withdrawn_unit_costs.json"

# The fork-point record each suite's campaign ran, pinned by content. Verified 2026-09-23: the
# (trace_sha, k, task_id) triples of all 408 records of each withdrawn population equal these
# files' 34 points exactly (0 file-only, 0 record-only).
FORKS: dict[str, tuple[str, str]] = {
    "tau2_retail": (
        "conf/forks/tau2_retail_test.recovered34.json",
        "b1fc9e669c3a6206e5c885d75f9a31e1460bbf81641ce7119c09dd4ddb278245",
    ),
    "tau2_airline": (
        "conf/forks/tau2_airline_test.recovered34.json",
        "651a8c2e5ce94339e5ae70bc0c575fe3d1bdbdd888029193fbd75a82c6d18b3c",
    ),
}
SUITES: tuple[str, ...] = tuple(FORKS)
# The withdrawn population each suite's cost history comes from -- one code_version per suite,
# never pooled. Airline's 172 slots at 87fbf71d are a superseded population and are not used.
WITHDRAWN: dict[str, dict[str, Any]] = {
    "tau2_retail": {"code_version": "ecdbb88ad83c98ebf97f9fde608ad0d105844157", "n_units": 408},
    "tau2_airline": {"code_version": "0276ce8116511a6f8df8cb7ae7fcc56710911229", "n_units": 408},
}
# Which withdrawn cell a new pin is costed from: the comparator from the comparator, both trained
# seeds from the (seed-0) trained arm. A different checkpoint behaves differently; this is a prior.
HISTORY_ARM = {pin: arm for pin, arm in PINS}
# The labels the withdrawn manifests carry: variant name + the first 8 hex of its template's sha256
# (pi_run.stages.tau2_runner.apply_prompt_variant). Read off 113 airline manifests at 87fbf71d, the
# only withdrawn manifests on this machine; every prompt_hashes entry there matches HEAD's
# templates. A different digest means a different prompt, so `check` refuses it before a launch.
PROMPT_VARIANT_IDS = {"tau2_base": "tau2_base-3006f5bf", "tau2_stop": "tau2_stop-6b1a91aa"}
_VARIANT_TEMPLATE = {
    "tau2_base": "inquirer_prompted_tau2_base",
    "tau2_stop": "inquirer_prompted_tau2",
}

# Exceptions after which no further unit may run: the results would be wrong, not missing.
ABORT_ON = {"FirewallError", "ReconcileError", "CanaryHit", "BrokenProcessPool"}


class PlanError(RuntimeError):
    """A precondition of the design does not hold; nothing may be launched."""


class PreflightFailed(RuntimeError):
    """Stage 2 could not prove its tunnel, proxy or routes; no suite was started."""


# ------------------------------------------------------------------------------ the plan
def load_points(suite: str, root: Path = ROOT) -> list[dict[str, Any]]:
    """The 34 recorded fork points, refused if the file is not the one the campaign ran."""
    if suite not in FORKS:
        raise PlanError(f"unknown suite {suite!r}; known: {sorted(FORKS)}")
    rel, want = FORKS[suite]
    raw = (root / rel).read_bytes()
    got = hashlib.sha256(raw).hexdigest()
    if got != want:
        raise PlanError(
            f"{rel} has sha256 {got}, not {want} -- the fork points are not the ones the "
            "withdrawn campaign ran, so this would be a different design"
        )
    doc = json.loads(raw)
    pts = doc["fork_points"]
    if len(pts) != int(doc["n_fork_points"]):
        raise PlanError(f"{rel}: {len(pts)} points but n_fork_points={doc['n_fork_points']}")
    return [
        {"trace_sha": str(p["trace_sha"]), "k": int(p["k"]), "task_id": str(p["task_id"])}
        for p in pts
    ]


def _task_key(tid: str) -> tuple:
    return (int(tid) if tid.isdigit() else 10**9, tid)


def pilot_points(suite: str, n: int, root: Path = ROOT) -> list[dict[str, Any]]:
    """THE PILOT'S SELECTION, DECLARED. Rank each point within its task by (k, trace_sha); order
    all 34 by (that rank, task_id as an integer); take the first `n`. So every task contributes
    one point before any contributes a second -- a plain task-id sort put both airline points of
    N=2 on ONE task. Deterministic and fixed before any pilot result exists, so the pilot cannot
    be steered toward points that look good."""
    by_task: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for p in load_points(suite, root):
        by_task[p["task_id"]].append(p)
    ranked = [
        (rank, _task_key(tid), p)
        for tid, ps in by_task.items()
        for rank, p in enumerate(sorted(ps, key=lambda q: (q["k"], q["trace_sha"])))
    ]
    pts = [p for _r, _t, p in sorted(ranked, key=lambda x: (x[0], x[1]))]
    if not 1 <= n <= len(pts):
        raise PlanError(f"--pilot {n} is outside 1..{len(pts)}")
    return pts[:n]


def parse_shard(text: str | None) -> tuple[int, int] | None:
    """`K/N` with 1 <= K <= N (LSF array indices are 1-based), or None for the whole plan."""
    if not text:
        return None
    m = re.fullmatch(r"(\d+)/(\d+)", text)
    if not m or not 1 <= int(m.group(1)) <= int(m.group(2)):
        raise PlanError(f"--shard {text!r} is not K/N with 1 <= K <= N")
    return int(m.group(1)), int(m.group(2))


def _slot(u: Mapping[str, Any]) -> tuple:
    """A unit's key WITHOUT its pin -- the three arms of one slot are one pairing unit -- ordered
    (seed, variant, trace, k, task) ON PURPOSE: see shard_units."""
    return (u["seed"], u["variant"], u["trace_sha"], u["k"], u["task_id"])


def shard_units(units: Sequence[Mapping[str, Any]], shard: tuple[int, int] | None) -> list:
    """Shard K of N: the slots sorted by their full key, every N-th from K-1, each slot with ALL
    its pins. Deterministic (a sort, not the plan's order), disjoint and covering (a partition of
    the sorted slots by index mod N), and a stopped shard leaves complete triples, never one arm.

    (seed, variant) LEADS THE SORT KEY. Led by the fork point instead, each point is a run of six
    consecutive slots (3 seeds x 2 variants), and dealing round-robin at N=6 or N=12 handed every
    shard ONE (seed, variant) for all its points -- measured by the test that asks each shard for
    both variants. Led by (seed, variant), each block of 34 slots is dealt across all N shards,
    so for N <= 34 every shard carries every seed and every variant, and shards cost alike."""
    if shard is None:
        return list(units)
    k, n = shard
    mine = set(sorted({_slot(u) for u in units})[k - 1 :: n])
    return [u for u in units if _slot(u) in mine]


def plan_units(
    suite: str,
    root: Path = ROOT,
    pilot: int | None = None,
    shard: tuple[int, int] | None = None,
) -> list[dict[str, Any]]:
    """Every unit, INTERLEAVED: point, then seed, then variant, then pin.

    Adjacent units are the three arms of one (point, seed, variant) slot, so a campaign stopped by
    the spend cap or an outage leaves mostly complete triples rather than one finished arm and two
    empty ones. With `pilot`, the same design on `pilot_points(suite, pilot)` at PILOT_SEEDS; with
    `shard`, only that shard's slots (`shard_units`).
    """
    points = pilot_points(suite, pilot, root) if pilot else load_points(suite, root)
    seeds = PILOT_SEEDS if pilot else SEEDS
    units = [
        {
            "suite": suite,
            "pin": pin,
            "arm_id": arm,
            "variant": variant,
            "seed": seed,
            "task_id": p["task_id"],
            "k": p["k"],
            "trace_sha": p["trace_sha"],
        }
        for p in points
        for seed in seeds
        for variant in VARIANTS
        for pin, arm in PINS
    ]
    return shard_units(units, shard)


def counts(units: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    c = collections.Counter(f"{u['pin']}|{u['arm_id']}|{u['variant']}" for u in units)
    return dict(sorted(c.items()))


# ------------------------------------------------------------------------------ cost history
def _q(xs: Sequence[float], p: float) -> float:
    s = sorted(xs)
    i = (len(s) - 1) * p
    lo = int(i)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (i - lo)


def _dist(xs: Sequence[float]) -> dict[str, float]:
    return {
        "median": round(statistics.median(xs), 4),
        "p90": round(_q(xs, 0.9), 4),
        "mean": round(statistics.fmean(xs), 4),
        "max": round(max(xs), 4),
    }


def _metered(r: Mapping[str, Any]) -> float:
    return float(r["usage"]["usd"]) + float(r.get("user_sim_usd") or 0)


def history_stats(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per (suite, arm, variant) cost and wall per unit, from status.json records.

    `usd_metered` = usage.usd (every ledger call, a cache hit charged as if it missed) plus
    user_sim_usd (the priced user-simulator messages). `usd_billed` = the recorded usd_billed
    (cache-miss ledger dollars + user_sim_usd + stock_agent_usd): what that campaign's cache state
    actually sent to the gateway. Refuses a cell that mixes code_versions.
    """
    cells: dict[str, list[Mapping[str, Any]]] = collections.defaultdict(list)
    for r in records:
        cells[f"{r['suite_id']}|{r['arm_id']}|{r['key'][9]}"].append(r)
    out: dict[str, dict[str, Any]] = {}
    for key, rs in sorted(cells.items()):
        cvs = sorted({str(r.get("code_version") or "") for r in rs})
        if len(cvs) != 1:
            raise PlanError(f"cell {key} pools code_versions {cvs}; disaggregate first")
        out[key] = {
            "n": len(rs),
            "code_version": cvs[0],
            "usd_metered": _dist([_metered(r) for r in rs]),
            "usd_billed": _dist([float(r.get("usd_billed") or 0) for r in rs]),
            "user_sim_unpriced_msgs_per_unit": round(
                statistics.fmean(int(r.get("user_sim_unpriced_msgs") or 0) for r in rs), 3
            ),
            "stock_agent_usd_total": round(
                sum(float(r.get("stock_agent_usd") or 0) for r in rs), 6
            ),
            "wall_s": _dist([float(r["wall_ms"]) / 1000 for r in rs]),
            "n_turns_median": statistics.median(int(r["n_turns"]) for r in rs),
            "n_calls_median": statistics.median(int(r["n_calls"]) for r in rs),
            "status": dict(collections.Counter(str(r.get("status")) for r in rs)),
        }
    return out


def point_stats(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per (suite, arm, variant, trace, k) means over the withdrawn seeds -- what a pilot on a few
    named points is costed from, since a cell mean would price the pilot at the average point."""
    cells: dict[str, list[Mapping[str, Any]]] = collections.defaultdict(list)
    for r in records:
        key = f"{r['suite_id']}|{r['arm_id']}|{r['key'][9]}|{r['foreign_trace_sha']}|{r['foreign_prefix_k']}"
        cells[key].append(r)
    return {
        key: {
            "n": len(rs),
            "usd_metered_mean": round(statistics.fmean(_metered(r) for r in rs), 4),
            "usd_billed_mean": round(
                statistics.fmean(float(r.get("usd_billed") or 0) for r in rs), 4
            ),
            "wall_s_mean": round(statistics.fmean(float(r["wall_ms"]) / 1000 for r in rs), 2),
        }
        for key, rs in sorted(cells.items())
    }


def population_summary(records: Sequence[Mapping[str, Any]], shards: int) -> dict[str, Any]:
    """The whole population's span, and what a pooled estimator predicts for it -- the estimator's
    only available validation. `shards` is how many units the campaign had in flight."""
    wall = sum(float(r["wall_ms"]) for r in records) / 1000
    span = max(float(r["finished_at"]) for r in records) - min(
        float(r["started_at"]) for r in records
    )
    ids = "\n".join(sorted(str(r["run_id"]) for r in records))
    return {
        "n": len(records),
        "run_ids_sha256_16": hashlib.sha256(ids.encode()).hexdigest()[:16],
        "sum_wall_h": round(wall / 3600, 3),
        "observed_span_h": round(span / 3600, 3),
        "predicted_span_h_at_inflight": round(wall / shards / 3600, 3),
        "inflight_assumed": shards,
    }


def estimate(
    units: Sequence[Mapping[str, Any]], concurrency: int, costs: Mapping[str, Any]
) -> dict[str, Any]:
    """Expected dollars and wall for `units`. Each unit is costed from the withdrawn mean of ITS
    point in its HISTORY_ARM's cell (three seeds), so a pilot on named points is priced as those
    points, and a full plan sums to the cell means exactly."""
    points = costs["points"]
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    tail_s = 0.0
    for u in units:
        cell = f"{u['suite']}|{HISTORY_ARM[u['pin']]}|{u['variant']}"
        p = points[f"{cell}|{u['trace_sha']}|{u['k']}"]
        r = rows.setdefault(
            (u["pin"], u["variant"]),
            {
                "pin": u["pin"],
                "variant": u["variant"],
                "n": 0,
                "costed_from": f"{cell}@{costs['cells'][cell]['code_version'][:8]}",
                "usd_metered": 0.0,
                "usd_billed": 0.0,
                "wall_h": 0.0,
            },
        )
        r["n"] += 1
        r["usd_metered"] += p["usd_metered_mean"]
        r["usd_billed"] += p["usd_billed_mean"]
        r["wall_h"] += p["wall_s_mean"] / 3600
        tail_s = max(tail_s, costs["cells"][cell]["wall_s"]["max"])
    out_rows = [
        {**r, **{f: round(r[f], 2) for f in ("usd_metered", "usd_billed", "wall_h")}}
        for _, r in sorted(rows.items())
    ]
    wall_h = sum(r["wall_h"] for r in rows.values())
    return {
        "rows": out_rows,
        "usd_metered": round(sum(r["usd_metered"] for r in rows.values()), 2),
        "usd_billed": round(sum(r["usd_billed"] for r in rows.values()), 2),
        "unit_wall_h": round(wall_h, 2),
        "concurrency": concurrency,
        "wall_h_pooled": round(wall_h / concurrency, 2),
        "wall_h_upper": round(wall_h / concurrency + tail_s / 3600, 2),
    }


# ------------------------------------------------------------------------------ preconditions
def local_route_port(config_text: str, pins: Sequence[str]) -> int:
    """The loopback port the proxy config sends every pinned Qwen route to -- the tunnel's local
    end. Derived from the config the proxy will start from, never typed: all pins must have a
    hosted_vllm route and all must share one loopback api_base."""
    import yaml

    doc = yaml.safe_load(config_text) or {}
    bases: dict[str, str] = {}
    for m in doc.get("model_list") or []:
        params = m.get("litellm_params") or {}
        if str(params.get("model") or "").startswith("hosted_vllm/"):
            bases[str(m.get("model_name"))] = str(params.get("api_base") or "")
    missing = [p for p in pins if p not in bases]
    if missing:
        raise PlanError(f"the proxy config has no hosted_vllm route for {missing}")
    distinct = sorted({bases[p] for p in pins})
    if len(distinct) != 1:
        raise PlanError(
            f"the pinned routes point at different api_bases {distinct}; one serve expected"
        )
    mt = re.fullmatch(r"http://127\.0\.0\.1:(\d+)/v1", distinct[0])
    if not mt:
        raise PlanError(
            f"api_base {distinct[0]!r} is not a literal loopback http://127.0.0.1:<port>/v1"
        )
    return int(mt.group(1))


def price_table_missing(path: Path, models: Iterable[str]) -> list[str]:
    rows = json.loads(path.read_text())["models"]
    return [m for m in models if m not in rows]


def missing_traces(points: Iterable[Mapping[str, Any]], trace_root: Path) -> list[str]:
    """Trace shas no JSONL under `trace_root` carries -- each would raise inside its unit."""
    want = {str(p["trace_sha"]) for p in points}
    for path in sorted(trace_root.glob("*/*.jsonl")):
        text = path.read_text()
        want = {s for s in want if s not in text}
        if not want:
            break
    return sorted(want)


def prompt_variant_problems() -> list[str]:
    """The labels THIS tree would stamp, against the withdrawn campaign's -- before any spend."""
    from pinq import promptlib

    out = []
    for variant, template in _VARIANT_TEMPLATE.items():
        label = f"{variant}-{promptlib.sha(template)[:8]}"
        if label != PROMPT_VARIANT_IDS[variant]:
            out.append(
                f"{template} renders {label}, the withdrawn campaign rendered "
                f"{PROMPT_VARIANT_IDS[variant]}: a different prompt is a different design"
            )
    return out


_SHA256 = re.compile(r"[0-9a-f]{64}")


def serve_record_problems(record: Mapping[str, Any], serve: str | None) -> list[str]:
    """The serve record (scripts/hpc/serve_map.py `served_record`, schema pinq.serve.v1) is the only
    weight provenance the unregistered s1/s2 names have: no conf/checkpoints.json row, so
    `adapter_sha` in every manifest is null. It must name a sha256 for both trained adapters, list
    all three pins as served, and -- when `serve` (host:port) is given -- be the record of THAT
    serve."""
    out = []
    adapters = record.get("adapters") or {}
    for pin in TRAINED:
        sha = str((adapters.get(pin) or {}).get("sha256") or "")
        if not _SHA256.fullmatch(sha):
            out.append(f"no sha256 for {pin} in the serve record")
    served = set(record.get("served") or [])
    missing = [p for p, _ in PINS if p not in served]
    if missing:
        out.append(f"the serve record does not list {missing} as served")
    if not serve:
        return out
    host, _, port = serve.rpartition(":")
    if str(record.get("port")) != port:
        out.append(
            f"the serve record is for port {record.get('port')!r}, the tunnel reaches {port}"
        )
    if str(record.get("host") or "").split(".")[0] != host.split(".")[0]:
        out.append(
            f"the serve record is for host {record.get('host')!r}, the tunnel reaches {host}"
        )
    return out


def probe_record_problems(record: Mapping[str, Any], proxy_url: str) -> list[str]:
    """The thinking probe's `--out` document (scripts/tau2_concordance/thinking_probe.py) must say
    PASS, for THIS job's proxy, with a clean route for every pinned Qwen model. A probe of another
    proxy proves nothing about this one: chat_template_kwargs live in the proxy's config, outside
    run identity and the call cache key."""
    out = []
    if record.get("verdict") != "PASS":
        out.append(f"probe verdict is {record.get('verdict')!r}, not 'PASS'")
    if record.get("proxy_url") != proxy_url:
        out.append(f"probe was of proxy {record.get('proxy_url')!r}, this run uses {proxy_url!r}")
    routes = {str(r.get("model")): r for r in record.get("routes") or []}
    for pin, _arm in PINS:
        r = routes.get(pin)
        if r is None:
            out.append(f"probe has no route for {pin}")
        elif r.get("refuse_reason") or r.get("thinking_detected"):
            out.append(f"probe route {pin}: {r.get('refuse_reason') or 'thinking detected'}")
    return out


def job_label(env: Mapping[str, str], shard: tuple[int, int] | None) -> str:
    """`<LSB_JOBID>[.<LSB_JOBINDEX>]` inside LSF, else the shard and a UTC stamp. A resubmitted
    shard gets a new job id, so no job's records overwrite another's."""
    jid = env.get("LSB_JOBID", "")
    if jid:
        idx = env.get("LSB_JOBINDEX", "")
        return f"{jid}.{idx}" if idx and idx != "0" else jid
    part = f"shard{shard[0]}of{shard[1]}" if shard else "all"
    return f"{part}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"


def stage_job(suite_root: Path, label: str, probe: Path, serve: Path) -> Path:
    """`<suite_root>/_jobs/<label>/` holding this job's own copies of the records it ran under.
    Refused if it exists: two jobs must never write one record."""
    d = suite_root / "_jobs" / label
    if d.exists():
        raise PlanError(f"{d} already exists; a job's records are never overwritten")
    d.mkdir(parents=True)
    shutil.copyfile(probe, d / "thinking_probe.json")
    shutil.copyfile(serve, d / "serve_record.json")
    return d


def manifest_problems(
    manifest: Mapping[str, Any], *, pin: str, variant: str, code_version: str, pilot: bool = False
) -> list[str]:
    """What a unit RECORDED that differs from what was requested. Read back per unit because a
    drift is invisible from the launcher's side: 408 units once shipped with the wrong frozen roles
    while every route probe passed."""
    pins = manifest.get("pins") or {}

    def mid(role: str) -> str:
        v = pins.get(role)
        return str(v.get("model_id") if isinstance(v, dict) else v or "")

    want = {
        "pins.inquirer": (mid("inquirer"), pin),
        "pins.drafter": (mid("drafter"), FROZEN_ROLES["PI_MODEL_DRAFTER"]),
        "pins.answerer": (mid("answerer"), FROZEN_ROLES["PI_MODEL_ANSWERER"]),
        "upstream_pins.user_sim": (
            str((manifest.get("upstream_pins") or {}).get("user_sim") or ""),
            FROZEN_ROLES["PI_MODEL_USERSIM"],
        ),
        "budget_cap": (manifest.get("budget_cap"), BUDGET_CAP),
        "max_turns": (manifest.get("max_turns"), MAX_TURNS),
        "dirty": (manifest.get("dirty"), False),
        "code_version": (manifest.get("code_version"), code_version),
        "pilot_flag": (bool(manifest.get("pilot_flag")), pilot),
    }
    out = [f"{f}={got!r}, requested {exp!r}" for f, (got, exp) in want.items() if got != exp]
    pv = str(manifest.get("prompt_variant_id") or "")
    if pv != PROMPT_VARIANT_IDS[variant]:
        out.append(
            f"prompt_variant_id={pv!r}, the withdrawn campaign's is {PROMPT_VARIANT_IDS[variant]!r}"
        )
    return out


def resume_problems(launch: Mapping[str, Any], now: Mapping[str, Any]) -> list[str]:
    """A root may only be resumed under the identity it was launched with. The commit and the
    proxy URL are in run identity (code_version, base_url_sha): changing either re-runs everything
    under new run_ids and pools two populations. The pilot setting decides which root a result
    belongs to: a pilot root is never resumed as a campaign root, nor the reverse."""
    out = []
    if launch.get("pilot") != now.get("pilot"):
        out.append(
            f"pilot: recorded {launch.get('pilot')!r}, now {now.get('pilot')!r} -- a pilot root is "
            "never resumed as a campaign root (and a campaign root never as a pilot)"
        )
    for k in ("git_sha", "suites", "proxy_url", "cache_root"):
        if launch.get(k) != now.get(k):
            out.append(f"{k}: recorded {launch.get(k)!r}, now {now.get(k)!r}")
    was = (launch.get("serve_record") or {}).get("adapters") or {}
    for pin in TRAINED:
        if was.get(pin) != (now.get("adapters") or {}).get(pin):
            out.append(f"{pin}: served weights changed since launch ({was.get(pin)!r})")
    return out


# ------------------------------------------------------------------------------ the driver
_FORKS_MOD = None


def _forks_module() -> Any:
    """scripts/run_tau2_forks.py -- the withdrawn campaign's own spec builder and unit call."""
    global _FORKS_MOD
    if _FORKS_MOD is None:
        spec = importlib.util.spec_from_file_location(
            "run_tau2_forks", ROOT / "scripts" / "run_tau2_forks.py"
        )
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        _FORKS_MOD = mod
    return _FORKS_MOD


def _unit(spec: Any, pin: str) -> dict[str, Any]:
    """One unit in a pool worker. A worker runs one task at a time, so setting the inquirer pin in
    its environment here is what that unit's MeteredClient reads at construction."""
    os.environ["PI_MODEL_INQUIRER"] = pin
    return _forks_module()._run_unit(spec)


def build_work(
    suite: str,
    runs_root: Path,
    cache_root: Path,
    gi: Any,
    pilot: int | None = None,
    shard: tuple[int, int] | None = None,
) -> list[tuple[Any, str, str]]:
    forks = _forks_module()
    points = {(p["trace_sha"], p["k"], p["task_id"]): p for p in load_points(suite)}
    work = []
    for u in plan_units(suite, pilot=pilot, shard=shard):
        (spec,) = forks.build_specs(
            [points[(u["trace_sha"], u["k"], u["task_id"])]],
            suite=suite,
            arms=[u["arm_id"]],
            seeds=[u["seed"]],
            max_turns=MAX_TURNS,
            budget_cap=BUDGET_CAP,
            code_version=gi.sha,
            dirty=gi.dirty,
            dirty_files=gi.dirty_files,
            runs_root=str(runs_root / u["pin"]),
            cache_root=str(cache_root),
            prompt_variant=u["variant"],
        )
        if pilot:
            # pilot_flag is in run identity and ELIGIBLE refuses it: a pilot unit can neither
            # share a run_id with a campaign unit nor reach a reported table.
            spec = dataclasses.replace(spec, pilot=True)
        work.append((spec, u["pin"], u["variant"]))
    return work


def check_env(proxy_url: str, price_table: str) -> list[str]:
    """The environment AFTER it is fully assembled -- a guard placed before a `.env` source is
    one the source can undo (scripts/tau2_campaign/load_env.sh)."""
    env = os.environ
    bad = []
    if env.get("PI_GOLD_ROOT"):
        bad.append("PI_GOLD_ROOT is set; a rollout worker must not be able to read gold")
    if env.get("LITELLM_BASE_URL") != proxy_url:
        bad.append("LITELLM_BASE_URL is not the launch's proxy URL")
    if env.get("PI_PRICE_TABLE") != price_table:
        bad.append("PI_PRICE_TABLE is not the launch's price table")
    for k, v in FROZEN_ROLES.items():
        if env.get(k) != v:
            bad.append(f"{k} is not {v}")
    for k in ("LITELLM_API_KEY", "TAU2_DATA_DIR"):
        if not env.get(k):
            bad.append(f"{k} is unset")
    return bad


def _effective(res: Mapping[str, Any]) -> str:
    st = str(res.get("status"))
    return str(res.get("prior_status") or st) if st == "resumed" else st


def drive(
    work: Sequence[tuple[Any, str, str]],
    ex: Any,
    unit_fn: Any,
    *,
    spend_cap: float,
    max_consecutive_failures: int,
    code_version: str,
    billed_usd: Any,
    pilot: bool = False,
    log: Any = print,
) -> tuple[collections.Counter, float, str]:
    """Submit every unit, read results as they land, and stop SCHEDULING on the first of:

      abort       a raise in ABORT_ON, a unit whose manifest recorded pins other than requested,
                  the attempt's first non-resumed unit not ok, or N consecutive non-ok units;
      spend cap   billed dollars >= the cap. A STOP-LOSS, not a ceiling: units already running
                  finish, bill and are counted here. A ProcessPoolExecutor also marks
                  EXTRA_QUEUED_CALLS (1) queued unit as running, so the total can pass the cap by
                  up to (workers + 1) units -- about $1 each at the withdrawn p90.

    Returns (tally by (pin, variant, status), billed dollars this attempt, stop reason or "").
    """
    from concurrent.futures import as_completed

    n = len(work)
    tally: collections.Counter = collections.Counter()
    billed, done, bad_run, first_new = 0.0, 0, 0, True
    stop = ""
    t0 = time.time()
    futures = {ex.submit(unit_fn, spec, pin): (spec, pin, variant) for spec, pin, variant in work}
    seen: set = set()
    try:
        for fut in as_completed(futures):
            seen.add(fut)
            spec, pin, variant = futures[fut]
            try:
                res = fut.result()
            except Exception as exc:  # noqa: BLE001 - classified here, never swallowed silently
                name = type(exc).__name__
                if name in ABORT_ON:
                    stop = f"abort: {name}: {str(exc)[:300]}"
                    break
                res = {"status": f"raised:{name}", "error": str(exc)[:300], "run_id": ""}
            done += 1
            resumed = str(res.get("status")) == "resumed"
            eff = _effective(res)
            tally[(pin, variant, eff)] += 1
            billed += 0.0 if resumed else billed_usd(res)
            mf = Path(spec.runs_root) / str(res.get("run_id") or "-") / "manifest.json"
            if res.get("run_id") and mf.is_file():
                probs = manifest_problems(
                    json.loads(mf.read_text()),
                    pin=pin,
                    variant=variant,
                    code_version=code_version,
                    pilot=pilot,
                )
                if probs:
                    stop = f"abort: {mf} recorded {probs}"
                    break
            log(
                f"  [{done}/{n}] {eff:<8} {pin:<38} {variant} seed={spec.seed} "
                f"task={spec.task_id:<4} k={spec.foreign_prefix_k:<3} billed=${billed:.2f} "
                f"elapsed={(time.time() - t0) / 3600:.2f}h"
                + (f"  err={str(res.get('error'))[:160]}" if eff != "ok" else "")
            )
            if not resumed and first_new:
                first_new = False
                if eff != "ok":
                    stop = f"abort: the first unit of this attempt finished {eff}, not ok"
                    break
            bad_run = 0 if eff == "ok" else bad_run + 1
            if bad_run >= max_consecutive_failures:
                stop = f"abort: {bad_run} consecutive non-ok units (an outage?); resume later"
                break
            if billed >= spend_cap:
                stop = f"spend cap: ${billed:.2f} >= ${spend_cap:.2f}"
                break
    finally:
        ex.shutdown(wait=True, cancel_futures=True)
        # Units already running when the loop broke have finished and billed by now; count them,
        # so the printed total is the invoice and not the invoice minus the overshoot.
        for fut, (spec, pin, variant) in futures.items():
            if fut in seen or fut.cancelled():
                continue
            try:
                res = fut.result(timeout=0)
            except BaseException:  # noqa: BLE001 - its spend is unrecorded; say so in the tally
                tally[(pin, variant, "raised_after_stop")] += 1
                continue
            tally[(pin, variant, _effective(res))] += 1
            billed += 0.0 if str(res.get("status")) == "resumed" else billed_usd(res)
    return tally, billed, stop


def check_run_records(a: argparse.Namespace) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """The records a `run` is launched under, checked before the environment, git or any spend.
    Returns (probe record, serve record)."""
    runs_root, cache_root = Path(a.runs_root), Path(a.cache_root)
    if not (runs_root.is_absolute() and cache_root.is_absolute()):
        raise PlanError("runs and cache roots must be absolute")
    probe = json.loads(Path(a.probe_record).read_text())
    probs = probe_record_problems(probe, a.proxy_url)
    if probs:
        raise PlanError(f"the probe record {a.probe_record} is not a PASS for this proxy: {probs}")
    serve = json.loads(Path(a.serve_record).read_text())
    probs = serve_record_problems(serve, a.serve)
    if probs:
        raise PlanError(f"the serve record {a.serve_record} cannot vouch for s1/s2: {probs}")
    return probe, serve


def run_design_problems(suite: str, trace_root: Path, price_table: Path) -> list[str]:
    """What `launch_local.sh`'s `check` proves before a local launch, proved again inside `run`,
    because a HPC job never passes through that script. `trace_root` is the runner's own
    TRACE_ROOT, which is RELATIVE: a job whose cwd is not the staged tree finds no traces and every
    unit raises inside its dialogue."""
    out = prompt_variant_problems()
    tr = missing_traces(load_points(suite), trace_root)
    if tr:
        out.append(f"{len(tr)} fork traces absent under {trace_root.resolve()} (cwd {Path.cwd()})")
    miss = price_table_missing(price_table, [p for p, _ in PINS] + list(FROZEN_ROLES.values()))
    if miss:
        out.append(f"price table {price_table} has no row for {miss}")
    return out


def run_campaign(a: argparse.Namespace) -> int:
    """ONE suite, or one shard of it. Its own process -- under `stage2` locally, under an LSF job
    on HPC -- so its pool, cap and abort are its own. Every shard of a suite writes into the SAME
    `<runs-root>/<pin>/`; its job records go to `<runs-root>/_jobs/<label>/`."""
    from concurrent.futures import ProcessPoolExecutor

    from pi_run.sweep import billed_usd

    shard = parse_shard(a.shard)
    _probe, serve = check_run_records(a)
    runs_root, cache_root = Path(a.runs_root), Path(a.cache_root)
    bad = check_env(a.proxy_url, a.price_table)
    if bad:
        raise PlanError("; ".join(bad))
    if a.spend_cap <= 0:
        raise PlanError("--spend-cap must be > 0: a cap of 0 still bills one unit per worker")
    from pi_run.stages.tau2_runner import TRACE_ROOT

    design = run_design_problems(a.suite, Path(TRACE_ROOT), Path(a.price_table))
    if design:
        raise PlanError("; ".join(design))
    gi = _forks_module()._git_state()
    if gi.dirty:
        raise PlanError(f"tree is dirty ({list(gi.dirty_files)[:5]}); runs would be stamped dev-")
    work = build_work(a.suite, runs_root, cache_root, gi, pilot=a.pilot, shard=shard)
    label = a.job_label or job_label(os.environ, shard)
    job = stage_job(runs_root, label, Path(a.probe_record), Path(a.serve_record))
    units = plan_units(a.suite, pilot=a.pilot, shard=shard)
    (job / "JOB.json").write_text(
        json.dumps(
            {
                "label": label,
                "suite": a.suite,
                "shard": a.shard,
                "pilot": a.pilot,
                "code_version": gi.sha,
                "host": os.uname().nodename,
                "LSB_JOBID": os.environ.get("LSB_JOBID"),
                "LSB_JOBINDEX": os.environ.get("LSB_JOBINDEX"),
                "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "proxy_url": a.proxy_url,
                "cache_root": str(cache_root),
                "price_table": {"path": a.price_table, "sha256": sha256_file(Path(a.price_table))},
                "probe_record_sha256": sha256_file(job / "thinking_probe.json"),
                "serve_record_sha256": sha256_file(job / "serve_record.json"),
                "served_adapters": {
                    p: (serve.get("adapters") or {}).get(p, {}).get("sha256") for p in TRAINED
                },
                "concurrency": a.concurrency,
                "spend_cap_usd": a.spend_cap,
                "n_units": len(units),
                "counts": counts(units),
                "units": units,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        f"run: {a.suite} {len(work)} units{' (PILOT)' if a.pilot else ''}"
        f"{f' shard {a.shard}' if shard else ''}, concurrency {a.concurrency}, "
        f"cap ${a.spend_cap:.2f}, code {gi.sha[:12]}, job {job}",
        flush=True,
    )
    t0 = time.time()
    tally, billed, stop = drive(
        work,
        ProcessPoolExecutor(max_workers=a.concurrency),
        _unit,
        spend_cap=a.spend_cap,
        max_consecutive_failures=a.max_consecutive_failures,
        code_version=gi.sha,
        billed_usd=billed_usd,
        pilot=bool(a.pilot),
        log=lambda s: print(s, flush=True),
    )
    ok = sum(v for (_p, _v, s), v in tally.items() if s == "ok")
    summary = {
        "suite": a.suite,
        "shard": a.shard,
        "label": label,
        "pilot": a.pilot,
        "code_version": gi.sha,
        "n_planned": len(work),
        "n_ok": ok,
        "billed_usd_this_attempt": round(billed, 4),
        "stop_reason": stop,
        "wall_h": round((time.time() - t0) / 3600, 3),
        "by_pin_variant_status": {"|".join(map(str, k)): v for k, v in sorted(tally.items())},
    }
    (job / "RUN_SUMMARY.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    if stop.startswith("abort"):
        return 11
    if stop:
        return 10
    return 0 if ok == len(work) else 12


# ------------------------------------------------------------------------------ stage 2
def attempt_path(root: Path, name: str, ext: str, attempt: str) -> Path:
    """`<root>/<name>.<ext>` for the first launch, `<root>/<name>.<attempt>.<ext>` for a resume --
    the same rule launch_local.sh uses to find the preflight marker."""
    return root / (f"{name}.{ext}" if attempt == "initial" else f"{name}.{attempt}.{ext}")


@dataclasses.dataclass
class Stage2Config:
    suites: list[str]
    root: Path
    cache_root: Path
    proxy_url: str
    serve: str
    ssh_login: str
    serve_record: Path
    concurrency: int
    spend_cap: float
    pilot: int | None
    attempt: str
    git_sha: str
    python: str = sys.executable
    tunnel_wait_s: int = 45
    config: Path = ROOT / PROXY_CONFIG_REL
    price_table: Path = ROOT / PRICE_TABLE_REL

    @property
    def proxy_port(self) -> int:
        return int(self.proxy_url.rsplit(":", 1)[1])

    def path(self, name: str, ext: str) -> Path:
        return attempt_path(self.root, name, ext, self.attempt)


def _listeners(port: int) -> list[int]:
    r = subprocess.run(
        ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"], capture_output=True, text=True
    )
    return [int(x) for x in r.stdout.split()]


class Services:
    """The things stage 2 starts, each exactly once, and stops. Tests pass a fake."""

    def __init__(self, c: Stage2Config) -> None:
        self.c = c
        self.tunnel: subprocess.Popen | None = None
        self.proxy_pid: int | None = None
        self.suites: list[subprocess.Popen] = []

    def start_tunnel(self) -> int:
        c = self.c
        port = local_route_port(c.config.read_text(), [p for p, _ in PINS])
        if _listeners(port):
            raise PreflightFailed(f"local port {port} already has a listener {_listeners(port)}")
        log = open(c.path("tunnel", "log"), "ab")  # noqa: SIM115 - held for the tunnel's lifetime
        self.tunnel = subprocess.Popen(
            ["ssh", "-N", "-o", "BatchMode=yes", "-o", "ExitOnForwardFailure=yes",
             "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=4",
             "-L", f"127.0.0.1:{port}:{c.serve}", c.ssh_login],
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )  # fmt: skip
        for _ in range(c.tunnel_wait_s):
            if self.tunnel.pid in _listeners(port):
                return self.tunnel.pid
            if self.tunnel.poll() is not None:
                raise PreflightFailed(f"the ssh tunnel exited rc={self.tunnel.returncode}")
            time.sleep(1)
        raise PreflightFailed(f"the ssh tunnel is not listening on {port} after {c.tunnel_wait_s}s")

    def start_proxy(self) -> int:
        c = self.c
        if _listeners(c.proxy_port):
            raise PreflightFailed(f"proxy port {c.proxy_port} already has a listener")
        env = dict(
            os.environ,
            PINQ_TAU2_PORT=str(c.proxy_port),
            PINQ_TAU2_PROXY_LOG=str(c.path("proxy", "log")),
        )
        r = subprocess.run(
            ["bash", str(ROOT / PROXY_UP_REL)], env=env, capture_output=True, text=True
        )
        print(r.stdout + r.stderr, flush=True)
        m = re.search(r"^proxy_up: pid=(\d+) ", r.stdout, re.M)
        if m:
            self.proxy_pid = int(m.group(1))
        if r.returncode != 0 or not m:
            raise PreflightFailed(f"proxy_up.sh exited {r.returncode} (pid {self.proxy_pid})")
        return self.proxy_pid

    def probe(self) -> None:
        c = self.c
        probe = ROOT / PROBE_REL
        if not probe.is_file():
            raise PreflightFailed(f"no thinking probe at {probe}; no launch without it")
        models = [x for p, _ in PINS for x in ("--model", p)]
        rc = subprocess.run(
            [c.python, str(probe), "--proxy-url", c.proxy_url, "--config", str(c.config),
             *models, "--out", str(c.path("thinking_probe", "json"))],
        ).returncode  # fmt: skip
        if rc != 0:
            raise PreflightFailed(f"thinking_probe.py exited {rc}: thinking is not provably off")

    def route_proof(self) -> None:
        names = sorted({m.removeprefix("openai/") for m in FROZEN_ROLES.values()})
        res = route_proof(self.c.proxy_url, names, os.environ.get("LITELLM_API_KEY", ""))
        self.c.path("route_proof", "json").write_text(json.dumps(res, indent=2))
        bad = [r for r in res if not r["ok"]]
        if bad:
            raise PreflightFailed(f"gateway route proof failed: {bad}")

    def start_suite(self, suite: str) -> subprocess.Popen:
        c = self.c
        sroot = c.root / suite
        cmd = [c.python, str(Path(__file__).resolve()), "run", "--suite", suite,
               "--runs-root", str(sroot), "--cache-root", str(c.cache_root),
               "--proxy-url", c.proxy_url, "--price-table", str(c.price_table),
               "--concurrency", str(c.concurrency), "--spend-cap", str(c.spend_cap),
               "--probe-record", str(c.path("thinking_probe", "json")),
               "--serve-record", str(c.path("serve_record", "json")), "--serve", c.serve]  # fmt: skip
        if c.pilot:
            cmd += ["--pilot", str(c.pilot)]
        log = open(attempt_path(sroot, "run", "log", c.attempt), "ab")  # noqa: SIM115
        p = subprocess.Popen(
            cmd, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )  # fmt: skip
        self.suites.append(p)
        return p

    def wait(self, handle: subprocess.Popen) -> int:
        return handle.wait()

    def stop_all(self) -> None:
        def _kill(fn: Any, pid: int, sig: int) -> None:
            try:
                fn(pid, sig)
            except (ProcessLookupError, PermissionError):
                pass

        for p in self.suites:  # only still running if stage 2 itself is being torn down
            if p.poll() is None:
                _kill(os.killpg, p.pid, signal.SIGTERM)
        for sig in (signal.SIGTERM, signal.SIGKILL):
            if self.proxy_pid:
                _kill(os.kill, self.proxy_pid, sig)
            if self.tunnel is not None and self.tunnel.poll() is None:
                _kill(os.killpg, self.tunnel.pid, sig)
            time.sleep(2 if sig == signal.SIGTERM else 0)
        print(f"stage2: stopped proxy {self.proxy_pid} and tunnel "
              f"{self.tunnel.pid if self.tunnel else None}", flush=True)  # fmt: skip


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write_launch_records(c: Stage2Config, tunnel_pid: int, proxy_pid: int) -> None:
    """One top-level LAUNCH record and one per suite root."""
    record = json.loads(c.serve_record.read_text())
    copied = c.path("serve_record", "json")
    shutil.copyfile(c.serve_record, copied)
    costs = json.loads(COSTS_JSON.read_text())
    top_name = c.path("LAUNCH", "json").name
    top = {
        "suites": c.suites,
        "attempt": c.attempt,
        "launched_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_sha": c.git_sha,
        "tree": str(ROOT),
        "pilot": c.pilot,
        "proxy_url": c.proxy_url,
        "proxy_config": {"path": str(c.config), "sha256": sha256_file(c.config)},
        "proxy_pid": proxy_pid,
        "tunnel": {
            "local_port": local_route_port(c.config.read_text(), [p for p, _ in PINS]),
            "target": c.serve,
            "login": c.ssh_login,
            "pid": tunnel_pid,
        },
        "price_table": {"PI_PRICE_TABLE": str(c.price_table), "sha256": sha256_file(c.price_table)},
        "serve_record": {
            "source": str(c.serve_record),
            "copied_to": str(copied),
            "sha256": sha256_file(copied),
            "job": record.get("job"),
            "host": record.get("host"),
            "port": record.get("port"),
            "adapters": {
                p: (record.get("adapters") or {}).get(p, {}).get("sha256") for p in TRAINED
            },
        },
        "cache_root": str(c.cache_root),
        "suite_roots": {s: str(c.root / s) for s in c.suites},
        "env_present": {
            k: bool(os.environ.get(k))
            for k in (
                "LITELLM_API_KEY",
                "PI_AGENT_LITELLM_API_KEY",
                "TAU2_DATA_DIR",
                "PINQ_GATEWAY_BASE_URL",
            )
        },
        "env_absent_required": {"PI_GOLD_ROOT": not os.environ.get("PI_GOLD_ROOT")},
        "frozen_roles": FROZEN_ROLES,
        "pins": [{"pin": p, "arm_id": arm} for p, arm in PINS],
        "design": {
            "max_turns": MAX_TURNS,
            "budget_cap": BUDGET_CAP,
            "k": "UnitSpec default (5)",
            "seeds": list(PILOT_SEEDS if c.pilot else SEEDS),
            "variants": list(VARIANTS),
            "prompt_variant_ids": PROMPT_VARIANT_IDS,
        },
        "thinking_probe": str(c.path("thinking_probe", "json")),
        "route_proof": str(c.path("route_proof", "json")),
        "concurrency_per_suite": c.concurrency,
        "spend_cap_per_suite": c.spend_cap,
    }
    c.path("LAUNCH", "json").write_text(json.dumps(top, indent=2) + "\n")
    for s in c.suites:
        sroot = c.root / s
        sroot.mkdir(parents=True, exist_ok=True)
        units = plan_units(s, pilot=c.pilot)
        rel, fsha = FORKS[s]
        per = {
            "suite": s,
            "pilot": c.pilot,
            "attempt": c.attempt,
            "git_sha": c.git_sha,
            "top_level": str(c.root / top_name),
            "proxy_url": c.proxy_url,
            "runs_roots": {pin: str(sroot / pin) for pin, _ in PINS},
            "forks_file": {"path": rel, "sha256": fsha},
            "pilot_points": pilot_points(s, c.pilot) if c.pilot else None,
            "withdrawn": WITHDRAWN[s],
            "concurrency": c.concurrency,
            "spend_cap_usd": c.spend_cap,
            "plan": {"n_units": len(units), "counts": counts(units), "units": units},
            "estimate": estimate(units, c.concurrency, costs),
        }
        attempt_path(sroot, "LAUNCH", "json", c.attempt).write_text(
            json.dumps(per, indent=2) + "\n"
        )
    if c.pilot:
        (c.root / "PILOT_ROOT_NOT_FOR_ANALYSIS.txt").write_text(
            f"pilot of {c.pilot} fork points per cell, seeds {list(PILOT_SEEDS)}; every unit "
            "carries pilot_flag=true. Never read by the analysis; never resumed as a campaign.\n"
        )


def stage2(c: Stage2Config, services: Any, log: Any = print) -> int:
    """Tunnel, proxy, probe and route proof ONCE; then every suite in parallel; then teardown.

    Returns 8 if the preflight failed (no suite started), else 0 if every suite returned 0, else
    the largest suite return code. Every suite is waited for before anything is stopped.
    """
    ok_mark, fail_mark = c.path("PREFLIGHT_OK", "txt"), c.path("PREFLIGHT_FAILED", "txt")
    try:
        try:
            tunnel_pid = services.start_tunnel()
            proxy_pid = services.start_proxy()
            services.probe()
            services.route_proof()
            write_launch_records(c, tunnel_pid, proxy_pid)
        except BaseException as exc:
            fail_mark.write_text(f"{type(exc).__name__}: {exc}\n")
            log(f"stage2: PREFLIGHT FAILED: {type(exc).__name__}: {exc}")
            if isinstance(exc, (PreflightFailed, PlanError, OSError)):
                return 8
            raise
        ok_mark.write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ\n", time.gmtime()))
        log(f"stage2: preflight passed; starting {c.suites}")
        handles = {s: services.start_suite(s) for s in c.suites}
        rcs = {s: services.wait(h) for s, h in handles.items()}
        c.path("STAGE2_RESULT", "json").write_text(json.dumps(rcs, indent=2) + "\n")
        log(f"stage2: suites returned {rcs} (0 ok, 10 spend cap, 11 abort, 12 incomplete)")
        return max(rcs.values()) if any(rcs.values()) else 0
    finally:
        services.stop_all()


def _stage2_main(a: argparse.Namespace) -> int:
    c = Stage2Config(
        suites=a.suites.split(","),
        root=Path(a.root),
        cache_root=Path(a.cache_root),
        proxy_url=a.proxy_url,
        serve=a.serve,
        ssh_login=a.ssh_login,
        serve_record=Path(a.serve_record),
        concurrency=a.concurrency,
        spend_cap=a.spend_cap,
        pilot=a.pilot,
        attempt=a.attempt,
        git_sha=a.git_sha,
    )

    def _term(signum: int, _frame: Any) -> None:
        raise SystemExit(128 + signum)  # so `finally` tears the tunnel and proxy down

    # SIGTERM only. SIGHUP stays IGNORED, as nohup left it: handling it here would let the
    # launching shell's exit end the campaign, which is exactly what nohup is there to prevent.
    signal.signal(signal.SIGTERM, _term)
    return stage2(c, Services(c), log=lambda s: print(s, flush=True))


# ------------------------------------------------------------------------------ probes
def route_proof(
    proxy_url: str, models: Sequence[str], api_key: str, attempts: int = 3
) -> list[dict[str, Any]]:
    """One 1-token completion per gateway role through the proxy. Only a completion proves a route;
    /v1/models lies in both directions. 404/401/403 are permanent and not retried."""
    import urllib.error
    import urllib.request

    out = []
    for m in models:
        rec: dict[str, Any] = {"model": m, "http": 0, "ok": False, "body_head": ""}
        for i in range(attempts):
            body = json.dumps(
                {"model": m, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1}
            ).encode()
            req = urllib.request.Request(
                proxy_url.rstrip("/") + "/v1/chat/completions",
                data=body,
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=120) as r:  # noqa: S310 - loopback proxy
                    rec["http"], text = int(r.status), r.read().decode("utf-8", "replace")
            except urllib.error.HTTPError as e:
                rec["http"], text = (
                    int(e.code),
                    (e.read().decode("utf-8", "replace") if e.fp else ""),
                )
            except Exception as e:  # noqa: BLE001 - a transport failure is a retry, then a refusal
                rec["http"], text = 0, f"{type(e).__name__}: {e}"
            rec["body_head"] = re.sub(r"sk-[A-Za-z0-9._-]+", "sk-<redacted>", text[:200])
            if rec["http"] == 200:
                rec["ok"] = True
                break
            if rec["http"] in (401, 403, 404):
                break
            if i + 1 < attempts:
                time.sleep(5)
        out.append(rec)
    return out


# ------------------------------------------------------------------------------ CLI
def _print_plan(
    suite: str,
    concurrency: int,
    spend_cap: float | None,
    pilot: int | None,
    shard_text: str | None = None,
) -> None:
    shard = parse_shard(shard_text)
    whole = plan_units(suite, pilot=pilot)
    units = plan_units(suite, pilot=pilot, shard=shard)
    pts = pilot_points(suite, pilot) if pilot else load_points(suite)
    costs = json.loads(COSTS_JSON.read_text())
    label = f"PILOT of {pilot} fork points, seeds {list(PILOT_SEEDS)}" if pilot else "full"
    print(
        f"{suite} [{label}]: {len(pts)} fork points over {len({p['task_id'] for p in pts})} "
        f"tasks, variants {list(VARIANTS)}, max_turns {MAX_TURNS}, budget_cap {BUDGET_CAP}"
    )
    print(f"forks file {FORKS[suite][0]} sha256 {FORKS[suite][1]}")
    if shard:
        print(
            f"SHARD {shard[0]}/{shard[1]}: {len(units)} of {len(whole)} units, "
            f"{len({_slot(u) for u in units})} of {len({_slot(u) for u in whole})} slots"
        )
    print("units:")
    for u in units:
        print(
            f"  {u['pin']:<38} {u['arm_id']:<18} {u['variant']:<9} seed={u['seed']} "
            f"task={u['task_id']:<4} k={u['k']:<3} trace={u['trace_sha'][:16]}"
        )
    print(f"counts (pin | arm_id | variant), {suite}:")
    for k, v in counts(units).items():
        print(f"  {k:<70} {v}")
    print(
        f"  TOTAL {len(units)}  (withdrawn: {WITHDRAWN[suite]['n_units']} over 2 arms at "
        f"{WITHDRAWN[suite]['code_version'][:8]})"
        + (f"  [shard {shard[0]}/{shard[1]} of {len(whole)}]" if shard else "")
    )
    e = estimate(units, concurrency, costs)
    print(f"estimate, {suite}, each unit costed from its point's withdrawn mean:")
    for r in e["rows"]:
        print(
            f"  {r['pin']:<38} {r['variant']:<9} n={r['n']:<4} metered ${r['usd_metered']:>7.2f}  "
            f"billed ${r['usd_billed']:>7.2f}  unit-wall {r['wall_h']:>6.2f}h  from {r['costed_from']}"
        )
    print(
        f"  TOTAL metered ${e['usd_metered']:.2f}  billed ${e['usd_billed']:.2f}  wall at "
        f"concurrency {concurrency}: {e['wall_h_pooled']:.2f}h (upper {e['wall_h_upper']:.2f}h)"
    )
    if spend_cap is not None:
        print(
            f"  spend cap ${spend_cap:.2f} = {spend_cap / e['usd_metered']:.2f}x the metered estimate"
        )
    for n in (2, 4):
        pe = estimate(plan_units(suite, pilot=n), concurrency, costs)
        print(
            f"  pilot N={n}: {6 * n} units, metered ${pe['usd_metered']:.2f}  billed "
            f"${pe['usd_billed']:.2f}  wall {pe['wall_h_upper']:.2f}h (upper) at concurrency {concurrency}"
        )
    print("plan-only: nothing launched")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("plan")
    p.add_argument("--suite", required=True, choices=SUITES)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--spend-cap", type=float, default=None)
    p.add_argument("--pilot", type=int, default=None)
    p.add_argument("--shard", default=None, help="K/N, 1-based")

    p = sub.add_parser("stats")
    p.add_argument("--records", nargs="+", required=True)
    p.add_argument("--out", required=True)

    p = sub.add_parser("check")
    p.add_argument("--suite", required=True, choices=SUITES)
    p.add_argument("--config", required=True)
    p.add_argument("--price-table", required=True)
    p.add_argument("--trace-root", required=True)

    p = sub.add_parser("tunnel-port")
    p.add_argument("--config", required=True)

    p = sub.add_parser("serve-record-check")
    p.add_argument("--record", required=True)
    p.add_argument("--serve", required=True)

    p = sub.add_parser("resume-check")
    for f in ("launch-json", "git-sha", "suites", "proxy-url", "cache-root", "serve-record"):
        p.add_argument(f"--{f}", required=True)
    p.add_argument("--pilot", type=int, default=None)

    p = sub.add_parser("stage2")
    for f in ("suites", "root", "cache-root", "proxy-url", "serve", "ssh-login", "serve-record",
              "attempt", "git-sha"):  # fmt: skip
        p.add_argument(f"--{f}", required=True)
    p.add_argument("--concurrency", type=int, required=True)
    p.add_argument("--spend-cap", type=float, required=True)
    p.add_argument("--pilot", type=int, default=None)

    p = sub.add_parser("run")
    p.add_argument("--suite", required=True, choices=SUITES)
    for f in ("runs-root", "cache-root", "proxy-url", "price-table"):
        p.add_argument(f"--{f}", required=True)
    p.add_argument("--concurrency", type=int, required=True)
    p.add_argument("--spend-cap", type=float, required=True)
    p.add_argument("--pilot", type=int, default=None)
    p.add_argument("--max-consecutive-failures", type=int, default=6)
    p.add_argument("--shard", default=None, help="K/N, 1-based; omit for the whole suite")
    p.add_argument("--probe-record", required=True, help="thinking_probe.py --out, must be PASS")
    p.add_argument("--serve-record", required=True, help="the serve's SERVED.<job>.json")
    p.add_argument("--serve", default=None, help="host:port the record must name, if given")
    p.add_argument("--job-label", default=None, help="default <LSB_JOBID>[.<LSB_JOBINDEX>]")

    a = ap.parse_args(argv)
    try:
        if a.cmd == "plan":
            _print_plan(a.suite, a.concurrency, a.spend_cap, a.pilot, a.shard)
            return 0
        if a.cmd == "stats":
            recs, sources = [], []
            for f in a.records:
                raw = Path(f).read_bytes()
                rs = json.loads(raw)
                recs += rs
                sources.append(
                    {
                        "file": Path(f).name,
                        "sha256": hashlib.sha256(raw).hexdigest(),
                        "n": len(rs),
                        "population": population_summary(rs, shards=12),
                    }
                )
            doc = {
                "_what": "per-unit cost and wall of the withdrawn tau2 transfer campaign, one "
                "code_version per suite; written by `rerun.py stats`",
                "sources": sources,
                "cells": history_stats(recs),
                "points": point_stats(recs),
            }
            Path(a.out).write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
            print(f"wrote {a.out}: {len(doc['cells'])} cells, {len(doc['points'])} points")
            return 0
        if a.cmd == "tunnel-port":
            print(local_route_port(Path(a.config).read_text(), [pin for pin, _ in PINS]))
            return 0
        if a.cmd == "serve-record-check":
            probs = serve_record_problems(json.loads(Path(a.record).read_text()), a.serve)
            for pr in probs:
                print(f"serve-record: {pr}", file=sys.stderr)
            return 6 if probs else 0
        if a.cmd == "resume-check":
            rec = json.loads(Path(a.serve_record).read_text())
            now = {
                "git_sha": a.git_sha,
                "suites": a.suites.split(","),
                "proxy_url": a.proxy_url,
                "cache_root": a.cache_root,
                "pilot": a.pilot,
                "adapters": {
                    p: (rec.get("adapters") or {}).get(p, {}).get("sha256") for p in TRAINED
                },
            }
            probs = resume_problems(json.loads(Path(a.launch_json).read_text()), now)
            for pr in probs:
                print(f"resume: {pr}", file=sys.stderr)
            return 3 if probs else 0
        if a.cmd == "check":
            problems = []
            try:
                local_route_port(Path(a.config).read_text(), [pin for pin, _ in PINS])
            except PlanError as e:
                problems.append(str(e))
            miss = price_table_missing(
                Path(a.price_table), [p for p, _ in PINS] + list(FROZEN_ROLES.values())
            )
            if miss:
                problems.append(f"price table {a.price_table} has no row for {miss}")
            tr = missing_traces(load_points(a.suite), Path(a.trace_root))
            if tr:
                problems.append(
                    f"{len(tr)} fork traces absent under {a.trace_root}: {[s[:16] for s in tr]}"
                )
            import pinq  # the interpreter must resolve THIS tree's package, not an installed copy

            if not str(Path(pinq.__file__).resolve()).startswith(str((ROOT / "src").resolve())):
                problems.append(f"pinq resolves to {pinq.__file__}, not {ROOT / 'src'}")
            problems += prompt_variant_problems()
            for pr in problems:
                print(f"check: {pr}", file=sys.stderr)
            if not problems:
                print(
                    f"check: {a.suite} config routes, price table, {len(load_points(a.suite))} "
                    "traces, pinq path and prompt variants ok"
                )
            return 6 if problems else 0
        if a.cmd == "stage2":
            return _stage2_main(a)
        if a.cmd == "run":
            return run_campaign(a)
    except PlanError as e:
        print(f"rerun: REFUSING: {e}", file=sys.stderr)
        return 6
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
