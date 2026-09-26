"""Run the prompted gpt-oss-120b questioner arm on the STAGED ee9afda2 harness (RULES amendment 11).

WHY A DRIVER OUTSIDE THE STAGED TREE. The fixed campaign's 1224 units ran at ee9afda2, staged
clean on HPC. The new arm must carry the same code_version, so the staged tree cannot be edited --
not even rerun.py's PINS, which would dirty it (every run stamped `dev-`) or move its HEAD (a second
code_version). This file lives anywhere else, imports the staged tree's own rerun.py
(`--tree`), and adds exactly one pin at runtime:

    ("openai/aws/gpt-oss-120b", "inquirer_prompted")   -- the 8B comparator's arm and prompt,
                                                          a larger prompted model as questioner

It enumerates ONLY that pin's units, with rerun.py's own planner: 34 fork points x 2 prompt
variants x 3 seeds = 204 per suite, 408 in all, sharded exactly as the campaign (`--shard K/N`
covers the same slots campaign shard K/N did). Every UnitSpec is built by rerun.py's
`build_work`, hence by the staged `run_tau2_forks.build_specs`, and run through rerun.py's own
`drive` (spend cap, first-unit gate, consecutive-failure breaker, per-unit manifest readback) and
`_unit`. The one difference is `runs_root`: this arm's units go to `<runs-root>/STORE_DIR/`,
a single directory name rather than the pin's slashes, in a root of its own (amendment 11: never
mixed into the ee9afda2 launch record). runs_root is not in run identity.

THE CODE VERSION IS THE STAGED TREE'S. It comes from `git_info(<tree>)`, through rerun.py's own
`_git_state`, never from where this file sits, and `run` refuses unless it equals
`--campaign-sha` (default ee9afda2) and the tree is clean. `run` also refuses unless `pinq`,
`pi_run` and `pinq_expt` are imported from `<tree>/src` -- otherwise another harness would run
under the staged tree's code_version, and nothing downstream could tell.

THE QUESTIONER MUST NOT BE TRUNCATED. gpt-oss-120b spends reasoning tokens inside max_tokens.
`questioner-probe` sends ONE real questioner request -- the arm's actual prompt on a planned
unit's actual first state, through the harness's own MeteredClient and the campaign proxy, at the
questioner's own max_tokens -- and refuses unless the content is non-empty and
completion_tokens < 0.9 x max_tokens. It records the reasoning-token count and the LSF job that
wrote it. It runs on HPC inside the job that will use it, never in a test: tests inject the
transport.

WHICH RECORDS A JOB NEEDS DEPENDS ON THE REGIME (`check_records`). If no planned pin is a
`hosted_vllm` route of `--proxy-config`, the job is GATEWAY-ONLY: no role in a unit calls a Qwen
route, so the Qwen thinking probe and the serve record say nothing about it (and the job needs no
GPU); it requires only a questioner-probe PASS written by THIS job (same LSB_JOBID/LSB_JOBINDEX,
or, outside LSF, written after --job-start-epoch) against THIS proxy (byte-equal URL), for the same
model, max_tokens and harness sha. If any planned pin is local, rerun.py's full requirements apply
as well. The regime is written into ARM_LAUNCH.json and JOB.json.

This file's own sha256 is written into every job's JOB.json and into the arm's ARM_LAUNCH.json,
which every shard of one suite must agree with (same driver, harness sha, proxy, pilot).

    python run_extra_pin.py plan --tree <staged> [--suite S] [--shard K/N]
    python run_extra_pin.py questioner-probe --tree <staged> --suite S --proxy-url U --out F
    python run_extra_pin.py run --tree <staged> --suite S --shard K/N --runs-root R ...
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import importlib
import json
import os
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Callable, Iterator, Mapping, Sequence

EXTRA_PIN = "openai/aws/gpt-oss-120b"
EXTRA_ARM = "inquirer_prompted"
CAMPAIGN_SHA = "ee9afda24c73065715e685541fb423e22d3e7688"
STORE_DIR = "openai__aws__gpt-oss-120b"
CAP_FRACTION = 0.9
HARNESS_PACKAGES = ("pinq", "pi_run", "pinq_expt")


class ExtraPinError(RuntimeError):
    """A precondition of the extra arm does not hold; nothing may run."""


def driver_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


# ------------------------------------------------------------------------------ the staged tree
def load_rerun(tree: Path) -> ModuleType:
    """The staged tree's rerun.py, imported under the module name `rerun`.

    A real module name on sys.path, not a path-loaded one, because rerun's `drive` pickles
    `rerun._unit` into pool workers, which import it by name. Loaded afresh on each call, so the
    module is always the one under `tree`."""
    path = tree.resolve() / "scripts" / "tau2_rerun" / "rerun.py"
    if not path.is_file():
        raise ExtraPinError(f"no rerun.py at {path}; --tree must be the staged harness tree")
    d = str(path.parent)
    sys.path[:] = [d, *(p for p in sys.path if p != d)]
    sys.modules.pop("rerun", None)
    importlib.invalidate_caches()
    rr = importlib.import_module("rerun")
    if Path(rr.__file__).resolve() != path:
        raise ExtraPinError(f"`rerun` resolved to {rr.__file__}, not {path}")
    return rr


def harness_src_problems(tree: Path) -> list[str]:
    """The harness packages must be the staged tree's: its git sha is what every manifest will
    carry as code_version, so any other src would run under a label that is not its own."""
    src = (tree.resolve() / "src").resolve()
    out = []
    for name in HARNESS_PACKAGES:
        mod = importlib.import_module(name)
        origin = Path(mod.__file__).resolve()
        if src not in origin.parents:
            out.append(f"{name} is imported from {origin}, not from {src}; set PYTHONPATH={src}")
    return out


@contextlib.contextmanager
def extra_pin_only(rr: ModuleType) -> Iterator[None]:
    """rerun.PINS is exactly the extra pin for the duration -- so the staged planner enumerates
    that pin's units and no other -- and is restored after, whatever happens."""
    saved = rr.PINS
    rr.PINS = ((EXTRA_PIN, EXTRA_ARM),)
    try:
        yield
    finally:
        rr.PINS = saved


def plan(rr: ModuleType, suite: str, *, pilot: int | None = None, shard: Any = None) -> list:
    with extra_pin_only(rr):
        return rr.plan_units(suite, pilot=pilot, shard=shard)


def build(
    rr: ModuleType,
    suite: str,
    runs_root: Path,
    cache_root: Path,
    gi: Any,
    *,
    pilot: int | None = None,
    shard: Any = None,
) -> list[tuple[Any, str, str]]:
    """rerun.build_work for the extra pin alone, with runs_root moved to `<runs-root>/STORE_DIR`
    -- the only field that differs from what rerun.py would build with the pin in PINS."""
    with extra_pin_only(rr):
        work = rr.build_work(suite, runs_root, cache_root, gi, pilot=pilot, shard=shard)
    store = str(runs_root / STORE_DIR)
    return [(dataclasses.replace(spec, runs_root=store), pin, v) for spec, pin, v in work]


# ------------------------------------------------------------------------------ the preflight
def first_state_messages(prefix: Sequence[Any], is_valid: Callable[[Any], bool]) -> list[Any]:
    """The DriverState messages at the agent's first rollout of a fork whose prefix ends on a
    customer message: tau2's Orchestrator hands the agent `history[:-1]` filtered by
    `is_valid_agent_history_message`, then the driver appends the last message as it arrives."""
    last = prefix[-1] if prefix else None
    if last is None or getattr(last, "role", "") != "user" or getattr(last, "tool_calls", None):
        raise ExtraPinError("the prefix does not end on a customer message; its first state needs "
                            "the user simulator and cannot be built without a call")  # fmt: skip
    return [m for m in prefix[:-1] if is_valid(m)] + [last]


def probe_unit(
    rr: ModuleType,
    suite: str,
    load_prefix: Callable[[Mapping[str, Any]], Sequence[Any]],
    shard: Any,
) -> tuple[Mapping[str, Any], Sequence[Any]]:
    """The first planned unit, in the planner's sorted slot order, whose prefix ends on a
    customer message (all 68 recorded fork points do, measured 2026-09-23)."""
    for u in sorted(plan(rr, suite, shard=shard), key=rr._slot):
        prefix = load_prefix(u)
        last = prefix[-1] if prefix else None
        if last is not None and last.role == "user" and not getattr(last, "tool_calls", None):
            return u, prefix
    raise ExtraPinError(f"no planned {suite} unit has a prefix ending on a customer message")


class _Capture:
    """The client the questioner holds during the probe: every call goes to the harness's own
    MeteredClient; the payload, text and telemetry are kept for the record."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.calls: list[dict[str, Any]] = []

    def complete(self, **kw: Any) -> Any:
        payload = self._inner.request_payload(**kw)
        text, tel = self._inner.complete(**kw)
        self.calls.append({"payload": payload, "text": text, "tel": tel})
        return text, tel

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def questioner_probe(
    *,
    unit: Mapping[str, Any],
    prefix: Sequence[Any],
    suite_obj: Any,
    runner: ModuleType,
    arms: ModuleType,
    client: Any,
    ledger: Any,
    is_valid: Callable[[Any], bool],
    proxy_url: str,
) -> dict[str, Any]:
    """ONE questioner call on the unit's first state, exactly as the harness makes it: the arm
    built by `arms.build`, the variant applied by `apply_prompt_variant`, the view by
    `dialogue_view`, then `reset(view, seed)` and `act(State(view=view))` -- whose `_complete`
    asks `client` for the questioner's own max_tokens. `unit_chars` is not passed to `build`:
    it reaches only drafters and the answerer, never an Inquirer (arms.build's docstring)."""
    from pinq.types import State

    tid = str(unit["task_id"])
    arm = arms.get(EXTRA_ARM)
    cap = _Capture(client)
    c = arms.build(arm, llm=cap, retriever=None, recorder=ledger, tools=suite_obj.tool_schemas(tid))
    variant_id = runner.apply_prompt_variant(c.inquirer, str(unit["variant"]))
    view = runner.dialogue_view(suite_obj, tid, first_state_messages(prefix, is_valid))
    c.inquirer.reset(view, int(unit["seed"]))
    action = c.inquirer.act(State(view=view))
    if len(cap.calls) != 1:
        raise ExtraPinError(f"the questioner made {len(cap.calls)} calls on its first state, not 1")
    call = cap.calls[0]
    payload, text, tel = call["payload"], str(call["text"]), call["tel"]
    max_tokens = int(payload["max_tokens"])
    prompt = str(payload["messages"][-1]["content"])
    problems = []
    if not text.strip():
        problems.append("the content is empty")
    if tel.tok_completion >= CAP_FRACTION * max_tokens:
        problems.append(
            f"completion_tokens {tel.tok_completion} >= {CAP_FRACTION} x max_tokens {max_tokens}"
        )
    return {
        "verdict": "REFUSE" if problems else "PASS",
        "problems": problems,
        "model": payload["model"],
        "proxy_url": proxy_url,
        "max_tokens": max_tokens,
        "content_budget": _act_max_tokens(),
        "reasoning_headroom": round(max_tokens / _act_max_tokens(), 4),
        "completion_tokens": tel.tok_completion,
        "reasoning_tokens": tel.tok_reasoning,
        "prompt_tokens": tel.tok_prompt,
        "usd": tel.usd,
        "content_chars": len(text),
        "content_preview": text[:160],
        "action": type(action).__name__,
        "malformed": int(getattr(c.inquirer, "malformed", 0)),
        "prompt_variant_id": variant_id,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "prompt_chars": len(prompt),
        "request_sha": tel.request_sha,
        "unit": dict(unit),
    }


def _act_max_tokens() -> int:
    from pinq_expt.policies.base import ACT_MAX_TOKENS

    return int(ACT_MAX_TOKENS)


def questioner_record_problems(
    rec: Mapping[str, Any], *, proxy_url: str, max_tokens: int, harness_sha: str
) -> list[str]:
    """The questioner-probe record a `run` is launched under: a PASS of THIS model, through THIS
    proxy, at THIS max_tokens (a different reasoning headroom is a different request), on THIS
    harness sha."""
    want = {
        "verdict": "PASS",
        "model": EXTRA_PIN,
        "proxy_url": proxy_url,
        "max_tokens": max_tokens,
        "harness_sha": harness_sha,
    }
    return [
        f"questioner probe {k}={rec.get(k)!r}, this run needs {v!r}"
        for k, v in want.items()
        if rec.get(k) != v
    ]


# ------------------------------------------------------------------------------ the regime
GATEWAY_ONLY = "gateway-only: questioner probe"
LOCAL = "local: thinking probe + serve record + questioner probe"


def local_routes(config_text: str) -> set[str]:
    """The model names the proxy config serves from a LOCAL vLLM (`hosted_vllm/*`). Everything
    else it routes to the gateway."""
    import yaml

    doc = yaml.safe_load(config_text) or {}
    return {
        str(m.get("model_name"))
        for m in doc.get("model_list") or []
        if str((m.get("litellm_params") or {}).get("model") or "").startswith("hosted_vllm/")
    }


def regime_for(pins: Sequence[str], local: set[str]) -> str:
    """GATEWAY-ONLY when no planned questioner pin is served locally: then no role in a unit
    calls a Qwen route (the tool retriever's selector uses the DRAFTER's client, and the drafter,
    answerer and user simulator are gateway -- verified at ee9afda2, tool_retriever.py:216), so
    neither the Qwen thinking probe nor a serve record says anything about these units, and the
    job needs no GPU. LOCAL as soon as one planned pin is a hosted_vllm route."""
    return LOCAL if set(pins) & local else GATEWAY_ONLY


def _idx(v: Any) -> str:
    s = str(v or "").strip()
    return "" if s in ("", "0") else s


def from_this_job(
    rec: Mapping[str, Any], path: Path, *, env: Mapping[str, str], job_start: float | None
) -> list[str]:
    """A record copied from another job is never accepted. Inside LSF the record must carry this
    job's LSB_JOBID and LSB_JOBINDEX (the probe writes both). Outside LSF it must carry no job id
    and be written after `job_start` (--job-start-epoch), which is then required."""
    jid = str(env.get("LSB_JOBID") or "")
    got = (str(rec.get("LSB_JOBID") or ""), _idx(rec.get("LSB_JOBINDEX")))
    if jid:
        want = (jid, _idx(env.get("LSB_JOBINDEX")))
        return [] if got == want else [f"the questioner probe record is from job {got}, not {want}"]
    if got[0]:
        return [f"the questioner probe record is from LSF job {got}; this process is not in it"]
    if job_start is None:
        return ["outside LSF a record's freshness needs --job-start-epoch; none was given"]
    if path.stat().st_mtime < job_start:
        return [f"the questioner probe record ({path}) predates this job's start"]
    return []


def check_records(
    a: argparse.Namespace,
    rr: ModuleType,
    pins: Sequence[str],
    local: set[str],
    *,
    env: Mapping[str, str],
    job_start: float | None,
    harness_sha: str,
    max_tokens: int,
) -> dict[str, Any]:
    """Which records this job must hold, by regime, all checked before git or any spend.

    LOCAL keeps rerun.py's full requirements -- `check_run_records` (a thinking-probe PASS for
    this proxy and a serve record vouching for s1/s2) -- and the probe must cover every local
    planned pin. Both regimes require, for a gateway questioner pin, a fresh questioner-probe
    PASS written by THIS job against THIS proxy (byte-equal URL)."""
    for name in ("runs_root", "cache_root"):
        if not Path(getattr(a, name)).is_absolute():
            raise ExtraPinError(f"--{name.replace('_', '-')} must be absolute")
    regime = regime_for(pins, local)
    out: dict[str, Any] = {"regime": regime, "probe": None, "serve": None, "qrec": None}
    if regime == LOCAL:
        if not a.probe_record or not a.serve_record:
            raise ExtraPinError(
                f"planned pins {sorted(set(pins) & local)} are local: the Qwen thinking-probe "
                "record and the serve record are required (--probe-record, --serve-record)"
            )
        probe, serve = rr.check_run_records(a)
        covered = {str(r.get("model")) for r in probe.get("routes") or []}
        missing = sorted((set(pins) & local) - covered)
        if missing:
            raise ExtraPinError(f"the thinking probe does not cover the local pins {missing}")
        out.update(probe=probe, serve=serve)
    gateway_questioners = sorted(set(pins) - local)
    if gateway_questioners:
        if not a.questioner_probe_record:
            raise ExtraPinError(
                f"{gateway_questioners} run through the gateway: a questioner-probe PASS written "
                "by this job is required (--questioner-probe-record)"
            )
        qpath = Path(a.questioner_probe_record)
        qrec = json.loads(qpath.read_text())
        probs = questioner_record_problems(
            qrec, proxy_url=a.proxy_url, max_tokens=max_tokens, harness_sha=harness_sha
        )
        probs += from_this_job(qrec, qpath, env=env, job_start=job_start)
        if probs:
            raise ExtraPinError("; ".join(probs))
        out["qrec"] = qrec
    return out


# ------------------------------------------------------------------------------ the run
def _arm_launch(path: Path, fields: Mapping[str, Any]) -> None:
    """Create the arm's launch record once per suite root, atomically; every later shard must
    agree with it on everything that decides which population it belongs to."""
    try:
        with open(path, "x") as f:
            json.dump(dict(fields), f, indent=2)
            f.write("\n")
        return
    except FileExistsError:
        pass
    have = json.loads(path.read_text())
    keys = ("driver_sha256", "harness_sha", "pin", "arm_id", "proxy_url", "pilot", "store_dir",
            "regime")  # fmt: skip
    bad = {k: (have.get(k), fields[k]) for k in keys if have.get(k) != fields[k]}
    if bad:
        raise ExtraPinError(f"{path} was written for another population: {bad}")


def _stage_job(runs_root: Path, label: str, records: Mapping[str, str | None]) -> Path:
    """`<runs-root>/_jobs/<label>/` with this job's own copies of the records it ran under.
    Refused if it exists: two jobs never write one record."""
    d = runs_root / "_jobs" / label
    if d.exists():
        raise ExtraPinError(f"{d} already exists; a job's records are never overwritten")
    d.mkdir(parents=True)
    for name, src in records.items():
        if src:
            (d / name).write_bytes(Path(src).read_bytes())
    return d


def run_extra(
    a: argparse.Namespace,
    rr: ModuleType,
    *,
    executor: Any = None,
    unit_fn: Any = None,
    env: Mapping[str, str] | None = None,
    log: Callable[[str], None] = print,
) -> int:
    """rerun.run_campaign's sequence, step for step, for the extra pin; the records it requires
    are decided by the regime (`check_records`)."""
    from concurrent.futures import ProcessPoolExecutor

    from pi_run.sweep import billed_usd
    from pinq.budget import BudgetLedger
    from pinq_adapters.llm.litellm_client import MeteredClient

    env = os.environ if env is None else env
    shard = rr.parse_shard(a.shard)
    runs_root, cache_root = Path(a.runs_root), Path(a.cache_root)
    mixed = [p for p, _ in rr.PINS if (runs_root / p).exists()]
    if mixed:
        raise ExtraPinError(f"{runs_root} holds campaign pins {mixed}; this arm has its own root")
    bad = rr.check_env(a.proxy_url, a.price_table)
    if bad:
        raise ExtraPinError("; ".join(bad))
    if a.spend_cap <= 0:
        raise ExtraPinError("--spend-cap must be > 0: a cap of 0 still bills one unit per worker")
    from pi_run.stages.tau2_runner import TRACE_ROOT

    design = rr.run_design_problems(a.suite, Path(TRACE_ROOT), Path(a.price_table))
    design += [
        f"price table has no row for {m}"
        for m in rr.price_table_missing(Path(a.price_table), [EXTRA_PIN])
    ]
    if design:
        raise ExtraPinError("; ".join(design))
    gi = rr._forks_module()._git_state()
    if gi.dirty:
        raise ExtraPinError(f"staged tree is dirty ({list(gi.dirty_files)[:5]})")
    if gi.sha != a.campaign_sha:
        raise ExtraPinError(f"staged tree is at {gi.sha}, not the campaign sha {a.campaign_sha}")
    max_tokens = MeteredClient(BudgetLedger(cap=1))._budget(_act_max_tokens())
    units = plan(rr, a.suite, pilot=a.pilot, shard=shard)
    config = Path(a.proxy_config)
    rec = check_records(
        a,
        rr,
        sorted({u["pin"] for u in units}),
        local_routes(config.read_text()),
        env=env,
        job_start=a.job_start_epoch,
        harness_sha=gi.sha,
        max_tokens=max_tokens,
    )
    work = build(rr, a.suite, runs_root, cache_root, gi, pilot=a.pilot, shard=shard)
    me = driver_sha256()
    runs_root.mkdir(parents=True, exist_ok=True)
    _arm_launch(
        runs_root / "ARM_LAUNCH.json",
        {
            "driver": str(Path(__file__).resolve()),
            "driver_sha256": me,
            "tree": str(rr.ROOT),
            "harness_sha": gi.sha,
            "rerun_py_sha256": hashlib.sha256(Path(rr.__file__).read_bytes()).hexdigest(),
            "pin": EXTRA_PIN,
            "arm_id": EXTRA_ARM,
            "suite": a.suite,
            "store_dir": STORE_DIR,
            "proxy_url": a.proxy_url,
            "pilot": a.pilot,
            "regime": rec["regime"],
            "rules": "artifacts/tau2_rerun_20260923/RULES.md amendment 11",
            "plan": plan(rr, a.suite, pilot=a.pilot),
        },
    )
    label = f"gptoss120b-{a.job_label or rr.job_label(env, shard)}"
    local = rec["regime"] == LOCAL
    job = _stage_job(
        runs_root,
        label,
        {
            "questioner_probe.json": a.questioner_probe_record,
            "thinking_probe.json": a.probe_record if local else None,
            "serve_record.json": a.serve_record if local else None,
        },
    )
    qrec, serve = rec["qrec"] or {}, rec["serve"] or {}
    (job / "JOB.json").write_text(
        json.dumps(
            {
                "label": label,
                "regime": rec["regime"],
                "driver": str(Path(__file__).resolve()),
                "driver_sha256": me,
                "tree": str(rr.ROOT),
                "harness_sha": gi.sha,
                "pin": EXTRA_PIN,
                "arm_id": EXTRA_ARM,
                "store": str(runs_root / STORE_DIR),
                "suite": a.suite,
                "shard": a.shard,
                "pilot": a.pilot,
                "host": os.uname().nodename,
                "LSB_JOBID": env.get("LSB_JOBID"),
                "LSB_JOBINDEX": env.get("LSB_JOBINDEX"),
                "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "proxy_url": a.proxy_url,
                "proxy_config": {
                    "path": str(config),
                    "sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
                },
                "cache_root": str(cache_root),
                "price_table": {
                    "path": a.price_table,
                    "sha256": hashlib.sha256(Path(a.price_table).read_bytes()).hexdigest(),
                },
                "questioner_max_tokens": max_tokens,
                "questioner_probe": {
                    k: qrec.get(k)
                    for k in ("verdict", "completion_tokens", "reasoning_tokens", "max_tokens")
                },
                "served_adapters": {
                    p: (serve.get("adapters") or {}).get(p, {}).get("sha256") for p in rr.TRAINED
                }
                if local
                else None,
                "concurrency": a.concurrency,
                "spend_cap_usd": a.spend_cap,
                "n_units": len(units),
                "units": units,
            },
            indent=2,
        )
        + "\n"
    )
    log(f"run: {a.suite} {EXTRA_PIN} {len(work)} units, shard {a.shard}, {rec['regime']}, "
        f"cap ${a.spend_cap:.2f}, harness {gi.sha[:12]}, driver {me[:12]}, job {job}")  # fmt: skip
    t0 = time.time()
    tally, billed, stop = rr.drive(
        work,
        executor or ProcessPoolExecutor(max_workers=a.concurrency),
        unit_fn or rr._unit,
        spend_cap=a.spend_cap,
        max_consecutive_failures=a.max_consecutive_failures,
        code_version=gi.sha,
        billed_usd=billed_usd,
        pilot=bool(a.pilot),
        log=log,
    )
    ok = sum(v for (_p, _v, s), v in tally.items() if s == "ok")
    summary = {
        "suite": a.suite,
        "shard": a.shard,
        "label": label,
        "regime": rec["regime"],
        "pin": EXTRA_PIN,
        "harness_sha": gi.sha,
        "driver_sha256": me,
        "n_planned": len(work),
        "n_ok": ok,
        "billed_usd_this_attempt": round(billed, 4),
        "stop_reason": stop,
        "wall_h": round((time.time() - t0) / 3600, 3),
        "by_pin_variant_status": {"|".join(map(str, k)): v for k, v in sorted(tally.items())},
    }
    (job / "RUN_SUMMARY.json").write_text(json.dumps(summary, indent=2))
    log(json.dumps(summary, indent=2))
    if stop.startswith("abort"):
        return 11
    if stop:
        return 10
    return 0 if ok == len(work) else 12


# ------------------------------------------------------------------------------ CLI
def _probe_main(a: argparse.Namespace, rr: ModuleType) -> int:
    from tau2.agent.base_agent import is_valid_agent_history_message

    from pi_run.stages import tau2_runner as runner
    from pi_run.worker import load_suite
    from pinq.budget import BudgetLedger
    from pinq_adapters.llm.litellm_client import MeteredClient
    from pinq_expt import arms

    if os.environ.get("LITELLM_BASE_URL") != a.proxy_url:
        raise ExtraPinError(
            "LITELLM_BASE_URL is not --proxy-url; the probe must use the campaign proxy"
        )
    os.environ["PI_MODEL_INQUIRER"] = EXTRA_PIN  # what rerun._unit sets for this pin's units
    gi = rr._forks_module()._git_state()
    if gi.dirty or gi.sha != a.campaign_sha:
        raise ExtraPinError(
            f"staged tree at {gi.sha} dirty={gi.dirty}; need clean {a.campaign_sha}"
        )
    traces = str(rr.ROOT / runner.TRACE_ROOT)

    def load_prefix(u: Mapping[str, Any]) -> Sequence[Any]:
        spec = SimpleNamespace(foreign_trace_sha=u["trace_sha"], foreign_prefix_k=u["k"])
        return runner.load_prefix(spec, root=traces)

    unit, prefix = probe_unit(rr, a.suite, load_prefix, rr.parse_shard(a.shard))
    ledger = BudgetLedger(cap=rr.BUDGET_CAP)
    rec = questioner_probe(
        unit=unit,
        prefix=prefix,
        suite_obj=load_suite(a.suite, ""),
        runner=runner,
        arms=arms,
        client=MeteredClient(ledger, temperature=0.0),  # exactly as run_tau2_unit builds it
        ledger=ledger,
        is_valid=is_valid_agent_history_message,
        proxy_url=a.proxy_url,
    )
    rec.update(
        {
            "harness_sha": gi.sha,
            "driver_sha256": driver_sha256(),
            "tree": str(rr.ROOT),
            "at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            # WHICH JOB WROTE IT: `run` accepts only its own job's record (from_this_job).
            "LSB_JOBID": os.environ.get("LSB_JOBID"),
            "LSB_JOBINDEX": os.environ.get("LSB_JOBINDEX"),
        }
    )
    Path(a.out).write_text(json.dumps(rec, indent=2, default=str) + "\n")
    print(f"questioner-probe: {rec['verdict']} {EXTRA_PIN} completion {rec['completion_tokens']}"
          f"/{rec['max_tokens']} reasoning {rec['reasoning_tokens']} content {rec['content_chars']} "
          f"chars action {rec['action']}" + (f" -- {rec['problems']}" if rec["problems"] else ""))  # fmt: skip
    if rec["malformed"]:
        print("questioner-probe: WARNING the reply broke the JSON action contract (malformed)")
    return 0 if rec["verdict"] == "PASS" else 8


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "questioner-probe", "run"):
        p = sub.add_parser(name)
        p.add_argument("--tree", required=True, help="the staged harness tree (ee9afda2)")
        p.add_argument("--shard", default=None, help="K/N, 1-based")
        p.add_argument("--pilot", type=int, default=None)
        p.add_argument("--campaign-sha", default=CAMPAIGN_SHA)
        if name == "plan":
            p.add_argument("--suite", action="append", default=None)
        else:
            p.add_argument("--suite", required=True)
            p.add_argument("--proxy-url", required=True)
    p = sub.choices["questioner-probe"]
    p.add_argument("--out", required=True)
    p = sub.choices["run"]
    for f in ("runs-root", "cache-root", "price-table", "proxy-config"):
        p.add_argument(f"--{f}", required=True)
    # Required or not by REGIME (check_records): gateway-only needs the questioner probe alone.
    for f in ("questioner-probe-record", "probe-record", "serve-record"):
        p.add_argument(f"--{f}", default=None)
    p.add_argument("--job-start-epoch", type=float, default=None, help="outside LSF only")
    p.add_argument("--serve", default=None)
    p.add_argument("--concurrency", type=int, required=True)
    p.add_argument("--spend-cap", type=float, required=True)
    p.add_argument("--max-consecutive-failures", type=int, default=6)
    p.add_argument("--job-label", default=None)
    a = ap.parse_args(argv)

    tree = Path(a.tree)
    try:
        rr = load_rerun(tree)
        if a.cmd == "plan":
            total = 0
            for suite in a.suite or list(rr.SUITES):
                units = plan(rr, suite, pilot=a.pilot, shard=rr.parse_shard(a.shard))
                total += len(units)
                for u in units:
                    print(f"  {u['pin']:<24} {u['arm_id']:<18} {u['variant']:<9} seed={u['seed']} "
                          f"task={u['task_id']:<4} k={u['k']:<3} trace={u['trace_sha'][:16]}")  # fmt: skip
                for k, v in rr.counts(units).items():
                    print(f"{suite}: {k} {v}")
                print(f"{suite}: TOTAL {len(units)}" + (f" [shard {a.shard}]" if a.shard else ""))
            print(f"ALL SUITES: {total} units, pin {EXTRA_PIN}; plan only, nothing launched")
            return 0
        src = harness_src_problems(tree)
        if src:
            raise ExtraPinError("; ".join(src))
        if a.cmd == "questioner-probe":
            return _probe_main(a, rr)
        return run_extra(a, rr, log=lambda s: print(s, flush=True))
    except Exception as e:  # noqa: BLE001 - only the two refusal classes are caught below
        refusals = (ExtraPinError, getattr(sys.modules.get("rerun"), "PlanError", ExtraPinError))
        if not isinstance(e, refusals):
            raise
        print(f"run_extra_pin: REFUSING: {e}", file=sys.stderr)
        return 6


if __name__ == "__main__":
    raise SystemExit(main())
