"""The prompted gpt-oss-120b questioner arm (RULES amendment 11), driven from outside the staged
tree. NOTHING HERE LAUNCHES OR CALLS A MODEL: units run through fake unit functions in a thread
pool, and the questioner probe goes through the harness's real MeteredClient with its transport
(`_dispatch`) replaced.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
DRIVER = ROOT / "scripts" / "tau2_rerun" / "run_extra_pin.py"
_spec = importlib.util.spec_from_file_location("run_extra_pin", DRIVER)
drv = importlib.util.module_from_spec(_spec)
sys.modules["run_extra_pin"] = drv
_spec.loader.exec_module(drv)

EXTRA = drv.EXTRA_PIN
SUITES = ("tau2_retail", "tau2_airline")
PROXY = "http://127.0.0.1:4062"
GI = SimpleNamespace(sha="abc123", dirty=False, dirty_files=())


@pytest.fixture
def rr():
    return drv.load_rerun(ROOT)


# ------------------------------------------------------------------------------ (a) specs
@pytest.mark.parametrize("suite", SUITES)
@pytest.mark.parametrize("shard", [None, (3, 6)])
def test_specs_equal_reruns_with_the_pin_in_pins_except_runs_root(rr, monkeypatch, suite, shard):
    work = drv.build(rr, suite, Path("/r"), Path("/c"), GI, shard=shard)
    campaign_pins = rr.PINS
    assert (EXTRA, "inquirer_prompted") not in campaign_pins  # the driver restored PINS
    monkeypatch.setattr(rr, "PINS", campaign_pins + ((EXTRA, "inquirer_prompted"),))
    want = {
        s.key: s
        for s, pin, _v in rr.build_work(suite, Path("/r"), Path("/c"), GI, shard=shard)
        if pin == EXTRA
    }
    assert len(want) == len(work) == (204 if shard is None else 34)
    for spec, pin, _v in work:
        assert pin == EXTRA and spec.runs_root == f"/r/{drv.STORE_DIR}"
        assert dataclasses.asdict(spec) == dataclasses.asdict(
            dataclasses.replace(want[spec.key], runs_root=spec.runs_root)
        )


# ------------------------------------------------------------------------------ (b) plan
def test_plan_is_408_units_204_per_suite_the_campaigns_slots(rr):
    total = 0
    for suite in SUITES:
        units = drv.plan(rr, suite)
        total += len(units)
        assert len(units) == 204
        assert {(u["pin"], u["arm_id"]) for u in units} == {(EXTRA, "inquirer_prompted")}
        assert rr.counts(units) == {
            f"{EXTRA}|inquirer_prompted|tau2_base": 102,
            f"{EXTRA}|inquirer_prompted|tau2_stop": 102,
        }
        base = {rr._slot(u) for u in rr.plan_units(suite) if u["pin"] == "qwen3-8b-base"}
        assert {rr._slot(u) for u in units} == base
        for k in range(1, 7):  # shard K of the arm is shard K of the campaign, slot for slot
            mine = {rr._slot(u) for u in drv.plan(rr, suite, shard=(k, 6))}
            theirs = {rr._slot(u) for u in rr.plan_units(suite, shard=(k, 6))}
            assert mine == theirs
    assert total == 408


def test_plan_cli_prints_408(capsys):
    assert drv.main(["plan", "--tree", str(ROOT)]) == 0
    out = capsys.readouterr().out
    assert "tau2_retail: TOTAL 204" in out and "tau2_airline: TOTAL 204" in out
    assert "ALL SUITES: 408 units" in out


# ------------------------------------------------------------------------------ (d) prices
def test_price_table_prices_gpt_oss_120b_as_a_questioner():
    from pinq_adapters.llm.pricing import PriceTable

    path = ROOT / "scripts" / "price_tables" / "2026-09.json"
    rows = json.loads(path.read_text())["models"]
    assert EXTRA in rows and rows[EXTRA]["input"] > 0 and rows[EXTRA]["output"] > 0
    table = PriceTable.load(path)
    # a questioner-shaped call: reasoning is a SUBSET of completion and is billed once
    usd = table.usd(EXTRA, tok_prompt=3000, tok_completion=900, tok_reasoning=700)
    assert usd == pytest.approx(3000 * 0.15e-6 + 900 * 0.75e-6)


# ------------------------------------------------------------------------------ (c) code_version
FILES = (
    "scripts/tau2_rerun/rerun.py",
    "scripts/tau2_rerun/withdrawn_unit_costs.json",
    "scripts/run_tau2_forks.py",
    "conf/forks/tau2_retail_test.recovered34.json",
    "conf/forks/tau2_airline_test.recovered34.json",
)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "-c", "user.email=t@example.invalid", "-c", "user.name=t", *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@pytest.fixture
def staged(tmp_path):
    """A clean git repo standing in for the staged tree: the planner's own files only."""
    repo = tmp_path / "staged"
    for f in FILES:
        (repo / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / f, repo / f)
    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "staged")
    return repo, _git(repo, "rev-parse", "HEAD")


def _manifest_for(rr, spec, pin):
    return {
        "code_version": spec.code_version,
        "dirty": spec.dirty,
        "budget_cap": spec.budget_cap,
        "max_turns": spec.max_turns,
        "pilot_flag": spec.pilot,
        "prompt_variant_id": rr.PROMPT_VARIANT_IDS[spec.prompt_variant],
        "pins": {
            "inquirer": {"model_id": pin},
            "drafter": {"model_id": "openai/aws/claude-sonnet-5"},
            "answerer": {"model_id": "openai/aws/claude-sonnet-5"},
        },
        "upstream_pins": {"user_sim": "openai/aws/gpt-oss-120b"},
    }


def _fake_unit(rr):
    """What run_tau2_unit writes, in the only respect under test: the manifest's code_version is
    the spec's, which build_specs took from `_git_state()`."""

    def unit(spec, pin):
        rid = hashlib.sha256(repr(spec.key).encode()).hexdigest()[:32]
        d = Path(spec.runs_root) / rid
        d.mkdir(parents=True)
        (d / "manifest.json").write_text(json.dumps(_manifest_for(rr, spec, pin)))
        return {"status": "ok", "run_id": rid, "usd_billed": 0.01}

    return unit


def _qrec(max_tokens, harness, jobid="910400", idx="3", **over):
    rec = {
        "verdict": "PASS",
        "model": EXTRA,
        "proxy_url": PROXY,
        "max_tokens": max_tokens,
        "harness_sha": harness,
        "completion_tokens": 310,
        "reasoning_tokens": 250,
        "LSB_JOBID": jobid,
        "LSB_JOBINDEX": idx,
    }
    rec.update(over)
    return rec


def _gateway_config(path: Path, local=("qwen3-8b-base",)) -> Path:
    routes = [
        {"model_name": m, "litellm_params": {"model": f"hosted_vllm/{m}", "api_base": "x"}}
        for m in local
    ] + [{"model_name": "*", "litellm_params": {"model": "openai/*"}}]
    path.write_text(json.dumps({"model_list": routes}))
    return path


@pytest.fixture
def job(staged, tmp_path, monkeypatch):
    """A staged tree, an environment assembled the way the wrapper does it, and a fresh root."""
    repo, sha = staged
    rr = drv.load_rerun(repo)
    monkeypatch.chdir(repo)
    traces = repo / "data" / "traces" / "tau2" / "src"  # untracked: git_info ignores it
    traces.mkdir(parents=True)
    shas = [p["trace_sha"] for s in SUITES for p in rr.load_points(s)]
    (traces / "t.jsonl").write_text("\n".join(json.dumps({"trace_sha": s}) for s in shas))
    table = str(ROOT / "scripts" / "price_tables" / "2026-09.json")
    env = {
        "LITELLM_BASE_URL": PROXY,
        "PI_PRICE_TABLE": table,
        "LITELLM_API_KEY": "k",
        "TAU2_DATA_DIR": str(tmp_path),
        "PI_MODEL_DRAFTER": "openai/aws/claude-sonnet-5",
        "PI_MODEL_ANSWERER": "openai/aws/claude-sonnet-5",
        "PI_MODEL_USERSIM": "openai/aws/gpt-oss-120b",
    }
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    for k in ("PI_GOLD_ROOT", "PI_LLM_REASONING_HEADROOM", "LSB_JOBID", "LSB_JOBINDEX"):
        monkeypatch.delenv(k, raising=False)
    from pinq.budget import BudgetLedger
    from pinq_adapters.llm.litellm_client import MeteredClient

    max_tokens = MeteredClient(BudgetLedger(cap=1))._budget(drv._act_max_tokens())
    q = tmp_path / "questioner_probe.json"
    q.write_text(json.dumps(_qrec(max_tokens, sha)))
    a = argparse.Namespace(
        tree=str(repo),
        suite="tau2_retail",
        shard="2/6",
        pilot=None,
        campaign_sha=sha,
        proxy_url=PROXY,
        runs_root=str(tmp_path / "arm" / "tau2_retail"),
        cache_root=str(tmp_path / "arm.cache"),
        price_table=table,
        proxy_config=str(_gateway_config(tmp_path / "proxy.yaml")),
        questioner_probe_record=str(q),
        probe_record=None,
        serve_record=None,
        job_start_epoch=None,
        serve=None,
        concurrency=1,
        spend_cap=100.0,
        max_consecutive_failures=6,
        job_label="t",
    )
    lsf = {"LSB_JOBID": "910400", "LSB_JOBINDEX": "3"}
    return SimpleNamespace(rr=rr, sha=sha, repo=repo, a=a, env=lsf, max_tokens=max_tokens)


def _run(j, **kw):
    return drv.run_extra(
        j.a,
        j.rr,
        executor=ThreadPoolExecutor(max_workers=1),
        unit_fn=_fake_unit(j.rr),
        env=kw.pop("env", j.env),
        log=lambda s: None,
    )


def test_code_version_is_the_staged_trees_git_not_the_drivers(job):
    here = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    assert job.sha != here  # the driver sits in a different repository, at a different sha
    assert job.rr.ROOT == job.repo.resolve()
    assert _run(job) == 0
    store = Path(job.a.runs_root) / drv.STORE_DIR
    manifests = [json.loads(p.read_text()) for p in store.glob("*/manifest.json")]
    assert len(manifests) == 34  # shard 2/6 of the arm's 204
    assert {m["code_version"] for m in manifests} == {job.sha}
    assert {m["pins"]["inquirer"]["model_id"] for m in manifests} == {EXTRA}
    launch = json.loads((Path(job.a.runs_root) / "ARM_LAUNCH.json").read_text())
    assert launch["harness_sha"] == job.sha and launch["regime"] == drv.GATEWAY_ONLY
    assert launch["driver_sha256"] == hashlib.sha256(DRIVER.read_bytes()).hexdigest()
    jobrec = json.loads((Path(job.a.runs_root) / "_jobs" / "gptoss120b-t" / "JOB.json").read_text())
    assert jobrec["driver_sha256"] == launch["driver_sha256"] and jobrec["harness_sha"] == job.sha
    assert jobrec["regime"] == drv.GATEWAY_ONLY and jobrec["n_units"] == 34


def test_a_tree_at_another_sha_is_refused(job):
    job.a.campaign_sha = "ee9afda24c73065715e685541fb423e22d3e7688"  # CONTROL: the real campaign
    with pytest.raises(drv.ExtraPinError, match="not the campaign sha"):
        _run(job)
    assert not (Path(job.a.runs_root) / drv.STORE_DIR).exists()


# ------------------------------------------------------------------------------ the regime
def test_gateway_only_plan_with_a_pass_questioner_probe_runs_without_qwen_records(job):
    assert job.a.probe_record is None and job.a.serve_record is None
    assert _run(job) == 0


def test_gateway_only_plan_without_the_questioner_probe_refuses(job):
    job.a.questioner_probe_record = None
    with pytest.raises(drv.ExtraPinError, match="questioner-probe PASS"):
        _run(job)


def test_a_questioner_record_from_another_job_refuses(job):
    with pytest.raises(drv.ExtraPinError, match="from job"):
        _run(job, env={"LSB_JOBID": "910399", "LSB_JOBINDEX": "3"})
    with pytest.raises(drv.ExtraPinError, match="from job"):
        _run(job, env={"LSB_JOBID": "910400", "LSB_JOBINDEX": "4"})  # another array element
    assert not (Path(job.a.runs_root) / drv.STORE_DIR).exists()


def test_outside_lsf_freshness_is_the_mtime_after_job_start(job, tmp_path):
    q = Path(job.a.questioner_probe_record)
    rec = json.loads(q.read_text())
    rec.update(LSB_JOBID=None, LSB_JOBINDEX=None)
    q.write_text(json.dumps(rec))
    with pytest.raises(drv.ExtraPinError, match="job-start-epoch"):
        _run(job, env={})
    job.a.job_start_epoch = time.time() + 3600  # the record predates this "job"
    with pytest.raises(drv.ExtraPinError, match="predates"):
        _run(job, env={})
    job.a.job_start_epoch = q.stat().st_mtime - 1  # CONTROL
    assert _run(job, env={}) == 0


@pytest.mark.parametrize(
    "field,value",
    [
        ("verdict", "REFUSE"),
        ("proxy_url", PROXY + "/"),
        ("model", "qwen3-8b-base"),
        ("max_tokens", 400),
        ("harness_sha", "0" * 40),
    ],  # fmt: skip
)
def test_questioner_record_must_match_this_job_byte_for_byte(job, field, value):
    q = Path(job.a.questioner_probe_record)
    rec = json.loads(q.read_text())
    rec[field] = value
    q.write_text(json.dumps(rec))
    with pytest.raises(drv.ExtraPinError, match=field):
        _run(job)


def test_mixed_plan_without_the_qwen_probe_refuses(job):
    local = drv.local_routes(Path(job.a.proxy_config).read_text())
    pins = [EXTRA, "qwen3-8b-base"]
    assert drv.regime_for(pins, local) == drv.LOCAL
    assert drv.regime_for([EXTRA], local) == drv.GATEWAY_ONLY
    with pytest.raises(drv.ExtraPinError, match="thinking-probe record and the serve record"):
        drv.check_records(
            job.a, job.rr, pins, local, env=job.env, job_start=None,
            harness_sha=job.sha, max_tokens=job.max_tokens,
        )  # fmt: skip


def test_mixed_plan_needs_a_probe_covering_its_local_pin(job, tmp_path):
    """With the Qwen records present rerun.py's own checks run, and the thinking probe must name
    every local planned pin."""
    local = {"qwen3-8b-base", "qwen3-8b-dpo-stacked-notdone-both-s1",
             "qwen3-8b-dpo-stacked-notdone-both-s2", "qwen3-8b-extra"}  # fmt: skip
    serve = tmp_path / "serve.json"
    serve.write_text(json.dumps({
        "served": [p for p, _ in job.rr.PINS],
        "adapters": {p: {"sha256": "a" * 64} for p in job.rr.TRAINED},
    }))  # fmt: skip
    probe = tmp_path / "probe.json"
    routes = [
        {"model": p, "refuse_reason": None, "thinking_detected": False} for p, _ in job.rr.PINS
    ]
    probe.write_text(json.dumps({"verdict": "PASS", "proxy_url": PROXY, "routes": routes}))
    job.a.probe_record, job.a.serve_record = str(probe), str(serve)
    kw = dict(env=job.env, job_start=None, harness_sha=job.sha, max_tokens=job.max_tokens)
    out = drv.check_records(job.a, job.rr, [EXTRA, "qwen3-8b-base"], local, **kw)
    assert out["regime"] == drv.LOCAL and out["qrec"]["verdict"] == "PASS"
    with pytest.raises(drv.ExtraPinError, match="does not cover"):
        drv.check_records(job.a, job.rr, [EXTRA, "qwen3-8b-extra"], local, **kw)
    probe.write_text(json.dumps({"verdict": "REFUSE", "proxy_url": PROXY, "routes": routes}))
    with pytest.raises(job.rr.PlanError, match="PASS"):
        drv.check_records(job.a, job.rr, [EXTRA, "qwen3-8b-base"], local, **kw)


def test_a_second_shard_with_another_driver_or_regime_refuses(job, tmp_path):
    assert _run(job) == 0
    launch = Path(job.a.runs_root) / "ARM_LAUNCH.json"
    rec = json.loads(launch.read_text())
    rec["driver_sha256"] = "0" * 64
    launch.write_text(json.dumps(rec))
    job.a.shard, job.a.job_label = "3/6", "t3"
    with pytest.raises(drv.ExtraPinError, match="another population"):
        _run(job)


def test_the_arm_refuses_a_root_holding_campaign_pins(job):
    (Path(job.a.runs_root) / "qwen3-8b-base").mkdir(parents=True)
    with pytest.raises(drv.ExtraPinError, match="its own root"):
        _run(job)


# ------------------------------------------------------------------------------ the harness src
def test_harness_src_must_be_the_staged_trees(tmp_path):
    assert drv.harness_src_problems(ROOT) == []  # this process imports this tree's src
    probs = drv.harness_src_problems(tmp_path)
    assert len(probs) == len(drv.HARNESS_PACKAGES)


# ------------------------------------------------------------------------------ the probe
def _suite():
    return SimpleNamespace(
        suite_id="tau2_retail",
        instructions="You are a retail support agent.",
        corpus_id="tau2_retail",
        corpus_hash="h" * 16,
        tool_schemas=lambda tid: [],
    )


def _prefix():
    return [
        SimpleNamespace(role="user", content="Hi, I want to return the mechanical keyboard.",
                        tool_calls=None),
        SimpleNamespace(role="assistant", content="Sure, what is your email?", tool_calls=None),
        SimpleNamespace(role="user", content="It is noah.brown@example.com and order W9", tool_calls=None),
    ]  # fmt: skip


def _probe(monkeypatch, text, completion, reasoning, variant="tau2_base"):
    from pi_run.stages import tau2_runner as runner
    from pinq.budget import BudgetLedger
    from pinq_adapters.llm import litellm_client as lc
    from pinq_expt import arms

    monkeypatch.delenv("PI_LLM_REASONING_HEADROOM", raising=False)
    env = {
        "PI_MODEL_INQUIRER": EXTRA,
        "LITELLM_BASE_URL": PROXY,
        "LITELLM_API_KEY": "k",
        "PI_PRICE_TABLE": str(ROOT / "scripts" / "price_tables" / "2026-09.json"),
    }
    ledger = BudgetLedger(cap=16)
    client = lc.MeteredClient(ledger, temperature=0.0, env=env)
    sent = []

    def transport(payload):
        sent.append(payload)
        return lc._Response(text=text, tok_prompt=2100, tok_completion=completion,
                            tok_reasoning=reasoning, tok_cached=0, http_status=200)  # fmt: skip

    client._dispatch = transport
    unit = {"suite": "tau2_retail", "pin": EXTRA, "arm_id": "inquirer_prompted",
            "variant": variant, "seed": 0, "task_id": "15", "k": 3, "trace_sha": "t"}  # fmt: skip
    rec = drv.questioner_probe(
        unit=unit, prefix=_prefix(), suite_obj=_suite(), runner=runner, arms=arms,
        client=client, ledger=ledger, is_valid=lambda m: True, proxy_url=PROXY,
    )  # fmt: skip
    return rec, sent


ASK = '{"action": "ask", "question": "What is the status of order W9?", "target": "kb"}'


def test_questioner_probe_passes_a_short_real_reply_and_records_reasoning(monkeypatch):
    rec, sent = _probe(monkeypatch, ASK, completion=310, reasoning=250)
    assert rec["verdict"] == "PASS" and rec["problems"] == []
    assert rec["reasoning_tokens"] == 250 and rec["completion_tokens"] == 310
    assert rec["max_tokens"] == 1200 == sent[0]["max_tokens"]  # 400 content x 3.0 headroom
    assert sent[0]["model"] == EXTRA and rec["model"] == EXTRA
    assert rec["action"] == "Ask" and rec["malformed"] == 0
    assert rec["prompt_variant_id"] == "tau2_base-3006f5bf"
    prompt = sent[0]["messages"][-1]["content"]
    assert "mechanical keyboard" in prompt and "noah.brown@example.com" in prompt
    assert rec["prompt_sha256"] == hashlib.sha256(prompt.encode()).hexdigest()


def test_questioner_probe_refuses_empty_content(monkeypatch):
    rec, _ = _probe(monkeypatch, "", completion=1200, reasoning=1200)
    assert rec["verdict"] == "REFUSE"
    assert any("empty" in p for p in rec["problems"])
    assert any("completion_tokens 1200" in p for p in rec["problems"])


@pytest.mark.parametrize("completion,verdict", [(1079, "PASS"), (1080, "REFUSE"), (1150, "REFUSE")])
def test_questioner_probe_refuses_at_ninety_percent_of_max_tokens(monkeypatch, completion, verdict):
    rec, _ = _probe(monkeypatch, ASK, completion=completion, reasoning=completion - 60)
    assert rec["verdict"] == verdict


def test_questioner_probe_uses_the_stop_variants_template(monkeypatch):
    rec, _ = _probe(monkeypatch, ASK, completion=300, reasoning=200, variant="tau2_stop")
    assert rec["prompt_variant_id"] == "tau2_stop-6b1a91aa"


def test_first_state_is_the_prefix_as_the_orchestrator_hands_it():
    p = _prefix()
    assert drv.first_state_messages(p, lambda m: True) == p
    assert drv.first_state_messages(p, lambda m: m.role != "assistant") == [p[0], p[2]]
    with pytest.raises(drv.ExtraPinError, match="customer message"):
        drv.first_state_messages(p[:2], lambda m: True)


def test_probe_unit_is_the_first_planned_user_last_unit(rr):
    def loader(u):
        last = "assistant" if u["task_id"] == "55" else "user"
        return [SimpleNamespace(role=last, content="x", tool_calls=None)]

    u, _ = drv.probe_unit(rr, "tau2_retail", loader, None)
    ordered = sorted(drv.plan(rr, "tau2_retail"), key=rr._slot)
    assert u == next(x for x in ordered if x["task_id"] != "55")


def test_driver_file_has_no_side_effects_on_import():
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    r = subprocess.run(
        [sys.executable, "-c", f"import runpy; runpy.run_path({str(DRIVER)!r}, run_name='x')"],
        env=env, capture_output=True, text=True, timeout=120,
    )  # fmt: skip
    assert r.returncode == 0 and r.stdout == "", r.stderr
