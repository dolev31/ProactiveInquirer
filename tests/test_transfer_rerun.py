"""The tau2 transfer-rerun reader refuses what it must not read, and reads the rest by the task.

Pins `scripts/tau2_concordance/transfer_rerun.py` against `artifacts/tau2_rerun_20260923/RULES.md`.
Every refusal has a test that feeds it the one defect it exists for and requires exit 3 with the
reason printed, and `test_a_healthy_population_is_read` requires the same builder with no defect to
exit 0 without the word REFUS anywhere, so no refusal is an off switch. The known-answer tests
carry their arithmetic in the docstring, computed by hand before the reader existed.
"""

from __future__ import annotations

import copy
import functools
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "tau2_concordance"))
import extract_records  # noqa: E402
import transfer_rerun as tr  # noqa: E402

from pinq.ids import h  # noqa: E402

T, C = "inquirer_trained", "inquirer_prompted"
S1, S2 = "qwen3-8b-dpo-stacked-notdone-both-s1", "qwen3-8b-dpo-stacked-notdone-both-s2"
BASE = "qwen3-8b-base"
FRONTIER = "openai/aws/claude-sonnet-5"
# the serve record's prefixes (item 8): 16 hex of each adapter's weight sha256
S1_SHA = "c33d3e6e0b7fd666" + "0" * 48
S2_SHA = "8e8603830b153535" + "1" * 48
CV = "a" * 40
PROXY = "http://127.0.0.1:4001"
PROXY_CONFIG_SHA = "c" * 64
# The questioner's base_url_sha IS the probe proxy's, hashed the way the harness pins it:
# `MeteredClient.pin` -> h("base_url", base_url). Computed here by pinq.ids directly, so a test
# that passes also says the reader's route through the harness agrees with this one.
URL_LOCAL = h("base_url", PROXY)
URL_GATEWAY = "2" * 64
FAST_BOOT = ["--boot-resamples", "40,80,120", "--printed-interval", "200:0"]
PREFIX = 2  # env calls inherited from the fork prefix


def manifest(model: str, **over) -> dict:
    m = {
        "code_version": CV,
        "dirty": False,
        "budget_cap": 16,
        "k": 5,
        "pins": {
            "answerer": {"role": "answerer", "model_id": FRONTIER, "base_url_sha": URL_GATEWAY},
            "drafter": {"role": "drafter", "model_id": FRONTIER, "base_url_sha": URL_GATEWAY},
            "inquirer": {
                "role": "inquirer",
                "model_id": model,
                "base_url_sha": URL_LOCAL,
                "adapter_sha": None,
            },
        },
        "upstream_pins": {"user_sim": "openai/aws/gpt-oss-120b"},
    }
    m.update(over)
    return m


def rec(
    arm: str,
    model: str,
    *,
    task: str = "t1",
    trace: str = "aaa",
    k: int = 4,
    seed: int = 0,
    variant: str = "tau2_base",
    suite: str = "tau2_retail",
    reward: float = 1.0,
    calls: int = 5,
    asks: int = 3,
    followups: int = 4,
    status: str = "ok",
) -> dict:
    return {
        "status": status,
        "run_id": f"{arm[9:13]}-{model[-2:]}-{suite[5:]}-{variant[5:]}-{task}-{trace}-{k}-{seed}",
        "key": [suite, task, arm, seed, "", -1, -1, trace, k, variant, ""],
        "suite_id": suite,
        "task_id": task,
        "arm_id": arm,
        "seed": seed,
        "foreign_trace_sha": trace,
        "foreign_prefix_k": k,
        "code_version": CV,
        "dirty": False,
        "native": {"tau_reward": reward},
        "n_env_calls": calls,
        "n_asks": asks,
        "n_user_turns": 2 + followups,
        "n_prefix_user_turns": 2,
        "n_transfer_to_human": 0,
        # THE GATED SHAPE (enforcement lane ba48ab0 + item 7). `calls` env calls: the first
        # PREFIX are the inherited prefix (free); the live ones end with one GENERIC call
        # (`calculate`, uncharged). The gate charged the live non-GENERIC ok calls, and the
        # ledger holds asks + that charge.
        # AMENDMENT 7: separate budgets. The asks ledger (spent.retrieval_calls) holds the asks
        # only; the gate charges its own tool_call_cap and never the ledger.
        "budget_enforcement": "refuse_and_tell_split",
        "budget_cap": 16,
        "tool_call_cap": 16,
        "n_refused_budget": 0,
        "refused_budget_calls": [],
        "budget_exempt_tools": ["calculate", "transfer_to_human_agents"],
        "n_prefix_env_calls": PREFIX,
        "n_live_env_calls": calls - PREFIX,
        "n_live_env_calls_generic": 1,
        "n_live_env_calls_nongeneric": calls - PREFIX - 1,
        "n_live_env_calls_nongeneric_ok": calls - PREFIX - 1,
        "n_gate_charged": calls - PREFIX - 1,
        "gate_census_agrees": True,
        "stop_reason": "policy_stop",
        "n_assistant_rollouts": 2,
        # the enforcement lane's rollout-1 and first-live-call fields: the asks came first, then
        # the first live call executed with the ledger at `asks`
        "rollout1_stop_reason": "policy_stop",
        "rollout1_n_asks": asks,
        "first_live_call_executed": True,
        "spent_before_first_live_call": float(asks),
        "spent": {"retrieval_calls": float(asks)},
        "_env_calls_from_outcome": calls,
        "_env_tool_counts": {"get_user_details": calls - 1, "calculate": 1},
        "_env_requestor_counts": {"assistant": calls},
        "_env_call_seq": [["get_user_details", True]] * (calls - 1) + [["calculate", True]],
        "_ask_retrieved_uids": [[] for _ in range(asks)],
        # distinct questions: 3 shared content tokens pairwise, below the guard's 4
        "_ask_questions": [f"What is record {i} of task {task}?" for i in range(asks)],
        "_manifest": manifest(model),
    }


def healthy(suites=("tau2_retail", "tau2_airline"), variants=("tau2_base", "tau2_stop")) -> list:
    """Every declared cell holds two tasks x one fork point x one run seed, all three models."""
    out = []
    for suite in suites:
        for variant in variants:
            for task, trace in (("t1", "aaa"), ("t2", "bbb")):
                kw = dict(suite=suite, variant=variant, task=task, trace=trace)
                out += [
                    rec(C, BASE, reward=0.0, **kw),
                    rec(T, S1, reward=1.0, **kw),
                    rec(T, S2, reward=1.0 if task == "t1" else 0.0, **kw),
                ]
    return out


def serve_record(tmp_path: Path, adapters=None, name="SERVED.910128.json", job="910128") -> Path:
    """The pinq.serve.v1 shape of LSF 910128's record (item 8): gpu-n595:8302, written
    2026-09-22T21:28:03Z, each adapter 349243752 bytes. Only the prefixes are real."""
    p = tmp_path / name
    p.parent.mkdir(parents=True, exist_ok=True)
    if adapters is None:
        adapters = {S1: {"sha256": S1_SHA}, S2: {"sha256": S2_SHA}}
    p.write_text(
        json.dumps(
            {
                "schema": "pinq.serve.v1",
                "written_at": "2026-09-22T21:28:03+00:00",
                "job": job,
                "host": "gpu-n595",
                "port": 8302,
                "base_model": "Qwen/Qwen3-8B",
                "served_base": BASE,
                "served": [BASE, *adapters],
                "adapters": {
                    m: {"path": f"/proj/pinq/user/artifacts/{m}", "bytes": 349243752, **a}
                    for m, a in adapters.items()
                },
            }
        )
    )
    return p


def probe_record(
    tmp_path: Path,
    models=(BASE, S1, S2),
    verdict="PASS",
    name="probe.json",
    config_sha=PROXY_CONFIG_SHA,
    proxy=PROXY,
    config_path="<scratch>/fixed.yaml",
) -> Path:
    """The thinking probe's record, in the shape of 1106c39 `probe_fixed.json`."""
    p = tmp_path / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(
            {
                "verdict": verdict,
                "proxy_url": proxy,
                "config_path": config_path,
                "config_sha256": config_sha,
                "routes": [
                    {
                        "model": m,
                        "chat_template_kwargs": {"enable_thinking": False},
                        "completion_tokens": 2,
                        "max_tokens": 32,
                        "thinking_detected": False,
                        "refuse_reason": None,
                        "content_preview": "READY",
                    }
                    for m in models
                ],
            }
        )
    )
    return p


def plan_unit(r: dict) -> dict:
    """One `scripts/tau2_rerun/rerun.py::plan_units` row for the unit a record ran."""
    k = r["key"]
    return {
        "suite": k[0],
        "pin": r["_manifest"]["pins"]["inquirer"]["model_id"],
        "arm_id": k[2],
        "variant": k[9],
        "seed": k[3],
        "task_id": k[1],
        "k": k[8],
        "trace_sha": k[7],
    }


def launch_records(tmp_path: Path, records, extra_units=()) -> list[Path]:
    """One LAUNCH.json per suite, in the launcher's shape (0212fe0 `launch_record`)."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    # a record with no manifest has no pin to plan it under; the manifest refusal names it first
    units = [plan_unit(r) for r in records if "_manifest" in r] + list(extra_units)
    paths = []
    for suite in sorted({u["suite"] for u in units}):
        us = [u for u in units if u["suite"] == suite]
        p = tmp_path / f"LAUNCH.{suite}.json"
        p.write_text(
            json.dumps(
                {
                    "suite": suite,
                    "runs_roots": {pin: f"/abs/{suite}/{pin}" for pin in (BASE, S1, S2)},
                    "pins": [{"pin": BASE, "arm_id": C}, {"pin": S1, "arm_id": T}],
                    "design": {"budget_cap": 16, "seeds": [0, 1, 2]},
                    "plan": {"n_units": len(us), "units": us},
                }
            )
        )
        paths.append(p)
    return paths


#: every task id any fixture uses
TASK_IDS = ("t1", "t2", "t3", "t9", "A", "B", "C", "D", "E", "F")
DB = {
    "retail": {
        "users": {"u1": {"name": "Ann"}, "u2": {"name": "Bo"}},
        "orders": {"#W1": {"status": "pending"}},
        "products": {},
    },
    "airline": {
        "users": {"u1": {"name": "Ann"}, "u2": {"name": "Bo"}},
        "reservations": {"R1": {"status": "booked"}},
        "flights": {},
    },
}
#: default gold: one READ naming pre-dialogue user u1
GOLD_ACTIONS = [{"name": "get_user_details", "arguments": {"user_id": "u1"}}]


def tau2_data(tmp_path: Path, gold: dict | None = None) -> Path:
    """A synthetic TAU2_DATA_DIR in the harness's layout: tau2/domains/<domain>/{db,tasks}.json.
    `gold` maps (domain, task id) to that task's gold actions."""
    root = tmp_path / "tau2data"
    for dom, db in DB.items():
        d = root / "tau2" / "domains" / dom
        d.mkdir(parents=True, exist_ok=True)
        (d / "db.json").write_text(json.dumps(db))
        tasks = [
            {
                "id": tid,
                "evaluation_criteria": {"actions": (gold or {}).get((dom, tid), GOLD_ACTIONS)},
            }
            for tid in TASK_IDS
        ]
        (d / "tasks.json").write_text(json.dumps(tasks))
    return root


def gold_uid(suite: str, table: str = "users", rid: str = "u1") -> str:
    from pinq_adapters.tau2 import airline_units, retail_units

    if suite == "tau2_retail":
        return retail_units.RetailIndex.from_db(DB["retail"]).uid(table, rid)
    return airline_units.AirlineIndex.from_db(DB["airline"]).uid(table, rid)


def run(tmp_path, records, *extra, serve=True, probe=True, launch=True, gold=None, capsys=None):
    f = tmp_path / "records.json"
    f.write_text(json.dumps(records))
    argv = ["--records", str(f), *FAST_BOOT, "--out", str(tmp_path / "out.json")]
    argv += ["--tau2-data-dir", str(tau2_data(tmp_path, gold))]
    if serve:
        argv += ["--serve-record", str(serve_record(tmp_path))]
    if probe:
        argv += ["--probe-record", str(probe_record(tmp_path))]
    if launch:
        for p in launch_records(tmp_path, records):
            argv += ["--launch-record", str(p)]
    argv += list(extra)
    code = tr.main(argv)
    out = capsys.readouterr().out if capsys is not None else ""
    return code, out


def refused(code: int, out: str, needle: str) -> None:
    """Exit 3, and the reason IN THE REFUSING LINE. Matching anywhere in stdout let a refusal for
    another reason pass whenever the needle sat in the tmp path, which carries the test's name
    (found 2026-09-23: '.../test_a_probe_on_another_proxy_th0/records.json' matched 'proxy')."""
    assert code == tr.EXIT_REFUSED, f"expected refusal (exit 3), got {code}:\n{out}"
    lines = [ln for ln in out.splitlines() if "REFUSING:" in ln]
    assert lines, out
    assert needle in lines[0], f"refusal reason {needle!r} not in {lines[0]!r}"


# ------------------------------------------------------------------ the refusals must not be blanket


def test_a_healthy_population_is_read(tmp_path, capsys):
    code, out = run(tmp_path, healthy(), capsys=capsys)
    assert code == 0, out
    assert "REFUS" not in out
    rep = json.loads((tmp_path / "out.json").read_text())
    assert rep["stamp"] is None
    assert len(rep["cells"]) == 8, "2 suites x 2 variants x 2 seeds"
    assert {c["seed_label"] for c in rep["cells"]} == {"s1", "s2"}


# ------------------------------------------------------------------ refusals: the store


def test_an_empty_store_is_refused(tmp_path, capsys):
    refused(*run(tmp_path, [], capsys=capsys), "empty")


def test_a_declared_cell_with_zero_pairs_is_refused(tmp_path, capsys):
    recs = [r for r in healthy() if not (r["key"][9] == "tau2_stop" and r["arm_id"] == C)]
    refused(*run(tmp_path, recs, capsys=capsys), "zero pairs")


def test_two_code_versions_are_refused_with_per_sha_counts(tmp_path, capsys):
    recs = healthy()
    recs[0]["code_version"] = "b" * 40
    recs[0]["_manifest"]["code_version"] = "b" * 40
    code, out = run(tmp_path, recs, capsys=capsys)
    refused(code, out, "code_version")
    assert "aaaaaaaa: 23" in out and "bbbbbbbb: 1" in out, out


def test_status_and_manifest_code_versions_that_disagree_are_refused(tmp_path, capsys):
    recs = healthy()
    recs[0]["_manifest"]["code_version"] = "b" * 40
    refused(*run(tmp_path, recs, capsys=capsys), "code_version")


def test_a_dirty_run_is_refused(tmp_path, capsys):
    recs = healthy()
    recs[3]["_manifest"]["dirty"] = True
    refused(*run(tmp_path, recs, capsys=capsys), "dirty")


def test_an_absent_dirty_flag_is_refused_not_read_as_clean(tmp_path, capsys):
    recs = healthy()
    del recs[3]["dirty"]
    refused(*run(tmp_path, recs, capsys=capsys), "dirty")


def test_two_base_url_shas_for_one_role_are_refused(tmp_path, capsys):
    recs = healthy()
    recs[1]["_manifest"]["pins"]["drafter"]["base_url_sha"] = "9" * 64
    refused(*run(tmp_path, recs, capsys=capsys), "base_url_sha")


def test_a_record_without_its_manifest_is_refused(tmp_path, capsys):
    recs = healthy()
    del recs[2]["_manifest"]
    refused(*run(tmp_path, recs, capsys=capsys), "manifest")


# ------------------------------------------------------------------ refusals: the budget


def test_a_run_without_refuse_and_tell_is_refused(tmp_path, capsys):
    recs = healthy()
    del recs[4]["budget_enforcement"]
    refused(*run(tmp_path, recs, capsys=capsys), "budget_enforcement")


def test_a_run_with_another_enforcement_mode_is_refused(tmp_path, capsys):
    recs = healthy()
    recs[4]["budget_enforcement"] = "post_hoc"
    refused(*run(tmp_path, recs, capsys=capsys), "budget_enforcement")


def test_the_enforcement_field_is_read_from_the_manifest_too(tmp_path, capsys):
    """The enforcement lane may stamp it on the manifest rather than the status."""
    recs = healthy()
    for r in recs:
        del r["budget_enforcement"]
        r["_manifest"]["budget_enforcement"] = "refuse_and_tell_split"
    code, out = run(tmp_path, recs, capsys=capsys)
    assert code == 0, out


def test_a_post_hoc_overrun_is_refused(tmp_path, capsys):
    recs = healthy()
    recs[5]["spent"]["post_hoc_budget_overrun"] = 1.0
    refused(*run(tmp_path, recs, capsys=capsys), "post_hoc_budget_overrun")


def test_spend_above_the_cap_is_refused(tmp_path, capsys):
    recs = healthy()
    recs[5]["spent"]["retrieval_calls"] = 17.0
    refused(*run(tmp_path, recs, capsys=capsys), "above the cap")


def _asked_nothing(r: dict) -> dict:
    """The shape of the two 120B runs of amendment 12 (e4efc025, d785b202): the questioner stopped
    before its first question, so the ledger never created `retrieval_calls`."""
    r["n_asks"] = 0
    r["rollout1_n_asks"] = 0
    r["n_turns"] = 0
    r["spent_before_first_live_call"] = 0.0
    r["spent"] = {"llm_calls": 3.0, "tok_total": 12492.0, "usd": 0.021}
    r["_ask_retrieved_uids"] = []
    r["_ask_questions"] = []
    return r


def test_a_run_that_asked_nothing_reads_its_absent_ledger_key_as_zero(tmp_path, capsys):
    """Amendment 12: the absence is a true zero only on a run that asked nothing."""
    recs = healthy()
    _asked_nothing(recs[0])  # the comparator, retail/base t1
    code, out = run(tmp_path, recs, capsys=capsys)
    assert code == 0, out
    assert "REFUS" not in out
    line = [ln for ln in out.splitlines() if "amendment 12" in ln]
    assert line and "1 ok run" in line[0] and recs[0]["run_id"] in line[0], out


def test_an_absent_ledger_key_on_a_run_that_asked_is_still_refused(tmp_path, capsys):
    recs = healthy()
    del recs[5]["spent"]["retrieval_calls"]
    refused(*run(tmp_path, recs, capsys=capsys), "absent")


def test_an_absent_ledger_key_with_a_recorded_question_is_refused(tmp_path, capsys):
    recs = healthy()
    _asked_nothing(recs[0])["_ask_questions"] = ["What is the order id?"]
    refused(*run(tmp_path, recs, capsys=capsys), "absent")


def test_an_absent_ledger_key_with_a_turn_is_refused(tmp_path, capsys):
    recs = healthy()
    _asked_nothing(recs[0])["n_turns"] = 1
    refused(*run(tmp_path, recs, capsys=capsys), "absent")


def test_an_absent_ledger_key_without_a_turn_count_is_refused(tmp_path, capsys):
    recs = healthy()
    del _asked_nothing(recs[0])["n_turns"]
    refused(*run(tmp_path, recs, capsys=capsys), "absent")


def test_an_empty_ledger_is_refused_not_read_as_zero(tmp_path, capsys):
    recs = healthy()
    _asked_nothing(recs[0])["spent"] = {}
    refused(*run(tmp_path, recs, capsys=capsys), "absent")


def test_a_run_whose_manifest_cap_is_not_the_declared_cap_is_refused(tmp_path, capsys):
    recs = healthy()
    recs[5]["_manifest"]["budget_cap"] = 8
    refused(*run(tmp_path, recs, capsys=capsys), "budget_cap")


def test_a_status_cap_that_is_not_the_declared_cap_is_refused(tmp_path, capsys):
    """ba48ab0 writes budget_cap on the status too; the two routes must agree with the design."""
    recs = healthy()
    recs[5]["budget_cap"] = 8
    refused(*run(tmp_path, recs, capsys=capsys), "budget_cap")


def test_a_refused_count_that_disagrees_with_its_list_is_refused(tmp_path, capsys):
    recs = healthy()
    recs[5]["n_refused_budget"] = 2
    recs[5]["refused_budget_calls"] = [{"tool_name": "get_order_details", "mutating": False}]
    refused(*run(tmp_path, recs, capsys=capsys), "refused_budget_calls")


def test_env_calls_above_the_cap_are_not_a_refusal(tmp_path, capsys):
    """REGRESSION PIN (passes on the prior code too). n_env_calls counts the free prefix and the
    uncharged GENERIC calls, so it may exceed 16 on a correctly gated unit; only the ledger is
    capped. 25 env calls, 12 of them prefix, 12 live charged + 1 GENERIC, 3 asks -> spent 15."""
    recs = healthy()
    r = recs[4]
    r.update(
        n_env_calls=25,
        _env_calls_from_outcome=25,
        n_prefix_env_calls=12,
        n_live_env_calls=13,
        n_gate_charged=12,
        _env_tool_counts={"get_user_details": 24, "calculate": 1},
        _env_requestor_counts={"assistant": 25},
        _env_call_seq=[["get_user_details", True]] * 24 + [["calculate", True]],
    )
    r["spent"]["retrieval_calls"] = 15.0
    code, out = run(tmp_path, recs, capsys=capsys)
    assert code == 0, out


def test_a_negative_spend_residual_is_refused(tmp_path, capsys):
    """CHANGED with amendment 7: under separate budgets the gate no longer charges the ledger, so
    the residual is r = spent - (n_asks - rejected_user_asks), with no gate term; refused only when
    NEGATIVE (expansion sub-retrievals make it positive)."""
    recs = healthy()
    recs[4]["spent"]["retrieval_calls"] -= 1.0
    recs[4]["spent_before_first_live_call"] = recs[4]["spent"]["retrieval_calls"]  # consistent
    refused(*run(tmp_path, recs, capsys=capsys), "residual")


def test_a_positive_residual_is_expansion_spend_and_is_printed_per_arm(tmp_path, capsys):
    """Two expansion sub-retrievals on one s1 run: r = 2 there, 0 elsewhere. Not a refusal."""
    recs = healthy()
    s1_run = next(r for r in recs if r["_manifest"]["pins"]["inquirer"]["model_id"] == S1)
    s1_run["spent"]["retrieval_calls"] += 2.0
    code, out = run(tmp_path, recs, capsys=capsys)
    assert code == 0, out
    res = json.loads((tmp_path / "out.json").read_text())["provenance"]["residual_by_arm"]
    assert res["s1"] == {"0": 7, "2": 1}
    assert res["comparator"] == {"0": 8}
    assert any("residual" in ln and "s1" in ln for ln in out.splitlines())


def test_rejected_user_asks_are_not_charged_asks(tmp_path, capsys):
    """spent.rejected_user_asks leave n_asks but never reach the ledger. Under the split the gate
    charges its own budget, so: 3 asks, 1 rejected -> spent 2 and r = 2 - (3 - 1) = 0."""
    recs = healthy()
    recs[4]["spent"]["rejected_user_asks"] = 1.0
    recs[4]["spent"]["retrieval_calls"] = 2.0
    recs[4]["spent_before_first_live_call"] = 2.0  # the two charged asks came first
    code, out = run(tmp_path, recs, capsys=capsys)
    assert code == 0, out


def test_a_gate_charge_off_by_one_from_the_recount_is_refused(tmp_path, capsys):
    """Identity (b), the instrument that can disagree with the gate: n_gate_charged against the ok,
    non-GENERIC env calls from index n_prefix_env_calls on, recounted by the reader. Off by one,
    with the ledger moved to match so identity (a) still holds."""
    recs = healthy()
    recs[4]["n_gate_charged"] += 1
    recs[4]["spent"]["retrieval_calls"] += 1.0
    code, out = run(tmp_path, recs, capsys=capsys)
    refused(code, out, "recount")
    assert "gate 3, recount 2" in out, out


def test_a_charged_generic_call_is_refused_by_the_recount(tmp_path, capsys):
    """GENERIC tools (calculate, transfer_to_human_agents) are uncharged: a gate that charged the
    live `calculate` shows as n_gate_charged one above the recount."""
    recs = healthy()
    recs[4]["n_gate_charged"] = 3  # 2 live get_user_details + the GENERIC calculate
    recs[4]["spent"]["retrieval_calls"] = 3.0 + 3
    refused(*run(tmp_path, recs, capsys=capsys), "recount")


def test_a_failed_live_call_is_not_in_the_recount(tmp_path, capsys):
    """A call the environment failed is not charged (BudgetGate: 'A FAILED CALL IS NOT CHARGED')."""
    recs = healthy()
    r = recs[4]
    r["_env_call_seq"] = [["get_user_details", True]] * 3 + [
        ["get_user_details", False],
        ["calculate", True],
    ]
    r["n_gate_charged"] = 1
    r["spent"]["retrieval_calls"] = 3.0 + 1
    code, out = run(tmp_path, recs, capsys=capsys)
    assert code == 0, out


def test_a_gate_census_disagreement_is_refused(tmp_path, capsys):
    recs = healthy()
    recs[4]["gate_census_agrees"] = False
    refused(*run(tmp_path, recs, capsys=capsys), "gate_census_agrees")


def test_exempt_tools_other_than_the_maps_generic_set_are_refused(tmp_path, capsys):
    """A gate that exempted more (or less) than conf/tau2/tool_types.json says is not the rule."""
    recs = healthy()
    recs[4]["budget_exempt_tools"] = ["calculate"]
    refused(*run(tmp_path, recs, capsys=capsys), "budget_exempt_tools")


def test_a_run_without_the_gate_fields_is_refused(tmp_path, capsys):
    recs = healthy()
    del recs[4]["n_gate_charged"]
    refused(*run(tmp_path, recs, capsys=capsys), "n_gate_charged")


def test_a_run_without_its_env_call_sequence_is_refused(tmp_path, capsys):
    recs = healthy()
    del recs[4]["_env_call_seq"]
    refused(*run(tmp_path, recs, capsys=capsys), "_env_call_seq")


# ------------------------------------------------------------------ refusals: the weights


def test_a_manifest_adapter_sha_off_the_expected_prefix_is_refused(tmp_path, capsys):
    recs = healthy()
    trained_s1 = next(r for r in recs if r["_manifest"]["pins"]["inquirer"]["model_id"] == S1)
    trained_s1["_manifest"]["pins"]["inquirer"]["adapter_sha"] = "80a56b74" + "0" * 56
    refused(*run(tmp_path, recs, capsys=capsys), "adapter")


def test_no_adapter_sha_and_no_serve_record_is_refused(tmp_path, capsys):
    refused(*run(tmp_path, healthy(), serve=False, capsys=capsys), "serve record")


def test_a_trained_model_absent_from_the_serve_record_is_refused(tmp_path, capsys):
    recs = healthy()
    f = tmp_path / "records.json"
    f.write_text(json.dumps(recs))
    sp = serve_record(tmp_path, adapters={S1: {"sha256": S1_SHA}})
    code = tr.main(
        ["--records", str(f), *FAST_BOOT, "--serve-record", str(sp)]
        + ["--probe-record", str(probe_record(tmp_path))]
    )
    refused(code, capsys.readouterr().out, "serve record")


def test_a_serve_record_sha_off_the_expected_prefix_is_refused(tmp_path, capsys):
    recs = healthy()
    f = tmp_path / "records.json"
    f.write_text(json.dumps(recs))
    sp = serve_record(tmp_path, adapters={S1: {"sha256": S2_SHA}, S2: {"sha256": S2_SHA}})
    code = tr.main(
        ["--records", str(f), *FAST_BOOT, "--serve-record", str(sp)]
        + ["--probe-record", str(probe_record(tmp_path))]
    )
    refused(code, capsys.readouterr().out, "c33d3e6e")


def test_a_trained_run_pinning_an_undeclared_model_is_refused(tmp_path, capsys):
    recs = healthy()
    stray = rec(T, "qwen3-8b-dpo-stacked-notdone-both", task="t9", trace="zzz")
    refused(*run(tmp_path, recs + [stray], capsys=capsys), "undeclared")


def test_a_comparator_run_pinning_another_model_is_refused(tmp_path, capsys):
    """Selecting the comparator by arm_id alone pools model pins (memory: 16 pins once)."""
    recs = healthy()
    other = rec(C, "qwen3-8b-sft", task="t9", trace="zzz")
    refused(*run(tmp_path, recs + [other], capsys=capsys), "comparator")


# ------------------------------------------------------------------ refusals: the thinking probe


def test_a_missing_probe_record_is_refused(tmp_path, capsys):
    refused(*run(tmp_path, healthy(), probe=False, capsys=capsys), "probe record")


def test_a_probe_record_that_is_not_pass_is_refused(tmp_path, capsys):
    f = tmp_path / "records.json"
    f.write_text(json.dumps(healthy()))
    code = tr.main(
        ["--records", str(f), *FAST_BOOT, "--serve-record", str(serve_record(tmp_path))]
        + ["--probe-record", str(probe_record(tmp_path, verdict="REFUSE"))]
    )
    refused(code, capsys.readouterr().out, "verdict")


def _argv(tmp_path, records, probes, serves):
    f = tmp_path / "records.json"
    f.write_text(json.dumps(records))
    argv = ["--records", str(f), *FAST_BOOT, "--out", str(tmp_path / "out.json")]
    argv += ["--tau2-data-dir", str(tau2_data(tmp_path))]
    for x in probes:
        argv += ["--probe-record", str(x)]
    for x in serves:
        argv += ["--serve-record", str(x)]
    for lp in launch_records(tmp_path, records):
        argv += ["--launch-record", str(lp)]
    return argv


def test_the_probe_must_cover_the_comparator_and_the_trained_models_only(tmp_path, capsys):
    """CHANGED 2026-09-23 (item 4; CLAUDE.md rule 4): the probe used to be required to cover every
    model in the role pins, the frontier drafter and answerer included. The required set is now
    exactly the comparator plus the trained models; gateway roles are route-proved separately."""
    recs = healthy()
    code = tr.main(_argv(tmp_path, recs, [probe_record(tmp_path)], [serve_record(tmp_path)]))
    out = capsys.readouterr().out
    assert code == 0, out
    code = tr.main(
        _argv(tmp_path, recs, [probe_record(tmp_path, models=(BASE, S1))], [serve_record(tmp_path)])
    )
    out = capsys.readouterr().out
    refused(code, out, "probe record")
    assert S2 in out, "the uncovered model must be named"


def test_a_run_pinned_outside_the_probed_set_is_refused(tmp_path, capsys):
    """Any run whose questioner pin is not the comparator or a declared trained model."""
    recs = healthy()
    other = rec("self_ask", "qwen3-8b-sft", task="t9", trace="zzz")
    refused(*run(tmp_path, recs + [other], capsys=capsys), "outside the required set")


def test_a_probe_on_another_proxy_than_the_runs_is_refused(tmp_path, capsys):
    """Item 5: the probe's proxy_url, hashed by the harness's own MeteredClient.pin route, must equal
    every run's pins.inquirer.base_url_sha -- or the probe proved a proxy the units did not use."""
    recs = healthy()
    probe = probe_record(tmp_path, proxy="http://127.0.0.1:4999")
    code = tr.main(_argv(tmp_path, recs, [probe], [serve_record(tmp_path)]))
    refused(code, capsys.readouterr().out, "hashes to")


def test_two_probe_jobs_that_agree_pass_and_their_union_covers(tmp_path, capsys):
    """HPC sharding: each LSF job writes its own probe. Job 1 probed base and s1, job 2 s2."""
    recs = healthy()
    j1 = probe_record(tmp_path, models=(BASE, S1), name="_jobs/1/thinking_probe.json")
    j2 = probe_record(tmp_path, models=(S2,), name="_jobs/2/thinking_probe.json")
    code = tr.main(_argv(tmp_path, recs, [j1, j2], [serve_record(tmp_path)]))
    out = capsys.readouterr().out
    assert code == 0, out
    rep = json.loads((tmp_path / "out.json").read_text())
    assert len(rep["provenance"]["probe_records"]) == 2


def test_probe_jobs_with_different_config_shas_are_refused(tmp_path, capsys):
    recs = healthy()
    j1 = probe_record(tmp_path, name="_jobs/1/thinking_probe.json")
    j2 = probe_record(tmp_path, name="_jobs/2/thinking_probe.json", config_sha="d" * 64)
    code = tr.main(_argv(tmp_path, recs, [j1, j2], [serve_record(tmp_path)]))
    refused(code, capsys.readouterr().out, "config_sha256")


def test_probe_jobs_on_two_proxy_urls_are_refused(tmp_path, capsys):
    recs = healthy()
    j1 = probe_record(tmp_path, name="_jobs/1/thinking_probe.json")
    j2 = probe_record(tmp_path, name="_jobs/2/thinking_probe.json", proxy="http://127.0.0.1:4002")
    code = tr.main(_argv(tmp_path, recs, [j1, j2], [serve_record(tmp_path)]))
    refused(code, capsys.readouterr().out, "proxy_url")


def test_one_refuse_probe_among_passes_is_refused(tmp_path, capsys):
    recs = healthy()
    bad = probe_record(tmp_path, name="_jobs/1/thinking_probe.json", verdict="REFUSE")
    ok1 = probe_record(tmp_path, name="_jobs/2/thinking_probe.json")
    ok2 = probe_record(tmp_path, name="_jobs/3/thinking_probe.json")
    code = tr.main(_argv(tmp_path, recs, [bad, ok1, ok2], [serve_record(tmp_path)]))
    refused(code, capsys.readouterr().out, "verdict")


def test_probe_and_serve_records_are_found_under_a_jobs_directory(tmp_path, capsys):
    """`<root>/<suite>/_jobs/*/thinking_probe.json` and `.../SERVED.*.json`, by directory or glob."""
    recs = healthy()
    jobs = tmp_path / "root" / "tau2_retail" / "_jobs"
    probe_record(jobs, models=(BASE, S1), name="911001/thinking_probe.json")
    probe_record(jobs, models=(S2,), name="911002/thinking_probe.json")
    serve_record(jobs, name="911001/SERVED.911001.json", job="911001")
    serve_record(jobs, name="911002/SERVED.911002.json", job="911002")
    code = tr.main(_argv(tmp_path, recs, [jobs], [str(jobs / "*" / "SERVED.*.json")]))
    out = capsys.readouterr().out
    assert code == 0, out
    rep = json.loads((tmp_path / "out.json").read_text())
    assert len(rep["provenance"]["probe_records"]) == 2
    assert [r["job"] for r in rep["provenance"]["serve_records"]] == ["911001", "911002"]


def test_a_probe_location_that_holds_no_record_is_refused(tmp_path, capsys):
    recs = healthy()
    empty = tmp_path / "root" / "tau2_retail" / "_jobs"
    empty.mkdir(parents=True)
    code = tr.main(_argv(tmp_path, recs, [empty], [serve_record(tmp_path)]))
    refused(code, capsys.readouterr().out, "no probe record found")


def test_two_serve_jobs_that_agree_pass(tmp_path, capsys):
    recs = healthy()
    a = serve_record(tmp_path, name="_jobs/1/SERVED.911001.json", job="911001")
    b = serve_record(tmp_path, name="_jobs/2/SERVED.911002.json", job="911002")
    code = tr.main(_argv(tmp_path, recs, [probe_record(tmp_path)], [a, b]))
    out = capsys.readouterr().out
    assert code == 0, out


def test_an_adapter_sha_that_differs_between_serve_jobs_is_refused(tmp_path, capsys):
    """Both shas carry the expected 16-hex prefix, so only the cross-job agreement can catch it."""
    recs = healthy()
    a = serve_record(tmp_path, name="_jobs/1/SERVED.911001.json", job="911001")
    other_s1 = "c33d3e6e0b7fd666" + "9" * 48
    b = serve_record(
        tmp_path,
        adapters={S1: {"sha256": other_s1}, S2: {"sha256": S2_SHA}},
        name="_jobs/2/SERVED.911002.json",
        job="911002",
    )
    code = tr.main(_argv(tmp_path, recs, [probe_record(tmp_path)], [a, b]))
    refused(code, capsys.readouterr().out, "sha256")


def test_the_expected_prefixes_are_the_serve_records_16_hex(tmp_path, capsys):
    """Item 8: c33d3e6e0b7fd666 (s1) and 8e8603830b153535 (s2). A sha agreeing on the first 8 hex
    only must refuse."""
    assert [p for _, _, p in tr.TRAINED_SEEDS] == ["c33d3e6e0b7fd666", "8e8603830b153535"]
    recs = healthy()
    short = serve_record(
        tmp_path,
        adapters={S1: {"sha256": "c33d3e6e" + "0" * 56}, S2: {"sha256": S2_SHA}},
    )
    code = tr.main(_argv(tmp_path, recs, [probe_record(tmp_path)], [short]))
    refused(code, capsys.readouterr().out, "c33d3e6e0b7fd666")


def test_the_output_records_every_probe_and_serve_record_it_read(tmp_path, capsys):
    """Each probe's file sha256, config_sha256 and per-route chat_template_kwargs, and each serve
    record's job, host, port and written_at, go into the JSON."""
    recs = healthy()
    code = tr.main(_argv(tmp_path, recs, [probe_record(tmp_path)], [serve_record(tmp_path)]))
    assert code == 0, capsys.readouterr().out
    prov = json.loads((tmp_path / "out.json").read_text())["provenance"]
    (pr,) = prov["probe_records"]
    assert pr["config_sha256"] == PROXY_CONFIG_SHA and len(pr["file_sha256"]) == 64
    assert {r["model"]: r["chat_template_kwargs"] for r in pr["routes"]} == {
        m: {"enable_thinking": False} for m in (BASE, S1, S2)
    }
    assert prov["proxy_url_hash_equals_every_questioner_base_url_sha"] is True
    (sr,) = prov["serve_records"]
    assert (sr["job"], sr["host"], sr["port"]) == ("910128", "gpu-n595", 8302)
    assert sr["written_at"] == "2026-09-22T21:28:03+00:00"
    assert sr["adapters"] == {S1: S1_SHA, S2: S2_SHA}


def test_a_pass_probe_with_a_thinking_route_is_refused(tmp_path, capsys):
    """An internally inconsistent record: PASS beside a route that detected thinking."""
    p = probe_record(tmp_path)
    rec_ = json.loads(p.read_text())
    rec_["routes"][0]["thinking_detected"] = True
    p.write_text(json.dumps(rec_))
    f = tmp_path / "records.json"
    f.write_text(json.dumps(healthy()))
    code = tr.main(
        ["--records", str(f), *FAST_BOOT, "--serve-record", str(serve_record(tmp_path))]
        + ["--probe-record", str(p)]
    )
    refused(code, capsys.readouterr().out, "thinking_detected")


# ------------------------------------------------------------------ refusals: the endpoint


def test_an_ok_run_without_a_reward_is_refused_not_read_as_failure(tmp_path, capsys):
    """fork_paired.succeeded reads an absent tau_reward as 0; the actuator says absence is absence."""
    recs = healthy()
    del recs[1]["native"]["tau_reward"]
    refused(*run(tmp_path, recs, capsys=capsys), "tau_reward")


def test_a_key_that_disagrees_with_its_fields_is_refused(tmp_path, capsys):
    recs = healthy()
    recs[1]["key"][8] = 99
    refused(*run(tmp_path, recs, capsys=capsys), "key")


def test_env_call_routes_that_disagree_are_refused(tmp_path, capsys):
    recs = healthy()
    recs[1]["_env_calls_from_outcome"] = 9
    refused(*run(tmp_path, recs, capsys=capsys), "n_env_calls")


def test_an_unmapped_tool_is_refused_not_counted_as_zero(tmp_path, capsys):
    recs = healthy()
    recs[1]["_env_tool_counts"] = {"not_a_real_tool": 5}
    refused(*run(tmp_path, recs, capsys=capsys), "no declared type")


# ------------------------------------------------------------------ absence is counted, never dropped


def test_a_missing_arm_is_counted_per_arm_and_variant_with_a_worst_case_line(tmp_path, capsys):
    recs = [
        r
        for r in healthy()
        if not (
            r["task_id"] == "t2" and r["key"][9] == "tau2_base" and r["suite_id"] == "tau2_retail"
        )
        or r["arm_id"] == T
    ]
    code, out = run(tmp_path, recs, capsys=capsys)
    assert code == 0, out
    assert "comparator missing at 1 slot" in out, out
    assert "WORST CASE" in out
    worst = [ln for ln in out.splitlines() if "WORST CASE" in ln]
    assert worst and all(" of 2, " in ln and "split" in ln for ln in worst), worst
    rep = json.loads((tmp_path / "out.json").read_text())
    cell = next(
        c
        for c in rep["cells"]
        if c["suite"] == "tau2_retail" and c["variant"] == "tau2_base" and c["seed_label"] == "s1"
    )
    assert cell["census"]["control_missing"] == 1
    assert cell["census"]["n_slots"] == 2
    assert cell["n_pairs"] == 1


def test_a_non_ok_unit_is_counted_and_imputed(tmp_path, capsys):
    recs = healthy()
    bad = next(
        r
        for r in recs
        if r["arm_id"] == T
        and r["_manifest"]["pins"]["inquirer"]["model_id"] == S1
        and r["key"][9] == "tau2_base"
        and r["suite_id"] == "tau2_retail"
        and r["task_id"] == "t1"
    )
    bad["status"] = "error"
    for f in ("native", "spent", "n_env_calls", "n_asks"):
        bad.pop(f)
    code, out = run(tmp_path, recs, capsys=capsys)
    assert code == 0, out
    rep = json.loads((tmp_path / "out.json").read_text())
    cell = next(
        c
        for c in rep["cells"]
        if c["suite"] == "tau2_retail" and c["variant"] == "tau2_base" and c["seed_label"] == "s1"
    )
    assert cell["non_ok"] == {"trained": {"error": 1}, "control": {}}
    assert cell["census"]["trained_missing"] == 1
    worst = cell["guard"]["sensitivity"]["worst_for_trained"]
    # success guard. t1: trained imputed 0 against comparator 0 -> 0; t2: 1 - 0 -> +1. Mean 0.5.
    assert worst["task_level"]["mean_delta"] == pytest.approx(0.5)
    best = cell["guard"]["sensitivity"]["worst_for_comparator"]
    assert best["task_level"]["mean_delta"] == pytest.approx(1.0)
    # primary follow-ups (4 everywhere): the missing trained unit is imputed at the cell's observed
    # range, so both worst cases read 0 here -- and the imputation range is recorded
    pw = cell["primary"]["sensitivity"]["worst_for_trained"]
    assert pw["imputation_range"] == [4.0, 4.0]


def test_an_absent_prefix_turn_count_refuses_now_that_it_is_the_primary(tmp_path, capsys):
    """CHANGED with amendment 7 (CLAUDE.md rule 4): follow-up turns were a secondary, and an absent
    n_prefix_user_turns read NOT RECORDED. They are now the PRIMARY; follow_ups() reads an absent
    prefix as 0, which would charge a fork for its own prefix, so absence refuses."""
    recs = healthy()
    del recs[1]["n_prefix_user_turns"]
    refused(*run(tmp_path, recs, capsys=capsys), "n_prefix_user_turns")


def test_an_absent_env_call_count_reads_not_recorded_not_a_traceback(tmp_path, capsys):
    recs = healthy()
    for f in (
        "n_env_calls",
        "_env_calls_from_outcome",
        "_env_tool_counts",
        "_env_requestor_counts",
    ):
        del recs[1][f]
    code, out = run(tmp_path, recs, capsys=capsys)
    assert code == 0, out
    assert "[env tool calls (n_env_calls)] NOT RECORDED" in out


def test_a_malformed_probe_route_is_refused(tmp_path, capsys):
    p = probe_record(tmp_path)
    rec_ = json.loads(p.read_text())
    rec_["routes"].append("not-a-route")
    p.write_text(json.dumps(rec_))
    f = tmp_path / "records.json"
    f.write_text(json.dumps(healthy()))
    code = tr.main(
        ["--records", str(f), *FAST_BOOT, "--serve-record", str(serve_record(tmp_path))]
        + ["--probe-record", str(p)]
    )
    refused(code, capsys.readouterr().out, "malformed")


def test_an_absent_refused_count_refuses_a_gated_run_and_reads_not_recorded_on_legacy(
    tmp_path, capsys
):
    """CHANGED 2026-09-23 with the final enforcement fields (ba48ab0; CLAUDE.md rule 4): this test
    read an absent n_refused_budget as NOT RECORDED on any run. On a refuse_and_tell run the field is
    final and must equal len(refused_budget_calls), so absence there is a refusal; NOT RECORDED is
    what a legacy population, which predates the field, still prints."""
    recs = healthy()
    for r in recs:
        del r["n_refused_budget"]
    refused(*run(tmp_path, recs, capsys=capsys), "n_refused_budget")

    code = tr.main(_legacy_argv(tmp_path, _legacy_records()) + ["--validation-legacy-population"])
    out = capsys.readouterr().out
    assert code == 0, out
    assert "n_refused_budget NOT RECORDED" in out


# ------------------------------------------------------------------ known answers


def _known_answer_records() -> list:
    """One suite, one variant, both seeds; the comparator is shared.

    slot (task, trace, k, run seed)   C   s1  s2
    A a1 4 0                          0   1   0
    A a2 6 0                          0   0   1
    B b1 4 0                          1   0   1
    B b1 4 1                          1   0   --   (s2 never ran here)
    C c1 4 0                          1   1   0
    D d1 4 0                          0   1   1

    s1 pair deltas: A {+1, 0}, B {-1, -1}, C {0}, D {+1}  (6 pairs)
      task deltas A +0.5, B -1, C 0, D +1 -> up 2, down 1, tied 1; mean +0.125
      sign test 2-1 over 3: p = 2 * (1 + 3) / 8 = 1.0; floor (3-0) = 2 / 8 = 0.25
      pair 2x2 (NOT QUOTABLE): both 1 (C), trained_only 2 (a1, D), control_only 2 (B x2), neither 1
      pair-level mean 0/6 = 0, which is NOT the task-level mean.
    s2 pair deltas: A {0, +1}, B {0}, C {-1}, D {+1}  (5 pairs)
      task deltas A +0.5, B 0, C -1, D +1 -> up 2, down 1, tied 1; mean +0.125
    seeds-pooled, seeds averaged WITHIN a task first:
      A (0.5 + 0.5)/2 = 0.5, B (-1 + 0)/2 = -0.5, C (0 - 1)/2 = -0.5, D (1 + 1)/2 = 1
      -> up 2, down 2, tied 0; mean 0.5/4 = +0.125; floor (4-0) = 2/16 = 0.125
      Pooling PAIRS instead would give B = (-1 - 1 + 0)/3 = -0.6667 and a mean of +0.0833.

    CORRECTED 2026-09-23 (CLAUDE.md rule 4): the first draft of this docstring read s2's D as 0
    and expected (up, down, tied) = (1, 1, 2), mean -0.125. The reader returned (2, 1, 1); the
    table above gives D: C=0, s2=1, i.e. +1. The test encoded a wrong belief, not the code.
    """
    rows = [
        ("A", "a1", 4, 0, 0.0, 1.0, 0.0),
        ("A", "a2", 6, 0, 0.0, 0.0, 1.0),
        ("B", "b1", 4, 0, 1.0, 0.0, 1.0),
        ("B", "b1", 4, 1, 1.0, 0.0, None),
        ("C", "c1", 4, 0, 1.0, 1.0, 0.0),
        ("D", "d1", 4, 0, 0.0, 1.0, 1.0),
    ]
    out = []
    for task, trace, k, seed, c, s1, s2 in rows:
        kw = dict(task=task, trace=trace, k=k, seed=seed)
        out.append(rec(C, BASE, reward=c, **kw))
        out.append(rec(T, S1, reward=s1, **kw))
        if s2 is not None:
            out.append(rec(T, S2, reward=s2, **kw))
    return out


def test_known_answer_task_level_sign_counts_floor_and_delta(tmp_path, capsys):
    code, out = run(
        tmp_path,
        _known_answer_records(),
        "--suites",
        "tau2_retail",
        "--variants",
        "tau2_base",
        capsys=capsys,
    )
    assert code == 0, out
    rep = json.loads((tmp_path / "out.json").read_text())
    # CHANGED with amendment 7: task success is the co-primary GUARD; the primary is follow-ups
    s1 = next(c for c in rep["cells"] if c["seed_label"] == "s1")
    t = s1["guard"]["task_level"]
    assert (t["n_tasks"], t["up"], t["down"], t["tied"]) == (4, 2, 1, 1)
    assert t["mean_delta"] == pytest.approx(0.125)
    assert t["sign_test_p"] == pytest.approx(1.0)
    assert t["sign_test_floor"] == pytest.approx(0.25)
    p = s1["guard"]["pair_level_NOT_QUOTABLE"]
    assert (p["both"], p["trained_only"], p["control_only"], p["neither"]) == (1, 2, 2, 1)
    assert p["mean_delta"] == pytest.approx(0.0)
    assert s1["n_pairs"] == 6

    s2 = next(c for c in rep["cells"] if c["seed_label"] == "s2")
    t2 = s2["guard"]["task_level"]
    assert (t2["up"], t2["down"], t2["tied"]) == (2, 1, 1)
    assert t2["mean_delta"] == pytest.approx(0.125)
    assert s2["census"]["trained_missing"] == 1, "s2 never ran at B/b1/4/seed1"

    # CHANGED 2026-09-23: Table 1's code decides (pooled_symmetric, table1_by_seed.py:75-83):
    # every training seed's pairs go into ONE task mean, so the default is pair_mean and the
    # pooled line reads +0.0833 (B = (-1 - 1 + 0)/3), not the seed_mean +0.125.
    pooled = rep["pooled"][0]["guard"]["task_level"]
    assert (pooled["up"], pooled["down"], pooled["tied"]) == (2, 2, 0)
    assert pooled["split"] == "2-2 of 4, 0 tied"
    assert pooled["mean_delta"] == pytest.approx(1 / 12), "pairs of every seed in one task mean"
    assert pooled["task_deltas"]["B"] == pytest.approx(-2 / 3)
    assert pooled["sign_test_floor"] == pytest.approx(0.125)
    assert "NOT QUOTABLE" in out
    assert "seeds-pooled" in out


def test_the_bootstrap_is_read_at_every_declared_resample_count_and_seed(tmp_path, capsys):
    code, _ = run(
        tmp_path,
        _known_answer_records(),
        "--suites",
        "tau2_retail",
        "--variants",
        "tau2_base",
        capsys=capsys,
    )
    assert code == 0
    rep = json.loads((tmp_path / "out.json").read_text())
    ivs = rep["cells"][0]["primary"]["task_level"]["intervals"]
    assert sorted((i["n_boot"], i["seed"]) for i in ivs) == sorted(
        (n, s) for n in (40, 80, 120) for s in (101, 202, 303)
    )
    assert rep["rules"]["boot_seeds"] == [101, 202, 303]


# ------------------------------------------------------------------ amendment 1 (7d08c06)


def _known(tmp_path, capsys, *extra):
    code, out = run(
        tmp_path,
        _known_answer_records(),
        "--suites",
        "tau2_retail",
        "--variants",
        "tau2_base",
        *extra,
        capsys=capsys,
    )
    assert code == 0, out
    return json.loads((tmp_path / "out.json").read_text()), out


def test_every_cell_quotes_its_informative_split_beside_its_interval(tmp_path, capsys):
    """Amendment 1: `6-1 of 25, 18 tied` next to the interval, so an interval over 7 informative
    tasks cannot read like one over 25. Known answer: s1 is 2 up, 1 down, 1 tied of 4."""
    rep, out = _known(tmp_path, capsys, "--printed-interval", "10000:0")
    s1 = next(c for c in rep["cells"] if c["seed_label"] == "s1")
    assert s1["guard"]["task_level"]["split"] == "2-1 of 4, 1 tied"
    assert rep["pooled"][0]["guard"]["task_level"]["split"] == "2-2 of 4, 0 tied"
    # the primary (follow-ups, 4 everywhere in this fixture) is tied at every task
    assert s1["primary"]["task_level"]["split"] == "0-0 of 4, 4 tied"
    lines = [ln for ln in out.splitlines() if "10000 resamples, seed 0" in ln]
    assert any("2-1 of 4, 1 tied" in ln for ln in lines), lines


def test_the_printed_interval_is_10000_at_seed_0_by_table_1s_path(tmp_path, capsys):
    """The printed interval is n_boot 10000, seed 0, and is the interval paired_difference gives
    on the per-task arm means -- the call Table 1 makes -- to float noise."""
    from pi_eval.stats.inference import paired_difference

    rep, _ = _known(tmp_path, capsys, "--printed-interval", "10000:0")
    s1 = next(c for c in rep["cells"] if c["seed_label"] == "s1")
    pi = s1["guard"]["task_level"]["printed_interval"]
    assert (pi["n_boot"], pi["seed"]) == (10_000, 0)
    # s1 per-task arm means from the known-answer table: (trained, comparator)
    a = {"A": 0.5, "B": 0.0, "C": 1.0, "D": 1.0}
    b = {"A": 0.0, "B": 1.0, "C": 1.0, "D": 0.0}
    est = paired_difference(a, b, n_boot=10_000, seed=0)
    assert pi["lo"] == pytest.approx(est.ci_lo, abs=1e-9)
    assert pi["hi"] == pytest.approx(est.ci_hi, abs=1e-9)
    assert rep["rules"]["printed_interval"] == {"n_boot": 10_000, "seed": 0}


def test_the_declared_verdict_ladder_is_the_default():
    a = tr.build_parser().parse_args(["--records", "x.json"])
    assert a.boot_resamples == "1000,10000,50000"
    assert a.boot_seeds == "101,202,303"
    assert a.printed_interval == "10000:0"
    assert a.holm_family == "suite_variant_seed"
    assert a.seed_pooling == "pair_mean"


def test_the_record_states_what_the_interval_is(tmp_path, capsys):
    rep, out = _known(tmp_path, capsys)
    s = rep["rules"]["interval_statement"]
    for phrase in ("over tasks", "weights tasks equally", "excludes training-seed variance"):
        assert phrase in s, phrase
    assert s in out


def test_the_seed_pooling_rule_is_named_and_its_alternative_printed_when_unbalanced(
    tmp_path, capsys
):
    """pair_mean -- table1_by_seed.pooled_symmetric's arithmetic, every seed's pairs in one task
    mean -- is the DEFAULT; seed_mean is printed beside it whenever the two differ. They agree when
    each seed has the same pairs in a task and differ here, where s2 never ran B/b1/4/seed1:
    pair_mean B = (-1 - 1 + 0)/3, mean +0.0833; seed_mean B = (-1 + 0)/2 = -0.5, mean +0.125."""
    rep, out = _known(tmp_path, capsys)
    p = rep["pooled"][0]
    assert p["seed_pooling"] == "pair_mean"
    g = p["guard"]
    assert g["task_level"]["mean_delta"] == pytest.approx(1 / 12)
    assert g["alternative"]["seed_pooling"] == "seed_mean"
    assert g["alternative"]["mean_delta"] == pytest.approx(0.125)
    assert p["n_tasks_unbalanced_across_seeds"] == 1
    assert "under seed_mean the mean delta is +0.1250" in out

    rep2, _ = _known(tmp_path, capsys, "--seed-pooling", "seed_mean")
    q = rep2["pooled"][0]
    assert q["seed_pooling"] == "seed_mean"
    assert q["guard"]["task_level"]["mean_delta"] == pytest.approx(0.125)
    assert q["guard"]["task_level"]["task_deltas"]["B"] == pytest.approx(-0.5)


def test_decided_requires_every_interval_at_the_deciding_count_to_exclude_zero():
    def iv(n, lo, hi):
        return {"n_boot": n, "seed": 0, "lo": lo, "hi": hi}

    assert tr.decided([iv(50_000, 0.01, 0.2)] * 3, 50_000) is True
    assert tr.decided([iv(50_000, -0.2, -0.01)] * 3, 50_000) is True
    mixed = [iv(50_000, 0.01, 0.2), iv(50_000, 0.01, 0.2), iv(50_000, -0.001, 0.2)]
    assert tr.decided(mixed, 50_000) is False, "one 50k interval touching zero leaves it undecided"
    # smaller counts never decide, whatever they say
    assert tr.decided([iv(1_000, 0.01, 0.2), iv(50_000, -0.1, 0.2)], 50_000) is False


def test_a_bound_on_zero_up_to_float_noise_does_not_decide():
    """Task deltas over paired binary outcomes put bootstrap mass exactly at 0, and the BCa
    percentile returns it with float noise: the legacy retail reading (VALIDATION ONLY) printed
    lower bounds of -1.84e-18, -2.22e-18 and 0.0 at 1k/10k. A +1e-18 must not read as 'excludes 0'.
    """
    tiny = [{"n_boot": 50_000, "seed": s, "lo": 1e-17, "hi": 0.2} for s in (101, 202, 303)]
    assert tr.decided(tiny, 50_000) is False
    tiny_neg = [{"n_boot": 50_000, "seed": s, "lo": -0.2, "hi": -1e-17} for s in (101, 202, 303)]
    assert tr.decided(tiny_neg, 50_000) is False


def test_disagreeing_criteria_are_named_not_resolved():
    """The legacy retail reading (VALIDATION ONLY) had every 50k BCa interval excluding 0 while
    the task sign test read 6-1, p = 0.125, Holm 0.25. The reader names the disagreement; which
    criterion governs is the paper lane's rule, not a choice the reader makes silently."""
    assert tr.criteria_note(True, 0.25) == "DISAGREE: interval decided, Holm sign test not < 0.05"
    assert tr.criteria_note(False, 0.01) == "DISAGREE: Holm sign test < 0.05, interval undecided"
    assert tr.criteria_note(True, 0.01) == "agree"
    assert tr.criteria_note(False, 0.5) == "agree"


def test_holm_adjusts_over_the_listed_family():
    adj = tr.holm([0.01, 0.04, 0.03, 0.5])
    assert adj == pytest.approx([0.04, 0.09, 0.09, 0.5])
    assert tr.holm([float("nan"), 0.01]) == pytest.approx([1.0, 0.02])


def test_the_holm_family_is_printed_with_its_members(tmp_path, capsys):
    code, out = run(tmp_path, healthy(), capsys=capsys)
    assert code == 0, out
    assert "Holm family (8 cells" in out
    rep = json.loads((tmp_path / "out.json").read_text())
    assert len(rep["holm"]["family"]) == 8


def test_the_output_never_names_a_winner(tmp_path, capsys):
    code, out = run(tmp_path, healthy(), capsys=capsys)
    assert code == 0
    low = out.lower()
    for word in ("wins", "improves", "better", "worse", "beats"):
        assert word not in low, word


# ------------------------------------------------------------------ the validation flag


#: what the withdrawn (pre-gate) status.json never carried; the legacy fixture drops all of it,
#: as the real ecdbb88a / 0276ce81 records lack it
GATED_ERA_FIELDS = (
    "_manifest",
    "budget_enforcement",
    "budget_cap",
    "budget_exempt_tools",
    "n_refused_budget",
    "refused_budget_calls",
    "n_prefix_env_calls",
    "n_live_env_calls",
    "n_live_env_calls_generic",
    "n_live_env_calls_nongeneric",
    "n_live_env_calls_nongeneric_ok",
    "n_gate_charged",
    "gate_census_agrees",
    "rollout1_stop_reason",
    "rollout1_n_asks",
    "first_live_call_executed",
    "spent_before_first_live_call",
    "_env_call_seq",
    "tool_call_cap",
    "n_transfer_to_human",
    "_ask_retrieved_uids",
)


def _legacy_records() -> list:
    recs = []
    for task, trace in (("t1", "aaa"), ("t2", "bbb")):
        for arm, reward in ((C, 0.0), (T, 1.0)):
            r = rec(arm, BASE, task=task, trace=trace, reward=reward)
            for f in GATED_ERA_FIELDS:
                r.pop(f)
            r["spent"] = {"retrieval_calls": 25.0, "post_hoc_budget_overrun": 1.0}
            recs.append(r)
    return recs


def _legacy_argv(tmp_path, recs):
    f = tmp_path / "legacy.json"
    f.write_text(json.dumps(recs))
    return [
        "--records",
        str(f),
        *FAST_BOOT,
        "--suites",
        "tau2_retail",
        "--variants",
        "tau2_base",
        "--trained",
        "s0=qwen3-8b-dpo-stacked-notdone-both:80a56b74",
        "--out",
        str(tmp_path / "legacy_out.json"),
    ]


def test_the_legacy_population_is_refused_without_the_flag(tmp_path, capsys):
    code = tr.main(_legacy_argv(tmp_path, _legacy_records()))
    assert code == tr.EXIT_REFUSED, capsys.readouterr().out


def test_the_validation_flag_stamps_every_line_and_the_json(tmp_path, capsys):
    code = tr.main(_legacy_argv(tmp_path, _legacy_records()) + ["--validation-legacy-population"])
    out = capsys.readouterr().out
    assert code == 0, out
    lines = [ln for ln in out.splitlines() if ln.strip()]
    assert lines and all(tr.VALIDATION_STAMP in ln for ln in lines), [
        ln for ln in lines if tr.VALIDATION_STAMP not in ln
    ]
    rep = json.loads((tmp_path / "legacy_out.json").read_text())
    assert rep["stamp"] == tr.VALIDATION_STAMP
    assert all(c["stamp"] == tr.VALIDATION_STAMP for c in rep["cells"])
    assert "budget_enforcement" in " ".join(rep["disabled_checks"])


def test_the_validation_flag_does_not_disable_the_provenance_refusals(tmp_path, capsys):
    """It disables the enforcement / weights / probe refusals ONLY. Two code_versions still refuse."""
    recs = _legacy_records()
    recs[0]["code_version"] = "b" * 40
    code = tr.main(_legacy_argv(tmp_path, recs) + ["--validation-legacy-population"])
    out = capsys.readouterr().out
    refused(code, out, "code_version")
    assert all(tr.VALIDATION_STAMP in ln for ln in out.splitlines() if ln.strip())


def test_the_validation_flag_still_refuses_a_dirty_run(tmp_path, capsys):
    recs = _legacy_records()
    recs[0]["dirty"] = True
    code = tr.main(_legacy_argv(tmp_path, recs) + ["--validation-legacy-population"])
    refused(code, capsys.readouterr().out, "dirty")


# ------------------------------------------------------------------ extract_records


def _write_run(
    root: Path,
    run_id: str,
    status: dict,
    manifest_: dict,
    env_calls: list,
    asks=None,
    llm_calls=None,
    questions=None,
) -> Path:
    d = root / run_id
    d.mkdir(parents=True)
    if llm_calls is not None:  # calls.jsonl: one CallTelemetry row per LLM call
        (d / "calls.jsonl").write_text("".join(json.dumps(x) + "\n" for x in llm_calls))
    (d / "status.json").write_text(json.dumps(status))
    (d / "manifest.json").write_text(json.dumps(manifest_))
    (d / "outcome.json").write_text(json.dumps({"env_calls": env_calls}))
    if asks is not None:  # turns.jsonl: one row per ask, as the runner writes it
        rows = [
            {"action_kind": "ask", "turn_idx": i, "retrieved_uids": u}
            | ({"question": questions[i]} if questions is not None else {})
            for i, u in enumerate(asks)
        ]
        (d / "turns.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
    return d


def test_extract_embeds_the_manifest_and_requestor_counts(tmp_path):
    calls = [
        {"tool_name": "get_user_details", "requestor": "assistant", "ok": True},
        {"tool_name": "get_order_details", "requestor": "assistant", "ok": False},
        {"tool_name": "toggle_airplane_mode", "requestor": "user", "ok": True},
    ]
    _write_run(tmp_path, "r1", {"run_id": "r1", "status": "ok"}, manifest(BASE), calls)
    recs = extract_records.extract(tmp_path)
    assert len(recs) == 1
    r = recs[0]
    assert r["_manifest"]["pins"]["inquirer"]["model_id"] == BASE
    assert r["_env_requestor_counts"] == {"assistant": 2, "user": 1}
    assert r["_env_calls_from_outcome"] == 3
    # ORDERED, with the ok flag: the gate recount reads it from index n_prefix_env_calls on
    assert r["_env_call_seq"] == [
        ["get_user_details", True],
        ["get_order_details", False],
        ["toggle_airplane_mode", True],
    ]


def test_extract_reads_a_symlink_farm(tmp_path):
    """Isolated stores are symlink farms; Python 3.12's rglob does not descend a symlinked dir.

    Measured before the fix: `Path(farm).rglob("status.json")` returned [] on a farm whose one
    entry is a symlink to a real run directory, so the extract read an isolated store as EMPTY.
    """
    real = tmp_path / "real"
    _write_run(real, "r1", {"run_id": "r1", "status": "ok"}, manifest(BASE), [])
    farm = tmp_path / "farm"
    farm.mkdir()
    (farm / "r1").symlink_to(real / "r1", target_is_directory=True)
    assert len(extract_records.extract(farm)) == 1


def _write_store(root: Path, records, misplace: dict | None = None, pin_dir=None) -> None:
    """The launcher's layout (0212fe0): `<root>/<suite>/<pin>/<run_id>/`. `misplace` maps a run_id
    to the pin sub-root it is (wrongly) written under; `pin_dir` maps a pin to its directory name
    (the extra-pin driver's STORE_DIR)."""
    for r in records:
        st = {k: v for k, v in r.items() if not k.startswith("_")}
        calls = [
            {"tool_name": name, "ok": ok, "requestor": "assistant"}
            for name, ok in r["_env_call_seq"]
        ]
        pin = (misplace or {}).get(r["run_id"], r["_manifest"]["pins"]["inquirer"]["model_id"])
        _write_run(
            root / r["suite_id"] / (pin_dir(pin) if pin_dir else pin),
            r["run_id"],
            st,
            r["_manifest"],
            calls,
            asks=r["_ask_retrieved_uids"],
            llm_calls=r.get("_llm_calls"),
            questions=r.get("_ask_questions"),
        )


def _store_argv(tmp_path, root, records):
    argv = ["--store", str(root), *FAST_BOOT, "--serve-record", str(serve_record(tmp_path))]
    argv += ["--tau2-data-dir", str(tau2_data(tmp_path))]
    argv += ["--probe-record", str(probe_record(tmp_path)), "--out", str(tmp_path / "o.json")]
    for lp in launch_records(tmp_path, records):
        argv += ["--launch-record", str(lp)]
    return argv


def test_the_store_flag_reads_the_pin_sub_roots(tmp_path, capsys):
    """`--store <root>` reads `<root>/<suite>/<pin>/` for every declared suite and pin."""
    root = tmp_path / "runs"
    recs = healthy()
    _write_store(root, recs)
    (root / "tau2_retail" / "_jobs" / "job1").mkdir(parents=True)  # not a pin: listed, not read
    code = tr.main(_store_argv(tmp_path, root, recs))
    out = capsys.readouterr().out
    assert code == 0, out
    rep = json.loads((tmp_path / "o.json").read_text())
    assert rep["source"]["kind"] == "store"
    assert rep["provenance"]["n_records"] == 24
    assert rep["source"]["layout"] == "<root>/<suite>/<pin>/"
    assert len(rep["source"]["sub_roots"]) == 6, "2 suites x 3 pins"
    assert rep["source"]["not_read"] == ["tau2_retail/_jobs"]


def test_a_run_under_another_pins_sub_root_is_refused(tmp_path, capsys):
    root = tmp_path / "runs"
    recs = healthy()
    s1_run = next(r for r in recs if r["_manifest"]["pins"]["inquirer"]["model_id"] == S1)
    _write_store(root, recs, misplace={s1_run["run_id"]: S2})
    code = tr.main(_store_argv(tmp_path, root, recs))
    refused(code, capsys.readouterr().out, "sub-root")


def test_a_missing_pin_sub_root_is_refused(tmp_path, capsys):
    root = tmp_path / "runs"
    recs = [r for r in healthy() if r["_manifest"]["pins"]["inquirer"]["model_id"] != S2]
    _write_store(root, recs)
    code = tr.main(_store_argv(tmp_path, root, recs))
    refused(code, capsys.readouterr().out, "sub-root")


def test_extracted_records_keep_their_sub_root_and_are_checked_against_it(tmp_path, capsys):
    """Records extracted from `<root>` carry `_relpath` = `<suite>/<pin>/<run_id>`, so the same
    pin check holds on a `--records` read."""
    root = tmp_path / "runs"
    recs = healthy()
    s1_run = next(r for r in recs if r["_manifest"]["pins"]["inquirer"]["model_id"] == S1)
    _write_store(root, recs, misplace={s1_run["run_id"]: S2})
    got = extract_records.extract(root)
    rel = {r["run_id"]: r["_relpath"] for r in got}
    assert rel[s1_run["run_id"]] == f"tau2_retail/{S2}/{s1_run['run_id']}"
    code, out = run(tmp_path, got, capsys=capsys)
    refused(code, out, "sub-root")


def test_an_s2_run_labelled_as_the_comparator_is_refused(tmp_path, capsys):
    """REGRESSION PIN (passes on the prior code). The only way _pair_forks' arm-id keying could
    pair s1 with s2 is an s2 run carrying the comparator's arm id; the pin check refuses it."""
    recs = healthy()
    stray = rec(C, S2, task="t9", trace="zzz")
    refused(*run(tmp_path, recs + [stray], capsys=capsys), "comparator")


def test_s1_and_s2_at_a_slot_without_the_comparator_make_no_pair(tmp_path, capsys):
    """REGRESSION PIN. s1 and s2 both at retail/base/t2, comparator absent: neither cell pairs there."""
    recs = [
        r
        for r in healthy()
        if not (
            r["suite_id"] == "tau2_retail"
            and r["key"][9] == "tau2_base"
            and r["task_id"] == "t2"
            and r["arm_id"] == C
        )
    ]
    code, out = run(tmp_path, recs, capsys=capsys)
    assert code == 0, out
    rep = json.loads((tmp_path / "out.json").read_text())
    for c in rep["cells"]:
        if c["suite"] == "tau2_retail" and c["variant"] == "tau2_base":
            assert c["n_pairs"] == 1 and c["census"]["control_missing"] == 1, c["seed_label"]


def test_planned_but_absent_slots_count_as_missing(tmp_path, capsys):
    """A slot the launch planned and the store never saw is invisible to the store; the LAUNCH.json
    makes it countable. Planned: retail/base task t3 at every pin; stored: nothing at t3.
    s1 worst case for trained: t1 +1, t2 +1, t3 imputed 0 - 1 = -1 -> mean 1/3, split 2-1 of 3."""
    recs = healthy()
    extra = [
        {
            "suite": "tau2_retail",
            "pin": pin,
            "arm_id": arm,
            "variant": "tau2_base",
            "seed": 0,
            "task_id": "t3",
            "k": 4,
            "trace_sha": "ccc",
        }
        for pin, arm in ((BASE, C), (S1, T), (S2, T))
    ]
    f = tmp_path / "records.json"
    f.write_text(json.dumps(recs))
    argv = ["--records", str(f), *FAST_BOOT, "--out", str(tmp_path / "out.json")]
    argv += ["--serve-record", str(serve_record(tmp_path))]
    argv += ["--probe-record", str(probe_record(tmp_path))]
    argv += ["--tau2-data-dir", str(tau2_data(tmp_path))]
    for lp in launch_records(tmp_path, recs, extra_units=extra):
        argv += ["--launch-record", str(lp)]
    code = tr.main(argv)
    out = capsys.readouterr().out
    assert code == 0, out
    rep = json.loads((tmp_path / "out.json").read_text())
    s1 = next(
        c
        for c in rep["cells"]
        if c["suite"] == "tau2_retail" and c["variant"] == "tau2_base" and c["seed_label"] == "s1"
    )
    assert s1["census"]["n_planned_slots"] == 3
    assert s1["census"]["n_absent_from_store"] == 1
    assert s1["census"]["both_missing"] == 1
    w = s1["guard"]["sensitivity"]["worst_for_trained"]["task_level"]  # success, the guard
    assert w["split"] == "2-1 of 3, 0 tied"
    assert w["mean_delta"] == pytest.approx(1 / 3)


def test_a_run_outside_the_launch_plan_is_refused(tmp_path, capsys):
    recs = healthy()
    f = tmp_path / "records.json"
    f.write_text(json.dumps(recs))
    argv = ["--records", str(f), *FAST_BOOT, "--out", str(tmp_path / "out.json")]
    argv += ["--serve-record", str(serve_record(tmp_path))]
    argv += ["--probe-record", str(probe_record(tmp_path))]
    argv += ["--tau2-data-dir", str(tau2_data(tmp_path))]
    for lp in launch_records(tmp_path, recs[1:]):  # the plan does not hold recs[0]
        argv += ["--launch-record", str(lp)]
    refused(tr.main(argv), capsys.readouterr().out, "launch plan")


def test_a_declared_suite_without_a_launch_record_is_refused(tmp_path, capsys):
    refused(*run(tmp_path, healthy(), launch=False, capsys=capsys), "launch record")


# ------------------------------------------------------------------ the stopping block and --pilot


def no_live_call(r: dict, asks: int = 10) -> dict:
    """A unit whose agent executed no live tool call; its questioner asked `asks` times, below the
    ask cap, so it is NOT starved under separate budgets."""
    r.update(
        n_asks=asks,
        n_env_calls=PREFIX,
        _env_calls_from_outcome=PREFIX,
        _env_tool_counts={"get_user_details": PREFIX},
        _env_requestor_counts={"assistant": PREFIX},
        _env_call_seq=[["get_user_details", True]] * PREFIX,
        n_live_env_calls=0,
        n_live_env_calls_generic=0,
        n_live_env_calls_nongeneric=0,
        n_live_env_calls_nongeneric_ok=0,
        n_gate_charged=0,
        stop_reason="max_turns",
        rollout1_stop_reason="max_turns",
        rollout1_n_asks=asks,
        first_live_call_executed=False,
        spent_before_first_live_call=None,
        _ask_retrieved_uids=[[] for _ in range(asks)],
    )
    r["spent"]["retrieval_calls"] = float(asks)
    return r


def starving(r: dict) -> dict:
    """The exact definition's starved unit: the asks reached the cap and no live call executed."""
    return no_live_call(r, asks=16)


def _stopping_population() -> list:
    """The comparator never makes a live call (10 asks each); s1 does not at retail/base/t1."""
    recs = healthy()
    for r in recs:
        model = r["_manifest"]["pins"]["inquirer"]["model_id"]
        first_s1 = (
            model == S1
            and r["suite_id"] == "tau2_retail"
            and r["key"][9] == "tau2_base"
            and r["task_id"] == "t1"
        )
        if model == BASE or first_s1:
            no_live_call(r)
    return recs


def test_the_stopping_block_reads_zero_starved_by_design_under_separate_budgets(tmp_path, capsys):
    """CHANGED with amendment 7: separate budgets, so starvation is 0 by design and printed so;
    the block still reports rollout-1 stop and asks, executed non-GENERIC calls and the censored
    calls-before-first-call."""
    code, out = run(tmp_path, _stopping_population(), capsys=capsys)
    assert code == 0, out
    rep = json.loads((tmp_path / "out.json").read_text())
    rows = {(r["suite"], r["variant"], r["arm"]): r for r in rep["stopping"]}
    comp = rows[("tau2_retail", "tau2_base", "comparator")]
    assert comp["n_ok"] == 2
    # RENAMED (coordinator, after 9e2c761): a DESCRIPTIVE stopping measure, not a starvation
    assert comp["frac_asks_capped"] == 0.0
    assert (comp["n_asks_capped"], comp["n_asks_capped_defined"]) == (0, 2)
    assert comp["asks_capped_basis"] == ["exact"]
    assert comp["tool_calls_blocked_by_asks"] == "0 by design (separate budgets)"
    assert comp["leak_refused_below_tool_cap"] == 0
    assert comp["frac_zero_executed_nongeneric"] == 1.0
    assert comp["frac_final_rollout_policy_stop"] == 0.0
    assert comp["median_asks_per_unit"] == 10
    s1 = rows[("tau2_retail", "tau2_base", "s1")]
    assert comp["frac_rollout1_policy_stop"] == 0.0 and s1["frac_rollout1_policy_stop"] == 0.5
    assert comp["median_rollout1_asks"] == 10 and s1["median_rollout1_asks"] == 6.5
    assert comp["calls_before_first_executed_tool_call_quartiles"] == [10.0, 10.0, 10.0]
    assert comp["n_no_live_call"] == 2
    assert comp["calls_before_first_quartiles_acting_only"] is None
    assert "STOPPING" in out
    assert (
        "tool calls blocked by asks: 0 by design (separate budgets); leak check: refused calls "
        "below tool cap = 0" in out
    )
    assert "asks reached the ask cap before the first tool call" in out


def test_pilot_prints_only_the_stopping_block_and_status_counts_and_skips_every_bootstrap(
    tmp_path, capsys, monkeypatch
):
    def no_bootstrap(*a, **k):
        raise AssertionError("--pilot must not bootstrap")

    monkeypatch.setattr(tr, "cluster_bootstrap", no_bootstrap)
    recs = _stopping_population()
    recs[1]["status"] = "error"
    recs[1]["error"] = "TimeoutError: unit exceeded 900s"
    code, out = run(tmp_path, recs, "--pilot", capsys=capsys)
    assert code == 0, out
    lines = [ln for ln in out.splitlines() if ln.strip()]
    assert all(tr.PILOT_STAMP in ln for ln in lines), [
        ln for ln in lines if tr.PILOT_STAMP not in ln
    ]
    for absent in ("PRIMARY", "BCa", "verdict ladder", "Holm", "SEEDS-POOLED", "SECONDARY"):
        assert absent not in out, absent
    assert "STOPPING" in out
    assert "TimeoutError: unit exceeded 900s" in out
    rep = json.loads((tmp_path / "out.json").read_text())
    assert rep["pilot"] is True and rep["stamp"] == tr.PILOT_STAMP
    assert "cells" not in rep and rep["stopping"]
    st = {(r["suite"], r["variant"], r["arm"]): r for r in rep["status_counts"]}
    assert st[("tau2_retail", "tau2_base", "s1")]["statuses"] == {"error": 1, "ok": 1}


def test_pilot_still_refuses_a_population_it_must_not_read(tmp_path, capsys):
    recs = _stopping_population()
    recs[0]["dirty"] = True
    code, out = run(tmp_path, recs, "--pilot", capsys=capsys)
    refused(code, out, "dirty")


# ------------------------------------------------------------------ amendment 4: the secondaries


def _with_first_call_spend(recs: list) -> list:
    """The field amendment 4's new secondary needs: ledger spend before the first executed live
    tool call. The healthy units spend 3 asks first; the no-live-call units carry null."""
    for r in recs:
        r.setdefault("spent_before_first_live_call", 3.0)
    return recs


A4 = [
    "asks reached the ask cap before the first tool call",
    "rollout-1 STOP (rollout1_stop_reason == policy_stop)",
    "zero executed non-GENERIC calls",
    "calls spent before the first executed tool call",
]


def test_every_amendment_4_secondary_has_both_arms_and_a_task_level_interval(tmp_path, capsys):
    """Both arms' levels, the paired task-level difference, and ONLY the 10k seed-0 interval (the
    1k/10k/50k ladder is the primary's). Pooled within task across s1 and s2 as the primary is.
    retail/base/s1: comparator made no live call at t1 or t2 (10 asks), s1 none at t1."""
    code, out = run(
        tmp_path,
        _with_first_call_spend(_stopping_population()),
        "--printed-interval",
        "10000:0",
        capsys=capsys,
    )
    assert code == 0, out
    rep = json.loads((tmp_path / "out.json").read_text())
    s1 = next(
        c
        for c in rep["cells"]
        if c["suite"] == "tau2_retail" and c["variant"] == "tau2_base" and c["seed_label"] == "s1"
    )
    for name in A4:
        e = s1["secondary"][name]
        assert e["defined_on_all_units"] is True
        assert e["printed_interval"]["n_boot"] == 10_000 and e["printed_interval"]["seed"] == 0
        assert "intervals" not in e, "the ladder is for the primary only"
    stop = s1["secondary"][A4[1]]
    # s1: t1 max_turns, t2 policy_stop; comparator max_turns at both
    assert (stop["trained_level"], stop["control_level"]) == (0.5, 0.0)
    st = s1["secondary"][A4[0]]
    assert (st["trained_level"], st["control_level"]) == (0.0, 0.0), "0 by design"
    # quoted calls-before-first-call, censored at total spend: s1 t1 10 (censored), t2 3;
    # comparator 10, 10 (both censored). Inclusive quartiles of [3, 10]: 4.75, 6.5, 8.25.
    calls = s1["secondary"][A4[3]]
    assert calls["task_pair_deltas"] == {"t1": [0.0], "t2": [-7.0]}
    assert calls["trained_quartiles"] == [4.75, 6.5, 8.25]
    assert calls["control_quartiles"] == [10.0, 10.0, 10.0]
    assert (calls["trained_n_censored"], calls["control_n_censored"]) == (1, 2)
    assert calls["descriptive_conditional_on_both_acting"]["n_pairs"] == 0
    pooled = next(
        p for p in rep["pooled"] if p["suite"] == "tau2_retail" and p["variant"] == "tau2_base"
    )
    # pair_mean: t1 s1 0, s2 3 - 10 = -7 -> -3.5; t2 s1 -7, s2 -7 -> -7
    assert pooled["secondary"][A4[3]]["task_deltas"] == {"t1": -3.5, "t2": -7.0}
    assert pooled["secondary"][A4[0]]["printed_interval"]["n_boot"] == 10_000
    assert "SECONDARY" in out


def test_the_first_call_fields_refuse_on_a_gated_run_and_read_not_recorded_on_legacy(
    tmp_path, capsys
):
    """CHANGED 2026-09-23 (CLAUDE.md rule 4): an absent spent_before_first_live_call read as NOT
    RECORDED on any run. The enforcement lane now writes it, so on a gated run its absence refuses;
    NOT RECORDED is kept for legacy populations only."""
    recs = _stopping_population()
    del recs[4]["spent_before_first_live_call"]
    refused(*run(tmp_path, recs, capsys=capsys), "spent_before_first_live_call")

    code = tr.main(_legacy_argv(tmp_path, _legacy_records()) + ["--validation-legacy-population"])
    out = capsys.readouterr().out
    assert code == 0, out
    rep = json.loads((tmp_path / "legacy_out.json").read_text())
    row = rep["stopping"][0]
    assert row["frac_asks_capped"] is None and row["frac_rollout1_policy_stop"] is None
    assert row["leak_refused_below_tool_cap"] is None
    assert row["calls_before_first_executed_tool_call_quartiles"] == "NOT RECORDED"
    assert rep["cells"][0]["secondary"][A4[3]] is None
    assert f"[{A4[3]}] NOT RECORDED" in out


@pytest.mark.parametrize(
    "field", ["rollout1_stop_reason", "rollout1_n_asks", "first_live_call_executed"]
)
def test_a_gated_run_without_a_rollout1_or_first_call_field_is_refused(tmp_path, capsys, field):
    recs = healthy()
    del recs[4][field]
    refused(*run(tmp_path, recs, capsys=capsys), field)


def test_a_first_call_flag_that_contradicts_its_spend_is_refused(tmp_path, capsys):
    """first_live_call_executed True carries a spend; False carries null. Either way round refuses."""
    recs = healthy()
    recs[4]["spent_before_first_live_call"] = None
    refused(*run(tmp_path, recs, capsys=capsys), "first_live_call_executed")
    recs = _stopping_population()
    recs[0]["spent_before_first_live_call"] = 16.0  # a no-live-call comparator: flag False
    refused(*run(tmp_path, recs, capsys=capsys), "first_live_call_executed")


def test_spend_before_the_first_call_above_the_total_spend_is_refused(tmp_path, capsys):
    recs = healthy()
    recs[4]["spent_before_first_live_call"] = recs[4]["spent"]["retrieval_calls"] + 1
    refused(*run(tmp_path, recs, capsys=capsys), "exceeds")


def test_starved_is_exact_where_the_old_derivation_was_wrong(tmp_path, capsys):
    """A comparator unit that asks 5 times, executes a GENERIC `calculate` live (uncharged), then
    asks 11 more: spent 16, gate 0. The old (spent - gate) >= cap derivation calls it starved;
    the exact rule does not, because a live call executed with only 5 spent."""
    recs = healthy()
    r = next(
        x
        for x in recs
        if x["arm_id"] == C
        and x["suite_id"] == "tau2_retail"
        and x["key"][9] == "tau2_base"
        and x["task_id"] == "t1"
    )
    r.update(
        n_asks=16,
        n_env_calls=PREFIX + 1,
        _env_calls_from_outcome=PREFIX + 1,
        _env_tool_counts={"get_user_details": PREFIX, "calculate": 1},
        _env_requestor_counts={"assistant": PREFIX + 1},
        _env_call_seq=[["get_user_details", True]] * PREFIX + [["calculate", True]],
        n_live_env_calls=1,
        n_live_env_calls_generic=1,
        n_live_env_calls_nongeneric=0,
        n_live_env_calls_nongeneric_ok=0,
        n_gate_charged=0,
        first_live_call_executed=True,
        spent_before_first_live_call=5.0,
    )
    r["spent"]["retrieval_calls"] = 16.0
    code, out = run(tmp_path, recs, capsys=capsys)
    assert code == 0, out
    rows = {
        (x["suite"], x["variant"], x["arm"]): x
        for x in json.loads((tmp_path / "out.json").read_text())["stopping"]
    }
    comp = rows[("tau2_retail", "tau2_base", "comparator")]
    assert (comp["n_asks_capped"], comp["n_asks_capped_defined"]) == (0, 2)


def test_the_primary_unit_set_is_identical_with_and_without_the_secondaries(
    tmp_path, capsys, monkeypatch
):
    """No code path may filter units by starvation or any other post-treatment variable: the
    primary's pairs, run ids and task deltas are the same when the secondaries are not computed."""
    recs = _with_first_call_spend(_stopping_population())
    code, _ = run(tmp_path, recs, capsys=capsys)
    assert code == 0
    full = json.loads((tmp_path / "out.json").read_text())

    monkeypatch.setattr(tr, "stopping_block", lambda *a, **k: ([], []))
    monkeypatch.setattr(tr, "_secondary", lambda *a, **k: {})
    code, _ = run(tmp_path, recs, capsys=capsys)
    assert code == 0
    bare = json.loads((tmp_path / "out.json").read_text())

    def primary(rep):
        return [
            (
                c["suite"],
                c["variant"],
                c["seed_label"],
                c["n_pairs"],
                c["run_ids"],
                c["primary"]["task_level"]["task_deltas"],
                c["guard"]["task_level"]["task_deltas"],
            )
            for c in rep["cells"]
        ] + [
            (p["suite"], p["variant"], p["task_level"]["task_deltas"], p["guard"]["task_level"])
            for p in rep["pooled"]
        ]

    assert primary(full) == primary(bare)
    assert bare["stopping"] == [] and full["stopping"]


AID_TAIL = "(descriptive; amendment 4's shared-budget wording rule is superseded by amendment 7's separate budgets)"


def test_the_wording_aid_is_descriptive_and_always_printed():
    """CHANGED after 8cdf0a1 (the coordinator's label fix): amendment 4's wording rule was about the
    comparator exhausting the SHARED budget on questions; amendment 7 removed the shared budget, so
    the aid is descriptive, says the rule is superseded, and prints at every share, not only > 0.5."""
    rows = [
        {
            "suite": "s",
            "variant": "v",
            "arm": "comparator",
            "n_asks_capped": 2,
            "n_asks_capped_defined": 2,
        },
        {"suite": "s", "variant": "v", "arm": "s1", "n_asks_capped": 1, "n_asks_capped_defined": 2},
        {"suite": "s", "variant": "v", "arm": "s2", "n_asks_capped": 0, "n_asks_capped_defined": 2},
    ]
    assert tr.wording_note(rows, "s", "v", ["s1"]) == (
        "asks reached the ask cap before the first tool call: comparator 2/2, questioner 1/2 "
        + AID_TAIL
    )
    assert "questioner 1/4" in tr.wording_note(rows, "s", "v", ["s1", "s2"])
    calm = [dict(r, n_asks_capped=0) for r in rows]
    assert tr.wording_note(calm, "s", "v", ["s1"]) == (
        "asks reached the ask cap before the first tool call: comparator 0/2, questioner 0/2 "
        + AID_TAIL
    )
    for rows_ in (rows, calm):
        assert "wording rule applies" not in tr.wording_note(rows_, "s", "v", ["s1"])


QUOTED = "defined on all units (censored at total spend where no live call executed)"
DESCR = "conditional on both arms acting — descriptive, not quoted"


def test_the_quoted_calls_difference_keeps_a_pair_where_the_comparator_never_acted(
    tmp_path, capsys
):
    """Amendment 6: restricting the paired difference to pairs where BOTH arms acted conditions on a
    post-treatment variable. retail/base/s1: the comparator never acts at t1 (10 asks, no live
    call); s1 acts at both tasks with 3 spent. Quoted, censored at total spend: t1 3 - 10 = -7,
    t2 3 - 3 = 0 -> mean -3.5 over BOTH pairs. Descriptive, both acting: t2 only, delta 0."""
    recs = healthy()
    comp_t1 = next(
        r
        for r in recs
        if r["arm_id"] == C
        and r["suite_id"] == "tau2_retail"
        and r["key"][9] == "tau2_base"
        and r["task_id"] == "t1"
    )
    no_live_call(comp_t1)
    code, out = run(tmp_path, recs, "--printed-interval", "10000:0", capsys=capsys)
    assert code == 0, out
    rep = json.loads((tmp_path / "out.json").read_text())
    s1 = next(
        c
        for c in rep["cells"]
        if c["suite"] == "tau2_retail" and c["variant"] == "tau2_base" and c["seed_label"] == "s1"
    )
    calls = s1["secondary"][A4[3]]
    assert calls["task_pair_deltas"] == {"t1": [-7.0], "t2": [0.0]}
    assert calls["mean_delta"] == pytest.approx(-3.5)
    assert (calls["n_pairs"], calls["n_defined_pairs"]) == (2, 2)
    assert calls["printed_interval"]["n_boot"] == 10_000
    d = calls["descriptive_conditional_on_both_acting"]
    assert d["n_pairs"] == 1 and d["task_deltas"] == {"t2": 0.0} and d["mean_delta"] == 0.0
    assert "printed_interval" not in d and "intervals" not in d, "descriptive: no interval"
    lines = [ln for ln in out.splitlines() if f"[{A4[3]}]" in ln]
    assert any(QUOTED in ln and "paired delta=-3.5000" in ln for ln in lines), lines
    assert any(DESCR in ln and "1 pair" in ln for ln in lines), lines
    pooled = next(
        p for p in rep["pooled"] if p["suite"] == "tau2_retail" and p["variant"] == "tau2_base"
    )
    # pair_mean over ALL pairs: t1 (s1 -7, s2 3 - 10 = -7) -> -7; t2 (0, 0) -> 0
    assert pooled["secondary"][A4[3]]["task_deltas"] == {"t1": -7.0, "t2": 0.0}
    assert pooled["secondary"][A4[3]]["n_defined_pairs"] == {"s1": 2, "s2": 2}


def test_a_partially_defined_secondary_pools_by_defined_pairs_and_says_so():
    """The paper lane's rule, kept for any secondary defined on only some units: pair_mean weighs
    each seed by its DEFINED pairs, and the form and the per-seed counts are printed and recorded.
    s1 is defined on 1 of 2 pairs (t2: 0), s2 on 2 of 2 (t1: 0, t2: 2): t1 = 0, t2 = (0 + 2)/2."""

    def entry(deltas, n_pairs):
        return {
            "n_pairs": n_pairs,
            "n_defined_pairs": sum(len(v) for v in deltas.values()),
            "defined_on_all_units": sum(len(v) for v in deltas.values()) == n_pairs,
            "task_pair_deltas": deltas,
        }

    cells = [
        {"seed_label": "s1", "secondary": {"x": entry({"t2": [0.0]}, 2)}},
        {"seed_label": "s2", "secondary": {"x": entry({"t1": [0.0], "t2": [2.0]}, 2)}},
    ]
    e = tr._pooled_secondary(cells, "x", "pair_mean", (200, 0))
    assert e["pooling"] == "pair_mean"
    assert e["n_defined_pairs"] == {"s1": 1, "s2": 2} and e["n_pairs"] == {"s1": 2, "s2": 2}
    assert e["defined_on_all_units"] is False
    assert e["task_deltas"] == {"t1": 0.0, "t2": 1.0}
    assert tr.pooled_secondary_form(e) == "pair_mean over defined pairs: s1 1 of 2, s2 2 of 2"
    full = {**e, "defined_on_all_units": True}
    assert tr.pooled_secondary_form(full) == "pair_mean, defined on all units"


# ------------------------------------------------------------------ amendment 7: the primary and its guard

A7_HIT = "gold-object hit rate per ask"
A7_RECALL = "gold-object recall"
A7_ASKS = "asks per unit"


def _fu_records() -> list:
    """Follow-up turns after the fork point, known answer. retail/base, run seed 0.

    slot            C   s1  s2
    A a1 4          5   3   4
    A a2 6          4   4   2
    B b1 4          2   6   2

    s1 deltas: A {-2, 0} -> -1; B {+4} -> +4. up 1, down 1; mean +1.5
    s2 deltas: A {-1, -2} -> -1.5; B {0} -> 0. up 0, down 1, tied 1; mean -0.75
    seeds-pooled pair_mean: A (-2 + 0 - 1 - 2)/4 = -1.25; B (4 + 0)/2 = +2 -> mean +0.375
    """
    rows = [("A", "a1", 4, 5, 3, 4), ("A", "a2", 6, 4, 4, 2), ("B", "b1", 4, 2, 6, 2)]
    out = []
    for task, trace, k, c, s1, s2 in rows:
        kw = dict(task=task, trace=trace, k=k)
        out += [
            rec(C, BASE, followups=c, **kw),
            rec(T, S1, followups=s1, **kw),
            rec(T, S2, followups=s2, **kw),
        ]
    return out


def _one_cell(rep, suite="tau2_retail", variant="tau2_base", seed="s1"):
    return next(
        c
        for c in rep["cells"]
        if c["suite"] == suite and c["variant"] == variant and c["seed_label"] == seed
    )


def test_the_primary_is_follow_up_turns_after_the_fork_point(tmp_path, capsys):
    code, out = run(
        tmp_path, _fu_records(), "--suites", "tau2_retail", "--variants", "tau2_base", capsys=capsys
    )
    assert code == 0, out
    rep = json.loads((tmp_path / "out.json").read_text())
    t = _one_cell(rep)["primary"]["task_level"]
    assert t["task_deltas"] == {"A": -1.0, "B": 4.0}
    assert (t["up"], t["down"], t["tied"]) == (1, 1, 0) and t["mean_delta"] == pytest.approx(1.5)
    t2 = _one_cell(rep, seed="s2")["primary"]["task_level"]
    assert t2["task_deltas"] == {"A": -1.5, "B": 0.0}
    assert t2["mean_delta"] == pytest.approx(-0.75)
    pooled = rep["pooled"][0]["task_level"]
    assert pooled["task_deltas"] == {"A": -1.25, "B": 2.0}
    assert pooled["mean_delta"] == pytest.approx(0.375)
    assert rep["rules"]["primary"].startswith("follow-up user turns after the fork point")
    assert rep["rules"]["co_primary_guard"].startswith("task success")
    assert "PRIMARY follow-up user turns after the fork point" in out
    assert "CO-PRIMARY GUARD task success" in out


def _headline_population(success_drop: bool = False) -> list:
    """Six retail/base tasks, one fork each: both trained seeds need 2 fewer follow-ups than the
    comparator everywhere; success is equal (1 and 1) unless `success_drop`, where it is 0 vs 1."""
    out = []
    for i, task in enumerate(("A", "B", "C", "D", "E", "F")):
        kw = dict(task=task, trace=f"x{i}", k=4)
        out += [
            rec(C, BASE, followups=5, reward=1.0, **kw),
            rec(T, S1, followups=3, reward=0.0 if success_drop else 1.0, **kw),
            rec(T, S2, followups=3, reward=0.0 if success_drop else 1.0, **kw),
        ]
    return out


def test_the_headline_is_yes_when_follow_ups_decide_negative_and_success_is_non_inferior(
    tmp_path, capsys
):
    code, out = run(
        tmp_path,
        _headline_population(),
        "--suites",
        "tau2_retail",
        "--variants",
        "tau2_base",
        capsys=capsys,
    )
    assert code == 0, out
    rep = json.loads((tmp_path / "out.json").read_text())
    h = _one_cell(rep)["headline"]
    assert h["text"] == "reduces the human in the loop: YES"
    assert h["follow_up_decided_negative"] is True and h["success_ni_passes"] is True
    assert h["success_ni_margin"] == -0.10
    assert rep["pooled"][0]["headline"]["text"] == "reduces the human in the loop: YES"
    assert "reduces the human in the loop: YES" in out
    # the success interval is always printed, with the NI verdict
    assert any("success NI guard at -0.10" in ln and "PASS" in ln for ln in out.splitlines())


def test_the_headline_names_the_condition_that_failed(tmp_path, capsys):
    # success lost on every task: the NI guard fails
    code, out = run(
        tmp_path,
        _headline_population(success_drop=True),
        "--suites",
        "tau2_retail",
        "--variants",
        "tau2_base",
        capsys=capsys,
    )
    assert code == 0, out
    h = _one_cell(json.loads((tmp_path / "out.json").read_text()))["headline"]
    assert h["text"].startswith("reduces the human in the loop: NO")
    assert "success non-inferiority fails" in h["text"] and h["follow_up_decided_negative"] is True
    # follow-ups tied everywhere: not decided negative
    code, out = run(tmp_path, healthy(), capsys=capsys)
    assert code == 0, out
    h = _one_cell(json.loads((tmp_path / "out.json").read_text()))["headline"]
    assert "follow-up delta not DECIDED negative" in h["text"]
    # healthy(): s1 succeeds where the comparator fails (+1 at both tasks), so the NI guard passes
    assert h["success_ni_passes"] is True
    assert "success non-inferiority" not in h["text"]


def test_the_success_ni_margin_is_a_named_constant():
    assert tr.SUCCESS_NI_MARGIN == -0.10


def test_hand_offs_are_printed_per_arm_beside_the_primary(tmp_path, capsys):
    recs = healthy()
    comp = next(
        r
        for r in recs
        if r["arm_id"] == C
        and r["suite_id"] == "tau2_retail"
        and r["key"][9] == "tau2_base"
        and r["task_id"] == "t1"
    )
    comp["n_transfer_to_human"] = 1
    code, out = run(tmp_path, recs, capsys=capsys)
    assert code == 0, out
    ho = _one_cell(json.loads((tmp_path / "out.json").read_text()))["handoffs"]
    assert ho == {"trained": {"total": 0, "n_units": 2}, "control": {"total": 1, "n_units": 2}}
    assert (
        "hand-offs (n_transfer_to_human): trained 0 over 2 units, comparator 1 over 2 units" in out
    )


def test_absent_hand_offs_refuse_a_gated_run_and_read_not_recorded_on_legacy(tmp_path, capsys):
    """CHANGED after 9e2c761: n_transfer_to_human is part of the refuse_and_tell_split status
    (lane A 4e09a07), so on a gated run its absence refuses; NOT RECORDED is for legacy only."""
    recs = healthy()
    for r in recs:
        del r["n_transfer_to_human"]
    refused(*run(tmp_path, recs, capsys=capsys), "n_transfer_to_human")

    code = tr.main(_legacy_argv(tmp_path, _legacy_records()) + ["--validation-legacy-population"])
    out = capsys.readouterr().out
    assert code == 0, out
    assert "hand-offs (n_transfer_to_human): NOT RECORDED" in out


# ------------------------------------------------------------------ amendment 7: separate budgets


def test_the_shared_budget_regime_is_refused_now(tmp_path, capsys):
    recs = healthy()
    recs[4]["budget_enforcement"] = "refuse_and_tell"
    refused(*run(tmp_path, recs, capsys=capsys), "budget_enforcement")


def test_a_run_without_its_tool_call_cap_is_refused(tmp_path, capsys):
    recs = healthy()
    del recs[4]["tool_call_cap"]
    refused(*run(tmp_path, recs, capsys=capsys), "tool_call_cap")


def test_two_tool_call_caps_are_refused(tmp_path, capsys):
    recs = healthy()
    recs[4]["tool_call_cap"] = 8
    refused(*run(tmp_path, recs, capsys=capsys), "tool_call_cap values")


def test_a_gate_charge_above_the_tool_call_cap_is_refused(tmp_path, capsys):
    recs = healthy()
    for r in recs:
        r["tool_call_cap"] = 1  # every healthy unit charged 2 live calls
    refused(*run(tmp_path, recs, capsys=capsys), "exceeds tool_call_cap")


def test_a_refused_tool_call_under_the_tool_cap_is_a_leak(tmp_path, capsys):
    """Under separate budgets the gate refuses only when the TOOL budget is spent; a refusal at 2 of
    16 means the asks' budget reached the tool calls."""
    recs = healthy()
    recs[4]["n_refused_budget"] = 1
    recs[4]["refused_budget_calls"] = [{"tool_name": "get_order_details", "mutating": False}]
    refused(*run(tmp_path, recs, capsys=capsys), "leak")


def test_asks_at_their_cap_with_no_tool_call_read_under_separate_budgets(tmp_path, capsys):
    """CHANGED after 9e2c761 (the coordinator's correction, CLAUDE.md rule 4): the starvation
    refusal is REMOVED. Under separate budgets the asks reaching their own cap is legitimate, with
    or without a tool call after it. 16 asks and no tool call READS, and is counted."""
    recs = healthy()
    starving(recs[0])  # comparator retail/base/t1: 16 asks, no live tool call
    code, out = run(tmp_path, recs, capsys=capsys)
    assert code == 0, out
    rows = {
        (x["suite"], x["variant"], x["arm"]): x
        for x in json.loads((tmp_path / "out.json").read_text())["stopping"]
    }
    comp = rows[("tau2_retail", "tau2_base", "comparator")]
    assert (comp["n_asks_capped"], comp["n_asks_capped_defined"]) == (1, 2)


def test_asks_at_their_cap_then_an_executed_tool_call_read(tmp_path, capsys):
    """16 asks, THEN an executed tool call: under separate budgets that is the split holding. It
    READS, it is counted by the descriptive measure, and the leak check reads 0."""
    recs = healthy()
    recs[4] = rec(T, S1, asks=16, task="t2", trace="bbb")  # spent 16, before-first 16, a call ran
    code, out = run(tmp_path, recs, capsys=capsys)
    assert code == 0, out
    rows = {
        (x["suite"], x["variant"], x["arm"]): x
        for x in json.loads((tmp_path / "out.json").read_text())["stopping"]
    }
    s1 = rows[("tau2_retail", "tau2_base", "s1")]
    assert (s1["n_asks_capped"], s1["leak_refused_below_tool_cap"]) == (1, 0)
    sec = _one_cell(json.loads((tmp_path / "out.json").read_text()))["secondary"]
    assert sec["asks reached the ask cap before the first tool call"]["task_pair_deltas"] == {
        "t1": [0.0],
        "t2": [1.0],
    }


def test_retrieval_k_other_than_5_is_refused(tmp_path, capsys):
    recs = healthy()
    recs[0]["_manifest"]["k"] = 3
    refused(*run(tmp_path, recs, capsys=capsys), "retrieval k")


# ------------------------------------------------------------------ amendment 7: gold-object hit rate and recall


def _hit_population() -> list:
    """retail/base; every task's gold object is users/u1 (GOLD_ACTIONS). G = gold uid, - = miss.

    unit                 asks (retrieved per ask)
    t1 comparator        [G], [-], [-]        hit 1/3
    t1 s1                [G], [G], [G]        hit 3/3
    t2 comparator        (asked nothing)      hit UNDEFINED; recall 0
    t2 s1                [-], [-], [-]        hit 0/3

    s1 hit-rate pairs: t1 1 - 1/3 = +2/3; t2 DROPPED (comparator asked nothing): 1 of 2 defined
    s1 recall pairs:   t1 1 - 1 = 0; t2 0 - 0 = 0
    s1 asks per unit:  t1 3 - 3 = 0; t2 3 - 0 = +3
    shares of asks hit: s1 3/6 = 0.5, comparator 1/3
    """
    g = gold_uid("tau2_retail")
    recs = healthy()
    for r in recs:
        if r["suite_id"] != "tau2_retail" or r["key"][9] != "tau2_base":
            continue
        model = r["_manifest"]["pins"]["inquirer"]["model_id"]
        if model == BASE and r["task_id"] == "t1":
            r["_ask_retrieved_uids"] = [[g], [], []]
        elif model == BASE and r["task_id"] == "t2":
            r.update(n_asks=0, rollout1_n_asks=0, spent_before_first_live_call=0.0)
            r["spent"]["retrieval_calls"] = 0.0
            r["_ask_retrieved_uids"] = []
        elif model == S1 and r["task_id"] == "t1":
            r["_ask_retrieved_uids"] = [[g], [g, "other"], [g]]
    return recs


def test_gold_object_hit_rate_and_recall_known_answer(tmp_path, capsys):
    code, out = run(tmp_path, _hit_population(), capsys=capsys)
    assert code == 0, out
    rep = json.loads((tmp_path / "out.json").read_text())
    sec = _one_cell(rep)["secondary"]
    hit = sec[A7_HIT]
    assert (hit["n_pairs"], hit["n_defined_pairs"]) == (2, 1)
    assert hit["task_pair_deltas"] == {"t1": [pytest.approx(2 / 3)]}
    assert hit["dropped_units"] == {"trained_asked_nothing": 0, "comparator_asked_nothing": 1}
    assert hit["dropped_pairs_no_gold_object"] == 0
    assert hit["trained_share_of_asks_hit"] == pytest.approx(0.5)
    assert hit["control_share_of_asks_hit"] == pytest.approx(1 / 3)
    assert sec[A7_RECALL]["task_pair_deltas"] == {"t1": [0.0], "t2": [0.0]}
    assert sec[A7_RECALL]["defined_on_all_units"] is True
    assert sec[A7_ASKS]["task_pair_deltas"] == {"t1": [0.0], "t2": [3.0]}
    assert "comparator asked nothing: 1 unit(s) dropped" in out
    assert "read together, never alone" in out


def test_a_task_with_no_pre_dialogue_gold_object_has_no_hit_rate_or_recall(tmp_path, capsys):
    recs = healthy()
    code, out = run(tmp_path, recs, gold={("retail", "t2"): []}, capsys=capsys)
    assert code == 0, out
    sec = _one_cell(json.loads((tmp_path / "out.json").read_text()))["secondary"]
    assert sec[A7_HIT]["dropped_pairs_no_gold_object"] == 1
    assert (sec[A7_RECALL]["n_pairs"], sec[A7_RECALL]["n_defined_pairs"]) == (2, 1)
    # t2 has no gold object: its asks (3 per arm) are outside the hit share, and counted
    hit = sec[A7_HIT]
    assert (
        hit["trained_asks_in_tasks_without_gold"],
        hit["control_asks_in_tasks_without_gold"],
    ) == (
        3,
        3,
    )
    assert "asks in tasks without gold objects: trained 3, comparator 3" in out


def test_the_deleted_metric_name_never_appears(tmp_path, capsys):
    """Amendment 7's sign-off: never 'precision' (CLAUDE.md lists Proactive Precision as deleted).
    Named without the word: pytest's tmp path carries the test name into the output."""
    code, out = run(tmp_path, _hit_population(), capsys=capsys)
    assert code == 0
    assert "precision" not in out.lower()
    assert "precision" not in (tmp_path / "out.json").read_text().lower()


def test_a_gated_run_without_its_per_ask_uids_is_refused(tmp_path, capsys):
    recs = healthy()
    del recs[4]["_ask_retrieved_uids"]
    refused(*run(tmp_path, recs, capsys=capsys), "_ask_retrieved_uids")


def test_a_task_absent_from_the_tau2_tasks_is_refused(tmp_path, capsys):
    recs = healthy()
    for r in recs:
        if r["task_id"] == "t2":
            r["task_id"] = "zz"
            r["key"][1] = "zz"
    refused(*run(tmp_path, recs, capsys=capsys), "tasks.json")


def test_a_gated_reading_without_tau2_data_is_refused(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("TAU2_DATA_DIR", raising=False)
    recs = healthy()
    f = tmp_path / "records.json"
    f.write_text(json.dumps(recs))
    argv = ["--records", str(f), *FAST_BOOT, "--serve-record", str(serve_record(tmp_path))]
    argv += ["--probe-record", str(probe_record(tmp_path))]
    for lp in launch_records(tmp_path, recs):
        argv += ["--launch-record", str(lp)]
    refused(tr.main(argv), capsys.readouterr().out, "--tau2-data-dir")


def test_a_record_a_gold_write_creates_is_not_a_gold_object(tmp_path):
    """The index is the PRE-dialogue db.json. A WRITE that modifies an existing record counts (the
    record existed before the dialogue); an id no pre-dialogue table holds -- as a created record's
    would be -- resolves to nothing."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "tau2_concordance"))
    import gold_objects

    root = tau2_data(
        tmp_path,
        gold={
            ("retail", "t1"): [
                {"name": "cancel_pending_order", "arguments": {"order_id": "#W1", "reason": "x"}},
                {"name": "get_order_details", "arguments": {"order_id": "#W999"}},
                {"name": "get_user_details", "arguments": {"user_id": "u1"}},
            ],
            ("airline", "t1"): [
                {"name": "book_reservation", "arguments": {"user_id": "u1", "origin": "SFO"}},
                {"name": "cancel_reservation", "arguments": {"reservation_id": "NEWRES"}},
            ],
        },
    )
    g = gold_objects.GoldObjects(root)
    assert g.uids("tau2_retail", "t1") == {
        gold_uid("tau2_retail", "orders", "#W1"),
        gold_uid("tau2_retail"),
    }
    assert g.summary("tau2_retail", "t1") == {
        "n_gold_actions": 3,
        "n_gold_write_actions": 1,
        "n_gold_objects": 2,
    }
    assert g.uids("tau2_airline", "t1") == {gold_uid("tau2_airline")}


def test_the_gold_object_loader_is_imported_by_the_reader_and_by_no_rollout_module():
    """Amendment 7's firewall: the reader computes gold objects at READING time. Nothing under src/
    names the loader, and importing the rollout worker and the tau2 runner does not load it."""
    import re
    import subprocess

    root = Path(__file__).resolve().parents[1]
    reader = (root / "scripts" / "tau2_concordance" / "transfer_rerun.py").read_text()
    assert re.search(r"^(import gold_objects|from gold_objects import)", reader, flags=re.M)
    hits = [str(q) for q in (root / "src").rglob("*.py") if "gold_objects" in q.read_text()]
    assert hits == [], hits
    probe = (
        "import sys; import pi_run.worker, pi_run.stages.tau2_runner; "
        "print(sorted(m for m in sys.modules if 'gold_objects' in m or 'tau2_concordance' in m))"
    )
    r = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        env={**__import__("os").environ, "PYTHONPATH": str(root / "src")},
    )
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "[]", r.stdout


def test_extract_embeds_the_per_ask_retrieved_uids(tmp_path):
    _write_run(
        tmp_path,
        "r1",
        {"run_id": "r1", "status": "ok"},
        manifest(BASE),
        [],
        asks=[["a", "b"], []],
    )
    (tmp_path / "r1" / "turns.jsonl").write_text(
        (tmp_path / "r1" / "turns.jsonl").read_text()
        + json.dumps({"action_kind": "answer", "retrieved_uids": ["x"]})
        + "\n"
    )
    (r,) = extract_records.extract(tmp_path)
    assert r["_ask_retrieved_uids"] == [["a", "b"], []]


def test_a_gated_reading_never_says_wording_rule_applies_or_starved(tmp_path, capsys):
    """The label fix after 8cdf0a1. Every comparator unit's asks reach the cap before any tool call
    (16 asks, no tool call) -- the share the old aid fired on. On a gated reading no output line
    and no JSON key or value says 'wording rule applies' or 'starv', and 'shared' appears only in
    the aid's statement that the shared-budget rule is superseded."""
    recs = healthy()
    for r in recs:
        if r["arm_id"] == C:
            starving(r)
    code, out = run(tmp_path, recs, capsys=capsys)
    assert code == 0, out
    js = (tmp_path / "out.json").read_text().lower()
    low = out.lower()
    assert "wording rule applies" not in low and "wording rule applies" not in js
    assert "starv" not in low and "starv" not in js
    shared = [ln for ln in out.splitlines() if "shared" in ln.lower()]
    assert shared and all(AID_TAIL in ln for ln in shared), shared
    prim = [
        ln
        for ln in out.splitlines()
        if "PRIMARY follow-up user turns after the fork point -- task level" in ln
    ]
    assert prim and all(
        "asks reached the ask cap before the first tool call: comparator 2/2, questioner 0/2" in ln
        for ln in prim
    ), prim


# ------------------------------------------------------------------ amendment 9: where the tool cap bound

A9 = "units with a refused tool call (n_refused_budget > 0)"


def test_the_tool_cap_binding_share_is_printed_beside_the_success_guard(tmp_path, capsys):
    """Amendment 9: a success gap must be readable against where the 16-call tool cap bound.
    retail/base/s1 at t1 reaches the tool cap (16 live charged calls) and has 1 call refused --
    legitimate, the cap was reached, so not a leak. Per arm: trained 1/2 units, comparator 0/2.
    Paired task deltas t1 +1, t2 0 (mean +0.5); seeds-pooled pair_mean: t1 (1 + 0)/2, t2 0."""
    recs = healthy()
    capped = rec(T, S1, calls=19)  # prefix 2 + 16 charged + 1 calculate: n_gate_charged 16
    capped["n_refused_budget"] = 1
    capped["refused_budget_calls"] = [{"tool_name": "get_order_details", "mutating": False}]
    assert recs[1]["run_id"] == capped["run_id"]
    recs[1] = capped
    code, out = run(tmp_path, recs, capsys=capsys)
    assert code == 0, out
    rep = json.loads((tmp_path / "out.json").read_text())
    tc = _one_cell(rep)["guard"]["refused_tool_call_units"]
    assert (tc["trained_units_refused"], tc["control_units_refused"], tc["n_pairs"]) == (1, 0, 2)
    assert tc["task_pair_deltas"] == {"t1": [1.0], "t2": [0.0]}
    assert tc["mean_delta"] == pytest.approx(0.5)
    assert tc["printed_interval"]["seed"] == 0
    pooled = next(
        p for p in rep["pooled"] if p["suite"] == "tau2_retail" and p["variant"] == "tau2_base"
    )
    pt = pooled["guard"]["refused_tool_call_units"]
    assert pt["task_deltas"] == {"t1": 0.5, "t2": 0.0}
    # per arm, pooled by the same rule as the delta: trained t1 (1+0)/2, t2 0 -> 0.25; comparator 0
    assert (pt["trained_level"], pt["control_level"]) == (pytest.approx(0.25), 0.0)
    assert pt["trained_level"] - pt["control_level"] == pytest.approx(pt["mean_delta"])
    assert pt["trained_units_refused"] == {"s1": 1, "s2": 0}
    assert pt["control_units_refused"] == {"s1": 0, "s2": 0}
    lines = out.splitlines()
    i = next(k for k, ln in enumerate(lines) if "--- tau2_retail / tau2_base / s1 " in ln)
    ni = next(k for k in range(i, len(lines)) if "success NI guard" in lines[k])
    assert f"[{A9}] trained 1/2 units, comparator 0/2 units" in lines[ni + 1], lines[ni + 1]
    assert "descriptive" in lines[ni + 1]
    j = next(k for k, ln in enumerate(lines) if "--- tau2_retail / tau2_base / seeds-pooled" in ln)
    pni = next(k for k in range(j, len(lines)) if "success NI guard" in lines[k])
    want = "trained 0.2500 (s1 1/2, s2 0/2 units), comparator 0.0000 (s1 0/2, s2 0/2 units)"
    assert f"[{A9}] seeds-pooled" in lines[pni + 1] and want in lines[pni + 1], lines[pni + 1]
    assert "paired delta=+0.2500" in lines[pni + 1] and "descriptive" in lines[pni + 1]


def test_the_tool_cap_share_is_not_recorded_rather_than_zero_without_the_field(tmp_path, capsys):
    """A run that does not record n_refused_budget has no tool-cap share: NOT RECORDED, never 0/N.
    Exercised on the legacy validation population, whose records predate the split budget."""
    recs = _legacy_records()
    assert all(tr.N_REFUSED_BUDGET_FIELD not in r for r in recs), "the fixture must lack the field"
    code = tr.main(_legacy_argv(tmp_path, recs) + ["--validation-legacy-population"])
    out = capsys.readouterr().out
    assert code == 0, out
    rep = json.loads((tmp_path / "legacy_out.json").read_text())
    assert all(c["guard"]["refused_tool_call_units"] is None for c in rep["cells"])
    assert f"[{A9}] NOT RECORDED -- not zero" in out


# ------------------------------------------------------------------ amendment 10: several proxy configs

LOCAL = {"enable_thinking": False}


def tracked_config() -> dict:
    """`conf/serving/litellm.tau2.yaml` at ee9afda2, with its vLLM port rewritten as the wrapper does:
    four hosted_vllm routes and two gateway patterns with two keys each."""
    local = [
        {
            "model_name": m,
            "litellm_params": {
                "model": f"hosted_vllm/{m}",
                "api_base": "http://127.0.0.1:8412/v1",
                "api_key": "unused-by-vllm",
                "extra_body": {"chat_template_kwargs": dict(LOCAL)},
            },
        }
        for m in (BASE, "qwen3-8b-dpo-stacked-notdone-both", S1, S2)
    ]
    gateway = [
        {
            "model_name": pattern,
            "litellm_params": {
                "model": "openai/*",
                "api_base": "os.environ/PINQ_GATEWAY_BASE_URL",
                "api_key": f"os.environ/{key}",
            },
        }
        for pattern in ("openai/*", "*")
        for key in ("LITELLM_API_KEY", "PI_AGENT_LITELLM_API_KEY")
    ]
    return {
        "model_list": local + gateway,
        "litellm_settings": {"num_retries": 0, "request_timeout": 600},
        "general_settings": {"disable_spend_logs": True},
    }


def generated_config() -> dict:
    """Links 2+: the tracked file plus, per gateway pattern, deployments identical but for
    api_key: os.environ/LITELLM_API_KEY_PARALLEL_{1,2,3}."""
    cfg = tracked_config()
    extra = []
    for pattern in ("openai/*", "*"):
        (first,) = [
            e
            for e in cfg["model_list"]
            if e["model_name"] == pattern
            and e["litellm_params"]["api_key"] == "os.environ/LITELLM_API_KEY"
        ]
        for i in (1, 2, 3):
            e = copy.deepcopy(first)
            e["litellm_params"]["api_key"] = f"os.environ/LITELLM_API_KEY_PARALLEL_{i}"
            extra.append(e)
    cfg["model_list"] += extra
    return cfg


def config_job(tmp_path, job, cfg, archive=True, models=(BASE, S1, S2)) -> Path:
    """One LSF job: its proxy config archived NEXT TO its probe record, under the basename of the
    probe's config_path, and the probe's config_sha256 computed as thinking_probe.py computes it."""
    text = yaml.safe_dump(cfg, sort_keys=False)
    name = f"litellm.tau2.{job}.yaml"
    d = tmp_path / "_jobs" / str(job)
    d.mkdir(parents=True, exist_ok=True)
    if archive:
        (d / name).write_text(text)
    return probe_record(
        tmp_path,
        models=models,
        name=f"_jobs/{job}/thinking_probe.json",
        config_sha=hashlib.sha256(text.encode()).hexdigest(),
        config_path=f"<job-tmpdir>/{name}",
    )


def _check_line(out: str, letter: str) -> str:
    (line,) = [ln for ln in out.splitlines() if ln.strip().startswith(f"({letter})")]
    return line


def test_two_configs_differing_only_in_gateway_api_keys_are_read(tmp_path, capsys):
    """Amendment 10: link 1 on the tracked config, links 2+ on the generated one. Two shas, one
    population. The shas, the job count on each and every check's verdict are printed."""
    recs = healthy()
    j1 = config_job(tmp_path, 911001, tracked_config())
    j2 = config_job(tmp_path, 912001, generated_config())
    j3 = config_job(tmp_path, 912002, generated_config())
    code = tr.main(_argv(tmp_path, recs, [j1, j2, j3], [serve_record(tmp_path)]))
    out = capsys.readouterr().out
    assert code == 0, out
    shas = {json.loads(j.read_text())["config_sha256"] for j in (j1, j2, j3)}
    assert len(shas) == 2, "non-vacuity: the two configs must actually differ"
    rep = json.loads((tmp_path / "out.json").read_text())
    ag = rep["provenance"]["config_agreement"]
    assert ag["jobs_per_config_sha256"] == {
        json.loads(j1.read_text())["config_sha256"]: 1,
        json.loads(j2.read_text())["config_sha256"]: 2,
    }
    assert {k: v["verdict"] for k, v in ag["checks"].items()} == {
        "a": "PASS",
        "b": "PASS",
        "c": "PASS",
        "d": "PASS",
    }
    assert ag["checks"]["c"]["n_entries_differing"] == 6
    for letter in "abcd":
        assert "PASS" in _check_line(out, letter), out
    for sha, n in ag["jobs_per_config_sha256"].items():
        assert f"config_sha256 {sha[:16]}: {n} job" in out, out


def _mut_thinking(cfg):
    cfg["model_list"][0]["litellm_params"]["extra_body"]["chat_template_kwargs"] = {
        "enable_thinking": True
    }


def _mut_local_api_base(cfg):
    cfg["model_list"][2]["litellm_params"]["api_base"] = "http://127.0.0.1:8413/v1"


def _mut_literal_gateway_key(cfg):
    cfg["model_list"][-1]["litellm_params"]["api_key"] = "sk-literal-not-an-env-name"


def _mut_gateway_api_base(cfg):
    cfg["model_list"][-1]["litellm_params"]["api_base"] = "https://elsewhere.example/v1"


def _mut_settings(cfg):
    cfg["litellm_settings"]["num_retries"] = 2


def _mut_router_section(cfg):
    cfg["router_settings"] = {"routing_strategy": "usage-based-routing"}


@pytest.mark.parametrize(
    "mutate, failing",
    [
        (_mut_thinking, "b"),
        (_mut_local_api_base, "b"),
        (_mut_literal_gateway_key, "c"),
        (_mut_gateway_api_base, "c"),
        (_mut_settings, "d"),
        (_mut_router_section, "d"),
    ],
    ids=[
        "differing-chat_template_kwargs",
        "differing-hosted_vllm-api_base",
        "gateway-api_key-not-an-env-name",
        "gateway-differs-beyond-api_key",
        "differing-litellm_settings",
        "a-section-only-one-config-has",
    ],
)
def test_two_configs_that_differ_beyond_gateway_keys_are_refused(tmp_path, capsys, mutate, failing):
    recs = healthy()
    other = generated_config()
    mutate(other)
    j1 = config_job(tmp_path, 911001, tracked_config())
    j2 = config_job(tmp_path, 912001, other)
    code = tr.main(_argv(tmp_path, recs, [j1, j2], [serve_record(tmp_path)]))
    out = capsys.readouterr().out
    refused(code, out, "config_sha256")
    assert "FAIL" in _check_line(out, failing), out
    assert "PASS" in _check_line(out, "a"), "the archives are present and match"


def test_a_missing_archived_config_is_refused(tmp_path, capsys):
    recs = healthy()
    j1 = config_job(tmp_path, 911001, tracked_config())
    j2 = config_job(tmp_path, 912001, generated_config(), archive=False)
    code = tr.main(_argv(tmp_path, recs, [j1, j2], [serve_record(tmp_path)]))
    out = capsys.readouterr().out
    refused(code, out, "config_sha256")
    line = _check_line(out, "a")
    assert "FAIL" in line and "litellm.tau2.912001.yaml" in line, line


def test_an_archive_whose_sha_is_not_its_probe_records_is_refused(tmp_path, capsys):
    """(a) is not satisfied by finding a file: the archive's bytes must hash to the probe's sha."""
    recs = healthy()
    j1 = config_job(tmp_path, 911001, tracked_config())
    j2 = config_job(tmp_path, 912001, generated_config())
    arch = j2.parent / "litellm.tau2.912001.yaml"
    arch.write_text(arch.read_text() + "# edited after the probe\n")
    code = tr.main(_argv(tmp_path, recs, [j1, j2], [serve_record(tmp_path)]))
    out = capsys.readouterr().out
    refused(code, out, "config_sha256")
    assert "FAIL" in _check_line(out, "a"), out


def test_one_config_sha_needs_no_archive_and_is_read_as_before(tmp_path, capsys):
    """Unchanged behaviour: one config_sha256 across the probe records requires no archive."""
    recs = healthy()
    j1 = probe_record(tmp_path, models=(BASE, S1), name="_jobs/1/thinking_probe.json")
    j2 = probe_record(tmp_path, models=(S2,), name="_jobs/2/thinking_probe.json")
    code = tr.main(_argv(tmp_path, recs, [j1, j2], [serve_record(tmp_path)]))
    out = capsys.readouterr().out
    assert code == 0, out
    assert "one config_sha256" in out
    assert not [ln for ln in out.splitlines() if ln.strip().startswith("(a)")]
    rep = json.loads((tmp_path / "out.json").read_text())
    assert rep["provenance"]["config_agreement"] is None


# ------------------------------------------------------------------ amendment 11: a prompted 120B arm

#: THE DRIVER'S OWN FILE (lane D, b3a81d8, sha256 bd8665e9): the pin, its store directory, the
#: regime string and the campaign sha are read from it, and the arm's records are written with its
#: own functions (_arm_launch, _stage_job, questioner_probe). Loaded by path, as lane D's tests do,
#: independently of the reader's own loader.
REPO = Path(__file__).resolve().parents[1]
DRIVER = REPO / "scripts" / "tau2_rerun" / "run_extra_pin.py"
_dspec = importlib.util.spec_from_file_location("run_extra_pin_reader_tests", DRIVER)
DRV = importlib.util.module_from_spec(_dspec)
_dspec.loader.exec_module(DRV)
G120 = DRV.EXTRA_PIN
CAMPAIGN = DRV.CAMPAIGN_SHA
QMAX = 1200  # the questioner's EFFECTIVE max_tokens: 400 content x 3.0 headroom (lane D's test)
A11_LIMIT = "limitation of the arm (amendment 11)"


def inq(tok, model=G120, status=200) -> dict:
    return {
        "actor": "inquirer",
        "model": model,
        "tok_completion": tok,
        "tok_reasoning": 0,
        "http_status": status,
    }


def g120_records(at_cap=0, suites=("tau2_retail", "tau2_airline")) -> list:
    """The 120B arm: `inquirer_prompted` pinned to G120, 8 runs of 5 questioner calls each (40),
    `at_cap` of them at 1200 tokens. Every run also carries a drafter call AT the cap, which the
    share must not count. Success t1 1, t2 0; 3 follow-up turns."""
    out = []
    for suite in suites:
        for variant in ("tau2_base", "tau2_stop"):
            for task, trace in (("t1", "aaa"), ("t2", "bbb")):
                r = rec(
                    C,
                    G120,
                    suite=suite,
                    variant=variant,
                    task=task,
                    trace=trace,
                    reward=1.0 if task == "t1" else 0.0,
                    followups=3,
                )
                r["_llm_calls"] = [inq(100) for _ in range(5)] + [
                    {
                        "actor": "drafter",
                        "model": FRONTIER,
                        "tok_completion": QMAX,
                        "http_status": 200,
                    }
                ]
                out.append(r)
    for i in range(at_cap):
        out[i]["_llm_calls"][0] = inq(QMAX if i % 2 else int(0.9 * QMAX))  # 1080 is AT the cap
    return out


def at_campaign(recs: list) -> list:
    """The amendment-11 population runs at the campaign sha: the driver refuses any other tree. Only
    the fixture's default code_version moves, so a test's own defect survives."""
    for r in recs:
        if r["code_version"] == CV:
            r["code_version"] = CAMPAIGN
        if "_manifest" in r and r["_manifest"].get("code_version") == CV:
            r["_manifest"]["code_version"] = CAMPAIGN
    return recs


ASK = '{"action": "ask", "question": "What is the status of order W9?", "target": "kb"}'


@functools.lru_cache(maxsize=None)
def _driver_probe_core(completion: int, reasoning: int) -> str:
    """`run_extra_pin.questioner_probe` itself: the arm's real prompt on a first state, through the
    harness's MeteredClient with only its transport replaced (lane D's `_probe`). Once per shape."""
    from pi_run.stages import tau2_runner as runner
    from pinq.budget import BudgetLedger
    from pinq_adapters.llm import litellm_client as lc
    from pinq_expt import arms

    saved = os.environ.pop("PI_LLM_REASONING_HEADROOM", None)
    try:
        env = {
            "PI_MODEL_INQUIRER": G120,
            "LITELLM_BASE_URL": PROXY,
            "LITELLM_API_KEY": "k",
            "PI_PRICE_TABLE": str(REPO / "scripts" / "price_tables" / "2026-09.json"),
        }
        ledger = BudgetLedger(cap=16)
        client = lc.MeteredClient(ledger, temperature=0.0, env=env)
        client._dispatch = lambda payload: lc._Response(
            text=ASK,
            tok_prompt=2100,
            tok_completion=completion,
            tok_reasoning=reasoning,
            tok_cached=0,
            http_status=200,
        )
        suite_obj = SimpleNamespace(
            suite_id="tau2_retail",
            instructions="You are a retail support agent.",
            corpus_id="tau2_retail",
            corpus_hash="h" * 16,
            tool_schemas=lambda tid: [],
        )
        prefix = [
            SimpleNamespace(role="user", content="I want to return a keyboard.", tool_calls=None),
            SimpleNamespace(role="assistant", content="Sure, what is your email?", tool_calls=None),
            SimpleNamespace(role="user", content="noah@example.com, order W9", tool_calls=None),
        ]
        unit = {
            "suite": "tau2_retail",
            "pin": G120,
            "arm_id": C,
            "variant": "tau2_base",
            "seed": 0,
            "task_id": "t1",
            "k": 4,
            "trace_sha": "aaa",
        }
        rec = DRV.questioner_probe(
            unit=unit,
            prefix=prefix,
            suite_obj=suite_obj,
            runner=runner,
            arms=arms,
            client=client,
            ledger=ledger,
            is_valid=lambda m: True,
            proxy_url=PROXY,
        )
    finally:
        if saved is not None:
            os.environ["PI_LLM_REASONING_HEADROOM"] = saved
    return json.dumps(rec, default=str)


def driver_qrec(
    jobid: str, idx: str, *, suite="tau2_retail", completion=310, reasoning=250, **over
) -> dict:
    """The record `run_extra_pin.py questioner-probe` writes: questioner_probe's own dict plus the
    fields `_probe_main` adds (harness_sha, driver_sha256, tree, at_utc, LSB_JOBID, LSB_JOBINDEX).
    Its `unit` is a planned unit OF THE SUITE probed: one job's retail and airline records differ
    (found 2026-09-23: with the unit fixed to retail, both suites' records were byte-identical and
    the content-keyed count read 2 records for 4 jobs)."""
    rec = json.loads(_driver_probe_core(completion, reasoning))
    rec["unit"] = {**rec["unit"], "suite": suite}
    rec.update(
        harness_sha=CAMPAIGN,
        driver_sha256=DRV.driver_sha256(),
        tree="<staged ee9afda2>",
        at_utc="2026-09-23T13:00:00Z",
        LSB_JOBID=jobid,
        LSB_JOBINDEX=idx,
    )
    rec.update(over)
    return rec


def arm_launch_fields(suite: str, recs, **over) -> dict:
    """The fields run_extra writes into ARM_LAUNCH.json (run_extra_pin.py, `_arm_launch` call)."""
    f = {
        "driver": str(DRIVER),
        "driver_sha256": DRV.driver_sha256(),
        "tree": "<staged ee9afda2>",
        "harness_sha": CAMPAIGN,
        "rerun_py_sha256": "r" * 64,
        "pin": G120,
        "arm_id": C,
        "suite": suite,
        "store_dir": DRV.STORE_DIR,
        "proxy_url": PROXY,
        "pilot": None,
        "regime": DRV.GATEWAY_ONLY,
        "rules": "artifacts/tau2_rerun_20260923/RULES.md amendment 11",
        "plan": [plan_unit(r) for r in recs],
    }
    f.update(over)
    return f


def job_record(suite: str, label: str, jobid: str, idx: str, n_units: int, **over) -> dict:
    """The fields run_extra writes into each job's JOB.json (run_extra_pin.py, JOB.json block)."""
    f = {
        "label": label,
        "regime": DRV.GATEWAY_ONLY,
        "driver": str(DRIVER),
        "driver_sha256": DRV.driver_sha256(),
        "tree": "<staged ee9afda2>",
        "harness_sha": CAMPAIGN,
        "pin": G120,
        "arm_id": C,
        "store": f"<arm>/{suite}/{DRV.STORE_DIR}",
        "suite": suite,
        "shard": "1/1",
        "pilot": None,
        "host": "gpu-n700",
        "LSB_JOBID": jobid,
        "LSB_JOBINDEX": idx,
        "started_at_utc": "2026-09-23T13:00:05Z",
        "proxy_url": PROXY,
        "questioner_max_tokens": QMAX,
        "questioner_probe": {
            "verdict": "PASS",
            "completion_tokens": 310,
            "reasoning_tokens": 250,
            "max_tokens": QMAX,
        },
        "served_adapters": None,
        "n_units": n_units,
    }
    f.update(over)
    return f


#: two LSF array elements per suite
JOBS = (("910500", "1"), ("910500", "2"))


def _none(*_a):
    return {}


def arm_root(
    tmp_path,
    cmp_recs,
    *,
    jobs=JOBS,
    launch_over=_none,
    qrec_over=_none,
    job_over=_none,
    qprobe=True,
    pin_dir=None,
    wrapper=False,
) -> Path:
    """The 120B arm's root, laid out by the DRIVER'S OWN FUNCTIONS: units under
    `<arm>/<suite>/STORE_DIR/`, `<arm>/<suite>/ARM_LAUNCH.json` by `_arm_launch`, and each job's
    questioner probe record staged by `_stage_job` into `<arm>/<suite>/_jobs/gptoss120b-<label>/`
    beside its JOB.json. No thinking probe anywhere: the regime is gateway-only."""
    arm = tmp_path / "arm"
    for suite in sorted({r["suite_id"] for r in cmp_recs}):
        runs_root = arm / suite
        recs = [r for r in cmp_recs if r["suite_id"] == suite]
        _write_store(arm, recs, pin_dir=pin_dir or (lambda pin: DRV.STORE_DIR))
        DRV._arm_launch(
            runs_root / "ARM_LAUNCH.json", arm_launch_fields(suite, recs, **launch_over(suite))
        )
        for jobid, idx in jobs:
            label = f"gptoss120b-{suite[5:]}-{jobid}-{idx}"
            src = tmp_path / "qsrc" / f"{label}.json"
            src.parent.mkdir(parents=True, exist_ok=True)
            src.write_text(
                json.dumps(driver_qrec(jobid, idx, suite=suite, **qrec_over(suite, jobid, idx)))
            )
            if (
                wrapper
            ):  # the HPC wrapper's own copy: $ARM/_wrapper/<LSB_JOBID>/...<suite>.k<k>.json
                w = arm / "_wrapper" / jobid / f"questioner_probe.{suite}.k{idx}.json"
                w.parent.mkdir(parents=True, exist_ok=True)
                w.write_bytes(src.read_bytes())
            job = DRV._stage_job(
                runs_root, label, {"questioner_probe.json": str(src) if qprobe else None}
            )
            (job / "JOB.json").write_text(
                json.dumps(
                    job_record(suite, label, jobid, idx, len(recs), **job_over(suite, jobid, idx))
                )
            )
    return arm


def run_120(
    tmp_path,
    main_recs,
    cmp_recs,
    *extra,
    cmp_launch=True,
    probe_models=(BASE, S1, S2),
    capsys=None,
    wrapper_glob=False,
    **arm_kw,
):
    """The amendment-11 reading: trained s1/s2 (and the 8B base) from the main source, the 120B
    arm from the root the driver lays out, its ARM_LAUNCH.json found by directory. The thinking
    probe is the main population's and covers the LOCAL pins only; the 120B arm's jobs each carry
    a questioner probe record instead."""
    at_campaign(main_recs)
    at_campaign(cmp_recs)
    arm = arm_root(tmp_path, cmp_recs, wrapper=wrapper_glob, **arm_kw)
    f = tmp_path / "records.json"
    f.write_text(json.dumps(main_recs))
    argv = ["--records", str(f), *FAST_BOOT, "--out", str(tmp_path / "out.json")]
    argv += ["--tau2-data-dir", str(tau2_data(tmp_path))]
    argv += ["--serve-record", str(serve_record(tmp_path))]
    argv += ["--probe-record", str(probe_record(tmp_path, models=probe_models))]
    for lp in launch_records(tmp_path, main_recs):
        argv += ["--launch-record", str(lp)]
    argv += ["--comparator-store", str(arm), "--comparator-model", G120]
    if cmp_launch:
        argv += ["--comparator-launch-record", str(arm)]
    if wrapper_glob:
        argv += ["--questioner-probe-record", f"{arm}/_wrapper/*/questioner_probe.*.k*.json"]
    argv += list(extra)
    code = tr.main(argv)
    out = capsys.readouterr().out if capsys is not None else ""
    return code, out


def test_extract_embeds_the_questioner_calls(tmp_path):
    """calls.jsonl rows of the questioner role, in order: [model, tok_completion, http_status].
    Other roles are not embedded; an absent calls.jsonl is None, never []."""
    rows = [
        inq(90),
        {"actor": "drafter", "model": FRONTIER, "tok_completion": 7},
        inq(0, status=500),
    ]
    _write_run(
        tmp_path / "a", "r1", {"run_id": "r1", "status": "ok"}, manifest(G120), [], llm_calls=rows
    )
    _write_run(tmp_path / "b", "r2", {"run_id": "r2", "status": "ok"}, manifest(G120), [])
    (r1,) = extract_records.extract(tmp_path / "a")
    (r2,) = extract_records.extract(tmp_path / "b")
    assert r1["_inquirer_calls"] == [[G120, 90, 200], [G120, 0, 500]]
    assert r2["_inquirer_calls"] is None


def test_a_comparator_store_is_paired_against_the_trained_seeds(tmp_path, capsys):
    """Amendment 11: the 120B arm from its own root, paired against trained s1/s2 from the main
    source. Success: s1 1 everywhere, the 120B 1 at t1 and 0 at t2, so retail/base/s1 reads
    t1 0, t2 +1 -- the 8B base (0 everywhere) would read t1 +1, t2 +1."""
    code, out = run_120(tmp_path, healthy(), g120_records(), capsys=capsys)
    assert code == 0, out
    rep = json.loads((tmp_path / "out.json").read_text())
    assert rep["rules"]["contrast"] == "trained vs prompted gpt-oss-120b"
    assert rep["rules"]["comparator_model"] == G120
    assert rep["source"]["comparator_store"]["kind"] == "store"
    assert len(rep["cells"]) == 8
    # found 2026-09-23: a loop's `label` shadowed the contrast and stamped "s1"/"s2" on the cells
    assert {c["contrast"] for c in rep["cells"]} == {"trained vs prompted gpt-oss-120b"}
    assert rep["holm"]["contrast"] == "trained vs prompted gpt-oss-120b"
    assert "vs prompted gpt-oss-120b: 2 pairs" in out
    cell = next(
        c
        for c in rep["cells"]
        if (c["suite"], c["variant"], c["seed_label"]) == ("tau2_retail", "tau2_base", "s1")
    )
    assert cell["guard"]["task_level"]["task_deltas"] == {"t1": 0.0, "t2": 1.0}
    prompted = [rid for c in rep["cells"] for rid in c["run_ids"] if rid.startswith("prom-")]
    assert prompted and all(rid.startswith("prom-0b-") for rid in prompted), "120B runs only"
    assert "CONTRAST: trained vs prompted gpt-oss-120b" in out
    holm_line = next(ln for ln in out.splitlines() if "Holm family (8 cells" in ln)
    assert "trained vs prompted gpt-oss-120b" in holm_line
    assert all("reference" not in m for m in rep["holm"]["family"])


def _set_cv(r):
    r["code_version"] = "b" * 40
    r["_manifest"]["code_version"] = "b" * 40


def _set_dirty(r):
    r["dirty"] = True


def _set_url(r):
    r["_manifest"]["pins"]["inquirer"]["base_url_sha"] = "9" * 64


@pytest.mark.parametrize(
    "mutate, needle",
    [(_set_cv, "code_version values"), (_set_dirty, "dirty"), (_set_url, "base_url_sha")],
    ids=["code_version", "dirty", "base_url_sha"],
)
def test_the_provenance_refusals_span_both_stores(tmp_path, capsys, mutate, needle):
    """One code_version, clean, one base_url_sha per role -- across BOTH stores together: a
    defect in the comparator store alone refuses the reading."""
    cmp = g120_records()
    mutate(cmp[0])
    refused(*run_120(tmp_path, healthy(), cmp, capsys=capsys), needle)


def test_the_comparator_pin_in_the_main_source_is_refused(tmp_path, capsys):
    main = healthy() + [rec(C, G120, task="t9", trace="zzz")]
    refused(*run_120(tmp_path, main, g120_records(), capsys=capsys), "--comparator-store")


def test_the_comparator_store_needs_its_own_launch_record(tmp_path, capsys):
    refused(
        *run_120(tmp_path, healthy(), g120_records(), cmp_launch=False, capsys=capsys),
        "--comparator-launch-record",
    )


def test_the_thinking_probe_is_required_only_for_local_pins(tmp_path, capsys):
    """The gateway pin is not thinking-probed; a local pin missing from the probe still refuses."""
    code, out = run_120(tmp_path, healthy(), g120_records(), capsys=capsys)
    assert code == 0, out
    assert f"gateway pin(s) not thinking-probed: ['{G120}']" in out
    refused(
        *run_120(tmp_path / "x", healthy(), g120_records(), probe_models=(BASE, S2), capsys=capsys),
        "do not cover",
    )


@pytest.mark.parametrize(
    "over, needle",
    [
        ({"verdict": "REFUSE"}, "verdict"),
        ({"model": "openai/aws/gpt-5"}, "questioner probe record"),
        ({"max_tokens": None}, "max_tokens"),
        ({"completion_tokens": 1100}, "completion_tokens"),
        ({"harness_sha": "0" * 40}, "harness_sha"),
        ({"proxy_url": PROXY + "/"}, "proxy_url"),
    ],
    ids=[
        "not-pass",
        "another-model",
        "no-max_tokens",
        "pass-but-at-the-cap",
        "another-harness",
        "another-proxy",
    ],
)
def test_a_bad_questioner_probe_record_in_one_job_is_refused(tmp_path, capsys, over, needle):
    """The driver's own record in every job, and ONE job's record altered: airline element 2."""

    def one_job(suite, jobid, idx):
        return over if (suite, idx) == ("tau2_airline", "2") else {}

    refused(*run_120(tmp_path, healthy(), g120_records(), qrec_over=one_job, capsys=capsys), needle)


def test_a_gateway_questioner_without_any_probe_record_is_refused(tmp_path, capsys):
    """No job directory and no --questioner-probe-record: nothing shows the 120B was not truncated."""
    refused(
        *run_120(tmp_path, healthy(), g120_records(), jobs=(), capsys=capsys),
        "--questioner-probe-record",
    )


def test_a_job_without_its_questioner_probe_record_is_refused(tmp_path, capsys):
    refused(
        *run_120(tmp_path, healthy(), g120_records(), qprobe=False, capsys=capsys),
        "holds no questioner_probe.json",
    )


def test_every_jobs_questioner_probe_is_read_and_recorded(tmp_path, capsys):
    """One record PER JOB, found in the driver's job directories: 2 suites x 2 array elements."""
    code, out = run_120(tmp_path, healthy(), g120_records(), capsys=capsys)
    assert code == 0, out
    rep = json.loads((tmp_path / "out.json").read_text())
    recs = rep["provenance"]["questioner_probe"]["records"][G120]
    assert len(recs) == 4
    assert sorted((r["LSB_JOBID"], r["LSB_JOBINDEX"]) for r in recs) == sorted(
        [("910500", "1"), ("910500", "2")] * 2
    )
    assert {r["harness_sha"] for r in recs} == {CAMPAIGN}
    assert {r["reasoning_tokens"] for r in recs} == {250}
    assert rep["provenance"]["questioner_probe"]["max_tokens"] == {G120: QMAX}


# ------------------------------------------------------------------ amendment 11, the driver's layout


def test_the_store_directory_is_the_drivers(tmp_path, capsys):
    """The driver writes units to `<runs-root>/STORE_DIR/` (the pin with '/' as '__'), pinned here
    against the driver's OWN OUTPUT (the runs_root of every spec `build` makes). A store laid out
    at the pin's slashes -- this reader's belief before b3a81d8 was read -- is refused."""
    assert tr.comparator_store_dir(G120) == "openai__aws__gpt-oss-120b" == DRV.STORE_DIR
    rr = DRV.load_rerun(REPO)
    gi = SimpleNamespace(sha="abc123", dirty=False, dirty_files=())
    work = DRV.build(rr, "tau2_retail", Path("/r"), Path("/c"), gi)
    assert {Path(spec.runs_root).name for spec, _pin, _v in work} == {tr.comparator_store_dir(G120)}
    code, out = run_120(tmp_path, healthy(), g120_records(), capsys=capsys)
    assert code == 0, out
    subs = json.loads((tmp_path / "out.json").read_text())["source"]["comparator_store"]
    assert {s["dir"] for s in subs["sub_roots"]} == {DRV.STORE_DIR}
    nested = tmp_path / "nested"
    refused(
        *run_120(nested, healthy(), g120_records(), pin_dir=lambda pin: pin, capsys=capsys),
        "missing",
    )


def test_a_run_in_the_store_whose_manifest_names_another_pin_is_refused(tmp_path, capsys):
    """The directory is the driver's name for the pin; the manifest must still name the pin."""
    cmp = g120_records()
    cmp[0]["_manifest"]["pins"]["inquirer"]["model_id"] = "openai/aws/gpt-5"
    refused(*run_120(tmp_path, healthy(), cmp, capsys=capsys), "sub-root")


def test_the_arm_launch_record_is_printed_and_recorded(tmp_path, capsys):
    code, out = run_120(tmp_path, healthy(), g120_records(), capsys=capsys)
    assert code == 0, out
    launch = json.loads((tmp_path / "out.json").read_text())["provenance"]["comparator_launch"]
    assert [x["suite"] for x in launch] == ["tau2_retail", "tau2_airline"]
    assert {x["driver_sha256"] for x in launch} == {DRV.driver_sha256()}
    assert {x["regime"] for x in launch} == {DRV.GATEWAY_ONLY}
    assert {x["harness_sha"] for x in launch} == {CAMPAIGN}
    assert {x["n_units"] for x in launch} == {4}
    assert f"driver_sha256 {DRV.driver_sha256()[:16]}" in out
    assert f"regime {DRV.GATEWAY_ONLY!r}" in out


@pytest.mark.parametrize(
    "over, needle",
    [
        ({"harness_sha": "0" * 40}, "harness_sha"),
        ({"regime": "local: thinking probe + serve record + questioner probe"}, "regime"),
        ({"store_dir": "openai/aws/gpt-oss-120b"}, "store_dir"),
        ({"pin": "openai/aws/gpt-5"}, "is for pin"),
        ({"arm_id": "inquirer_trained"}, "arm_id"),
    ],
    ids=["another-harness", "local-regime", "another-store-dir", "another-pin", "another-arm"],
)
def test_an_arm_launch_record_off_the_declared_arm_is_refused(tmp_path, capsys, over, needle):
    refused(
        *run_120(tmp_path, healthy(), g120_records(), launch_over=lambda s: over, capsys=capsys),
        needle,
    )


def test_two_suites_launched_by_two_drivers_are_refused(tmp_path, capsys):
    def airline(suite):
        return {"driver_sha256": "0" * 64} if suite == "tau2_airline" else {}

    refused(
        *run_120(tmp_path, healthy(), g120_records(), launch_over=airline, capsys=capsys),
        "driver_sha256",
    )


@pytest.mark.parametrize(
    "over, needle",
    [
        ({"regime": "local: thinking probe + serve record + questioner probe"}, "JOB.json regime"),
        ({"driver_sha256": "0" * 64}, "JOB.json driver_sha256"),
        ({"harness_sha": "0" * 40}, "JOB.json harness_sha"),
        ({"LSB_JOBINDEX": "9"}, "LSB_JOB"),
    ],
    ids=["local-regime", "another-driver", "another-harness", "another-job"],
)
def test_a_job_off_its_arm_launch_record_is_refused(tmp_path, capsys, over, needle):
    def one_job(suite, jobid, idx):
        return over if (suite, idx) == ("tau2_retail", "1") else {}

    refused(*run_120(tmp_path, healthy(), g120_records(), job_over=one_job, capsys=capsys), needle)


def test_no_qwen_probe_is_demanded_of_the_comparator_jobs(tmp_path, capsys):
    """The 120B arm is gateway-only by design: its jobs hold no thinking probe and no serve record,
    and the only --probe-record is the main population's, covering the Qwen pins."""
    code, out = run_120(tmp_path, healthy(), g120_records(), capsys=capsys)
    assert code == 0, out
    assert not list((tmp_path / "arm").rglob("thinking_probe*.json"))
    assert not list((tmp_path / "arm").rglob("serve_record*.json"))
    assert f"gateway pin(s) not thinking-probed: ['{G120}']" in out


def test_the_reader_loads_the_drivers_own_file():
    drv = tr.extra_pin_driver()
    assert Path(drv.__file__).resolve() == DRIVER.resolve()
    assert drv.driver_sha256() == hashlib.sha256(DRIVER.read_bytes()).hexdigest()


def _token_cap(tmp_path, capsys, cmp):
    code, out = run_120(tmp_path, healthy(), cmp, capsys=capsys)
    assert code == 0, out
    return out, json.loads((tmp_path / "out.json").read_text())["questioner_token_cap"]


def test_the_token_cap_share_at_exactly_five_percent_is_not_a_limitation(tmp_path, capsys):
    """2 of 40 questioner calls at the cap (one at 1080 = 0.9 x 1200, one at 1200) is 5.00%, which
    does not EXCEED 0.05. The eight drafter calls at 1200 are not questioner calls: counted, the
    share would be 10/48."""
    out, tc = _token_cap(tmp_path, capsys, g120_records(at_cap=2))
    assert (tc["n_calls"], tc["n_at_cap"]) == (40, 2)
    assert tc["share"] == pytest.approx(0.05)
    assert tc["threshold_tokens"] == pytest.approx(1080)
    assert tc["limitation"] is False
    assert "QUESTIONER TOKEN CAP" in out and A11_LIMIT not in out


def test_a_token_cap_share_above_five_percent_is_a_limitation_of_the_arm(tmp_path, capsys):
    out, tc = _token_cap(tmp_path, capsys, g120_records(at_cap=3))
    assert (tc["n_calls"], tc["n_at_cap"]) == (40, 3)
    assert tc["limitation"] is True
    assert A11_LIMIT in out


def test_a_failed_questioner_call_is_not_in_the_share(tmp_path, capsys):
    cmp = g120_records(at_cap=2)
    cmp[5]["_llm_calls"].append(inq(0, status=500))
    out, tc = _token_cap(tmp_path, capsys, cmp)
    assert (tc["n_calls"], tc["n_at_cap"], tc["n_not_200_excluded"]) == (40, 2, 1)


def test_a_questioner_call_to_another_model_is_refused(tmp_path, capsys):
    """Filtering by model would silently drop it; a questioner call off the pin is a misroute.
    (The needle was "model" at first and passed before any code existed, on the comparator-pin
    refusal's "pools model pins": too loose to be a test.)"""
    cmp = g120_records()
    cmp[0]["_llm_calls"][1] = inq(100, model="openai/aws/gpt-5")
    refused(*run_120(tmp_path, healthy(), cmp, capsys=capsys), "name a model other than the pin")


def test_a_comparator_run_without_calls_is_refused(tmp_path, capsys):
    cmp = g120_records()
    del cmp[0]["_llm_calls"]
    refused(*run_120(tmp_path, healthy(), cmp, capsys=capsys), "calls.jsonl")


def test_the_8b_base_against_the_120b_is_descriptive_with_no_claim(tmp_path, capsys):
    """Amendment 11 point 2. Delta = 8B base minus 120B: success t1 0-1, t2 0-0 (mean -0.5);
    follow-ups 4 - 3 = +1 at every task. Printed interval only; in no Holm family."""
    code, out = run_120(tmp_path, healthy(), g120_records(), capsys=capsys)
    assert code == 0, out
    rep = json.loads((tmp_path / "out.json").read_text())
    refc = rep["reference_contrast"]
    assert refc["label"] == "descriptive, no claim"
    assert (refc["reference_model"], refc["comparator_model"]) == (BASE, G120)
    assert len(refc["cells"]) == 4 and all(c["n_pairs"] == 2 for c in refc["cells"])
    c0 = next(
        c for c in refc["cells"] if (c["suite"], c["variant"]) == ("tau2_retail", "tau2_base")
    )
    assert c0["endpoints"]["task success"]["task_deltas"] == {"t1": -1.0, "t2": 0.0}
    assert c0["endpoints"]["task success"]["mean_delta"] == pytest.approx(-0.5)
    fu = c0["endpoints"]["follow-up user turns after the fork point"]
    assert fu["task_deltas"] == {"t1": 1.0, "t2": 1.0}
    assert "prompted qwen3-8b-base vs prompted gpt-oss-120b -- descriptive, no claim" in out


def test_the_8b_reading_is_unchanged_by_amendment_11(tmp_path, capsys):
    code, out = run(tmp_path, healthy(), capsys=capsys)
    assert code == 0, out
    rep = json.loads((tmp_path / "out.json").read_text())
    assert rep["rules"]["contrast"] == "trained vs prompted qwen3-8b-base"
    assert rep["questioner_token_cap"] is None and rep["reference_contrast"] is None
    assert "QUESTIONER TOKEN CAP" not in out and "descriptive, no claim" not in out


def test_a_comparator_store_for_a_main_source_pin_is_refused(tmp_path, capsys):
    """--comparator-store with the comparator left at its default (the 8B base, also the reference
    read from the main source) would pool one pin from two roots."""
    refused(
        *run_120(tmp_path, healthy(), g120_records(), "--comparator-model", BASE, capsys=capsys),
        "would pool it",
    )


def test_the_wrappers_copies_are_read_by_glob_and_counted_once(tmp_path, capsys):
    """The HPC wrapper writes each job's record to `$ARM/_wrapper/<LSB_JOBID>/questioner_probe.
    <suite>.k<k>.json` and the driver copies the same bytes into the job's directory. Read by the
    glob as well, a byte-identical copy is one record: 4 jobs, 4 records -- not 8."""
    code, out = run_120(tmp_path, healthy(), g120_records(), wrapper_glob=True, capsys=capsys)
    assert code == 0, out
    qp = json.loads((tmp_path / "out.json").read_text())["provenance"]["questioner_probe"]
    assert len(qp["records"][G120]) == 4
    assert qp["n_byte_identical_copies"] == 4
    assert f"questioner probe records for {G120}: 4 PASS" in out


def test_a_refuse_record_in_the_wrappers_glob_is_refused(tmp_path, capsys):
    """All must be PASS for the pin: a wrapper record from a job that refused before staging (so in
    no job directory) still refuses the reading when the glob reads it."""
    arm = tmp_path / "arm"
    stray = arm / "_wrapper" / "910777" / "questioner_probe.tau2_retail.k1.json"
    stray.parent.mkdir(parents=True)
    stray.write_text(json.dumps(driver_qrec("910777", "1", verdict="REFUSE")))
    refused(
        *run_120(tmp_path, healthy(), g120_records(), wrapper_glob=True, capsys=capsys), "verdict"
    )


# ------------------------------------------------------------------ amendment 8: exploratory ask quality

EXPL = "EXPLORATORY (declared in amendment 8 after one interim look)"
AQ_RETRIEVE = "share of asks retrieving >= 1 record"
AQ_EXACT = "exact repeat rate"
AQ_NEAR = "near-exact repeat rate"
AQ = (AQ_RETRIEVE, AQ_EXACT, AQ_NEAR)

Q1 = "What is the status, items and shipping address of order #W1?"
Q2 = "what is the status,  items and shipping address of ORDER #W1?"  # Q1, lowercased + spaces
Q3 = "For order #W1: what are its items, status and shipping address?"  # a reword, not exact
Q4 = "What is the customer's email for verification?"
SHORT = "Find order #W2"  # repeated verbatim: exact, but 2 content tokens -- under the guard's 4


def test_extract_embeds_each_asks_question_in_turn_order(tmp_path):
    """turns.jsonl `question` on `action_kind == "ask"` rows, ordered by turn_idx (written here out
    of order, with a non-ask row between). An ask row with no question makes the list None, never
    a guessed "", and so does an absent turns.jsonl."""
    d = _write_run(tmp_path / "a", "r1", {"run_id": "r1", "status": "ok"}, manifest(BASE), [])
    rows = [
        {"action_kind": "ask", "turn_idx": 2, "question": "third", "retrieved_uids": []},
        {"action_kind": "ask", "turn_idx": 0, "question": "first", "retrieved_uids": ["u"]},
        {"action_kind": "stop", "turn_idx": 3},
        {"action_kind": "ask", "turn_idx": 1, "question": "second", "retrieved_uids": []},
    ]
    (d / "turns.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
    (r1,) = extract_records.extract(tmp_path / "a")
    assert r1["_ask_questions"] == ["first", "second", "third"]
    d2 = _write_run(tmp_path / "b", "r2", {"run_id": "r2", "status": "ok"}, manifest(BASE), [])
    (d2 / "turns.jsonl").write_text(json.dumps({"action_kind": "ask", "turn_idx": 0}) + "\n")
    _write_run(tmp_path / "c", "r3", {"run_id": "r3", "status": "ok"}, manifest(BASE), [])
    (r2,) = extract_records.extract(tmp_path / "b")
    (r3,) = extract_records.extract(tmp_path / "c")
    assert r2["_ask_questions"] is None and r3["_ask_questions"] is None


def _aq_population() -> list:
    """Known answers, retail/base, by hand:
    s1 t1 asks Q1 Q2 Q3 Q4, retrieving [u1] [] [u2] []: retrieve 2/4, exact 1/4 (Q2), near 2/4
       (Q2 and Q3 share all six content tokens with Q1; Q4 shares none). Comparator t1 asks Q4
       retrieving [u9]: 1, 0, 0. Deltas +(-0.5), +0.25, +0.5.
    s1 t2 asks nothing: undefined, the pair dropped and counted (trained 1).
    Comparator t2 asks SHORT twice, retrieving nothing: retrieve 0, exact 1/2, near 0 -- the guard
       needs 4 shared content tokens and SHORT has two ("order", "w2").
    s2 asks the fixture's distinct questions (retrieve 0, exact 0, near 0) everywhere:
       t1 deltas -1, 0, 0; t2 deltas 0, -0.5, 0.
    Seeds pooled within task (pair_mean): retrieve t1 (-0.5 - 1)/2, t2 0; exact t1 0.25/2, t2 -0.5;
       near t1 0.5/2, t2 0."""
    recs = healthy()

    def at(model, task):
        return next(
            i
            for i, r in enumerate(recs)
            if r["_manifest"]["pins"]["inquirer"]["model_id"] == model
            and r["task_id"] == task
            and r["suite_id"] == "tau2_retail"
            and r["key"][9] == "tau2_base"
        )

    r = rec(T, S1, task="t1", trace="aaa", asks=4, reward=1.0)
    r["_ask_questions"], r["_ask_retrieved_uids"] = [Q1, Q2, Q3, Q4], [["u1"], [], ["u2"], []]
    recs[at(S1, "t1")] = r
    recs[at(S1, "t2")] = rec(T, S1, task="t2", trace="bbb", asks=0, reward=1.0)
    r = rec(C, BASE, task="t1", trace="aaa", asks=1, reward=0.0)
    r["_ask_questions"], r["_ask_retrieved_uids"] = [Q4], [["u9"]]
    recs[at(BASE, "t1")] = r
    r = rec(C, BASE, task="t2", trace="bbb", asks=2, reward=0.0)
    r["_ask_questions"], r["_ask_retrieved_uids"] = [SHORT, SHORT], [[], []]
    recs[at(BASE, "t2")] = r
    return recs


def _aq(rep: dict, block: str, **where) -> dict:
    return next(
        c["measures"]
        for c in rep["exploratory_ask_quality"][block]
        if all(c[k] == v for k, v in where.items())
    )


def test_the_exploratory_ask_quality_measures_read_their_known_answers(tmp_path, capsys):
    code, out = run(tmp_path, _aq_population(), "--exploratory-ask-quality", capsys=capsys)
    assert code == 0, out
    rep = json.loads((tmp_path / "out.json").read_text())
    assert rep["exploratory_ask_quality"]["label"] == EXPL
    m1 = _aq(rep, "cells", suite="tau2_retail", variant="tau2_base", seed_label="s1")
    assert m1[AQ_RETRIEVE]["task_deltas"] == {"t1": -0.5}
    assert (m1[AQ_RETRIEVE]["trained_level"], m1[AQ_RETRIEVE]["control_level"]) == (0.5, 1.0)
    assert (m1[AQ_RETRIEVE]["n_pairs"], m1[AQ_RETRIEVE]["n_defined_pairs"]) == (2, 1)
    assert m1[AQ_RETRIEVE]["undefined_units"] == {"trained": 1, "control": 0}
    assert m1[AQ_EXACT]["task_deltas"] == {"t1": 0.25}
    assert m1[AQ_NEAR]["task_deltas"] == {"t1": 0.5}
    assert m1[AQ_NEAR]["printed_interval"]["seed"] == 0
    m2 = _aq(rep, "cells", suite="tau2_retail", variant="tau2_base", seed_label="s2")
    assert m2[AQ_RETRIEVE]["task_deltas"] == {"t1": -1.0, "t2": 0.0}
    assert m2[AQ_EXACT]["task_deltas"] == {"t1": 0.0, "t2": -0.5}
    assert m2[AQ_NEAR]["task_deltas"] == {"t1": 0.0, "t2": 0.0}, "a short exact repeat"
    pooled = _aq(rep, "pooled", suite="tau2_retail", variant="tau2_base")
    assert pooled[AQ_RETRIEVE]["task_deltas"] == {"t1": -0.75, "t2": 0.0}
    assert pooled[AQ_EXACT]["task_deltas"] == {"t1": 0.125, "t2": -0.5}
    assert pooled[AQ_NEAR]["task_deltas"] == {"t1": 0.25, "t2": 0.0}
    assert pooled[AQ_NEAR]["pooling"] == "pair_mean"


def test_every_exploratory_line_and_the_json_carry_the_label(tmp_path, capsys):
    code, out = run(tmp_path, _aq_population(), "--exploratory-ask-quality", capsys=capsys)
    assert code == 0, out
    mine = [ln for ln in out.splitlines() if any(n in ln for n in AQ)]
    assert len(mine) >= 3 * (8 + 4), "every cell and pooled line"
    assert all(EXPL in ln for ln in mine), [ln for ln in mine if EXPL not in ln][:3]
    aq = json.loads((tmp_path / "out.json").read_text())["exploratory_ask_quality"]
    assert aq["label"] == EXPL
    assert all(c["label"] == EXPL for c in aq["cells"] + aq["pooled"])


def test_the_near_exact_guard_is_the_repos_pair_paraphrase_guard_itself():
    """Imported, never reimplemented: the reader's guard IS pinq_train's function object."""
    from pinq_train.export import dataset

    assert tr.is_paraphrase is dataset.is_paraphrase
    assert tr.is_paraphrase(Q1, Q3) and not tr.is_paraphrase(SHORT, SHORT)


def test_the_exploratory_block_leaves_the_primary_reading_untouched(tmp_path, capsys):
    """The same records read with and without the block: every cell, pooled line and Holm column
    identical -- the primary's pair set is the pair set, whatever the block reads."""
    reps = []
    for sub, extra in (("off", ()), ("on", ("--exploratory-ask-quality",))):
        (tmp_path / sub).mkdir()
        code, out = run(tmp_path / sub, _aq_population(), *extra, capsys=capsys)
        assert code == 0, out
        reps.append(json.loads((tmp_path / sub / "out.json").read_text()))
    off, on = reps
    assert off["cells"] == on["cells"] and off["pooled"] == on["pooled"]
    assert off["holm"] == on["holm"]
    assert off["exploratory_ask_quality"] is None and on["exploratory_ask_quality"] is not None


def test_an_ask_quality_measure_without_its_field_is_not_recorded_not_zero(tmp_path, capsys):
    recs = _aq_population()
    del recs[0]["_ask_questions"]
    code, out = run(tmp_path, recs, "--exploratory-ask-quality", capsys=capsys)
    assert code == 0, out
    rep = json.loads((tmp_path / "out.json").read_text())
    m = _aq(rep, "cells", suite="tau2_retail", variant="tau2_base", seed_label="s1")
    assert m[AQ_EXACT] is None and m[AQ_NEAR] is None
    assert m[AQ_RETRIEVE] is not None
    assert any(f"[{AQ_EXACT}] NOT RECORDED" in ln and EXPL in ln for ln in out.splitlines())


def test_the_exploratory_block_reads_the_120b_arm_and_the_8b_against_it(tmp_path, capsys):
    """Through --comparator-store: trained vs prompted 120B in the cells, and the prompted 8B base
    against the 120B beside them. The 120B's questions are read back from its store's turns.jsonl."""
    code, out = run_120(
        tmp_path, healthy(), g120_records(), "--exploratory-ask-quality", capsys=capsys
    )
    assert code == 0, out
    aq = json.loads((tmp_path / "out.json").read_text())["exploratory_ask_quality"]
    assert {c["contrast"] for c in aq["cells"]} == {"trained vs prompted gpt-oss-120b"}
    assert len(aq["cells"]) == 8 and len(aq["pooled"]) == 4
    assert len(aq["reference"]) == 4
    assert {c["delta"] for c in aq["reference"]} == {f"{BASE} minus {G120}"}
    ref = next(
        c["measures"]
        for c in aq["reference"]
        if (c["suite"], c["variant"]) == ("tau2_retail", "tau2_base")
    )
    assert ref[AQ_EXACT]["task_deltas"] == {"t1": 0.0, "t2": 0.0}
    assert ref[AQ_EXACT]["n_defined_pairs"] == 2
    assert any("qwen3-8b-base minus gpt-oss-120b" in ln and EXPL in ln for ln in out.splitlines())
