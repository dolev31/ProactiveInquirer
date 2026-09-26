"""The HTTP surface itself, driven through the real ASGI app.

WHY THIS FILE EXISTS. `tests/test_serve.py` covers the handlers, which are plain functions
over dataclasses -- and they were all correct while the SERVER was completely unreachable.
`create_app` imported `fastapi.Request` inside the function body; with
`from __future__ import annotations` every annotation is a string, FastAPI resolved
`"Request"` against the module globals, found nothing, and classified the parameter as a
QUERY parameter. Every POST answered

    422 {"detail":[{"type":"missing","loc":["query","request"],"msg":"Field required"}]}

`GET /healthz` takes no parameters, so a health check passed while `/rollout`, `/score` and
`/retrieve` were all dead. Unit tests on the handlers cannot see that, and a trainer would
have discovered it as an unexplained 422 in the middle of a paid rollout campaign.

`fastapi` is in the `serve` extra, so this module skips when it is absent rather than making
the default gate depend on an extra. It needs no network, no key and no GPU.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pinq.wire import SPLIT_REFUSED_EXIT, to_dict

pytest.importorskip("fastapi", reason="serve extra not installed")
pytest.importorskip("httpx", reason="fastapi.testclient needs httpx")


def _client(tmp_path: Path, monkeypatch, *, gold: bool = True):
    """Build the app in ONE of its two roles.

    ONE PROCESS, ONE ROLE. A /score server needs PI_GOLD_ROOT set; a /rollout server needs it
    unset and `handle_rollout` re-asserts that on every request. Hosting both in one process
    is a configuration error, so these tests build the app twice rather than once -- which is
    also how the servers are actually started (see docs/TRAINING.md section 1).
    """
    from fastapi.testclient import TestClient

    from pi_eval.build.synth_build import build
    from pi_run.serve.app import ServerConfig, create_app
    from pinq_adapters.synth.suite import SynthSuite

    corpus, *_ = build(n_tasks=4, n_facets=2, depth=2, root=tmp_path)
    suite = SynthSuite(Path(corpus).parent)
    tid = str(suite.task_ids()[0])
    units = list(suite.retriever(tid)._units)

    runs = tmp_path / "runs"
    for run_id, split in (("train_run", "train"), ("test_run", "test")):
        d = runs / run_id
        d.mkdir(parents=True)
        (d / "manifest.json").write_text(
            json.dumps({"run_id": run_id, "suite_id": "synth", "task_id": tid, "split": split})
        )
        (d / "turns.jsonl").write_text(
            "\n".join(
                json.dumps({"turn_idx": i, "retrieved_uids": [u.uid], "new_uids": [u.uid]})
                for i, u in enumerate(units[:2])
            )
        )
        (d / "status.json").write_text(json.dumps({"stop_reason": "policy_stop", "wall_ms": 1}))
        (d / "ledger.jsonl").write_text(
            json.dumps({"currency": "retrieval_calls", "cumulative": 2.0})
        )

    if gold:
        monkeypatch.setenv("PI_GOLD_ROOT", str(tmp_path / "data" / "gold"))
    else:
        monkeypatch.delenv("PI_GOLD_ROOT", raising=False)
    cfg = ServerConfig(
        root=tmp_path, runs_root=runs, default_suite="synth", default_task=tid, code_version="test"
    )
    return TestClient(create_app(cfg)), tid, units


def test_every_endpoint_actually_accepts_a_json_body(tmp_path, monkeypatch):
    """The regression test for the 422. A POST must read its body, not a query string."""
    from pinq.wire import RetrieveRequest, ScoreRequest

    client, tid, units = _client(tmp_path, monkeypatch)

    health = client.get("/healthz")
    assert health.status_code == 200
    assert health.json()["gold_root_set"] is True

    scored = client.post(
        "/score", json=to_dict(ScoreRequest(episode_id="train_run", mode="prefix_ladder"))
    )
    assert scored.status_code == 200, scored.text
    body = scored.json()
    assert body["ok"] and body["split"] == "train"
    assert body["potential"] == sorted(body["potential"])
    assert len(body["turns"]) == 2

    got = client.post(
        "/retrieve",
        json=to_dict(
            RetrieveRequest(queries=[units[0].title], topk=2, suite_id="synth", task_id=tid)
        ),
    )
    assert got.status_code == 200, got.text
    assert set(got.json()["result"][0][0]["document"]) == {"id", "contents"}


def test_a_non_train_task_is_refused_with_403_over_http(tmp_path, monkeypatch):
    """403, not 400: the request was well formed and is refused on POLICY. A reviewer reading
    a proxy log must be able to tell that apart from a malformed request."""
    from pinq.wire import ScoreRequest

    client, _, _ = _client(tmp_path, monkeypatch)
    resp = client.post("/score", json=to_dict(ScoreRequest(episode_id="test_run")))
    assert resp.status_code == 403
    assert resp.json()["refused"] is True
    assert "not 'train'" in resp.json()["error"]
    # The CLI form of the same refusal carries its own exit code, and they mean one thing.
    assert SPLIT_REFUSED_EXIT == 3


def test_a_rollout_request_that_would_move_the_frozen_roles_is_409(tmp_path, monkeypatch):
    """One LITELLM_BASE_URL per process applies to EVERY role, so overriding it re-routes the
    frozen Drafter and Answerer along with the policy."""
    from pinq.wire import RolloutRequest

    monkeypatch.delenv("LITELLM_BASE_URL", raising=False)
    client, tid, _ = _client(tmp_path, monkeypatch, gold=False)
    resp = client.post(
        "/rollout",
        json=to_dict(
            RolloutRequest(task_id=tid, suite_id="synth", policy_base_url="http://elsewhere")
        ),
    )
    assert resp.status_code == 409
    assert "policy_model" in resp.json()["error"]


def test_a_rollout_server_that_can_read_gold_refuses_to_run(tmp_path, monkeypatch):
    """The firewall, re-asserted at the endpoint. A rollout worker that can read gold makes
    every number it produces unusable, so this propagates rather than becoming a warning."""
    from pi_run.worker import FirewallError
    from pinq.wire import RolloutRequest

    client, tid, _ = _client(tmp_path, monkeypatch, gold=True)
    with pytest.raises(FirewallError, match="rollout worker"):
        client.post("/rollout", json=to_dict(RolloutRequest(task_id=tid, suite_id="synth")))
