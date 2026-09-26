"""The trainer's side of the seam: HTTP, stdlib only, no shared memory with the scorer.

WHY urllib AND NOT httpx. `pinq_train` has to be importable in a training container whose
dependency set is dictated by torch and vllm pins. Every dependency this package adds is a
dependency that can conflict with those, so the client that talks to the scorer is 80 lines
of `urllib.request` rather than a library.

WHY THE CLIENT REFUSES TO START IN A PROCESS THAT CAN READ GOLD. The whole point of the seam
is that the trainer asks a server for a number instead of computing it. A trainer that had
`PI_GOLD_ROOT` set could compute it — and then the seam is decoration and nothing about the
firewall is true any more. `assert_trainer_process()` runs on construction; it is the mirror
image of `pi_run.worker.assert_firewall`, which refuses the opposite condition.

WHY 403 IS NOT AN ERROR TO RETRY. The server refuses on policy: a task outside `train`, or a
gold-exposed episode. Retrying it is meaningless and burying it in a generic exception is
worse, so it gets its own class with the payload attached, and a training loop that catches
`ServerRefused` and continues has to say so out loud.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from pinq.wire import (
    Health,
    RetrievedDoc,
    RetrieveRequest,
    RetrieveResponse,
    RolloutRequest,
    RolloutResponse,
    ScoreRequest,
    ScoreResponse,
    TurnScore,
    from_dict,
    to_dict,
)

# The status codes pi_run.serve.app promises. Duplicated deliberately: the trainer and the
# server are separate processes that must not share an import, and tests/test_serve.py pins
# these against the server's own constants.
SPLIT_REFUSED_STATUS = 403
FROZEN_ROLES_STATUS = 409
NO_GOLD_ROOT_STATUS = 503


class TrainerFirewallError(RuntimeError):
    """PI_GOLD_ROOT is set in a trainer process. The seam is not advisory."""


class ServerRefused(RuntimeError):
    """The server declined on policy (403/409). Never retry; fix the request."""

    def __init__(self, status: int, payload: Mapping[str, Any], url: str) -> None:
        super().__init__(f"{url} refused with {status}: {payload.get('error', payload)}")
        self.status = status
        self.payload = dict(payload)


class ServerUnavailable(RuntimeError):
    """The server is misconfigured or down (503, connection refused). Retrying may help."""


def assert_trainer_process(env: Mapping[str, str] | None = None) -> None:
    e = env if env is not None else os.environ
    if e.get("PI_GOLD_ROOT"):
        raise TrainerFirewallError(
            "PI_GOLD_ROOT is set in this process. A trainer that can read gold does not need "
            "the seam, and every number it produces afterwards is unfalsifiable. Unset it and "
            "point the client at a scoring server instead."
        )


# MEASURED, not guessed. Over 5,641 recorded musique rollouts the wall time was median 64s,
# p90 223s, p99 379s, max 717s -- so the previous 120s default sat between the median and the
# p90 and abandoned roughly a quarter of rollouts. The server kept working and finished; only
# the client stopped waiting, so the work was paid for and thrown away.
#
# A CLIENT PATIENCE SETTING, NOT A BUDGET. `unit_timeout_s` is what stops a runaway episode
# server-side; this only decides how long to wait for one that is still going.
DEFAULT_SEAM_TIMEOUT_S = 900.0


def _timeout_from(env: Mapping[str, str] | None = None) -> float:
    """`PI_SEAM_TIMEOUT_S`, because a slow proxy is an operational condition rather than a
    code change. Garbage falls back rather than taking a search down over a typo."""
    e = env if env is not None else os.environ
    raw = str(e.get("PI_SEAM_TIMEOUT_S", "")).strip()
    if not raw:
        return DEFAULT_SEAM_TIMEOUT_S
    try:
        return max(1.0, float(raw))
    except (TypeError, ValueError):
        return DEFAULT_SEAM_TIMEOUT_S


@dataclass(frozen=True, slots=True)
class Endpoint:
    base_url: str = "http://127.0.0.1:8077"
    timeout: float = DEFAULT_SEAM_TIMEOUT_S

    def url(self, path: str) -> str:
        return f"{self.base_url.rstrip('/')}{path}"


class SeamClient:
    """One client for all four endpoints, across ONE OR TWO servers.

    THE FIREWALL MAKES A SINGLE PROCESS IMPOSSIBLE, and both sides say so. `serve/rollout.py`:
    "If someone hosts both endpoints in one process, this endpoint must still refuse to run,
    because a rollout worker that can read gold makes every number it produces unusable" -- so
    a gold-set server REFUSES /rollout. And /score without gold returns 503 by design.

    So a scoring server cannot roll out and a rollout server cannot score, while this client
    posted all four endpoints to one `base_url`: whichever way it was pointed, half its calls
    failed. Rung 0 needs both, which is consistent with rung 0 never having run -- all 5,291
    recorded runs carry `prompt_variant_id = "v1"`.

    `rollout_url` defaults to `base_url`, so a single-server caller is unchanged and only a
    caller that needs the split has to know it exists.

    THE SPLIT FOLLOWS WHAT EACH ENDPOINT TOUCHES, not convenience: /score and /healthz go to
    the SCORING server (healthz is how a caller reads `gold_root_set`, a property of that
    server), while /rollout and /retrieve go to the ROLLOUT server -- retrieval touches the
    corpus, and routing it to the gold-bearing process would ask that process to serve corpus
    text.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8077",
        *,
        rollout_url: str | None = None,
        timeout: float | None = None,
    ) -> None:
        assert_trainer_process()
        timeout = _timeout_from() if timeout is None else float(timeout)
        self.ep = Endpoint(base_url=base_url, timeout=timeout)
        self.rollout_ep = Endpoint(base_url=rollout_url or base_url, timeout=timeout)

    # ------------------------------------------------------------------ transport

    def _request(
        self, path: str, payload: Mapping[str, Any] | None, *, ep: Endpoint | None = None
    ) -> dict[str, Any]:
        endpoint = ep or self.ep
        url = endpoint.url(path)
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="GET" if data is None else "POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=endpoint.timeout) as fh:  # noqa: S310
                return json.loads(fh.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")
            try:
                parsed = json.loads(body)
            except json.JSONDecodeError:
                parsed = {"error": body}
            if exc.code in (SPLIT_REFUSED_STATUS, FROZEN_ROLES_STATUS):
                raise ServerRefused(exc.code, parsed, url) from None
            if exc.code == NO_GOLD_ROOT_STATUS:
                raise ServerUnavailable(f"{url}: {parsed.get('error', body)}") from None
            raise RuntimeError(f"{url} -> {exc.code}: {parsed.get('error', body)}") from None
        except urllib.error.URLError as exc:
            raise ServerUnavailable(f"{url} is not reachable: {exc.reason}") from None

    # ------------------------------------------------------------------ endpoints

    def healthz(self) -> Health:
        return from_dict(Health, self._request("/healthz", None))

    def score(self, episode_id: str, *, mode: str = "prefix_ladder") -> ScoreResponse:
        req = ScoreRequest(episode_id=episode_id, mode=mode)  # type: ignore[arg-type]
        return score_from_payload(self._request("/score", to_dict(req)))

    def rollout(self, req: RolloutRequest) -> RolloutResponse:
        return from_dict(
            RolloutResponse, self._request("/rollout", to_dict(req), ep=self.rollout_ep)
        )

    def retrieve(
        self,
        queries: Sequence[str],
        *,
        topk: int = 5,
        suite_id: str | None = None,
        task_id: str | None = None,
        return_scores: bool = True,
    ) -> RetrieveResponse:
        req = RetrieveRequest(
            queries=tuple(queries),
            topk=topk,
            return_scores=return_scores,
            suite_id=suite_id,
            task_id=task_id,
        )
        return retrieve_from_payload(self._request("/retrieve", to_dict(req), ep=self.rollout_ep))


# --------------------------------------------------------------------------- decoding


def score_from_payload(payload: Mapping[str, Any]) -> ScoreResponse:
    """`from_dict` is one level deep on purpose (see pinq.wire); the nesting is decoded here.

    Tuples, not lists: `ScoreResponse` is frozen and is used as a dict key in the candidate
    tables, and a list field would make it unhashable at exactly the wrong moment.
    """
    turns = tuple(from_dict(TurnScore, t) for t in payload.get("turns", ()))
    resp = from_dict(ScoreResponse, payload)
    return type(resp)(
        **{
            **{k: v for k, v in vars_of(resp).items()},
            "turns": turns,
            "q_ladder": tuple(payload.get("q_ladder", ())),
            "potential": tuple(payload.get("potential", ())),
        }
    )


def retrieve_from_payload(payload: Mapping[str, Any]) -> RetrieveResponse:
    return RetrieveResponse(
        result=tuple(
            tuple(from_dict(RetrievedDoc, d) for d in batch) for batch in payload.get("result", ())
        )
    )


def vars_of(obj: Any) -> dict[str, Any]:
    """`vars()` does not work on a slotted dataclass; this does, without importing dataclasses
    machinery at every call site."""
    from dataclasses import fields

    return {f.name: getattr(obj, f.name) for f in fields(obj)}
