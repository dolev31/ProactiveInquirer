"""POST /rollout — one episode, executed by a process that cannot read gold.

WHY THE POLICY ARRIVES AS A `base_url`. `RolloutRequest.policy_base_url` names an
OpenAI-compatible endpoint, and this module's only job is to make that endpoint the
Inquirer's pin for the duration of one rollout. A vLLM server holding a LoRA adapter, a
prompted open model and a frontier API are then literally the same code path, which is what
makes the trained / untrained / frontier families comparable at all. There is deliberately no
`if trained:` branch anywhere: an untested branch in the middle of the comparison the paper
is about would be the comparison's largest uncontrolled term.

HOW THE PIN IS APPLIED. `pi_run.worker.run_unit` builds its own `MeteredClient`, which reads
its role pins and its base URL from the environment — one env var per ROLE, so "the Answerer
is frozen across all arms" is enforced by construction. So this module sets exactly the
Inquirer's pin, runs the unit, and restores the previous environment. It does not reach into
the worker: the worker is the single place a rollout happens, and forking a second one here
for the trained family would mean the trained arm ran through code no other arm did.

WHY A BASE URL IS REFUSED BY DEFAULT AND A MODEL NAME IS NOT. There is ONE `LITELLM_BASE_URL`
per process and `MeteredClient` applies it to every role. Overriding it therefore re-routes
the frozen Drafter and the frozen Answerer along with the policy, which silently voids the
only invariant the trained-vs-prompted comparison has ("the sole difference is the Inquirer's
pin"). The supported way to reach a trained checkpoint is to register it as a MODEL NAME on
the proxy that already serves the frozen roles and send `policy_model`; then exactly one role
moves and `ModelPin.base_url_sha` stays equal across the families, as `pinq.types.ModelPin`
says it should. `allow_base_url_override=True` exists for the case where the endpoint really
does serve every pinned role, and it is a decision a human makes once, out loud, when
starting the server — not something a request can grant itself.

WHY A PROMPT OVERLAY. Rung 0 optimises the Inquirer PROMPT, and a candidate prompt has to
reach the policy through the same path every arm uses, or the winning prompt was selected
under a harness the confirmatory run does not share. `prompt_overlay` points
`pinq.promptlib` at a directory of candidate templates; the template's bytes are hashed into
`RunManifest.prompt_hashes` exactly as the shipped one would be, so a candidate rollout has a
different `run_id` and can never be confused with a baseline rollout.

THE FIREWALL IS RE-ASSERTED HERE. A `/score` server has `PI_GOLD_ROOT` set. If someone hosts
both endpoints in one process, this endpoint must still refuse to run, because a rollout
worker that can read gold makes every number it produces unusable. `assert_firewall()` is
called before anything else and is not catchable into a warning.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Mapping

from pinq.wire import RolloutRequest, RolloutResponse

# One env var per ROLE, mirroring pinq_adapters.llm.litellm_client.ROLE_ENV. Only the
# Inquirer's is ever overridden here: the Drafter and the frozen Answerer must stay pinned
# to whatever the experiment pinned them to, or "the only difference is the policy" stops
# being true.
INQUIRER_MODEL_ENV = "PI_MODEL_INQUIRER"
BASE_URL_ENV = "LITELLM_BASE_URL"
API_KEY_ENV = "LITELLM_API_KEY"
# Read by pinq.promptlib. A directory of candidate templates that shadows the shipped ones.
PROMPT_OVERLAY_ENV = "PI_PROMPT_OVERLAY"


class CorpusNotFound(FileNotFoundError):
    pass


class FrozenRolesWouldMove(RuntimeError):
    """A request asked to change the base URL, which moves every role, not just the policy."""


def resolve_corpus(root: str | Path, suite_id: str) -> Path:
    """Locate the frozen corpus for a suite.

    Ambiguity is an error rather than a guess: two corpus hashes are two different frozen
    corpora, and silently picking the newer one changes `corpus_hash` inside every run
    identity the server then produces.
    """
    from pi_run.worker import SELF_SOURCED

    base = Path(root) / "data" / "corpora" / suite_id
    if suite_id in SELF_SOURCED:
        return base  # tau2/pare/drgym source their own data; the adapter raises if it is absent
    cands = sorted(
        (d for d in base.glob("*") if (d / "tasks.jsonl").exists()), key=lambda d: d.name
    )
    if not cands:
        raise CorpusNotFound(f"no built corpus for suite {suite_id!r} under {base}")
    if len(cands) > 1:
        raise CorpusNotFound(f"{len(cands)} corpora under {base}: {[c.name for c in cands]}")
    return cands[0]


def check_base_url(req: RolloutRequest, *, allow_base_url_override: bool = False) -> None:
    """Refuse a base URL that would re-route the frozen roles. See the module docstring."""
    want = req.policy_base_url
    if not want or allow_base_url_override or want == os.environ.get(BASE_URL_ENV):
        return
    raise FrozenRolesWouldMove(
        f"policy_base_url={want!r} would also re-route the Drafter and the frozen Answerer: "
        f"{BASE_URL_ENV} is process-wide, not per role. Register the checkpoint as a model "
        "name on the proxy that already serves the frozen roles and send policy_model "
        "instead, or start the server with --allow-base-url-override if that endpoint really "
        "does serve every pinned role."
    )


@contextmanager
def policy_pin(req: RolloutRequest) -> Iterator[None]:
    """Point the Inquirer role at this request's endpoint, then put the environment back.

    Restoring matters because one server process serves many requests: a leaked pin would
    make the NEXT rollout run on the previous request's policy, and nothing downstream could
    detect it — the manifest would record the pin the worker actually used and it would be
    the wrong one for that request.
    """
    saved: dict[str, str | None] = {}

    def _set(key: str, value: str | None) -> None:
        saved.setdefault(key, os.environ.get(key))
        if value:
            os.environ[key] = value
        else:
            os.environ.pop(key, None)

    if req.policy_model:
        _set(INQUIRER_MODEL_ENV, req.policy_model)
    if req.policy_base_url:
        _set(BASE_URL_ENV, req.policy_base_url)
        # A local vLLM server accepts any key; the field exists so the same code path reaches
        # a gated frontier endpoint without a branch.
        if not os.environ.get(API_KEY_ENV):
            _set(API_KEY_ENV, "local")
    if req.prompt_overlay:
        _set(PROMPT_OVERLAY_ENV, req.prompt_overlay)
    try:
        yield
    finally:
        for key, old in saved.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old


def build_spec(
    req: RolloutRequest,
    *,
    root: str | Path,
    runs_root: str | Path,
    cache_root: str | Path,
    code_version: str = "",
    dirty: bool = True,
    dirty_files: tuple[str, ...] = (),
):
    """RolloutRequest -> UnitSpec. Plain data in, plain data out."""
    from pi_run.worker import UnitSpec

    return UnitSpec(
        suite_id=req.suite_id,
        corpus_dir=str(resolve_corpus(root, req.suite_id)),
        task_id=req.task_id,
        arm_id=req.arm_id,
        seed=req.seed,
        runs_root=str(runs_root),
        cache_root=str(cache_root),
        max_turns=req.max_turns,
        k=req.k,
        budget_cap=req.budget_cap,
        code_version=code_version,
        dirty=dirty,
        dirty_files=dirty_files,
        # Was DROPPED here. The field existed on RolloutRequest, was documented as passed
        # "through to an OpenAI-compatible endpoint verbatim", and reached nothing: a caller
        # asking for temperature 1.0 silently got greedy decoding and no error.
        sampling=dict(req.sampling or {}),
    )


def handle_rollout(
    req: RolloutRequest,
    *,
    root: str | Path = ".",
    runs_root: str | Path = "runs",
    cache_root: str | Path = "cache",
    code_version: str = "",
    dirty: bool = True,
    dirty_files: tuple[str, ...] = (),
    allow_base_url_override: bool = False,
) -> RolloutResponse:
    from pi_run.worker import assert_firewall, run_unit

    assert_firewall()  # a rollout server with PI_GOLD_ROOT set is a breach, not a warning
    check_base_url(req, allow_base_url_override=allow_base_url_override)
    spec = build_spec(
        req,
        root=root,
        runs_root=runs_root,
        cache_root=cache_root,
        code_version=code_version,
        dirty=dirty,
        dirty_files=dirty_files,
    )
    with policy_pin(req):
        res: Mapping[str, object] = run_unit(spec)
    spent: Mapping[str, object] = res.get("spent") or {}  # type: ignore[assignment]
    usage: Mapping[str, object] = res.get("usage") or {}  # type: ignore[assignment]
    return RolloutResponse(
        run_id=str(res.get("run_id", "")),
        status=str(res.get("status", "")),
        suite_id=req.suite_id,
        task_id=req.task_id,
        arm_id=req.arm_id,
        seed=req.seed,
        n_turns=int(res.get("n_turns", 0) or 0),
        n_asks=int(res.get("n_asks", 0) or 0),
        stop_reason=str(res.get("stop_reason", "")),
        retrieval_calls=float(spent.get("retrieval_calls", 0.0) or 0.0),  # type: ignore[arg-type]
        tok_total=int(usage.get("tok_total", 0) or 0),  # type: ignore[arg-type]
        usd=float(usage.get("usd", 0.0) or 0.0),  # type: ignore[arg-type]
        wall_ms=int(res.get("wall_ms", 0) or 0),
        run_dir=str(Path(runs_root) / str(res.get("run_id", ""))),
        error=str(res.get("error", "") or ""),
    )
