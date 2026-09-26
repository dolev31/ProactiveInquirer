"""The HTTP surface: /rollout, /score, /retrieve, /healthz.

WHY THE HANDLERS ARE PLAIN FUNCTIONS AND THIS FILE IS THIN. Everything above is testable with
no server, no port and no `fastapi` installed — `handle_score` takes a dataclass and returns
a dataclass. This module only translates JSON to those calls and exceptions to status codes.
That split is what lets the split-refusal be tested as a unit rather than as an integration,
and a refusal that is only tested through a socket is a refusal nobody runs in CI.

`fastapi` is imported INSIDE `create_app()`. It lives in the `serve` extra, and the default
test run must not depend on an extra to import a module it does not start.

ONE PROCESS, ONE ROLE. A `/score` server needs `PI_GOLD_ROOT` set; a `/rollout` server needs
it unset. Hosting both in one process is therefore a configuration error, and `/healthz`
reports `gold_root_set` so it is a visible one. `/rollout` re-asserts the firewall on every
request anyway, because a guarantee with one enforcement point is one bug away from nothing.

THE STATUS CODES ARE PART OF THE CONTRACT. A trainer reads them without parsing prose:

    403  refused on policy — a split violation or a gold-exposed episode. Never retry.
    409  the request would move a frozen role (see rollout.check_base_url). Never retry.
    404  no such episode / no gold graph for that task. Never retry.
    503  this process cannot score at all: PI_GOLD_ROOT is unset. Fix the server, then retry.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pinq.wire import (
    Health,
    RetrieveRequest,
    RolloutRequest,
    ScoreRequest,
    from_dict,
    to_dict,
)

from .retrieve import RetrieverPool, SuiteNotConfigured, TaskNotSpecified, handle_retrieve
from .rollout import CorpusNotFound, FrozenRolesWouldMove, handle_rollout
from .score import EpisodeNotFound, NoGoldForTask, SplitRefused, handle_score

# 403, not 400: the request was well formed and is refused on policy. A reviewer reading a
# proxy log must be able to tell "the trainer asked for a test task" apart from "the trainer
# sent nonsense", because only one of those voids an experiment.
SPLIT_REFUSED_STATUS = 403
FROZEN_ROLES_STATUS = 409
NO_GOLD_ROOT_STATUS = 503


class ServerConfig:
    """Everything the endpoints need, resolved once at start-up.

    A config object rather than module globals: two servers (a rollout one and a scoring one)
    routinely run side by side on one machine with different roots, and module globals make
    that a source of cross-talk that only shows up under load.
    """

    def __init__(
        self,
        *,
        root: str | Path = ".",
        runs_root: str | Path = "runs",
        cache_root: str | Path = "cache",
        graph_version: str = "v1",
        default_suite: str | None = None,
        default_task: str | None = None,
        code_version: str = "",
        dirty: bool = True,
        dirty_files: tuple[str, ...] = (),
        allow_base_url_override: bool = False,
    ) -> None:
        self.root = Path(root)
        self.runs_root = Path(runs_root)
        self.cache_root = Path(cache_root)
        self.graph_version = graph_version
        self.code_version = code_version
        self.dirty = dirty
        self.dirty_files = tuple(dirty_files)
        self.allow_base_url_override = allow_base_url_override
        self.pool = RetrieverPool(root=root, default_suite=default_suite, default_task=default_task)
        self.default_suite = default_suite

    def health(self) -> Health:
        suites: tuple[str, ...] = ()
        corpora = self.root / "data" / "corpora"
        if corpora.exists():
            suites = tuple(sorted(d.name for d in corpora.iterdir() if d.is_dir()))
        return Health(
            ok=True,
            code_version=self.code_version,
            gold_root_set=bool(os.environ.get("PI_GOLD_ROOT")),
            suites=suites,
        )


def create_app(cfg: ServerConfig | None = None) -> Any:
    """Build the ASGI app. `fastapi` is required only here.

    THE HANDLERS TAKE `body: dict`, NOT `request: Request`, AND THAT IS LOAD-BEARING. This
    module has `from __future__ import annotations`, so every annotation is a STRING at
    runtime and FastAPI resolves it with `get_type_hints(func, func.__globals__)`. `Request`
    was imported inside this function, so it was in no namespace FastAPI could see: the
    annotation failed to resolve, FastAPI classified the parameter as a QUERY parameter, and
    every POST returned `422 {"loc": ["query", "request"], "msg": "Field required"}` -- the
    whole HTTP surface was unreachable. `/healthz` takes no parameters, so a health check
    passed while nothing else did, which is how it survived unnoticed.

    `dict` resolves from builtins under any import discipline, and a single non-scalar
    parameter is a JSON body to FastAPI, which is exactly what these endpoints want.
    tests/test_serve_http.py drives all four endpoints through TestClient so the next person
    to reach for `Request` finds out in CI rather than in a training run.
    """
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse

    conf = cfg or ServerConfig()
    app = FastAPI(title="pi_run.serve", version="1")

    @app.get("/healthz")
    def healthz() -> Any:
        return to_dict(conf.health())

    @app.post("/rollout")
    async def rollout(body: dict) -> Any:
        req = from_dict(RolloutRequest, body)
        try:
            resp = handle_rollout(
                req,
                root=conf.root,
                runs_root=conf.runs_root,
                cache_root=conf.cache_root,
                code_version=conf.code_version,
                dirty=conf.dirty,
                dirty_files=conf.dirty_files,
                allow_base_url_override=conf.allow_base_url_override,
            )
        except FrozenRolesWouldMove as exc:
            return JSONResponse(
                {"error": str(exc), "refused": True}, status_code=FROZEN_ROLES_STATUS
            )
        except CorpusNotFound as exc:
            return JSONResponse({"error": str(exc)}, status_code=404)
        return to_dict(resp)

    @app.post("/score")
    async def score(body: dict) -> Any:
        # Imported here, not at module scope: this is the one exception type that belongs to
        # the gold package, and a /rollout-only process must never load it.
        from pi_eval.gold import GoldAccessError

        req = from_dict(ScoreRequest, body)
        try:
            resp = handle_score(req, runs_root=conf.runs_root, graph_version=conf.graph_version)
        except SplitRefused as exc:
            return JSONResponse(
                {"error": str(exc), "refused": True, "split_assert": req.split_assert},
                status_code=SPLIT_REFUSED_STATUS,
            )
        except GoldAccessError as exc:
            # Not a 500: the request was fine and the SERVER is misconfigured. A trainer that
            # retried a 500 forever would look like a hang; this says what to fix.
            return JSONResponse({"error": str(exc)}, status_code=NO_GOLD_ROOT_STATUS)
        except (EpisodeNotFound, NoGoldForTask) as exc:
            return JSONResponse({"error": str(exc)}, status_code=404)
        return to_dict(resp)

    @app.post("/retrieve")
    async def retrieve(body: dict) -> Any:
        req = from_dict(RetrieveRequest, body)
        try:
            resp = handle_retrieve(req, pool=conf.pool)
        except (SuiteNotConfigured, TaskNotSpecified) as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        return to_dict(resp)

    return app


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - starts a server
    import argparse

    import uvicorn

    p = argparse.ArgumentParser(prog="pi-serve")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8077)
    p.add_argument("--root", default=".")
    p.add_argument("--runs-root", default="runs")
    p.add_argument("--cache-root", default="cache")
    p.add_argument("--graph-version", default="v1")
    p.add_argument("--suite", default=None, help="default suite for /retrieve")
    p.add_argument("--task", default=None, help="default task for /retrieve")
    p.add_argument(
        "--allow-base-url-override",
        action="store_true",
        help="permit policy_base_url, which re-routes EVERY role, not just the Inquirer",
    )
    a = p.parse_args(argv)
    # PROVENANCE FROM GIT, NOT HARDCODED. `ServerConfig` defaults to code_version="" and
    # dirty=True and this passed neither, so EVERY rollout served over HTTP carried no code
    # version at all and a `dev-` run_id -- which `pi agg` mechanically excludes from every
    # reported table, with no flag anywhere to change it. Derived here exactly as
    # `pi_run.manifest.build_manifest` derives it for `pi run`, so a served rollout is
    # identified by the same rule as a local one: dirty tree -> dev-, clean tree -> a real id.
    from pi_run.manifest import git_info

    g = git_info(a.root)
    cfg = ServerConfig(
        root=a.root,
        runs_root=a.runs_root,
        cache_root=a.cache_root,
        graph_version=a.graph_version,
        default_suite=a.suite,
        default_task=a.task,
        code_version=g.sha,
        dirty=g.dirty,
        dirty_files=g.dirty_files,
        allow_base_url_override=a.allow_base_url_override,
    )
    print(
        f"code_version={g.sha[:12]} dirty={g.dirty}" + (" (runs will be dev-)" if g.dirty else "")
    )
    if cfg.health().gold_root_set:
        print(
            "PI_GOLD_ROOT is SET: this process may score, and must NOT be used to serve "
            "/rollout. Run the rollout server from a shell where it is unset."
        )
    uvicorn.run(create_app(cfg), host=a.host, port=a.port)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
