"""`pi suites` --- the map from benchmark to role, and a readiness check that actually runs.

THE QUESTION THIS ANSWERS. "Which data trained this, and which data measured it." Until this
command existed the answer required reading `worker.load_suite`, `pinq.splitting`,
`conf/grids/*.yaml` and `pi_eval.prereg` and hoping they agreed, and the first time anyone
checked, they did not: wiki2 was believed to carry no endpoint while sitting in the pool of
six declared secondary tests.

WHY `validate` CONSTRUCTS THINGS INSTEAD OF READING A MANIFEST. `pi data status` already
reports what is on disk, and on-disk-ness is not readiness: a corpus can be present and the
adapter still raise, a gold graph can be present and still describe a different corpus hash
than the one the adapter loads. Both of those are silent, and both produce a table full of
zeros rather than an error. So this walks the same path a rollout walks --- construct the
suite, take a view, run one retrieval --- and reports where it stops.

IT SPENDS NOTHING. No LLM client is constructed anywhere in this file. A suite whose only
failure is a missing API key is reported as `needs-key`, not as broken, because those are
different problems with different fixes.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from pi_run.suites import REGISTRY, audit, audit_against_prereg, evaluation_suites, training_sources


def _repo_root(a: argparse.Namespace) -> Path:
    from pi_run.manifest import repo_root

    return Path(getattr(a, "root", None) or repo_root())


def _loadable() -> frozenset[str]:
    from pi_run.worker import CORPUS_BACKED, SELF_SOURCED

    return frozenset(CORPUS_BACKED) | frozenset(SELF_SOURCED)


def _corpus_dir(root: Path, suite_id: str) -> Path | None:
    """The newest content-addressed corpus directory for a suite, or None.

    Content-addressed means the directory NAME is the hash of its contents, so picking the
    newest is picking a specific corpus rather than an arbitrary one -- and `validate` then
    reports which hash it picked, so a mismatch against gold is visible instead of inferred.
    """
    d = root / "data" / "corpora" / suite_id
    if not d.is_dir():
        return None
    subs = sorted((p for p in d.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime)
    return subs[-1] if subs else None


# --------------------------------------------------------------------------------- pi suites map


def cmd_suites_map(a: argparse.Namespace) -> int:
    """Print the two lists the plan turns on: what is mined, and what is measured."""
    mined, measured = training_sources(), evaluation_suites()

    rows = []
    for sid, s in REGISTRY.items():
        rows.append(
            {
                "suite": sid,
                "status": s.status,
                "source": s.source,
                "mines_training_data": s.mines_training_data,
                "carries_endpoints": s.carries_endpoints,
                "endpoint_status": s.endpoint_status,
                "eval_only": s.eval_only,
                "axis": list(s.axis),
                "required_env": list(s.required_env),
            }
        )
    if getattr(a, "json", False):
        print(
            json.dumps({"suites": rows, "mined": list(mined), "measured": list(measured)}, indent=2)
        )
        return 0

    print(f"{'suite':13s} {'stat':7s} {'src':7s} {'MINE':5s} {'EVAL':5s} {'endpoint':13s} axis")
    print("-" * 92)
    for r in rows:
        mine = "yes" if r["mines_training_data"] else "-"
        ev = "yes" if r["carries_endpoints"] else "-"
        if r["eval_only"]:
            ev = "ONLY"
        print(
            f"{r['suite']:13s} {r['status']:7s} {r['source']:7s} {mine:5s} {ev:5s} "
            f"{r['endpoint_status']:13s} {','.join(r['axis'])}"
        )
    print("-" * 92)
    print(f"TRAINING DATA is mined from : {', '.join(mined)}")
    print(f"ENDPOINTS are measured on   : {', '.join(measured)}")
    print()
    print("MINE=yes means the suite's `train` split is exported into data/rl/.")
    print("EVAL=ONLY means it is in EVAL_ONLY_SUITES: split_of() returns 'test' for every")
    print("  task, so it can never contribute a training row and its number is transfer.")
    print("endpoint=exploratory means the suite is NOT named in pi_eval.prereg and its")
    print("  numbers may carry a CI but never a p-value.")
    return 0


# ------------------------------------------------------------------------------- pi suites audit


def _confirmatory_suites_for(a: argparse.Namespace) -> tuple[frozenset[str], str]:
    """(the suites that may carry a p-value, and which document said so).

    RESOLVED, NOT ASSUMED. The registry describes the programme the paper is running, and
    `pi_eval.prereg`'s module-level endpoints are the PROMPTED era's -- the trained programme
    seals its own stage file. So the audit takes the document: `--stage1` if named, else a
    SEALED `prereg/stage1.json` if one exists, else the module's declarations, and it prints
    which. A named file that does not exist is an error, never a fallback: checking a
    different programme and printing `clean` about it is the failure this whole command is
    for.
    """
    from pi_eval.prereg import confirmatory_suites

    named = getattr(a, "stage1", None)
    if named:
        p = Path(named)
        if not p.exists():
            raise FileNotFoundError(p)
        return confirmatory_suites(json.loads(p.read_text())), str(p)
    sealed = _repo_root(a) / "prereg" / "stage1.json"
    if sealed.exists():
        return confirmatory_suites(json.loads(sealed.read_text())), str(sealed)
    return confirmatory_suites(), "pi_eval.prereg (no sealed stage 1)"


def cmd_suites_audit(a: argparse.Namespace) -> int:
    """Every disagreement between the registry and the code it describes. Exit 1 on drift."""
    # The scorer's gold-free set, fetched here rather than imported by `pi_run.suites`: that
    # module has to stay importable in a rollout worker, where PI_GOLD_ROOT is unset by design.
    # A failure to reach it is reported, never swallowed -- the same treatment the
    # preregistration cross-check gets below, and for the same reason.
    gold_free: frozenset[str] | None = None
    gold_free_problem: str | None = None
    try:
        from pi_eval.score import GOLD_FREE_SUITES

        gold_free = GOLD_FREE_SUITES
    except Exception as exc:  # noqa: BLE001
        gold_free_problem = f"could not cross-check the scorer's gold-free suites: {exc!r}"

    problems = audit(_loadable(), gold_free)
    if gold_free_problem:
        problems.append(gold_free_problem)

    source = "pi_eval.prereg"
    try:
        named, source = _confirmatory_suites_for(a)
        problems += audit_against_prereg(named)
    except FileNotFoundError as exc:
        print(f"registry audit: the stage 1 you named does not exist: {exc}")
        return 2
    except Exception as exc:  # noqa: BLE001 -- reported, never swallowed
        problems.append(f"could not cross-check the preregistration: {exc!r}")

    if not problems:
        print(f"registry audit: clean (endpoint roles checked against {source})")
        return 0
    print(f"registry audit: {len(problems)} problem(s) (endpoint roles checked against {source})")
    for p in problems:
        print(f"  - {p}")
    if source.startswith("pi_eval.prereg"):
        print(
            "\nNo stage 1 is sealed, so the endpoint roles were checked against "
            "`pi_eval.prereg`'s own declarations -- the PROMPTED programme. The registry "
            "describes the programme the paper is RUNNING. If those are different documents, "
            "`pi suites audit --stage1 <path>` checks against the one that governs; sealing "
            "that stage 1 makes it the default. A role disagreement here is a real one and "
            "not a plumbing failure: it says a suite may carry a p-value that no sealed "
            "document has authorised, or the reverse."
        )
    return 1


# ---------------------------------------------------------------------------- pi suites validate


def _validate_one(root: Path, suite_id: str, *, k: int, graph_version: str) -> dict[str, Any]:
    """Walk a rollout's path far enough to know the suite is real, without spending anything."""
    spec = REGISTRY[suite_id]
    r: dict[str, Any] = {
        "suite": suite_id,
        "status": spec.status,
        "adapter": None,
        "n_tasks": None,
        "view": None,
        "retriever": None,
        "n_retrieved": None,
        "corpus_hash": None,
        "gold_graphs": None,
        # Only a gold_free suite fills this; None elsewhere keeps the --json shape stable.
        "gold_answers": None,
        "gold_nodes": None,
        "gold_required": None,
        "gold_gaps_expected": None,
        "recorded_runs": None,
        "corpus_hash_matches_gold": None,
        "split": None,
        "n_templated": None,
        "latent_share": None,
        "blockers": [],
    }

    missing = [v for v in spec.required_env if not os.environ.get(v)]
    if missing:
        r["blockers"].append(f"needs-env: {', '.join(missing)}")

    if suite_id not in _loadable():
        r["blockers"].append("not constructible via worker.load_suite (replay-only or planned)")
        return r

    cdir = _corpus_dir(root, suite_id)
    if cdir is not None:
        r["corpus_hash"] = cdir.name

    # --- the adapter ------------------------------------------------------------------
    try:
        from pi_run.worker import load_suite

        suite = load_suite(suite_id, str(cdir) if cdir else "")
        tids = tuple(suite.task_ids())
        r["adapter"], r["n_tasks"] = "ok", len(tids)
    except Exception as exc:  # noqa: BLE001
        r["adapter"] = f"FAIL: {type(exc).__name__}: {exc}"
        r["blockers"].append("adapter does not construct")
        return r

    if not tids:
        r["blockers"].append("adapter constructs but yields zero tasks")
        return r

    # --- a view, which is where a leak would surface -----------------------------------
    #
    # For an orchestrator-driven suite the assertion is INVERTED. `view()` must refuse,
    # because the only text it could return is the customer's roleplay script; a view that
    # succeeds there has handed the agent the user-private partition and every ceiling
    # number computed afterwards is meaningless. So a raise is the pass and a success is the
    # blocker -- the same shape as `GoldAccessError` being the firewall working.
    orchestrated = spec.driven_by == "orchestrator"
    # A GYM-DRIVEN SUITE IS THE OTHER HALF OF THE SAME IDEA, AND IT SPLITS THE TWO LEGS.
    # `userbench` HAS an honest view -- upstream's `initial_description`, the user's
    # incomplete opening -- so the view leg is checked normally and a refusal there would be
    # a real failure. What it does not have is a free retrieval: its `[search]` is an action
    # inside TravelEnv scored by an LLM judge, so `retriever()` refuses by design and this
    # file must not construct one, both because a refusal is not a break-in and because this
    # command is documented to spend nothing.
    priced_retrieval = spec.driven_by in ("orchestrator", "gym")
    try:
        v = suite.view(tids[0])
        if orchestrated:
            r["view"] = "FAIL: view() returned instead of refusing"
            r["blockers"].append(
                "view() SUCCEEDED on an orchestrator-driven suite -- the roleplay script "
                "reached the agent and the user-private partition is void"
            )
        else:
            r["view"] = "ok" if getattr(v, "question", None) else "ok (no question text)"
    except Exception as exc:  # noqa: BLE001
        if orchestrated and type(exc).__name__ == "Tau2NeedsOrchestrator":
            r["view"] = "refuses correctly (Tau2NeedsOrchestrator)"
        else:
            r["view"] = f"FAIL: {type(exc).__name__}: {exc}"
            r["blockers"].append("view() raises")

    # --- one retrieval -----------------------------------------------------------------
    if priced_retrieval:
        # Reached only through a driver that talks to a paid user simulator -- tau2's
        # Orchestrator, or `UserBenchSuite.session()`. Out of scope for a check that must
        # cost nothing; `pi verify tau2 --replay-gold` is the free instrument that exercises
        # tau2 for real.
        r["retriever"] = (
            "n/a (orchestrator-driven; see `pi verify tau2`)"
            if orchestrated
            else "n/a (gym-driven; retrieval is the env's own scored [search] action)"
        )
    else:
        try:
            q = getattr(suite.view(tids[0]), "question", "") or "test query"
            units = suite.retriever(tids[0]).search(q, k)
            r["retriever"], r["n_retrieved"] = "ok", len(tuple(units))
            if r["n_retrieved"] == 0:
                r["blockers"].append("retriever returns nothing for the task's own question")
        except Exception as exc:  # noqa: BLE001
            name = type(exc).__name__
            if name == "SearchCacheMiss" and spec.retrieval_is_replay:
                # NOT A BLOCKER, and calling it one was wrong. The cache holds the sub-queries
                # the recorded runs issued, never the raw task question this probe sends, so a
                # miss here is expected even when the suite is fully exercised. `validate`
                # reported drgym unrunnable on this basis while 342 scored runs sat on disk.
                r["retriever"] = "replay cache (probe query uncached, which is expected)"
            elif name == "SearchCacheMiss":
                r["retriever"] = "cache-miss (offline replay; needs PI_DRGYM_ONLINE=1 to record)"
                r["blockers"].append("search cache does not cover this query")
            else:
                r["retriever"] = f"FAIL: {name}: {exc}"
                r["blockers"].append("retriever raises")

    # --- for a replay suite, the real readiness question: are there recorded runs? -------
    if spec.retrieval_is_replay:
        try:
            import json as _json

            n_runs = 0
            for m in (root / "runs").glob("*/manifest.json"):
                try:
                    if _json.loads(m.read_text()).get("suite_id") == suite_id:
                        n_runs += 1
                except Exception:  # noqa: BLE001 -- a half-written manifest is not this check
                    continue
            r["recorded_runs"] = n_runs
            if n_runs == 0:
                r["blockers"].append(
                    "replay suite with no recorded runs: nothing to replay, and the cache "
                    "cannot be rebuilt without a deliberate online pass"
                )
        except Exception as exc:  # noqa: BLE001
            r["blockers"].append(f"could not count recorded runs: {exc!r}")

    # --- the split, over every task ----------------------------------------------------
    try:
        from pi_run.manifest import template_id_of
        from pinq.splitting import split_of

        # THE TEMPLATE ID IS NOT OPTIONAL HERE. `split_of` hashes on it where a suite has
        # one, so omitting it reports a DIFFERENT split than the runner will actually use --
        # measured on musique, 461/125/214 without it against 456/132/212 with it. A
        # readiness check that disagrees with the runner about which tasks are trainable is
        # worse than no readiness check.
        counts = {"train": 0, "dev": 0, "test": 0}
        n_templated = 0
        for t in tids:
            tpl = template_id_of(suite, str(t))
            n_templated += tpl is not None
            counts[split_of(suite_id, str(t), tpl)] += 1
        r["split"] = counts
        r["n_templated"] = n_templated
        if REGISTRY[suite_id].mines_training_data and counts["train"] == 0:
            r["blockers"].append("declared a training source but has no train tasks")
        if REGISTRY[suite_id].carries_endpoints and counts["test"] == 0:
            r["blockers"].append("declared to carry an endpoint but has no test tasks")
    except Exception as exc:  # noqa: BLE001
        r["blockers"].append(f"split failed: {exc!r}")

    # --- gold, and whether it describes THIS corpus -------------------------------------
    #
    # "NO NEED-GRAPH" AND "NO GOLD" ARE DIFFERENT, AND THIS USED TO PRINT THE SECOND FOR BOTH.
    # userbench genuinely has no gold file: it is scored inside its own gym against the task's
    # option ids. frames has no need-graph and DOES have gold -- one answer per task, which is
    # the only thing `pi score` measures it on. Reporting "n/a" there means nobody notices
    # when the answer file is missing or describes a different corpus, and a stale pairing
    # mis-scores every answer metric the suite has. `pi score` refuses that mismatch, but it
    # refuses after a campaign has been paid for. This is the same check, beforehand.
    if not spec.uses_gold_graph and not spec.gold_free:
        r["gold_graphs"] = "n/a (scored inside its own environment, not against a need-graph)"
        return r
    if spec.gold_free:
        try:
            from pi_eval.gold import load_graphs

            graphs = load_graphs(suite_id, graph_version)
            answered = [g for g in graphs.values() if g.gold_answer]
            r["gold_graphs"] = f"n/a (gold-free: {len(graphs)} answers, no need-graph)"
            r["gold_answers"] = len(answered)
            missing = sorted(set(tids) - {g.gold_task_key for g in answered})
            if missing:
                r["blockers"].append(
                    f"{len(missing)} task(s) have no gold answer, so they would score a silent "
                    f"zero on the only metrics this suite has: {missing[:5]}"
                )
            hashes = {g.gold_corpus_hash for g in graphs.values() if g.gold_corpus_hash}
            if r["corpus_hash"] and hashes and r["corpus_hash"] not in hashes:
                r["blockers"].append(
                    f"gold describes corpus {sorted(hashes)} but the adapter loaded "
                    f"{r['corpus_hash']} -- every answer would be scored against another build"
                )
            # A gold_free suite that ships NODES is a registry lie, and the scorer would then
            # be suppressing structure that actually exists.
            with_nodes = sorted(g.gold_task_key for g in graphs.values() if g.gold_nodes)
            if with_nodes:
                r["blockers"].append(
                    f"declared gold_free but {len(with_nodes)} graph(s) carry nodes: "
                    f"{with_nodes[:5]} -- the scorer is suppressing real structure"
                )
        except Exception as exc:  # noqa: BLE001
            r["gold_graphs"] = f"FAIL: {type(exc).__name__}: {exc}"
            r["blockers"].append(
                "gold answers do not load (PI_GOLD_ROOT unset is expected in a worker)"
            )
        return r
    try:
        from pi_eval.gold import load_graphs

        # load_graphs returns a dict keyed by gold_task_key, and a graph's nodes live on
        # `gold_nodes` -- every gold field carries the prefix, which is the type wall doing
        # its job: a name that cannot collide with a TaskView field cannot be smuggled.
        graphs = load_graphs(suite_id, graph_version)
        r["gold_graphs"] = len(graphs)
        nodes = [n for g in graphs.values() for n in g.gold_nodes]
        r["gold_nodes"] = len(nodes)

        # THE LATENT SHARE, over REQUIRED nodes only. Optional nodes are the 2,269 that carry
        # no evidence span, and counting them would let a suite look deep because its
        # annotation was incomplete.
        req = [n for n in nodes if getattr(n, "gold_partition", None) == "required"]
        deep = [n for n in req if (getattr(n, "gold_depth", None) or 0) >= 1]
        r["latent_share"] = round(len(deep) / len(req), 4) if req else None
        r["gold_required"] = len(req)

        hashes = {
            getattr(g, "gold_corpus_hash", None)
            for g in graphs.values()
            if getattr(g, "gold_corpus_hash", None)
        }
        expected_gaps = len(spec.gold_gap_tasks)
        if isinstance(r["n_tasks"], int):
            unexplained = (r["n_tasks"] - r["gold_graphs"]) - expected_gaps
            r["gold_gaps_expected"] = expected_gaps
            if unexplained > 0:
                r["blockers"].append(
                    f"{unexplained} task(s) have no gold graph beyond the "
                    f"{expected_gaps} documented -- they would score as a silent miss, "
                    f"not as an error"
                )
            elif unexplained < 0:
                # More gold than tasks, or a documented gap that filled itself in. Either way
                # the declaration no longer describes the data and should not be trusted.
                r["blockers"].append(
                    f"gold covers more tasks than the {expected_gaps} documented gaps allow "
                    f"for; the gold_gap_tasks declaration is stale"
                )
        if r["corpus_hash"] and hashes:
            r["corpus_hash_matches_gold"] = r["corpus_hash"] in hashes
            if not r["corpus_hash_matches_gold"]:
                r["blockers"].append(
                    f"gold describes corpus {sorted(hashes)} but the adapter loaded "
                    f"{r['corpus_hash']} -- every gold uid would silently miss"
                )
    except Exception as exc:  # noqa: BLE001
        r["gold_graphs"] = f"FAIL: {type(exc).__name__}: {exc}"
        r["blockers"].append("gold does not load (PI_GOLD_ROOT unset is expected in a worker)")

    return r


def cmd_suites_validate(a: argparse.Namespace) -> int:
    root = _repo_root(a)
    want = (
        [a.suite]
        if getattr(a, "suite", None)
        else [s for s in REGISTRY if REGISTRY[s].status == "live"]
    )

    results = [_validate_one(root, s, k=a.k, graph_version=a.graph_version) for s in want]

    if getattr(a, "json", False):
        print(json.dumps(results, indent=2, default=str))
    else:
        print(
            f"{'suite':13s} {'tasks':>7s} {'gold':>7s} {'latent':>7s} {'train/dev/test':>18s}  blockers"
        )
        print("-" * 100)
        for r in results:
            sp = r["split"] or {}
            spl = f"{sp.get('train', '-')}/{sp.get('dev', '-')}/{sp.get('test', '-')}"
            lat = f"{r['latent_share']:.1%}" if isinstance(r["latent_share"], float) else "-"
            g = r["gold_graphs"]
            # "FAIL" for anything non-integer was wrong for a suite that has no gold by
            # design: userbench is scored inside its own gym and reported FAIL forever.
            gold = (
                g
                if isinstance(g, int)
                else ("n/a" if isinstance(g, str) and g.startswith("n/a") else "FAIL")
            )
            n = r["n_tasks"] if r["n_tasks"] is not None else "-"
            print(
                f"{r['suite']:13s} {str(n):>7s} {str(gold):>7s} {lat:>7s} {spl:>18s}  "
                f"{'; '.join(r['blockers']) if r['blockers'] else 'ready'}"
            )
        print("-" * 100)
        bad = [r for r in results if r["blockers"]]
        print(f"{len(results) - len(bad)} of {len(results)} suites ready")

    return 1 if any(r["blockers"] for r in results) else 0


def register(sub: argparse._SubParsersAction) -> None:
    """Attach `pi suites` to the top-level dispatch table."""
    p = sub.add_parser("suites", help="what each benchmark is FOR, and whether it is ready")
    s = p.add_subparsers(dest="sub", required=True)

    m = s.add_parser("map", help="which suites are mined for training, which carry endpoints")
    m.add_argument("--json", action="store_true")
    m.set_defaults(fn=cmd_suites_map)

    au = s.add_parser("audit", help="disagreements between the registry, the loader and the prereg")
    au.add_argument("--root")
    au.add_argument(
        "--stage1",
        help="stage-1 payload whose endpoint roles the registry is checked against "
        "(default: a sealed prereg/stage1.json, else pi_eval.prereg's own declarations)",
    )
    au.set_defaults(fn=cmd_suites_audit)

    v = s.add_parser("validate", help="construct each adapter, take a view, retrieve once")
    v.add_argument("--suite", choices=sorted(REGISTRY))
    v.add_argument("--root")
    v.add_argument("--k", type=int, default=5, help="retrieval depth for the smoke retrieval")
    v.add_argument("--graph-version", default="v1")
    v.add_argument("--json", action="store_true")
    v.set_defaults(fn=cmd_suites_validate)

    el = s.add_parser(
        "eligibility", help="which split the runs feeding the paper's tables come from"
    )
    el.add_argument("--root")
    el.add_argument("--parquet", help="directory holding runs.parquet")
    el.add_argument("--json", action="store_true")
    el.set_defaults(fn=cmd_suites_eligibility)

    rg = s.add_parser("range", help="does each metric have room to move, and is it measured")
    rg.add_argument("--root")
    rg.add_argument("--parquet", help="directory holding runs.parquet and scores.parquet")
    rg.add_argument(
        "--scorer-hash",
        required=True,
        help="the ONE scorer_hash to read; a metric's range is a property of its scorer",
    )
    rg.add_argument(
        "--metric",
        action="append",
        help="restrict to this metric name (repeatable); overrides the family filter",
    )
    rg.add_argument("--json", action="store_true")
    rg.set_defaults(fn=cmd_suites_range)


# ------------------------------------------------------------------------- pi suites eligibility

# The predicate `pi_eval.report` uses to decide which runs may reach a table, minus the two
# clauses that need columns a plain parquet read does not carry. Duplicated deliberately and
# narrowly: this command must be able to say what the report WOULD include without importing
# the report's machinery, and `test_suites_eligibility.py` pins the shared clauses equal.
#
# IN TWO PIECES, AND THE SPLIT IS THE POINT. `_ELIGIBLE_CORE_ANY_SPLIT` is the population that
# WAS eligible before 2026-09-15, when `ELIGIBLE` gained the held-out clause; `_ELIGIBLE_CORE`
# is what is eligible now. Filtered on `split = 'test'` alone this command could only ever
# print zeros under train and dev and report "clean" -- true, and useless, because the number
# the decision record needs is what the clause REMOVES. Both strings exist; the report below is
# the difference between them.
_HELD_OUT = "r.split = 'test'"
_ELIGIBLE_CORE_ANY_SPLIT = (
    "r.status = 'ok' AND r.gold_exposed = FALSE AND r.is_dev_run = FALSE "
    "AND r.dirty = FALSE AND r.pilot_flag = FALSE AND r.canary_hit = FALSE "
    "AND r.exploratory = FALSE AND r.firewall_ok = TRUE "
    "AND r.counterfactual_kind = 'none'"
)
_ELIGIBLE_CORE = f"{_ELIGIBLE_CORE_ANY_SPLIT} AND {_HELD_OUT}"

# `_ELIGIBLE_CORE_ANY_SPLIT`'s nine clauses, spelled out again as (label, fragment) so the
# eligibility breakdown below can attribute a removed row to exactly ONE clause rather than
# to the predicate as a whole. `test_the_core_clause_list_matches_the_core_string` pins this
# list equal to `_ELIGIBLE_CORE_ANY_SPLIT` so the two cannot drift apart the way the report's
# and the command's predicates once could have.
_CORE_CLAUSES: tuple[tuple[str, str], ...] = (
    ("status_ok", "r.status = 'ok'"),
    ("gold_exposed", "r.gold_exposed = FALSE"),
    ("is_dev_run", "r.is_dev_run = FALSE"),
    ("dirty", "r.dirty = FALSE"),
    ("pilot_flag", "r.pilot_flag = FALSE"),
    ("canary_hit", "r.canary_hit = FALSE"),
    ("exploratory", "r.exploratory = FALSE"),
    ("firewall_ok", "r.firewall_ok = TRUE"),
    ("counterfactual_kind", "r.counterfactual_kind = 'none'"),
)

# THE TWO CLAUSES THAT EXCLUDE A ROW FOR WHAT IT **IS**, not for what went wrong with it. A
# forked rollout (`exploratory`, by campaign convention -- see `pi_eval.fork_report`'s own
# docstring: "ELIGIBLE rightly excludes forks from the main tables, they inherit a foreign
# prefix, so this protocol needs its own report") and a counterfactual rollout (`prefix`,
# `loo`, `forced_continue`, `candidate_branch`, `shapley_perm`, `random_q_branch` --
# `pinq.types.CounterfactualKind`) are both alternate rollouts OF an existing run rather than
# an independent arm run on the task, and pooling either under the task's plain task_id would
# be exactly the identifier collision the firewall section of CLAUDE.md warns about for forks
# specifically. Re-running the suite on `split: test` cannot admit these rows: the new run
# would carry the identical flag for the identical reason and the clause would refuse it
# again. Checked against the live store on 2026-09-17: no suite's test population is
# currently zeroed out by `counterfactual_kind` alone, but the shape of the exclusion is the
# same as `exploratory`'s and it gets the same treatment on the chance it ever is.
_BY_DESIGN_CLAUSES: dict[str, str] = {
    "exploratory": (
        "flagged exploratory -- a canary, a rehearsal, an oracle probe, or (in practice, for "
        "the suites this has been seen on) a forked rollout that exists to be looked at, not "
        "pooled into a confirmatory table. A forked run reaches a table through "
        "`pi_eval.fork_report`'s own paired-run provenance, not through this predicate -- see "
        "that module's docstring."
    ),
    "counterfactual_kind": (
        "a counterfactual rollout (prefix/loo/forced_continue/candidate_branch/shapley_perm/"
        "random_q_branch), not an independent arm run on this task_id. It reaches a table, if "
        "at all, through whatever report was built for that counterfactual protocol, not "
        "through this predicate."
    ),
}


def cmd_suites_eligibility(a: argparse.Namespace) -> int:
    """Which split do the runs that feed the paper's tables actually come from?

    WHY THIS COMMAND EXISTS. `pi_eval.report.ELIGIBLE` filtered on status, gold exposure, dirty
    trees, pilots, canaries, the firewall and reconciliation -- and **not on split**. Neither
    does `conf/grids/tier1_confirmatory.yaml`, `tier2_confirmatory.yaml` or `tier3_drgym.yaml`,
    none of which declares one; only `tier1_trained.yaml` and `latent_trainset.yaml` do.

    Nothing is contaminated while every arm is prompted -- a policy that never trained cannot
    have trained on these tasks. Two things break the moment one is:

      * `inquirer_trained` compared against `inquirer_prompted` on rows drawn from `train` is
        a comparison on the trained arm's own training data;
      * `tier1_trained` DOES declare `split: test`, so the trained arm and the prompted arm
        would be measured on different populations, and pooling or comparing across the two
        grids is invalid regardless of contamination.

    Both failures are silent. The table renders, the CI is tight, and nothing in the output
    says which split it came from.

    WHAT CHANGED ON 2026-09-15 (plan v4 SS8.6). `ELIGIBLE` gained `AND r.split = 'test'`, so
    the mixture this command used to report can no longer reach a table, and it exits 0. It
    still runs, and it still prints per suite, because the question it answers is now the next
    one: how much did the clause remove, and from where. That number changes every time a sweep
    lands, and a suite whose eligible set went to ZERO is a suite that now reports nothing --
    a gap in the evidence rather than a contamination, and one that must be visible rather than
    inferred from a missing row in a rendered table.
    """
    root = _repo_root(a)
    parquet = Path(a.parquet or (root / "scores" / "parquet")) / "runs.parquet"
    if not parquet.exists():
        print(f"no runs.parquet at {parquet}; run `pi compact` first")
        return 2

    import duckdb

    con = duckdb.connect()
    # THE PRE-CLAUSE POPULATION, grouped by the split the clause reads. Every row in it passes
    # every other check, so its train and dev columns are exactly what the held-out clause
    # removes, and its test column is what is left to report.
    rows = con.execute(
        f"SELECT r.suite_id, r.split, count(*) AS n FROM '{parquet}' r "
        f"WHERE {_ELIGIBLE_CORE_ANY_SPLIT} GROUP BY 1, 2 ORDER BY 1, 2"
    ).fetchall()

    by_suite: dict[str, dict[str, int]] = {}
    for suite, split, n in rows:
        by_suite.setdefault(suite, {})[split] = n

    if getattr(a, "json", False):
        print(json.dumps(by_suite, indent=2, sort_keys=True))
        return 0

    # THE TWO WAYS A SUITE CAN "REPORT NOTHING", priced separately. `raw_test_rows` counts
    # test-split rows with NO clause applied at all -- a suite absent from it never had a
    # test-split row on disk, full stop, and the re-run advisory is the correct remedy.
    # `clause_removed[label][suite]` is the MARGINAL count for one clause: rows that satisfy
    # every OTHER core clause and fail only this one, restricted to split='test'. A suite whose
    # zero comes from a positive count under a by-design clause (`_BY_DESIGN_CLAUSES`) has rows
    # that exist, carry the test split already, and are excluded because of what they ARE --
    # re-running cannot change that, because the clause does not read the split.
    raw_test_rows: dict[str, int] = dict(
        con.execute(
            f"SELECT r.suite_id, count(*) FROM '{parquet}' r WHERE r.split = 'test' GROUP BY 1"
        ).fetchall()
    )
    clause_removed: dict[str, dict[str, int]] = {}
    for label, frag in _CORE_CLAUSES:
        others = " AND ".join(f for lbl, f in _CORE_CLAUSES if lbl != label)
        q = (
            f"SELECT r.suite_id, count(*) FROM '{parquet}' r "
            f"WHERE r.split = 'test' AND {others} AND NOT ({frag}) GROUP BY 1"
        )
        clause_removed[label] = dict(con.execute(q).fetchall())

    print("Runs eligible to reach a paper table, by split.")
    print(f"{'suite':13s} {'train':>8s} {'dev':>8s} {'test':>8s}   verdict")
    print("-" * 74)
    dropped = 0
    absent: list[str] = []  # zero test-split rows on disk, of any kind
    by_design: list[str] = []  # test rows exist; excluded because of what they ARE
    by_design_detail: dict[str, tuple[str, int, int]] = {}
    unexplained: list[str] = []  # test rows exist; excluded by accidental clauses only
    unexplained_detail: dict[str, str] = {}
    for suite in sorted(by_suite):
        c = by_suite[suite]
        tr, dv, te = c.get("train", 0), c.get("dev", 0), c.get("test", 0)
        held = tr + dv
        dropped += held
        if held == 0:
            verdict = "held out only"
        elif te == 0:
            raw_te = raw_test_rows.get(suite, 0)
            if raw_te == 0:
                verdict = f"REPORTS NOTHING: {held} dropped, 0 test rows exist"
                absent.append(suite)
            else:
                per_clause = {lbl: clause_removed[lbl].get(suite, 0) for lbl, _ in _CORE_CLAUSES}
                hits = {
                    lbl: n for lbl, n in per_clause.items() if lbl in _BY_DESIGN_CLAUSES and n > 0
                }
                if hits:
                    lbl = max(hits, key=hits.get)
                    verdict = f"EXCLUDED BY DESIGN ({lbl}): {hits[lbl]} of {raw_te} test row(s)"
                    by_design.append(suite)
                    by_design_detail[suite] = (lbl, hits[lbl], raw_te)
                else:
                    verdict = f"REPORTS NOTHING: 0 of {raw_te} test row(s) eligible"
                    unexplained.append(suite)
                    unexplained_detail[suite] = ", ".join(
                        f"{lbl}={n}" for lbl, n in per_clause.items() if n
                    )
        else:
            verdict = f"{held} dropped by the split clause"
        print(f"{suite:13s} {tr:8d} {dv:8d} {te:8d}   {verdict}")
    print("-" * 74)
    # NOT A REFUSAL ANY MORE, and the exit code is the decision. While `ELIGIBLE` ignored split
    # this command was the only thing between a train row and a published table, so it exited
    # 1. The predicate refuses those rows itself now, in one place, for every caller -- so an
    # exit code here would fire on a store that merely HOLDS training runs, which is every
    # store, and a check that fires on correct behaviour is a check that gets disabled. THE SAME
    # ARGUMENT applies to the by-design case below, harder: it is not merely correct behaviour
    # but the FIREWALL working (forks and counterfactual rollouts refused on purpose), and a
    # tool that exits non-zero for that trains people to ignore its exit code the first time it
    # fires on something that needed no action -- which is every time a fork campaign runs.
    print(
        f"`pi_eval.report.ELIGIBLE` carries `AND {_HELD_OUT}` since 2026-09-15, so the "
        f"{dropped} train/dev row(s)\nabove cannot reach a table. Those two columns are what "
        "the clause removes, not a warning."
    )
    if absent:
        print(
            f"{len(absent)} suite(s) have NO test-split rows on disk at all: "
            f"{', '.join(absent)}.\nA gap in the evidence, not a contamination: the grids ran "
            "the tasks, and the rows they\nproduced are not held out. Re-run those suites on "
            "`split: test` before citing them."
        )
    if by_design:
        print(
            f"{len(by_design)} suite(s) report nothing BY DESIGN -- their test rows exist and "
            "already carry the test split, and are excluded because of what they ARE, not "
            "because they are absent:"
        )
        for suite in by_design:
            lbl, n, raw_te = by_design_detail[suite]
            print(f"  {suite}: {n} of {raw_te} test row(s) are {_BY_DESIGN_CLAUSES[lbl]}")
        print(
            "Launching these suites again on `split: test` produces the identical exclusion "
            "and admits nothing: the clause reads what the run is, not which split it landed "
            "on. Do not size or launch a re-run for these -- follow the remedy named above."
        )
    if unexplained:
        print(
            f"{len(unexplained)} suite(s) have test-split rows on disk that are excluded by "
            "other (non-by-design) clauses and report nothing:"
        )
        for suite in unexplained:
            print(f"  {suite}: {unexplained_detail[suite]}")
        print(
            "These are not by-design exclusions; investigate the clause(s) named above before "
            "deciding whether a re-run would help."
        )
    print("every eligible run is held out")
    return 0


# ------------------------------------------------------------------------------ pi suites range

# The arm roles the range rule is stated over. The CEILING is an arm handed the evidence it
# would otherwise have had to ask for, the FLOOR is an arm that never asks, and between them
# sits everything a policy can possibly buy on this metric. A preregistered endpoint on a
# metric whose ceiling and floor nearly touch is powered to detect an effect the MEASUREMENT
# cannot express, and nothing in a rendered table says so: the row appears, the CI is tight.
RANGE_PROMPTED_ARM = "inquirer_prompted"
RANGE_CEILING_ARMS = ("oracle_vreq", "gold_evidence")
RANGE_FLOOR_ARM = "drafter_only"

# The plan's rule, both halves. `span` is in the metric's own units, which is why the printed
# table names them: on a [0,1] recall 0.2 is twice the headroom Gate 2 asks of the oracle arm;
# on a COUNT (breadth_components) it is 0.2 of a component and the rule barely bites.
RANGE_MIN_SPAN = 0.2
RANGE_MIN_NONNAN = 0.9

# Families whose metrics are scores of the policy. `operational` is excluded because wall_ms
# and usd are machine artifacts that are never compared across arms (CLAUDE.md), and `frontier`
# and `qvalue` are indexed ladders whose points are not per-task scores. `--metric` overrides
# this for a deliberate look at one name.
RANGE_FAMILIES = frozenset({"quality", "discovery", "structure", "latent", "human"})


def _range_metrics(con, scores: Path, scorer_hash: str, wanted) -> list[str]:
    """The metric names to check, in table order.

    Read from the DATA rather than from the register: a metric that is declared and never
    emitted has no range to report, and one emitted under another name would be invisible
    to a hard-coded list.
    """
    from pi_eval.score import define

    names = [
        str(r[0])
        for r in con.execute(
            f"SELECT DISTINCT metric_name FROM '{scores}' WHERE scorer_hash = ? ORDER BY 1",
            [scorer_hash],
        ).fetchall()
    ]
    if wanted:
        return [n for n in names if n in set(wanted)]
    out = []
    for n in names:
        if "#" in n:  # one point of a ladder, not a per-task score
            continue
        d = define(n)
        if d is not None and not d.indexed and d.family in RANGE_FAMILIES:
            out.append(n)
    return out


def cmd_suites_range(a: argparse.Namespace) -> int:
    """Per (suite, metric): the span an arm can move in, and how much of the suite it covers.

    READ-ONLY, and pinned to ONE `scorer_hash`. A metric's range is a property of the scorer
    that produced it -- re-scoring under a changed definition writes new rows beside the old
    ones under a new hash, so pooling the two would average two definitions of one name.
    The population is `_ELIGIBLE_CORE`'s held-out split only: a range measured over training
    rollouts would answer a different question than the one a confirmatory endpoint asks.
    """
    root = _repo_root(a)
    parquet = Path(a.parquet or (root / "scores" / "parquet"))
    for name in ("runs.parquet", "scores.parquet"):
        if not (parquet / name).exists():
            print(f"no {name} at {parquet}; run `pi compact` first")
            return 2

    import duckdb

    con = duckdb.connect()
    runs, scores = parquet / "runs.parquet", parquet / "scores.parquet"
    metrics = _range_metrics(con, scores, a.scorer_hash, getattr(a, "metric", None))

    # The MEAN ignores NaN and the DENOMINATOR does not: an absent row and a NaN row say the
    # same thing -- this task was not measured -- and neither may enter the mean as a 0.
    means = con.execute(
        f"SELECT r.suite_id, s.metric_name, r.arm_id, avg(s.value) "
        f"FROM '{scores}' s JOIN '{runs}' r USING(run_id) "
        f"WHERE s.scorer_hash = ? AND {_ELIGIBLE_CORE} "
        "AND s.value IS NOT NULL AND NOT isnan(s.value) GROUP BY 1, 2, 3",
        [a.scorer_hash],
    ).fetchall()
    by_arm: dict[tuple[str, str], dict[str, float]] = {}
    for suite, metric, arm, mean in means:
        by_arm.setdefault((str(suite), str(metric)), {})[str(arm)] = float(mean)

    # Coverage is counted over TASKS of the arm under test, not over score rows: a task whose
    # row was never written is exactly the case the share exists to expose.
    total_tasks = {
        str(r[0]): int(r[1])
        for r in con.execute(
            f"SELECT r.suite_id, count(DISTINCT r.task_id) FROM '{runs}' r "
            f"WHERE r.arm_id = ? AND {_ELIGIBLE_CORE} GROUP BY 1",
            [RANGE_PROMPTED_ARM],
        ).fetchall()
    }
    measured = {
        (str(r[0]), str(r[1])): int(r[2])
        for r in con.execute(
            f"SELECT r.suite_id, s.metric_name, count(DISTINCT r.task_id) "
            f"FROM '{scores}' s JOIN '{runs}' r USING(run_id) "
            f"WHERE s.scorer_hash = ? AND r.arm_id = ? AND {_ELIGIBLE_CORE} "
            "AND s.value IS NOT NULL AND NOT isnan(s.value) GROUP BY 1, 2",
            [a.scorer_hash, RANGE_PROMPTED_ARM],
        ).fetchall()
    }

    rows = []
    for suite, metric in sorted({k for k in by_arm if k[1] in set(metrics)}):
        arms = by_arm[(suite, metric)]
        ceiling_arm = next((x for x in RANGE_CEILING_ARMS if x in arms), None)
        ceiling = arms.get(ceiling_arm) if ceiling_arm else None
        floor = arms.get(RANGE_FLOOR_ARM)
        span = (ceiling - floor) if (ceiling is not None and floor is not None) else None
        n_tasks = total_tasks.get(suite, 0)
        share = (measured.get((suite, metric), 0) / n_tasks) if n_tasks else float("nan")

        if n_tasks and share < RANGE_MIN_NONNAN:
            verdict = "not measured"
        elif span is None:
            verdict = "no ceiling arm" if ceiling is None else "no floor arm"
        elif span < RANGE_MIN_SPAN:
            verdict = "no range"
        else:
            verdict = "ok"
        rows.append(
            {
                "suite": suite,
                "metric": metric,
                "prompted": arms.get(RANGE_PROMPTED_ARM),
                "ceiling": ceiling,
                "ceiling_arm": ceiling_arm,
                "floor": floor,
                "span": span,
                "nonnan_share": share,
                "n_tasks": n_tasks,
                "verdict": verdict,
            }
        )

    bad = [r for r in rows if r["verdict"] in ("no range", "not measured")]
    if getattr(a, "json", False):
        print(json.dumps({"scorer_hash": a.scorer_hash, "rows": rows}, indent=2, sort_keys=True))
        return 1 if bad else 0

    def _f(v: object) -> str:
        return "     --" if v is None or (isinstance(v, float) and v != v) else f"{v:7.3f}"

    print(f"Range of each metric under scorer_hash {a.scorer_hash[:12]}.")
    print(
        f"ceiling = first of {'/'.join(RANGE_CEILING_ARMS)} that ran; floor = {RANGE_FLOOR_ARM}; "
        f"arm under test = {RANGE_PROMPTED_ARM}."
    )
    print(
        f"{'suite':13s} {'metric':28s} {'arm':>7s} {'ceil':>7s} {'floor':>7s} "
        f"{'span':>7s} {'cover':>6s}   verdict"
    )
    print("-" * 100)
    for r in rows:
        cover = "    --" if r["nonnan_share"] != r["nonnan_share"] else f"{r['nonnan_share']:6.2f}"
        print(
            f"{r['suite']:13s} {r['metric']:28s} {_f(r['prompted'])} {_f(r['ceiling'])} "
            f"{_f(r['floor'])} {_f(r['span'])} {cover}   {r['verdict']}"
        )
    print("-" * 100)
    if not bad:
        print(f"every metric clears span >= {RANGE_MIN_SPAN} and coverage >= {RANGE_MIN_NONNAN}")
        return 0
    print(
        f"{len(bad)} (suite, metric) pair(s) fail the range rule. A metric whose ceiling and "
        f"floor are less than {RANGE_MIN_SPAN} apart cannot express the effect an endpoint on "
        f"it is powered to detect, and one measured on under {RANGE_MIN_NONNAN:.0%} of tasks "
        "reports over a different population from the one the grid selected. Neither is "
        "visible in a rendered table."
    )
    return 1
