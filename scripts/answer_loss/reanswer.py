"""Lane L3, part B (re-answer): is the recipe's evidence gain lost in the frozen answering path?

Implements the part-B primary of `artifacts/answer_loss_20260923/DECLARATION.md` (c2bdf77):

  (0)   Claude Opus 5, closed book (question only) -- the headroom floor, one call per task.
  (i)   the frozen Answerer EXACTLY as pinned in the population's manifests
        (`pins.answerer.model_id` = `openai/aws/gpt-oss-120b`, asserted per run), called through
        the harness's own `FrozenLLMAnswerer` + `CachingClient(MeteredClient)` on each state's
        evidence and draft, with the harness's 1,200-char rendering -- the CONTROL.
  (iii) Claude Opus 5 on the state's evidence ONLY, untruncated, told to answer only from the
        evidence or say "unknown".

Each of (i) and (iii) is read at two states per run: the arm's OWN STOP (final evidence, final
draft) and the EQUAL-SPEND MATCHED PREFIX (for each pair (training seed, task, rollout seed),
k = min(n_asks recipe, n_asks comparator); the evidence retrieved in the first k turns and the
draft recorded at turn k-1 -- both arms truncated, the symmetric rule of Table 1).

PRIMARY: DiD = (recipe - comparator under iii) - (recipe - comparator under i), paired over
tasks (training seeds averaged within the task, rollout seeds into the task), clusters =
template_id where present, 10k intervals, DECIDED only if every 50k interval (101/202/303)
excludes zero.

Every request is canary-scanned before it is sent (and `CachingClient` scans the control's
again on its own path). Every response is cached on disk under `<out-dir>/cache/`, keyed by the
sha256 of the request bytes, so a rerun is free and the record replays.

Usage (repo root):
    set -a; source .env; set +a
    PI_GOLD_ROOT=$PWD/data/gold .venv/bin/python -m scripts.answer_loss.reanswer --dry-run
    PI_GOLD_ROOT=$PWD/data/gold .venv/bin/python -m scripts.answer_loss.reanswer --full
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from pi_eval.canary import load as load_canaries
from pi_eval.canary import scan_text
from pinq.types import Draft, Evidence
from pinq_expt.components import ANSWER_MAX_TOKENS, UNIT_CHARS, render_evidence

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "artifacts" / "answer_loss_20260923"
RUNS = REPO / "runs"
BASE_URL = "http://127.0.0.1:4030"
OPUS = "openai/aws/claude-opus-5"
CONTROL_MODEL = "openai/aws/gpt-oss-120b"
# Claude Opus 5 spends hidden reasoning tokens INSIDE max_tokens (the reply carries
# `reasoning_content`; usage reports no reasoning_tokens). MEASURED in this lane's first full pass
# at 300 (reanswer_rows.full_cap300.jsonl): 416 of 5,425 replies finished 'length', 395 of them
# with EMPTY content (98 of 200 MuSiQue closed-book calls; 104 finished 'length'), and were scored
# as wrong answers. Re-sent at 4,000, six of them used
# 248-1,428 completion tokens and all finished with 'stop'. 8,000 leaves headroom, and
# `usable_content` refuses to score an empty cap-truncated reply whatever the cap.
OPUS_MAX_TOKENS = 8000
# MEASURED in the 12:26 full pass at 8000: 5 of 2,976 distinct Opus requests (all MuSiQue
# closed-book) spent all 8,000 completion tokens and returned no content. A request is re-sent at
# the next cap ONLY when its reply at the current cap was empty and cut by the cap; a reply usable
# at 8000 is never re-sent, so every finished request keeps its key and its cache entry.
OPUS_MAX_TOKENS_LADDER = (OPUS_MAX_TOKENS, 32000)
CANARY_PREFIX = "PINQCANARY"
DRY_RUN_TASKS = 20

EVIDENCE_ONLY = """Answer the question using ONLY the evidence below. Do not use any knowledge that the evidence does not state. If the evidence does not contain enough information to answer, reply with exactly: unknown

QUESTION
{question}

EVIDENCE
{evidence}

Answer in at most {word_cap} words. Give the answer itself as a bare span, with no preamble, no restatement of the question and no explanation. If it is a yes/no question, answer yes or no."""

CLOSED_BOOK = """Answer the question from your own knowledge. Give your best answer even if you are not certain.

QUESTION
{question}

Answer in at most {word_cap} words. Give the answer itself as a bare span, with no preamble, no restatement of the question and no explanation. If it is a yes/no question, answer yes or no."""

# --------------------------------------------------------------------------- pure


def render_full(ev: Evidence) -> str:
    """The harness's own rendering (`render_evidence`: canonical uid order, `[uid12] title`
    header) with the per-unit cut lifted -- so (iii) differs from the harness's view in the
    window and nothing else."""
    return render_evidence(ev, unit_chars=10**9)


def prefix_uids(turns: Sequence[Mapping[str, Any]], k: int) -> set[str]:
    """Union of `retrieved_uids` over the first k turns, in `turn_idx` order."""
    ordered = sorted(turns, key=lambda t: int(t["turn_idx"]))
    if not 1 <= k <= len(ordered):
        raise ValueError(f"prefix k={k} outside 1..{len(ordered)}")
    out: set[str] = set()
    for t in ordered[:k]:
        out.update(str(u) for u in (t.get("retrieved_uids") or ()))
    return out


def word_cap(text: str, cap: int) -> str:
    """`FrozenLLMAnswerer`'s own cap, applied to every condition so answer length is held."""
    return " ".join(text.split()[:cap])


class CanaryRefused(RuntimeError):
    """A request carried a gold canary (or anything bearing the canary prefix). Never sent."""


def assert_no_canary(text: str, canaries: frozenset[str], *, where: str) -> None:
    """Refuse on any registered nonce (`pi_eval.canary.scan_text`), and also on the bare
    prefix: every nonce carries it, so this is strictly more conservative than the registry."""
    if CANARY_PREFIX not in text:
        return
    hits = scan_text(text, set(canaries), where=where)
    raise CanaryRefused(
        f"{where}: canary prefix in request"
        + (f" ({len(hits)} registered nonce(s))" if hits else " (unregistered token)")
    )


class ReplyTruncated(RuntimeError):
    """The token cap ran out before the model emitted any content: no answer was given."""


def usable_content(text: str, finish_reason: str | None) -> str:
    """The reply's content, or raise if the cap ran out before any content. An empty reply cut
    by the cap is an instrument failure (an error row), never a wrong answer."""
    if not (text or "").strip() and finish_reason == "length":
        raise ReplyTruncated("empty content with finish_reason 'length': max_tokens exhausted")
    return text


def evidence_only_prompt(question: str, ev: Evidence, *, word_cap: int) -> str:
    return EVIDENCE_ONLY.format(question=question, evidence=render_full(ev), word_cap=word_cap)


def closed_book_prompt(question: str, *, word_cap: int) -> str:
    return CLOSED_BOOK.format(question=question, word_cap=word_cap)


def request_key(req: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(req, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def did_inputs(
    r3: Mapping[str, float],
    c3: Mapping[str, float],
    r1: Mapping[str, float],
    c1: Mapping[str, float],
) -> tuple[dict[str, float], dict[str, float]]:
    """Per task, (recipe iii - recipe i) and (comparator iii - comparator i): their paired
    difference IS (recipe - comparator under iii) - (recipe - comparator under i)."""
    keys = sorted(set(r3) & set(c3) & set(r1) & set(c1))
    return {k: r3[k] - r1[k] for k in keys}, {k: c3[k] - c1[k] for k in keys}


def template_sha(t: str) -> str:
    return hashlib.sha256(t.encode()).hexdigest()


# --------------------------------------------------------------------------- clients


class OpusClient:
    """OpenAI-compatible chat call through the local litellm proxy, disk-cached by
    sha256(request bytes). Temperature 0; no seed (not every provider route honours one)."""

    def __init__(self, cache_dir: Path, canaries: frozenset[str], *, api_key: str) -> None:
        self.cache_dir = cache_dir
        self.canaries = canaries
        self.api_key = api_key
        self.lock = threading.Lock()
        self.n_scanned = 0
        self._key_locks: dict[str, threading.Lock] = {}

    def _lock_for(self, key: str) -> threading.Lock:
        with self.lock:
            return self._key_locks.setdefault(key, threading.Lock())

    def complete(self, prompt: str, *, where: str) -> dict[str, Any]:
        truncated: ReplyTruncated | None = None
        for cap in OPUS_MAX_TOKENS_LADDER:
            req = {
                "model": OPUS,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": cap,
            }
            body = json.dumps(req, ensure_ascii=False)
            assert_no_canary(body, self.canaries, where=where)
            with self.lock:
                self.n_scanned += 1
            key = request_key(req)
            # one request in flight per key: two identical requests racing would both be paid
            # for and the second would overwrite the first's cached text
            try:
                with self._lock_for(key):
                    res = self._complete_locked(req, key)
            except ReplyTruncated as exc:
                truncated = exc
                continue
            return {**res, "max_tokens": cap}
        assert truncated is not None
        raise truncated

    def _complete_locked(self, req: Mapping[str, Any], key: str) -> dict[str, Any]:
        import httpx

        p = self.cache_dir / key[:2] / f"{key}.json"
        if p.exists():
            rec = json.loads(p.read_text())
            usable_content(rec["text"], rec.get("finish_reason"))
            return {**rec, "cache_hit": True}
        last: str = ""
        for attempt in range(8):
            try:
                r = httpx.post(
                    f"{BASE_URL}/v1/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=req,
                    timeout=180,
                )
            except httpx.TransportError as exc:
                last = repr(exc)
                time.sleep(min(60, 4 * 2**attempt))
                continue
            if r.status_code in (400, 401, 403, 404):
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
            if r.status_code != 200:
                last = f"HTTP {r.status_code}: {r.text[:200]}"
                time.sleep(min(60, 4 * 2**attempt))
                continue
            j = r.json()
            choice = j["choices"][0]
            rec = {
                "request_key": key,
                "request": req,
                "text": choice["message"].get("content") or "",
                "finish_reason": choice.get("finish_reason"),
                "model_returned": j.get("model"),
                "usage": j.get("usage"),
                "attempts": attempt + 1,
            }
            # cached even when unusable, so the record of what came back is kept; the raise
            # below makes it an error row
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(f".tmp{threading.get_ident()}")
            tmp.write_text(json.dumps(rec, ensure_ascii=False))
            os.replace(tmp, p)
            usable_content(rec["text"], rec.get("finish_reason"))
            return {**rec, "cache_hit": False}
        raise RuntimeError(f"exhausted retries: {last}")


class ScannedLLM:
    """Wraps the harness's `CachingClient`: scans the exact messages before the call and keeps
    the request sha, so the control's request can be compared byte-for-byte with the one the
    recorded run's Answerer sent (`calls.jsonl` `request_sha`)."""

    def __init__(self, inner: Any, canaries: frozenset[str], *, where: str) -> None:
        self._inner = inner
        self._canaries = canaries
        self._where = where
        self.last_key: str | None = None

    def pin(self, role: str) -> Any:
        return self._inner.pin(role)

    def complete(self, **kw: Any) -> tuple[str, Any]:
        assert_no_canary(
            json.dumps(kw["messages"], ensure_ascii=False), self._canaries, where=self._where
        )
        self.last_key = self._inner.key_for(**kw)
        return self._inner.complete(**kw)


def control_call(
    view: Any,
    ev: Evidence,
    draft_text: str,
    seed: int,
    gw: Any,
    canaries: frozenset[str],
    where: str,
) -> dict[str, Any]:
    from scripts.answerer_strength import lib as as_lib

    from pinq.budget import BudgetLedger
    from pinq_expt.components import FrozenLLMAnswerer

    ledger = BudgetLedger(cap=10**9)
    inner = as_lib._client_for(CONTROL_MODEL, "answerer", gw, ledger)
    wrap = ScannedLLM(inner, canaries, where=where)
    ans = FrozenLLMAnswerer(llm=wrap).answer(
        view, ev, Draft(text=draft_text), seed=seed, ledger=ledger
    )
    tel = ledger.calls[-1]
    return {
        "text": ans.text,
        "request_sha": wrap.last_key,
        "cache_hit": bool(tel.cache_hit),
        "model": tel.model,
        "tok_prompt": tel.tok_prompt,
        "tok_completion": tel.tok_completion,
        "usd": tel.usd,
    }


# --------------------------------------------------------------------------- population and states


def read_ids(path: Path) -> list[str]:
    return [
        ln.split()[0]
        for ln in path.read_text().splitlines()
        if ln.strip() and not ln.startswith("#")
    ]


def load_population(limit_tasks: int | None) -> dict[str, Any]:
    """Runs, their turns, their reconstructed views and evidence, and the states to answer."""
    from scripts.answer_loss.decompose import COHORT, RECIPE, SEEDREP, SUITES
    from scripts.answerer_strength import lib as as_lib

    pop: dict[str, Any] = {"suites": {}, "states": {}, "views": {}, "checks": {}}
    n_sha_draft_bad = n_hash_bad = 0
    for suite in SUITES:
        rec_ids = {
            s: read_ids(SEEDREP / "run_ids" / f"run_ids.{m}.{suite}.txt") for s, m in RECIPE.items()
        }
        cmp_ids = read_ids(COHORT / "cohort" / f"run_ids.prompted.{suite}.txt")
        runs: dict[str, dict[str, Any]] = {}
        for arm, ids in (("s1", rec_ids["s1"]), ("s2", rec_ids["s2"]), ("comparator", cmp_ids)):
            for rid in ids:
                d = RUNS / rid
                man = json.loads((d / "manifest.json").read_text())
                if man["pins"]["answerer"]["model_id"] != CONTROL_MODEL:
                    raise SystemExit(f"{rid}: answerer pin {man['pins']['answerer']['model_id']}")
                turns = [
                    json.loads(x) for x in (d / "turns.jsonl").read_text().splitlines() if x.strip()
                ]
                turns.sort(key=lambda t: int(t["turn_idx"]))
                if [int(t["turn_idx"]) for t in turns] != list(range(len(turns))) or not turns:
                    raise SystemExit(f"{rid}: turns not 0..n-1")
                calls = [
                    json.loads(x) for x in (d / "calls.jsonl").read_text().splitlines() if x.strip()
                ]
                ans_calls = [c for c in calls if c.get("actor") == "answerer"]
                rec = as_lib.load_run_record(RUNS, rid, arm=arm)
                runs[rid] = {
                    "rid": rid,
                    "arm": "recipe" if arm != "comparator" else "comparator",
                    "tseed": arm if arm != "comparator" else None,
                    "task": rec.task_id,
                    "rseed": rec.seed,
                    "n": len(turns),
                    "turns": turns,
                    "record": rec,
                    "recorded_answer_request_sha": ans_calls[-1]["request_sha"]
                    if ans_calls
                    else None,
                    "inquirer_model": man["pins"]["inquirer"]["model_id"],
                }
        tasks = sorted({r["task"] for r in runs.values()})
        if limit_tasks is not None:
            # the first `limit_tasks` tasks in sorted order, plus -- if none of them does -- the
            # first task whose recorded final evidence carries a unit longer than the window, so
            # the dry run's untruncation assertion has a paragraph to bite on
            chosen = tasks[:limit_tasks]
            if not any(_has_long_unit(r, suite) for r in runs.values() if r["task"] in set(chosen)):
                extra = next(
                    (
                        t
                        for t in tasks[limit_tasks:]
                        if any(_has_long_unit(r, suite) for r in runs.values() if r["task"] == t)
                    ),
                    None,
                )
                if extra is not None:
                    chosen = [*chosen, extra]
            tasks = sorted(chosen)
            runs = {k: v for k, v in runs.items() if v["task"] in set(tasks)}
        pop["suites"][suite] = {"tasks": tasks, "runs": runs}
        # views and final evidence, checked against the recorded subset hash
        for rid, r in runs.items():
            view, ev = as_lib.build_view_and_evidence(
                r["record"], corpora_root=REPO / "data" / "corpora"
            )
            if r["turns"][-1]["subset_hash_after"] != ev.subset_hash:
                n_hash_bad += 1
                raise SystemExit(f"{rid}: last turn subset_hash_after != final evidence hash")
            if Draft(text=r["turns"][-1]["draft_text"]).sha != r["turns"][-1]["draft_sha"]:
                n_sha_draft_bad += 1
            pop["views"][rid] = view
            r["units"] = {
                u.uid: u
                for u in as_lib._units_by_uid(
                    as_lib.get_suite(suite, r["record"].corpus_dir, REPO / "data" / "corpora"),
                    (suite, r["record"].corpus_dir, r["task"]),
                ).values()
            }
            add_state(pop, suite, rid, None, view, ev, r["turns"][-1]["draft_text"])
        # matched-prefix states, one per (training seed, task, rollout seed) pair
        cmp_by = {(r["task"], r["rseed"]): r for r in runs.values() if r["arm"] == "comparator"}
        pairs = []
        for r in runs.values():
            if r["arm"] != "recipe":
                continue
            c = cmp_by[(r["task"], r["rseed"])]
            k = min(r["n"], c["n"])
            sid_r = prefix_state(pop, suite, r, k)
            sid_c = prefix_state(pop, suite, c, k)
            pairs.append(
                {
                    "tseed": r["tseed"],
                    "task": r["task"],
                    "rseed": r["rseed"],
                    "k": k,
                    "recipe_run": r["rid"],
                    "comparator_run": c["rid"],
                    "own": (f"{r['rid']}@final", f"{c['rid']}@final"),
                    "prefix": (sid_r, sid_c),
                }
            )
        pop["suites"][suite]["pairs"] = pairs
    pop["checks"]["n_final_hash_mismatch"] = n_hash_bad
    pop["checks"]["n_final_draft_sha_mismatch"] = n_sha_draft_bad
    return pop


def _has_long_unit(r: Mapping[str, Any], suite: str) -> bool:
    from scripts.answerer_strength import lib as as_lib

    rec = r["record"]
    units = as_lib._units_by_uid(
        as_lib.get_suite(suite, rec.corpus_dir, REPO / "data" / "corpora"),
        (suite, rec.corpus_dir, rec.task_id),
    )
    return any(len(units[u].text) > UNIT_CHARS for u in rec.cited_uids if u in units)


def add_state(
    pop: dict[str, Any],
    suite: str,
    rid: str,
    k: int | None,
    view: Any,
    ev: Evidence,
    draft_text: str,
) -> str:
    sid = f"{rid}@final" if k is None else f"{rid}@k{k}"
    if sid not in pop["states"]:
        pop["states"][sid] = {
            "sid": sid,
            "suite": suite,
            "rid": rid,
            "k": k,
            "view": view,
            "ev": ev,
            "draft": draft_text,
        }
    return sid


def prefix_state(pop: dict[str, Any], suite: str, r: dict[str, Any], k: int) -> str:
    if k == r["n"]:
        return f"{r['rid']}@final"
    uids = prefix_uids(r["turns"], k)
    ev = Evidence.of([r["units"][u] for u in sorted(uids)])
    t = r["turns"][k - 1]
    if ev.subset_hash != t["subset_hash_after"]:
        raise SystemExit(f"{r['rid']}@k{k}: prefix evidence hash != turn {k - 1} subset_hash_after")
    if Draft(text=t["draft_text"]).sha != t["draft_sha"]:
        raise SystemExit(f"{r['rid']}@k{k}: draft text does not hash to the recorded draft_sha")
    return add_state(pop, suite, r["rid"], k, pop["views"][r["rid"]], ev, t["draft_text"])


# --------------------------------------------------------------------------- execution


def run_calls(
    pop: dict[str, Any], out_dir: Path, rows_path: Path, *, workers_control: int, workers_opus: int
) -> dict[str, Any]:
    from scripts.answerer_strength import lib as as_lib

    from pinq_adapters.llm.pricing import PriceTable

    api_key = os.environ.get("LITELLM_API_KEY", "")
    if not api_key:
        raise SystemExit("LITELLM_API_KEY is not set: source .env first")
    canaries = frozenset(load_canaries(root=REPO))
    gw = as_lib.Gateway(
        base_url=BASE_URL,
        api_key=api_key,
        cache_root=str(out_dir / "cache" / "control"),
        price_table=PriceTable.load(),
        env=dict(os.environ),
    )
    opus = OpusClient(out_dir / "cache" / "opus", canaries, api_key=api_key)

    jobs_control, jobs_opus = [], []
    for sid, st in pop["states"].items():
        jobs_control.append(("control", sid))
        jobs_opus.append(("evidence_only", sid))
    seen_tasks = set()
    for sid, st in pop["states"].items():
        tk = (st["suite"], pop["suites"][st["suite"]]["runs"][st["rid"]]["task"])
        if tk in seen_tasks:
            continue
        seen_tasks.add(tk)
        jobs_opus.append(("closed_book", sid))

    lock = threading.Lock()
    counts = {"ok": 0, "error": 0, "cache_hit": 0}
    t0 = time.time()
    total = len(jobs_control) + len(jobs_opus)

    def do(job: tuple[str, str]) -> dict[str, Any]:
        cond, sid = job
        st = pop["states"][sid]
        run = pop["suites"][st["suite"]]["runs"][st["rid"]]
        view = st["view"]
        row: dict[str, Any] = {
            "condition": cond,
            "sid": sid,
            "suite": st["suite"],
            "task": run["task"],
        }
        try:
            if cond == "control":
                res = control_call(
                    view, st["ev"], st["draft"], run["rseed"], gw, canaries, where=f"control {sid}"
                )
                row.update(res)
            elif cond == "evidence_only":
                prompt = evidence_only_prompt(view.question, st["ev"], word_cap=view.word_cap)
                res = opus.complete(prompt, where=f"evidence_only {sid}")
                row.update(
                    {
                        "text": word_cap(res["text"], view.word_cap),
                        "raw_text": res["text"],
                        "request_key": res["request_key"],
                        "cache_hit": res["cache_hit"],
                        "model_returned": res.get("model_returned"),
                        "finish_reason": res.get("finish_reason"),
                        "usage": res.get("usage"),
                        "max_tokens": res.get("max_tokens"),
                        "n_units": len(st["ev"].units),
                        "max_unit_chars": max((len(u.text) for u in st["ev"].units), default=0),
                        "long_units_whole": all(u.text in prompt for u in st["ev"].units),
                    }
                )
            else:
                prompt = closed_book_prompt(view.question, word_cap=view.word_cap)
                res = opus.complete(prompt, where=f"closed_book {sid}")
                row.update(
                    {
                        "text": word_cap(res["text"], view.word_cap),
                        "raw_text": res["text"],
                        "request_key": res["request_key"],
                        "cache_hit": res["cache_hit"],
                        "model_returned": res.get("model_returned"),
                        "finish_reason": res.get("finish_reason"),
                        "usage": res.get("usage"),
                        "max_tokens": res.get("max_tokens"),
                    }
                )
            row["ok"] = True
        except CanaryRefused as exc:
            row.update({"ok": False, "error": f"CANARY: {exc}"})
        except Exception as exc:  # noqa: BLE001 -- a failed call is a row, not a crashed sweep
            row.update({"ok": False, "error": repr(exc)[:500]})
        return row

    rows: list[dict[str, Any]] = []
    with (
        rows_path.open("w") as fh,
        ThreadPoolExecutor(workers_control) as ex_c,
        ThreadPoolExecutor(workers_opus) as ex_o,
    ):
        futs = [ex_c.submit(do, j) for j in jobs_control] + [ex_o.submit(do, j) for j in jobs_opus]
        for fut in as_completed(futs):
            row = fut.result()
            with lock:
                rows.append(row)
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                counts["ok" if row["ok"] else "error"] += 1
                counts["cache_hit"] += 1 if row.get("cache_hit") else 0
                n = counts["ok"] + counts["error"]
                if n % 250 == 0 or n == total:
                    print(
                        f"[{time.strftime('%H:%M:%S')}] {n}/{total} ok={counts['ok']} err={counts['error']} "
                        f"cache_hit={counts['cache_hit']} {time.time() - t0:.0f}s",
                        flush=True,
                    )
    return {
        "rows": rows,
        "counts": counts,
        "n_jobs_control": len(jobs_control),
        "n_jobs_opus": len(jobs_opus),
        "n_opus_requests_scanned": opus.n_scanned,
        "n_canaries_registered": len(canaries),
    }


# --------------------------------------------------------------------------- scoring and reading


def score_rows(
    pop: dict[str, Any], rows: list[dict[str, Any]]
) -> dict[tuple[str, str], dict[str, Any]]:
    from scripts.answerer_strength import lib as as_lib

    from pi_eval.gold import load_graphs

    graphs = {s: load_graphs(s, "v1") for s in pop["suites"]}
    out = {}
    for row in rows:
        if not row.get("ok"):
            continue
        g = graphs[row["suite"]][row["task"]]
        sc = as_lib.score_answer(row["text"], g)
        row["f1"] = sc["answer_token_f1"]
        row["correct"] = sc["contains_answer"]
        row["unknown"] = row["text"].strip().lower().rstrip(".") == "unknown"
        out[(row["condition"], row["sid"])] = row
    return out


def stored_scores(pop: dict[str, Any]) -> dict[str, dict[str, float]]:
    from scripts.answer_loss.decompose import COHORT, SEEDREP, Store

    out: dict[str, dict[str, float]] = {}
    for store_path in (SEEDREP / "scores_parquet", COHORT / "scores_parquet"):
        st = Store(store_path)
        for suite, sp in pop["suites"].items():
            ids = [
                rid
                for rid in sp["runs"]
                if (store_path == SEEDREP / "scores_parquet")
                == (sp["runs"][rid]["arm"] == "recipe")
            ]
            for name in ("answer_token_f1", "answer_correct"):
                for rid, v in st.metric(ids, name).items():
                    out.setdefault(rid, {})[name] = v
    return out


def reading(
    pop: dict[str, Any],
    scored: Mapping[tuple[str, str], Any],
    stored: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    from scripts.answer_loss.decompose import pair_task_means, paired_reading
    from scripts.answerer_strength import lib as as_lib

    from pi_eval.gold import load_graphs

    res: dict[str, Any] = {}
    for suite, sp in pop["suites"].items():
        runs = sp["runs"]
        templates = {}
        for rid in runs:
            man = json.loads((RUNS / rid / "manifest.json").read_text())
            templates[runs[rid]["task"]] = man.get("template_id")
        clusters = (
            {t: str(v) for t, v in templates.items() if v} if any(templates.values()) else None
        )
        suite_out: dict[str, Any] = {
            "n_tasks": len(sp["tasks"]),
            "clusters": "template_id" if clusters else "task",
        }

        def val(cond: str, sid: str, metric: str) -> float | None:
            row = scored.get((cond, sid))
            return None if row is None else float(row[metric])

        per: dict[str, Any] = {}
        for reading_name in ("own", "prefix"):
            for metric in ("f1", "correct"):
                maps = {}
                for cond in ("control", "evidence_only"):
                    pairs = [
                        (
                            p["task"],
                            val(cond, p[reading_name][0], metric),
                            val(cond, p[reading_name][1], metric),
                        )
                        for p in sp["pairs"]
                    ]
                    a, b = pair_task_means(pairs)
                    maps[cond] = (a, b)
                    per[f"{reading_name}::{cond}::{metric}"] = paired_reading(
                        a, b, clusters=clusters
                    )
                a_did, b_did = did_inputs(
                    maps["evidence_only"][0],
                    maps["evidence_only"][1],
                    maps["control"][0],
                    maps["control"][1],
                )
                per[f"{reading_name}::DiD::{metric}"] = paired_reading(
                    a_did, b_did, clusters=clusters
                )
        # stored contrast on the same pairs (own stop), and the control's reproduction of it
        for metric, name in (("f1", "answer_token_f1"), ("correct", "answer_correct")):
            pairs = [
                (p["task"], stored[p["recipe_run"]][name], stored[p["comparator_run"]][name])
                for p in sp["pairs"]
            ]
            a_s, b_s = pair_task_means(pairs)
            per[f"own::stored::{metric}"] = paired_reading(a_s, b_s, clusters=clusters)
            pairs_c = [
                (
                    p["task"],
                    val("control", p["own"][0], metric),
                    val("control", p["own"][1], metric),
                )
                for p in sp["pairs"]
            ]
            a_c, b_c = pair_task_means(pairs_c)
            a_d, b_d = did_inputs(a_c, b_c, a_s, b_s)
            per[f"own::control_minus_stored::{metric}"] = paired_reading(
                a_d, b_d, clusters=clusters
            )
        # (0) closed book: one call per task; (iii) - (0) per arm, own stop
        graphs = load_graphs(suite, "v1")
        cb = {}
        for (cond, sid), row in scored.items():
            if cond == "closed_book" and row["suite"] == suite:
                cb[row["task"]] = row
        suite_out["closed_book"] = {
            "n_tasks": len(cb),
            "mean_f1": sum(r["f1"] for r in cb.values()) / len(cb) if cb else float("nan"),
            "mean_correct": sum(r["correct"] for r in cb.values()) / len(cb)
            if cb
            else float("nan"),
        }
        for metric in ("f1", "correct"):
            cbm = {t: float(r[metric]) for t, r in sorted(cb.items())}
            for arm_i, arm in ((0, "recipe"), (1, "comparator")):
                pairs = [
                    (p["task"], val("evidence_only", p["own"][arm_i], metric), 0.0)
                    for p in sp["pairs"]
                ]
                a, _ = pair_task_means(pairs)
                per[f"own::evidence_only_minus_closed_book::{arm}::{metric}"] = paired_reading(
                    a, cbm, clusters=clusters
                )
        # descriptive: unknown rate, reproduction
        repro = {
            "n": 0,
            "exact": 0,
            "request_sha_match": 0,
            "request_sha_known": 0,
            "correct_agree": 0,
        }
        for rid, r in runs.items():
            row = scored.get(("control", f"{rid}@final"))
            if row is None:
                continue
            repro["n"] += 1
            repro["exact"] += int(row["text"] == r["record"].recorded_answer_text)
            if r["recorded_answer_request_sha"]:
                repro["request_sha_known"] += 1
                repro["request_sha_match"] += int(
                    row["request_sha"] == r["recorded_answer_request_sha"]
                )
            repro["correct_agree"] += int(float(row["correct"]) == stored[rid]["answer_correct"])
        suite_out["control_reproduction"] = {
            **repro,
            "exact_rate": repro["exact"] / repro["n"] if repro["n"] else float("nan"),
            "request_sha_match_rate": repro["request_sha_match"] / repro["request_sha_known"]
            if repro["request_sha_known"]
            else float("nan"),
            "answer_correct_agreement_rate": repro["correct_agree"] / repro["n"]
            if repro["n"]
            else float("nan"),
        }
        # scorer lock: score_answer on the RECORDED answer text reproduces the stored metrics
        bad = 0
        for rid, r in runs.items():
            sc = as_lib.score_answer(r["record"].recorded_answer_text, graphs[r["task"]])
            if (
                abs(sc["answer_token_f1"] - stored[rid]["answer_token_f1"]) > 1e-12
                or abs(sc["contains_answer"] - stored[rid]["answer_correct"]) > 1e-12
            ):
                bad += 1
        suite_out["scorer_lock"] = {"n_runs": len(runs), "n_mismatch": bad}
        unk = {}
        for cond in ("evidence_only",):
            for reading_name in ("own", "prefix"):
                for arm_i, arm in ((0, "recipe"), (1, "comparator")):
                    xs = [scored.get((cond, p[reading_name][arm_i])) for p in sp["pairs"]]
                    xs = [x for x in xs if x is not None]
                    unk[f"{reading_name}::{arm}"] = (
                        sum(x["unknown"] for x in xs) / len(xs) if xs else float("nan")
                    )
        suite_out["evidence_only_unknown_rate_pairs"] = unk
        ks = [p["k"] for p in sp["pairs"]]
        suite_out["prefix"] = {
            "n_pairs": len(ks),
            "mean_k": sum(ks) / len(ks) if ks else float("nan"),
            "n_pairs_recipe_prefix_is_own_stop": sum(
                1 for p in sp["pairs"] if p["prefix"][0] == p["own"][0]
            ),
            "n_pairs_comparator_prefix_is_own_stop": sum(
                1 for p in sp["pairs"] if p["prefix"][1] == p["own"][1]
            ),
        }
        suite_out["readings"] = per
        res[suite] = suite_out
    return res


def long_unit_check(pop: dict[str, Any], scored: Mapping[tuple[str, str], Any]) -> dict[str, Any]:
    """The (iii) request must carry a >1,200-char paragraph whole: read back from the cached
    request bytes, not from the prompt builder."""
    out = {"n_states_with_long_unit": 0, "n_checked_from_cache": 0, "n_whole": 0, "example": None}
    for (cond, sid), row in scored.items():
        if cond != "evidence_only" or row.get("max_unit_chars", 0) <= UNIT_CHARS:
            continue
        out["n_states_with_long_unit"] += 1
        p = OUT / "cache" / "opus" / row["request_key"][:2] / f"{row['request_key']}.json"
        sent = json.loads(p.read_text())["request"]["messages"][0]["content"]
        out["n_checked_from_cache"] += 1
        long_units = [u for u in pop["states"][sid]["ev"].units if len(u.text) > UNIT_CHARS]
        whole = all(u.text in sent for u in long_units)
        out["n_whole"] += int(whole)
        if out["example"] is None:
            u = long_units[0]
            out["example"] = {
                "sid": sid,
                "uid": u.uid[:12],
                "unit_chars": len(u.text),
                "whole_in_request": whole,
                "tail_in_request": u.text[-80:] in sent,
            }
    return out


def _finish_reasons(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for r in rows:
        if r["condition"] == "control" or not r.get("ok"):
            continue
        d = out.setdefault(r["condition"], {})
        k = str(r.get("finish_reason"))
        d[k] = d.get(k, 0) + 1
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--full", action="store_true")
    ap.add_argument("--workers-control", type=int, default=16)
    ap.add_argument("--workers-opus", type=int, default=16)
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    limit = DRY_RUN_TASKS if args.dry_run else None
    tag = "dryrun" if args.dry_run else "full"

    print(f"[{time.strftime('%H:%M:%S')}] loading population ({tag})", flush=True)
    pop = load_population(limit)
    print(
        f"[{time.strftime('%H:%M:%S')}] {sum(len(s['runs']) for s in pop['suites'].values())} runs, "
        f"{len(pop['states'])} states, checks {pop['checks']}",
        flush=True,
    )
    rows_path = OUT / f"reanswer_rows.{tag}.jsonl"
    calls = run_calls(
        pop, OUT, rows_path, workers_control=args.workers_control, workers_opus=args.workers_opus
    )
    scored = score_rows(pop, calls["rows"])
    stored = stored_scores(pop)
    out = {
        "provenance": {
            "declaration": "artifacts/answer_loss_20260923/DECLARATION.md (c2bdf77)",
            "command": f"PI_GOLD_ROOT=$PWD/data/gold .venv/bin/python -m scripts.answer_loss.reanswer --{'dry-run' if args.dry_run else 'full'}",
            "mode": tag,
            "limit_tasks_per_suite": limit,
            "base_url": BASE_URL,
            "control_model_pin": CONTROL_MODEL,
            "control_prompt": "answerer_frozen (pinq.promptlib), FrozenLLMAnswerer, ANSWER_MAX_TOKENS=%d"
            % ANSWER_MAX_TOKENS,
            "opus_model": OPUS,
            "opus_max_tokens": OPUS_MAX_TOKENS,
            "opus_max_tokens_ladder": list(OPUS_MAX_TOKENS_LADDER),
            "temperature": 0,
            "evidence_only_template_sha256": template_sha(EVIDENCE_ONLY),
            "closed_book_template_sha256": template_sha(CLOSED_BOOK),
            "window_chars": UNIT_CHARS,
            "scorer": "scripts.answerer_strength.lib.score_answer (pi_eval.metrics.quality) against pi_eval.gold v1",
            "rows": str(rows_path.relative_to(REPO)),
            "cache": "artifacts/answer_loss_20260923/cache/{control,opus}",
        },
        "population_checks": pop["checks"],
        "calls": {k: v for k, v in calls.items() if k != "rows"},
        "errors": [r for r in calls["rows"] if not r.get("ok")][:50],
        "models_returned": sorted(
            {
                str(r.get("model_returned"))
                for r in calls["rows"]
                if r.get("condition") != "control" and r.get("ok")
            }
        ),
        "control_models_recorded": sorted(
            {
                str(r.get("model"))
                for r in calls["rows"]
                if r.get("condition") == "control" and r.get("ok")
            }
        ),
        "long_unit_check": long_unit_check(pop, scored),
        "opus_finish_reasons": _finish_reasons(calls["rows"]),
        "opus_max_tokens_used": {
            f"{c}::{m}": sum(
                1
                for r in calls["rows"]
                if r.get("ok") and r["condition"] == c and r.get("max_tokens") == m
            )
            for c in ("evidence_only", "closed_book")
            for m in OPUS_MAX_TOKENS_LADDER
        },
        "n_empty_answers_by_condition": {
            c: sum(
                1
                for r in calls["rows"]
                if r.get("ok") and r["condition"] == c and not r["text"].strip()
            )
            for c in ("control", "evidence_only", "closed_book")
        },
        "per_suite": reading(pop, scored, stored),
    }
    path = OUT / ("reanswer_dryrun.json" if args.dry_run else "reanswer.json")
    path.write_text(json.dumps(out, indent=1, sort_keys=True, default=str) + "\n")
    print(f"wrote {path}", flush=True)
    for suite, so in out["per_suite"].items():
        rd = so["readings"]
        print(
            suite,
            "repro",
            {
                k: so["control_reproduction"][k]
                for k in ("n", "exact_rate", "request_sha_match_rate")
            },
            "scorer_lock",
            so["scorer_lock"],
        )
        for r in ("own", "prefix"):
            for m in ("f1", "correct"):
                c = rd[f"{r}::DiD::{m}"]
                print(
                    f"  {r} DiD {m}: {c['point']:+.4f} [{c['ci_lo']:+.4f},{c['ci_hi']:+.4f}] {c['verdict']}"
                )
    return 0


if __name__ == "__main__":
    sys.exit(main())
