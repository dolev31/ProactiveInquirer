"""Rebuild `runs/` from the parquet compaction plus the response cache. No API calls.

WHY THIS EXISTS. `rm -rf` took the whole run corpus (~34k directories). There is no snapshot
and no backup. But `pi compact` had written every field a run directory holds into columnar
form, and the LLM response cache is keyed on `request_sha`, so the directories are
RECONSTRUCTIBLE rather than lost:

  manifest.json  <- runs.parquet (+ branch_turn_idx, which only turns.parquet carries)
  turns.jsonl    <- turns.parquet   (incl. draft_text, subset_hash_before/after, response_text)
  ledger.jsonl   <- ledger.parquet
  status.json    <- runs.parquet    (stop_reason, wall_ms, usage)
  outcome.json   <- cache[answerer request_sha]["text"], plus cited_unit_ids, which is
                    `tuple(u.uid for u in ev.units)` in components.py -- every retrieved unit,
                    so it is the union of the run's retrieved_uids and needs no cache at all.

THE ONE FIELD THAT IS NOT IN THE COMPACTION: `prompt_hashes`. `_pins_sha` digests
(model_pin_hash, prompt_hashes) and the exporter refuses to pair across instruments -- a
guard that was measured to matter (266 cross-era pairs, newer side chosen 35.3%, p=1e-6).
Recomputing today's prompt hashes for every run would ERASE the era distinction and silently
re-admit exactly those pairs. So the era is carried by `code_version` instead, which changes
at least as often as the prompts do: it splits some cohorts that shared prompts, which
refuses pairs that would have been fine, and never merges two that differed. Conservative in
the only direction that is safe.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
PARQ = REPO / "scores" / "parquet"
CACHE = REPO / "cache"


def _l(v):
    """parquet gives numpy arrays for list columns; json cannot serialise them."""
    if v is None:
        return []
    if hasattr(v, "tolist"):
        return list(v.tolist())
    return list(v) if isinstance(v, (list, tuple)) else []


def _s(v):
    if v is None or (isinstance(v, float) and v != v):
        return ""
    return str(v)


def _i(v, d=0):
    try:
        if v is None or (isinstance(v, float) and v != v):
            return d
        return int(v)
    except (TypeError, ValueError):
        return d


def restore(out: Path, parq: Path = PARQ, cache: Path = CACHE, limit: int = 0) -> dict:
    """Rebuild run directories under `out`. Returns counts; makes no network calls."""
    runs = pd.read_parquet(parq / "runs.parquet")
    turns = pd.read_parquet(parq / "turns.parquet")
    ledger = pd.read_parquet(parq / "ledger.parquet")
    calls = pd.read_parquet(parq / "calls.parquet")
    ans = calls[calls.actor == "answerer"][["run_id", "request_sha"]].drop_duplicates("run_id")
    ans_sha = dict(zip(ans.run_id, ans.request_sha))

    turns_by = {k: v.sort_values("turn_idx") for k, v in turns.groupby("run_id", sort=False)}
    led_by = {k: v.sort_values("row_idx") for k, v in ledger.groupby("run_id", sort=False)}

    OUT = out
    LIMIT = limit
    CACHE_ = cache
    OUT.mkdir(parents=True, exist_ok=True)
    n = miss_ans = no_turns = 0
    rows = runs.itertuples(index=False)
    for r in rows:
        if LIMIT and n >= LIMIT:
            break
        rid = str(r.run_id)
        d = OUT / rid
        d.mkdir(exist_ok=True)
        tdf = turns_by.get(rid)
        trows = []
        if tdf is None:
            no_turns += 1
        else:
            for t in tdf.itertuples(index=False):
                trows.append(
                    {
                        "turn_idx": _i(t.turn_idx),
                        "action_kind": _s(t.action_kind) or "ask",
                        "question": _s(t.question),
                        "question_id": _s(t.question_id),
                        "rationale": _s(t.rationale),
                        "target": _s(t.target),
                        "parent_uids": _l(t.parent_uids),
                        "retrieved_uids": _l(t.retrieved_uids),
                        "new_uids": _l(t.new_uids),
                        "n_retrieved": _i(t.n_retrieved),
                        "n_new": _i(t.n_new),
                        "response_text": _s(t.response_text),
                        "draft_sha": _s(t.draft_sha),
                        "draft_text": _s(t.draft_text),
                        "subset_hash_before": _s(t.subset_hash_before),
                        "subset_hash_after": _s(t.subset_hash_after),
                        "usage": {
                            "tok_prompt": _i(t.usage_tok_prompt),
                            "tok_completion": _i(t.usage_tok_completion),
                            "tok_total": _i(t.usage_tok_total),
                            "usd": float(t.usage_usd or 0.0),
                            "wall_ms": _i(t.usage_wall_ms),
                            "n_calls": _i(t.usage_n_calls),
                        },
                    }
                )
        (d / "turns.jsonl").write_text("\n".join(json.dumps(x) for x in trows))

        # branch_turn_idx lives ONLY on the turn rows, never on runs.parquet.
        bti = None
        if tdf is not None and len(tdf):
            v = tdf.iloc[0].get("branch_turn_idx")
            if v is not None and v == v:
                bti = int(v)
        man = {
            "run_id": rid,
            "semantic_hash": _s(r.semantic_hash),
            "model_pin_hash": _s(r.model_pin_hash),
            # See the module docstring: era carried by code_version, not by today's prompts.
            "prompt_hashes": {"__rehydrated_code_version": _s(r.code_version)},
            "suite_id": _s(r.suite_id),
            "task_id": _s(r.task_id),
            "arm_id": _s(r.arm_id),
            "policy_id": _s(r.policy_id),
            "seed": _i(r.seed),
            "split": _s(r.split),
            "corpus_hash": _s(r.corpus_hash),
            "corpus_dir": _s(r.corpus_dir),
            "template_id": None if _s(r.template_id) == "" else _s(r.template_id),
            "budget_cap": None
            if r.budget_cap is None or r.budget_cap != r.budget_cap
            else int(r.budget_cap),
            "max_turns": None
            if r.max_turns is None or r.max_turns != r.max_turns
            else int(r.max_turns),
            "word_cap": _i(r.word_cap, 30),
            "code_version": _s(r.code_version),
            "dirty": bool(r.dirty),
            "is_dev_run": bool(r.is_dev_run),
            "exploratory": bool(r.exploratory),
            "gold_exposed": bool(r.gold_exposed),
            "branch_of_run_id": None if _s(r.branch_of_run_id) == "" else _s(r.branch_of_run_id),
            "branch_turn_idx": bti,
            "grid_name": _s(r.grid_name),
            "status": _s(r.status) or "ok",
            "stop_reason": _s(r.stop_reason),
            "rehydrated_from": "scores/parquet + cache",
        }
        (d / "manifest.json").write_text(json.dumps(man, indent=1, sort_keys=True))

        ldf = led_by.get(rid)
        lrows = []
        if ldf is not None:
            for x in ldf.itertuples(index=False):
                lrows.append(
                    {
                        "currency": _s(x.currency),
                        "charged": float(x.charged or 0.0),
                        "cumulative": float(x.cumulative or 0.0),
                        "turn_idx": _i(x.turn_idx),
                        "cap": float(x.cap) if x.cap == x.cap else None,
                        "hard": bool(x.hard),
                    }
                )
        (d / "ledger.jsonl").write_text("\n".join(json.dumps(x) for x in lrows))

        (d / "status.json").write_text(
            json.dumps(
                {
                    "ok": _s(r.status) in ("", "ok"),
                    "status": _s(r.status) or "ok",
                    "error": _s(r.error),
                    "stop_reason": _s(r.stop_reason),
                    "wall_ms": _i(r.wall_ms),
                    "usage": {
                        "tok_prompt": _i(r.tok_prompt),
                        "tok_completion": _i(r.tok_completion),
                        "tok_reasoning": _i(r.tok_reasoning),
                        "tok_cached": _i(r.tok_cached),
                        "tok_total": _i(r.tok_total),
                        "usd": float(r.usd or 0.0),
                    },
                }
            )
        )

        text = None
        sha = ans_sha.get(rid)
        if sha:
            p = CACHE_ / str(sha)[:2] / f"{sha}.json"
            if p.exists():
                try:
                    text = json.loads(p.read_text()).get("text")
                except json.JSONDecodeError:
                    text = None
        if text is None:
            miss_ans += 1
        # cited_unit_ids is every retrieved unit (components.py:603), so it is the union of
        # the run's own retrieved_uids -- derived, not cached.
        cited: list[str] = []
        seen: set[str] = set()
        for t in trows:
            for u in t["retrieved_uids"]:
                if u not in seen:
                    seen.add(u)
                    cited.append(u)
        (d / "outcome.json").write_text(
            json.dumps(
                {
                    "answer": None
                    if text is None
                    else {
                        "text": text,
                        "evidence_hash": _s(r.answer_evidence_hash),
                        "cited_unit_ids": cited,
                        "n_words": _i(r.answer_n_words),
                        "stop_reason": _s(r.stop_reason),
                    },
                    "env_calls": [],
                    "env_final_hashes": {},
                    "native": {},
                    "artifacts": {},
                    "stop_reason": _s(r.stop_reason),
                    "subset_hash": _s(r.subset_hash),
                }
            )
        )
        n += 1
        if n % 2000 == 0:
            print(f"  {n:,} runs rehydrated", flush=True)

    return {"runs": n, "no_answer_text": miss_ans, "no_turns": no_turns, "out": str(OUT)}


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "runs"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    res = restore(out, limit=limit)
    print(json.dumps(res, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
