"""runs/ -> scores/parquet/. Single-threaded, schema-asserted, idempotent.

This module runs in the SCORING process, which is the one process permitted to import
pi_eval. That is why the frozen column contract lives in pi_eval.schema and is imported here
but never from pi_run.worker: a rollout worker with gold-side code in its address space would
make the process split decorative.

Single-threaded on purpose. Compaction is I/O over a few thousand small files and finishes in
seconds; parallelising it would buy nothing and would reintroduce exactly the ordering
nondeterminism the sweep runner goes out of its way to eliminate. Run directories are walked
in sorted order so two compactions of the same runs/ produce byte-comparable tables.

Rows are passed to the validator EXACTLY as they were written, with only run_id added. No
key filtering, no defaulting: a field the writer invented must fail the schema assertion
rather than being quietly dropped here, because dropping it is how a real column disappears
between the run and the table. The one adjustment allowed is `_migrate`, which RENAMES a field
whose name changed and BACKFILLS a field that did not yet exist -- both of which preserve
every value, neither of which can invent one.

"WITH ONLY run_id ADDED" IS AN OWNERSHIP CLAIM, AND IT HAS TO BE ENFORCED. A run id is the
name of the directory, and for four days it was not: `{"run_id": run_id, **row}` lets the
record's own key win, and 21,001 rehydrated directories carry `"run_id": null` in every turn
record. See `_keyed`, and the gate that now counts what that cost.

THE WRITE IS ALL OR NOTHING. Seven tables are validated and staged before any of them is
renamed into place, because a compaction that commits some tables and abandons the rest does
not fail -- it produces a parquet directory whose tables describe two different corpora and
whose every downstream inner join reads it without complaint. See the staging block for what
that cost the last time it happened.
"""

from __future__ import annotations

import json
import os
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

try:
    import pyarrow.parquet as pq
except ImportError as _exc:  # pragma: no cover - names the extra instead of the module
    from pinq.extras import missing

    raise missing("pyarrow", why="pi compact writes scores/parquet") from _exc

from pi_eval import schema as sch

DEFAULT_OUT = "scores/parquet"


@dataclass(frozen=True, slots=True)
class CompactResult:
    out_dir: str
    counts: dict[str, int]
    run_dirs: int
    skipped: tuple[str, ...]
    schema_hash: str
    # Score rows whose run is not in `runs.parquet`. Every reporting query INNER JOINs the two,
    # so an orphan is a row that exists on disk and appears in no table -- invisible unless it
    # is counted here.
    orphaned_score_rows: int = 0
    # Runs whose turns.jsonl holds records and whose turn rows reached no table. The mirror
    # image of an orphan, and the one this pipeline had no instrument for: see `_keyed`.
    runs_with_turns_on_disk_but_none_compacted: tuple[str, ...] = ()
    # Prior rows the merge dropped because their run_id was NULL or "": unjoinable by
    # construction and unreplaceable by any future pass. Per table, and never silent.
    prior_rows_dropped_with_no_run_id: Mapping[str, int] = types.MappingProxyType({})

    def as_dict(self) -> dict[str, Any]:
        return {
            "out_dir": self.out_dir,
            "counts": dict(self.counts),
            "run_dirs": self.run_dirs,
            "skipped": list(self.skipped),
            "schema_hash": self.schema_hash,
            "orphaned_score_rows": self.orphaned_score_rows,
            "n_runs_with_turns_on_disk_but_none_compacted": len(
                self.runs_with_turns_on_disk_but_none_compacted
            ),
            # Capped: the operator needs to be able to go and look, not to read 21,001 ids.
            "runs_with_turns_on_disk_but_none_compacted": list(
                self.runs_with_turns_on_disk_but_none_compacted[:20]
            ),
            "prior_rows_dropped_with_no_run_id": dict(self.prior_rows_dropped_with_no_run_id),
        }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _n_records(path: Path) -> int:
    """Non-empty lines, counted from the FILE and not from the parsed rows.

    Deliberately a second, independent read. The loss gate below compares this against the
    run ids that reached the committed table; derived from the same parse, it would be a
    quantity that could not take the other value, and a check that cannot fail is
    indistinguishable from a clean pass.
    """
    if not path.exists():
        return 0
    with path.open("rb") as fh:
        return sum(1 for line in fh if line.strip())


def _keyed(table: str, row: dict[str, Any], **owned: Any) -> dict[str, Any]:
    """Attach the columns the COMPACTION owns, overriding what the record carries.

    THIS IS NOT `{"run_id": run_id, **row}`, AND THE DIFFERENCE COST 21,001 RUNS. A dict
    literal lets the LATER key win, so `{"run_id": run_id, **row}` reads as "add the run id"
    and means "let the record overwrite it". Measured on the shared tree 2026-09-18: 21,001
    run directories rehydrated on 2026-09-03 (`manifest.rehydrated_from == "scores/parquet +
    cache"`) hold parquet-shaped turn records whose first key is `"run_id": null`, and all
    134,370 of their turn rows reached `turns.parquet` with a NULL run_id. `validate_rows` is
    strict about a key being PRESENT and silent about its value, so nothing raised; every
    reporting query INNER JOINs on run_id, so the rows were in the table and in no result.
    `pi score` read those runs as n_turns=0 / n_asks=0 / evidence_coverage=0.0, and 382,916 of
    L0.6's 656,301 scorer-hash inequalities are that and nothing else.

    The DIRECTORY is the authority: a run id is the name of the directory the records were
    read from. A record that names a DIFFERENT run is a directory holding two runs' rows,
    which is a fact worth a stack trace rather than a silent relabelling, so it raises.
    """
    out = dict(row)
    for key, value in owned.items():
        carried = out.get(key)
        if carried is not None and carried != "" and carried != value:
            raise sch.SchemaViolation(
                f"table {table!r}: record carries {key}={carried!r} but was read from the "
                f"directory of {value!r}. A run directory holding another run's rows is not "
                "something this reader may relabel."
            )
        out[key] = value
    return out


def _s(v: Any) -> str | None:
    return None if v is None else str(v)


def _run_row(
    manifest: dict[str, Any],
    status: dict[str, Any],
    outcome: dict[str, Any],
    turn_records: Sequence[Mapping[str, Any]] = (),
) -> dict:
    usage = status.get("usage") or {}
    spent = status.get("spent") or {}
    rec = status.get("reconcile") or {}
    answer = (outcome.get("answer") or {}) if outcome else {}
    hashes = (outcome.get("env_final_hashes") or {}) if outcome else {}
    run_id = str(manifest["run_id"])
    # A MISSING KEY IS NOT A MEASURED ZERO, AND HERE IT WAS RECOVERABLE.
    #
    # `status.get("n_turns", 0)` reads a run that never wrote the field as a run that took no
    # turns. 21,001 directories on the shared tree are restores written by
    # `scripts/restore_corpus.py`, whose status.json is exactly
    # {ok, status, error, stop_reason, wall_ms, usage} -- no n_turns, no n_asks -- so every one
    # of them carried n_turns=0 next to a turns.jsonl holding up to 24 records. That 0 agreed
    # with the missing turn rows and made the loss look like a property of the runs.
    #
    # `worker` writes `n_turns = len(traj.turns)` and one turns.jsonl record per `traj.turns`,
    # and `n_asks = sum(isinstance(t.action, Ask))` against `action_kind == "ask"`, so both are
    # EXACTLY recoverable from the records. Status still wins when it is there: a run that
    # crashed between writing status and writing its turns is a real disagreement, not a
    # rounding error. `n_evidence`, `n_calls` and `unique_docs` are NOT repaired -- those
    # directories carry no evidence.jsonl and no calls.jsonl, so there is nothing to count and
    # a repaired value would be invented.
    n_turns = int(status["n_turns"]) if status.get("n_turns") is not None else len(turn_records)
    n_asks = (
        int(status["n_asks"])
        if status.get("n_asks") is not None
        else sum(1 for r in turn_records if r.get("action_kind") == "ask")
    )
    return {
        "run_id": run_id,
        "semantic_hash": str(manifest.get("semantic_hash", "")),
        "model_pin_hash": str(manifest.get("model_pin_hash", "")),
        "suite_id": str(manifest["suite_id"]),
        "task_id": str(manifest["task_id"]),
        "arm_id": str(manifest["arm_id"]),
        "policy_id": str(manifest.get("policy_id", "")),
        "seed": int(manifest.get("seed", 0)),
        "split": str(manifest.get("split", "")),
        "corpus_hash": str(manifest.get("corpus_hash", "")),
        "corpus_dir": str(manifest.get("corpus_dir", "")),
        "grid_sha256": str(manifest.get("grid_sha256", "")),
        "grid_name": str(manifest.get("grid_name", "")),
        "template_id": _s(manifest.get("template_id")),
        "budget_cap": int(manifest.get("budget_cap", 0)),
        "max_turns": int(manifest.get("max_turns", 0)),
        "word_cap": int(manifest.get("word_cap", 0)),
        "pilot_flag": bool(manifest.get("pilot_flag", False)),
        "exploratory": bool(manifest.get("exploratory", False)),
        "gold_exposed": bool(manifest.get("gold_exposed", False)),
        "counterfactual_kind": str(manifest.get("counterfactual_kind", "none")),
        "prefix_of_run_id": _s(manifest.get("prefix_of_run_id")),
        "prefix_k": manifest.get("prefix_k"),
        "branch_of_run_id": _s(manifest.get("branch_of_run_id")),
        # THE FORK'S STATE KEY, AND WHY IT IS HERE. `pinq_train.export.dataset.state_key` is
        # (suite, task, branch_of_run_id, branch_turn_idx); this column used to hold the first
        # half and drop the second, so a corpus rebuilt from the compaction had every fork
        # candidate in a group of one and could produce no preference pair at all. That is not
        # hypothetical: `runs/` was lost, 22,607 directories were rebuilt from these tables
        # with zero state_mismatch, and only the 8,695 forks whose turn could be read back out
        # of a surviving export were usable. `branch_seed` says WHICH candidate of the state a
        # run is. The compaction is in practice the backup, so a field the exporter needs and
        # this table drops is data that only looks safe.
        "branch_turn_idx": manifest.get("branch_turn_idx"),
        "branch_seed": manifest.get("branch_seed"),
        # THE INSTRUMENT. `_pins_sha` digests (model_pin_hash, prompt_hashes) and the exporter
        # refuses to pair across instruments -- measured: 266 cross-era pairs, newer side
        # chosen 35.3%, p = 1e-6. With only the model pin surviving, a rebuild silently
        # re-admits exactly those pairs. Canonical JSON so the column compares across runs;
        # "" when the run carried none, which a legacy run does and is not the same as {}.
        "prompt_hashes": (
            json.dumps(dict(sorted((manifest.get("prompt_hashes") or {}).items())))
            if manifest.get("prompt_hashes")
            else ""
        ),
        "permutation_id": manifest.get("permutation_id"),
        "prompt_variant_id": str(manifest.get("prompt_variant_id", "v1")),
        "train_id_set_hash": _s(manifest.get("train_id_set_hash")),
        "code_version": str(manifest.get("code_version", "")),
        "dirty": bool(manifest.get("dirty", False)),
        # Mechanically excludable, which is the entire point of the dev- prefix.
        "is_dev_run": run_id.startswith("dev-"),
        "canary_hit": bool(manifest.get("canary_hit", False)),
        "firewall_ok": bool(manifest.get("firewall_ok", True)),
        "concurrency": int(manifest.get("concurrency", 1)),
        "status": str(status.get("status", "unknown")),
        "error": _s(status.get("error")),
        "stop_reason": str(status.get("stop_reason", "")),
        "n_turns": n_turns,
        "n_asks": n_asks,
        "n_calls": int(status.get("n_calls", 0)),
        "n_evidence": int(status.get("n_evidence", 0)),
        # NULL, not 0, when the suite has no user simulator -- which is every suite but tau2.
        # `pi_eval.metrics.environment.user_turns` reads this as NaN and declines to emit,
        # because a 0 here is the BEST possible score on the anticipation endpoint and would
        # be awarded to every musique run for having no user to speak.
        "n_user_turns": (
            None if status.get("n_user_turns") is None else int(status["n_user_turns"])
        ),
        # NULL, NOT 0, WHEN ABSENT. A fork that genuinely had no prefix user turns and a suite
        # that never records the field are different facts. `environment.prefix_user_turns`
        # defaults to 0.0 on a missing key, which is how every forked run came to be charged
        # for its prefix's user turns with nothing on the row to show it had happened.
        "n_prefix_user_turns": (
            None
            if status.get("n_prefix_user_turns") is None
            else int(status["n_prefix_user_turns"])
        ),
        # NULL rather than "": unlike the count above, there is no measured value an empty
        # string could be confused with, and ELIGIBLE admits a run on `IS NULL OR = ''`, so
        # both spellings read as "one of ours". NULL is written because it is unambiguous.
        "foreign_trace_sha": str(manifest.get("foreign_trace_sha") or "") or None,
        "unique_docs": int(status.get("unique_docs", 0)),
        "retrieval_calls": float(spent.get("retrieval_calls", 0.0)),
        "tok_prompt": int(usage.get("tok_prompt", 0)),
        "tok_completion": int(usage.get("tok_completion", 0)),
        "tok_reasoning": int(usage.get("tok_reasoning", 0)),
        "tok_cached": int(usage.get("tok_cached", 0)),
        "tok_total": int(usage.get("tok_total", 0)),
        "usd": float(usage.get("usd", 0.0)),
        "wall_ms": int(usage.get("wall_ms", 0)),
        "reconciled": bool(status.get("reconciled", False)),
        "reconciled_tokens": bool(rec.get("tokens_ok", False)),
        "reconciled_docs": bool(rec.get("docs_ok", False)),
        "answer_n_words": int(answer.get("n_words", 0)),
        "answer_evidence_hash": str(answer.get("evidence_hash", "")),
        "subset_hash": str(outcome.get("subset_hash", "")) if outcome else "",
        # The reward substrate. `Actuator.final_hashes()` promises exactly these two keys and
        # promises "" rather than None for a missing one; the same promise is kept here so the
        # column is non-nullable end to end.
        "env_db_hash": str(hashes.get("db", "")),
        "env_user_db_hash": str(hashes.get("user_db", "")),
    }


def _native_rows(manifest: dict[str, Any], outcome: dict[str, Any]) -> list[dict[str, Any]]:
    """Outcome.native -> one row per (run, key).

    A non-numeric value is DROPPED rather than coerced: `float("ok")` raises, `float(True)` is
    1.0, and a native field that is not a number is not a measurement this table can hold. The
    drop is silent by design — the alternative is a compaction that dies on one suite's
    diagnostic string and takes the other five suites' rows with it.
    """
    rows: list[dict[str, Any]] = []
    for key, value in sorted((outcome.get("native") or {}).items()):
        try:
            v = float(value)
        except (TypeError, ValueError):
            continue
        rows.append(
            {
                "run_id": str(manifest["run_id"]),
                "suite_id": str(manifest["suite_id"]),
                "key": str(key),
                "value": v,
            }
        )
    return rows


def iter_run_dirs(runs_root: Path) -> Iterator[Path]:
    if not runs_root.exists():
        return
    for d in sorted(runs_root.iterdir()):
        if d.is_dir() and (d / "status.json").exists():
            yield d


# Column renames this reader accepts from artifacts written by an older commit. The parquet is
# merged with what is already on disk, so a rename that only touched the WRITER would make the
# strict schema reject every previously compacted row -- turning a naming fix into a demand that
# everyone rebuild.
_RENAMED: dict[str, dict[str, str]] = {
    # `response_sha` held the Drafter's reply TEXT and always did: measured on the shipped
    # turns.parquet, 53 non-empty values and 0 matching [0-9a-f]{16,64}, while the CALLS table
    # used the same name for a real digest. Renamed, contents unchanged.
    "turns": {"response_sha": "response_text"},
    # NOTE: `draft_text` is not a RENAME -- a pre-field run carries a draft_sha and no text,
    # so there is nothing to rename it from. It is BACKFILLED empty by `_migrate` instead,
    # and the two guards then divide the work cleanly:
    #
    #   * the schema is STRUCTURAL -- every declared column present, so a NULL cannot make an
    #     aggregate silently skip rows. 273 run dirs predate the field; refusing them here
    #     kills compaction for the whole repo.
    #   * `cmd_train.render_state` is SEMANTIC -- it refuses exactly `draft_sha and not
    #     draft_text`, so an empty string is the SIGNAL it keys on, not a disguise. Those
    #     turns stay unexportable until re-run, which is the intended outcome.
}

# Fields added after runs were already on disk. Backfilled to the type's empty value so the
# structural contract holds; a consumer that needs the real value tests for empty itself.
_BACKFILL: dict[str, dict[str, object]] = {
    "turns": {"draft_text": ""},
    # `EVIDENCE.text` is written only for a request-addressed (`call:`) unit -- see the column's
    # comment in pi_eval.schema for the measurement. Every corpus-addressed row, and every row
    # written before the field existed, gets "" so the STRUCTURAL half of the contract holds:
    # a declared column missing from a row would otherwise reject the whole corpus, and
    # NULL-filling it would let an aggregate skip those rows without saying so. This backfill
    # covers the read-back path too, so an evidence.parquet compacted before the column existed
    # still merges instead of demanding that `runs/` -- in practice the only backup -- be
    # rebuilt.
    "evidence": {"text": ""},
}


def _migrate(table: str, row: dict) -> dict:
    ren = _RENAMED.get(table)
    fill = _BACKFILL.get(table)
    if not ren and not fill:
        return row
    out = dict(row)
    for name, empty in (fill or {}).items():
        out.setdefault(name, empty)
    if not ren:
        return out
    for old_name, new_name in ren.items():
        if old_name in out and new_name not in out:
            out[new_name] = out.pop(old_name)
        else:
            out.pop(old_name, None)
    return out


def compact(
    runs_root: str | Path = "runs",
    out_dir: str | Path = DEFAULT_OUT,
    *,
    include_dev: bool = True,
    replace: bool = False,
) -> CompactResult:
    runs_root = Path(runs_root)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows: dict[str, list[dict[str, Any]]] = {name: [] for name in sch.POPULATED}
    skipped: list[str] = []
    n_dirs = 0
    # run_id -> non-empty lines in its turns.jsonl, read from the FILE. Only directories this
    # pass actually ingested are in here: a dir skipped for a missing manifest.json produces no
    # rows BY DESIGN, and counting it would make the gate fire on an intended condition, which
    # is how a check gets disabled.
    turn_lines_on_disk: dict[str, int] = {}

    for d in iter_run_dirs(runs_root):
        n_dirs += 1
        status = _read_json(d / "status.json")
        manifest_path = d / "manifest.json"
        if not manifest_path.exists():
            skipped.append(d.name)
            continue
        manifest = _read_json(manifest_path)
        run_id = str(manifest["run_id"])
        if not include_dev and run_id.startswith("dev-"):
            skipped.append(d.name)
            continue

        outcome = _read_json(d / "outcome.json") if (d / "outcome.json").exists() else {}
        turn_records = _read_jsonl(d / "turns.jsonl")
        rows["runs"].append(_run_row(manifest, status, outcome, turn_records))

        turn_lines_on_disk[run_id] = _n_records(d / "turns.jsonl")
        for r in turn_records:
            # `response_sha` is the PRE-RENAME key for `response_text`. The field was renamed
            # because it holds the Drafter's reply TEXT and always did -- measured on the
            # shipped turns.parquet, 53 non-empty values and 0 matching [0-9a-f]{16,64}, in a
            # schema whose CALLS table uses the same name for a real digest. The contents did
            # not change, so run directories already on disk stay readable rather than being
            # rejected by the strict schema.
            rows["turns"].append(_keyed("turns", _migrate("turns", r), run_id=run_id))
        for r in _read_jsonl(d / "calls.jsonl"):
            rows["calls"].append(_keyed("calls", r, run_id=run_id))
        for r in _read_jsonl(d / "evidence.jsonl"):
            # `text` is present only on the `call:` units of runs rolled between 2026-09-06 and
            # 2026-09-09, and absent everywhere else; `_migrate` supplies "" for the absent case
            # so both eras satisfy one schema. Nothing is dropped here -- see _BACKFILL.
            rows["evidence"].append(_keyed("evidence", _migrate("evidence", r), run_id=run_id))
        for i, r in enumerate(_read_jsonl(d / "ledger.jsonl")):
            rows["ledger"].append(_keyed("ledger", r, run_id=run_id, row_idx=i))
        for r in outcome.get("env_calls", []) or []:
            rows["env_calls"].append(_keyed("env_calls", r, run_id=run_id))
        rows["native"] += _native_rows(manifest, outcome)

    counts: dict[str, int] = {}
    # ALL OR NOTHING ACROSS THE SEVEN TABLES, AND THIS IS THE WHOLE POINT OF THE STAGING.
    #
    # `sch.POPULATED` is walked in order and each table used to be written the instant it
    # validated. So a SchemaViolation on the FOURTH table committed the first three and
    # abandoned the last four -- and the failure is not "the compaction did not run", it is a
    # parquet directory describing two different corpora at once. Measured on the live tree:
    # `evidence` refused an undeclared `text` column on 2026-09-15, and runs / turns / calls
    # were left stamped 11:11 while evidence / env_calls / ledger / native stayed at 2026-09-09
    # 00:26. For six days `pi score`, `pi train gate`, the contamination checks and `pi render`
    # inner-joined a fresh `runs` against a six-day-old `evidence` and `ledger`. Nothing
    # errored: each table was internally valid, and only their disagreement was wrong.
    #
    # Staged beside the target and renamed at the end, rather than held in memory and written
    # at the end: `os.replace` is atomic per file and the whole batch takes microseconds, while
    # buffering seven arrow tables for a 70,000-run corpus is hundreds of megabytes that the
    # merge path (which already materialises each prior table as python dicts) cannot spare.
    # The failure this exists to contain -- an undeclared column -- happens in `to_table`,
    # before any rename, so a violation now leaves every table byte-for-byte as it was.
    staged: list[tuple[Path, Path]] = []
    all_validated = False
    compacted_turn_ids: set[str] = set()
    dropped_unkeyed: dict[str, int] = {}
    try:
        for name in sch.POPULATED:
            # MERGED BY run_id, NOT REPLACED.
            #
            # Compaction owns the six run-side tables, and "owns" used to mean "overwrites with
            # whatever --runs-root currently holds". But `scores.parquet` is deliberately
            # preserved (see EMPTY_BY_DESIGN below), and every reporting query INNER JOINs
            # scores -> runs. So compacting a DIFFERENT runs-root into the same parquet --
            # re-running one suite, pointing at a subset, recovering from a partial sweep --
            # silently orphaned every score row whose run had just been evicted from `runs`.
            #
            # Measured: sweep A -> compact -> score gives 3 runs and 228 joinable score rows.
            # Compacting sweep B into the same directory gives 3 runs, 228 score rows, and 0
            # joinable. Every table built from it is empty, and nothing anywhere errors.
            #
            # Merging makes the run side behave like the score side: additive, keyed on run_id,
            # idempotent. A row for a run this pass DID see is replaced by the fresh one; a row
            # for a run it did not see is kept. `replace=True` restores the old behaviour for
            # the one case that wants it -- deliberately dropping runs -- and says what it
            # orphaned.
            # AND A PRIOR ROW WITH NO run_id IS DROPPED, NOT KEPT.
            #
            # The merge keeps a prior row when its run_id is not in the set this pass saw. No
            # run is named `None`, so a row whose key is NULL is never in that set, can never
            # be replaced by a fresh one, and joins to nothing: it would survive every
            # compaction there will ever be. Measured on the live store, 2026-09-18: the first
            # re-compaction under `_keyed` added all 134,370 recovered turn rows AND kept all
            # 134,370 NULL-keyed ones beside them. The two are content-identical modulo run_id
            # -- `except` in both directions over the other 33 columns returns 0 of 134,370 --
            # so the prior row carries nothing the fresh one does not, and what it does carry
            # is a wrong `count(*)` waiting for someone who does not inner-join.
            #
            # DROPPED WITH A COUNT, never silently: a row vanishing from a table is exactly
            # the class of event this module exists to make loud.
            fresh = rows[name]
            path = out / f"{name}.parquet"
            if path.exists() and not replace:
                seen = {str(r.get("run_id", "")) for r in fresh}
                prior = []
                n_unkeyed = 0
                for r in pq.read_table(path).to_pylist():
                    rid = r.get("run_id")
                    if rid is None or rid == "":
                        n_unkeyed += 1
                        continue
                    if str(rid) not in seen:
                        prior.append(r)
                if n_unkeyed:
                    dropped_unkeyed[name] = n_unkeyed
                prior = [_migrate(name, r) for r in prior]
                fresh = prior + fresh
            table = sch.to_table(name, fresh)  # raises SchemaViolation on any drift
            if name == "turns":
                compacted_turn_ids = {i for i in table.column("run_id").to_pylist() if i}
            tmp = path.parent / f"{path.name}.tmp-{os.getpid()}"
            pq.write_table(table, tmp)
            staged.append((tmp, path))
            counts[name] = table.num_rows
        all_validated = True
    finally:
        # A staging file must not outlive the attempt that wrote it. Left behind, it is a file
        # a later glob or a later rename can pick up -- the same tearing, one indirection out.
        if not all_validated:
            for tmp, _ in staged:
                tmp.unlink(missing_ok=True)
    # Every table validated. The renames are the commit.
    for tmp, path in staged:
        os.replace(tmp, path)

    for name in sch.EMPTY_BY_DESIGN:
        # Declared here, POPULATED BY pi score. Compaction owns the six run-side tables and
        # nothing else: an existing file is left byte-for-byte alone, because rewriting it
        # empty would silently discard every score row the moment a new rollout landed —
        # exactly the append-only guarantee pi_eval.score exists to make.
        path = out / f"{name}.parquet"
        if path.exists():
            counts[name] = pq.read_metadata(path).num_rows
            continue
        # First compaction: write the empty file so a downstream query is a SELECT returning
        # zero rows rather than a missing file.
        pq.write_table(sch.empty_table(name), path)
        counts[name] = 0

    # THE INVARIANT, CHECKED RATHER THAN ASSUMED: no score row may become unreachable. This is
    # reported even when merging, because a --replace, a deleted run directory or an
    # --exclude-dev can all orphan rows, and an orphan is invisible in every table it should
    # have appeared in.
    orphaned = 0
    scores_path = out / "scores.parquet"
    if scores_path.exists():
        live = {str(r.get("run_id")) for r in pq.read_table(out / "runs.parquet").to_pylist()}
        orphaned = sum(
            1 for r in pq.read_table(scores_path).to_pylist() if str(r.get("run_id")) not in live
        )

    # THE LOSS GATE. Turn lines exist on disk for this run and the committed table holds no row
    # keyed to it: the rows are gone from every inner join, and nothing else in this pipeline
    # says so -- `runs.n_turns` comes from status.json, not from the rows, so it agrees with the
    # loss instead of contradicting it. Measured against the FILE, not against the parse the
    # rows came from (see `_n_records`). A run whose turns.jsonl is genuinely empty is never in
    # `turn_lines_on_disk` at all, so a no-ask arm -- drafter_only is 780 of 780 empty on tau2,
    # 836 of 836 on musique -- cannot make this fire.
    lost = tuple(
        sorted(rid for rid, n in turn_lines_on_disk.items() if n and rid not in compacted_turn_ids)
    )

    return CompactResult(
        out_dir=str(out),
        counts=counts,
        orphaned_score_rows=orphaned,
        run_dirs=n_dirs,
        skipped=tuple(skipped),
        schema_hash=sch.schema_hash(),
        runs_with_turns_on_disk_but_none_compacted=lost,
        prior_rows_dropped_with_no_run_id=types.MappingProxyType(dict(dropped_unkeyed)),
    )


def read_counts(out_dir: str | Path = DEFAULT_OUT) -> dict[str, int]:
    out = Path(out_dir)
    return {
        name: pq.read_metadata(out / f"{name}.parquet").num_rows
        for name in sch.TABLES
        if (out / f"{name}.parquet").exists()
    }


def table_names() -> Sequence[str]:
    return tuple(sch.TABLES)
