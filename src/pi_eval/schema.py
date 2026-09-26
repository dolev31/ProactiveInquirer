"""The frozen parquet column contract. Explicit pyarrow schemas, asserted by `pi compact`.

WHY THIS IS STRICT IN BOTH DIRECTIONS. `pi compact` fails on an unknown column AND on a
missing one. Neither failure mode is pedantry:

  * An UNKNOWN column means the writer learned a field the reader does not know about.
    Parquet will happily accept it, DuckDB will happily union it in as NULL for every older
    file, and the first join against a table with a partially-populated key produces a silent
    Cartesian blow-up. The symptom appears three tables downstream as a number that is merely
    surprising, not obviously wrong.

  * A MISSING column means a run wrote fewer fields than the contract promises. Filling it
    with NULL would let `mean(usd)` quietly skip those rows, and a cost table that omits the
    expensive arm is worse than no cost table.

`matches`, `judgments` and `scores` are declared here, created EMPTY by `pi compact` and
POPULATED by `pi score`. Compaction never rewrites an existing one, so the append-only
guarantee in `pi_eval.score` survives a new rollout landing after a scoring pass.

Column-name discipline: gold-side columns keep their gold_ prefix, so a join that accidentally
mixes a run-side and gold-side field cannot silently unify on a shared name.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

try:
    import pyarrow as pa
except ImportError as _exc:  # pragma: no cover - names the extra instead of the module
    from pinq.extras import missing

    raise missing("pyarrow", why="the frozen parquet column contracts") from _exc

from pinq.ids import canon, h

SCHEMA_VERSION = "v1"

_str = pa.string()
_i32 = pa.int32()
_i64 = pa.int64()
_f64 = pa.float64()
_bool = pa.bool_()
_strs = pa.list_(pa.string())


class SchemaViolation(ValueError):
    """A row does not match the frozen contract. Always fatal to the compaction."""


# --------------------------------------------------------------------------- run side

RUNS = pa.schema(
    [
        ("run_id", _str),  # PK. 'dev-' prefix => produced from a dirty tree, excluded by the agg
        ("semantic_hash", _str),
        ("model_pin_hash", _str),
        ("suite_id", _str),
        ("task_id", _str),
        ("arm_id", _str),
        ("policy_id", _str),
        ("seed", _i32),
        ("split", _str),
        ("corpus_hash", _str),
        # The content-addressed DIRECTORY the corpus lived in. A different string from
        # corpus_hash and for a different purpose: this is what `load_questions` opens.
        ("corpus_dir", _str),
        ("template_id", _str),
        ("budget_cap", _i32),
        ("max_turns", _i32),
        ("word_cap", _i32),
        ("pilot_flag", _bool),
        ("exploratory", _bool),  # a run that exists to be looked at is inadmissible
        ("gold_exposed", _bool),
        ("counterfactual_kind", _str),
        ("prefix_of_run_id", _str),
        ("prefix_k", _i32),
        ("branch_of_run_id", _str),
        # THE OTHER HALF OF THE FORK'S STATE KEY. `state_key` is (suite, task,
        # branch_of_run_id, branch_turn_idx); the runs table carried only the first, so a
        # corpus rebuilt from these files had every candidate in a group of one and could
        # produce no preference pair. The turns table declares `branch_turn_idx` already and
        # `compact` never populated it (0 of 134,378 rows). See
        # tests/test_compaction_keeps_the_fork_key.py for what that cost.
        ("branch_turn_idx", _i32),
        # int64: a branch seed is a 32-bit-overflowing hash (measured 3,229,923,555 on the
        # live corpus), and pyarrow refuses it as an int32 rather than truncating -- which is
        # the schema guard doing its job.
        ("branch_seed", _i64),
        # `_pins_sha` digests (model_pin_hash, prompt_hashes); the exporter refuses to pair
        # across instruments, and with only the model surviving a rebuild re-admits exactly
        # the cross-era pairs that guard was measured to catch. Canonical JSON.
        ("prompt_hashes", _str),
        ("permutation_id", _i32),
        ("prompt_variant_id", _str),
        ("train_id_set_hash", _str),
        ("code_version", _str),
        ("dirty", _bool),
        ("is_dev_run", _bool),
        ("canary_hit", _bool),
        ("firewall_ok", _bool),
        ("concurrency", _i32),
        ("status", _str),
        ("error", _str),
        ("stop_reason", _str),
        ("n_turns", _i32),
        ("n_asks", _i32),
        ("n_calls", _i32),
        ("n_evidence", _i32),
        ("n_user_turns", _i32),  # NULL on every suite without a user simulator
        # HOW MUCH OF THAT DIALOGUE THE RUN INHERITED. `score_run` reads this parquet and
        # never the run directory, so a field compaction does not carry is unreachable
        # however faithfully status.json recorded it -- and `environment.prefix_user_turns`
        # falls back to 0.0 when absent, which silently charges every forked run for the
        # user turns of the prefix it was handed. `fork_report.follow_ups` is
        # `n_user_turns - n_prefix_user_turns`, so this column is half of the headline.
        # NULL and not 0 when absent, for the reason on the line above it.
        ("n_prefix_user_turns", _i32),
        # The only way a table reader can tell a forked or trace-derived run from one of
        # ours. `report.ELIGIBLE` refuses a run on exactly this column, because a fork
        # inherits a foreign prefix and must never pool with our own arms under one task_id.
        ("foreign_trace_sha", _str),
        ("grid_sha256", _str),  # PROVENANCE: which grid file, exact bytes. Not identity.
        ("grid_name", _str),
        ("unique_docs", _i32),
        ("retrieval_calls", _f64),
        # Denormalized rollups. `calls` carries the fine grain; these exist so the metric
        # contract is a groupby on one table rather than a join for every cost number.
        ("tok_prompt", _i64),
        ("tok_completion", _i64),
        ("tok_reasoning", _i64),
        ("tok_cached", _i64),
        ("tok_total", _i64),
        ("usd", _f64),
        ("wall_ms", _i64),
        ("reconciled", _bool),
        ("reconciled_tokens", _bool),
        ("reconciled_docs", _bool),
        ("answer_n_words", _i32),
        ("answer_evidence_hash", _str),
        ("subset_hash", _str),
        # Outcome.env_final_hashes, under the two keys `pinq_adapters.tau2.Tau2Actuator`
        # promises. These are the SUBSTRATE of the tau2 primary endpoint: the reward is
        # equality of the agent-DB and user-DB hashes against a gold environment, and a table
        # that carried the verdict but not the hashes could not answer "which of the two
        # halves differed?" without a re-run. "" for every suite with no environment, never
        # NULL, so a join on them cannot silently drop a row.
        ("env_db_hash", _str),
        ("env_user_db_hash", _str),
    ]
)

# Outcome.native, LONG rather than wide, and this is the only shape that works. A suite's
# native reward is not one number: tau2 alone produces `tau_reward`, `db_reward`, `db_match`
# and one `breakdown.<component>` per reward component, and PARE and DRGym will bring their
# own. A column per key would mean a schema migration — and a new schema_hash, and therefore
# a new scorer_hash on every existing row — every time a suite added a component.
#
# `key` is the suite's own name for the quantity, verbatim, so `tau_reward` in
# `pi_eval.score.METRICS` and `tau_reward` in `Tau2Actuator.attach_reward` are the same string
# and a rename on either side is a row that stops joining rather than a number that quietly
# changes meaning.
NATIVE = pa.schema(
    [
        ("run_id", _str),
        ("suite_id", _str),
        ("key", _str),
        ("value", _f64),
    ]
)

TURNS = pa.schema(
    [
        ("run_id", _str),
        ("turn_idx", _i32),
        ("action_kind", _str),
        ("question", _str),
        ("question_id", _str),
        ("rationale", _str),
        ("target", _str),
        ("parent_uids", _strs),  # SELF-REPORTED: audited, never scored
        ("retrieved_uids", _strs),  # OBJECTIVE ev(q_t): every attribution metric reads this
        ("new_uids", _strs),
        ("n_retrieved", _i32),
        ("n_new", _i32),
        ("response_text", _str),  # the Drafter's reply TEXT, not a digest; see pinq.types.Turn
        ("draft_sha", _str),
        ("draft_text", _str),  # D_t as TEXT; the sha cannot reconstruct a prompt. See pinq.types
        ("subset_hash_before", _str),
        ("subset_hash_after", _str),
        ("branch_of_run_id", _str),
        ("branch_turn_idx", _i32),
        ("candidate_id", _str),
        ("candidate_rank", _i32),
        ("forced", _bool),
        ("policy_conf", _f64),
        ("depth_pred", _i32),
        ("bm25_hits", _i32),
        ("spec_bits", _f64),
        ("usage_tok_prompt", _i64),
        ("usage_tok_completion", _i64),
        ("usage_tok_reasoning", _i64),
        ("usage_tok_cached", _i64),
        ("usage_tok_total", _i64),
        ("usage_usd", _f64),
        ("usage_wall_ms", _i64),
        ("usage_n_calls", _i32),
    ]
)

CALLS = pa.schema(
    [
        ("run_id", _str),
        ("call_id", _str),
        ("actor", _str),
        ("model", _str),
        ("provider", _str),
        ("request_sha", _str),  # == the cache key; joins runs to cache/<xx>/<sha>.json
        ("response_sha", _str),
        ("tok_prompt", _i64),
        ("tok_completion", _i64),
        ("tok_reasoning", _i64),
        ("tok_cached", _i64),
        ("usd", _f64),
        ("wall_ms", _i64),  # ARTIFACT: never a cross-arm claim
        ("ttft_ms", _i64),  # ARTIFACT
        ("cache_hit", _bool),
        ("retries", _i32),
        ("rate_limit_stall_ms", _i64),
        ("http_status", _i32),
        ("provider_error_code", _str),
        ("attempt", _i32),
    ]
)

EVIDENCE = pa.schema(
    [
        ("run_id", _str),
        ("uid", _str),
        ("corpus_id", _str),
        ("doc_id", _str),
        ("span", _str),
        ("title", _str),
        ("score", _f64),
        ("n_chars", _i32),
        # THE ONLY COPY OF A REQUEST-ADDRESSED UNIT'S BYTES. `worker.evidence_rows` does not
        # persist text, and its reason -- "the corpus is content-addressed and frozen, so a uid
        # plus corpus_hash already identifies the bytes" -- is exactly right for a corpus record
        # and exactly wrong for a `call:<tool>:<hash>` unit, which is in NO corpus and is
        # identified by nothing else. Those runs wrote `text` anyway and compaction refused
        # them, so it is declared here rather than dropped.
        #
        # MEASURED before declaring it, because "legitimate content" and "bulk passage text
        # that would bloat the parquet" are the two possible answers and they have opposite
        # treatments: over all 48,299 run directories carrying an evidence.jsonl, 2,051 carry
        # `text` at all -- 3,244 rows, and 3,244 of 3,244 have a doc_id beginning `call:`.
        # 1.54 MB total, mean 476 chars, zero empty. Content, not bloat: dropping it deletes
        # the only surviving copy of 3,244 tool results and leaves those runs unexportable,
        # because `render_state` rebuilds held evidence from the corpus and a `call:` uid it
        # cannot find is the `state_mismatch` measured on 8 of the first 14 fork runs.
        #
        # "" -- never NULL -- for a corpus-addressed unit, backfilled by `compact._BACKFILL` on
        # both the run-directory and the read-back path. The empty string is unambiguous here
        # in a way it is not everywhere: 0 of 3,244 `call:` texts are empty, so "" says "this
        # unit is identified by its uid in a frozen corpus", never "the text was lost".
        ("text", _str),
        ("first_turn_idx", _i32),
    ]
)

ENV_CALLS = pa.schema(
    [
        ("run_id", _str),
        ("seq", _i32),
        ("turn_idx", _i32),
        ("requestor", _str),
        ("tool_name", _str),
        ("kwargs_json", _str),
        ("ok", _bool),
        ("result_digest", _str),
        ("mutating", _bool),
        ("unlock_required", _bool),  # tau2's discoverable tools: the binary prerequisite edge
        ("unlock_satisfied", _bool),
        ("latency_ms", _i64),
    ]
)

LEDGER = pa.schema(
    [
        ("run_id", _str),
        ("row_idx", _i32),
        ("currency", _str),
        ("charged", _f64),
        ("cumulative", _f64),
        ("turn_idx", _i32),
        ("cap", _f64),
        ("hard", _bool),
    ]
)

# --------------------------------------------------------------------------- declared empty

MATCHES = pa.schema(
    [
        ("run_id", _str),
        ("suite_id", _str),
        ("task_id", _str),
        ("node_id", _str),
        ("match_kind", _str),  # none | ask | resolve | use
        ("matched_turn_idx", _i32),
        ("matcher_id", _str),
        ("matcher_family", _str),
        ("matcher_score", _f64),
        ("threshold", _f64),
        ("graph_version", _str),
        ("cited_uids", _strs),
    ]
)

# RECONCILED WITH `pi_eval.judges.types.Judgment`: one column per dataclass field, same names,
# same order. `Judgment.to_row()` / `judgment_from_row()` round-trip through it, and
# tests/test_schema_judgments.py asserts the two field sets are equal, so a field added on one
# side without the other is a test failure rather than a silently dropped measurement.
#
# The first draft of this table was declared before the judges existed and guessed wrong in
# four ways, each of which would have destroyed information the judges actually produce:
#
#   * NO `criterion` COLUMN. All three judges write Judgments -- key-point labels, citation
#     support labels and six 0-10 quality ratings -- into this one table. Without the criterion
#     they are indistinguishable, so `avg(score)` would average a 0/1 recall indicator with a
#     Likert rating and return a number with no unit.
#   * NO `key_point_id`. That is the UNIT two judges must agree ON (paired.rating_unit_id) and
#     the unit incremental KPR deduplicates on. Without it a Krippendorff alpha can only be
#     computed over task means, which reports agreement between judges that disagree about
#     every individual point.
#   * `winner`/`tie` INSTEAD OF `pref_sign`. Two encodings of one verdict, the second derivable
#     from the first: `pref_sign` is +1/-1/0 from A's point of view and 0 IS the tie, including
#     the order-inconsistent pair that `resolve_pair()` forces to one. A stored `winner` string
#     is a copy that can disagree with the sign it came from.
#   * NO `label` AND NO `score`. The absolute graders return a LABEL ("Supported",
#     "full_support") and a score, and a preference-only table can hold neither -- which is
#     exactly why `judgments.parquet` stayed at 0 rows while all three judges were callable.
#
# Nullable columns are precisely the fields typed `| None`: absolute grading has no `run_id_b`
# and no `pref_sign`, and a NULL there says "not a contest" where "" or 0 would say "tie".
JUDGMENTS = pa.schema(
    [
        ("judgment_id", _str),
        ("run_id_a", _str),
        ("run_id_b", _str),  # NULL => absolute grading, not a contest
        ("suite_id", _str),
        ("task_id", _str),
        ("criterion", _str),
        ("order", _str),  # 'ab' | 'ba'; both are always run, inconsistent pairs -> tie
        ("judge_family", _str),
        ("judge_model", _str),
        ("judge_prompt_sha", _str),
        ("temperature", _f64),
        ("pref_sign", _i32),  # -1 / 0 / +1 from A's point of view; NULL when absolute
        ("magnitude", _f64),
        ("label", _str),
        ("score", _f64),
        ("key_point_id", _str),
        ("len_a_words", _i32),  # recorded on EVERY judgment: length is the null explanation
        ("len_b_words", _i32),
        ("retest_group_id", _str),  # sigma_J test-retest grouping
        ("paraphrase_id", _str),
        ("judge_pin", _str),
        ("response_sha", _str),
    ]
)

SCORES = pa.schema(
    [
        ("run_id", _str),
        ("metric_name", _str),
        ("scorer_hash", _str),  # h(metric_defs, graph_hash, matcher_hash, judge_pins)
        ("value", _f64),
        ("ci_lo", _f64),
        ("ci_hi", _f64),
        ("n", _i32),
        ("graph_version", _str),
        ("notes", _str),
    ]
)


TABLES: dict[str, pa.Schema] = {
    "runs": RUNS,
    "turns": TURNS,
    "calls": CALLS,
    "evidence": EVIDENCE,
    "env_calls": ENV_CALLS,
    "ledger": LEDGER,
    "native": NATIVE,
    "matches": MATCHES,
    "judgments": JUDGMENTS,
    "scores": SCORES,
}

# Written by `pi compact`. `matches`, `judgments` and `scores` are written by `pi score`,
# which is why compaction creates them empty and then leaves them alone forever after.
POPULATED: tuple[str, ...] = (
    "runs",
    "turns",
    "calls",
    "evidence",
    "env_calls",
    "ledger",
    "native",
)
EMPTY_BY_DESIGN: tuple[str, ...] = ("matches", "judgments", "scores")


def schema_for(name: str) -> pa.Schema:
    try:
        return TABLES[name]
    except KeyError:
        raise SchemaViolation(f"unknown table {name!r}; known: {sorted(TABLES)}") from None


def schema_hash() -> str:
    """Fingerprint of the whole contract. Belongs in scorer_hash so a column change writes
    new score rows instead of quietly re-labelling old ones."""
    return h(
        "schema",
        SCHEMA_VERSION,
        canon({n: [(f.name, str(f.type)) for f in s] for n, s in sorted(TABLES.items())}),
    )


def validate_rows(name: str, rows: Sequence[Mapping[str, Any]]) -> pa.Schema:
    """Strict in both directions. See the module docstring for why NULL-filling is banned."""
    schema = schema_for(name)
    allowed = set(schema.names)
    for i, row in enumerate(rows):
        keys = set(row)
        unknown = keys - allowed
        if unknown:
            raise SchemaViolation(
                f"table {name!r} row {i}: unknown column(s) {sorted(unknown)}. "
                "Add them to pi_eval.schema first — an undeclared column joins as NULL and "
                "the damage surfaces three tables downstream."
            )
        missing = allowed - keys
        if missing:
            raise SchemaViolation(
                f"table {name!r} row {i}: missing column(s) {sorted(missing)}. "
                "NULL-filling would let aggregates silently skip these rows."
            )
        # A FRACTIONAL FLOAT IN AN INTEGER COLUMN IS TRUNCATED BY pyarrow, NOT REFUSED.
        # `to_table` stored 7.9 as 7 -- a wrong value in the shape of a right one, in a module
        # whose own docstring says it is "strict in both directions". A float reaching an
        # integer column means a bug upstream (a mean where a count belongs, a ratio where a
        # turn index belongs), and truncating it deletes the evidence.
        #
        # An EXACT float is fine: JSON has one number type, so 7.0 arrives for an int all the
        # time and converting it loses nothing.
        for f in schema:
            v = row[f.name]
            if isinstance(v, bool) or not isinstance(v, float):
                continue
            if pa.types.is_integer(f.type) and v != int(v):
                raise SchemaViolation(
                    f"table {name!r} row {i}: column {f.name!r} is declared {f.type} but got "
                    f"the fractional float {v!r}. pyarrow would store {int(v)} and say nothing. "
                    "Fix the producer: a fraction here is a value of the wrong kind, not a "
                    "value needing rounding."
                )
    return schema


def to_table(name: str, rows: Sequence[Mapping[str, Any]]) -> pa.Table:
    schema = validate_rows(name, rows)
    cols = {f.name: [r[f.name] for r in rows] for f in schema}
    return pa.Table.from_pydict(cols, schema=schema)


def empty_table(name: str) -> pa.Table:
    return to_table(name, [])
