"""Shared machinery for turning a tau2 relational DB into evidence units. VIEW SIDE.

WHY THIS EXISTS SEPARATELY FROM `retail_units`. Retail was first and is left exactly as it is:
it is referenced by name in `retail_build`, in the registry note and in two tests, and a
refactor that moved it would make those citations lie. Airline and telecom need the same
arithmetic over differently-shaped databases, so the arithmetic moved here and retail did not.

WHAT ALL THREE DOMAINS SHARE. A record's uid is `evidence_uid(corpus_id, "<table>:<id>",
"0:<len>")`; the corpus hash is order-independent over `(doc_id, title, sha256(record_text))`;
a call's evidence is read from its ARGUMENTS and never from its result, because `tau2_runner`
stores a `result_digest` and never the payload. An id naming no record contributes nothing --
a fabricated uid can never be matched by gold, so it lands as a silent zero rather than as a
miss anyone could see.

WHAT THEY DO NOT SHARE. The DB shape (retail and airline are dicts of dicts keyed by record
id; telecom's TOML tables are ARRAYS whose records carry their own `*_id` field), the argument
names that address a table, and the one indirection each domain has -- retail's `item_id` names
a variant inside a product, telecom's `phone_number` names a line, airline's `flight_number`
hides one level down inside a list of dicts.

`pinq_adapters` may never import `pi_eval` (contract 1), so `record_doc_id` and `record_text`
are duplicated on the gold side and held equal by test.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Callable, Mapping, Sequence

from pinq.ids import corpus_hash as _corpus_hash
from pinq.ids import evidence_uid


def record_text(record: Mapping[str, Any]) -> str:
    """Canonical JSON. SORTED KEYS, because the uid depends on the length -- two
    serialisations of one record must not mint two uids."""
    return json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def record_doc_id(table: str, record_id: str) -> str:
    """`<table>:<record_id>`. Mirrored gold-side; the agreement is asserted by test."""
    return f"{table}:{record_id}"


def record_title(table: str, record_id: str) -> str:
    return f"{table}/{record_id}"


def tables_from_lists(
    db: Mapping[str, Any], id_field: Mapping[str, str]
) -> dict[str, dict[str, Any]]:
    """Normalise TOML array-tables into `{table: {record_id: record}}`.

    Telecom's `db.toml` holds every table as an array of records, each carrying its own
    `<table>_id`. A mapping-shaped loop over that iterates integer positions and indexes
    nothing, so the normalisation happens once, here, before any uid is minted. A record
    missing its id field is skipped rather than keyed by position: a positional key would be
    stable only until upstream reorders the file.
    """
    out: dict[str, dict[str, Any]] = {}
    for table, rows in db.items():
        field = id_field.get(table)
        if field is None or not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            continue
        bucket: dict[str, Any] = {}
        for row in rows:
            if isinstance(row, Mapping) and row.get(field) is not None:
                bucket[str(row[field])] = row
        if bucket:
            out[table] = bucket
    return out


def index_tables(
    tables: Mapping[str, Mapping[str, Any]], corpus_id: str
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """`(uids, text_sha, titles)` keyed by doc_id, in sorted order."""
    uids: dict[str, str] = {}
    text_sha: dict[str, str] = {}
    titles: dict[str, str] = {}
    for table in sorted(tables):
        for rid in sorted(tables[table]):
            text = record_text(tables[table][rid])
            doc = record_doc_id(table, rid)
            uids[doc] = evidence_uid(corpus_id, doc, f"0:{len(text)}")
            text_sha[doc] = hashlib.sha256(text.encode()).hexdigest()
            titles[doc] = record_title(table, rid)
    return uids, text_sha, titles


def corpus_hash_of(
    uids: Mapping[str, str], titles: Mapping[str, str], text_sha: Mapping[str, str]
) -> str:
    """Must equal the gold builder's. Order-independent over (doc_id, title, sha)."""
    return _corpus_hash((doc, titles[doc], text_sha[doc]) for doc in uids)


def units_from_tables(
    tables: Mapping[str, Mapping[str, Any]], corpus_id: str
) -> tuple[dict[str, Any], ...]:
    """Every record as an evidence unit -- the corpus, at the grain a tool call returns.

    Record-level rather than field-level for the reason the paragraph suites are
    paragraph-level: a gold uid the environment can never emit is a node that can never be
    resolved, and no tau2 tool returns a single field.
    """
    out: list[dict[str, Any]] = []
    for table in sorted(tables):
        for rid in sorted(tables[table]):
            text = record_text(tables[table][rid])
            out.append(
                {
                    "uid": evidence_uid(corpus_id, record_doc_id(table, rid), f"0:{len(text)}"),
                    "corpus_id": corpus_id,
                    "doc_id": record_doc_id(table, rid),
                    "span": f"0:{len(text)}",
                    "title": record_title(table, rid),
                    "text": text,
                }
            )
    return tuple(out)


def walk_call_arguments(
    arguments: Mapping[str, Any],
    resolve: Callable[[str, str], str | None],
    *,
    descend_dicts: bool = False,
) -> tuple[str, ...]:
    """Uids for the records a call's arguments name. Sorted and deduplicated.

    Deduplicated because `retrieved_uids` is set-hashed downstream and counted as
    `n_retrieved`: a repeated id would inflate the redundancy penalty for a call that read one
    record. Sorted so a replay of the same call produces the same row.

    `descend_dicts` exists for airline and is off by default because it changes what retail
    and telecom would credit. Airline's `flight_number` NEVER appears as a top-level argument
    -- it appears only inside `flights=[{"flight_number": ..., "date": ...}]` -- so without
    one level of descent, 30 of airline's 142 gold actions resolve to nothing. The descent
    applies the SAME resolver to the nested keys, which is what keeps `passengers` and
    `payment_methods` (also lists of dicts, 13 and 10 gold actions) correctly worthless:
    `first_name`, `dob`, `payment_id` and `amount` address no table.
    """
    if not arguments:
        return ()
    found: set[str] = set()

    def consider(name: str, value: Any, depth: int) -> None:
        if isinstance(value, Mapping):
            if descend_dicts and depth == 0:
                for k, v in value.items():
                    consider(k, v, depth + 1)
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                consider(name, item, depth)
            return
        if isinstance(value, (str, int, float)) and not isinstance(value, bool):
            uid = resolve(name, str(value))
            if uid is not None:
                found.add(uid)

    for name, value in arguments.items():
        consider(name, value, 0)
    return tuple(sorted(found))


def uids_from_env_calls(
    calls: Sequence[Mapping[str, Any]],
    resolve_call: Callable[[str, Mapping[str, Any]], Sequence[str]],
) -> tuple[str, ...]:
    """Every record a turn's recorded `EnvCall`s read, in first-seen order.

    `kwargs_json` is the canonical form `tau2_runner` already stores, so this reconstructs
    `retrieved_uids` from the run log with no access to any result.
    """
    seen: list[str] = []
    known: set[str] = set()
    for call in calls:
        raw = call.get("kwargs_json") or "{}"
        try:
            args = json.loads(raw) if isinstance(raw, str) else dict(raw)
        except (ValueError, TypeError):
            continue
        if not isinstance(args, Mapping):
            continue
        for uid in resolve_call(str(call.get("tool_name") or ""), args):
            if uid not in known:
                known.add(uid)
                seen.append(uid)
    return tuple(seen)


def evidence_units(records: Sequence[Mapping[str, Any]]) -> tuple[Any, ...]:
    """Unit dicts -> `EvidenceUnit`s, for the exporter.

    The dict shape is what the gold builders and `units_from_db` share, and it is the right
    shape there. Its one other consumer, `cmd_train._task_units`, does
    `{u.uid: u for u in suite.units(task_id)}` -- so a dict arrives as
    `AttributeError: 'dict' object has no attribute 'uid'`, which the exporter catches PER RUN
    and reports only as a count. The export then succeeds with a smaller shard and nothing says
    which runs were dropped. Converting once, here, is what stops the two shapes meeting again.
    """
    from pinq.types import EvidenceUnit

    return tuple(
        EvidenceUnit.make(
            corpus_id=str(r["corpus_id"]),
            doc_id=str(r["doc_id"]),
            span=str(r["span"]),
            title=str(r.get("title") or ""),
            text=str(r.get("text") or ""),
        )
        for r in records
    )
