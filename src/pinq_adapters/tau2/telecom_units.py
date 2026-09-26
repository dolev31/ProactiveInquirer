"""Which telecom DB records a tool call retrieved. VIEW SIDE -- reads the environment, not gold.

TWO STRUCTURAL DIFFERENCES FROM RETAIL AND AIRLINE.

THE TABLES ARE TOML ARRAYS, NOT MAPPINGS. `db.toml` holds plans, devices, lines, customers and
bills as arrays of records, each carrying its own `<table>_id`. A mapping-shaped loop over an
array iterates integer positions and indexes nothing, so the tables are normalised to
`{record_id: record}` once, before any uid is minted, and a record missing its id field is
skipped rather than keyed by position -- a positional key stays stable only until upstream
reorders the file.

THERE IS A SECOND DATABASE AND IT IS DELIBERATELY NOT A CORPUS. `user_db.toml` is the phone's
own state, coupled to the agent DB by `TelecomEnvironment.sync_tools`. The agent cannot read
it: a fact living only there reaches the agent by the CUSTOMER SAYING IT, which is exactly the
user-private partition the discoverability ceiling is defined over. Indexing it would move
those facts to the KB-discoverable side and silently raise the ceiling that bounds what any
autonomous inquirer could reach -- a change to the headline claim, made by an import. This
module never opens the file.

WHAT MINTS NOTHING. Most telecom gold actions are the CUSTOMER toggling the handset -- 393 of
the 516 gold actions in the base split are `requestor: user` (MEASURED). `toggle_airplane_mode`
and `set_network_mode_preference` take no record key and retrieve nothing for the agent, which
is the honest accounting: the agent instructed, it did not read.

`phone_number` resolves through the index rather than through `_ARG_TABLE`, the way retail's
`item_id` does: it is how a customer names a line, but the record is the line.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .db_units import (
    corpus_hash_of,
    index_tables,
    record_doc_id,
    record_text,
    tables_from_lists,
    units_from_tables,
    walk_call_arguments,
)
from .db_units import uids_from_env_calls as _uids_from_env_calls

CORPUS_ID = "tau2_telecom"

# The id field each array-table keys itself by.
_ID_FIELD: Mapping[str, str] = {
    "plans": "plan_id",
    "devices": "device_id",
    "lines": "line_id",
    "customers": "customer_id",
    "bills": "bill_id",
}

_ARG_TABLE: Mapping[str, str] = {
    "customer_id": "customers",
    "line_id": "lines",
    "device_id": "devices",
    "plan_id": "plans",
    "bill_id": "bills",
}

# Not in `_ARG_TABLE`: it names a line indirectly, so it resolves through the index.
_PHONE_ARGS = ("phone_number",)


@dataclass(frozen=True, slots=True)
class TelecomIndex:
    """The agent DB, indexed for the two questions a call answers: does this record exist,
    and which line has this phone number?"""

    uids: Mapping[str, str]
    text_sha: Mapping[str, str]
    titles: Mapping[str, str]
    phone_to_line: Mapping[str, str]
    units: Mapping[str, dict[str, Any]]

    @classmethod
    def from_db(cls, db: Mapping[str, Any]) -> "TelecomIndex":
        tables = tables_from_lists(db, _ID_FIELD)
        uids, text_sha, titles = index_tables(tables, CORPUS_ID)
        phone_to_line = {
            str(rec["phone_number"]): str(rid)
            for rid, rec in (tables.get("lines") or {}).items()
            if isinstance(rec, Mapping) and rec.get("phone_number") is not None
        }
        units = {u["uid"]: u for u in units_from_tables(tables, CORPUS_ID)}
        return cls(uids, text_sha, titles, phone_to_line, units)

    @classmethod
    def from_dir(cls, domain_dir: Path) -> "TelecomIndex":
        """Reads `db.toml` ONLY. `user_db.toml` is not a corpus; see the module docstring."""
        with (Path(domain_dir) / "db.toml").open("rb") as fh:
            return cls.from_db(tomllib.load(fh))

    def uid(self, table: str, record_id: str) -> str:
        return self.uids[record_doc_id(table, record_id)]

    def unit_by_uid(self, uid: str) -> dict[str, Any] | None:
        return self.units.get(uid)

    def corpus_hash(self) -> str:
        return corpus_hash_of(self.uids, self.titles, self.text_sha)

    def _resolve(self, name: str, raw: str) -> str | None:
        table = _ARG_TABLE.get(name)
        if table is not None:
            return self.uids.get(record_doc_id(table, raw))
        if name in _PHONE_ARGS:
            line_id = self.phone_to_line.get(raw)
            if line_id is not None:
                return self.uids.get(record_doc_id("lines", line_id))
        return None


def uids_for_call(
    index: TelecomIndex, tool_name: str, arguments: Mapping[str, Any]
) -> tuple[str, ...]:
    """Records this call read, as evidence uids. Sorted, deduplicated, arguments only."""
    return walk_call_arguments(arguments, index._resolve)


def units_from_db(db: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    return units_from_tables(tables_from_lists(db, _ID_FIELD), CORPUS_ID)


def uids_from_env_calls(index: TelecomIndex, calls: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    return _uids_from_env_calls(calls, lambda name, args: uids_for_call(index, name, args))


__all__ = [
    "CORPUS_ID",
    "TelecomIndex",
    "record_doc_id",
    "record_text",
    "uids_for_call",
    "uids_from_env_calls",
    "units_from_db",
]
