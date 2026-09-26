"""Which airline DB records a tool call retrieved. VIEW SIDE -- reads the environment, not gold.

FROM THE ARGUMENTS, NEVER FROM THE RESULT, for the reason `retail_units` states: a tau2 tool
result is a customer record, `tau2_runner` stores only a `result_digest`, and that is a
constraint to satisfy rather than work around.

THE ONE THING AIRLINE NEEDS THAT RETAIL DOES NOT. `flight_number` NEVER appears as a top-level
argument in airline's gold actions. It appears only nested inside a list of dicts --
`flights=[{"flight_number": "HAT005", "date": "2024-05-20"}]` -- in 30 of airline's 142 gold
actions (MEASURED from tasks.json). `retail_units.uids_for_call` skips any Mapping value
outright, so a copy of it credits no flight evidence at all, and the failure is invisible:
downstream it reads as "the policy retrieved nothing relevant", not as an error.

WHAT MUST STILL MINT NOTHING. `search_direct_flight(origin, destination, date)` names no
record. It is a search, and the flights it returns are learned from the RESULT, which this
module never touches; crediting it would hand the policy evidence it did not obtain by key.
`passengers` and `payment_methods` are also lists of dicts (13 and 10 gold actions) and stay
worthless because the descent applies the same `_ARG_TABLE`, which has no entry for
`first_name`, `dob`, `payment_id` or `amount`.

db.json IS NOT GOLD. It is the environment the agent acts in; the gold is
`evaluation_criteria`, which lives in tasks.json and which this module never opens.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .db_units import (
    corpus_hash_of,
    index_tables,
    record_doc_id,
    record_text,
    units_from_tables,
    walk_call_arguments,
)
from .db_units import uids_from_env_calls as _uids_from_env_calls

CORPUS_ID = "tau2_airline"

# Which argument names a record in which table. A tool absent from every entry reads nothing
# (`calculate`, `transfer_to_human_agents`, `search_direct_flight`) and correctly yields no
# evidence. `flight_number` is reachable only through the nested descent; see the docstring.
_ARG_TABLE: Mapping[str, str] = {
    "reservation_id": "reservations",
    "user_id": "users",
    "flight_number": "flights",
}


@dataclass(frozen=True, slots=True)
class AirlineIndex:
    """The DB, indexed for the one question a call answers: does this record exist?

    Airline needs no equivalent of retail's `item_to_product`. Every id it accepts names a
    top-level record directly; the only indirection is positional, not semantic.
    """

    uids: Mapping[str, str]  # doc_id -> uid
    text_sha: Mapping[str, str]  # doc_id -> sha256(record_text)
    titles: Mapping[str, str]
    units: Mapping[str, dict[str, Any]]  # uid -> unit, so a keyed read can mint the record

    @classmethod
    def from_db(cls, db: Mapping[str, Mapping[str, Any]]) -> "AirlineIndex":
        uids, text_sha, titles = index_tables(db, CORPUS_ID)
        units = {u["uid"]: u for u in units_from_tables(db, CORPUS_ID)}
        return cls(uids, text_sha, titles, units)

    @classmethod
    def from_dir(cls, domain_dir: Path) -> "AirlineIndex":
        return cls.from_db(json.loads((Path(domain_dir) / "db.json").read_text()))

    def uid(self, table: str, record_id: str) -> str:
        return self.uids[record_doc_id(table, record_id)]

    def unit_by_uid(self, uid: str) -> dict[str, Any] | None:
        return self.units.get(uid)

    def corpus_hash(self) -> str:
        return corpus_hash_of(self.uids, self.titles, self.text_sha)

    def _resolve(self, name: str, raw: str) -> str | None:
        table = _ARG_TABLE.get(name)
        if table is None:
            return None
        return self.uids.get(record_doc_id(table, raw))


def uids_for_call(
    index: AirlineIndex, tool_name: str, arguments: Mapping[str, Any]
) -> tuple[str, ...]:
    """Records this call read, as evidence uids. Sorted, deduplicated, arguments only."""
    return walk_call_arguments(arguments, index._resolve, descend_dicts=True)


def units_from_db(db: Mapping[str, Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    return units_from_tables(db, CORPUS_ID)


def uids_from_env_calls(index: AirlineIndex, calls: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    return _uids_from_env_calls(calls, lambda name, args: uids_for_call(index, name, args))


__all__ = [
    "CORPUS_ID",
    "AirlineIndex",
    "record_doc_id",
    "record_text",
    "uids_for_call",
    "uids_from_env_calls",
    "units_from_db",
]
