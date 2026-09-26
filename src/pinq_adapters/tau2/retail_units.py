"""Which DB records a retail tool call retrieved. VIEW SIDE — reads the environment, not gold.

FROM THE ARGUMENTS, NEVER FROM THE RESULT
    A tau2 tool result is a customer record: `tau2_runner` stores a `result_digest` and never
    the payload, and that is a constraint to satisfy rather than work around. Retail's tools
    are key-addressed — MEASURED, 458 of 550 required calls (83.3%) name a record directly in
    their arguments — so what a call read is recoverable from what it ASKED FOR. Nothing in
    this module touches a tool result.

    The 75 `find_user_id_by_*` calls are the deliberate exception. They return an id rather
    than taking one, so naming their record would mean reading the result. They contribute no
    evidence here; the keyed call that follows is what records the user record, which is the
    honest accounting — the agent did not hold the record until it looked it up.

WHY THIS DUPLICATES `pi_eval.build.retail_build`
    `pinq_adapters` may never import `pi_eval` (contract 1). `record_doc_id` and `record_text`
    therefore exist on both sides, and `tests/test_retail_units.py` asserts they agree on every
    one of the 1,550 records and on the corpus hash. This is the same arrangement
    `common.unit_uid` and `EvidenceUnit.make()` already have for the paragraph suites, for the
    same reason: a uid mismatch is invisible in every downstream number, because it reads as
    "the policy retrieved nothing relevant" rather than as an error.

    db.json IS NOT GOLD. It is the environment the agent acts in, so reading it here crosses
    no firewall — the gold is `evaluation_criteria`, which lives in tasks.json and which this
    module never opens.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from pinq.ids import corpus_hash as _corpus_hash
from pinq.ids import evidence_uid

CORPUS_ID = "tau2_retail"

# Which argument names a record in which table. A tool absent from every entry reads nothing
# (`calculate`, `transfer_to_human_agents`) and correctly yields no evidence: crediting it
# would hand the policy evidence it never obtained.
_ARG_TABLE: Mapping[str, str] = {
    "order_id": "orders",
    "user_id": "users",
    "product_id": "products",
}

# `item_id` names a VARIANT, which is not a top-level record. The record a call for it reads
# is the product holding that variant, so it resolves through the index rather than directly.
_ITEM_ARGS = ("item_id", "item_ids", "new_item_ids")


def record_text(record: Mapping[str, Any]) -> str:
    """Canonical JSON. SORTED KEYS, because the uid depends on the length — two serialisations
    of one record must not mint two uids."""
    return json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def record_doc_id(table: str, record_id: str) -> str:
    """`<table>:<record_id>`. Mirrored in `pi_eval.build.retail_build`; the two are asserted
    equal on every record by the agreement test."""
    return f"{table}:{record_id}"


@dataclass(frozen=True, slots=True)
class RetailIndex:
    """The DB, indexed for the two questions a call answers: does this record exist, and which
    product holds this item?"""

    uids: Mapping[str, str]  # doc_id -> uid
    text_sha: Mapping[str, str]  # doc_id -> sha256(record_text), for the corpus hash
    item_to_product: Mapping[str, str]
    titles: Mapping[str, str]
    # uid -> unit dict, the shape AirlineIndex and TelecomIndex already expose. Without it
    # `ToolRetriever._units_for`'s keyed branch -- guarded on `callable(getattr(index,
    # "unit_by_uid", None))` -- was unreachable for retail, so every keyed read fell through to
    # the request-addressed `call:<tool>:<hash>` fallback instead of minting the records it
    # named. Our retail runs recorded 0.4 evidence rows per run against airline's 2.0.
    # `from_db` already computed this text and kept only its sha.
    units: Mapping[str, dict[str, Any]] = field(default_factory=dict)

    def unit_by_uid(self, uid: str) -> dict[str, Any] | None:
        return self.units.get(uid)

    @classmethod
    def from_db(cls, db: Mapping[str, Mapping[str, Any]]) -> "RetailIndex":
        uids: dict[str, str] = {}
        text_sha: dict[str, str] = {}
        titles: dict[str, str] = {}
        item_to_product: dict[str, str] = {}
        units: dict[str, dict[str, Any]] = {}
        for table in sorted(db):
            for rid in sorted(db[table]):
                rec = db[table][rid]
                doc = record_doc_id(table, rid)
                text = record_text(rec)
                span = f"0:{len(text)}"
                uid = evidence_uid(CORPUS_ID, doc, span)
                uids[doc] = uid
                text_sha[doc] = hashlib.sha256(text.encode()).hexdigest()
                titles[doc] = f"{table}/{rid}"
                # The span here is the one hashed into the uid above. A unit whose span
                # disagreed with it would never match gold.
                units[uid] = {
                    "uid": uid,
                    "corpus_id": CORPUS_ID,
                    "doc_id": doc,
                    "span": span,
                    "title": titles[doc],
                    "text": text,
                }
                if table == "products":
                    for item_id in rec.get("variants") or {}:
                        item_to_product[str(item_id)] = str(rid)
        return cls(uids, text_sha, item_to_product, titles, units)

    @classmethod
    def from_dir(cls, domain_dir: Path) -> "RetailIndex":
        return cls.from_db(json.loads((Path(domain_dir) / "db.json").read_text()))

    def uid(self, table: str, record_id: str) -> str:
        return self.uids[record_doc_id(table, record_id)]

    def corpus_hash(self) -> str:
        """Must equal `retail_build.corpus_hash_of`. Order-independent over
        (doc_id, title, sha256(record_text)) — the shape `pinq.ids.corpus_hash` expects."""
        return _corpus_hash((doc, self.titles[doc], self.text_sha[doc]) for doc in self.uids)


def uids_for_call(
    index: RetailIndex, tool_name: str, arguments: Mapping[str, Any]
) -> tuple[str, ...]:
    """Records this call read, as evidence uids. Sorted and deduplicated.

    Deduplicated because `retrieved_uids` is set-hashed downstream and counted as
    `n_retrieved`: a repeated id would inflate the redundancy penalty for a call that read one
    record. Sorted so a replay of the same call produces the same row.

    An id that names no record yields nothing. A fabricated uid can never be matched by gold,
    so it would land as a silent zero rather than as a miss anyone could see.
    """
    if not arguments:
        return ()
    found: set[str] = set()
    for name, value in arguments.items():
        values = value if isinstance(value, (list, tuple)) else [value]
        for item in values:
            if isinstance(item, (Mapping, list, tuple)):
                continue
            raw = str(item)
            table = _ARG_TABLE.get(name)
            if table is not None:
                doc = record_doc_id(table, raw)
                if doc in index.uids:
                    found.add(index.uids[doc])
            elif name in _ITEM_ARGS:
                pid = index.item_to_product.get(raw)
                if pid is not None:
                    found.add(index.uid("products", pid))
    return tuple(sorted(found))


def units_from_db(db: Mapping[str, Mapping[str, Any]]) -> tuple[dict, ...]:
    """Every DB record as an evidence unit. The corpus, at the grain a tool call returns.

    Record-level rather than field-level for the reason the paragraph suites are
    paragraph-level: a gold uid the environment can never emit is a node that can never be
    resolved, and no retail tool returns a single field.
    """
    out: list[dict] = []
    for table in sorted(db):
        for rid in sorted(db[table]):
            rec = db[table][rid]
            text = record_text(rec)
            out.append(
                {
                    "uid": evidence_uid(CORPUS_ID, record_doc_id(table, rid), f"0:{len(text)}"),
                    "corpus_id": CORPUS_ID,
                    "doc_id": record_doc_id(table, rid),
                    "span": f"0:{len(text)}",
                    "title": f"{table}/{rid}",
                    "text": text,
                }
            )
    return tuple(out)


def uids_from_env_calls(index: RetailIndex, calls: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    """Every record a turn's recorded `EnvCall`s read, in first-seen order.

    `kwargs_json` is the canonical form `tau2_runner` already stores, so this reconstructs
    `retrieved_uids` from the run log with no access to any result.
    """
    seen: list[str] = []
    known: set[str] = set()
    for call in calls:
        args = call.get("kwargs_json")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                continue
        if not isinstance(args, Mapping):
            continue
        for uid in uids_for_call(index, str(call.get("tool_name") or ""), args):
            if uid not in known:
                known.add(uid)
                seen.append(uid)
    return tuple(seen)
