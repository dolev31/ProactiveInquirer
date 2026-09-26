"""Convlog suite, VIEW SIDE — the split-key half only.

This is deliberately not a full Suite yet. The corpus-reading half (task_ids, view, units,
retriever, actuator over data/corpora/convlog/<hash>/) is a later task; writing it now would
mean guessing at a public record shape before pi_eval/build/convlog_parse.py's output has
been reviewed, and re-guessing later is how a corpus and its gold drift apart (see
pi_eval/build/common.py's doc_id/unit_uid discussion for what that costs).

WHY template_id_of EXISTS AT ALL, and why it collapses to the session id: pinq.splitting
buckets every task by `template_id or task_id`, on the premise that two instantiations of one
template are near-duplicates and must land on the same side of the train/dev/test wall. A
convlog "template" is a session — two decision points mined from the same conversation share
its task, its repository and its user's habits, so letting them straddle the wall is exactly
the leak `split_key` exists to prevent. Task ids are minted as "<session_id>:<dp_index>"
(see pi_eval.build.convlog_parse), so recovering the session id is just taking the part
before the colon.
"""

from __future__ import annotations


class ConvlogSuite:
    @staticmethod
    def template_id_of(task_id: str) -> str:
        session_id, _, _dp_index = task_id.partition(":")
        return session_id
