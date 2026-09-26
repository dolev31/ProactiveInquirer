"""The corpus, re-read as UNPAIRED (state, action, label) signals. Pure; no torch, no trl.

WHAT THIS IS FOR. The DPO rung trains on a pair and therefore uses each pair ONCE, as a
contrast. Every absolute fact the corpus states about a single action is discarded in the
process: that a rejected question was a bad question at that state, that an ASK at a state gold
says was finished is a wrong action whatever it is compared against, that a STOP at a state that
was not finished is wrong on its own terms. KTO scores one completion at a time against the
reference policy, so those facts become rows. MEASURED on `data/rl/{sft,pairs}.jsonl`
(2026-09-15, 43,837 SFT rows + 47,989 pairs): 80,766 distinct signals, of which 33,481 are
undesirable -- 41.5% of a dataset the sequential recipe cannot express at all.

THE DIRECTION RULE IS ONE SENTENCE AND IT IS DELIBERATELY NOT A TABLE OF KINDS:

    every SFT row is desirable; the CHOSEN side of a pair is desirable and the REJECTED side
    undesirable, whatever kind of pair it is.

Written as a table over `pair_kind` x `stop_source` it would need a new branch every time the
exporter grows a kind, and the branch that was not added would silently emit the wrong sign --
on the `ask_stop_synth` family that is 28,613 of the 80,766 rows, i.e. the sign error would be
most of the dataset and no loss curve would show it. So `stop_source` is CARRIED, never branched
on: `synthesised_notdone` needs no change here, and neither will the next one. `source_kind`
names which family a row came from so the manifest can still be read per-family.

WHAT IS NOT INHERITED FROM THE DPO LOADER, AND WHY. The LENGTH GUARD does not carry over. It
exists because a preference model shown two completions side by side can learn "longer question
wins"; KTO never sees two completions together -- each row is scored alone against the reference
-- so the confound has no mechanism here, and importing the guard would delete rows on an
argument that does not apply to this loss. `assert_same_state` DOES carry over, for a different
reason than it has in DPO: nothing here needs the V(s_t) cancellation, but a pair carrying two
different states would have both of its rows emitted under one of them.

THE CONFLICT IS REAL AND IT IS NOT A DATA BUG. A (state, action) that is desirable in one pair
and undesirable in another is a contradiction KTO cannot represent, because it has no partner to
relativise against. MEASURED on the live corpus: 1,475 such keys, 1,098 of them from the ask_ask
tournament alone -- candidate B beats C at a state and loses to A at the same state, which is
pairwise ranking working exactly as designed. `on_conflict` is therefore a DECLARED choice that
rides in `cfg.sha`, not a default: "refuse" is the safe reading (the dataset is not convertible
as it stands) and "drop" removes BOTH sides of every conflicting key. Neither keeps one label.
Keeping one would pick a side of a contradiction the corpus does not settle, and the picked side
would then be an absolute claim nothing measured.
"""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

from pinq.actions import STOP_ACTION_JSON, action_kind_of
from pinq_train.rung1_sft.train import HeldOutDataset, assert_train_split
from pinq_train.rung2_dpo.train import PAIR_KINDS, assert_same_state, pair_kind_of

# WHERE A ROW CAME FROM AND WHICH SIDE IT WAS. The label is a function of the side alone (see
# the module docstring); these names exist so the manifest can report per-family counts, and so
# a sign error shows up as a count in the wrong column rather than as nothing at all.
SFT_ROW = "sft_row"
PAIR_CHOSEN = "pair_chosen"  # ask_ask winner
PAIR_REJECTED = "pair_rejected"  # ask_ask loser
# The STOP-bearing families. "synth" is the historical name for the group (`ask_stop_synth` is
# 31,309 of the 33,062 STOP-bearing pairs); `stop_source` on the row is what separates a
# RECORDED stop from a synthesised one, and it is carried on every row for exactly that reason.
SYNTH_STOP_CHOSEN = "synth_stop_chosen"  # stop at a state gold says was done
SYNTH_ASK_REJECTED = "synth_ask_rejected"  # ask at a state gold says was done
SYNTH_ASK_CHOSEN = "synth_ask_chosen"  # ask at a state that was NOT done
SYNTH_STOP_REJECTED = "synth_stop_rejected"  # stop at a state that was NOT done
SOURCE_KINDS = (
    SFT_ROW,
    PAIR_CHOSEN,
    PAIR_REJECTED,
    SYNTH_STOP_CHOSEN,
    SYNTH_ASK_REJECTED,
    SYNTH_ASK_CHOSEN,
    SYNTH_STOP_REJECTED,
)

# What a row must be able to name. AGENTS.md rule 1: a row that cannot name the instrument that
# scored it is not a training row, and a KTO row is one step further from its origin than a pair
# is -- the pair it came from no longer exists once the sides are split.
PROVENANCE_FIELDS = (
    "suite_id",
    "task_id",
    "run_id",
    "turn_idx",
    "scorer_hash",
    "graph_version",
    "split",
    "code_version",
    "arm_id",
)

ON_CONFLICT = ("refuse", "drop")


class ConflictingLabels(RuntimeError):
    """One (state, action) carries both labels, so KTO is asked to learn a contradiction.

    Fatal under the default. The two readings of such a key -- "desirable, because something
    beat it once" and "undesirable, because something else beat it" -- are both wrong: pairwise
    evidence orders candidates, it does not place them on an absolute scale, and KTO's objective
    is absolute. `.n_conflicts` and `.report` carry the counts so a caller can print them.
    """

    def __init__(self, message: str, *, n_conflicts: int, report: dict[str, Any]) -> None:
        super().__init__(message)
        self.n_conflicts = n_conflicts
        self.report = report


class NonCanonicalStop(ValueError):
    """A STOP whose bytes are not `pinq.actions.STOP_ACTION_JSON`.

    Fatal, and more so here than in DPO. A DPO pair shows the two spellings side by side, where
    a separator difference is a surface cue the loss can exploit; a KTO row is scored alone, so
    two spellings of one action are simply TWO COMPLETIONS, one of which the policy is trained
    to produce and the serving parser has never been shown to accept as the canonical stop.
    """


class UnparseableAction(ValueError):
    """An action `pinq.actions.action_kind_of` cannot read as an ask or a stop.

    Refused rather than dropped. As a desirable row it would train the policy to emit bytes the
    loop's own parser rejects; as an undesirable one it would spend a gradient on teaching the
    policy not to do something it has no way of doing. Either way the corpus has a row nothing
    downstream can interpret, which is a fact about the export, not about this converter.
    """


def state_sha(state_text: str) -> str:
    """The state's identity in the dedupe key. Hashed because a state is thousands of tokens."""
    return hashlib.sha256(state_text.encode("utf-8")).hexdigest()


def balanced_weights(n_desirable: int, n_undesirable: int) -> tuple[float, float]:
    """KTO's two class weights, set so the two total masses are EQUAL at mean weight 1.

    `w_d * n_d == w_u * n_u`, and `(w_d * n_d + w_u * n_u) / (n_d + n_u) == 1`. The second
    clause is what makes this the right normalisation rather than one of the two obvious ones.
    Pinning `w_d = 1` and lifting the minority scales the whole loss up; pinning `w_u = 1` and
    damping the majority scales it down. Either changes the EFFECTIVE LEARNING RATE of the run
    as a side effect of its class balance, so a balanced arm and an unbalanced arm would differ
    in two things at once. Rung 1 makes the same argument about `stop_weight` and
    `mean_row_weight` (see `SFTConfig`), and this is that argument for a binary label.

    Refuses an empty class rather than clamping it: with no undesirable rows there is no mass to
    balance, and a weight that "balances" a class with no members is a number describing nothing.
    trl's own recommended band (KTO paper Eq. 8, checked in `KTOTrainer._prepare_dataset`) is
    `1 <= (w_d n_d) / (w_u n_u) <= 1.33`, so the equal-mass point sits at the bottom of it.
    """
    if n_desirable <= 0:
        raise ValueError(
            f"0 desirable rows (n_undesirable={n_undesirable}): there is no mass to balance "
            "against, and a KTO run with no desirable signal has no direction to move in."
        )
    if n_undesirable <= 0:
        raise ValueError(
            f"0 undesirable rows (n_desirable={n_desirable}): the whole point of RC4 is the "
            "undesirable half of the corpus, and without it this is SFT with a reference model."
        )
    total = n_desirable + n_undesirable
    return total / (2.0 * n_desirable), total / (2.0 * n_undesirable)


def _check_action(action: str, *, stop_action: str, where: str) -> str:
    kind = action_kind_of(action)
    if kind == "unparseable":
        raise UnparseableAction(
            f"{where}: {action[:120]!r} is neither an ask nor a stop to "
            "`pinq.actions.action_kind_of`, which is the loop's own parser. A row the policy "
            "could not legally emit cannot be a KTO target, and cannot be a KTO negative either."
        )
    if kind == "stop" and action != stop_action:
        raise NonCanonicalStop(
            f"{where}: a STOP whose bytes are {action[:120]!r}, not the one constant "
            f"{stop_action!r}. KTO scores each completion alone, so a second spelling is a "
            "second completion -- and `pinq.actions` exists so there is exactly one."
        )
    return kind


def _sides(pair: Mapping[str, Any], kind: str, stop_action: str) -> tuple[str, str]:
    """(chosen source_kind, rejected source_kind). The LABEL is the side; this is only the name."""
    if kind == "ask_ask":
        return PAIR_CHOSEN, PAIR_REJECTED
    # Every STOP-bearing kind, present and future. Which of the two sides is the STOP is read
    # off the bytes, never off `stop_source` -- that field says where the stop came from, not
    # which way the pair points, and branching on it is how a new value gets the sign wrong.
    if str(pair.get("chosen_json")) == stop_action:
        return SYNTH_STOP_CHOSEN, SYNTH_ASK_REJECTED
    return SYNTH_ASK_CHOSEN, SYNTH_STOP_REJECTED


def _row(
    src: Mapping[str, Any],
    *,
    state: str,
    action: str,
    label: bool,
    source_kind: str,
    pair_kind: str = "",
    candidate_run_id: str = "",
) -> dict[str, Any]:
    row: dict[str, Any] = {
        # WHAT TRL CONSUMES. `KTOTrainer` reads exactly these three column names
        # (trl 1.13.0, `kto_trainer.py:225-226`); everything below them is provenance that
        # `train()` strips before it builds the `Dataset`.
        "prompt": state,
        "completion": action,
        "label": label,
        "source_kind": source_kind,
        "state_sha": state_sha(state),
        "action_kind": action_kind_of(action),
        "pair_kind": pair_kind,
        "stop_source": str(src.get("stop_source") or ""),
        # The CANDIDATE's run, which is not `run_id`: `run_id` is the state's (so an SFT row and
        # a pair side at one state agree), and this is the rollout the side itself came from.
        # "" on a synthesised side, which is not a rollout and has no run id.
        "candidate_run_id": candidate_run_id,
    }
    for k in PROVENANCE_FIELDS:
        v = src.get(k)
        row[k] = "" if v is None else v
    return row


def build_kto_rows(
    sft_rows: Sequence[Mapping[str, Any]],
    pair_rows: Sequence[Mapping[str, Any]],
    *,
    include_pair_kinds: Sequence[str] = PAIR_KINDS,
    stop_action: str = STOP_ACTION_JSON,
    on_conflict: str = "refuse",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The SFT rows and the pairs file, as one list of labelled (state, action) signals.

    PURE and total: no file is read, nothing is written, and every number the manifest will
    report about the dataset is returned beside it. `include_pair_kinds` defaults to ALL THREE
    kinds -- unlike the DPO loader, which defaults to the sampled ones -- because RC4 exists to
    use the synthetic STOP family the headline discards.

    ORDER IS PART OF THE CONTRACT. SFT rows are emitted first, then the pairs in file order,
    chosen side before rejected. The dedupe keeps the FIRST occurrence, so an action that is
    both an SFT target and a pair's chosen side is recorded as `sft_row` -- the SFT row is an
    absolute judgement (the candidate cleared the value floor) and the pair side is a relative
    one, so when the two name the same bytes the stronger claim is the one worth recording.
    """
    if on_conflict not in ON_CONFLICT:
        raise ValueError(f"on_conflict={on_conflict!r}: expected one of {ON_CONFLICT}")
    if not set(include_pair_kinds) <= set(PAIR_KINDS):
        raise ValueError(
            f"include_pair_kinds={tuple(include_pair_kinds)!r}: expected a subset of {PAIR_KINDS}"
        )
    # THE WALL, FIRST. Present-and-wrong is refused and absent is waved through and counted --
    # rung 1's rule, reused rather than restated so the two rungs cannot drift on what the
    # split field means. Named per file, because "the wrong file" is the case this catches.
    _assert_train_split(sft_rows, where="the SFT rows")
    _assert_train_split(pair_rows, where="the pairs")

    emitted: list[dict[str, Any]] = []
    drops: dict[str, int] = {}
    for i, r in enumerate(sft_rows):
        state = str(r["state_text"])
        action = str(r["action_json"])
        _check_action(action, stop_action=stop_action, where=f"sft row {i}")
        emitted.append(_row(r, state=state, action=action, label=True, source_kind=SFT_ROW))
    for i, p in enumerate(pair_rows):
        assert_same_state(p)
        kind = pair_kind_of(p)
        if kind not in set(include_pair_kinds):
            drops["pair_kind"] = drops.get("pair_kind", 0) + 1
            continue
        state = str(p.get("state_text") or p["chosen_state_text"])
        chosen, rejected = str(p["chosen_json"]), str(p["rejected_json"])
        _check_action(chosen, stop_action=stop_action, where=f"pair {i} chosen")
        _check_action(rejected, stop_action=stop_action, where=f"pair {i} rejected")
        cs, rs = _sides(p, kind, stop_action)
        emitted.append(
            _row(
                p,
                state=state,
                action=chosen,
                label=True,
                source_kind=cs,
                pair_kind=kind,
                candidate_run_id=str(p.get("chosen_run_id") or ""),
            )
        )
        emitted.append(
            _row(
                p,
                state=state,
                action=rejected,
                label=False,
                source_kind=rs,
                pair_kind=kind,
                candidate_run_id=str(p.get("rejected_run_id") or ""),
            )
        )

    return _resolve(
        emitted,
        on_conflict=on_conflict,
        drops=drops,
        include_pair_kinds=tuple(include_pair_kinds),
    )


def _assert_train_split(rows: Sequence[Mapping[str, Any]], *, where: str) -> None:
    """Rung 1's wall, with the FILE named. `assert_train_split` counts rows and cannot say which
    of the two inputs they came from, and "the wrong file" is the case the check exists for."""
    try:
        assert_train_split(list(rows))  # type: ignore[arg-type]
    except HeldOutDataset as exc:
        raise HeldOutDataset(f"in {where}: {exc}") from exc


def _key(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    """(suite, task, state, action). NOT run_id or turn_idx.

    The same question asked at the same state in two runs is ONE signal about that state, and
    counting it twice would weight a popular state by how many rollouts happened to visit it --
    which is a property of the sweep, not of the corpus. Suite and task are in the key because
    a state text is only unique within one; two suites could in principle render the same bytes.
    """
    return (
        str(row["suite_id"]),
        str(row["task_id"]),
        str(row["state_sha"]),
        str(row["completion"]),
    )


def _resolve(
    emitted: Sequence[dict[str, Any]],
    *,
    on_conflict: str,
    drops: dict[str, int],
    include_pair_kinds: tuple[str, ...],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    labels: dict[tuple[str, str, str, str], set[bool]] = {}
    for row in emitted:
        labels.setdefault(_key(row), set()).add(bool(row["label"]))
    conflicts = {k for k, v in labels.items() if len(v) > 1}
    conflict_rows = [r for r in emitted if _key(r) in conflicts]

    if conflicts and on_conflict == "refuse":
        examples = _conflict_examples(conflict_rows, conflicts)
        raise ConflictingLabels(
            f"{len(conflicts)} (state, action) keys carry BOTH labels "
            f"({len(conflict_rows)} rows). KTO scores a completion alone against the "
            "reference, so it has no way to represent 'better than C and worse than A at the "
            "same state' -- which is what the ask_ask tournament records. First "
            f"{len(examples)}: {examples}. Either fix the export, or declare "
            "on_conflict='drop', which removes BOTH sides of every conflicting key and counts "
            "them in the manifest. Nothing here keeps one label: that would turn a "
            "contradiction the corpus does not settle into an absolute claim.",
            n_conflicts=len(conflicts),
            report={
                "on_conflict": on_conflict,
                "n_emitted_before_dedupe": len(emitted),
                "n_conflicts": len(conflicts),
                "n_conflict_rows": len(conflict_rows),
                "conflict_examples": examples,
            },
        )

    kept: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    deduped: dict[str, int] = {}
    for row in emitted:
        key = _key(row)
        if key in conflicts:
            continue
        if key in kept:
            deduped[row["source_kind"]] = deduped.get(row["source_kind"], 0) + 1
            continue
        kept[key] = row
    rows = list(kept.values())

    n_d = sum(1 for r in rows if r["label"])
    n_u = len(rows) - n_d
    report: dict[str, Any] = {
        "n_rows": len(rows),
        "n_desirable": n_d,
        "n_undesirable": n_u,
        "undesirable_share": (n_u / len(rows)) if rows else 0.0,
        "n_by_source_kind": _counts(rows, "source_kind"),
        "n_by_pair_kind": _counts(rows, "pair_kind"),
        "n_by_stop_source": _counts(rows, "stop_source"),
        "n_by_action_kind": _counts(rows, "action_kind"),
        "suites": sorted({str(r["suite_id"]) for r in rows}),
        "n_states": len({(r["suite_id"], r["task_id"], r["state_sha"]) for r in rows}),
        # WHAT THE FILE HELD BEFORE THIS FUNCTION TOUCHED IT. A dedupe that removes 52,792 of
        # 139,815 rows without saying so is a dedupe nobody can reconcile against the export.
        "n_emitted_before_dedupe": len(emitted),
        "n_deduped": sum(deduped.values()),
        "n_deduped_by_source_kind": dict(sorted(deduped.items())),
        "n_conflicts": len(conflicts),
        "n_conflict_rows": len(conflict_rows),
        "n_conflict_rows_dropped": len(conflict_rows) if on_conflict == "drop" else 0,
        "conflict_examples": _conflict_examples(conflict_rows, conflicts),
        "on_conflict": on_conflict,
        "n_dropped_by_filter": dict(sorted(drops.items())),
        "include_pair_kinds": list(include_pair_kinds),
        # How much of the two guards above was vacuous, in rung 1's sense: a row carrying no
        # split was waved past the wall, and a row carrying no code_version cannot be placed
        # in a loop cohort. Both are silent unless counted.
        "n_rows_without_split": sum(1 for r in rows if not r["split"]),
        "n_rows_without_code_version": sum(1 for r in rows if not r["code_version"]),
    }
    # THE WEIGHTS THE ROWS THEMSELVES IMPLY, computed here so `pi train kto` has one number to
    # pin on the config rather than a formula to re-derive. Absent when a class is empty --
    # `preflight` refuses that case with its own message rather than one about weights.
    if n_d and n_u:
        w_d, w_u = balanced_weights(n_d, n_u)
        report["balanced_desirable_weight"] = w_d
        report["balanced_undesirable_weight"] = w_u
    return rows, report


def _counts(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rows:
        v = str(r.get(field) or "")
        if v:
            out[v] = out.get(v, 0) + 1
    return dict(sorted(out.items()))


def _conflict_examples(
    conflict_rows: Sequence[Mapping[str, Any]],
    conflicts: set[tuple[str, str, str, str]],
    *,
    limit: int = 3,
) -> list[dict[str, Any]]:
    """A few conflicting keys, named well enough to find them in the export."""
    out: list[dict[str, Any]] = []
    for key in sorted(conflicts)[:limit]:
        rows = [r for r in conflict_rows if _key(r) == key]
        out.append(
            {
                "suite_id": key[0],
                "task_id": key[1],
                "state_sha": key[2][:12],
                "action": key[3][:90],
                "desirable_from": sorted({r["source_kind"] for r in rows if r["label"]}),
                "undesirable_from": sorted({r["source_kind"] for r in rows if not r["label"]}),
            }
        )
    return out
