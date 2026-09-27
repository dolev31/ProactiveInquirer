#!/usr/bin/env python3
"""Keep the checkpoints that the dev verdicts say are best.

    keep_best.py --run RUN --eval-dir DIR --keep-dir DIR --source DIR [--source DIR ...]

Read every `<eval-dir>/<RUN>.ckpt-<tag>.tierA.json` written by `pi train eval-offline`
(scripts/hpc/eval_checkpoints.sh), choose a winner per criterion, and make sure the winner's
adapter is copied under `<keep-dir>/<criterion>/step-<tag>/` before the trainer's
`save_total_limit` deletes it. Two criteria, both from the held-out sample and both explained
in plan v3 §8.5:

  by_nll      the smallest dev NLL per token (the SFT objective on rows never trained on)
  by_stop2x2  the largest balanced STOP accuracy, (P(STOP | done) + P(ASK | not done)) / 2,
              the pair of cells tier A reads; a threshold shift moves them in opposite
              directions, which is why neither alone is the criterion

The untrained base (tag `0`) is never a candidate. `final` counts as the largest step. Ties go
to the later step (more training at equal dev quality). `--source` directories are searched
for `checkpoint-<tag>/` (the evaluator's node-local stage, the trainer's own --out); the tag
`final` is the adapter at the --out root itself. A winner whose adapter is no longer anywhere
on disk is recorded as `unavailable` and the previously kept copy stays: the record says which
tag SHOULD be there, so the loss is visible rather than silent.

ONE DIRECTORY PER STEP, AND NOTHING IS EVER OVERWRITTEN. The first version of this script kept
one flat `<keep-dir>/<criterion>/` and rewrote it whenever a later checkpoint won, which makes
the copy exactly as perishable as the checkpoint it was made to outlive. MEASURED 2026-09-16:
`qwen3-8b-sft-sw05`'s by_stop2x2 winner at step 1200 was registered in conf/checkpoints.json by
its adapter sha (cd0a1884...) at 20:00; at 20:24 the final checkpoint won the criterion and
overwrote `best/by_stop2x2/`; checkpoint-1200 had already rotated out of the trainer's
retention, so the registered weights then existed NOWHERE and that row had to be deleted. A
registry row naming weights that no longer exist is CONTRIBUTING.md rule 1 failing after the fact.

So the layout is:

  <keep-dir>/<criterion>/step-<tag>/   the adapter files plus TAG, one per step ever selected;
                                      written once, then never deleted, moved or rewritten
  <keep-dir>/<criterion>/CURRENT       one line: the tag that currently wins the criterion
  <keep-dir>/SELECTION.json            the pass's verdict, plus `history` per criterion: every
                                       step ever selected, with its metric AT SELECTION TIME

WHAT THAT COSTS. ~350 MB per kept step for the 8B r32 adapter (the optimizer state is not
selection material), against a 100 GB home quota. The winner changes a handful of times per
run, not once per checkpoint -- rung1-8b-headline selected three distinct by_nll steps over 27
checkpoints -- so the expected cost is a few GB per run, and the worst case (every checkpoint
wins its criterion in turn) is 27 x 2 x 350 MB ~= 19 GB. Nothing here prunes: a directory this
script has written is the only remaining copy of weights that may already be registered, and
deciding it is expendable is a decision for a human with the registry in front of them.

MIGRATION. A `<criterion>/adapter_model.safetensors` left by the overwriting version is copied
once into `step-<TAG>/` (its TAG file says which step it is) and then left alone -- both
copies stay, because the flat one may be what a `PINQ_LORA` map or a registry row is pointing
at. `SELECTION.json` records the migration in that criterion's history.

Stdlib only: this runs inside the evaluator job on the trainer venv, but must not depend on it.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")
OPTIONAL_FILES = (
    "README.md",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "chat_template.jinja",
)
CRITERIA = ("by_nll", "by_stop2x2")


def _step(tag: str) -> int:
    return sys.maxsize if tag == "final" else int(tag)


def _nll(v: dict) -> float | None:
    """Dev NLL per token, or None when absent or not finite.

    Same exposure as `_stop2x2`: a NaN loss is not None, and `min()` over NaN keys returns the
    first element just as `max()` does. A diverged checkpoint writing NaN would then win the
    criterion outright.
    """
    x = (v.get("sft_nll") or {}).get("nll_per_token")
    if x is None:
        return None
    f = float(x)
    return f if math.isfinite(f) else None


def _stop2x2(v: dict) -> float | None:
    """The balanced STOP accuracy, or None when either cell is missing OR NOT FINITE.

    NaN IS NOT None, AND THAT DISTINCTION SELECTED A CHECKPOINT BY DICT ORDER. When
    `PI_REFERENCE_ASK` is unset, `eval_checkpoints.sh` logs a warning and continues (unlike
    `eval_offline.sh`, which dies), every STOP row is skipped, and `p_stop_given_done` is written
    as NaN -- measured 2026-09-19 on all six E39-8b verdicts (n_done 0.0, n_not_done 350,
    n_skipped_no_reference 650). The `is None` guard did not fire, this returned `(nan+1.0)/2 =
    nan`, the candidate passed `select_best`'s `is not None` filter, and `max()` over all-NaN keys
    returns the FIRST element because every NaN comparison is False. Reversing the insertion order
    of the same measurements changed the winner from `200` to `600`, which
    tests/test_keep_best_rejects_nonfinite.py pins.

    A criterion that reports a winner chosen by iteration order is worse than one that reports no
    winner: the registry row would name weights selected by nothing. So a non-finite cell
    disqualifies the candidate, `by_stop2x2` comes back None, and `by_nll` -- which is unaffected
    -- still decides.
    """
    s = v.get("stop_confusion") or {}
    a, b = s.get("p_stop_given_done"), s.get("p_ask_given_not_done")
    if a is None or b is None:
        return None
    fa, fb = float(a), float(b)
    if not (math.isfinite(fa) and math.isfinite(fb)):
        return None
    return (fa + fb) / 2.0


METRIC_FN = {"by_nll": _nll, "by_stop2x2": _stop2x2}


def unrankable_tags(verdicts: dict[str, dict], crit: str) -> list[str]:
    """Candidates whose cells are PRESENT but whose metric is not a finite number.

    `_nll` and `_stop2x2` return None both when a cell is missing and when it is non-finite, so
    `select_best` drops those candidates -- correctly, since ranking on NaN selected by dict order
    (ed03455). But it drops them SILENTLY, and `by_stop2x2: None` then records `status: "none"`,
    which is exactly what a run with zero verdicts records. Those are opposite situations: the
    three granite runs had nothing scored at all, while E39-8b s0/s1/s2 had three scored verdicts
    whose `p_stop_given_done` was NaN because `PI_REFERENCE_ASK` was unset.

    `pair_accuracy`'s `_finite` convention is the rule this follows: an empty cell must not print
    what a measured value prints, because the two readings are opposite. Here it matters more,
    because the next person will not know the environment variable exists -- so the record has to
    say WHY there is no winner and name the candidates it could not rank, rather than leaving a
    bare None that looks like "nothing to do".
    """
    fn = METRIC_FN[crit]
    out = []
    for tag, v in verdicts.items():
        if tag == "0":
            continue  # the untrained base is never a candidate
        if fn(v) is None and _raw_cells_present(v, crit):
            out.append(tag)
    return sorted(out)


def _raw_cells_present(v: dict, crit: str) -> bool:
    """Whether the verdict HAS the fields the metric reads, whatever their values."""
    if crit == "by_nll":
        return (v.get("sft_nll") or {}).get("nll_per_token") is not None
    s = v.get("stop_confusion") or {}
    return s.get("p_stop_given_done") is not None and s.get("p_ask_given_not_done") is not None


def select_best(verdicts: dict[str, dict]) -> dict[str, str | None]:
    """The winning tag per criterion, or None when no checkpoint qualifies."""
    cands = {t: v for t, v in verdicts.items() if t != "0"}
    out: dict[str, str | None] = {}
    nll = [(t, _nll(v)) for t, v in cands.items() if _nll(v) is not None]
    out["by_nll"] = min(nll, key=lambda tv: (tv[1], -_step(tv[0])))[0] if nll else None
    stop = [(t, _stop2x2(v), _nll(v)) for t, v in cands.items() if _stop2x2(v) is not None]
    out["by_stop2x2"] = (
        max(stop, key=lambda t: (t[1], -(t[2] if t[2] is not None else float("inf")), _step(t[0])))[
            0
        ]
        if stop
        else None
    )
    return out


def load_verdicts(run: str, eval_dir: Path) -> dict[str, dict]:
    prefix = f"{run}.ckpt-"
    out: dict[str, dict] = {}
    for p in sorted(eval_dir.glob(f"{prefix}*.tierA.json")):
        tag = p.name[len(prefix) : -len(".tierA.json")]
        try:
            out[tag] = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue  # a verdict still being written; next pass
    return out


def find_adapter(tag: str, sources: list[Path]) -> Path | None:
    for src in sources:
        cand = src if tag == "final" else src / f"checkpoint-{tag}"
        if all((cand / f).is_file() for f in ADAPTER_FILES):
            return cand
    return None


def _copy_adapter(src: Path, dst: Path, tag: str) -> None:
    """Copy one adapter into a step directory that does not exist yet.

    The `FileExistsError` is the guarantee, not an inconvenience: the whole point of this
    layout is that a copy already made is never a candidate for replacement, and a function
    that could be passed an existing `dst` would be one edit away from the bug it replaced.
    TAG is written INSIDE the staging directory, so the rename publishes a directory that
    already names its own step -- a job killed mid-copy leaves `step-<tag>.partial`, which no
    reader looks at, rather than weights whose provenance is a file that was never written.
    """
    if dst.exists():
        raise FileExistsError(f"{dst} already exists; keep_best never rewrites a kept copy")
    tmp = dst.with_name(dst.name + ".partial")
    shutil.rmtree(tmp, ignore_errors=True)  # a previous job's abandoned staging, never a copy
    tmp.mkdir(parents=True)
    for f in ADAPTER_FILES + OPTIONAL_FILES:
        if (src / f).is_file():
            shutil.copy2(src / f, tmp / f)
    (tmp / "TAG").write_text(tag + "\n")
    tmp.rename(dst)  # appears complete or not at all


def step_dir(crit_dir: Path, tag: str) -> Path:
    return crit_dir / f"step-{tag}"


def _is_kept(d: Path) -> bool:
    return all((d / f).is_file() for f in ADAPTER_FILES)


def _migrate_flat(crit_dir: Path) -> str | None:
    """Copy a flat `<criterion>/` adapter from the overwriting version into `step-<TAG>/`.

    Returns the tag migrated, or None when there is nothing to migrate or it has already been
    done. NOTHING IS DELETED: the flat copy stays where it is, because a `PINQ_LORA` map or a
    conf/checkpoints.json row may be pointing at that exact path, and because the flat copy is
    now frozen (this script writes only step directories from here on) it is safe to leave.
    A flat adapter with no TAG names no step and is left alone rather than guessed at.
    """
    if not _is_kept(crit_dir):
        return None
    tag_file = crit_dir / "TAG"
    tag = tag_file.read_text().strip() if tag_file.is_file() else ""
    if not tag:
        return None
    dst = step_dir(crit_dir, tag)
    if dst.exists():
        return None  # migrated by an earlier pass; once means once
    _copy_adapter(crit_dir, dst, tag)
    return tag


def _history_entry(tag: str, metrics: dict | None, when: str) -> dict:
    """One step that won a criterion, and the metric it won at.

    `metrics` is recorded AT SELECTION TIME and never refreshed: the history has to be
    readable after the verdict files have been archived, and a metric re-derived later is a
    different measurement from the one the selection was made on.
    """
    return {"tag": tag, "metrics": metrics, "selected_at": when, "kept": False, "dir": None}


def keep(*, run: str, eval_dir: Path, keep_dir: Path, sources: list[Path]) -> dict:
    verdicts = load_verdicts(run, eval_dir)
    best = select_best(verdicts)
    keep_dir.mkdir(parents=True, exist_ok=True)
    sel_path = keep_dir / "SELECTION.json"
    previous = json.loads(sel_path.read_text()) if sel_path.is_file() else {}
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    selection: dict = {
        "run": run,
        "eval_dir": str(eval_dir),
        "criteria": {
            "by_nll": "min sft_nll.nll_per_token; ties -> later step; tag 0 excluded",
            "by_stop2x2": "max (p_stop_given_done + p_ask_given_not_done)/2; ties -> lower nll, later step",
        },
        "layout": "<criterion>/step-<tag>/ per step ever selected, written once and never "
        "rewritten; <criterion>/CURRENT names the current winner",
        "n_verdicts": len(verdicts),
        "updated": now,
    }
    for crit in CRITERIA:
        tag = best.get(crit)
        crit_dir = keep_dir / crit
        prev_entry = previous.get(crit) or {}
        # Carried forward, not rebuilt: SELECTION.json is rewritten every pass, so a step that
        # won at 20:00 and lost at 20:24 is only recoverable from the record it was written in.
        history = [dict(h) for h in (prev_entry.get("history") or [])]
        by_tag = {h.get("tag"): h for h in history}

        entry: dict = {"tag": tag, "kept_tag": None, "status": "none"}
        # NO WINNER IS TWO DIFFERENT SITUATIONS. See `unrankable_tags`: nothing scored at all, or
        # scored and unrankable. Only the second names tags, and it must not read as the first.
        if tag is None:
            cannot = unrankable_tags(verdicts, crit)
            if cannot:
                entry["status"] = "unrankable"
                entry["unrankable_tags"] = cannot
                entry["unrankable_reason"] = (
                    f"{len(cannot)} candidate(s) carry the cells this criterion reads, but the "
                    "metric is not a finite number. For by_stop2x2 the usual cause is an unset "
                    "PI_REFERENCE_ASK, which skips every STOP row and writes "
                    "p_stop_given_done as NaN. No selection was made rather than an arbitrary one."
                )

        migrated = _migrate_flat(crit_dir)
        if migrated is not None:
            entry["migrated_flat_to"] = f"step-{migrated}"
            if migrated not in by_tag:
                h = _history_entry(
                    migrated,
                    prev_entry.get("metrics") if prev_entry.get("kept_tag") == migrated else None,
                    previous.get("updated") or now,
                )
                h["kept"], h["dir"] = True, f"step-{migrated}"
                h["note"] = "migrated from the flat <criterion>/ copy, which was left in place"
                history.append(h)
                by_tag[migrated] = h

        cur_file = crit_dir / "CURRENT"
        current = cur_file.read_text().strip() if cur_file.is_file() else None
        if current is None and migrated is not None:
            current = migrated
            cur_file.write_text(current + "\n")
        entry["kept_tag"] = current

        if tag is not None:
            entry["metrics"] = {
                "nll_per_token": _nll(verdicts[tag]),
                "stop2x2": _stop2x2(verdicts[tag]),
            }
            dst = step_dir(crit_dir, tag)
            if _is_kept(dst):
                entry["kept_tag"] = tag  # copied by an earlier pass; not touched again
                entry["status"] = "kept"
            elif dst.exists():
                # Present but not a complete adapter. Not this script's to delete (it did not
                # write it) and not this script's to trust. Named, and left exactly as found.
                entry["status"] = "incomplete"
            else:
                src = find_adapter(tag, sources)
                if src is None:
                    entry["status"] = "unavailable"  # the trainer deleted it before we copied
                else:
                    _copy_adapter(src, dst, tag)
                    entry["kept_tag"] = tag
                    entry["status"] = "copied"
                    entry["copied_from"] = str(src)
            h = by_tag.get(tag)
            if h is None:
                h = _history_entry(tag, entry["metrics"], now)
                history.append(h)
                by_tag[tag] = h
            if entry["kept_tag"] == tag:
                h["kept"], h["dir"] = True, f"step-{tag}"
                if current != tag:
                    cur_file.write_text(tag + "\n")
                    current = tag
        entry["current"] = current
        entry["history"] = history
        selection[crit] = entry
        prev = prev_entry.get("tag")
        if prev is not None and prev != tag:
            selection[crit]["previous_tag"] = prev
    sel_path.write_text(json.dumps(selection, indent=2) + "\n")
    return selection


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--eval-dir", required=True, type=Path)
    ap.add_argument("--keep-dir", required=True, type=Path)
    ap.add_argument(
        "--source", action="append", type=Path, default=[], help="where checkpoint-<tag>/ may be"
    )
    a = ap.parse_args(argv)
    sel = keep(run=a.run, eval_dir=a.eval_dir, keep_dir=a.keep_dir, sources=a.source)
    for crit in CRITERIA:
        e = sel[crit]
        steps = ",".join(h["tag"] for h in e["history"] if h["kept"])
        print(
            f"keep_best {crit}: tag={e['tag']} current={e['current']} status={e['status']} "
            f"metrics={e.get('metrics')} kept_steps=[{steps}]"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
