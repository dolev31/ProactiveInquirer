"""ONE walk of `runs/`, TWO SFT files: the rule of record, and the answer-node variant.

WHY NOT TWO `pi train export` INVOCATIONS. `runs/` is shared and live -- peers write into it
while this runs -- so two walks minutes apart can see different run directories. Every row that
one walk saw and the other did not would then appear in the label-shift table as a decision the
rule changed. One `collect_rows`, two `export_sft` calls over the SAME list, is the only way the
table means what its column headers say. It is also 174k run directories of scoring that do not
have to happen twice.

WHAT IS HELD IDENTICAL. Everything `export_sft` takes except `stop_label`: the same rows, the
same cohort file and mode, the same margin threshold, the same split, the same arm allowlist,
the same `turn0_any_cohort`. `--assert-unchanged` then re-checks that claim against the two
manifests rather than asserting it in prose.

THE DIRECTION IS FIXED BEFORE THE DATA IS SEEN. An answer node is a required node, so
`done_before is True` implies `answer_node_covered_before is True`; the variant's STOP set is a
superset of v1's, and the only states that can move STOP -> ASK are those whose answer node is
UNKNOWN. `--refuse-unexpected-direction` (default on) makes an observation to the contrary stop
the export instead of quietly writing a file that contradicts the invariant it was built on.

USAGE (gold-side; `PI_GOLD_ROOT` must be readable -- this is not a rollout path):

    PYTHONPATH=<worktree>/src <repo>/.venv/bin/python -m scripts.answer_node_stop.export_variants \\
        --root <repo> --runs-root <repo>/runs --gold-root <repo>/data/gold \\
        --cohort-file <repo>/conf/cohorts/fixed_loop.json --cohort fixed --turn0-any-cohort \\
        --include-arm inquirer_prompted --tau 0.05 --sigma-j 0.0 \\
        --out <repo>/data/rl/answer_node_20260919
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--root", required=True, help="repo root: corpora, conf and the default runs")
    p.add_argument("--runs-root", default=None, help="default <root>/runs")
    p.add_argument("--gold-root", required=True, help="sets PI_GOLD_ROOT for this process")
    p.add_argument(
        "--out", required=True, help="a NEW directory; both files and manifests land here"
    )
    p.add_argument("--graph-version", default="v1")
    p.add_argument("--split", default="train", choices=("train", "dev"))
    p.add_argument("--include-arm", action="append", default=None)
    p.add_argument("--cohort", default="any", choices=("any", "fixed", "old"))
    p.add_argument("--cohort-file", default=None)
    p.add_argument("--turn0-any-cohort", action="store_true")
    p.add_argument("--tau", type=float, default=0.05)
    p.add_argument("--sigma-j", type=float, default=0.0)
    p.add_argument("--max-examples-per-task", type=int, default=None)
    p.add_argument(
        "--no-verify-state",
        action="store_true",
        help="do not. `render_state`'s hash check is what makes `state_text` the prompt the "
        "policy saw; it is here only so the flag the CLI has is not silently missing.",
    )
    p.add_argument(
        "--refuse-unexpected-direction",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="refuse if any state moves STOP -> ASK for a reason other than an unknown answer "
        "node. Default on: that transition is impossible under the subset invariant, so seeing "
        "one means the invariant broke and every number downstream is about something else.",
    )
    p.add_argument("--report", default=None, help="write the label-shift report as JSON here too")
    return p


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    os.environ["PI_GOLD_ROOT"] = str(Path(a.gold_root).resolve())

    from scripts.answer_node_stop import label_shift as ls

    from pi_run.cmd_train import (
        DEFAULT_TRAINABLE_ARMS,
        _cohort,
        _stamp_arms,
        _train,
        collect_rows,
    )

    dataset = _train("export.dataset")
    rung1 = _train("rung1_sft")

    root = Path(a.root).resolve()
    runs_root = Path(a.runs_root).resolve() if a.runs_root else root / "runs"
    out_dir = Path(a.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    margin = rung1.margin_threshold(a.tau, a.sigma_j)
    include_arms = frozenset(a.include_arm) if a.include_arm else DEFAULT_TRAINABLE_ARMS
    cohort, cohort_sha = _cohort(a.cohort_file)

    print(f"runs        {runs_root}", flush=True)
    print(f"arms        {sorted(include_arms)}", flush=True)
    print(
        f"cohort      {a.cohort}  turn0_any={bool(a.turn0_any_cohort)}  sha={cohort_sha[:12]}",
        flush=True,
    )
    print(f"margin      {margin:.4f}  (tau {a.tau}, sigma_j {a.sigma_j})", flush=True)

    t0 = time.time()
    rows, skipped = collect_rows(
        runs_root,
        root,
        graph_version=a.graph_version,
        weights=_train("reward").RewardWeights(),
        verify_state=not a.no_verify_state,
        include_arms=include_arms,
        split=a.split,
    )
    walk_s = time.time() - t0
    n_with_field = sum(1 for r in rows if "answer_node_covered_before" in r)
    print(f"rows        {len(rows)} in {walk_s / 60:.1f} min", flush=True)
    print(
        f"            {n_with_field} carry answer_node_covered_before"
        f" ({len(rows) - n_with_field} unknown)",
        flush=True,
    )
    if skipped:
        print(
            "not exported: " + ", ".join(f"{k}={v}" for k, v in sorted(skipped.items())), flush=True
        )

    common = dict(
        margin_threshold=margin,
        max_examples_per_task=a.max_examples_per_task,
        split=a.split,
        cohort=cohort,
        cohort_file_sha=cohort_sha,
        cohort_mode=a.cohort,
        turn0_any_cohort=bool(a.turn0_any_cohort),
    )
    # TWO EXPORTS OVER ONE ROW LIST. `export_sft` consumes `rows` as an iterable and does not
    # mutate it (it copies into `by_state`), so the second call sees exactly what the first did.
    ex_v1, man_v1 = dataset.export_sft(rows, stop_label="done_before", **common)
    ex_an, man_an = dataset.export_sft(rows, stop_label="answer_node_covered_before", **common)
    # The SAME stamp the CLI applies, from the same function: `export_sft` takes rows and has
    # never seen a run directory, so the arm refusals counted during the walk reach the manifest
    # only here. A second implementation is a second place for `included_arms` to be wrong.
    for man in (man_v1, man_an):
        _stamp_arms(man, include_arms, skipped)

    sh = ls.shift(ex_v1, ex_an)
    print("\n--- label shift (state level, joined on the state key) ---", flush=True)
    print(sh.render(), flush=True)

    if a.refuse_unexpected_direction and sh.stop_to_ask > man_an.n_answer_node_unknown:
        print(
            f"REFUSED: {sh.stop_to_ask} states moved STOP -> ASK against at most "
            f"{man_an.n_answer_node_unknown} whose answer node is unknown. An answer node is a "
            "required node, so `done_before` implies it is covered and that transition cannot "
            "happen for any other reason. Nothing was written.",
            file=sys.stderr,
        )
        return 3

    paths: dict[str, str] = {}
    for name, items, man in (("sft", ex_v1, man_v1), ("sft.answer_node", ex_an, man_an)):
        p = dataset.write_jsonl(out_dir / f"{name}.jsonl", items, man)
        n_stop, n_rows, share = ls.stop_share(items)
        wt = ls.within_task_distinct3(items)
        paths[name] = str(p)
        print(f"\n{name}.jsonl  n={n_rows}  sha256={_sha256(Path(p))[:16]}", flush=True)
        print(f"  sft_stop_label        {man.sft_stop_label}", flush=True)
        print(f"  train_id_set_hash     {man.train_id_set_hash}", flush=True)
        print(f"  stop rows / share     {n_stop} / {share:.4f}  (ceiling 0.80)", flush=True)
        print(f"  within-task distinct3 {wt:.4f}  (floor 0.65)", flush=True)
        print(f"  n_answer_node_unknown {man.n_answer_node_unknown}", flush=True)
        print(f"  n_no_target_dropped   {man.n_no_target_dropped}", flush=True)

    invariant = [
        "n_cohort_refused",
        "n_answer_leak_dropped",
        "n_binary_gold_hedged_nan",
        "n_states_mixed_cap",
        "n_state_done_disagree",
        "n_answer_node_state_disagree",
        "n_answer_node_unknown",
        "n_done_before_unknown",
        "n_arm_refused",
    ]
    held = ls.unchanged_counters(man_v1, man_an, invariant)
    print("\nlabel-invariant counters (equal on both manifests, checked):", flush=True)
    for k, v in held.items():
        print(f"  {k:<32} {v}", flush=True)
    moved = ls.manifest_deltas(man_v1, man_an)
    print("counters that moved:", flush=True)
    for k, (x, y) in moved.items():
        print(f"  {k:<32} {x} -> {y}", flush=True)

    report = {
        "runs_root": str(runs_root),
        "n_rows": len(rows),
        "n_rows_with_answer_node_field": n_with_field,
        "walk_seconds": round(walk_s, 1),
        "skipped": skipped,
        "shift": {
            "stop_to_stop": sh.stop_to_stop,
            "stop_to_ask": sh.stop_to_ask,
            "ask_to_stop": sh.ask_to_stop,
            "ask_to_ask": sh.ask_to_ask,
            "only_in_v1": sh.only_in_v1,
            "only_in_variant": sh.only_in_variant,
            "n_v1": sh.n_v1,
            "n_variant": sh.n_variant,
            "stop_share_v1": sh.stop_share_v1,
            "stop_share_variant": sh.stop_share_variant,
        },
        "files": paths,
        "manifest_deltas": {k: list(v) for k, v in moved.items()},
        "label_invariant": dict(held),
        "v1": {
            "train_id_set_hash": man_v1.train_id_set_hash,
            "n_examples": man_v1.n_examples,
            "sha256": _sha256(Path(paths["sft"])),
        },
        "variant": {
            "train_id_set_hash": man_an.train_id_set_hash,
            "n_examples": man_an.n_examples,
            "sha256": _sha256(Path(paths["sft.answer_node"])),
        },
    }
    dest = Path(a.report) if a.report else out_dir / "label_shift.json"
    dest.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {dest}", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover - a driver
    raise SystemExit(main())
