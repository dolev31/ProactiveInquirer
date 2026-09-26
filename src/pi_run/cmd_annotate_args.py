"""`pi annotate`'s argument surface, kept free of pi_eval.

Every `pi` process builds its whole parser before it builds a process pool, and on Linux (fork,
the Python 3.12 default) a pool child inherits its parent's sys.modules whole. `cmd_annotate`
imports pi_eval.annotate at module scope, so registering from there put pi_eval into every
rollout worker on the cluster, and `pi verify firewall` reported pi_eval_imported true there.
macOS spawns, which re-imports only __main__ and hid it; eebf52a moved the import into
`build_parser` and was right only under spawn. So the handlers are resolved from `cmd_annotate`
when a subcommand runs, and the two pi_eval constants this surface shows are mirrored here and
pinned equal by tests/test_runtime.py.
"""

from __future__ import annotations

import argparse
from typing import Any, Callable

DEFAULT_OUT_VERSION = "v1h"

# Mirrors of pi_eval.annotate.A6_MIN_CANDIDATES and pi_eval.annotate.TASK_TYPES.
A6_MIN_CANDIDATES = 3
TASK_TYPES = (
    "A1",
    "A2",
    "A3_node",
    "A3_edge",
    "A3_match",
    "A3_missing",
    "A4",
    "A5",
    "A6",
    "A7",
    "B1",
    "B1b",
    "B2",
    "B3",
    "B4",
)


def _handler(name: str) -> Callable[[argparse.Namespace], Any]:
    def run(a: argparse.Namespace) -> Any:
        from pi_run import cmd_annotate

        return getattr(cmd_annotate, name)(a)

    run.__name__ = name
    return run


def register(sub: argparse._SubParsersAction) -> None:
    """Attach `pi annotate` to the top-level dispatch table."""
    p = sub.add_parser("annotate", help="human annotation campaign: export, import, review")
    s = p.add_subparsers(dest="sub", required=True)

    e = s.add_parser(
        "export", help="sample a blinded bundle from gold + recorded runs (NEEDS gold)"
    )
    e.add_argument("--suite", required=True)
    e.add_argument(
        "--a3-suite",
        default=None,
        help="draw A3_node/A3_edge/A3_match/A3_missing (and attention checks) from a "
        "different suite than --suite; default is --suite itself",
    )
    e.add_argument("--graph-version", default="v1")
    e.add_argument("--gold-root", default=None, help="sets PI_GOLD_ROOT for this process")
    e.add_argument("--root", default=None)
    e.add_argument("--runs-root", default=None)
    e.add_argument("--parquet", default=None, help="default <root>/scores/parquet")
    e.add_argument("--pairs", default=None, help="default <root>/data/rl/pairs.jsonl")
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--n-a1", type=int, default=60)
    e.add_argument("--n-a2", type=int, default=100)
    e.add_argument("--n-a3-node", type=int, default=150)
    e.add_argument("--n-a3-edge", type=int, default=80)
    e.add_argument("--n-a3-match", type=int, default=100)
    e.add_argument("--n-a4", type=int, default=120)
    e.add_argument(
        "--a5-blind",
        action="store_true",
        help=(
            "build A5 items that withhold BOTH the final answer and the unresolved-need list. "
            "The default variant shows the gold frontier and the verdict tracks it: 0.0%% of "
            "264 items said 'should have asked more' when the list was empty against 100%% "
            "when it named three or more, and the separation against blinded answer "
            "correctness falls from 24.5 points to 9.0 once that list is held fixed."
        ),
    )
    e.add_argument(
        "--n-a5",
        type=int,
        default=80,
        help="STOP-decision items, one per policy_stop run of --suite (never --a3-suite)",
    )
    e.add_argument(
        "--n-a6",
        type=int,
        default=40,
        help=f"candidate-RANKING items: one per forked state with >= {A6_MIN_CANDIDATES} "
        "distinct candidates, each yielding C(k,2) pairwise preferences from one annotation "
        "slot. Read from the same --pairs file A2 is drawn from",
    )
    e.add_argument(
        "--n-a7",
        type=int,
        default=0,
        help="anticipation items: A2's pairs re-asked as a per-candidate reaches_unstated "
        "judgment plus an anticipation preference, stratified half/half on the mechanical "
        "is_latent flag they relabel. Read from the same --pairs file A2 is drawn from",
    )
    e.add_argument(
        "--n-a7-foils",
        type=int,
        default=0,
        help="planted A7 items whose coin-flipped side is the task question verbatim (honest "
        "reaches judgment: stays_stated); marked ONLY in the key, never gated -- see "
        "pi_eval.annotate._collect",
    )
    e.add_argument(
        "--n-attention",
        type=int,
        default=0,
        help="planted attention-check items (task's question paired with an unrelated task's "
        "need); marked ONLY in the key, never gated -- see pi_eval.annotate._collect",
    )
    e.add_argument("--out", default=None, help="default <gold_root>/human/<suite>/bundles")
    e.set_defaults(fn=_handler("cmd_annotate_export"))

    i = s.add_parser(
        "import", help="records -> gold graph version + gates + judgments (NEEDS gold)"
    )
    i.add_argument("--suite", required=True)
    i.add_argument("--bundle", required=True)
    i.add_argument("--key", required=True)
    i.add_argument("--records", action="append", required=True)
    i.add_argument("--out-version", default=DEFAULT_OUT_VERSION)
    i.add_argument("--min-annotators", type=int, default=2)
    i.add_argument("--allow-majority", action="store_true")
    i.add_argument("--n-missing-adjudicated", type=int, default=None)
    i.add_argument("--parquet", default=None, help="default <root>/scores/parquet")
    i.add_argument("--gold-root", default=None, help="sets PI_GOLD_ROOT for this process")
    i.add_argument("--root", default=None)
    i.add_argument("--dry-run", action="store_true")
    i.set_defaults(fn=_handler("cmd_annotate_import"))

    r = s.add_parser(
        "review", help="read-only agreement + degeneracy report, runnable mid-collection"
    )
    r.add_argument("--bundle", required=True)
    r.add_argument("--key", default=None)
    r.add_argument("--records", action="append", required=True)
    r.add_argument("--json", action="store_true")
    r.set_defaults(fn=_handler("cmd_annotate_review"))

    m = s.add_parser(
        "llm",
        help="automatic LLM annotation pass over a bundle (no --key, no PI_GOLD_ROOT)",
    )
    m.add_argument("--bundle", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--model", default=None, help="default PI_MODEL_ANNOTATOR")
    m.add_argument("--seed", type=int, default=0)
    m.add_argument("--limit", type=int, default=None)
    m.add_argument("--task-type", action="append", choices=TASK_TYPES, default=None)
    m.add_argument("--max-usd", type=float, default=None)
    m.add_argument("--cache-root", default=None)
    m.add_argument("--resume", action="store_true")
    m.add_argument(
        "--overwrite",
        action="store_true",
        help="intentionally discard a non-empty --out and start this pass over",
    )
    m.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="annotation calls in flight at once (default 1: sequential, byte-for-byte "
        "unchanged). Uses a THREAD pool, not the process pool `pi run` sweeps use -- this "
        "pass is I/O-bound on one HTTP endpoint with no per-item state to isolate, unlike a "
        "rollout unit; see _run_llm_pass_concurrent's docstring. With N>1: (1) --max-usd can "
        "OVERSHOOT by up to N-1 calls already in flight when the cap trips -- dispatch of new "
        "work stops immediately, in-flight calls are not cancelled or refunded; (2) --out is "
        "still written incrementally, one JSON object per completed call, and is re-sorted by "
        "item_id once the pass finishes normally, so two runs at different concurrencies "
        "produce byte-identical output regardless of which call happened to finish first.",
    )
    m.set_defaults(fn=_handler("cmd_annotate_llm"))
