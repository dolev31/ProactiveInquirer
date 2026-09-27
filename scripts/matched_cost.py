#!/usr/bin/env python3
"""P13 MATCHED COST: the tier-B gate re-read at equal retrieval cost.

    PI_GOLD_ROOT="$PWD/data/gold" .venv/bin/python scripts/matched_cost.py \
        --gate artifacts/gate --out figures

WHAT THE GATE COMPARED, AND WHY IT IS NOT THE PAPER'S ENDPOINT. The tier-B dev gate
(`artifacts/gate/*.json`) contrasts each trained checkpoint against the prompted base at
retrieval cap 8, paired within task. Every checkpoint LOSES on `evidence_coverage`, `cad_ge2`
and `facet_breadth` there -- and asks half to a quarter as many questions. The paper's primary
endpoint is an ORDERED PAIR, (delta coverage, delta retrieval calls), and its claim is "more
required evidence at EQUAL retrieval cost". A cap-8 contrast charges the trained policy for
stopping early and credits it with nothing for the cost it saved, so it cannot decide that
claim in either direction.

WHY A PREFIX LADDER IS A LEGITIMATE MATCHED-COST COMPARATOR. `Inquirer.act(s: State)` takes no
ledger, no context and no budget (`src/pinq/protocols.py`), so the base policy's k-th question
is a function of the state alone. The first k questions of a cap-8 prompted run are therefore
the questions a cap-k prompted run would have asked, and its coverage after k asks is that
run's terminal coverage. A budget-aware policy would break this, which is exactly why the
protocol forbids one.

WHAT IS REUSED RATHER THAN REIMPLEMENTED. Everything that touches gold:

  * `pi_eval.matcher.base.MechanicalMatcher` (matcher_id `mechanical_v3`) decides RESOLVE, run
    once per PREFIX against the same `pi_eval.score` parquet shims the scorer itself uses;
  * `pi_eval.metrics.discovery.evidence_coverage` is Q;
  * `pi_eval.score._gold_uids` picks the required gold spans Q divides by;
  * `pi_eval.stats.inference.paired_difference` produces every interval (BCa cluster bootstrap)
    and every p (cluster-level sign flip).

`cad_ge2` is the |V_d|-weighted C@d>=2 of `pinq_train.gate.cad_ge2_by_run`, which over depths
d>=2 reduces algebraically to (# depth>=2 nodes resolved) / (# depth>=2 nodes) -- restated here
in node terms because the prefix version has no `scores.parquet` rows to weight.

THE INSTRUMENT IS PROVED IDENTICAL, NOT ASSERTED. `verify_instrument` recomputes the FULL
trajectory from these ladders and compares it to what the verdicts' own scorer wrote:
`frontier_q#k` for every k, `evidence_coverage`, and `cad#d` / `cad_n#d` for every depth. Any
disagreement raises `InstrumentMismatch` and no table is printed. A matched-cost delta computed
with a slightly different matcher would look exactly like a result.

THE SECOND COST BASIS, AND WHY THE RETRIEVAL AXIS NEEDED ONE. A longer question costs the same
one retrieval call, so matched-k cannot answer "was the coverage gain bought with verbosity?" --
and the trained arms ask visibly longer questions (18.07 words against the base's 12.99 on
strategyqa). So the base is also charged the Inquirer BUDGET the trained arm spends, on two
bases, reported side by side and never averaged:

  * `matched_gen_tokens` -- what a token bill charges: per-call `tok_completion + tok_reasoning`
    from `calls.parquet` WHERE `actor = 'inquirer'` (the same column `pinq_train.gate` resolves
    `--baseline-model-id` through). MEASURED on artifacts/gate/n1 and /8b2-t20: the Inquirer's
    `tok_reasoning` is 0 on all 7,522 of its calls, so "completion including reasoning" and
    "completion minus reasoning" are the SAME number here and cannot disagree. They will diverge
    the day the Inquirer pin is reasoning-capable, which is why they are computed apart.
  * `matched_question_words` -- the LENGTH CONTROL, denominated in the unit the objection is
    stated in. It is a labelled PROXY for the question's token count: no tokenizer is installed
    in this venv, so the question's real tokens are not recoverable without adding one.

WHY THE LENGTH CONTROL IS NOT `tok_completion`. Because of a measurement, not a preference. The
Inquirer's completion is `question` + `rationale` + `target`, and on artifacts/gate/n1 the BASE's
rationale is the longer half (131 chars against the trained arm's 77) while its question is the
shorter one (79 against 110). Its completion is therefore 80 tokens per call against the trained
arm's 56 -- more generated tokens for a SHORTER question. A budget denominated in completion
hands the base a rationale allowance and biases the test toward "the gain survives", which is
the one failure a length control exists to prevent.

HOW A CALL IS ATTRIBUTED TO AN ASK. `calls.parquet` carries `actor` but no `turn_idx`; the
ledger carries `turn_idx` but no `actor`. The bridge is measured, not assumed: on all 1,395 runs
of artifacts/gate/n1/parquet_qwen3-8b-dpo-stacked-notdone-both the calls file order reproduces
the ledger's charge order `(tok_prompt, tok_completion)` value for value, and the i-th Inquirer
call sits at `turn_idx == i`, with `n_inquirer_calls == n_asks + 1` on every run of both gates,
both arms and both stop reasons -- one call per ask plus one terminal decision call.
`build_ladders` refuses a run that breaks that count rather than charging a budget to the wrong
rung.

WHAT THIS SCRIPT WILL NOT DO. It makes no LLM call, rolls out nothing, and writes no figure it
did not measure; a source parquet that is absent is a refusal, never an empty axis.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import glob
import hashlib
import json
import math
import os
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import duckdb

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:  # a checkout without an editable install
    sys.path.insert(0, str(REPO / "src"))

from pi_eval.gold import GoldGraph, load_graphs  # noqa: E402  (after the sys.path fix)
from pi_eval.matcher.base import MechanicalMatcher  # noqa: E402
from pi_eval.metrics.discovery import evidence_coverage  # noqa: E402
from pi_eval.score import _Act, _gold_uids, _Outcome, _Traj, _Turn  # noqa: E402
from pi_eval.stats.inference import cluster_bootstrap, paired_difference  # noqa: E402
from pinq.ids import canon, h  # noqa: E402

TOOL = "scripts/matched_cost.py"
FIGURE_ID = "P13_matched_cost"
GRAPH_VERSION = "v1"
BASE_ARM = "inquirer_prompted"
TRAINED_ARM = "inquirer_trained"
CAP = 8  # the gate's retrieval cap, and the top rung of the ladder
N_BOOT = 1000
N_PERM = 1000
SEED = 0
RESOLVE_RANK = 2  # MatchRecord.rank >= 2 is RESOLVE (or USE); see pi_eval.metrics.structure

# `parquet_<tag>/` -> the checkpoint whose 465 runs it holds. One parquet per checkpoint
# because `pi train gate` selects the checkpoint arm by grid name alone; see
# artifacts/gate/README.md for why pooling them would be silent. The pass-1 gate (2026-09-15
# morning) named its three directories by a short tag rather than by the model id; that
# layout is recognised by `PASS1_CHECKPOINTS` so the committed P13 digests stay put, and
# every later gate names `parquet_<model>/` after the registry id its verdicts carry.
PASS1_CHECKPOINTS: tuple[tuple[str, str], ...] = (
    ("parquet_sft", "qwen3-8b-sft"),
    ("parquet_headline", "qwen3-8b-sft-headline"),
    ("parquet_headline_stop", "qwen3-8b-sft-headline-stop"),
)
_PASS1_BY_TAG = {tag[len("parquet_") :]: model for tag, model in PASS1_CHECKPOINTS}
TABLES = ("runs", "turns", "scores")
# Read for the Inquirer budget, and hashed into provenance like every other input. Kept out of
# `TABLES` because that tuple is the runs/turns/scores contract the ladder itself is built from.
CALLS_TABLE = "calls"
INQUIRER_ACTOR = "inquirer"  # pinq.types.Actor; the value pinq_train.gate also filters on
TOKEN_COLUMNS = ("actor", "tok_completion", "tok_reasoning")

# The budget bases. `matched_k`, `matched_k_plus_1` and `cap8` are priced in ASKS and are not
# here. Reported side by side, never averaged: they answer different objections.
COST_BASES: tuple[str, ...] = (
    "matched_gen_tokens",
    "matched_question_words",
    "matched_question_tokens",
)


def discover_checkpoints(gate: Path) -> tuple[tuple[str, str], ...]:
    """The `(parquet dir name, model id)` pairs a gate directory holds, read off the disk.

    THE RULE: `parquet_<model>/` beside `<model>.<suite>.json`, the verdict `pi train gate`
    wrote for that checkpoint, so the id every table and figure prints is the id the verdict
    itself carries. A pass-1 short tag (`sft`, `headline`, `headline_stop`) is accepted only
    when no verdict carries the tag verbatim and one carries its pass-1 model id. A parquet
    directory that resolves to neither is a REFUSAL: guessing a model id here would label a
    matched-cost delta with the wrong checkpoint and nothing downstream could tell.

    ORDER: exactly the pass-1 trio keeps the pass-1 order (the committed P13 provenance digest
    hashes the sources in that order); anything else is sorted by model id.
    """
    dirs = sorted(p for p in Path(gate).glob("parquet_*") if p.is_dir())
    if not dirs:
        raise Refused(f"{gate}: no parquet_*/ directory (one per gated checkpoint)")
    found: list[tuple[str, str]] = []
    for d in dirs:
        tag = d.name[len("parquet_") :]
        if any(Path(gate).glob(f"{tag}.*.json")):
            model = tag
        elif tag in _PASS1_BY_TAG and any(Path(gate).glob(f"{_PASS1_BY_TAG[tag]}.*.json")):
            model = _PASS1_BY_TAG[tag]
        else:
            raise Refused(
                f"{d}: no verdict named {tag}.<suite>.json beside it. Name the directory "
                "parquet_<model id> and keep `pi train gate`'s <model id>.<suite>.json verdicts "
                "in the same gate directory."
            )
        found.append((d.name, model))
    if set(found) == set(PASS1_CHECKPOINTS):
        return PASS1_CHECKPOINTS
    return tuple(sorted(found, key=lambda tm: tm[1]))


class Refused(RuntimeError):
    """A refusal, never a warning. Exits non-zero so nothing downstream reads a half table."""


class InstrumentMismatch(Refused):
    """The prefix ladder disagrees with the scorer that produced the verdicts."""


COST_BASIS_NOTES: Mapping[str, str] = {
    "matched_gen_tokens": "calls.parquet actor='inquirer', tok_completion + tok_reasoning "
    "(GENERATED tokens; tok_prompt excluded -- the prompt is the transcript, not the "
    "Inquirer's own writing)",
    "matched_question_words": "turns.parquet question, whitespace-delimited word count. "
    "A word count standing beside the token count as its ROBUSTNESS CHECK and "
    "human-readable gloss. It is the unit the length objection is stated in, and unlike tok_completion it "
    "counts only text the retriever sees (not the rationale, of which the BASE writes "
    "more).",
    "matched_question_tokens": "turns.parquet question, tokenized by the Inquirer pin's "
    "OWN tokenizer (tokenizers.Tokenizer.from_file on a LOCAL tokenizer.json; never "
    "downloaded, never fetched from the cluster). THE PRIMARY length control and the "
    "primary cost axis: the physically meaningful budget is the text the Drafter "
    "receives, tokenized the way the model tokenizes it. `matched_question_words` is the "
    "robustness check on it and the human-readable gloss, not a competing answer. This "
    "primacy was fixed before either number was computed.",
    "measured_degeneracy": "matched_gen_tokens starves the base on both gates measured "
    "so far (asks_base 0.04-0.86, against asks_base ~= asks_trained under matched_k), "
    "because the base's generated tokens per ask are ~1.6-2x the trained arms'. It is "
    "reported as the cost axis it is, and is not the length control.",
}


# --------------------------------------------------------------------------- ladders


@dataclass(frozen=True, slots=True)
class Ladder:
    """One run, scored at every prefix of its own trajectory.

    `cov[k]` is required-evidence coverage after k asks and `cad2_hit[k]` the number of
    depth>=2 needs RESOLVED after k asks. Both are indexed 0..n_asks inclusive, so `cov[0]` is
    the pre-question level and `cov[n_asks]` is what the scorer stored for the whole run.
    """

    run_id: str
    suite_id: str
    task_id: str
    cluster_id: str
    arm_id: str
    seed: int
    n_asks: int
    stop_reason: str
    cov: tuple[float, ...]
    cad2_hit: tuple[int, ...]
    cad2_n: int
    cad_hit: Mapping[int, tuple[int, ...]]  # depth -> hits at each prefix (for verification)
    cad_n: Mapping[int, int]
    # The Inquirer's own spend, per call and per ask. `gen_tok` has n_asks+1 entries (one call
    # per ask plus the terminal decision call), `q_words` has n_asks. Empty when no budget was
    # supplied, and empty REFUSES rather than reading as 0 -- see `gen_tokens_at`.
    gen_tok: tuple[int, ...] = ()
    q_words: tuple[int, ...] = ()
    q_tok: tuple[int, ...] = ()  # the same questions in the pin's OWN tokens, when one is local

    @property
    def key(self) -> str:
        return f"{self.suite_id}/{self.task_id}"

    def asks_at(self, k: int) -> int:
        """Retrieval calls a cap-k run of this policy would have billed.

        `min`, not `k`: a run that stopped at 3 under cap 8 stops at 3 under cap 5 too -- the
        policy is budget-blind, so the cap cannot make it ask more. Charging it k would invent
        spend it never made, in the very comparison whose subject is spend.
        """
        return min(k, self.n_asks)

    def coverage_at(self, k: int) -> float:
        return self.cov[self.asks_at(k)]

    def cad2_at(self, k: int) -> float:
        if not self.cad2_n:
            return float("nan")  # no depth>=2 need on this task: absent, never 0.0
        return self.cad2_hit[self.asks_at(k)] / self.cad2_n

    def calls_at(self, k: int) -> int:
        """Inquirer calls a cap-k run of this policy would have billed: its asks plus its one
        terminal decision call. `min(k, n_asks) + 1`, for `asks_at`'s reason -- a budget-blind
        policy makes the same calls under a smaller cap, and never more of them."""
        return self.asks_at(k) + 1

    def gen_tokens_at(self, k: int) -> float:
        """Cumulative Inquirer generated tokens (completion + reasoning) through prefix k."""
        if not self.gen_tok:
            raise Refused(
                f"{self.run_id}: no Inquirer token curve on this ladder. A missing curve read as "
                "0 tokens would give every base run an unbounded budget and print the cap-8 "
                "delta under a matched-token label. Pass `inquirer_gen_tokens` to build_ladders."
            )
        return float(sum(self.gen_tok[: self.calls_at(k)]))

    def question_words_at(self, k: int) -> float:
        """Cumulative words in the questions a cap-k run would have ASKED. Per ask, not per
        call: the terminal decision call emits no question, so it spends no question words."""
        if not self.q_words and self.n_asks:
            raise Refused(
                f"{self.run_id}: no Inquirer question-word curve on this ladder; see "
                "`gen_tokens_at` for why an absent curve is not a zero."
            )
        return float(sum(self.q_words[: self.asks_at(k)]))

    def question_tokens_at(self, k: int) -> float:
        """`question_words_at` in the Inquirer pin's own tokens. The real unit the words proxy
        stands in for; priced by the identical rule, so any disagreement is about the UNIT."""
        if not self.q_tok and self.n_asks:
            raise Refused(
                f"{self.run_id}: no Inquirer question-token curve on this ladder; see "
                "`gen_tokens_at` for why an absent curve is not a zero."
            )
        return float(sum(self.q_tok[: self.asks_at(k)]))

    def cost_at(self, k: int, basis: str) -> float:
        if basis == "matched_gen_tokens":
            return self.gen_tokens_at(k)
        if basis == "matched_question_words":
            return self.question_words_at(k)
        if basis == "matched_question_tokens":
            return self.question_tokens_at(k)
        raise Refused(f"{basis!r} is not an Inquirer cost basis; known: {COST_BASES}")

    def k_within(self, budget: float, basis: str) -> int:
        """The largest prefix this run could have reached on `budget`, in 0..n_asks.

        Clamped at `n_asks` on purpose: a budget that outruns the trajectory buys the WHOLE
        trajectory and nothing beyond it -- the policy has no further questions to ask, and
        extrapolating the ladder past its last rung would invent coverage. A budget too small
        for even the first call buys k=0, the pre-question level: an honest rung, not a dropped
        pair and not rung 1 at a discount.
        """
        best = 0
        for k in range(self.n_asks + 1):
            if self.cost_at(k, basis) <= budget + 1e-9:
                best = k
            else:
                break
        return best

    def done_at(self, k: int) -> bool:
        """`pi_run.cmd_train._done`'s threshold, restated: coverage complete, NaN is not done."""
        c = self.coverage_at(k)
        return (not math.isnan(c)) and c >= 1.0 - 1e-12


def build_ladders(
    runs: Sequence[Mapping[str, Any]],
    turns: Mapping[str, Sequence[_Turn]],
    graphs: Mapping[str, Mapping[str, GoldGraph]],
    *,
    matcher: MechanicalMatcher | None = None,
    inquirer_gen_tokens: Mapping[str, Sequence[int]] | None = None,
    question_words: Mapping[str, Sequence[int]] | None = None,
    question_tokens: Mapping[str, Sequence[int]] | None = None,
) -> dict[str, Ladder]:
    """One `Ladder` per run, by running the REAL matcher on every prefix of the trajectory.

    The answer is deliberately not loaded: `MechanicalMatcher` only consults citations to
    promote RESOLVE to USE, and every rung at or above RESOLVE counts the same here. Confirmed
    on the gate parquets, where no match record carries `use` at all.
    """
    mt = matcher or MechanicalMatcher()
    out: dict[str, Ladder] = {}
    for r in runs:
        run_id, suite, task = str(r["run_id"]), str(r["suite_id"]), str(r["task_id"])
        graph = graphs.get(suite, {}).get(task)
        if graph is None:
            raise Refused(f"{run_id}: no gold graph for {suite}/{task} at {GRAPH_VERSION}")
        ts = list(turns.get(run_id, ()))
        if len(ts) != int(r["n_asks"]):
            raise Refused(f"{run_id}: {len(ts)} turn rows against n_asks={r['n_asks']}")
        gen_tok, q_words, q_tok = (), (), ()
        if inquirer_gen_tokens is not None:
            calls = inquirer_gen_tokens.get(run_id)
            if calls is None:
                raise Refused(f"{run_id}: no {INQUIRER_ACTOR} call rows for a run being charged")
            # THE MEASURED INVARIANT, enforced: one call per ask plus one terminal decision call,
            # true on all 1,395 runs of every gate parquet dir. Break it and call i is no longer
            # ask i, so the budget is charged to the wrong rung with no symptom.
            if len(calls) != len(ts) + 1:
                raise Refused(
                    f"{run_id}: {len(calls)} {INQUIRER_ACTOR} calls against n_asks={len(ts)}. The "
                    "per-ask attribution the token curve is built on requires exactly n_asks+1 "
                    "(one call per ask plus the terminal decision call)."
                )
            gen_tok = tuple(int(x) for x in calls)
        if question_words is not None:
            # `.get(run_id, ())` and not `[run_id]`: a run that asked NOTHING has no turn rows,
            # so `turn_shims` never makes it a key, and an absent entry there is the correct
            # empty curve rather than a missing measurement. The length check below is what
            # still catches a genuinely absent one, for every run that did ask.
            words = question_words.get(run_id, ())
            if len(words) != len(ts):
                raise Refused(
                    f"{run_id}: {len(words)} question-word counts against n_asks={len(ts)}"
                )
            q_words = tuple(int(x) for x in words)
        if question_tokens is not None:
            toks = question_tokens.get(run_id, ())  # `.get`: a 0-ask run has no turn rows
            if len(toks) != len(ts):
                raise Refused(
                    f"{run_id}: {len(toks)} question-token counts against n_asks={len(ts)}"
                )
            q_tok = tuple(int(x) for x in toks)
        gold_uids = set(_gold_uids(graph))
        depths = {
            n.gold_node_id: n.gold_depth for n in graph.gold_nodes if n.gold_depth is not None
        }
        cad_n: dict[int, int] = {}
        for d in depths.values():
            cad_n[d] = cad_n.get(d, 0) + 1
        cov: list[float] = []
        cad_hit: dict[int, list[int]] = {d: [] for d in cad_n}
        seen: set[str] = set()
        for k in range(len(ts) + 1):
            if k:
                seen |= set(ts[k - 1].retrieved_uids)
            cov.append(evidence_coverage(set(seen), gold_uids))
            traj = _Traj(turns=tuple(ts[:k]), outcome=_Outcome(answer=None))
            resolved = {
                m.node_id
                for m in mt.match(graph=graph, trajectory=traj, run_id=run_id)
                if m.rank >= RESOLVE_RANK
            }
            for d in cad_n:
                cad_hit[d].append(sum(1 for i, dd in depths.items() if dd == d and i in resolved))
        cad2_n = sum(n for d, n in cad_n.items() if d >= 2)
        cad2_hit = [sum(cad_hit[d][k] for d in cad_n if d >= 2) for k in range(len(ts) + 1)]
        out[run_id] = Ladder(
            run_id=run_id,
            suite_id=suite,
            task_id=task,
            cluster_id=str(r.get("template_id") or task),
            arm_id=str(r["arm_id"]),
            seed=int(r["seed"]),
            n_asks=len(ts),
            stop_reason=str(r.get("stop_reason") or ""),
            cov=tuple(cov),
            cad2_hit=tuple(cad2_hit),
            cad2_n=cad2_n,
            cad_hit={d: tuple(v) for d, v in cad_hit.items()},
            cad_n=dict(cad_n),
            gen_tok=gen_tok,
            q_words=q_words,
            q_tok=q_tok,
        )
    return out


def verify_instrument(
    ladders: Mapping[str, Ladder], stored: Mapping[str, Mapping[str, float]]
) -> dict[str, int]:
    """Refuse unless the ladder's LAST rung is what the verdicts' scorer wrote.

    Three independent comparisons, because each can fail on its own: `frontier_q#k` pins the
    whole prefix curve, `evidence_coverage` pins the terminal Q the gate's delta is built from,
    and `cad#d` / `cad_n#d` pin the matcher's RESOLVE decision per depth. Rule 2: this check is
    the reason a "matched-cost delta" here is the same measurement as the one it re-reads.
    """
    checked = {"frontier_q": 0, "evidence_coverage": 0, "cad": 0}
    bad: list[str] = []
    for run_id, lad in sorted(ladders.items()):
        rows = stored.get(run_id, {})
        for k in range(lad.n_asks + 1):
            v = rows.get(f"frontier_q#{k}")
            if v is None:
                continue  # score() omits the point when Q is undefined; it is not a zero
            checked["frontier_q"] += 1
            if not _close(v, lad.cov[k]):
                bad.append(f"{run_id} frontier_q#{k}: stored {v!r} ladder {lad.cov[k]!r}")
        cov = rows.get("evidence_coverage")
        if cov is not None:
            checked["evidence_coverage"] += 1
            if not _close(cov, lad.cov[-1]):
                bad.append(f"{run_id} evidence_coverage: stored {cov!r} ladder {lad.cov[-1]!r}")
        elif not math.isnan(lad.cov[-1]):
            bad.append(f"{run_id} evidence_coverage: no stored row, ladder {lad.cov[-1]!r}")
        for d, n_d in sorted(lad.cad_n.items()):
            checked["cad"] += 1
            want_c, want_n = rows.get(f"cad#{d}"), rows.get(f"cad_n#{d}")
            got = lad.cad_hit[d][-1] / n_d
            if want_c is None or want_n is None:
                bad.append(f"{run_id} cad#{d}: no stored row")
            elif not _close(want_c, got) or not _close(want_n, float(n_d)):
                bad.append(
                    f"{run_id} cad#{d}: stored ({want_c!r},{want_n!r}) ladder ({got!r},{n_d})"
                )
    if bad:
        raise InstrumentMismatch(
            f"the prefix ladder disagrees with the stored scorer on {len(bad)} value(s); "
            "no matched-cost number is reportable until they agree. First five:\n  "
            + "\n  ".join(bad[:5])
        )
    return checked


def _close(a: float, b: float, tol: float = 1e-12) -> bool:
    if math.isnan(a) and math.isnan(b):
        return True
    return abs(float(a) - float(b)) <= tol


# --------------------------------------------------------------------------- contrasts


@dataclass(frozen=True, slots=True)
class Contrast:
    suite_id: str
    checkpoint: str
    metric: str
    comparator: str  # matched_k | matched_k_plus_1 | cap8
    n_tasks: int
    n_clusters: int
    trained_mean: float
    base_mean: float
    delta: float
    ci_lo: float
    ci_hi: float
    p_value: float | None
    trained_asks: float
    base_asks: float
    # PAIRS, not tasks: one (suite, task, seed) pair per count, while `n_tasks` above counts
    # tasks. The two denominators differ by the number of seeds, and a reader comparing them
    # without that in hand is reading a mix. See `contrast`.
    n_base_short: int  # (suite, task, seed) pairs where the base ran out of turns before k
    # The budgets that were EQUATED, so a reader can see what was charged rather than trust the
    # word "matched". NaN for the ask-priced comparators, which equate no budget.
    trained_cost: float = float("nan")
    base_cost: float = float("nan")
    # A digest of the exact (suite/task) keys that entered the pairing, so a re-derivation that
    # silently dropped or added one cannot agree on n AND on this.
    pair_keys_hash: str = ""

    def row(self) -> list[str]:
        return [
            self.suite_id,
            self.checkpoint,
            self.metric,
            self.comparator,
            str(self.n_tasks),
            f"{self.trained_asks:.3f}",
            f"{self.base_asks:.3f}",
            _num(self.trained_cost),
            _num(self.base_cost),
            f"{self.trained_mean:.4f}",
            f"{self.base_mean:.4f}",
            f"{self.delta:+.4f}",
            f"[{self.ci_lo:+.4f}, {self.ci_hi:+.4f}]",
            "n/a" if self.p_value is None else f"{self.p_value:.4f}",
            str(self.n_base_short),
        ]


CONTRAST_COLUMNS = (
    "suite",
    "checkpoint",
    "metric",
    "comparator",
    "n",
    "asks_trained",
    "asks_base",
    "cost_trained",
    "cost_base",
    "trained",
    "base",
    "delta",
    "ci95",
    "p",
    "base_short",
)


def _value(lad: Ladder, metric: str, k: int) -> float:
    return lad.coverage_at(k) if metric == "evidence_coverage" else lad.cad2_at(k)


def _num(x: float) -> str:
    return "--" if math.isnan(x) else f"{x:.1f}"


def contrast(
    trained: Sequence[Ladder],
    base: Mapping[tuple[str, str, int], Ladder],
    *,
    suite_id: str,
    checkpoint: str,
    metric: str,
    comparator: str,
    offset: int,
) -> Contrast:
    """One paired contrast. `offset` is added to the trained run's ask count to pick the base
    prefix; `comparator == 'cap8'` pins the base at the gate's cap instead.

    THE PAIR IS `(suite, task, seed)` AND THE UNIT IS THE TASK. A trained run is charged against
    the SAME seed's baseline run, and the seeds of one task are then averaged into that task,
    which is the key `paired_difference` resamples.

    WHY TWO HOPS AND NOT ONE, stated because the two available conventions give different
    numbers on an unbalanced population and the choice is not free. Two seeds of one task are
    two measurements of one task, not two tasks: entering them as separate units would weight a
    task by how many of its seeds happened to complete -- 350 baseline runs over 176 MuSiQue
    tasks on the held-out store, so some tasks carry one seed and some two -- and make the
    estimand depend on rollout bookkeeping, which is a machine artifact. It is also
    `pinq_train.gate`'s own rule (`_paired_by_task_map`, and `_matched_cost`: "seeds averaged
    within a task before the task enters the bootstrap"), which is what lets `reconcile` compare
    this n against the verdict's n at all: under seed-averaging both are task counts.

    THE BELIEF THAT CHANGED, and what it cost (CONTRIBUTING.md rule 4). Until 2026-09-18 this
    docstring read: "Pairing is on (suite, task, seed) -- the gate's own key. The trained arms
    carry seed 0 only, so this is seed 0 against seed 0 and no seed is averaged into the other."
    Only the FIRST half was ever true of the code: the comparator was looked up by
    `(suite, task, seed)`, but the pair was then stored under `Ladder.key` -- `suite/task` --
    so on a two-seed population each task's second pair silently overwrote its first and the
    aggregate was one arbitrarily chosen seed (whichever the run-id ordering put last) at the
    full task count. The claim about the data is now false: the held-out QA store
    (`artifacts/testsplit_qa/scores_parquet`) runs `inquirer_trained` at seeds 0 AND 1 on all
    three suites, as does `inquirer_prompted`. Against its published verdicts the collapse
    missed `evidence_coverage` by +0.00166 / +0.00786 / -0.00226 (MuSiQue / StrategyQA / wiki2)
    and `cad_ge2` by -0.00217 / +0.02679, at IDENTICAL n -- which is why n could not catch it and
    `reconcile` had to. It refused, correctly, and no matched-cost number was reportable on the
    test split until this was fixed.
    """
    pairs: dict[tuple[str, str, int], list[Mapping[str, float]]] = {}
    n_short = 0
    budget = comparator in COST_BASES
    for lad in trained:
        peer = base.get((lad.suite_id, lad.task_id, lad.seed))
        if peer is None:
            continue
        if budget:
            # The trained arm's OWN spend is the budget, and the base is charged the largest
            # prefix it fits in. Both sides are priced by the same accessor, so neither is
            # measured with a rule the other was not.
            spend = lad.cost_at(lad.n_asks, comparator)
            k = peer.k_within(spend, comparator)
        else:
            spend = float("nan")
            k = CAP if comparator == "cap8" else lad.n_asks + offset
        va, vb = _value(lad, metric, lad.n_asks), _value(peer, metric, k)
        if math.isnan(va) or math.isnan(vb):
            continue
        pairs.setdefault((lad.suite_id, lad.task_id, lad.seed), []).append(
            {
                "va": va,
                "vb": vb,
                "trained_asks": float(lad.n_asks),
                "base_asks": float(peer.asks_at(k)),
                "trained_cost": spend,
                # NaN for the ask-priced comparators, which equate no budget; `_fmean` drops it,
                # so an unpriced comparator reports `--` rather than a cost of 0.
                "base_cost": peer.cost_at(k, comparator) if budget else float("nan"),
            }
        )
        if budget:
            # The budget outran the whole trajectory: the base was charged all of it and no more.
            n_short += int(spend > peer.cost_at(peer.n_asks, comparator) + 1e-9)
        else:
            n_short += int(peer.n_asks < k)
    by_task = _seed_averaged(pairs)
    a = {key: rec["va"] for key, rec in by_task.items()}
    b = {key: rec["vb"] for key, rec in by_task.items()}
    est = paired_difference(a, b, clusters=None, n_boot=N_BOOT, n_perm=N_PERM, seed=SEED)
    return Contrast(
        suite_id=suite_id,
        checkpoint=checkpoint,
        metric=metric,
        comparator=comparator,
        n_tasks=len(a),
        n_clusters=len(a),  # task-clustered: one task per unit, whatever its seed count
        trained_mean=_fmean(a.values()),
        base_mean=_fmean(b.values()),
        delta=est.point,
        ci_lo=est.ci_lo,
        ci_hi=est.ci_hi,
        p_value=est.p_value,
        # Task-weighted like the metric means above, so one row does not mix a per-task delta
        # with a per-pair cost.
        trained_asks=_fmean(rec["trained_asks"] for rec in by_task.values()),
        base_asks=_fmean(rec["base_asks"] for rec in by_task.values()),
        n_base_short=n_short,
        trained_cost=_fmean(rec["trained_cost"] for rec in by_task.values()),
        base_cost=_fmean(rec["base_cost"] for rec in by_task.values()),
        pair_keys_hash=h("pairs", canon(sorted(a))),
    )


PAIR_FIELDS: tuple[str, ...] = (
    "va",
    "vb",
    "trained_asks",
    "base_asks",
    "trained_cost",
    "base_cost",
)


def _seed_averaged(
    pairs: Mapping[tuple[str, str, int], Sequence[Mapping[str, float]]],
) -> dict[str, dict[str, float]]:
    """`(suite, task, seed)` pairs -> one record per TASK, keyed `suite/task`, in two hops.

    Hop one averages the pairs that share a `(suite, task, seed)`: there are none on any
    population measured so far (no gate parquet or test-split store holds two runs of one arm at
    one task and seed), and averaging them is what `pinq_train.gate` does with the several run
    ids one of its keys can hold. Hop two averages the seeds into the task.

    THE KEY IS `Ladder.key` UNCHANGED. `pair_keys_hash` digests these strings and is recorded in
    `figures/P13_matched_cost/*.json` and `artifacts/n1/matched_cost.*.json`; putting the seed
    into the string would move every published provenance digest without moving any number.
    """
    out: dict[str, list[Mapping[str, float]]] = {}
    for (suite, task, _seed), vals in sorted(pairs.items()):
        out.setdefault(f"{suite}/{task}", []).append(
            {f: _fmean(v[f] for v in vals) for f in PAIR_FIELDS}
        )
    return {
        key: {f: _fmean(rec[f] for rec in recs) for f in PAIR_FIELDS}
        for key, recs in sorted(out.items())
    }


def _fmean(xs: Iterable[float]) -> float:
    vals = [float(x) for x in xs if not math.isnan(float(x))]
    return statistics.fmean(vals) if vals else float("nan")


def level(values: Sequence[float]) -> tuple[float, float, float]:
    """Mean and a task-clustered BCa interval for one arm's level (not a difference)."""
    units = [[v] for v in values if not math.isnan(v)]
    if not units:
        return (float("nan"), float("nan"), float("nan"))
    return cluster_bootstrap(units, n_boot=N_BOOT, seed=SEED)


def outcome_split(
    trained: Sequence[Ladder],
    base: Mapping[tuple[str, str, int], Ladder],
    metric: str,
    *,
    offset: int = 0,
) -> dict[str, int]:
    """Win / tie / loss of the trained run against the base prefix at the SAME cost."""
    out = {"n": 0, "win": 0, "tie": 0, "loss": 0}
    for lad in trained:
        peer = base.get((lad.suite_id, lad.task_id, lad.seed))
        if peer is None:
            continue
        va = _value(lad, metric, lad.n_asks)
        vb = _value(peer, metric, lad.n_asks + offset)
        if math.isnan(va) or math.isnan(vb):
            continue
        out["n"] += 1
        out["tie" if _close(va, vb) else ("win" if va > vb else "loss")] += 1
    return out


def early_stop_split(
    trained: Sequence[Ladder], base: Mapping[tuple[str, str, int], Ladder]
) -> dict[str, int]:
    """Why a cap-8 loss happened: a bad question, or a question the base paid extra for?

    `done_at_stop` is `pi_run.cmd_train._done` applied to the state the STOP was taken in --
    required-evidence coverage complete. Among cap-8 LOSSES it is 0 by arithmetic (a run below
    a coverage that is at most 1 cannot itself be at 1); it is reported anyway, with its n,
    because the question "did it stop because it was done?" is answered by the OTHER two rows:
    a loss that is a tie-or-win at matched k was bought with the base's extra asks, and one
    that is still behind at matched k is a worse question per ask.
    """
    out = {
        "n_paired": 0,
        "n_done_at_stop": 0,
        "n_lose_at_cap8": 0,
        "lose_done_at_stop": 0,
        "lose_tie_or_win_at_k": 0,
        "lose_behind_at_k": 0,
        "n_win_or_tie_at_cap8": 0,
    }
    for lad in trained:
        peer = base.get((lad.suite_id, lad.task_id, lad.seed))
        if peer is None:
            continue
        va, v8 = lad.coverage_at(lad.n_asks), peer.coverage_at(CAP)
        if math.isnan(va) or math.isnan(v8):
            continue
        out["n_paired"] += 1
        out["n_done_at_stop"] += int(lad.done_at(lad.n_asks))
        if va < v8 - 1e-12:
            out["n_lose_at_cap8"] += 1
            out["lose_done_at_stop"] += int(lad.done_at(lad.n_asks))
            vk = peer.coverage_at(lad.n_asks)
            out["lose_tie_or_win_at_k"] += int(va >= vk - 1e-12)
            out["lose_behind_at_k"] += int(va < vk - 1e-12)
        else:
            out["n_win_or_tie_at_cap8"] += 1
    return out


MATCHED_COST = "matched_cost"

# WHICH CRITERION FIELD HOLDS WHICH NUMBER, and why it is not simply `value`. `pi train gate`
# has defaulted to `--coverage-rule matched_cost` since 4e3055b. Under that rule it REPLACES
# `criteria.evidence_coverage.value` with the matched-k delta and moves the cap-8 one it
# replaced to `cap8_value` (`pinq_train.gate`, the D9 block). No other criterion is moved:
# `cad_ge2` stays the cap-8 contrast under both rules -- T19b (ff73a4b) only makes it
# report-only and copies the same number into `cap8_cad_ge2_delta`, which is the field to
# prefer because it says what the number is denominated in.
MOVED_BY_MATCHED_COST: tuple[str, ...] = ("evidence_coverage",)
CAP8_FIELD: Mapping[str, str] = {
    "evidence_coverage": "cap8_value",
    "cad_ge2": "cap8_cad_ge2_delta",
}


def _cap8_n(
    verdict: Mapping[str, Any], crit: Mapping[str, Any], *, metric: str, suite_id: str
) -> int:
    """The n that belongs to the cap-8 delta of a `matched_cost` verdict.

    `criteria.<metric>.n` counts the MATCHED-k pairs there; the cap-8 pair count is
    `matched_cost.by_suite[<suite>].cap8_n_tasks`. They are equal on every gate measured so far
    and are not equal by construction -- a task whose baseline ladder omits the rung at k drops
    out of one population and not the other -- so read the one that goes with the value.
    """
    blk = ((verdict.get(MATCHED_COST) or {}).get("by_suite") or {}).get(suite_id) or {}
    n = blk.get("cap8_n_tasks")
    return int(crit["n"]) if n is None else int(n)


def _locked_value(x: Any) -> float:
    """A verdict field read as a number, with NULL meaning "this verdict measured nothing here".

    NaN, never 0.0. `_close` treats NaN as equal only to NaN, so a null cell locks the
    recomputation on ALSO being empty: a recomputed delta of any real number against a null one
    is a disagreement about the population and is refused. Reading null as 0.0 would instead
    make a real recomputation look like a difference of that size from nothing.

    A verdict of exactly this shape exists: a suite whose gold graph has no node at depth two or
    more carries `criteria.cad_ge2.cap8_cad_ge2_delta: null` with `n: 0` (wiki2, in
    artifacts/testsplit_qa/verdicts/stacked-notdone.wiki2.json -- 0 of its 12,576 gold tasks
    carry a depth>=2 node, so the criterion is ABSENT there and not zero). Before this,
    `float(None)` raised TypeError out of `_locked_field` and the whole analysis died with a
    traceback after printing its tables -- in a script whose stated contract is that a missing
    input is a refusal and never an empty axis.
    """
    return float("nan") if x is None else float(x)


def _locked_field(
    verdict: Mapping[str, Any], name: str, *, metric: str, comparator: str, suite_id: str
) -> tuple[str, float, int] | None:
    """`(field, value, n)` this contrast is locked against, or None when the verdict locks none.

    THE FAILURE THIS EXISTS TO STOP is reading one measurement as another. Against a
    `matched_cost` verdict the cap-8 comparator used to be compared with `value`, which is the
    matched-k delta there, and a correct recomputation was refused for it: measured on
    artifacts/gate/17b, `REFUSED: ... verdict 0.1092 n=132 vs recomputed 0.0537 n=132`, where
    0.0537 is that very verdict's `cap8_value`.
    """
    crit = (verdict.get("criteria") or {}).get(metric)
    if crit is None:
        raise Refused(f"{name}: no `{metric}` criterion")
    rule = str(verdict.get("coverage_rule") or "cap8")
    moved = rule == MATCHED_COST and metric in MOVED_BY_MATCHED_COST
    field = CAP8_FIELD.get(metric, "")
    if comparator == "matched_k":
        # The lock is two-sided only where the verdict carries a matched-k number of its own.
        return ("value", _locked_value(crit["value"]), int(crit["n"])) if moved else None
    if comparator != "cap8":
        return None
    if moved:
        if field not in crit:
            raise Refused(
                f"{name}: `coverage_rule` is {MATCHED_COST}, so `criteria.{metric}.value` is the "
                f"matched-k delta and the cap-8 delta belongs in `{field}` -- which this verdict "
                f"does not carry. Re-run `pi train gate` on these parquets, or reconcile against "
                f"a verdict written with --coverage-rule cap8. Reading `value` as the cap-8 "
                f"number here would compare two different measurements and blame the parquets."
            )
        return (
            field,
            _locked_value(crit[field]),
            _cap8_n(verdict, crit, metric=metric, suite_id=suite_id),
        )
    if rule == MATCHED_COST:
        if field and field in crit:
            return (field, _locked_value(crit[field]), int(crit["n"]))
        if str(crit.get("comparator") or "cap8") != "cap8":
            raise Refused(
                f"{name}: `criteria.{metric}.comparator` is {crit.get('comparator')!r}, so "
                f"`value` is not the cap-8 delta, and `{field or 'a cap-8 field'}` is not there "
                f"to hold it. A cap-8 row reconciled against it would be a category error."
            )
    return ("value", _locked_value(crit["value"]), int(crit["n"]))


def reconcile(gate: Path, contrasts: Sequence[Contrast]) -> list[dict[str, Any]]:
    """Tie the recomputed rows to the published verdicts, value for value.

    The `cap8` comparator is the pre-`matched_cost` gate's own contrast recomputed from the
    parquets by different code, so it must land on the number
    `artifacts/gate/<model>.<suite>.json` already carries -- point estimate and n both. If it
    does not, either this script's pairing differs from `pinq_train.gate`'s or the parquets are
    not the ones the verdict came from, and in either case NONE of the matched-cost rows above
    it are trustworthy. Hence a refusal, not a note.

    AGAINST A `matched_cost` VERDICT THE LOCK IS TWO-SIDED, because the verdict now carries
    both numbers. The cap-8 row is checked against `cap8_value` and the `matched_k` row against
    `value`, and each row records the field it read. One-sided would not do: a cap-8 delta that
    lands on `cap8_value` proves the PAIRING agrees and says nothing about the baseline prefix
    rung the matched-k delta is a difference from -- which is the number this table reports.

    Only the point and n are compared. The intervals legitimately differ: the verdicts use a
    plain percentile bootstrap over per-task deltas, this script the BCa cluster bootstrap in
    `pi_eval.stats.inference`, and an interval is a property of its estimator.
    """
    rows: list[dict[str, Any]] = []
    bad: list[str] = []
    seen: dict[Path, Mapping[str, Any]] = {}
    for c in contrasts:
        path = gate / f"{c.checkpoint}.{c.suite_id}.json"
        if path not in seen:
            if not path.exists():
                raise Refused(f"no verdict at {path} to reconcile the cap-8 contrast against")
            seen[path] = json.loads(path.read_text())
        locked = _locked_field(
            seen[path], path.name, metric=c.metric, comparator=c.comparator, suite_id=c.suite_id
        )
        if locked is None:
            continue
        field, want, want_n = locked
        agree = _close(want, c.delta, 1e-9) and want_n == c.n_tasks
        rows.append(
            {
                "verdict": path.name,
                "metric": c.metric,
                "comparator": c.comparator,
                "field": field,
                "verdict_value": want,
                "recomputed": c.delta,
                "verdict_n": want_n,
                "recomputed_n": c.n_tasks,
                "agree": agree,
            }
        )
        if not agree:
            bad.append(
                f"{path.name}/{c.metric} [{c.comparator} vs criteria.{c.metric}.{field}]: "
                f"verdict {want!r} n={want_n} vs recomputed {c.delta!r} n={c.n_tasks}"
            )
    if bad:
        raise Refused(
            "a recomputed contrast does not reproduce the published verdict it re-reads, so no "
            "matched-cost row derived from the same pairing is reportable:\n  " + "\n  ".join(bad)
        )
    return rows


# --------------------------------------------------------------------------- inputs


# `matched_question_tokens`'s tokenizer, resolved OFFLINE or not at all. `transformers` is not
# installed in this venv; `tokenizers` is, and `Tokenizer.from_file` needs only a tokenizer.json,
# so the real token count costs one local file read and no network. The default location is the
# HF hub cache entry for the Inquirer pin's family -- `calls.model` is `qwen3-8b-base` for the
# base arm and `qwen3-8b-dpo-*` / `qwen3-8b-sft-*` for the trained ones, all Qwen3-8B derivatives
# sharing one tokenizer. Never downloaded: absent means the comparator is OMITTED and said to be,
# never silently priced at zero.
TOKENIZER_ENV = "PI_QUESTION_TOKENIZER"  # path to a tokenizer.json, overriding the search
TOKENIZER_GLOBS: tuple[str, ...] = (
    "~/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/*/tokenizer.json",
)


def find_question_tokenizer(env: Mapping[str, str] | None = None) -> Path | None:
    """A local `tokenizer.json` for the Inquirer pin, or None. Reads the disk; never the network."""
    override = (env if env is not None else os.environ).get(TOKENIZER_ENV, "").strip()
    if override:
        p = Path(override).expanduser()
        if not p.is_file():
            raise Refused(f"{TOKENIZER_ENV}={override!r} is not a file")
        return p
    for pat in TOKENIZER_GLOBS:
        hits = sorted(glob.glob(os.path.expanduser(pat)))
        if hits:
            return Path(hits[0])
    return None


def question_tokens(turns: Mapping[str, Sequence[_Turn]], tokenizer: Path) -> dict[str, list[int]]:
    """Tokens per ASKED question, by the Inquirer pin's own tokenizer. The real unit, not a proxy.

    `add_special_tokens=False`: the budget is the question's own tokens, and a chat template's
    role markers are a constant per call that neither arm chose to write.
    """
    from tokenizers import Tokenizer  # local, so a missing wheel is not an import-time failure

    tok = Tokenizer.from_file(str(tokenizer))
    out: dict[str, list[int]] = {}
    for run_id, ts in turns.items():
        texts = [str(t.action.text or "") for t in ts]
        enc = tok.encode_batch(texts, add_special_tokens=False) if texts else []
        out[run_id] = [len(e.ids) for e in enc]
    return out


def active_cost_bases(tokenizer: Path | None) -> tuple[str, ...]:
    """The budget bases this run can price. `matched_question_tokens` needs a local tokenizer;
    without one it is dropped BY NAME in the report rather than reported against 0 tokens."""
    return (
        COST_BASES
        if tokenizer is not None
        else tuple(b for b in COST_BASES if b != "matched_question_tokens")
    )


def read_inquirer_calls(parquet_dir: Path) -> dict[str, list[int]]:
    """`{run_id: [generated tokens per Inquirer call, in file order]}` from `calls.parquet`.

    WHERE THE COUNTS LIVE, established by reading. `calls.parquet` is the fine grain: one row per
    LLM call, carrying `actor` (`pinq.types.Actor`: inquirer | drafter | answerer | ...) and
    `tok_prompt` / `tok_completion` / `tok_reasoning` / `tok_cached`. `runs.parquet` carries only
    the whole-run rollup with no actor, and `turns.parquet`'s `usage_tok_*` is the WHOLE turn --
    verified on run 00081fed: `usage_tok_total` at turn 0 is 3119, which is the Inquirer's 690
    plus both Drafter calls, so reading it as an Inquirer budget would charge the base for the
    Drafter's spend. `ledger.parquet` has the `turn_idx` but no actor. So: this table, this
    actor.

    GENERATED, not billed: `tok_completion + tok_reasoning`, excluding `tok_prompt`. The prompt is
    the transcript the environment handed back, not anything the Inquirer chose to write, and a
    prompt-inclusive budget is dominated by how many turns have accumulated rather than by
    verbosity -- which is the quantity under test. `tok_completion` EXCLUDES `tok_reasoning` in
    this schema (verified on the Drafter, whose 924+124+71 sums to its `tok_total` of 1119), so
    the two are added rather than one taken as the other.

    A missing column, or a NULL in one, is a REFUSAL. Summing an absent column to 0 would hand
    every base run an unbounded budget and reprint the cap-8 delta as a matched-token result.
    """
    path = Path(parquet_dir) / f"{CALLS_TABLE}.parquet"
    if not path.exists():
        raise Refused(
            f"missing {path}: per-call Inquirer token counts are not recoverable without it, and "
            "no proxy is substituted for them here"
        )
    con = duckdb.connect()
    cols = {
        d[0] for d in con.execute(f"SELECT * FROM read_parquet('{path}') LIMIT 0").description or ()
    }
    absent = [c for c in TOKEN_COLUMNS if c not in cols]
    if absent:
        raise Refused(
            f"{path} carries no {', '.join(absent)} column (has: {', '.join(sorted(cols))}). The "
            "Inquirer's per-call token counts are what the matched-token comparator charges the "
            "base; an absent column summed to 0 is an unbounded budget wearing a number."
        )
    rows = _rows(
        con,
        "SELECT run_id, tok_completion, tok_reasoning FROM (SELECT *, row_number() OVER () AS _rn "
        f"FROM read_parquet('{path}')) WHERE actor = '{INQUIRER_ACTOR}' ORDER BY _rn",
    )
    if not rows:
        raise Refused(f"{path}: no row with actor = {INQUIRER_ACTOR!r}")
    out: dict[str, list[int]] = {}
    for r in rows:
        c, g = r["tok_completion"], r["tok_reasoning"]
        if c is None or g is None:
            raise Refused(
                f"{path}: a {INQUIRER_ACTOR} call of run {r['run_id']} has NULL "
                f"tok_completion/tok_reasoning ({c!r}/{g!r}). NULL is not 0 tokens."
            )
        out.setdefault(str(r["run_id"]), []).append(int(c) + int(g))
    return out


def question_words(turns: Mapping[str, Sequence[_Turn]]) -> dict[str, list[int]]:
    """Words per ASKED question, from `turns.question` -- a labelled PROXY for its token count.

    LOUDLY A PROXY. The question's real tokens would need the Inquirer pin's tokenizer, and no
    tokenizer is installed in this venv, so this counts whitespace-delimited words instead. Two
    things make it the right proxy rather than a convenient one: it is the exact unit the
    objection is stated in (18.07 words against 12.99), and it counts only text that reaches the
    retriever -- `tok_completion` also pays for the `rationale`, which the retriever never sees
    and which the BASE writes more of.
    """
    return {
        run_id: [len(str(t.action.text or "").split()) for t in ts] for run_id, ts in turns.items()
    }


def read_tables(parquet_dir: Path) -> dict[str, list[dict[str, Any]]]:
    for t in TABLES:
        if not (parquet_dir / f"{t}.parquet").exists():
            raise Refused(f"missing {parquet_dir / t}.parquet")
    con = duckdb.connect()
    for t in TABLES:
        con.execute(f"CREATE VIEW {t} AS SELECT * FROM read_parquet('{parquet_dir}/{t}.parquet')")
    runs = _rows(
        con,
        "SELECT run_id, suite_id, task_id, template_id, arm_id, seed, n_asks, n_turns, "
        "stop_reason, retrieval_calls, split, status, code_version "
        "FROM runs WHERE status = 'ok' ORDER BY run_id",
    )
    turns = _rows(
        con,
        "SELECT run_id, turn_idx, question, retrieved_uids FROM turns ORDER BY run_id, turn_idx",
    )
    scores = _rows(
        con,
        "SELECT run_id, metric_name, value, scorer_hash, graph_version FROM scores "
        "WHERE metric_name = 'evidence_coverage' OR metric_name LIKE 'frontier_q#%' "
        "OR metric_name LIKE 'cad#%' OR metric_name LIKE 'cad_n#%'",
    )
    return {"runs": runs, "turns": turns, "scores": scores}


def _rows(con, query: str) -> list[dict[str, Any]]:
    cur = con.execute(query)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def turn_shims(turns: Sequence[Mapping[str, Any]]) -> dict[str, list[_Turn]]:
    """The same parquet shim `pi_eval.score` hands the matcher, so the matcher cannot tell
    which caller built it."""
    out: dict[str, list[_Turn]] = {}
    for t in turns:
        out.setdefault(str(t["run_id"]), []).append(
            _Turn(
                turn_idx=int(t["turn_idx"]),
                retrieved_uids=tuple(t.get("retrieved_uids") or ()),
                new_uids=(),
                action=_Act(text=str(t.get("question") or "")),
            )
        )
    for v in out.values():
        v.sort(key=lambda t: t.turn_idx)
    return out


def stored_scores(scores: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for s in scores:
        out.setdefault(str(s["run_id"]), {})[str(s["metric_name"])] = float(s["value"])
    return out


def sha256_file(p: Path) -> str:
    with open(p, "rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


# --------------------------------------------------------------------------- report


def text_table(title: str, columns: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    widths = [
        max(len(str(c)), *(len(str(r[i])) for r in rows)) if rows else len(str(c))
        for i, c in enumerate(columns)
    ]
    head = "  ".join(str(c).ljust(w) for c, w in zip(columns, widths))
    body = ["  ".join(str(v).ljust(w) for v, w in zip(r, widths)) for r in rows]
    rule = "-" * len(head)
    return "\n".join([title, rule, head, rule, *body, ""])


def readme_source(gate: Path) -> Path | None:
    """The gate root's README.md if it exists, else None. Documentation, not an input the numbers
    depend on: it is hashed into the provenance record when present and recorded as absent when
    not (measured 2026-09-15 on artifacts/gate/4b: every table printed, then `--json` died on the
    missing file and no provenance record was written)."""
    p = Path(gate) / "README.md"
    return p if p.is_file() else None


def analyse(gate: Path) -> dict[str, Any]:
    """Every number this script reports, from the parquets up. Pure except for file reads."""
    graphs: dict[str, dict[str, GoldGraph]] = {}
    per_dir: dict[str, dict[str, Any]] = {}
    ladders: dict[str, Ladder] = {}
    checked = {"frontier_q": 0, "evidence_coverage": 0, "cad": 0}
    scorer_hashes: set[str] = set()
    graph_versions: set[str] = set()
    code_versions: set[str] = set()
    splits_seen: set[str] = set()
    sources: list[Path] = []
    checkpoints = discover_checkpoints(gate)
    # Resolved ONCE, before any number is computed, so the set of comparators cannot depend on
    # which of them looked better. Absent => `matched_question_tokens` is dropped by name.
    tokenizer = find_question_tokenizer()
    bases = active_cost_bases(tokenizer)

    for tag, model in checkpoints:
        d = gate / tag
        tabs = read_tables(d)
        sources += [d / f"{t}.parquet" for t in TABLES] + [d / f"{CALLS_TABLE}.parquet"]
        for s in tabs["scores"]:
            scorer_hashes.add(str(s["scorer_hash"]))
            graph_versions.add(str(s["graph_version"]))
        for r in tabs["runs"]:
            code_versions.add(str(r["code_version"]))
            splits_seen.add(str(r.get("split") or ""))
            suite = str(r["suite_id"])
            if suite not in graphs:
                graphs[suite] = load_graphs(suite, GRAPH_VERSION)
                sources.append(_graph_path(suite))
        shims = turn_shims(tabs["turns"])
        built = build_ladders(
            tabs["runs"],
            shims,
            graphs,
            inquirer_gen_tokens=read_inquirer_calls(d),
            question_words=question_words(shims),
            question_tokens=None if tokenizer is None else question_tokens(shims, tokenizer),
        )
        got = verify_instrument(built, stored_scores(tabs["scores"]))
        for k, v in got.items():
            checked[k] += v
        per_dir[tag] = {"model": model, "ladders": built}
        for run_id, lad in built.items():
            prev = ladders.get(run_id)
            if prev is not None and (prev.cov, prev.cad2_hit, prev.cad2_n, prev.gen_tok) != (
                lad.cov,
                lad.cad2_hit,
                lad.cad2_n,
                lad.gen_tok,
            ):
                raise Refused(f"{run_id}: the same run scores differently in two parquet dirs")
            ladders[run_id] = lad

    if len(scorer_hashes) != 1:
        raise Refused(f"expected ONE scorer_hash across the gate parquets, got {scorer_hashes}")
    if len(graph_versions) != 1:
        raise Refused(f"expected ONE graph_version, got {graph_versions}")
    by_arm: dict[str, Any] = {BASE_ARM: [], TRAINED_ARM: {}}
    for tag, model in checkpoints:
        for lad in per_dir[tag]["ladders"].values():
            if lad.arm_id == TRAINED_ARM:
                by_arm[TRAINED_ARM].setdefault(model, []).append(lad.run_id)
    by_arm[BASE_ARM] = sorted(x.run_id for x in ladders.values() if x.arm_id == BASE_ARM)

    base = {
        (lad.suite_id, lad.task_id, lad.seed): lad
        for lad in ladders.values()
        if lad.arm_id == BASE_ARM
    }
    suites = sorted({lad.suite_id for lad in ladders.values()})

    contrasts: list[Contrast] = []
    splits: list[dict[str, Any]] = []
    early: list[dict[str, Any]] = []
    for tag, model in checkpoints:
        for suite in suites:
            trained = sorted(
                (
                    lad
                    for lad in per_dir[tag]["ladders"].values()
                    if lad.arm_id == TRAINED_ARM and lad.suite_id == suite
                ),
                key=lambda x: x.run_id,
            )
            if not trained:
                continue
            for metric in ("evidence_coverage", "cad_ge2"):
                for comparator, offset in (
                    ("matched_k", 0),
                    ("matched_k_plus_1", 1),
                    ("cap8", 0),
                    *((c, 0) for c in bases),
                ):
                    contrasts.append(
                        contrast(
                            trained,
                            base,
                            suite_id=suite,
                            checkpoint=model,
                            metric=metric,
                            comparator=comparator,
                            offset=offset,
                        )
                    )
                splits.append(
                    {
                        "suite_id": suite,
                        "checkpoint": model,
                        "metric": metric,
                        **outcome_split(trained, base, metric),
                    }
                )
            early.append(
                {"suite_id": suite, "checkpoint": model, **early_stop_split(trained, base)}
            )

    frontier = _frontier(ladders, per_dir, suites, checkpoints)
    budgets = _budgets(ladders, per_dir, suites, checkpoints)
    sources += [gate / f"{model}.{suite}.json" for _tag, model in checkpoints for suite in suites]
    readme = readme_source(gate)
    if readme is not None:
        sources.append(readme)
    return {
        "checkpoints": [list(tm) for tm in checkpoints],
        "contrasts": contrasts,
        "reconciliation": reconcile(gate, contrasts),
        "splits": splits,
        "early": early,
        "frontier": frontier,
        "budgets": budgets,
        "cost_bases_active": list(bases),
        "question_tokenizer": None if tokenizer is None else str(tokenizer),
        "checked": checked,
        "scorer_hash": sorted(scorer_hashes)[0],
        "graph_version": sorted(graph_versions)[0],
        "code_versions": sorted(code_versions),
        "splits_seen": sorted(splits_seen),
        "gate": str(gate),
        "run_ids_by_arm": by_arm,
        "run_ids": sorted(ladders),
        "n_tasks": len({(x.suite_id, x.task_id) for x in ladders.values()}),
        "n_runs": {
            "base": sum(1 for x in ladders.values() if x.arm_id == BASE_ARM),
            "trained": sum(1 for x in ladders.values() if x.arm_id == TRAINED_ARM),
        },
        "sources": sorted({str(p) for p in sources}),
        "readme_present": readme is not None,
        "ladders": ladders,
        "per_dir": per_dir,
    }


def _graph_path(suite: str) -> Path:
    from pi_eval.gold import gold_root

    return gold_root() / "graphs" / suite / f"{GRAPH_VERSION}.jsonl"


def _budgets(
    ladders: Mapping[str, Ladder],
    per_dir: Mapping[str, Mapping[str, Any]],
    suites: Sequence[str],
    checkpoints: Sequence[tuple[str, str]],
) -> list[dict[str, Any]]:
    """Mean Inquirer spend per arm and per suite, on both bases, at each arm's own stop.

    This is the row that lets a reader check the equating rather than take the word "matched" on
    trust: it prints what the trained arm spent, what the base spent over its whole cap-8
    trajectory, and the per-ask rate on each basis. The base rows are SEED 0 only, and the series
    name says so -- for `_frontier`'s reason, and with `_frontier`'s caveat: on a trained arm that
    carries two seeds this is one seed of the paired population rather than all of it, so read the
    budgets as the seed-0 slice they are labelled as and the deltas from T1.
    """
    out: list[dict[str, Any]] = []
    nan = float("nan")

    def row(series: str, kind: str, suite: str, runs: Sequence[Ladder]) -> dict[str, Any]:
        asks = _fmean([float(x.n_asks) for x in runs])
        gen = _fmean([x.gen_tokens_at(x.n_asks) for x in runs])
        words = _fmean([x.question_words_at(x.n_asks) for x in runs])
        toks = _fmean([x.question_tokens_at(x.n_asks) for x in runs]) if runs[0].q_tok else nan
        return {
            "suite_id": suite,
            "series": series,
            "kind": kind,
            "n_runs": len(runs),
            "asks": asks,
            "gen_tokens": gen,
            "question_words": words,
            "question_tokens": toks,
            "gen_tokens_per_ask": gen / asks if asks else nan,
            "words_per_ask": words / asks if asks else nan,
            "tokens_per_ask": toks / asks if asks else nan,
        }

    for suite in suites:
        b = [
            lad
            for lad in ladders.values()
            if lad.arm_id == BASE_ARM and lad.suite_id == suite and lad.seed == 0
        ]
        if b:
            out.append(row(f"{BASE_ARM} (seed 0, whole cap-{CAP})", "base", suite, b))
        for tag, model in checkpoints:
            t = [
                lad
                for lad in per_dir[tag]["ladders"].values()
                if lad.arm_id == TRAINED_ARM and lad.suite_id == suite
            ]
            if t:
                out.append(row(model, "trained", suite, t))
    return out


def _frontier(
    ladders: Mapping[str, Ladder],
    per_dir: Mapping[str, Mapping[str, Any]],
    suites: Sequence[str],
    checkpoints: Sequence[tuple[str, str]],
) -> list[dict[str, Any]]:
    """The base's ladder as a curve, the checkpoints as points. One row per plotted marker.

    The base rows are SEED 0 and SEED 1, drawn as two separate curves and never averaged into
    one, so a reader is never shown a two-seed mean against a one-seed point. That is the whole
    of the claim now: the parenthesis "-- the population the paired deltas use --" stood here
    until 2026-09-18 and was true only while every trained arm carried seed 0 alone. It does not
    hold on the held-out store, where the trained arm carries both seeds and `contrast` pairs
    each against its own seed before averaging the task, so the paired population is BOTH curves.
    """
    out: list[dict[str, Any]] = []
    for suite in suites:
        # 0 and 1 are the seeds every store measured so far carries (four gate roots and the
        # held-out QA store: baseline at both, checkpoints at seed 0 on the gates and at both on
        # the held-out store). A run at a third seed would be absent from THIS table -- the paired
        # deltas in T1 would still include it, since `contrast` reads the seed off the run.
        for seed in (0, 1):
            runs = [
                lad
                for lad in ladders.values()
                if lad.arm_id == BASE_ARM and lad.suite_id == suite and lad.seed == seed
            ]
            if not runs:
                continue
            for k in range(CAP + 1):
                cov = [lad.coverage_at(k) for lad in runs]
                cad = [lad.cad2_at(k) for lad in runs]
                c_mean, c_lo, c_hi = level(cov)
                d_mean, d_lo, d_hi = level(cad)
                out.append(
                    {
                        "suite_id": suite,
                        "series": f"qwen3-8b-base prefix (seed {seed})",
                        "kind": "base_prefix",
                        "seed": seed,
                        "k": k,
                        "asks": _fmean([float(lad.asks_at(k)) for lad in runs]),
                        "coverage": c_mean,
                        "coverage_lo": c_lo,
                        "coverage_hi": c_hi,
                        "cad_ge2": d_mean,
                        "cad_ge2_lo": d_lo,
                        "cad_ge2_hi": d_hi,
                        "n_runs": len(runs),
                        "n_cad": sum(1 for v in cad if not math.isnan(v)),
                    }
                )
        for tag, model in checkpoints:
            runs = [
                lad
                for lad in per_dir[tag]["ladders"].values()
                if lad.arm_id == TRAINED_ARM and lad.suite_id == suite
            ]
            if not runs:
                continue
            cov = [lad.coverage_at(lad.n_asks) for lad in runs]
            cad = [lad.cad2_at(lad.n_asks) for lad in runs]
            c_mean, c_lo, c_hi = level(cov)
            d_mean, d_lo, d_hi = level(cad)
            out.append(
                {
                    "suite_id": suite,
                    "series": model,
                    "kind": "trained_stop",
                    # The seed this marker IS, or None when it pools more than one. It was the
                    # literal 0 whatever the runs carried; on the two-seed held-out store that
                    # labelled a both-seeds marker as seed 0. Single-seed roots still read 0, so
                    # no recorded frontier row moves.
                    "seed": seeds[0] if len(seeds := sorted({x.seed for x in runs})) == 1 else None,
                    "k": None,
                    "asks": _fmean([float(lad.n_asks) for lad in runs]),
                    "coverage": c_mean,
                    "coverage_lo": c_lo,
                    "coverage_hi": c_hi,
                    "cad_ge2": d_mean,
                    "cad_ge2_lo": d_lo,
                    "cad_ge2_hi": d_hi,
                    "n_runs": len(runs),
                    "n_cad": sum(1 for v in cad if not math.isnan(v)),
                }
            )
    return out


# ------------------------------------------------- the citable provenance block

# WHY A SECOND BLOCK, beside the figure's own `provenance.json`. That one describes a FIGURE; a
# paper table cites NUMBERS, and rule 1 says every reported value names its run identity, scorer
# hash and graph version. The matched-question-token and matched-question-word deltas are the
# paper's answer to the verbosity objection and were sitting behind a not-ready marker for want
# of exactly this file.
#
# SHAPE MATCHED, NOT INVENTED. The key names and the digest discipline are
# `pi_eval.report.provenance`'s. Two deliberate departures, each stated in the block itself:
#
#   * NO `generated_at_utc`. That function excludes the timestamp from its DIGEST so two
#     identical aggregations an hour apart compare equal. A block a table cites needs the
#     stronger property -- the BYTES equal -- so the clock is omitted entirely rather than
#     written and then excused. A wall clock is a machine artifact: recorded, never compared.
#   * NO `eligibility_predicate` / `prereg_verified`. `pi_eval.report.ELIGIBLE` filters on
#     `split = 'test'`; this analysis reads the tier-B DEV gate and applies no such filter.
#     Naming a filter that did not run is the failure that field's own comment warns about, so
#     the split actually present is recorded instead.
PROVENANCE_ID = "P13_matched_cost_numbers"
PROVENANCE_FILE = "matched_cost_provenance.json"
DEGENERATE_BASES: Mapping[str, str] = {
    "matched_gen_tokens": (
        "DEGENERATE ON THIS DATA, not citable as a coverage result. The base spends its "
        "completion on a longer rationale than the trained arms (131 chars against 77) while "
        "asking a SHORTER question, so its generated tokens per ask are ~90 against their "
        "~49-61. A generated-token budget therefore buys the base 0.04-0.86 asks and the delta "
        "is measured against a base that asked nothing -- read `base_asks` in this row. It is "
        "reported as the cost axis it is; the length control is matched_question_tokens."
    )
}


def _required(res: Mapping[str, Any], field: str, value: Any) -> Any:
    """A provenance field is present and non-blank, or the block is REFUSED.

    `analyse` pins `len(scorer_hashes) == 1`, and an empty string is a set of size one -- so a
    parquet whose `scorer_hash` column was written blank passed that check and would have
    produced a block with a hole where the identity goes. A hole invites exactly the false
    confidence this block exists to prevent, so it is a refusal rather than an omitted key.
    """
    if value is None or (isinstance(value, str) and not value.strip()) or value == []:
        raise Refused(
            f"{PROVENANCE_ID}: `{field}` is {value!r}. A number without provenance is not a "
            "result (CONTRIBUTING.md rule 1); no matched-cost provenance block is written without it."
        )
    return value


def matched_cost_provenance(res: Mapping[str, Any]) -> dict[str, Any]:
    """The block a paper table cites, GENERATED from the analysis result and never typed.

    Every value here is read off `res` -- the object `analyse()` returned -- so the block cannot
    drift from the numbers it describes, and it carries no clock so two emissions of the same
    analysis are byte-identical.
    """
    scorer = _required(res, "scorer_hash", res.get("scorer_hash"))
    gv = _required(res, "graph_version", res.get("graph_version"))
    codes = sorted({c for c in res.get("code_versions") or () if str(c).strip()})
    _required(res, "code_version", codes)
    files = {
        (str(Path(p).relative_to(REPO)) if str(p).startswith(str(REPO)) else str(p)): sha256_file(
            Path(p)
        )
        for p in res.get("sources") or ()
    }
    body: dict[str, Any] = {
        "table_id": PROVENANCE_ID,
        "tool": f"{TOOL} --gate {res.get('gate', '')}",
        "gate": str(res.get("gate", "")),
        "query": (
            "runs/turns/scores/calls from each <gate>/parquet_*/; gold graphs at "
            f"graph_version={gv}; MechanicalMatcher {MechanicalMatcher.matcher_id} re-run on "
            "every trajectory prefix; the base's prefix ladder charged at the trained arm's own "
            "retrieval count (matched_k) and at its own Inquirer budget (see cost_bases)"
        ),
        "scorer_hash": scorer,
        "graph_version": gv,
        "code_version": codes,
        "split": sorted(res.get("splits_seen") or ()),
        "matcher_id": MechanicalMatcher.matcher_id,
        "run_ids": sorted(res.get("run_ids") or ()),
        "n_run_ids": len(set(res.get("run_ids") or ())),
        # BOTH SIDES, by run id. A matched-cost delta is a paired difference; a block that named
        # only a pooled run list could not show which population stood on either side of it.
        "run_ids_by_arm": {
            BASE_ARM: sorted((res.get("run_ids_by_arm") or {}).get(BASE_ARM) or ()),
            TRAINED_ARM: {
                model: sorted(ids)
                for model, ids in sorted(
                    ((res.get("run_ids_by_arm") or {}).get(TRAINED_ARM) or {}).items()
                )
            },
        },
        "n_tasks": int(res.get("n_tasks") or 0),
        "n_runs": dict(res.get("n_runs") or {}),
        "resampling": {
            "n_boot": N_BOOT,
            "n_perm": N_PERM,
            "seed": SEED,
            "unit": "(suite_id, task_id)",
            "method": "BCa cluster bootstrap + cluster-level sign flip "
            "(pi_eval.stats.inference.paired_difference)",
        },
        "instrument_checks": dict(res.get("checked") or {}),
        "reconciliation": list(res.get("reconciliation") or ()),
        "cost_bases_active": list(res.get("cost_bases_active") or ()),
        # The token unit is unreproducible without naming the tokenizer that produced it.
        "question_tokenizer": res.get("question_tokenizer"),
        "cost_bases": COST_BASIS_NOTES,
        "primary_length_control": "matched_question_tokens",
        "robustness_check": "matched_question_words",
        "comparators": [
            {
                "suite_id": c.suite_id,
                "checkpoint": c.checkpoint,
                "metric": c.metric,
                "comparator": c.comparator,
                "n": c.n_tasks,
                "n_clusters": c.n_clusters,
                # WHICH pairs, not just how many: a digest of the exact (suite/task) keys that
                # entered, so a re-derivation that silently dropped one cannot agree.
                "pair_keys_hash": c.pair_keys_hash,
                "trained_mean": c.trained_mean,
                "base_mean": c.base_mean,
                "delta": c.delta,
                "ci_lo": c.ci_lo,
                "ci_hi": c.ci_hi,
                "p_value": c.p_value,
                "trained_asks": c.trained_asks,
                "base_asks": c.base_asks,
                "trained_cost": c.trained_cost,
                "base_cost": c.base_cost,
                "n_base_short": c.n_base_short,
                "degenerate": c.comparator in DEGENERATE_BASES,
                "degenerate_reason": DEGENERATE_BASES.get(c.comparator, ""),
            }
            for c in res.get("contrasts") or ()
        ],
        # Mean question tokens and words per arm and per base, so a reader sees the budgets that
        # were equated rather than trusting the word "matched".
        "inquirer_budgets": list(res.get("budgets") or ()),
        "inputs": files,
        "inputs_hash": h("inputs", canon(sorted(files.items()))),
        "departures_from_pi_eval_report_provenance": [
            "no generated_at_utc: a block a table cites must be BYTE-identical across two "
            "emissions of the same analysis, and a wall clock defeats that. The digest is the "
            "identity.",
            "no eligibility_predicate/prereg_verified: pi_eval.report.ELIGIBLE filters on "
            "split='test' and this reads the tier-B DEV gate, so naming it would name a filter "
            "that did not run. `split` records what is actually present.",
        ],
    }
    body["provenance_digest"] = h("prov", canon(body))
    return body


def write_provenance(res: Mapping[str, Any], path: Path) -> Path:
    """Serialise the block deterministically: sorted keys, fixed indent, one trailing newline."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(matched_cost_provenance(res), indent=2, sort_keys=True) + "\n")
    return p


# --------------------------------------------------------------------------- figure


PALETTE = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000")
MARKERS = ("o", "s", "^", "D", "v", "P", "X")
GREY = "#8C8C8C"


def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": ":",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
            "savefig.bbox": "tight",
        }
    )
    return plt


def draw(res: Mapping[str, Any], out: Path) -> Path:
    """One panel per (metric, suite): the base's prefix ladder, the checkpoints as points."""
    plt = _mpl()
    rows = res["frontier"]
    suites = sorted({r["suite_id"] for r in rows})
    metrics = (("coverage", "required-evidence coverage"), ("cad_ge2", "C@d$\\geq$2"))
    fig, axes = plt.subplots(
        len(metrics), len(suites), figsize=(6.9, 6.0), sharex=True, squeeze=False
    )
    for i, (metric, ylab) in enumerate(metrics):
        for j, suite in enumerate(suites):
            ax = axes[i][j]
            lad = sorted(
                (r for r in rows if r["suite_id"] == suite and r["kind"] == "base_prefix"),
                key=lambda r: (r["seed"], r["k"]),
            )
            for seed, col, ls in ((0, PALETTE[6], "-"), (1, GREY, "--")):
                pts = [r for r in lad if r["seed"] == seed]
                if not pts:
                    continue
                xs = [r["asks"] for r in pts]
                ys = [r[metric] for r in pts]
                ax.plot(
                    xs,
                    ys,
                    color=col,
                    linestyle=ls,
                    marker="o" if seed == 0 else None,
                    markersize=3,
                    linewidth=1.3,
                    zorder=2,
                    label=f"qwen3-8b-base prefix, seed {seed}" if i + j == 0 else None,
                )
                if seed == 0:
                    ax.fill_between(
                        xs,
                        [r[f"{metric}_lo"] for r in pts],
                        [r[f"{metric}_hi"] for r in pts],
                        color=col,
                        alpha=0.12,
                        linewidth=0.0,
                        zorder=1,
                    )
                    # ABOVE-LEFT of the curve. Below it is where the checkpoint markers and
                    # their whiskers sit, and a rung label sitting on a measured point is how
                    # a reader mis-reads which x a marker is at.
                    for r in pts:
                        if r["k"] in (1, 2, 4, 6, 8):
                            ax.annotate(
                                f"k={r['k']}",
                                (r["asks"], r[metric]),
                                textcoords="offset points",
                                xytext=(-4, 7),
                                fontsize=6,
                                color=col,
                            )
            for m, (_tag, model) in enumerate(res["checkpoints"]):
                pt = next(
                    (r for r in rows if r["suite_id"] == suite and r["series"] == model), None
                )
                if pt is None:
                    continue
                ax.errorbar(
                    [pt["asks"]],
                    [pt[metric]],
                    yerr=[[pt[metric] - pt[f"{metric}_lo"]], [pt[f"{metric}_hi"] - pt[metric]]],
                    color=PALETTE[m],
                    marker=MARKERS[m + 1],
                    markersize=7,
                    markeredgecolor="white",
                    markeredgewidth=0.7,
                    linewidth=1.0,
                    capsize=2.5,
                    zorder=4 + m,
                    label=model if i + j == 0 else None,
                )
            n = next(
                (
                    r["n_runs"]
                    for r in rows
                    if r["suite_id"] == suite and r["kind"] == "base_prefix"
                ),
                0,
            )
            if i == 0:
                ax.set_title(f"{suite}  (n={n} tasks)", fontsize=9)
            if j == 0:
                ax.set_ylabel(ylab, fontsize=8)
            if i == len(metrics) - 1:
                ax.set_xlabel("mean retrieval calls (= asks)", fontsize=8)
            ax.tick_params(labelsize=7)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.suptitle(
        "P13  Matched cost: the prompted base's prefix ladder vs each checkpoint's own stop",
        fontsize=9.5,
    )
    # The legend gets its own reserved strip. Laid over the axes it covered the bottom row's
    # x-label, which is the axis the whole argument is about.
    fig.tight_layout(rect=(0, 0.11, 1, 0.965))
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=7,
        bbox_to_anchor=(0.5, 0.005),
    )
    d = out / FIGURE_ID
    d.mkdir(parents=True, exist_ok=True)
    fig.savefig(d / f"{FIGURE_ID}.pdf")
    fig.savefig(d / f"{FIGURE_ID}.png")
    plt.close(fig)
    return d


def provenance(res: Mapping[str, Any]) -> dict[str, Any]:
    files = {
        (str(Path(p).relative_to(REPO)) if str(p).startswith(str(REPO)) else str(p)): sha256_file(
            Path(p)
        )
        for p in res["sources"]
    }
    body: dict[str, Any] = {
        "table_id": FIGURE_ID,
        "measured": True,
        "readme_present": bool(res.get("readme_present", False)),
        "tool": f"{TOOL} --gate artifacts/gate",
        "inputs": files,
        "inputs_hash": h("inputs", canon(sorted(files.items()))),
        "query": (
            "runs/turns/scores from each artifacts/gate/parquet_*/; gold graphs at "
            f"graph_version={res['graph_version']}; MechanicalMatcher "
            f"{MechanicalMatcher.matcher_id} re-run on every trajectory prefix"
        ),
        "run_ids": res["run_ids"],
        "n_run_ids": len(res["run_ids"]),
        "scorer_hash": [res["scorer_hash"]],
        "graph_version": [res["graph_version"]],
        "code_version": res["code_versions"],
        "matcher_id": MechanicalMatcher.matcher_id,
        "bootstrap": {
            "n_resamples": N_BOOT,
            "n_perm": N_PERM,
            "seed": SEED,
            "unit": "(suite_id, task_id)",
            "method": "BCa cluster bootstrap + cluster-level sign flip "
            "(pi_eval.stats.inference.paired_difference)",
        },
        "instrument_checks": dict(res["checked"]),
        "n_runs": dict(res["n_runs"]),
        "contrasts": [asdict(c) for c in res["contrasts"]],
        "reconciliation": list(res["reconciliation"]),
        "outcome_splits": res["splits"],
        "early_stop": res["early"],
        "frontier": res["frontier"],
        "inquirer_budgets": res["budgets"],
        "cost_bases_active": res["cost_bases_active"],
        "question_tokenizer": res["question_tokenizer"],
        "cost_bases": COST_BASIS_NOTES,
    }
    body["provenance_digest"] = h("prov", canon(body))
    body["generated_at_utc"] = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    return body


CAPTION = """
**P13 — Matched cost.** The tier-B dev gate compared three Qwen3-8B checkpoints
(`inquirer_trained`) against the prompted base `qwen3-8b-base` (`inquirer_prompted`) at
retrieval cap 8, and every checkpoint lost on coverage while asking far fewer questions. The
paper's endpoint is the ordered pair (Δ coverage, Δ retrieval calls), so the comparator here is
the base's own **prefix ladder**: because `Inquirer.act(s)` takes no budget, the first *k*
questions of a cap-8 prompted run are the questions a cap-*k* run would have asked, and its
coverage after *k* asks is that run's terminal coverage. Black line: the base at *k* = 0…8 asks
(base seed 0 solid, grey dashed seed 1, shown so seed noise is visible rather than averaged
away; the paired deltas use every seed the TRAINED arm carries, each charged against the same
seed's baseline and averaged within its task). Coloured markers: each checkpoint at its own natural stop. Bands and
whiskers are task-clustered BCa 95% intervals ({n_boot} resamples, seed {seed}). Coverage is
`pi_eval.metrics.discovery.evidence_coverage` over the required gold spans; C@d≥2 is the
|V_d|-weighted C@d over depths ≥ 2, decided by `MechanicalMatcher` `{matcher}` at every prefix.
Every ladder's last rung reproduces the verdicts' own scorer bit for bit
(`scorer_hash {scorer}`, `graph_version {gv}`): {n_fq} `frontier_q#k` values, {n_cov}
`evidence_coverage` values and {n_cad} `cad#d` values checked, 0 disagreements. Drawn from
`{gate_layout}` (sha256 of every file in
`provenance.json`); {n_base} base runs ({n_base_seeds} seed(s) x {n_tasks} tasks) and
{n_trained} checkpoint runs ({n_per_ckpt} per checkpoint, seed(s) {trained_seeds}), all
`status='ok'`, `split={splits}`, `budget_cap=8`. This analysis reproduces all {n_locks}
published verdict values it is locked against, in `{gate_dir}/*.json`, to ten decimals -- each
against the criterion field that says which quantity it is (`value` is the matched-k delta under
`coverage_rule matched_cost`, `cap8_value` / `cap8_cad_ge2_delta` the cap-8 one) -- and that is
what ties the matched-cost rows to the gate they re-read.
"""


def _rel(p: Path) -> str:
    """A path as the repo would cite it. Relative under the repo, else as given -- `caption.md` is
    one of the `.md` files `scripts/check_no_home_paths.sh` greps, so a repo-local gate must not
    be written into it absolutely."""
    return str(p.relative_to(REPO)) if str(p).startswith(str(REPO)) else str(p)


def _seeds(res: Mapping[str, Any], arm: str) -> list[int]:
    """The seeds an arm actually carries in this analysis, for the caption to state."""
    return sorted({lad.seed for lad in res["ladders"].values() if lad.arm_id == arm})


def write_figure(res: Mapping[str, Any], out: Path) -> Path:
    d = draw(res, out)
    prov = provenance(res)
    (d / "provenance.json").write_text(json.dumps(prov, indent=2, sort_keys=True) + "\n")
    write_provenance(res, d / PROVENANCE_FILE)
    (d / "caption.md").write_text(
        CAPTION.strip().format(
            n_boot=N_BOOT,
            seed=SEED,
            matcher=MechanicalMatcher.matcher_id,
            scorer=res["scorer_hash"][:12],
            gv=res["graph_version"],
            n_fq=res["checked"]["frontier_q"],
            n_cov=res["checked"]["evidence_coverage"],
            n_cad=res["checked"]["cad"],
            n_base=res["n_runs"]["base"],
            n_trained=res["n_runs"]["trained"],
            n_tasks=res["n_tasks"],
            n_per_ckpt=res["n_runs"]["trained"] // len(res["checkpoints"]),
            # The gate this ran on, not the one the caption was first written for. It said
            # `artifacts/gate/` and "all twelve published verdict values" whatever it had been
            # pointed at; on the held-out store that is a wrong directory and a wrong count.
            gate_layout=", ".join(
                f"{_rel(Path(res['gate']) / tag)}/" for tag, _m in res["checkpoints"]
            ),
            gate_dir=_rel(Path(res["gate"])),
            n_locks=len(res["reconciliation"]),
            # DERIVED, not asserted. This caption said "2 seeds x N dev tasks ... seed 0 ...
            # split='dev'" while the seed counts and the split were sitting in `res`: true of
            # the pass-1 dev gate it was written against, false the moment the same tool is
            # pointed at the two-seed held-out store (see `contrast`).
            n_base_seeds=len(_seeds(res, BASE_ARM)),
            trained_seeds=", ".join(str(x) for x in _seeds(res, TRAINED_ARM)) or "none",
            splits=", ".join(f"'{x}'" for x in res["splits_seen"]),
        )
        + "\n"
    )
    return d


def render(res: Mapping[str, Any]) -> str:
    parts: list[str] = []
    parts.append(
        text_table(
            "T0  Reconciliation: the recomputed rows against the published verdicts "
            "(`field` is the criterion field each was checked against)",
            (
                "verdict",
                "metric",
                "comparator",
                "field",
                "verdict_value",
                "recomputed",
                "n_verdict",
                "n_here",
                "agree",
            ),
            [
                [
                    r["verdict"],
                    r["metric"],
                    r["comparator"],
                    r["field"],
                    f"{r['verdict_value']:+.10f}",
                    f"{r['recomputed']:+.10f}",
                    str(r["verdict_n"]),
                    str(r["recomputed_n"]),
                    "yes" if r["agree"] else "NO",
                ]
                for r in res["reconciliation"]
            ],
        )
    )
    parts.append(
        text_table(
            "T1  Paired matched-cost deltas (trained at its own stop vs base prefix). READ "
            "`asks_base` BEFORE any budget row: on artifacts/gate/n1 and /8b2-t20 the "
            "`matched_gen_tokens` basis is DEGENERATE -- the base generates ~90 tokens per ask "
            "against the trained arms' ~49-61 (a longer rationale, not a longer question), so a "
            "generated-token budget buys it 0.04-0.86 asks and the delta is against a base that "
            "asked nothing. `matched_question_words` is the length control that answers the "
            "verbosity objection, and `matched_question_tokens` -- the same questions in the pin's own "
            "tokens -- is the PRIMARY one: the budget that means something physically is the text "
            "the Drafter receives, tokenized as the model tokenizes it. Words are the robustness "
            "check on that unit, fixed as such before either number was computed.",
            CONTRAST_COLUMNS,
            [c.row() for c in res["contrasts"]],
        )
    )
    parts.append(
        text_table(
            "T2  Frontier: mean metric vs mean retrieval calls",
            (
                "suite",
                "series",
                "k",
                "asks",
                "coverage",
                "cov_ci95",
                "cad_ge2",
                "cad_ci95",
                "n_runs",
                "n_cad",
            ),
            [
                [
                    r["suite_id"],
                    r["series"],
                    "--" if r["k"] is None else str(r["k"]),
                    f"{r['asks']:.3f}",
                    f"{r['coverage']:.4f}",
                    f"[{r['coverage_lo']:.4f}, {r['coverage_hi']:.4f}]",
                    f"{r['cad_ge2']:.4f}",
                    f"[{r['cad_ge2_lo']:.4f}, {r['cad_ge2_hi']:.4f}]",
                    str(r["n_runs"]),
                    str(r["n_cad"]),
                ]
                for r in res["frontier"]
            ],
        )
    )
    parts.append(
        text_table(
            "T3  Trained vs base prefix at the SAME k: win / tie / loss",
            ("suite", "checkpoint", "metric", "n", "win", "tie", "loss", "tie_share", "win_share"),
            [
                [
                    s["suite_id"],
                    s["checkpoint"],
                    s["metric"],
                    str(s["n"]),
                    str(s["win"]),
                    str(s["tie"]),
                    str(s["loss"]),
                    f"{s['tie'] / s['n']:.3f}" if s["n"] else "--",
                    f"{s['win'] / s['n']:.3f}" if s["n"] else "--",
                ]
                for s in res["splits"]
            ],
        )
    )
    parts.append(
        text_table(
            "T4  Early stop: what a cap-8 coverage loss was made of",
            (
                "suite",
                "checkpoint",
                "n_paired",
                "done_at_stop",
                "lose_cap8",
                "lose_done",
                "lose_tie_or_win_at_k",
                "lose_behind_at_k",
            ),
            [
                [
                    e["suite_id"],
                    e["checkpoint"],
                    str(e["n_paired"]),
                    f"{e['n_done_at_stop']} ({e['n_done_at_stop'] / e['n_paired']:.3f})"
                    if e["n_paired"]
                    else "--",
                    str(e["n_lose_at_cap8"]),
                    str(e["lose_done_at_stop"]),
                    f"{e['lose_tie_or_win_at_k']}"
                    + (
                        f" ({e['lose_tie_or_win_at_k'] / e['n_lose_at_cap8']:.3f})"
                        if e["n_lose_at_cap8"]
                        else ""
                    ),
                    f"{e['lose_behind_at_k']}"
                    + (
                        f" ({e['lose_behind_at_k'] / e['n_lose_at_cap8']:.3f})"
                        if e["n_lose_at_cap8"]
                        else ""
                    ),
                ]
                for e in res["early"]
            ],
        )
    )
    parts.append(
        text_table(
            "T5  Inquirer budgets being equated (each arm at its OWN stop; base = whole cap-8 "
            "trajectory). `question_tokens` is the PRIMARY unit -- the question text in the "
            "Inquirer pin's own tokenizer, read from a LOCAL tokenizer.json (see "
            "`question_tokenizer` in provenance); `question_words` is the robustness check on it "
            "and the human-readable gloss. `gen_tokens` is tok_completion+tok_reasoning, and the "
            "Inquirer's tok_reasoning is 0 on every call in these gates, so it is also "
            "completion-minus-reasoning; it is DEGENERATE as a budget (see T1) because the base "
            "spends its completion on a longer rationale, not a longer question.",
            (
                "suite",
                "series",
                "kind",
                "n_runs",
                "asks",
                "gen_tokens",
                "gen_tok/ask",
                "question_tokens",
                "q_tok/ask",
                "question_words",
                "words/ask",
            ),
            [
                [
                    r["suite_id"],
                    r["series"],
                    r["kind"],
                    str(r["n_runs"]),
                    f"{r['asks']:.3f}",
                    f"{r['gen_tokens']:.1f}",
                    f"{r['gen_tokens_per_ask']:.1f}",
                    f"{r['question_tokens']:.1f}",
                    f"{r['tokens_per_ask']:.1f}",
                    f"{r['question_words']:.1f}",
                    f"{r['words_per_ask']:.1f}",
                ]
                for r in res["budgets"]
            ],
        )
    )
    return "\n".join(parts)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--gate", default=str(REPO / "artifacts/gate"), help="the gate artifact dir")
    ap.add_argument("--out", default=str(REPO / "figures"), help="figures root")
    ap.add_argument("--json", default=None, help="also write the full record here")
    ap.add_argument("--no-figure", action="store_true", help="tables only")
    ap.add_argument(
        "--provenance",
        default=None,
        help="write the citable matched-cost provenance block here. It is written into "
        f"the figure directory as {PROVENANCE_FILE} too, so a --no-figure run can still "
        "emit it.",
    )
    a = ap.parse_args(argv)
    try:
        res = analyse(Path(a.gate))
    except Refused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    print(render(res))
    if not a.no_figure:
        d = write_figure(res, Path(a.out))
        print(f"figure: {d.relative_to(REPO) if str(d).startswith(str(REPO)) else d}")
    if a.provenance:
        q = write_provenance(res, Path(a.provenance))
        print(f"provenance: {q.relative_to(REPO) if str(q).startswith(str(REPO)) else q}")
    if a.json:
        Path(a.json).write_text(json.dumps(provenance(res), indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
