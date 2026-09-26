"""POST /score — the ONLY way a trainer learns what an episode was worth.

THE REFUSAL IS THE FEATURE. This process runs with `PI_GOLD_ROOT` set; the trainer runs with
it unset. So the trainer cannot read gold by import, by file, or by accident — it can only
ask, and this module decides. Everything below follows from taking that seriously:

  * a `task_id` outside `train` is REFUSED, with `SPLIT_REFUSED_EXIT` (3, not 1: a CI job
    must be able to tell "you asked for the one thing that voids the experiment" apart from
    "something broke");
  * the refusal reads the split that was STAMPED IN THE RUN'S OWN MANIFEST at rollout time.
    Recomputing it here would let a later edit to the bucket function silently reclassify
    finished runs, which is the failure this whole apparatus exists to prevent;
  * eval-only suites are refused by name as well as by split, because `tau2` and `pare` are
    the zero-shot transfer targets and "it happened to hash into train" is not a reason to
    burn them;
  * a GOLD-EXPOSED episode is refused too. `gold_evidence` and `oracle_vreq` are ceilings
    whose questions were selected where gold is readable; training on them is oracle
    distillation wearing an experiment's clothes, and it would inflate the trained arm by an
    amount no ablation could recover.

WHAT IS RETURNED, AND WHAT IS NOT. Components, never a reward. `pi_run` may not import
`pinq_train` (import-linter contract 4), so the weights physically cannot live here — and
that constraint is the right one anyway: re-weighting a finished rollout set must be
arithmetic over stored numbers, exactly as re-pricing a sweep is arithmetic over stored token
counts. A reward computed server-side would make every weight change a re-score.

WHAT Q AND PHI ARE. Identical to `pi_eval.score`, by calling the same function:

    Q(E) = |E n gold_ev_uids| / |gold_ev_uids|       (pi_eval.metrics.discovery)

over the REQUIRED partition. `potential` is that same coverage evaluated at each prefix, so
Phi is a function of retrieved EvidenceUnits and of nothing else. That is the anti-gaming
property the reward depends on: a policy cannot move Phi by rewriting its question text,
only by causing different documents to come back.

A TASK WITH NO GOLD GRAPH IS REFUSED, NOT SCORED AS ZERO. `evidence_coverage` divides by
|gold| and returns NaN on an empty denominator; a NaN that reaches a reward propagates into
a gradient and is invisible in every summary statistic that skips it. `NoGoldForTask` is
raised instead, which is loud and has an obvious remedy (build the graph, or stop asking).

`phi_loo` is returned for diagnostics and is deliberately NOT what the trainer optimizes.
Two reasons, and the second is the one that matters: a leave-one-out value is not available
online, so training on it trains on an oracle; and the phi_LOO the PAPER reports is computed
over a re-drafted ANSWER (pi_eval.metrics.qvalue.phi_loo), not over coverage. The field here
is the coverage-space analogue, useful for a sanity plot and for nothing else.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal, Sequence

from pinq.budget import HARD_CURRENCY
from pinq.ids import h
from pinq.splitting import EVAL_ONLY_SUITES as _EVAL_ONLY_SUITES
from pinq.wire import SPLIT_REFUSED_EXIT, ScoreRequest, ScoreResponse, TurnScore, to_dict

# Refused by NAME, before any hashing. These are the zero-shot transfer targets: tau2's 97
# tasks are far too few to split, and "transfers to an action-consequential environment it
# was never trained on" is a much stronger claim than a within-suite gain.
#
# THE DUPLICATION IS GONE, AND THE COMMENT THAT DEFENDED IT WAS WRONG. It said the literal
# had to be retyped here because contract 4 forbids `pi_run` importing `pinq_train`. That
# much is true and is not the whole story: the set is also defined in `pinq.splitting`,
# which is stdlib-only and which this module already imports for `HARD_CURRENCY` and `h`.
# So the server and the trainer can share ONE definition without either importing the other,
# and the price the comment was paying was being paid for nothing. tests/test_serve.py now
# pins that they are the same object rather than that two literals happen to match.
EVAL_ONLY_SUITES: frozenset[str] = _EVAL_ONLY_SUITES

DEFAULT_GRAPH_VERSION = "v1"

# Bumped when the MEASUREMENT changes — the Q definition, the partition, the prefix rule.
# It rides into ScoreResponse.scorer_hash, so a stored training value can always be traced to
# the function that produced it (rule 1: a number without provenance is not a result).
SCORER_VERSION = "serve-score-1"

# The counter a policy increments when it emits prose where JSON was asked for. It reaches
# the ledger through the write-only Recorder (pinq.protocols.Recorder), which is why a
# malformed rate is reportable without the policy being able to read anything back.
MALFORMED_CURRENCY = "malformed"


class SplitRefused(RuntimeError):
    """The scorer was asked for a task it must not score.

    Carries the exit code so a CLI wrapper does not have to remember it, and so the HTTP
    layer can map one exception class to one status code.
    """

    exit_code = SPLIT_REFUSED_EXIT


class EpisodeNotFound(FileNotFoundError):
    pass


class NoGoldForTask(FileNotFoundError):
    """No gold graph for this (suite, task, graph_version).

    A separate class from EpisodeNotFound because the remedy is different: the episode is
    fine, the measuring instrument is missing.
    """


# --------------------------------------------------------------------------- run artifacts


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def episode_dir(runs_root: str | Path, episode_id: str) -> Path:
    """`episode_id` IS the run_id, which IS the run directory name.

    A traversal guard, not paranoia: this function is reachable from an HTTP body, and
    `../../etc` is a directory name until someone checks.
    """
    if "/" in episode_id or "\\" in episode_id or episode_id in ("", ".", ".."):
        raise EpisodeNotFound(f"{episode_id!r} is not a run id")
    d = Path(runs_root) / episode_id
    if not (d / "manifest.json").exists():
        raise EpisodeNotFound(f"no run {episode_id!r} under {runs_root}")
    return d


def assert_scorable(manifest: dict[str, Any], *, split_assert: str = "train") -> None:
    """Layer 4 of the split firewall. Raises `SplitRefused`, never returns a bool.

    A predicate would invite `if scorable(...)` at a call site that forgets the else.

    THE COMPARISON TARGET IS NOT THE CALLER'S TO CHOOSE. `handle_score` passes
    `req.split_assert` -- a field of the request body -- and this compared the run's split
    against it. `pinq.wire.ScoreRequest` types that field `Literal["train"]` and its docstring
    says "the server refuses anything else", but a Literal is a type annotation, not a runtime
    check, and `from_dict` deserializes whatever the body contains. Reproduced end to end:

        from_dict(ScoreRequest, {"episode_id": "e1", "split_assert": "test"})   -> accepted
        assert_scorable({... "split": "test"}, split_assert="test")             -> ALLOWED

    So the one guard standing between a trainer and a held-out episode was parameterised by the
    party it exists to defend against, and wire.py's own comment gives the reason that is
    fatal: "a check the trainer performs on itself is a check the trainer can skip". This is
    that check, moved to the server and then handed back.

    The field stays -- it makes the refusal visible in a request body a reviewer can read, which
    is why it was introduced -- but only one value is now accepted, and any other is refused
    before the run's split is even looked at.
    """
    if split_assert != "train":
        raise SplitRefused(
            f"split_assert={split_assert!r} is not a value a client may ask for. This server "
            "scores TRAINING episodes; the comparison target is fixed at 'train' because a "
            "firewall whose threshold the caller supplies is not a firewall. If an eval-side "
            "caller genuinely needs another split, that is a server-side configuration change "
            "with a reviewer attached, not a request field."
        )
    suite = str(manifest.get("suite_id", ""))
    task = str(manifest.get("task_id", ""))
    if suite in EVAL_ONLY_SUITES:
        raise SplitRefused(
            f"{suite}/{task}: {suite} is eval-only by construction. Scoring it for a trainer "
            "would destroy the zero-shot transfer claim that is the reason it is in the paper."
        )
    split = str(manifest.get("split", ""))
    if split != split_assert:
        raise SplitRefused(
            f"{suite}/{task}: run is in split {split!r}, not {split_assert!r}. "
            "Contamination here is unrecoverable after the fact and invisible in the output."
        )
    if manifest.get("gold_exposed") or manifest.get("canary_hit"):
        raise SplitRefused(
            f"{suite}/{task}: run {manifest.get('run_id', '')!r} is gold-exposed "
            f"(gold_exposed={bool(manifest.get('gold_exposed'))}, "
            f"canary_hit={bool(manifest.get('canary_hit'))}). A ceiling arm's questions were "
            "chosen where gold is readable; training on them is oracle distillation."
        )


# --------------------------------------------------------------------------- the measurement


def _gold_uids(
    suite_id: str, task_id: str, graph_version: str, manifest: dict[str, Any] | None = None
) -> frozenset[str]:
    """Gold spans of the REQUIRED partition, for one task.

    pi_eval is imported HERE and not at module scope: `pi_run.serve` also hosts /rollout,
    which must never pull gold-reading code into a rollout worker's address space.
    """
    from pi_eval.gold import load_graphs

    graphs = load_graphs(suite_id, graph_version)
    g = graphs.get(task_id)
    if g is None:
        raise NoGoldForTask(
            f"no gold graph for {suite_id}/{task_id} at graph_version={graph_version!r}. "
            "Refused rather than scored: coverage over an empty gold set is NaN, and a NaN "
            "reward is invisible in every statistic that skips it."
        )
    # GOLD MUST DESCRIBE THE CORPUS THIS RUN USED. Gold built against corpus A scoring a run
    # rolled against corpus B has the same task ids and different evidence uids, so every match
    # silently misses: coverage is 0.0 for every episode, and the trainer receives a uniform
    # zero reward indistinguishable from a policy that never retrieves anything. `pi_eval.score`
    # already refuses this (runs_skipped_corpus_mismatch); this server -- the path a GRPO loop
    # actually runs through -- did not check at all. Empty on either side means "unknown", which
    # is not an error: a suite whose corpus is upstream and unhashed must stay scoreable.
    # DIRECTORY AGAINST DIRECTORY, which is the only comparison that can ever be equal.
    #
    # This compared `gold_corpus_hash` against the manifest's `corpus_hash`, and those are two
    # different hash functions over two different inputs: `gold_corpus_hash` is `write_corpus`'s
    # return value and IS the corpus directory name (16 hex), while `corpus_hash` is
    # `pinq.ids.corpus_hash` -- a domain-separated set hash over (doc_id, title, sha256(text))
    # triples (64 hex). Measured: musique gold 38f5afb69fb7ea18 against musique runs
    # c19b505a0ba4fe09..., so EVERY run was flagged mismatched and `collect_rows` returned 0
    # rows with 137 `no_gold` skips -- which zeroes rungs 0, 1 and 2 simultaneously.
    #
    # The eval-side twin at `pi_eval.score` documents this exact mistake in a comment and
    # compares `corpus_dir`. I added this copy of the guard in f4f76e5 and used the wrong field.
    # An empty value on either side still means "unknown" and is allowed.
    want = str(getattr(g, "gold_corpus_hash", "") or "")
    got = str((manifest or {}).get("corpus_dir") or "")
    if want and got and want != got:
        raise NoGoldForTask(
            f"{suite_id}/{task_id}: gold was built against corpus {want[:12]} but this run "
            f"used {got[:12]}. Scoring across that boundary returns 0.0 for every episode -- "
            "indistinguishable from a policy that retrieves nothing -- so it is refused rather "
            "than reported. Rebuild gold for this corpus."
        )

    uids = frozenset(u for n in g.required() for u in n.gold_ev_uids)
    if not uids:
        raise NoGoldForTask(
            f"{suite_id}/{task_id}: the gold graph has no REQUIRED evidence spans, so Q is "
            "undefined for it (0/0). Fix the graph or exclude the task."
        )
    return uids


def _coverage(uids: set[str], gold: set[str]) -> float:
    from pi_eval.metrics.discovery import evidence_coverage

    return evidence_coverage(uids, gold)


def _ledger_totals(rows: Sequence[dict[str, Any]]) -> tuple[float, int]:
    """(hard-currency spend, malformed count). Cumulative, so the last row wins."""
    n_ret = 0.0
    malformed = 0.0
    for r in rows:
        cur = r.get("currency")
        if cur == HARD_CURRENCY:
            n_ret = float(r.get("cumulative", n_ret))
        elif cur == MALFORMED_CURRENCY:
            malformed = float(r.get("cumulative", malformed))
    return n_ret, int(malformed)


def scorer_hash(graph_version: str) -> str:
    """Identity of the MEASUREMENT, not of the run. Joins a stored value to the function."""
    return h("scorer", SCORER_VERSION, graph_version, "coverage/required")


def handle_score(
    req: ScoreRequest,
    *,
    runs_root: str | Path = "runs",
    graph_version: str = DEFAULT_GRAPH_VERSION,
) -> ScoreResponse:
    """Measure one recorded TRAINING episode. The trainer-facing door, and the only wired one.

    Pure over the artifacts plus gold: no clock, no network.
    """
    d = episode_dir(runs_root, req.episode_id)
    manifest = _read_json(d / "manifest.json")
    assert_scorable(manifest, split_assert=req.split_assert)
    return _measure(
        manifest, d, episode_id=req.episode_id, mode=req.mode, graph_version=graph_version
    )


def assert_heldout(manifest: dict[str, Any], *, split: str) -> None:
    """The held-out twin of `assert_scorable`. Raises `SplitRefused`, never returns a bool.

    THE THRESHOLD IS STILL NOT THE CALLER'S TO INVENT. `assert_scorable` fixes it at "train"
    because a firewall whose threshold the request supplies is not a firewall. Here the caller is
    server-side code with a reviewer attached -- which is the remedy that docstring names -- but
    "server-side" is not "unbounded": only the two HELD-OUT splits exist here. `train` belongs to
    `assert_scorable`, where the refusal that keeps a trainer off held-out data lives, and letting
    this function answer for it would put two guards on one question.

    Everything else is refused for exactly the reasons `assert_scorable` gives, unchanged by the
    split: an eval-only suite is the zero-shot transfer target whatever bucket it hashed into, and
    a ceiling arm chose its questions where gold is readable whether or not the task is held out.

    The split compared against is the one STAMPED IN THE RUN'S OWN MANIFEST, for the same reason
    the train side reads the stamp: recomputing it here would let a later edit to the bucket
    function silently reclassify finished runs.
    """
    if split not in ("dev", "test"):
        raise SplitRefused(
            f"split={split!r} is not a held-out split. This entry point exists to score dev and "
            "test episodes for model selection and reporting; a training episode goes through "
            "`handle_score`, which is where the trainer-facing refusal lives."
        )
    suite = str(manifest.get("suite_id", ""))
    task = str(manifest.get("task_id", ""))
    if suite in EVAL_ONLY_SUITES:
        raise SplitRefused(
            f"{suite}/{task}: {suite} is eval-only by construction and is scored through the "
            "transfer benchmark's own evaluator, never through this one."
        )
    stamped = str(manifest.get("split", ""))
    if stamped != split:
        raise SplitRefused(
            f"{suite}/{task}: run is in split {stamped!r}, not {split!r}. Selecting a checkpoint "
            "on the split it is later reported on is unrecoverable after the fact and invisible "
            "in the output."
        )
    if manifest.get("gold_exposed") or manifest.get("canary_hit"):
        raise SplitRefused(
            f"{suite}/{task}: run {manifest.get('run_id', '')!r} is gold-exposed "
            f"(gold_exposed={bool(manifest.get('gold_exposed'))}, "
            f"canary_hit={bool(manifest.get('canary_hit'))}). A ceiling arm's questions were "
            "chosen where gold is readable; measuring it as a held-out baseline is a ceiling "
            "reported as a policy."
        )


def score_heldout(
    req: ScoreRequest,
    *,
    split: Literal["dev", "test"],
    runs_root: str | Path = "runs",
    graph_version: str = DEFAULT_GRAPH_VERSION,
) -> ScoreResponse:
    """Measure one recorded HELD-OUT episode. Deliberately not reachable over the wire.

    WHY THIS EXISTS. 1,130 dev rollouts of the prompted policy and 816 dev forks are on disk and
    no exporter can see them, because the scorer refuses anything that is not `train` -- correctly.
    Without a dev export there is no dev NLL, no STOP 2x2 and no dev pair accuracy, so model
    selection would have to happen on the test split, which voids every confirmatory table.

    WHY IT IS A FUNCTION AND NOT A REQUEST FIELD. That was the design that failed:
    `assert_scorable` once compared the run's split against `req.split_assert`, so a body reading
    `{"split_assert": "test"}` was honoured end to end. Its docstring states the only acceptable
    remedy -- "a server-side configuration change with a reviewer attached, not a request field" --
    and this is that, taken literally. `pi_run.serve.app` does not import this name, does not route
    to it, and `tests/test_heldout_export.py` asserts both by AST. A trainer holding a socket
    cannot reach it; a gold-side CLI holding an import can.

    `split` is a KEYWORD with no default. A default would make the held-out door openable by
    forgetting an argument, which is how every guard of this shape eventually fails.
    """
    d = episode_dir(runs_root, req.episode_id)
    manifest = _read_json(d / "manifest.json")
    assert_heldout(manifest, split=split)
    return _measure(
        manifest, d, episode_id=req.episode_id, mode=req.mode, graph_version=graph_version
    )


def _measure(
    manifest: dict[str, Any],
    d: Path,
    *,
    episode_id: str,
    mode: str,
    graph_version: str,
) -> ScoreResponse:
    """THE measurement. One body, so a dev number and a train number are the same instrument.

    Extracted from `handle_score` when the held-out door was added. Two copies would drift, and
    the dev value that SELECTS a checkpoint would quietly stop being comparable with the train
    value that produced it -- both still floats in the right range, with nothing to notice.

    It takes `episode_id` and `mode` rather than the request, so it is visible at a glance that
    `split_assert` plays no part in the measurement: the split decides WHETHER to measure, never
    HOW. Called only after one of the two assertions above has passed.
    """
    suite_id = str(manifest["suite_id"])
    task_id = str(manifest["task_id"])
    gold = set(_gold_uids(suite_id, task_id, graph_version, manifest))

    turns = sorted(_read_jsonl(d / "turns.jsonl"), key=lambda t: int(t["turn_idx"]))
    ledger = _read_jsonl(d / "ledger.jsonl")
    status = _read_json(d / "status.json") if (d / "status.json").exists() else {}

    # Phi at each prefix k = 0..K, over RETRIEVED EvidenceUnits only. Monotone by
    # construction: evidence accumulates, so coverage never decreases, which is what makes
    # the potential differences downstream non-negative without any clipping.
    seen: set[str] = set()
    potential = [_coverage(seen, gold)]
    for t in turns:
        seen |= set(t.get("retrieved_uids") or ())
        potential.append(_coverage(seen, gold))

    all_uids = frozenset(seen)
    base = potential[-1]
    turn_scores: list[TurnScore] = []
    for t in turns:
        ev = frozenset(t.get("retrieved_uids") or ())
        phi_loo = None if not ev else base - _coverage(set(all_uids - ev), gold)
        turn_scores.append(
            TurnScore(
                turn_idx=int(t["turn_idx"]),
                n_retrieved=len(ev),
                n_new=len(t.get("new_uids") or ()),
                phi_loo=phi_loo,
                qid=str(t.get("question_id", "")),
            )
        )

    n_ret, malformed = _ledger_totals(ledger)
    usage = status.get("usage") or {}
    stop_reason = str(status.get("stop_reason", manifest.get("stop_reason", "")))

    resp = ScoreResponse(
        episode_id=episode_id,
        split=str(manifest.get("split", "")),
        ok=True,
        q_terminal=base,
        q_ladder=tuple(potential),
        potential=tuple(potential),
        turns=tuple(turn_scores),
        stopped=stop_reason == "policy_stop",
        stop_reason=stop_reason,
        n_ret=n_ret,
        tok_total=int(usage.get("tok_total", 0) or 0),
        wall_ms=int(status.get("wall_ms", 0) or 0),
        n_malformed=malformed,
        scorer_hash=scorer_hash(graph_version),
        graph_version=graph_version,
    )

    if mode == "terminal":
        return replace(resp, q_ladder=(potential[0], potential[-1]))
    if mode in ("prefix_ladder", "loo"):
        return resp
    # stop_probe forces one question PAST k_hat and re-measures. That is a new rollout, not a
    # re-read of a finished one, so it is refused rather than approximated: a fabricated
    # probe would average into the stopping table indistinguishably from a real one.
    return replace(
        resp,
        supported=False,
        note=(
            "mode='stop_probe' requires a forced-continue ROLLOUT (counterfactual_kind="
            "'forced_continue'), which this endpoint cannot synthesise from stored artifacts. "
            "Issue it through POST /rollout and score the resulting episode."
        ),
    )


# --------------------------------------------------------------------------- CLI


def main(argv: Sequence[str] | None = None) -> int:
    """`python -m pi_run.serve.score <run_id>` — the refusal, observable as an exit code.

    This exists so the split guarantee is checkable from a shell script and from CI without
    standing a server up. Exit 3 means refused; exit 0 means scored.
    """
    p = argparse.ArgumentParser(prog="pi-score-episode")
    p.add_argument("episode_id")
    p.add_argument("--runs-root", default="runs")
    p.add_argument("--mode", default="terminal")
    p.add_argument("--graph-version", default=DEFAULT_GRAPH_VERSION)
    p.add_argument("--split-assert", default="train")
    a = p.parse_args(argv)
    req = ScoreRequest(
        episode_id=a.episode_id,
        mode=a.mode,  # type: ignore[arg-type]
        split_assert=a.split_assert,  # type: ignore[arg-type]
    )
    try:
        resp = handle_score(req, runs_root=a.runs_root, graph_version=a.graph_version)
    except SplitRefused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return SPLIT_REFUSED_EXIT
    except (EpisodeNotFound, NoGoldForTask, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(to_dict(resp), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
