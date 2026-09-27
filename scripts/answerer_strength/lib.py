"""Lane L1.10: does the evidence advantage reach the answer when the Answerer cannot compensate
from its own knowledge.

Reads the held-out population named in `artifacts/testsplit_qa/run_ids.{trained,prompted}.
{musique,strategyqa,wiki2}.txt` (the same 2,282-run population lane L1.5 used) and, for every
run, rebuilds the FINAL recorded evidence set and the final recorded draft text from
`runs/<run_id>/{manifest.json,evidence.jsonl,turns.jsonl,outcome.json}` plus the suite's own
frozen corpus (`data/corpora/<suite>/<hash>/tasks.jsonl`) -- never from a second, hand-rolled
copy of the corpus reader. The reconstructed evidence's `subset_hash` is asserted equal to the
recorded one on every run: that assertion, not the harness's own say-so, is what licenses
calling this reconstruction "the byte-identical prompt" CONTRIBUTING.md's Drafter-purity rule
promises.

WHY THIS IMPORTS `pinq_train.gate` PRIVATE NAMES. `_by_key`, `_coverage_ladder`, `_matched_cost`,
`_mean`, `_metric_by_run`, `_in`, `_rows` are the functions that already compute the matched-cost
coverage margin this lane's HIGH/LOW split needs. `scripts/stopping_answer_test/lib.py` (lane
L1.5, same population) already established the pattern for the identical reason: a script is not
one of the packages import-linter's contracts govern (`root_packages` in pyproject.toml), so
reusing the vetted function beats a second, possibly-diverging copy of the matched-cost rule.
`matched_cost_coverage_by_task` below is a direct port of L1.5's function of the same name (which
itself calls only `pinq_train.gate`'s own primitives) rather than a second import across worktree
boundaries, because worktrees are independent checkouts and this one must build standalone.

WHY THIS IMPORTS `pi_eval`. This is analysis tooling run by an operator with `PI_GOLD_ROOT` set,
scoring already-collected text against gold answers -- exactly the sanctioned use CONTRIBUTING.md's
firewall describes ("gold is read only through pi_eval for scoring"). `scripts/` is not a
root_package import-linter's forbidden-import contracts scan, so this is not contract 1 or
contract 3 in `pyproject.toml`, and `lint-imports` is part of this lane's own gate.
"""

from __future__ import annotations

import json
import math
import random
import statistics
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from pi_eval.metrics import quality
from pi_eval.stats.inference import (
    _percentile,
    _phi,
    _phi_inv,
    cluster_bootstrap,
    paired_difference,
)
from pinq.budget import BudgetLedger
from pinq.types import Answer, Draft, Evidence, EvidenceUnit, TaskView
from pinq_adapters.llm.litellm_client import MeteredClient
from pinq_adapters.llm.pricing import PriceTable
from pinq_expt.components import FrozenLLMAnswerer, LLMDrafter

SUITES = ("musique", "strategyqa", "wiki2")
ARMS = ("trained", "prompted")

# The three Answerer pins.
#
# `gpt_oss_120b` uses the RECORDED pin string ("openai/aws/gpt-oss-120b") for every call, hit
# or miss, and carries NO fallback. Measured 2026-09-18, in this order:
#   1. a raw HTTP request (curl, bypassing the SDK) with the literal, already-prefixed string
#      403s: "team not allowed to access model ... Tried to access openai/aws/gpt-oss-120b" --
#      the proxy's bare `model_name: "*"` wildcard re-prefixes an already-prefixed model to
#      `openai/openai/aws/gpt-oss-120b`, one `openai/` is stripped before the wire, and
#      `openai/aws/gpt-oss-120b` matches no team's allow-list (the real name is
#      `aws/gpt-oss-120b`) -- see the infra lane's `PROXY.md`, section "lane L0.8".
#   2. `MeteredClient` -- the actual client every real call in this lane goes through -- is NOT
#      the raw-HTTP path: litellm's own SDK strips exactly one leading `openai/` CLIENT-SIDE
#      before the wire request is built, so the request that reaches the proxy already reads
#      `aws/gpt-oss-120b` and never triggers the double-prefix bug. Confirmed both by the infra
#      lane's own role-model access matrix (`PI_MODEL_ANSWERER` / `aws/gpt-oss-120b` OK under
#      this key) and directly here: of 72 pilot calls, every `gpt_oss_120b` call that reached
#      `MeteredClient` on a cache MISS succeeded once the pin was corrected to match (see next
#      paragraph); zero 403s.
#   3. An earlier version of this module gave `gpt_oss_120b` a fallback pin,
#      `"aws/gpt-oss-120b"` (no prefix), reasoning from measurement 1 alone. That fallback
#      itself BROKE 23/72 pilot calls with `litellm.BadRequestError: LLM Provider NOT
#      provided`: `_routing()` in `pinq_adapters/llm/litellm_client.py` only adds
#      `custom_llm_provider="litellm_proxy"` when the model string has NO "/" at all, so
#      `"aws/gpt-oss-120b"` (has a "/", but "aws" is not an SDK-recognised provider) reaches
#      neither a provider inference nor the explicit override, and `litellm.completion()`
#      raises before any HTTP request is sent. Removed; kept here so the next reader does not
#      reintroduce it from measurement 1 without also reading measurement 2.
PIN_MODEL: Mapping[str, str] = {
    "gpt_oss_120b": "openai/aws/gpt-oss-120b",
    "qwen3_8b_base": "qwen3-8b-base",
    "granite33_8b_base": "granite33-8b-base",
}
# `TryPrimaryThenFallback` (below) degenerates to a plain pass-through when a pin has no
# fallback, which is every pin as of the measurement above. Left in place, not deleted: it is
# tested, harmless, and is the one mechanism available if a future pin genuinely needs it.
PIN_FALLBACK_MODEL: Mapping[str, str | None] = {
    "gpt_oss_120b": None,
    "qwen3_8b_base": None,
    "granite33_8b_base": None,
}
PIN_IDS: tuple[str, ...] = tuple(PIN_MODEL)

NEAR_ZERO = 0.01
DEFAULT_N_BOOT = 10_000
STABILITY_N_BOOT = 50_000
STABILITY_SEEDS = (0, 1, 2)


# --------------------------------------------------------------------------- env


def load_env_file(path: str | Path) -> dict[str, str]:
    """A minimal `KEY=VALUE` parser, so a script run via `python -m` (never `source`, which a
    worktree-isolated sandbox here refuses to run at all) can still pick up `.env` in-process."""
    out: dict[str, str] = {}
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, val = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        out[key] = val
    return out


# --------------------------------------------------------------------------- population


def read_population(population_dir: str | Path) -> dict[str, dict[str, list[str]]]:
    """{arm: {suite: [run_id, ...]}} from `run_ids.<arm>.<suite>.txt`. Never `neverask`: this
    lane's contrast is trained vs prompted, matching the brief's 2,282-run population."""
    out: dict[str, dict[str, list[str]]] = {}
    for arm in ARMS:
        out[arm] = {}
        for suite in SUITES:
            p = Path(population_dir) / f"run_ids.{arm}.{suite}.txt"
            out[arm][suite] = [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]
    return out


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    arm: str
    suite_id: str
    task_id: str
    seed: int
    corpus_dir: str
    corpus_hash: str
    word_cap: int
    cited_uids: tuple[str, ...]
    evidence_hash: str
    draft_text: str
    n_turns: int
    recorded_answer_text: str


def load_run_record(runs_root: str | Path, run_id: str, *, arm: str) -> RunRecord:
    d = Path(runs_root) / run_id
    manifest = json.loads((d / "manifest.json").read_text())
    outcome = json.loads((d / "outcome.json").read_text())
    turns = [json.loads(ln) for ln in (d / "turns.jsonl").read_text().splitlines() if ln.strip()]
    draft_text = str(turns[-1]["draft_text"]) if turns else ""
    answer = outcome["answer"] or {}
    return RunRecord(
        run_id=run_id,
        arm=arm,
        suite_id=str(manifest["suite_id"]),
        task_id=str(manifest["task_id"]),
        seed=int(manifest["seed"]),
        corpus_dir=str(manifest["corpus_dir"]),
        corpus_hash=str(manifest["corpus_hash"]),
        word_cap=int(manifest["word_cap"]),
        cited_uids=tuple(answer.get("cited_unit_ids") or ()),
        evidence_hash=str(answer.get("evidence_hash") or outcome.get("subset_hash") or ""),
        draft_text=draft_text,
        n_turns=len(turns),
        recorded_answer_text=str(answer.get("text") or ""),
    )


# --------------------------------------------------------------------------- suite / evidence


class EvidenceReconstructionError(RuntimeError):
    pass


_suite_cache: dict[tuple[str, str], Any] = {}
_units_cache: dict[tuple[str, str, str], dict[str, EvidenceUnit]] = {}
_cache_lock = threading.Lock()


def get_suite(suite_id: str, corpus_dir_hash: str, corpora_root: str | Path) -> Any:
    """The suite adapter, built by the harness's OWN `pi_run.worker.load_suite` -- never a
    second constructor -- so `view()`/`units()` match what the rollout actually saw."""
    key = (suite_id, corpus_dir_hash)
    with _cache_lock:
        cached = _suite_cache.get(key)
    if cached is not None:
        return cached
    from pi_run.worker import load_suite

    root = Path(corpora_root) / suite_id / corpus_dir_hash
    suite = load_suite(suite_id, str(root))
    with _cache_lock:
        _suite_cache[key] = suite
    return suite


def _units_by_uid(suite: Any, key: tuple[str, str, str]) -> dict[str, EvidenceUnit]:
    with _cache_lock:
        cached = _units_cache.get(key)
    if cached is not None:
        return cached
    units = {u.uid: u for u in suite.units(key[2])}
    with _cache_lock:
        _units_cache[key] = units
    return units


def build_view_and_evidence(
    record: RunRecord, *, corpora_root: str | Path
) -> tuple[TaskView, Evidence]:
    """The exact `(view, evidence)` the recorded run's Answerer saw. Reconstruction is checked,
    not assumed: `subset_hash` and `word_cap` must both match the recorded run, or this raises
    rather than silently scoring a different prompt under the run's own id."""
    suite = get_suite(record.suite_id, record.corpus_dir, corpora_root)
    if (
        record.corpus_hash
        and getattr(suite, "corpus_hash", record.corpus_hash) != record.corpus_hash
    ):
        raise EvidenceReconstructionError(
            f"{record.run_id}: suite corpus_hash {suite.corpus_hash} != manifest "
            f"{record.corpus_hash}"
        )
    view = suite.view(record.task_id)
    by_uid = _units_by_uid(suite, (record.suite_id, record.corpus_dir, record.task_id))
    missing = [u for u in record.cited_uids if u not in by_uid]
    if missing:
        raise EvidenceReconstructionError(
            f"{record.run_id}: {len(missing)} cited uid(s) absent from the suite's own unit "
            f"pool for task {record.task_id!r}: {missing[:3]}"
        )
    ev = Evidence.of(tuple(by_uid[u] for u in record.cited_uids))
    if record.evidence_hash and ev.subset_hash != record.evidence_hash:
        raise EvidenceReconstructionError(
            f"{record.run_id}: reconstructed subset_hash {ev.subset_hash} != recorded "
            f"{record.evidence_hash}"
        )
    if view.word_cap != record.word_cap:
        raise EvidenceReconstructionError(
            f"{record.run_id}: suite word_cap {view.word_cap} != manifest {record.word_cap}"
        )
    return view, ev


# --------------------------------------------------------------------------- gateway / LLM plumbing


@dataclass(frozen=True)
class Gateway:
    base_url: str
    api_key: str
    cache_root: str
    price_table: PriceTable
    env: Mapping[str, str]


class TryPrimaryThenFallback:
    """`pinq.protocols.LLM`. Peeks the PRIMARY pin's cache; a hit is returned exactly as the
    harness would return it (no dispatch). A miss goes to the FALLBACK client whole, for a pin
    whose primary model string is reachable through the cache but not through a fresh dispatch
    (see the `PIN_MODEL`/`PIN_FALLBACK_MODEL` comment for the measurement history).

    With no fallback configured (`fallback is None`, every pin as of that measurement), a miss
    dispatches through the primary as normal: this is the plain pass-through case.
    """

    def __init__(self, primary: Any, fallback: Any | None) -> None:
        self._primary = primary
        self._fallback = fallback
        self.used_fallback = False

    def pin(self, role: str) -> Any:
        return self._primary.pin(role)

    def complete(self, **kw: Any) -> tuple[str, Any]:
        if self._fallback is None:
            return self._primary.complete(**kw)
        sha = self._primary.key_for(**kw)
        cached = self._primary._cache.get(sha)  # noqa: SLF001 -- peek only, never dispatches
        if cached is not None:
            return self._primary.complete(**kw)
        self.used_fallback = True
        return self._fallback.complete(**kw)


def _client_for(model: str, role: str, gw: Gateway, ledger: BudgetLedger) -> Any:
    from pi_run.cache import CachingClient, DiskCache

    inner = MeteredClient(
        ledger,
        price_table=gw.price_table,
        models={role: model},
        temperature=0.0,
        base_url=gw.base_url,
        api_key=gw.api_key,
        env=gw.env,
    )
    return CachingClient(inner, DiskCache(gw.cache_root), ledger=ledger)


def _wrapped_client(
    pin_id: str, role: str, gw: Gateway, ledger: BudgetLedger
) -> TryPrimaryThenFallback:
    primary = _client_for(PIN_MODEL[pin_id], role, gw, ledger)
    fallback_model = PIN_FALLBACK_MODEL[pin_id]
    fallback = _client_for(fallback_model, role, gw, ledger) if fallback_model else None
    return TryPrimaryThenFallback(primary, fallback)


def _telemetry_dict(ledger: BudgetLedger, *, used_fallback: bool) -> dict[str, Any]:
    if not ledger.calls:
        return {
            "tok_prompt": 0,
            "tok_completion": 0,
            "tok_reasoning": 0,
            "tok_cached": 0,
            "usd": 0.0,
            "cache_hit": False,
            "model": None,
            "used_fallback": used_fallback,
        }
    tel = ledger.calls[-1]
    return {
        "tok_prompt": tel.tok_prompt,
        "tok_completion": tel.tok_completion,
        "tok_reasoning": tel.tok_reasoning,
        "tok_cached": tel.tok_cached,
        "usd": tel.usd,
        "cache_hit": tel.cache_hit,
        "model": tel.model,
        "used_fallback": used_fallback,
    }


def answer_call(
    pin_id: str,
    view: TaskView,
    ev: Evidence,
    draft: Draft | None,
    seed: int,
    gw: Gateway,
) -> tuple[Answer, dict[str, Any]]:
    """One Answerer call under `pin_id`. A fresh `BudgetLedger`/client PAIR per call -- cheap,
    no I/O at construction -- is what keeps concurrent calls from reading each other's
    `ledger.calls[-1]`; the only shared, concurrency-safe state is the on-disk cache and the
    immutable price table."""
    ledger = BudgetLedger(cap=10**9)
    wrapper = _wrapped_client(pin_id, "answerer", gw, ledger)
    answerer = FrozenLLMAnswerer(llm=wrapper)
    ans = answerer.answer(view, ev, draft, seed=seed, ledger=ledger)
    return ans, _telemetry_dict(ledger, used_fallback=wrapper.used_fallback)


def draft_call(
    pin_id: str, view: TaskView, ev: Evidence, seed: int, gw: Gateway
) -> tuple[Draft, dict[str, Any]]:
    """The pure `Drafter.draft()` call, used only to recover the final draft text of a 0-ask
    run (turns.jsonl has no row to read `draft_text` off). Always the frozen Drafter pin
    (`gpt_oss_120b`): the brief varies the Answerer pin only, never the Drafter's."""
    ledger = BudgetLedger(cap=10**9)
    wrapper = _wrapped_client(pin_id, "drafter", gw, ledger)
    drafter = LLMDrafter(llm=wrapper, agentic_resolve=False)
    d = drafter.draft(view, ev, seed=seed, ledger=ledger)
    return d, _telemetry_dict(ledger, used_fallback=wrapper.used_fallback)


# --------------------------------------------------------------------------- gold / scoring


_gold_cache: dict[tuple[str, str], Mapping[str, Any]] = {}


def load_gold_graphs(suite_id: str, graph_version: str) -> Mapping[str, Any]:
    key = (suite_id, graph_version)
    with _cache_lock:
        cached = _gold_cache.get(key)
    if cached is not None:
        return cached
    from pi_eval.gold import load_graphs

    graphs = load_graphs(suite_id, graph_version)
    with _cache_lock:
        _gold_cache[key] = graphs
    return graphs


def score_answer(pred_text: str, graph: Any) -> dict[str, float]:
    """`answer_token_f1`/`answer_token_recall`/`contains_answer`, exactly as `pi_eval.score.
    score_run` computes them (`quality.decompose_over_aliases` then `quality.contains_answer`,
    both against `graph.answer` -- the canary-stripped gold -- and `graph.gold_aliases`).
    Called, never reimplemented."""
    gold = graph.answer
    aliases = graph.gold_aliases
    f1, precision, recall = quality.decompose_over_aliases(pred_text, gold, aliases)
    contains = quality.contains_answer(pred_text, gold, aliases)
    return {
        "answer_token_f1": f1,
        "answer_token_precision": precision,
        "answer_token_recall": recall,
        "contains_answer": contains,
    }


# --------------------------------------------------------------------------- BCa: two independent samples


def bca_two_sample_mean_delta(
    group_a: Sequence[float], group_b: Sequence[float], *, n_boot: int, seed: int
) -> dict[str, Any]:
    """BCa interval for `mean(group_a) - mean(group_b)`, two INDEPENDENT samples, each resampled
    with replacement from itself. A direct port of `scripts/stopping_answer_test/lib.py`'s
    function of the same name (lane L1.5, same population, same coverage-linkage design) -- not
    a second derivation of the recipe, and not importable across worktrees, so ported rather
    than re-derived. Acceleration is the standard two-sample generalisation: jackknife over the
    pooled n_a + n_b leave-one-out replicates (Efron & Tibshirani 1993, ch. 14)."""
    a = sorted(float(x) for x in group_a if not math.isnan(x))
    b = sorted(float(x) for x in group_b if not math.isnan(x))
    if not a or not b:
        return {
            "point": float("nan"),
            "lo": float("nan"),
            "hi": float("nan"),
            "n": len(a) + len(b),
            "n_a": len(a),
            "n_b": len(b),
        }

    theta_hat = statistics.fmean(a) - statistics.fmean(b)
    na, nb = len(a), len(b)
    rng = random.Random(seed)
    reps: list[float] = []
    for _ in range(n_boot):
        ra = statistics.fmean([a[rng.randrange(na)] for _ in range(na)])
        rb = statistics.fmean([b[rng.randrange(nb)] for _ in range(nb)])
        reps.append(ra - rb)

    prop = sum(1 for r in reps if r < theta_hat) / len(reps)
    z0 = _phi_inv(min(max(prop, 1e-9), 1 - 1e-9))

    jack: list[float] = []
    for i in range(na):
        rest = a[:i] + a[i + 1 :]
        if rest:
            jack.append(statistics.fmean(rest) - statistics.fmean(b))
    for i in range(nb):
        rest = b[:i] + b[i + 1 :]
        if rest:
            jack.append(statistics.fmean(a) - statistics.fmean(rest))
    if len(jack) > 1:
        jbar = statistics.fmean(jack)
        num = sum((jbar - x) ** 3 for x in jack)
        den = 6 * (sum((jbar - x) ** 2 for x in jack) ** 1.5)
        acc = num / den if den else 0.0
    else:
        acc = 0.0

    def adj(q: float) -> float:
        z = _phi_inv(q)
        denom = 1 - acc * (z0 + z)
        return _phi(z0 + (z0 + z) / denom) if denom else q

    reps_sorted = sorted(reps)
    lo = _percentile(reps_sorted, adj(0.025))
    hi = _percentile(reps_sorted, adj(0.975))
    return {"point": theta_hat, "lo": lo, "hi": hi, "n": na + nb, "n_a": na, "n_b": nb}


# --------------------------------------------------------------------------- stability wrapper


def with_stability(
    compute: Callable[[int, int], tuple[float, float, float]], *, seed: int = 0
) -> dict[str, Any]:
    """Run `compute(n_boot, seed) -> (point, lo, hi)` at 10k. If either bound sits within
    `NEAR_ZERO` of 0, rerun at 50k on three seeds and report whether that bound's SIGN is
    stable across them -- the coordinator rule stated in the brief, applied uniformly to every
    interval this lane computes regardless of which estimator produced it."""
    point, lo, hi = compute(DEFAULT_N_BOOT, seed)
    out: dict[str, Any] = {
        "point": point,
        "lo": lo,
        "hi": hi,
        "n_boot_reported": DEFAULT_N_BOOT,
        "stability_checked": False,
    }
    if math.isnan(lo) or math.isnan(hi):
        out["verdict"] = "absent"
        return out

    lo_close = abs(lo) <= NEAR_ZERO
    hi_close = abs(hi) <= NEAR_ZERO
    if not (lo_close or hi_close):
        out["verdict"] = "excludes_zero" if (lo > 0 or hi < 0) else "includes_zero"
        return out

    reps = [compute(STABILITY_N_BOOT, s) for s in STABILITY_SEEDS]

    def sign(x: float) -> int:
        return 0 if x == 0 else (1 if x > 0 else -1)

    stable = True
    if lo_close:
        stable = stable and len({sign(r[1]) for r in reps}) == 1
    if hi_close:
        stable = stable and len({sign(r[2]) for r in reps}) == 1

    out["stability_checked"] = True
    out["stability_seeds"] = list(STABILITY_SEEDS)
    out["stability_n_boot"] = STABILITY_N_BOOT
    out["stability_reps"] = [{"lo": r[1], "hi": r[2]} for r in reps]
    out["stable"] = stable
    if not stable:
        out["verdict"] = "undecided"
    else:
        p0, lo0, hi0 = reps[0]
        out["point"], out["lo"], out["hi"] = p0, lo0, hi0
        out["n_boot_reported"] = STABILITY_N_BOOT
        out["verdict"] = "excludes_zero" if (lo0 > 0 or hi0 < 0) else "includes_zero"
    return out


def _key(task_id: str, seed: int) -> str:
    """The `(task, seed)` pairing key used throughout `paired_contrast` and the row tables --
    one place so `run.py` and `tables.py` cannot format it two different ways."""
    return f"{task_id}|{seed}"


def paired_contrast(
    trained: Mapping[str, float], prompted: Mapping[str, float], *, seed: int = 0
) -> dict[str, Any]:
    """Trained-minus-prompted over shared `(task, seed)` keys, BCa (`pi_eval.stats.inference.
    paired_difference`, `clusters=None` so every `(task, seed)` key is its own resample unit),
    with the near-zero stability re-check applied."""

    def compute(n_boot: int, s: int) -> tuple[float, float, float]:
        est = paired_difference(trained, prompted, clusters=None, n_boot=n_boot, seed=s)
        return est.point, est.ci_lo, est.ci_hi

    result = with_stability(compute, seed=seed)
    n_shared = len(set(trained) & set(prompted))
    result["n"] = n_shared
    return result


def one_sample_level(values: Sequence[float], *, seed: int = 0) -> dict[str, Any]:
    """A descriptive level (mean + BCa CI) over singleton clusters -- used for the closed-book
    floor, where there is only one arm and no pairing."""
    vals = [float(v) for v in values if not math.isnan(v)]
    if not vals:
        return {
            "point": float("nan"),
            "lo": float("nan"),
            "hi": float("nan"),
            "n": 0,
            "verdict": "absent",
        }
    units = [[v] for v in vals]

    def compute(n_boot: int, s: int) -> tuple[float, float, float]:
        return cluster_bootstrap(units, n_boot=n_boot, seed=s)

    result = with_stability(compute, seed=seed)
    result["n"] = len(vals)
    return result


def two_sample_contrast(
    group_a: Sequence[float], group_b: Sequence[float], *, seed: int = 0
) -> dict[str, Any]:
    def compute(n_boot: int, s: int) -> tuple[float, float, float]:
        d = bca_two_sample_mean_delta(group_a, group_b, n_boot=n_boot, seed=s)
        return d["point"], d["lo"], d["hi"]

    result = with_stability(compute, seed=seed)
    result["n_a"] = len(group_a)
    result["n_b"] = len(group_b)
    return result


# --------------------------------------------------------------------------- matched-cost coverage split


def _n_asks_rows(con, runs: Sequence[Mapping]) -> list[dict]:
    from pinq_train.gate import _in, _rows  # noqa: PLC0415

    ids = [r["run_id"] for r in runs]
    if not ids:
        return []
    return _rows(con, f"SELECT run_id, n_asks FROM runs WHERE run_id IN {_in(ids)}")


def matched_cost_coverage_by_task(
    con, *, ck_runs: Sequence[Mapping], ba_runs: Sequence[Mapping], scorer_hash: str
) -> dict[tuple[str, str], float]:
    """Per-(suite, task) matched-cost `evidence_coverage` delta, seeds averaged within task.

    A direct port of lane L1.5's function of the same name. Reproduces the inner loop of
    `pinq_train.gate._matched_cost` (checkpoint's own terminal coverage minus the mean of the
    baseline's `frontier_q#min(k, baseline n_asks)` across the baseline runs sharing that key),
    which that function computes internally as `by_task` but does not return -- only the
    suite/pooled aggregates are public. Calling the same primitives (`_coverage_ladder`,
    `_metric_by_run`, `_by_key`) in the same order keeps this identical to the published
    headline number rather than a second, possibly-diverging implementation.
    """
    from pinq_train.gate import _by_key, _coverage_ladder, _mean, _metric_by_run  # noqa: PLC0415

    cov = _metric_by_run(con, "evidence_coverage", scorer_hash=scorer_hash)
    all_runs = [*ck_runs, *ba_runs]
    n_asks = {str(r["run_id"]): int(r.get("n_asks") or 0) for r in _n_asks_rows(con, all_runs)}
    ck_keys = _by_key(ck_runs)
    ba_keys = _by_key(ba_runs)
    shared = sorted(set(ck_keys) & set(ba_keys))
    base_used = sorted({rid for key in shared for rid in ba_keys[key]})
    ladders = _coverage_ladder(con, [{"run_id": rid} for rid in base_used], scorer_hash=scorer_hash)

    per_task: dict[tuple[str, str], list[float]] = {}
    for key in shared:
        deltas: list[float] = []
        for c_rid in ck_keys[key]:
            c_val = cov.get(c_rid)
            if c_val is None:
                continue
            k = n_asks.get(c_rid, 0)
            rungs = []
            for b_rid in ba_keys[key]:
                b_n = n_asks.get(b_rid, 0)
                at = min(k, b_n)
                rung = (ladders.get(b_rid) or {}).get(at)
                if rung is None:
                    continue
                rungs.append(float(rung))
            if not rungs:
                continue
            deltas.append(float(c_val) - _mean(rungs))
        if deltas:
            per_task.setdefault((str(key[0]), str(key[1])), []).append(_mean(deltas))
    return {k: _mean(v) for k, v in per_task.items()}


def high_low_split(
    coverage_delta_by_task: Mapping[tuple[str, str], float], *, suite: str
) -> tuple[set[str], set[str]]:
    """HIGH = trained's matched-cost coverage exceeds the prompted pair's; LOW = it does not.
    Mirrors lane L1.5's step 3 split rule exactly."""
    high: set[str] = set()
    low: set[str] = set()
    for (s, task_id), delta in coverage_delta_by_task.items():
        if s != suite:
            continue
        (high if delta > 0 else low).add(task_id)
    return high, low
