# Reproducing the numbers

## What "reproducible" means here, precisely

**Cache-replayable, not bitwise deterministic.** Temperature 0 is not bitwise deterministic
under batched inference — two identical requests to the same provider can differ. So the
reproducibility artifact is the **content-addressed request cache**: given it, every number
in the paper regenerates with no network and no API keys. Without it you will get *similar*
numbers, not identical ones, and the README says so rather than implying otherwise.

**Check it, do not assume it.** `pi cache verify` joins calls.parquet to the cache and
reports, per recorded call, whether the cache still holds the response the run received:

```bash
.venv/bin/pi cache verify        # exit 1 if any call diverged
```

Measured 2026-08-29: **41 diverged of 11,681**, `missing_from_cache: 0` — disagreement, not
absence. Cause: two workers issuing byte-identical payloads concurrently get different text
(see the paragraph above), the cache keeps the first, and the loser used to return its own
answer anyway — so those calls replay to a different response than the run recorded. Fixed
at the source in 95d6524: a losing racer now adopts the canonical text and response_sha, and
the event is counted per run as `cache_races`.

**Runs recorded before that fix keep their divergence**, because the artifact says what
happened rather than what we wish had happened. Concretely: 41 of 11,681 calls (0.35%),
touching 14 of the 300 P3 runs. Those runs' published numbers are what was measured — they
come from the recorded artifacts, not from a replay — but replaying *those* runs from the
cache reproduces them only up to those calls. Every run rolled after 95d6524 is exact.

## The three-stage handoff

```
rollout  →  runs/<run_id>/{status,manifest,outcome}.json turns.jsonl calls.jsonl ledger.jsonl
compact  →  scores/parquet/{runs,turns,calls,evidence,env_calls,ledger}.parquet
score    →  scores/parquet/{matches,judgments,scores}.parquet     keyed by scorer_hash
aggregate→  tables/*.tex + figures/*.pdf + provenance.json
```

Re-scoring never re-rolls. `scorer_hash = h(metric_defs, graph_hash, matcher_hash, judge_pins)`,
so swapping a matcher or revising a graph **adds rows under a new hash** and leaves the old
ones byte-identical — which is what makes the matcher/judge-swap audit a `groupby` rather
than a two-week rerun.

## From nothing to a number, with no keys

```bash
make venv
make gate      # ruff, format, import contracts, tests, path guard
make smoke     # synth suite end-to-end: no network, no keys, no dollars
```

## Reproducing a real table

```bash
export TAU2_DATA_DIR=/path/to/tau2-bench/data          # the wheel does NOT ship data/
export PARE_BENCHMARK_SPLITS_DIR=/path/to/pare/data/splits   # nor does PARE's

.venv/bin/pi data fetch --suite musique --verify-sha   # pinned sha256; idempotent
.venv/bin/pi data build --suite musique                # raw -> corpus + gold, hop-stratified
.venv/bin/pi data status                               # what is on disk, per suite
.venv/bin/pi gold stats                                # nodes, edges, depth, provenance

.venv/bin/pi run --sweep conf/grids/tier1_confirmatory.yaml --concurrency 16
.venv/bin/pi compact
PI_JUDGE_CLIENT=pi_run.judge_client:judge \
  .venv/bin/pi score --gold-root "$PWD/data/gold"   # without this, kpr_incremental (P3) has no rows
.venv/bin/pi agg --primary --assert-no-gold-exposed
.venv/bin/pi render --all
```

`--assert-no-gold-exposed` exits non-zero if any oracle/ceiling run or any `dev-` prefixed
run reached a primary table. That is the mechanical guarantee, not a convention.

Every command above exists. That sentence is load-bearing: this file used to document
`pi data fetch` and `pi gold stats` while neither was implemented, so a reader concluded the
pipeline was reproducible and had no way to find out otherwise until they typed it.
`docs/RUNBOOK.md` §2 is the long form, including what each suite can and cannot build.

## tau2 gold, and why `pi data` does not build it

tau2 is **self-sourced**: the 698 knowledge documents and the 97 tasks come from a
`sierra-research/tau2-bench@v1.0.1` checkout through `TAU2_DATA_DIR`, so there is nothing for
`pi data fetch` to fetch and no corpus for this repo to mint. Minting one would give the same
bytes two different `corpus_hash` values and make the adapter's uids and gold's disagree,
which is invisible in every downstream number (it reads as "the policy retrieved nothing
relevant"). The gold graph is therefore built directly:

```bash
export TAU2_DATA_DIR=/path/to/tau2-bench/data
.venv/bin/python -c "import sys; sys.path.insert(0,'src'); \
from pathlib import Path; from pi_eval.build.tau2_build import build; \
print(build(root=Path('.')))"
# -> data/gold/graphs/tau2/v1.jsonl, 97 graphs, corpus_hash == Tau2Suite().corpus_hash
```

The version is `v1`, not `tau2/v1`: the suite is already the directory, and a suite-scoped
version wrote `graphs/tau2/tau2/v1.jsonl` while `pi score --graph-version v1` looked in
`graphs/tau2/v1.jsonl` and skipped every tau2 run as "no graph".

**`template_id` is derived, because upstream ships none.** The tau2 primary endpoint is
clustered at `template_id`, and upstream's only per-task label (`description.purpose`) is the
string `"Task: task_017"` — 97 distinct values, i.e. no clustering at all. `Tau2Suite.
template_id` derives one from the task's `required_documents`: the sorted set of *topic
stems* (the document id minus its `doc_` prefix and its trailing `_NNN` serial, truncated
after the second underscore-delimited component), joined by `+` and prefixed `doc:`. A task
requiring no document falls back to `solo:<task_id>`, its own cluster. Measured on v1.0.1
this yields **18 clusters over 97 tasks** (largest 27, median 3.5). The derivation reads a
gold-bearing field, so it is minted suite-side and travels in the `RunManifest`; it must never
be routed through `view()`, and `make_view` would raise if it were.

## The data stage, in one paragraph

`pi data` and `pi gold` drive the **same** builder pass — a corpus and its gold graph are two
halves of one split, and producing them separately is how the two drift — and differ only in
what they report. `fetch` re-verifies a cached file rather than re-downloading it and prints
the digest it checked *and whether that digest was a hard pin or a trust-on-first-use
sidecar*; there is deliberately **no `--no-verify`**. `build` writes to a content-addressed
corpus directory, so two builds from one raw input land on the same `corpus_hash` and the same
path instead of shadowing each other.

`pi gold mine` drives S0–S7 over recorded rollouts and **refuses a pool too thin to support
the contamination controls** (< 2 generator cells or < 2 model families), emitting the task
with `admissible=false` and the reason rather than dropping it — a systematically-skipped
stratum has to stay visible. `pi gold validate` prints every gate with its written
consequence, and prints `NOT RUN` plus the missing input for the gates whose inputs are human
annotations this repository does not yet have.

## What can go wrong, and how you will know

| Symptom | Meaning |
|---|---|
| `GoldAccessError: PI_GOLD_ROOT is unset` in a worker | **The firewall working.** Rollout processes are not permitted to read gold. |
| `pi verify firewall` exits 1 | A canary nonce appeared in a serialized request. Gold text reached a model; the run is void. There is no benign explanation. |
| `run_id` starts with `dev-` | Dirty working tree. The run is mechanically excluded from every reported table. |
| `reconciled_docs: false` | A Drafter retrieved inside `resolve()` without metering. Budget parity is broken; the arm is not comparable. |
| `purity_violation_rate > 0` | `draft()` is not a pure function of the evidence subset. φ_LOO, the prefix ladder and the stop test are all undefined until it is. |
| `below_noise_floor: true` | The effect is under `2σ̂_J/√n`. It is printed, not claimed. |
