# Runbook — from a clean machine to a paper number

Every command in this file has been run and its output pasted or summarised. Where a step is
not yet possible, it says so and says why, rather than documenting an aspiration.
(`docs/REPRODUCE.md` once documented `pi data fetch` and `pi gold stats`, neither of which
existed; that is the failure mode this file is written against.)

---

## 0. Environment, once

```bash
make venv                       # .[dev,run,analysis] -- everything `make gate` imports
uv pip install -e ".[suites,judge]"   # only if you are building corpora or judging
make gate                       # ruff, format, 4 import contracts, tests, path guard. Must exit 0.
```

`make gate` is the only definition of "working". It is what CI runs, in CI's order. Check its
exit code with `echo $?` — piping it into `tail` throws the status away.

### Credentials

`.env` is gitignored and is the only place secrets live. The client **refuses to default a
model pin**, because defaulting would let two arms silently run on different models and still
be compared.

```dotenv
LITELLM_BASE_URL=https://litellm.gateway.example.com     # VPN-gated
LITELLM_API_KEY=...
PI_MODEL_INQUIRER=openai/aws/gpt-oss-120b
PI_MODEL_DRAFTER=openai/aws/gpt-oss-120b
PI_MODEL_ANSWERER=openai/aws/gpt-oss-120b
PI_MODEL_USERSIM=openai/aws/gpt-oss-120b
PI_MODEL_JUDGE=openai/aws/gpt-oss-120b
```

**No throttle or cap belongs in `.env`.** Both are flags: `pi run --concurrency 16
--spend-cap 80`. A value parked in `.env` is chosen once and then governs silently for weeks
— `PI_MAX_CONCURRENCY=2`, set for one flaky evening, outranked `--concurrency 16` and turned
a 9-hour grid into a 74-hour one. The flags now win, so the vars remain usable as a guard rail
for an unattended run, and `.env` ships them commented out.

```bash
set -a; . ./.env; set +a
.venv/bin/pi env doctor        # live call per pinned model; prints no secrets
```

`pi env doctor` **makes a real call**: one 32-token completion per distinct pinned model,
deduplicated across roles. It exits **2** when any model is unreachable, so it works as a
runbook step rather than as a comment. It previously checked only that an env var was
non-empty — which is equally true of a typo, an expired key, a proxy that is down, and a VPN
that is not connected, i.e. of all four reasons a sweep dies at 02:00. `--no-probe` restores
the offline report; `make gate` uses it.

It also reports the three PREREQUISITE variables (`TAU2_DATA_DIR`,
`PARE_BENCHMARK_SPLITS_DIR`, `DRGYM_API_KEY`) whose absence otherwise surfaces minutes into a
sweep, from inside an adapter constructor.

**The proxy is VPN-gated.** Without the tunnel the host does not resolve. A drop mid-sweep is
survivable: `litellm.APIError` is retryable and the ladder spans ~148 s (three rate-limit
windows). **Resume is the default** — re-issuing the identical command re-runs failed units
and skips successful ones. There is no `--resume` flag; the flag is `--no-resume`, which
re-runs everything.

Resume is per-commit. `code_version` is inside `semantic_hash` is inside `run_id`, so **any
commit changes every run id** and a sweep cannot be resumed across one. Freeze the tree before
a campaign.

### Upstream checkouts that ship no data

Two suites are self-sourced and neither wheel ships a `data/` directory:

```bash
export TAU2_DATA_DIR=/path/to/tau2-bench/data          # 698 documents, 97 tasks (asserted)
export PARE_BENCHMARK_SPLITS_DIR=/path/to/pare/data/splits   # 143 scenarios (asserted)
```

Without them the adapter raises with the remedy in the message, e.g.

```
PareSuiteUnavailable: pare is installed but its data is not:
  .../site-packages/data/splits/full.txt does not exist. The wheel ships no data/ directory.
  Clone deepakn97/pare and export PARE_BENCHMARK_SPLITS_DIR=<checkout>/data/splits.
```

`pinq_adapters.pare.available()` returns that same string, and every live-PARE test is
`@pytest.mark.integration`, so the default `make gate` run neither needs the checkout nor
pretends to have one.

---

## 1. Prove the machine works, with no keys and no money

```bash
make smoke      # build synth -> run 3 arms x 5 tasks -> compact -> verify firewall
```

The `synth` suite constructs its need-graph *first* and derives the corpus from it, so
coverage, depth and per-question value all have closed-form answers. If a metric is wrong,
this is where it shows, in seconds, for nothing.

---

## 2. Data and gold

`pi data` owns raw → corpus + gold for the five buildable suites; `pi gold` owns everything
downstream of the graph. They drive the *same* builder pass — a corpus and its gold graph are
two halves of one split — and differ only in what they report.

```bash
.venv/bin/pi data status [--json]                 # what is on disk, per suite
.venv/bin/pi data fetch --suite musique --verify-sha
.venv/bin/pi data build  --suite strategyqa [--limit N] [--raw-dir DIR]
.venv/bin/pi gold build  --suite strategyqa       # the same pass, reported from the gold side
.venv/bin/pi gold stats                           # nodes, edges, depth, provenance, orphans
```

`pi data status`, run on 2026-08-24 after building all five:

```
suite       raw              corpus                           gold
------------------------------------------------------------------
synth       generated        cb91322cbc8e n=12                v1 g=12 v=108
musique     4 files 519M     934e38adf6ac n=800               v1 g=800 v=2660
strategyqa  1 files 3M       ee8291a32bb7 n=2290              v1 g=2290 v=6720
wiki2       5 files 1G       309cefe9519b n=12576             v1 g=12576 v=31120
drgym       1002 files 6M    18fd59f66a9e n=976, +1 more      v1 g=976 v=16156
------------------------------------------------------------------
pare        not driven by `pi data`: self-sourced from a pare checkout (...)
tau2        not driven by `pi data`: self-sourced from a tau2-bench checkout (...)

raw counts exclude .sha256 sidecars; verified sidecars: musique=3/4, strategyqa=1/1,
wiki2=2/5, drgym=1001/1002
```

The corpus column is capped at one entry; `--json` prints every built corpus. `pi gold stats`
is the same inventory from the graph side (nodes, edges, facets, depth histogram, orphan rate,
provenance) and recomputes depth from the stored edges rather than reading it off the record.

**Every build arms the firewall's fourth layer.** `write_graphs` is the single choke point
every real builder passes through, so it mints a per-task canary nonce and appends it to
`data/canaries/`. Until 2026-08-24 minting existed only in `synth_build`, which meant
`pi verify firewall` scanned the cache against a registry covering the one suite that carries
no real corpus text — it reported "clean" on musique, strategyqa, wiki2, drgym and tau2, i.e.
on every suite where a leak could happen, because there was nothing there to find. The
registry now holds **16,744 nonces across all six suites**, and planting one in a cache entry
makes `pi verify firewall` exit 1 (verified on a *musique* nonce, not a synthetic one).

**The nonce is embedded in `gold_answer`, not only in its own field.** That is what makes the
layer able to fire at all: a nonce sitting in a sibling `gold_canary` column is unreachable,
because no code path that leaks gold would carry that column along with the text it leaked. The
carrier is `gold_answer` specifically because it is the most dangerous string to leak *and* the
only gold string that never legitimately reaches a model — the judges are handed `gold_text`
key points and `pi gold questions` emits `gold_text` plus `gold_aliases`, so a nonce in either
would fire on a sanctioned path, and a detector that cries wolf is a detector that gets
disabled. Use `GoldGraph.answer`, which strips it; the raw field is the exception.

**And it fires at the boundary, not only in a forensic scan afterwards — this is the part that
was wrong for most of a day.** `pi verify firewall` scans the *response cache*, and a cache
record carries the response text plus the request's **sha**, never the request itself (that
minimality is what makes two racing writers produce byte-identical files). A gold answer
interpolated into a **prompt** is therefore written nowhere the scan can see it, so the scan
could only ever catch the rarer event of a model echoing gold back in its reply. An earlier
verification here planted a nonce in the cached *response* and reported success; it was
checking the wrong direction.

`pinq.tripwire` now scans every serialized request inside the client, before dispatch — where
the request still exists, has not been sent and has not been billed. Verified end to end on
**musique**, through the real client and the repository's own 16,751-nonce registry: a prompt
containing a real gold answer raises `CanaryLeak`, and a clean prompt passes straight through.
The forensic cache scan is kept as a second net for the echo case.

It runs on every request of every unit, so the common case is one substring scan for
`PINQCANARY_` and nothing else; only if that appears is anything looked up. `pinq.tripwire` is
stdlib-only and lives in `pinq` rather than `pi_eval`, so no gold-reading code goes anywhere
near the rollout path.

Set `PI_CANARY_SALT` in `.env`. Without it every rebuild mints fresh nonces: the registry is
append-only so nothing stops being scanned, but the build stops being reproducible and the
registry grows without bound. It is secret-ish — knowing the salt lets anyone recompute the
nonces and strip them — which is why it lives in `.env` and `data/canaries/` is gitignored.

Three properties are load-bearing:

- **Idempotent.** `fetch` re-verifies a cached file instead of re-downloading it (a warm
  `--suite musique` fetch is 0.3 s, not 519 MB), and `build` writes to a **content-addressed**
  corpus directory, so building twice from the same raw input yields the same `corpus_hash`
  and the same path rather than a second corpus that shadows the first.
- **It says what it verified**, and with which strength: a **hard pin** (a digest recorded in
  the builder) or a **trust-on-first-use sidecar**. Collapsing those into one "ok" would
  overstate the weaker one.
- **There is no `--no-verify`.** A flag that skips the checksum is a flag that gets passed the
  first time an honest download fails. If upstream genuinely republished, bump the pin and the
  suite version together.

`--verify-sha` is the explicit spelling of the default and changes nothing.

### Retrieval sensitivity: run this before spending anything

```bash
python scripts/probe_retrieval.py --suite musique --n 300     # 0 = PASS, 1 = FAIL, 3 = inapplicable
```

Zero tokens, seconds. It measures the share of tasks holding gold evidence a sub-question
reaches and the full question misses — i.e. the headroom inquiry exists to exploit. Re-measured
2026-08-24 at n=300, on the corpora that will actually run (**bold** = the pinned k):

| suite | k=2 | k=3 | k=5 | k=8 |
|---|---|---|---|---|
| musique | 66% | 68% | **65%** | 63% |
| strategyqa | **45%** | 28% | 23% | 19% |
| wiki2 | 72% | **58%** | 39% | 21% |

**The musique row moved twice, and both moves were the sample rather than the suite.**

| | reading | why |
|---|---|---|
| 60-task corpus | 63/67/67/62 | head-N of a hop-sorted upstream file → entirely 2hop, accidentally easy |
| 800 tasks, as first written | 49/50/47/44 | `_stratified_order` selected round-robin and then `sorted()` the result, so the *file* stayed hop-ordered and every prefix was 2hop-heavy |
| 800 tasks, order preserved | **66/68/65/63** | every prefix hop-balanced — this is the corpus the grids read |

The middle row is the instructive one. The selection was stratified and the written order was
not, and **every consumer takes a prefix**: `task_ids()` returns file order and `pi run --n N`
slices it. Measured with the sort in place, the first 40 ids were 100% 2hop and the first 200
contained no 4-hop task at all — so `tier1_pilot` (n=120) and `tier1_confirmatory` (n=200)
would both have run an effectively 2hop sample on a corpus built precisely to stop that. The
retrieval probe failing at n=40 (20% against a 30% floor) is what surfaced it.

Deeper tasks hold more sub-question-only evidence, which is why removing the 2-hop skew raised
the whole curve. musique's is flat over k ∈ [2,5] (3pp at n=300, SE ≈ 2.7pp), so the pinned
k=5 sits inside the optimum rather than at it, and is kept because `tier1_trained` and
`tier2_confirmatory` must share it — otherwise trained-vs-prompted is also a comparison of
retrieval budget.

**`k` is not a free parameter.** As it grows the full question's top-k absorbs more gold and
leaves less for a sub-question to add, so a single grid-wide `k=5` would make StrategyQA
unmeasurable *while looking like an ordinary null*. `conf/grids/tier1_confirmatory.yaml`
therefore pins `k_by_suite: {musique: 5, strategyqa: 2, wiki2: 3}`.

On `synth` and `tau2` the probe reports **INAPPLICABLE** (exit 3), not FAIL: a gold node there
is a fact or a document title, and the token that actually keys retrieval lives in the
document rather than in the need text.

**MuSiQue selection is hop-stratified, and must stay that way.** The upstream files are
hop-sorted — the first 13,900 train and 1,179 dev records are all 2hop — so head-N truncation
yields an all-2hop sample whose graphs are depth ≤ 1, which makes `coverage_at_depth_ge2`
structurally unmeasurable at any n. A 40-task head-N build had depth histogram `{0:40, 1:40}`:
not one depth-2 node, and nothing in the output said so. `_stratified_order` selects
round-robin across hop classes instead; `tests/test_adapters_musique.py` holds it.

The built corpus is **800 tasks / 2,660 nodes**, depth histogram `{0:1196, 1:800, 2:530,
3:134}`, orphan rate 0.000 — so `coverage_at_depth_ge2` has 664 nodes to be measured on. It
was 60 tasks until 2026-08-24, which `tier1_pilot` (n=120) and `tier1_confirmatory` (n=200)
would both have exhausted; a pilot that consumes its own suite burns the ids it is supposed to
protect.

### What each suite can and cannot build

| suite | `fetch` | `build` | note |
|---|---|---|---|
| synth | nothing to fetch | generated from a seed | the closed-form suite; `make smoke` builds it |
| musique | pinned jsonl + the exclusion-list zip | full split | hop-stratified selection; `--limit` is safe |
| strategyqa | pinned zip | 2,290 tasks / 6,720 nodes / 4,483 edges | ~6 s |
| wiki2 | pinned parquet | dev = 12,576 tasks / 31,120 nodes | depth ≤ 1: `comparison` items are flat by construction |
| drgym | pinned query list | **gold only** | see below |
| tau2 | — | **gold only**, not via `pi data` | self-sourced from a checkout; `pi_eval.build.tau2_build.build()` writes `data/gold/graphs/tau2/v1.jsonl` (97 graphs). See docs/REPRODUCE.md |
| pare | — | — | self-sourced; scenarios, no corpus |

**DRGym builds its gold but not its corpus.** The key points at the pinned commit *are* the
gold and are fetched one file per query (~12 s per 25 files, so ~8 min for all 1,000). The
*corpus* is a hosted open-web index behind an API key: `/search` and `/fineweb/search` both
return 401 without one, so there is no document set to build and `gold_ev_uids` is empty by
construction. Resolution on this suite is judged, never matched by uid.

### The store lock

`scores/parquet` is shared. `pi compact` rewrites its six run-side tables and `pi score`
appends to `scores.parquet` beside them, so two lanes writing it at once do not tear the
files — `os.replace` is atomic per file — they silently discard each other. That is not
hypothetical: on 2026-09-18 two lanes compacted the same directory seventeen seconds apart,
the later batch of renames threw away the earlier lane's whole result, and because the earlier
lane was running a pre-fix compactor the store carried 134,370 turn rows keyed to NULL again
for the twenty minutes in between. Nothing errored.

So both commands REFUSE to write the shared store without the lock, and exit non-zero:

```bash
LOCK="$PWD/scores/parquet/.store.lock"          # or $PI_STORE_LOCK, if you set one
until mkdir "$LOCK" 2>/dev/null; do sleep 30; done
echo "$(date) <your lane> <what you are doing>" > "$LOCK/owner"

.venv/bin/pi compact                             # prints: shared store is locked by ...

rm "$LOCK/owner"; rmdir "$LOCK"
```

`mkdir` is the atomic step; the `owner` file is what makes a held lock nameable, and a lock
directory without one reads as NOT held, so a crashed lane is visible rather than blocking.
The default is `<out>/.store.lock`, beside the store — **not** a per-session scratchpad, because
a lock no other session can name is not a lock, and that is precisely how both lanes above
believed they held one. `PI_STORE_LOCK` overrides the path. An isolated `--out` (the frames
convention, `artifacts/frames/FRAMES.md`) is not the shared store and needs no lock.

### Mining a graph from recorded runs

`pi gold mine` drives S0–S7 (`pi_eval.mining`) over this repository's own recorded rollouts.
It needs three things: completed runs, a scored parquet, and the public corpus.

```bash
.venv/bin/pi compact
.venv/bin/pi score --gold-root "$PWD/data/gold"
.venv/bin/pi gold mine --suite synth
.venv/bin/pi gold validate --suite synth
```

**It refuses a thin pool, loudly.** `pool_is_admissible` demands ≥ 2 generator cells and ≥ 2
model families before a candidate need may be promoted, because one generator's habits would
otherwise manufacture a phantom need. A task below that is emitted with `admissible=false`
and the reason, its refusal is printed, and if *no* task clears the bar the command exits 1:

```
  s0         REFUSED: only 1 distinct generator cells (< 2)
  ...
admissible=0/12 refused=12
EXIT 1: no task produced an admissible pool.
```

That is the command working. Widen the factorial — a second model family, a second policy
form — and re-run the sweep. A pool made only of `fake_*` arms has exactly one family
(`llm_free`) and can never clear it on its own.

Two pins ride into every mined `graph_version` and neither is a default:

- `nli_pin` is `exact_entail@stdlib` unless a real NLI model is injected. String identity is a
  **weaker** canonicalizer than the pinned model the design calls for — it splits needs a real
  model would merge, so node counts run high and promotion rates run low. It is recorded so no
  mined graph can be mistaken for one produced under a real entailment model.
- With no probe policy wired, **no intervention edge can be minted**. Order alone may only
  veto an edge, never establish one, so a mined graph without a probe carries mechanical edges
  or none. That is the honest outcome, not a gap.

`pi gold validate` prints every gate with its **written consequence**, and prints `NOT RUN`
plus the missing input for every gate whose input this repository cannot produce on its own
(human-gold node recall, edge precision, matcher kappa, contamination delta). A gate list that
silently shrinks when data is missing is a gate list that always passes.

---

## 3. Run

```bash
# a grid is the unit the preregistration freezes; its content hash rides into the manifest
.venv/bin/pi run --sweep conf/grids/tier1_pilot.yaml --concurrency 16 --spend-cap 150

# replay arms (random_q, parallel_replay) need a prior sweep's questions
.venv/bin/pi run --sweep conf/grids/tier1_confirmatory.yaml \
    --seed-questions-from runs --seed-questions-arm inquirer_prompted

# the CEILING arms take an explicit list instead, or the headroom gate is a tautology
.venv/bin/pi run --sweep conf/grids/tier1_oracle.yaml --questions-from-file data/gold/q_musique.json
```

### The ceiling arms, and the headroom gate that costs nothing

```bash
PI_GOLD_ROOT="$PWD/data/gold" .venv/bin/pi gold questions --suite musique \
    --mode oracle_vreq --verify-recall 5 --n 200 --out data/gold/q_musique_vreq.json
```

`tier1_oracle` used to seed BOTH ceiling arms from
`recorded_questions(arm_id="inquirer_prompted")` — the treatment's own questions. That made
`gold_evidence` input-identical to `parallel_replay`, `oracle_vreq` a second copy of it, and
the pre-spend headroom gate a tautology: it compared the treatment against itself and could
only ever report that the treatment had no headroom over itself. **A ceiling seeded from the
arm it bounds is not a ceiling.**

The two modes are now different ceilings, because they bound different things:

| mode | construction | bounds |
|---|---|---|
| `oracle_vreq` | one question per **required** node, decontextualized, in prerequisite-topological order. No answer text. | perfect question **selection and sequencing**, ordinary retrieval |
| `gold_evidence` | the same, plus each node's answer alias appended, forcing retrieval onto the gold paragraph | the **Answerer** — what is reachable when the evidence is simply handed over |

MuSiQue writes later hops as `When was #1 founded?`, where `#1` is hop 1's answer;
`questions_for` substitutes the prior hop's alias, which is exactly what a policy with perfect
selection would have in hand at that point. Left unresolved, the "ceiling" would score *below*
the arm it bounds.

**`--verify-recall K` measures the headroom for zero tokens**, before any of Gate 2's $40 is
spent. It reports the share of each task's required gold evidence the questions actually
retrieve at the grid's `k`, against the task's own full question as baseline. Measured
2026-08-24, musique, n=200, k=5:

| question source | mean gold-evidence recall | tasks at full recall |
|---|---|---|
| the task's own full question (what `drafter_only` sees) | **0.547** | — |
| `oracle_vreq` | **0.947** | 175/200 (88%) |
| `gold_evidence` | **0.998** | 199/200 (100%) |

Two things follow, and both were unknown before the command existed. The ceiling arms *are*
ceilings — `gold_evidence` reaches essentially all the gold evidence, so it genuinely bounds
the Answerer. And there is **+0.40 of evidence recall** between what the full question
retrieves and what a perfectly-chosen question sequence retrieves: that gap is the headroom
inquiry exists to exploit, and it is measured rather than assumed.

| flag | why it exists |
|---|---|
| `--concurrency N` | worker processes. **Overrides** `PI_MAX_CONCURRENCY`. The binding constraint is provider TPM, not CPU. |
| `--spend-cap USD` | campaign cap, checked in the parent as results return. Overrides `PI_SPEND_CAP_USD`. Remaining units are **cancelled, not run**; re-issue to continue, and a resumed unit bills nothing so the cap does not re-trip on it. |
| `--unit-timeout S` | wall-clock allowance per unit (default 1800). A wedged unit becomes `status=timeout` instead of holding a worker for hours. |
| `--questions-from-file` | an explicit `{task_id: [question, ...]}`. Wins over mining a prior sweep. |
| `--no-resume` | re-run units that already have a `status.json`. Resume is otherwise the default. |

A sweep prints progress every 25 units — done/total, ok/err, billed dollars, ETA, and the last
*distinct* error. Silence on a paid job is not neutral: it is what turns a 20-minute diagnosis
into a 3-hour one.

### tau2 is a dialogue and is driven, not run flat

```bash
export TAU2_DATA_DIR=<checkout>/data
.venv/bin/pi verify tau2 --replay-gold                        # banking: 97/97. Zero tokens.
.venv/bin/pi verify tau2 --replay-gold --suite tau2_retail    # retail: 112/112, 2 skipped.
.venv/bin/pi run --suite tau2 --arm drafter_only --n 3
```

Both tau2 suites replay, and both must be checked: they route through ONE driver, so a defect
on either side is invisible from the other. Retail's 2 skips are tasks 24 and 57, which ship
`"actions": []` upstream because their correct outcome is a DB that does not change; they are
exactly the 2 of 114 for which the gold builder emits no graph.

`pi run --suite tau2` dispatches to `pi_run.stages.tau2_runner`, which builds the upstream
Orchestrator, a user simulator (`PI_MODEL_USERSIM`) and our agent, runs the dialogue to
termination and grades it with **upstream's own evaluator**. The flat path is unreachable and
should stay that way: the only task text tau2 ships is the customer's roleplay script.

Three properties, each of which is the difference between a number and a wrong number:

- **Actions travel as messages.** `EnvironmentEvaluator` replays the transcript into a fresh
  environment and compares DB hashes, so a tool call executed against the live env but absent
  from the transcript is invisible to the grader — `tau_reward` would be 0 for every arm no
  matter what the policy did, and P1 would read as a flat null rather than as an unwired path.
- **The view is what the customer actually said**, accumulated over the dialogue. Never
  `user_scenario.instructions`.
- **One ledger per unit**, not per assistant turn: `budget_cap` is 16 retrieval calls for a
  *task*, and a fresh 16 per user turn would break budget parity in the treatment's favour.

### tier2 is priced, and must be TWO invocations

Measured 2026-08-29 on the first live tau2 rollouts at grid parameters
(`--budget-cap 8 --max-turns 16 --k 5`, seeds 0-2), 6 units:

| arm | mean usd/unit | range | mean wall | user turns |
|---|---|---|---|---|
| `inquirer_prompted` | $0.0317 | $0.0044-$0.0459 | 98 s | 3-4 |
| `drafter_only` | $0.0125 | $0.0110-$0.0141 | 127 s | 17-19 |

User-sim traffic is priced separately from the policy's (it is not in the ledger; see
`user_sim_usd`) at about **$0.00031 per user turn** — so `drafter_only` carries the LARGER
harness bill despite the smaller policy bill, because a drafter that never asks makes the
customer do the talking.

tier2 is 97 tasks x 3 seeds x 7 arms = **2,037 units**. Extrapolating the five unmeasured
arms at the `inquirer_prompted` rate gives **roughly $60**, and at the observed per-unit
maximum roughly $85. Treat that as a range and not a quote: it extrapolates five arms from
two measured ones, and this project's small-sample cost projections have been wrong by 18%,
4x, 3.6x and -4x. Wall clock at concurrency 8 is about 8 hours.

**Two invocations, in this order.** The grid loop reads replay questions from prior
manifests filtered by `suite_id` (`cli.py:433-440`), and on a first tau2 run there are
none — so all 291 `parallel_replay` units fail at build. Run the six other arms first, then
`parallel_replay` seeded with `--seed-questions-from runs`.

### Gate 1, run 2026-08-24

Six live dialogues against the proxy, **$0.36 total**. The driver works at full scale.

| | |
|---|---|
| messages per dialogue | 42–102 |
| assistant rollouts per unit | 8–21 |
| executed env calls | 12–38 |
| agent tool-call errors | **0** |
| units reconciled | 6 of 6 |
| `tau_reward` | non-null on every unit, `0.0` on all of them |

`pi compact` → `pi score` → `pi agg --primary` all run on those rows: `tau_reward` reaches
`scores.parquet` as a real scored metric, and the primary table prints `(no eligible rows)`
plus **"Not emitted (no data): tau2:tau_reward:inquirer_prompted-vs-self_inquire"** — naming
the contrast it could not compute rather than emitting an empty cell. `provenance.json` carries
`scorer_hash`, `graph_version`, `run_ids`, the SQL and the eligibility predicate, and shows
*two* `code_version`s, which is the per-commit run-identity discipline made visible.

**The zeros are a policy result, not a plumbing one.** On tasks 036/043/045 the treatment
executes real tools — `log_verification`, `get_user_information_by_email`,
`get_credit_card_accounts_by_user` — but never crosses the `unlock_discoverable_agent_tool` →
`call_discoverable_agent_tool` edge those tasks are graded on. The driver is separately proven
able to score 1.0 when the plan contains the gold actions, so a 0.0 here is a real 0.0. That
the no-inquiry *and* the prompted-inquiry arm both fail to discover tools is the L2 story in
miniature.

**`pi verify tau2 --replay-gold` before any tau2 spend.** It replays each task's own
evaluation criteria and checks the gold DB hash comes back; measured 2026-08-24: **97/97**. It
is the only check that separates a broken harness from a bad policy, and until it existed
`tau_reward` had never been validated against upstream even once.

**Run it for retail too (`--suite tau2_retail`); measured 2026-09-02: 112/112.** It could not
be run at all before that date -- the command built a banking suite unconditionally -- and
retail's grading path was in fact broken in three separate ways that only this check sees. A
task the harness cannot GRADE is now reported under `errors` and fails the check, instead of
being folded into `skipped` alongside the tasks that have nothing to grade.

Measured shape of the 97 tasks (`test_the_shape_of_tau2s_gold_actions_is_what_the_design_assumes`):

| | count |
|---|---|
| gold actions, assistant / user | 853 / 102 |
| of the assistant's: `call_` + `unlock_discoverable_agent_tool` | **730 (86%)** |
| tasks assistant-only / both / user-only | 48 / 42 / 7 |
| tasks with a DB reward basis (i.e. emitting `tau_reward`) | 88 of 97 |

**`tau_reward` is the tau2 primary endpoint (P1) and its secondary (S2).** It was
`task_success` — every required need resolved — for as long as nothing emitted the hash
reward, and the preregistration said in a comment that the line would change in a visible diff
once it did. It has: the driver grades with upstream's own `EnvironmentEvaluator` and
`pi verify tau2 --replay-gold` reproduces the gold DB hash on 97/97. `task_success` was not
merely a placeholder but the wrong measurement for this suite — it is a resolution-coverage
proxy, and **an agent can resolve every need and change nothing**, which on a benchmark whose
success criterion is the customer's account state is the difference that matters. The
denominator is 88 of 97; the 9 ACTION-basis tasks are graded and withheld, because an ENV
evaluator hands them a free 1.0.

The first row is the reason tau2 is the L2 suite: 86% of what the agent is graded on *is* the
read-then-unlock-then-call prerequisite edge. The last two are limits to state rather than
hide — 7 tasks are graded entirely on customer actions the agent can only enable, and the 9
ACTION-basis tasks are withheld from the endpoint because an ENV evaluator hands them a free
1.0 that is not a measurement.

`random_q` draws from **other** tasks: sampled from its own task's pool it is not a null but a
weaker copy of the treatment, and it refuses with `EmptyPool` rather than quietly being one.

**Grid E moves exactly one slot.** `pare_baseline`, `pare_plus_inquirer_prompted` and
`pare_plus_verbosity` share one executor — the same `LLMDrafter`, the same frozen answerer,
the same `PareActuator` — and differ only in what occupies PARE's goal-inference stage. The
kill switch is the deliberate exception: it moves the *drafter*, because "the effect is
length" is the hypothesis it exists to test. Scoring is PARE's own oracle validation folded
into `native()`, so the grid adds no judge and no gold, which is why it is exploratory and
spends none of the multiplicity budget. `pare_baseline` is Observe-Execute with no inserted
inquiry stage on that same executor; upstream's own runner is reachable as
`PareSuite.run_upstream` and its number is an **upstream reference**, not this grid's control
— comparing against a differently-implemented executor would confound the Inquirer with the
actuator.

**Reading the summary.** `ok: false` / `failed: N` is top-level, and resumed units report the
status they were resumed *from*. A sweep in which every unit failed once reported
`{"resumed": N}` and read as a success — the next stage then produced an empty table instead
of an error.

`usd` and `usd_billed` are both printed and are meant to differ. `usd` charges cache hits
**as-if**, which is what keeps two arms comparable however warm the cache happened to be;
`usd_billed` is the invoice, and it is the number the spend cap reads. Their difference is the
cache's savings.

### The RNR ladder had one rung, in the middle

`ASK ≥ RESOLVE ≥ USE` is the ladder the matcher docstring is built on. Both ends were dead.

**The bottom.** The ASK branch tested `re.findall(r"\b[KD][0-9A-F]{8}\b", node.gold_text)` —
the synthetic suite's opaque ids. Those live in the corpus documents and the task *question*;
`gold_text` reads `"the value for facet 0 at step 0 is V000"`. Measured: **0 nodes on synth,
musique, strategyqa, wiki2 and tau2**.

**The top.** Both shipped Answerers set `cited_unit_ids=tuple(u.uid for u in ev.units)` — they
cite *every* retrieved unit, unconditionally — so `ev <= cited` held whenever
`ev <= retrieved`, and `kind` was always `"use"`, never `"resolve"`. Confirmed on live musique
rollouts: every matched need came back `use`.

So `RNR_ask == RNR_resolve == RNR_use`, and the identity held vacuously in both directions.

A natural-language need now matches ASK when at least `ASK_MIN_COVERAGE` (0.5) of its
distinctive terms appear in what was asked, and never on fewer than `ASK_MIN_TERMS` (2). Both
floors are deliberate: ASK is the cheapest rung to game, and the count floor stops one shared
word crediting an ask across a suite while the share floor stops a long need being credited by
two incidental words. Reach after the change: **95–100% of nodes**, against 0%.

USE is **capped at RESOLVE when the citation set names everything retrieved** — such a list
carries no information about which need mattered, and the ladder's strongest claim must not
rest on an instrument that says nothing. Cite a strict subset and the rung becomes real again
with no other change: the instrument was never the problem, the Answerer's behaviour was.

The thresholds are instrument choices, so they are named constants, and `matcher_id` is
`mechanical_v2` — it rides into `matcher_hash` → `scorer_hash`, so rows scored under the old
instrument can never be pooled with rows scored under this one.

### Known gaps, stated

- **PARE's oracle verdict has no caller.** `PareActuator.attach_validation` exists and nothing
  invokes it, so a PARE run writes no `oracle_success` row — downstream indistinguishable from
  a scenario the policy failed. Not fixed because **Grid E is cut**: exploratory, no
  confirmatory test, oracle reproduction never demonstrated. Un-cutting it means writing the
  driver that runs a scenario to completion and calls it, the same shape as
  `pi_run/stages/tau2_runner.py`. `tests/test_adapters_pare.py` pins this and fails the moment
  a caller appears.
- **`STOP_SHARE_CEILING` is not calibrated.** No rung-1 dataset has been built, so 0.80 is
  chosen to catch collapse rather than to tune anything. The first real export should confirm
  it or move it in a visible diff.

Three things an adversarial review surfaced that are **not** fixed, recorded here rather than
discovered later:

- ~~The unit watchdog is arm-independent.~~ **Measured and closed 2026-08-24.** Ten live units:

  | suite | arm | n | max wall | median |
  |---|---|---|---|---|
  | musique | `drafter_only` | 2 | 17 s | 14 s |
  | musique | `inquirer_prompted` | 2 | 86 s | 57 s |
  | tau2 | `drafter_only` | 3 | 300 s | 165 s |
  | tau2 | `inquirer_prompted` | 3 | **726 s** | 426 s |

  The treatment runs 5× the baseline on musique and 2.4× on tau2, so at the old 1800 s cap it
  was the only arm that could realistically time out — and one rate-limit ladder is ~148 s.
  `PI_UNIT_TIMEOUT_S` now defaults to **3600 s**, 5× the slowest unit ever observed, and
  `summarize()` reports `timeout_margin` and warns once the slowest unit passes half the cap.
  Note that `report.estimate` pairs on the intersection of the two arms' task sets, so a
  missing treatment unit drops that task from *both* arms: the residual risk is not an
  arm-asymmetric bias but a selection toward easier tasks.
- ~~σ_J's provenance is not checked against the judge that produced the scores.~~ **Closed.**
  `pi score` now refuses when `sigma_j.provenance.json`'s `judge_model` differs from the judge
  producing the scores. A floor from one model applied to another is not conservative in a
  knowable direction: a quieter judge's floor lets a noisier judge's effects through, and a
  noisier judge's floor suppresses a quieter judge's real ones. An *absent* sidecar still
  scores — before σ_J is measured the floor is +inf and every judged effect is flagged, and
  refusing there would make the flag unreachable.
- ~~The judge has no spend cap.~~ **Closed.** `--judge-spend-cap` / `PI_JUDGE_SPEND_CAP_USD`,
  same precedence rule as `--spend-cap`. Exceeding it **writes nothing**: `score` writes its
  rows only after judging returns, and a half-judged parquet is worse than an unjudged one —
  some arms would carry judge-derived rows and others would not, and the comparison between
  them would silently be over different subsets.

---

## 4. Score, aggregate, render

```bash
.venv/bin/pi compact
.venv/bin/pi score --gold-root "$PWD/data/gold"     # the ONLY stage that may read gold
.venv/bin/pi agg --primary --assert-no-gold-exposed
.venv/bin/pi render --all
```

- **A dirty git tree gives every run a `dev-` id, and the aggregator mechanically excludes
  `dev-` runs from every reported table.** Empty tables usually mean an uncommitted tree, not
  a bug. Commit, re-run.
- `--assert-no-gold-exposed` exits non-zero if a gold-exposed, `dev-`, pilot or canary-hit run
  reached a primary table.
- Re-scoring under a new `scorer_hash` **adds** rows; it never re-rolls and never mutates.
- `render` refuses on provenance drift or a hand-edited `.tex`.

---

## 5. The order that matters

Cheap things that can end the paper come first. Total spend to the decision: **under $8.**

| # | step | why it is first |
|---|---|---|
| 1 | `python scripts/probe_retrieval.py --suite musique --n 50` | Zero tokens. If every arm retrieves the same evidence, every Δ is 0.00 ± 0.00 and it looks exactly like a null. |
| 2 | `tier1_oracle` (400 units, ~$1.60) | If `F1(gold_evidence) − F1(drafter_only) < 0.10` there is no headroom and no policy can demonstrate anything. |
| 3 | `tier1_pilot` (840 units, ~$3.76) | Kill-switch day. Five contrasts, each with a written consequence in `prereg/`. |
| 4 | seal `prereg` | `python scripts/seal_prereg.py seal-stage1 && git tag prereg-stage1`. Write-once. |
| 5 | `tier1_confirmatory` | Only after 1–4. |

**Do not seal the preregistration before the pilot, and do not run the confirmatory grid
before the pilot.** That sequencing is the one error that wastes the whole campaign.

---

## 6. Diagnosing a failure

| symptom | meaning |
|---|---|
| `GoldAccessError: PI_GOLD_ROOT is unset` in a worker | **The firewall working.** Rollouts may not read gold. |
| `pi verify firewall` exits 1 | A canary nonce reached a serialized request. The run is void; there is no benign explanation. |
| `run_id` starts with `dev-` | Dirty tree. Excluded from every reported table. |
| `LLMConfigError: role 'x' has no model pin` | Set `PI_MODEL_X`. It will not guess. |
| `EmptyPool` from `random_q` | Seed from a sweep covering more than one task. |
| `PareSuiteUnavailable` / tau2 "Data directory does not exist" | The wheel ships no `data/`. Set `PARE_BENCHMARK_SPLITS_DIR` / `TAU2_DATA_DIR`. |
| `pi gold mine` exits 1 with `REFUSED` | The trace pool is too thin for the contamination controls. Not a bug — see §2. |
| `sha256 mismatch for data/raw/...` | Upstream changed the file, or the cache is corrupt. Delete and refetch; if upstream genuinely republished, the suite_version must be bumped with the pin. |
| `reconciled_docs: false` | A Drafter retrieved inside `resolve()` without metering. Budget parity is broken. |
| `below_noise_floor: true` | Effect under `2σ̂_J/√n`. Printed, never claimed. |
| tables empty but runs succeeded | Almost always a dirty tree (`dev-` exclusion). |
