# Data sources: licence status and how to obtain each

Nothing upstream is redistributed in this repository. Each builder downloads to `data/raw/`,
verifies a sha256, and splits the result into a **public corpus** (`data/corpora/`, the only
thing an adapter may read) and a **gold graph** (`data/gold/`, read only by `pi_eval`).

All statuses below were verified by direct API call on 2026-08-23.

| Source | Licence | Access | Notes |
|---|---|---|---|
| MuSiQue | CC BY 4.0 | HF `dgslibisey/MuSiQue` (musique-ANS v1.0), ungated | `question_decomposition` `#N` placeholders are human-authored prerequisite edges. Exclude the ids in `dev_test_singlehop_questions_v1.0.json`. |
| StrategyQA | code MIT; AI2 zip reachable (200); HF `ChilleD/StrategyQA` tagged MIT, ungated | direct zip or HF | 2,780 questions. Decomposition is *strategic and not stated* in the question — the cleanest test of horizontal proactivity. Binary answers, so no primary endpoint. |
| 2WikiMultihopQA | Apache-2.0 | HF `xanhho/2WikiMultihopQA` | Entity graph, not a sub-question graph. |
| FRAMES | Apache-2.0 | HF `google/frames-benchmark`, 824 rows, test only | Document-level evidence only, no decomposition. |
| τ-Knowledge (`banking_knowledge`) | **MIT, code *and* data** | `git+https://github.com/sierra-research/tau2-bench@v1.0.1` — **not on PyPI (404)** | 698 documents, 97 tasks (both verified). `tasks.json` is **stale for 13 tasks**; always glob `tasks/task_*.json`. Pin retrieval variant `bm25`. |
| τ-Knowledge **data** | MIT | **not shipped by the package** | The pip/git install provides `tau2` code but no `data/` directory: it logs `Data directory does not exist` and needs `TAU2_DATA_DIR` pointed at a repo checkout. Verified against a checkout: 698 documents, 97 tasks. |
| PARE | MIT | `git+https://github.com/deepakn97/pare@main`, dist name `proactive-agents-research-environment` | 143 scenarios, 9 FSM apps. Python 3.12+. |
| UserBench (`travelgym`) | **Apache-2.0, code *and* data** | `git clone https://github.com/SalesforceAIResearch/UserBench`, pinned at commit `80506d2ab484cab843e60a2401ff3e0290d05b87` — **not on PyPI** | 3,122 scenarios in `travelgym/data/*.json` (~200 MB, 8 files), split 2,651 train / 471 test by upstream's own `data/{env}_multiturn/{train,test}.parquet` index. The three headline envs travel22/33/44 give **255 test tasks** (101/87/67) — all verified against the checkout. **EVAL-ONLY** (`pinq.splitting.EVAL_ONLY_SUITES`): the 2,651 train scenarios are reachable in one config line and must never be. Its user simulator is a **paid LLM call** that returns `None` on failure and is then scored as reward 0.0 with a canned reply — see the note below. |
| Researchy Questions | CDLA-Permissive-2.0 | HF `corbyrosset/researchy_questions`, ungated | `gpt4_decomposition` is model-generated → weak supervision only. |
| DRGym agentic-search logs | CC BY 4.0 | HF `cx-cmu/deepresearchgym-agentic-search-logs`, ungated | 14.3M ordered queries: the edge-mining substrate. |
| DeepResearchGym search API | — | `https://clueweb22.us`, **API key required** | Re-verified 2026-08-24: `/health` = 200, both `/search` and `/fineweb/search` = **401** without a key (`{"detail":"Invalid or missing API Key..."}`). The public FAQ claiming FineWeb is keyless is stale. Email `deepresearchgym@cmu.edu`. Without a key, `DrGymSuite(root, offline=True)` replays the search cache. |
| DRGym eval harness | **NO LICENCE FILE** (licence API → 404, re-verified 2026-08-24) | `github.com/cxcscmu/deepresearch_benchmarking`, pinned at commit `d4d2433309d9da9be637c365618e7e01f8f9205a` | Public ≠ licensed. Its judge prompts are **reimplemented** here from visible text, never vendored, with a published fidelity delta (`pi_eval.judges.fidelity`). Its committed `key_point/*_aggregated.json` files are the gold key points: 1000 files, 1:1 with the 1000 query ids, of which **24 are empty** (`key_points: []`) and are filtered and counted by the builder → 976 tasks. `queries/researchy_queries_sample_doc_click.jsonl` sha256 `39c97a79da4be948e53a90cdf73c30c664388b39df1f9f5947a123ed6d7a1568`. |
| ClueWeb22 | signed institutional licence | `lemurproject.org/clueweb22/obtain.php` | Weeks of lead time. **Not used** — FineWeb only. |
| Claude Code session logs (`convlog`) | **none — private, collaborator consent on file** | not distributed; ingested from a Drive folder by `scripts/convlog_fetch.py`, or a hand-downloaded zip | Real agent-user sessions: the only source here where the user channel was open. Scrubbed at build (`pi_eval.build.scrub`, fail-closed). **Never released** (D16); raw and corpus are both git-ignored. Requires `--consent-sha`, matching the sha256 of `docs/consent/convlog.md` (derived at build time by `pi_eval.build.consent.recorded_for`, never pasted into a constant). See docs/CONVLOG.md. |
| HotpotQA | CC BY-SA 4.0 | HF `hotpotqa/hotpot_qa` | Not used: 2 hops means max depth 1, so the depth axis degenerates. |

## How to obtain and build each suite

Every suite below is driven by `pi data` / `pi gold`. There is no Python one-liner to
remember, and there is no `--no-verify`.

```bash
.venv/bin/pi data fetch  --suite <suite>          # download + verify the pinned sha256
.venv/bin/pi data build  --suite <suite> [--limit N]   # raw -> corpus + gold
.venv/bin/pi data status                          # raw / corpus / gold per suite
.venv/bin/pi gold stats  [--suite <suite>]        # nodes, edges, depth, provenance, orphans
```

`fetch` is idempotent: a cached file is re-verified, not re-downloaded, and the output names
the digest AND whether it was checked against a **hard pin** (a constant recorded in the
builder) or a **trust-on-first-use sidecar**. `build` writes a content-addressed corpus
directory, so a rebuild from the same raw input produces the same `corpus_hash`.

Measured by building each suite on 2026-08-24 (`pi gold stats`, depth recomputed from the
stored edges rather than read off the record):

| suite | tasks | nodes | edges | facets | depth histogram | orphan rate |
|---|---|---|---|---|---|---|
| synth | 12 | 108 | 72 | 36 | `0:36, 1:36, 2:36` | 0.000 |
| musique (60-task slice) | 60 | 200 | 140 | 60 | `0:90, 1:60, 2:40, 3:10` | 0.000 |
| strategyqa | 2,290 | 6,720 | 4,483 | 2,272 | `0:3721, 1:2320, 2:615, 3:60, 4:4` | 0.000 |
| wiki2 (dev) | 12,576 | 31,120 | 11,956 | 11,956 | `0:19164, 1:11956` | 0.000 |
| drgym | 976 | 16,156 | 0 | 0 | `0:16156` | 0.000 |

Notes that are not decoration:

- **MuSiQue selection is hop-stratified** (`_stratified_order`). The upstream files are
  hop-sorted, so head-N truncation yields an all-2hop sample whose graphs are depth ≤ 1 and
  makes `coverage_at_depth_ge2` structurally unmeasurable at any n. Measured: a 40-task
  head-N build had depth histogram `{0:40, 1:40}` and not one depth-2 node. The 60-task
  stratified slice above reaches depth 3.
- **StrategyQA edges are all mechanical.** Re-measured on the full train split 2026-08-24:
  2,999 of 6,720 steps carry an explicit `#N`, and the prose-back-reference fallback fires on
  **0** of 6,720 (every step matching the prose pattern already carries a `#N`, so the
  mechanical branch claims it first). An earlier docstring claimed 1; the builder now records
  the measured 0. Depth from this suite is a lower bound.
- **DRGym builds its gold but not its corpus.** 1,000 `key_point/*_aggregated.json` files at
  the pinned commit, 24 of them empty → **976 tasks / 16,156 key points**, exactly as the
  builder's docstring predicted. The graph is flat by construction: `original_point_number`
  is merge provenance, not dependency, so it is preserved as an audit sidecar and never
  emitted as edges. The *search corpus* needs an API key (401 without one), so
  `gold_ev_uids` is empty and resolution on this suite is judged, never matched by uid.
- **2Wiki is depth ≤ 1 on dev.** Its edges come from object → subject linkage inside a
  single reasoning type, and `comparison` questions are flat by construction, so the whole
  dev split has depth histogram `{0: 19164, 1: 11956}` and no depth-2 node. That is the
  correct shape for an entity graph, not a missing one — but it means `coverage_at_depth_ge2`
  is not measurable on this suite and MuSiQue is the only real suite that carries that axis.
- **tau2, PARE and UserBench are not driven by `pi data`.** All three are self-sourced from
  an upstream checkout (`TAU2_DATA_DIR`, `PARE_BENCHMARK_SPLITS_DIR`, `USERBENCH_DIR`), and
  `pi data status` prints a row saying so rather than omitting them. PARE's `full.txt` split
  was re-verified on 2026-08-24: **143 scenario ids**, matching `N_SCENARIOS` and
  `conf/grids/pare_transfer.yaml`.

## UserBench: how to fetch it, and the three things that will bite you

Nothing is vendored. The ~200 MB of scenarios lives in the upstream repository and stays
there; this adapter reads it in place.

```bash
git clone https://github.com/SalesforceAIResearch/UserBench.git
git -C UserBench checkout 80506d2ab484cab843e60a2401ff3e0290d05b87
export USERBENCH_DIR="$PWD/UserBench"
uv pip install -e "$USERBENCH_DIR"          # installs the `travelgym` package
```

`USERBENCH_DIR` is the repository **root**, not the `travelgym` package inside it; both the
split index (`data/<env>_multiturn/test.parquet`) and the scenarios
(`travelgym/data/travelgym_data_NN.json`) are resolved from it. With the variable unset the
adapter raises and prints these four lines rather than fabricating a corpus, for the reason
in `pinq_adapters/userbench/_probe.py`: an empty corpus reads downstream as a policy that
found nothing.

`uv pip install -e` pulls upstream's full `install_requires`, which includes `together`,
`google-genai`, `boto3`, `pandas` and `fastparquet` — every one of them only for `eval.py`,
which this adapter does not use. `--no-deps` alongside `gymnasium numpy pyyaml openai
pyarrow` is enough to construct a `TravelEnv`, and is what the live episodes below were run
under.

### What a live episode needs beyond the checkout

`USERBENCH_DIR` alone gets you `task_ids()`, `view()` and `pi suites validate` — all of which
read parquet and JSON and spend nothing. An **episode** additionally talks to a user
simulator, and three variables decide who that is:

| variable | what it does |
|---|---|
| `PI_MODEL_USERSIM` | the model that answers as the user. **Required; never defaulted.** |
| `LITELLM_BASE_URL` | the proxy that serves it. Optional — absent means OpenAI direct. |
| `LITELLM_API_KEY` (or `OPENAI_API_KEY` when no proxy) | the key for *that* endpoint. |

```bash
export PI_MODEL_USERSIM="openai/aws/gpt-oss-120b"   # the project's user-simulator role pin
# LITELLM_BASE_URL / LITELLM_API_KEY come from .env
```

Three things about this are load-bearing, and all three were discovered by running an
episode rather than by reading upstream:

- **The pin is required.** `TravelGymConfig.model_name` defaults to `"gpt-4o"` and `api_key`
  to `$OPENAI_API_KEY`. An adapter that set neither ran this suite's simulated user on a
  model the rest of the paper does not use, and recorded that nowhere. Against this
  project's own proxy it is a 403 on every call → `model_call` returns `None` → the failure
  counter refuses the run (loud, and dead); against an endpoint that *does* serve gpt-4o it
  would have produced numbers quietly. `SimulatorPin.from_env` refuses instead, naming the
  variable — the same rule as `MeteredClient.model_for`.
- **`openai/` is stripped.** `openai/aws/gpt-oss-120b` is a *litellm* routing spec: the
  prefix selects the OpenAI-compatible handler and never reaches the wire. travelgym calls
  the OpenAI SDK directly, so the adapter sends `aws/gpt-oss-120b`, which is what the
  proxy's own `/v1/models` lists. Exactly one prefix is removed, and only `openai/` —
  `azure/gpt-oss-120b` is also a real id on that proxy.
- **The endpoint can only be set through the environment.** `prompts.model_call` builds
  `OpenAI(api_key=...)` with **no** `base_url`, so the only lever is `OPENAI_BASE_URL`.
  `UserBenchSession` sets it for the duration of a session (and restores it afterwards),
  next to the failure counter it arms, so a stale value from another tool cannot silently
  redirect the user simulator to an endpoint the run record does not name. `LITELLM_BASE_URL`
  is stored without a path because litellm appends one itself; the adapter appends `/v1`,
  which the OpenAI SDK does not.

`UserBenchSuite.session(task_id)` is the only door to an episode. `pi run --suite userbench`
**refuses** — `driven_by="gym"` in `pi_run.suites.REGISTRY` — and prints what to use instead
before it starts a sweep.

Three properties of upstream that the adapter exists to handle, each verified by reading the
code at the pinned commit:

- **A failed user-simulator call scores 0.0 and looks exactly like a real result.**
  `travelgym/env/prompts.py:model_call` returns `None` after three failed attempts — and
  also returns `None` without retrying when the model's text does not parse as JSON. Every
  caller sits inside a `try` that swallows it: `evaluate_action` falls back to a canned reply
  with reward 0.0, and `eval.py:rollout` catches again and returns `[0]`. So a run with a bad
  key, an exhausted quota or a rate limit produces a complete, well-formed, **entirely zero**
  results file with no traceback and a zero exit code. The `[search]` branch is worse: its
  fallback string is the same one the env emits for its own *deliberate* every-5th-search
  error, so the feedback text cannot distinguish the two. `pinq_adapters.userbench.simulator`
  counts `model_call`'s return value and `UserBenchSession.step` **refuses to emit a result**
  if any call came back unusable. A UserBench number is admissible only with
  `simulator_failures: 0` recorded next to it.
- **The answer key is in the observation and on the env object.**
  `observation["task_description"]` is the full `scenario` prose, in which every preference is
  written out; `info["preferences_summary"]` lists the ones not yet elicited; and
  `env.state_list["remaining_best_options"]` / `remaining_correct_options` are the ids
  `[answer]` is scored against by **pure set membership**, with no judge. Measured over the
  1,462 preferences in the 255 test tasks, by content-word overlap: `scenario` covers 52.7%
  of a preference on average and makes 334 of 1,462 at least 80% recoverable;
  `initial_description` covers 6.0% and makes **zero** at least 80% recoverable. The adapter
  shows an agent `initial_description` and, per turn, `observation["feedback"]` — which is
  exactly and only what upstream's own baseline agent reads — through an `ObservationView`
  whose field names are disjoint from every raw key.
- **`eval.py` is not used.** `--one_choice` is declared `type=bool`, so argparse evaluates
  `bool("<the string>")` and *any* value yields True while omitting the flag yields False and
  silently overrides the config default of True — the flag cannot express one of its two
  states. Measured against the declaration at the pinned commit: no flag → `False`;
  `--one_choice False` → `True`; `--one_choice 0` → `True`; `--one_choice no` → `True`.
  (An earlier version of this note also claimed that `rollout()`'s `hash_id = hash(data)`
  raises `TypeError: unhashable type: 'dict'` outside its `try`. **That is wrong and has been
  retracted**: `eval.py` defines its own `def hash(data)` at module scope — a sha256 over
  `json.dumps(data)` — which shadows the builtin, and it succeeds on a real row. Re-measured
  on `data/travel22_multiturn/test.parquet` row 0: upstream's `hash` returns
  `c08397c741c03905…`; only the *builtin* raises. The line is still outside the `try`, and
  would still be an unguarded failure if `data` ever stopped being JSON-serializable, but it
  is not a live defect and this file should not have said it was.)
  The adapter drives `TravelEnv` directly. It also pins `config.seed`, because
  `TravelEnv.__init__` seeds `random` and `numpy.random` from it *before* loading tasks;
  left unset, `travelgym.env.task_data` seeds numpy once at import and each later env
  construction continues that stream, so the second env built in a process sees a different
  option order than the first.

## What this repository releases

- The mined information-need graphs, with per-node provenance
  (`human_composed` / `bench_author` / `mechanical` / `trace_mined` / `llm_elicited` /
  `click_evidence` / `human_rated`), ablation verdicts, and confidence.
- The content-addressed request cache (keys redacted), which is what makes results
  *cache-replayable*. Note that temperature 0 is **not** bitwise deterministic under batched
  inference; "replayable" means cache-replayable, and the README says so.
- Never: upstream document text from any corpus whose licence does not permit it. Gold
  records carry ids, URLs and offsets only.
