# Training track — the rung ladder

Every command in this file has been run and its output pasted, or it says plainly that it has
not. The rungs that need a GPU have **never been executed**: this project has no CUDA device
(Apple M1 Max) and no rented one has been booked. What that means concretely, rung by rung, is
in the "status" column of the table below and repeated in each rung's own section.

The governing constraint: **`pinq_train` never imports `pi_eval`, and nothing imports
`pinq_train`.** The trainer reaches a score only over HTTP, through `src/pi_run/serve/`. Four
import-linter contracts hold it, and `make gate` prints `Contracts: 4 kept, 0 broken`.

---

## 0. The ladder at a glance

| rung | what it does | GPU | plan cost | status |
|---|---|---|---|---|
| **0** | GEPA prompt search over the Inquirer template | none | $600 / 6,000 rollouts | **ships**; runs from a laptop; dry-run verified |
| **1** | rejection-sampling SFT, LoRA, masked loss | 1×80 GB | ~180 GPU-h ≈ $450 | everything except `trainer.train()` is tested; `train()` never run |
| **2** | DPO/KTO on same-state pairs | 1×80 GB | ~25 GPU-h ≈ $60 | same |
| **3** | GRPO | 8×80 GB | $2,400, **hard kill 2026-09-12** | **scaffold only**; `train()` raises |

```bash
.venv/bin/pi train status      # the ladder, which deps are present, what is on disk
```

Rung 0 is **not optional**. It produces the Inquirer prompt that the untrained and the
frontier families both run, which is what makes "trained vs prompted" a comparison of model
pins rather than a prompt contest. Skip it and the headline claim reduces to "our fine-tune
beat a prompt we did not try very hard on".

---

## 1. The seam, and why the trainer cannot cheat

```
  trainer process                        scoring process
  PI_GOLD_ROOT UNSET                     PI_GOLD_ROOT SET
  pinq_train.client.SeamClient  ──HTTP──▶  pi_run.serve
    refuses to construct if                 POST /rollout   one episode
    PI_GOLD_ROOT is set                     POST /score     components, never a reward
                                            POST /retrieve  Search-R1's contract verbatim
                                            GET  /healthz   which role this process may play
```

Three properties, each with a test in `tests/test_serve.py`:

- **`/score` refuses any task outside `train`**, with exit code **3** from the CLI form (`1`
  means "something broke"; `3` means "you asked for the one thing that voids the experiment").
  It reads the split **stamped in the run's own manifest**, not a recomputation — a later edit
  to a bucket function must not silently reclassify finished runs.
- **`tau2` and `pare` are refused by name**, before any hashing. They are the zero-shot
  transfer targets; "it happened to hash into train" is not a reason to burn them.
- **A gold-exposed episode is refused even inside `train`.** `gold_evidence` and `oracle_vreq`
  chose their questions where gold is readable; training on them is oracle distillation.

The server returns **components, never a reward**: `pi_run` may not import `pinq_train`, so the
weights physically cannot live server-side — and that is also the right design, because
re-weighting a finished rollout set should be arithmetic over stored numbers rather than a
re-score of every episode.

Start the two servers in two shells:

```bash
# scoring side — gold readable, MUST NOT serve /rollout
PI_GOLD_ROOT="$PWD/data/gold" .venv/bin/python -m pi_run.serve.app --port 8077

# rollout side — gold NOT readable
env -u PI_GOLD_ROOT .venv/bin/python -m pi_run.serve.app --port 8078 --suite musique
```

`GET /healthz` reports `gold_root_set` so the misconfiguration (both roles in one process) is
visible rather than something to debug — and `/rollout` re-asserts it, so a rollout request to a
gold-readable process raises rather than quietly producing an unusable episode.

> **Bug found and fixed while building this track.** The HTTP surface had never worked.
> `create_app` imported `fastapi.Request` inside the function body; with
> `from __future__ import annotations` every annotation is a string, FastAPI resolved
> `"Request"` against the module globals, found nothing, and classified the parameter as a
> **query** parameter. Every POST answered
> `422 {"loc":["query","request"],"msg":"Field required"}`. `GET /healthz` takes no parameters,
> so a health check passed while `/rollout`, `/score` and `/retrieve` were all dead — and the
> handler unit tests passed too, because the handlers were correct. The handlers now take
> `body: dict`, which resolves from builtins under any import discipline, and
> `tests/test_serve_http.py` drives all four endpoints through `TestClient` so the next person
> to reach for `Request` finds out in CI rather than in a paid rollout campaign.

---

## 2. Splits

```
bucket = f(suite_id, template_id or task_id) % 100     0–59 train, 60–74 dev, 75–99 test
```

Hashing on `template_id` where one exists is what stops two instantiations of one template
landing on opposite sides of the wall. `tau2` and `pare` contribute **zero** training examples
by construction (`pinq_train.split.EVAL_ONLY_SUITES`), so their columns are genuine zero-shot
transfer rather than a within-suite gain.

> **Fixed. Recorded because the number it cost is worth remembering.** There used to be
> **two** bucket functions that did not agree: `pi_run.manifest.split_of` hashed
> `h("split", suite_id, key)` while `pinq_train.split.bucket` hashed
> `sha256(f"{suite_id}|{key}")`. A run was *stamped* with the first at rollout time and
> *re-checked* against the second at export time, so a task could be `train` for the runner
> and `test` for the exporter. Nothing ever leaked — the exporter intersects, so the failure
> was lossy rather than unsafe — but it silently discarded **66 of 109** otherwise usable
> decision points, and it was a disagreement about which tasks are trainable, which had to be
> settled before anything was trained and before the prereg was sealed.
>
> There is now **one** definition, `pinq.splitting`, which both modules delegate to. Verified
> two ways and re-measured on this repository's runs: `split_disagreement` returns `{}` over
> all 143 usable decision points, and `tests/test_train_export.py` checks both that the two
> agree over 500 task ids and — structurally — that neither hashes on its own, since two
> functions that agree today can drift tomorrow.
>
> `pi train export` still prints the count as a `WARNING` rather than a `SplitViolation`
> tally, because an opaque tally reads like a contamination attempt being blocked, which is
> the one thing it is not. The count is now structurally zero; it is kept because it is the
> check that would catch a future re-divergence.

### 2.1 The id set, and the reader it did not have

Every export writes the **set itself** next to its manifest as `train_ids.<hash16>.txt` — the
sorted `suite_id/task_id` lines whose set-hash is the `train_id_set_hash` the manifest reports
(the same 16 hex `pi train export` prints, e.g. `train_id_set_hash=263ae507f7a25db5` below), and
`write_jsonl` refuses to write a sidecar whose ids do not reproduce that hash. A checkpoint's
row in `conf/checkpoints.json` carries that hash, so `pi_run.manifest.build_manifest` stamps it
onto any run whose **Inquirer** pin is a registered deployment — and `None` onto every run whose
Inquirer is not, which is every run made so far; it is deliberately outside
`RunManifest.semantic_hash`, so stamping it renames nothing. `pi agg` then resolves each stamped
hash to its id file and **refuses** the aggregation if a trained arm's task is in the set
(`reason=train_id_set`) *or* if the file cannot be found at all — `--ids-dir` points at the
sidecars when they are not under `data/rl/`.

#### Lineage rows: what a rung-2 checkpoint was actually fitted on

A rung-1 row's `dataset_sha` and `train_id_set_hash` are **one quantity** — the adapter was
fitted on exactly one file. A DPO adapter was not. Rung 2 is *initialised* from rung 1
(`DPOConfig.validate` refuses an empty `adapter` under either reference mode, because DPO on
the raw base optimises a preference the base has no policy to express), so its weights saw the
SFT export's tasks **and** the pairs export's — while its row names only the pairs export.

Measured 2026-09-15 on the shipped exports:

| export | SFT ids | pairs ids | SFT − pairs | pairs − SFT | union |
|---|---:|---:|---:|---:|---:|
| `data/rl/headline/` | 2,525 | 1,027 | **1,499** | 1 | 2,526 |
| `data/rl/` (pooled) | 2,664 | 1,933 | **732** | 1 | 2,665 |

So `qwen3-8b-dpo-headline-control` stamped its runs with a hash resolving to 1,027 of the 2,526
tasks its weights had seen. The **split** predicate above still covered those rows; the id-set
canary beside it was blind to the other **1,499** — and that is precisely the exposure layer 4
of the firewall exists for, since layers 1–3 stop leakage through *code* and only the canary
catches leakage through a *string* such as a served model name.

The canary also covers what the split predicate structurally cannot see — a task exported into
a training file and since re-bucketed, where `split_of` answers `test` today and is *right*
while the weights still saw it. That argument is structural, not a count. Measured 2026-09-15,
**0** of the 2,525 SFT ids and **0** of the 1,027 pairs ids are `test` or `dev` today.

> **A retracted number, and the call that produced it.** An earlier draft of this section said
> 94 of the SFT ids were `test` today. That came from `split_of(suite_id, task_id)` with no
> `template_id`. `bucket()` hashes `template_id or task_id`, and musique's ids carry template
> ids — so for its 453 SFT ids (and 337 pairs ids) the bare call hashes the wrong string.
> Every one of the 148 SFT ids and 114 pairs ids it mislabels is musique; joined to
> `scores/parquet/runs.parquet` for the template id, all of them are `train`. The other 2,072
> SFT ids (strategyqa, wiki2, synth) carry no template id at all, so there the bare call is
> already the right one. **The reader was never affected**: `train_split_violations` keys on
> the parquet's `template_id` column, which is the same quantity `bucket()` wants. This is the
> same false alarm the prereg draft hit on 2026-09-15.

The fix keeps `dataset_sha` meaning what it always did — *this* row's own export — and makes
`train_id_set_hash` the **union over the lineage**, with a new `init_from` field naming the row
it was unioned with. `scripts/hpc/register_checkpoint.py --init-from <model id>` computes it
with `pinq_train.split.id_set_hash` (the exporter's own function, imported rather than written
again) and writes the union's preimage as a `train_ids.<hash16>.txt` sidecar beside the dataset
manifest — so the stamp still resolves by the same filename rule, with no index in between.
`--init-from` is **required** at rung ≥ 2 and **refused** at rung 1; the loader enforces the
same rule and additionally refuses an `init_from` that is unregistered, that names a **higher**
rung (lineage does not run backwards: a DPO pass cannot have started from a GRPO), or whose
chain does not reach a base case. Termination is checked by *walking* the chain with a visited
set to a row with no `init_from` or to the literal `base`; a cycle, including a row naming
itself, is a refusal. It is deliberately **not** enforced by demanding a strictly lower parent
rung: **DPO-on-DPO is still rung 2**, and a second DPO pass from the
`qwen3-8b-dpo-headline-control` adapter is a planned arm, so that rule would have made a
legitimate checkpoint unregisterable. Because an init row's own `train_id_set_hash` is
*already* a union when that row is itself a lineage row, such a stacked arm recurses for free
and no row needs more than one union.

The one rung ≥ 2 row with no ancestor is `pi train rung2 --reference base` (the `dpo_from_base`
ablation, a fresh LoRA on the untrained base). It takes the literal `--init-from base`: no
union, `train_id_set_hash` equals `dataset_sha` as at rung 1, and the row records *why* rather
than leaving the field off. Where the run dir has a `rung2.manifest.json`, the flag is
cross-checked against its `reference_policy.reference` in both directions — `init_from` is the
only field on a row that a reader cannot re-derive from the served artifact.

Existing rows are migrated per row with
`--backfill-init-from <init model id> --name <row> --dataset-manifest <its export manifest>`;
plain `--backfill` refuses a rung ≥ 2 row that names no `init_from` rather than copying
`dataset_sha` across, which would write the subset stamp all of this exists to stop. Nothing
about run identity moves: `train_id_set_hash` is absent from `RunManifest.semantic_hash`, so
rewriting it on a row that has already served renames no finished run.

> **Why a second check, when `split_of` already refuses non-test tasks.** The split predicate
> is a statement about the split *function*, so it catches a task that is train or dev
> **today**. It cannot catch a task that was exported into a training file and has since moved
> — a re-bucketed suite, a `template_id` minted after the export, an export taken before
> `EVAL_ONLY_SUITES` grew. After the move `split_of` answers `test` and it is *right*: the wall
> is where it says it is, and the checkpoint still saw the task. Only the recorded id set can
> answer "did these weights see it?". Before 2026-09-15 `pinq_train.split` called the hash
> layer 3 of five, "so a later run can prove which ids a checkpoint saw", and nothing ever
> asked it to prove anything: no preimage on disk, no stamp on a run, and a reader that
> selected the column and ignored it.

---

## 3. `pi train export` — recorded runs → SFT rows and preference pairs

This is a **gold-side** command, like `pi score`: it attaches a measured value to every
decision point, so it needs `PI_GOLD_ROOT`. The *trainer* never does this; it asks a server.

```bash
.venv/bin/pi train export --kind both --gold-root "$PWD/data/gold" --tau 0.005
```

Observed on this repository's 50 recorded synth runs (2026-08-24; the absolute run
path is elided below because `scripts/check_no_home_paths.sh` forbids home paths in
committed files):

```
runs <repo>/runs  rows 109  margin_threshold 0.0050
  WARNING: 66/109 rows {'synth': 66} are 'train' under pi_run.manifest.split_of ... and NOT
  'train' under pinq_train.split.bucket ...
  not exported: split_refused=15
sft   data/rl/sft.jsonl  n=43  suites=['synth']
      train_id_set_hash=263ae507f7a25db5  refused={'SplitViolation:synth': 66}
pairs data/rl/pairs.jsonl  n=0  suites=[]
      0 pairs: a preference pair needs TWO candidates at one state.
```

**`state_text` is the prompt the policy actually saw**, re-rendered from the recorded artifacts
through `pinq.promptlib` — not a summary and not an approximation:

- `Evidence` is deduplicated and sorted by uid, so the evidence block is a function of the
  retrieved uid **set**, which is exactly what `Turn.subset_hash_before` records;
- `Turn.response_sha` holds the Drafter's actual reply text, so `render_history` reproduces the
  Q/A history verbatim;
- **`state.draft` is NOT `None` inside the loop, and this line used to say it was.**
  `pinq.loop` drafts on every turn so the next `act()` can see `D_t` — that is the two-agent
  mechanism — and stamps `draft_sha` on the turn. Measured on the recorded runs: 149 of 335
  turns carry one, 26 of them at turn 0. Rendering `(no draft yet)` for those would make
  `state_text` something the policy never saw on 44% of turns, so `render_state` now REFUSES
  such a turn instead. The draft text is not persisted (only its sha), so to export those turns
  you must first persist `Turn.draft_text` the way `response_text` already is, or resolve it
  from the response cache by `request_sha`.

The reconstruction is **checked**: `render_state` re-derives `subset_hash_before` and refuses
the run on a mismatch. All 109 rows above passed that check. `--no-verify-state` exists and
should not be used.

**`--tau` and `--sigma-j` are measurements, not knobs.** `tau` is the 35th percentile of pilot
`phi_LOO` (`pinq_train.reward.tau_from_pilot`) and `sigma_j` is the judge's measured standard
deviation. The acceptance rule is

```
accept the argmax candidate  iff  value > tau + 1.5 * sigma_J * sqrt(2)
```

— the `sqrt(2)` because the thresholded quantity is a *difference* of two judged values and so
carries two independent noise draws.

### 3.1 The STOP target is a gold question, not a sampling accident

This section used to end: "Below the floor the target is **STOP**: 'this question was not
measurably better than not asking' is what a tie should teach." **That was wrong**, and it is
worth stating plainly because it shaped the dataset. A tie among the eight sampled candidates
is a fact about the *sampler*. On a state where required evidence is still missing and all
eight happened to ask badly, the rule taught the policy to stop exactly where it should have
kept going — and `stop_share` on the live SFT reached 0.668 largely that way.

Whether stopping is right is a property of the **task**: was the required evidence already in
hand *before* this decision? Rows carry that now (`coverage_before`, `done_before`, computed
gold-side in `pi_run.cmd_train`; the episode-final `evidence_coverage` cannot answer it,
because it calls a state done exactly when the ASK taken there is what completed it). The rule
is `stop_rule = "gold_coverage_v1"`, recorded in the manifest:

| state | target | `label_rule` | counter |
| --- | --- | --- | --- |
| done before the decision | STOP | `stop_done` | `n_stop_done_before_dedupe` |
| not done, best ASK clears the floor | that ASK | `ask_clears_floor` | — |
| not done, nothing clears the floor | **no row** | — | `n_no_target_dropped` |

The third row is the point: there is no target to teach, and rung 3's reward already punishes
the undershoot. `frontier_size == 0` is a *second opinion* (a matcher statement over uids
against coverage's statement over required units) and is counted in `n_done_frontier_disagree`,
never decisive. `done_before` absent is UNKNOWN, never "not done".

### 3.2 One STOP shape, and ASK-vs-STOP pairs

There were **four** serialisations of one action (the prompt's lowercase `stop`, the exporter's
compact `{"action":"STOP"}`, the convlog shard's invented `"rationale": "report"`, and an
unreachable branch in `cmd_train`). A DPO loss that sees chosen and rejected side by side can
learn the *separator style* as a cue for stopping, so the shape is now one constant in
`pinq.actions` (`STOP_ACTION_JSON`), with `ask_action_json` and `action_kind_of` beside it.

A recorded STOP is now a **row**: `pinq.loop` breaks on a `Stop` without stamping a `Turn`
(correct — it records what was retrieved), but `status.json` carries `stop_reason` and the last
turn carries `subset_hash_after`, so `rows_from_run` renders and verifies the state the STOP was
decided in. Before this, the 8.4% of fork candidates that stopped at the fork turn (1,988 of
23,583) were invisible to both exporters. **A budget or `max_turns` stop gets no row** — that is
the harness's decision, not the policy's, and teaching STOP there teaches the cap.

`export_pairs` therefore emits two `pair_kind`s:

- **`ask_ask`** — two questions, as before.
- **`ask_stop`** — a question against stopping. STOP **never** wins on value: its value is 0.0
  by construction, so on a state whose questions all came out negative a value comparison hands
  STOP every pair and teaches "stop whenever asking is expensive". The STOP direction needs a
  gold reason (`done_before`, or the outcome key); the ASK direction needs the same noise floor
  as §3.1. Neither supported → no pair (`n_ask_stop_undecided`).

The **length guard does not apply to `ask_stop`** and cannot: a STOP is 18 bytes and a question
is 40–120 characters. The asymmetry is what the two actions *are*, so it is made visible
(`pair_kind` on every row, a per-kind split in rung 2's report, `DPOConfig.include_pair_kinds`)
rather than hidden behind a guard that would delete the category. It is still enforced for
`ask_ask`. A state that gold says was done and where no candidate stopped gets **one**
synthesised STOP pair, outside `MAX_PAIRS_PER_STATE`, marked `stop_source="synthesised"`.

**`gold_coverage_v2`: the not-done contrast.** The rule above has a hole, and §5.3.1 measures
what it cost: every rung-1 checkpoint scores `P(ASK | not done)` 0.83–0.91 against the prompted
base's 0.96 — they stop at 9–17% of states where required evidence was still missing. Nothing
in the corpus ever said *"you are not done, so do not stop, even though no sampled question was
measurably good"*. §3.1's third row is why: those states clear no noise floor, so the SFT rule
drops them (24,538 of them on `data/rl/sft.manifest.json`, `n_no_target_dropped`) and the pairs
rule sees only whatever the candidates happened to sample. That refusal is right about an
*imitation target* — the right question there is one none of the eight candidates asked — and a
preference asserts strictly less than a target does: not "ask this", but "asking beat stopping
here", which `done_before is False` establishes on its own.

`pi train export --kind pairs --stop-rule gold_coverage_v2` adds exactly that mirror. At a state
gold says was **not** done, where at least one candidate ASKed and **no** candidate stopped, it
emits **one** pair: the best-available ASK (highest `value`, tie-broken on a digest of the
candidate's identity so input order cannot decide it) **chosen** over `STOP_ACTION_JSON`
**rejected** — `pair_kind="ask_stop_synth"`, `stop_source="synthesised_notdone"`,
`decided_by="not_done"`, `label_source="rule"`, outside `MAX_PAIRS_PER_STATE` and exempt from
the length guard like every other STOP kind. Three guards, each for a reason already in this
section: `done_before is None` is **unknown, never "not done"**, so it emits nothing; a state
where a candidate *actually stopped* keeps its recorded pair (or the rule's recorded refusal to
order it) and gets no synthesis, because marking a side `synthesised` whose bytes a rollout did
emit is a false provenance claim; and a not-done state with no ASK at all emits nothing, since
there is exactly one way to stop and no canonical way to ask.

| manifest field | what it counts |
| --- | --- |
| `stop_rule` | `gold_coverage_v1` or `gold_coverage_v2` — which rule wrote this file |
| `n_stop_pairs_synth_notdone` | v2 pairs emitted; outside the cap, so this equals the row count |
| `n_notdone_states_no_ask` | not-done states where every candidate stopped → nothing emitted |

`n_stop_pairs_synth_notdone` is deliberately **not** folded into `n_stop_pairs_synth`: that one
is STOP-chosen-because-done and this one is ASK-chosen-because-not-done, and the two pull the
policy in opposite directions. Both carry `pair_kind="ask_stop_synth"` — `pair_kind_of`
re-derives the kind from the payloads and a derived constant is a derived constant — so
`stop_source` is the only witness to the direction. A consumer that wants the corpus as it was
can exclude the single key `decided_by="not_done"` (`DPOConfig.exclude_decided_by`).

**v1 remains the default of record** until the user switches: every artifact under `data/rl/`
was exported under it, and a v2 export is a *different dataset* that must not share a manifest
with one. The flag touches `export_pairs` only — `export_sft` has no branch it changes, so under
`--kind both` the SFT manifest still reads `gold_coverage_v1` and the command prints a NOTE
saying so, because two manifests in one directory naming different rules otherwise reads as a bug.

### 3.3 `--rank`: two exports, and only one of them can measure the claim

```bash
pi train export --kind pairs --rank outcome        # -> pairs.jsonl              (CONTROL)
pi train export --kind pairs --rank anticipation   # -> pairs.anticipation.jsonl (TRAIN ON)
```

`_anticipates` ranks a candidate that took a need *at the moment it became nameable*
(`newly_reachable`, a trajectory property) above one that did not — below outcome, above speed,
since speed is a proxy for the same thing. It is a flag rather than the default because
`scripts/validate_pairs.py` reports latent pursuit as an **emergent** property of ranking on
outcome, speed and gain; the moment anticipation is a ranking key that headline is true by
construction. The two orderings never share a filename or a manifest, `rank_rule` is recorded,
and `validate_pairs` reads it and labels its own output accordingly. `_anticipates` is a no-op
at turn-0 forks (a depth-0 need has no prerequisite) — about 46% of live states.

### 3.4 The retrieval budget on a fork

`sample-candidates` copies `budget_cap`/`max_turns` from the parent manifest unless
`--budget-cap`/`--max-turns` say otherwise. Overriding them is legitimate *because the policy is
budget-blind by type* (`Inquirer.act(s)` takes no ledger, and the cap is a `FORBIDDEN_PLACEHOLDER`
in every prompt), so the state at the fork turn is cap-invariant; only the **episode** labels
depend on the cap. Both values are in `semantic_hash`, so a re-fork at a larger cap is a distinct
run that cannot collide with the smaller-cap candidates.

The exporter therefore **refuses to pair across cohorts** (`n_cross_cap_dropped`): a cap-24
candidate beats a cap-8 twin on `answer_correct` and `turns_to_complete` by construction, and
`pins_sha` does not carry the cap. Both values ride on every row and every pair.

**Preference pairs need candidate sampling.** A finished sweep records one action per state, so
`export_pairs` correctly yields zero. Pairs come from rung 1's N=8-per-state candidate rollouts.

---

## 4. Rung 0 — GEPA prompt search (no GPU, ships)

A genetic-Pareto search: evaluate candidates on a fixed task set through the seam, keep every
candidate that is best on **at least one** task, sample parents in proportion to how many tasks
they own, and mutate by showing an LLM the parent prompt plus its worst measured traces.

**Why a Pareto front and not an argmax on the mean.** Averaging over tasks throws away the fact
that different prompts win on different tasks. A mutation that fixes ten hard tasks and costs a
hair on forty easy ones loses on the mean and is exactly the mutation worth keeping.

**Cost is printed before anything is spent.** A search that discovers its own price after eight
hours has already spent it.

```
$ .venv/bin/pi train rung0 --dry-run
rung 0 (GEPA prompt search) -- PROJECTED COST, before any rollout
  rollouts          4,240
  prompt tokens     1,072,720,000
  completion tokens 37,312,000
  rates             $0.15/Mtok in, $0.75/Mtok out  [2026-08.json v2026-08]
  USD, nothing cached   $188.89
  USD, 27% prefix cached  $145.45
  (a ceiling: the naive per-rollout token draw takes no credit for the response cache)
  config_sha        ab1fb2ba29f2c036
  seed prompt       inquirer_prompted.txt  sha 814658eaa2f09eb7

dry run: nothing was rolled out and nothing was spent.
```

Without `--dry-run` and without `--yes` it prints the same block and **exits 2**. Candidate
counts are `--candidates`, `--generations`, `--n-tasks`, `--n-val-tasks`, `--minibatch`.

Two structural guards, applied before any tokens are spent (`accept_mutation`):

- a proposal that **drops a `{{placeholder}}`** is rejected — it would render an *empty*
  evidence or history section, a silent ablation nobody declared, which `render()` cannot even
  raise on because the hole is gone rather than unresolved;
- a proposal that **adds a forbidden placeholder** (`budget`, `cap`, `turns_left`, `arm`) is
  rejected — `Inquirer.act(s)` being budget-blind by type is worth nothing if the string handed
  to the model announces the cap.

**A candidate reaches the policy as a file.** `RolloutRequest.prompt_overlay` points
`pinq.promptlib` at a directory of candidate templates, so the candidate is rendered by the same
policy class through the same loop as the shipped template — and because `promptlib.sha()`
hashes the bytes actually loaded, a candidate rollout carries different `prompt_hashes` and a
different `run_id`. A candidate can never be confused with a baseline rollout.

**The reflector was verified live, once** (2026-08-24, `aws/gpt-oss-120b` through the remote
LiteLLM proxy). One call, parent = the shipped `inquirer_prompted.txt`, two synthetic measured
traces:

```
model: aws/gpt-oss-120b base_url: https://litellm.gateway.example.com
returned chars: 2341
accept_mutation -> True | ok
```

That is one call and no more; the search itself has not been run. Note the reason the call
returned content at all: gpt-oss bills **reasoning tokens inside `max_tokens`**, so
`OpenAIReflector` multiplies its request by `REASONING_HEADROOM = 3`. Without it a request for
1,200 content tokens comes back with 1,200 reasoning tokens and an **empty** content field, and
`accept_mutation` correctly rejects it as `empty proposal` — the mutation silently becomes "no
change" and a generation is wasted paying for it.

**Where the winner goes.** `conf/prompts/rung0/inquirer_prompted.txt`, plus a manifest carrying
the config sha, the lineage and the front. It is an **overlay directory**, not an edit to
`src/pinq/prompts/`: the untrained, trained and frontier families all point
`PI_PROMPT_OVERLAY` at it, so the three share one prompt by construction.

**One honest deviation from the plan.** The plan says rung 0 is optimised on musique+wiki2 *dev*
and validated on drgym dev. The seam refuses any task outside `train`
(`ScoreRequest.split_assert` is `Literal["train"]`), so this implementation optimises on train
and breaks Pareto ties on a **held-out slice of train** (`--n-val-tasks`). Selecting on dev
would require a scoring endpoint that answers for dev, and such an endpoint, once it exists, is
reachable by every later rung too. The winner is validated on dev afterwards through the
ordinary gold-side eval path (`pi run` + `pi score`), which no trainer can call.

**Go/no-go into rung 1.** Ship the winner only if its held-out mean beats the seed prompt's by
more than the paired noise floor, *and* it passes `pi run` on a dev slice without an increase in
malformed-action rate. A prompt that scores better in the search and parses worse in the loop
has moved the measurement, not the policy.

---

## 5. Rung 1 — rejection-sampling SFT

```bash
.venv/bin/pi train tokstats --dataset data/rl/sft.jsonl --tokenizer <pin>   # sets --max-seq-len
.venv/bin/pi train rung1 --base-model <pin> --dataset data/rl/sft.jsonl \
    --tau <measured> --sigma-j <measured>            # preflight only, no GPU needed
.venv/bin/pi train rung1 ... --train --acknowledge-untested    # needs CUDA
```

**A preempted run resumes; it does not restart.** Both rungs call
`trainer.train(resume_from_checkpoint=...)` with the highest-numbered `checkpoint-N/` under
`--out` that carries a `trainer_state.json` (`pinq_train.resume.latest_checkpoint`; a directory
without that file is a job killed mid-write and is skipped). `--save-steps` sets the cadence —
200 by default, ~3,200 rows at the shipped effective batch of 16 — which is what bounds the loss
from a preemption; `--no-resume` starts fresh, which is what a changed dataset or changed
hyperparameters need. `save_steps` is **in** `cfg.sha` because it changes what `--out` contains;
`resume` is **not**, because a resumed run and the run it continues are one experiment and must
carry one identity. Which checkpoint was picked up is recorded as `resumed_from` in the rung's
manifest, beside the config rather than inside it. This exists because the cluster's reservation
is shared and jobs are preempted (`docs/HPC_RUNBOOK.md` §5, §7); rung 1 is ~180 GPU-h.

**The loss mask is the whole rung.** An example is a multi-thousand-token state followed by a
short JSON action. The state was written by the harness and the Drafter; only the action was
written by the Inquirer. An unmasked loss spends better than 95% of its gradient teaching the
model to reproduce retrieved documents, the checkpoint drifts toward a document language model,
and the evaluation shows a policy that got *worse at deciding* for reasons no metric in the
paper can name.

`tests/test_train_rungs.py::test_loss_mask_sums_to_the_inquirer_json_length` asserts on a
**real recorded decision point** (`tests/fixtures/train/golden_episode.json`, copied from
`runs/dev-97f3de…`) that the number of supervised positions equals the tokenizer length of the
Inquirer's JSON and not one token more.

### 5.1 The trainer and the server must tokenize the same bytes

`chat_template=True` is the default. `build_chat_masked_example` renders the state as one **user**
message and the action as the **assistant** turn, through the model's own chat template, with
`enable_thinking=False` passed to *both* renderings. The prompt rendering must be a strict prefix
of the full one — `TemplateNotPrefix` otherwise — because the mask is built from the prompt's
*length*, and a template that rewrites the join would supervise part of the chat header while
every count in the report still looked correct.

Why it matters: vLLM serves this policy through `/v1/chat/completions`, which adds role headers
and an end-of-turn token. Training on the raw concatenation and serving the templated string
measures every dev NLL on a prompt the policy never sees — and that discrepancy raises no error
anywhere. Nothing is appended to the target: the last supervised token is whatever end-of-turn id
the template emitted, **not** `eos_token_id`, which on some models is a different token the
server's stop criteria do not watch.

Two consequences follow:

- **Over-length rows are refused, not truncated** (`OverLength`). `truncate_left` drops tokens
  from the front, which on a templated sequence is `<|im_start|>user`; a row without its header is
  a row the server can never produce. `train()` counts them as `n_over_length_refused` in the
  manifest. Size the window with `pi train tokstats` first — and note its default `--field
  state_text` measures the **state alone**, with the action and the header on top.
- **Packing is off, and `validate()` refuses packing together with the template.** A packed window
  lets example *i+1* attend to example *i*'s evidence, which is not a sequence the server could
  ever render.

The one property no laptop can check — that these ids equal a real server's — is
`tests/test_chat_template_identity.py::test_serving_tokens_equal_training_tokens`, skipped unless
`PI_VLLM_URL` is set. Run it against the rented box **before spending anything**.

### 5.2 The rows are weighted the way the exporter weighted them

`sample_weight` is `1/sqrt(rows in that task and kind)`, normalised to mean 1 within the kind.
Three musique tasks contribute 107 ASK states each; unweighted, the policy learns those three
tasks' phrasing. The alternative — a per-task cap — deleted 65% of the ASK rows, every one a state
no kept row represented.

The batch loss is `sum_i w_i L_i / sum_i w_i` with `L_i` the per-example token-mean CE over the
supervised positions. **This does not decompose across micro-batches**, and at the shipped
hyperparameters (batch 1 × accumulation 16) each micro-batch holds exactly one example, so a naive
per-micro-batch weighted mean computes `w·m/w = m` and the weights cancel *exactly* — a silent
no-op with an unchanged loss curve. `accum_scale` therefore normalises by the accumulation
**window** (`per_device_batch × grad_accum`), which is valid only because the dataset's mean weight
is 1; `test_the_exported_dataset_has_mean_weight_one` is the measurement that licenses it, and
`n_unweighted_rows` in the preflight report counts the rows that were defaulted to 1.0 rather than
measured. `num_items_in_batch` is honoured by its *presence*, which is what tells the helper
whether `Trainer` will divide by `gradient_accumulation_steps`; its value is a token count and our
normalisation is per example, so the value itself is not used.

Config, from the plan: LoRA `r=32 alpha=64` (`alpha = 2r` is a coupling, not two free knobs — a
rank sweep must not also sweep the effective learning rate), `lr 1e-4`, 2 epochs, bf16, gradient
checkpointing, `group_by_length=False` (length-ordered batches are a curriculum nobody chose).

The `TrainingArguments` kwargs are built by `training_argument_kwargs`, which is pure and is tested
against both field sets without transformers installed. `[train]` allows `transformers>=4.42`, i.e.
both sides of the 5.0 break, and 5.0 removed two of the names this trainer passed: `warmup_ratio`
(folded into `warmup_steps`, which is now a float where a value in [0, 1) is a ratio — so it is a
branch and **not** a rename, because `warmup_steps=0.03` on 4.x truncates to no warmup at all) and
`group_by_length` (gone outright, so under 5.x the key is simply not emitted). It also passes
`remove_unused_columns=False`: `_Rows` is a plain torch `Dataset`, so the Trainer otherwise wraps
the collator in a `RemoveColumnsCollator` keyed on `inspect.signature(model.forward)` and drops
`sample_weight` — the one column the weighted loss exists to read — before `compute_loss` sees it.

`SFTConfig.validate()` refuses an empty `base_model` and an unmeasured `tau`/`sigma_j`.
`include_arms` defaults to `{inquirer_prompted}` and is **recorded** in `cfg.sha` and the manifest;
the exporter is what enforces it, since this rung sees rows rather than runs.

### 5.2.1 The loss is computed only where it is supervised

The mask leaves exactly the **action** supervised, and the action is the tail of the sequence, so
every logit before it is multiplied by zero. `_weighted_ce` nevertheless materialised `[T, V]`
three times — the slice, the `view`, and cross_entropy's own copy — and the model's own
`ForCausalLMLoss` added a fourth by calling `logits.float()` on the whole block. With
V = 151,936 one of those tensors is 3.1 GB at T = 5,120.

`forward_for_loss` passes `logits_to_keep` (transformers ≥ 4.45) so the model slices the hidden
states *before* the LM head, and drops `labels` from the model call — the model's own loss was
thrown away here anyway, and shifting full labels against a tail of logits is a shape error.
MEASURED, Qwen3-0.6B on CPU in fp32, one forward + backward at T = 5,120 with a 128-token action,
peak RSS in a fresh process per arm:

| arm | logits | peak RSS | loss |
| --- | --- | --- | --- |
| full | `(1, 5120, 151936)` | 32.667 GB | 10.118711 |
| kept span | `(1, 129, 151936)` | 15.527 GB | 10.118711 |

17.14 GB, 52.5%, same number to every printed digit. Three things keep it a correctness change
rather than a performance one: the span is `max(n_action) + 1` **read off the labels** (the `+1`
is the shift — keeping `n_action` exactly drops the first supervised token silently);
`weighted_ce` aligns the labels to the logits the model actually **returned**, so a wrapper that
swallows the kwarg is merely not optimised; and a tail that does not line up raises. Where the
kwarg is absent (`transformers` 4.42–4.44) the old call runs unchanged.

### 5.2.2 The grids of record, and why they are files

`conf/train/rung1_8b.json` and `conf/train/rung2_8b.json` are the 8B run's hyperparameters, with
a `_why_*` field beside every number saying where it came from. `--config PATH` pre-fills them and
**any flag overrides it**, so a sweep is reproducible from the command line that ran it:

```bash
pi train rung1 --config conf/train/rung1_8b.json --base-model Qwen/Qwen3-8B --tau <measured> --sigma-j <measured>
pi train rung2 --config conf/train/rung2_8b.json --base-model Qwen/Qwen3-8B --adapter artifacts/rung1
```

A key nothing consumes is **refused**, not ignored: a typo in a grid file would otherwise train at
the default under a `cfg.sha` that names the value nobody applied — the same failure as a flag
nothing forwards, one layer up. Keys starting with `_` are documentation and are skipped.

`max_seq_len` is **5,120** in both, and that is a measurement rather than a round number: on the
real tokenizer the longest exported state is 4,963 tokens and the longest templated example ~5,045,
so 5,120 is the next multiple of 1,024 that truncates nothing. The previous 12,288 was a guess and
cost 2.4× the activation memory of the longest row that exists. `max_prompt_len` is 4,608 — well
above any action, since a question is 40–120 characters — and on trl 1.x it is *recorded rather
than applied* (`prompt_length_hook`; there is no separate prompt budget there).

Neither file names `base_model`, and rung 1's names neither `tau` nor `sigma_j`: there is no
default base model, and the two thresholds must be measured before the dataset is built.

The flags each rung now exposes, all of which reach `cfg.sha`:

| rung 1 | rung 2 |
| --- | --- |
| `--config`, `--epochs`, `--max-seq-len`, `--learning-rate`, `--per-device-batch`, `--grad-accum`, `--lora-r` (alpha is 2r and is not separate), `--bf16/--no-bf16`, `--chat-template/--no-chat-template` | `--config`, `--epochs`, `--max-seq-len`, `--max-prompt-len`, `--learning-rate`, `--per-device-batch`, `--grad-accum`, `--beta`, `--bf16/--no-bf16`, `--include-pair-kind` (repeatable), `--reference` |

### 5.2.3 The `--stop-weight` knob

`pi train rung1 --stop-weight <float>` (`PINQ_STOP_WEIGHT` in `scripts/hpc/rung1.sh`) multiplies
**every STOP row's `sample_weight`** by the given value before the accumulation window is
normalised (`SFTConfig.stop_weight`, `row_weight` in `src/pinq_train/rung1_sft/train.py`). STOP
is decided off the **action bytes** — `pinq.actions.action_kind_of(row["action_json"]) == "stop"`
(`is_stop_row`) — the same call the exporter's `_is_stop_row` makes, because those bytes are what
the mask supervises; measured on `data/rl/sft.jsonl` (43,837 rows, 2026-09-15), the three
candidate fields (`label_rule == "stop_done"`, the `is_stop` flag, and the action bytes) agree on
all 43,837 rows, so which field decides is a choice about which field cannot drift, not about
today's corpus.

**Unset is not `1.0`.** `stop_weight` sits in `SHA_OMIT_WHEN_NONE`, so at its absent default it
is dropped from `cfg.sha` entirely and every already-trained rung-1 run's identity is unmoved;
set it to any value, 1.0 included, and it enters `cfg.sha` like any other field — "the knob was
considered and set to a no-op" is a different run from "the knob did not exist." The motivating
observation (commit `2aa419d`): every rung-1 checkpoint stops far earlier than the prompted base
when run live (mean asks 3.0–3.5 vs 6.25 on musique dev, 1.3–1.5 vs 5.6 on strategyqa) and loses
evidence coverage doing it, and lowering the STOP share of the *file* from 68% to 39% (the
headline export) did not change that — leaving "the STOP rows take too much of the gradient" as
the other lever, which is what this knob tests directly.

**What preflight reports.** `stop_loss_share(rows, use_sample_weight=, stop_weight=)` is called
by both `preflight` and `build_examples`, so the number recorded beside the checkpoint is the
number it actually trained on, not a second implementation of it. It returns:

- `stop_share_of_loss` — the STOP share **of the weighted loss**, which is not the STOP share of
  the file: the exporter normalises `sample_weight` to mean 1 within each kind, so the two are
  equal only at the absent default (measured on `data/rl/sft.jsonl`: 0.6784679608549855 of rows,
  0.6784679608549308 of the loss); a `stop_weight` separates them on purpose.
- `mean_row_weight` — reported beside it because `accum_scale` normalises the accumulation window
  by `per_device_batch × grad_accum` on the strength of the dataset's mean weight being 1; a
  `stop_weight` breaks that by a uniform factor (the `SFTConfig` docstring gives ~0.66 at 0.5 on
  the pooled 68%-STOP file; commit `2aa419d` measured ~0.80 at 0.5 on the 39%-STOP headline file)
  that AdamW's per-parameter normalisation largely absorbs but that is now a recorded number
  rather than an assumption. The per-example *ratio* the ablation is actually about is untouched
  by it.
- `n_stop_by_action_json` — beside `dataset_report`'s own `n_stop` (the `is_stop` flag count), so
  the two predicates are visible in one report.

**The three refusals** (`SFTConfig.validate()`), all at `pi train rung1` invocation time, before
any GPU-hour is spent:

1. `stop_weight <= 0` — zero would delete every STOP row from the gradient while leaving it in
   every count the manifest reports; a negative weight would train the opposite of what those
   rows say while the loss curve still goes down.
2. `stop_weight` set but `--no-sample-weight` (`use_sample_weight=False`) — the trainer would be
   the stock `Trainer`, handed no weight column at all: the knob would be in `cfg.sha` and the
   manifest and reach no gradient.
3. `stop_weight` set but `pack_sequences=True` — a packed window concatenates rows and carries no
   single row's weight, so it trains at 1.0 by construction and the multiplier would silently
   apply to nothing.

### 5.2.4 The curriculum knob

`pi train rung1 --curriculum {shuffled,depth_easy_first,depth_hard_first}` decides **in what
order** the rows are shown (`SFTConfig.curriculum`, `order_examples` in
`src/pinq_train/rung1_sft/curriculum.py`). `shuffled` is the default, the headline recipe and
the null arm: every epoch is a full shuffle, which is what every rung-1 run of record trained
under. The other two order the **first** epoch by the depth of the need the supervised ASK
resolved and shuffle every epoch after it.

**The axis is the target's depth, not the state's.** A row's stratum comes from `latent_depth` —
S0 = depth 0, S1 = depth 1, S2 = depth ≥ 2 — which for an ASK row is the gold node that ask
resolved, i.e. the thing the policy had to anticipate. `latent_depth == -1` is a *sentinel* and
not a depth (no required node was resolved at that turn), so those rows join S0, where they are
the same "nothing deeper was needed here" case as a genuine root; a missing or unreadable label
is the same case. Three strata rather than one per depth because depth ≥ 3 is 4.6% of the ASK
rows, and a stratum that small has its first-epoch position decided by sampling noise.

**STOP rows are dealt, not read.** A STOP row supervises no question, so it has no target-need
depth; its `latent_depth` describes the state it stopped in. If it were read as a depth, the STOP
rows would pile into one stratum and epoch 1 would front-load them — the arm would then be an
ordering intervention *and* a STOP-share schedule, reported as one number. They are assigned at
random (a uniform permutation, seeded by `SFTConfig.seed`, because a STOP row carries no depth
signal to decide it by) in **proportion to each stratum's ASK mass**, which is the deal under
which every stratum carries the file's STOP share exactly: with `a_k` ASK rows in stratum k and
`s_k = S·a_k/A` STOP rows dealt to it, the share is `(S/A)/(1 + S/A) = S/(A+S)` for every k, for
any depth distribution. An equal three-way deal would not do this — measured on the unbalanced
3,000-row corpus in `tests/test_curriculum.py`, proportional lands every stratum within 0.00024
of the file's 0.400000 while equal gives 0.286 / 0.444 / 0.571, off by as much as 0.171.

**The suite is not an ordering key.** Rows of every domain are shuffled together inside a
stratum: a suite-blocked epoch would be a domain curriculum wearing a depth curriculum's name,
and nothing the checkpoint records could tell them apart afterwards. Measured at seed 0 over
epoch 1 of `depth_easy_first`, the longest single-suite run is 7 against the log₃(3000) ≈ 7.3 a
uniform shuffle predicts.

**Exposure is identical in all three arms** — every row appears exactly once per epoch, i.e.
exactly `epochs` times — so the arms differ in order and in nothing else. A schedule that
re-weighted by repeating a stratum would be a different dataset, and the difference it bought in
the checkpoint could not be attributed to ordering. Only the **first** epoch is ordered; epochs
2..n are full shuffles, because repeating the ramp every epoch would confound "shallow first"
with "shallow n times, each time right before the deep rows".

**How the epoch boundary maps onto the sequence.** `Trainer` draws a fresh `RandomSampler`
permutation at every epoch boundary, so an order handed to it survives exactly one epoch. Under a
curriculum the whole `epochs × N` sequence is therefore materialised as **one** dataset in
`train()`, `num_train_epochs` is **1** (`training_argument_kwargs`), and the trainer class gains
an `_Ordered` mixin whose `_get_train_sampler` returns a `SequentialSampler`. The LR schedule is
the same schedule: total optimiser steps are `floor(epochs·N / (batch·accum))` against the null's
`epochs · floor(N / (batch·accum))` — equal when that division is exact and within `epochs − 1`
steps of it otherwise — and `learning_rate`, `warmup_ratio` and the checkpoint cadence are
untouched. `n_train_sequences` is recorded beside `n_packed` in the manifest so the multiplication
is a number rather than an inference. `group_by_length` stays `False` under every curriculum:
ordering by evidence size is a second ordering intervention, and an arm that ran both cannot be
read.

**Unset is the null arm's own name.** `curriculum` sits in `SHA_OMIT_WHEN_DEFAULT` rather than
`SHA_OMIT_WHEN_NONE`, because its "off" is not `None` — `shuffled` is a real schedule with a
spelling and a row in the table. At that default it is dropped from `cfg.sha` entirely and every
already-trained rung-1 run's identity is unmoved (measured at the pre-change commit `f3f3f0f`:
`SFTConfig(base_model="Qwen/Qwen3-8B", tau=0.05, sigma_j=0.01).sha` is
`44eee2b7d306bdb27ae7bc3b9626c847dfb523f1f0d8b5c21bae4f83b1fdcc08` before and after). Unlike
`stop_weight`, there is no third state to record: "the knob was considered and set to shuffled"
and "the knob did not exist" produce the same rows in the same order from the same seed, so two
identities would split one experiment rather than distinguish two. Any other value enters
`cfg.sha` like any other field (`depth_easy_first` → `616b1960bcdf…`, `depth_hard_first` →
`69b4934119f9…` on that same config).

**The two refusals** (`SFTConfig.validate()`), both at `pi train rung1` invocation time:

1. a name that is not one of the three — refused rather than defaulted, because an unrecognised
   name would train the shuffled null arm while sitting in `cfg.sha`, in the manifest and in the
   preflight report as an ablation: the arm would measure a difference of zero and the zero would
   be believed.
2. a curriculum together with `pack_sequences=True` — a packed window is a first-fit
   concatenation of whole rows, so the row stops being the unit the trainer steps over and an
   ordering defined on rows cannot survive it.

**What preflight reports.** `curriculum`, unconditionally, beside `stop_weight`. Everything else
in that report — the row count, the STOP share, the diversity statistics, the weights — is
identical between an arm and its null by construction, so a preflight that did not name the
schedule would print the same screen for both and leave only a sha to tell them apart by eye.
`train()` adds `n_train_sequences` (the `epochs × N` the trainer's single pass holds) beside
`n_packed` (what one epoch of the file is).

`order_examples(rows, curriculum, seed, epochs)` is a pure function over plain dicts and is
exercised end to end without a GPU in `tests/test_curriculum.py`, including the wiring: the test
recovers the permutation the `Trainer` double was actually handed from its tokens and compares it
to the sequence the function computes.

### 5.3 The wall, and the gates as they are actually enforced

`preflight` raises `HeldOutDataset` on any row stamped with a split other than `train`. A row with
**no** `split` key is legacy and is waved through, counted as `n_rows_without_split`; an **empty**
split is refused, because it means the exporter looked and could not tell.

Two gates run on the **dataset**, both inside `preflight`, both raising rather than warning:

| gate | value | what it catches |
|---|---|---|
| within-task distinct-3 | **≥ 0.65** | the questions stopped depending on the state |
| STOP share | **≤ 0.80** | a dataset that trains a policy to stop immediately |

The within-task measure is the one that decides. Pooled distinct-3 is **reported beside it as a
diagnostic only**: it scores the gold answer key — a perfectly state-dependent policy by
construction — at 0.2250, so gating on it would reject exactly the data it exists to protect.
`SFTConfig.distinct3_floor` sets the floor the *pooled diagnostic* is reported against; it does
not add a second gate, and reading it as one would mean trusting a threshold that cannot fire.
Measured baselines for the 0.65 choice: gold 0.9985, our export 0.7787, ten canned templates
emitted regardless of state 0.6033, fully collapsed 0.2571.

The gates on the **checkpoint's own dev generations** — malformed-action rate against the prompted
baseline, within-task distinct-3, and the question-length equivalence test — are a separate
command: see `pi train gate`. The malformed gate passes at `rate <= max(baseline_rate,
MALFORMED_TOLERANCE)`, `MALFORMED_TOLERANCE = 0.005`, so a single stray event on an n≈300 dev
slice against a zero-event baseline (measured: 1/333 = 0.003 on `qwen3-8b-dpo-pooled-control`
strategyqa) is not read as a regression; the verdict's `malformed_tolerance` block records the
tolerance, the checkpoint's raw event count and run count, and both rates, so the rounding-level
case stays auditable rather than silently passing.

**An arm with zero runs refuses rather than gates (T19d, 2026-09-16):** if the checkpoint or the
baseline arm selects no `status='ok'` runs from its grid, `run_gate` raises `EmptyArm` — naming
the parquet dir, the grid, the arm and the suite — instead of computing every criterion against
the other arm alone, and the CLI prints `GATE REFUSED: ...`, exits non-zero and leaves `--out`
unwritten. Every verdict that IS written now also carries both arms' raw run counts,
`n_checkpoint_runs` and `n_baseline_runs`, in the top-level `selection` block.

**A partial arm refuses too (`IncompleteArm`, 2026-09-17):** having some `status='ok'` runs is
not the same as having the right ones. Today six of eight gate arms were only partially run —
a staging step upstream had selected checkpoint runs for some but not all of the baseline's
tasks — and only a human noticing kept them out of a verdict; `EmptyArm` above does not catch
this, since a partial arm is not empty, every criterion still computes over the paired subset,
and the printed verdict reads exactly as if it covered every run.

**Correction, same day — re-measured twice.** The first cut raised on `ck_key_set !=
ba_key_set` — set equality, either side missing a key the other has. A first re-measurement
against 156 real verdicts reported a common, legitimate counter-shape in 48 of them; both that
156 and that 48 were themselves wrong, and a second, careful re-measurement (pinned
2026-09-17T16:51:10+0300; every real, non-symlink `*.json` under `artifacts/gate`, recursive
and top-level — two subtrees, `8b1-rescored-t19c/` and `8b2-t19c/`, are entirely symlink farms,
`.json` entries included, and either double-count or under-count if walked naively) found 150
real verdicts, every one carrying a `pairing` block: **all 150** have a baseline arm with
strictly more keys than the checkpoint (`n_unpaired_baseline > 0`; one instance —
`qwen3-8b-sft.musique.json` — pairs 132, sets aside 132 unpaired-baseline, 0
unpaired-checkpoint, from a second baseline-only seed), and **zero** have a checkpoint key the
baseline lacks (`n_unpaired_checkpoint` is 0 in all 150). This is not 150 near-misses of the
same failure: baselines are deliberately run at more seeds than the checkpoints compared
against them, pairing intersects on the shared keys and records what it set aside, and that is
the design working — a reader who does not know this would misread 150 of 150 as an alarm
rather than the normal case it is. Only a checkpoint key the baseline lacks is the real hazard,
since that is a checkpoint run that would be silently excluded from every criterion with
nothing in the verdict distinguishing it from a complete gate — a hazard this check stays armed
for even though it has not yet occurred once in 150 real verdicts. `run_gate` now raises
`IncompleteArm` only when the checkpoint's key set is **not a subset** of the baseline's — a
baseline with extra keys is paired and counted exactly as before, never refused — checked once,
before a single criterion runs, at the same point as `EmptyArm`, and naming both arm names,
both counts, and a few example task ids on the checkpoint-only side of the gap. Skipped for a
fork grid (T16b), whose pairing key is `(foreign_trace_sha, n_prefix_user_turns, seed)` rather
than `(suite_id, task_id, seed)` and which `_fork_pairs` already refuses in its own shape.

**The count is checked exactly now (`UnderfilledArm`, 2026-09-17).** The subset rule above
does not catch a checkpoint that is short of its *own* grid's declared tasks but still a subset
of the baseline's (e.g. staged on 40 of a 50-task grid whose baseline happens to cover all 50)
— a subset is a subset, and `IncompleteArm` has nothing to say about it. The original incident
above was a shortfall in *how many* tasks ran, not a question of *which*, so that is exactly
what `run_gate` now checks: given `--grids-root`, it reads the checkpoint's own grid file — the
same read `_k_by_suite` already makes for `k_matches_baseline`, parsed as plain YAML data, no
new import (`pinq_train → pi_run` is unused rather than disallowed by any of the four
`import-linter` contracts, and this path never creates the edge regardless) — and compares the
checkpoint's paired, per-suite, distinct-task count against that grid's own declared `n_tasks`
(or `len(task_ids)`), raising `UnderfilledArm` when any suite comes up short and naming the
arm, the grid, and each short suite's paired-vs-declared count. What stays unreachable is the
missing tasks' *identity*: naming them needs the grid's corpus-ordered id list (`Grid.select`,
`data/corpora`), a dependency this module does not have and should not acquire, for the same
reasons `IncompleteArm`'s check gives above. **One honest residual, documented rather than
closed:** a grid that selects by `split` alone, with no `n_tasks` or `task_ids` cap, declares no
count to check against; the verdict's `task_count.reason` is stamped with that fact rather than
a guess, and this does not raise on that shape. No grid under `conf/grids/` has this shape
today — all 32 checked 2026-09-17 declare one or the other — so this is a residual against a
shape that does not yet exist, not a live gap; the null-stamp-with-a-reason discipline already
in force here (see `k_matches_baseline`, `_k_by_suite`) is the remedy if one ever does.

A related near-miss an hour later in the sibling lane: eight gate verdicts came back `passed:
false` on every criterion, and the cause was not the checkpoints — their `scores.parquet` held
zero rows because the score step had never run, against 107,135 rows in the reference, so every
criterion computed a value anyway and the verdict read as the checkpoint failing rather than as
never having been scored; `run_gate` now raises `UnscoredArm` the same way when either arm's
selected runs carry zero score rows, naming the arm, its run count, the score-row count found,
and the fix (`pi score` on that parquet). None of these three checks — `IncompleteArm`,
`UnscoredArm`, `UnderfilledArm` — adds a field to the hashed `criteria` block; each raises
before a verdict is built, so any counts they report belong beside
`n_checkpoint_runs`/`n_baseline_runs` in the unhashed `selection` block or, for `task_count`
itself, beside the similarly unhashed `k_matches_baseline`, and
`test_the_cap8_rule_reproduces_the_previous_verdict_bit_for_bit` still passes unchanged.

**Fallback if it does not clear the gates.** Fall back to rung 0's prompt, report the SFT attempt
as a **negative result with the distinct-3 number and the dev delta attached**, and drop
`inquirer_trained` from the trained grids rather than reporting it. A grid row for an arm that
failed its own gate is worse than no row.

### 5.3.1 The online STOP 2×2

`pinq_train.gate._stop_2x2` is part of the **online** Tier-B dev gate (`pinq_train.gate`, "does
this checkpoint earn a test-split rollout" — checkpoint *selection*, not measurement; run via
`pi train gate`). Unlike a run-level summary, **the unit is a decision point**, `(run, t)`, not a
run: one episode offers the policy `n_asks + 1` decisions, `t = 0..n_asks`, and collapsing them
into one 2×2 per run would hide, e.g., a run that asked six questions after it was already done
and then stopped — it would score 1.0 with the six wasted questions invisible. This mirrors the
offline Tier-A `stop_confusion` (§ above), so the two tiers count the same events.

- **Done axis.** `frontier_q#t >= DONE_AT` with `DONE_AT = 1.0 - 1e-12` — required-evidence
  coverage the policy *holds when it decides* at `t`, read from `scores.frontier_q#k` (built by
  merging each turn's retrieved uids in order; `runs.parquet` itself has no coverage column).
  This is the same threshold the export's `done_before` uses, so "done" cannot mean two things
  across the offline and online tiers.
- **Stop axis.** ASK at every `t < n_asks` (the episode has a turn there); STOP at `t = n_asks`
  **only** when `runs.stop_reason == "policy_stop"`.
- **Forced stops are excluded from the 2×2 and counted separately**, as `n_forced_stops`. Under
  `stop_reason` of `"budget"` or `"max_turns"` the *harness* halted the episode — `pinq.loop`
  breaks on the refused retrieval charge or exhausts its turn budget — and the policy was never
  consulted at that state, so it carries no decision. The run-level predecessor counted a forced
  halt as an ASK, which credited the retrieval cap with the policy's own judgement.

**Why `stop_undershoot` could not be the done axis, and was (commit `f348feb`).** The metric
`pi_eval.score` defines as `scores.stop_undershoot = max(0, k* - k_hat)`, with `k*` the argmax of
the prefix-coverage ladder, is **identically 0 on every run ever scored** — 1,395 of 1,395 on
each of the three gate parquets — because that ladder is monotone in `k` (each turn's retrieved
uids only add to a growing set), so `k* <= k_hat` always. Reading the done axis off it therefore
made the "not done" cell empty by construction on every checkpoint: every verdict reported
`n_not_done = 0` and `p_ask_given_not_done = NaN`, and the gate failed for "one cell has no runs"
rather than on an actual measurement. `pi_eval.score` itself is untouched — `stop_undershoot`
cannot be made non-tautological in question units for a run that never reached done, and
changing it is a scorer-hash decision, out of scope for a training-side module contract 3
already forbids from importing `pi_eval`.

All four cells are recorded, plus `n_states`, `n_runs`, `n_forced_stops`,
`n_skipped_no_coverage` (a state with no required gold evidence, hence no ladder point — absence,
not zero), and `mean_asks_after_done` (questions bought at states already done, averaged over
runs — reported, never gated).

**No `stop_2x2` cell from a verdict written before `f348feb` may be quoted.** Every such verdict
used the degenerate `stop_undershoot` done axis and reports `n_not_done = 0` /
`p_ask_given_not_done = NaN` regardless of what the checkpoint actually did; it is not a smaller
sample of the same quantity, it is a different, empty-by-construction one. Re-run on the three
existing gate parquets (scorer `12910e78bb46c73c`, graph `v1`, verdicts under
`artifacts/gate/rerun-stop2x2/`, every other criterion byte-identical to the old verdicts): the
base scores `P(STOP|done) 0.15`, `P(ASK|not done) 0.96`, with 1.66 (musique) / 3.42 (strategyqa)
mean asks after done; every rung-1 checkpoint scores `P(STOP|done) 0.88–0.95`,
`P(ASK|not done) 0.83–0.91`, with 0.02–0.10 mean asks after done — the over-stopping result,
now counted per decision point rather than per run.

**The 2×2 is no longer gated under `--coverage-rule matched_cost`** (D9 follow-up, 2026-09-15):
those very numbers are why — a base with `P(STOP|done) 0.15` has `P(ASK|not done) 0.96` *because*
it never stops, so "both cells ≥ the base" is unreachable by any policy that stops at all. Every
cell above is still recorded; only the gating moves, and stopping is selected by the N1 rule
outside the gate. See §5.3.2.

### 5.3.2 The matched-cost re-read

The tier-B gate above contrasts each checkpoint against the prompted base **at a fixed retrieval
cap (8)**, paired within task — and every rung-1 checkpoint loses on `evidence_coverage` there
while asking half to a quarter as many questions. That is not the paper's endpoint: the claim is
"more required evidence at **equal retrieval cost**", an ordered pair (Δ coverage, Δ retrieval
calls), and a cap-8 contrast charges the trained policy for stopping early while crediting it
nothing for the cost it saved.

`scripts/matched_cost.py` re-reads the same gate artifacts against a **matched-cost** comparator
instead:
```bash
PI_GOLD_ROOT="$PWD/data/gold" .venv/bin/python scripts/matched_cost.py \
    --gate artifacts/gate --out figures
```
Because `Inquirer.act(s: State)` takes no ledger, no context and no budget (§1, "the seam"), the
base policy's *k*-th question is a function of the state alone — so the first *k* questions of a
cap-8 prompted run are exactly the questions a cap-*k* run would have asked, and that run's
coverage after *k* asks is its terminal coverage. The base's own **prefix ladder** is therefore a
legitimate matched-cost comparator, and each checkpoint is read at its own natural stop against
the base prefix at the same *k*. (A budget-aware `Inquirer.act` would break this construction,
which is one more reason that parameter does not exist — see "Things that look like details".)

**Two provenance locks gate whether a table is printed at all**, not just whether one is
trusted:

1. **Instrument identity.** `verify_instrument` recomputes the *full* trajectory from the
   prefix ladders and compares it, value by value, against what the verdicts' own scorer wrote —
   `frontier_q#k` at every `k`, `evidence_coverage`, and `cad#d`/`cad_n#d` at every depth. Any
   disagreement raises `InstrumentMismatch` and no table is printed: a matched-cost delta
   computed with a slightly different matcher would look exactly like a result otherwise.
2. **Reproduction.** The ladder's cap-8 column (its last rung) must reproduce all twelve
   published verdict values in `artifacts/gate/*.json` to ten decimals, with identical `n` —
   what ties the matched-cost rows to the gate they re-read.

Measured on `artifacts/gate` (scorer `12910e78bb46c73c`, graph `v1`, `mechanical_v3`, seed 0,
cap 8, dev split), both locks held (23,003 `frontier_q#k` / 4,185 `evidence_coverage` / 10,305
`cad#d` values checked, 0 disagreements) and, trained at its own stop vs. the base prefix at the
same *k*, task-clustered BCa 95%:

| suite | evidence_coverage Δ |
| --- | --- |
| musique (n=132) | sft +0.042 [−0.012, +0.094]; headline +0.049 [+0.003, +0.092]; headline-stop +0.065 [+0.018, +0.105] |
| strategyqa (n=333) | sft +0.064 [+0.039, +0.089]; headline +0.080 [+0.050, +0.108]; headline-stop +0.059 [+0.033, +0.084] |

`cad_ge2`: all six matched-*k* intervals span 0. One extra base question (`k+1`) puts the base
back ahead on both metrics. Reading (commit `c4f36e1`): per question asked, the trained policies
retrieve as much or more required evidence than the base; what they lose at cap 8 they lose by
stopping, not by asking worse questions. Figure `figures/P13_matched_cost/` (`caption.md`,
`provenance.json` with the sha256 of every input) carries the full frontier and the complete
contrast table; this section does not restate it.

**Since D9 (2026-09-15) this comparator is the gate's own COVERAGE criterion**, not only a
re-read: `pi train gate --coverage-rule matched_cost` (the default; `cap8` restores the previous
rule) gates `evidence_coverage` on the trained checkpoint at its own stop *k* against the base's
prefix at the same *k*, stamps `coverage_rule` in the verdict, and reports the cap-8 delta it
replaces per suite in a `matched_cost` block as the ungated cost line — every other criterion,
`cad_ge2` included, is the cap-8 contrast exactly as before. **One criterion changes with it**:
under `matched_cost` the `facet_breadth` **no-loss criterion is report-only** (`facet_rule:
"report_only"`; its value, CI, *n* and the no-loss outcome it would have had all stay in the
JSON), because facet breadth scales with the number of asks exactly as the cap-8 coverage delta
does — the 4B trio loses −0.075 to −0.121 on it at cap 8 while being at or ahead of the base per
question asked — and unlike coverage it cannot be moved to matched *k*, since `scores.parquet`
carries `facet_breadth` as one terminal metric with no `#k` ladder. Tier B is checkpoint
*selection*, so gating it here would reject every trained checkpoint for asking less; the Tier-C
preregistered endpoint is untouched and remains a reported secondary at cap 8 with the
matched-cost caveat. **The difference from the script is the instrument lock**: `scripts/matched_cost.py` proves the ladder
is the verdicts' instrument by re-running `MechanicalMatcher` against gold on every prefix, which
contract 3 forbids a `pinq_train` module, so the gate checks instead the one thing parquet alone
can decide — that every baseline ladder it reads ends at that run's own `evidence_coverage` — and
raises `LadderInconsistent` rather than reporting a number if it does not. (This is also why only
coverage moves: `scores` carries a `frontier_q#k` ladder and no `facet_breadth#k` or `cad#d,k`,
so a matched-*k* C@d≥2 or facet breadth cannot be read from the parquet at all.) Measured on
`artifacts/gate/parquet_{sft,headline,headline_stop}/`, the gate's six matched-cost rows equal
P13's `matched_k` contrasts exactly (delta, both CI bounds, *n*, mean *k*), and under
`--coverage-rule cap8` every criterion of all six verdicts is reproduced value for value.

#### Three more criteria move with it (D9 follow-up, 2026-09-15)

Moving *one* criterion to matched cost left the verdict being decided by three others that fail
for the comparator, not for the checkpoint. Under `--coverage-rule matched_cost` **only** —
`cap8` is unchanged in all three, which `test_the_cap8_rule_reproduces_the_previous_verdict_bit_for_bit`
pins by sha256 — the gate now also:

1. **`cad_ge2` → report-only** (`cad_rule: "report_only"`). C@d≥2 is a recall over depth-≥2
   nodes, so it rises with the number of asks exactly as cap-8 coverage and facet breadth do; at
   cap 8 all twelve verdicts lose, −0.035 to −0.177, while asking a quarter to a half as many
   questions as the base. **At matched *k* the loss is not there**: P13's six 8B intervals all
   span 0 (musique +0.029 [−0.076, +0.129], −0.012 [−0.112, +0.088], +0.053 [−0.047, +0.147];
   strategyqa +0.038 [−0.038, +0.108], +0.059 [−0.024, +0.134], +0.005 [−0.081, +0.081]).
   **But the gate cannot compute that number.** `scores.parquet` carries `cad#d`/`cad_n#d` as
   *terminal per-depth* rows and no ladder over prefixes — only `frontier_q` is indexed by *k* —
   and a prefix C@d≥2 needs the node → `gold_depth` map, which is gold and which contract 3
   forbids a `pinq_train` module. It is not recoverable from the compacted tables either:
   `matches.parquet` gives the turn each node resolved, but the per-depth rows are *aggregates*,
   and on **114 of the 465 dev tasks more than one node → depth map reproduces every stored
   `cad#d` row**, with the resulting per-*k* spread reaching 0.48 of the metric's range at *k*=1.
   So the value, its CI, *n*, `no_loss` (the verdict the no-loss rule would give) and
   `cap8_cad_ge2_delta` all stay in the JSON, ungated, with `matched_k_lives_in` pointing at
   `scripts/matched_cost.py` / `figures/P13_matched_cost/provenance.json`.
2. **`length_equivalence` → one-sided non-inferiority** (`length_rule: "not_longer"`). The
   criterion exists for one confound: a **longer** question retrieves more by accident, so an
   unchecked length gain reads as a question-quality gain. A **shorter** question cannot produce
   that error — it can only make the checkpoint's own coverage and depth numbers harder to earn —
   so the lower test is dropped and the upper one kept unchanged (same 0.10 margin, same relative
   scale, same 0.05 one-sided bound). 8 of the 12 verdicts failed the two-sided TOST, **every one
   of them because the trained questions are shorter** (8B headline musique [−0.171, −0.100]; 8B
   sft musique [−0.239, −0.183]; the widest *upper* bound across all twelve is +0.092, inside the
   margin). The two-sided verdict is kept as `tost_equivalent`. Verbosity still fails: the upper
   bound is the TOST's.
3. **`stop_2x2` → report-only** (`stop_rule: "report_only"`). The **prompted base is a degenerate
   comparator on this axis**: P(STOP | done) is 0.1525 / 0.1460 at 8B and 0.0066 / 0.0081 at 4B,
   so P(ASK | not done) is 0.9642 / 0.9633 / 0.9985 / 0.9994 — ~1 *because* the base essentially
   never stops. "Both cells ≥ the base" therefore asks a real stopper to keep asking as often as a
   policy that cannot stop, and all twelve verdicts failed it on the second cell while every
   checkpoint beat the base on the first (0.88–1.00 against 0.007–0.15). Every cell, count and
   column is kept, with `both_cells_ge_baseline` holding the cap-8 verdict. **Stopping is still
   selected**, by the N1 rule — P(ASK | not done) rise ≥ 0.03 against the *reference checkpoint*
   (a comparator that can stop) and P(STOP | done) ≥ 0.85 — outside this gate.

Measured on the three 8B parquets and the three 4B ones, two suites each, under
`--coverage-rule matched_cost`: **7 of 12 now pass**, and the failures are `evidence_coverage` on
the four musique cells whose matched-cost CI still contains 0 (8B sft +0.042 [−0.012, +0.094];
the three 4B) and `malformed` on `qwen3-8b-sft.strategyqa`. Before this change all twelve failed,
every one of them on `cad_ge2` and `stop_2x2`.

### 5.3.3 Fork-grid checkpoint selection (T16b)

`pi train gate` auto-detects a **tau2 fork grid** — every selected checkpoint run carrying a
non-empty `foreign_trace_sha` — and, only then, adds two criteria: `fork_followups`, GATED,
paired mean follow-ups (checkpoint − baseline, `n_user_turns - n_prefix_user_turns`) with a
BCa CI clustered by fork point (seeds averaged within a fork point first), failing when the
lower CI bound is **> 0** (a confidently-measured rise in the turns a continuation needs); and
`fork_reward`, REPORT-ONLY (`gated: false`), the paired mean `tau_reward` delta over the same
pairs, carrying the same `fewer`/`more` counts as `fork_followups` so a reader sees whether a
reward gain was bought with more turns or fewer. Both replicate `pi_eval.fork_report` **by
value** (contract 3 forbids importing it) — `pinq_train.gate.follow_ups`,
`pinq_train.gate.sign_test_p` and `pinq_train.gate.fork_paired_engagement` — with one forced
deviation: the pairing key is `(foreign_trace_sha, n_prefix_user_turns, seed)`, not
`(foreign_trace_sha, foreign_prefix_k, seed)`, because `foreign_prefix_k` is on the fork's
manifest.json and was never a `runs.parquet` column. `n_prefix_user_turns` is verified, not
assumed, to substitute for it: on the 204 real runs behind
`docs/reports/forks_tau2_retail_test.json`, two of the 32 forked traces are cut at two different
`foreign_prefix_k` each and the substitute key still separates every one of them, and
`test_fork_paired_engagement_replicates_the_retail_headline` pins the reproduction of that
report's pooled diff_mean (−4.313725490196078) and fewer/more (71/24) bit for bit. On a grid
that is not a fork grid, neither key is added to `criteria`, so every other verdict — including
`test_the_cap8_rule_reproduces_the_previous_verdict_bit_for_bit` — is untouched.

### 5.3.4 Where the N1 selection rule lives

"The N1 rule" cited above (§5.3.2's stop criterion; run per candidate × suite together with a
matched-cost margin criterion (c) that reads the same `coverage_rule` block §5.3.2 describes) is
`scripts/select_checkpoint.py` — not a paper appendix, not a notebook, not a session scratchpad.
It decides, per candidate × suite, criteria (a′)/(b)/(c) plus a quality floor, gating each
criterion on `scorer_match` first: a scorer-hash mismatch between candidate and reference is a
refusal for that criterion, not a comparison, the same discipline `verify_instrument` enforces in
§5.3.2. `tests/test_select_checkpoint.py` runs the tool's own `--self-test` as a pytest case
rather than a script a reviewer has to remember to invoke by hand, and adds one fixture-backed
test per refusal — a pre-`f348feb` stop-2×2 block (§5.3.1), a scorer-hash mismatch across
checkpoints, and a `coverage_rule` block in a shape nobody wrote a reader for — each against a
static JSON fixture under `tests/fixtures/select_checkpoint/` a reviewer can open and diff
directly, rather than only the self-test's in-memory ones. It was promoted out of a session
scratchpad because it decides which checkpoint this paper reports, and a rule that decides a
reported number cannot live somewhere that can vanish with the session that wrote it.

---

## 6. Rung 2 — DPO/KTO on same-state pairs

**Why same-state pairing is the point and not a convenience.** Write the advantage as
`A(s_t,a) = Q(s_t,a) − V(s_t)`. Terminal-reward policy gradient never observes `V(s_t)`; it has
to estimate it from returns collected across *different* tasks. On these suites that estimate is
hopeless for a measurable reason: the between-task variance of `V(s_t)` is set by task
difficulty, and the REML decomposition of the pilot shows task variance dominating arm variance
by roughly an order of magnitude. The signal arrives buried under a baseline error several times
its size, and more rollouts shrink noise as `1/√n` without touching the bias in a mis-specified
baseline.

DPO on two candidates drawn at the **same** `s_t` does not estimate `V(s_t)` — it cancels it
exactly. The objective depends only on the difference of the two log-ratios, and `s` is
*literally the same token sequence* in both terms, so everything that is a function of the state
alone (task difficulty, evidence-pool size, the drafter's prior, `V(s_t)` itself) appears
identically in both and subtracts to zero. Nothing has to be modelled, tuned or estimated. That
is why this rung costs ~25 GPU-h and rung 3 costs $2,400.

Two invariants, both re-checked at load time (`load_pairs`):

- **`assert_same_state` refuses a cross-state pair.** Fatal, not filterable: a mixed dataset
  trains a model whose objective is partly the within-state contrast and partly an uncontrolled
  cross-task comparison, and no metric on the training curve can tell you which part moved.
- **The length guard survives into training** (`|Δlen| ≤ 40` chars). Without it the preference
  model learns "longer question wins" — the same confound that makes a naive LLM judge unusable,
  imported straight into the policy's weights. It is re-checked rather than trusted because the
  pairs file is a plain jsonl that can be filtered, concatenated or hand-edited after export.

### 6.1 What the reference policy is, and why it is not a merge

Rung 2's claim is "DPO moved the policy away from the SFT checkpoint", and that sentence has
content only if `pi_ref` **is** rung 1. TRL computes `pi_ref` by disabling the adapter it trains.
The trainer used to load rung 1's LoRA onto the base *and* hand `DPOTrainer` a fresh
`peft_config`, so "the adapter" named two things and which one got disabled was a property of the
installed trl/peft versions. Both readings train, report a falling loss and a rising reward
margin, and write a checkpoint; only one makes the claim true, and no artifact recorded which
happened.

`DPOConfig.reference` now names which of the three unambiguous readings a run used, and is in
`cfg.sha`.

**`reference: "adapter"` — the default.** One base in memory, rung 1's LoRA loaded **twice**: a
trainable copy that DPO optimises and a frozen copy that is `pi_ref`. Nothing is merged, so
nothing can be rounded away, and the reference is selected by a *name* this repository writes
down. Its LoRA hyperparameters are therefore rung 1's — the policy adapter *is* rung 1's adapter
— and `rung2.manifest.json` records `reference_policy.lora` read from rung 1's
`adapter_config.json` rather than from `DPOConfig.lora`, which describes nothing this run did.

```bash
pi train rung2 --base-model Qwen/Qwen3-8B --adapter artifacts/rung1 --pairs data/rl/pairs.jsonl
```

**Why merging is not the default: 44.7%.** A LoRA fold is exact only in a dtype whose ulp is
below the update, and rung 1's is not (§6.1 measurement below). `MERGE_DTYPE = "float32"` fixes
the storage, and a merged 8B base is then ~32 GB instead of ~16 GB — which rung 2 must load. The
two-adapter recipe costs one base in memory, loses nothing, and is the standard recipe.

**Two names are decided by the installed libraries, not by taste**, and both are measured:

| name | value | why it cannot be anything else |
| --- | --- | --- |
| `POLICY_ADAPTER` | `"default"` | trl 1.13.0's `DPOTrainer.__init__` builds the frozen copy itself with `model.peft_config["default"]` and a parameter-wise copy of every `.default.` tensor — the name is *indexed*. And `PeftModel.save_pretrained` writes any other name into a **subdirectory**, so rung 2's checkpoint would land in `artifacts/rung2/train/`. |
| `REFERENCE_ADAPTER` | `"ref"` | trl 1.13.0's reference forward is `use_adapter(model, adapter_name="ref" if "ref" in model.peft_config else None)`. Under any other name TRL finds nothing, **disables every adapter**, and `pi_ref` is the bare base — silently. |

MEASURED, trl 1.13.0: `DPOConfig` declares 137 fields and `model_adapter_name`/`ref_adapter_name`
are **not** among them; they were removed. `dpo_argument_kwargs` therefore probes for them, passes
them where they exist, falls back to the name convention where they do not, refuses
(`UnknownReferenceHook`) where neither can be found, and records which applied as
`reference_adapter_hook` in the manifest. And because intent is not observation,
`verify_frozen_reference` runs on the assembled model *after* `DPOTrainer` is constructed and
raises unless every policy LoRA tensor has a reference counterpart that is **equal to it and
frozen** — the silent-wrong-reference bug caught before the first optimiser step rather than
after a week of GPU time.

**`reference: "merged"` — kept, for a fused checkpoint.** Rung 1 folded into the weights, the only
adapter in the process is the fresh LoRA DPO trains, and `merged_base_sha` names the weights. Use
it to serve one directory with no adapter at inference time, or when rung 2's LoRA shape must
differ from rung 1's:

```bash
pi train merge --base-model Qwen/Qwen3-8B --adapter artifacts/rung1 --out artifacts/rung1-merged
pi train rung2 --reference merged --merged-base artifacts/rung1-merged --adapter artifacts/rung1 \
  --pairs data/rl/pairs.jsonl
```

`--merged-base` is refused under `--reference adapter`: a directory that is read and then ignored
is worse than one that is not named.

**`reference: "base"` — the ablation, and the arm is `dpo_from_base`.** No rung 1 anywhere in the
run: the policy is a **fresh** LoRA from `DPOConfig.lora` on the **untrained** base, and `pi_ref`
is that same untrained base. What it measures is **the preference label alone, without
imitation** — plan v4 A14 / E39, the arm that asks whether the SFT rung is *necessary* before
preference learning. It is not `"merged"` with the merge left out: `merge_adapter=False` under
`"merged"` says `base_model` is *already* the policy (merged elsewhere, or a full-weight
fine-tune), which is a claim about a checkpoint that was trained, and this arm's whole point is
that its starting policy was not.

```bash
pi train rung2 --reference base --base-model Qwen/Qwen3-8B --pairs data/rl/pairs.jsonl
```

`validate()` refuses a non-empty `adapter` (`AmbiguousReference` — a base reference with an
adapter named is ambiguous: *which* one is the policy? the run would hold rung 1's LoRA and the
fresh LoRA DPO trains, and "DPO with no SFT rung" is false under the first reading and
unverifiable under the second), `merge_adapter=True`, and a non-empty `merged_base_sha` — the
adapter mode's three refusals, mirrored. `--merged-base` is refused too, and for a *worse* reason
than under `adapter`: that flag resolves **into** `base_model`, so a merge output named here
would be loaded as the base and the run would start from rung 1's weights under the name of the
arm that has no rung 1 in it.

The mechanism is TRL's `None` branch — the same line the adapter mode depends on, read for its
other outcome. With no adapter called `"ref"` anywhere, `use_adapter(model, adapter_name="ref" if
"ref" in model.peft_config else None)` takes `None` and `use_adapter(None)` is
`model.disable_adapter()` (trl 1.13.0, `trainer/utils.py:1261`), so the reference forward runs on
the bare base. `base_reference_hook` probes for that line *before the weights load* and refuses
(`UnknownReferenceHook`) a TRL that does not show it — a field probe is deliberately **not**
accepted as a fallback here, because this repository has only trl 1.13.0 installed and cannot
measure what an older one does with neither adapter name passed. The manifest records
`reference_adapter_hook: "trl_disable_adapters"` and
`reference_policy: {"reference": "base", "base_model": …, "adapter": null, "lora_source":
"DPOConfig.lora"}`.

And `verify_base_reference` runs after `DPOTrainer` is constructed, asserting the **opposite** of
what `verify_frozen_reference` asserts: that **no** `"ref"` adapter exists (in the parameters or
in `peft_config` — TRL selects on the config, so an adapter registered under that name captures
the reference forward whether or not it carries a tensor), that the `None` branch is the one the
installed TRL takes, and that the policy's `lora_B` is **zero** at step 0 — `is_the_base` true,
which is the *requirement* here and a *bug* under `adapter`, where it would mean rung 1 never
trained. It is handed `trainer.model`, not the model `train()` built: a `peft_config` makes trl
1.13.0 replace the model with `get_peft_model(model, peft_config)`, and the local object is a bare
base with no adapter and no `peft_config`, on which every check would pass by finding nothing.

**Its β and lr are NOT comparable to the adapter mode's.** The DPO KL term is anchored to
`pi_ref`, and `pi_ref` here is a *different model* — the untrained base rather than the SFT
checkpoint. The same β therefore buys a different constraint and the same learning rate moves a
policy that starts somewhere else, so a `dpo_from_base` row and an adapter-mode row at matched β
and lr are not a controlled comparison of the two references; either tune them per arm, or state
in the table that they were held fixed across a reference change and that the KL anchor differs.
`reference` is in `cfg.sha` precisely so the two cannot be joined as one run.

**What an adapter's sha covers.** `merge.adapter_sha` hashes `adapter_model.safetensors` (or
`.bin`) and `adapter_config.json` **at the root of the directory, by name** — not the tree. It was
a suffix sweep over `rglob("*")`, which swept in everything the trainer leaves beside the adapter:
`checkpoint-*/` (one every `save_steps`, each with its own weights and an
`optimizer.pt`), `training_args.bin`, `rung1.manifest.json`, the model card. MEASURED on the smoke
run: **deleting an intermediate checkpoint changed `adapter_sha` while the adapter did not change a
byte.** That sha sits inside `ModelPin.key` → `model_pin_hash` → `semantic_hash` → `run_id`, so a
field that moves when nothing moved renames finished runs in a repository whose tables cite run
ids. A directory with no adapter weights at its root is refused rather than hashed.

`pi train merge` writes `merge.manifest.json` (base, adapter path, adapter sha, output sha) and
prints the output sha; that sha is `DPOConfig.merged_base_sha`, read from the manifest by
`--merged-base` rather than retyped. `validate()` refuses `merge_adapter=True` with an empty
`merged_base_sha` — a run that cannot name its reference weights cannot make the claim — and
raises `AmbiguousReference` on `merge_adapter=False` with a non-empty `adapter`, which is the
two-adapter configuration by name. The sha covers the `*.safetensors` files only: `config.json`
carries `transformers_version`, so hashing it would give two shas for one set of weights merged
on two boxes. `train()` reconciles `merged_base_sha` against the manifest and records
`merged_base.verified`; a base with no manifest (a hub id) is recorded unverified, not asserted.
`label_smoothing` (conservative DPO, 0.2 on the control arm) reaches TRL's config; the loss
itself is §6.2.

**The merge dtype is a correctness knob, not a storage preference.** Qwen3 ships
`torch_dtype: "bfloat16"` and transformers 5.x makes `from_pretrained` default to the
*checkpoint's* dtype where 4.x upcast to float32 — so the fold ran in bf16 and saved bf16, and a
bf16 weight has eight mantissa bits. MEASURED on the Mac smoke run (Qwen3-0.6B, LoRA r=32, six
optimiser steps): mean |LoRA delta| 2.819e-05 against a typical |W| of 2.363e-02 whose bf16 ulp is
9.229e-05, i.e. the update is a third of one representable step; **44.7% of `sum|delta|` did not
survive the save** (6862.69 of 12413.99) and 307,036,664 weights had a non-zero fp32 delta rounded
exactly to zero. `test_reference_logprobs_equal_a_freshly_loaded_merged_model` — written for this
property and never run until `PI_SMOKE_MODEL` was set — failed at 2.74e-02 against its 1e-04
tolerance. It is *storage*, not arithmetic: folding in fp32 and storing bf16 keeps the same 55.3%
to the digit. `merge_adapter` therefore loads and stores at `MERGE_DTYPE = "float32"` (100% of the
delta survives; the test passes), records `dtype` in the manifest, and takes `dtype=` for a run
that knowingly wants bf16. The cost is real — a merged 8B base is ~32 GB rather than ~16 GB, and
rung 2 loads that — so this is a decision to revisit with a *measured* delta from the full run
rather than a six-step one.

### 6.2 Which pairs a run trains on

`ask_ask` only is the **primary** arm (`include_pair_kinds`, default the sampled kinds): the 236
STOP-winning pairs cannot teach stopping, and the 1,356 ASK-winning ones import the "longer wins"
gradient the length guard exists to keep out. Every filter is a `DPOConfig` field, so it is in
`cfg.sha` — two runs over different subsets of one file must not claim one identity — and every
drop is counted into `rung2.manifest.json`:

| field | what it selects | note |
| --- | --- | --- |
| `include_pair_kinds` | the decision being trained | `ask_stop` is a separate population, not noise inside `ask_ask` |
| `include_label_sources` | which **file** the pair came from (rule, rater) | |
| `exclude_decided_by` | who **ordered** the pair | 532 rows on `pairs.rater.jsonl` are `label_source=rule` **and** `decided_by=rater`; filtering the label source excludes none of them |
| `include_code_versions` | the cohort that produced the rows | allowlist; a row with no `code_version` is dropped and counted apart, under `code_version_missing` |
| `min_q_distinctness` | the diversity floor (`--min-q-distinctness`) | paraphrase < related < different; refused if no row carries the field |
| `exclude_suites` | suites | |

`rows_for(cfg)` is the single mapping from a config to its rows, and `pi train rung2` hands the
trainer the list it preflighted rather than asking it to re-read the file. This closes a real
divergence: the CLI preflighted without `include_label_sources` while `train()` loaded with it,
so one `cfg.sha` covered two datasets and the printed report described neither reliably.

**The STOP-share floor.** `min_stop_chosen_share` (`--min-stop-chosen-share`, default 0.0, in
`cfg.sha`) is a floor on `n_stop_chosen / n_pairs`, reported as `stop_chosen_share`; `preflight`
raises `TooFewStopPairs` below it. A run that includes `ask_stop` to teach stopping, over a file
that is almost all
ASK-wins, otherwise trains the opposite of what its config asks for while every loss curve looks
healthy — the same argument as rung 1's STOP-share ceiling. The default refuses nothing, which is
right for the `ask_ask`-only arm whose share is 0 by construction.

**The objective, and which names the installed library will run.** `loss_type` is a **tuple**,
default `("sigmoid",)`, and TRL 1.x sums the named terms with `loss_weights`. All three fields
below are in `cfg.sha`; `loss_weights` and `ld_alpha` sit in rung 2's `SHA_OMIT_WHEN_NONE`, and a
**single-element** `loss_type` renders as the bare string the old `loss_type: str` rendered — so
the default config keeps the `config_sha` it had before this field changed shape (measured at
`f3f3f0f`: `DPOConfig(base_model="Qwen/Qwen3-8B", adapter="artifacts/rung1").sha` is
`63dcf89f421d…` before and after, pinned by
`tests/test_rung2_objective.py::test_the_default_objective_hashes_exactly_what_the_old_string_hashed`).

| field | flag | what it is for |
| --- | --- | --- |
| `loss_type` | `--loss-type` (repeatable) | the loss terms. `sigmoid` is plain DPO; `ipo` is the length-normalised squared-margin form, which regularises the "keep separating" failure DPO is prone to on near-duplicate pairs; `discopop` blends the logistic and exponential components by log-ratio magnitude; `sft` is the RPO-style **NLL anchor on the chosen side** — not a preference term at all, but the thing that stops a DPO run from separating the two sides by walking *both* of them away from the SFT checkpoint |
| `loss_weights` | `--loss-weights 1.0,0.5` | one weight per term, zipped positionally. Omitted is TRL's own `1.0` each. The MPO shape is a preference term plus `sft` at a smaller weight |
| `ld_alpha` | `--ld-alpha` | length-desensitised DPO: the weight on the log-probabilities of the tokens **past the shared prefix** of the two sides. `1.0` applies no weighting (plain DPO), `0.0` masks that tail entirely. A second lever on the length confound the `|Δlen| ≤ 40` guard bounds from the other side |

`kto_pair`, the **other** value the old string field accepted, is **gone**. MEASURED on trl
1.13.0: the loss ladder in `trl/trainer/dpo_trainer.py` (lines 1472–1589) dispatches exactly
`sigmoid, hinge, ipo, exo_pair, nca_pair, robust, bco_pair, sppo_hard, aot, aot_unpaired,
apo_zero, apo_down, discopop, sft, sigmoid_norm` and nothing else, so a run configured with
`kto_pair` validated, preflighted, loaded an 8B base and two adapters, and *then* raised
`Unknown loss type` from inside the loss. KTO on these pairs is `trl.KTOTrainer` — a different
trainer with a different dataset shape — and is not reachable from this rung. `LOSS_TYPES` is
that measured set; `installed_loss_types()` re-reads the ladder off the installed source at run
time, and `dpo_argument_kwargs` refuses a name that source does not dispatch.

**The objective may not be silently dropped.** `max_prompt_length` and `chat_template_kwargs`
are omitted on a TRL that lacks them, because the run is still the run the config describes
without them. `loss_weights` and `ld_alpha` are not: omitting either trains **plain DPO** under a
`cfg.sha` that names a mixture or LD-DPO, with every curve looking healthy — so a TRL that does
not declare them raises `UnsupportedObjective` instead, and `dpo_argument_kwargs` now runs
*before* the weights load rather than after. What was passed is recorded in the manifest as
`objective_hooks` (the comma-separated names actually sent) and `loss_type_hook`
(`list` / `str`, suffixed `_unchecked` when the installed ladder could not be read — an
unreadable source is "could not tell", never "supports nothing").

**Held-out data.** `load_pairs` raises `HeldOutDataset` on any row stamped with a split other
than `train`. A row that makes no claim is not a violation — the whole existing corpus predates
the field — but the dev exporter stamps `split: "dev"`, so the guard fires on exactly the file an
operator points `--pairs` at by mistake.

**Chat template.** `chat_template` and `enable_thinking` are recorded on the config; the
tokenisation at DPO time is TRL's, through `processing_class`. `dpo_argument_kwargs` probes the
installed `trl.DPOConfig` for a chat-template hook, passes `chat_template_kwargs={"enable_thinking":
...}` when one exists, and records which path it took as `chat_template_hook` in the manifest.
MEASURED on trl 1.13.0: there is **no** hook, so the recorded value is `tokenizer_default`, the
tokenizer's own template governs, and Qwen3's default emits thinking — the deployment must serve it
with `enable_thinking=False` explicitly (§8).

**Prompt budget.** `dpo_argument_kwargs` probes for `max_prompt_length` the same way and records
`prompt_length_hook`. MEASURED on trl 1.13.0: TRL 1.x removed it and keeps only `max_length` plus
`truncation_mode`, so the recorded value is `max_length_only`. The two regimes truncate
**differently** and are not the same experiment: a separate prompt budget trimmed the state and
kept the action, whereas `truncation_mode="keep_start"` over the concatenation trims from the right,
which is where the action is. TRL 1.x drops a row whose prompt alone fills `max_length`, so the
"supervise nothing" case is caught; a prompt just under it still loses part of its action, which is
an argument for sizing `max_length` from `pi train tokstats` with headroom rather than to the p99.

**Go/no-go into rung 3.** Rung 2 must beat rung 1 on the dev slice by more than the paired noise
floor, *and* the win must survive the length control (compare mean question length across arms;
if the trained arm's questions grew, the gain is the confound the guard was supposed to stop).

**Fallback.** Ship rung 1's checkpoint and report rung 2 as attempted-and-flat. A flat DPO on
same-state pairs is itself an informative sentence: it says the candidate sampler was not
producing meaningfully different actions at a state, which is a statement about the policy's
entropy, not about the method.

### 6.3 KTO over the unpaired signals (RC4 `kto_unpaired`)

`src/pinq_train/rung2_kto/` is the library for the RC4 combination arm
(`plans/2026-09-14-training-programme-v3.md` §5b). **The CLI is not wired here** — `pi train kto`
is a separate task; this section describes `convert.build_kto_rows`, `train.KTOConfig` and
`train.preflight`, which are what it will call.

**What the arm is for.** §6 opens with the reason the headline is a within-state contrast: it
cancels `V(s_t)` exactly. That argument is not retracted. It says which estimator has low
variance, not which rows carry information — and the corpus states facts about *single* actions
that no pairing can express. KTO scores one completion at a time against the reference, so each
of those becomes a row, at the cost of the variance DPO avoids (its loss depends on `V(s_t)`
through the reference's own log-probability rather than cancelling it). That trade is what RC4
measures, against `dpo_control` on Tier B. It is a combination arm, not a replacement.

**The direction rule is one sentence, and deliberately not a table of kinds.**

> Every SFT row is desirable; the **chosen** side of a pair is desirable and the **rejected**
> side undesirable, whatever kind of pair it is.

Written as a table over `pair_kind` × `stop_source` it would need a new branch every time the
exporter grows a kind, and the branch nobody added would emit the wrong sign. `stop_source` is
therefore **carried, never branched on**: `synthesised_notdone` (§4.2's `dpo_notdone` export)
needs no change in the converter. Which side of a STOP-bearing pair is the STOP is read off the
bytes. `source_kind` on every row names the family so the manifest stays readable per-family:
`sft_row`, `pair_chosen`/`pair_rejected` (ask_ask), `synth_stop_chosen`/`synth_ask_rejected` (a
STOP-chosen pair: stopping at a state gold says was done, and the ask that was wrong there), and
`synth_ask_chosen`/`synth_stop_rejected` (an ASK-chosen one: a stop at a state that was *not*
done). The length guard does **not** carry over from §6.2 — it exists because a preference model
shown two completions side by side can learn "longer question wins", and KTO never sees two
completions together, so importing it would delete rows on an argument that does not apply.

**MEASURED on `data/rl/{sft,pairs}.jsonl` (2026-09-15, 43,837 SFT rows + 47,989 pairs):**

| | rows |
| --- | --- |
| emitted before dedupe | 139,815 |
| collapsed by dedupe | 52,792 |
| dropped as conflicting | 6,257 (1,475 keys) |
| **kept** | **80,766** (47,285 desirable / 33,481 undesirable, 41.5% undesirable) |
| `sft_row` | 43,577 |
| `synth_ask_rejected` | 28,613 |
| `pair_rejected` / `pair_chosen` | 4,688 / 2,965 |
| `synth_ask_chosen` / `synth_stop_chosen` / `synth_stop_rejected` | 428 / 315 / 180 |

**Dedupe.** The key is `(suite_id, task_id, sha256(state_text), action_json)` — *not* `run_id` or
`turn_idx`: the same question at the same state in two runs is one signal about that state, and
counting it twice weights a state by how many rollouts happened to visit it. The STOP action is
one 18-byte constant, so every synthetic pair at a state names the same `(state, STOP)` row;
31,156 of the 52,792 collapses are exactly that. First occurrence wins and SFT rows are emitted
first, so an action that is both an SFT target and a pair's chosen side is recorded as `sft_row`
— the SFT row is an absolute judgement (the candidate cleared the value floor), the pair side a
relative one.

**The conflict rule, and why the plan's expectation was wrong.** A `(state, action)` that is
desirable in one pair and undesirable in another is a contradiction KTO cannot represent: it has
no partner to relativise against. The task that specified this arm expected such keys not to
exist. **They do: 1,475 of them, 1,098 from the ask_ask tournament alone** — candidate B beats C
at a state and loses to A at the same state, which is pairwise ranking working as designed, not
corrupt data. So `on_conflict` is a declared choice in `cfg.sha`:

- `"refuse"` (default) raises `ConflictingLabels` with the count and three named examples. On
  today's export this refuses the whole corpus, which is the honest reading — it is not
  convertible as it stands.
- `"drop"` removes **both** sides of every conflicting key (6,257 rows) and counts them.

Neither keeps one label. Keeping one would turn a contradiction the corpus does not settle into
an absolute claim nothing measured. A run of RC4 on today's file must therefore declare
`on_conflict="drop"`, and its `cfg.sha` records that it did.

**The weight rule.** `balanced_weights(n_d, n_u)` returns `w_d = (n_d+n_u)/2n_d`,
`w_u = (n_d+n_u)/2n_u`, so `w_d·n_d == w_u·n_u` **and the mean row weight is 1**. The second
clause is what makes this the right normalisation rather than either obvious one: pinning
`w_d = 1` and lifting the minority scales the whole loss up, pinning `w_u = 1` and damping the
majority scales it down, and either changes the run's *effective learning rate* as a side effect
of its class balance — so a balanced arm and an unbalanced one would differ in two things at
once. §5.2.3 makes the same argument about `stop_weight` and `mean_row_weight`. On the numbers
above: `w_d = 0.854034`, `w_u = 1.206147`. trl's recommended band (KTO paper Eq. 8, checked in
`KTOTrainer._prepare_dataset`) is a mass ratio in `[1, 1.33]`, so the equal-mass point sits at
the bottom of it.

The two weights are **plain fields on `KTOConfig`**, therefore in `cfg.sha`, and are *not*
derived inside `train()`: a weight the trainer computed for itself is a weight the run's identity
cannot state, and two runs over two exports would then share one sha while optimising two
different balances. `preflight` refuses a config whose weights do not equalise the masses of the
rows it was handed (`ImbalancedMasses`, tolerance 1%), names the pair to pass, and accepts
`require_balanced_masses=False` as a declared arm. `min_undesirable_share` is the floor for an
arm whose claim is about the negative half; 0.0 refuses nothing.

**The reference policy is §6.1's, imported rather than restated.** One base in memory, rung 1's
LoRA loaded twice — a trainable `default` and a frozen `ref` — with `reference_hook`,
`verify_frozen_reference`, `adapter_sha` and `adapter_lora_spec` taken from `rung2_dpo.train`.
MEASURED: `trl/trainer/kto_trainer.py:1280` reads `use_adapter(unwrapped_model,
adapter_name="ref" if "ref" in unwrapped_model.peft_config else None)`, three occurrences — the
same line DPO's probe already recognises, so one probe covers both trainers. The **merged**
reference is not offered here: nothing in RC4 asks for a fused serving checkpoint, and §6.1's
44.7% measurement applies unchanged.

**What the installed TRL accepts.** MEASURED on trl 1.13.0 (`.venv-train`, 2026-09-15):
`KTOConfig` declares 129 fields. `beta` (`trl/trainer/kto_config.py:201`), `desirable_weight`
(`:208`) and `undesirable_weight` (`:215`) are among them; `max_prompt_length`,
`chat_template_kwargs`, `label_smoothing`, `model_adapter_name` and `ref_adapter_name` are not.
`kto_argument_kwargs` is a pure probe for the same reason `dpo_argument_kwargs` is, and records
`truncation_hook` (`max_length_right` on 1.13.0 — the sequence is truncated from the right,
which is where the action is), `chat_template_hook` and `reference_adapter_hook` in the manifest.
The dataset columns are `prompt` / `completion` / `label` (`kto_trainer.py:225-226`); the
provenance block on each converted row stays out of the `Dataset` and reaches
`rung2_kto.manifest.json` as aggregate counts.

**Hardware honesty.** No KTO run of any size has been executed in this repository. `train()`
raises `NotValidatedOnHardware` unless the caller acknowledges that; everything else in the
module runs on a laptop and is covered by `tests/test_kto_convert.py` and
`tests/test_kto_config.py`.

---

## 7. Rung 3 — GRPO. Optional, unvalidated, hard kill 2026-09-12

**Nothing in `src/pinq_train/rung3_grpo/` has ever been run.** It is a config, the reward wiring
(`group_rewards`), and a written description of the rollout-server contract. `train()` raises
`Rung3NotValidated`, and says so:

> rung 3 is a scaffold: config, reward wiring and the rollout-server contract, and nothing else.
> No GRPO step has ever been executed in this repository and no claim about its behaviour is
> supported by anything here.

`assert_go()` enforces two gates: an **explicit written go**, and the **kill date**, which is a
constant in code (`KILL_DATE = date(2026, 9, 12)`) rather than a note in a plan, because a kill
date that lives in a document gets renegotiated at 2 a.m. on the day it fires.

The one computation that is implemented and tested is the group-relative advantage. It does
**not** divide by the group's standard deviation (Dr-GRPO's correction): dividing up-weights
groups that happened to be homogeneous, which here means up-weighting the *easy* prompts —
exactly the prompts where the treatment has the least room to show anything. Small per step,
systematic across every step.

**The fallback, committed in advance.** If the reward curve is not monotone over 300 steps by
the kill date, kill it and **ship the learning curve as a reported negative result with a
diagnosis** — a figure, the reward decomposition showing which term failed to move, and the
variance argument from §6. Not a footnote and not "future work". The strongest version of this
paper containing no trained model at all is already a complete paper; rung 3 is upside, and
treating upside as a requirement is how a deadline is missed for a result nobody needed.

---

## 8. Renting the GPU

There is no local CUDA device. The plan's dollar figures imply **≈ $2.50 per GPU-hour** for a
single 80 GB card (180 GPU-h ≈ $450; 25 GPU-h ≈ $60), which is the on-demand range for an H100
80 GB on the usual providers. **Re-check the live price before booking** — these are the rates
the plan was costed at, not quotes obtained today:

| provider | shape | notes |
|---|---|---|
| RunPod (Secure Cloud) | 1×H100 80 GB | per-second billing; persistent volume survives a pod stop, which is what makes an interrupted rung resumable |
| Lambda Labs | 1×H100 80 GB / 8×H100 | fixed hourly, no spot; simplest for rung 3's 8-card shape |
| Vast.ai | 1×A100 80 GB or 1×H100 | cheapest, interruptible; acceptable for rung 1/2, **not** for rung 3 |
| Modal / Together | serverless H100 | per-second; convenient for rung 2's ~25 h, priciest per hour |

Three operational notes that cost real money if missed:

- **Book by the hour, not by the day.** Rung 1's 180 GPU-h is *compute* time; the dataset build
  and the eval run on the laptop. Renting across the whole week multiplies the bill by ~4.
- **The dataset is small and the base model is not.** Pull the base weights once onto a
  persistent volume; re-downloading a 60 GB checkpoint on every pod start is 20 minutes of paid
  idle each time.
- **Sleep killed two earlier attempts on the local machine.** On a rented box the equivalent is
  an interruptible instance being reclaimed mid-epoch. Both rungs checkpoint every
  `--save-steps` steps (200 by default) and resume from the latest one under `--out`, so a
  reclaimed instance costs at most that; treat spot capacity as unavailable for rung 3, which
  has no trainer at all.

**A sizing fact the plan does not state.** 180 GPU-h on **one** 80 GB card constrains the
trainable base to roughly ≤ 30 B parameters in bf16 with LoRA. `gpt-oss-120b` — the model pinned
for the *frozen* roles through the remote proxy — does **not** fit that budget. The trained arm's
base must therefore be chosen and pinned explicitly (`SFTConfig.base_model` has no default, and
`validate()` refuses an empty one), and it will not be the same model the frozen roles run. That
is fine for the experiment — the comparison is between two *Inquirer* pins with everything else
held fixed — but it must be stated in the paper rather than discovered by a reader.

---

## 9. What gets released, and under which licence

| artifact | contents | licence |
|---|---|---|
| **Code** (`src/pinq_train/`, `src/pi_run/serve/`) | the ladder, the seam, the reward | Apache-2.0 (this repo's `LICENSE`) |
| **Dataset** (`data/rl/sft.jsonl`, `pairs.jsonl`) | rendered states + Inquirer actions + measured values | **per-suite shards**, each under its upstream's licence — see below |
| **LoRA adapters** (rung 1, rung 2) | adapter weights only, not merged | Apache-2.0, **subject to the base model's licence** |
| **Rung-0 prompt** (`conf/prompts/rung0/`) | the optimised Inquirer template + lineage manifest | Apache-2.0 |

**The dataset must be shipped as per-suite shards, not as one file.** A rendered `state_text`
embeds upstream paragraphs verbatim, so each shard inherits its source's terms and the terms
differ:

- `musique` — **CC BY 4.0** (attribution)
- `strategyqa` — **MIT** (code repo MIT; the AI2 data zip ships under the same terms)
- `wiki2` (2WikiMultiHopQA) — **Apache-2.0**
- `drgym` (key points / agentic-search logs) — **CC BY 4.0**
- `synth` — generated by this repository, **Apache-2.0**
- `tau2`, `pare` — **not in the dataset at all.** Eval-only by construction; the exporter
  refuses them by name and the scorer refuses them again.

Merging the shards into one file would put the strictest term on all of it and make attribution
unresolvable per row. Each shard therefore ships with its own `LICENSE` and a `manifest.json`
carrying `train_id_set_hash`, `scorer_hash`, `graph_version` and the reward `weights_sha` — so a
reader can prove which ids a checkpoint saw and under which measurement they were valued.

**Adapters, not merged weights.** A merged checkpoint redistributes the base model and inherits
every restriction attached to it; an adapter is a small diff that a user applies to weights they
obtained themselves. Release the adapter plus the exact base model id and revision.

**What is deliberately not released.** `data/gold/` and the canary nonces. Publishing gold would
make every future evaluation on these suites unfalsifiable, which is the same reason the firewall
exists in code.

---

## 10. Verification, as run on 2026-08-24

```bash
make fmt && make gate ; echo "GATE=$?"      # 0, and "Contracts: 4 kept, 0 broken"
.venv/bin/python -m pytest -q
.venv/bin/pi train status
.venv/bin/pi train export --kind both --gold-root "$PWD/data/gold"
.venv/bin/pi train rung0 --dry-run
```

`pi train rung1 --train` and `pi train rung2 --train` are **not** in that list and have not been
run. They require a CUDA device that does not exist here, and both refuse to start without
`--acknowledge-untested`, which is a statement that the caller knows the first execution is also
the first test.

## Annotation must be keyed to identity, never to `pair_id`

`pair_id` is a digest the exporter computes; it is NOT stable across re-exports. **MEASURED
2026-09-04**: a re-export moved `pairs.jsonl` from 3,648 to 27,266 rows and changed *every*
`pair_id` — **0 of 27,266** matched a previously rated set, while the same comparisons were
still present in the file. Joining on identity instead recovered **354 of 435**.

A pair's identity is the comparison it encodes:

    (suite_id, task_id, branch_of_run_id, branch_turn_idx, turn_idx,
     chosen_run_id, rejected_run_id)

`scripts/curate_pairs_with_a2.py` joins on that. It takes `rated=<the export the rating was
built on>` so a stored `pair_id` can be resolved to an identity once, after which the verdicts
survive any later re-export.

**Re-apply curation after every export.** `pi train export` overwrites `data/rl/pairs.jsonl`
and carries no annotation forward — three passes of A2 verdicts were lost this way before the
cause was understood. The verdicts themselves live in `artifacts/validation/` and are committed,
so a loss is always recoverable; what is lost is the join, and only if the join was keyed to
`pair_id`.
