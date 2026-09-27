# GPU runbook — from an empty rented box to a registered checkpoint

For someone renting an A100/H100 for the first time, with no other context on this repo. Every
command was checked against `.venv/bin/pi train <sub> --help` on this checkout (main @
`0b9b108a4b98`, 2026-09-12) or the source file cited beside it; every number names its file
(CONTRIBUTING.md rule 1), and a number measured for this runbook has its command and real output
pasted, dated 2026-09-12. Keep `plans/2026-09-11-training-regime.md` and `docs/TRAINING.md` open
beside this file — it gives commands, not rationale.

**Corrections pass 2026-09-15, on main @ `4e3055b` (the window-1 merge).** §7 (the registry's
sixth field), §8 (the merge-window rule, the D9 coverage rule, spend accounting) and §9 (resume,
`drafter_only` pins, `--spend-cap`) carry numbers measured on that commit, each stated beside the
claim it supports. Everything dated 2026-09-12/14 above is unchanged and still names its own
source.

## 1. Before renting (on the Mac)

Check every line below before paying for an hour of GPU time. As re-measured on 2026-09-14 (after
the on-policy growth campaign's export chain landed on top of the 2026-09-12 arm-allowlist
re-export and dev export), all five are **true** — the file sizes below are this campaign's
numbers; the two that were not true as of 2026-09-12 are recorded below with the command that
closed each one, kept as the way to reproduce this state on a fresh checkout rather than as an
outstanding step.

```bash
uv run --no-sync ruff check . && uv run --no-sync ruff format --check .
.venv/bin/lint-imports        # must print: Contracts: 4 kept, 0 broken
.venv/bin/pi suites audit     # must print: registry audit: clean
.venv/bin/python -m pytest -q
bash scripts/check_no_home_paths.sh    # "main gate green" per CONTRIBUTING.md
.venv/bin/pi train status              # ladder, split, what's on disk, what's missing
```
- **`data/rl/{sft,pairs,pairs.rater,pairs.reaches}.jsonl` present.** MEASURED 2026-09-14: yes,
  43,837 / 47,989 / 48,860 / 48,059 lines (`pi train status`, matches each `*.manifest.json`'s
  `n_examples`/`n_pairs`; was 35,869 / 38,858 / 39,173 / 38,917 post-allowlist, 2026-09-12,
  archived at `data/rl/archive-20260913-growth/`). `sft.manifest.json`'s `included_arms` is
  still `["inquirer_prompted"]` (measured directly from `data/rl/sft.manifest.json`; the
  **pre-allowlist** counts were 44,624 / 50,730 / 51,252 / 50,843, archived at
  `data/rl/archive-20260912-preallowlist/`). Its `suites` list still names `synth` — that is not
  a leftover: `synth` is a suite, not an arm, and 22 of its rows survive the allowlist as genuine
  `inquirer_prompted` rows (`RESULT data/rl/sft.jsonl`, `suites {..., 'synth': 22, ...}`,
  unchanged by the growth campaign, which added no synth rows). Reproduce with: `pi train export
  --kind both --gold-root "$PWD/data/gold" --include-arm inquirer_prompted`, then confirm
  `included_arms == ["inquirer_prompted"]` in the manifest.
- **`data/rl/dev/` present.** MEASURED 2026-09-14: yes. `data/rl/dev/sft.dev.jsonl` is 6,975 rows
  (`sft.dev.manifest.json` `n_examples`) and `data/rl/dev/pairs.dev.jsonl` is 4,962 rows
  (`pairs.dev.manifest.json` `n_pairs`, of which 970 are `ask_ask`), both `"split": "dev"` — was
  5,265 / 3,538 (371 `ask_ask`) before the growth campaign grew the dev-split run pool from 1,946
  to 3,570 run directories (2026-09-12 → 2026-09-14). Reproduce with: `pi train export --kind
  both --split dev --gold-root "$PWD/data/gold"` (the gold-side entry point that refuses non-dev
  rows — docs/TRAINING.md §2).
- **`pi train verify-prefs` passes.** MEASURED 2026-09-14, both rater-derived arms, both
  `MATCH`, exit 0 (was MEASURED 2026-09-12 at the smaller pre-campaign shas/counts noted inline):
  ```bash
  .venv/bin/pi train verify-prefs --pairs-manifest data/rl/pairs.rater.manifest.json --prefs data/rl/a6_preferences.jsonl      # sha 0fce532025625f94, 4,610 verdicts (was 1bc9b1c9709c27da, 2,429 verdicts, 2026-09-12)
  .venv/bin/pi train verify-prefs --pairs-manifest data/rl/pairs.reaches.manifest.json --prefs data/rl/reaches_preferences.jsonl  # sha c15251f7ecb6cf5e, 1,426 verdicts (was 52b139bde71c79d6, 720 verdicts, 2026-09-12)
  ```
- **`conf/checkpoints.json` empty.** MEASURED 2026-09-12: `pi train status` reports "(0
  registered)". Correct: an entry for a model id that already ran renames every run under that
  id (the file's own `_why_it_is_empty`).

## 2. Box setup

**Pin the commit, don't float on `main`.** Record `git rev-parse HEAD` on the Mac immediately
before renting and check out that exact sha on the box — `main` moves under a 12-week plan; this
runbook's own `0b9b108a4b98...` will be stale by the time you read it.

```bash
git clone <this repo's url> pinq && cd pinq && git checkout <the sha you just recorded>
uv venv --python 3.12
uv pip install -e ".[train]"   # torch>=2.3 transformers>=4.42 trl>=0.9 peft>=0.11 datasets>=2.19
                                # (pyproject.toml train extra — exactly what rungs 1-2 import)
```
**vLLM in its own venv.** `[train]` deliberately excludes it (`vllm` is under `train-grpo`,
rung-3-only). Serving pins its own torch build; mixing it with the trainer's breaks both:
```bash
uv venv .venv-vllm --python 3.12 && uv pip install --python .venv-vllm/bin/python "vllm>=0.9"
```
**Copy data by rsync, not git** — `data/rl/`, `data/gold/`, `data/corpora/`, `cache/`, `.env`
are all gitignored, so a clone alone gets none of them:
```bash
rsync -avz data/rl/*.jsonl data/rl/dev/ box:pinq/data/rl/     # rung 1/2 need only this
rsync -avz data/corpora data/gold cache box:pinq/data/        # for the offline/dev-gate evals later
rsync -avz .env box:pinq/.env                                 # never `git add` it; mode 600 both ends
```
**`.env` is never sourced automatically** — no dotenv loader exists here, so a missing key hangs
in retry backoff instead of raising:
```bash
set -a; . ./.env; set +a        # docs/RUNBOOK.md's convention
.venv/bin/pi env doctor         # live call per pinned model; exits 2 if any is unreachable
```
**In any rollout or serving shell, `PI_GOLD_ROOT` must stay unset.** `pi_eval.gold.gold_root()`
raises `GoldAccessError` if it isn't: "If you are in a rollout worker, this exception is the
firewall working correctly" (`src/pi_eval/gold.py`). Only Mac-side `pi train export`, `pi
score` and `pi train gate` set it, each for one process only.

## 3. Throughput probe (first hour)

```bash
.venv/bin/pi train tokstats --dataset data/rl/sft.jsonl --tokenizer Qwen/Qwen3-8B
```
Expect the numbers plan I.5 already measured on the real tokenizer (2026-09-12): state p50
1,841 / p99 3,901 / p99.9 4,459 / max **4,963** tokens (templated adds 20-82).
`conf/train/rung1_8b.json`'s `max_seq_len: 5120` is the next multiple of 1,024 above that max —
if your box disagrees, stop and find out why before sizing GPU memory to the wrong dataset.

**A 50-step timed run.** Neither `rung1` nor `--config` exposes a step limit (verified against
`--help` — there is no `--max-steps`), so shape the row count instead: effective batch is
`per_device_batch(1) x grad_accum(16) = 16`, so 800 rows at `--epochs 1` is exactly 50 steps.
```bash
shuf -n 800 data/rl/sft.jsonl > /tmp/probe_sft.jsonl
time .venv/bin/pi train rung1 --config conf/train/rung1_8b.json --base-model Qwen/Qwen3-8B \
    --tau 0.05 --sigma-j 0.0 --dataset /tmp/probe_sft.jsonl --epochs 1 --out /tmp/probe_out \
    --train --acknowledge-untested
```
`pi train tokstats --dataset /tmp/probe_sft.jsonl --tokenizer Qwen/Qwen3-8B` gives the probe's
own mean state-token count; add ~60-100 for action + template overhead (plan I.5). Then:
```
tokens_processed ≈ 50 steps x 16 x mean_tokens_per_row
tokens/s         = tokens_processed / (wall-clock seconds from `time`)
hours_full_run   = 179,000,000 tokens [plan I.15: SFT, 2 epochs] / (tokens/s x 3600)
```
**Write the measured tokens/s into `plans/2026-09-11-training-regime.md` §I.15** — its heading
says "the week-2 throughput probe replaces the GPU column": replace the FLOP-model estimate
(~1,950 tok/s A100 / ~5,150 tok/s H100) with what you measured and recompute the hour columns.

## 4. Rung 1

**Confirm tau/sigma_j from the manifest — don't retype the plan's numbers on faith.**
`data/rl/sft.manifest.json` records `margin_threshold` (0.05) and `rank_rule` ("outcome"). Since
`margin_threshold = tau + 1.5*sigma_J*sqrt(2)` (`src/pinq_train/export/dataset.py` ~line 714)
and `outcome` is the mechanical ranker (no judge, so `sigma_J = 0`), `tau = margin_threshold =
0.05` directly.

**Preflight** (no GPU; re-run after step 1's re-export changes `n_examples`):
```bash
.venv/bin/pi train rung1 --config conf/train/rung1_8b.json --base-model Qwen/Qwen3-8B \
    --tau 0.05 --sigma-j 0.0
```
MEASURED on the Mac, 2026-09-12 (pre-allowlist export, Apple M1 Max, no CUDA): `config_sha
71936163991c2b567d28c9d251197edb8db0086db97df82154384b28930b2ace`, `stop_share 0.6695` (gate
<= 0.80, **pass**), `within_task_distinct_3_of_dataset 0.7525` (gate >= 0.65, **pass**), pooled
`distinct_3 0.4002` (diagnostic only — docs/TRAINING.md §5.3). This is a property of the
TRAINING FILE, not the checkpoint — every arm trained on this export prints this same 0.7525
regardless of base model or seed. The box, run against the same config and dataset, must
reproduce this exact `config_sha`, or something differs from what shipped here.

**Train** (needs CUDA):
```bash
.venv/bin/pi train rung1 --config conf/train/rung1_8b.json --base-model Qwen/Qwen3-8B \
    --tau 0.05 --sigma-j 0.0 --train --acknowledge-untested
```
The adapter lands in `artifacts/rung1` (`--out`'s default); `rung1.manifest.json` beside it
records `cfg.sha`, `n_over_length_refused`, `n_unweighted_rows` and the two gates above under
`dataset`. Two more gates run later — that's `pi train gate`, §8 — on the checkpoint's OWN dev
generations (malformed rate vs. baseline, a fresh within-task distinct-3 measured on what this
particular checkpoint asks, question-length equivalence). That distinct-3 is a different
number from `within_task_distinct_3_of_dataset` above, and the gate verdict's own
`distinct3_scope` field says so.

## 5. Tier A offline evaluation

```bash
.venv/bin/pi train eval-offline --checkpoint artifacts/rung1 \
    --dev-sft data/rl/dev/sft.dev.jsonl --dev-pairs data/rl/dev/pairs.dev.jsonl \
    --chat-template --out artifacts/eval/rung1.tierA.json
```
An adapter directory has no tokenizer of its own; `tokenizer_source()` (`src/pi_run/cmd_train.py`)
reads `adapter_config.json`'s `base_model_name_or_path` and borrows the base's — a LoRA cannot
change the tokenizer, and peft's `save_pretrained` writes no tokenizer files. `--reference-ask`
supplies the ASK string a STOP row is compared against; omit it and STOP rows are skipped and
counted rather than scored against nothing.

The written JSON: `checkpoint`, `tokenizer` (which one scored it), `dev_sft`, `dev_pairs`,
`n_sft_rows`, `n_pairs`, `chat_template`, `enable_thinking`, `sft_nll`, `stop_confusion`,
`pair_accuracy`.

**Pass criteria (plan I.9, Tier A):** dev NLL below the untrained base; dev pair accuracy > 0.5
with a CI excluding 0.5 for DPO arms and above the SFT checkpoint's; both STOP-2x2 cells no
worse than base; malformed rate <= baseline; distinct-3 >= 0.65; question length within margin.
Failing means the data did not teach the label — report it before buying any rollout.

**The length-sign split (plan v4 §4 / rev-2 §4), on every rung-2 checkpoint.**
`pair_accuracy` carries `acc_by_len_sign` — dev pair accuracy split by the sign of the
chosen-minus-rejected *question* length, as `chosen_longer`, `chosen_shorter` and `equal`, each
`{acc, n}` — plus `len_sign_gap`, `acc(chosen_longer) − acc(chosen_shorter)` (`null` when either
side is empty). The sign is recomputed from `chosen_json`/`rejected_json` through the exporter's
own `_action_text` parse, not read off `len_delta`, which is an `abs()` and carries no sign; the
split covers ask_ask pairs only, because a STOP is 18 bytes and would put every ask_stop pair on
one side by construction. Read it because the pooled `acc` cannot: a policy that learned "prefer
the longer question" scores 1.0 on one half and 0.0 on the other, and the two cancel to the same
0.5 that "learned nothing" prints. **Kill rule, applied by the reviewer and not by this code:**
`chosen_shorter` accuracy below 0.50 while `chosen_longer` is above 0.60 means the policy learned
length, and the checkpoint does not proceed to Tier B whatever its pooled accuracy says. Quote
both cells with their `n` — a gap read off two cells of a handful of pairs each is noise.

## 6. Rung 2

One command per label arm, all sharing `--reference adapter` against rung 1's frozen adapter.
**Give each arm its own `--out`** — the default (`artifacts/rung2`) is shared by all three, and
a second invocation would silently overwrite the first.
```bash
.venv/bin/pi train rung2 --config conf/train/rung2_8b.json --base-model Qwen/Qwen3-8B \
    --reference adapter --adapter artifacts/rung1 --pairs data/rl/pairs.jsonl \
    --out artifacts/rung2-control --train --acknowledge-untested

.venv/bin/pi train rung2 --config conf/train/rung2_8b.json --base-model Qwen/Qwen3-8B \
    --reference adapter --adapter artifacts/rung1 --pairs data/rl/pairs.rater.jsonl \
    --include-label-source rater --out artifacts/rung2-rater --train --acknowledge-untested

.venv/bin/pi train rung2 --config conf/train/rung2_8b.json --base-model Qwen/Qwen3-8B \
    --reference adapter --adapter artifacts/rung1 --pairs data/rl/pairs.reaches.jsonl \
    --out artifacts/rung2-reaches --train --acknowledge-untested
```
`--include-pair-kind` is not repeated: `rung2_8b.json` already sets `include_pair_kinds:
["ask_ask"]` and `--config` pre-fills it. `--include-label-source rater` is the rater arm only —
it restricts `pairs.rater.jsonl` to rows whose `label_source` is `rater`. `--reference adapter`
(above) is the one to use; `--reference merged` (`pi train merge --base-model ... --adapter
artifacts/rung1 --out artifacts/rung1-merged`, then `rung2 --reference merged --merged-base
artifacts/rung1-merged`) is the fused-checkpoint alternative, needed only to serve rung 1 with no
adapter at inference or to change rung 2's LoRA shape — docs/TRAINING.md §6.1.

`verify_frozen_reference` runs after `DPOTrainer` is built and raises unless every policy LoRA
tensor has a reference counterpart equal to it and frozen — the wrong-reference bug caught
before the first optimiser step (docs/TRAINING.md §6.1). Two hooks are recorded rather than
applied, both MEASURED on trl 1.13.0: `chat_template_hook: tokenizer_default` (trl 1.x has no
such hook; the tokenizer's own template governs) and `prompt_length_hook: max_length_only` (trl
1.x dropped `max_prompt_length`; only `max_length` + `truncation_mode` exist). The two adapter
names are fixed by trl's own internals: the trainable copy is `"default"`, the frozen reference
is `"ref"` — any other name and trl nests the checkpoint under `train/` or disables every adapter.

## 7. Serving

```bash
# on the box, PI_GOLD_ROOT unset
.venv-vllm/bin/vllm serve Qwen/Qwen3-8B --served-model-name qwen3-8b-base \
    --enable-lora --max-lora-rank 32 --max-loras 4 \
    --lora-modules qwen3-8b-sft=artifacts/rung1 qwen3-8b-dpo-control=artifacts/rung2-control \
                   qwen3-8b-dpo-rater=artifacts/rung2-rater qwen3-8b-dpo-reaches=artifacts/rung2-reaches \
    --max-model-len 8192
```
(plan I.14, verbatim; **not verified against `vllm --help`** — vLLM isn't installed on this Mac.
Check it against your installed vLLM's own `--help` before the first real serve.)

Author a LiteLLM proxy on the box (`conf/serving/litellm.yaml` — new, doesn't exist yet): one
deployment per local checkpoint plus a passthrough for the frozen roles:
```yaml
model_list:
  - model_name: qwen3-8b-base       # one block per --served-model-name / --lora-modules key,
    litellm_params: {model: hosted_vllm/qwen3-8b-base, api_base: http://127.0.0.1:8000/v1,   # same api_base,
                      extra_body: {chat_template_kwargs: {enable_thinking: false}}}           # same extra_body
  - model_name: openai/aws/gpt-oss-120b   # PASSTHROUGH, unchanged, no extra_body:
    litellm_params: {model: openai/aws/gpt-oss-120b, api_base: <the team gateway's own base_url>}
```
`extra_body` is per-client, so `enable_thinking: false` belongs on each **local** block, never
globally — on the gpt-oss-120b passthrough it would reach the frozen Drafter/Answerer/user-sim
too. One `LITELLM_BASE_URL` (the tunnel below) serves every role for a sweep; only
`PI_MODEL_INQUIRER` changes between invocations — never override `LITELLM_BASE_URL` per-arm.

**SSH tunnel from the Mac** (runs, cache, provenance stay on the Mac — plan I.14). Two
ports: the LiteLLM proxy (default 4000) is what every sweep talks to, so the frozen roles reach
the team gateway through it; vLLM's own port (8000) is tunnelled only for the `/tokenize`
identity check, which speaks to vLLM directly:
```bash
ssh -N -L 4000:localhost:4000 -L 8000:localhost:8000 you@the-box
```
**Before spending anything on a sweep**, the identity check (plan I.13.1):
```bash
PI_VLLM_URL=http://127.0.0.1:8000 pytest tests/test_chat_template_identity.py -k serving
```
(`PI_VLLM_MODEL` defaults `qwen3-8b-base`, `PI_VLLM_TOKENIZER` defaults `Qwen/Qwen3-8B` —
override for a different checkpoint.) Compares the server's `/tokenize` ids against the
trainer's; a mismatch means every dev NLL is measured on a prompt the policy never sees.

**Register each checkpoint in `conf/checkpoints.json` at deploy time, never retroactively** (an
entry for an id that already ran renames every run under that id): `{"<served-name>":
{"adapter_sha": "<from rungN.manifest.json>", "training_manifest_sha": "<that run's manifest
sha>", "base_model": "Qwen/Qwen3-8B", "rung": 1|2, "dataset_sha": "<the ExportManifest sha it was
fitted on>", "train_id_set_hash": "<the export manifest's own train_id_set_hash>"}}`.
`scripts/price_tables/2026-09.json` already carries `$0` rows for
`qwen3-{8b,4b}-base`, `qwen3-{8b,4b}-sft` and `qwen3-8b-dpo-{control,rater,reaches}` — reuse
those served names, or `PriceTable.rates` raises `LLMConfigError` and the sweep refuses to start.

**`train_id_set_hash` is the sixth registry field, and it is REQUIRED.** The paragraph here
said T2 had not landed and the required set was still five; T2 and T2b landed on the
integration branch and it is six, so a row without it no longer loads. RE-MEASURED on this
branch: `src/pinq_adapters/llm/checkpoints.py` ~43 declares `REQUIRED_FIELDS = (adapter_sha,
training_manifest_sha, base_model, rung, dataset_sha, train_id_set_hash)`, and all 14 rows of
`conf/checkpoints.json` carry all six — 13 rows with six keys, one with seven. The seventh is
`init_from`, which T2b makes required at **rung >= 2** and refuses at rung 1: there
`train_id_set_hash` is the UNION over the lineage while `dataset_sha` keeps naming that row's
own export, so the two names are two quantities and the loader stops demanding they agree.
Register with `--init-from <model id>` at rung >= 2 and the union and its sidecar are written
for you; an existing row is migrated with `--backfill-init-from`. It is the **id-set** hash and
never a file sha: `id_set_hash` is
`sha256("\n".join(sorted(set(ids))))` (`src/pinq_train/split.py` ~77), so it proves MEMBERSHIP and
is insensitive to row order and to reformatting. `docs/HPC_WAVES.md` §1 draws the same line from
the other side: a wave's `dataset_sha256` IS a file sha, and is a different quantity over
different bytes.

**The id-set sidecars.** `train_ids.<hash16>.txt` sits beside each export manifest and holds the
sorted, unique `suite_id/task_id` lines the hash is taken over. Written 2026-09-15 for the
datasets the registry names, and re-verified on that date by reading each file, recomputing
`id_set_hash` over its lines, and comparing against both the `<hash16>` in its own name and the
manifest beside it:

| sidecar | ids | manifests carrying that `train_id_set_hash` |
| --- | --- | --- |
| `data/rl/train_ids.037f94b5bec536e2.txt` | 2,664 | `data/rl/sft.manifest.json` |
| `data/rl/headline/train_ids.c8b89f4dc1520c6f.txt` | 2,525 | `data/rl/headline/sft.manifest.json` |
| `data/rl/headline/train_ids.c1444a77617b402d.txt` | 1,027 | `data/rl/headline/pairs{,.rater,.reaches}.manifest.json` |
| `data/rl/train_ids.f3632ac09d84fc1c.txt` | 1,933 | `data/rl/pairs{,.rater,.reaches}.manifest.json` |
| `data/rl/headline/train_ids.11bd5de6d558be20.txt` | 2,526 | none — see below |

The last one is not an export's sidecar and has no manifest: it is the LINEAGE union written by
`register_checkpoint.py --init-from` (the headline SFT set's 2,525 ids with the pairs set's
1,027, union 2,526), and what names it is the `train_id_set_hash` of a rung-2 REGISTRY ROW
rather than a manifest. It resolves by the same filename rule, which is the point of writing a
preimage at all. RE-MEASURED on this branch, all five agreed on every check: each line unique,
each file already sorted, and the recomputed hash's first 16 hex equal to the filename's and —
for the four export sidecars — to the manifest's `train_id_set_hash`.

A re-export chain in flight also writes under `data/rl/`, so a sidecar can appear here that no
registry row or manifest names yet: `data/rl/panel2/` is being written by the panel-2 re-export
and its files are not installed. Read the table as the sidecars that are IN USE, not as
everything on disk. The
sidecar is a convenience for READING the ids; **the reader keys on the hash**, so a sidecar
that is regenerated or reformatted is still the same dataset as long as the id set is — and a
sidecar whose name no longer matches its contents is a broken file, not a new dataset.

## 8. Tier B dev gate

**Commit first.** `RunManifest.run_id` prefixes `dev-` when the tree is dirty
(`src/pi_run/manifest.py`), and a `dev-` run is training-only, never an eval arm — running this
sweep from an uncommitted tree burns GPU-hours and API spend on rows the paper cannot use.

**`pi run --sweep` refuses a dirty tree instead of banner-and-proceed (2026-09-17).** Today a
400-unit evaluation sweep was launched while three unrelated tracked files sat uncommitted in
this checkout; every one of the 400 runs came out `dev-` stamped — training-only, per above,
inadmissible as an eval arm — and the loss was invisible until the manifests were read
afterward, because the DIRTY banner `pi run` already prints is emitted once per suite, after
real per-unit planning, deep in a long log, and the operator missed it. `--sweep` on a dirty
tree now exits non-zero before a single suite is dispatched, naming the dirty files and both
ways forward — commit or set aside the changes, or launch from a pinned clean worktree
(`git worktree add --detach`) — and `--allow-dirty` restores exactly today's prior behaviour
for the one legitimate case, deliberately collecting `dev-` training data. A plain
(non-`--sweep`) `pi run` is unchanged either way. `GitInfo` already carries `dirty` and
`dirty_files`, which is every fact about the tree provenance needs regardless of which side of
the refusal a launch took, so no field records whether `--allow-dirty` itself was passed.

**`--sweep` also refuses a silently-discarded `--task` (2026-09-17, found by another lane).**
`grid_flag_conflicts` already refused a conflicting `--seeds`/`--n`/`--max-turns`/`--k`/
`--concurrency` against the grid's own values (431bada onward); it never looked at `--task`,
and the grid loop's `sub.task = list(grid.task_ids) or []` discards an explicit `--task`
unconditionally -- there is no shape in which it survives, which makes this worse than the
other five. A grid that declares its own `task_ids` overwrites `--task` with them; a grid that
declares none (the common case -- most grids select by `n_tasks`/`split` instead) overwrites it
with `[]`, which then falls through to `n_tasks`-based selection, so `pi run --sweep <200-task
grid> --task one_task` ran all 200 while its operator believed they had pinned one, with
nothing in the banner to catch it -- exactly the failure that would corrupt the one-unit
validation sweep this runbook's earlier sections use to prove a tree is clean before a large
campaign. `grid_flag_conflicts` now takes the grid's `task_ids` and refuses both dangerous
shapes: an explicit `--task` that disagrees with the grid's own declared `task_ids`, and an
explicit `--task` against a grid that declares none at all (naming the `n_tasks`-based count it
would have run instead). `--task` uses `action="append", default=[]` rather than the `None`
sentinel the other five flags use, since it is repeatable -- the explicit-vs-default test here
is "non-empty list" vs `[]`, not "not None" vs `None`, and a passed-but-empty `--task` (there is
no such CLI shape today, but the guard does not assume one) would still read as unpassed, same
discipline as the other five, different sentinel because that is the sentinel this flag has.

**The list was the defect; the check is now a diff (2026-09-18).** Adding `--task` by hand made
seven flags checked out of the seventeen fields the grid loop assigned, and the eighth would
have been found the same way -- by someone whose campaign had already run. Three measurements
closed it as a class rather than an instance:

* `grid_flag_conflicts(Namespace(budget_cap=4), <a cap-8 grid>)` returned `()`, so
  `pi run --sweep conf/grids/frames_trained.yaml --budget-cap 4` was accepted, the flag was
  discarded, and the campaign ran at 8. `budget_cap` is the sole axis of the spend curve
  (`frames_trained_cap{4,8,12,16,24}` differ in that field and in nothing else), so the wrong
  cap is not a slow run, it is a mislabelled point on a published curve. The checker's own
  docstring named `budget_cap` among the fields the assignment broke.
* by AST, the loop assigned 17 fields and the checker inspected 6. `--split` and
  `--task-offset` had never been named in any list: a discarded `--split` reports the wrong
  population (`pi_eval.report.ELIGIBLE` filters on `split = 'test'`), and a discarded
  `--task-offset` silently re-runs a slice a pilot already burned. Nothing was checked but
  never assigned, so there was no check that could never fire.
* `--k` was CHECKED AND STILL DISCARDED: the check compared it with `grid.k` while the loop
  assigned `grid.k_for(suite_id)`. THREE COUNTS, all real, and the first report of this gave
  the narrowest one in the widest one's sentence. NINE grids declare `k_by_suite` tracked at
  `main`; TEN in a working tree that also has the untracked `frontier_trained_musique_cap24`;
  and on FOUR of them a per-suite value actually differs from the grid-wide `k`. The other five
  or six declare it redundantly, every entry equal to `k`, so they take the per-suite path
  without the value changing. Nine or ten is the mechanism exposure, four is where the check
  and the assignment could disagree and a wrong `k` could be recorded. The four does not move
  between the two trees, because the tenth grid is one of the redundant ones;
  `tier1_confirmatory`'s is `{strategyqa: 2, wiki2: 3}` against `k: 5`, so `--k 5` was accepted
  and strategyqa ran at 2. A set comparison alone would not have found this one.

So there is no longer a list of inspected fields. Every grid override lives in
`grid_child_namespace`, and `grid_discarded_flags` DIFFS the operator's namespace against the
one that function will dispatch, per suite, before the first suite runs. A field becomes
checked at the moment it becomes assigned. `--arm`, `--spend-cap` and `--sweep` are the three
exemptions, each carrying its reason in `GRID_DIFF_EXEMPT` (a filter, a cap that is honoured
grid-wide, and structural recursion control), and a test walks `cmd_run`'s syntax tree so a
grid assignment added outside that one function fails the suite. Replayed against the flag
sets real launches use, over every grid on disk (312 combinations): 0 newly refused, 0
refusals lost.

TWO FLAGS THE PARSER CANNOT DISTINGUISH, and neither is fixable inside the check. The refusal
rests on comparing a dest against the parser's own default, which is why `--seeds`, `--n`,
`--max-turns`, `--k`, `--budget-cap`, `--split`, `--suite` and `--concurrency` all default to
`None` (`resolved_run_defaults` applies the real values on the non-`--sweep` path). But
`--task-offset` defaults to `0`, so an explicit `--task-offset 0` is byte-identical to an
unpassed one, and `--pilot` is `store_true`, so only an explicit `--pilot` is visible. Both are
blind in exactly the direction where the typed value already equals the default, so neither can
cause a WRONG refusal -- each can only miss one. Giving `--task-offset` a `None` default would
close it and would also change what a standalone `pi run` does, so it is a decision rather than
a fix and is recorded as one, asserted in
`test_the_two_flags_whose_typed_value_argparse_cannot_distinguish`.

**`--unit-timeout`'s help text drifted from its own default (2026-09-17, found independently by
two lanes).** The literal in the help string read `1800`; `DEFAULT_UNIT_TIMEOUT_S`
(`src/pi_run/worker.py`) is `3600` -- raised 2x in a prior, measured fix
(`test_the_unit_timeout_default_clears_the_slowest_unit_ever_measured`: 3600s is 5x the 726s
slowest unit ever observed, and 1800s left a tau2 treatment unit plus a few rate-limit ladders
no headroom at all) that never touched the CLI help text describing it. The help string now
interpolates `DEFAULT_UNIT_TIMEOUT_S` with an f-string instead of restating a copy of its
value, so the two cannot drift apart again the way they just did.

**Kill orphaned workers before relaunching (measured today).** A partially-died sweep can leave
worker processes running past the parent that launched them, and relaunching into the same
`--runs-root` without killing them first lets two processes write the same run directory — a
correctness bug, since whichever write lands last silently wins, not a performance one.

**And commit only in an announced merge gap.** `code_version` is inside `semantic_hash` and so
inside `run_id` (`src/pinq/types.py` ~633, the `semantic_hash` dict), which means a sweep
restarted after a commit mints a whole new run set instead of continuing the old one. It does
not even need a restart to go wrong: `pi run` spawns worker processes continuously and each one
imports from disk as it starts, so a merge into `src/` while **any** session's sweep is running
puts new code under the old `code_version` stamp — a run set that is no longer one thing and
cannot be made into one afterwards. Hence the rule: **merges to `main` happen only in announced
gaps, with no sweep running in any session.** Docs, tests and new files nothing imports yet may
land at any time, with a heads-up. MEASURED 2026-09-15 on the FRAMES arms, which were run inside
such a gap: one `code_version` (`f3f3f0f`) across all 1,648 manifests, both arms, and no `dev-`
run id among them.

```bash
set -a; . ./.env; set +a
export LITELLM_BASE_URL=http://127.0.0.1:4000   # the proxy tunnel; DRAFTER/ANSWERER/USERSIM/JUDGE stay
                                                 # at gpt-oss-120b, through the passthrough
for suite in musique strategyqa wiki2; do
  PI_MODEL_INQUIRER=qwen3-8b-sft .venv/bin/pi run --sweep conf/grids/dev_select_${suite}.yaml
done
.venv/bin/pi compact
.venv/bin/pi score --gold-root "$PWD/data/gold"
for suite in musique strategyqa wiki2; do
  .venv/bin/pi train gate --parquet-dir scores/parquet \
      --grid-name dev_select_${suite} --baseline-grid-name dev_baseline_${suite} \
      --checkpoint-arm inquirer_trained --baseline-arm inquirer_prompted \
      --out artifacts/gate/qwen3-8b-sft.${suite}.json
done
```
(`dev_baseline_{musique,strategyqa,wiki2}` are the `grid_name`s already run under
`conf/grids/reroll/dev_*.yaml`; `--baseline-grid-name` must be passed explicitly since the
checkpoint and baseline grid names never match.) `pi run` here has `PI_GOLD_ROOT` unset by
construction (§2); `pi score` needs it.

Each verdict JSON carries `"passed": true|false` (exit 0/1). **Pass criteria (plan I.9, Tier
B):** paired delta vs. prompted@same-base on evidence coverage and coverage at depth >= 2, CI
excluding 0. **Cost:** ~$5/checkpoint across the three suites (plan I.8). This is checkpoint
*selection*; the test split stays untouched until one checkpoint per arm passes here.

**Since D9 (2026-09-15) the COVERAGE criterion is matched-cost, and it is the default.** The
rule and its justification are `docs/TRAINING.md` §5.3.2; what changes at the keyboard:

- `pi train gate` takes `--coverage-rule {cap8,matched_cost}` and **defaults to `matched_cost`**
  (MEASURED 2026-09-15: `src/pi_run/cmd_train.py` ~3113 declares `choices=("cap8", "matched_cost")`,
  `default="matched_cost"`). The loop above therefore runs the new rule with no flag typed.
- What it contrasts: the **trained arm at its own stop *k*** against the **base's own prefix
  ladder at the same *k*** (`scores.frontier_q#k`) — not against the base's whole cap-8
  episode. The cap-8 delta is not lost: it is reported per suite in
  `matched_cost.by_suite[*].cap8_coverage_delta` as the **cost line** of an ordered pair, and is
  not gated.
- The verdict JSON gains three things: `coverage_rule` (which rule produced the number), a
  top-level `matched_cost` block (`pooled`, `by_suite`, mean *k* on both sides, the instrument
  check, the column definitions), and `facet_rule`. Under `matched_cost`, `facet_breadth` is
  **report-only** (`facet_rule: "report_only"`, `"gated": false`; its value, CI, *n* and the
  no-loss outcome it would have had all stay in the JSON) — which is why "no loss on facet
  breadth" has come out of the pass criteria above.
- **Pass `--coverage-rule cap8` to reproduce a pre-D9 verdict bit for bit.** That path reads no
  ladder, so it is also the way to gate a parquet that has no `frontier_q#k` column.
- An inconsistent ladder is a refusal, not a warning: the gate raises `LadderInconsistent`
  unless every baseline ladder it reads ends at that run's own `evidence_coverage`, and tells
  you to re-score or to pass `--coverage-rule cap8`.
- **Two gated criteria did not move, and both still fail every checkpoint.** Read
  `docs/TRAINING.md` §5.3.2's closing paragraph before you read a `"passed": false` as a
  verdict on the checkpoint.

**Spend: `usd_billed` is the per-run number, and `terminal_usage` is not.** Each run
directory's `status.json` carries both, and they answer different questions:

- `usd_billed` — the invoice, `sum(call.usd for call in ledger.calls if not call.cache_hit)`
  (`src/pi_run/worker.py` ~667). **This is the authoritative per-run spend**: it is what
  `--spend-cap` counts down and what a sweep's summary sums.
- `reconcile.turn_usage` / `reconcile.terminal_usage` — the two halves of the **token-parity
  check**, not a bill. `turn_usage` folds the per-turn usage, which is where the Inquirer's ask
  turns live; `terminal_usage` is everything charged to no turn — the STOP `act()`, the final
  draft and the Answerer. They exist so `turns + terminal == ledger` can be asserted, and both
  count cache hits, which `usd_billed` excludes.

So **`reconcile.terminal_usage.usd` is not the cost of a run**: on an Inquirer-bearing arm it
omits every ask. MEASURED 2026-09-15 over the 824 `inquirer_prompted` FRAMES runs named in
`artifacts/frames/run_ids.inquirer_prompted.txt`, summing those fields out of each run's
`status.json` and `calls.jsonl`:

| field | USD |
| --- | --- |
| `usd_billed` (the invoice) | **8.8653** |
| `reconcile.turn_usage.usd` | 9.6335 |
| `reconcile.terminal_usage.usd` | 2.0616 |
| turn + terminal (= every call, cache hits included) | 11.6951 |
| the terminal calls alone, cache hits excluded — like for like against `usd_billed` | **1.2579** |

Read the terminal half alone and this arm reports **$1.26 against $8.87 actually billed**, a
7.0x under-report — because the Inquirer's 5,199 calls ($3.7336 billed) and the drafts they
trigger are all in `turn_usage`. By actor over those same runs, billed: drafter $4.5740 (9,574
calls), inquirer $3.7336 (5,199), answerer $0.5577 (824). For scale, the `drafter_only` twin of
that arm — same 824 tasks, no Inquirer — really did bill $1.6427 over its own 824 runs, so
terminal-only makes an Inquirer-bearing arm report roughly its own comparator's invoice.

## 9. Test grids (only after a checkpoint passes Tier B)

Run once per selected checkpoint, test split only:
**One environment variable pins the Inquirer role for every arm in an invocation.** A grid that
lists both `inquirer_prompted` and `inquirer_trained` therefore runs once per arm with `--arm`,
each time with the `PI_MODEL_INQUIRER` that arm needs; the arm id is inside the run id, so the
two invocations never collide and resume keeps them apart.

**There is no `--resume` flag.** Resume is the DEFAULT, and `--no-resume` is how you opt out:
re-issuing any `pi run` in this runbook picks up where it stopped and bills nothing for a unit
that already has a `status.json`. MEASURED 2026-09-15:
```console
$ .venv/bin/pi run --resume
pi: error: unrecognized arguments: --resume       # exit 2
```
(`src/pi_run/cli.py` ~532 passes `resume=not a.no_resume`; the flag itself is declared at ~1959
as `--no-resume`, "re-run units that already have status".)

```bash
G=conf/grids
# teacher pin: the prompted arm at gpt-oss-120b (and drafter_only, which has no Inquirer)
PI_MODEL_INQUIRER=openai/aws/gpt-oss-120b .venv/bin/pi run --sweep $G/tier1_trained_qa_teacher.yaml --arm drafter_only
PI_MODEL_INQUIRER=openai/aws/gpt-oss-120b .venv/bin/pi run --sweep $G/tier1_trained_qa_teacher.yaml --arm inquirer_prompted
# fair baseline: the same grid body, the untrained base as the prompted Inquirer
PI_MODEL_INQUIRER=qwen3-8b-base           .venv/bin/pi run --sweep $G/tier1_trained_qa_base.yaml --arm inquirer_prompted
# the trained arm: one invocation per selected checkpoint
PI_MODEL_INQUIRER=qwen3-8b-sft            .venv/bin/pi run --sweep $G/tier1_trained_qa_base.yaml --arm inquirer_trained
PI_MODEL_INQUIRER=qwen3-8b-dpo-control    .venv/bin/pi run --sweep $G/tier1_trained_qa_base.yaml --arm inquirer_trained
# kill switches carry no trained arm; the Inquirer-bearing switches run at the base pin
PI_MODEL_INQUIRER=qwen3-8b-base           .venv/bin/pi run --sweep $G/tier1_trained_killswitch.yaml
# frontier: both arms, so twice per cap
for cap in 4 8 12 16; do
  PI_MODEL_INQUIRER=qwen3-8b-base        .venv/bin/pi run --sweep $G/frontier_trained_musique_cap${cap}.yaml --arm inquirer_prompted
  PI_MODEL_INQUIRER=qwen3-8b-dpo-control .venv/bin/pi run --sweep $G/frontier_trained_musique_cap${cap}.yaml --arm inquirer_trained
done
```
`pi run --sweep ... --arm X` is the existing per-arm filter (`src/pi_run/cli.py`, grid dispatch).
Two invocations of one grid under two pins produce distinct `run_id`s because `model_pin_hash`
is inside `semantic_hash` (`src/pinq/types.py`).

**`drafter_only` is the exception, and the second pin is free.** That arm's Inquirer is
`NeverAsk` (`src/pinq_expt/arms.py`; `src/pinq_expt/fakes.py` ~87), which is not LLM-backed, and
`collect_pins` reads pins off the components actually built (`src/pi_run/worker.py` ~670) — so a
`drafter_only` manifest carries **no inquirer pin at all**. `model_pin_hash` hashes only the
roles present, so this arm's `run_id`s are **identical under every `PI_MODEL_INQUIRER`**, which
is exactly true of what it ran. Two consequences worth knowing before you read a table:

- Running the base-pin grid after the teacher-pin grid **resumes** `drafter_only`'s rows and
  bills **$0**. That is correct behaviour, not a failure — and not evidence that the base-pin
  invocation produced anything of its own.
- **Select `drafter_only` rows by run-id list, never by `grid_name`.** One set of rows exists,
  under whichever grid name reached it first; asking for "the base grid's `drafter_only`" by
  name gets you the teacher grid's rows, or none.

MEASURED 2026-09-15 on the 824 FRAMES `drafter_only` manifests: pins `{answerer, drafter}` on
824 of 824, with no `inquirer` key on any of them — while the 824 `inquirer_prompted`
manifests beside them carry all three.

**`--spend-cap` is the GRID's budget, not each suite's.** `pi run --sweep` calls the runner once
per suite and threads the *remaining* budget through, so a three-suite grid at `--spend-cap 50`
bills at most $50 rather than $150; the last suite gets whatever the first two left, and a suite
reached with nothing left is refused rather than run at a zero cap (`src/pi_run/cli.py` ~338-382,
whose comment records the bug this fixed). The cap is **not** in `semantic_hash`
(`src/pinq/types.py` ~604), so stopping a sweep at the cap and re-issuing it re-enters the same
`run_id`s and resumes: the cap changes what gets run, never what a run IS.

`tier1_trained_qa_{teacher,base}` carry `arms: [drafter_only, inquirer_prompted,
inquirer_trained]` at `k_by_suite {musique 5, strategyqa 2, wiki2 3}` (matching
`tier1_confirmatory`, not the dev gate's uniform k=5); $11.76/API-Inquirer arm, $6.94/
local-Inquirer arm (plan I.10). `tier1_trained_killswitch` omits `inquirer_trained` (read against
the rows above) and omits wiki2 (its costliest switches are decidable on musique+strategyqa
alone at n=200 — the file's own CI table). The four `frontier_trained_musique_cap*` files differ
only in `budget_cap`; $32 for all four together. `frames_trained.yaml` (external, gold-free,
`n_tasks: 824`) runs the same two-pin way once a checkpoint is otherwise done.

**The contamination check:** `pi_eval.report.train_split_violations` refuses any trained-arm row
recomputed off anything but `test` — it's what makes the commit-first rule in §8 non-optional.

## 10. What to record after each step

For every rung-1/rung-2 run and every sweep, write down — don't leave it "inferable":

| what | where it comes from |
| --- | --- |
| `cfg.sha` (`config_sha`) | printed by `pi train rung1`/`rung2`'s own preflight/train JSON |
| the manifest | `rung1.manifest.json` / `rung2.manifest.json`, beside the adapter in `--out` |
| `adapter_sha` | `merge.manifest.json` if merged, else the adapter dir's `*.safetensors` + `adapter_config.json` at its root (not the tree — `checkpoint-*/` subdirs must not move it) |
| the `runs/` directory a sweep wrote into | `pi run`'s `--runs-root` (default under the repo root) |
| what a sweep actually billed | each run's `status.json` `usd_billed`, summed — **not** `reconcile.terminal_usage.usd` (§8) |
| `run_id`, `scorer_hash`, `graph_version` per reported number | CONTRIBUTING.md rule 1 — missing any one, it doesn't go in a table |

Then place the number: throughput -> plan I.15's table (§3 says which cell); Tier A/B verdicts
-> plan I.9's pass/fail read for that tier; Tier C contrasts -> the row of plan I.9's table they
answer (trained-vs-prompted, DPO-vs-SFT, control-vs-rater-vs-reaches, kill switches, frontier,
transfer), then into the paper's tables via `pi render`, each with the `provenance.json` beside it.
