# Composed MuSiQue pairs (`musique_x2`)

Declared before any data in `artifacts/composed_pairs_20260923/DECLARATION.md` (commit c2bdf77).
The suite is eval-only and exploratory. It is built by `pi_eval.build.compose_build` into an
isolated root, never into the shared `data/`.

| file | role |
|---|---|
| `predict.py` | the additivity prediction: per composed task and arm, the constituents' solo levels in the Table 3 population, weighted by node count. Locked first against `table1_by_seed.json`. |
| `analyze.py` | breadth (of 2), coverage and dwr at equal realized spend (the symmetric seed-matched rule, with the recipe's seeds averaged within the task), a paired bootstrap clustered on the pair, and the interference DiD. Locked first against `table1_by_seed.json`. |
| `hpc_campaign.sh` | the campaign as one LSF CPU job on HPC (see below) |
| `conf/grids/musique_x2_20260923.yaml` | 96 tasks x seeds 0,1, cap 16, max_turns 16, k 5, split test |

## Running the campaign on HPC

The campaign runs on HPC, not on the Mac. That is the user's decision. It runs as one CPU job (no
GPU). The job runs a litellm proxy on exactly `http://127.0.0.1:4030`, because `base_url_sha` is
part of run identity: one URL string gives one identity. The proxy uses the staged tree's committed
`conf/serving/litellm.yaml`. The job rewrites the `api_base` of three routes only
(`qwen3-8b-base` and the recipe's `-s1` and `-s2`) to serve 917484 at `gpu-n563:8302`. The frozen
roles (`openai/aws/gpt-oss-120b`) go to the gateway. The gateway keys come from
`/u/user/.pinq_env`. That file must be mode 600, and the job never prints the keys.
`PI_GOLD_ROOT` is unset for every rollout. `PI_CANARY_SALT` must be set in the env file.
`PI_CANARY_ROOT` is the staged isolated root.

The campaign runs three streams. Each has its own runs root and cache, and 192 units:

| stream | `--arm` | `PI_MODEL_INQUIRER` |
|---|---|---|
| s1 | inquirer_trained | qwen3-8b-dpo-stacked-notdone-both-s1 |
| s2 | inquirer_trained | qwen3-8b-dpo-stacked-notdone-both-s2 |
| base | inquirer_prompted | qwen3-8b-base |

Before any spend, the job refuses if any of these fails:
- the staged tree is dirty, or its `git_info` is not the merged sha;
- the isolated root's four sha256 values are not the declared build;
- a canary appears on the corpus side;
- port 4030 is already bound;
- the upstream does not publish all three questioners;
- a route does not return a completion, or a qwen route answers with `<think>`;
- the suite does not load 96 tasks.

After the first finished unit of each stream, the job reads that unit's manifest back. It checks
the arm, the pins, the `code_version`, `dirty`, the split, the grid, the canary and the status. On
a mismatch it kills the job.

At the end, the job reads back every unit, runs `pi verify firewall` over each cache and counts the
`status == ok` units. It writes `.CAMPAIGN_COMPLETE` only at 3 x 192. It tars the runs roots,
caches and wrapper records to `/proj/pinq/user/musique_x2-<sha8>/`.

The default is `--concurrency 2` per stream, so 6 concurrent questioner requests. Serve 917484
also carries other traffic, so the total is kept under 8. The spend cap is $40 per stream.
`pi run` resumes by default, so after a kill you run `submit` again.

### Commands

`<SHA>` is the full 40-character merged sha. Run each step only after the previous one succeeds.

`stage` reads the isolated root at `/private/tmp/musique_x2_root` (set `PINQ_X2_ROOT` to use
another path), and a reboot can clear that directory. If it is gone, rebuild it with the command
below. `PI_CANARY_SALT` must come from `.env`. On 2026-09-23 a rebuild made this way was byte-identical
to the original in all five files. `stage` refuses any other build.

```bash
set -a; source .env; set +a; unset PI_GOLD_ROOT
.venv/bin/python -m pi_eval.build.compose_build \
  --musique-corpus data/corpora/musique/38f5afb69fb7ea18/tasks.jsonl \
  --musique-gold data/gold/graphs/musique/v1.jsonl \
  --train-ids data/rl/notdone_v2/train_ids.d6750e34239f398f.txt --root /private/tmp/musique_x2_root
```

```bash
# 1. On the Mac, from the repository. SSH_AUTH_SOCK must be the socket that holds the HPC certificate.
bash scripts/composed_pairs/hpc_campaign.sh stage <SHA>

# 2. On the HPC login node (ssh user@login-node.example.com).
bash /proj/pinq/user/musique_x2-<sha8>/tree/scripts/composed_pairs/hpc_campaign.sh submit <SHA>
bjobs -w -J pinq-x2-<sha8>
tail -f /proj/pinq/user/musique_x2-<sha8>/logs/job.<jobid>.out

# 3. On the Mac, after the job ends. This unpacks into /private/tmp/musique_x2_campaign/<sha8>/.
bash scripts/composed_pairs/hpc_campaign.sh pull <SHA>
```

### Scoring and reading, on the Mac

Score with the code that ran the units: a clean worktree at `<SHA>`.
`PI_GOLD_ROOT` is set in the scoring and analysis processes only.

```bash
D=/private/tmp/musique_x2_campaign/<sha8>; X2=/private/tmp/musique_x2_root
git worktree add --detach /private/tmp/x2-score <SHA>; WT=/private/tmp/x2-score
PY=<main>/.venv/bin/python
for L in s1 s2 base; do
  PYTHONPATH=$WT/src $PY -m pi_run.cli compact --root $WT --runs-root $D/runs_$L --out $D/scores_$L
  PI_GOLD_ROOT=$X2/data/gold PI_CANARY_ROOT=$X2 PYTHONPATH=$WT/src $PY -m pi_run.cli score --root $WT \
    --parquet $D/scores_$L --runs-root $D/runs_$L --gold-root $X2/data/gold \
    --corpora-root $X2/data/corpora --graph-version v1 --allow-no-judge
done
PI_GOLD_ROOT=$X2/data/gold PYTHONPATH=$WT/src $PY $WT/scripts/composed_pairs/analyze.py \
  --arm s1=$D/scores_s1:$D/runs_s1:inquirer_trained:qwen3-8b-dpo-stacked-notdone-both-s1 \
  --arm s2=$D/scores_s2:$D/runs_s2:inquirer_trained:qwen3-8b-dpo-stacked-notdone-both-s2 \
  --arm base=$D/scores_base:$D/runs_base:inquirer_prompted:qwen3-8b-base \
  --prediction $WT/artifacts/composed_pairs_20260923/prediction.json \
  --lock-gold-root <main>/data/gold --artifacts-root <main> \
  --out <main>/artifacts/composed_pairs_20260923/composed.json
```

`analyze.py` refuses in these cases:
- its Table 1 lock fails;
- an arm mixes code versions, base URLs or models;
- an arm contains dev-, dirty or canary-hit runs;
- a graph is missing.

A cell over fewer than two clusters, or with a zero-width interval, reads DEGENERATE and is never
DECIDED. `answer_correct` is not a declared metric here, and it cannot be read on this suite. The
composed gold answer is `(1) A (2) B` with no composed aliases, so an answer that says "Italy" for
"Italian Republic" scores 0.
