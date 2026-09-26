# ProactiveInquirer

Code for measuring and training **proactive information-seeking** in tool-using agents: going after
information the user never asked for.

A tool-using agent usually does what it is asked, yet a task often needs information the user never
mentions. This repository studies *what* an agent pursues unasked, in two forms:

- **Horizontal proactivity** pursues a need the current state already names. A customer gives a name
  and a ZIP code, so the agent looks up the account.
- **Vertical proactivity** pursues a need that only newly found evidence names. The account lists
  the order, the order names the product, and the product lists the size-8 variant.

It contains three things:

1. **Need graphs and metrics.** Each task's required evidence is a graph recovered from the
   benchmark's own decomposition, with an edge wherever one need can be named only after another is
   found. A run is scored from its transcript against that graph, with no model judge, and agents
   are compared after the same number of questions.
2. **Q&D (questioner and drafter).** The agent loop pairs a questioner, which asks one question at a
   time or stops, with a frozen drafter, which folds the retrieved evidence into a draft. Training
   forks a run at one state, continues it after several candidate questions, and prefers the
   question whose continuation retrieves more of the required evidence. No reward model or judge is
   involved.
3. **The experiment stack.** Benchmark adapters, a sweep runner with a content-addressed response
   cache, gold-only scoring behind an import firewall, statistics, and a training ladder (prompt
   search, SFT, DPO).

## Quick start

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/). After the install, `make gate` and
`make smoke` need no API keys.

```bash
git clone <repository-url> ProactiveInquirer && cd ProactiveInquirer
uv venv --python 3.12
uv pip install -e ".[dev,run,suites,analysis,tau2]"
make gate     # ruff, format, the 4 import contracts, the suite audit, unit tests, path guard
make smoke    # end to end on a synthetic suite
```

`make smoke` builds a synthetic suite whose need graphs have closed-form answers, runs three offline
arms on it, compacts and scores the runs, and checks the gold firewall. `make gate` is what CI runs.

Optional extras: `train` (the GPU stack for SFT and DPO), `judge` (model-judged metrics), `mine`,
`serve` and `pare`. See `pyproject.toml`.

## Configuration

Model calls go through LiteLLM, so any OpenAI-compatible endpoint works. Copy the template, fill in
the endpoint and one model pin per role, and load it:

```bash
cp .env.example .env
set -a; . ./.env; set +a
.venv/bin/pi env doctor      # one short live call per pinned model; prints no secrets
```

`.env` is git-ignored and is the only place credentials live. The client never defaults a model pin,
so two arms cannot silently run on different models.

## Data

No benchmark text is redistributed here. Each suite is downloaded, checked against a pinned sha256,
and split into a public corpus (`data/corpora/`, the only thing an agent may read) and a gold need
graph (`data/gold/`, read only by the scorer):

```bash
.venv/bin/pi data fetch --suite musique
.venv/bin/pi data build --suite musique
.venv/bin/pi data status
.venv/bin/pi gold stats --suite musique
.venv/bin/pi suites map        # which suites train the questioner and which measure it
```

The suites are MuSiQue, StrategyQA and 2WikiMultiHopQA (multi-hop question answering), FRAMES and
composed MuSiQue pairs (evaluation only), τ²-bench retail and airline (customer service with a
simulated customer, which needs `TAU2_DATA_DIR`), and a synthetic calibration suite. `docs/DATA.md`
gives each source's licence and build steps, and `docs/SUITES.md` records which role each suite
plays.

## Running an experiment

A sweep runs (task × arm × seed) units and records every model call in the response cache:

```bash
.venv/bin/pi run --suite musique --split test --n 200 --seeds 0 1 \
    --arm inquirer_prompted --arm inquirer_trained
.venv/bin/pi compact                          # runs/ -> scores/parquet/
PI_GOLD_ROOT=data/gold .venv/bin/pi score     # the only step that reads gold
.venv/bin/pi agg --primary                    # tables, each with a provenance record
```

`inquirer_prompted` is the base model prompted as the questioner, and `inquirer_trained` is a trained
questioner served at `PI_MODEL_INQUIRER`. The other arms (controls such as `inquirer_depth1` and
`random_q`, baselines such as `self_ask`, `ircot` and `par2_rag`, and ceilings such as
`gold_evidence`) are defined in `src/pinq_expt/arms.py`.

## Training Q&D

`pi train` runs the ladder. Each step has a dry preflight, and the GPU steps need
`uv pip install -e ".[train]"`.

| Command | Step |
|---|---|
| `pi train frontier-states` | Find the recorded states where a choice of question existed |
| `pi train sample-candidates` | Fork each state into candidate continuations |
| `pi train export` | Turn recorded runs into SFT rows and preference pairs (reads gold) |
| `pi train rung0` | Prompt search over the questioner template (no GPU) |
| `pi train rung1` | Rejection-sampling SFT |
| `pi train rung2` | DPO on same-state preference pairs |
| `pi train eval-offline` | Dev NLL, stop decisions and pair accuracy for a checkpoint |
| `pi train status` | What is on disk and what is missing |

`docs/TRAINING.md` explains each rung, and `docs/GPU_RUNBOOK.md` goes from an empty rented GPU box to
a registered, served checkpoint (`scripts/hpc/register_checkpoint.py`, `scripts/hpc/serve.sh`).

## Using the loop in your own agent

`pinq` depends on nothing outside the standard library. Implement the `Retriever`, `Drafter` and
`Answerer` protocols from `pinq.protocols` and pass them to `run_loop`:

```python
from pinq.budget import BudgetLedger
from pinq.loop import run_loop

trajectory = run_loop(
    view=task_view,  # what the agent may see of the task
    inquirer=questioner,  # asks one question or stops
    retriever=my_retriever,
    drafter=my_drafter,  # must be pure in (view, evidence, seed)
    answerer=my_answerer,
    ledger=BudgetLedger(cap=8),
    max_turns=16,
)
```

## Working rules

Code comments cite these as "CLAUDE.md rule N", after the maintainers' working-rules file.

1. A number without provenance is not a result. Every reported value names its `run_id`,
   `scorer_hash` and `graph_version`.
2. A test that passes without the fix is not a test. Write the failing test before the fix.
3. Never state a measurement you did not take.
4. Never weaken a test to make it pass. Either the code is wrong or the test encodes a wrong belief,
   and that is what gets fixed.

## Rules the code enforces

- **The gold firewall.** `pi_eval` is gold-only, and nothing outside it may import it. Four layers
  hold this: an import-linter contract, a type wall (the task view shares no field with a gold node),
  a process split (rollout workers run with `PI_GOLD_ROOT` unset), and canary nonces on gold records,
  scanned in every serialized request.
- **A pure drafter.** `Drafter.draft()` depends only on the view, the evidence and the seed, so every
  change in what the agent holds is caused by a question.
- **Reproducible by cache.** Temperature 0 is not bitwise deterministic, so the content-addressed
  request cache is the reproducibility artifact, and `pi cache verify` checks it
  (`docs/REPRODUCE.md`).

## Repository layout

| Path | Contents |
|---|---|
| `src/pinq/` | The agent loop, types, budget ledger and protocols |
| `src/pinq_adapters/` | One adapter per benchmark, and the LLM client. Reads `data/corpora/` only |
| `src/pinq_expt/` | The arms that are compared, including controls and baselines |
| `src/pi_eval/` | Gold-only: need graphs, matchers, metrics and statistics |
| `src/pi_run/` | The `pi` CLI: sweeps, cache, manifests, scoring and reports |
| `src/pinq_train/` | The training ladder |
| `conf/` | Sweep grids, training and serving configs, cohorts, the checkpoint registry |
| `scripts/` | Tools beside the CLI, and per-study analyses (indexed in `scripts/README.md`) |
| `tests/` | The test suite |
| `docs/` | Data, suites, metrics, training, reproduction and design decisions |
| `prereg/` | The judge's frozen noise floors (σ_J), which the statistics read |

Not included: built data, experiment outputs (`runs/`, `scores/`, `artifacts/`), the paper, and the
cluster job scripts the experiments were launched with.

## Documentation

| File | Read it for |
|---|---|
| `docs/DATA.md` | Each data source, its licence, and how to fetch and build it |
| `docs/SUITES.md` | Which suites train the questioner and which measure it |
| `docs/PROACTIVITY_METRICS.md` | The metrics and what each one measures |
| `docs/RUNBOOK.md` | From a clean machine to a result, command by command |
| `docs/REPRODUCE.md` | What reproducible means here, and how to check it |
| `docs/TRAINING.md` | The training ladder, rung by rung |
| `docs/GPU_RUNBOOK.md` | From an empty rented GPU box to a served checkpoint |
| `docs/DECISIONS.md` | The judgement calls the code depends on, with their defaults |
| `docs/CONVLOG.md` | The design record of the `convlog` suite, which is not released |

## License

Apache-2.0 (see `LICENSE`). Benchmarks keep their own licences and are not redistributed here. To
cite the software, use `CITATION.cff`.
