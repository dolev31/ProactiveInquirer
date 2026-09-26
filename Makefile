# The commands a stranger types. Everything a reviewer needs is one of these.
.DEFAULT_GOAL := help
PY := .venv/bin/python
RUFF := .venv/bin/ruff

.PHONY: help venv gate fmt test smoke prereg-draft prereg-verify paper clean-cache

help:  ## show this
	@grep -hE '^[a-z0-9-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-16s %s\n",$$1,$$2}'

venv:  ## create the env: dev + the extras `make gate` and `make smoke` actually import
	uv venv --python 3.12 && uv pip install -e ".[dev,run,analysis]"
	@echo "note: suite extras are separate and large. tau2: uv pip install -e '.[tau2]'"

gate:  ## everything CI enforces, in CI's order
	$(RUFF) check .
	$(RUFF) format --check .
	.venv/bin/lint-imports
	@# A suite that the runner can build and that nobody declared a role for, or one declared
	@# both a training source and eval-only, is a training-set leak with no other symptom.
	@# Cheap: no gold, no network, no keys.
	.venv/bin/pi suites audit
	$(PY) -m pytest -q -m "not integration and not slow"
	bash scripts/check_no_home_paths.sh

fmt:  ## autofix
	$(RUFF) check --fix . && $(RUFF) format .

test:  ## full suite including integration (needs TAU2_DATA_DIR and downloads)
	$(PY) -m pytest -q

smoke:  ## end-to-end on the synthetic suite: no network, no keys, no dollars
	@# BUILDS INTO ITS OWN ROOT. The corpora tree is content-addressed, but the GOLD tree is
	@# not -- it is one file per (suite, graph_version) and the last build wins. So a 5-task
	@# smoke build used to overwrite data/gold/graphs/synth/v1.jsonl while a 12-task corpus
	@# sat beside it, and `pi score` then refused every synth run with a corpus mismatch. A
	@# smoke test must not be able to corrupt the tree it is smoke-testing.
	@rm -rf .smoke && mkdir -p .smoke
	$(PY) -c "import sys;sys.path.insert(0,'src');\
from pathlib import Path;from pi_eval.build.synth_build import build;\
print(build(n_tasks=5,n_facets=3,depth=3,root=Path('.smoke'))[0].parent)" > .smoke/corpus
	.venv/bin/pi run --suite synth --arm fake_chain --arm fake_drafter_only --arm fake_depth1 \
	    --n 5 --corpus "$$(cat .smoke/corpus)" --runs-root .smoke/runs --cache-root .smoke/cache
	.venv/bin/pi compact --runs-root .smoke/runs --out .smoke/parquet
	@# PI_JUDGE_CLIENT=...judge_replay is MEANT to exercise the judging path -- factory,
	@# harness, parser, schema preflight -- with a replay-only client that RAISES on a cache
	@# miss instead of dispatching.
	@#
	@# IT DOES NOT DO SO TODAY, and the claim above was wrong. `pi score` prints
	@#     judging: "refused: PI_JUDGE_CLIENT set but PI_MODEL_JUDGE is not"
	@# and the recipe still exits 0, so the smoke LOOKS like it covers judging and does not.
	@# Setting PI_MODEL_JUDGE would clear that guard and then raise on the first cache miss,
	@# because judge_replay is cache-only by design. Real coverage needs a small seeded judge
	@# cache committed as a fixture, and seeding it needs one provider call -- blocked while
	@# the LiteLLM proxy is unreachable. Left honest rather than made to look green.
	@#
	@# Without it this recipe exits 1 and `pi verify firewall` below never runs: `pi score`
	@# refuses a silent judge skip because a judge-derived PRIMARY (kpr_incremental) would
	@# otherwise have no rows. I introduced that refusal in 75ee0f6 and it broke smoke here.
	PI_JUDGE_CLIENT=pi_run.judge_client:judge_replay \
	    .venv/bin/pi score --parquet .smoke/parquet --runs-root .smoke/runs \
	    --gold-root "$$PWD/.smoke/data/gold"
	.venv/bin/pi verify firewall --cache-root .smoke/cache
	@rm -rf .smoke

prereg-draft:  ## print the preregistration without sealing anything
	$(PY) scripts/seal_prereg.py draft

prereg-verify:  ## recompute every sealed digest
	$(PY) scripts/seal_prereg.py verify

paper:  ## regenerate every table and figure from parquet, no API keys
	.venv/bin/pi render --all

clean-cache:  ## drop the LLM cache (keeps runs/ and scores/)
	rm -rf cache && echo "cache dropped; reruns will hit the provider"
