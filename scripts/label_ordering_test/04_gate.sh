#!/bin/bash
# Step 4a of 4: the nine cells and the three pooled cells. Run from the repo root.
#
# `rc=1` from `pi train gate` is a verdict that did not PASS, not a failure to compute -- these
# three all fail on `distinct3`, a level against a fixed floor that carries no interval and is
# not the endpoint here. Capture the code directly, never after a pipe.
set -u
: "${PI_WORK:?set PI_WORK to a private scratch directory}"
REPO=$(pwd)
mkdir -p "$PI_WORK/verdicts"
for arm in qwen3-8b-dpo-headline-control qwen3-8b-dpo-headline-rater qwen3-8b-dpo-headline-reaches; do
  .venv/bin/pi train gate --parquet-dir "$PI_WORK/store_$arm" \
    --grid-name tier1_trained_qa_base --baseline-grid-name tier1_trained_qa_base \
    --grids-root "$REPO/conf/grids" \
    --checkpoint-arm inquirer_trained --baseline-arm inquirer_prompted \
    --baseline-model-id qwen3-8b-base \
    --coverage-rule matched_cost --bootstrap-seed 0 --n-resamples 1000 \
    --out "$PI_WORK/verdicts/$arm.matched_cost"
  echo "$arm rc=$?"
done
