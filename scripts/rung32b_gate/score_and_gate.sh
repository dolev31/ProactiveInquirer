#!/usr/bin/env bash
# Score the PRIVATE parquet and gate both suites. Never touches scores/parquet.
set -uo pipefail
MAIN=$HOME/PycharmProjects/ProactiveInquirer
WT=$HOME/PycharmProjects/pinq-wt-rung32b-20260919
A=$MAIN/artifacts/rung32b_gate_20260919
FARM=$A/farm
PI="$MAIN/.venv/bin/pi"
set -a; . "$MAIN/.env"; set +a

echo "##### pi score (isolated store; --allow-no-judge: the gated criteria are gold-derived,"
echo "##### not judge-derived, which is how every other rung on this ladder was scored) #####"
"$PI" score --parquet "$FARM/scores_parquet" --runs-root "$FARM/runs" \
      --gold-root "$MAIN/data/gold" --corpora-root "$MAIN/data/corpora" \
      --allow-no-judge
echo "pi score exit=$?"

for SUITE in musique strategyqa; do
  for N in 10000; do
    echo "##### pi train gate  suite=$SUITE  n_resamples=$N #####"
    "$PI" train gate \
      --parquet-dir "$FARM/scores_parquet" \
      --grid-name "dev_select_${SUITE}" \
      --baseline-grid-name "dev_baseline_${SUITE}" \
      --checkpoint-arm inquirer_trained \
      --baseline-arm inquirer_prompted \
      --baseline-model-id qwen3-32b-base \
      --checkpoint-model-id qwen3-32b-sft-headline \
      --grids-root "$WT/conf/grids" \
      --n-resamples "$N" \
      --out "$A/qwen3-32b-sft-headline.${SUITE}.n${N}.json"
    echo "gate exit=$?  suite=$SUITE n=$N"
  done
done
