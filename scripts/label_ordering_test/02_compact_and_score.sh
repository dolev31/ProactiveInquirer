#!/bin/bash
# Step 2 of 4. Run from the repo root. No gateway spend: compacting and scoring are local, and
# judging is skipped deliberately (PI_JUDGE_CLIENT unset -> judge_pins []), which is the same
# judge state the development reading of these arms used.
#
# PI_GOLD_ROOT is EMPTY in .env on purpose -- a rollout worker that can read gold is the firewall
# failing -- so the scorer's gold root is passed explicitly here instead.
set -u
: "${PI_WORK:?set PI_WORK to a private scratch directory}"
REPO=$(pwd)
set -a; . "$REPO/.env" || { echo "DIE: .env did not source"; exit 3; }; set +a
unset PI_JUDGE_CLIENT
.venv/bin/pi compact --root "$REPO" --runs-root "$PI_WORK/runs_iso" --out "$PI_WORK/parquet"
echo "compact_rc=$?"
.venv/bin/pi score --root "$REPO" --parquet "$PI_WORK/parquet" \
  --runs-root "$PI_WORK/runs_iso" --corpora-root "$REPO/data/corpora" \
  --gold-root "$REPO/data/gold" --graph-version v1 --allow-no-judge
echo "score_rc=$?"
