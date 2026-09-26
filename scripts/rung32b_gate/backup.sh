#!/usr/bin/env bash
# git protects source; a backup protects artefacts. artifacts/ is untracked by construction
# (artifacts/gate/README.md: "nothing here is committed"), so this directory's only durable
# copy is the backup. The farm is symlinks into the shared runs/ -- copied as LINKS, not
# dereferenced, because dereferencing would duplicate 1,395 run dirs; the run ids themselves
# are in run_ids.txt, which is what makes the population recoverable.
set -euo pipefail
A=$HOME/PycharmProjects/ProactiveInquirer/artifacts/rung32b_gate_20260919
B=$HOME/pi-corpus-backup/rung32b-gate-20260919
mkdir -p "$B"
rsync -a --no-links --exclude 'farm/runs/' "$A"/ "$B"/
# the parquet store IS real data and must come across
[ -d "$A/farm/scores_parquet" ] && rsync -a "$A/farm/scores_parquet"/ "$B/farm/scores_parquet"/
echo "=== backup contents ==="; find "$B" -maxdepth 2 -type f | sed "s|$B/||" | sort
echo "=== sizes ==="; du -sh "$B"
