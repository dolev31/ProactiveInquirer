#!/usr/bin/env bash
# Lane S2, SECONDARY row: score the 32B DPO FINAL checkpoint with the published paired runner.
#
# Submitted chained on `ended(902609) && ended(905198)` -- the trainer AND the preserve job -- because
# preserve copies the final adapter to step-final only AFTER the trainer is gone. Chaining on the
# trainer alone could run before that copy exists.
#
# REFUSES (exit 3) rather than scoring an intermediate when no final adapter exists: an intermediate
# scored under a "final" label is exactly the kind of number this lane was set up to prevent.
#
# RECORDS whether the run was ever RESUMED. On 2026-09-22 the rung-2 DPO trainer (resume=True by
# default) was shown to reload the INIT weights on resume while keeping the step counter, so a resumed
# run's "final" is its init plus the remaining steps. A resumed 32B final would read a ~zero DPO gain for
# the wrong reason, so this job states it loudly instead of letting the number stand alone.
set -euo pipefail
PRES=/proj/pinq/user/artifacts/preserved/rung2-32b-headline-notdone-both
TRAIN_OUT=/proj/pinq/user/artifacts/rung2-32b-headline-notdone-both
LANE=/proj/pinq/user/lane_s2
RUNNER="$LANE/lmpaired_run.sh"
DEST="$LANE/eval_paired"
LABEL=qwen3-32b-dpo-notdone-both-final
mkdir -p "$DEST"
log() { echo "$(date -u +%FT%TZ)  $*"; }

if [ -f "$PRES/step-final/adapter_model.safetensors" ]; then CK="$PRES/step-final"
elif [ -f "$TRAIN_OUT/adapter_model.safetensors" ]; then CK="$TRAIN_OUT"
else log "REFUSING: no final adapter at $PRES/step-final or $TRAIN_OUT"; exit 3; fi

log "final checkpoint: $CK"
log "adapter sha256: $(sha256sum "$CK/adapter_model.safetensors" | cut -d' ' -f1)"
log "runner sha256:  $(sha256sum "$RUNNER" | cut -d' ' -f1)"
if [ -f "$TRAIN_OUT/rung2.manifest.json" ]; then
  RES=$(python3 -c "import json;print(json.load(open('$TRAIN_OUT/rung2.manifest.json')).get('resumed_from'))")
  if [ "$RES" != "None" ]; then
    log "WARNING: THIS RUN WAS RESUMED from $RES -- its final may be its INIT plus the remaining steps."
    log "WARNING: check the weight distance to the checkpoint it resumed from before reading any gain."
  else
    log "resumed_from: None (a fresh run, not exposed to the resume defect)"
  fi
else
  log "WARNING: no rung2.manifest.json under $TRAIN_OUT, so resume status is UNKNOWN (not assumed fresh)"
fi
log "global_step: $(python3 -c "import json;print(json.load(open('$CK/trainer_state.json')).get('global_step'))" 2>/dev/null || echo UNKNOWN)"
bash "$RUNNER" "$CK" "$DEST/$LABEL.paired.json" "$LABEL"
log "scored -> $DEST/$LABEL.paired.json"
