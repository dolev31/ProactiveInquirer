#!/usr/bin/env bash
# Lane L6.1: rung 1 on a named SFT corpus, for BOTH arms of the label contrast.
#
# ONE SCRIPT, TWO ARMS, AND THAT IS THE POINT. The arms differ in the STOP label and in nothing
# else, so they must not differ in the file that trains them either: two near-identical scripts
# is how an arm quietly acquires a hyperparameter the other does not have. The dataset and the
# label it MUST carry come from the environment; everything else is fixed here.
#
#   PI_L61_ARM=answernode   -> qwen3-8b-sft-headline-answernode, sft.answer_node.jsonl,
#                              manifest must say sft_stop_label = answer_node_covered_before
#   PI_L61_ARM=control      -> qwen3-8b-sft-headline-refresh,    sft.jsonl,
#                              manifest must say sft_stop_label = done_before
#
# WHY A CONTROL ARM EXISTS. The published `qwen3-8b-sft-headline` was fitted on the 2026-09-14
# corpus (14,302 rows). This lane's 2026-09-19 re-export of the SAME RULE over the SAME cohort
# yields 14,150 -- 152 ASK rows short, 1.06%, because the walk crossed a shared live `runs/` and
# recorded 1,454 RuntimeError, 46 state_mismatch and 733 no_gold run skips. `train_id_set_hash`
# and `n_stop` (5,541) reproduce exactly, so the task population is identical and it is the ASK
# rows that are short. Against the published headline the variant therefore differs in the label
# AND in 1.06% of its ASK rows; against `-refresh` it differs in the label alone.
#
# SUBMIT (docs/HPC_RUNBOOK.md 3; -U <reservation> is the advance reservation, not the queue).
# EXPORT the variables -- LSF copies the submitting shell's environment, and `-env` splits its
# own value on commas, so a JSON reference ask passed that way arrives truncated:
#   export PI_L61_ARM=answernode
#   export PI_REFERENCE_ASK='{"action": "ASK", "question": "...", "rationale": ""}'
#   bsub -U <reservation> -q normal -n 8 -M 96000 -W 1800 -gpu "num=1:mode=exclusive_process" \
#     -J pinq-rung1-l61-answernode -P pinq -o ~/logs/rung1-l61-answernode.%J.out \
#     bash /proj/pinq/user/ablation/train_rung1_answernode.sh
#
# THE HYPERPARAMETERS ARE NOT RESTATED HERE. `conf/train/rung1_8b.json` is the grid the
# imitation headline trained under (`artifacts/rung1-8b-headline/rung1.manifest.json`: lr 1e-4,
# 2 epochs, LoRA r32/alpha64/dropout 0.05, max_seq_len 5120, per_device_batch 1, grad_accum 16,
# bf16, seed 0, curriculum unset -> shuffled). Naming them a second time in this file is how the
# two arms would come to differ in something nobody meant to change.
set -euo pipefail
source "$HOME/ProactiveInquirer/scripts/hpc/common.sh"

ARM="${PI_L61_ARM:?PI_L61_ARM must be 'answernode' or 'control' -- there is no default, because
a default here would silently train one arm twice under two names}"
DATA_DIR="$REPO/data/rl/answer_node_20260919"
case "$ARM" in
  answernode)
    RUN=rung1-8b-headline-answernode
    DATASET="$DATA_DIR/sft.answer_node.jsonl"
    EXPECT_LABEL=answer_node_covered_before
    ;;
  control)
    RUN=rung1-8b-headline-refresh
    DATASET="$DATA_DIR/sft.jsonl"
    EXPECT_LABEL=done_before
    ;;
  *) echo "FATAL: PI_L61_ARM=$ARM is neither 'answernode' nor 'control'" >&2; exit 2 ;;
esac

hpc_banner "pinq-rung1-l61-$ARM"
require_venv

MANIFEST="${DATASET%.jsonl}.manifest.json"
CONFIG="$REPO/conf/train/rung1_8b.json"
BASE_MODEL=Qwen/Qwen3-8B
OUT="${PINQ_ARTIFACTS:-/proj/pinq/user/artifacts}/$RUN"
SNAP="$OUT-snapshots"
EVAL_OUT="${PINQ_ARTIFACTS:-/proj/pinq/user/artifacts}/eval/$RUN.tierA.json"
DEV_SFT="${PI_L61_DEV_SFT:-$REPO/data/rl/dev/sft.dev.jsonl}"
DEV_PAIRS="$REPO/data/rl/dev/pairs.dev.jsonl"
TAU=0.05
SIGMA_J=0.0

# THE OFFLINE STOP 2x2 NEEDS A REFERENCE ASK OR IT RETURNS HALF A TABLE. A STOP row carries no
# ask of its own, so without one every STOP row is SKIPPED and `p_stop_given_done` comes back
# `nan` -- which reads as a finding about the model rather than as an omission in the harness.
# MEASURED: all 60 Tier-A verdicts on this cluster carry `reference_ask: null`, so that cell has
# never once been populated. Refused here rather than defaulted: the string is a measurement
# choice (it sets what STOP is being preferred OVER), it must be IDENTICAL across the two arms
# for their contrast to mean anything, and a default invented in this file would be invisible.
: "${PI_REFERENCE_ASK:?PI_REFERENCE_ASK must be exported before bsub -- without it every STOP row
is skipped and p_stop_given_done is nan. It must be the SAME string for both arms.}"

[ -f "$DATASET" ]  || { echo "FATAL: no dataset at $DATASET" >&2; exit 1; }
[ -f "$MANIFEST" ] || { echo "FATAL: no manifest at $MANIFEST" >&2; exit 1; }
mkdir -p "$OUT" "$SNAP"

# THE MANIFEST IS CHECKED, NOT TRUSTED. `rung1.sh` checks `rank_rule` and `margin_threshold`
# for the same reason; this adds the one field that is the whole point of the run. A dataset
# whose manifest says `done_before` here would train the control under the variant's name, and
# every number downstream would be about the wrong arm while looking completely normal.
python3 - "$MANIFEST" "$TAU" "$EXPECT_LABEL" <<'PY' || exit 1
import json, sys
m = json.load(open(sys.argv[1]))
tau = float(sys.argv[2])
expect = sys.argv[3]
bad = []
if m.get("sft_stop_label") != expect:
    bad.append(f"sft_stop_label={m.get('sft_stop_label')!r}, expected {expect!r}")
if m.get("rank_rule") != "outcome":
    bad.append(f"rank_rule={m.get('rank_rule')!r}, expected 'outcome'")
if abs(float(m.get("margin_threshold", -1)) - tau) > 1e-12:
    bad.append(f"margin_threshold={m.get('margin_threshold')!r}, expected {tau}")
if m.get("split") != "train":
    bad.append(f"split={m.get('split')!r}, expected 'train'")
if bad:
    print("FATAL: dataset manifest does not describe this arm:", *bad, sep="\n  ", file=sys.stderr)
    raise SystemExit(1)
print(f"dataset OK: n_examples={m['n_examples']} label={m['sft_stop_label']} "
      f"train_id_set_hash={m['train_id_set_hash'][:16]} cohort={m.get('cohort')}")
PY
echo "dataset rows: $(wc -l < "$DATASET")"
echo "dataset sha : $(sha256sum "$DATASET" | cut -d' ' -f1)"

# START THE COPIER, THEN TRAP ON THE NEXT LINE. `save_total_limit` is hardcoded to 2, so
# without this only the last two checkpoints survive and "checkpoint selection" is a choice
# among two. Traps do not stack here and a late one has already cost this project two A100s
# for nine hours (see LAUNCHER_TRAPS.md).
/proj/pinq/user/ablation/ckpt_copier.sh "$OUT" "$SNAP" 120 30 &
COPIER=$!
trap 'rc=$?; kill '"$COPIER"' 2>/dev/null; wait '"$COPIER"' 2>/dev/null; echo "[trap] rc=$rc copier stopped $(date -u +%FT%TZ)"; exit $rc' EXIT INT TERM
echo "copier pid=$COPIER -> $SNAP"

# --train IS REQUIRED. Without it `pi train rung1` runs PREFLIGHT ONLY and exits 0, which is
# the worst shape a failure takes: indistinguishable from success in every signal a supervisor
# checks. The assertion after this is what actually decides whether training happened.
"$PI" train rung1 \
  --config "$CONFIG" \
  --base-model "$BASE_MODEL" \
  --tau "$TAU" --sigma-j "$SIGMA_J" \
  --dataset "$DATASET" \
  --out "$OUT" \
  --seed 0 \
  --train --acknowledge-untested || { echo "FATAL: rung1 failed" >&2; exit 1; }

[ -f "$OUT/adapter_model.safetensors" ] || { echo "FATAL: exited 0 but no adapter at $OUT -- the preflight-only path; --train was not honoured" >&2; exit 1; }
[ -f "$OUT/rung1.manifest.json" ]       || { echo "FATAL: exited 0 but no rung1.manifest.json at $OUT" >&2; exit 1; }
SNAPN=$(ls -1d "$SNAP"/checkpoint-* 2>/dev/null | wc -l)
echo "ASSERTED: adapter present, manifest present, $SNAPN checkpoint snapshots captured"
echo "adapter sha (the REGISTERED digest, weights alone, not adapter_sha.txt which covers config too):"
sha256sum "$OUT/adapter_model.safetensors"
ls -1 "$SNAP" || true

python3 - "$OUT/rung1.manifest.json" <<'PY' || true
import json, sys
m = json.load(open(sys.argv[1]))
d = m.get("dataset", {})
print("  n_examples            =", d.get("n_examples"))
print("  n_stop / stop_share   =", d.get("n_stop"), "/", d.get("stop_share"))
print("  within_task_distinct3 =", d.get("within_task_distinct_3_of_dataset"))
print("  config_sha            =", d.get("config_sha"))
PY

# ---------------------------------------------------------------- Tier A offline eval
# RUN HERE, not left to `rung1.sh`, because this script calls `pi train rung1` directly. It is
# the FIRST number this lane can have: a held-out stop 2x2 over decision rows, needing no serve
# slot, no proxy restart, no sweep and no spend.
#
# `--reference-ask` IS PASSED. Without it every STOP row is skipped and `p_stop_given_done` is
# `nan`; the refusal at the top of this file is what guarantees the string exists, and it is the
# same string for both arms so the contrast between them is a contrast and not two scales.
#
# WHICH QUESTION THIS 2x2 ANSWERS, STATED EXACTLY, because it is NOT the one the arm name
# suggests. `pinq_train.eval_offline.stop_confusion` reads `row.get("done_before")` and nothing
# else -- the condition field is hardcoded (checked, `eval_offline.py:168`). So re-exporting the
# dev rows under this arm's STOP label does NOT move what the table conditions on: it changes
# which rows are STOP targets, hence which ask strings get scored, and leaves both cells keyed on
# POOLED completeness. The answer-node-conditioned 2x2 needs `stop_confusion` to be able to
# condition on `answer_node_covered_before`, which is a code change, and it is run as a separate
# step after this job rather than pretending this one delivered it.
#
# What THIS table does deliver, and it is worth having: both arms scored against the identical
# dev file and the identical reference ask, so `p_stop_given_done` and `p_ask_given_not_done`
# are directly comparable between them. That is a measurement of "does the variant stop more",
# which is prediction 1's premise. `PI_L61_DEV_SFT` exists so the later, correctly-conditioned
# pass can point at the re-exported file without this script changing.
mkdir -p "$(dirname "$EVAL_OUT")"
echo "Tier A: dev_sft=$DEV_SFT"
echo "Tier A: reference ask = $PI_REFERENCE_ASK"
"$PI" train eval-offline \
  --checkpoint "$OUT" --dev-sft "$DEV_SFT" --dev-pairs "$DEV_PAIRS" \
  --chat-template --reference-ask "$PI_REFERENCE_ASK" --out "$EVAL_OUT" \
  || die "Tier-A eval-offline failed -- the adapter and manifest are already written under $OUT and are left in place"
echo "Tier A verdict:"
cat "$EVAL_OUT"

# A `nan` here means the reference ask did not reach the scorer after all. It is the failure this
# whole block exists to prevent, so it is checked rather than eyeballed in a log nobody re-reads.
python3 - "$EVAL_OUT" <<'PY' || exit 1
import json, math, sys
d = json.load(open(sys.argv[1]))
sc = d.get("stop_confusion", {})
n_skip = sc.get("n_skipped_no_reference", 0)
p = sc.get("p_stop_given_done")
if n_skip or p is None or (isinstance(p, float) and math.isnan(p)):
    print(f"FATAL: the stop 2x2 is half empty -- p_stop_given_done={p}, n_done={sc.get('n_done')}, "
          f"n_skipped_no_reference={n_skip}. The reference ask did not reach the scorer.",
          file=sys.stderr)
    raise SystemExit(1)
print(f"stop 2x2 LIVE: p_stop_given_done={p:.4f} (n={sc.get('n_done')}), "
      f"p_ask_given_not_done={sc.get('p_ask_given_not_done'):.4f} (n={sc.get('n_not_done')})")
PY
