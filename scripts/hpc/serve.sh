#!/usr/bin/env bash
# vLLM on a compute node, one base + every adapter the CALLER names, in the FOREGROUND so LSF
# holds the job for its whole wall limit. docs/HPC_RUNBOOK.md 6 for the topology,
# docs/GPU_RUNBOOK.md 7 for the command.
#
# THE MAP IS AN INPUT, NOT A COMMITTED ARRAY (changed 2026-09-15). This script used to carry
#   DIRS=(rung1-8b rung2-8b-control ...)   NAMES=(qwen3-8b-sft qwen3-8b-dpo-control ...)
# so serving a newly finished checkpoint meant editing a tracked file -- and on the cluster a
# tracked edit is an uncommitted tree, which `hpc_banner` refuses to start under. It also
# SKIPPED a directory that held no adapter and served the base alone, so a sweep pinned to the
# missing name 404ed mid-run after the queue wait was already spent. Both are gone: the map
# comes from the environment, and a path that is not an adapter is a refusal.
#
#   PINQ_LORA="name=path[,name=path...]"   paths absolute, or relative to $ART
#   PINQ_LORA_FILE=<file>                  the same, one pair per line, '#' comments allowed
#   PINQ_SERVED_BASE   default qwen3-8b-base   what the base is published as
#   PINQ_BASE_MODEL    default Qwen/Qwen3-8B   the weights the adapters are applied to
#   PINQ_MAX_MODEL_LEN default 16384
#   PI_VLLM_PORT       default 8000
#   PINQ_TENSOR_PARALLEL  UNSET -- set to N to shard the base across N cards
#
# ONE INSTANCE PER BASE IS NOT ONE CARD PER INSTANCE (added 2026-09-18). Until today this
# script could only express `-gpu "num=1"`: nothing under scripts/hpc/ spelled
# `--tensor-parallel-size`, and `--max-model-len` was a single number measured FOR THE 8B.
# Qwen3-32B does not fit that shape. Measured from the cluster's own HF snapshot
# (/proj/pinq/user/hf/hub/models--Qwen--Qwen3-32B): 61.02 GiB of bf16 weights against the 8B's
# 16.31, and 256 KiB of KV per token (2 . 64 layers . 8 kv-heads . 128 head-dim . 2 bytes,
# config.json) against the 8B's 144 KiB -- so at --max-model-len 16384 ONE full-length sequence
# costs 4.00 GiB of KV, out of the ~11 GiB a single 80 GiB card has left after the weights.
# Across TWO cards the weights are 30.51 GiB each and the KV is 128 KiB/token/card, which
# restores the headroom 16384 was chosen under. artifacts/rung32b_serve_20260918/RESULT.md has
# the full arithmetic and its provenance.
#
#   export PINQ_TENSOR_PARALLEL=2
#   bsub ... -gpu "num=2:mode=exclusive_process" -R "span[hosts=1]" ...
#
# THE TWO NUMBERS MUST AGREE, and nothing can check that from here: `num=` is LSF's and this is
# vLLM's. Too few cards is an OOM after the queue wait; too many is a hang in NCCL. `span
# [hosts=1]` is not optional -- LSF counts `num=` PER HOST, and tensor parallelism inside one
# vLLM process cannot cross a node.
#
# UNSET IS NOT A VALUE. With PINQ_TENSOR_PARALLEL unset this script emits no `--tensor-parallel`
# argument at all and the argv is byte-identical to the one every ARGS.<job>.txt on disk already
# records -- the same rule `SHA_OMIT_WHEN_DEFAULT` applies to a config sha. An unconditional
# `--tensor-parallel "${PINQ_TENSOR_PARALLEL:-}"` would pass an EMPTY STRING on every 8B serve,
# which is why the append below is guarded; tests/test_serve_tensor_parallel.py pins both halves.
#
# POINT THE MAP AT FROZEN WEIGHTS, NOT AT `best/<criterion>/`. `scripts/hpc/keep_best.py`
# REWRITES `<run>/best/by_nll/` whenever a later checkpoint wins the dev criterion -- rung1-8b's
# by_stop2x2 moved from 1200 to 3800 between 2026-09-15 00:23 and 08:05. A served name whose
# weights change underneath it is exactly what `conf/checkpoints.json` exists to prevent, so a
# REGISTERED name must be served from a copy that nobody rewrites: `~/pinq-serve/adapters/<name>/`
# is that convention (qwen3-8b-sft is served from qwen3-8b-sft-step2600 for this reason). A
# `best/` path is fine for an unregistered exploratory serve -- and either way SERVED.<jobid>.json
# records the sha256 that was actually loaded, so "which weights answered" is answerable after
# the fact rather than assumed.
#
# ONE INSTANCE PER BASE. An adapter can only be applied to the base it was trained on, so the
# 4B and 1.7B checkpoints need their own job at their own port (8001, 8002 by convention) with
# PINQ_BASE_MODEL/PINQ_SERVED_BASE/PI_VLLM_PORT set. Mixing them in one instance is not a
# configuration this script can express, and vLLM would reject the load anyway.
#
# SUBMIT (docs/HPC_RUNBOOK.md 6, plus the map). EXPORT the map rather than passing `-env`:
# LSF copies the submitting shell's environment into the job, and `-env` splits ITS OWN value
# on commas, so a two-adapter map passed that way arrives truncated at the first comma.
#   export PINQ_LORA="qwen3-8b-sft=rung1-8b/best/by_nll,qwen3-8b-sft-headline=rung1-8b-headline"
#   bsub -U <reservation> -q normal -n 8 -M 64000 -W 10000 -gpu "num=1:mode=exclusive_process" \
#     -J pinq-serve -P pinq -o ~/logs/serve.%J.out bash scripts/hpc/serve.sh
# then, on the Mac, with H read out of $ART/serve/HOST:
#   ssh -N -L 8000:H:8000 user@login-node.example.com
#
# THE vLLM FLAGS. --served-model-name, --enable-lora, --max-lora-rank, --max-loras,
# --lora-modules, --max-model-len and --port were transcribed from docs/GPU_RUNBOOK.md 7 and
# plan I.14 and were NOT verified locally (no vLLM on the Mac); they have since been run --
# vLLM 0.29.0, LSF job 775651 on gpu-n629 -- with `--lora-modules name=path` and
# --max-model-len 16384 accepted. The spelling of --lora-modules has changed between vLLM
# releases; artifacts/setup/versions.txt records which version setup_env.sh installed.
#
# WHY --max-model-len IS 16384 AND NOT THE PLAN'S 8192. Measured on the 80 GB card: the 8B at
# bf16 is 16.31 GiB of weights and leaves 54.46 GiB of KV cache, which holds 16k tokens at the
# concurrency a dev sweep asks for. 8192 was a transcription that had never been run.
#
# WHY THE LiteLLM PROXY IS NOT HERE. docs/HPC_RUNBOOK.md 6: the proxy stays on the Mac because
# it holds the team-gateway credentials in `.env`, and the frozen Drafter/Answerer/user-sim
# roles reach the gateway through it. The cluster serves local checkpoints and nothing else,
# so no hpc script ever needs a credential (scripts/hpc/common.sh never sources `.env`).
#
# BEFORE ANY PAID SWEEP, from the Mac once the tunnel is up (docs/HPC_RUNBOOK.md 6):
#   PI_VLLM_URL=http://127.0.0.1:8000 pytest tests/test_chat_template_identity.py -k serving
# A /tokenize mismatch means every dev NLL was measured on a prompt the policy never sees.
set -euo pipefail
# shellcheck source=scripts/hpc/common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
hpc_banner "pinq-serve"

BASE_MODEL="${PINQ_BASE_MODEL:-Qwen/Qwen3-8B}"
SERVED_BASE="${PINQ_SERVED_BASE:-qwen3-8b-base}"
MAX_MODEL_LEN="${PINQ_MAX_MODEL_LEN:-16384}"
# No `:-1`. The absent state has no spelling on the command line -- see the header.
TENSOR_PARALLEL="${PINQ_TENSOR_PARALLEL:-}"
PORT="${PI_VLLM_PORT:-8000}"
JOBID="${LSB_JOBID:-nojob}"
SERVE_DIR="$ART/serve"
HOSTFILE="$SERVE_DIR/HOST"
SERVED_JSON="$SERVE_DIR/SERVED.$JOBID.json"
ARGS_FILE="$SERVE_DIR/ARGS.$JOBID.txt"
MAP_PY="$(dirname "${BASH_SOURCE[0]}")/serve_map.py"

[ -x "$VLLM_VENV/bin/python" ] || die "$VLLM_VENV missing -- run scripts/hpc/setup_env.sh first"
require_file "$MAP_PY"

# "Activate" as asked, then call the venv's own binary. The activation is what makes
# `python`/`vllm` on PATH belong to .venv-vllm for anything the server shells out to; the
# absolute path is what makes THIS line independent of whether it worked. `set +u` around the
# source because an activate script is not written to be read under `set -u`.
set +u
# shellcheck disable=SC1091  # generated by `uv venv`; not in the repo
source "$VLLM_VENV/bin/activate"
set -u
VLLM_BIN="$VLLM_VENV/bin/vllm"
[ -x "$VLLM_BIN" ] || die "$VLLM_BIN not executable -- did the vllm install finish? see artifacts/setup/versions.txt"
log "vllm: $VLLM_BIN"

# ---------------------------------------------------------------- the LoRA map
# THE SERVED NAMES ARE NOT FREE. scripts/price_tables/2026-09.json must carry a $0 row for
# every name here, and conf/checkpoints.json should carry its provenance
# (scripts/hpc/register_checkpoint.py writes both), or `PriceTable.rates` raises
# LLMConfigError and the sweep refuses to start -- docs/GPU_RUNBOOK.md 7. Nothing on this box
# can check that: the price table lives on the Mac, with the sweep.
mkdir -p "$SERVE_DIR"
SERVE_PY="$VLLM_VENV/bin/python"   # stdlib only; any python would do
MAPARGS=(
  --art "$ART"
  --base-model "$BASE_MODEL"
  --served-base "$SERVED_BASE"
  --port "$PORT"
  --max-model-len "$MAX_MODEL_LEN"
)
# APPENDED ONLY WHEN SET, so an 8B serve builds the same MAPARGS it always built. An empty
# string here would reach serve_map.py as `--tensor-parallel ''`, and an empty argument is one
# vLLM refuses after the GPU is already held.
if [ -n "${TENSOR_PARALLEL:-}" ]; then
  MAPARGS+=(--tensor-parallel "$TENSOR_PARALLEL")
  log "PINQ_TENSOR_PARALLEL=$TENSOR_PARALLEL -- this must equal the bsub -gpu \"num=N\""
  log "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
fi
log "PINQ_LORA=${PINQ_LORA:-<unset>}  PINQ_LORA_FILE=${PINQ_LORA_FILE:-<unset>}"
if [ -z "${PINQ_LORA:-}" ] && [ -z "${PINQ_LORA_FILE:-}" ]; then
  log "WARNING: no adapter map -- serving $SERVED_BASE alone."
  log "         --enable-lora is kept so adapters can be added later without a restart;"
  log "         every PI_MODEL_INQUIRER except $SERVED_BASE will 404 until then."
fi

# The map is validated FIRST, before anything is announced: a bad path must cost a second,
# not a GPU-hour. `set -e` plus the redirect means a refusal here ends the job with the
# python message in the log, and the argv itself stays on disk as provenance.
"$SERVE_PY" "$MAP_PY" args "${MAPARGS[@]}" > "$ARGS_FILE"
VLLM_ARGS=()
while IFS= read -r line; do VLLM_ARGS+=("$line"); done < "$ARGS_FILE"
[ "${#VLLM_ARGS[@]}" -gt 0 ] || die "serve_map printed no arguments -- refusing to exec vllm bare"
log "vllm serve argv ($ARGS_FILE):"
printf '  %s\n' "${VLLM_ARGS[@]}"

# ---------------------------------------------------------------- announce the host
# The Mac cannot see LSF's scheduling decision, and docs/HPC_RUNBOOK.md 6's tunnel needs the
# host by name. Written BEFORE the server binds so the tunnel can be prepared while vLLM loads
# 16 GB of weights; the port is only reachable once the server logs that it is up.
# `--lora-modules` is the last flag serve_map emits, so every argument after it is a
# name=path pair -- no guessing from the shape of the string.
SERVED_LIST="$SERVED_BASE"
in_loras=0
for a in "${VLLM_ARGS[@]}"; do
  if [ "$a" = "--lora-modules" ]; then in_loras=1; continue; fi
  if [ "$in_loras" = 1 ]; then SERVED_LIST="$SERVED_LIST ${a%%=*}"; fi
done
{
  echo "$(hostname) $PORT"
  echo "# written by scripts/hpc/serve.sh at $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  echo "# job=$JOBID  git_head=$(git_head)"
  echo "# served: $SERVED_LIST"
  echo "# record: $SERVED_JSON"
  echo "# Mac: ssh -N -L $PORT:$(hostname):$PORT user@login-node.example.com"
} > "$HOSTFILE"
log "wrote $HOSTFILE:"
cat "$HOSTFILE"

# ---------------------------------------------------------------- what WAS served
# conf/checkpoints.json says which weights SHOULD answer to a name. This says which ones did:
# one sha256 per adapter, plus the host and port the tunnel reached. It happens BEFORE the
# exec, because after the exec this shell no longer exists.
#
# IT COSTS A FULL READ OF EVERY ADAPTER (~350 MB each) AND THE ELAPSED TIME IS LOGGED, because
# that cost is not always small: reading ONE of these files on login-node took 10 minutes on
# 2026-09-15 (~0.6 MB/s; the home fileset under load). docs/HPC_RUNBOOK.md measures /proj at
# 2.3 GB/s from a compute node, which is the configuration this is meant to run in. If this
# step is ever slow enough to matter, the answer is to put the adapters on $PINQ_PROJ -- not to
# drop the hash, which is the only thing that can tell a sweep which weights answered it.
log "hashing the adapters into $SERVED_JSON"
SHA_T0=$SECONDS
"$SERVE_PY" "$MAP_PY" record "${MAPARGS[@]}" \
  --host "$(hostname)" --job "$JOBID" --git-head "$(git_head)" --out "$SERVED_JSON"
log "hashed in $((SECONDS - SHA_T0))s"
log "served record:"
cat "$SERVED_JSON"

# ---------------------------------------------------------------- serve, in the foreground
# exec, so the server IS the job: LSF's wall clock, `bkill` and the job's stdout all apply to
# the process that is actually serving, with no shell in between to swallow a signal.
log "starting vLLM on port $PORT (foreground; LSF holds the job until it exits)"
exec "$VLLM_BIN" serve "${VLLM_ARGS[@]}"
