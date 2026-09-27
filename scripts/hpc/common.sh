#!/usr/bin/env bash
# Shared setup for every HPC job script. SOURCED, never executed.
#
# WHAT THIS FILE OWNS. Every hpc/*.sh begins `source scripts/hpc/common.sh`, and after that
# line the environment is fully determined: no job script sets a path, a CUDA variable or a
# Hugging Face variable of its own. One place to read means one place to be wrong.
#
# `.env` IS NEVER SOURCED HERE, and no hpc script may source it. The cluster holds no
# team-gateway credentials and rungs 1-2 make no API call: the trainers read jsonl off GPFS and
# weights out of $HF_HOME. docs/GPU_RUNBOOK.md 2 sources `.env` for `pi env doctor` and the
# Tier-B sweeps -- those run on the Mac, behind the tunnel (docs/HPC_RUNBOOK.md 6), not here.
#
# `PI_GOLD_ROOT` IS UNSET UNCONDITIONALLY. Nothing in scripts/hpc runs a rollout today, so
# nothing here would trip `pi_eval.gold.gold_root()`. It is unset anyway, because the day a
# rollout does move to the cluster the firewall must already be standing and not need a diff:
# CONTRIBUTING.md, "a process split: rollout workers run with PI_GOLD_ROOT unset, so gold_root()
# raises -- that raise is the firewall working".

# Refuse to be run instead of sourced: `bash common.sh` would set variables in a shell that
# exits one line later and report success for having done nothing.
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
  echo "common.sh is sourced, not executed: 'source scripts/hpc/common.sh'" >&2
  exit 2
fi

set -euo pipefail

# The repo root, from THIS file's location -- not $PWD and not $HOME/ProactiveInquirer. The
# runbook's bsub lines cd into the repo first, but a job resubmitted by hand from ~ would
# otherwise write artifacts into the home directory and find no data.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export REPO
cd "$REPO"

# Weights on shared GPFS, not on a node's local disk: docs/HPC_RUNBOOK.md 1 ("home /u/user on
# a 195 TB shared GPFS, visible from compute nodes") and 4 ("Keep HF_HOME=~/hf so weights live
# on shared storage"). A per-node cache would re-download 16 GB on every new host.
# THE PROJECT FILESET. HPC created /proj/pinq (500 GB, group proj_pinq, GPFS ess03, 2.3 GB/s from a
# compute node, measured 2026-09-15 by job 776793) after the home fileset's 100 GB hard quota
# stalled the 4B download and killed a checkpoint write. Weights are read from the project copy
# when it exists (identical snapshots; the home copy is deleted once no running job reads it),
# and every NEW run's artifacts go under $PINQ_ARTIFACTS (the launcher passes
# /proj/pinq/user/artifacts; the default stays the repo's artifacts/ so a job submitted the old
# way lands where it always did).
PINQ_PROJ="${PINQ_PROJ:-/proj/pinq/user}"
if [ -z "${HF_HOME:-}" ] && [ -d "$PINQ_PROJ/hf/hub" ]; then HF_HOME="$PINQ_PROJ/hf"; fi
export HF_HOME="${HF_HOME:-$HOME/hf}"
ART="${PINQ_ARTIFACTS:-$REPO/artifacts}"
export ART

# OFFLINE IS THE DEFAULT, and it is the training default on purpose. A trainer that can reach
# huggingface.co can also silently fetch a revision that is not the one setup_env.sh pinned
# into the cache, and the run would be named after weights nobody chose. `hf_online` (below)
# lifts it for exactly the two steps that must download: the wheel install and the weight
# pull, both in setup_env.sh.
export HF_HUB_OFFLINE=1

export PYTHONUNBUFFERED=1
# Silences the fork warning and, more usefully, keeps the fast tokenizer from spawning a
# thread pool per dataloader worker on a 128-core node.
export TOKENIZERS_PARALLELISM=false

# The toolkit, from the cluster's own module tree. docs/HPC_RUNBOOK.md 1 lists
# /opt/share/cuda-12.{6,8.1,9}; 12.6 is the one these scripts pin, so a wheel that compiles a
# kernel at import time compiles it against a version that does not move under us.
CUDA_HOME="${CUDA_HOME:-/opt/share/cuda-12.6}"
export CUDA_HOME
if [ -d "$CUDA_HOME" ]; then
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
fi

# See the header. Unset, not set-to-empty: `gold_root()` tests presence.
unset PI_GOLD_ROOT || true

PY="$REPO/.venv/bin/python"
PI="$REPO/.venv/bin/pi"
# A venv is generated and ignored, so a PINNED worktree -- which carries neither untracked nor
# ignored files -- never has one, and `serve.sh` then dies at its own guard before vLLM starts
# (that is how LSF 895633 ended, 4 s in). Overridable for the same reason ART is, one line up.
VLLM_VENV="${PINQ_VLLM_VENV:-$REPO/.venv-vllm}"
UV="${UV:-$HOME/.local/bin/uv}"
export PY PI VLLM_VENV UV

log() { printf '%s  %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*"; }

die() { log "FATAL: $*"; exit 1; }

# Lift the offline pin for a download step. Used twice, both in setup_env.sh.
hf_online() { unset HF_HUB_OFFLINE; log "HF_HUB_OFFLINE lifted (download step)"; }
hf_offline() { export HF_HUB_OFFLINE=1; log "HF_HUB_OFFLINE=1 (offline)"; }

# THE BANNER, AND THE DIRTY-TREE REFUSAL.
#
# Everything printed here is provenance for whatever the job writes afterwards: which host,
# which GPUs, which commit. The refusal is the part that matters. An adapter trained from an
# uncommitted tree is an adapter whose code version cannot be named, and CONTRIBUTING.md rule 1 is
# that a number without provenance is not a result -- the same reason `RunManifest.run_id`
# prefixes `dev-` on a dirty tree (docs/GPU_RUNBOOK.md 8) and the same reason a `dev-` rung is
# training-only. Untracked files are NOT dirty (`--untracked-files=no`): artifacts/, logs/ and
# the probe sample are untracked by construction, and a job that dirtied the tree by running
# would refuse its own resubmission.
hpc_banner() {
  log "job:       ${1:-hpc}"
  log "host:      $(hostname)"
  log "gpus:      $(nvidia-smi -L 2>/dev/null | tr '\n' ';' || echo 'nvidia-smi: not available')"
  log "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
  log "repo:      $REPO"
  log "git HEAD:  $(git -C "$REPO" rev-parse HEAD)"
  log "HF_HOME=$HF_HOME  HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-<unset>}  CUDA_HOME=$CUDA_HOME"
  log "artifacts: $ART"
  log "PI_GOLD_ROOT=${PI_GOLD_ROOT:-<unset, as required>}"
  local dirty
  dirty="$(git -C "$REPO" status --porcelain --untracked-files=no)"
  if [ -n "$dirty" ]; then
    log "working tree is DIRTY -- refusing to start:"
    printf '%s\n' "$dirty" >&2
    die "commit (or stash) these paths first; an artifact from an uncommitted tree cannot name its code version (CONTRIBUTING.md rule 1)"
  fi
  log "working tree clean at $(git -C "$REPO" rev-parse --short HEAD)"
}

git_head() { git -C "$REPO" rev-parse HEAD; }

gpu_name() { nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo unknown; }

# `.venv` is built by setup_env.sh and by nothing else; every other script needs it to exist.
require_venv() {
  [ -x "$PY" ] || die "$PY missing -- run scripts/hpc/setup_env.sh first"
  [ -x "$PI" ] || die "$PI missing -- run scripts/hpc/setup_env.sh first"
}

require_file() { [ -f "$1" ] || die "missing $1 ${2:+($2)}"; }
