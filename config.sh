#!/usr/bin/env bash
# shellcheck shell=bash disable=SC2034  # a config file: several vars are consumed by the scripts that source it
# Shared settings for start.sh, stop.sh, status.sh and scripts/*.sh. A setting's value comes from
# the first of these that sets it:
#   1. the environment: `PORT=9000 ./scripts/start.sh`
#   2. scripts/local.sh (this setup's own values; sourced as bash), then ./.env (KEY=value lines,
#      read, never run): both are yours, not the repository's; where both set a key, local.sh wins
#   3. the defaults below
#
# This recipe serves GLM-5.3-Flash (NVIDIA's NVFP4 checkpoint) on the native TensorFold CUDA
# engine ("tensorfold-native") at tensor parallel over two GB10 hosts, after PR #804 (first
# shipped in the v1.0.x line). Everything runs in Docker; no Python engine and no patch stack.
_cfg_root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [[ -f "$_cfg_root/scripts/local.sh" ]]; then
  declare -A _cfg_env=()
  while IFS= read -r _n; do _cfg_env[$_n]=${!_n}; done < <(compgen -e)
  source "$_cfg_root/scripts/local.sh"
  for _n in "${!_cfg_env[@]}"; do [[ "${!_n-}" == "${_cfg_env[$_n]}" ]] || export "$_n=${_cfg_env[$_n]}"; done
  unset _cfg_env
fi
if [[ -f "$_cfg_root/.env" ]]; then
  while IFS= read -r _line || [[ -n "$_line" ]]; do
    [[ "$_line" =~ ^[[:space:]]*(export[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]] || continue
    _key=${BASH_REMATCH[2]}; _value=${BASH_REMATCH[3]}
    if [[ "$_value" =~ ^\"([^\"]*)\"[[:space:]]*(#.*)?$ || "$_value" =~ ^\'([^\']*)\'[[:space:]]*(#.*)?$ ]]; then
      _value=${BASH_REMATCH[1]}
    else
      _value=${_value%%#*}; _value=${_value%"${_value##*[![:space:]]}"}
    fi
    [[ -n "${!_key+set}" ]] || export "$_key=$_value"
  done < "$_cfg_root/.env"
fi
unset _n _line _key _value

# ---------------------------------------------------------------------------
# TensorFold version. Releases are frequent; "latest-v1" (the default) resolves to the newest
# v1.* release at prepare time and records the resolved tag and commit in the prepared marker,
# so an upstream release changes nothing until you re-run prepare. Pin instead with e.g.
# TF_REF=v1.0.12 (a tag) or TF_COMMIT=<sha> (an exact tree; wins over TF_REF).
# ---------------------------------------------------------------------------
TF_REF="${TF_REF:-latest-v1}"
TF_COMMIT="${TF_COMMIT:-}"
TF_REPO="${TF_REPO:-https://github.com/ashhart/TensorFold.git}"

# The GLM-5.3-Flash NVFP4 checkpoint (ModelOpt, read as stored; PR #804's family reads no other
# format). The revision below is the one already present in both Sparks' Hugging Face caches.
# Alternatives:
#   423acf37... - the revision PR #804's receipts were measured on (download it if you want the
#                 exact validated tree; see docs/VALIDATION.md)
#   da920bb0... - upstream main's revision (adds layer 45 to the ignore list; the engine reads
#                 tensors as stored, so later revisions are expected to work)
# The ranks cross-check every checkpoint byte against each other at start, so a copy that
# differs on one host stops both before serving.
MODEL_ID="${MODEL_ID:-nvidia/GLM-5.3-Flash-NVFP4}"
MODEL_REVISION="${MODEL_REVISION:-09b04e5e74bca08ca8549fc736d4cdd8624bfde3}"

# The two GB10s: this machine (the head) serves rank 0 and the API; WORKER (ssh target,
# key-based) runs rank 1. Rank 1 is started first, rank 0 serves HTTP (the native engine's
# rank-over-host pattern; docs/DELTAS.md). TP=3 is not servable on the native engine yet
# (it waits on upstream's multi-rank bootstrap fabric), so TP is fixed at 2.
TP=2
WORKER="${WORKER:-giles@brain03}"

# The ConnectX-7 rails between the Sparks (one /30 subnet per cable). NCCL's bootstrap listens
# on the head's rail address; NCCL discovers the rest from there. Both rails are cabled on this
# site, so the default split exchange works and TF_GLM_PREFILL_REDUCE=gather is NOT needed
# (that is for fabrics where some rank pairs cannot connect).
HEAD_FABRIC_ADDR="${HEAD_FABRIC_ADDR:-192.168.2.1}"      # brain02, enp1s0f1np1
WORKER_FABRIC_ADDR="${WORKER_FABRIC_ADDR:-192.168.2.2}"  # brain03, enp1s0f1np1
NCCL_PORT="${NCCL_PORT:-29741}"
# Comma-separated netdevs NCCL may use over the link; both rails of this site. NCCL_IB_HCA is
# left empty on purpose (all RoCE devices); set it in local.sh to pin devices.
NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-enp1s0f1np1,enP2p1s0f1np1}"
NCCL_IB_HCA="${NCCL_IB_HCA:-}"
# Extra NCCL debug, for a first bring-up: set NCCL_DEBUG=INFO in local.sh and watch the logs.

# Docker. The builder image compiles tensorfold-native (Zig + nvcc, sm_121); the runtime image
# carries the binary, its libraries and NCCL and is what both ranks run. The builder needs a
# CUDA devel base with nvcc for aarch64: nvcr.io/nvidia/pytorch:26.07-py3 is already pulled on
# the head and ships both, but any arm64 CUDA 13.x devel image works (set BASE_IMAGE).
ZIG_VERSION="${ZIG_VERSION:-0.17.0}"                     # TensorFold's .zig-version at v1.0.x
SM_ARCH="${SM_ARCH:-121}"                                # GB10
BASE_IMAGE="${BASE_IMAGE:-nvcr.io/nvidia/pytorch:26.07-py3}"
IMAGE_NAME="${IMAGE_NAME:-glm53-native-tf}"   # the runtime image; prepare.sh tags it <name>:<resolved version>
CONTAINER_NAME="${CONTAINER_NAME:-glm53-native-tf}"      # the same name on both Sparks

# Checkpoint and binary locations (host paths; the container mounts the HF cache read-write so
# the snapshot layout, blob names and hashes stay exactly as Hugging Face wrote them).
HF_CACHE="${HF_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}}"
MODEL_SNAPSHOT="$HF_CACHE/hub/models--${MODEL_ID//\//--}/snapshots/$MODEL_REVISION"

# Serving. The context the PR's receipts ran at; larger windows fit but are unmeasured on this
# pair (the engine's bf16 KV takes more per token than the old stack's fp8 latent cache) —
# docs/VALIDATION.md has the fit test. PARALLEL is the stream count (the lanes core; the PR
# served --parallel 2 and 4, and admitted 8).
SERVED_NAME="${SERVED_NAME:-GLM-5.3-Flash-NVFP4}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8888}"
CONTEXT="${CONTEXT:-40960}"
PARALLEL="${PARALLEL:-4}"
DRAFTS="${DRAFTS:-1}"            # 0 passes --no-drafts (the serial reference; MTP is the only drafter here)
API_KEY="${API_KEY:-}"           # recommended; also required by upstream for --slide-style features
MEMORY_LIMIT_GB="${MEMORY_LIMIT_GB:-}"   # -> TENSORFOLD_CUDA_MEMORY_LIMIT_GB, empty: engine default

# Runtime state: pidfiles, logs (each rank's docker log saved gzipped at stop, LOG_KEEP kept),
# and the prepared marker that start.sh compares to decide whether prepare.sh must run again.
STATE_DIR="${STATE_DIR:-$HOME/.local/state/glm53-native}"
LOG_DIR="${LOG_DIR:-$HOME/.cache/glm53-native/logs}"
LOG_KEEP="${LOG_KEEP:-10}"

# Colours only on a terminal.
_c() { [[ -t "$1" ]] && printf '\033[%sm' "$2" || true; }
log()  { printf '%s[%s]%s %s\n' "$(_c 1 '1;36')" "$(basename "$0")" "$(_c 1 0)" "$*"; }
warn() { printf '%s[%s] WARN:%s %s\n' "$(_c 2 '1;33')" "$(basename "$0")" "$(_c 2 0)" "$*" >&2; }
die()  { printf '%s[%s] ERROR:%s %s\n' "$(_c 2 '1;31')" "$(basename "$0")" "$(_c 2 0)" "$*" >&2; exit 1; }

worker() { ssh -o BatchMode=yes "$WORKER"; }

# The docker run contract, shared by both ranks (the same as the EXL3 recipe's: host networking
# for the CX7 rails, the RDMA devices, locked memory for NCCL's registered buffers).
RUN_ARGS=(--gpus all --ipc=host --network host --shm-size 16g --device /dev/infiniband
          --cap-add IPC_LOCK --ulimit memlock=-1 --ulimit stack=67108864)

# Environment passed into both ranks: offline serving from the mounted cache, plus every
# TENSORFOLD_/TF_GLM_/NCCL_ variable this recipe sets.
env_args() {
  ENV_ARGS=(-e HF_HUB_OFFLINE=1 -e NCCL_SOCKET_IFNAME="$NCCL_SOCKET_IFNAME")
  [[ -z "$NCCL_IB_HCA" ]] || ENV_ARGS+=(-e NCCL_IB_HCA="$NCCL_IB_HCA")
  [[ -z "$MEMORY_LIMIT_GB" ]] || ENV_ARGS+=(-e TENSORFOLD_CUDA_MEMORY_LIMIT_GB="$MEMORY_LIMIT_GB")
  while IFS='=' read -r name _; do ENV_ARGS+=(-e "$name=${!name}"); done \
    < <(env | grep -E '^(TENSORFOLD|TF_GLM)_[A-Z0-9_]+=' || true)
}

# The serve arguments both ranks take (rank 0 adds --name/--host/--port in start.sh). The ranks
# must agree on window and streams: each allocates its own state before rank 0 admits.
serve_args() {
  local args=(serve "/root/.cache/huggingface/hub/models--${MODEL_ID//\//--}/snapshots/$MODEL_REVISION"
              --context "$CONTEXT" --parallel "$PARALLEL")
  [[ "$DRAFTS" == 1 ]] || args+=(--no-drafts)
  echo "${args[@]}"
}

# The prepared marker: a one-line description of everything a start depends on. start.sh re-runs
# prepare.sh whenever the config fingerprint no longer matches (a new upstream release, another
# revision, another worker, a different window or stream count).
config_line() {   # the fingerprint: everything except the resolved version
  echo "model=$MODEL_ID@$MODEL_REVISION worker=$WORKER context=$CONTEXT parallel=$PARALLEL head=$HEAD_FABRIC_ADDR"
}
prepared_line() { # the fingerprint plus what prepare resolved (written to $STATE_DIR/prepared)
  echo "tf=$TF_RESOLVED_REF@$TF_RESOLVED_COMMIT image=$IMAGE_REF $(config_line)"
}
