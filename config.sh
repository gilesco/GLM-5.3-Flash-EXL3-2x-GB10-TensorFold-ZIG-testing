#!/usr/bin/env bash
# shellcheck shell=bash disable=SC2034  # a config file: several vars are consumed by the scripts that source it
# Shared settings for start.sh, stop.sh, status.sh and scripts/*.sh. A setting's value comes from
# the first of these that sets it:
#   1. the environment: `PORT=9000 ./scripts/start.sh`
#   2. scripts/local.sh (this setup's own values; sourced as bash), then ./.env (KEY=value lines,
#      read, never run): both are yours, not the repository's; where both set a key, local.sh wins
#   3. the defaults below
#
# This recipe serves GLM-5.3-Flash on the native TensorFold CUDA engine ("tensorfold-native")
# at tensor parallel over two GB10 hosts, from the EXL3 checkpoint line — the AZ-Focus
# variant by default (the (user) hardmode-bar call, 2026-10-27), with the DFlash2 drafter —
# the two-rank path upstream measured for 1.1.0 and onwards. This line is EXL3
# only: the NVFP4 route has its own repo (GLM-5.3-Flash-NVFP4-2x-GB10-TensorFold-v1-testing).
# Everything runs in Docker; no Python engine and no patch stack.
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

# The served default checkpoint: AxonZeta's AZ-Focus EXL3 (topk=8 by design — the
# (user) hardmode-bar default; a local checkpoint on both hosts, see docs/SITE.md, not in
# git). The stock/top4cfg cells serve Mia-AiLab's EXL3 one (4-bit mcg routed experts, BF16
# elsewhere) — production's checkpoint, already in this site's Hugging Face caches. This
# line is EXL3 only; the NVFP4 route has its own repo. The ranks cross-check every
# checkpoint byte against each other at start, so a copy that differs on one host stops
# both before serving.
MODEL_ID="${MODEL_ID:-local/GLM-5.3-Flash-AZ-Focus-EXL3}"
MODEL_REVISION="${MODEL_REVISION:-main}"   # the qualified optimum (2026-10-27, (user): hardmode is the quality bar — the Focus's 92 run-level vs production's 94; the top4cfg (89/76% subset) remains receipted as the perf leader)
                             # the site's top4cfg variant of this checkpoint — num_experts_per_tok
                             # 4, byte-surgical vs the base snapshot (one line differs). Receipts:
                             # perf campaign-20261024-135458 (solo 108% of production, c2 103%,
                             # c4 78%, slope 0.593) and the battery campaign-20261024-142659
                             # (§2 gates ALL PASS, tool-eval 93/100 above stock's 92; hardmode
                             # 89 vs production's 91). The base revision stays in the caches;
                             # serve stock with MODEL_REVISION=078455ffe6472f9a52fbc1139f58b9db2881b25c.
                             # The original base revision comment: present in this
                             # site's HF caches on both hosts (see docs/SITE.md, not in git)

# The DFlash2 drafter (incoai's weights, CC BY-NC-ND 4.0 — non-commercial terms; production
# runs the same revision). DRAFTS=1 passes --drafter with this snapshot when it is present on
# the head (the EXL3 path upstream measured; give both ranks the same drafter); when the
# snapshot is missing the server drafts with the checkpoint's MTP head instead, with a warning.
# DRAFTS=0 still passes --no-drafts (the serial reference).
DFLASH2_ID="${DFLASH2_ID:-incoai/GLM-5.3-Flash-DFlash2}"
DFLASH2_REVISION="${DFLASH2_REVISION:-bf582e4eacc1810f76656d1811693ff6c6737d2a}"

# The two GB10s: this machine (the head) serves rank 0 and the API; WORKER (ssh target,
# key-based) runs rank 1. Rank 1 is started first, rank 0 serves HTTP (the native engine's
# rank-over-host pattern; docs/DELTAS.md). TP=3 is not servable on the native engine yet
# (it waits on upstream's multi-rank bootstrap fabric), so TP is fixed at 2.
TP=2
# Site identity — REQUIRED, set in scripts/local.sh (git-ignored; local.sh.example has the
# shape, docs/SITE.md on a prepared site has its real values — that file is not in git).
WORKER="${WORKER:-}"                     # rank 1's ssh target (key-based), e.g. user@worker-host

# The CX7 rails between the pair (one private subnet per cable — /24s are the easy choice;
# see local.sh.example for how to find the interfaces and assign the addresses). NCCL's
# bootstrap listens on the head's rail address; NCCL discovers the rest from there. With
# both rails cabled the default split exchange works and TF_GLM_PREFILL_REDUCE=gather is
# NOT needed (that is for fabrics where some rank pairs cannot connect).
HEAD_FABRIC_ADDR="${HEAD_FABRIC_ADDR:-}"      # the head's rail address (NCCL bootstraps here)
WORKER_FABRIC_ADDR="${WORKER_FABRIC_ADDR:-}"  # the worker's rail address
NCCL_PORT="${NCCL_PORT:-29741}"
# Comma-separated netdevs NCCL may use over the link (both rails on a two-rail site).
# NCCL_IB_HCA is left empty on purpose (all RoCE devices); set it in local.sh to pin devices.
NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-}"
NCCL_IB_HCA="${NCCL_IB_HCA:-}"
# Extra NCCL debug, for a first bring-up: set NCCL_DEBUG=INFO in local.sh and watch the logs.

# Docker. The builder image compiles tensorfold-native (Zig + nvcc, sm_121); the runtime image
# carries the binary, its libraries and NCCL and is what both ranks run. The builder needs a
# CUDA devel base with nvcc for aarch64: nvcr.io/nvidia/pytorch:26.07-py3 is already pulled on
# the head and ships both, but any arm64 CUDA 13.x devel image works (set BASE_IMAGE).
ZIG_VERSION="${ZIG_VERSION:-0.17.0}"                     # TensorFold's .zig-version (still 0.17.0 at v1.1.0)
SM_ARCH="${SM_ARCH:-121}"                                # GB10
BASE_IMAGE="${BASE_IMAGE:-nvcr.io/nvidia/pytorch:26.07-py3}"
IMAGE_NAME="${IMAGE_NAME:-glm53-native-tf}"   # the runtime image; prepare.sh tags it <name>:<resolved version>
CONTAINER_NAME="${CONTAINER_NAME:-glm53-native-tf}"      # the same name on both Sparks

# Checkpoint and binary locations (host paths; the container mounts the HF cache read-write so
# the snapshot layout, blob names and hashes stay exactly as Hugging Face wrote them).
HF_CACHE="${HF_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}}"
MODEL_SNAPSHOT="$HF_CACHE/hub/models--${MODEL_ID//\//--}/snapshots/$MODEL_REVISION"

# Serving. The defaults ARE the qualified optimum (2026-10-23 campaign receipts in
# docs/TOOL_EVAL_TESTING.md): dense4fp8 at the 524,288 window, 4 streams — all §2 gates
# PASS, tool-eval 92 (= the production-parity cell's), throughput leader, ~2 GiB cheaper
# than dense8. PARALLEL: 5 admits at 512K on window-mode fp8 (the engine's own arithmetic);
# the KV-pages branch admits 8 with paging but excludes fp8 KV — docs/ECOSYSTEM.md.
SERVED_NAME="${SERVED_NAME:-GLM-5.3-Flash-EXL3}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"             # the same port production serves on: clients keep one endpoint,
                                 # and the two stacks are mutually exclusive (start.sh's guard
                                 # refuses a busy port instead of fighting for it)
CONTEXT="${CONTEXT:-524288}"
PARALLEL="${PARALLEL:-4}"
DRAFTS="${DRAFTS:-1}"            # 0 passes --no-drafts (the serial reference); 1 drafts, with
                                 # DFlash2 when its snapshot is present (DFLASH2_* above), else MTP
RECALL="${RECALL:-0}"            # 1 passes --recall on the server (1.0.12): drafts copy from text
                                 # the server has already read; verified-exact, its own size cap.
                                 # For agent loops that re-read text (2.4-3x upstream on repeats)
VISION="${VISION:-1}"            # 1 passes --vision --vision-image-tokens 2048 (GLM-5.3-Flash
                                 # image input: data-URL image_url parts in user messages;
                                 # production's parity line: a 1.05 GiB tower with 1.41 GiB of
                                 # workspace, 2048 tokens a picture, 16384 a clip — the clip
                                 # cap is the engine's own). Needs libcublasLt.so.13 in the
                                 # image (the Dockerfile copies it from the builder's CUDA 13
                                 # toolkit — round 52's fix for the round-51 refusal). The
                                 # (user) call: vision is a want, enabled as a standing
                                 # default at the 524288 window; no perf testing.
PROMPT_CACHE_GIB="${PROMPT_CACHE_GIB:-}"  # --prompt-cache-gib on the server: the kept-states
                                 # store. Round 47 receipt: at this window (524288x4) the
                                 # engine served its own 2.6 GiB default EVEN WITH the flag
                                 # =16 in argv (campaign-20261028-113509/21-cachegib.md) —
                                 # budget-cap or ignored, unresolved; the store is NOT free
                                 # to grow here. --spill-gib (v1.1.8) is the alternative:
                                 # evict kept states to disk instead of dropping them.
SPILL_GIB="${SPILL_GIB:-}"       # --spill-gib on the server (v1.1.8): evicted kept states
                                 # are written to disk (one I/O worker) instead of dropped
API_KEY="${API_KEY:-}"           # recommended; also required by upstream for --slide-style features
MEMORY_LIMIT_GB="${MEMORY_LIMIT_GB:-}"   # -> TENSORFOLD_CUDA_MEMORY_LIMIT_GB, empty: engine default

# Decode-side packing knobs (shipped in v1.1.0; 4, 6, 8 or 16 bits; empty = 16/bf16).
# The defaults are the measured optimum: dense4 is the speed leader at every depth
# (4 < 6 < 8 < 16 bits, monotonic — the v1.1.2+RoCE matrix) and dense4fp8 is the
# qualified serving config (all §2 gates PASS, tool-eval 92 = parity, throughput lead).
# Exactness holds within one configuration (drafted = plain there); packing is not
# bf16-exact, which is why each config's gates run on itself. PDL=1 turns on
# programmatic dependent launch on the mm kernels — it HURT dense4 in the matrix
# (−13/−11/−8% solo); keep 0.
DENSE_BITS="${DENSE_BITS:-4}"
HEAD_BITS="${HEAD_BITS:-4}"
PDL="${PDL:-0}"
# KV_CACHE: the latent cache's storage (config.sh → serve_args → --kv-cache). fp8 is
# v1.1.2's prepared fp8 latent cache and now the DEFAULT: the §2 exactness gates PASSED on
# it at 524288 (dense8+fp8 and dense4fp8, content form — VALIDATION §2 receipt), it halves
# per-stream KV bytes (the 512K ladder fits at 4–5 streams on it), and dense4fp8 is the
# qualified serving config. NOTE: fp8 KV and TF_KV_PAGES are mutually exclusive on the
# KV-pages branch (bf16 is its only paged path — docs/ECOSYSTEM.md).
KV_CACHE="${KV_CACHE:-fp8}"

# PROMPT_K: v1.1.1 regroups prompt-projection K slices (one K slice at every operator row
# count), which can change replies versus 1.1.0 — including long or resumed prompts. "shape"
# selects K slices by shape instead (the 1.1.0 policy family); it needs TF_GLM_MM=1 on every
# rank, which is the engine default. Empty: 1.1.1's default.
PROMPT_K="${PROMPT_K:-}"

# PATCHES: optional overlay of local .patch files applied to the TensorFold tree at image
# build (see patches/README.md for the policy: upstream-first, receipts required, rebase
# cost owned). Empty = pristine upstream. Names are files in patches/, space or comma
# separated; prepare.sh stages them into the build.
# The qualified default (2026-10-24, (user)): glm-topk4b — the glm family's reference
# shape moves to topk=4 (the parse's `want` pin + the EXL3 experts' module global), which
# the top4cfg checkpoint requires. Receipts: perf campaign-20261024-135458 (solo 108% of
# production, c2 103%, c4 78%, slope 0.593) and the battery campaign-20261024-142659
# (§2 gates ALL PASS, tool-eval 93/100 above stock's 92, hardmode 89 vs production's 91).
PATCHES="${PATCHES-}"   # '-' not ':-'; the default is EMPTY (the Focus serves pristine upstream: topk=8 by design, no patch — the topk4-pinned parse rejects its field-less config with NotGlm53Flash, whose weightShare catch bills the whole checkpoint to one rank). The top4cfg serve (the prior default) is MODEL_ID=Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold MODEL_REVISION=top4cfg PATCHES=glm-topk4b.patch

# TEMPERATURE: the server's default sampling temperature (stock default 1.0). TensorFold
# drafts with DFlash2 on greedy (temperature 0) requests only — agent frameworks that send
# no temperature get the 1.0 default and no DFlash2 drafts (jayleaton's measured set:
# 4.36 -> 5.73 tokens/step, ~68 -> ~83 tok/s on the same hardware class with
# --temperature=0). Set 0 for agent serving; clients that send their own temperature are
# unaffected. Empty: the engine default.
TEMPERATURE="${TEMPERATURE:-}"

# VISION: 1 passes --vision (the checkpoint's vision tower serves images; rank 0 encodes,
# the others receive). VISION_IMAGE_TOKENS bounds the per-image token budget (--vision-image-tokens).
# UNQUALIFIED: no vision receipts exist on this line yet — the vision gate round
# (PERF_PLAN) validates before any default flips to 1. Empty: the engine default (off).
VISION="${VISION:-}"
VISION_IMAGE_TOKENS="${VISION_IMAGE_TOKENS:-}"

# CPU pinning for both ranks (as --cpuset-cpus on the containers). On a GB10 the engine's CUDA
# sync thread and NCCL's progress thread spin at ~100% CPU each and heat the CPU's thermal
# zones (five of seven ACPI zones are the CPU's), which a heat watch eventually punishes
# (ashhart/TensorFold#1033: as-shipped, a 40-reply run stopped at the 7th at 94.7 C; pinned to
# the efficiency cores it finished 40/40 at -1.1% decode / -2.7% prefill). This site's GB10s
# are the same 20-core SoC; the issue's map is 0-4,10-14. Empty: all cores (as shipped).
CPUSET="${CPUSET:-}"

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

worker() { ssh -o BatchMode=yes "$WORKER" "$@"; }  # args ARE the remote command (a bare ssh
                                                   # would eat our stdin as shell input)

# The docker run contract, shared by both ranks (the same as the EXL3 recipe's: host networking
# for the CX7 rails, the RDMA devices, locked memory for NCCL's registered buffers). The
# tensorfold-cache mount keeps the engine's file-digest cache (v1.1.0's #900) across restarts:
# without it every start re-reads the whole checkpoint for the byte verify — measured here at
# 624.8 s (163.6 GiB); with the cache a warm start skips it (0.1 s upstream).
RUN_ARGS=(--gpus all --ipc=host --network host --shm-size 16g --device /dev/infiniband
          --cap-add IPC_LOCK --ulimit memlock=-1 --ulimit stack=67108864
          -v "$STATE_DIR/tensorfold-cache:/root/.cache/tensorfold")
[[ -z "$CPUSET" ]] || RUN_ARGS+=(--cpuset-cpus "$CPUSET")

# Environment passed into both ranks: offline serving from the mounted cache, plus every
# TENSORFOLD_/TF_GLM_/NCCL_ variable this recipe sets.
env_args() {
  ENV_ARGS=(-e HF_HUB_OFFLINE=1 -e NCCL_SOCKET_IFNAME="$NCCL_SOCKET_IFNAME")
  [[ -z "$NCCL_IB_HCA" ]] || ENV_ARGS+=(-e NCCL_IB_HCA="$NCCL_IB_HCA")
  [[ -z "$MEMORY_LIMIT_GB" ]] || ENV_ARGS+=(-e TENSORFOLD_CUDA_MEMORY_LIMIT_GB="$MEMORY_LIMIT_GB")
  # Every other NCCL_* (channels, passthrough, debug), TENSORFOLD_*, TF_GLM_*, the engine's
  # TF_HEAT_* bands, the TF_ROCE_* gather settings and TF_KV_* (#1163's paging knob) pass
  # through to both ranks.
  while IFS='=' read -r name _; do ENV_ARGS+=(-e "$name=${!name}"); done \
    < <(env | grep -E '^(TENSORFOLD|TF_GLM|TF_HEAT|TF_ROCE|TF_KV|NCCL)_[A-Z0-9_]+=' || true)
}

# The serve arguments both ranks take (rank 0 adds --name/--host/--port in start.sh). The ranks
# must agree on window and streams: each allocates its own state before rank 0 admits.
serve_args() {
  local args=(serve "/root/.cache/huggingface/hub/models--${MODEL_ID//\//--}/snapshots/$MODEL_REVISION"
              --context "$CONTEXT" --parallel "$PARALLEL")
  [[ -z "$TEMPERATURE" ]] || args+=(--temperature "$TEMPERATURE")
  [[ -z "$KV_CACHE" ]] || args+=(--kv-cache "$KV_CACHE")
  # VISION=1: serve the checkpoint's vision tower (347 model.visual.* tensors; rank 0
  # encodes, the others receive). Unqualified until the vision gate round runs —
  # PERF_PLAN's vision section carries the plan.
  [[ "$VISION" != 1 ]] || args+=(--vision)
  [[ -z "$VISION_IMAGE_TOKENS" ]] || args+=(--vision-image-tokens "$VISION_IMAGE_TOKENS")
  if [[ "$DRAFTS" == 1 ]]; then
    local dflash="/root/.cache/huggingface/hub/models--${DFLASH2_ID//\//--}/snapshots/$DFLASH2_REVISION"
    if [[ -e "$HF_CACHE/hub/models--${DFLASH2_ID//\//--}/snapshots/$DFLASH2_REVISION/config.json" ]]; then
      args+=(--drafter "$dflash")   # the same folder on both ranks; the engine checks that
    else
      warn "DFlash2 snapshot $DFLASH2_REVISION missing on the head; drafts fall back to MTP"
    fi
  else
    args+=(--no-drafts)
  fi
  echo "${args[@]}"
}

# The prepared marker: a one-line description of everything a start depends on. start.sh re-runs
# prepare.sh whenever the config fingerprint no longer matches (a new upstream release, another
# revision, another worker, a different window or stream count).
config_line() {   # the fingerprint: everything except the resolved version
  echo "model=$MODEL_ID@$MODEL_REVISION worker=$WORKER context=$CONTEXT parallel=$PARALLEL head=$HEAD_FABRIC_ADDR patches=${PATCHES// /,}"
}
prepared_line() { # the fingerprint plus what prepare resolved (written to $STATE_DIR/prepared)
  echo "tf=$TF_RESOLVED_REF@$TF_RESOLVED_COMMIT image=$IMAGE_REF $(config_line)"
}

# --- optional knob exports (must sit after die's definition; before the site guard) ----------
# The engine reads the TF_GLM_ names directly; exporting here means local.sh can use the short
# forms and env_args' TENSORFOLD_/TF_GLM_ forwarding carries them into both ranks.
case "$KV_CACHE" in ""|fp8) ;; *) die "KV_CACHE must be empty (bf16, the engine default) or fp8 (v1.1.2's prepared latent cache; quality unqualified upstream)" ;; esac
case "$DENSE_BITS" in ""|4|6|8|16) ;; *) die "DENSE_BITS must be empty (engine default 16) or 4, 6 or 8" ;; esac
case "$HEAD_BITS" in ""|4|6|8|16) ;; *) die "HEAD_BITS must be empty (engine default 16) or 4, 6 or 8" ;; esac
[[ -z "$DENSE_BITS" ]] || export TF_GLM_DENSE_BITS="$DENSE_BITS"
[[ -z "$HEAD_BITS" ]] || export TF_GLM_HEAD_BITS="$HEAD_BITS"
[[ "$PDL" == 0 ]] || export TF_GLM_PDL="$PDL"
case "$PROMPT_K" in ""|shape) ;; *) die "PROMPT_K must be empty (v1.1.1 default) or shape" ;; esac
[[ -z "$PROMPT_K" ]] || export TF_GLM_PROMPT_K="$PROMPT_K"
# The DSA indexer's two dedicated CUDA kernels (#1449, v1.1.8): upstream's contract is
# byte-equal scoring; our receipts (rounds 47-48 — campaign-20261028-113509/20-indexdepth.md,
# campaign-20261028-141916/22-ixgates.md): cold-TTFT -2.6% at 131k → -5.5% at 393k with the
# off-arm slopes rising vs flat on-arm, §2 gates ALL PASS, the duel within noise, loading/
# decode/d0 unchanged. ADOPTED as the standing default (round 49); unset to compare.
export TF_GLM_INDEX_CUDA="${TF_GLM_INDEX_CUDA:-1}"
case "$TEMPERATURE" in ""|*) [[ -z "$TEMPERATURE" ]] || [[ "$TEMPERATURE" =~ ^[0-9]+(\.[0-9]+)?$ ]] || die "TEMPERATURE must be a number (0 enables DFlash2 drafts on every request; empty: the engine default 1.0)" ;; esac

# --- site identity must be set (scripts/local.sh) -------------------------------------------
# Fail loudly rather than guessing a worker or a fabric address: a wrong guess rank-1s into
# nothing, or worse, onto the wrong host.
for _req in WORKER HEAD_FABRIC_ADDR WORKER_FABRIC_ADDR NCCL_SOCKET_IFNAME; do
  [[ -n "${!_req}" ]] || die "site setting $_req is empty — copy scripts/local.sh.example to
scripts/local.sh and fill it in (a prepared site's values are in docs/SITE.md, not in git)"
done
unset _req
