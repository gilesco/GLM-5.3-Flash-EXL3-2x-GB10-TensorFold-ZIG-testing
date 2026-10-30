#!/usr/bin/env bash
# Serve GLM-5.3-Flash (EXL3) on the two GB10s with TensorFold's native CUDA engine, one rank
# per host: rank 1 on WORKER (started first, it follows), rank 0 here serving the API on $PORT.
# Both ranks run the same runtime image as detached containers named $CONTAINER_NAME;
# ./scripts/stop.sh stops them, ./scripts/status.sh reports.
#
# Usage: scripts/start.sh [extra tensorfold-native serve args...]   (run on the head Spark)
#   DRY_RUN=1 scripts/start.sh   print both ranks' docker commands, change nothing, need nothing
# Re-runs scripts/prepare.sh first when the config fingerprint no longer matches the prepared
# marker (a new upstream v1.* release, another revision, another worker, other window/streams).
# Instrument hooks (session phases): RUN_IMAGE launches this image ref instead of the
# prepared one; RANK0_WRAP prefixes rank 0's container command (e.g. a profiler).
# RANK1_WRAP (round 113) wraps rank 1 the same way: a wrapper on rank 0 alone makes the
# engines' program fingerprint (the executable's sha256, all-gathered at rank start)
# differ and BOTH ranks refuse — a profiled pair must be symmetric, so the wrapped
# rank 1 also follows RUN_IMAGE (the same instrument image; the caller ships it to the
# worker first). Both are start.sh-level — they never trigger a re-prepare.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
source ./config.sh
DRY="${DRY_RUN:-0}"

# The resolved version the prepared marker recorded is the pin that gets served; a dry run
# shows it when one exists and a placeholder when it does not (it never prepares or builds).
resolved_from_marker() {
  local _tf
  _tf=$(sed -n 's/.*tf=\([^@]*\)@\([^ ]*\).*/\1 \2/p' "$STATE_DIR/prepared" 2>/dev/null) || return 1
  [[ -n "$_tf" ]] || return 1
  TF_RESOLVED_REF=${_tf%% *}; TF_RESOLVED_COMMIT=${_tf##* }
  IMAGE_REF="${IMAGE_NAME}:${TF_RESOLVED_REF}"
}
if [[ "$DRY" == 1 ]]; then
  resolved_from_marker || IMAGE_REF="${IMAGE_NAME}:<unprepared>"
fi

if [[ "$DRY" != 1 ]]; then
  current_config="$(bash -c 'source ./config.sh; config_line')"
  if [[ ! -f "$STATE_DIR/prepared.cfg" ]] || [[ "$(cat "$STATE_DIR/prepared.cfg")" != "$current_config" ]]; then
    log "prepared state is missing or differs from the config - running scripts/prepare.sh"
    ./scripts/prepare.sh
  fi
  resolved_from_marker || die "prepared marker has no resolved version; run scripts/prepare.sh"
  log "TensorFold $TF_RESOLVED_REF ($TF_RESOLVED_COMMIT)"
  command -v docker >/dev/null || die "docker is required here (this script runs on the head Spark)"
  docker image inspect "$IMAGE_REF" >/dev/null 2>&1 || { log "image $IMAGE_REF missing - running prepare"; ./scripts/prepare.sh; }
  [[ -z "$(docker ps -q --filter "name=^/${CONTAINER_NAME}$")" ]] || die "a $CONTAINER_NAME container is already running here (./scripts/stop.sh first)"
  # Fail fast when the hosts are busy: the production EXL3 stack (glm53-flash-tf) holds both
  # GPUs and serves on its own port (production 8000, the testing stack 8001 — round 117);
  # nvidia-smi reports unified memory as [N/A], so busy is read from MemAvailable instead.
  if ss -tln 2>/dev/null | grep -q ":$PORT "; then
    die "port $PORT is already listening on the head - the production stack probably still serves it; stop that first (docs/LAUNCH.md)"
  fi
  mem_busy() { # $1 empty for the head, else an ssh target; true when unified memory is mostly held
    local avail
    avail=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
    [[ -n "$avail" && "$avail" -lt 30000 ]]
  }
  if mem_busy; then
    die "the head has only $(awk '/MemAvailable/{printf "%.0f GiB", $2/1048576}' /proc/meminfo) available - the production stack (glm53-flash-tf) is probably still up; stop it first (docs/LAUNCH.md)"
  fi
  if worker "$(declare -f mem_busy); mem_busy"; then
    die "the worker ($WORKER) is low on available memory - the production rank is probably still up; stop it first (docs/LAUNCH.md)"
  fi
fi

env_args
read -r -a ARGS <<< "$(serve_args)"
rank0_flags=(--name "$SERVED_NAME" --host "$HOST" --port "$PORT")
[[ -z "$API_KEY" ]] || rank0_flags+=(--api-key "$API_KEY")
[[ "$RECALL" != 1 ]] || rank0_flags+=(--recall)
[[ -z "$PROMPT_CACHE_GIB" ]] || rank0_flags+=(--prompt-cache-gib "$PROMPT_CACHE_GIB")
[[ -z "$SPILL_GIB" ]] || rank0_flags+=(--spill-gib "$SPILL_GIB")
# Vision at production's parity limits (its serve line: a 1.05 GiB tower with 1.41 GiB of
# workspace; at most 2048 tokens a picture, 16384 a clip — 128 frames at 2/s is the
# engine's own clip cap, and --vision-video-tokens already defaults to 16384).
# Vision is a RANK-AGREED setting: both ranks must carry the same --vision (the engine's
# rank cross-check refuses a split — receipt: campaign-20261028-153511, "rank 1's vision
# differs from rank 0's"). The flags above are genuinely rank-0-side (the prompt cache
# and the spill store live on rank 0); vision is not.
rank_both_flags=()
[[ "$VISION" != 1 ]] || rank_both_flags+=(--vision --vision-image-tokens 2048 --chat-template /opt/tensorfold/share/glm5-next-vision.jinja)
# CHAT_TEMPLATE=effort-tail: a rank-agreed --chat-template override (the template file is
# staged into the image by the Dockerfile). Text-only for now — VISION=1 already sets its
# own --chat-template and the two would collide; merging the effort-tail fix into the
# vision template is the follow-up.
# CHAT_TEMPLATE=effort-tail + VISION=1 needs no extra flag: the vision template
# carries the effort-tail fix itself (round 77), so the vision serve already has it.
if [[ "$VISION" != 1 && "$CHAT_TEMPLATE" == "effort-tail" ]]; then
  rank_both_flags+=(--chat-template /opt/tensorfold/share/glm5-next-effort-tail.jinja)
fi
if [[ "$VISION" != 1 && "$CHAT_TEMPLATE" == "effort-tail-tr" ]]; then
  rank_both_flags+=(--chat-template /opt/tensorfold/share/glm5-next-effort-tail-tr.jinja)
fi

rank_env() { # $1 rank id, $2 name of the array to append to; the ranks rendezvous on the
             # head's rail address
  local -n _out=$2
  _out+=(-e "TF_GLM_RANK=$1" -e "TF_GLM_WORLD=$TP" -e "NCCL_COMM_ID=$HEAD_FABRIC_ADDR:$NCCL_PORT")
}
mount_args=(-v "$HF_CACHE:/root/.cache/huggingface")
RANK1_ENV=(); RANK0_ENV=(); rank_env 1 RANK1_ENV; rank_env 0 RANK0_ENV

# instrument hooks (the :- defaults keep set -u happy when nothing is passed).
# RUN_IMAGE overrides rank 0's image only — rank 1 always runs the prepared image:
# the committed instrument image exists on the head alone (a worker-side copy would
# need a 2 GB docker save/load over ssh), and rank 1's engine bits are identical.
# RANK0_WRAP makes the wrapper binary the container ENTRYPOINT — the image's own
# entrypoint (the engine) and the serve args follow as the wrapper's arguments.
# Prepending to the command instead hands the wrapper's flags to the engine, which
# then prints its usage (the round-95 window's exact failure).
RANK0_WRAP="${RANK0_WRAP:-}"; RUN_IMAGE="${RUN_IMAGE:-}"; RANK1_WRAP="${RANK1_WRAP:-}"; RANK1_IMAGE="${RANK1_IMAGE:-}"
PREPARED_IMAGE_REF="$IMAGE_REF"
RANK0_WRAP_ARR=(); RANK1_WRAP_ARR=(); IMG_EP=""
if [[ -n "$RANK0_WRAP" ]]; then
  read -r -a RANK0_WRAP_ARR <<< "$RANK0_WRAP"
  IMG_EP="$(docker image inspect --format '{{join .Config.Entrypoint " "}}' "$PREPARED_IMAGE_REF" 2>/dev/null || true)"
  [[ -n "$IMG_EP" ]] || die "RANK0_WRAP set but $PREPARED_IMAGE_REF has no entrypoint to wrap"
fi
if [[ -n "$RANK1_WRAP" ]]; then
  read -r -a RANK1_WRAP_ARR <<< "$RANK1_WRAP"
  [[ -n "$IMG_EP" ]] || { IMG_EP="$(docker image inspect --format '{{join .Config.Entrypoint " "}}' "$PREPARED_IMAGE_REF" 2>/dev/null || true)"; [[ -n "$IMG_EP" ]] || die "RANK1_WRAP set but $PREPARED_IMAGE_REF has no entrypoint to wrap"; }
fi
[[ -z "$RUN_IMAGE" ]] || IMAGE_REF="$RUN_IMAGE"
# a wrapped rank 1 runs the SAME image as the wrapped rank 0 (the program fingerprints
# must agree); RANK1_IMAGE overrides rank 1's image WITHOUT a wrap (the fingerprint
# hashes /proc/self/exe — the engine binary, identical from the same image whether or
# not ncu wraps rank 0 — the round-113 census topology); unwrapped and unoverridden
# rank 1 keeps the prepared image (the old rule)
RANK1_IMAGE_REF="$PREPARED_IMAGE_REF"
[[ -z "$RANK1_WRAP" && -z "$RANK1_IMAGE" ]] || RANK1_IMAGE_REF="$IMAGE_REF"
[[ -z "$RANK1_IMAGE" ]] || RANK1_IMAGE_REF="$RANK1_IMAGE"
if [[ -n "$RANK1_WRAP" ]]; then
  # rank 1's command is a STRING through ssh — a remote shell parses it. The wrap's
  # tokens (the ncu --kernel-name regex carries |) must be shell-quoted or the remote
  # bash executes the regex's pipes (the round-113 window's exact failure).
  RANK1_WRAP_STR=""
  for _tok in "${RANK1_WRAP_ARR[@]}"; do RANK1_WRAP_STR+=" $(printf '%q' "$_tok")"; done
  rank1_cmd="docker rm -f $CONTAINER_NAME 2>/dev/null; docker run -d --name $CONTAINER_NAME ${RUN_ARGS[*]} --cap-add SYS_ADMIN --entrypoint ${RANK1_WRAP_ARR[0]} ${ENV_ARGS[*]} ${RANK1_ENV[*]} ${mount_args[*]} $RANK1_IMAGE_REF${RANK1_WRAP_STR} ${IMG_EP} ${ARGS[*]} ${rank_both_flags[*]}"
else
  rank1_cmd="docker rm -f $CONTAINER_NAME 2>/dev/null; docker run -d --name $CONTAINER_NAME ${RUN_ARGS[*]} ${ENV_ARGS[*]} ${RANK1_ENV[*]} ${mount_args[*]} $PREPARED_IMAGE_REF ${ARGS[*]} ${rank_both_flags[*]}"
fi
if [[ -n "$RANK0_WRAP" ]]; then
  # the wrapper needs GPU performance-counter permission — this driver loads with
  # RmProfilingAdminOnly=1, and CAP_SYS_ADMIN satisfies it (proven on hardware: a
  # tiny matmul capture produced duration rows under the same cap, round 97)
  rank0_cmd=(docker run -d --name "$CONTAINER_NAME" "${RUN_ARGS[@]}" --cap-add SYS_ADMIN --entrypoint "${RANK0_WRAP_ARR[0]}" "${ENV_ARGS[@]}" "${RANK0_ENV[@]}" "${mount_args[@]}" "$IMAGE_REF" "${RANK0_WRAP_ARR[@]:1}" ${IMG_EP} "${ARGS[@]}" "${rank_both_flags[@]}" "${rank0_flags[@]}")
else
  rank0_cmd=(docker run -d --name "$CONTAINER_NAME" "${RUN_ARGS[@]}" "${ENV_ARGS[@]}" "${RANK0_ENV[@]}" "${mount_args[@]}" "$IMAGE_REF" "${ARGS[@]}" "${rank_both_flags[@]}" "${rank0_flags[@]}")
fi

if [[ "$DRY" == 1 ]]; then
  echo "[dry-run] rank 1 on $WORKER:  $rank1_cmd"
  echo "[dry-run] rank 0 here:        ${rank0_cmd[*]}"
  exit 0
fi

log "rank 1 on $WORKER (starts first; it follows rank 0's steps)"
worker "$rank1_cmd" >/dev/null || die "could not start rank 1 on $WORKER"

log "rank 0 here: serving $MODEL_ID@$MODEL_REVISION at $HOST:$PORT (container $CONTAINER_NAME)"
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
"${rank0_cmd[@]}" >/dev/null

log "waiting for /health (weights load in minutes; the ranks first verify each other's checkpoint bytes)"
deadline=$(( $(date +%s) + ${HEALTH_TIMEOUT_S:-900} ))
until curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; do
  if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
    echo; warn "rank 0's container exited; its log:"; docker logs --tail 30 "$CONTAINER_NAME" 2>&1 | sed 's/^/  /'
    warn "rank 1's log: ssh $WORKER docker logs --tail 30 $CONTAINER_NAME"; exit 1
  fi
  if (( $(date +%s) > deadline )); then
    warn "timed out waiting for /health after ${HEALTH_TIMEOUT_S:-900} s"
    warn "rank 0: docker logs -f $CONTAINER_NAME   rank 1: ssh $WORKER docker logs -f $CONTAINER_NAME"
    exit 1
  fi
  sleep 5
done
log "up: http://127.0.0.1:$PORT/v1  (model $SERVED_NAME)"
log "stop: ./scripts/stop.sh    status: ./scripts/status.sh    logs: docker logs -f $CONTAINER_NAME"
