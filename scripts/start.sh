#!/usr/bin/env bash
# Serve GLM-5.3-Flash (NVFP4) on the two GB10s with TensorFold's native CUDA engine, one rank
# per host: rank 1 on WORKER (started first, it follows), rank 0 here serving the API on $PORT.
# Both ranks run the same runtime image as detached containers named $CONTAINER_NAME;
# ./scripts/stop.sh stops them, ./scripts/status.sh reports.
#
# Usage: scripts/start.sh [extra tensorfold-native serve args...]
#   DRY_RUN=1 scripts/start.sh   print both ranks' docker commands, change nothing
# Re-runs scripts/prepare.sh first when the config fingerprint no longer matches the prepared
# marker (a new upstream v1.* release, another revision, another worker, other window/streams).
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
source ./config.sh

current_config="$(bash -c 'source ./config.sh; config_line')"
if [[ ! -f "$STATE_DIR/prepared.cfg" ]] || [[ "$(cat "$STATE_DIR/prepared.cfg")" != "$current_config" ]]; then
  log "prepared state is missing or differs from the config - running scripts/prepare.sh"
  ./scripts/prepare.sh
fi
# Read back the resolved version the marker recorded: this is the pin that gets served.
_tf=$(sed -n 's/.*tf=\([^@]*\)@\([^ ]*\).*/\1 \2/p' "$STATE_DIR/prepared")
[[ -n "$_tf" ]] || die "prepared marker has no resolved version; run scripts/prepare.sh"
TF_RESOLVED_REF=${_tf%% *}; TF_RESOLVED_COMMIT=${_tf##* }
IMAGE_REF="${IMAGE_NAME}:${TF_RESOLVED_REF}"
log "TensorFold $TF_RESOLVED_REF ($TF_RESOLVED_COMMIT)"

docker image inspect "$IMAGE_REF" >/dev/null 2>&1 || { log "image $IMAGE_REF missing - running prepare"; ./scripts/prepare.sh; }
[[ -z "$(docker ps -q --filter "name=^/${CONTAINER_NAME}$")" ]] || die "a $CONTAINER_NAME container is already running here (./scripts/stop.sh first)"

env_args
read -r -a ARGS <<< "$(serve_args)"
rank0_flags=(--name "$SERVED_NAME" --host "$HOST" --port "$PORT")
[[ -z "$API_KEY" ]] || rank0_flags+=(--api-key "$API_KEY")

rank_env() { # $1 rank id; the ranks rendezvous on the head's rail address
  printf ' -e TF_GLM_RANK=%s -e TF_GLM_WORLD=%s -e NCCL_COMM_ID=%s:%s' "$1" "$TP" "$HEAD_FABRIC_ADDR" "$NCCL_PORT"
}
mount_args="-v $HF_CACHE:/root/.cache/huggingface"

rank1_cmd="docker rm -f $CONTAINER_NAME 2>/dev/null; docker run -d --name $CONTAINER_NAME ${RUN_ARGS[*]} ${ENV_ARGS[*]}$(rank_env 1) $mount_args $IMAGE_REF ${ARGS[*]}"
rank0_cmd=(docker run -d --name "$CONTAINER_NAME" "${RUN_ARGS[@]}" "${ENV_ARGS[@]}" $(rank_env 0) $mount_args "$IMAGE_REF" "${ARGS[@]}" "${rank0_flags[@]}")

if [[ "${DRY_RUN:-0}" == 1 ]]; then
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
