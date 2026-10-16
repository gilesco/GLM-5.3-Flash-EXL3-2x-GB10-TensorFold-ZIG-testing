#!/usr/bin/env bash
# Stop both ranks and save each one's log to $LOG_DIR (gzipped, the newest $LOG_KEEP kept),
# as the EXL3 recipe's stop.sh did: docker rm throws a container's log away, so without this a
# crash's log is gone at the next start. Stops the worker's rank first, then the head's, so a
# dying rank never leaves its peer waiting (either rank ending takes the other down anyway).
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
source ./config.sh
command -v docker >/dev/null || die "docker is required here (this script runs on the head Spark)"
mkdir -p "$LOG_DIR"

save_log() { # $1 host ("local" or ssh target), $2 rank, $3 label
  local out
  out="$LOG_DIR/$(date +%Y%m%d-%H%M%S)-$2.log.gz"
  if [[ "$1" == local ]]; then docker logs "$CONTAINER_NAME" 2>&1 | gzip > "$out" || return 0
  else ssh -o BatchMode=yes "$1" "docker logs $CONTAINER_NAME 2>&1" | gzip > "$out" || return 0; fi
  echo "$3 log saved: $out"
}
keep_newest() { ls -t "$LOG_DIR"/rank-$1.log.gz 2>/dev/null | tail -n +$(( LOG_KEEP + 1 )) | xargs -r rm -f; }

if ssh -o BatchMode=yes "$WORKER" "docker ps -q --filter name=^/${CONTAINER_NAME}\$ | grep -q ."; then
  save_log "$WORKER" 1 "rank 1 ($WORKER)"; keep_newest 1
  ssh -o BatchMode=yes "$WORKER" "docker rm -f $CONTAINER_NAME" >/dev/null && echo "rank 1 stopped on $WORKER"
else
  echo "rank 1 not running on $WORKER"
fi
if docker ps -q --filter "name=^/${CONTAINER_NAME}$" | grep -q .; then
  save_log local 0 "rank 0 (head)"; keep_newest 0
  docker rm -f "$CONTAINER_NAME" >/dev/null && echo "rank 0 stopped on the head"
else
  echo "rank 0 not running on the head"
fi
