#!/usr/bin/env bash
# What is running where: both ranks' containers, /health, and each rank's last log lines.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
source ./config.sh
command -v docker >/dev/null || die "docker is required here (this script runs on the head Spark)"

echo "== containers =="
echo "head ($(hostname -s)): $(docker ps -a --filter "name=^/${CONTAINER_NAME}$" --format '{{.Status}} (image {{.Image}})')"
echo "worker ($WORKER): $(ssh -o BatchMode=yes "$WORKER" "docker ps -a --filter name=^/${CONTAINER_NAME}\$ --format '{{.Status}} (image {{.Image}})'")"
echo "== health (http://127.0.0.1:$PORT) =="
curl -fsS -m 5 "http://127.0.0.1:$PORT/health" 2>/dev/null | head -c 2000 && echo || echo "no answer (server down or still loading)"
echo "== last log lines =="
echo "-- rank 0 (head)"; docker logs --tail 3 "$CONTAINER_NAME" 2>&1 | sed 's/^/  /' || echo "  (no container)"
echo "-- rank 1 ($WORKER)"; ssh -o BatchMode=yes "$WORKER" "docker logs --tail 3 $CONTAINER_NAME 2>&1" | sed 's/^/  /' || true
