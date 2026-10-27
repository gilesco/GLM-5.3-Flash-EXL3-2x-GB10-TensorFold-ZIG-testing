#!/usr/bin/env bash
# Window block: wait for production to stop, run this window's qualification
# phases (the campaign), then stop the testing stack so production can return.
# This script NEVER stops or modifies production — it only waits for it to go
# (the campaign's own guard refuses to start GPU phases while production is up;
# this wrapper turns that refusal into a wait).
#
#   Launch detached (survives everything, incl. the agent's session):
#     cd ~/GLM-5.3-Flash-EXL3-2x-GB10-TensorFold-v1.1-testing
#     setsid nohup scripts/run-window-block.sh </dev/null >/tmp/window-block.console 2>&1 &
#
#   Console:  /tmp/window-block.console  (+ the campaign's receipts under results/)
#   Override the phases:  PHASES="preflight top4qual summary" scripts/run-window-block.sh
set -uo pipefail
cd "$(dirname "$0")/.."
# export is the load-bearing word: without it PHASES stays a shell variable and
# the campaign never sees it (round 36's block ran the full default phase list
# because of exactly this — a 4.7 h accident the receipts salvaged).
export PHASES="${PHASES:-preflight top4qual focusqual summary}"
STEPS="${STEPS:-900}"   # 15 s per step: ~3.75 h ceiling for the production stop

echo "=== window block $(date): PHASES=$PHASES — waiting for production (glm53-flash-tf) to stop ==="
for _ in $(seq 1 "$STEPS"); do
  docker ps --format '{{.Names}}' 2>/dev/null | grep -qx glm53-flash-tf || break
  sleep 15
done
if docker ps --format '{{.Names}}' 2>/dev/null | grep -qx glm53-flash-tf; then
  echo "=== production still up after the wait ceiling — aborting; nothing was started ==="
  exit 1
fi
echo "=== production is down at $(date) — launching the campaign ==="
scripts/run-qualification.sh
RC=$?
echo "=== campaign wrapped (rc=$RC) at $(date) — stopping the testing stack ==="
scripts/stop.sh </dev/null >/dev/null 2>&1 || true
echo "=== the port is free — production can be restarted."
echo "=== receipts: results/ (the newest campaign-* dir; bench runs under run-*/ inside it) ==="
