#!/usr/bin/env bash
# scripts/autonomous-window.sh — the self-contained window (round 90; (user): the
# agent may stop/restart production for a staged window, and THIS script is the only
# thing allowed to). Run detached from the agent: it stops production (the agent,
# served by production, goes dark mid-launch), runs the window phases, and restarts
# production on EVERY exit path — success, failure, or interrupt. Multi-hour/multi-
# day safe: nothing here depends on a live agent session; the cron safety net
# (scripts/autonomous-cron.sh) watches the flag below.
#   setsid nohup scripts/autonomous-window.sh "preflight ncurouter summary" </dev/null >/tmp/autonomous-launch.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
PHASES="${1:-preflight summary}"
CONSOLE="${2:-/tmp/autonomous.console}"
STATE=/tmp/autonomous-testing.flag
PROD=glm53-flash-tf
restore() {
  docker start "$PROD" >/dev/null 2>&1 || true
  rm -f "$STATE"
  echo "[$(date '+%F %T')] production restored (window rc=${1:-?})" >> "$CONSOLE"
}
trap 'restore $?' EXIT
[ -f "$STATE" ] && { echo "[$(date '+%F %T')] a window is already running ($STATE exists) — refusing" >> "$CONSOLE"; exit 1; }
touch "$STATE"
sleep 10   # let the launching ssh detach; the agent (served by production) dies at the stop
echo "[$(date '+%F %T')] autonomous window starting: $PHASES" >> "$CONSOLE"
docker stop "$PROD"
echo "[$(date '+%F %T')] production stopped — the window runs" >> "$CONSOLE"
PHASES="$PHASES" FORCE=1 scripts/run-window-block.sh </dev/null >>"$CONSOLE" 2>&1
rc=$?
echo "[$(date '+%F %T')] window finished rc=$rc" >> "$CONSOLE"
exit $rc
