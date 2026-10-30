#!/usr/bin/env bash
# scripts/autonomous-window.sh — the self-contained window (round 90; production
# lifecycle corrected round 92: it now stops and starts production through the
# -tip checkout's stop.sh/start.sh, which handle BOTH ranks — the head and the
# worker — and save each rank's log; the raw docker stop/start this script used
# before only touched the head rank, which is why the early autonomous attempts
# did not work). (user): the agent may stop/restart production for a staged
# window, and THIS script is the only thing allowed to. Run detached from the
# agent: it stops production (the agent, served by production, goes dark
# mid-launch), runs the window phases, and restarts production on EVERY exit
# path — success, failure, or interrupt. Multi-hour/multi-day safe: nothing
# here depends on a live agent session; the head's cron crash net
# (scripts/autonomous-cron.sh) and the agent host's poke (scripts/autonomous-
# poke.sh) cover the aftermath.
#   setsid nohup scripts/autonomous-window.sh "preflight ncurouter summary" </dev/null >/tmp/autonomous-launch.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
[ -f scripts/local.sh ] && source scripts/local.sh
: "${PROD_TIP:?set PROD_TIP (the production checkout dir name) in scripts/local.sh}"
PHASES="${1:-preflight summary}"
CONSOLE="${2:-/tmp/autonomous.console}"
STATE=/tmp/autonomous-testing.flag
TIP="$HOME/$PROD_TIP"
note() { echo "[$(date '+%F %T')] $*" >> "$CONSOLE"; }
restore() {
  note "production restore: the tip's start.sh (both ranks, waits for the API)"
  if ! "$TIP/start.sh" >>"$CONSOLE" 2>&1; then
    note "START.SH FAILED — production may be DOWN; the cron crash-net retries every 5 min"
  fi
  rm -f "$STATE"
  note "production restored (window rc=${1:-?})"
}
trap 'restore $?' EXIT
[ -f "$STATE" ] && { note "a window is already running ($STATE exists) — refusing"; exit 1; }
touch "$STATE"
sleep 30   # let the launching ssh detach and the agent's closing message flush; the agent (served by production) dies at the stop
note "autonomous window starting: $PHASES"
"$TIP/stop.sh" >>"$CONSOLE" 2>&1
note "production stopped (both ranks) — the window runs"
PHASES="$PHASES" FORCE=1 scripts/run-window-block.sh </dev/null >>"$CONSOLE" 2>&1
rc=$?
note "window finished rc=$rc"
exit $rc
