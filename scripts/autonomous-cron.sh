#!/usr/bin/env bash
# scripts/autonomous-cron.sh — the head's crash net (cron every 5 min). If
# production is DOWN and no window is running (no flag): restart it through the
# -tip checkout's start.sh (both ranks) and log it — production must never stay
# down. The wake/poke half lives on the agent host now (scripts/autonomous-
# poke.sh, the opencode server's durable prompt inbox); this file only guards
# production's lifecycle.
set -uo pipefail
cd "$(dirname "$0")/.."
exec 9>/tmp/autonomous-cron.lock
flock -n 9 || exit 0
[ -f scripts/local.sh ] && source scripts/local.sh
: "${PROD_TIP:?set PROD_TIP (the production checkout dir name) in scripts/local.sh}"
STATE=/tmp/autonomous-testing.flag
LOG="$HOME/autonomous-cron.log"
TIP="$HOME/$PROD_TIP"
if ! docker ps --format "{{.Names}}" | grep -qx glm53-flash-tf; then
  if [ ! -f "$STATE" ]; then
    echo "[$(date '+%F %T')] production down with no active window — restarting via the tip's start.sh" >> "$LOG"
    "$TIP/start.sh" >> "$LOG" 2>&1 || echo "[$(date '+%F %T')] start.sh rc=$? — will retry next tick" >> "$LOG"
  else
    echo "[$(date '+%F %T')] production down, window in flight — hands off" >> "$LOG"
  fi
fi
