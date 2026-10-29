#!/usr/bin/env bash
# scripts/autonomous-cron.sh — the safety net + the wake marker (cron, every 30 min).
# 1. If production is DOWN, no window is running (no flag), and the port is not being
#    freed by a wrap: restart production and log it — production must never stay down.
# 2. If production is UP and a newer campaign receipt exists than the last-digested
#    marker: write $HOME/.opencode-poke with its path — the agent reads that file first
#    on any wake. (The opencode CLI poke is wired when its headless invocation lands.)
set -uo pipefail
PROD=glm53-flash-tf
STATE=/tmp/autonomous-testing.flag
MARKER="$HOME/.opencode-last-digested"
POKE="$HOME/.opencode-poke"
LOG="$HOME/autonomous-cron.log"
if ! docker ps --format "{{.Names}}" | grep -qx "$PROD"; then
  if [ ! -f "$STATE" ]; then
    echo "[$(date '+%F %T')] production down with no active window — restarting" >> "$LOG"
    docker start "$PROD" >> "$LOG" 2>&1 || true
  else
    echo "[$(date '+%F %T')] production down, window in flight — hands off" >> "$LOG"
  fi
fi
NEW=$(ls -td "$HOME"/GLM-5.3-Flash-EXL3-2x-GB10-TensorFold-v1.1-testing/results/campaign-* 2>/dev/null | head -1)
if [ -n "$NEW" ] && [ ! -f "$MARKER" -o "$NEW" -nt "$MARKER" ]; then
  echo "$NEW" > "$POKE"
fi
