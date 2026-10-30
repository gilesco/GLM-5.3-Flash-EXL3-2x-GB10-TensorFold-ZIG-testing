#!/usr/bin/env bash
# scripts/autonomous-launch.sh — the ONLY way a window is launched (the round-89
# inline-chain incident; the round-93 double-launch race where one launch's flag
# rm stripped the flag another window had just taken, so the crash-net started
# production mid-window). Syncs the head to origin/main, refuses when a window
# is already running, and starts the window detached. The agent calls exactly
# this, via one ssh, as the LAST action of its turn:
#   ssh <head> 'cd <repo> && scripts/autonomous-launch.sh "preflight ncurouter summary"'
set -uo pipefail
cd "$(dirname "$0")/.."
STATE=/tmp/autonomous-testing.flag
[ -f "$STATE" ] && { echo "a window is already running ($STATE) — refusing"; exit 1; }
git fetch origin -q || true
git merge --ff-only origin/main -q || echo "sync to origin/main failed — launching the checkout as-is"
setsid nohup scripts/autonomous-window.sh "${1:-preflight summary}" "${2:-/tmp/autonomous.console}" </dev/null >/tmp/autonomous-launch.log 2>&1 &
sleep 1
echo "launched: ${1:-preflight summary}"
