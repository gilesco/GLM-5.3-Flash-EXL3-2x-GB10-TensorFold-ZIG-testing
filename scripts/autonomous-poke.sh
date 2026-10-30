#!/usr/bin/env bash
# scripts/autonomous-poke.sh — runs on the agent host (cron every 5 min); wakes
# the agent after a testing window. If production on the head is UP (container
# running AND its /health answers, port read from the tip's own config) and the
# newest campaign receipt dir on the head has not been poked yet: POST
# /api/session/{id}/prompt to the opencode server (v2.x — the unit's name, the
# server URL and the basic-auth password all come from git-ignored local.sh /
# the systemd unit itself; nothing site-specific is committed). The prompt
# endpoint admits the input durably and schedules the agent loop, so the wake
# works while the agent sits idle-dark. The head's ssh target comes from
# scripts/local.sh in THIS clone (git-ignored site config).
set -uo pipefail
cd "$(dirname "$0")/.."
[ -f scripts/local.sh ] && source scripts/local.sh
: "${SSH_TARGET:?set SSH_TARGET (user@head) in scripts/local.sh}"
: "${PROD_TIP:?set PROD_TIP (the production checkout dir name) in scripts/local.sh}"
: "${OPENCODE_API:?set OPENCODE_API (the opencode server API base URL) in scripts/local.sh}"
: "${OPENCODE_UNIT:?set OPENCODE_UNIT (the opencode server systemd unit path) in scripts/local.sh}"
PROJ="GLM-5.3-Flash-EXL3-2x-GB10-TensorFold-v1.1-testing"   # this repo's dir name, same on both hosts
MARKER="$HOME/.opencode-last-poked"
LOG="$HOME/autonomous-poke.log"
PW=$(grep -oP 'OPENCODE_SERVER_PASSWORD=\K.*' "$OPENCODE_UNIT" 2>/dev/null | tr -d '"')
[ -n "$PW" ] || { echo "$(date '+%F %T') no web password — cannot poke" >> "$LOG"; exit 1; }
API="$OPENCODE_API"
AUTH="opencode:$PW"
# 1. production fully up on the head? (container AND its health endpoint)
if ! ssh -o BatchMode=yes -o ConnectTimeout=10 "$SSH_TARGET" \
     "docker ps --format '{{.Names}}' | grep -qx glm53-flash-tf && cd ~/$PROD_TIP 2>/dev/null && source ./scripts/config.sh >/dev/null 2>&1; curl -s -m 5 -o /dev/null \"http://127.0.0.1:\$PORT/health\"" 2>/dev/null; then
  exit 0   # down: a window in flight or a crash — the head's cron crash-net handles it
fi
# a window in flight (even with production crash-net-restored mid-window) — hands off
if ssh -o BatchMode=yes "$SSH_TARGET" "test -f /tmp/autonomous-testing.flag" 2>/dev/null; then
  exit 0
fi
# 2. newest campaign receipt dir on the head
NEW=$(ssh -o BatchMode=yes "$SSH_TARGET" "ls -td ~/$PROJ/results/campaign-* 2>/dev/null | head -1") || exit 0
[ -n "$NEW" ] || exit 0
[ "$(basename "$NEW")" = "$(basename "$(cat "$MARKER" 2>/dev/null)")" ] && exit 0   # already poked about this one
# only completed campaigns wake the agent (the wrapper's last log line is " done.")
if ! ssh -o BatchMode=yes "$SSH_TARGET" "tail -1 '$NEW/campaign.log' 2>/dev/null | grep -q ' done\.'" 2>/dev/null; then
  exit 0   # still running (or crashed hard) — nothing to read yet
fi
# 3. the agent's session in this project (newest first)
SID=$(curl -s -m 10 -u "$AUTH" "$API/session?directory=$PWD&order=desc" \
      | python3 -c "import json,sys; d=json.load(sys.stdin); r=d.get('data') or []; print(r[0]['id'] if r else '')" 2>/dev/null) || true
[ -n "$SID" ] || { echo "$(date '+%F %T') no session found in $PWD" >> "$LOG"; exit 1; }
# 4. the poke (durable prompt inbox — the agent loop runs once the backend serves)
MSG="[autonomous poke] A testing window finished on the head: $NEW (console /tmp/autonomous.console). Production is back up. Read that campaign's receipts and the console tail, digest per AGENTS.md the same sitting (LAUNCH.md append, the ledger, anchors/chart when a qualified fact changed), then continue the perf plan (docs/PERF_PLAN.md next step)."
RC=$(curl -s -m 20 -u "$AUTH" -X POST "$API/session/$SID/prompt" -H 'Content-Type: application/json' \
     -d "$(python3 -c 'import json,sys; print(json.dumps({"text": sys.argv[1]}))' "$MSG")" \
     -o /dev/null -w "%{http_code}") || RC=000
if [ "$RC" = "200" ]; then
  echo "$NEW" > "$MARKER"
  echo "$(date '+%F %T') poked session $SID about $NEW" >> "$LOG"
else
  echo "$(date '+%F %T') poke POST rc=$RC — will retry next tick" >> "$LOG"
fi
