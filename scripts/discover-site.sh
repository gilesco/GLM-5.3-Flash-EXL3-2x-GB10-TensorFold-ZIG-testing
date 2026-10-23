#!/usr/bin/env bash
# Discover this site's configuration and write scripts/local.sh — or, with
# REDISCOVER=1, update the site keys of an existing one (a timestamped backup is
# kept and every other entry is left alone). prepare.sh runs this automatically
# when local.sh is missing; it can also be run on its own:
#   ./scripts/discover-site.sh               # generate local.sh if missing
#   REDISCOVER=1 ./scripts/discover-site.sh  # re-detect and update the site keys
# What it detects:
#   - the cabled CX7 HCAs and their netdevs (only links that are actually up),
#   - the rails' IPv4 addresses; unassigned rails are reported with the exact
#     `ip addr add` commands for both hosts (discovery does not assign addresses
#     itself — applying them is a root decision on each host),
#   - the worker: ssh as the same user to the head's rail subnet's .2 (the
#     point-to-point convention), then reads the worker's rail addresses there.
#     Override the guess with WORKER_TARGET=user@host on the command line.
# Only five keys are ever written: WORKER, HEAD_FABRIC_ADDR, WORKER_FABRIC_ADDR,
# NCCL_SOCKET_IFNAME, NCCL_IB_HCA.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."

log() { printf '[discover-site] %s\n' "$*"; }
die() { printf '[discover-site] ERROR: %s\n' "$*" >&2; exit 2; }

LOCAL_SH=scripts/local.sh
UPDATING=0
if [[ -f $LOCAL_SH ]]; then
  if [[ "${REDISCOVER:-0}" != 1 ]]; then
    log "$LOCAL_SH exists — nothing to discover (REDISCOVER=1 re-detects)"
    exit 0
  fi
  UPDATING=1
  cp "$LOCAL_SH" "$LOCAL_SH.bak.$(date +%Y%m%d-%H%M%S)"
  log "backed up the previous $LOCAL_SH"
fi

# --- 1. the cabled rails here (HCA + netdev pairs with carrier up), sorted so the
#        head's and worker's first rail are picked the same way on both hosts ------
cabled() {
  local ibdev netdev
  for ibdev in /sys/class/infiniband/*; do
    [[ -d $ibdev/device/net ]] || continue
    for netdev in $(ls "$ibdev/device/net" 2>/dev/null); do
      [[ $(cat "/sys/class/net/$netdev/carrier" 2>/dev/null || echo 0) == 1 ]] \
        && echo "$(basename "$ibdev") $netdev"
    done
  done | sort
}
mapfile -t LOCAL_PAIRS < <(cabled)
((${#LOCAL_PAIRS[@]})) || die "no cabled RoCE port found (a ConnectX-7 link that is up). Cable the pair first, or fill scripts/local.sh in from local.sh.example."
HCAS=(); NETDEVS=()
for p in "${LOCAL_PAIRS[@]}"; do HCAS+=("${p%% *}"); NETDEVS+=("${p##* }"); done
log "cabled rails here: ${NETDEVS[*]} (HCAs: ${HCAS[*]})"

# --- 2. the rails' addresses -----------------------------------------------------
addr_of() { ip -4 -o addr show dev "$1" scope global 2>/dev/null | awk 'NR==1{print $4}' | cut -d/ -f1; }
HEAD1=$(addr_of "${NETDEVS[0]}")
if [[ -z $HEAD1 ]]; then
  cat >&2 <<EOF
[discover-site] The first cabled rail (${NETDEVS[0]}) has no IPv4 address yet.
[discover-site] Assign one on EACH host (root), then re-run:
    here:      sudo ip addr add 192.168.10.1/24 dev ${NETDEVS[0]}
    the worker: sudo ip addr add 192.168.10.2/24 dev ${NETDEVS[0]}
  (any private /24 works — pick one per cable; local.sh.example has the walkthrough)
EOF
  exit 2
fi
WORKER_TARGET=${WORKER_TARGET:-}
if [[ -z $WORKER_TARGET && $UPDATING == 1 ]]; then
  WORKER_TARGET=$(grep -m1 '^WORKER=' "$LOCAL_SH" | cut -d= -f2- || true)
fi
if [[ -n $WORKER_TARGET ]] && ! ssh -o BatchMode=yes -o ConnectTimeout=5 "$WORKER_TARGET" true 2>/dev/null; then
  log "the existing WORKER=$WORKER_TARGET does not answer; falling back to detection"
  WORKER_TARGET=""
fi
if [[ -z $WORKER_TARGET ]]; then
  WORKER_TARGET="$USER@${HEAD1%.*}.2"
  log "probing the point-to-point convention: ssh $WORKER_TARGET (override with WORKER_TARGET=...)"
fi
ssh -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new \
    "$WORKER_TARGET" '[[ -d /sys/class/infiniband ]]' 2>/dev/null \
  || die "cannot ssh to $WORKER_TARGET (key-based). Set up the key, or write scripts/local.sh by hand (local.sh.example) and re-run ./scripts/prepare.sh."

# --- 3. the worker's rails (same platform: expect the same cabled netdev names) ---
mapfile -t WORKER_PAIRS < <(ssh -o BatchMode=yes "$WORKER_TARGET" 'bash -s' < <(declare -f cabled; echo cabled))
((${#WORKER_PAIRS[@]})) || die "could not detect the worker's cabled rails over ssh (rdma tools missing there?)"
[[ "${WORKER_PAIRS[0]##* }" == "${NETDEVS[0]}" ]] \
  || die "the worker's first cabled netdev is '${WORKER_PAIRS[0]##* }' but the head's is '${NETDEVS[0]}' — the rails' naming should match on two GB10s; wire the config by hand (local.sh.example)."
worker_addr_of() {
  ssh -o BatchMode=yes "$WORKER_TARGET" "ip -4 -o addr show dev '$1' scope global 2>/dev/null" | awk 'NR==1{print $4}' | cut -d/ -f1
}
WORKER1=$(worker_addr_of "${NETDEVS[0]}")
[[ -n $WORKER1 ]] || { cat >&2 <<EOF
[discover-site] The worker's rail (${NETDEVS[0]}) has no IPv4 address yet — run on the worker:
    sudo ip addr add $(echo "$HEAD1" | sed 's/\.[0-9]*$/.2/')/24 dev ${NETDEVS[0]}
  then re-run.
EOF
  exit 2; }

# --- 4. the checks ---------------------------------------------------------------
ping -c1 -W2 "$WORKER1" >/dev/null 2>&1 \
  || log "WARN: the worker's rail address $WORKER1 does not answer ping — if ICMP is filtered this is fine (ssh reached it); prepare's preflight re-checks the link either way"

# --- 5. write the five keys ------------------------------------------------------
if ((UPDATING)); then
  write() { local k=$1 v=$2
    if grep -qE "^$k=" "$LOCAL_SH"; then sed -i "s|^$k=.*|$k=$v|" "$LOCAL_SH"
    else printf '%s=%s\n' "$k" "$v" >> "$LOCAL_SH"; fi; }
else
  cat > "$LOCAL_SH" <<EOF
# scripts/local.sh — this site's identity (git-ignored).
# Generated by scripts/discover-site.sh; REDISCOVER=1 ./scripts/prepare.sh re-detects
# and updates these keys, keeping any other entries you add. The full walkthrough:
# scripts/local.sh.example.
EOF
  write() { printf '%s=%s\n' "$1" "$2" >> "$LOCAL_SH"; }
fi
write WORKER "$WORKER_TARGET"
write HEAD_FABRIC_ADDR "$HEAD1"
write WORKER_FABRIC_ADDR "$WORKER1"
write NCCL_SOCKET_IFNAME "$(IFS=,; echo "${NETDEVS[*]}")"
write NCCL_IB_HCA "$(IFS=,; echo "${HCAS[*]}")"

log "wrote $LOCAL_SH:"
grep -E '^(WORKER|HEAD_FABRIC_ADDR|WORKER_FABRIC_ADDR|NCCL_SOCKET_IFNAME|NCCL_IB_HCA)=' "$LOCAL_SH" | sed 's/^/    /'
log "next: ./scripts/prepare.sh"
