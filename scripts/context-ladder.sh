#!/usr/bin/env bash
# Climb the context ladder on the head Spark: for each window size, restart the server at
# PARALLEL=1 with the long-context memory knobs, probe long-context retrieval with the
# bench's needle test, and record the engine's own memory accounting. A rung the admission
# refuses is a recorded outcome, not a failure (docs/CONTEXT_500K_PLAN.md).
#
#   scripts/context-ladder.sh                          # 131072 262144 400000 500000
#   scripts/context-ladder.sh 500000                   # one rung
#   RESERVE_GIB=8 PROMPT_CACHE_GIB=1.5 PARALLEL=1 ...  # the knobs the plan computes
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
source ./config.sh
command -v docker >/dev/null || die "docker is required (run this on the head Spark)"
BENCH="$HOME/tool-eval-bench/.venv/bin/tool-eval-bench"
[[ -x "$BENCH" ]] || die "tool-eval-bench not found at $BENCH"

RESERVE_GIB="${RESERVE_GIB:-8}"            # -> TENSORFOLD_MEMORY_RESERVE_GIB (default here: 8)
PROMPT_CACHE_GIB="${PROMPT_CACHE_GIB:-1.5}"  # --prompt-cache-gib (default here: 1.5)
LADDER_PARALLEL="${LADDER_PARALLEL:-1}"    # the 500K math fits one stream (bf16 KV)

RUN_ID="ctx-$(date +%Y%m%d-%H%M%S)"
OUT="results/$RUN_ID"; mkdir -p "$OUT"
SUMMARY="$OUT/SUMMARY.md"
echo "# context ladder $RUN_ID (reserve ${RESERVE_GIB} GiB, prompt cache ${PROMPT_CACHE_GIB} GiB, parallel $LADDER_PARALLEL)" > "$SUMMARY"

RUNGS=("$@")
(( ${#RUNGS[@]} )) || RUNGS=(131072 262144 400000 500000)

for ctx in "${RUNGS[@]}"; do
  echo "=== rung $ctx ==="
  ./scripts/stop.sh < /dev/null || true
  if ! CONTEXT="$ctx" PARALLEL="$LADDER_PARALLEL" \
       TENSORFOLD_MEMORY_RESERVE_GIB="$RESERVE_GIB" PROMPT_CACHE_GIB="$PROMPT_CACHE_GIB" \
       ./scripts/start.sh < /dev/null; then
    echo "## $ctx" >> "$SUMMARY"; echo "- REFUSED by admission (see /tmp or the start log)" >> "$SUMMARY"
    continue
  fi
  sleep 5
  docker logs "$CONTAINER_NAME" 2>&1 | grep -m1 "CUDA sm_121" > "$OUT/$ctx.server.log" || true
  HEALTH=$(curl -s -m 5 "http://127.0.0.1:$PORT/health")
  echo "## $ctx" >> "$SUMMARY"
  echo "- server: $(cat "$OUT/$ctx.server.log")" >> "$SUMMARY"
  echo "- health: $HEALTH" >> "$SUMMARY"

  # Long-context retrieval + decode rate at this window: one needle buried at ~half depth.
  "$BENCH" run --needle-only --base-url "http://127.0.0.1:$PORT/v1" --model "$SERVED_NAME" \
    --format openai --seed 42 --needle-lengths "$ctx" --needle-depths 2 --no-live \
    --timeout 600 --label "ctx-$ctx" --output-dir "$OUT/needle-$ctx" \
    --json-file "$OUT/needle-$ctx/result.json" 2>&1 | tee "$OUT/$ctx.needle.log" \
    | grep -iE "needle|tok/s|ttft|pass|fail" | sed 's/^/- /' >> "$SUMMARY" || \
    echo "- needle run incomplete (see $OUT/$ctx.needle.log)" >> "$SUMMARY"
done

echo; echo "Ladder complete: $SUMMARY"
cat "$SUMMARY"
