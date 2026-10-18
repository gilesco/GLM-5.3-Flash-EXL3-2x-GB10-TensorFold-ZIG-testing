#!/usr/bin/env bash
# Run the tool-eval-bench matrix against this stack (on the head host).
#
# Each configuration restarts the server (interrupting clients), waits for /health, then runs
# the bench with IDENTICAL options every time (docs/TOOL_EVAL_TESTING.md). Results land in
# results/<run-id>/<label>/. Usage:
#
#   scripts/run-bench-matrix.sh                    # all four configurations
#   scripts/run-bench-matrix.sh baseline recall    # a subset
#   NO_BENCH=1 scripts/run-bench-matrix.sh recall6 # restart + serve only (a manual bench run follows)
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
source ./config.sh
command -v docker >/dev/null || die "docker is required (run this on the head Spark)"
BENCH="$HOME/tool-eval-bench/.venv/bin/tool-eval-bench"
[[ -x "$BENCH" ]] || die "tool-eval-bench not found at $BENCH"

RUN_ID="run-$(date +%Y%m%d-%H%M%S)"
OUT="results/$RUN_ID"; mkdir -p "$OUT"
: > "$OUT/SUMMARY.md"
echo "# tool-eval-bench matrix $RUN_ID" >> "$OUT/SUMMARY.md"
echo "bench options are fixed across every configuration; the server configuration is the variable." >> "$OUT/SUMMARY.md"

# The configurations: label = the optimization under test (one variable changed per config).
# The bench options never change. $4 (optional) is extra env for the restart, space-separated
# KEY=VALUE pairs (see config.sh's knobs: DENSE_BITS, HEAD_BITS, PDL).
declare -A CFG_PARALLEL CFG_RECALL CFG_ENV
set_config() { CFG_PARALLEL[$1]=$2; CFG_RECALL[$1]=$3; CFG_ENV[$1]=${4:-}; }
set_config baseline  4 0                       # today's default
set_config pdl       4 0 "PDL=1"               # + programmatic dependent launch on the mm kernels
set_config dense8    4 0 "DENSE_BITS=8 HEAD_BITS=8"   # + dense/head packed to 8 bits at load
set_config parallel6 6 0                       # + two more lanes (6 x 0.71 = 4.3 GiB of streams, EXL3's memory model)
set_config parallel8 8 0                       # + four more lanes (5.7 GiB of streams)
set_config recall    4 1                       # + --recall (drafts copy from previously-read text)
set_config recall6   6 1                       # both

SELECTED=("$@")
(( ${#SELECTED[@]} )) || SELECTED=(baseline pdl dense8 parallel6 parallel8 recall recall6)

for label in "${SELECTED[@]}"; do
  [[ -n "${CFG_PARALLEL[$label]:-}" ]] || die "unknown configuration: $label (baseline | pdl | dense8 | parallel6 | parallel8 | recall | recall6)"
  echo "=== [$label] restart: PARALLEL=${CFG_PARALLEL[$label]} RECALL=${CFG_RECALL[$label]} ${CFG_ENV[$label]:-} ==="
  ./scripts/stop.sh < /dev/null || true
  read -r -a _envpairs <<< "${CFG_ENV[$label]:-}"
  env "${_envpairs[@]}" PARALLEL="${CFG_PARALLEL[$label]}" RECALL="${CFG_RECALL[$label]}" \
    ./scripts/start.sh < /dev/null

  docker logs "$CONTAINER_NAME" 2>&1 | grep -m1 "CUDA sm_121" > "$OUT/$label.server.log" || true
  echo "## $label" >> "$OUT/SUMMARY.md"
  echo "- server: $(cat "$OUT/$label.server.log")" >> "$OUT/SUMMARY.md"

  if [[ "${NO_BENCH:-0}" == 1 ]]; then
    echo "- (NO_BENCH: no bench run; server left up)" >> "$OUT/SUMMARY.md"
    continue
  fi

  echo "=== [$label] bench: $(date) ==="
  # Phased (docs/TOOL_EVAL_TESTING.md): quick performance first; the scenario suites run
  # only on the configuration the quick phase validates. Options otherwise identical.
  BENCH_ARGS=(run
    --base-url "http://127.0.0.1:$PORT/v1"
    --model "$SERVED_NAME"
    --format openai
    --tokenizer "$MODEL_ID"   # the checkpoint's own tokenizer files (offline: read from the cache)
    --seed 42
    --perf
    --parallel 4
    --timeout 240
    --no-live
    --label "$label"
    --output-dir "$OUT/$label"
    --json-file "$OUT/$label/result.json")
  case "${PHASE:-perf}" in
    perf)     BENCH_ARGS+=(--perf-only) ;;            # minutes: llama-benchy pp/tg only
    tools)    : ;;                                     # the 69 standard scenarios + perf
    hardmode) BENCH_ARGS+=(--hardmode) ;;              # + the 28 Hard-Mode scenarios
    *) die "PHASE is perf | tools | hardmode" ;;
  esac
  "$BENCH" "${BENCH_ARGS[@]}" 2>&1 | tee "$OUT/$label.bench.log" | tail -25

  # The score and the perf lines, for the summary table.
  grep -hoE "[0-9]+/[0-9]+ (scenarios|passed)|[0-9]+/[0-9]+ points|pp tok/s|tg tok/s|TTFT[^,]*" \
    "$OUT/$label.bench.log" | sort -u | sed 's/^/- /' >> "$OUT/SUMMARY.md" || true
  # Recall carries state across requests; clear it so the next configuration starts clean.
  curl -s -X DELETE "http://127.0.0.1:$PORT/v1/recall" >/dev/null 2>&1 || true
done

echo; echo "Matrix complete. Results: $OUT (summary: $OUT/SUMMARY.md)"
tail -30 "$OUT/SUMMARY.md"
