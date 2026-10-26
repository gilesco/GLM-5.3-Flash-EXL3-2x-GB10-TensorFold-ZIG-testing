#!/usr/bin/env bash
# Run the tool-eval-bench matrix against this stack (on the head host).
#
# Each configuration restarts the server (interrupting clients), waits for /health, then runs
# the bench with IDENTICAL options every time (docs/TOOL_EVAL_TESTING.md). Results land in
# results/<run-id>/<label>/. Usage:
#
#   scripts/run-bench-matrix.sh                    # the current search set (six configurations)
#   scripts/run-bench-matrix.sh baseline recall    # a subset (any label below)
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
set_config top4      4 0 "PATCHES=glm-topk4b.patch MODEL_REVISION=top4cfg"  # the AZ Experiment 1 serve (real topk=4; glm-topk4b carries the reference-shape change)
set_config azexl3   4 0 "PATCHES=glm-topk4b.patch MODEL_ID=local/GLM-5.3-Flash-AZ-EXL3 MODEL_REVISION=main"  # the AZ retrained selector (the requant output)
set_config focus    4 0 "MODEL_ID=AxonZeta/GLM-5.3-Flash-AZ-Focus-EXL3 MODEL_REVISION=d6922a95301c2ec274d96fb8ddb81f32f7b7f9be DENSE_BITS=4 HEAD_BITS=4 KV_CACHE=fp8 CONTEXT=524288"  # AxonZeta's reasoning-efficiency EXL3 (mcg-coded, integer bits; topk=8 by design — no patch)
set_config pdl       4 0 "PDL=1"               # + programmatic dependent launch on the mm kernels
set_config dense8    4 0 "DENSE_BITS=8 HEAD_BITS=8"   # + dense/head packed to 8 bits at load
set_config dense6    4 0 "DENSE_BITS=6 HEAD_BITS=6"   # + packed to 6 bits (dense8 beat baseline by ~13%)
set_config dense4    4 0 "DENSE_BITS=4 HEAD_BITS=4"   # + packed to 4 bits (quality check matters here)
set_config parallel6 6 0                       # + two more lanes (6 x 0.71 = 4.3 GiB of streams, EXL3's memory model)
set_config parallel8 8 0                       # + four more lanes (5.7 GiB of streams)
set_config recall    4 1                       # + --recall (drafts copy from previously-read text)
set_config recall6   6 1                       # both

# --- the 2026-10-22 search set (after the v1.1.2 + RoCE matrix; see TOOL_EVAL_TESTING.md) ---
set_config roce0       4 0 "TF_ROCE=0"
#   attribution control: the engine's RoCE gather off (NCCL's RDMA transport keeps running
#   via NCCL_IB_HCA) — splits the 1.1.2 engine gain from the transport gain.
set_config chan4       4 0 "NCCL_PASSTHROUGH=1 NCCL_MIN_NCHANNELS=4 NCCL_MAX_NCHANNELS=4"
#   + the measured NCCL channel set on the now-working RoCE path (jayleaton W19: prefill +1.5%,
#   4 MiB all-gather 322 -> 142 us; never tested here because the transport was TCP before).
set_config dense4pdl   4 0 "DENSE_BITS=4 HEAD_BITS=4 PDL=1"
#   the leader + PDL (PDL was +3.6 solo on baseline, worse at depth — worth one cell on dense4).
set_config dense4chan4 4 0 "DENSE_BITS=4 HEAD_BITS=4 NCCL_PASSTHROUGH=1 NCCL_MIN_NCHANNELS=4 NCCL_MAX_NCHANNELS=4"
#   the leader + the channel set (the combination candidate if both cells win).
set_config dense4h8    4 0 "DENSE_BITS=4 HEAD_BITS=8"
#   mixed bits: 4-bit dense linears, 8-bit head — a quality/speed compromise point (the head
#   is the sampling path; denser bits there may buy back exactness at little speed cost).
set_config fp8         4 0 "KV_CACHE=fp8"
#   + v1.1.2's prepared fp8 latent cache: halves KV bytes (memory model, big windows) —
#   speed unknown on EXL3, quality unqualified upstream: bench-only until gated.
set_config dense4fp8   4 0 "DENSE_BITS=4 HEAD_BITS=4 KV_CACHE=fp8 CONTEXT=524288"
#   the leader + fp8 caches at the goal window — the ladder candidate: dense4's speed with
#   fp8's halved per-token KV. The 2026-10-23 campaign ran exactly this (its shell happened
#   to export CONTEXT=524288); now explicit so the cell can't silently run at 40960.

# --- production-parity cells (the real-world anchor) ----------------------------------------
# Production is the -tip fork on the head (this site's path: docs/SITE.md, not in git):
# Python engine 0.6.6, same EXL3 checkpoint, DENSE=fp8 (dense linears fp8), KV=fp8 (DSA latent
# e4m3), COMM=roce, CONTEXT=524288, DFlash2 drafter, TP=2; admission 87.45/101.21 GiB, cache
# slots 524304. These cells mirror it knob-for-knob on the native engine so the bench numbers
# compare like with like (bf16-KV 40K cells are NOT production parity).
set_config parity      4 0 "DENSE_BITS=8 HEAD_BITS=8 KV_CACHE=fp8 CONTEXT=524288"
#   the knob-for-knob mirror of production: fp8-dense-equivalent bits, fp8 latent cache,
#   the 512K window, same drafter. The real-world comparison cell.
set_config parity4fp8  4 0 "DENSE_BITS=4 HEAD_BITS=4 KV_CACHE=fp8 CONTEXT=524288"
#   parity's window with the dense4 compression — the ladder candidate at full window.
set_config nodraft     4 0 "DENSE_BITS=8 HEAD_BITS=8 KV_CACHE=fp8 CONTEXT=524288 DRAFTS=0"
#   parity minus the drafter — quantifies what DFlash2 buys on the native engine (the
#   forum's spec-decode thread claims it is the difference between 14 and 23 tok/s).

SELECTED=("$@")
(( ${#SELECTED[@]} )) || SELECTED=(roce0 chan4 dense4pdl dense4chan4 dense4h8 fp8)

for label in "${SELECTED[@]}"; do
  [[ -n "${CFG_PARALLEL[$label]:-}" ]] || die "unknown configuration: $label (baseline | pdl | dense4 | dense6 | dense8 | parallel6 | parallel8 | recall | recall6 | roce0 | chan4 | dense4pdl | dense4chan4 | dense4h8 | fp8 | dense4fp8 | parity | parity4fp8 | nodraft)"
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
  PYTHONUNBUFFERED=1 "$BENCH" "${BENCH_ARGS[@]}" 2>&1 | tee "$OUT/$label.bench.log" | tail -25

  # The score and the perf lines, for the summary table.
  grep -hoE "[0-9]+/[0-9]+ (scenarios|passed)|[0-9]+/[0-9]+ points|pp tok/s|tg tok/s|TTFT[^,]*" \
    "$OUT/$label.bench.log" | sort -u | sed 's/^/- /' >> "$OUT/SUMMARY.md" || true
  # Recall carries state across requests; clear it so the next configuration starts clean.
  curl -s -X DELETE "http://127.0.0.1:$PORT/v1/recall" >/dev/null 2>&1 || true
done

echo; echo "Matrix complete. Results: $OUT (summary: $OUT/SUMMARY.md)"
tail -30 "$OUT/SUMMARY.md"
