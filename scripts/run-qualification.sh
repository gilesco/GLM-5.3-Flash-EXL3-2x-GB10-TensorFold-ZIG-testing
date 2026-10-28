#!/usr/bin/env bash
# The 0.6.x -> 1.1.x replication campaign: one blank run on the head Spark that
# tests everything needed to work out the way forward for this repo.
#
#   bash scripts/run-qualification.sh            # everything (~3 h)
#   PHASES="parity kvqual" bash scripts/run-qualification.sh   # a subset
#   FORCE=1 ...                                  # re-run even completed phases
#
# What it does, in order (each phase lands a receipt in results/campaign-<id>/):
#   preflight  guards (production untouched), marker, RoCE, CPU clocks, disk
#   reference  one greedy probe captured from production 0.6.6 if it is still up
#              (read-only, a single request) — the cross-engine output reference
#   prodqual   validate production 0.6.6 with the same instruments while it is up:
#              determinism / repeat / concurrent / seeded probes (read-only — the
#              server is never restarted or modified) + the same-harness tool-eval
#              and perf runs, giving the native engine's 92 its like-for-like
#              production baseline. Prompts to stop production only if GPU phases
#              are also selected.
#   parity     the knob-for-knob mirror of production on this engine:
#              DENSE_BITS=8 HEAD_BITS=8 KV_CACHE=fp8 CONTEXT=524288 + DFlash2,
#              benched with the fixed matrix options (PARALLEL=4, fallback 2)
#   candidate  dense4fp8 at the same window — the ladder candidate's cell
#   kvqual     VALIDATION §2 gates on dense8+fp8 KV at 512K, content form
#              (determinism, drafted=plain, resumed=fresh, concurrent=solo,
#              sampled-seed determinism; the native engine is byte-nondeterministic,
#              so gates compare normalized content — VALIDATION §2 receipt)
#   kvqual4    the same gates on the ladder candidate dense4fp8 (its own receipt)
#   throughput perf-only bench tables (llama-benchy pp/tg) for parity + dense4fp8
#   kvqual4P   the gates on PAGED bf16 KV (TF_KV_PAGES=on): the KV-pages branch pages the
#              bf16 KV path ONLY — fp8 KV and paging are mutually exclusive there (the
#              engine exits on the combination; 87e4417e receipt, campaign-20261023-165319)
#   throughputP the perf-only bench on paged bf16 KV (dense4, 512K window)
#   climbP     PARALLEL=5/6/8 under paging: pages allocate on demand, so admission
#              is about live paged usage, not streams × window
#              at 512K — the speed numbers the first campaign's full-eval cells lack
#   climb      dense4fp8 at 512K with PARALLEL=6 then 8: does admission take it,
#              and what does aggregate decode do (refusal = recorded outcome)
#   depth      the depth gate the forum thread taught us: seeded fact+needle
#              probes at 8k/64k/192k/384k on the fp8-KV 512K config (production's
#              93.9% Cyrillic score is the community bar)
#   cliff      the concurrency cliff: 4-stream decode with vs without one
#              structured-output request in the batch (the draft-truncation
#              mechanism MaTeZZ documented on the vLLM stacks)
#   prefill    corrected probe 3: a ~14k-token prompt's prefill rate
#   rung       the 524288x1 ladder rung: admission line + a depth probe
#   summary    CAMPAIGN.md: verdict per gate + the way-forward recommendations
#
# Safety: this script never touches production (glm53-flash-tf). If production is
# running it captures the reference probe and then STOPS and asks you to stop
# production yourself; with PROMPT=0 it refuses and exits instead.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
source ./config.sh
command -v docker >/dev/null || die "docker is required (run this on the head Spark)"
command -v python3 >/dev/null || die "python3 is required"
BENCH="$HOME/tool-eval-bench/.venv/bin/tool-eval-bench"
[[ -x "$BENCH" ]] || die "tool-eval-bench not found at $BENCH"

RUN_ID="campaign-$(date +%Y%m%d-%H%M%S)"
OUT="results/$RUN_ID"; mkdir -p "$OUT"
LOG="$OUT/campaign.log"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
phase_done() { [[ -f "$OUT/.$1.done" ]] && [[ "${FORCE:-0}" != 1 ]]; }

PHASES="${PHASES:-preflight reference prodqual parity candidate kvqual kvqual4 throughput climb kvqual4P throughputP climbP knobs duel confirm depth cliff prefill rung summary}"
want() { case " $PHASES " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }
# Do the selected phases need the testing stack on the GPUs? (everything except the
# read-only production passes and the summary — those run against production AS IS)
NEEDS_GPU=0
for _p in parity candidate kvqual kvqual4 throughput climb kvqual4P throughputP climbP knobs duel confirm prof4 top4 routerks2 routerpart top4qual azqual focusqual hmtrials settingsab markspatch v118 indexdepth cachegib ixgates gibdisc spillgib depth cliff prefill rung; do
  want "$_p" && NEEDS_GPU=1
done

# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------
prod_running() {   # production containers up here and on the worker?
  docker ps --format '{{.Names}}' | grep -qx 'glm53-flash-tf' && return 0
  ssh -o BatchMode=yes -o ConnectTimeout=5 "$WORKER" \
    'docker ps --format "{{.Names}}" | grep -qx glm53-flash-tf' 2>/dev/null && return 0
  return 1
}

restart() {        # restart the testing stack with env pairs: restart K=V ...
  local kv rc=0
  if prod_running; then die "production (glm53-flash-tf) is running — stop it first; this script never touches it"; fi
  log "restarting the testing stack (${*:-defaults})"
  ./scripts/stop.sh < /dev/null >/dev/null 2>&1 || true
  if [[ $# -gt 0 ]]; then env "$@" ./scripts/start.sh < /dev/null >>"$LOG" 2>&1 || rc=$?
  else ./scripts/start.sh < /dev/null >>"$LOG" 2>&1 || rc=$?; fi
  log "serving: $(serve_line)"
  log "$(roce_state)"
  # propagate start.sh's exit code even when called in an if-context (set -e is
  # suppressed there) — the climb phase depends on refusals being visible
  return $rc
}

restart_fb() {     # restart with PARALLEL=4, falling back to PARALLEL=2 on refusal
  local retry=() kv
  if restart "$@"; then return 0; fi
  log "start failed — retrying with PARALLEL=2"
  for kv in "$@"; do [[ "$kv" == PARALLEL=* ]] || retry+=("$kv"); done
  restart "${retry[@]}" PARALLEL=2
}

roce_state() {     # did the RoCE gather come up on this launch?
  if docker logs "$CONTAINER_NAME" 2>&1 | grep -q "VerbsUnavailable\|NCCL carries every gather"; then
    echo "RoCE DOWN (fallback warning present)"
  elif docker logs "$CONTAINER_NAME" 2>&1 | grep -q "RoCE all-gather"; then
    echo "RoCE up: $(docker logs "$CONTAINER_NAME" 2>&1 | grep -m1 'RoCE all-gather' | sed 's/^info: //')"
  else
    echo "RoCE: no gather lines in the log"
  fi
}

# The duel's five-probe arm, shared by the duel phase and the session phases
# (prof4/top4) so every instrument reads the same way everywhere.
DUEL_PROMPT="Write a 300-word technical summary of a harbor crane's maintenance cycle. Do not use any formatting."

duel_arm() {   # $1 engine tag ("prod"|"ours"|<cell>), $2 the model id to address
  local tag="$1" model="$2" H
  H="http://127.0.0.1:$PORT"
  # the probe set, generated once per tag
  SERVED_MODEL="$model" PROMPT="$DUEL_PROMPT" FILL_WORDS="$FILL_WORDS" python3 - "$OUT" "$tag" <<'PYEOF'
import json, os, sys
out, tag, prompt = sys.argv[1], sys.argv[2], os.environ["PROMPT"]
model = os.environ["SERVED_MODEL"]
base = {"model": model, "temperature": 0}
for d in (0, 4096, 8192):
    if d == 0:
        body = dict(base, messages=[{"role": "user", "content": prompt}])
    else:
        words = os.environ["FILL_WORDS"].split(); n = d // 4
        fill = " ".join(w if i % 11 == 0 else w + f"-{i % 89}" for i, w in
                        ((i, words[i % len(words)]) for i in range(n)))
        body = dict(base, messages=[{"role": "user", "content": fill + "\n\n" + prompt}])
    json.dump(dict(body, max_tokens=2048), open(f"{out}/duel-{tag}-d{d}-2048.json", "w"))
    json.dump(dict(body, max_tokens=1),   open(f"{out}/duel-{tag}-d{d}-1.json", "w"))
PYEOF
    curl -s -m 10 "$H/health" > "$OUT/duel-$tag-health-before.json" || true
    # instrument 1+5: the solo twins (twice — stability), then c2 and c4 concurrently
    for r in a b; do
      S=$(date +%s.%N)
      curl -s -m 300 "$H/v1/chat/completions" -H 'Content-Type: application/json' \
        --data @"$OUT/duel-$tag-d0-2048.json" > "$OUT/duel-$tag-solo-$r.json" || true
      E=$(date +%s.%N); echo "$r $S $E" >> "$OUT/duel-$tag-walls.txt"
    done
    for c in 2 4; do
      S=$(date +%s.%N)
      for i in $(seq 1 "$c"); do { curl -s -m 300 "$H/v1/chat/completions" -H 'Content-Type: application/json' \
        --data @"$OUT/duel-$tag-d0-2048.json" > "$OUT/duel-$tag-c$c-$i.json" || true; } & done; wait
      E=$(date +%s.%N); echo "c$c $S $E" >> "$OUT/duel-$tag-walls.txt"
    done
    # instrument 2: the streamed twin (per-chunk rhythm)
    SERVED_MODEL="$model" python3 - "$OUT" "$tag" "$H" "$DUEL_PROMPT" <<'PYEOF'
import json, os, sys, time, urllib.request
out, tag, url, prompt = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
body = json.load(open(f"{out}/duel-{tag}-d0-2048.json")); body["stream"] = True
req = urllib.request.Request(url + "/v1/chat/completions",
                             data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json"})
t0 = time.monotonic(); gaps = []; prev = None; first = None
try:
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            s = line.strip()
            if not s.startswith(b"data:") or s == b"data: [DONE]":
                continue
            now = time.monotonic()
            if prev is None: first = now - t0
            else: gaps.append(now - prev)
            prev = now
except Exception as e:
    print(f"- duel-{tag} stream failed: {e}"); raise SystemExit(0)
gaps.sort()
pick = lambda p: gaps[min(len(gaps) - 1, int(len(gaps) * p))] if gaps else 0
json.dump({"chunks": len(gaps) + 1, "first_chunk_ms": (first or 0) * 1000,
           "mean_itl_ms": sum(gaps) / len(gaps) * 1000 if gaps else 0,
           "p50_itl_ms": pick(.5) * 1000, "p95_itl_ms": pick(.95) * 1000},
          open(f"{out}/duel-{tag}-itl.json", "w"))
PYEOF
    # instrument 3: the prefill slope (max_tokens=1, three depths)
    for d in 0 4096 8192; do
      S=$(date +%s.%N)
      curl -s -m 300 "$H/v1/chat/completions" -H 'Content-Type: application/json' \
        --data @"$OUT/duel-$tag-d$d-1.json" > "$OUT/duel-$tag-prefill-$d.json" || true
      E=$(date +%s.%N); echo "p$d $S $E" >> "$OUT/duel-$tag-walls.txt"
    done
    curl -s -m 10 "$H/health" > "$OUT/duel-$tag-health-after.json" || true
    # the receipt block for this engine
    python3 - "$OUT" "$tag" >> "$P" <<'PYEOF'
import glob, json, sys
out, tag = sys.argv[1], sys.argv[2]
walls = {}
for line in open(f"{out}/duel-{tag}-walls.txt"):
    k, s, e = line.split(); walls[k] = float(e) - float(s)
rates = []
for r in ("a", "b"):
    try:
        d = json.load(open(f"{out}/duel-{tag}-solo-{r}.json"))
        rates.append(d["usage"]["completion_tokens"] / walls[r])
    except Exception: pass
if len(rates) == 2:
    print("- solo decode: %.1f / %.1f t/s (stability Δ %.1f%%)" % (rates[0], rates[1], abs(rates[0]-rates[1])/max(rates)*100))
for c in (2, 4):
    try:
        tot = sum(json.load(open(f))["usage"]["completion_tokens"]
                  for f in glob.glob(f"{out}/duel-{tag}-c{c}-*.json"))
        print("- c%d aggregate: %.1f t/s" % (c, tot / walls[f"c{c}"]))
    except Exception: pass
try:
    itl = json.load(open(f"{out}/duel-{tag}-itl.json"))
    print("- ITL per chunk: mean %.1f ms, p50 %.1f, p95 %.1f (%d chunks, first %.0f ms)" % (
        itl["mean_itl_ms"], itl["p50_itl_ms"], itl["p95_itl_ms"], itl["chunks"], itl["first_chunk_ms"]))
except Exception: pass
tt = {}
for d in (0, 4096, 8192):
    tt[d] = walls.get(f"p{d}", 0) * 1000
if all(tt.values()):
    print("- TTFT (max_tokens=1): d0 %.0f ms; d4096 %.0f; d8192 %.0f → slope %.3f ms/token, fixed overhead %.0f ms" % (
        tt[0], tt[4096], tt[8192], (tt[8192]-tt[0])/8192, tt[0]))
try:
    b = json.load(open(f"{out}/duel-{tag}-health-before.json")); a = json.load(open(f"{out}/duel-{tag}-health-after.json"))
    for k in ("drafted", "accepted", "rounds"):
        if isinstance(b, dict) and isinstance(a, dict) and k in b and k in a:
            d = a.get(k, 0) - b.get(k, 0)
            if d: print("- %s Δ: %d" % (k, d))
except Exception: pass
PYEOF
}

c4_reprobe() {     # $1 tag: 4 fresh concurrent 2048-token generations on the d0 body;
                   # prints the aggregate and leaves the engine's last 64-call prof
                   # window = this probe, so a following prof-line extraction reads c4
  local tag="$1" S E
  [[ -s "$OUT/duel-$tag-d0-2048.json" ]] || { echo "- c4 re-probe skipped: $OUT/duel-$tag-d0-2048.json missing (the duel arm did not run under this tag)"; return 0; }
  S=$(date +%s.%N)
  for i in 1 2 3 4; do { curl -s -m 300 "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' --data @"$OUT/duel-$tag-d0-2048.json" \
    > "$OUT/duel-$tag-c4b-$i.json" || true; } & done
  wait
  E=$(date +%s.%N)
  python3 - "$S" "$E" "$OUT" "$tag" <<'PYEOF'
import glob, json, sys
s, e, out, tag = float(sys.argv[1]), float(sys.argv[2]), sys.argv[3], sys.argv[4]
tot = sum(json.load(open(f))["usage"]["completion_tokens"] for f in glob.glob(f"{out}/duel-{tag}-c4b-*.json"))
print("- c4 re-probe (fresh, so the prof window below is this probe): %.1f t/s aggregate" % (tot / (e - s)))
PYEOF
}

clock_mhz() {      # delivered CPU clock (MaTeZZ's acpi_cppc probe), 0 if unreadable
  local host="$1" out snippet
  snippet=$(cat <<'EOF'
f=/sys/devices/system/cpu/cpu5/acpi_cppc/feedback_ctrs
[[ -e $f ]] || { echo 0; exit 0; }
taskset -c 5 timeout 4 sh -c 'while :; do :; done' &
sleep 1; a=$(cat "$f"); sleep 2; b=$(cat "$f"); wait
echo "$a $b" | tr -d 'a-z:' | awk '{m=int(1000*($4-$2)/($3-$1)); print (m>100 && m<6000)?m:0}'
EOF
)
  if [[ "$host" == local ]]; then out=$(bash -c "$snippet" 2>/dev/null) || out=0
  else out=$(ssh -o BatchMode=yes -o ConnectTimeout=5 "$host" bash -c "$snippet" 2>/dev/null) || out=0; fi
  echo "${out:-0}"
}

serve_line() {     # the sm_121 admission line of the running server, verbatim
  docker logs "$CONTAINER_NAME" 2>&1 | grep -m1 "CUDA sm_121" || echo "(no server line)"
}

bench_cell() {     # the fixed matrix bench options against the running server
  local label="$1" dest="$2" rc=0
  mkdir -p "$dest"
  PYTHONUNBUFFERED=1 "$BENCH" run \
    --base-url "http://127.0.0.1:$PORT/v1" \
    --model "$SERVED_NAME" \
    --format openai \
    --tokenizer "$MODEL_ID" \
    --seed 42 --perf --parallel 2 --timeout 240 --no-live \
    --label "$label" --output-dir "$dest" --json-file "$dest/result.json" \
    2>&1 | tee "$dest/bench.log" | tail -20 || rc=$?
  # a non-zero bench exit must not kill the campaign (set -e) — record it and move on
  [[ $rc -ne 0 ]] && echo "- NOTE: bench exited $rc (its receipts may still be complete)" || true
  return 0
}

chat() {           # one chat completion; $1 = request file, optional $2 = curl timeout
  curl -s -m "${2:-900}" "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' --data @"$1"
}

bench_perf() {     # $1 label, $2 dest, $3 client parallel (default 4) — the matrix's
                   # quick perf mode (--perf-only: llama-benchy pp/tg, minutes per cell).
                   # Prints a short live tail, then the saved report's results table.
  local label="$1" dest="$2" clients="${3:-4}" report rc=0
  mkdir -p "$dest"
  PYTHONUNBUFFERED=1 "$BENCH" run \
    --base-url "http://127.0.0.1:$PORT/v1" \
    --model "$SERVED_NAME" \
    --format openai \
    --tokenizer "$MODEL_ID" \
    --seed 42 --perf --perf-only --parallel "$clients" --timeout 240 --no-live \
    --label "$label" --output-dir "$dest" --json-file "$dest/result.json" \
    2>&1 | tee "$dest/bench.log" | tail -4 || rc=$?
  # a non-zero bench exit must not kill the campaign (set -e) — record it and move on
  [[ $rc -ne 0 ]] && echo "- NOTE: bench exited $rc (its report may still be complete)" || true
  report=$(ls -t "$dest"/2026/*/*--*.md "$dest"/2026/*/*/*--*.md 2>/dev/null | head -1) || true
  if [[ -n "${report:-}" ]]; then sed -n '/^| Test/,/^$/p' "$report" || true; fi
  return 0
}

sha_msg() {        # sha256 of the whole response message, canonical JSON (raw-file sha on parse failure)
  python3 -c '
import hashlib, json, sys
raw = open(sys.argv[1], "rb").read()
try:
    m = json.loads(raw)
    print(hashlib.sha256(json.dumps(m, sort_keys=True).encode()).hexdigest())
except Exception:
    print(hashlib.sha256(raw).hexdigest())
' "$1"
}

content_of() {
  python3 -c '
import json, sys
try:
    m = json.load(open(sys.argv[1]))
    print(m["choices"][0]["message"].get("content") or "")
except Exception:
    print("")
' "$1"
}

content_sha() {    # sha256 of the normalized CONTENT text (whitespace-folded, reasoning excluded).
                   # The native engine is byte-level nondeterministic at greedy (2026-10-23 receipt:
                   # g1 raw shas differ while content matches), so the §2 gates bind on content,
                   # with raw-sha equality recorded as a bonus line when it holds.
  python3 -c '
import hashlib, json, re, sys
try:
    m = json.load(open(sys.argv[1]))
    c = m["choices"][0]["message"].get("content") or ""
except Exception:
    c = ""
print(hashlib.sha256(re.sub(r"\s+", " ", c).strip().encode()).hexdigest())
' "$1"
}

req_file() {       # $1 dest, $2 prompt text, rest: extra curl body fields as JSON obj
  local dest="$1" prompt="$2" extra="${3:-}"
  SERVED_MODEL="$SERVED_NAME" MSG_PROMPT="$prompt" MSG_EXTRA="$extra" python3 - "$dest" <<'EOF'
import json, os, sys
body = {
    "model": os.environ["SERVED_MODEL"],
    "messages": [{"role": "user", "content": os.environ["MSG_PROMPT"]}],
    "temperature": 0, "max_tokens": 512,
}
if os.environ["MSG_EXTRA"]:
    body.update(json.loads(os.environ["MSG_EXTRA"]))
json.dump(body, open(sys.argv[1], "w"))
EOF
}

# A seeded filler that tokenizes ~4 chars/token without the degenerate-repeat trap.
FILL_WORDS="harbor lantern gravel meadow tideline quarry bramble inkwell thicket ferry
cistern gable linden sluice cobble rafter minnow heather bracken cellars
windlass tarn larch furrow beck gantry lintel osier marl culvert"

filler_words() {  # $1 = approximate token count -> N words on stdout
  python3 - "$1" "$FILL_WORDS" <<'EOF'
import random, sys
n, words = int(sys.argv[1]), sys.argv[2].split()
random.seed(1234)
out = []
for i in range(n):
    w = words[i % len(words)]
    out.append(f"{w}{i%97}" if i % 11 == 0 else w)
print(" ".join(out))
EOF
}

# ---------------------------------------------------------------------------
# phase: preflight
# ---------------------------------------------------------------------------
if want preflight && ! phase_done preflight; then
  log "=== preflight"
  P="$OUT/00-preflight.md"; : > "$P"
  # prodqual (if selected) runs against production first and does its own handover
  # prompt — preflight only demands the stop when no production pass is selected
  if prod_running && [[ "$NEEDS_GPU" == 1 ]] && ! want prodqual; then
    if [[ "${PROMPT:-1}" == 1 ]]; then
      log "PRODUCTION IS RUNNING. Nothing in this script touches it, but the bench"
      log "needs the GPUs. Stop production yourself (the -tip stack's stop), then"
      read -r -p "press Enter once production is stopped (Ctrl-C to abort): " _
    else
      die "production is running and PROMPT=0 — stop it first, then re-run"
    fi
  elif prod_running; then
    log "production is up — read-only phases only (reference/prodqual); no restarts selected"
  fi
  prod_running && [[ "$NEEDS_GPU" == 1 ]] && ! want prodqual && die "production still running — this script will not continue"
  {
    echo "# preflight $RUN_ID"
    echo "- prepared marker: $(cat "$STATE_DIR/prepared" 2>/dev/null || echo MISSING)"
    echo "- head clocks delivered: $(clock_mhz local) MHz (healthy X925 ~3900, floor ~1378)"
    echo "- worker clocks delivered: $(clock_mhz "$WORKER") MHz"
    echo "- head RoCE devices: $(ibdev2netdev 2>/dev/null | head -4 | tr '\n' '; ' || echo 'ibdev2netdev missing')"
    echo "- NCCL_IB_HCA: ${NCCL_IB_HCA:-<empty>}"
    echo "- disk free head: $(df -BG /home | awk 'NR==2{print $4}')"
  } >> "$P"
  [[ -n "${NCCL_IB_HCA:-}" ]] || log "WARNING: NCCL_IB_HCA empty — the RoCE gather will not come up"
  touch "$OUT/.preflight.done"
fi

# ---------------------------------------------------------------------------
# phase: reference (production 0.6.6 output capture; read-only, one request)
# ---------------------------------------------------------------------------
REF_PROMPT="List the first eight prime numbers in order, comma separated, then state the sum."
if want reference && ! phase_done reference; then
  log "=== reference"
  P="$OUT/01-reference.md"; : > "$P"
  if prod_running; then
    req_file "$OUT/ref-req.json" "$REF_PROMPT"
    if curl -s -m 120 "http://127.0.0.1:$PORT/v1/chat/completions" \
         -H 'Content-Type: application/json' --data @"$OUT/ref-req.json" \
         > "$OUT/reference-prod.json"; then
      {
        echo "# production reference (0.6.6 Python engine, read-only single request)"
        echo "- sha256: $(sha_msg "$OUT/reference-prod.json")"
        echo '- content: '"$(content_of "$OUT/reference-prod.json" | head -c 400)"
      } >> "$P"
      log "production reference captured"
    else
      echo "# production reference: request failed (production up but not answering?)" >> "$P"
    fi
  else
    echo "# production reference: production was not running at campaign start — skipped" >> "$P"
    log "production not up; reference skipped (comparisons will be internal only)"
  fi
  touch "$OUT/.reference.done"
fi

# ---------------------------------------------------------------------------
# phase: prodqual — validate production 0.6.6 with the same instruments, read-only.
# Runs while production is UP, before any testing-stack restart. The server is probed
# and never restarted or modified: gates-style probes (determinism, same-conversation
# repeat, concurrent-vs-solo, seeded sampling — drafted=plain has no arm here since
# production's server flags are fixed) plus the same-harness tool-eval and perf runs,
# so 92 on the native engine gets its like-for-like production baseline.
# ---------------------------------------------------------------------------
if want prodqual && ! phase_done prodqual; then
  log "=== prodqual"
  P="$OUT/02b-prodqual.md"; : > "$P"
  if ! prod_running; then
    echo "# prodqual: production not running — skipped" >> "$P"
    touch "$OUT/.prodqual.done"
  else
    PROD_MODEL=$(curl -s -m 10 "http://127.0.0.1:$PORT/v1/models" | python3 -c '
import json, sys
try: print(json.load(sys.stdin)["data"][0]["id"])
except Exception: print("")') || PROD_MODEL=""
    PROD_MODEL="${PROD_MODEL:-$SERVED_NAME}"
    SAVED_SERVED="$SERVED_NAME"
    {
      echo "# prodqual (production 0.6.6 Python engine — probed read-only, never restarted or modified)"
      echo "- production model id: $PROD_MODEL"
      echo "- NOTE: this is client load on a live server; benches share it with real traffic"
    } >> "$P"
    # pg1 determinism: production is documented byte-stable — verify with our own probe
    SERVED_NAME="$PROD_MODEL" req_file "$OUT/prod-g1-req.json" "$REF_PROMPT"
    for r in a b; do
      curl -s -m 120 "http://127.0.0.1:$PORT/v1/chat/completions" \
        -H 'Content-Type: application/json' --data @"$OUT/prod-g1-req.json" \
        > "$OUT/prod-g1-$r.json" || true
    done
    if [[ "$(sha_msg "$OUT/prod-g1-a.json")" == "$(sha_msg "$OUT/prod-g1-b.json")" ]]; then
      echo "- PASS pg1 determinism: production greedy runs BYTE-identical (the 0.6.x property the native engine lacks — this is the control)" >> "$P"
    elif [[ "$(content_sha "$OUT/prod-g1-a.json")" == "$(content_sha "$OUT/prod-g1-b.json")" ]]; then
      echo "- NOTE pg1 determinism: production greedy content-identical, raw bytes differ (unexpected for 0.6.6 — worth a look)" >> "$P"
    else
      echo "- FAIL pg1 determinism: production greedy CONTENT differs across two runs" >> "$P"
    fi
    # pg3 same-conversation repeat (the prompt-cache path, read-only form)
    T1="Remember these three codes: alpha-7391, bravo-2250, charlie-8118. Reply with exactly: codes noted."
    T2="What were the three codes? Answer with just the three codes, comma separated."
    SERVED_NAME="$PROD_MODEL" T1="$T1" python3 - "$OUT" <<'EOF'
import json, os, sys
json.dump({"model": os.environ["SERVED_NAME"],
           "messages": [{"role": "user", "content": os.environ["T1"]}],
           "temperature": 0, "max_tokens": 1024}, open(f"{sys.argv[1]}/prod-g3-turn1.json", "w"))
EOF
    curl -s -m 300 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
      --data @"$OUT/prod-g3-turn1.json" > "$OUT/prod-g3-turn1-resp.json" || true
    R1=$(content_of "$OUT/prod-g3-turn1-resp.json")
    SERVED_NAME="$PROD_MODEL" T1="$T1" T2="$T2" R1="$R1" python3 - "$OUT" <<'EOF'
import json, os, sys
msgs = [{"role": "user", "content": os.environ["T1"]},
        {"role": "assistant", "content": os.environ["R1"]},
        {"role": "user", "content": os.environ["T2"]}]
json.dump({"model": os.environ["SERVED_NAME"], "messages": msgs,
           "temperature": 0, "max_tokens": 1024}, open(f"{sys.argv[1]}/prod-g3-conv.json", "w"))
EOF
    for r in x y; do
      curl -s -m 300 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
        --data @"$OUT/prod-g3-conv.json" > "$OUT/prod-g3-$r.json" || true
    done
    A=$(content_of "$OUT/prod-g3-x.json"); B=$(content_of "$OUT/prod-g3-y.json")
    if [[ "$A" == "$B" && -n "$A" ]]; then
      echo "- PASS pg3 same-conversation repeat: identical content on both sends (answer: $(echo "$A" | head -c 60))" >> "$P"
    else
      echo "- FAIL pg3 same-conversation repeat: '$A' vs '$B'" >> "$P"
    fi
    # pg4 concurrent = solo (read-only form)
    curl -s -m 120 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
      --data @"$OUT/prod-g1-req.json" > "$OUT/prod-g4-solo.json" || true
    for i in 1 2 3 4; do { curl -s -m 120 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
      --data @"$OUT/prod-g1-req.json" > "$OUT/prod-g4-c$i.json" || true; } & done; wait
    S=$(content_sha "$OUT/prod-g4-solo.json"); OK=1
    for i in 1 2 3 4; do [[ "$(content_sha "$OUT/prod-g4-c$i.json")" == "$S" ]] || OK=0; done
    if [[ "$OK" == 1 ]]; then
      echo "- PASS pg4 concurrent=solo (content): 4 concurrent outputs identical in content to solo" >> "$P"
    else
      echo "- FAIL pg4 concurrent=solo: a concurrent output differs in content from solo" >> "$P"
    fi
    # pg5 seeded sampling twice
    SERVED_NAME="$PROD_MODEL" python3 - "$OUT" <<'EOF'
import json, os, sys
json.dump({"model": os.environ["SERVED_NAME"],
           "messages": [{"role": "user", "content": "List the first eight prime numbers in order, comma separated, then state the sum."}],
           "temperature": 0.7, "top_p": 0.95, "seed": 42, "max_tokens": 512},
          open(f"{sys.argv[1]}/prod-g5-req.json", "w"))
EOF
    for r in a b; do
      curl -s -m 120 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
        --data @"$OUT/prod-g5-req.json" > "$OUT/prod-g5-$r.json" || true
    done
    if [[ "$(content_sha "$OUT/prod-g5-a.json")" == "$(content_sha "$OUT/prod-g5-b.json")" ]]; then
      echo "- PASS pg5 sampled determinism: seeded samples content-identical across two runs" >> "$P"
    else
      echo "- NOTE pg5 sampled determinism: seeded samples differ on production" >> "$P"
    fi
    # the same-harness benches (the head-to-head numbers)
    log "prodqual: tool-eval on production (~25 min of client load)"
    SERVED_NAME="$PROD_MODEL" bench_cell prod-reference "$OUT/prod-reference" >> "$P"
    python3 - "$OUT/prod-reference/result.json" >> "$P" <<'EOF'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    print("- production tool-eval final_score: %s (%s), %s points of %s" % (
        d.get("final_score"), d.get("rating", ""), d.get("scores", {}).get("total_points"),
        d.get("scores", {}).get("max_points")))
except Exception as e:
    print("- prod tool-eval result parse failed: %s" % e)
EOF
    log "prodqual: perf table on production (~10 min of client load)"
    SERVED_NAME="$PROD_MODEL" bench_perf prod-perf "$OUT/prod-perf" 4 >> "$P"
    SERVED_NAME="$SAVED_SERVED"   # bash persists VAR=x func assignments — restore it
    touch "$OUT/.prodqual.done"
    # hand the GPUs over if restart phases are selected
    if [[ "$NEEDS_GPU" == 1 ]] && prod_running; then
      if [[ "${PROMPT:-1}" == 1 ]]; then
        log "production probes complete. Stop production yourself (the -tip stack's stop), then"
        read -r -p "press Enter once production is stopped (Ctrl-C to abort): " _
      fi
      prod_running && die "production still running — the GPU phases will not start"
    fi
  fi
fi

# ---------------------------------------------------------------------------
# phase: parity + candidate (the bench cells; restart with admission fallback)
# ---------------------------------------------------------------------------
run_cell() {       # $1 label, $2.. env pairs; PARALLEL=4 first, fallback 2
  local label="$1"; shift
  local P="$OUT/$label.md"; : > "$P"
  log "=== cell $label: restart (parallel 4, fallback 2)"
  if ! restart_fb "$@"; then
    echo "# $label: server refused to start at parallel 4 and 2" >> "$P"; return 1; fi
  echo "# $label" >> "$P"; echo "- server: $(serve_line)" >> "$P"; echo "- $(roce_state)" >> "$P"
  log "bench: $label"
  bench_cell "$label" "$OUT/$label" >> "$P" 2>&1
  grep -A11 "^| Test" "$OUT/$label"/2026/*/*--*.md "$OUT/$label"/2026/*/*/*--*.md 2>/dev/null | head -13 >> "$P" || true
}

if want parity && ! phase_done parity; then
  run_cell parity DENSE_BITS=8 HEAD_BITS=8 KV_CACHE=fp8 CONTEXT=524288 PARALLEL=4
  touch "$OUT/.parity.done"
fi

if want candidate && ! phase_done candidate; then
  run_cell dense4fp8 DENSE_BITS=4 HEAD_BITS=4 KV_CACHE=fp8 CONTEXT=524288 PARALLEL=4
  touch "$OUT/.candidate.done"
fi

# ---------------------------------------------------------------------------
# VALIDATION §2 gates, content form — shared by kvqual (dense8) and kvqual4 (dense4fp8)
# ---------------------------------------------------------------------------
run_gates() {      # $1 tag (receipt file prefix), $2.. the gated config's env pairs
  local tag="$1"; shift
  if ! restart_fb "$@" PARALLEL=4; then
    echo "- SKIP: server refused to start for the gate config" >> "$P"
    return 1
  fi

  # g1 determinism: the same greedy probe twice — content is the gate, raw bytes are the FYI
  req_file "$OUT/g-$tag-req.json" "$REF_PROMPT"
  chat "$OUT/g-$tag-req.json" > "$OUT/g-$tag-1a.json"; chat "$OUT/g-$tag-req.json" > "$OUT/g-$tag-1b.json"
  local CA CB BA BB
  CA=$(content_sha "$OUT/g-$tag-1a.json"); CB=$(content_sha "$OUT/g-$tag-1b.json")
  BA=$(sha_msg "$OUT/g-$tag-1a.json"); BB=$(sha_msg "$OUT/g-$tag-1b.json")
  if [[ "$CA" != "$CB" ]]; then
    echo "- FAIL g1 determinism: greedy CONTENT differs across two runs on the same server" >> "$P"
  elif [[ "$BA" == "$BB" ]]; then
    echo "- PASS g1 determinism: two greedy runs byte-identical" >> "$P"
  else
    echo "- PASS g1 determinism (content): two greedy runs identical in content; raw bytes differ" >> "$P"
    echo "  (byte-level engine nondeterminism — reasoning/metadata bytes; the 0.6.x Python engine was byte-stable, the native one is not)" >> "$P"
  fi

  # g2 drafted = plain (the 0.6.x guarantee, in its semantic form: content, not bytes)
  cp "$OUT/g-$tag-1a.json" "$OUT/g-$tag-2-drafted.json"  # drafted arm = captured with DFlash2 on (g1 server)
  if ! restart "$@" PARALLEL=4 DRAFTS=0; then
    echo "- SKIP g2 drafted=plain: server refused to start with DRAFTS=0" >> "$P"
    return 1
  fi
  chat "$OUT/g-$tag-req.json" > "$OUT/g-$tag-2-plain.json"
  if [[ "$(content_sha "$OUT/g-$tag-2-drafted.json")" == "$(content_sha "$OUT/g-$tag-2-plain.json")" ]]; then
    echo "- PASS g2 drafted=plain (content): DFlash2-drafted and --no-drafts greedy outputs identical" >> "$P"
    [[ "$(sha_msg "$OUT/g-$tag-2-drafted.json")" != "$(sha_msg "$OUT/g-$tag-2-plain.json")" ]] && \
      echo "  (raw bytes differ — that is the g1 byte-level nondeterminism, not the drafter)" >> "$P"
  else
    echo "- FAIL g2 drafted=plain: CONTENT differs between drafted and plain" >> "$P"
    diff <(content_of "$OUT/g-$tag-2-drafted.json") <(content_of "$OUT/g-$tag-2-plain.json") | head -5 >> "$P" || true
  fi

  # g3 resumed = fresh (the prompt cache's snap hooks) — the SAME 3-message conversation
  # sent twice: arm 1 computes every token (cold cache, just restarted), arm 2 is a full
  # prefix-cache hit. The 0.6.x gate, in its semantic form. (The first campaign's version
  # wrongly compared two DIFFERENT prompts — turn 2 alone vs turn 1+2 — unrunnable by design.)
  if ! restart_fb "$@" PARALLEL=4; then
    echo "- SKIP g3 resumed=fresh: server refused to start" >> "$P"
    return 1
  fi
  local T1 T2 R1 A B
  T1="Remember these three codes: alpha-7391, bravo-2250, charlie-8118. Reply with exactly: codes noted."
  T2="What were the three codes? Answer with just the three codes, comma separated."
  SERVED_MODEL="$SERVED_NAME" T1="$T1" python3 - "$OUT" "$tag" <<'EOF'
import json, os, sys
json.dump({"model": os.environ["SERVED_MODEL"],
           "messages": [{"role": "user", "content": os.environ["T1"]}],
           "temperature": 0, "max_tokens": 1024}, open(f"{sys.argv[1]}/g-{sys.argv[2]}-3-turn1.json", "w"))
EOF
  chat "$OUT/g-$tag-3-turn1.json" > "$OUT/g-$tag-3-turn1-resp.json"
  R1=$(content_of "$OUT/g-$tag-3-turn1-resp.json")
  SERVED_MODEL="$SERVED_NAME" T1="$T1" T2="$T2" R1="$R1" python3 - "$OUT" "$tag" <<'EOF'
import json, os, sys
msgs = [{"role": "user", "content": os.environ["T1"]},
        {"role": "assistant", "content": os.environ["R1"]},
        {"role": "user", "content": os.environ["T2"]}]
json.dump({"model": os.environ["SERVED_MODEL"], "messages": msgs,
           "temperature": 0, "max_tokens": 1024}, open(f"{sys.argv[1]}/g-{sys.argv[2]}-3-conv.json", "w"))
EOF
  chat "$OUT/g-$tag-3-conv.json" > "$OUT/g-$tag-3-computed.json"
  chat "$OUT/g-$tag-3-conv.json" > "$OUT/g-$tag-3-cached.json"
  A=$(content_of "$OUT/g-$tag-3-computed.json"); B=$(content_of "$OUT/g-$tag-3-cached.json")
  if [[ "$A" == "$B" && -n "$A" ]]; then
    echo "- PASS g3 resumed=fresh (content): cache-hit reply identical to the computed reply (answer: $(echo "$A" | head -c 60))" >> "$P"
  else
    echo "- FAIL g3 resumed=fresh: computed='$A' cached='$B'" >> "$P"
  fi
  for arm in computed cached; do
    python3 - "$OUT/g-$tag-3-$arm.json" "$arm" >> "$P" <<'EOF'
import json, sys
try:
    u = json.load(open(sys.argv[1]))["usage"]
    print("- g3 %s arm: cached_tokens=%s" % (sys.argv[2],
          (u.get("prompt_tokens_details") or {}).get("cached_tokens")))
except Exception:
    pass
EOF
  done

  # g4 concurrent = solo (content form)
  chat "$OUT/g-$tag-req.json" > "$OUT/g-$tag-4-solo.json"
  for i in 1 2 3 4; do { chat "$OUT/g-$tag-req.json" > "$OUT/g-$tag-4-c$i.json" || true; } & done; wait
  local S OK BYTES i
  S=$(content_sha "$OUT/g-$tag-4-solo.json"); OK=1; BYTES=0
  for i in 1 2 3 4; do
    [[ "$(content_sha "$OUT/g-$tag-4-c$i.json")" == "$S" ]] || OK=0
    [[ "$(sha_msg "$OUT/g-$tag-4-c$i.json")" == "$(sha_msg "$OUT/g-$tag-4-solo.json")" ]] || BYTES=$((BYTES+1))
  done
  if [[ "$OK" == 1 ]]; then
    echo "- PASS g4 concurrent=solo (content): all 4 concurrent outputs identical in content to solo" >> "$P"
    [[ "$BYTES" -gt 0 ]] && echo "  ($BYTES/4 differ in raw bytes — the g1 byte-level nondeterminism)" >> "$P"
  else
    echo "- FAIL g4 concurrent=solo: at least one concurrent output differs in CONTENT from solo" >> "$P"
  fi

  # g5 sampled determinism (seeded, twice)
  SERVED_MODEL="$SERVED_NAME" python3 - "$OUT" "$tag" <<'EOF'
import json, os, sys
json.dump({"model": os.environ["SERVED_MODEL"],
           "messages": [{"role": "user", "content": "List the first eight prime numbers in order, comma separated, then state the sum."}],
           "temperature": 0.7, "top_p": 0.95, "seed": 42, "max_tokens": 512},
          open(f"{sys.argv[1]}/g-{sys.argv[2]}-5-req.json", "w"))
EOF
  chat "$OUT/g-$tag-5-req.json" > "$OUT/g-$tag-5a.json"; chat "$OUT/g-$tag-5-req.json" > "$OUT/g-$tag-5b.json"
  if [[ "$(sha_msg "$OUT/g-$tag-5a.json")" == "$(sha_msg "$OUT/g-$tag-5b.json")" ]]; then
    echo "- PASS g5 sampled determinism: seeded sample byte-identical across two runs" >> "$P"
  elif [[ "$(content_sha "$OUT/g-$tag-5a.json")" == "$(content_sha "$OUT/g-$tag-5b.json")" ]]; then
    echo "- NOTE g5 sampled determinism: seeded samples identical in content, raw bytes differ" >> "$P"
  else
    echo "- NOTE g5 sampled determinism: seeded samples differ (sampling is not seed-reproducible on this engine; recorded, not gating)" >> "$P"
  fi
}

# phase: kvqual — the gates on dense8+fp8 KV at the 512K window
# ---------------------------------------------------------------------------
if want kvqual && ! phase_done kvqual; then
  log "=== kvqual (dense8)"
  P="$OUT/03-kvqual.md"; : > "$P"
  echo "# kvqual (VALIDATION §2 on fp8 KV, dense8, 512K window; content-form gates — the native engine is byte-nondeterministic, see g1)" >> "$P"
  if run_gates dense8 DENSE_BITS=8 HEAD_BITS=8 KV_CACHE=fp8 CONTEXT=524288; then
    touch "$OUT/.kvqual.done"
  else
    echo "- kvqual incomplete (a restart was refused) — rerun this phase" >> "$P"
  fi
fi

# phase: kvqual4 — the same gates on the ladder candidate dense4fp8
# ---------------------------------------------------------------------------
if want kvqual4 && ! phase_done kvqual4; then
  log "=== kvqual4 (dense4fp8)"
  P="$OUT/03b-kvqual4.md"; : > "$P"
  echo "# kvqual4 (VALIDATION §2 on fp8 KV, DENSE_BITS=4, 512K window; the ladder candidate's own gates)" >> "$P"
  if run_gates dense4fp8 DENSE_BITS=4 HEAD_BITS=4 KV_CACHE=fp8 CONTEXT=524288; then
    touch "$OUT/.kvqual4.done"
  else
    echo "- kvqual4 incomplete (a restart was refused) — rerun this phase" >> "$P"
  fi
fi

# phase: throughput — the perf-only bench (llama-benchy pp/tg) for both fp8 cells at 512K.
# The first campaign's bench call accidentally ran the full tool-eval (quality 92) —
# this phase collects the speed tables those receipts lack.
# ---------------------------------------------------------------------------
if want throughput && ! phase_done throughput; then
  log "=== throughput (perf-only: parity + dense4fp8 at 512K)"
  P="$OUT/03c-throughput.md"; : > "$P"
  echo "# throughput (perf-only bench, 512K window, fp8 KV)" >> "$P"
  if restart_fb DENSE_BITS=8 HEAD_BITS=8 KV_CACHE=fp8 CONTEXT=524288 PARALLEL=4; then
    echo "## parity (dense8+fp8)" >> "$P"; echo "- server: $(serve_line)" >> "$P"
    bench_perf parity "$OUT/parity-perf" 4 >> "$P"
  else
    echo "- parity config refused to start" >> "$P"
  fi
  if restart_fb DENSE_BITS=4 HEAD_BITS=4 KV_CACHE=fp8 CONTEXT=524288 PARALLEL=4; then
    echo "## dense4fp8" >> "$P"; echo "- server: $(serve_line)" >> "$P"
    bench_perf dense4fp8 "$OUT/dense4fp8-perf" 4 >> "$P"
  else
    echo "- dense4fp8 config refused to start" >> "$P"
  fi
  touch "$OUT/.throughput.done"
fi

# phase: climb — how many 512K streams fit on dense4fp8? (the plan's rule: a refusal is
# a recorded outcome, not an OOM — no fallback here; PARALLEL=4 needs 14.4 GiB of the
# 15.8 GiB the first campaign saw free, so 6 and 8 should both fit)
# ---------------------------------------------------------------------------
if want climb && ! phase_done climb; then
  log "=== climb (dense4fp8 at 512K, PARALLEL=5/6/8)"
  P="$OUT/03d-climb.md"; : > "$P"
  echo "# climb (dense4fp8 + fp8 KV + 524288; client parallel = server streams)" >> "$P"
  echo "- the engine's admission arithmetic at this window: 8 streams need 25.29 GiB, budget leaves ~15.8 — it suggests --parallel 5 (5 x 3.16 = 15.81 GiB, knife-edge); 6+ is blocked by the engine's 12.2 GiB reserve (no knob exposed)" >> "$P"
  for N in 5 6 8; do
    if restart DENSE_BITS=4 HEAD_BITS=4 KV_CACHE=fp8 CONTEXT=524288 PARALLEL=$N; then
      echo "## PARALLEL=$N" >> "$P"; echo "- server: $(serve_line)" >> "$P"; echo "- $(roce_state)" >> "$P"
      bench_perf "climb$N" "$OUT/climb$N" $N >> "$P"
    else
      # the usual refusal cause at 512K is the kept-states store's worst-case sizing;
      # retry once with a trimmed prompt cache before recording the refusal
      echo "- PARALLEL=$N refused at defaults; retrying with PROMPT_CACHE_GIB=4" >> "$P"
      if restart DENSE_BITS=4 HEAD_BITS=4 KV_CACHE=fp8 CONTEXT=524288 PARALLEL=$N PROMPT_CACHE_GIB=4; then
        echo "## PARALLEL=$N (prompt cache trimmed to 4 GiB)" >> "$P"; echo "- server: $(serve_line)" >> "$P"; echo "- $(roce_state)" >> "$P"
        bench_perf "climb$N" "$OUT/climb$N" $N >> "$P"
      else
        echo "- PARALLEL=$N refused at 524288 even with a 4 GiB prompt cache — recorded outcome" >> "$P"
      fi
    fi
  done
  touch "$OUT/.climb.done"
fi

# ---------------------------------------------------------------------------
# phase: kvqual4P — VALIDATION §2 gates on PAGED GLM KV (the #1163/#1211 branch):
# dense4bf16 at the 512K window, TF_KV_PAGES=on, KV_CACHE unset (bf16 is the only
# paged path — the engine exits on fp8 KV × paging, 87e4417e receipt,
# campaign-20261023-165319). First a NO-PAGING control at the same window so the
# receipt answers "would it admit anyway": window-mode bf16 holds ~6.3 GiB per
# stream, so PARALLEL=4 should refuse (~2 streams fit); paging allocates 512-row
# pages on demand, so gate-short prompts may admit 4.
# ---------------------------------------------------------------------------
if want kvqual4P && ! phase_done kvqual4P; then
  log "=== kvqual4P (paged dense4bf16)"
  P="$OUT/05a-kvqual4P.md"; : > "$P"
  echo "# kvqual4P (VALIDATION §2 on PAGED bf16 KV, DENSE_BITS=4, 512K window; TF_KV_PAGES=on)" >> "$P"
  if restart DENSE_BITS=4 HEAD_BITS=4 CONTEXT=524288 PARALLEL=4; then
    echo "- window-mode control (no TF_KV_PAGES): bf16 KV admitted PARALLEL=4 at 512K (the expected refusal did NOT happen — recorded)" >> "$P"
  else
    echo "- window-mode control (no TF_KV_PAGES): bf16 at 512K refused PARALLEL=4 — the window arithmetic holds; paging is the only route to parity" >> "$P"
    docker logs "$CONTAINER_NAME" 2>&1 | tail -2 >> "$P"
  fi
  if run_gates dense4bf16P DENSE_BITS=4 HEAD_BITS=4 CONTEXT=524288 TF_KV_PAGES=on; then
    touch "$OUT/.kvqual4P.done"
  else
    echo "- kvqual4P incomplete (a restart was refused) — rerun this phase" >> "$P"
  fi
fi

# ---------------------------------------------------------------------------
# phase: throughputP — perf-only bench on PAGED bf16 KV (dense4, 512K window):
# the same packing as dense4fp8 with only the KV dtype+layout different, so the
# tables compare directly against the fp8-window receipts in TOOL_EVAL_TESTING.md.
# ---------------------------------------------------------------------------
if want throughputP && ! phase_done throughputP; then
  log "=== throughputP (perf-only: paged dense4bf16 at 512K)"
  P="$OUT/05b-throughputP.md"; : > "$P"
  echo "# throughputP (perf-only bench, 512K window, PAGED bf16 KV, DENSE_BITS=4)" >> "$P"
  if restart_fb DENSE_BITS=4 HEAD_BITS=4 CONTEXT=524288 TF_KV_PAGES=on PARALLEL=4; then
    echo "- server: $(serve_line)" >> "$P"; echo "- $(roce_state)" >> "$P"
    bench_perf "dense4bf16P" "$OUT/dense4bf16P" 4 >> "$P"
  else
    echo "- dense4bf16P config refused to start (paged bf16 at 512K, PARALLEL=4)" >> "$P"
  fi
  touch "$OUT/.throughputP.done"
fi

# ---------------------------------------------------------------------------
# phase: climbP — PARALLEL=5/6/8 under paging. The window-mode ceiling sat at
# "the engine's 12.2 GiB reserve blocks 6+"; demand-based pages only hold what is
# actually live, so this is the branch's real payoff question.
# ---------------------------------------------------------------------------
if want climbP && ! phase_done climbP; then
  log "=== climbP (paged dense4bf16, PARALLEL=5/6/8)"
  P="$OUT/05c-climbP.md"; : > "$P"
  echo "# climbP (PAGED bf16 KV + dense4 + 524288; client parallel = server streams)" >> "$P"
  echo "- paging allocates 512-row pages on demand: admission is about live usage (a full 512K stream holds ~6.3 GiB bf16), not streams × window" >> "$P"
  for N in 5 6 8; do
    if restart DENSE_BITS=4 HEAD_BITS=4 CONTEXT=524288 TF_KV_PAGES=on PARALLEL=$N; then
      echo "## PARALLEL=$N" >> "$P"; echo "- server: $(serve_line)" >> "$P"; echo "- $(roce_state)" >> "$P"
      bench_perf "climbP$N" "$OUT/climbP$N" $N >> "$P"
    else
      echo "- PARALLEL=$N refused (paged bf16; the engine's last lines follow)" >> "$P"
      docker logs "$CONTAINER_NAME" 2>&1 | tail -2 >> "$P"
    fi
  done
  touch "$OUT/.climbP.done"
fi

# ---------------------------------------------------------------------------
# phase: knobs — the untested TF_GLM_ decode/prefill knobs (2026-10-23 source
# scan of the pin): TF_GLM_DRAFT_CONFIDENCE (where a DFlash2 chain's chance
# product stops it, default 0.7 — the decode acceptance trade), TF_GLM_PROMPT_
# STEPS=0 (prompt passes run whole — no other stream's round between chunks),
# TF_GLM_OVERLAP_PIECES=6 (a prompt chunk's RoCE exchanges in 6 row pieces,
# default 3), TF_GLM_KDA_DECODE_WIDE=0 (KDA decode off the three-kernel chain).
# Each cell is a TIMED restart — the engine's own "loaded in Ns" line is the
# loading-time metric (production 0.6.6 loads in 93.7 s; our restart walls run
# ~265 s — the load-nodraft cell isolates the drafter's share). Baseline: the
# dense4fp8 receipts (campaign-20261023-093827, and the 185520 A/B round).
# ---------------------------------------------------------------------------
if want knobs && ! phase_done knobs; then
  log "=== knobs (untested TF_GLM_ decode/prefill knobs, dense4fp8 @ 512K)"
  P="$OUT/06-knobs.md"; : > "$P"
  echo "# knobs (dense4fp8 + fp8 KV + 524288 x4; each cell = timed restart + bench matrix)" >> "$P"
  echo "- loading-time anchor: production 0.6.6 loads in 93.7 s (its serving line); our restart walls run ~265 s" >> "$P"
  knob_cell() {   # $1 label, $2.. env pairs — timed restart, the engine's load line, bench
    local label="$1" S E loaded
    shift
    S=$(date +%s)
    if restart "$@"; then
      E=$(date +%s)
      loaded=$(docker logs "$CONTAINER_NAME" 2>&1 | grep -oE "loaded in [0-9.]+s" | tail -1)
      echo "## $label — wall $((E-S)) s, engine ${loaded:-"(no loaded-in line)"}" >> "$P"
      echo "- server: $(serve_line)" >> "$P"; echo "- $(roce_state)" >> "$P"
      bench_perf "knob-$label" "$OUT/knob-$label" 4 >> "$P"
    else
      echo "## $label — refused to start" >> "$P"
      docker logs "$CONTAINER_NAME" 2>&1 | tail -2 >> "$P"
    fi
  }
  knob_cell conf055  DENSE_BITS=4 HEAD_BITS=4 KV_CACHE=fp8 CONTEXT=524288 PARALLEL=4 TF_GLM_DRAFT_CONFIDENCE=0.55
  knob_cell conf085  DENSE_BITS=4 HEAD_BITS=4 KV_CACHE=fp8 CONTEXT=524288 PARALLEL=4 TF_GLM_DRAFT_CONFIDENCE=0.85
  knob_cell steps0   DENSE_BITS=4 HEAD_BITS=4 KV_CACHE=fp8 CONTEXT=524288 PARALLEL=4 TF_GLM_PROMPT_STEPS=0
  knob_cell overlap6 DENSE_BITS=4 HEAD_BITS=4 KV_CACHE=fp8 CONTEXT=524288 PARALLEL=4 TF_GLM_OVERLAP_PIECES=6
  knob_cell kdawide0 DENSE_BITS=4 HEAD_BITS=4 KV_CACHE=fp8 CONTEXT=524288 PARALLEL=4 TF_GLM_KDA_DECODE_WIDE=0
  # the loading-time cells: the drafter's share of our ~265 s (packing share is read
  # off the engine's own lines; a bf16 cell is not expressible via env pairs — an
  # unset would be needed, so packing is inferred from the engine's load lines)
  knob_cell load-nodraft DENSE_BITS=4 HEAD_BITS=4 KV_CACHE=fp8 CONTEXT=524288 PARALLEL=4 DRAFTS=0
  touch "$OUT/.knobs.done"
fi

# ---------------------------------------------------------------------------
# phase: prof — TF_GLM_PROF=1: where a forward's GPU time goes. The engine's
# cuda_prof.zig sums CUDA-event sections per forward and logs averages every 64
# calls per draft-window profiler; probes at max_tokens 2048 generate ~78 calls
# per window, so every window reports. The concurrency question: our aggregate
# decode FALLS with streams (48.3 solo -> 41.3 c2 -> 38.7 c4 @d0) while
# production's RISES (51.6 -> 67.3 -> 60.5). Sections to watch: wall-vs-gpu per
# call (scheduling bubbles), gather_a/gather_b (RoCE), kda/mla (attention over
# depth), dense/router/routed (the MoE path), mtp (the drafter).
# ---------------------------------------------------------------------------
if want prof && ! phase_done prof; then
  log "=== prof (TF_GLM_PROF=1, the concurrency diagnosis)"
  P="$OUT/03e-prof.md"; : > "$P"
  echo "# prof (TF_GLM_PROF=1, dense4fp8 + fp8 KV + 524288, PARALLEL=4)" >> "$P"
  restart_fb DENSE_BITS=4 HEAD_BITS=4 KV_CACHE=fp8 CONTEXT=524288 PARALLEL=4 TF_GLM_PROF=1
  python3 - "$OUT" "$FILL_WORDS" <<'EOF'
import json, random, sys
words, seed = sys.argv[2].split(), 4242
for tag, depth in (("d0", 0), ("d4096", 4096)):
    random.seed(seed + depth)
    n = depth // 4
    body = " ".join(words[i % len(words)] + (f"-{i%89}" if i % 13 == 0 else "") for i in range(n))
    q = "Write a 300-word technical summary of a harbor crane's maintenance cycle. Do not use any formatting."
    prompt = q if depth == 0 else body + "\n\n" + q
    for variant in ("solo", "c4"):
        json.dump({"model": "GLM-5.3-Flash-EXL3",
                   "messages": [{"role": "user", "content": prompt}],
                   "temperature": 0, "max_tokens": 2048},
                  open(f"{sys.argv[1]}/prof-{variant}-{tag}-req.json", "w"))
EOF
  curl -s -m 10 "http://127.0.0.1:$PORT/health" > "$OUT/prof-health-before.json" || true
  for variant in solo c4; do
    for tag in d0 d4096; do
      START=$(date +%s.%N)
      if [[ "$variant" == solo ]]; then
        chat "$OUT/prof-solo-$tag-req.json" > "$OUT/prof-solo-$tag-resp.json" || true
      else
        for i in 1 2 3 4; do { chat "$OUT/prof-c4-$tag-req.json" > "$OUT/prof-c4-$tag-$i.json" || true; } & done; wait
      fi
      ELAPSED=$(python3 -c "print(f'{$(date +%s.%N)-$START:.2f}')")
      echo "$variant $tag $ELAPSED" >> "$OUT/prof-walls.txt"
      log "prof probe $variant $tag done (${ELAPSED}s)"
    done
  done
  curl -s -m 10 "http://127.0.0.1:$PORT/health" > "$OUT/prof-health-after.json" || true
  python3 - "$OUT" >> "$P" <<'EOF'
import glob, json, sys
out = sys.argv[1]
walls = {}
for line in open(f"{out}/prof-walls.txt"):
    v, tag, t = line.split()
    walls[f"{v} {tag}"] = float(t)
for variant in ("solo", "c4"):
    for tag in ("d0", "d4096"):
        total = 0
        for f in glob.glob(f"{out}/prof-{variant}-{tag}-resp*.json") or glob.glob(f"{out}/prof-{variant}-{tag}-[0-9]*.json"):
            try: total += json.load(open(f))["usage"]["completion_tokens"]
            except Exception: pass
        t = walls.get(f"{variant} {tag}", 0)
        rate = f"{total/t:,.1f} tok/s aggregate" if t > 0 and total else "rate n/a"
        print(f"- {variant} {tag}: {t:.1f}s wall, {total} completion tokens -> {rate}")
try:
    b = json.load(open(f"{out}/prof-health-before.json")); a = json.load(open(f"{out}/prof-health-after.json"))
    for k in ("drafted", "accepted", "rounds"):
        if isinstance(b, dict) and k in b: print(f"- health {k}: {b.get(k)} -> {a.get(k)}")
except Exception as e:
    print(f"- health delta unavailable: {e}")
EOF
  echo "## head prof lines (per draft-window section averages, ms/call)" >> "$P"
  docker logs "$CONTAINER_NAME" 2>&1 | grep "glm prof" | tail -24 >> "$P" || true
  echo "## worker prof lines" >> "$P"
  ssh -o BatchMode=yes "$WORKER" "docker logs $CONTAINER_NAME 2>&1 | grep 'glm prof' | tail -24" >> "$P" 2>/dev/null || true
  touch "$OUT/.prof.done"
fi

# ---------------------------------------------------------------------------
# phase: depth — the forum-taught gate: fact+needle probes at 8k/64k/192k/384k
# ---------------------------------------------------------------------------
if want depth && ! phase_done depth; then
  log "=== depth"
  P="$OUT/04-depth.md"; : > "$P"
  echo "# depth gate (fp8 KV, dense8, 512K window; community bar: production 93.9%)" >> "$P"
  restart DENSE_BITS=8 HEAD_BITS=8 KV_CACHE=fp8 CONTEXT=524288 PARALLEL=2
  python3 - "$OUT" "$FILL_WORDS" <<'EOF'
import json, random, sys
out, words, seed = sys.argv[1], sys.argv[2].split(), 987
facts = [("the loading-bay access code", "XJ-4821"),
         ("the maintenance vendor's name", "Ostergaard"),
         ("the coolant batch number", "B-2214-C"),
         ("the inspection due date", "March 3rd")]
for depth in (8000, 64000, 192000, 384000):
    random.seed(seed + depth)
    n = depth // 4                      # ~4 chars/token with the trailing spaces
    body = []
    anchor = (n * 3) // 5               # facts sit ~60% deep, as in grown sessions
    for i in range(n):
        body.append(words[i % len(words)] + (f"-{i%89}" if i % 13 == 0 else ""))
        if i == anchor:
            for j, (q, a) in enumerate(facts[:3]):
                body.append(f" Record {j}: {q} is {a}. ")
    prompt = " ".join(body) + (
        "\n\nFrom the records above, answer in one line each:\n"
        + "\n".join(f"{j+1}. What is {q}?" for j, (q, _) in enumerate(facts[:3]))
        + "\nAnswer with the three values only."
    )
    json.dump({"model": "GLM-5.3-Flash-EXL3",
               "messages": [{"role": "user", "content": prompt}],
               "temperature": 0, "max_tokens": 512}, open(f"{out}/depth-{depth}-req.json", "w"))
    json.dump(facts[:3], open(f"{out}/depth-{depth}-facts.json", "w"))
EOF
  for depth in 8000 64000 192000 384000; do
    log "depth probe at $depth"
    chat "$OUT/depth-$depth-req.json" > "$OUT/depth-$depth-resp.json" || true
    C=$(content_of "$OUT/depth-$depth-resp.json" 2>/dev/null || echo "")
    HITS=0; TOTAL=0
    while IFS= read -r ans; do
      TOTAL=$((TOTAL+1)); [[ "$C" == *"$ans"* ]] && HITS=$((HITS+1))
    done < <(python3 -c 'import json,sys; print("\n".join(a for _,a in json.load(open(sys.argv[1]))))' "$OUT/depth-$depth-facts.json")
    echo "- depth $depth: $HITS/$TOTAL facts retrieved" >> "$P"
  done
  touch "$OUT/.depth.done"
fi

# ---------------------------------------------------------------------------
# phase: cliff — 4-stream decode with vs without one structured-output request
# ---------------------------------------------------------------------------
if want cliff && ! phase_done cliff; then
  log "=== cliff"
  P="$OUT/05-cliff.md"; : > "$P"
  echo "# concurrency cliff: one structured request in the batch (MaTeZZ's mechanism)" >> "$P"
  python3 - "$OUT" <<'EOF'
import json, sys
plain = {"model": "GLM-5.3-Flash-EXL3",
         "messages": [{"role": "user", "content": "Write a 200-word technical description of a harbor crane. Do not use any formatting."}],
         "temperature": 0, "max_tokens": 256}
json.dump(plain, open(f"{sys.argv[1]}/cliff-plain.json", "w"))
schema = dict(plain)
schema["response_format"] = {"type": "json_schema", "json_schema": {
    "name": "crane", "strict": True,
    "schema": {"type": "object", "properties": {"description": {"type": "string"}},
               "required": ["description"], "additionalProperties": False}}}
json.dump(schema, open(f"{sys.argv[1]}/cliff-schema.json", "w"))
EOF
  for variant in plain schema; do
    START=$(date +%s.%N)
    for i in 1 2 3 4; do
      if [[ "$variant" == schema && $i -eq 4 ]]; then F="$OUT/cliff-schema.json"; else F="$OUT/cliff-plain.json"; fi
      { chat "$F" > "$OUT/cliff-$variant-$i.json" || true; } &
    done; wait
    ELAPSED=$(python3 -c "print(f'{$(date +%s.%N)-$START:.1f}')")
    echo "- batch of 4 with one $variant request: ${ELAPSED}s wall for 4x256 tokens" >> "$P"
  done
  echo "- verdict: if the schema batch is much slower, the structured-request draft-truncation cliff exists here too" >> "$P"
  touch "$OUT/.cliff.done"
fi

# ---------------------------------------------------------------------------
# phase: prefill — corrected probe 3 (~14k tokens)
# ---------------------------------------------------------------------------
if want prefill && ! phase_done prefill; then
  log "=== prefill"
  P="$OUT/06-prefill.md"; : > "$P"
  # this phase only probes — make sure a server is actually up (the climb phase may
  # have ended on a refusal with the stack down)
  curl -s -m 5 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 || \
    restart_fb DENSE_BITS=4 HEAD_BITS=4 KV_CACHE=fp8 CONTEXT=524288 PARALLEL=4
  python3 - "$OUT" "$FILL_WORDS" <<'EOF'
import json, sys
words, n = sys.argv[2].split(), 14000
body = " ".join(words[i % len(words)] + (f"-{i%83}" if i % 17 == 0 else "") for i in range(n))
json.dump({"model": "GLM-5.3-Flash-EXL3",
           "messages": [{"role": "user", "content": body + "\n\nReply with exactly: read complete."}],
           "temperature": 0, "max_tokens": 1}, open(f"{sys.argv[1]}/prefill-req.json", "w"))
EOF
  T=$(curl -s -o "$OUT/prefill-resp.json" -m 900 -w '%{time_total}' \
        "http://127.0.0.1:$PORT/v1/chat/completions" \
        -H 'Content-Type: application/json' --data @"$OUT/prefill-req.json") || true
  python3 - "$OUT/prefill-resp.json" "${T:-0}" >> "$P" <<'EOF'
import json, sys
try:
    u = json.load(open(sys.argv[1]))["usage"]
    pt = u.get("prompt_tokens") or 0
    t = float(sys.argv[2] or 0)
    rate = f" = {pt/t:,.0f} tok/s prefill" if t > 0.5 and pt else ""
    print(f"- usage: {pt:,} prompt tokens; wall {t:.1f}s (max_tokens=1, so wall is prefill + one decode step){rate}")
    print(f"- cached: {u.get('prompt_tokens_details')}")
except Exception as e:
    print(f"- prefill probe failed to parse: {e} (curl wall {sys.argv[2]}s)")
EOF
  echo "- compare with the v1.1.0 ~625 t/s and the pp2048 cells (~1,030 t/s) in TOOL_EVAL_TESTING.md" >> "$P"
  touch "$OUT/.prefill.done"
fi

# ---------------------------------------------------------------------------
# phase: rung — the 524288x1 goal rung
# ---------------------------------------------------------------------------
if want rung && ! phase_done rung; then
  log "=== rung 524288x1"
  P="$OUT/07-rung.md"; : > "$P"
  echo "# 524288 x PARALLEL=1 (the goal rung; production-parity window)" >> "$P"
  restart DENSE_BITS=8 HEAD_BITS=8 KV_CACHE=fp8 CONTEXT=524288 PARALLEL=1
  echo "- server: $(serve_line)" >> "$P"; echo "- $(roce_state)" >> "$P"
  chat "$OUT/depth-8000-req.json" > "$OUT/rung-probe.json" || true
  C=$(content_of "$OUT/rung-probe.json" 2>/dev/null || echo "")
  HITS=0
  while IFS= read -r ans; do [[ "$C" == *"$ans"* ]] && HITS=$((HITS+1)); done \
    < <(python3 -c 'import json,sys; print("\n".join(a for _,a in json.load(open(sys.argv[1]))))' "$OUT/depth-8000-facts.json")
  echo "- 8k-depth probe at the rung: $HITS/3 facts retrieved" >> "$P"
  touch "$OUT/.rung.done"
fi

# ---------------------------------------------------------------------------
# phase: duel — the sharp, short head-to-head. The SAME probes on production
# (read-only, run first while production is up) and on our stack (after one
# restart). Five instruments, ~4 min per engine, every number a direct
# measurement of the thing it names — no bench rows, no derivation:
#   1. solo decode rate, TWICE (the stability test): greedy, 2048 tokens,
#      completion/wall — the pair's delta is the instrument's noise floor;
#   2. inter-token latency from the SSE stream (per-chunk mean/p50/p95 — the
#      decode rhythm, where the drafter's accept/verify pattern shows);
#   3. prefill slope: max_tokens=1 at d0/d4096/d8192 — the d0 TTFT is the fixed
#      overhead, the deltas give ms/token of prefill;
#   4. the drafter's acceptance counters from /health (deltas around the decode
#      pair): drafted/accepted/rounds;
#   5. c2/c4 aggregate decode (the honest concurrency instrument, measured —
#      not derived — on both engines).
# Production arm skipped with a note when production is down; the handover
# prompt follows prodqual's pattern when it is up.
# ---------------------------------------------------------------------------
if want duel && ! phase_done duel; then
  log "=== duel (the sharp head-to-head: production first, then ours)"
  P="$OUT/07-duel.md"; : > "$P"
  echo "# duel (five direct instruments, same probes on both engines; no bench rows)" >> "$P"

  if prod_running; then
    PROD_MODEL=$(curl -s -m 10 "http://127.0.0.1:$PORT/v1/models" | python3 -c '
import json, sys
try: print(json.load(sys.stdin)["data"][0]["id"])
except Exception: print("")') || PROD_MODEL=""
    PROD_MODEL="${PROD_MODEL:-$SERVED_NAME}"
    echo "## production 0.6.6 (read-only; model $PROD_MODEL)" >> "$P"
    duel_arm prod "$PROD_MODEL"
    if [[ "${PROMPT:-1}" == 1 ]]; then
      log "production probed. NOW stop production (the -tip stack's stop.sh) — watch for its 'Stopped' lines and only THEN press Enter (Ctrl-C to abort):"
      read -r -p "…: " _
    fi
    prod_running && die "production still running — the ours arm needs the GPUs"
  else
    echo "- production not running: the prod arm skipped (run the duel again with production up for the head-to-head)" >> "$P"
  fi
  restart DENSE_BITS=4 HEAD_BITS=4 KV_CACHE=fp8 CONTEXT=524288 PARALLEL=4
  echo "## ours (dense4fp8 @ 512K x4)" >> "$P"
  duel_arm ours "$SERVED_NAME"
  touch "$OUT/.duel.done"
fi

# ---------------------------------------------------------------------------
# phase: confirm — the knob sweep's one sharp lead, reproduced or retired:
# TF_GLM_PROMPT_STEPS=0 took c2 @ d8192 aggregate 17.3 → 33.1 t/s (1.9×) in
# campaign-20261023-200302 on a row that is tight across five rounds. Two
# independent restarts + benches decide; the receipt prints the comparison
# against the dense4fp8 baseline explicitly.
# ---------------------------------------------------------------------------
if want confirm && ! phase_done confirm; then
  log "=== confirm (steps0 twice: does the 1.9x c2@d8192 lead reproduce?)"
  P="$OUT/08-confirm.md"; : > "$P"
  echo "# confirm (TF_GLM_PROMPT_STEPS=0, dense4fp8 @ 512K x4; two independent restarts)" >> "$P"
  echo "- the lead being tested: c2 @ d8192 aggregate 17.3 → 33.1 t/s (campaign-20261023-200302)" >> "$P"
  for r in a b; do
    S=$(date +%s)
    if restart DENSE_BITS=4 HEAD_BITS=4 KV_CACHE=fp8 CONTEXT=524288 PARALLEL=4 TF_GLM_PROMPT_STEPS=0; then
      echo "## run $r (load $(( $(date +%s) - S )) s)" >> "$P"
      bench_perf "confirm-$r" "$OUT/confirm-$r" 4 >> "$P"
    else
      echo "## run $r — refused to start" >> "$P"
      docker logs "$CONTAINER_NAME" 2>&1 | tail -2 >> "$P"
    fi
  done
  python3 - "$OUT" >> "$P" <<'PYEOF'
import glob, sys
out = sys.argv[1]
rows = {}
for f in sorted(glob.glob(f"{out}/confirm-*/2026/*/*/*.md")):
    run = f.split("/confirm-")[1].split("/")[0]
    for line in open(f):
        for probe in ("| pp2048 tg128 @ d0 |", "| pp2048 tg128 @ d8192 c2 |", "| pp2048 tg128 @ d8192 c4 |", "| pp2048 tg128 @ d4096 c2 |"):
            if line.startswith(probe):
                cells = [c.strip() for c in line.split("|")[1:-1]]
                rows[f"{run} {probe.replace('| pp2048 tg128 @','').strip()}"] = cells[1]
for k in sorted(rows):
    print(f"- {k}: tg {rows[k]} t/s")
print("- baseline (dense4fp8 receipts): d0 49.9; d4096 c2 24.0; d8192 c2 17.3; d8192 c4 13.3")
print("- the lead reproduces if d8192 c2 lands near 33 again; near 17 retires it")
PYEOF
  touch "$OUT/.confirm.done"
fi

# ---------------------------------------------------------------------------
# phase: prof4 (SESSION-SPECIFIC for the Front 1 investigation in
# docs/PERF_PLAN.md; retires when the concurrency front is decided). Two cells
# on the base model, both with TF_GLM_PROF=1 so the engine's verify lines
# print rows-per-call:
#   a) default sharing — the full duel arm, then a fresh c4 re-probe: the last
#      64-call verify window's row count answers whether our verify rounds
#      batch across streams at concurrency (Lever A);
#   b) TF_GLM_SHARED=0 (each window in its own forward, cuda_lanes.zig:46) —
#      the c4 aggregate vs the duel's 99.8 t/s says whether sharing helps or
#      thrashes at concurrency (Lever B).
# Duel baselines quoted from campaign-20261023-221204: solo 55.2 t/s, c2 72.1,
# c4 99.8, slope 0.662 ms/tok.
# ---------------------------------------------------------------------------
if want prof4 && ! phase_done prof4; then
  log "=== prof4 (verify batching at c4, then TF_GLM_SHARED=0)"
  P="$OUT/09-prof4.md"; : > "$P"
  echo "# prof4 (Front 1: verify batching + TF_GLM_SHARED=0; dense4fp8 @ 512K x4, PROF on)" >> "$P"
  echo "- duel baseline: solo 55.2 t/s, c2 72.1, c4 99.8, slope 0.662 ms/tok (campaign-20261023-221204)" >> "$P"
  if restart DENSE_BITS=4 HEAD_BITS=4 KV_CACHE=fp8 CONTEXT=524288 PARALLEL=4 TF_GLM_PROF=1; then
    echo "## cell a: default sharing" >> "$P"
    duel_arm shared "$SERVED_NAME"
    c4_reprobe shared
    echo "- verify lines (last 64-call window = the c4 re-probe; rows/call is the batching answer):" >> "$P"
    docker logs "$CONTAINER_NAME" 2>&1 | grep "glm prof verify" | tail -4 >> "$P" || true
  else
    echo "## cell a — refused to start" >> "$P"
    docker logs "$CONTAINER_NAME" 2>&1 | tail -5 >> "$P" || true
  fi
  if restart DENSE_BITS=4 HEAD_BITS=4 KV_CACHE=fp8 CONTEXT=524288 PARALLEL=4 TF_GLM_PROF=1 TF_GLM_SHARED=0; then
    echo "## cell b: TF_GLM_SHARED=0 (windows alone)" >> "$P"
    duel_arm s0 "$SERVED_NAME"
    c4_reprobe s0
    echo "- verify lines (windows alone; the wall/rows shape vs cell a is the thrash test):" >> "$P"
    docker logs "$CONTAINER_NAME" 2>&1 | grep "glm prof verify" | tail -4 >> "$P" || true
  else
    echo "## cell b — refused to start" >> "$P"
    docker logs "$CONTAINER_NAME" 2>&1 | tail -5 >> "$P" || true
  fi
  touch "$OUT/.prof4.done"
fi

# ---------------------------------------------------------------------------
# phase: top4 (SESSION-SPECIFIC; PERF_PLAN Front 2 Lever A, Experiment 1 — the
# AZ checkpoint's routing change alone, on our own checkpoint; retires when
# the AZ lever is decided). Builds a symlinked variant snapshot with
# num_experts_per_tok: 4 (blobs stay shared, so the ranks' byte cross-check
# passes; the MODEL_REVISION change re-runs prepare, which rsyncs the variant
# to the worker), serves it with PROF on, and runs the duel's probes plus a
# fresh c4 re-probe for the prof window. Serving viability is itself a
# receipt: an admission refusal prints its arithmetic and the phase records
# it. Perf only — the quality battery (gates + tool-eval + structured-output
# + TEB) is the quality round's job and is deliberately NOT run here.
# ---------------------------------------------------------------------------
if want top4 && ! phase_done top4; then
  log "=== top4 (AZ Experiment 1: top-4 routing on our own checkpoint)"
  P="$OUT/10-top4.md"; : > "$P"
  echo "# top4 (num_experts_per_tok 8 -> 4 on the base EXL3 checkpoint; dense4fp8 @ 512K x4, PROF on)" >> "$P"
  echo "- duel baseline (topk=8): solo 55.2 t/s, c2 72.1, c4 99.8, slope 0.662 ms/tok (campaign-20261023-221204)" >> "$P"
  VARIANT_REV="top4cfg"
  VARIANT_DIR="$HF_CACHE/hub/models--${MODEL_ID//\//--}/snapshots/$VARIANT_REV"
  # The variant's config must be a REAL file, never a symlink into the shared blobs:
  # cp -P preserves the snapshot's symlinks, and an edit through the symlink either
  # aliases the base checkpoint's config or silently fails to persist (campaign
  # 115242: the variant served topk=8 three times while declaring 4). Write a real
  # file and verify it reads 4 before anything serves.
  if [[ ! -f "$VARIANT_DIR/config.json" || -L "$VARIANT_DIR/config.json" ]]; then
    log "building the variant snapshot (symlinks into the shared blobs + a REAL edited config)"
    mkdir -p "$VARIANT_DIR"
    cp -P "$MODEL_SNAPSHOT"/* "$VARIANT_DIR"/ || die "variant snapshot build failed"
    rm -f "$VARIANT_DIR/config.json"
    # Surgical BYTE replacement — never a json round-trip: the engine's config parser
    # rejected a python-dumped config, and weightShare's catch-return-total then billed
    # the FULL checkpoint (163.6 GiB) to the budget (campaign 125609's refusal).
    python3 - "$MODEL_SNAPSHOT/config.json" "$VARIANT_DIR/config.json" <<'EOF'
import sys
data = open(sys.argv[1], "rb").read()
old = b'"num_experts_per_tok": 8'
assert data.count(old) == 1, f"expected exactly one occurrence, found {data.count(old)}"
open(sys.argv[2], "wb").write(data.replace(old, b'"num_experts_per_tok": 4'))
print("wrote a real config (byte-surgical): num_experts_per_tok 8 -> 4")
EOF
  fi
  python3 - "$VARIANT_DIR/config.json" <<'EOF'
import json, sys
tc = json.load(open(sys.argv[1])).get("text_config", {})
v = tc.get("num_experts_per_tok")
assert v == 4, f"the variant config reads topk={v} — refusing to serve an unedited variant"
print("variant config verified: num_experts_per_tok = 4")
EOF
  # prepare.sh only rsyncs a snapshot it downloaded itself; a locally-built variant
  # must be pushed to the worker explicitly (the ranks cross-check bytes at start).
  rsync -a "$VARIANT_DIR/" "$WORKER:$VARIANT_DIR/" || die "variant rsync to the worker failed"
  S=$(date +%s)
  if restart DENSE_BITS=4 HEAD_BITS=4 KV_CACHE=fp8 CONTEXT=524288 PARALLEL=4 MODEL_REVISION="$VARIANT_REV" TF_GLM_PROF=1; then
    echo "## serving (load $(( $(date +%s) - S )) s; the base load ran ~257 s)" >> "$P"
    TOP4_MODEL=$(curl -s -m 10 "http://127.0.0.1:$PORT/v1/models" | python3 -c '
import json, sys
try: print(json.load(sys.stdin)["data"][0]["id"])
except Exception: print("")') || TOP4_MODEL=""
    TOP4_MODEL="${TOP4_MODEL:-$SERVED_NAME}"
    duel_arm top4 "$SERVED_NAME"
    c4_reprobe top4
    echo "- acceptance counters (the topk=4 serve; ours ran 77.2%% at topk=8, production 62.7%%):" >> "$P"
    docker logs "$CONTAINER_NAME" 2>&1 | grep -oE "accepted=[0-9]+/[0-9]+ rounds=[0-9]+[^$]*" | tail -4 >> "$P" || true
    curl -s -m 5 "http://127.0.0.1:$PORT/metrics" 2>/dev/null | grep -E "mtp_drafted_total|mtp_accepted_total" | head -2 >> "$P" || true
    echo "- verify lines (routed share vs the knobs receipt: 18.45 ms of 40.08 at 2 rows, topk=8):" >> "$P"
    docker logs "$CONTAINER_NAME" 2>&1 | grep "glm prof verify" | tail -4 >> "$P" || true
  else
    echo "## the variant did NOT serve — the refusal/failure is the receipt (container log tail):" >> "$P"
    docker logs "$CONTAINER_NAME" 2>&1 | tail -15 >> "$P" || true
  fi
  touch "$OUT/.top4.done"
fi

# ---------------------------------------------------------------------------
# phase: routerks2 (SESSION-SPECIFIC; PERF_PLAN Front 2 Lever A2 — the
# prompt-router pathology A/B; retires when the lever is decided). Two cells
# on the default revision, both PROF-on:
#   a) stock — the A/B baseline: the duel arm, the 2,048-row chunk prof lines,
#      and the drafter acceptance lines (request log + /metrics) for Front 1;
#   b) the patched build — PATCHES=glm-router-ks2.patch,glm-slot-leak.patch
#      (the slot-leak fix rides along, bit-invisible) — the same probes plus
#      the content-identity gate: the patched greedy reply must equal the
#      stock cell's (the router-ks change is bit-visible by design; the gate
#      decides the patch, like #1079).
# Baselines: duel 221204 (solo 55.2, c4 99.8, slope 0.662); chunk prof from
# 145449 + the 235008 salvage (router 910–1125 of 1641–2040 ms per chunk).
# ---------------------------------------------------------------------------
if want routerks2 && ! phase_done routerks2; then
  log "=== routerks2 (the prompt-router A/B: stock, then the ks2 patch)"
  P="$OUT/11-routerks2.md"; : > "$P"
  echo "# routerks2 (Front 2 Lever A2: the prompt router; dense4fp8 @ 512K x4, PROF on)" >> "$P"
  echo "- baselines: duel 221204 solo 55.2 / c4 99.8 / slope 0.662; chunk prof router 910-1125 of 1641-2040 ms (145449, 235008)" >> "$P"
  routerks2_cell() {  # $1 tag: the duel arm + the chunk prof + the c4 re-probe
    local tag="$1"
    duel_arm "$tag" "$SERVED_NAME"
    c4_reprobe "$tag"
    echo "- chunk prof lines (the d8192 prefill pass's 2048-row chunks; the router share is the A/B read):" >> "$P"
    docker logs "$CONTAINER_NAME" 2>&1 | grep "glm prof prompt chunk" | tail -3 >> "$P" || true
  }
  if restart DENSE_BITS=4 HEAD_BITS=4 KV_CACHE=fp8 CONTEXT=524288 PARALLEL=4 TF_GLM_PROF=1; then
    echo "## cell a: stock" >> "$P"
    routerks2_cell stock
    echo "- acceptance lines (the c4 probe's request log; the Front 1 counters):" >> "$P"
    docker logs "$CONTAINER_NAME" 2>&1 | grep -oE "accepted=[0-9]+/[0-9]+ rounds=[0-9]+[^$]*" | tail -6 >> "$P" || true
    curl -s -m 5 "http://127.0.0.1:$PORT/metrics" 2>/dev/null | grep -iE "draft|accept|round" | head -6 >> "$P" || echo "- (no /metrics drafter lines)" >> "$P"
    cp "$OUT/duel-stock-solo-a.json" "$OUT/gate-stock-reply.json"
  else
    echo "## cell a — refused to start" >> "$P"
    docker logs "$CONTAINER_NAME" 2>&1 | tail -5 >> "$P" || true
  fi
  if restart DENSE_BITS=4 HEAD_BITS=4 KV_CACHE=fp8 CONTEXT=524288 PARALLEL=4 TF_GLM_PROF=1 PATCHES="glm-router-ks2.patch,glm-slot-leak.patch"; then
    echo "## cell b: patched (glm-router-ks2 + glm-slot-leak)" >> "$P"
    routerks2_cell patched
    python3 - "$OUT" >> "$P" <<'PYEOF'
import hashlib, json, sys
out = sys.argv[1]
try:
    a = json.load(open(f"{out}/gate-stock-reply.json"))
    b = json.load(open(f"{out}/duel-patched-solo-a.json"))
    ha = hashlib.sha256(json.dumps(a["choices"], sort_keys=True).encode()).hexdigest()
    hb = hashlib.sha256(json.dumps(b["choices"], sort_keys=True).encode()).hexdigest()
    print("- content gate (greedy reply, stock vs patched): %s" % ("IDENTICAL" if ha == hb else "DIFFERS - the ks change flipped a token; the patch dies here"))
except Exception as e:
    print("- content gate could not run: %s" % e)
PYEOF
  else
    echo "## cell b — the build or the serve failed (the receipt):" >> "$P"
    docker logs "$CONTAINER_NAME" 2>&1 | tail -5 >> "$P" || true
  fi
  touch "$OUT/.routerks2.done"
fi

# ---------------------------------------------------------------------------
# phase: routerpart (SESSION-SPECIFIC; PERF_PLAN Front 2 Lever A2, probe two —
# retires with the lever). Forces the UNSTAGED router kernel (router_part) for
# every launch: same ks slicing and summation order as router_tile, so the math
# should be bit-identical — the content gate (vs the 115242 stock reply)
# verifies, and the chunk-cost read then splits the pathology:
#   part ≈ tile (~1100 ms/chunk)  → both kernels share it → replace with a
#          real GEMM (the cuda_mm class);
#   part >> tile                  → the tile is the better kernel and the fix
#          is its K-depth/w-layout.
# glm-slot-leak.patch rides this build (bit-invisible). Baselines: 115242 cell
# a (solo 55.3, c4 101.8, slope 0.659, chunk router 1096 ms of 1967 gpu).
# ---------------------------------------------------------------------------
if want routerpart && ! phase_done routerpart; then
  log "=== routerpart (the router kernel A/B: force the unstaged fallback)"
  P="$OUT/12-routerpart.md"; : > "$P"
  echo "# routerpart (Front 2 A2 probe two: router_part forced; dense4fp8 @ 512K x4, PROF on)" >> "$P"
  echo "- baselines: 115242 cell a solo 55.3 / c4 101.8 / slope 0.659; chunk router 1096 ms of 1967 gpu" >> "$P"
  if restart DENSE_BITS=4 HEAD_BITS=4 KV_CACHE=fp8 CONTEXT=524288 PARALLEL=4 TF_GLM_PROF=1 PATCHES="glm-router-part.patch,glm-slot-leak.patch"; then
    duel_arm rpart "$SERVED_NAME"
    c4_reprobe rpart
    echo "- chunk prof lines (the A/B read vs 115242's router 1096 ms):" >> "$P"
    docker logs "$CONTAINER_NAME" 2>&1 | grep "glm prof prompt chunk" | tail -3 >> "$P" || true
    python3 - "$OUT" >> "$P" <<'PYEOF'
import hashlib, json, sys
out = sys.argv[1]
try:
    a = json.load(open("results/campaign-20261024-115242/duel-stock-solo-a.json"))
    b = json.load(open(f"{out}/duel-rpart-solo-a.json"))
    ha = hashlib.sha256(json.dumps(a["choices"], sort_keys=True).encode()).hexdigest()
    hb = hashlib.sha256(json.dumps(b["choices"], sort_keys=True).encode()).hexdigest()
    print("- content gate (greedy reply, 115242 stock vs rpart): %s" % ("IDENTICAL - the swap is bit-safe; the cost read above decides the pathology" if ha == hb else "DIFFERS - the swap is NOT bit-safe (the summation-order reasoning was wrong); treat the cost read as suspect"))
except Exception as e:
    print("- content gate could not run: %s" % e)
PYEOF
  else
    echo "## cell — the build or the serve failed (the receipt):" >> "$P"
    docker logs "$CONTAINER_NAME" 2>&1 | tail -5 >> "$P" || true
  fi
  touch "$OUT/.routerpart.done"
fi

# ---------------------------------------------------------------------------
# phase: top4qual (SESSION-SPECIFIC; the quality battery on real topk=4 —
# PERF_PLAN Front 2 Lever A's gate; retires with the lever's decision).
# The (user) bar: a new checkpoint/model behavior means seriously more quality
# checks. The battery on the top4cfg serve (PATCHES=glm-topk4b):
#   1) the §2 gates, content form (greedy determinism, drafted==plain,
#      cache-hit==computed, concurrency, seeded reproducibility);
#   2) the full tool-eval suite (run-bench-matrix PHASE=tools — the
#      structured-output scenarios ride it) — the reference is the stock
#      dense4fp8 cell's 92/100 (125/136 pts, 0 errors);
#   3) hardmode (the Hard-Mode suite — 35 scenarios on bench v3.0.1);
#   4) the duel arm + c4 reprobe — the perf probes for the scoreboard refresh.
# A battery failure sends the lever to the AZ full checkpoint (the retrained
# selector) instead of a default change.
# ---------------------------------------------------------------------------
if want top4qual && ! phase_done top4qual; then
  log "=== top4qual (the quality battery on real topk=4)"
  P="$OUT/13-top4qual.md"; : > "$P"
  echo "# top4qual (the quality battery on real topk=4; dense4fp8 @ 512K x4)" >> "$P"
  echo "- reference: production's v3.0.1 anchors (round 35): tool-eval 96/100 (133/138), hardmode subset 63/70 = 90%; the top4cfg's own rows are v1.1.2-era (93/89, old bench) until this battery" >> "$P"
  echo "- the model behavior changes by design (the unretrained top-4 selector): the gates and the score decide" >> "$P"
  run_gates top4cfg MODEL_REVISION=top4cfg PATCHES=glm-topk4b.patch DENSE_BITS=4 HEAD_BITS=4 KV_CACHE=fp8 CONTEXT=524288 || true
  # the duel arm + c4 reprobe (round 36 addition): the scoreboard's solo/c2/c4/TTFT
  # rows need the top4cfg's own v1.1.6 steady-state probes against production's
  # fresh duel receipt (round 33 addendum) — the scores alone don't move those rows.
  if restart MODEL_REVISION=top4cfg PATCHES=glm-topk4b.patch DENSE_BITS=4 HEAD_BITS=4 KV_CACHE=fp8 CONTEXT=524288 PARALLEL=4; then
    T4MODEL=$(curl -s -m 10 "http://127.0.0.1:$PORT/v1/models" | python3 -c '
import json, sys
try: print(json.load(sys.stdin)["data"][0]["id"])
except Exception: print("")') || T4MODEL=""
    T4MODEL="${T4MODEL:-$SERVED_NAME}"
    duel_arm top4cfg "$T4MODEL"
    c4_reprobe top4cfg
  else
    echo "- the top4cfg serve refused on restart — the receipt (container log tail):" >> "$P"
    docker logs "$CONTAINER_NAME" 2>&1 | tail -8 >> "$P" || true
  fi
  echo "- the tool-eval suite rides the hardmode run below (the (user) hardmode-only directive: no separate short run)" >> "$P"
  echo "- hardmode (the Hard-Mode suite):" >> "$P"
  if PHASE=hardmode scripts/run-bench-matrix.sh top4 > "$OUT/bench-hardmode.log" 2>&1; then
    grep -E "/100|pts|errors|score|PASS|FAIL" "$OUT/bench-hardmode.log" | head -10 >> "$P" || tail -10 "$OUT/bench-hardmode.log" >> "$P"
  else
    echo "- the hardmode bench FAILED — its tail:" >> "$P"; tail -8 "$OUT/bench-hardmode.log" >> "$P"
  fi
  touch "$OUT/.top4qual.done"
fi

# ---------------------------------------------------------------------------
# phase: azqual (SESSION-SPECIFIC; the battery + the probes on the AZ-EXL3
# serve — the retrained selector, Experiment 2; retires with the AZ route's
# decision). The serve: local/GLM-5.3-Flash-AZ-EXL3 (the requant output, placed
# by scripts/az-window.sh) under the topk=4 reference (glm-topk4b). The battery
# mirrors top4qual's: §2 gates, the full tool-eval, hardmode; then the duel arm
# for the perf probes. The references: top4cfg's 135458/142659 receipts.
# ---------------------------------------------------------------------------
if want azqual && ! phase_done azqual; then
  log "=== azqual (the AZ retrained selector: the battery + the probes)"
  P="$OUT/14-azqual.md"; : > "$P"
  echo "# azqual (the AZ-EXL3 serve: the retrained top-4 selector; dense4fp8 @ 512K x4)" >> "$P"
  echo "- references: top4cfg perf (135458: solo 63.6, c2 90.0, c4 120.4, slope 0.593) and battery (142659: gates PASS, tool-eval 93, hardmode 89)" >> "$P"
  AZENV="MODEL_ID=local/GLM-5.3-Flash-AZ-EXL3 MODEL_REVISION=main PATCHES=glm-topk4b.patch"
  run_gates azexl3 $AZENV DENSE_BITS=4 HEAD_BITS=4 KV_CACHE=fp8 CONTEXT=524288 || true
  if restart $AZENV DENSE_BITS=4 HEAD_BITS=4 KV_CACHE=fp8 CONTEXT=524288 PARALLEL=4; then
    AZMODEL=$(curl -s -m 10 "http://127.0.0.1:$PORT/v1/models" | python3 -c '
import json, sys
try: print(json.load(sys.stdin)["data"][0]["id"])
except Exception: print("")') || AZMODEL=""
    AZMODEL="${AZMODEL:-$SERVED_NAME}"
    duel_arm azexl3 "$AZMODEL"
    c4_reprobe azexl3
  else
    echo "- the AZ serve refused — the receipt (container log tail):" >> "$P"
    docker logs "$CONTAINER_NAME" 2>&1 | tail -8 >> "$P" || true
  fi
  echo "- the tool-eval suite rides the hardmode run below (the (user) hardmode-only directive)" >> "$P"
  echo "- hardmode (the Hard-Mode suite):" >> "$P"
  if PHASE=hardmode scripts/run-bench-matrix.sh azexl3 > "$OUT/bench-hardmode.log" 2>&1; then
    grep -E "/100|pts|errors|score|PASS|FAIL" "$OUT/bench-hardmode.log" | head -10 >> "$P" || true
  else
    echo "- the hardmode bench FAILED — its tail:" >> "$P"; tail -8 "$OUT/bench-hardmode.log" >> "$P"
  fi
  touch "$OUT/.azqual.done"
fi

# ---------------------------------------------------------------------------
# phase: focusqual (SESSION-SPECIFIC; the battery + the probes on AxonZeta's
# AZ-Focus-EXL3 — the reasoning-efficiency checkpoint of the thinking-
# efficiency track; retires with that track's decision). The serve: the
# AxonZeta checkpoint as downloaded (mcg-coded, integer bits — admitted by the
# engine; topk=8 by design, no patch). The battery mirrors top4qual's: §2
# gates, the full tool-eval, hardmode; then the duel arm for the perf probes.
# The references: production (tool-eval 95, hardmode 91) and top4cfg (93/89).
# ---------------------------------------------------------------------------
if want focusqual && ! phase_done focusqual; then
  log "=== focusqual (AxonZeta's AZ-Focus-EXL3: the battery + the probes)"
  P="$OUT/15-focusqual.md"; : > "$P"
  echo "# focusqual (the AZ-Focus-EXL3 serve: the reasoning-efficiency checkpoint; dense4fp8 @ 512K x4)" >> "$P"
  echo "- references: production's v3.0.1 anchors (round 35): tool-eval 96/100 (133/138), hardmode subset 63/70 = 90%; the Focus's own perf receipts: duel 94.3 c2 / 150.6 c4 (round 33 A/B)" >> "$P"
  FOCUSENV="MODEL_ID=local/GLM-5.3-Flash-AZ-Focus-EXL3 MODEL_REVISION=main PATCHES="
  run_gates focus $FOCUSENV DENSE_BITS=4 HEAD_BITS=4 KV_CACHE=fp8 CONTEXT=524288 || true
  if restart $FOCUSENV DENSE_BITS=4 HEAD_BITS=4 KV_CACHE=fp8 CONTEXT=524288 PARALLEL=4; then
    FOCMODEL=$(curl -s -m 10 "http://127.0.0.1:$PORT/v1/models" | python3 -c '
import json, sys
try: print(json.load(sys.stdin)["data"][0]["id"])
except Exception: print("")') || FOCMODEL=""
    FOCMODEL="${FOCMODEL:-$SERVED_NAME}"
    duel_arm focus "$FOCMODEL"
    c4_reprobe focus
  else
    echo "- the Focus serve refused — the receipt (container log tail):" >> "$P"
    docker logs "$CONTAINER_NAME" 2>&1 | tail -8 >> "$P" || true
  fi
  echo "- the tool-eval suite rides the hardmode run below (the (user) hardmode-only directive)" >> "$P"
  echo "- hardmode (the Hard-Mode suite):" >> "$P"
  if PHASE=hardmode scripts/run-bench-matrix.sh focus > "$OUT/bench-hardmode.log" 2>&1; then
    grep -E "/100|pts|errors|score|PASS|FAIL" "$OUT/bench-hardmode.log" | head -10 >> "$P" || true
  else
    echo "- the hardmode bench FAILED — its tail:" >> "$P"; tail -8 "$OUT/bench-hardmode.log" >> "$P"
  fi
  touch "$OUT/.focusqual.done"
fi

# ---------------------------------------------------------------------------
# phase: hmtrials (SESSION-SPECIFIC; the beat-94 variance check behind the
# (user) hardmode bar — retires once the variant spread around the Focus's 92
# is known). Greedy sampling is content-deterministic per variant seed, so
# re-running the same seed reproduces the score exactly; the spread that
# matters is ACROSS scenario variants. Three hardmode runs at --variant-seed
# 7/8/9 (plus round 37's default-seed 92) give the score's spread and the
# per-scenario flip list: if 90-93 is the band, the 2-point gap to
# production's 94 is reach; if 86-88, 92 was a lucky draw.
# ---------------------------------------------------------------------------
if want hmtrials && ! phase_done hmtrials; then
  log "=== hmtrials (hardmode across variant seeds on the served default)"
  P="$OUT/16-hmtrials.md"; : > "$P"
  echo "# hmtrials (the hardmode suite across variant seeds; the served default)" >> "$P"
  echo "- references: production run-level 94 (round 35, default variants); the Focus 92 (round 37, default variants)" >> "$P"
  if restart DENSE_BITS=4 HEAD_BITS=4 KV_CACHE=fp8 CONTEXT=524288 PARALLEL=4; then
    HM=$(curl -s -m 10 "http://127.0.0.1:$PORT/v1/models" | python3 -c '
import json, sys
try: print(json.load(sys.stdin)["data"][0]["id"])
except Exception: print("")') || HM=""
    HM="${HM:-$SERVED_NAME}"
    OUTD="$OUT/hmtrials"; mkdir -p "$OUTD"
    for vs in 7 8 9; do
      echo "- variant-seed $vs:" >> "$P"
      PYTHONUNBUFFERED=1 "$BENCH" run --base-url "http://127.0.0.1:$PORT" \
        --model "$HM" --format openai --tokenizer "$MODEL_ID" \
        --seed 42 --variant-seed "$vs" --perf --parallel 2 --timeout 240 \
        --hardmode --label "hm-vs$vs" --output-dir "$OUTD" \
        --json-file "$OUTD/hm-vs$vs.json" > "$OUTD/hm-vs$vs.log" 2>&1 || true
      grep -oE '"final_score": [0-9]+' "$OUTD/hm-vs$vs.json" >> "$P" 2>/dev/null || true
      grep -oE '"scenario_id": "TC-[0-9]+", "status": "fail"' "$OUTD/hm-vs$vs.json" 2>/dev/null | grep -oE "TC-[0-9]+" | sort -u | tr '\n' ' ' >> "$P" || true
      echo >> "$P"
      tail -2 "$OUTD/hm-vs$vs.log" >> "$P" 2>/dev/null || true
    done
  else
    echo "- the serve refused — the receipt (container log tail):" >> "$P"
    docker logs "$CONTAINER_NAME" 2>&1 | tail -8 >> "$P" || true
  fi
  touch "$OUT/.hmtrials.done"
fi

# ---------------------------------------------------------------------------
# phase: settingsab (SESSION-SPECIFIC; the sampling-settings A/B — retires once
# the recommended-settings receipt exists). The bench runs greedy (its default
# temperature is 0.0) while GLM-5.3-Flash's recommended agentic settings are
# temperature 1.0 / top_p 0.95 / max reasoning effort (Z.ai docs, Together,
# Unsloth — max effort is the model's default, so sampling is the only lever
# the bench exposes). One hardmode run at the recommended settings on the
# served default compares against the greedy receipts: the Focus 92 (round 37)
# and production 94 (round 35 — production was ALSO measured greedy, so any
# lift is ours to take without touching the comparison's fairness).
# ---------------------------------------------------------------------------
if want settingsab && ! phase_done settingsab; then
  log "=== settingsab (hardmode at the recommended sampling settings)"
  P="$OUT/17-settingsab.md"; : > "$P"
  echo "# settingsab (hardmode: greedy vs temperature 1.0 / top_p 0.95)" >> "$P"
  echo "- references: the Focus greedy 92 (round 37); production greedy 94 (round 35)" >> "$P"
  if restart DENSE_BITS=4 HEAD_BITS=4 KV_CACHE=fp8 CONTEXT=524288 PARALLEL=4; then
    SM=$(curl -s -m 10 "http://127.0.0.1:$PORT/v1/models" | python3 -c '
import json, sys
try: print(json.load(sys.stdin)["data"][0]["id"])
except Exception: print("")') || SM=""
    SM="${SM:-$SERVED_NAME}"
    OUTD="$OUT/settingsab"; mkdir -p "$OUTD"
    PYTHONUNBUFFERED=1 "$BENCH" run --base-url "http://127.0.0.1:$PORT" \
      --model "$SM" --format openai --tokenizer "$MODEL_ID" \
      --seed 42 --temperature 1.0 --top-p 0.95 --perf --parallel 2 --timeout 240 \
      --hardmode --label "hm-rec" --output-dir "$OUTD" \
      --json-file "$OUTD/hm-rec.json" > "$OUTD/hm-rec.log" 2>&1 || true
    grep -oE '"final_score": [0-9]+' "$OUTD/hm-rec.json" >> "$P" 2>/dev/null || true
    tail -2 "$OUTD/hm-rec.log" >> "$P" 2>/dev/null || true
    echo "- NOTE: sampled runs are not content-deterministic — one run is a draw, not a receipt; the greedy 92 stays the deterministic anchor. A lift here is a harness-settings receipt, not an engine change." >> "$P"
  else
    echo "- the serve refused — the receipt (container log tail):" >> "$P"
    docker logs "$CONTAINER_NAME" 2>&1 | tail -8 >> "$P" || true
  fi
  touch "$OUT/.settingsab.done"
fi

# ---------------------------------------------------------------------------
# phase: markspatch (SESSION-SPECIFIC; the round-34 frontier fix — retires with
# the marks-policy decision). Serves PATCHES=glm-prefix-marks.patch: planned
# families keep grid-aligned interior prompt states, so a later request sharing
# a token prefix resumes instead of re-prefilling (cross-request prefixes never
# matched under the conversation-structure marks). Instruments: the §2 gates
# first (exactness around the new resume path), then the duel arm — the TTFT
# slope at depth is the direct instrument. Baseline without the patch (round
# 37): solo 57.4, c2 94.4, c4 153.3, TTFT d0 203 ms, slope 0.665 ms/tok
# (d8192: 5,647 ms).
# ---------------------------------------------------------------------------
if want markspatch && ! phase_done markspatch; then
  log "=== markspatch (glm-prefix-marks.patch: gates + duel)"
  P="$OUT/18-markspatch.md"; : > "$P"
  echo "# markspatch (the interior-marks patch: gates + duel vs the round-37 baseline)" >> "$P"
  echo "- baseline (no patch, round 37): solo 57.4/57.5, c2 94.4, c4 153.3, TTFT d0 203 ms, slope 0.665 ms/tok, d8192 5,647 ms" >> "$P"
  echo "- the round-40 verdict: gates PASS but the duel didn't move — the duel's single cold requests can't see the resume; this run adds the two-shot probe (the same long prompt twice: shot2 << shot1 is the resume working)" >> "$P"
  # the unpatched probe first (the pristine serve's signature, probe-before-anything: cold readings)
  if restart MODEL_ID=local/GLM-5.3-Flash-AZ-Focus-EXL3 MODEL_REVISION=main PATCHES= DENSE_BITS=4 HEAD_BITS=4 KV_CACHE=fp8 CONTEXT=524288 PARALLEL=4; then
    PRMODEL=$(curl -s -m 10 "http://127.0.0.1:$PORT/v1/models" | python3 -c '
import json, sys
try: print(json.load(sys.stdin)["data"][0]["id"])
except Exception: print("")') || PRMODEL=""
    PRMODEL="${PRMODEL:-$SERVED_NAME}"
    echo "- the A/A/B probe, UNPATCHED (probe-first: A1 cold, A2 same-prompt resume, B1 cross-prompt):" >> "$P"
    scripts/twoshot-probe.sh "http://127.0.0.1:$PORT" "$PRMODEL" 2 >> "$P" 2>&1 || echo "- the probe failed" >> "$P"
    echo "- the UNPATCHED duel follows (the pages-vs-patch discriminator: its d4096/d8192 TTFTs come AFTER the probe's fill prefill — if they drop like the patched arm's did, the pages do the work and the patch is redundant):" >> "$P"
    duel_arm focusmp "$PRMODEL"
    c4_reprobe focusmp
  fi
  if run_gates focusmp MODEL_ID=local/GLM-5.3-Flash-AZ-Focus-EXL3 MODEL_REVISION=main PATCHES=glm-prefix-marks.patch DENSE_BITS=4 HEAD_BITS=4 KV_CACHE=fp8 CONTEXT=524288; then
    if restart MODEL_ID=local/GLM-5.3-Flash-AZ-Focus-EXL3 MODEL_REVISION=main PATCHES=glm-prefix-marks.patch DENSE_BITS=4 HEAD_BITS=4 KV_CACHE=fp8 CONTEXT=524288 PARALLEL=4; then
      MPMODEL=$(curl -s -m 10 "http://127.0.0.1:$PORT/v1/models" | python3 -c '
import json, sys
try: print(json.load(sys.stdin)["data"][0]["id"])
except Exception: print("")') || MPMODEL=""
      MPMODEL="${MPMODEL:-$SERVED_NAME}"
      echo "- the A/A/B probe, PATCHED (probe-first: cold readings — the round-41 confound removed):" >> "$P"
      scripts/twoshot-probe.sh "http://127.0.0.1:$PORT" "$MPMODEL" 2 >> "$P" 2>&1 || echo "- the probe failed" >> "$P"
      duel_arm focusmp "$MPMODEL"
      c4_reprobe focusmp
      echo "- the A/A/B probe, PATCHED post-duel (A1 shows whether the duel's fill states resume — the cross-fill case):" >> "$P"
      scripts/twoshot-probe.sh "http://127.0.0.1:$PORT" "$MPMODEL" 1 >> "$P" 2>&1 || echo "- the probe failed" >> "$P"
    else
      echo "- the patched serve refused on restart — the receipt (container log tail):" >> "$P"
      docker logs "$CONTAINER_NAME" 2>&1 | tail -8 >> "$P" || true
    fi
  else
    echo "- the patched serve refused for the gates — the receipt (container log tail):" >> "$P"
    docker logs "$CONTAINER_NAME" 2>&1 | tail -8 >> "$P" || true
  fi
  touch "$OUT/.markspatch.done"
fi

# ---------------------------------------------------------------------------
# phase: v118 (SESSION-SPECIFIC; the v1.1.8 adoption battery — retires with the
# adoption decision). The pin is bumped to v1.1.8 (812fb040) with upstream's
# fork rule opted in (PATCHES=glm-fork-rule.patch, snap_fork #1291). The marks
# patch is RETIRED (round 43: the pages do the resume; the patch is redundant).
# The battery: (1) the §2 gates on the new engine — exactness first;
# (2) the A/A/B probe on v1.1.8+fork — vs the stock readings (A2 ~1,434 /
# B1 ~1,435 ms): does the fork rule improve the cross-prompt case?
# (3) the INDEX_CUDA A/B — the duel with TF_GLM_INDEX_CUDA unset vs =1 (the
# DSA indexer kernels, #1449: upstream measured cold-TTFT −2%/−9%/−16% at
# 131k/524k/1M; our duel slope is the instrument at our depths);
# (4) hardmode at --parallel 2 — the score receipt on v1.1.8 (receipts-
# equal-or-better vs the Focus's 92 decides the adoption).
# ---------------------------------------------------------------------------
if want v118 && ! phase_done v118; then
  log "=== v118 (the v1.1.8 adoption battery: gates + the fork probe + the INDEX_CUDA A/B + hardmode)"
  P="$OUT/19-v118.md"; : > "$P"
  echo "# v118 (engine v1.1.8 + the fork rule: the adoption battery)" >> "$P"
  echo "- references: v1.1.6 Focus receipts — hardmode 92 (rounds 37/40); the duel slope 0.665 cold / ~0.156 after a same-fill prefill (rounds 41-43); the A/A/B stock readings A2 ~1,434 / B1 ~1,435 ms; production's resume reference 65 ms" >> "$P"
  VENV="MODEL_ID=local/GLM-5.3-Flash-AZ-Focus-EXL3 MODEL_REVISION=main PATCHES=glm-fork-rule.patch"
  if run_gates focus118 $VENV DENSE_BITS=4 HEAD_BITS=4 KV_CACHE=fp8 CONTEXT=524288; then
    if restart $VENV DENSE_BITS=4 HEAD_BITS=4 KV_CACHE=fp8 CONTEXT=524288 PARALLEL=4; then
      V1MODEL=$(curl -s -m 10 "http://127.0.0.1:$PORT/v1/models" | python3 -c '
import json, sys
try: print(json.load(sys.stdin)["data"][0]["id"])
except Exception: print("")') || V1MODEL=""
      V1MODEL="${V1MODEL:-$SERVED_NAME}"
      echo "- the A/A/B probe on v1.1.8 + the fork rule (vs the stock A2 ~1,434 / B1 ~1,435):" >> "$P"
      scripts/twoshot-probe.sh "http://127.0.0.1:$PORT" "$V1MODEL" 2 >> "$P" 2>&1 || echo "- the probe failed" >> "$P"
      duel_arm focus118 "$V1MODEL"
      c4_reprobe focus118
    else
      echo "- the v1.1.8 serve refused on restart — the receipt (container log tail):" >> "$P"
      docker logs "$CONTAINER_NAME" 2>&1 | tail -8 >> "$P" || true
    fi
    if restart $VENV TF_GLM_INDEX_CUDA=1 DENSE_BITS=4 HEAD_BITS=4 KV_CACHE=fp8 CONTEXT=524288 PARALLEL=4; then
      V2MODEL=$(curl -s -m 10 "http://127.0.0.1:$PORT/v1/models" | python3 -c '
import json, sys
try: print(json.load(sys.stdin)["data"][0]["id"])
except Exception: print("")') || V2MODEL=""
      V2MODEL="${V2MODEL:-$SERVED_NAME}"
      echo "- the INDEX_CUDA=1 duel (the DSA indexer kernels; compare the TTFT slope to the unset arm above):" >> "$P"
      duel_arm focus118ix "$V2MODEL"
      c4_reprobe focus118ix
    else
      echo "- the INDEX_CUDA serve refused — the receipt (container log tail):" >> "$P"
      docker logs "$CONTAINER_NAME" 2>&1 | tail -8 >> "$P" || true
    fi
    if restart $VENV DENSE_BITS=4 HEAD_BITS=4 KV_CACHE=fp8 CONTEXT=524288 PARALLEL=4; then
      HM=$(curl -s -m 10 "http://127.0.0.1:$PORT/v1/models" | python3 -c '
import json, sys
try: print(json.load(sys.stdin)["data"][0]["id"])
except Exception: print("")') || HM=""
      HM="${HM:-$SERVED_NAME}"
      OUTD="$OUT/v118"; mkdir -p "$OUTD"
      echo "- hardmode on v1.1.8+fork (--parallel 2, the (user) bench shape):" >> "$P"
      PYTHONUNBUFFERED=1 "$BENCH" run --base-url "http://127.0.0.1:$PORT" \
        --model "$HM" --format openai --tokenizer "$MODEL_ID" \
        --seed 42 --perf --parallel 2 --timeout 240 \
        --hardmode --label "v118-hm" --output-dir "$OUTD" \
        --json-file "$OUTD/v118-hm.json" > "$OUTD/v118-hm.log" 2>&1 || true
      grep -oE '"final_score": [0-9]+' "$OUTD/v118-hm.json" >> "$P" 2>/dev/null || true
      tail -2 "$OUTD/v118-hm.log" >> "$P" 2>/dev/null || true
    fi
  else
    echo "- the v1.1.8 serve refused for the gates — the receipt (container log tail):" >> "$P"
    docker logs "$CONTAINER_NAME" 2>&1 | tail -8 >> "$P" || true
  fi
  touch "$OUT/.v118.done"
fi

# ---------------------------------------------------------------------------
# phase: indexdepth (SESSION-SPECIFIC; retires once the INDEX_CUDA receipt is
# in). The owed round-45 item: TF_GLM_INDEX_CUDA=1 (#1449 — the DSA indexer
# kernels) A/B at depth. The 8k duel cannot resolve it (upstream's own curve
# puts the gains at 131k/524k/1M: -2%/-9%/-16% cold TTFT). The instrument: the
# TTFT ladder 65536..393216 (max_tokens=1) plus one 512-token decode row at
# 96k, on the standing config, each arm a fresh serve (identical pre-histories).
# The ladder fills are per-row distinct from token ~1 (decoration keyed on
# (i + depth)) — the duel's generator makes its rows token-prefixes of each
# other and is deliberately NOT reused: no ladder row resumes another's pages,
# and the prompt-cache counters double as the no-sharing proof (one miss per
# row).
# ---------------------------------------------------------------------------
if want indexdepth && ! phase_done indexdepth; then
  log "=== indexdepth (the TF_GLM_INDEX_CUDA A/B at depth: TTFT ladder 64k-384k)"
  P="$OUT/20-indexdepth.md"; : > "$P"
  echo "# indexdepth (TF_GLM_INDEX_CUDA A/B at depth — the owed round-45 receipt)" >> "$P"
  echo "- references: upstream #1449 claims cold-TTFT -2%/-9%/-16% at 131k/524k/1M; the 8k duel slope cannot resolve it (round 45); the v1.1.6-era cold duel slope ~0.665 ms/tok at 8k" >> "$P"
  echo "- walls are client-side (upload + tokenize + prefill); the fixed share is identical across arms, so the A/B deltas are valid" >> "$P"
  echo "- each arm: fresh serve → d0 (fixed overhead) → the 96k decode row → 64k..384k ascending, one cold prefill each" >> "$P"
  IX_STANDING="MODEL_ID=local/GLM-5.3-Flash-AZ-Focus-EXL3 MODEL_REVISION=main PATCHES= DENSE_BITS=4 HEAD_BITS=4 KV_CACHE=fp8 CONTEXT=524288 PARALLEL=4"
  indexdepth_arm() {   # $1 tag (off|on), $2.. env pairs (word-split intentionally)
    local tag="$1" model d S E c0 c1
    shift
    S=$(date +%s)
    if ! restart "$@"; then
      echo "## arm $tag — refused to start" >> "$P"
      docker logs "$CONTAINER_NAME" 2>&1 | tail -4 >> "$P" || true
      return 0
    fi
    E=$(date +%s)
    echo "## arm $tag — restart wall $((E-S)) s, engine $(docker logs "$CONTAINER_NAME" 2>&1 | grep -oE 'loaded in [0-9.]+s' | tail -1)" >> "$P"
    model=$(curl -s -m 10 "http://127.0.0.1:$PORT/v1/models" | python3 -c '
import json, sys
try: print(json.load(sys.stdin)["data"][0]["id"])
except Exception: print("")' 2>/dev/null) || model=""
    model="${model:-$SERVED_NAME}"
    SERVED_MODEL="$model" python3 - "$OUT" "$tag" <<'PYEOF'
import json, os, sys
out, tag = sys.argv[1], sys.argv[2]
words = ("harbor lantern gravel meadow tideline quarry bramble inkwell thicket ferry "
         "cairn jetty saltbox windlass").split()
Q = "Write a 300-word technical summary of a harbor crane's maintenance cycle. Do not use any formatting."
for depth in (0, 98304, 65536, 131072, 196608, 262144, 327680, 393216):
    if depth == 0:
        content = Q
    else:
        n = depth // 4
        # per-row distinct from token ~1: the decoration is keyed on (i + depth),
        # so no ladder row is a token-prefix of another (the duel generator's rows
        # ARE — do not reuse it here).
        fill = " ".join(words[i % len(words)] + (f"-{(i + depth) % 89}" if i % 11 else "")
                        for i in range(n))
        content = fill + "\n\n" + Q
    json.dump({"model": os.environ["SERVED_MODEL"],
               "messages": [{"role": "user", "content": content}],
               "temperature": 0,
               "max_tokens": 512 if depth == 98304 else 1},
              open(f"{out}/ix-{tag}-d{depth}.json", "w"))
PYEOF
    c0=$(docker logs "$CONTAINER_NAME" 2>&1 | grep "prompt cache:" | tail -1 | grep -oE "hits [0-9]+, misses [0-9]+, evicted [0-9]+" || true)
    : > "$OUT/ix-$tag-walls.txt"
    for d in 0 98304 65536 131072 196608 262144 327680 393216; do
      log "indexdepth $tag at $d"
      S=$(date +%s.%N)
      chat "$OUT/ix-$tag-d$d.json" > "$OUT/ix-$tag-d$d-resp.json" || true
      E=$(date +%s.%N)
      echo "d$d $S $E" >> "$OUT/ix-$tag-walls.txt"
    done
    c1=$(docker logs "$CONTAINER_NAME" 2>&1 | grep "prompt cache:" | tail -1 | grep -oE "hits [0-9]+, misses [0-9]+, evicted [0-9]+" || true)
    python3 - "$OUT" "$tag" "$c0" "$c1" >> "$P" <<'PYEOF'
import sys
out, tag, c0, c1 = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
walls = {}
for line in open(f"{out}/ix-{tag}-walls.txt"):
    k, s, e = line.split(); walls[int(k[1:])] = (float(e) - float(s)) * 1000
actual = {}
for d, w in walls.items():
    try:
        actual[d] = json.load(open(f"{out}/ix-{tag}-d{d}-resp.json"))["usage"]["prompt_tokens"]
    except Exception:
        actual[d] = d
depths = sorted(d for d in walls if d and d != 98304)
print("- TTFT rows (max_tokens=1; nominal d → actual prompt_tokens): " +
      "; ".join("d%d(%d tok) %.0f ms" % (d, actual[d], walls[d]) for d in depths))
if 0 in walls and len(depths) > 1:
    slopes = [(walls[b] - walls[a]) / (actual[b] - actual[a]) for a, b in zip(depths, depths[1:])]
    print("- ladder marginal slopes (ms/actual-token): " + " ".join("%.3f" % s for s in slopes)
          + "  | overall cold slope %.3f (the 8k-era cold reference: 0.665)" % (
              (walls[depths[-1]] - walls[0]) / (actual[depths[-1]] - actual[0])))
if 98304 in walls and 0 in walls and depths:
    est = walls[0] + (walls[depths[0]] - walls[0]) / (actual[depths[0]] - actual[0]) * (actual[98304] - actual[0])
    print("- decode row d98304 (%d tok): wall %.0f ms; minus the prefill estimate (%.0f ms) ≈ %.1f tok/s over 512 (comparative)" % (
        actual[98304], walls[98304], est, 512 / max((walls[98304] - est) / 1000, 0.001)))
print("- prompt-cache counters before: %s → after: %s (misses should read 8: one per row — the no-sharing proof)" % (c0 or "n/a", c1 or "n/a"))
PYEOF
  }
  indexdepth_arm off $IX_STANDING
  indexdepth_arm on  $IX_STANDING TF_GLM_INDEX_CUDA=1
  python3 - "$OUT" >> "$P" <<'PYEOF'
import sys
out = sys.argv[1]
def walls(tag):
    w = {}
    for line in open(f"{out}/ix-{tag}-walls.txt"):
        k, s, e = line.split(); w[int(k[1:])] = (float(e) - float(s)) * 1000
    return w
try:
    off, on = walls("off"), walls("on")
except Exception:
    print("- comparison skipped (an arm failed to run)")
    raise SystemExit(0)
print("- the A/B (negative % = INDEX_CUDA=1 faster):")
for d in sorted(off):
    if d in on and off[d] and on[d]:
        print("  d%-7d off %8.0f ms | on %8.0f ms | %+6.1f%%" % (d, off[d], on[d], (on[d] - off[d]) / off[d] * 100))
PYEOF
  touch "$OUT/.indexdepth.done"
fi

# ---------------------------------------------------------------------------
# phase: cachegib (SESSION-SPECIFIC; retires with the PROMPT_CACHE_GIB cell).
# The round-45 queue item: under concurrent distinct-prompt load the 3.9 GiB
# kept-states store churns (the overnight bench read hits-0/evicted-113) and
# every re-visit re-prefills. The cell: a fixed churn load — 24 distinct ~64k
# fills, 8 rounds x 4 concurrent streams; rounds 1-6 cover all 24, rounds 7-8
# re-visit fills 1-8 — on the engine-default store vs PROMPT_CACHE_GIB=16.
# Pressure math: the default store thrashes once KV state exceeds ~2.6 KB/token
# (3.9 GiB / 1.54M tokens); at any plausible fp8 KV size the default arm
# re-prefills the re-visit rounds while a 16 GiB store holds all 24 states and
# resumes them. The counters + round walls are the receipt; a refusal at 16
# drops to 8 (recorded either way).
# ---------------------------------------------------------------------------
if want cachegib && ! phase_done cachegib; then
  log "=== cachegib (the PROMPT_CACHE_GIB churn cell: engine default vs 16 GiB)"
  P="$OUT/21-cachegib.md"; : > "$P"
  echo "# cachegib (the PROMPT_CACHE_GIB churn cell — the round-45 queue item)" >> "$P"
  echo "- the churn reference: the overnight bench read hits-0/evicted-113 under 4 concurrent chains" >> "$P"
  echo "- the load: 24 distinct ~64k-token fills, 8 rounds x 4 concurrent (rounds 1-6 cover all 24, rounds 7-8 re-visit fills 1-8); max_tokens=8 per request" >> "$P"
  CG_STANDING="MODEL_ID=local/GLM-5.3-Flash-AZ-Focus-EXL3 MODEL_REVISION=main PATCHES= DENSE_BITS=4 HEAD_BITS=4 KV_CACHE=fp8 CONTEXT=524288 PARALLEL=4"
  cg_counters() { docker logs "$CONTAINER_NAME" 2>&1 | grep "prompt cache:" | tail -1 | grep -oE "hits [0-9]+, misses [0-9]+, evicted [0-9]+" || true; }
  cachegib_arm() {   # $1 label, $2 GIB ("" = engine default)
    local label="$1" gib="$2" served c0 c1 model r s f S E
    served="$gib"
    if [[ -n "$gib" ]]; then
      if ! restart $CG_STANDING "PROMPT_CACHE_GIB=$gib"; then
        log "cachegib: PROMPT_CACHE_GIB=$gib refused — dropping to 8"
        served=8
        if ! restart $CG_STANDING "PROMPT_CACHE_GIB=8"; then
          echo "## arm $label — refused at 16 and at 8 (the ceiling itself is the receipt)" >> "$P"
          docker logs "$CONTAINER_NAME" 2>&1 | tail -4 >> "$P" || true
          return 0
        fi
      fi
    else
      if ! restart $CG_STANDING; then
        echo "## arm $label — refused to start" >> "$P"
        docker logs "$CONTAINER_NAME" 2>&1 | tail -4 >> "$P" || true
        return 0
      fi
    fi
    model=$(curl -s -m 10 "http://127.0.0.1:$PORT/v1/models" | python3 -c '
import json, sys
try: print(json.load(sys.stdin)["data"][0]["id"])
except Exception: print("")' 2>/dev/null) || model=""
    model="${model:-$SERVED_NAME}"
    SERVED_MODEL="$model" python3 - "$OUT" <<'PYEOF'
import json, os, sys
out = sys.argv[1]
words = ("harbor lantern gravel meadow tideline quarry bramble inkwell thicket ferry "
         "cairn jetty saltbox windlass").split()
for j in range(24):                       # 24 mutually-distinct fills (diverge at token ~1)
    n = 32768                             # ~64k tokens per fill
    fill = " ".join(words[i % len(words)] + (f"-{(i + j * 7) % 89}" if i % 11 else "")
                    for i in range(n))
    content = fill + "\n\nAnswer with one word: ready."
    for s in range(4):
        json.dump({"model": os.environ["SERVED_MODEL"],
                   "messages": [{"role": "user", "content": content}],
                   "temperature": 0, "max_tokens": 8},
                  open(f"{out}/cg-f{j}-s{s}.json", "w"))
PYEOF
    c0=$(cg_counters)
    echo "## arm $label (PROMPT_CACHE_GIB=${served:-engine default})" >> "$P"
    echo "- counters before: ${c0:-n/a}" >> "$P"
    for r in 1 2 3 4 5 6 7 8; do
      S=$(date +%s.%N)
      for s in 0 1 2 3; do
        f=$(( (r - 1) * 4 + s )); [[ $f -ge 24 ]] && f=$(( f - 24 ))
        { chat "$OUT/cg-f$f-s$s.json" > "$OUT/cg-$label-r$r-s$s-resp.json" 2>/dev/null || true; } &
      done
      wait
      E=$(date +%s.%N)
      c1=$(cg_counters)
      echo "round $r wall $(python3 -c "print(f'{$E-$S:.1f}')") s | counters: ${c1:-n/a}" >> "$P"
    done
    return 0
  }
  cachegib_arm default ""
  cachegib_arm gib16 16
  python3 - "$OUT" >> "$P" <<'PYEOF'
import json, re, sys
p = f"{sys.argv[1]}/21-cachegib.md"
arms, cur = {}, None
for line in open(p):
    m = re.match(r"## arm (\S+)", line)
    if m:
        cur = m.group(1); arms[cur] = []
        continue
    m = re.match(r"round (\d) wall ([\d.]+) s \| counters: hits (\d+), misses (\d+), evicted (\d+)", line)
    if m and cur:
        arms[cur].append((int(m.group(1)), float(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5))))
ran = False
for name, rs in arms.items():
    if not rs:
        continue
    ran = True
    cold = [w for r, w, *_ in rs if r <= 6]
    revisit = [w for r, w, *_ in rs if r >= 7]
    last = rs[-1]
    cm = sum(cold) / len(cold); rm = sum(revisit) / len(revisit)
    fill_tok = ""
    try:
        fill_tok = " | fill = %d prompt_tokens" % json.load(open(f"{sys.argv[1]}/cg-{name}-r1-s0-resp.json"))["usage"]["prompt_tokens"]
    except Exception:
        pass
    print("- %s: cold-round mean %.1f s | revisit-round mean %.1f s (%.0f%% of cold)%s | final counters hits %d, misses %d, evicted %d" % (
        name, cm, rm, rm / cm * 100, fill_tok, last[2], last[3], last[4]))
if not ran:
    print("- no round data (the arms did not run)")
PYEOF
  touch "$OUT/.cachegib.done"
fi

# ---------------------------------------------------------------------------
# phase: ixgates (SESSION-SPECIFIC; retires with the INDEX_CUDA adoption).
# The round-47 owed serve: §2 gates + the standard duel WITH TF_GLM_INDEX_CUDA=1
# — the exactness receipt the adoption needs (upstream's kernel contract is
# byte-equal scoring; our gates prove it on our serve). The perf side is
# already receipted at depth (round 47): -2.6%/-3.6%/-5.5% TTFT at
# 131k/262k/393k, decode/loading/d0 unchanged. Gates PASS here → config.sh
# flips the knob as the standing default (the flip is the digest sitting's
# edit, not this window's).
# ---------------------------------------------------------------------------
if want ixgates && ! phase_done ixgates; then
  log "=== ixgates (the INDEX_CUDA adoption: gates + the standard duel under the knob)"
  P="$OUT/22-ixgates.md"; : > "$P"
  echo "# ixgates (TF_GLM_INDEX_CUDA=1: the adoption gates + duel)" >> "$P"
  echo "- perf references (round 47, 20-indexdepth.md): TTFT -2.6%/-3.6%/-5.5% at 131k/262k/393k; decode 38.3→39.6 t/s (noise); loading unchanged" >> "$P"
  echo "- duel references (the v1.1.8 receipts, round 45): solo 57.5-57.9, c2 93.9, c4 152.4, d0 ~200 ms" >> "$P"
  IX_STANDING="MODEL_ID=local/GLM-5.3-Flash-AZ-Focus-EXL3 MODEL_REVISION=main PATCHES= DENSE_BITS=4 HEAD_BITS=4 KV_CACHE=fp8 CONTEXT=524288 PARALLEL=4"
  if run_gates ix $IX_STANDING TF_GLM_INDEX_CUDA=1; then
    if restart $IX_STANDING TF_GLM_INDEX_CUDA=1; then
      IMODEL=$(curl -s -m 10 "http://127.0.0.1:$PORT/v1/models" | python3 -c '
import json, sys
try: print(json.load(sys.stdin)["data"][0]["id"])
except Exception: print("")' 2>/dev/null) || IMODEL=""
      IMODEL="${IMODEL:-$SERVED_NAME}"
      echo "- the standard duel under the knob:" >> "$P"
      duel_arm ix "$IMODEL"
    else
      echo "- the INDEX_CUDA serve refused after the gates — log tail:" >> "$P"
      docker logs "$CONTAINER_NAME" 2>&1 | tail -4 >> "$P" || true
    fi
  else
    echo "- the gates FAILED under the knob — the adoption waits for upstream's byte-equal contract to show on our serve" >> "$P"
  fi
  touch "$OUT/.ixgates.done"
fi

# ---------------------------------------------------------------------------
# phase: gibdisc (SESSION-SPECIFIC; retires with the discriminator receipt).
# The round-47 open question: --prompt-cache-gib 16 reached the engine's argv
# yet the store served 2.6 GiB with default-identical counters. Arm 1: GIB=10
# at the standing budget — 10 fits inside the serve line's 13.4-GiB slack, so
# a budget cap would lift the store. If it still reads 2.6: arm 2 drops the
# window to 262144 (the engine's own arithmetic frees memory) — a lift there
# means the knob works and the standing budget binds; still 2.6 means the flag
# does not lift the store at all (upstream-bound data).
# ---------------------------------------------------------------------------
if want gibdisc && ! phase_done gibdisc; then
  log "=== gibdisc (does --prompt-cache-gib lift the kept-states store?)"
  P="$OUT/23-gibdisc.md"; : > "$P"
  echo "# gibdisc (the PROMPT_CACHE_GIB discriminator: budget-cap vs ignored)" >> "$P"
  echo "- round 47: --prompt-cache-gib 16 in argv, the store read 2.6 GiB, counters identical to default" >> "$P"
  GD_STANDING="MODEL_ID=local/GLM-5.3-Flash-AZ-Focus-EXL3 MODEL_REVISION=main PATCHES= DENSE_BITS=4 HEAD_BITS=4 KV_CACHE=fp8 CONTEXT=524288 PARALLEL=4"
  gd_cache() { docker logs "$CONTAINER_NAME" 2>&1 | grep -oE "a [0-9.]+ GiB prompt cache" | tail -1; }
  gd_serve() {   # $1 label, $2.. env pairs — serve, read the store figure
    if restart "$@"; then
      echo "- $1: $(gd_cache)" >> "$P"
      return 0
    fi
    echo "- $1: REFUSED — log tail:" >> "$P"
    docker logs "$CONTAINER_NAME" 2>&1 | tail -4 >> "$P" || true
    return 1
  }
  if gd_serve "arm1 GIB=10 @ 524288x4" $GD_STANDING PROMPT_CACHE_GIB=10; then
    if gd_cache | grep -qE "a 2\.6 GiB prompt cache"; then
      gd_serve "arm2 GIB=10 @ 262144x4 (the window freed)" \
        MODEL_ID=local/GLM-5.3-Flash-AZ-Focus-EXL3 MODEL_REVISION=main PATCHES= \
        DENSE_BITS=4 HEAD_BITS=4 KV_CACHE=fp8 CONTEXT=262144 PARALLEL=4 PROMPT_CACHE_GIB=10 || true
      gd_cache | grep -qE "a 2\.6 GiB prompt cache" \
        && echo "- arm2 STILL reads 2.6 — the flag does not lift the store here (upstream-bound)" >> "$P" \
        || echo "- arm2 LIFTED — the knob works; the standing 524288x4 budget is the binder" >> "$P"
    else
      echo "- arm1 LIFTED the store — the knob works; 16 was budget-capped at 524288x4" >> "$P"
    fi
  fi
  touch "$OUT/.gibdisc.done"
fi

# ---------------------------------------------------------------------------
# phase: spillgib (SESSION-SPECIFIC; retires with the spill receipt). The
# round-47 reframe: the kept-states store is budget-bound at 524288x4, so
# v1.1.8's --spill-gib (evict kept states to disk instead of dropping) is the
# alternative churn lever. Instrument: the round-47 churn load scaled to churn
# the 2.6 GiB store cheaply — 16 distinct ~56k-token fills, 6 rounds x 4
# concurrent streams (rounds 1-4 cover all 16, rounds 5-6 re-visit fills 1-8).
# The working set (~900k tokens) exceeds the store's ~810k-token hold (round
# 47: 7 kept x 116k states ≈ 2.6 GiB), so rounds 5-6 MUST come from somewhere:
# fast revisits = spill restores from disk; revisit = cold = spill does not
# (round 47's default arm is the control: revisit 99%+ of cold). One arm —
# the within-arm cover-vs-revisit ratio is the instrument.
# ---------------------------------------------------------------------------
if want spillgib && ! phase_done spillgib; then
  log "=== spillgib (the --spill-gib churn cell: restore vs re-prefill)"
  P="$OUT/24-spillgib.md"; : > "$P"
  echo "# spillgib (--spill-gib 32: do revisits restore from disk instead of re-prefilling?)" >> "$P"
  echo "- the control: round 47's default arm (no spill) read revisit rounds at 99%+ of cold (438 s rounds at ~116k fills)" >> "$P"
  echo "- the load: 16 distinct ~56k-token fills, 6 rounds x 4 concurrent (rounds 1-4 cover, 5-6 revisit fills 1-8); the working set ~900k tokens > the store's ~810k hold" >> "$P"
  SG_STANDING="MODEL_ID=local/GLM-5.3-Flash-AZ-Focus-EXL3 MODEL_REVISION=main PATCHES= DENSE_BITS=4 HEAD_BITS=4 KV_CACHE=fp8 CONTEXT=524288 PARALLEL=4"
  sg_counters() { docker logs "$CONTAINER_NAME" 2>&1 | grep "prompt cache:" | tail -1 | grep -oE "hits [0-9]+, misses [0-9]+, evicted [0-9]+" || true; }
  if restart $SG_STANDING SPILL_GIB=32; then
    model=$(curl -s -m 10 "http://127.0.0.1:$PORT/v1/models" | python3 -c '
import json, sys
try: print(json.load(sys.stdin)["data"][0]["id"])
except Exception: print("")' 2>/dev/null) || model=""
    model="${model:-$SERVED_NAME}"
    SERVED_MODEL="$model" python3 - "$OUT" <<'PYEOF'
import json, os, sys
out = sys.argv[1]
words = ("harbor lantern gravel meadow tideline quarry bramble inkwell thicket ferry "
         "cairn jetty saltbox windlass").split()
for j in range(16):                       # 16 mutually-distinct fills (diverge at token ~1)
    n = 16000                             # ~56k tokens per fill (the round-47 ratio: 3.5 tok/word)
    fill = " ".join(words[i % len(words)] + (f"-{(i + j * 7) % 89}" if i % 11 else "")
                    for i in range(n))
    content = fill + "\n\nAnswer with one word: ready."
    for s in range(4):
        json.dump({"model": os.environ["SERVED_MODEL"],
                   "messages": [{"role": "user", "content": content}],
                   "temperature": 0, "max_tokens": 8},
                  open(f"{out}/sg-f{j}-s{s}.json", "w"))
PYEOF
    echo "## the spill serve — $(docker logs "$CONTAINER_NAME" 2>&1 | grep -oE 'a [0-9.]+ GiB prompt cache' | tail -1)" >> "$P"
    c0=$(sg_counters)
    echo "- counters before: ${c0:-n/a}" >> "$P"
    for r in 1 2 3 4 5 6; do
      S=$(date +%s.%N)
      for s in 0 1 2 3; do
        f=$(( (r - 1) * 4 + s )); [[ $f -ge 16 ]] && f=$(( f - 16 ))
        { chat "$OUT/sg-f$f-s$s.json" > "$OUT/sg-r$r-s$s-resp.json" 2>/dev/null || true; } &
      done
      wait
      E=$(date +%s.%N)
      c1=$(sg_counters)
      echo "round $r wall $(python3 -c "print(f'{$E-$S:.1f}')") s | counters: ${c1:-n/a}" >> "$P"
    done
    python3 - "$OUT" >> "$P" <<'PYEOF'
import json, re, sys
p = f"{sys.argv[1]}/24-spillgib.md"
rounds = []
for line in open(p):
    m = re.match(r"round (\d) wall ([\d.]+) s", line)
    if m:
        rounds.append((int(m.group(1)), float(m.group(2))))
if not rounds:
    print("- no round data")
    raise SystemExit(0)
cover = [w for r, w in rounds if r <= 4]
revisit = [w for r, w in rounds if r >= 5]
cm = sum(cover) / len(cover); rm = sum(revisit) / len(revisit)
fill_tok = ""
try:
    fill_tok = " | fill = %d prompt_tokens" % json.load(open(f"{sys.argv[1]}/sg-r1-s0-resp.json"))["usage"]["prompt_tokens"]
except Exception:
    pass
print("- cover-round mean %.1f s | revisit-round mean %.1f s (%.0f%% of cover)%s" % (cm, rm, rm / cm * 100, fill_tok))
print("- the reading: revisit near-100% of cover = spill does NOT restore (matches round 47's default control); revisit well under = spill restores")
PYEOF
  else
    echo "## the spill serve REFUSED — the flag is not accepted (the refusal is the receipt; log tail):" >> "$P"
    docker logs "$CONTAINER_NAME" 2>&1 | tail -6 >> "$P" || true
  fi
  touch "$OUT/.spillgib.done"
fi

# ---------------------------------------------------------------------------
# phase: summary
# ---------------------------------------------------------------------------
if want summary; then
  log "=== summary"
  S="$OUT/CAMPAIGN.md"
  {
    echo "# Replication campaign $RUN_ID"
    echo
    echo "## Gates"
    grep -hE "^- (PASS|FAIL|NOTE|depth|batch|usage|server|RoCE|PARALLEL|.*refused)" "$OUT"/[0-9]*.md 2>/dev/null | head -60 || true
    echo
    echo "## Phase receipts"
    for f in "$OUT"/[0-9]*.md; do echo "### $(basename "$f")"; cat "$f"; echo; done
    echo "## Way forward"
    echo "- Content-form gates (VALIDATION §2 receipt, 2026-10-23): PASS = content-identical;"
    echo "  raw-byte differences are the engine's documented nondeterminism, not a cache fault."
    echo "- dense4fp8 vs parity: tool-eval 92 = 92 (first campaign); this campaign's"
    echo "  throughput + climb receipts pick the serving config and its stream count."
    echo "- kvqual4 PASS + throughput lead = dense4fp8 is the qualified ladder config;"
    echo "  feed the cliff/prefill receipts into the next upstream note."
  } > "$S"
  if ./scripts/render-perf-chart.sh --check >> "$S" 2>&1; then
    log "chart check: the committed SVG matches the anchors"
  else
    log "chart check: STALE vs the anchors — regenerate and commit (render-perf-chart.sh)"
  fi
  log "campaign complete: $S"
  tail -40 "$S"
fi
log "done."
