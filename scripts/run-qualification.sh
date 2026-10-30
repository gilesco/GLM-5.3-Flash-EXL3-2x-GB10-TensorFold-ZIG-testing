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
for _p in parity candidate kvqual kvqual4 throughput climb kvqual4P throughputP climbP knobs duel confirm prof4 top4 routerks2 routerpart top4qual azqual focusqual hmtrials pinbump settingsab ncurouter profchunk profload efforttail etreminder etadopt routergemm depth cliff prefill rung; do
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
  echo "# depth gate (the STANDING config: dense4fp8 + TF_GLM_INDEX_CUDA=1 + vision, 524288x4; community bar: production 93.9%)" >> "$P"
  echo "- references: the v1.1.6-era receipt 12/12 (campaign-20261027-124115, the top4cfg re-qual); this run re-qualifies the current default end to end" >> "$P"
  restart
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
# phase: visionen (SESSION-SPECIFIC; retires with the vision enablement
# receipt). The (user) call (round 51): vision is a want, enabled as a
# standing default at the 524288 window; NO perf testing — the receipt is
# functional: the serve admits at 524288x4 with --vision, the §2 gates PASS on
# the vision serve, and a smoke image request reads the image's content (a
# generated 64x64 four-quadrant PNG: TL=red TR=green BL=blue BR=white; the
# model must answer the asked quadrant's color). The tower loads BF16 (the
# checkpoint's -vb 16 recipe); the engine defaults 4 images x 4096 tokens.
# ---------------------------------------------------------------------------
if want visionen && ! phase_done visionen; then
  log "=== visionen (the vision enablement: the standing serve + gates + the smoke image)"
  P="$OUT/25-visionen.md"; : > "$P"
  echo "# visionen (--vision at 524288x4: the standing default's enablement receipt)" >> "$P"
  echo "- the (user) call: vision is a want, enabled as a default before anything else; no perf testing; 512k only" >> "$P"
  echo "- the pieces: the checkpoint's 347 BF16 model.visual.* tensors; the engine's GLM vision (families/glm/cuda_vision.zig); defaults 4 images x 4096 tokens" >> "$P"
  if restart_fb DENSE_BITS=4 HEAD_BITS=4 KV_CACHE=fp8 CONTEXT=524288 PARALLEL=4; then
    echo "- the serve line: $(serve_line)" >> "$P"
    model=$(curl -s -m 10 "http://127.0.0.1:$PORT/v1/models" | python3 -c '
import json, sys
try: print(json.load(sys.stdin)["data"][0]["id"])
except Exception: print("")' 2>/dev/null) || model=""
    model="${model:-$SERVED_NAME}"
    SERVED_MODEL="$model" python3 - "$OUT" <<'PYEOF'
import base64, json, os, struct, sys, urllib.request, zlib
out = sys.argv[1]
W = H = 64
quads = [(255, 0, 0), (0, 200, 0), (0, 80, 255), (255, 255, 255)]  # TL red, TR green, BL blue, BR white
rows = []
for y in range(H):
    row = bytearray()
    for x in range(W):
        row += bytes(quads[(y // 32) * 2 + (x // 32)])
    rows.append(b"\x00" + bytes(row))
raw = b"".join(rows)
def chunk(tag, data):
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))
png = (b"\x89PNG\r\n\x1a\n"
       + chunk(b"IHDR", struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0))
       + chunk(b"IDAT", zlib.compress(raw))
       + chunk(b"IEND", b""))
open(f"{out}/_quadrants.png", "wb").write(png)
print("smoke PNG built: 64x64 quadrants TL=red TR=green BL=blue BR=white (%d bytes)" % len(png))

# the vision suite: per manifest entry, a targeted question (max_tokens 64) and a
# long-form detail question (max_tokens 1024) scored against the agent-verified
# ground-truth detail list (assets/vision/manifest.json)
port = os.environ.get("PORT", "8000")
url_base = f"http://127.0.0.1:{port}"
manifest = json.load(open("assets/vision/manifest.json"))
R = []
for e in manifest:
    img = open(f"assets/vision/{e['file']}", "rb").read()
    mime = "image/png" if e["file"].endswith(".png") else "image/jpeg"
    durl = f"data:{mime};base64," + base64.b64encode(img).decode()
    for kind, qkey, mt in (("targeted", "question", 64), ("detail", "detail_question", 1024)):
        body = {"model": os.environ["SERVED_MODEL"], "temperature": 0, "max_tokens": mt,
                "messages": [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": durl}},
                    {"type": "text", "text": e[qkey]}]}]}
        req = urllib.request.Request(url_base + "/v1/chat/completions",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                d = json.loads(resp.read())
            c = d["choices"][0]["message"]["content"]
            R.append({"file": e["file"], "kind": kind, "answer": c,
                      "pt": d["usage"]["prompt_tokens"], "ct": d["usage"]["completion_tokens"]})
            print(f"  {e['file']} {kind}: asked, {d['usage']['prompt_tokens']} prompt / {d['usage']['completion_tokens']} out tokens")
        except Exception as ex:
            R.append({"file": e["file"], "kind": kind, "answer": "", "error": str(ex)})
            print(f"  {e['file']} {kind}: REQUEST FAILED ({ex})")
json.dump(R, open(f"{out}/vision-answers.json", "w"), indent=1)
PYEOF
    python3 - "$OUT" >> "$P" <<'PYEOF'
import json, re, sys
out = sys.argv[1]
manifest = {e["file"]: e for e in json.load(open("assets/vision/manifest.json"))}
answers = json.load(open(f"{out}/vision-answers.json"))
tgt_ok = det_cov = det_all = 0
for e in manifest.values():
    tgt = next((a for a in answers if a["file"] == e["file"] and a["kind"] == "targeted"), None)
    det = next((a for a in answers if a["file"] == e["file"] and a["kind"] == "detail"), None)
    print("## %s" % e["file"])
    if tgt and "error" not in tgt:
        hit = any(w in tgt["answer"].lower() for w in e["expect"])
        tgt_ok += hit
        print("- targeted: %r → %s (%d out tokens)" % (tgt["answer"][:120], "PASS" if hit else "FAIL", tgt["ct"]))
    else:
        print("- targeted: FAILED (%s)" % (tgt.get("error") if tgt else "missing"))
    if det and "error" not in det:
        low = det["answer"].lower()
        got, missed = [], []
        for lbl, alts in zip(e["detail_labels"], e["detail_expect"]):
            (got if any(w in low for w in alts) else missed).append(lbl)
        det_cov += len(got); det_all += len(e["detail_labels"])
        print("- detail (%d out tokens): %d/%d ground-truth details seen — %s" % (
            det["ct"], len(got), len(e["detail_labels"]), ", ".join(got) if got else "none"))
        if missed:
            print("- detail missed: %s" % ", ".join(missed))
        print("- detail answer: %s" % " ".join(det["answer"].split())[:400])
    else:
        print("- detail: FAILED (%s)" % (det.get("error") if det else "missing"))
    print("- ground truth: %s" % e["ground_truth"])
print("- the vision suite: targeted %d/%d, detail coverage %d/%d" % (
    tgt_ok, len(manifest), det_cov, det_all))
PYEOF
    run_gates vision DENSE_BITS=4 HEAD_BITS=4 KV_CACHE=fp8 CONTEXT=524288 PARALLEL=4
  else
    echo "- the vision serve REFUSED at 524288 (the PARALLEL=2 retry also failed or the receipt is the refusal) — log tail:" >> "$P"
    docker logs "$CONTAINER_NAME" 2>&1 | tail -6 >> "$P" || true
  fi
  touch "$OUT/.visionen.done"
fi


# ---------------------------------------------------------------------------
# phase: efforttail (SESSION-SPECIFIC; the effort-tail template A/B — retires
# with the verdict). Two legs in the SAME window (bernisse's methodology rule:
# a one-line template change moved median turn time 3.4 -> 4.5 s across weeks —
# same-session pairs only). Leg 1 = the baseline (the checkpoint's own template,
# the effort hint at the prompt top); leg 2 = CHAT_TEMPLATE=effort-tail (the hint
# just before the final assistant turn, user-message-ending only — NV forums
# 381350 posts 915/933: at long context the top hint is ignored, max reasoning
# 11-528 vs 1,900-7,000 tokens, and effort switches invalidate the prefix cache).
# Quality targets: the hardmode score vs the same-session baseline, and TC-51 /
# TC-92 — which fail deterministically on our line on BOTH checkpoints (round
# 72's flip data) while technigma is clean on both. VISION=0 on both legs (the
# effort-tail flag is text-only for now). The jinja render gate passed
# pre-window (four placement cases); the Dockerfile stages the template, so
# prepare runs EXPLICITLY (a Dockerfile change does not move the fingerprint —
# the round-27 lesson).
# ---------------------------------------------------------------------------
if want efforttail && ! phase_done efforttail; then
  log "=== efforttail (the effort-tail template A/B)"
  P="$OUT/27-efforttail.md"; : > "$P"
  echo "# efforttail (the effort-tail template A/B on the Focus: baseline vs tail)" >> "$P"
  echo "- the jinja render gate: PASS pre-window (user-ending / tool-result-ending / undefined / None)" >> "$P"
  echo "- methodology: same-session legs; fixed probes; VISION=0 both legs" >> "$P"
  log "efforttail: explicit prepare (the Dockerfile stages the template into the image)"
  if ! scripts/prepare.sh </dev/null >>"$LOG" 2>&1; then die "efforttail: the image rebuild (prepare) failed — see the log"; fi
  ETENV="MODEL_ID=local/GLM-5.3-Flash-AZ-Focus-EXL3 MODEL_REVISION=main PATCHES= DENSE_BITS=4 HEAD_BITS=4 KV_CACHE=fp8 CONTEXT=524288 PARALLEL=4"
  ETOUT="$OUT/efforttail"; mkdir -p "$ETOUT"
  for leg in baseline tail; do
    CT=""; [ "$leg" = tail ] && CT="CHAT_TEMPLATE=effort-tail"
    log "efforttail: leg $leg (CHAT_TEMPLATE=${CT:-empty})"
    if restart $ETENV VISION=0 $CT; then
      AM=$(curl -s -m 10 "http://127.0.0.1:$PORT/v1/models" | python3 -c '
import json, sys
try: print(json.load(sys.stdin)["data"][0]["id"])
except Exception: print("")' 2>/dev/null) || AM=""
      AM="${AM:-local/GLM-5.3-Flash-AZ-Focus-EXL3}"
      echo "- leg $leg (CHAT_TEMPLATE=${CT:-empty}): $(serve_line)" >> "$P"
      PYTHONUNBUFFERED=1 "$BENCH" run --base-url "http://127.0.0.1:$PORT" \
        --model "$AM" --format openai --tokenizer "$MODEL_ID" \
        --hardmode --label "et-$leg" --output-dir "$ETOUT" \
        --json-file "$ETOUT/et-$leg.json" > "$ETOUT/et-$leg.log" 2>&1 || true
      grep -o '"final_score": [0-9]*' "$ETOUT/et-$leg.json" >> "$P" 2>/dev/null || true
      grep -o '"scenario_id": "TC-51", "status": "[a-z]*"' "$ETOUT/et-$leg.log" >> "$P" 2>/dev/null || true
      grep -o '"scenario_id": "TC-92", "status": "[a-z]*"' "$ETOUT/et-$leg.log" >> "$P" 2>/dev/null || true
    else
      echo "- leg $leg: the serve REFUSED (log tail):" >> "$P"
      docker logs "$CONTAINER_NAME" 2>&1 | tail -4 >> "$P" || true
    fi
  done
  touch "$OUT/.efforttail.done"
fi

# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# phase: etreminder (SESSION-SPECIFIC; the tool-continuation reminder A/B —
# retires with the verdict). TC-51 fails deterministically on our line on both
# checkpoints (rounds 72-76) — the second template gap: reasoning collapse after
# tool results (technigma's reproduced failure: ZERO reasoning tokens on tool
# continuations at 38k/500k/900k; their reminder restored separate reasoning +
# correct plans, before/after 0/0/0 -> 240/212/194). Leg 1 = the adopted
# effort-tail (the round-79 receipt: 93 on the vision serve, TC-51 fail); leg 2 =
# effort-tail-tr (the same template + the tool-continuation reminder as a
# synthetic user turn, tool-ending only, opt-out
# chat_template_kwargs.tool_reasoning_reminder=false). The two fixes are
# mutually exclusive by condition (tool-ending vs user-ending). Same-session
# legs, VISION=0 both, hardmode + TC-51/TC-92 per leg; the render gate covers
# the reminder cases (fires once on tool-ending, silent on user-ending, the
# opt-out honored).
# ---------------------------------------------------------------------------
if want etreminder && ! phase_done etreminder; then
  log "=== etreminder (the tool-continuation reminder A/B)"
  P="$OUT/29-etreminder.md"; : > "$P"
  echo "# etreminder (the tool-continuation reminder A/B on the Focus: effort-tail vs effort-tail-tr)" >> "$P"
  echo "- the render gate: PASS pre-window (the reminder fires once on tool-ending, silent on user-ending, the opt-out honored, mutually exclusive with the effort hint)" >> "$P"
  echo "- the mechanism: technigma 23c7b03's before/after (0/0/0 -> 240/212/194 reasoning tokens on reproduced tool-continuation failures); reminder-off controls on THEIR fresh fixtures also passed — this A/B measures it on OUR bench" >> "$P"
  log "etreminder: explicit prepare (the new template stages into the image)"
  if ! scripts/prepare.sh </dev/null >>"$LOG" 2>&1; then die "etreminder: the image rebuild (prepare) failed — see the log"; fi
  ETENV="MODEL_ID=local/GLM-5.3-Flash-AZ-Focus-EXL3 MODEL_REVISION=main PATCHES= DENSE_BITS=4 HEAD_BITS=4 KV_CACHE=fp8 CONTEXT=524288 PARALLEL=4"
  ETOUT="$OUT/etreminder"; mkdir -p "$ETOUT"
  for leg in tail tail-tr; do
    CT="CHAT_TEMPLATE=effort-tail"; [ "$leg" = "tail-tr" ] && CT="CHAT_TEMPLATE=effort-tail-tr"
    log "etreminder: leg $leg ($CT)"
    if restart $ETENV VISION=0 $CT; then
      AM=$(curl -s -m 10 "http://127.0.0.1:$PORT/v1/models" | python3 -c '
import json, sys
try: print(json.load(sys.stdin)["data"][0]["id"])
except Exception: print("")' 2>/dev/null) || AM=""
      AM="${AM:-local/GLM-5.3-Flash-AZ-Focus-EXL3}"
      echo "- leg $leg ($CT): $(serve_line)" >> "$P"
      PYTHONUNBUFFERED=1 "$BENCH" run --base-url "http://127.0.0.1:$PORT" \
        --model "$AM" --format openai --tokenizer "$MODEL_ID" \
        --hardmode --label "et-$leg" --output-dir "$ETOUT" \
        --json-file "$ETOUT/et-$leg.json" > "$ETOUT/et-$leg.log" 2>&1 || true
      grep -o '"final_score": [0-9]*' "$ETOUT/et-$leg.json" >> "$P" 2>/dev/null || true
      grep -o '"scenario_id": "TC-51", "status": "[a-z]*"' "$ETOUT/et-$leg.log" >> "$P" 2>/dev/null || true
      grep -o '"scenario_id": "TC-92", "status": "[a-z]*"' "$ETOUT/et-$leg.log" >> "$P" 2>/dev/null || true
    else
      echo "- leg $leg: the serve REFUSED (log tail):" >> "$P"
      docker logs "$CONTAINER_NAME" 2>&1 | tail -4 >> "$P" || true
    fi
  done
  touch "$OUT/.etreminder.done"
fi

# ---------------------------------------------------------------------------
# phase: etadopt (SESSION-SPECIFIC; the adopted-default validation — retires
# with the receipt). Round 77 adopted CHAT_TEMPLATE=effort-tail as the standing
# default and merged the fix into the vision template; this window receipts the
# ADOPTED default on the vision path: the serve admits at the standing config
# (VISION=1 + the vision template carrying the effort-tail fix — the serve line
# names the template), the vision suite (assets/vision via the round-55 runner,
# now repoed at scripts/vision-suite-probe.py) matches its baseline (4/4
# targeted + 23/25 detail), and ONE hardmode run re-verifies the score on the
# vision serve (the round-76 receipts: 93 tail text-only / 92 baseline).
# Explicit prepare first: the vision template's CONTENT changed — a content
# change does not move the fingerprint (the round-27 lesson).
# ---------------------------------------------------------------------------
if want etadopt && ! phase_done etadopt; then
  log "=== etadopt (the adopted-default validation: vision + bench on the new template)"
  P="$OUT/28-etadopt.md"; : > "$P"
  echo "# etadopt (the adopted default: VISION=1 + the vision template with the effort-tail fix)" >> "$P"
  echo "- the receipts this window re-verifies: the vision suite 4/4 + 23/25 (round 55); hardmode 93 tail / 92 baseline (round 76)" >> "$P"
  log "etadopt: explicit prepare (the vision template's content changed)"
  if ! scripts/prepare.sh </dev/null >>"$LOG" 2>&1; then die "etadopt: the image rebuild (prepare) failed — see the log"; fi
  if restart DENSE_BITS=4 HEAD_BITS=4 KV_CACHE=fp8 CONTEXT=524288 PARALLEL=4; then
    echo "- the serve line: $(serve_line)" >> "$P"
    model=$(curl -s -m 10 "http://127.0.0.1:$PORT/v1/models" | python3 -c '
import json, sys
try: print(json.load(sys.stdin)["data"][0]["id"])
except Exception: print("")' 2>/dev/null) || model=""
    model="${model:-local/GLM-5.3-Flash-AZ-Focus-EXL3}"
    ETOUT="$OUT/etadopt"; mkdir -p "$ETOUT"
    if [ -f scripts/vision-suite-probe.py ]; then
      python3 scripts/vision-suite-probe.py "http://127.0.0.1:$PORT" "$ETOUT/vision.json" > "$ETOUT/vision.log" 2>&1 || true
      tail -6 "$ETOUT/vision.log" >> "$P" 2>/dev/null || true
    else
      echo "- the vision-suite runner is MISSING at scripts/vision-suite-probe.py" >> "$P"
    fi
    PYTHONUNBUFFERED=1 "$BENCH" run --base-url "http://127.0.0.1:$PORT" \
      --model "$model" --format openai --tokenizer "$MODEL_ID" \
      --hardmode --label "etadopt" --output-dir "$ETOUT" \
      --json-file "$ETOUT/etadopt.json" > "$ETOUT/etadopt.log" 2>&1 || true
    grep -o '"final_score": [0-9]*' "$ETOUT/etadopt.json" >> "$P" 2>/dev/null || true
    grep -o '"scenario_id": "TC-51", "status": "[a-z]*"' "$ETOUT/etadopt.log" >> "$P" 2>/dev/null || true
    grep -o '"scenario_id": "TC-92", "status": "[a-z]*"' "$ETOUT/etadopt.log" >> "$P" 2>/dev/null || true
  else
    echo "- the serve REFUSED (log tail):" >> "$P"
    docker logs "$CONTAINER_NAME" 2>&1 | tail -4 >> "$P" || true
  fi
  touch "$OUT/.etadopt.done"
fi


# ---------------------------------------------------------------------------
# phase: routergemm (SESSION-SPECIFIC; PERF_PLAN Front 2 Lever A2 — the GEMM
# replacement, retires with the lever). Two same-session legs on the v1.1.9 base
# (the pin moves 812fb040 -> 465e48b6 for this window; v1.1.9's GLM items are
# bit-exact upstream — same tokens). ONE prepare stages glm-router-gemm.patch
# into the image; the patch is INERT without TF_GLM_ROUTER_MM=1 (the dispatch
# needs the tiles, built only by the env-gated engine init), so:
#   leg stock: the patched binary, env off — the executed path is the hand-rolled
#              kernels: v1.1.9's true baseline (duel solo + c4 + chunk prof).
#   leg gemm:  TF_GLM_ROUTER_MM=1 — the staged chain router (tf_glm_router_mm:
#              glm_mm's 128-row cp.async shape per K slice, ONE continuous mma
#              chain per output — the hand-rolled kernels' exact bits by
#              construction; the content gate verifies byte-identity) + hardmode.
# Read: content gate IDENTICAL + the chunk router ms down vs leg stock = the
# lever lands; the hardmode score guards quality (93 expected, TC-92 pass).
# The mechanism receipt: stock glm_mm could NOT be called per-slice (its per-64-
# group acc structure differs from the router kernels' continuous chain — fp32
# non-associativity), hence the dedicated chain kernel; slice_sum is untouched.
# ---------------------------------------------------------------------------
if want routergemm && ! phase_done routergemm; then
  log "=== routergemm (Front 2 A2: the staged chain router A/B on v1.1.9)"
  P="$OUT/30-routergemm.md"; : > "$P"
  echo "# routergemm (the GEMM replacement A/B: v1.1.9 stock path vs the chain router)" >> "$P"
  echo "- the pin: TF_COMMIT 812fb040 (v1.1.8) -> 465e48b6 (v1.1.9); the patch diffs the v1.1.9 tree" >> "$P"
  echo "- compile receipt: fatbins + native build RC=0 in the builder image (nvcc sm_121, zig 0.17.0)" >> "$P"
  log "routergemm: explicit prepare (the pin moved + the patch overlay stages)"
  if ! PATCHES=glm-router-gemm.patch scripts/prepare.sh </dev/null >>"$LOG" 2>&1; then die "routergemm: prepare failed — see the log"; fi
  RGOUT="$OUT/routergemm"; mkdir -p "$RGOUT"
  RGENV="MODEL_ID=local/GLM-5.3-Flash-AZ-Focus-EXL3 MODEL_REVISION=main DENSE_BITS=4 HEAD_BITS=4 KV_CACHE=fp8 CONTEXT=524288 PARALLEL=4 VISION=0 TF_GLM_PROF=1"
  for leg in stock gemm; do
    if [ "$leg" = stock ]; then
      log "routergemm: leg stock (the patched binary, the chain path off)"
      RGX="$RGENV PATCHES=glm-router-gemm.patch"
    else
      log "routergemm: leg gemm (TF_GLM_ROUTER_MM=1)"
      RGX="$RGENV PATCHES=glm-router-gemm.patch TF_GLM_ROUTER_MM=1"
    fi
    if restart $RGX; then
      echo "- leg $leg: $(serve_line)" >> "$P"
      duel_arm "rg-$leg" "$SERVED_NAME"
      c4_reprobe "rg-$leg"
      echo "- leg $leg chunk prof (the router section read):" >> "$P"
      docker logs "$CONTAINER_NAME" 2>&1 | grep "glm prof prompt chunk" | tail -2 >> "$P" || true
      if [ "$leg" = gemm ]; then
        AM=$(curl -s -m 10 "http://127.0.0.1:$PORT/v1/models" | python3 -c '
import json, sys
try: print(json.load(sys.stdin)["data"][0]["id"])
except Exception: print("")' 2>/dev/null) || AM=""
        AM="${AM:-local/GLM-5.3-Flash-AZ-Focus-EXL3}"
        PYTHONUNBUFFERED=1 "$BENCH" run --base-url "http://127.0.0.1:$PORT" \
          --model "$AM" --format openai --tokenizer "$MODEL_ID" \
          --hardmode --label "rg-gemm" --output-dir "$RGOUT" \
          --json-file "$RGOUT/rg-gemm.json" > "$RGOUT/rg-gemm.log" 2>&1 || true
        grep -o '"final_score": [0-9]*' "$RGOUT/rg-gemm.json" >> "$P" 2>/dev/null || true
        grep -o '"scenario_id": "TC-51", "status": "[a-z]*"' "$RGOUT/rg-gemm.log" >> "$P" 2>/dev/null || true
        grep -o '"scenario_id": "TC-92", "status": "[a-z]*"' "$RGOUT/rg-gemm.log" >> "$P" 2>/dev/null || true
      fi
    else
      echo "- leg $leg: the serve REFUSED (log tail):" >> "$P"
      docker logs "$CONTAINER_NAME" 2>&1 | tail -4 >> "$P" || true
    fi
  done
  python3 - "$OUT" >> "$P" <<'RGEOF'
import hashlib, json, sys
out = sys.argv[1]
try:
    a = json.load(open(f"{out}/duel-rg-stock-solo-a.json"))
    b = json.load(open(f"{out}/duel-rg-gemm-solo-a.json"))
    ha = hashlib.sha256(json.dumps(a["choices"], sort_keys=True).encode()).hexdigest()
    hb = hashlib.sha256(json.dumps(b["choices"], sort_keys=True).encode()).hexdigest()
    print("- content gate (the hand-rolled path vs the chain router): %s" % ("IDENTICAL - the staged chain path is bit-safe; the chunk prof decides the lever" if ha == hb else "DIFFERS - NOT bit-safe: the patch dies (#1079 protocol) and the cost read is void"))
except Exception as e:
    print("- content gate could not run: %s" % e)
RGEOF
  touch "$OUT/.routergemm.done"
fi

# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# phase: ncurouter (SESSION-SPECIFIC; the Front 2 A2 follow-up instrument —
# retires with the diagnosis). Three kernel probes agree the router section's
# ~1100 ms/chunk is implementation-independent; this capture splits KERNEL TIME
# from GAP TIME with a real profiler. Vehicle (round 94 design): the capture
# runs INSIDE the testing stack's normal lifecycle — ncu is tar-copied into the
# patched serve, the image is committed, and a WRAPPED restart launches rank 0's
# engine under ncu (kernel-name filter, duration metric only, CSV to
# /tmp/rg-ncu.csv) while rank 1 comes up normally via start.sh and the TP pair
# forms. The round-93 container-swap design is dead on two structural facts:
# the worker rank exits when the head rank's container stops (no peer for a
# throwaway rank 0), and the model cache is a bind mount a swapped container
# does not inherit. One ~2k-token prompt fires one 2048-row chunk. Read: the
# per-launch duration DISTRIBUTION (boot/warmup/smoke/prompt launches mix
# shapes) vs the section's 1109 ms — a cheap cluster means the section is
# bubbles (host enqueue / span semantics); an expensive one says why.
# Fail-honest: an instrument refusal is a receipt, not a retry.
# ---------------------------------------------------------------------------
if want ncurouter && ! phase_done ncurouter; then
  log "=== ncurouter (the router-section profiler capture)"
  P="$OUT/31-ncurouter.md"; : > "$P"
  echo "# ncurouter (splitting the router section's 1109 ms into kernel time vs gap time)" >> "$P"
  echo "- the question: three kernel probes cost the same; is the section slow KERNELS or BUBBLES?" >> "$P"
  NOUT="$OUT/ncurouter"; mkdir -p "$NOUT"
  log "ncurouter: the patched serve up (the chain path on) for the commit"
  if restart DENSE_BITS=4 HEAD_BITS=4 KV_CACHE=fp8 CONTEXT=524288 PARALLEL=4 VISION=0 PATCHES=glm-router-gemm.patch TF_GLM_ROUTER_MM=1 TF_GLM_PROF=1; then
    # 1. copy the profiler into the running container (from the base image), then commit
    # the proven pipe: tar the profiler out of the base image and straight into the serve container
    NDIR=$(docker run --rm --entrypoint /bin/bash nvcr.io/nvidia/pytorch:26.07-py3 -c "ls -d /usr/local/cuda/nsight-compute* /opt/nvidia/nsight-compute* 2>/dev/null | head -1") || true
    if [ -n "$NDIR" ]; then
      docker run --rm --entrypoint /bin/bash nvcr.io/nvidia/pytorch:26.07-py3 -c "tar -cf - -C / usr/local/bin/ncu ${NDIR#/}" 2>/dev/null | docker exec -i "$CONTAINER_NAME" tar -xf - -C / 2>/dev/null || true
    fi
    if docker exec "$CONTAINER_NAME" test -x /usr/local/bin/ncu 2>/dev/null; then
      NSCD="$NDIR"
      docker exec "$CONTAINER_NAME" bash -c "NV_COMPUTE_PROFILER_DIR=$NDIR /usr/local/bin/ncu --version 2>&1 | head -2" > "$NOUT/ncu-version.txt" 2>&1 || true
      echo "- ncu in-container: $(head -1 "$NOUT/ncu-version.txt" 2>/dev/null || echo BROKEN)" >> "$P"
      # 2. the capture container: the engine under ncu, filtered to the router kernels.
      #    Generation is placeholder-based, NOT an f-string — the template carries a
      #    literal JSON request body whose braces killed the f-string generator in
      #    rounds 90/91 (ValueError: Invalid format specifier). The capture script
      #    self-waits for the serve, self-queries the served model id, and signals
      #    done; the commit happens AFTER the script is copied in (the capture
      #    container runs it from the committed image — the round-91 ordering bug).
      docker commit "$CONTAINER_NAME" rg-ncu-img >/dev/null 2>&1 || true
      # 2. the wrapped restart: rank 0's engine runs under ncu (kernel-name filter,
      #    duration metric only — minimal per-launch collection overhead), rank 1 comes
      #    up normally via start.sh and the TP pair forms. WAIT_TIMEOUT bounds the wait
      #    (ncu overhead slows the boot; a timeout is recorded, not retried blind).
      WRAP="ncu --kernel-name regex:tf_glm_router_mm|tf_glm_slice_sum --launch-count 512 --metrics gpu__time_duration.sum --csv --log-file /tmp/rg-ncu.csv"
      log "ncurouter: the wrapped restart (rank 0 under ncu)"
      if restart DENSE_BITS=4 HEAD_BITS=4 KV_CACHE=fp8 CONTEXT=524288 PARALLEL=4 VISION=0 PATCHES=glm-router-gemm.patch TF_GLM_ROUTER_MM=1 TF_GLM_PROF=1 RUN_IMAGE=rg-ncu-img RANK0_WRAP="$WRAP" WAIT_TIMEOUT=600; then
        # 3. the serve answered: fire one ~2k-token prompt (one 2048-row chunk), 3 tries
        MODEL=$(curl -s -m 5 "http://127.0.0.1:$PORT/v1/models" | python3 -c "import json,sys; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null)
        [ -n "$MODEL" ] || MODEL=local/GLM-5.3-Flash
        PTEXT="Summarize the following. $(printf 'The harbor crane maintenance cycle covers wire rope inspection, sheave wear, hydraulic fluid sampling, brake torque checks, boom deflection measurement, and corrosion protection. %.0s' $(seq 42))"
        for try in 1 2 3; do
          curl -s -m 300 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
            -d "$(python3 -c 'import json,sys; print(json.dumps({"model": sys.argv[1], "max_tokens": 8, "messages": [{"role": "user", "content": sys.argv[2]}]}))' "$MODEL" "$PTEXT")" \
            > "$NOUT/prompt-reply.json" 2>/dev/null || true
          [ -s "$NOUT/prompt-reply.json" ] && break
          sleep 10
        done
        echo "- prompt fired: $(wc -c < "$NOUT/prompt-reply.json" 2>/dev/null || echo 0) bytes reply (try ${try:-?})" >> "$P"
        # 4. settle, then finalize: TERM to ncu writes the report and takes the engine
        #    down (the capture is this phase's last GPU act; the campaign wrap stops the
        #    testing stack anyway). Then copy the CSV + the serve log out.
        sleep 45
        docker exec "$CONTAINER_NAME" pkill -TERM ncu 2>/dev/null || true
        for i in $(seq 1 12); do docker ps --format '{{.Names}}' 2>/dev/null | grep -qx "$CONTAINER_NAME" || break; sleep 10; done
        docker stop "$CONTAINER_NAME" >/dev/null 2>&1 || true
        sleep 5
        docker cp "$CONTAINER_NAME:/tmp/rg-ncu.csv" "$NOUT/rg-ncu.csv" 2>/dev/null \
          || echo "- the CSV did not come out of the serve container (fail-honest)" >> "$P"
        docker logs "$CONTAINER_NAME" > "$NOUT/serve-under-ncu.log" 2>&1 || true
        # 5. the read: per-launch duration DISTRIBUTION (boot/warmup/smoke/prompt
        #    launches mix shapes — the percentile spread carries the verdict, not the
        #    mean; the 2048-row chunk is the SLOW cluster)
        python3 - "$NOUT" >> "$P" <<'RGNCU3'
import csv, io, sys
out = sys.argv[1]
try:
    raw = open(f"{out}/rg-ncu.csv").read().splitlines()
    # ncu's ==PROF==/==ERROR== banners PRECEDE the CSV header — skip to it
    i = next(k for k, l in enumerate(raw) if l.startswith('"ID"'))
    rows = list(csv.DictReader(io.StringIO("\n".join(raw[i:]))))
except Exception as e:
    print(f"- the capture CSV is missing/broken: {e} — the serve log has the reason (fail-honest)") ; raise SystemExit
def dur(rs):
    v = []
    for r in rs:
        val = (r.get("Metric Value") or "").strip()
        try: v.append(float(val))
        except ValueError: pass
    return sorted(v)
def summ(name, rs):
    if not rs:
        print(f"- {name}: no launches captured"); return
    d = sorted(rs)
    n = len(d)
    q = lambda p: d[min(n - 1, int(n * p))]
    print(f"- {name}: {n} launches | min {d[0]:.0f}ns | p50 {q(0.5):.0f} | p90 {q(0.9):.0f} | p95 {q(0.95):.0f} | max {d[-1]:.0f}")
km = [r for r in rows if (r.get("Kernel Name") or "") == "tf_glm_router_mm"]
ks = [r for r in rows if (r.get("Kernel Name") or "") == "tf_glm_slice_sum"]
print(f"- rows in the CSV: {len(rows)} (the budget is 512 kernel rows + ncu banners)")
summ("router_mm", dur(km))
summ("slice_sum", dur(ks))
dm, ds = dur(km), dur(ks)
if dm and ds:
    k50 = (dm[int(len(dm)*0.5)] + ds[int(len(ds)*0.5)]) * 8 * 80 / 1e6
    k95 = (dm[min(len(dm)-1,int(len(dm)*0.95))] + ds[min(len(ds)-1,int(len(ds)*0.95))]) * 8 * 80 / 1e6
    print(f"- kernel time per 2048-row chunk (8 slices x 80 layers): p50 {k50:.1f} ms | p95 {k95:.1f} ms")
    print(f"- vs the section's ~1109 ms: kernels are {k50/1109:.1%} at p50 ({k95/1109:.1%} at p95) — the rest is GAPS (host enqueue / span semantics / cross-rank serialization)")
RGNCU3
      else
        echo "- the WRAPPED restart failed (ncu overhead or refusal) — the receipt (serve log tail):" >> "$P"
        docker logs "$CONTAINER_NAME" 2>&1 | tail -8 >> "$P" || true
      fi
    else
      echo "- the ncu copy FAILED (the base image's profiler did not come over) — instrument gap recorded, fail-honest" >> "$P"
    fi
  else
    echo "- the serve refused — the receipt (container log tail):" >> "$P"
    docker logs "$CONTAINER_NAME" 2>&1 | tail -8 >> "$P" || true
  fi
  touch "$OUT/.ncurouter.done"
fi

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# phase: profchunk (SESSION-SPECIFIC; retires with the stage breakdown). The
# clean stage-level instrument: the probe config + TF_GLM_PROF=1, NO ncu (the
# round-98 capture proved the engine's paired-event spans inflate ~7x under a
# profiler). One ~2k-token prompt fires one chunk; the serve log's span lines
# itemize every stage per layer in DEVICE time (paired same-stream events —
# zig/src/families/glm/cuda_prompt_trace.zig). The read keeps the LAST chunk's
# spans (earlier ones are boot warmup + the smoke test) and aggregates
# per-(site,stage) totals — naming which stage owns the chunk and how much of
# the wall is inter-stage gaps (the round-98 bubbles verdict, stage-resolved).
# ---------------------------------------------------------------------------
if want profchunk && ! phase_done profchunk; then
  log "=== profchunk (the clean stage-level chunk profile)"
  P="$OUT/32-profchunk.md"; : > "$P"
  echo "# profchunk (the chunk's stages in clean device-event spans, no ncu)" >> "$P"
  PROFOUT="$OUT/profchunk"; mkdir -p "$PROFOUT"
  if restart DENSE_BITS=4 HEAD_BITS=4 KV_CACHE=fp8 CONTEXT=524288 PARALLEL=4 VISION=0 PATCHES=glm-router-gemm.patch TF_GLM_ROUTER_MM=1 TF_GLM_PROF=1; then
    MODEL=$(curl -s -m 5 "http://127.0.0.1:$PORT/v1/models" | python3 -c "import json,sys; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null)
    [ -n "$MODEL" ] || MODEL=local/GLM-5.3-Flash
    PTEXT="Summarize the following. $(printf 'The harbor crane maintenance cycle covers wire rope inspection, sheave wear, hydraulic fluid sampling, brake torque checks, boom deflection measurement, and corrosion protection. %.0s' $(seq 42))"
    for try in 1 2 3; do
      curl -s -m 300 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
        -d "$(python3 -c 'import json,sys; print(json.dumps({"model": sys.argv[1], "max_tokens": 8, "messages": [{"role": "user", "content": sys.argv[2]}]}))' "$MODEL" "$PTEXT")" \
        > "$PROFOUT/prompt-reply.json" 2>/dev/null || true
      [ -s "$PROFOUT/prompt-reply.json" ] && break
      sleep 10
    done
    echo "- prompt fired: $(wc -c < "$PROFOUT/prompt-reply.json" 2>/dev/null || echo 0) bytes reply (try ${try:-?})" >> "$P"
    docker logs "$CONTAINER_NAME" > "$PROFOUT/serve-prof.log" 2>&1 || true
    python3 - "$PROFOUT" >> "$P" <<'RGPC'
import re, sys
log = open(f"{sys.argv[1]}/serve-prof.log", errors="replace").read()
stages = {}
chunks = []
for m in re.finditer(r"glm prompt stage rank=(\d+) slot=\d+ start=(\d+) rows=(\d+) layer=(\d+) piece=(\S+) site=(\S+) stage=(\S+) stream=(\S+) ms=([0-9.]+)", log):
    rank, start, rows, layer, piece, site, stage, stream, ms = m.groups()
    if rank != "0":
        continue
    stages.setdefault(int(start), {}).setdefault(f"{site}/{stage}", []).append(float(ms))
for m in re.finditer(r"glm prompt chunk rank=0 slot=\d+ start=(\d+) rows=(\d+) wall_ms=([0-9.]+) spans=(\d+)", log):
    chunks.append((int(m.group(1)), int(m.group(2)), float(m.group(3)), int(m.group(4))))
if not stages:
    print("- no stage spans found (TF_GLM_PROF lines missing) — the serve log has the reason (fail-honest)"); raise SystemExit
print(f"- chunk lines (start, rows, wall_ms, spans): {chunks[-3:]}")
last_start = max(stages)
v_by_key = stages[last_start]
rows_note = next((r for (st, r, w, sp) in chunks if st == last_start), "?")
wall = next((w for (st, r, w, sp) in chunks if st == last_start), 0.0)
print(f"- the LAST chunk (start={last_start}, rows={rows_note}, wall_ms={wall:.1f}):")
tot_all = 0.0
for key in sorted(v_by_key, key=lambda k: -sum(v_by_key[k])):
    v = sorted(v_by_key[key])
    tot = sum(v)
    tot_all += tot
    print(f"-   {key}: n={len(v)} total={tot:.1f} ms p50={v[len(v)//2]:.2f} max={v[-1]:.2f}")
print(f"- ALL stage span time (rank 0, this chunk): {tot_all:.1f} ms across {sum(len(v) for v in v_by_key.values())} spans")
print(f"- chunk wall: {wall:.1f} ms -> inter-stage gaps own {wall - tot_all:.1f} ms ({(wall - tot_all) / wall:.0%} of the wall)" if wall else "- no chunk line for the last start")
RGPC
  else
    echo "- the serve refused — the receipt (container log tail):" >> "$P"
    docker logs "$CONTAINER_NAME" 2>&1 | tail -8 >> "$P" || true
  fi
  touch "$OUT/.profchunk.done"
fi

# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# phase: profload (SESSION-SPECIFIC; retires with the concurrency-cost map).
# Round 100 resolved the round-85 "1109 ms router section" as a measurement
# artifact: the engine's paired-event spans absorb device-busy time when other
# streams share the GPU (the round-85 line was taken under 4-way load; the
# clean lone-chunk map shows the router stage at ~0.2 ms/layer and the expert
# GEMMs owning ~half the wall). This phase makes the artifact ITSELF the
# instrument: the same serve under TF_GLM_PROF=1 runs (1) one lone ~14k-token
# prefill and (2) four concurrent ~2k prefills (the c4 shape), and the read
# puts the engine's own `glm prof prompt chunk` aggregates side by side — the
# stage attribution under load vs lone, i.e. where concurrency cost lands.
# ---------------------------------------------------------------------------
if want profload && ! phase_done profload; then
  log "=== profload (lone vs 4-way prefill under the engine's prof spans)"
  P="$OUT/33-profload.md"; : > "$P"
  echo "# profload (the concurrency cost, stage-attributed by the engine's own aggregates)" >> "$P"
  PROFOUT="$OUT/profload"; mkdir -p "$PROFOUT"
  if restart DENSE_BITS=4 HEAD_BITS=4 KV_CACHE=fp8 CONTEXT=524288 PARALLEL=4 VISION=0 TF_GLM_PROF=1; then
    mkreq() { python3 - "$1" "$2" <<'PFEOF'
import json, sys
words = open("/usr/share/dict/words").read().split() if False else ("harbor crane maintenance cycle covers wire rope inspection sheave wear hydraulic fluid sampling brake torque checks boom deflection measurement corrosion protection schedule").split()
n = int(sys.argv[1])
body = " ".join(words[i % len(words)] + (f"-{i%83}" if i % 17 == 0 else "") for i in range(n))
json.dump({"model": "GLM-5.3-Flash-EXL3", "messages": [{"role": "user", "content": body + "\n\nReply with exactly: read complete."}], "temperature": 0, "max_tokens": 1}, open(sys.argv[2], "w"))
PFEOF
    }
    mkreq 14000 "$PROFOUT/req-14k.json"
    for i in 1 2 3 4; do mkreq 2000 "$PROFOUT/req-2k-$i.json"; done
    L1=$(curl -s -o "$PROFOUT/resp-14k.json" -m 900 -w '%{time_total}' "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' --data @"$PROFOUT/req-14k.json") || true
    echo "- lone 14k prefill wall: ${L1:-0} s" >> "$P"
    MARK1=$(docker logs "$CONTAINER_NAME" 2>&1 | wc -l)
    T0=$(date +%s.%N)
    for i in 1 2 3 4; do
      curl -s -o "$PROFOUT/resp-2k-$i.json" -m 900 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' --data @"$PROFOUT/req-2k-$i.json" &
    done
    wait
    T1=$(date +%s.%N)
    echo "- 4-way 2k prefill wall: $(python3 -c "print(f'{$T1-$T0:.2f}')") s (4 replies: $(wc -c "$PROFOUT"/resp-2k-*.json | tail -1 | awk '{print $1}') bytes total)" >> "$P"
    docker logs "$CONTAINER_NAME" > "$PROFOUT/serve-prof.log" 2>&1 || true
    python3 - "$PROFOUT" "$MARK1" >> "$P" <<'RGPL'
import re, sys
log = open(f"{sys.argv[1]}/serve-prof.log", errors="replace").read().splitlines()
mark = int(sys.argv[2])
chunk_res = []
for line in log[mark:]:
    m = re.search(r"glm prof prompt chunk to (\d+) rows(?: \(?(\d+) calls?\)?)?: wall ([0-9.]+) ms, gpu ([0-9.]+) ms a call; (.*)", line)
    if m:
        chunk_res.append(m.groups())
if not chunk_res:
    print("- no `glm prof prompt chunk` aggregate lines in the log tail — the serve log has the reason (fail-honest)"); raise SystemExit
for rows, calls, wall, gpu, stages in chunk_res[-6:]:
    print(f"- to {rows} rows ({calls or '1'} call(s)): wall {wall} ms, gpu {gpu} ms a call")
    print(f"    stages: {stages[:400]}")
print("- read: gpu > wall means overlapping streams; the stage list under 4-way load vs the")
print("  lone 14k leg shows where concurrency cost lands (enqueue-sensitive stages inflate first).")
RGPL
  else
    echo "- the serve refused — the receipt (container log tail):" >> "$P"
    docker logs "$CONTAINER_NAME" 2>&1 | tail -8 >> "$P" || true
  fi
  touch "$OUT/.profload.done"
fi

# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# phase: pinbump (SESSION-SPECIFIC; retires with the pin decision). The pin in
# scripts/local.sh (TF_COMMIT) does NOT trigger the auto-prepare — the config
# fingerprint excludes it (the round-104 window proved the marker stayed at
# 465e48b6 and the battery measured the OLD engine throughout). This phase
# prepares EXPLICITLY: prepare.sh builds the pinned commit (the compile loop;
# the kernel cache makes it incremental) and the marker MUST move to the pin —
# the battery that follows only counts if it did. Fail-honest otherwise.
# ---------------------------------------------------------------------------
if want pinbump && ! phase_done pinbump; then
  log "=== pinbump (the explicit prepare for the pinned commit)"
  P="$OUT/02-pinbump.md"; : > "$P"
  echo "# pinbump (prepare the pinned TF_COMMIT; verify the marker moved)" >> "$P"
  PIN=$(grep "^TF_COMMIT=" scripts/local.sh 2>/dev/null | cut -d= -f2 | tr -d '"' )
  echo "- the pin: TF_COMMIT=${PIN:-unset}" >> "$P"
  BEFORE=$(sed -n "s/.*tf=\([^@]*\)@\([^ ]*\).*/\2/p" "$STATE_DIR/prepared" 2>/dev/null || echo none)
  echo "- the marker before: $BEFORE" >> "$P"
  if [ -z "$PIN" ]; then
    echo "- no pin in scripts/local.sh — nothing to prepare (fail-honest)" >> "$P"
  elif ./scripts/prepare.sh >> "$LOG" 2>&1; then
    AFTER=$(sed -n "s/.*tf=\([^@]*\)@\([^ ]*\).*/\2/p" "$STATE_DIR/prepared" 2>/dev/null || echo none)
    echo "- the marker after: $AFTER" >> "$P"
    SHORT=$(echo "$PIN" | cut -c1-8)
    if echo "$AFTER" | grep -q "^$SHORT\|^$PIN"; then
      echo "- the prepare BUILT the pin — the battery runs on the new engine" >> "$P"
    else
      echo "- the marker did NOT move to the pin — the battery would measure the old engine (fail-honest)" >> "$P"
    fi
  else
    echo "- the prepare FAILED (the campaign log has the compile output) — the receipt:" >> "$P"
    tail -12 "$LOG" >> "$P" || true
  fi
  touch "$OUT/.pinbump.done"
fi

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
