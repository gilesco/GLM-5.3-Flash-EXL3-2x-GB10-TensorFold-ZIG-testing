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
#   parity     the knob-for-knob mirror of production on this engine:
#              DENSE_BITS=8 HEAD_BITS=8 KV_CACHE=fp8 CONTEXT=524288 + DFlash2,
#              benched with the fixed matrix options (PARALLEL=4, fallback 2)
#   candidate  dense4fp8 at the same window — the ladder candidate's cell
#   kvqual     VALIDATION §2 gates on the fp8-KV config: determinism, drafted=plain
#              (the 0.6.x "drafted replies still equal serial ones" guarantee),
#              resumed=fresh, concurrent=solo, sampled-seed determinism
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

PHASES="${PHASES:-preflight reference parity candidate kvqual depth cliff prefill rung summary}"
want() { case " $PHASES " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

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
  local kv
  if prod_running; then die "production (glm53-flash-tf) is running — stop it first; this script never touches it"; fi
  log "restarting the testing stack (${*:-defaults})"
  ./scripts/stop.sh < /dev/null >/dev/null 2>&1 || true
  if [[ $# -gt 0 ]]; then env "$@" ./scripts/start.sh < /dev/null >>"$LOG" 2>&1
  else ./scripts/start.sh < /dev/null >>"$LOG" 2>&1; fi
  log "serving: $(serve_line)"
  log "$(roce_state)"
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
  local label="$1" dest="$2"
  mkdir -p "$dest"
  PYTHONUNBUFFERED=1 "$BENCH" run \
    --base-url "http://127.0.0.1:$PORT/v1" \
    --model "$SERVED_NAME" \
    --format openai \
    --tokenizer "$MODEL_ID" \
    --seed 42 --perf --parallel 4 --timeout 240 --no-live \
    --label "$label" --output-dir "$dest" --json-file "$dest/result.json" \
    2>&1 | tee "$dest/bench.log" | tail -20
}

chat() {           # one chat completion; $1 = request file, optional $2 = curl timeout
  curl -s -m "${2:-900}" "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' --data @"$1"
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
  if prod_running; then
    if [[ "${PROMPT:-1}" == 1 ]]; then
      log "PRODUCTION IS RUNNING. Nothing in this script touches it, but the bench"
      log "needs the GPUs. Stop production yourself (the -tip stack's stop), then"
      read -r -p "press Enter once production is stopped (Ctrl-C to abort): " _
    else
      die "production is running and PROMPT=0 — stop it first, then re-run"
    fi
  fi
  prod_running && die "production still running — this script will not continue"
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
  grep -A11 "^| Test" "$OUT/$label"/2026/*/*/*.md 2>/dev/null | head -13 >> "$P" || true
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
# phase: kvqual — VALIDATION §2 gates on the fp8-KV config (dense8, 512K, DFlash2)
# ---------------------------------------------------------------------------
if want kvqual && ! phase_done kvqual; then
  log "=== kvqual"
  P="$OUT/03-kvqual.md"; : > "$P"
  echo "# kvqual (VALIDATION §2 on fp8 KV, dense8, 512K window)" >> "$P"
  restart_fb DENSE_BITS=8 HEAD_BITS=8 KV_CACHE=fp8 CONTEXT=524288 PARALLEL=4

  # g1 determinism: the same greedy probe twice
  req_file "$OUT/g1-req.json" "$REF_PROMPT"
  chat "$OUT/g1-req.json" > "$OUT/g1-a.json"; chat "$OUT/g1-req.json" > "$OUT/g1-b.json"
  if [[ "$(sha_msg "$OUT/g1-a.json")" == "$(sha_msg "$OUT/g1-b.json")" ]]; then
    echo "- PASS g1 determinism: two greedy runs identical ($(sha_msg "$OUT/g1-a.json" | head -c 16)...)" >> "$P"
  else
    echo "- NOTE g1 determinism: greedy runs differ (engine nondeterminism) — recorded, not gating" >> "$P"
  fi

  # g2 drafted = plain (the 0.6.x guarantee, replicated on the native engine)
  DRAFTED_SHA=$(sha_msg "$OUT/g1-a.json")
  restart DENSE_BITS=8 HEAD_BITS=8 KV_CACHE=fp8 CONTEXT=524288 PARALLEL=4 DRAFTS=0
  chat "$OUT/g1-req.json" > "$OUT/g2-plain.json"
  PLAIN_SHA=$(sha_msg "$OUT/g2-plain.json")
  if [[ "$DRAFTED_SHA" == "$PLAIN_SHA" ]]; then
    echo "- PASS g2 drafted=plain: DFlash2-drafted and --no-drafts greedy outputs identical" >> "$P"
  else
    echo "- FAIL g2 drafted=plain: drafted $(echo "$DRAFTED_SHA" | head -c 16)... vs plain $(echo "$PLAIN_SHA" | head -c 16)..." >> "$P"
    diff <(content_of "$OUT/g1-a.json") <(content_of "$OUT/g2-plain.json") | head -5 >> "$P" || true
  fi

  # g3 resumed = fresh (the prompt cache's snap hooks)
  restart_fb DENSE_BITS=8 HEAD_BITS=8 KV_CACHE=fp8 CONTEXT=524288 PARALLEL=4
  T1="Remember these three codes: alpha-7391, bravo-2250, charlie-8118. Reply with exactly: codes noted."
  T2="What were the three codes? Answer with just the three codes, comma separated."
  req_file "$OUT/g3-t2.json" "$T2"
  chat "$OUT/g3-t2.json" > "$OUT/g3-fresh.json"          # cold server: turn 2 alone
  req_file "$OUT/g3-t1.json" "$T1"; chat "$OUT/g3-t1.json" > /dev/null
  req_file "$OUT/g3-both.json" "$T1

$T2"
  chat "$OUT/g3-both.json" > "$OUT/g3-resumed.json"      # turn 2 after turn 1 (cache hit)
  FRESH_C=$(content_of "$OUT/g3-fresh.json"); RESUMED_C=$(content_of "$OUT/g3-resumed.json")
  if [[ "$FRESH_C" == "$RESUMED_C" ]]; then
    echo "- PASS g3 resumed=fresh: turn-2 answer identical with and without the cached prefix" >> "$P"
  else
    echo "- FAIL g3 resumed=fresh: fresh='$FRESH_C' resumed='$RESUMED_C'" >> "$P"
  fi

  # g4 concurrent = solo
  chat "$OUT/g1-req.json" > "$OUT/g4-solo.json"
  for i in 1 2 3 4; do chat "$OUT/g1-req.json" > "$OUT/g4-c$i.json" & done; wait
  SOLO_SHA=$(sha_msg "$OUT/g4-solo.json"); OK=1
  for i in 1 2 3 4; do [[ "$(sha_msg "$OUT/g4-c$i.json")" == "$SOLO_SHA" ]] || OK=0; done
  if [[ "$OK" == 1 ]]; then
    echo "- PASS g4 concurrent=solo: 4 concurrent greedy outputs identical to solo" >> "$P"
  else
    echo "- FAIL g4 concurrent=solo: at least one concurrent output differs from solo" >> "$P"
  fi

  # g5 sampled determinism (seeded, twice)
  python3 - "$OUT/g5-req.json" <<'EOF'
import json, sys
json.dump({"model": "GLM-5.3-Flash-EXL3",
           "messages": [{"role": "user", "content": "List the first eight prime numbers in order, comma separated, then state the sum."}],
           "temperature": 0.7, "top_p": 0.95, "seed": 42, "max_tokens": 512}, open(sys.argv[1], "w"))
EOF
  chat "$OUT/g5-req.json" > "$OUT/g5-a.json"; chat "$OUT/g5-req.json" > "$OUT/g5-b.json"
  if [[ "$(sha_msg "$OUT/g5-a.json")" == "$(sha_msg "$OUT/g5-b.json")" ]]; then
    echo "- PASS g5 sampled determinism: seeded sample identical across two runs" >> "$P"
  else
    echo "- NOTE g5 sampled determinism: sampled runs differ (recorded; greedy gates above are the binding ones)" >> "$P"
  fi
  touch "$OUT/.kvqual.done"
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
  python3 - "$OUT" "$FILL_WORDS" <<'EOF'
import json, sys
words, n = sys.argv[2].split(), 14000
body = " ".join(words[i % len(words)] + (f"-{i%83}" if i % 17 == 0 else "") for i in range(n))
json.dump({"model": "GLM-5.3-Flash-EXL3",
           "messages": [{"role": "user", "content": body + "\n\nReply with exactly: read complete."}],
           "temperature": 0, "max_tokens": 512}, open(f"{sys.argv[1]}/prefill-req.json", "w"))
EOF
  chat "$OUT/prefill-req.json" > "$OUT/prefill-resp.json" || true
  python3 - "$OUT/prefill-resp.json" >> "$P" <<'EOF'
import json, sys
try:
    u = json.load(open(sys.argv[1]))["usage"]
    print(f"- usage: {u.get('prompt_tokens')} prompt tokens (aim ~14k); cached: {u.get('prompt_tokens_details')}")
except Exception as e:
    print(f"- prefill probe failed to parse: {e}")
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
# phase: summary
# ---------------------------------------------------------------------------
if want summary; then
  log "=== summary"
  S="$OUT/CAMPAIGN.md"
  {
    echo "# Replication campaign $RUN_ID"
    echo
    echo "## Gates"
    grep -hE "^- (PASS|FAIL|NOTE|depth|batch|usage|- server|- RoCE)" "$OUT"/0*.md 2>/dev/null || true
    echo
    echo "## Phase receipts"
    for f in "$OUT"/0*.md; do echo "### $(basename "$f")"; cat "$f"; echo; done
    echo "## Way forward"
    echo "- Read the PASS/FAIL lines above against docs/VALIDATION.md §2 and the depth bar"
    echo "  (production 93.9%). fp8 KV passing g1-g5 replicates the 0.6.x guarantee on the"
    echo "  native engine; the depth column says whether the 512K window is servable."
    echo "- The parity vs dense4fp8 bench cells pick the ladder config; the cliff and"
    echo "  prefill receipts feed the next upstream note."
  } > "$S"
  log "campaign complete: $S"
  tail -40 "$S"
fi
log "done."
