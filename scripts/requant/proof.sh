#!/usr/bin/env bash
# The requant proof (the authorization gate). The capture exports hin for EVERY
# budgeted linear in one pass on the head (the full job reuses this dir);
# recipes are GENERATED from the captured key list — the converter's recipe
# rule demands full coverage of every budgeted tensor, so each host's recipe is
# the catch-all "*": 16 (unquantized store, no trellis cost) plus its half of
# the keys at 3; the reference recipe names every key at 3. With --hessians +
# hin files the conversion runs uncalibrated (no forward passes — no
# calibration state chain), so the encodes are pure quantize and the worker
# needs no kernel patch. Checks: every owner key present in the single-host
# reference with an identical size (one batched listing per host, joined
# locally); a deterministic sample (every 500th key plus each half's ends)
# byte-compared; one catch-all 16-bit file cross-checked; the precomputed-H
# sqnr read from the runs' own prints (the quality bar stays the final bench
# battery). A failure stops the arc here — almost nothing spent.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/../.."
source ./config.sh
export PATH="$HOME/.venv-exl3/bin:$PATH"
V=python
SRC="$MODEL_SNAPSHOT"
W=$(grep "^WORKER=" scripts/local.sh | cut -d= -f2)
WSRC=$(ssh -o BatchMode=yes "$W" "ls -d ~/.cache/huggingface/hub/models--local--GLM-5.3-Flash-AZ-Focus-EXL3/snapshots/main")
BASE=/home/giles/requant
HDIR="$BASE/hessians"
OUT="$BASE/proof"
mkdir -p "$HDIR" "$OUT"

echo "== [1/7] the capture: hin for every budgeted linear (the head; the full job reuses this)"
if [ ! -f "$OUT/capture.log" ] || ! grep -q "^KEY " "$OUT/capture.log"; then
  $V scripts/requant/capture_h.py --src "$SRC" --work "$BASE/work-capture" \
    --hessians "$HDIR" 2>&1 | tee "$OUT/capture.log"
fi
mapfile -t KEYS < <(grep "^KEY " "$OUT/capture.log" | cut -d" " -f2)
N=${#KEYS[@]}
[ "$N" -ge 4 ] || { echo " !! the capture produced $N keys — need 4+"; exit 1; }
HALF=$((N / 2))
du -sh "$HDIR"
echo "   keys: $N (the head encodes 0..$((HALF-1)), the worker $HALF..$((N-1)))"
printf '%s\n' "${KEYS[@]:0:HALF}" | sed 's/$/.safetensors/' > "$OUT/keys-a.txt"
printf '%s\n' "${KEYS[@]:HALF}" | sed 's/$/.safetensors/' > "$OUT/keys-b.txt"

echo "== [2/7] the recipes (generated; coverage is mandatory, the catch-all 16 carries the rest)"
{
  echo "tensors:"
  echo '  "*": 16'
  while IFS= read -r K; do printf '  "%s": 3\n' "${K%.safetensors}"; done < "$OUT/keys-a.txt"
} > "$OUT/recipe-a.yaml"
{
  echo "tensors:"
  echo '  "*": 16'
  while IFS= read -r K; do printf '  "%s": 3\n' "${K%.safetensors}"; done < "$OUT/keys-b.txt"
} > "$OUT/recipe-b.yaml"
{
  echo "tensors:"
  for K in "${KEYS[@]}"; do printf '  "%s": 3\n' "$K"; done
} > "$OUT/recipe-ref.yaml"
wc -l "$OUT"/recipe-*.yaml

echo "== [3/7] the hin + recipe sync to the worker (192.168.2.x)"
ssh -o BatchMode=yes "$W" "mkdir -p $BASE/proof"
rsync -a -e "ssh" "$HDIR/" "$W:$BASE/hessians/"
rsync -a -e "ssh" "$OUT/recipe-b.yaml" "$OUT/keys-b.txt" "$W:$BASE/proof/"
ssh -o BatchMode=yes "$W" "df -h /home/giles | tail -1"

echo "== [4/7] the split encodes (the worker launched first, in the background)"
ssh -o BatchMode=yes "$W" "nohup bash -c 'export PATH=\$HOME/.venv-exl3/bin:\$PATH; \
  python -m exllamav3.convert -i $WSRC -w $BASE/work-worker -o $BASE/proof/out-worker \
  -b 3 -cb mcg -eb 16 -hb 16 -mb 4 -vb 16 --hessians $BASE/hessians \
  -rcp $BASE/proof/recipe-b.yaml > $BASE/proof/enc-worker.console 2>&1; \
  echo WORKER-ENCODE-EXIT-\$? >> $BASE/proof/enc-worker.console' >/dev/null 2>&1 & echo worker-encode-launched"

echo "== [4/7] the split encodes (the head: keys 0..$((HALF-1)), blocking)"
$V -m exllamav3.convert -i "$SRC" -w "$BASE/work-head" -o "$OUT/out-head" \
  -b 3 -cb mcg -eb 16 -hb 16 -mb 4 -vb 16 --hessians "$HDIR" \
  -rcp "$OUT/recipe-a.yaml" 2>&1 | tee "$OUT/enc-head.log"

echo "== [5/7] the single-host reference (every key at 3, the same precomputed-H path; the worker's half overlaps)"
$V -m exllamav3.convert -i "$SRC" -w "$BASE/work-ref" -o "$OUT/out-ref" \
  -b 3 -cb mcg -eb 16 -hb 16 -mb 4 -vb 16 --hessians "$HDIR" \
  -rcp "$OUT/recipe-ref.yaml" 2>&1 | tee "$OUT/enc-ref.log"

echo "== waiting for the worker's encode"
for _ in $(seq 1 720); do
  if ssh -o BatchMode=yes "$W" "grep -q WORKER-ENCODE-EXIT $BASE/proof/enc-worker.console 2>/dev/null" 2>/dev/null; then break; fi
  ssh -o BatchMode=yes "$W" "pgrep -f exllamav3.convert >/dev/null" 2>/dev/null || { echo " !! the worker's encode process died without a marker"; break; }
  sleep 30
done
ssh -o BatchMode=yes "$W" "cat $BASE/proof/enc-worker.console" | tee "$OUT/enc-worker.log" | tail -3
grep -q "WORKER-ENCODE-EXIT-0" "$OUT/enc-worker.log" || { echo " !! the worker's encode did not exit 0"; exit 1; }

echo "== [6/7] the checks"
OK=1
# one batched listing + size dump per host; everything compared locally
ssh -o BatchMode=yes "$W" "cd $BASE/proof/out-worker/qtensors && stat -c'%n %s' *.safetensors" > "$OUT/worker-qt.sizes" 2>/dev/null \
  || { echo " !! the worker's qtensors dir is missing"; exit 1; }
( cd "$OUT/out-ref/qtensors" && stat -c'%n %s' *.safetensors ) > "$OUT/ref-qt.sizes"
( cd "$OUT/out-head/qtensors" && stat -c'%n %s' *.safetensors ) > "$OUT/head-qt.sizes"

check_half() { # $1 = keys file, $2 = sizes file, $3 = label
  local bad=0 K RS SS
  while IFS= read -r K; do
    RS=$(awk -v k="$K" '$1==k{print $2; found=1} END{if(!found)print "MISSING"}' "$OUT/ref-qt.sizes")
    SS=$(awk -v k="$K" '$1==k{print $2; found=1} END{if(!found)print "MISSING"}' "$2")
    if [ "$RS" = MISSING ] || [ "$SS" = MISSING ]; then
      echo " !! missing: $K (ref: $RS, $3: $SS)"; bad=$((bad+1))
    elif [ "$RS" != "$SS" ]; then
      echo " FAIL: $K size $RS vs $SS"; bad=$((bad+1))
    fi
  done < "$1"
  echo "   $3: $(( $(wc -l < "$1") - bad )) ok, $bad problems"
  [ "$bad" -eq 0 ]
}
check_half "$OUT/keys-a.txt" "$OUT/head-qt.sizes" "the head" || OK=0
check_half "$OUT/keys-b.txt" "$OUT/worker-qt.sizes" "the worker" || OK=0

# the byte sample: every 500th key plus the first and last five of each half
sample_check() { # $1 = keys file, $2 = fetch mode (head|worker)
  local LINES idx=0 BYTE=0 BYTEFAIL=0 K R S
  LINES=$(wc -l < "$1")
  while IFS= read -r K; do
    if [ "$idx" -ge 5 ] && [ "$idx" -lt $((LINES - 5)) ] && [ $((idx % 500)) -ne 0 ]; then idx=$((idx+1)); continue; fi
    R="$OUT/out-ref/qtensors/$K"
    if [ "$2" = head ]; then S="$OUT/out-head/qtensors/$K"; else
      S="$OUT/fetch-$K"; ssh -o BatchMode=yes "$W" "cat $BASE/proof/out-worker/qtensors/$K" > "$S" 2>/dev/null
    fi
    if [ -f "$R" ] && [ -f "$S" ]; then
      if cmp -s "$R" "$S"; then BYTE=$((BYTE+1)); else echo " FAIL: $K byte-differs"; BYTEFAIL=$((BYTEFAIL+1)); fi
    else echo " !! sample file missing: $K"; BYTEFAIL=$((BYTEFAIL+1)); fi
    idx=$((idx+1))
  done < "$1"
  echo "   $2 byte-compared: $BYTE files, $BYTEFAIL failures"
  [ "$BYTEFAIL" -eq 0 ]
}
sample_check "$OUT/keys-a.txt" head || OK=0
sample_check "$OUT/keys-b.txt" worker || OK=0

# one catch-all 16-bit file cross-checked (deterministic dequant on both hosts)
F16=$(grep -vFf "$OUT/keys-a.txt" "$OUT/head-qt.sizes" | head -1 | cut -d" " -f1 || true)
if [ -n "$F16" ]; then
  ssh -o BatchMode=yes "$W" "cat $BASE/proof/out-worker/qtensors/$F16" > "$OUT/fetch16" 2>/dev/null || true
  if [ -f "$OUT/fetch16" ] && cmp -s "$OUT/out-head/qtensors/$F16" "$OUT/fetch16"; then
    echo "   PASS: catch-all 16-bit $F16 identical on both hosts"
  else echo " FAIL: catch-all 16-bit cross-check: $F16"; OK=0; fi
fi

echo "== [7/7] the sqnr read + hygiene"
grep -iE "sqnr|cosine|proxy" "$OUT/enc-head.log" "$OUT/enc-worker.log" "$OUT/enc-ref.log" > "$OUT/sqnr-precomp.log" || true
echo "-- precomputed-H sqnr (head/worker/ref):"; head -12 "$OUT/sqnr-precomp.log" || true
rm -f "$OUT"/fetch-* "$OUT/fetch16"
( cd "$OUT/out-head/qtensors" && ls *.safetensors | grep -vFf "$OUT/keys-a.txt" || true ) | \
  while IFS= read -r F; do rm -f "$OUT/out-head/qtensors/$F"; done
ssh -o BatchMode=yes "$W" "cd $BASE/proof/out-worker/qtensors && ls *.safetensors | grep -vFf $BASE/proof/keys-b.txt | xargs -r rm -f"
echo "-- disk after cleanup:"; df -h /home/giles | tail -1
ssh -o BatchMode=yes "$W" "df -h /home/giles | tail -1"
[ "$OK" = 1 ] && echo "== PROOF PASS: the split merge is byte-correct (sampled)" || { echo "== PROOF FAIL"; exit 1; }
