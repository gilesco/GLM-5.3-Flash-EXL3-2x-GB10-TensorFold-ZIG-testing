#!/usr/bin/env bash
# The 2-module requant proof (the authorization gate): the capture exports hin
# for ALL expert modules in one pass (the full job reuses this dir); the head
# encodes one key, the worker another; the merge is checked byte-level against
# a single-host two-key reference run on the same precomputed-H path, and the
# precomputed-H sqnr is read against the online-H reference. A failure stops
# the arc here — almost nothing spent.
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

echo "== [1/6] the capture: hin for every expert module (the head; the full job reuses this)"
$V scripts/requant/capture_h.py --src "$SRC" --work "$BASE/work-capture" \
  --hessians "$HDIR" 2>&1 | tee "$OUT/capture.log"
mapfile -t KEYS < <(grep "^KEY " "$OUT/capture.log" | cut -d" " -f2)
[ "${#KEYS[@]}" -ge 2 ] || { echo " !! the capture produced ${#KEYS[@]} keys — need 2+"; exit 1; }
KA="${KEYS[0]}"; KB="${KEYS[$(( ${#KEYS[@]} / 2 ))]}"
echo "the proof keys: $KA / $KB" | tee "$OUT/keys.txt"

echo "== [2/6] the hin sync to the worker (192.168.2.x)"
du -sh "$HDIR"
rsync -a --delete -e "ssh" "$HDIR/" "$W:$BASE/hessians/"

echo "== [3/6] the recipes (one key each for the split; both for the reference)"
printf 'tensors:\n  "%s": 3\n' "$KA" > "$OUT/recipe-a.yaml"
printf 'tensors:\n  "%s": 3\n' "$KB" > "$OUT/recipe-b.yaml"
printf 'tensors:\n  "%s": 3\n  "%s": 3\n' "$KA" "$KB" > "$OUT/recipe-ref.yaml"

echo "== [4/6] the split encodes (the head: $KA)"
$V -m exllamav3.convert -i "$SRC" -w "$BASE/work-head" -o "$OUT/out-head" \
  -b 3 -cb mcg -eb 16 -hb 16 --hessians "$HDIR" \
  -rcp "$OUT/recipe-a.yaml" 2>&1 | tee "$OUT/enc-head.log"
echo "== [4/6] the split encodes (the worker: $KB)"
ssh -o BatchMode=yes "$W" "export PATH=\$HOME/.venv-exl3/bin:\$PATH; \
  $V -m exllamav3.convert -i $WSRC -w $BASE/work-worker -o $OUT/out-worker \
  -b 3 -cb mcg -eb 16 -hb 16 --hessians $BASE/hessians \
  -rcp $OUT/recipe-b.yaml" 2>&1 | tee "$OUT/enc-worker.log"

echo "== [5/6] the single-host reference (both keys, the same precomputed-H path)"
$V -m exllamav3.convert -i "$SRC" -w "$BASE/work-ref" -o "$OUT/out-ref" \
  -b 3 -cb mcg -eb 16 -hb 16 --hessians "$HDIR" \
  -rcp "$OUT/recipe-ref.yaml" 2>&1 | tee "$OUT/enc-ref.log"

echo "== [6/6] the checks"
OK=1
for pair in "$KA:head" "$KB:worker"; do
  K="${pair%%:*}"; H="${pair#*:}"
  F="$K.safetensors"
  R="$OUT/out-ref/qtensors/$F"
  if [ "$H" = head ]; then S="$OUT/out-head/qtensors/$F"; else
    S="$OUT/fetch-$F"; ssh -o BatchMode=yes "$W" "cat $OUT/out-worker/qtensors/$F" > "$S" 2>/dev/null
  fi
  [ -f "$R" ] && [ -f "$S" ] || { echo " !! missing output for $K ($R / $S)"; OK=0; continue; }
  if cmp -s "$R" "$S"; then echo " PASS: $F byte-identical to the single-host reference"; else
    echo " FAIL: $F differs ($(stat -c%s "$R") vs $(stat -c%s "$S") bytes)"; OK=0; fi
done
echo "-- the sqnr read: the precomputed-H runs vs the online-H reference"
grep -iE "sqnr|cosine|proxy" "$OUT/enc-head.log" "$OUT/enc-worker.log" > "$OUT/sqnr-precomp.log" || true
$V -m exllamav3.convert -i "$SRC" -w "$BASE/work-online" -o "$OUT/out-online" \
  -b 3 -cb mcg -eb 16 -hb 16 -rcp "$OUT/recipe-ref.yaml" 2>&1 | tee "$OUT/enc-online.log" | grep -iE "sqnr|cosine|proxy" > "$OUT/sqnr-online.log" || true
echo "-- precomputed-H:"; cat "$OUT/sqnr-precomp.log" || true
echo "-- online-H:"; cat "$OUT/sqnr-online.log" || true
[ "$OK" = 1 ] && echo "== PROOF PASS: the split merge is byte-correct" || { echo "== PROOF FAIL"; exit 1; }
