#!/usr/bin/env bash
# The AZ requant window — mcg edition (run after production is stopped): stream-
# dequant the AZ auto-round int4 checkpoint to fp16 (already done), exllamav3-
# convert it to EXL3 4bpw WITH -cb mcg, place it as a local model, sync the
# worker (retry-wrapped), then run the battery + the probes (azqual).
#
# Why -cb mcg: the native engine's admit() (families/glm/cuda_exl3.zig) reads
# 4-bit routed experts on the "mcg" codebook only, with `bits` as a JSON INTEGER.
# The 1.6.4 convert defaults to mul1 + float bits — campaign-20261026-010238's
# azqual refusal receipt. The mul1 output stays parked at az-az-exl3 until the
# mcg verdict.
# Hours of GPU work; receipts land in results/<campaign>/.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
source ./config.sh
V=~/.venv-exl3/bin/python
export PATH="$HOME/.venv-exl3/bin:$PATH"   # torch's JIT build finds the venv's ninja via PATH
AZSRC=$(echo ~/.cache/huggingface/hub/models--azampatti--GLM-5.3-Flash-AZ/snapshots/*/)
OUTDIR=/home/giles/az-az-fp16              # the fp16 intermediate (dequant done; az_conv_fix is idempotent on it)
WORK=/home/giles/az-az-work-mcg            # fresh: the mul1 qtensors are invalid for a mcg convert
EXLOUT=/home/giles/az-az-exl3-mcg          # fresh output dir
LOCAL=~/.cache/huggingface/hub/models--local--GLM-5.3-Flash-AZ-EXL3/snapshots/main

if [[ ! -f "$OUTDIR/model.safetensors.index.json" ]]; then
  echo "[az] 1/4 stream-dequant (cpu, ~1-2 h)"
  $V scripts/az_dequant.py "$OUTDIR"
else
  echo "[az] 1/4 dequant already done (index present)"
fi

if [[ ! -f "$EXLOUT/model.safetensors.index.json" ]]; then
  echo "[az] 1b conv1d fix (idempotent on the already-fixed fp16: the dim-2 guard no-ops)"
  ~/.venv-exl3/bin/python scripts/az_conv_fix.py "$OUTDIR"
  if [[ -d "$WORK/qtensors" ]]; then
    echo "[az] 2/4 exllamav3 convert — resuming the checkpointed work (qtensors present)"
    CVARG="-r"
  else
    echo "[az] 2/4 exllamav3 convert (gpu, ~12 h): -cb mcg -b 4 -hb 16 -eb 16 -vb 16"
    rm -rf "$WORK"; mkdir -p "$WORK" "$EXLOUT"
    CVARG=""
  fi
  # convert_model.py uses package-relative imports — run it as a module, never as a file.
  # -cb mcg: the engine's admit() contract (see header). -vb 16: the vision tower
  # stays unquantized (the AZ checkpoint keeps it bf16; the validated-tower default
  # would silently quantize it to 6bpw).
  $V -m exllamav3.conversion.convert_model $CVARG -i "$OUTDIR" -w "$WORK" -o "$EXLOUT" -cb mcg -b 4 -hb 16 -eb 16 -vb 16
else
  echo "[az] 2/4 convert already done (index present)"
fi
echo "[az] 2c engine-shape check (integer bits + mcg; byte-surgical normalization + assert)"
$V scripts/az_bits_fix.py "$EXLOUT"

echo "[az] 3/4 place as the local model + sync the worker (retry-wrapped)"
mkdir -p "$LOCAL"
find "$LOCAL" -mindepth 1 -delete          # fresh placement: the mcg bytes replace the refused mul1 wholesale
rsync -a "$EXLOUT"/ "$LOCAL"/
# The worker sync: 6 attempts, file-granular resume, ssh keepalives. The 4 silent
# single-invocation deaths of the 2026-10-26 night are beaten by retry+resume.
SC=/tmp/az-worker-sync.console; : > "$SC"
ssh -o BatchMode=yes "$WORKER" "mkdir -p $LOCAL"
rc=1
for a in 1 2 3 4 5 6; do
  echo "[sync] attempt $a start $(date +%H:%M:%S)" >> "$SC"
  rsync -a --partial --info=progress2 --timeout=120 \
    -e "ssh -o ServerAliveInterval=20 -o ServerAliveCountMax=6 -o BatchMode=yes" \
    "$LOCAL/" "$WORKER:$LOCAL/" >> "$SC" 2>&1
  rc=$?
  echo "[sync] attempt $a exit=$rc $(date +%H:%M:%S)" >> "$SC"
  [ "$rc" -eq 0 ] && break
  sleep 15
done
if [ "$rc" -ne 0 ]; then echo "[az] the worker sync failed after 6 attempts — see $SC"; exit 1; fi

echo "[az] 4/4 the battery + the probes (the campaign: azqual)"
exec env PHASES="azqual summary" ./scripts/run-qualification.sh
