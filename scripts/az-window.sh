#!/usr/bin/env bash
# The AZ requant window (run after production is stopped): stream-dequant the AZ
# auto-round int4 checkpoint to fp16, exllamav3-convert it to EXL3 4bpw, place it
# as a local model, sync the worker, then run the battery + the probes (azqual).
# Hours of GPU work; receipts land in results/<campaign>/.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
source ./config.sh
V=~/.venv-exl3/bin/python
export PATH="$HOME/.venv-exl3/bin:$PATH"   # torch's JIT build finds the venv's ninja via PATH
AZSRC=$(echo ~/.cache/huggingface/hub/models--azampatti--GLM-5.3-Flash-AZ/snapshots/*/)
OUTDIR=/home/giles/az-az-fp16
WORK=/home/giles/az-az-work
EXLOUT=/home/giles/az-az-exl3
LOCAL=~/.cache/huggingface/hub/models--local--GLM-5.3-Flash-AZ-EXL3/snapshots/main

if [[ ! -f "$OUTDIR/model.safetensors.index.json" ]]; then
  echo "[az] 1/4 stream-dequant (cpu, ~1-2 h)"
  $V scripts/az_dequant.py "$OUTDIR"
else
  echo "[az] 1/4 dequant already done (index present)"
fi

if [[ ! -f "$EXLOUT/model.safetensors.index.json" ]]; then
  echo "[az] 1b squeeze the conv1d weights ([dim,1,K] -> [dim,K]; exllamav3's update kernel is strict)"
  ~/.venv-exl3/bin/python scripts/az_squeeze_conv.py "$OUTDIR"
  if [[ -d "$WORK/qtensors" ]]; then
    echo "[az] 2/4 exllamav3 convert — resuming the checkpointed work (qtensors present)"
    CVARG="-r"
  else
    echo "[az] 2/4 exllamav3 convert (gpu, hours): -b 4 -hb 16 -eb 16"
    rm -rf "$WORK"; mkdir -p "$WORK" "$EXLOUT"
    CVARG=""
  fi
  # convert_model.py uses package-relative imports — run it as a module, never as a file.
  # -vb 16: the vision tower stays unquantized (the AZ checkpoint keeps it bf16; the
  # validated-tower default would silently quantize it to 6bpw).
  $V -m exllamav3.conversion.convert_model $CVARG -i "$OUTDIR" -w "$WORK" -o "$EXLOUT" -b 4 -hb 16 -eb 16 -vb 16
else
  echo "[az] 2/4 convert already done (index present)"
fi

echo "[az] 3/4 place as the local model + sync the worker"
mkdir -p "$LOCAL"
cp -a "$EXLOUT"/. "$LOCAL"/
rsync -a "$LOCAL/" "$WORKER:$LOCAL/"

echo "[az] 4/4 the battery + the probes (the campaign: azqual)"
exec env PHASES="azqual summary" ./scripts/run-qualification.sh
