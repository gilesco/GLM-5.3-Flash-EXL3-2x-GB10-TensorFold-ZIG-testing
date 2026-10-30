#!/usr/bin/env python3
"""Squeeze the conv1d weights in the dequantized fp16 dir: HF-style [dim, 1, K]
-> exllamav3's expected [dim, K]. Only rewrites the shards that hold conv1d
tensors; everything else is untouched (the convert's checkpointed work survives).
"""
import glob, json, os, sys
import torch
from safetensors.torch import load_file, save_file

OUT = sys.argv[1] if len(sys.argv) > 1 else "/home/<user>/az-az-fp16"
fixed, files = 0, 0
for f in sorted(glob.glob(OUT + "/*.safetensors")):
    t = load_file(f)
    hits = [k for k in t if "conv1d" in k and t[k].dim() == 3]
    if not hits:
        continue
    files += 1
    for k in hits:
        t[k] = t[k].squeeze(1).contiguous()
        fixed += 1
    tmp = f + ".tmp"
    save_file(t, tmp)
    os.replace(tmp, f)
    print(f"  {os.path.basename(f)}: squeezed {len(hits)} conv1d tensors", flush=True)
print(f"done: {fixed} tensors across {files} files", flush=True)
