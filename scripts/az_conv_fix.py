#!/usr/bin/env python3
"""Fix the conv1d tensors in the dequantized dir for exllamav3's GDN loader.

The canonical set (what TensorFold's native engine reads, weights.zig:372): the
BF16 splits {q,k,v}_conv1d.weight in [channels, 1, K]. The AZ checkpoint also
carries F32 mixed conv1d leftovers ([3c, 1, K]) that are NOT canonical — they
made the converter's split-cat produce 40960 channels against 24576 expected.
So: drop every F32 conv1d tensor; unsqueeze any 2-dim bf16 conv1d (a previous
squeeze's damage) back to [c, 1, K]. Idempotent.
"""
import glob, json, os, struct, sys
import torch
from safetensors.torch import load_file, save_file

OUT = sys.argv[1] if len(sys.argv) > 1 else "/home/giles/az-az-fp16"
dropped, unsqueezed = 0, 0
for f in sorted(glob.glob(OUT + "/*.safetensors")):
    t = load_file(f)
    changed = False
    for k in list(t.keys()):
        if "conv1d.weight" not in k:
            continue
        if t[k].dtype == torch.float32:
            del t[k]; changed = True; dropped += 1
        elif t[k].dim() == 2:
            t[k] = t[k].unsqueeze(1).contiguous(); changed = True; unsqueezed += 1
    if changed:
        tmp = f + ".tmp"
        save_file(t, tmp)
        os.replace(tmp, f)
print(f"dropped {dropped} F32 mixed convs, unsqueezed {unsqueezed} bf16 splits", flush=True)

# the index: rebuilt from the shards as they now stand (everything is 2-byte here)
names, sizes = {}, 0
for f in sorted(glob.glob(OUT + "/*.safetensors")):
    fh = open(f, "rb"); n = struct.unpack("<Q", fh.read(8))[0]
    hdr = json.loads(fh.read(n)); fh.close()
    for k, v in hdr.items():
        if k != "__metadata__":
            names[k] = os.path.basename(f)
            sizes += int(torch.Size(v["shape"]).numel()) * 2
ip = os.path.join(OUT, "model.safetensors.index.json")
idx = json.load(open(ip))
idx["weight_map"] = names
idx["metadata"]["total_size"] = sizes
json.dump(idx, open(ip, "w"))
print("index rewritten:", len(names), "tensors,", sizes >> 30, "GiB", flush=True)

