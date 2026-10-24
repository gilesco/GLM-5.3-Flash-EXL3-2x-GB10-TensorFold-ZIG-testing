#!/usr/bin/env python3
"""Stream-dequantize the AZ checkpoint (auto-round int4, GPTQ layout) to fp16.

Expert tensors arrive as (qweight [in/8, out] I32, qzeros [groups, out/8] I32,
scales [groups, out] F16) with group 128; everything else passes through.
Output: a plain fp16 HF dir (config minus quantization_config + a correct
safetensors index) that exllamav3's converter accepts. RAM stays per-shard.

The GPTQ zero-point convention (+1 offset or not) is auto-detected on the first
expert tensor: the wrong convention shifts every weight by exactly one scale
step, which shows as a huge |mean| — the right one dequantizes symmetric.
"""
import glob, json, os, struct, sys

import numpy as np
from safetensors.numpy import load_file, save_file

SRC = sorted(glob.glob("/home/giles/.cache/huggingface/hub/models--azampatti--GLM-5.3-Flash-AZ/snapshots/*/"))[0]
OUT = sys.argv[1] if len(sys.argv) > 1 else "/home/giles/az-az-fp16"
GROUP = 128
FLUSH_BYTES = 4 << 30

os.makedirs(OUT, exist_ok=True)

def unpack_i4(arr):  # [.., n] int32 -> [.., n*8] int8, low nibble first
    sh = arr.shape
    out = np.empty((*sh[:-1], sh[-1] * 8), dtype=np.int8)
    for i in range(8):
        out[..., i::8] = ((arr >> (4 * i)) & 0xF).astype(np.int8)
    return out

def dequant(qw, qz, sc):
    q = unpack_i4(qw).astype(np.float32)              # [in, out]
    z = unpack_i4(qz).astype(np.float32)              # [groups, out]
    g = q.shape[0] // GROUP
    q = q.reshape(g, GROUP, -1)
    z = z[:, None, :]
    s = sc.astype(np.float32)[:, None, :]
    w = (q - z) * s                                   # [groups, GROUP, out]
    if CONVENTION == "plus1":
        w = w - s
    return w.reshape(-1, w.shape[-1]).astype(np.float16).T.copy()  # [out, in] f16

CONVENTION = None
total = 0
shard_no = 0
acc, acc_bytes = {}, 0

def flush():
    global shard_no, acc, acc_bytes, total
    shard_no += 1
    name = f"model-{shard_no:05d}-of-NNNNN.safetensors"
    save_file(acc, os.path.join(OUT, name))
    shard_total = sum(int(np.prod(v.shape)) * v.itemsize for v in acc.values())
    total += shard_total
    print(f"  {name}: {len(acc)} tensors, {shard_total >> 30} GiB (running {total >> 30} GiB)", flush=True)
    acc, acc_bytes = {}, 0

files = sorted(glob.glob(SRC + "model-*.safetensors"))
print(f"source: {SRC} ({len(files)} shards)", flush=True)
for f in files:
    tensors = load_file(f)
    for k in list(tensors.keys()):
        v = tensors[k]
        if k.endswith(".qweight"):
            base = k[: -len(".qweight")]
            w = dequant(v, tensors[base + ".qzeros"], tensors[base + ".scales"])
            if CONVENTION is None:
                # decisive check on this tensor: mean |w| under each convention —
                # the wrong one leaves every weight offset by one scale step
                qw, qz, sc = v, tensors[base + ".qzeros"], tensors[base + ".scales"]
                q = unpack_i4(qw).astype(np.float32).reshape(-1, GROUP, w.shape[-1])
                z = unpack_i4(qz).astype(np.float32)[:, None, :]
                s = sc.astype(np.float32)[:, None, :]
                mp = float(np.abs((q - z) * s).mean())
                m1 = float(np.abs(((q - z) * s) - s).mean())
                CONVENTION = "plain" if mp <= m1 else "plus1"
                print(f"convention: {CONVENTION} (mean|w| plain {mp:.4g} vs plus1 {m1:.4g})", flush=True)
            acc[base + ".weight"] = w
            acc_bytes += w.nbytes
            del tensors[base + ".qzeros"], tensors[base + ".scales"]
        elif k.endswith(".qzeros") or k.endswith(".scales"):
            continue
        else:
            acc[k] = v.astype(np.float16) if v.dtype == np.float32 else v
            acc_bytes += acc[k].nbytes
        del tensors[k]
    if acc_bytes >= FLUSH_BYTES:
        flush()
    del tensors
flush()

# the index with the real shard count
shards = sorted(glob.glob(OUT + "/model-*-of-NNNNN.safetensors"))
n = len(shards)
weight_map, tsize = {}, 0
for i, sh in enumerate(shards, 1):
    real = os.path.join(OUT, f"model-{i:05d}-of-{n:05d}.safetensors")
    os.rename(sh, real)
    with open(real, "rb") as fh:
        hlen = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(hlen))
    for k, v in hdr.items():
        if k != "__metadata__":
            weight_map[k] = os.path.basename(real)
            tsize += int(np.prod(v["shape"])) * 2
json.dump({"metadata": {"total_size": tsize}, "weight_map": weight_map},
          open(os.path.join(OUT, "model.safetensors.index.json"), "w"))
cfg = json.load(open(SRC + "config.json"))
cfg.pop("quantization_config", None)
cfg.setdefault("text_config", {})["dtype"] = "float16"
json.dump(cfg, open(os.path.join(OUT, "config.json"), "w"), indent=2)
for extra in ("generation_config.json", "tokenizer_config.json", "tokenizer.json", "chat_template.jinja", "processor_config.json"):
    src = os.path.join(SRC, extra)
    if os.path.exists(src):
        os.system(f"cp '{src}' '{OUT}/'")
print(f"done: {OUT} ({total >> 30} GiB across {n} shards)", flush=True)
