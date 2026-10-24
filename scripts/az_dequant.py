#!/usr/bin/env python3
"""Stream-dequantize the AZ checkpoint (auto-round int4, GPTQ layout) to bf16.

Expert tensors arrive as (qweight [in/8, out] I32, qzeros [groups, out/8] I32,
scales [groups, out] F16) with group 128; everything else passes through.
Output: a plain bf16 HF dir (config minus quantization_config + a real
safetensors index) that exllamav3's converter accepts. RAM stays per-shard.

Runs on torch (numpy's safetensors loader cannot read BF16; torch can).
The GPTQ zero-point convention (+1 offset or not) auto-detects on the first
expert tensor: the wrong convention shifts every weight by exactly one scale
step — mean |w| decides.
"""
import glob, json, os, struct, sys

import torch
from safetensors.torch import load_file, save_file

SRC = sorted(glob.glob("/home/giles/.cache/huggingface/hub/models--azampatti--GLM-5.3-Flash-AZ/snapshots/*/"))[0]
OUT = sys.argv[1] if len(sys.argv) > 1 else "/home/giles/az-az-bf16"
GROUP = 128
FLUSH_BYTES = 4 << 30

os.makedirs(OUT, exist_ok=True)

def unpack_i4(t, axis):  # expand the packed axis by 8 (qweight packs the IN axis = 0; qzeros packs OUT = -1)
    sh = list(t.shape)
    sh[axis] = sh[axis] * 8
    out = torch.empty(sh, dtype=torch.int32, device=t.device)
    for i in range(8):
        sl = [slice(None)] * len(sh)
        sl[axis] = slice(i, None, 8)
        out[tuple(sl)] = (t >> (4 * i)) & 0xF
    return out

CONVENTION = None
weight_map = {}
total = 0
shard_no = 0
acc, acc_bytes = {}, 0

def flush():
    global shard_no, acc, acc_bytes, total
    shard_no += 1
    name = f"model-{shard_no:05d}-of-NNNNN.safetensors"
    save_file(acc, os.path.join(OUT, name))
    for k, v in acc.items():
        weight_map[k] = (name, list(v.shape), str(v.dtype).upper())
    total += acc_bytes
    print(f"  {name}: {len(acc)} tensors, {acc_bytes >> 30} GiB (running {total >> 30} GiB)", flush=True)
    acc, acc_bytes = {}, 0

def dequant(qw, qz, sc):
    q = unpack_i4(qw, 0).float()                       # [in, out]
    z = unpack_i4(qz, -1).float()                      # [groups, out]
    g = q.shape[0] // GROUP
    q = q.reshape(g, GROUP, -1)
    z = z[:, None, :]
    s = sc.float()[:, None, :]
    w = (q - z) * s                                    # [groups, GROUP, out]
    if CONVENTION == "plus1":
        w = w - s
    return w.reshape(-1, w.shape[-1]).t().contiguous().to(torch.bfloat16)  # [out, in]

files = sorted(glob.glob(SRC + "model-*.safetensors"))
print(f"source: {SRC} ({len(files)} shards)", flush=True)
for f in files:
    tensors = load_file(f)
    for k in list(tensors.keys()):
        v = tensors[k]
        if k.endswith(".qweight"):
            base = k[: -len(".qweight")]
            qw, qz, sc = v, tensors[base + ".qzeros"], tensors[base + ".scales"]
            if CONVENTION is None:
                # decisive check on this tensor: mean |w| under each convention —
                # the wrong one leaves every weight offset by one scale step
                q = unpack_i4(qw, 0).float().reshape(-1, GROUP, qw.shape[1])
                z = unpack_i4(qz, -1).float()[:, None, :]
                s = sc.float()[:, None, :]
                mp = float(((q - z) * s).abs().mean())
                m1 = float((((q - z) * s) - s).abs().mean())
                CONVENTION = "plain" if mp <= m1 else "plus1"
                print(f"convention: {CONVENTION} (mean|w| plain {mp:.4g} vs plus1 {m1:.4g})", flush=True)
            w = dequant(qw, qz, sc)
            acc[base + ".weight"] = w
            acc_bytes += w.numel() * 2
            del tensors[base + ".qzeros"], tensors[base + ".scales"]
        elif k.endswith(".qzeros") or k.endswith(".scales"):
            continue
        else:
            acc[k] = v
            acc_bytes += v.numel() * v.element_size()
        del tensors[k]
    if acc_bytes >= FLUSH_BYTES:
        flush()
    del tensors
flush()

n = shard_no
for i in range(1, n + 1):
    os.rename(os.path.join(OUT, f"model-{i:05d}-of-NNNNN.safetensors"),
              os.path.join(OUT, f"model-{i:05d}-of-{n:05d}.safetensors"))
meta = {}
tsize = 0
for k, (name, shape, dtype) in weight_map.items():
    real = name.replace("-of-NNNNN", f"-of-{n:05d}")
    meta[k] = {"dtype": dtype, "shape": shape, "data_offsets": [0, 0]}
    weight_map[k] = real
    tsize += int(torch.tensor(shape).prod().item()) * 2
json.dump({"metadata": {"total_size": tsize}, "weight_map": weight_map},
          open(os.path.join(OUT, "model.safetensors.index.json"), "w"))
cfg = json.load(open(SRC + "config.json"))
cfg.pop("quantization_config", None)
cfg.setdefault("text_config", {})["dtype"] = "bfloat16"
json.dump(cfg, open(os.path.join(OUT, "config.json"), "w"), indent=2)
for extra in ("generation_config.json", "tokenizer_config.json", "tokenizer.json", "chat_template.jinja", "processor_config.json"):
    src = os.path.join(SRC, extra)
    if os.path.exists(src):
        os.system(f"cp '{src}' '{OUT}/'")
print(f"done: {OUT} ({total >> 30} GiB across {n} shards)", flush=True)
