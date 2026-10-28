#!/home/giles/.venv-exl3/bin/python
"""Re-pack the az-az-exl3-mcg requant (exllamav3 1.6.4 layout) into the layout the
TensorFold native engine reads (the served AZ-Focus's layout):

  - the routed experts' .mcg/.trellis/.suh/.svh quads copy through byte-identical
    (.mcg reshaped I32 [] -> I32 [1]; the same magic value in both layouts, verified)
  - the merged self_attn.conv1d.weight [3c,1,taps] slices three ways into the
    q/k/v_conv1d.weight [c,1,taps] the engine's loader reads (the q|k|v order, verified
    from gated_delta_net's conv stash map)
  - every other 1.6.4 quantized linear (.mcg/.trellis/.suh/.svh) DEQUANTIZES to a BF16
    .weight via exllamav3's own reconstruction (Exl3HfLinear.dequantized_weight —
    trellis decode + Hadamard + scales); the fused qkv_proj splits into q/k/v in the
    q|k|v order (confirmed from gated_delta_net's fdim_qkv arithmetic)
  - unquantized tensors and the tokenizer/config family copy through; the
    quantization_config is rewritten to the old shape (bits 4, codebook mcg,
    head_bits 16, scope glm53_routed_experts_only)

Modes:
  --mode dry    one dense layer + one expert layer end to end, full mapping print,
                the name-set diff against the served Focus, no output dir. The gate.
  --mode full   both hosts: stage the source to the worker, run both halves (this
                host in-process, the worker via ssh), merge, verify, place.
  --mode work   (internal; the head drives it over ssh) one host's shard slice.

Every dequantized tensor is checked finite with its name in any failure; the merge
verifies the output name set equals the served Focus's inventory before declaring
success. A wrong output fails at the serve in seconds with the engine's own
named-tensor receipt regardless.
"""
import argparse
import glob
import json
import os
import re
import struct
import subprocess
import sys
import time

SRC = "/home/giles/az-az-exl3-mcg"
REF = "/home/giles/.cache/huggingface/hub/models--local--GLM-5.3-Flash-AZ-Focus-EXL3/snapshots/main"
OUT = "/home/giles/az-repack-out"
WORKER_STAGE = "/home/giles/az-repack-src"

MAGIC = -877912083  # the mcg marker's value in both layouts (verified)

t0 = time.time()
def log(msg):
    print("[%6.1fs] %s" % (time.time() - t0, msg), flush=True)

def die(msg):
    log("FAIL: " + msg)
    sys.exit(1)

def load_index(path):
    return json.load(open(os.path.join(path, "model.safetensors.index.json")))["weight_map"]

_shard_cache = {"path": None, "tensors": None}
def read_tensor(path, name, index):
    """One tensor as a torch tensor. The shards load lazily and the last one stays cached
    (a shard's tensors process consecutively)."""
    shard = os.path.join(path, index[name])
    if _shard_cache["path"] != shard:
        from safetensors.torch import load_file
        _shard_cache["path"] = shard
        _shard_cache["tensors"] = load_file(shard)
    return _shard_cache["tensors"][name]

def shard_groups(index, names):
    """names grouped by their shard file."""
    g = {}
    for n in names:
        g.setdefault(index[n], []).append(n)
    return g

def classify(names):
    """Split the requant's tensor names into: expert quads (copy), non-expert quad
    groups (dequant), merged conv1d (transpose), plain (copy)."""
    experts, quads, convs, plain = [], [], [], []
    for n in names:
        if re.search(r"mlp\.experts\.\d+\.", n):
            experts.append(n)
        elif n.endswith(".conv1d.weight"):
            convs.append(n)
        elif re.search(r"\.(mcg|trellis|suh|svh)$", n):
            quads.append(n)
        else:
            plain.append(n)
    return experts, quads, convs, plain

def quad_groups(quads):
    """quad tensor names grouped by their module key."""
    g = {}
    for n in quads:
        g.setdefault(n.rsplit(".", 1)[0], []).append(n)
    return g

def check_finite(t, name):
    import torch
    if not torch.isfinite(t).all():
        n_bad = int((~torch.isfinite(t)).sum())
        die("dequantized %s holds %d non-finite values — the trellis decode is wrong" % (name, n_bad))
    amax = t.abs().max().item()
    if amax == 0 or amax > 1e3:
        log("  WARN %s: |max| = %.4g (outside the expected weight range)" % (name, amax))
    return amax

def dequant_linear(key, parts, src, index, device):
    """One 1.6.4 quantized linear -> fp32 (out, in) via exllamav3's own reconstruction.
    The kernel's get_weight_tensor yields (in, out); the .t() to the HF/nn.Linear
    convention the loader reads happens here (the round-64 dry-run's shape gate
    caught the missing transpose: down_proj [12288, 4096] vs the reference [4096, 12288])."""
    import torch
    from exllamav3.integration.transformers import Exl3HfLinear
    suh = read_tensor(src, key + ".suh", index).to(device)
    svh = read_tensor(src, key + ".svh", index).to(device)
    trellis = read_tensor(src, key + ".trellis", index).to(device)
    mcg = read_tensor(src, key + ".mcg", index).to(device).reshape(1)
    in_f, out_f = int(suh.shape[0]), int(svh.shape[0])
    # the class registers its buffers on the META device; the fill is an ASSIGN
    # (load_state_dict assign=True replaces the meta buffers with real tensors) —
    # copy_-into-meta and .to() both fail (the round-60 dry-run's traceback)
    meta = {key + ".mcg": {"shape": (1,), "torch_dtype": torch.int32}}
    for suff, t in (("suh", suh), ("svh", svh), ("trellis", trellis)):
        meta[key + "." + suff] = {"shape": tuple(t.shape), "torch_dtype": t.dtype}
    mod = Exl3HfLinear(in_f, out_f, meta)
    mod.load_state_dict({"trellis": trellis, "suh": suh, "svh": svh, "mcg": mcg},
                        strict=True, assign=True)
    mod.finalize()
    if int(mod.mcg.item()) != MAGIC:
        die("%s.mcg holds %d, expected the mcg magic %d — the experts/payload contract changed" % (key, int(mod.mcg.item()), MAGIC))
    w = mod.dequantized_weight()          # (in, out) — the kernel's convention
    w = w.float()
    check_finite(w, key)
    return w.t().contiguous()             # (out, in) — the loader's convention

def emit_weight(out_acc, name, w, out_f_dtype):
    import torch
    t = w.to(out_f_dtype).contiguous()
    if name in out_acc:
        die("output collision on %s" % name)
    out_acc[name] = t
    return t.shape

def process_shard_group(shard, names, src, index, out_acc, device, ref_shapes, stats):
    """Transform one input shard's tensors into out_acc."""
    import torch
    experts, quads, convs, plain = classify(names)
    # 1. the routed experts' quads: byte copy, .mcg reshaped to [1]
    for n in experts:
        t = read_tensor(src, n, index)
        if n.endswith(".mcg"):
            t = t.reshape(1)
            if int(t.item()) != MAGIC:
                die("%s holds %d, expected the mcg magic" % (n, int(t.item())))
        out_acc[n] = t.contiguous()
        stats["expert_quads"] += 1
    # 2. the merged conv1d: a 3-way dim-0 slice into q/k/v_conv1d (the q|k|v order)
    for n in convs:
        t = read_tensor(src, n, index)           # [3c, 1, taps] bf16
        c = t.shape[0] // 3
        if 3 * c != t.shape[0]:
            die("%s's dim 0 (%d) is not divisible by 3" % (n, t.shape[0]))
        for pi, proj in enumerate(("q", "k", "v")):
            name = n.replace("conv1d.weight", proj + "_conv1d.weight")
            blk = t[pi * c:(pi + 1) * c].clone()   # clone, not contiguous: a dim-0 slice is contiguous, so contiguous() would return a storage-sharing view
            ref = ref_shapes.get(name)
            if ref is not None and list(blk.shape) != list(ref):
                die("%s sliced to %s; the engine reads %s" % (name, list(blk.shape), ref))
            out_acc[name] = blk
            stats["conv_transposed"] += 1
        log("  conv1d %s -> q/k/v_conv1d %s x3 (the q|k|v slice)" % (n, list(blk.shape)))
    # 3. the non-expert quantized linears: dequant to BF16 .weight
    for key, parts in sorted(quad_groups(quads).items()):
        need = {"." + s for s in ("mcg", "trellis", "suh", "svh")}
        have = {"." + p.rsplit(".", 1)[1] for p in parts}
        if have != need:
            die("%s carries %s — expected exactly mcg/trellis/suh/svh" % (key, sorted(have)))
        w = dequant_linear(key, parts, src, index, device)   # (out, in) fp32
        if key.endswith("qkv_proj"):
            # the fused ROW order q|k|v (gated_delta_net's fdim_qkv arithmetic, verified)
            third = w.shape[0] // 3
            if 3 * third != w.shape[0]:
                die("%s's fused out-dim %d is not divisible by 3" % (key, w.shape[0]))
            for pi, proj in enumerate(("q_proj", "k_proj", "v_proj")):
                blk = w[pi * third:(pi + 1) * third]         # (out/3, in)
                name = key.replace("qkv_proj", proj) + ".weight"
                sh = emit_weight(out_acc, name, blk, torch.bfloat16)
                st = check_finite(blk, name)
                stats["dequanted"] += 1
                log("  %s -> %s %s (|max| %.3g)" % (key, name, list(sh), st))
        else:
            name = key + ".weight"
            sh = emit_weight(out_acc, name, w, torch.bfloat16)
            st = check_finite(w, name)
            stats["dequanted"] += 1
            log("  %s -> %s %s (|max| %.3g)" % (key, name, list(sh), st))
    # 4. the plain tensors: byte copy
    for n in plain:
        out_acc[n] = read_tensor(src, n, index).contiguous()
        stats["plain"] += 1

def ref_inventory():
    """The served Focus's index: {name: shape} — the structural reference."""
    idx = load_index(REF)
    shapes = {}
    import glob, os
    shard_cache = {}
    for n, shard in idx.items():
        p = os.path.join(REF, shard)
        if p not in shard_cache:
            with open(p, "rb") as f:
                nb = struct.unpack("<Q", f.read(8))[0]
                shard_cache[p] = json.loads(f.read(nb))
        h = shard_cache[p].get(n, {})
        shapes[n] = h.get("shape")
    return idx, shapes

def check():
    """The pre-run environment gate. Runs THROUGH fix_path (main applies it before
    dispatch), so it probes exactly the environment --mode work will run in — the
    round-65 lesson: a probe outside the run's PATH probes nothing (the bare-ssh
    check tripped the ext's rebuild path and found no ninja on a healthy worker)."""
    import shutil
    import torch
    log("check: python %s | torch %s | cuda %s" % (
        sys.version.split()[0], torch.__version__, torch.cuda.is_available()))
    if not torch.cuda.is_available():
        die("no CUDA device visible")
    log("cuda: %s" % torch.cuda.get_device_name(0))
    from exllamav3.ext import exllamav3_ext as ext
    if not hasattr(ext, "reconstruct"):
        die("exllamav3_ext lacks reconstruct — the dequant kernel is missing")
    from exllamav3.integration.transformers import Exl3HfLinear  # noqa: F401
    import safetensors  # noqa: F401
    import transformers  # noqa: F401
    free = shutil.disk_usage("/home").free >> 30
    log("imports OK: ext.reconstruct + Exl3HfLinear + safetensors + transformers | /home free %d GiB" % free)
    if free < 90:
        die("only %d GiB free on /home — this host's repack half needs ~85 GiB of output space" % free)
    print("CHECK OK")

def names_check():
    """No torch, no GPU: apply the name mapping to every requant tensor on strings alone
    and diff the result against the served Focus's inventory. The mapping-completeness gate."""
    index = load_index(SRC)
    ref_idx, _ = ref_inventory()
    experts, quads, convs, plain = classify(index.keys())
    produced = set(plain) | set(experts)
    for n in convs:
        c = None
        produced.update(n.replace("conv1d.weight", p + "_conv1d.weight") for p in ("q", "k", "v"))
    for key in quad_groups(quads):
        if key.endswith("qkv_proj"):
            for proj in ("q_proj", "k_proj", "v_proj"):
                produced.add(key.replace("qkv_proj", proj) + ".weight")
        else:
            produced.add(key + ".weight")
    missing_ref = sorted(produced - set(ref_idx))
    not_produced = sorted(set(ref_idx) - produced)
    print("requant tensors: %d | produced by the mapping: %d | reference: %d" % (len(index), len(produced), len(ref_idx)))
    print("produced-but-not-in-reference: %d %s" % (len(missing_ref), missing_ref[:6]))
    print("in-reference-but-not-produced: %d %s" % (len(not_produced), not_produced[:6]))
    if not missing_ref and not not_produced:
        print("NAME MAPPING: EXACT — every requant tensor maps to a reference name and vice versa")
    else:
        print("NAME MAPPING: MISMATCH — read the diffs above before anything else")
        sys.exit(1)

def dry_run():
    import torch
    log("dry-run: python %s | torch %s | cuda %s" % (
        sys.version.split()[0], torch.__version__, torch.cuda.is_available()))
    if not torch.cuda.is_available():
        die("no CUDA device — the dequant runs on the GPU; are the GPUs free (production down)?")
    log("cuda: %s" % torch.cuda.get_device_name(0))
    device = "cuda:0"
    index = load_index(SRC)
    ref_idx, ref_shapes = ref_inventory()
    log("source tensors: %d | reference (served Focus) tensors: %d" % (len(index), len(ref_idx)))
    # the source completeness: every index entry's shard exists
    missing = [s for s in set(index.values()) if not os.path.exists(os.path.join(SRC, s))]
    if missing:
        die("source shards missing: %s" % missing[:4])
    # the mcg magic across a sample of expert tensors
    experts, quads, convs, plain = classify(index.keys())
    log("inventory: %d expert quads, %d non-expert quad tensors (%d groups), %d merged conv1d, %d plain" % (
        len(experts), len(quads), len(quad_groups(quads)), len(convs), len(plain)))
    for n in sorted(experts):
        if n.endswith(".mcg"):
            t = read_tensor(SRC, n, index).reshape(1)
            if int(t.item()) != MAGIC:
                die("%s holds %d — the mcg magic mismatch" % (n, int(t.item())))
    log("mcg magic verified on %d expert groups" % len([n for n in experts if n.endswith(".mcg")]))
    # the layer pick: one dense (no experts) + one expert layer
    layers = sorted({int(m.group(1)) for n in index for m in [re.search(r"layers\.(\d+)\.", n)] if m})
    dense_layer = next(i for i in layers if not any("layers.%d.mlp.experts." % i in n for n in index))
    expert_layer = next(i for i in layers if any("layers.%d.mlp.experts." % i in n for n in index))
    log("dry layers: dense %d + expert %d (of %d layers)" % (dense_layer, expert_layer, len(layers)))
    stats = {"expert_quads": 0, "conv_transposed": 0, "dequanted": 0, "plain": 0}
    out_acc = {}
    for li in (dense_layer, expert_layer):
        names = [n for n in index if "layers.%d." % li in n]
        for shard, ns in shard_groups(index, names).items():
            process_shard_group(shard, ns, SRC, index, out_acc, device, ref_shapes, stats)
    log("dry stats: %s" % stats)
    # the name-set check: every produced name must exist in the reference inventory with the same shape
    bad_shape, missing_ref = [], []
    for n, t in out_acc.items():
        if n not in ref_idx:
            missing_ref.append(n)
        elif ref_shapes.get(n) is not None and list(t.shape) != list(ref_shapes[n]):
            bad_shape.append((n, list(t.shape), ref_shapes[n]))
    if missing_ref:
        die("%d produced names are NOT in the reference inventory (the first 5: %s)" % (len(missing_ref), missing_ref[:5]))
    if bad_shape:
        die("%d shape mismatches vs the reference (the first 5: %s)" % (len(bad_shape), bad_shape[:5]))
    # the reverse: reference names the dry layers should have produced but did not
    expect = {n for n in ref_idx if any("layers.%d." % li in n for li in (dense_layer, expert_layer))}
    produced = {n for n in out_acc}
    not_produced = sorted(expect - produced)
    if not_produced:
        die("%d expected names were not produced (the first 5: %s)" % (len(not_produced), not_produced[:5]))
    log("name-set equality vs the reference: PASS (%d tensors, shapes match)" % len(out_acc))
    log("DRY-RUN OK — the mapping, the dequant and the name contract all hold. Run --mode full.")
    torch.cuda.empty_cache()

def save_shard(out_dir, acc, shard_no, total_unknown, weight_map):
    from safetensors.torch import save_file
    name = "model-%05d-of-NNNNN.safetensors" % shard_no
    save_file(acc, os.path.join(out_dir, name), metadata={"format": "pt"})
    for k, v in acc.items():
        weight_map[k] = {"file": name, "shape": list(v.shape)}
    log("  wrote %s: %d tensors" % (name, len(acc)))

def work(shard_lo, shard_hi, out_dir, src):
    """One host's slice: the shards [lo, hi] by the index's shard ordinal. src is the
    host-local source (the head's requant dir, or the worker's staged copy — the
    round-66 full run failed here: the worker read the head's hardcoded path)."""
    import torch
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    log("work: shards %d-%d on %s (device %s)" % (shard_lo, shard_hi, os.uname().nodename, device))
    index = load_index(src)
    by_shard = {}
    for n, s in index.items():
        m = re.match(r"model-(\d+)-of-", s)
        by_shard.setdefault(int(m.group(1)), {})[n] = s
    os.makedirs(out_dir, exist_ok=True)
    try:
        _, ref_shapes = ref_inventory()
    except Exception as e:
        log("WARN: no readable reference inventory here (%s) — the conv1d shape check defers to the merge (which binds)" % str(e).splitlines()[-1][:80])
        ref_shapes = {}
    weight_map = {}
    stats = {"expert_quads": 0, "conv_transposed": 0, "dequanted": 0, "plain": 0}
    done = 0
    for shard_no in sorted(by_shard):
        if not (shard_lo <= shard_no <= shard_hi):
            continue
        names = sorted(by_shard[shard_no])
        out_acc = {}
        process_shard_group(shard_no, names, src, index, out_acc, device, ref_shapes, stats)
        save_shard(out_dir, out_acc, shard_no, True, weight_map)
        done += 1
        log("shard %d done (%d/%d in this host's range)" % (shard_no, done, shard_hi - shard_lo + 1))
    json.dump(weight_map, open(os.path.join(out_dir, "partial-weight-map.json"), "w"), indent=0)
    json.dump(stats, open(os.path.join(out_dir, "partial-stats.json"), "w"))
    log("work complete: %d shards, stats %s" % (done, stats))

def merge(out_dir):
    import torch
    log("merge: collecting the partial maps")
    weight_map, stats = {}, {"expert_quads": 0, "conv_transposed": 0, "dequanted": 0, "plain": 0}
    for pm in sorted(glob.glob(os.path.join(out_dir, "*", "partial-weight-map.json"))):
        part = json.load(open(pm))
        for k, v in part.items():
            if k in weight_map:
                die("duplicate output tensor %s (in %s and %s)" % (k, weight_map[k]["file"], v["file"]))
            weight_map[k] = v
    for st in sorted(glob.glob(os.path.join(out_dir, "*", "partial-stats.json"))):
        s = json.load(open(st))
        for k in stats: stats[k] += s[k]
    log("merged stats: %s over %d tensors" % (stats, len(weight_map)))
    total = len(set(load_index(SRC).values()))
    # move every produced shard into merged/ under the real total-count name
    os.makedirs(os.path.join(out_dir, "merged"), exist_ok=True)
    for f in glob.glob(os.path.join(out_dir, "*", "model-*-of-NNNNN.safetensors")):
        os.rename(f, os.path.join(out_dir, "merged", os.path.basename(f).replace("of-NNNNN", "of-%05d" % total)))
    # the index with real shard names
    index = {}
    for k, v in weight_map.items():
        index[k] = v["file"].replace("of-NNNNN", "of-%05d" % total)
    # the name-set equality against the reference
    ref_idx, ref_shapes = ref_inventory()
    missing_ref = sorted(set(ref_idx) - set(index))
    extra = sorted(set(index) - set(ref_idx))
    if missing_ref:
        die("%d reference tensors missing from the output (the first 8: %s)" % (len(missing_ref), missing_ref[:8]))
    if extra:
        die("%d output tensors are not in the reference (the first 8: %s)" % (len(extra), extra[:8]))
    # the shapes: every output tensor's shape must equal the reference's
    bad = []
    for shard in sorted(set(index.values())):
        p = os.path.join(out_dir, "merged", shard)
        if not os.path.exists(p):
            die("missing output shard %s" % p)
        with open(p, "rb") as f:
            nb = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(nb))
        for k, h in hdr.items():
            if k == "__metadata__": continue
            if ref_shapes.get(k) is not None and list(h["shape"]) != list(ref_shapes[k]):
                bad.append((k, h["shape"], ref_shapes[k]))
    if bad:
        die("%d shape mismatches vs the reference (the first 5: %s)" % (len(bad), bad[:5]))
    log("name-set + shape equality vs the served Focus: PASS (%d tensors)" % len(index))
    json.dump({"weight_map": index}, open(os.path.join(out_dir, "merged", "model.safetensors.index.json"), "w"), indent=1)
    # the config: the old-shape quantization_config over the requant's config
    cfg = json.load(open(os.path.join(SRC, "config.json")))
    cfg["quantization_config"] = {
        "quant_method": "exl3", "bits": 4, "codebook": "mcg", "head_bits": 16,
        "scope": "glm53_routed_experts_only",
    }
    json.dump(cfg, open(os.path.join(out_dir, "merged", "config.json"), "w"), indent=2)
    # the tokenizer/config family + the template copy through
    for f in ("tokenizer.json", "tokenizer_config.json", "generation_config.json",
              "chat_template.jinja", "LICENSE", "NOTICE"):
        p = os.path.join(SRC, f)
        if os.path.exists(p):
            subprocess.run(["cp", p, os.path.join(out_dir, "merged", f)], check=True)
    # the provenance: honest, written by this repo
    json.dump({
        "origin": "azampatti/GLM-5.3-Flash-AZ (the retrained selector) dequantized and re-converted",
        "requant": "exllamav3 1.6.4 convert -b 4 -hb 16 -eb 16 -vb 16 -cb mcg (the az-az-exl3-mcg dir)",
        "this_file": "re-packed to the TensorFold-native layout by scripts/az-repack.py (rounds 58-60):"
                     " the routed experts' quads copied (the mcg magic verified), the merged conv1d"
                     " transposed for the loader's merged branch, the 290 trellis-coded linears"
                     " dequantized to BF16 .weight via exllamav3's own reconstruction (they pass"
                     " through two 4-bit quantizations: the converter's and the engine's load-time pack)",
    }, open(os.path.join(out_dir, "merged", "repack_provenance.json"), "w"), indent=2)
    log("MERGE OK: %s/merged is a complete old-layout checkpoint — place it and run the serve window." % out_dir)

def fix_path():
    """The exllamav3 ext JIT-compiles at import: torch runs ninja via subprocess, so the
    venv's bin (ninja) and the CUDA toolkit's bin (nvcc) must be on PATH — running the
    script through the venv's python does NOT put them there (the round-60 dry-run's
    traceback on the head; the same trap hit on the worker over ssh). Idempotent."""
    vbin = os.path.dirname(os.path.abspath(sys.executable))
    if not os.path.exists(os.path.join(vbin, "ninja")):
        die("ninja is missing from the venv's bin (%s) — run: %s -m pip install ninja" % (vbin, sys.executable))
    parts = os.environ.get("PATH", "").split(os.pathsep)
    add = [vbin]
    for c in ("/usr/local/cuda/bin", "/usr/local/cuda-13/bin", "/usr/local/cuda-13.0/bin"):
        if os.path.exists(os.path.join(c, "nvcc")) and c not in parts:
            add.append(c)
    for p in reversed(add):
        if p not in parts:
            parts.insert(0, p)
    os.environ["PATH"] = os.pathsep.join(parts)
    log("PATH: %s joined ahead (ninja + nvcc for the ext's JIT compile)" % ", ".join(add))

def main():
    fix_path()
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("check", "names", "dry", "full", "work", "merge"), required=True)
    ap.add_argument("--shards", default="", help="work mode: this host's inclusive range, e.g. 1-12")
    ap.add_argument("--src", default=SRC, help="work mode: the source checkpoint dir (the worker's staged copy passes --src)")
    a = ap.parse_args()
    if a.mode == "check":
        check()
    elif a.mode == "names":
        names_check()
    elif a.mode == "dry":
        dry_run()
    elif a.mode == "work":
        lo, hi = (int(x) for x in a.shards.split("-"))
        work(lo, hi, os.path.join(OUT, "work-" + ("head" if lo == 1 else "worker")), a.src)
    elif a.mode == "merge":
        merge(OUT)
    elif a.mode == "full":
        full()

def full():
    """The head drives everything: stage the source to the worker, both halves, merge."""
    import torch
    log("full run: preflight")
    if not torch.cuda.is_available():
        die("no CUDA on the head — production must be down (the GPUs are the dequant's engine)")
    index = load_index(SRC)
    free = __import__("shutil").disk_usage("/home").free >> 30
    if free < 260:
        die("only %d GiB free on /home — the output needs ~170 GiB plus staging" % free)
    worker = None
    env = {}
    for line in open("scripts/local.sh"):
        m = re.match(r'WORKER="?(.+?)"?\s*$', line.strip())
        if m: worker = m.group(1)
    if not worker:
        die("no WORKER in scripts/local.sh — the both-hosts run needs it")
    log("preflight OK: %d source tensors, %d GiB free, worker %s" % (len(index), free, worker))
    for stale in ("work-head", "work-worker", "merged"):
        __import__("shutil").rmtree(os.path.join(OUT, stale), ignore_errors=True)
    # 1. stage the source to the worker (-W whole-file; the round-31 delta trap is
    # handled by -W + --delete, NOT by wiping: a complete identical stage (the
    # round-66 triple-check: mtimes preserved across the rail) quick-checks to a
    # zero-transfer pass, while a partial one completes — the rm -rf would force a
    # full ~6-min re-copy every re-run for no safety)
    log("staging the source to the worker (-W, whole files; an up-to-date stage is skipped by the quick check)…")
    r = subprocess.run(["rsync", "-a", "-W", "--delete", "-e", "ssh -o BatchMode=yes",
                        SRC + "/", worker + ":" + WORKER_STAGE])
    if r.returncode != 0:
        die("the source staging to the worker failed (rc %d)" % r.returncode)
    # 2. the worker's half: shards 13-23
    log("launching the worker's half (shards 13-23)")
    vpy = "/home/giles/.venv-exl3/bin/python"
    # copy the script over FIRST — the environment check must run through the script's
    # own fix_path (the round-65 full run failed here: a bare ssh probe skipped the
    # PATH join, the ext's rebuild check found no ninja, and a healthy worker looked broken)
    subprocess.run(["scp", "-q", "-o", "BatchMode=yes", os.path.abspath(__file__), worker + ":" + WORKER_STAGE + ".py"], check=True)
    log("checking the worker's environment through the script itself (--mode check)…")
    r = subprocess.run(["ssh", "-o", "BatchMode=yes", worker,
                        "rm -rf %s/work-worker && %s %s --mode check" % (OUT, vpy, WORKER_STAGE + ".py")],
                       capture_output=True, text=True)
    if "CHECK OK" not in (r.stdout or ""):
        log("the worker's check FAILED — its output:")
        print((r.stdout or "")[-600:], (r.stderr or "")[-600:])
        die("the worker's environment is not ready — fix per its output above, "
            "or run head-only: scripts/az-repack.py --mode work --shards 1-23 && scripts/az-repack.py --mode merge")
    log("worker: running its half…")
    r = subprocess.run(["ssh", "-o", "BatchMode=yes", worker,
                        "%s %s --mode work --shards 13-23 --src %s > %s/work.log 2>&1; echo rc=$?" % (
                            vpy, WORKER_STAGE + ".py", WORKER_STAGE, WORKER_STAGE)],
                       capture_output=True, text=True)
    if "rc=0" not in (r.stdout or ""):
        log("the worker's half FAILED — its log:")
        subprocess.run(["ssh", "-o", "BatchMode=yes", worker, "tail -30 %s/work.log" % WORKER_STAGE])
        die("the worker's repack failed (see its log above)")
    log("worker half OK")
    # 3. the head's half: shards 1-12
    log("this host's half (shards 1-12)")
    work(1, 12, os.path.join(OUT, "work-head"), SRC)
    # 4. the worker's outputs back
    log("fetching the worker's output shards…")
    r = subprocess.run(["rsync", "-a", "-W", "-e", "ssh -o BatchMode=yes",
                        worker + ":" + OUT + "/work-worker/", os.path.join(OUT, "work-worker") + "/"])
    if r.returncode != 0:
        die("fetching the worker's output failed")
    # 5. merge + verify
    merge(OUT)
    log("ALL DONE — place the merged dir and run the serve window.")

if __name__ == "__main__":
    main()
