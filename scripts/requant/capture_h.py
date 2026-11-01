#!/usr/bin/env python3
"""The requant capture driver (the head): one calibration pass through the
SOURCE model, exporting per-key Hessian files (the hin format) for the
precomputed-H split encode. Mirrors exllamav3's conversion loop (load →
capture → advance) minus the quantize, using the converter's own machinery
(capture_module_parallel / advance_state_parallel) so the conventions match.

The hin file: {"hin": <packed row-major upper triangle of the H accumulator's
H buffer>} — the exact contract unpack_sym reads (the precomputed path sets
count=1, so the matrix is used as-is; no normalization ambiguity).
"""
import argparse, os, sys

VP = "/home/giles/.venv-exl3/lib/python3.12/site-packages"
sys.path.insert(0, VP)

import torch
from safetensors.torch import save_file

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--work", required=True)
ap.add_argument("--hessians", required=True, help="the output dir for <key>.safetensors hin files")
ap.add_argument("--bits", type=float, default=3.0)
A = ap.parse_args()

os.makedirs(A.hessians, exist_ok=True)
os.makedirs(A.work, exist_ok=True)

from exllamav3.conversion import convert_model as cm
args = cm.parser.parse_args(["-i", A.src, "-w", A.work, "-b", str(A.bits), "-cb", "mcg", "-eb", "16", "-hb", "16"])

with torch.inference_mode():
    torch.set_grad_enabled(False)
    devices = [0]
    config, model, mtp_model, vision_model, tokenizer, use_ref = cm.get_base_model(vars(args))
    assert use_ref, "the capture needs the reference-state architecture"

    state = cm.get_default_calibration(vars(args), tokenizer)
    original_input_ids = state.copy()
    quant_preserves = [{} for _ in range(len(state))]
    bad_rows = set()

    def get_preserve(si, params_):
        params_.update(quant_preserves[si])
        params_["quant_preserve"] = quant_preserves[si]

    def put_preserve(si, params_):
        quant_preserves[si] = params_["quant_preserve"]

    saved = []
    for idx, module in enumerate(model.modules):
        print(f" -- [{idx + 1}/{len(model.modules)}] {module.key}", flush=True)
        slicing = module.num_slices > 1
        for current_slice in range(module.num_slices):
            defer = module.can_defer_load()
            if defer:
                module.config.stc.begin_deferred_load()
            try:
                module.load(torch.device("cpu"), load_slice=current_slice if slicing else None, keep_source_weights=True)
            finally:
                if defer:
                    module.config.stc.close()

            if len(module.get_qmaps()) > 0:
                capture_H, capture_sens, ref_states = cm.capture_module_parallel(
                    model, [module], devices, None, state, original_input_ids,
                    get_preserve, put_preserve, slicing, current_slice,
                    f" -- Capturing: {module.key}", bad_rows)
                for m in module:
                    if not hasattr(m, "qmap") or m.qmap not in capture_H:
                        continue
                    if m.key in saved:
                        continue
                    H = capture_H[m.qmap]["H"].float().cpu()
                    n = m.in_features
                    mask = torch.ones((n, n), dtype=torch.bool).triu_()
                    save_file({"hin": H[mask]}, os.path.join(A.hessians, m.key + ".safetensors"))
                    saved.append(m.key)
                    print(f"    + hin: {m.key} ({n}x{n})", flush=True)
                del capture_H, capture_sens

            cm.advance_state_parallel(
                model, [module], devices, None, state, original_input_ids,
                get_preserve, put_preserve, {}, True, idx == len(model.modules) - 1,
                f" -- Advancing: {module.key}", bad_rows)
            try:
                module.unload()
            except Exception:
                pass

    print(f" -- saved {len(saved)} hin files to {A.hessians}")
    for k in saved:
        print(f"    KEY {k}")
