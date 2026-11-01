# REQUANT — the Focus 3bpw requant (round 131, authorized with conditions)

The task: requant the Focus checkpoint's routed experts from 4bpw to **3bpw
integer on the mcg codebook** (the engine's `admit()` reads GLM experts as
integer-bits-on-mcg only; the fractional widths are mul1 — so 3 is THE sub-4
target). Payoff: ~40 GB off the checkpoint, ~20-25 GiB device memory across
the pair. Quality bar: hardmode (the Focus cell's 92-93).

**The conditions (user, round 131)**: the job runs OFFLINE (no LLM serving
alongside a quant job), the work SPLITS across the head + the worker with transfers
over 192.168.2.x, and the job pauses every ~3 h — production relaunches,
progress is checked and doc-updated, then it continues. Authorization is
conditional on the parallelization being real.

## The design (grounded in the converter's own contracts)

The standard conversion is a SEQUENTIAL pipeline (each module's Hessian is
captured from the already-quantized upstream activations) — a naive module
split breaks it. The parallel enabler is the converter's first-class
precomputed-Hessian mode (`--hessians <dir>`): *tensors with hin skip the
calibration forward passes* — in that mode each module quantizes
independently from its own weights + Hessian file. The pipeline:

1. **The capture (the head, sequential, once)**: one calibration pass through
   the SOURCE model, exporting per-key Hessian files to `hessians/`.
2. **The encodes (split)**: the head and the worker each encode a disjoint half
   of the expert modules — via the converter's `-rcp` recipe YAMLs (per-key
   wildcard patterns at 3bpw, even/odd split) — each writing its own
   `qtensors/<key>.safetensors` tree.
3. **The merge (the head)**: the union of the two `qtensors/` trees + the
   passthrough tensors → the assembled 3bpw checkpoint → synced to both
   hosts (192.168.2.x) for the battery.
4. **The battery**: a reqqual window — the §2 gates, hardmode, the bench,
   the admission read (the serve line's model GiB).

Efficiency: capture 1× + encodes ½+½ ≈ 0.55-0.6× wall ≈ ~1.8× speedup; with
the ~7 h AZ precedent for this expert set, the job ≈ 4 h.

## The internals map (read this sitting — the driver's facts)

- The conversion core: `exllamav3/conversion/convert_model.py` (1702 lines).
  The main loop iterates `model.modules`; per module: load → capture (the
  state forward, `capture_module_parallel`) → quantize (`quantize_linears_*`)
  → advance the state (`advance_state_parallel`) — the state chain is why the
  naive split fails.
- `get_H_data(args, linear, capture_H, state)`: with `state` truthy the H
  comes from the online capture (`capture_H[linear.qmap]` — an accumulator
  DICT: H buffer + count + inf_nan + num_total); with `state=None` it builds
  from the `--hessians` file — `hin` (the packed upper triangle, row-major —
  `unpack_sym` is its reader) → `hd = linear.init_H_data(False)` +
  `hd.update({"H": hin, ...})`. The hin file is the stateless-encode
  interface.
- The CLI: `-i/-w/-o`, `-b/--bits` (FLOAT bpw), `-rcp/--recipe` (per-key
  YAML, wildcard patterns, later entries override), `--hessians` (the dir of
  `<key>.safetensors` with hin/hout/hout_diag), `--hessians_one_sided`,
  `-eb/-hb/-mb/-vb`, `-hq`. With hin files present the main sets
  `use_reference_state = False` — "no calibration state, no forward passes".
- The output: per-module `qtensors/<module.key>.safetensors`
  (`save_tensor(q_tensors, f"qtensors/{module.key}.safetensors", args)`) —
  the mergeable unit; two hosts writing disjoint keys merge by union.
- The H accumulator → hin extraction: THE REMAINING READ. The accumulator
  dict's H buffer + count must become the packed matrix the hin file carries
  (the count normalization convention is the thing to get right). Sources:
  capture_module_parallel's worker (the buffer's creation) and
  convert_exl3_group's consumption (what the encoder reads). ANSWERED: the
  accumulator is a plain sum (`H += hd["H"]`); the precomputed path sets
  `count: 1` — the hin matrix is used as-is, no normalization.
- The hin files' size: CORRECTED. The experts are per-expert Linears
  (`{expert_idx}` in the key), 288 per MoE layer × 42 MoE layers × 3 matrices,
  moe_intermediate 2048: gate/up ≈ 33.6 MB packed each, down ≈ 8.4 MB →
  ~920 GB of hin total (plus attention/dense, small). It fits both hosts'
  disks and syncs over the CX7 pair in minutes — but it is the arc's dominant
  artifact, and `capture_h.py` filters to `qbits_key == "bits"` Linears so
  head/MTP linears (their own bit args) never enter the recipes.
- The recipe rule (allocation.py `create_q_strategy_from_recipe`): EVERY
  budgeted tensor must be covered and every entry must match at least one
  budgeted tensor — a partial recipe is refused, not a partial quantization.
  So each host's recipe is generated from the captured key list: the
  catch-all `"*": 16` (unquantized store — dequant + I/O only, no trellis
  cost) plus its half of the keys at 3; the reference recipe names every key
  at 3. The merge takes each 3-bit file from its owner.
- The kernel wall, found and fixed: the all-experts capture presents
  topk = 288 (n_routed_experts) to `exl3_moe_gather`, whose
  `MOE_GATHER_MAX_TOPK` is 32 — `exl3_moe.cu:689` refused (the receipt:
  campaign-20261101-172539, layer 3's MoE gather). Site-local venv patch:
  the define raised to 512 (shared mem 6 KB/block, trivial; the original
  kept as `exl3_moe.cu.orig-site`). The worker needs NO patch: with hin
  files the encode runs uncalibrated (no forward passes at all).

## The proof gate (before any long job — the trap-avoidance)

1. The capture driver (scripts/requant/capture_h.py) exports hin for EVERY
   budgeted linear in one pass on the head — the full job reuses the dir; the
   proof costs no extra capture.
2. Recipes generated from the captured key list; the head encodes its half,
   the worker the other (one `--hessians` encode each, recipe-driven).
3. A single-host reference encodes EVERY key at 3 from the same hin.
4. The checks: (a) every owner key present in the reference at an identical
   size (batched listings); (b) a deterministic sample (~1 in 500 + ends)
   byte-compared; (c) one catch-all 16-bit file cross-checked between hosts;
   (d) the precomputed-H sqnr read from the runs' own prints — recorded as
   the baseline receipt; the online-H comparison is NOT in the gate (it would
   cost a full calibration capture; the quality bar stays the final bench
   battery). A check failure STOPS the arc — almost nothing spent.

## The segment ledger

| segment | what | status |
|---|---|---|
| 0 | the design + the internals map (this doc) | done |
| 1 | the capture driver + the proof | in flight (the failure ladder: cd, prepare/-vb, out_dir, stc close, gpu load, the top-k patch; relaunched) |
| 2+ | the capture pass, the split encodes (3 h segments, production up at each pause) | pending |
| final | the merge + the assembly + the cross-sync | pending |
| battery | the reqqual window (the gates + hardmode + the admission read) | pending |
