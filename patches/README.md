# Patch overlay policy

This directory carries **optional local patches** applied to the TensorFold tree at image
build. `config.sh`'s `PATCHES` knob selects them (empty = pristine upstream, the default and
the tested configuration); `scripts/prepare.sh` stages the named files into `image/patches/`
and the Docker build applies them right after the TensorFold checkout.

## The rules

1. **Upstream first.** A patch exists here only while upstream lacks the feature (check
   #840's carry-over tracker and the latest release notes). The moment an upstream release
   covers it, the patch is dropped and the version bump takes over.
2. **A receipt or it doesn't go in.** Each patch carries, in this file or the patch header:
   what it changes, the measured effect (config, hardware, cell-by-cell numbers from
   `run-bench-matrix.sh` or the engine's own counters), the exact TreeFold commit it was
   ported against, and who owns rebasing it at the next bump.
3. **One concern per patch.** A patch that bundles several changes cannot be measured or
   dropped independently.
4. **Rebase cost is the bill.** The ecosystem's 77-patch stack (jayleaton's repo) targets
   TensorFold 0.3.4 and buys ~3% over its own control — their headline numbers come from
   configuration, not patches. Port a patch only when its measured win is worth the
   rebase tax on every future version bump.

## Candidates under watch (upstream PR scan, 2026-10-23)

- **#1079 — DFlash2/lanes decode (+9–15% on qwen27, "replies unchanged")** — TESTED AND
  DROPPED (2026-10-23, `campaign-20261023-185520/`): all §2 gates PASS on the patched
  build ("replies unchanged" holds on our model), but the throughput A/B is a **wash** —
  solo d0 +3.3% (48.3→49.9), solo d4096 −8.6% (49.0→44.8), every other row ±2%, pp/TTFT
  unchanged. The qwen gains live in qwen's tree-draft kernels, not in the shared-lanes
  parts our chain drafting hits. Patch file removed per the rebase-tax rule; this entry
  is the receipt.
- **#1171 — qmmf half tiles (32-column, 16-row decode calls)** — bit-for-bit, gated by
  `TF_QMMF_HALF_BELOW`, but instantiated for MXFP8/BF16 **only — FP4 is excluded**, and
  our dense4/head4 packing is FP4-mode qmmf: only the residual BF16 lane calls would
  take the fast path. The diff does not apply at our pin (one registry hunk); expected
  gain small for us. Revisit when it lands in a release.
- **#1143 — Flash Next: NVFP4 experts on prompt tiles, decode-form sums, bit for bit** —
  the expert-batching-in-prompt-passes mechanism our prefill gap wants (we are 90%
  router+routed in prompt chunks), but Flash Next-family files. The glm/cuda port is
  the upstream ask (with #1080's 4096-row chunks).
- **#1078 (TF_DRAFT_NODES) is qwen-tree-only** — its knob lives in qwen's draft-frame
  staging, not the shared drafter; no path to GLM's DFlash2. Main since 1351ed16 is
  app/infra/Flash-Next/grammar — no glm/cuda perf movement; our pin stays current.

## Candidates under watch (from the ecosystem review, 2026-10-19)

- **fp8 latent KV for GLM CUDA** (jayleaton 0220) — halves KV bytes; **upstream 1.1.2 now
  prepares this in stock behind `--kv-cache fp8`** (BF16 default; native quality and device
  qualification pending). A patch would only be a backport; qualifying stock on our pair is
  the open item (see the 512K plan).
- **16-row verify windows** (jayleaton 0380) — more tokens per round; upstream's GLM window
  policy is per-release.
- **L2 weight prefetch in decode** (jayleaton 0460).
- **RoCE all-gather wiring** (jayleaton 0230/0350) — **resolved without a patch**: stock
  already wires the gather (`openRoce` at GLM startup, `TF_ROCE` defaults on); our blocker
  was the runtime image missing libibverbs/libmlx5 (rank logs warned
  "VerbsUnavailable; NCCL carries every gather"), fixed 2026-10-22 by installing
  libibverbs1 + rdma-core in the image and setting `NCCL_IB_HCA` on the site. Their +1.5%
  prefill gain was the NCCL channel set, also adoptable via env.

None of these are carried yet. Measure stock v1.1.2 + the RoCE/NCCL set first —
if the greedy cells land near the ecosystem's numbers, none of these is worth its tax.

## glm-router-ks2.patch (staged 2026-10-24, untested — PERF_PLAN Front 2 Lever A2)

One constant: `router_ks = 8 → 2` in `families/glm/cuda_glue.zig`. The prompt-router
section eats 910–1,125 ms of every 2,048-row prompt chunk (55–69% of chunk GPU time) for
a ~4.8 GFLOP GEMM (receipts: campaign-20261023-145449, the 235008 salvage, the 000708
cells); the same kernels cost 47 µs/layer at m=2 — the slowdown is superlinear in rows.
With `router_ks = 8` each tile block runs an 8-step mma pipeline (never reaches steady
state) and every x element is re-read 9×8=72 times per layer; `ks = 2` halves the
re-reads and quadruples each block's K-depth. Expected: chunk 1641 → ~600 ms, prefill
slope 0.662 → ~0.25 (production parity).

**Bit-visible by design** (the source comment: "the sliced and the single pass differ in
bits") — the §2 gates decide this patch, not hope: greedy-drafting-equals-serial must
PASS or the patch dies like #1079. Verification plan: build with
PATCHES="glm-router-ks2.patch", serve dense4fp8, prof one 2,048-row prompt chunk (the
router section + chunk wall), run the §2 gates, then the prefill slope probes.

## glm-slot-leak.patch (staged 2026-10-24, untested — upstream 7d330e24d, #1218)

One line in `core/lanes/engine.zig`'s `release`: a stream released between rounds left
its queued draw's handle unread, and GLM's ring keeps a queued round's slot until its
handle is read — so every completed or cancelled request leaked a slot until restart.
Long-serving sessions would erode toward fewer usable streams. Carried as a staged patch
until v1.1.4 ships it upstream. Bit-invisible (pure resource management); the §2 gates
run as usual when the next build carries it.

## glm-router-ks2.patch — RETIRED 2026-10-24 (content gate; receipt campaign-20261024-115242/11-routerks2.md)

The window ran the A/B clean (the agent silent): stock cell solo 55.3/55.1, c2 73.5,
c4 101.8, slope 0.659 — the duel baseline reproduces exactly. The patched cell: the
**content gate said DIFFERS** (the ks change flipped a greedy token — bit-visible as the
source warned) **and the router section did not move** (1096 → 1090 ms of the chunk —
the 8-step-pipeline hypothesis is rejected as the pathology's cause). The patched cell's
decode looked faster (solo 64.3, c4 119.6) but that is an acceptance lottery riding the
bit change (77→higher acceptance on this workload), not a mechanism — and it changes
replies, so it is unadoptable. The patch dies here; the pathology stays open with the
ks hypothesis eliminated (next candidates in PERF_PLAN Front 2 A2: microbench both
router kernels at m=2048, or route the prompt router through cuda_mm's GEMM).
glm-slot-leak.patch remains staged (bit-invisible, unaffected by this verdict).

## glm-router-part.patch (staged 2026-10-24, probe two — Front 2 A2)

Forces the unstaged router kernel (`router_part`) for every launch, replacing the
tile/ part branch in `families/glm/cuda_glue.zig`. Same ks slicing and summation order
as `router_tile`, so the math should be bit-identical — the content gate (vs the
115242 stock reply) verifies. The cost read splits the prompt-router pathology:
part ≈ tile (~1100 ms/chunk) → both kernels share it → replace with a real GEMM;
part >> tile → the tile is better and the fix is its K-depth/w-layout.

## glm-topk4b.patch (staged 2026-10-24, probe three v2 — Front 2 A2 / the AZ Experiment 1)

Renamed from glm-topk4.patch: the config fingerprint hashes the patch LIST (names), not
contents — the amended patch needs a new name to force a rebuild (the round-7 lesson's
sibling: session pins live in local.sh, and here a content change needs a name change).

Two lines: the struct default `topk: u32 = 8` → `4` in `families/glm/cuda_glue.zig`'s
parse (see below) and the module global `pub const top_k = 8` → `4` in
`families/glm/cuda_exl3.zig:128` — the EXL3 experts specialize for the routing width;
the first forward with slots=5 hit `error.UnsupportedTopK` there (campaign-20261024-13xx,
rank 1 stopped; rank 0 followed). The actual struct-default edit lives in
`families/glm/config.zig`. The
parse's validation snapshots the struct defaults BEFORE the json (`const want = c;`) and
requires `c.topk == want.topk` — any topk≠8 checkpoint is `error.NotGlm53Flash`, whose
silent catch in `weightShare` then bills the WHOLE 163.6 GiB to the budget (campaigns
125609 + 132628's refusals; the three earlier "top4" serves actually ran topk=8 because
the config edit never persisted, then the real topk=4 config was rejected here). With
the reference at topk=4 the kernels specialize for slots=5 and the variant serves.
**MEASURED 2026-10-24 (campaign-20261024-135458)**: routed 18.45 → 11.0 ms at 2 rows;
solo 63.5/63.6 t/s = **108% of production**; c2 103%; **c4 120.4 = 78% of production**
(was 65%); slope 0.593 (−10%); acceptance rose (4.8 tok/chunk vs 3.37). Outputs change
by design (the AZ Experiment 1: unretrained top-4 routing) — the content gate does not
apply to this phase; the acceptance counters and the quality battery do. **Not a
servable default** (config.sh stays topk=8) until the quality battery runs: §2 gates +
full tool-eval + structured-output + TEB. The AZ full checkpoint (the retrained
selector) is the quality-safe route to the same win.
## glm-router-gemm.patch (staged 2026-10-29, probe three v3 — Front 2 A2, the scoped fix)

Diffed against **v1.1.9 (465e48b6)** — the pin moves for this window; it does not apply to
1.1.8 (1.1.9 touched the same files). The prompt-router pathology's replacement: prompt
chunks (m ≥ 128, aligned) route through a new staged chain kernel instead of the
hand-rolled `router_tile`/`router_part` pair. Decoded into pieces:

- `tf_glm_router_mm` (glm_glue_router.cuh): one K slice per launch in glm_mm's prompt
  shape (128-row blocks, ROWS_FIRST for L2 sharing, 3-stage cp.async, PDL, 256 threads)
  with ONE continuous mma chain per output — the hand-rolled kernels' exact arithmetic
  (same canonical m16n8k16 fragments, same ascending k16 order from zero, `slice_sum`
  untouched), so the summed logits are bit-identical by construction; the content gate
  verifies. The gate tile rides the same shape as glm_mm16_rows (74112 B dynamic shared).
- `glm_mm_tile16s` (glm_mm.cu): tile16 with a strided source — each MoE gate tiles into
  8 contiguous [ne][d/8] K-slice layouts at engine init (a re-layout, same bytes;
  ~2.25 MB/layer, one buffer owned by the mm kernels).
- Wiring: `glue.router` gains a `tiles` param and the dispatch (decode-size rows,
  unaligned x and untiled gates keep the hand-rolled kernels); the MoE layer carries
  `rtiles`; the engine init builds the tiles under **TF_GLM_ROUTER_MM=1** (off by
  default — the patch is INERT without the env, so one image serves both A/B legs).
- Why not call stock glm_mm per slice: its per-64-group accumulator (fresh `d` per
  group, `acc += d`) rounds differently than the router kernels' continuous chain —
  fp32 non-associativity would fail the content gate (#1079 protocol).

Compile receipt: fatbins + native build RC=0 in the builder image (nvcc sm_121,
zig 0.17.0) at v1.1.9. Measured effect: PENDING the routergemm window (leg stock vs
leg gemm: content gate + chunk prof + hardmode). Baseline: the 1.1.8 cell's chunk
router 1096 ms of 1967 gpu (campaign-20261024-115242). Owner of rebasing at the next
bump: this line's agent.

## glm-hc-fuse.patch — the fused hcPre launch (Front 2's fusion lever, round 115)

- What: ONE kernel per hcPre replaces the two-kernel chain (tf_glm_hc_partial or
  hc_partial8, then tf_glm_hc_finish) — a block per row, 256 threads, the partials in
  shared memory in the finish's `part` row layout, `glm_hc::finish_row` inline. The
  finish body is UNMODIFIED (its `part` row offset moved to the callers); the fused
  kernel reproduces the chain's exact op order (partial8's per-step ops over all 16
  K-blocks, the dots reduced in ascending step order, the ss warp butterfly + the
  (0+2)+(1+3) combine) — bit-identical by construction; the content gate verifies.
- Gating: **TF_GLM_HC_FUSE=1** (off by default — the patch is INERT without the env,
  so one image serves both A/B legs). The finish/f32w twins gain `hc_pre_fused` /
  `hc_pre_fused_f32w` (the Set's comptime name binding picks them up).
- Design + census: patches/hc-fuse-design.md; results/campaign-20261030-134836/31-
  ncurouter.md (hc_partial p50 104 µs + hc_finish 54 µs, 4 launches/layer — the glue
  launch-count dominator; the fusion removes 2 launches/layer and the global `part`
  round-trip).
- Compile receipt: fatbins + native build RC=0 in the builder image (nvcc sm_121,
  zig 0.17.0) at v1.1.9; `git apply --check` clean on the pristine tag.
- Measured effect: PENDING the hcfuse window (leg stock vs leg fuse: content gate +
  chunk prof + duel + c4). Owner of rebasing at the next bump: this line's agent.

**Measured verdict (round 116, campaign-20261030-144836): NOT ADOPTED.** The content
gate is IDENTICAL (bit-safety by construction proven), but the perf regressed where it
mattered: solo decode −8.4% (53.9 → 49.4 t/s), c2 −8.3%, c4/prompt neutral. Mechanism:
the fusion serializes the 16 K-blocks (the chain's grid-y) into one block's sequential
loop — free at prompt occupancy (2048 blocks), costly at decode occupancy (1-16
blocks). TF_GLM_HC_FUSE stays off; the patch remains as the receipted design of a
correct-but-unprofitable fusion.

**Status correction (round 119): DEAD on v1.1.9 — does not apply** (developed
pre-v1.1.9, never rebased; `git apply` fails on the pristine tag). Even rebased it
would be near-inert on the native GLM serve: the cache rules are hardcoded
`planned = false` with no grid (glm_dsa_host.zig:46), so the patch's block never
fires — and its interior keeps address cross-prompt share-stops, not the forced
2048-aligned resume tail that owns the resumed-TTFT gap (round 119's decomposition:
the restore is at parity with production; the gap is the chunk granularity). The
round-41 "proven" claim is corrected: its 1,445 ms equals the unpatched behavior.
Rebasing is parked with the finer-grid trade-off (the prefill anchor vs the resume)
documented in LAUNCH.md round 119.

## glm-boot-marks.patch — the boot's phase marks (the loading lever, round 126)

- What: additive `std.log.info` timing at the GLM CUDA load's phase boundaries —
  the checkpoint open, every 10th weight layer, embed+norm+head, the kernel
  modules, the weights upload, the linear pack, the buffers. Logging only; no
  behavior or numerics change (the window's content gate binds the same
  content_sha as the pristine serve).
- Why: the boot's 221.5-s upload+pack bracket (campaign-20261030-210712's
  timestamped docker brackets: digest-end → pack-line) is 95% of the 233-s boot;
  the disk does 7 GB/s O_DIRECT on the shards (which re-read cold every boot —
  the loader's `ck.release()` drops pages so CUDA allocations keep room), so the
  phase is the loader itself, not storage. The marks split the loader's interior
  (read vs convert vs per-tensor upload vs pack).
- Ported against: v1.1.9 (465e48b6); `git apply --check` clean on the pinned
  tree. Owner: this line.
- Status: measurement instrument — passed via `RC_PATCHES` for its window; the
  standing config stays pristine.

## glm-pinned-upload.patch — the loader uploads through the pinned stage (round 126)

- What: `Loader.put()` chunks every tensor through the copy-source Stage's
  reusable 64 MiB PINNED buffer (cuMemHostAlloc, freed with the loader) instead
  of `DeviceBuffer.fromHost`'s synchronous path, which stages each ≤64 MiB chunk
  through a fresh PAGEABLE heap buffer (`copy_source.zig`:
  `const durable = std.heap.page_allocator`) and `cuMemcpyHtoD_v2`s from it.
  Failure contracts kept: drain-before-free on a failed staging, the stage's
  wait-before-reuse, the stage closed with the loader.
- Why: the boot's 45-layer loop is 184.5 s at ~444 MiB/s effective
  (campaign-20261030-212657's marks, ~4.1 s/layer dead-flat, GPU util 0-4%) —
  the pageable-copy regime; the disk does 7 GB/s O_DIRECT. Pinned copies run at
  fabric speed. The DeepSeek v41 loader took this exact pattern upstream
  (7fce15ce9, post-v1.1.9) — our port puts the GLM loader on the tree's own
  Stage machinery, which v1.1.9 already ships.
- Byte-identical by construction: the same bytes, the same device addresses;
  the window's content gate binds the outputs (identical content_sha to the
  pristine serve).
- Ported against: v1.1.9 (465e48b6); `git apply --check` clean. Owner: this line.
- Status: under measurement — passed via `RC_PATCHES` with glm-boot-marks.patch
  (the layer-loop marks show the effect); the standing config stays pristine
  until the receipt.

## glm-readahead.patch — one WILLNEED per tensor range at the loader's funnel (round 126)

- What: `Loader.add()` (the single funnel every tensor's bytes pass through)
  prefaults the exact byte range it is about to copy with one `MADV_POPULATE_READ`
  (22, raw syscall so the pinned std doesn't matter) before the copy. Advice only —
  anonymous memory (the no-mmap fallback) no-ops, errors ignored like `release()`'s
  own madvise. **v1 used WILLNEED and REGRESSED** (211.1 → 278.3 s boot,
  campaign-20261030-221626): WILLNEED's async readahead races the fault loop into
  duplicate I/O; POPULATE_READ is synchronous — each page read once.
- Why: the maps are `MADV_DONTNEED`ed after every layer
  (`safetensors.File.release` — unified memory needs the room), so each layer's
  ~1.8 GiB re-faults from disk in tensor-interleaved order and the kernel's
  readahead never ramps: ~3 s/layer of major faults held after the pinned-upload
  patch removed the copy component (campaign-20261030-215302: the loop 184.5 →
  162.9 s for a −0.48 s/layer copy win). A per-tensor WILLNEED starts the
  full-range read before the faults; a contiguous tensor read at fabric speed is
  ~0.3-0.5 s.
- Byte-identical by construction; the window's content gate binds the outputs.
- Ported against: v1.1.9 (465e48b6); `git apply --check` clean. Owner: this line.
- Status: under measurement — rides with glm-pinned-upload.patch and
  glm-boot-marks.patch in one window; the standing config stays pristine until
  the receipt.

## glm-boot-marks2.patch — the phase marks plus per-tensor timing (round 126)

- What: glm-boot-marks.patch's phase marks, plus `add()` marks the read (the
  map's faults + the host copy) per tensor and `put()` marks the device alloc +
  H2D per tensor — name, KiB, elapsed ms, ~4000 short lines a boot. Logging only.
- Why: both advice-based readahead variants REGRESSED (WILLNEED 211.1 → 278.3 s;
  POPULATE_READ → 289.6 s — campaigns -221626 and -224013), so the fault-read
  hypothesis needs direct evidence: the per-tensor marks decompose the 45-layer
  loop's ~4.1 s/layer into read vs upload. Runs ALONE (it patches the pristine
  loader; glm-pinned-upload.patch edits the same put() lines).
- Ported against: v1.1.9 (465e48b6); `git apply --check` clean. Owner: this line.
- Status: measurement instrument; supersedes glm-boot-marks.patch for this
  window; the standing config stays pristine.
