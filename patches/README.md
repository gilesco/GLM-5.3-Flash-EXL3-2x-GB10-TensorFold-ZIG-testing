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
