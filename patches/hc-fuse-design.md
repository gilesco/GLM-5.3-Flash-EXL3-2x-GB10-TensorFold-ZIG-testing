# patches/hc-fuse-design.md — the fused hc_pre kernel (round 114's census target, 2026-10-30)

The census (results/campaign-20261030-134836/31-ncurouter.md) named the target: the
hyper-connection pre chain fires TWO kernels per hcPre — tf_glm_hc_partial (or
hc_partial8 at decode rows) then tf_glm_hc_finish — 4 launches per layer, ~480 glue
launches per prompt pass, micro-scale kernels (p50 104 µs + 54 µs prompt-shaped).
The fusion: ONE kernel per hcPre, bit-identical by construction.

## Why this pair and not the router pair

- The router's slices are already z-batched in one launch (tf_glm_router_tile,
  grid.z = router_ks = 8). Fusing slice_sum in would serialize the 8 slices per
  block (~8× less parallelism, the tile kernel is 204 µs BECAUSE of its parallelism)
  or need a grid-wide ordered reduction (not bit-safe without grid sync).
- The hc chain's second kernel (finish) is the partial's ONLY consumer; nothing else
  reads `part`. The fusion is self-contained: the partials live in shared memory and
  die with the block.

## The bit-safety argument (by construction)

The fused kernel performs the SAME operations in the SAME order as the two-kernel
chain; only the storage of the intermediates changes (shared memory instead of
global `part`). Concretely, per row r:

1. The dots: per K-block b (0..15, ascending) and step s (0..kb/128, ascending), a
   128-thread group computes the 24 dots with the exact tiling of tf_glm_hc_partial
   (8 weight-rows m0 × 16 c-lanes; the fma chain order `d = x1*w1; d = fma(x0,w0,d);
   for j=2..7 fma`; the c-reduce `shfl_xor` o = 8,4,2,1; acc += per step, ascending).
   These are tf_glm_hc_partial8's exact per-step ops — its own comment carries the
   contract: "same ops, same order".
2. The sum of squares: per (r, b), one 128-thread group walks the block's elements
   (base + step*128 + t, ascending steps), warp-butterfly o = 16..1, then
   `(smem[0]+smem[2]) + (smem[1]+smem[3])` — the partial kernel's exact reduce.
3. The finish's math: glm_hc::finish's tail runs inline on the shared partials —
   mix = Σ_b dot_total[b][lane] (block-ascending, the finish's loop order), ss per
   block, the Sinkhorn rows/cols in `iters` rounds, the pre/post/comb writes, the
   normed row c[k] and the xs sums — byte-for-byte the template body, unchanged.
4. The f32w variant: the same kernel with the finish's template parameter W=float
   (two symbols, as finish/finish_f32w are today).

## The kernel

`tf_glm_hc_pre_fused` (+ `_f32w`), in zig/kernels/cuda/glm_glue_hc.cuh:

- Launch: grid = (rows), block = 1024 threads (8 groups of 128; grp = threadIdx.x>>7,
  t = threadIdx.x&127 — partial8's mapping).
- Loop b in 0..nb-1: group g covers step g of block b (base = b*kb + g*128):
  - g < 8: the 24-dot accumulation for (b, step g) → dots[b][g][24] in shared
    (nb × 8 × 24 f32 = 12 KB at nb=16 — static, no dynamic shared needed).
  - g == 0: the ss walk for block b → ss[b] in shared (the butterfly + the 4-warp
    smem order, per b).
- After the b loop (__syncthreads): the per-(b,dot) reduction acc = Σ_g dots[b][g][d]
  in group order (= the step order — partial8's reduction, extended over b) →
  dtot[b][24]; then the finish's tail inline: mix = Σ_b dtot[b][lane], the Sinkhorn,
  pre/post/comb, the normed row, xs.
- Registers: the finish's xu[2][S] uint4 loads and its c[16] math join after the
  dots loop — the kernel's register budget needs the compile loop's report (the
  contract record's `usage` fills from actuals, as the round-84 record notes).

## The plumbing (zig)

- zig/src/families/glm/cuda_glue.zig: `hcPre` gains a fused branch behind
  `TF_GLM_HC_FUSE=1` (off by default — the A/B's patched arm turns it on): one launch
  with the partial's args + the finish's args composed (x, fn, base, scale, nw, out,
  xs, post, comb, eps, hc_eps, nb, iters, rows). `part` is not passed.
- The kernel functions register in the glue Set next to hc_partial/hc_finish (the
  glm_glue image's kernel list — both fused symbols).
- zig/src/core/contract/glm_cuda_records.zig: new launch records for the two fused
  symbols (the resource `usage` copied from hc_finish's shape sibling until the
  contract run prints actuals — the round-84 pattern).
- The decode path rides the same kernel: rows<64 launches the same fused symbol with
  fewer blocks — the CUDA graphs then capture ONE node per hcPre instead of two.

## The A/B protocol (the user's standing decision, rounds 107/108)

- The patched arm: `PATCHES=glm-hc-fuse.patch TF_GLM_HC_FUSE=1`; the standing config
  otherwise identical (DENSE_BITS=4, HEAD_BITS=4, KV_CACHE=fp8, CONTEXT=524288,
  PARALLEL=2 for the bench legs, VISION on).
- 2+ runs per arm (the ~16% same-engine run variance); the definitive benchmark
  protocol (the full 104-suite + hardmode + perf, seed 42, timeout 240), same day.
- The content gate: bit-identical replies across the suite (the fused kernel must be
  arithmetic-invisible). Any content diff = the fusion is wrong; fix, don't adopt.
- The metrics that matter: c4 aggregate + c4@d8192 (the concurrency deficit is the
  target), prefill, decode solo, ITL; the anchors/chart update only on a qualified
  fact.

## Expected effect (honest)

- Prompt: ~160 fewer launches/pass + the inter-launch gaps; kernel time ~neutral
  (the same math; the global `part` round-trip of rows×512 f32 disappears). The glue
  stage's span should shrink by roughly a third of its launch count.
- Decode: the graphs shrink by one node per hcPre per layer (160 nodes per verify
  graph) — replay overhead drops; ITL the visible metric.
- Under 4-way load the enqueue-serialization relief is the real target (the round-101
  finding: the glue's launch stages inflate ~18× under load).
