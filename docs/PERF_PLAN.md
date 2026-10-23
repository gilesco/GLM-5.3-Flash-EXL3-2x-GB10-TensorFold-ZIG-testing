# Performance plan: closing the measured fronts

> **The bar (user, 2026-10-23):** surpass production 0.6.6 on model loading time,
> generation tok/s, and prefill rate. This document is the continuing-look home for
> that hunt: the measured scoreboard, the fronts in attack order, every lever with its
> mechanism and verification plan, and the research log. The mandate (user): look hard,
> keep looking, go outside the scope if necessary — everything documented.

## The scoreboard (the duel, 2026-10-23 — five direct probes, both engines)

| front | production | ours | ours as % | instrument |
| --- | --- | --- | --- | --- |
| solo decode | 58.7 t/s | 55.2 t/s | 94% | 2048-token greedy ×2, Δ 0.6% |
| c2 aggregate | 87.1 t/s | 72.1 t/s | 83% | 2×2048-token concurrent |
| **c4 aggregate** | **154.5 t/s** | **99.8 t/s** | **65%** | 4×2048-token concurrent |
| request overhead | 183 ms | 200 ms | 92% | TTFT at max_tokens=1, solo |
| prefill slope | 0.265 ms/tok | 0.662 ms/tok | 40% | TTFT deltas at d0/d4096/d8192 |
| streaming pace | 17.0 ms/tok | 17.0 ms/tok | 100% | per-token ITL (ours arrives in 3.4-token bursts) |
| loading (restart wall) | 93.7 s | ~265 s | 35% | the knobs phase; engine itself ~250 s |

Retired claims: the "concurrency cliff" (bench artifact, round 6), the "2-second fixed
overhead" (was concurrency queueing), the bench's shallow-depth concurrent rows as an
instrument (they swing up to 2× run to run — campaign-20261023-221204's confirm receipt;
single-run A/Bs on them are void as evidence; the duel's probes and the d8192 rows are
the stable instruments).

## Front 1: concurrency scaling (65% at c4) — the biggest gap

Production retains 66% of per-stream speed at 4× concurrency (154.5/4 = 38.6 vs solo
58.7); we retain 45% (99.8/4 = 24.95 vs 55.2). Our GPU is saturated with useful work
(wall ≈ gpu in every prof line), so this is about *how the work is batched*, not about
waiting.

- **Lever A — ANSWERED (2026-10-24, campaign-20261024-000708)**: our verify rounds
  **do batch across streams** — at c4 the shared windows run 8 rows (4 streams × ~2 draft
  rows) in one 76.5 ms round, while a solo stream's window is 2 rows/41 ms. Batching is
  NOT the missing piece; a #1155-style glm port is the wrong patch for this gap.
- **Lever B — ANSWERED, retired (same receipt)**: `TF_GLM_SHARED=0` (windows alone) at
  c4 = **51.6 t/s** vs sharing's 70–105 in the same sitting (and 4×52 ms per-stream rounds
  of unshared GPU work). Sharing IS our concurrency-scaling mechanism — the knob stays at
  its default.
- **What remains of Front 1**: with batching present and sharing load-bearing, the c4 gap
  (ours ~100 vs production 154.5, the clean duel receipt) lives in the *round economics*:
  acceptance at concurrency (production's /health shows 62.7% acceptance at 2.10
  drafts/round — ours comes from the request-log lines, since our /health carries no
  drafter counters), the cost tables that size shared windows
  (`cuda_lanes.zig`: alone 52.0 base + 6.8/row vs shared 76.0 + 4.1/row), and whether the
  draft block pass (1.6 ms, batched across streams) serializes against verify rounds
  instead of overlapping them. Next window: read the acceptance counters at c4 and the
  verify walls at 16/32 rows.
- **Methodology caveat (recorded with the receipt)**: this window's absolute numbers are
  polluted by the agent's own inference — the session is served by the engine under test,
  and the agent was generating during cell a's probes (solo 14.2/34.1 t/s, Δ58%, is the
  agent's traffic, not engine behavior). The duel receipt (campaign-20261023-221204)
  remains the clean baseline; probe windows now run with the agent silent.

## Front 2: prefill slope (40%) — the MoE prompt path

Prompt chunks are 90% router+routed (a 2,048-row chunk: router 910 + routed 793 of
1,892 gpu ms). Three levers, the first one new tonight:

- **Lever A2 — the prompt-router pathology (NEW, 2026-10-24 evening; likely the front's
  real cause)**: the **router section alone eats 910–1,125 ms of every 2,048-row prompt
  chunk — 55–69% of chunk GPU time** (three receipt lines agree: campaign-20261023-145449,
  the 235008 salvage, the 000708 cells). The same math is a 4.8 GFLOP bf16 GEMM
  (2048×4096×288) that production's cuBLAS path does in ~1 ms — production's *entire*
  chunk is 543 ms (slope 0.265); our router section alone is 2× that. Source at the pin:
  `glue.router` (families/glm/cuda_glue.zig:146, `router_ks = 8`) launches `router_tile`
  (glm_glue_bmm.cuh — a proper double-buffered cp.async GEMM); the alignment check
  (x_stride=4096 elements) always passes, so the unstaged fallback `router_part`
  (glm_glue_router.cuh — individual 4-byte loads per MMA step, no staging) should not
  run. Yet the section costs ~25 ms/layer (193 GFLOP/s effective, ~0.1% of tensor peak;
  the same kernels at m=2 cost 47 µs/layer — the slowdown is superlinear in rows).
  **Patch candidates**: (a) microbench both kernels at m=2048 and force the fast one
  (10-minute window experiment); (b) reuse the dense-projection GEMM machinery
  (cuda_mm, the 0.7 ms class) for the router; (c) raise the tile's K-depth (router_ks=8
  gives an 8-step pipeline that never reaches steady state). **Expected**: chunk
  1641 → ~600 ms, prefill slope 0.662 → ~0.25 (production parity), zero quality risk
  (same math, faster scheduling — the §2 gates verify). **This outranks the AZ lever:
  bigger, free, and no quality battery needed.**
- **Lever A — the AZ checkpoint (community, 2026-10-23) — (user) APPROVED, with a heavier
  quality battery**: `azampatti/GLM-5.3-Flash-AZ` routes top-4 of the original top-8
  (18B → ~12.5B active) with the shared experts *retrained* to keep the selection faithful
  ("95% exact same answer"), and ajvazan's quality-degradation test passed everything but
  one 8k-context case. On their Python stack: ~2,300 tok/s cold prefill, 50 tok/s decode
  to 150k. For us the routed path halved should cut a prompt chunk's routed share
  (~793 → ~400 ms) and verify's (~18.5 → ~9 ms): expected **~+25% prefill, ~+25–29%
  decode**. Base model is MIT. **The download is running** on the head (nohup, console
  `/tmp/az-download.console`, ~169 GiB, no GPU needed; 1.2 TB free ✓).
  **Loader verification (2026-10-23, source at our pin)**: the native loader is
  EXL3-only — no GPTQ/autoround path exists in `zig/src` (91 files reference exl3; the
  gptq hits are unrelated xpu files) — so AZ *as distributed* cannot serve natively.
  But the glm config parser reads `num_experts_per_tok` from the checkpoint
  (`families/glm/config.zig:163`) and the routing kernels take topk as a runtime config
  field, which splits the lever into two experiments:
  1. **top4-on-our-own-checkpoint (cheap, next window)**: a variant model dir (symlinks
     to the existing safetensors + a config edit to `num_experts_per_tok: 4`) — measures
     the pure compute saving and the *un-retrained* quality floor with zero download.
     If the engine refuses top4, the refusal prints its arithmetic — a receipt.
  2. **the full AZ (after download + requant)**: the AZ checkpoint needs dequant
     int4→bf16→EXL3 4-bit (exllamav3 convert; window-gated, hours; disk choreography:
     ~640 GB bf16 intermediate + 169 GB source + output against 1.2 TB free — stage it,
     delete the intermediate before the gates run).
  **The quality battery (user: a new checkpoint means seriously more quality checks)**:
  both experiments get §2 gates + the full tool-eval + the structured-output scenarios +
  hardmode TEB + the duel probes — a double-quantized modded checkpoint is maximally
  guilty until measured.
- **Lever B — upstream expert-prefill kernels**: #1143 (Flash Next NVFP4 experts on
  prompt tiles, decode-form sums) is the mechanism; the glm port remains the upstream ask.

## Front 3: loading (35%) — the packing cache

The engine loads in ~250 s across every knob cell (walls 251–269 s; orchestration only
~8–17 s; the drafter ~11 s). The cost is the native loader's convert-and-pack path
(dense4/head4 packing + fp8 KV prep at load). Fix shape: a packed-weights cache beside
the existing byte-verify digest cache — pack once per (checkpoint, packing config), reuse
on every start. Scope from `load_plan.zig`/`cuda_qpack.zig` at our pin.

**Scope verified 2026-10-23 (source at our pin, re-cloned after the site's reboot wiped
/tmp)**: the load flow is `families/glm/weights.zig` (581 lines) — parallel preads of the
shards, then pack transforms into the serving layouts (`fixed(..., .router_pack)`,
`.hc_pack`, stacked projections and expert mixes via the `Loader`'s `stack`/`mat`/`experts`
helpers; the EXL3 handling in `families/glm/cuda_exl3.zig`, 206 lines). The cache
precedent exists: `core/checkpoint_identity.zig` already caches per-file digests
("83/83 cached files" — the byte-verify line in every serve log). **Patch shape**: after
a load, serialize the packed weight slabs keyed by (checkpoint digest, packing config =
DENSE_BITS/HEAD_BITS/KV_CACHE); on a matching start, skip read-and-pack and restore
(I/O-only, ~30–40 s at NVMe speed — a ~4–5× load win). Storage ~170–200 GB per config —
fits, but the cache should be one-config-deep with the key visible in /health so a stale
config self-invalidates. Open engineering question for the patch window: whether the
pack runs on CPU before upload (cache skips it cleanly) or interleaved on GPU (the cache
must then capture the device layout).

## Research log

- 2026-10-23: upstream v1.1.3 released (21:25) — **no glm/cuda perf content** (one
  sampler change, ~1% unverified; a Nemotron-only draft-schedule change −17%/+13%);
  **(user) version bumps are perf-gated** — we stay at TF_COMMIT=1351ed16, the pin is in
  local.sh. The builder now builds and ships libtfgrammar (xgrammar v0.2.8) beside the
  binary — our source builds never carried it (structured output ran degraded; the
  quality round will test whether the 3-point tool-eval gap closes) — a builder fix at
  this pin, not a version bump.
- 2026-10-23: the forum (381350, 904 posts, active today): **azampatti/GLM-5.3-Flash-AZ**
  (the top-4 routing mod above) is the first community checkpoint mod that attacks
  per-token *compute* rather than quantization; ajvazan's quality pass makes it testable.
  Also: the request log lines carry per-request acceptance (`accepted=168/299 rounds=119`)
  — an instrument for the drafter's behavior. grebneke's finding: the reasoning-effort
  hint is ignored at depth when it sits at the prompt top (template-level, quality-adjacent).
- 2026-10-23: PR #1155 read — batched MTP drafting is flashnext-family-only (17 files);
  the glm port is the concurrency-scaling patch candidate.
- 2026-10-23 (later): **(user) approved the AZ attempt** — "a new checkpoint will mean
  seriously more quality checks" — the quality battery is expanded accordingly (Front 2
  Lever A). The download is running on the head; the loader check at our pin found the
  native engine EXL3-only but topk-dynamic, splitting the lever into the cheap
  top4-on-our-own-checkpoint experiment (no download) and the full-AZ EXL3 requant
  (window-gated, hours).
