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

- **Lever A — do our verify rounds batch across streams?** The duel's ITL shows our
  stream arriving in 3.4-token bursts (one DFlash2 verify round per ~57 ms). If four
  streams run four *separate* verify rounds instead of one 16-row batched round, that is
  the root cause. Test: `TF_GLM_PROF=1` at c4 on our stack (a GPU window) — the round
  lines will show rows-per-verify. If per-stream: patch candidate — batch the draft
  windows across streams (the mechanism upstream PR #1155 implements for Flash Next:
  `cuda_lanes_draft/verify/prompt_windows`, 17 files, flashnext-family only — a glm port
  to our `dflash*`/`cuda_lanes*` files).
- **Lever B — measure the other direction**: `TF_GLM_SHARED=0` (each window alone) at
  c4. If alone *beats* shared at c4, the shared-window path is thrashing, not helping.
- **Lever C — production's schedule**: the 0.6.x engine's shared-window logic is the
  reference behavior (the old repo's REBASE-NOTES/patches describe it); read it and
  compare with our round loop.

## Front 2: prefill slope (40%) — the MoE prompt path

Prompt chunks are 90% router+routed (a 2,048-row chunk: router 910 + routed 793 of
1,892 gpu ms). Two levers:

- **Lever A — the AZ checkpoint (community, 2026-10-23)**: `azampatti/GLM-5.3-Flash-AZ`
  routes top-4 of the original top-8 (18B → ~12.5B active) with the shared experts
  *retrained* to keep the selection faithful ("95% exact same answer"), and ajvazan's
  quality-degradation test passed everything but one 8k-context case. On their Python
  stack: ~2,300 tok/s cold prefill, 50 tok/s decode to 150k. For us the routed path
  halved should cut a prompt chunk's routed share (~793 → ~400 ms) and verify's
  (~18.5 → ~9 ms): expected **~+25% prefill, ~+25–29% decode**. Base model is MIT.
  **Verification plan (our instruments decide)**: download (~182 GB, 1.2 TB free ✓),
  serve on our native engine (its config declares the routing), run the §2 gates, the
  tool-eval (the Act II lesson: a modded checkpoint is guilty until measured), then the
  duel probes. Disk and window required.
- **Lever B — upstream expert-prefill kernels**: #1143 (Flash Next NVFP4 experts on
  prompt tiles, decode-form sums) is the mechanism; the glm port remains the upstream ask.

## Front 3: loading (35%) — the packing cache

The engine loads in ~250 s across every knob cell (walls 251–269 s; orchestration only
~8–17 s; the drafter ~11 s). The cost is the native loader's convert-and-pack path
(dense4/head4 packing + fp8 KV prep at load). Fix shape: a packed-weights cache beside
the existing byte-verify digest cache — pack once per (checkpoint, packing config), reuse
on every start. Scope from `load_plan.zig`/`cuda_qpack.zig` at our pin.

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
