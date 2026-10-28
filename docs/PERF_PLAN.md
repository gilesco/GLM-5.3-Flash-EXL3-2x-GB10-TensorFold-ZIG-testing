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
  (ours ~100 vs production 154.5, the clean duel receipt) lives in the *round economics*.
  **Acceptance is ANSWERED (2026-10-24, 115242/11-routerks2.md + production's /health):
  NOT our gap** — our drafter accepts **77.2%** (15660/20292 via our /metrics
  `tensorfold:mtp_drafted/accepted_total`) against production's 62.7% (1610/2569). The
  remaining question is the **shared-round cost scaling at c4**: our 8-row shared round
  costs 76.5 ms (vs 40 ms solo at 2 rows — the routed experts' marginal ~5.5 ms/row at
  topk=8), production's rhythm implies ~34 ms rounds at c4 against ~22 ms solo. Levers:
  the routed-expert kernel's marginal per-row cost (topk=4 via the AZ route cuts it in
  half — the top4 experiment), the cost tables that size shared windows
  (`cuda_lanes.zig`: alone 52.0 base + 6.8/row vs shared 76.0 + 4.1/row), and whether the
  draft block pass (1.6 ms, batched across streams) serializes against verify rounds
  instead of overlapping them.
- **Methodology caveat (recorded with the receipt)**: this window's absolute numbers are
  polluted by the agent's own inference — the session is served by the engine under test,
  and the agent was generating during cell a's probes (solo 14.2/34.1 t/s, Δ58%, is the
  agent's traffic, not engine behavior). The duel receipt (campaign-20261023-221204)
  remains the clean baseline; probe windows now run with the agent silent.

## Front 2: prefill slope (40%) — the MoE prompt path

Prompt chunks are 90% router+routed (a 2,048-row chunk: router 910 + routed 793 of
1,892 gpu ms). Three levers, the first one new tonight:

- **Lever A2 — the prompt-router pathology (ks hypothesis REJECTED 2026-10-24,
  campaign-20261024-115242/11-routerks2.md)**: the A/B ran clean (the agent silent —
  the stock cell reproduces the duel exactly: solo 55.3/55.1 Δ0.3%, c4 101.8, slope
  0.659). The `router_ks 8 → 2` patch **did not move the router section** (1096 → 1090
  ms of the chunk) and the **content gate flipped a token** — the patch died (#1079
  protocol). The +16% decode in the patched cell was an acceptance lottery on the
  bit change, not a mechanism. **The pathology remains open with the pipeline-depth
  hypothesis eliminated as the pathology's cause**: next candidates are (a) a 10-minute microbench of
  `router_tile` vs `router_part` at m=2048 (**DONE 2026-10-24, campaign-20261024-125609:
  the forced fallback costs the same — router 1140 vs 1096 ms — and the content gate says
  IDENTICAL: both kernels share the pathology; the hand-rolled mma approach itself is the
  bottleneck**), (b) **the GEMM replacement (the scoped fix)**, (c) the
  w-matrix layout/packing as the latency source. The prize stands: chunk 1641 → ~600 ms
  would put prefill at production parity with zero quality risk.
  **The GEMM replacement's design (scoped from `families/glm/cuda_mm.zig`, 175 lines)**:
  glm_mm is the engine's bf16/tensor-core linear machinery — "weights tiled once at load,
  any row count, **row-stable K slices**" (the row-stability is the bit-property that
  matters). The replacement: skip the `.router_pack` fix (keep the raw bf16 gate
  [288][4096]), call glm_mm **once per K-slice (ks=8 slices, x-sliced [m][512] ×
  w-sliced [288][512])**, then the existing `sliceSum` adds the 8 partials in order —
  preserving the current kernel's exact summation structure so the cross-build content
  gate (the repo's patch bar: "replies unchanged", the #1079 precedent) can pass.
  glm_mm's tile16 takes bf16 [n][k] directly; its split_target=192 launch shaping is the
  performance win (column tiles × slices aimed at the GB10's SMs, prompt blocks sharing
  weight tiles in L2). Files: `weights.zig` (keep the raw gate ref), `cuda_glue.zig`
  (the router() body), possibly `cuda_mm.zig` (an exported row-count-generic entry).
  Wiring ~30–50 lines; verify with the content gate + the chunk prof + the §2 gates.
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
  **Status 2026-10-24 (post-window)**: the variant served **topk=8 three times** — the
  config edit never persisted, because the snapshot's `config.json` is a *symlink into
  the shared blobs* (the edit aliased the base checkpoint or vanished; the early consoles
  that would show which were reboot-wiped). Fixed in the phase: the variant's config is
  now a REAL file, verified to read 4 before anything serves (verified on both hosts
  2026-10-24 12:48). The next window's top4 runs with zero prep. The 115242 top4 receipt
  doubles as a clean instrument validation — the duel baseline reproduces exactly with
  the agent silent (solo 55.8/55.7 Δ0.2%, c4 101.0, slope 0.660).
  **Experiment 1 MEASURED (2026-10-24, campaign-20261024-135458 — the real topk=4 serve,
  unretrained selector, after the full gate ladder: the parse's `want` pin + the EXL3
  module-global `top_k`, both patched in glm-topk4b.patch)**: routed 18.45 → 11.0 ms at
  2 rows (−40%; the weight-traffic union isn't slot-proportional, hence not −44%),
  verify 40 → 32.3 ms, 8-row shared rounds 76.5 → 53.3 ms. **Solo 63.5/63.6 (Δ0.2%) =
  108% of production's 58.7 — the generation-tok/s front crosses the bar at solo**; c2
  103%; **c4 120.4 = 78% of production (was 65%)**; slope 0.593 (−10%); acceptance rose
  (4.8 tok/chunk vs 3.37 — the fewer-expert target drafts easier). What remains: the
  quality battery (the unretrained selector is guilty until measured — config.sh stays
  topk=8 until it passes), and the AZ full checkpoint (the retrained selector) as the
  quality-safe route to the same win.
  **The battery RAN (2026-10-24, campaign-20261024-142659/13-top4qual.md)**: §2 gates
  ALL PASS (drafted=plain on the new model ✓); the full tool-eval **93/100 — above
  stock's 92** (production's anchor: 95); hardmode **89/100 vs production's 91** (the
  line's first hardmode datum — no stock reference exists). **The fork (user)**: adopt
  top-4-unretrained as the default now, or hold stock and route through the AZ full
  checkpoint (the retrained selector). **(user, 2026-10-24): ADOPTED** — config.sh's
  qualified defaults serve top4cfg + glm-topk4b (round 27; the scoreboard regenerated:
  solo 108%, c2 103%, c4 78%, streaming ~189%, prefill 45%, loading 27%).
  **Experiment 2 (the AZ full checkpoint — the quality-safe upgrade to the same lever)**:
  the download is COMPLETE (the Xet CAS survived the reboot; 170 G materialized,
  topk=4, 288 experts, auto-round int4 — verified 2026-10-24). The route: requant
  auto-round int4 → EXL3 4bpw (the engine's format; try exllamav3's streaming convert
  first — it may dequant per-layer with no intermediate; else dequant→bf16 ~640 GB
  intermediate with the disk choreography, then quant, then delete the intermediate),
  serve with glm-topk4b (the reference is already topk=4), then the battery + the perf
  probes. GPU work: hours, window-gated. If the battery passes, the default upgrades to
  the retrained selector.
  **Discovered en route (a separate lever, recorded)**: `AxonZeta/GLM-5.3-Flash-AZ-Focus-EXL3`
  — NOT azampatti's mod: AxonZeta's own weight edit against second-guessing (40–50%
  fewer thinking tokens at max effort, AIME 81.7 vs 75.0, GPQA up, SWE-bench 92 vs 95),
  **topk=8 intact, EXL3 4bpw native — serve-ready for our engine as-is**. For the
  agent-farm workload the fewer-thinking-tokens effect is an effective-throughput
  multiplier; a candidate experiment after the AZ route.
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

## The roadmap (2026-10-25, from the upstream sweep + bernisse's second-pair bake-off)

*Bernisse (forum post #926, a second 2×GB10 pair, same fixed prompts/thinking-off/T=0):
native 1.1.x prose 40.4 / code 62.3 / prefill64k 1,448 / 4× 217; **the fork (0.6.6+106)
49.1 / 91.6 / 2,483 / 639**; technigma NVFP4 43.0 / 67.4 / 2,933 / 359. Their native
4× (217) beats our measured 120 — workload/acceptance difference to reconcile; **the
fork's 639 aggregate and 91.6 code-decode make the PATCHED PYTHON STACK the field's
concurrency reference, not vanilla production.** They also independently confirm our
router diagnosis ("the router fix looks like the highest-leverage kernel work in the
field right now") and flag a quality instrument we've never run: **TC-92 (tenant
isolation) + TC-51 — the fork fails TC-92 deterministically; technigma is the only
clean stack (97).**

1. **Tonight**: the AZ battery digest (the convert lands ~20:00, `done.` ~21:00) — the
   retrained selector vs the top4cfg references.
2. **The thinking-efficiency track (quality + effective throughput)**: the AZ-Focus
   battery (test-ready on both hosts) + the `TF_GLM_CLEAR_THINKING` knob (#840, landed
   upstream Oct 25) — fewer thinking tokens at equal-or-better accuracy = the agent
   farm's effective throughput, quality-safe by construction.
3. **The router GEMM replacement** (Front 2 A2) — now field-confirmed as the
   highest-leverage kernel work; the 8-slice bit-stable design is scoped.
4. **The v1.1.5 candidate test**: released Oct 25; visible glm content = functional,
   no measured perf — the gate holds, BUT **#1218 (our slot-leak) is MERGED and slated**
   — the first correctness reason to test a bump. Protocol: build 1.1.5's tree,
   rebase glm-topk4b, run the §2 gates + the perf probes; adopt only receipts-equal-or-better.
5. **The pages×fp8 marriage** (#1163 mergeable-clean, #1211 stacks): when paging×fp8
   lands, the 512K window admits 8 streams ON fp8 — concurrency and cache bytes together.
6. **The concurrency references**: reproduce bernisse's native-217 cell (workload-matched:
   code-domain prompts, thinking off) and study the fork's scheduler for the 639
   mechanism; both re-anchor Front 1.
7. **The isolation quality round**: add TC-92/TC-51-class scenarios to our battery —
   drafted/serial equality does NOT cover tenant isolation (bernisse's caveat).

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

## Vision qualification (queued, post-requant — (user), 2026-10-24)

After the AZ requant completes: validate and qualify vision on this model. Scoped at
the pin: the serve flags exist (`--vision`, `--vision-image-tokens` — the CLI tests
cover the budget parsing); the checkpoint carries the tower (347 `model.visual.*`
tensors, bf16); rank 0 encodes, the others receive (the engine's comment). **Staged:
`VISION` / `VISION_IMAGE_TOKENS` knobs in config.sh wired into serve_args (both ranks).**
The qualification round: serve with VISION=1 → the §2 gates with the tower loaded →
a vision receipt (a real image prompt: describe it, read text from it — content checks)
→ the memory arithmetic (the tower's load cost in the serve line) → the AZ interplay
(the requant runs `-vb 16`: the tower stays unquantized ✓ staged in az-window.sh).
GPU window; ~20 min.

- 2026-10-24 (later): **the two-box conversion pattern (user asked)** — today's convert
  is single-GPU by design (exllamav3 quantizes layer-by-layer with calibration in one
  process; no layer-range/multi-host mode), so the worker cannot share the bottleneck.
  What the worker CAN take: (a) half of any future stream-dequant (the shard-parallel
  CPU+disk stage) with the rail merge, and (b) the NEXT lever's downloads — the
  AZ-Focus-EXL3 download started on the worker now (59 files, ~163 GiB, 1.7 T free)
  while the head's convert runs. The worker always joins the serving/battery as rank 1.
- 2026-10-25: **v1.1.5 released** (Oct 25) — the visible glm items functional, not perf
  (the pool-queue fix `2e33e297d`, `TF_GLM_CLEAR_THINKING` #840, the shared-gate
  saturation fix, the cold-prompt-state fix, `return_token_ids` instrumentation, the
  tool-call max_tokens fix) — the perf gate holds, but the **#1218 slot-leak is MERGED
  and slated** → the first correctness reason to run the v1.1.5 candidate test (with
  the glm-topk4b rebase). Also: NVIDIA's FP4-Local and XSFP4 checkpoint formats land in
  the engine's model catalog (the README's Pick-a-model row) — the checkpoint-class watch.
- 2026-10-25: **the forum (914 posts)**: bernisse's second-pair bake-off (the roadmap's
  numbers + the TC-92/TC-51 quality instrument + the fork-as-field-reference) — the
  roadmap section carries the synthesis. **kindlingai/gr8: a THIRD engine** (a
  from-scratch CUDA engine; GLM-5.3 full on TP=3-6 + the Flash TP=2 doc with caveats,
  "not final") — the ecosystem's third implementation; the watch: its numbers + the
  DFlash/vision roadmap. ajvazan daily-drives an EXLR8 quant on TensorFold with no
  issues — the checkpoint candidate class.

- 2026-10-27: **the v1.1.6 rebase ADOPTED** (round 33) — the engine at `eea9dc9c`; the A/B on the
  identical Focus serve: **c2 74.7→94.3 (+26%), c4 119.8→150.6 (+26%)**, hardmode 90→91 (ties
  production), everything else flat. The upstream #1322 KV-pages default transferred at +26% on
  EXL3. **The Focus's c2 gap vs the top4cfg is CLOSED** (94.3 ≥ 90.0) — the perf question narrows
  to the solo/c4 slope. The served default's scoreboard rows are v1.1.2-era — the top4cfg re-qual
  on v1.1.6 is the next window before the rows/chart move. The bench updated to v3.0.1 (the
  graphics + the web `serve` + hardmode to TC-104) — scores across the bench boundary wait for
  the re-baselines (production's in flight; our serves next window).

- 2026-10-27 (round 34): **the TTFT-at-depth frontier root-caused at source level** — the native
  engine's prompt cache is a token-prefix cache, but it keeps states only at conversation-structure
  marks (history / rewind / stable-diff / shared cuts), and a chat request's shared cuts cover only
  the system prefix. Shared prefixes inside user content never get a kept state → full prefill every
  request → the +0.663 ms/tok slope. The fix shape: keep grid-boundary states inside long prompts
  (spans are token-derived, so resumes stay exact) — a marks-policy patch experiment for the next
  window, gates battery around it. Full chain: LAUNCH.md round 34.

- 2026-10-27 (round 37): **the score comparison completed on v3.0.1** — the top4cfg ties
  production on tool-eval (96 = 96, 133/138 both) and leads every perf front (c2 124%, c4
  109%, solo 110%) but takes hard-mode 76% vs 90%; the Focus 95/86%. The bar's fronts:
  tok/s CROSSED; **prefill 78% and loading 37% remain the open fronts** (loading
  decomposition still pending), plus the TTFT-at-depth frontier (3%). The way-forward
  stack: (1) the marks-policy patch experiment for the frontier, (2) the loading
  decomposition from the restart logs, (3) the upstream watch for a marks-policy/prefix
  fix landing, (4) the (user) hardmode-trade decision on the default.

- 2026-10-27 (round 38): **the (user) hardmode-bar decision** — hardmode is THE quality
  measure; the served default swapped to the Focus (92 run-level vs production's 94). The
  chart re-anchored to the Focus cell (solo 98%, c2 109% CROSSED, c4 99%, ITL 102%, loading
  41%, prefill 67%, TTFT-at-depth 3%). The perf fronts keep their receipts: the top4cfg's
  crossed rows remain the documented perf leader; the open perf items are unchanged
  (TTFT-at-depth → the marks-policy patch; loading decomposition; prefill; the v1.1.8
  indexer A/B). The quality line's new track: the beat-94 plan (multi-trial variance, the
  AZ retrained selector un-park decision, the upstream EXL3-format watch).

- 2026-10-28 (round 40): the overnight verdicts — the variance check says the Focus's 92
  is stable (three variant seeds, one score); the recommended-settings cell scored 91 (no
  lift; greedy stays); the marks-patch passed every gate but the duel didn't move (the
  instrument can't see the cross-request resume — a same-prefix-repeat probe + the cache
  counters is the next experiment before the patch is judged). The bench moves to
  --parallel 2 for all future runs (the (user) round-39 call). The TTFT-at-depth frontier
  remains open pending the better instrument; the loading and prefill fronts unchanged.

- 2026-10-28 (round 41): **the two-shot instrument landed and re-wrote the frontier** —
  the marks-patch is proven (cross-prompt prefix resume, gates PASS, the fingerprint:
  patched pair-1 shot-1 1,445 ms resumed from the duel's kept states vs the unpatched
  cold 7,359 ms); identical-prompt resume exists unpatched (80%); production's resume is
  65 ms where ours is 1,440 ms (22× — the snapshot-restore cost is the new named gap);
  the duel's cross-length non-resume is junction tokenization (the instrument, not the
  patch). The frontier decomposes into: cold prefill (the indexer path → v1.1.8's
  TF_GLM_INDEX_CUDA) + restore cost (the snapshot copy — a source-level dig) + cache
  churn under load (the budget/eviction lever, a knobs cell). The patch's adoption
  (a standing PATCHES line) is the (user) call: exactness-safe, proven, helps repeated
  prefixes; it does not move the cold-prefill slope.
