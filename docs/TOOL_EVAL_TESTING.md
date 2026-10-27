# Test plan: performance first, then quality (phased)

> **2026-10-23: the campaign script (`scripts/run-qualification.sh`) is the primary
> instrument** — phased, resumable, production-guarded; this file's ledger (Status
> below) is where its receipts land. The `run-bench-matrix.sh` phases below remain as
> the manual single-config fallback. **(User) The bar: surpass production 0.6.6 on
> loading time, generation tok/s, and prefill rate.**

**Strategy:** quick performance measurements select the best server configuration; the
scenario suites (tool-calling, Hard Mode) run only on the winner. **No `--hardmode` in
early testing.**

**Rule of the campaign:** within a phase, the bench options are **fixed and identical for
every test**; the server configuration is the only variable. Results and session notes land
as commits in this repo.

## The matrix (server configurations under test)

| Label | `PARALLEL` | `RECALL` | Extra | What it isolates |
| --- | --- | --- | --- | --- |
| `baseline` | 4 | off | — | today's default — the reference point |
| `pdl` | 4 | off | `PDL=1` | programmatic dependent launch on the mm kernels |
| `dense8` | 4 | off | `DENSE_BITS=8 HEAD_BITS=8` | dense/head packed to 8 bits at load (memory + packed mm tiles; exactness gates hold *within* the config) |
| `dense6` | 4 | off | `DENSE_BITS=6 HEAD_BITS=6` | packed to 6 bits |
| `dense4` | 4 | off | `DENSE_BITS=4 HEAD_BITS=4` | packed to 4 bits — the 2026-10-22 leader (55.4 solo); quality gate owed |
| `roce0` | 4 | off | `TF_ROCE=0` | attribution control: the engine's RoCE gather off, NCCL's RDMA transport kept — splits the 1.1.2 engine gain from the transport gain |
| `chan4` | 4 | off | `NCCL_PASSTHROUGH=1 NCCL_MIN_NCHANNELS=4 NCCL_MAX_NCHANNELS=4` | the measured NCCL channel set on the RoCE path (untestable before the transport fix) |
| `dense4pdl` | 4 | off | `DENSE_BITS=4 HEAD_BITS=4 PDL=1` | the leader + PDL |
| `dense4chan4` | 4 | off | leader + `chan4`'s env | the combination candidate |
| `dense4h8` | 4 | off | `DENSE_BITS=4 HEAD_BITS=8` | mixed bits: 4-bit dense, 8-bit head (quality/speed compromise point) |
| `fp8` | 4 | off | `KV_CACHE=fp8` | v1.1.2's prepared fp8 latent cache — speed/memory probe; quality unqualified upstream, bench-only |
| `dense4fp8` | 4 | off | leader + `KV_CACHE=fp8` | the ladder candidate: dense4's speed, fp8's halved per-token KV |
| `parallel6` | 6 | off | — | two more lanes (EXL3 model: 0.71 GiB/stream, 13.3 GiB free after reserve at p4) |
| `parallel8` | 8 | off | — | four more lanes |
| `recall` | 4 | on | — | Recall alone (drafts copy from previously-read text; verified-exact by design) |
| `recall6` | 6 | on | — | both — the candidate "best" configuration |

## The phases

### Phase 1 — quick performance (default; minutes per configuration, not hours)

`PHASE=perf scripts/run-bench-matrix.sh` (the default): the bench runs **`--perf-only`** —
llama-benchy pp/tg throughput and latency against the endpoint, no scenarios. All four
configurations in well under an hour including restarts.

Fixed options (identical every run): `--seed 42 --perf --parallel 4 --timeout 240
--no-live --format openai --model "$SERVED_NAME"` (the config default is
`GLM-5.3-Flash-EXL3`; the tokenizer follows `MODEL_ID`) against `http://127.0.0.1:8000/v1`.

**Outcome:** the fastest configuration whose latency/throughput profile fits the traffic
(single-stream tg, concurrent aggregate, pp).

### Phase 2 — quality on the winner only

`PHASE=tools scripts/run-bench-matrix.sh <winner>`: the 69 standard tool-calling scenarios
plus `--perf` in the same run — quality and speed from one configuration, no Hard Mode.
Score within noise of the configuration's Phase-1 expectations and of production's anchor
(91/100 was achieved **with** Hard Mode; the standard-only comparison is the 69-scenario
score).

### Phase 3 — Hard Mode, last

`PHASE=hardmode scripts/run-bench-matrix.sh <winner>`: adds the 28 Hard-Mode scenarios
(the production anchor's 91/100 / 177-194 points shape). Only run once Phase 2 passes.

## Procedure

```sh
cd ~/GLM-5.3-Flash-EXL3-2x-GB10-TensorFold-v1.1-testing
PHASE=perf scripts/run-bench-matrix.sh                    # all four configurations
PHASE=perf scripts/run-bench-matrix.sh baseline recall    # or a subset
NO_BENCH=1 scripts/run-bench-matrix.sh recall6            # restart + serve only
```

- Each configuration **restarts the server** (interrupts clients — run in a quiet window),
  waits for `/health` (~4 min), then benches.
- Results land in `results/<run-id>/<label>/` (bench log, `result.json`, the server's own
  memory-model line); `SUMMARY.md` collects the score/perf lines.
- After each run the script clears the recall record, so no configuration inherits
  another's state.

## Reading the results

| Metric | Source | Anchor |
| --- | --- | --- |
| pp / tg tok/s, latency | `--perf` (llama-benchy) | production: 1,829 pp / 63.0 tg single; 78.2 / 67.1 tg at 2 and 4 concurrent (EXL3 stack — external reference) |
| Scenario score | Phase 2/3 output, `result.json` | production: 91/100, 177/194 points **with** Hard Mode |
| Server memory model | `<label>.server.log` | NVFP4-era anchor: model 93.72 GiB; streams 0.89 GiB at 4 / ~1.4 GiB at 6 — **re-derive from the EXL3 server's startup line** |

**Decision rule:** the Phase-1 winner = best tg/pp at acceptable latency; it earns Phase 2.
Phase 2's score within noise (±1 scenario / ~2 points) of expectation → it stands; a drop
→ fall back to `baseline` and re-test. Recall is exact by construction; `parallel6`'s only
risk is memory headroom (~1.2 GiB left — watch `<label>.server.log` and `/health`).

The context campaign (docs/CONTEXT_512K_PLAN.md) is separate and already performance-first
(needle probes); its ladder runs at `--parallel 1` and never touches the scenario suites.

## Status

- [x] RECALL / PROMPT_CACHE_GIB / knob support in the scripts
- [x] 7-config matrix run on **v1.1.0** (2026-10-18, `results/run-20261018-174541/`):
      **dense8 won** (solo 21.2 → 24.0 tok/s, −3.8 GiB model); pdl noise; parallel6/8 no
      gain; recall unrewarded by the fresh-prompt bench shape. Deep-prompt concurrency
      collapse (8.6 agg) structural — see docs/LAUNCH.md's session log.
- [x] ~~v1.1.1 re-baseline~~ superseded: v1.1.2 landed the same week (quick window measured
      1.1.1 greedy 24.7 / sampled 25.9 solo; 1.1.2 rebased over it).
- [x] **5-config matrix on v1.1.2 + RoCE (2026-10-22, `results/run-20261022-154406/`)** —
      every number moved; every restart came up on the RoCE gather (`2 HCA(s) … RDMA
      writes`, no fallback):

  | config | solo tg d0 / d4096 / d8192 | c4 agg d0 / d4096 | pp2048 | free / cache |
  |---|---:|---:|---:|---|
  | baseline | 36.4 / 38.9 / 35.1 | 37.8 / 19.6 | 1,026 | 13.3 / 11.2 GiB |
  | **dense4** | **55.4 / 54.7 / 51.4** | **42.6 / 21.1** | **1,053** | **19.4 / 16.0** |
  | dense6 | 47.1 / 47.0 / 47.2 | 41.3 / 20.6 | 1,042 | 18.4 / 16.0 |
  | dense8 | 43.8 / 41.0 / 43.0 | 40.3 / 20.1 | 1,032 | 17.4 / 15.3 |
  | pdl | 40.0 / 35.6 / 34.6 | 37.2 / 19.4 | 1,028 | 13.4 / 11.2 |

  vs v1.1.0 baseline (657 pp / 21.2 solo / 25.3 c4 / 8.6 c4@d8192): **prefill +56%,
  solo decode +72% (dense4 +161%), TTFT d0 3,117 → 1,995 ms**. Three changes bundled
  (engine 1.1.0→1.1.2, RoCE RDMA up, HCA set) — attribution needs a `TF_ROCE=0`
  control run. dense4's win is clean and monotonic (4 < 6 < 8 < 16 bits), unlike
  1.1.0's dense8-noise. The memory anchor recovered to 13.3/11.2 (1.1.1's 11.0/8.9
  was transient — likely #993's bounded snapshot pages).
- [x] **Search matrix on v1.1.2 + RoCE (2026-10-22, `results/run-20261022-172325/`)** —
  solo decode (d0/d4096/d8192), c4@d4096, pp2048, and the memory line:

  | config | solo tg | c4@d4096 | pp2048 | free / cache | verdict |
  |---|---|---:|---:|---|---|
  | roce0 | 34.8 / 33.5 / 31.8 | 19.3 | 1,028 | 13.4 / 11.3 | the engine's RoCE gather adds ~5–10% solo decode; prefill unchanged → the 657→1,026 prefill jump was NCCL RDMA (vs TCP), not the gather |
  | chan4 | 35.8 / 33.9 / 34.3 | 19.4 | 1,029 | 15.4 / 13.3 | no prefill gain here, decode ≤ baseline at depth; **frees ~2 GiB of NCCL workspace** — a memory knob, not a speed knob |
  | dense4pdl | 47.9 / 48.8 / 47.4 | 20.7 | 1,043 | 19.4 / 16.0 | **PDL hurts dense4** (−13/−11/−8% solo) — opposite sign from baseline; drop |
  | dense4chan4 | 49.8 / 51.9 / 44.7 | 20.8 | 1,044 | 21.3 / 16.0 | **the channel set costs dense4 ~10% solo** for +1.9 GiB — take only if a rung needs the memory |
  | dense4h8 | 45.3 / 47.2 / 47.9 | 20.8 | 1,030 | 19.3 / 16.0 | 8-bit head is *slower* than 4 (−18/−14/−7%) and +0.15 GiB — no reason over dense4 |
  | fp8 | 35.6 / 35.6 / 35.0 | 18.7 | 971 | 13.7 / 12.3 | fp8 caches work (stream 0.71→0.48 GiB); ≈ baseline speed −3% — **a memory knob with a small speed tax** |

  **Standing after two matrices: dense4 alone remains the speed leader; nothing stacks
  on it without cost. The two cheap knobs trade speed for memory (chan4 +2 GiB, fp8
  KV −32% stream bytes).**
- [x] ~~Bench the untested combination: `dense4fp8`~~ — done in the first campaign:
      tool-eval **92 = parity cell's 92** at 524288, 0 errors. Quality risk retired at
      this bench; throughput tables still owed (below).
- [x] ~~Structured-output cliff (MaTeZZ's mechanism)~~ — absent on native 1.1.x
      (schema batch 17.9s vs plain 21.2s). The general solo→c4 flatness is still open.
- [x] ~~Rerun the gates in their fixed content form~~ — done (round 2): **all PASS on
      dense8 AND dense4fp8**. Remaining rigor item: confirm the g3 cache hit via
      `/metrics` (`cached_tokens=0` in usage on both arms — field likely unwired).
- [x] ~~Throughput tables for the fp8 cells at 512K~~ — done (round 2, table above).
- [x] ~~**Qualify dense4**~~ — **done**: kvqual4 all-PASS + tool-eval 92 = parity +
      throughput lead. dense4fp8 is the qualified 512K config.
- [x] ~~Climb with the trimmed prompt cache~~ — done (entries below): the ceiling is the
      engine's own admission arithmetic, not a knob fight.
- [x] ~~Concurrency at depth — the one big open item~~ — resolved as a **bench artifact**
      (round-6 entry below): steady-state aggregate decode RISES with streams
      (c4 ≈ 99–110 t/s, matching production's derived 103.7); the tg128-at-depth rows
      were prefill-polluted by their own TTFT.
- [x] **KV-pages branch test — resolved** (#1163+#1211 @ `87e4417e`, image
      `glm53-native-tf:commit`): round 9 (`campaign-20261023-170125/`, 17:01–18:20):
      - **fp8 KV × `TF_KV_PAGES` are mutually exclusive** on the branch (the engine exits
        at startup; round-8A receipt `campaign-20261023-165319/`) — paging is bf16-KV only.
      - **All §2 gates PASS on paged bf16 at 524288 × PARALLEL=4** (content form: g1–g4
        PASS, g5 content-stable) — exactness holds under paging. The no-paging control
        refused with the engine's own arithmetic ("4 streams need 19.77 GiB, budget
        leaves 12.06: serve --parallel 2").
      - **Admission: 5, 6 and 8 streams at 512K** (window-mode: 5 on fp8, 2 on bf16). The
        pool is fixed at ~one full window (1028–1032 pages, 6.40–6.42 GiB), demand-
        allocated; each stream costs ~220 MiB fixed (180 without the drafter; DFlash2
        adds ~2.8 GiB of model). Depth × streams is the budget — "enough for one at a
        524288-token window".
      - **Bench (same instrument both configs)**: paged-bf16 leads the fp8-window config
        on every row — solo d0 51.4 vs 48.3 (+6%), c2 43.2 vs 41.3, c4 42.0 vs 38.7 (+9%),
        d8192 c4 14.1 vs 13.3, pp ~1,037 vs 1,025, TTFT ≈ equal. The solo row reaches
        production parity (51.0–51.4 vs production's 51.6). Rows stay the TTFT-polluted
        instrument; the steady-state probe on paged KV is owed for the true curve — and
        the bench loads at most c4 clients regardless of server streams, so the 5/6/8
        rungs measured ADMISSION, not >4-client load.
      - **Verdict**: a shallow-many concurrency option on bf16 — NOT a serving-config
        replacement (no fp8 KV → a deep 512K stream costs 2×, and the pool caps deep
        concurrency at one window). The head's pin came out; marker back to main
        (1351ed16). Upstream ask: paging × fp8 KV for glm/cuda.
- [x] **PR #1079 A/B — a wash, dropped** (`campaign-20261023-185520/`): the staged
      shared-lanes decode patch passed all §2 gates ("replies unchanged" holds here) but
      moved nothing that matters: solo d0 +3.3%, solo d4096 −8.6%, rest ±2%, pp/TTFT
      unchanged. Receipt in patches/README.md; knob removed from local.sh.
- [x] **Knob sweep — done, one sharp lead** (`campaign-20261023-200302/`, the `knobs`
      phase): `TF_GLM_PROMPT_STEPS=0` (prompt passes run whole) took **c2 @ d8192
      aggregate 17.3 → 33.1 t/s (1.9×)** — the d8192-c2 row is tight across five rounds,
      so the jump looks real; c4 unchanged, solo d0 dipped. conf085 solo +3.8% (weak —
      the duel's acceptance/ITL instruments decide). conf055/overlap6/kdawide0: no
      effect. **Loading decomposition**: the engine loads in 241.5–253.5 s across all
      cells (walls 251–269 s — orchestration is only ~8–17 s; the drafter ~11 s), so the
      2.7× gap to production's 93.7 s is the native loader's convert-and-pack path —
      patch candidate: a packed-weights cache beside the byte-verify digest cache.
      The drafter's decode value confirmed: 29.6 vs 49.9 t/s solo without it.
- [ ] **steps0 confirm cell** — one re-run of `TF_GLM_PROMPT_STEPS=0` (plus the duel
      phase with production up first) before the 1.9× goes in the anchors.

- [ ] **Loading time — behind 2.8× and unexplained** (part of the knobs phase):
      production 0.6.6 logs **"loaded in 93.7 s"**; our restart walls run ~265 s. The
      knobs phase records each cell's engine "loaded in" line vs the wall — isolating
      the drafter's share (load-nodraft cell), the packing share (the engine's own
      lines), and the orchestration overhead (wall − engine).
- [ ] Small hygiene: worker clock probe reads 0 via ssh (acpi_cppc path?); g3's
      `/metrics` cache-counter check (goshi's exporter maps `tensorfold_cached_tokens_total`).
- [ ] Phase 2/3 quality on **dense4fp8** (the qualified winner): tools-mode + hardmode
      TEB runs, and the failing-scenario comparison vs production (TC-68, TC-51).
- [ ] Small hygiene: worker clock probe reads 0 via ssh (acpi_cppc path?); consider
      pinning the upstream ref in prepare so receipts stop drifting with HEAD.
- [x] **One-command campaign** (`scripts/run-qualification.sh`, 2026-10-23): preflight →
      production reference probe → parity cells → fp8-KV exactness gates (VALIDATION §2)
      → depth probes → structured-output cliff → prefill probe → 524288 rung → summary.
      Phased (`.done` markers, `PHASES=` selectable), production-guarded, `bash -n` clean.
      This script IS the open items above: dense4fp8 + dense4 qualify in one window.
- [x] **Ecosystem research pass** (2026-10-23) — docs/ECOSYSTEM.md: repo-by-repo findings
      from the thread + READMEs. Adopt now: goshi's TensorFold metrics exporter (targets
      our engine line directly), GID_INDEX unpinned as invariant, clock probe (in the
      campaign). Measure & hand upstream: adaptive-k acceptance, prefill fairness,
      structured-output draft truncation (the cliff). Watch: EXLR8 exl3-trellis experts.
      Ease-of-use stance held: no second config system, no patch ladders.
- [x] **First campaign run** (2026-10-23 01:13–02:45, `results/campaign-20261023-011329/`)
      — the 512K window qualified end-to-end on the first overnight run:

  | phase | result |
  |---|---|
  | parity cell (dense8+fp8 KV+524288, PARALLEL=4) | **tool-eval 92/100** (125/136 pts, ★★★★★, 0 errors) |
  | candidate cell (dense4fp8+524288, PARALLEL=4) | **tool-eval 92/100** (identical points; deployability 68 vs 67, responsiveness 11 vs 10) |
  | depth (fp8 KV, dense8) | **12/12 facts** at 8k / 64k / 192k / 384k |
  | rung 524288 × PARALLEL=1 | fits: 3.61 GiB/stream fp8, 13.9 GiB left; 8k probe 3/3 |
  | gates g1–g5 | **byte-level gates are the wrong instrument on this engine** — see below |
  | cliff (4-stream + one schema request) | **absent**: schema batch 17.9s vs plain 21.2s — MaTeZZ's draft-truncation mechanism does not reproduce on native 1.1.x |
  | production reference | skipped (production was down) |

  **The gates need re-reading, not the KV cache.** g2/g3/g4 "failed" as raw-sha
  comparisons, but every captured output was **semantically identical** (the same
  primes+sum across drafted/plain and solo/c1–c4). Root cause (corrected after the
  prodqual control, see round 3 below): the raw-sha instrument hashes the whole
  response envelope, whose `id`/`created`/metadata fields differ on every request on
  ANY engine — it was never measuring token determinism. g3's empty reply was a probe
  bug (it compared two *different* prompts — turn 2 alone vs turn 1+2 — and 512
  max_tokens was eaten by 2,242 chars of thinking on the resumed arm).
  The script now gates on **normalized content** (`content_sha`), gives g3 the same
  conversation twice (cold compute vs full cache hit, with cached_tokens receipts), and
  times the prefill probe (`max_tokens=1`, wall ≈ prefill). Rerun `PHASES=kvqual prefill`
  to convert the three FAILs into their real verdicts — expected PASS on content for
  g2/g4 and a true cache-path test for g3.
  Two housekeeping findings: (1) the campaign rebuilt the image mid-run — start.sh
  re-prepares when the config fingerprint changes (the new KV_CACHE knob guaranteed it)
  and prepare.sh builds from upstream HEAD, so the campaign ran engine 1.1.2 at
  **1351ed16**, not the matrix's 4ff78441 (same engine version; noted for
  cross-receipt comparisons); (2) both fp8 cells served 524288 because the shell
  exported CONTEXT — dense4fp8's matrix cell now pins it explicitly.
  **Ladder verdict: the 512K goal is met and over-delivered** — dense4fp8 serves
  524288 × PARALLEL=4 with 15.8 GiB free (a PARALLEL=6/8 climb is affordable), quality
  identical to the production-parity config, ~2 GiB cheaper, and it should inherit
  dense4's solo-decode lead. Missing numbers: throughput tables for the fp8 cells at
  512K (the campaign's bench call ran the full tool-eval instead of `--perf-only`) —
  one `run-bench-matrix.sh parity dense4fp8` window fills that in.
- [x] **Round 2** (`results/campaign-20261023-093827/`, 2026-10-23 09:38): gates, speed,
      climb — **dense4fp8 is the qualified 512K serving config**.

  **Gates — all PASS on both configs** (content form, VALIDATION §2 receipt): g1
  token-deterministic (`choices` byte-identical across runs — see the round-3
  correction), g2 drafted=plain ✓, g3 cache-hit=computed ✓ (both arms answer
  "alpha-7391, bravo-2250, charlie-8118"), g4 concurrent=solo ✓, g5 seeded sampling
  content-stable. **The 0.6.x replication proof is now on paper for the native line.**
  One rigor caveat: g3's `cached_tokens=0` on *both* arms — the engine doesn't populate
  the usage field, so the gate passed on content while the cache-hit path itself went
  unverified; a `/metrics` cache-counter check closes that (goshi's exporter maps
  `tensorfold_cached_tokens_total`).

  **Throughput at 524288 × PARALLEL=4, fp8 KV** (pp2048 tg128, t/s):

  | cell | solo d0 / d4096 / d8192 | c2 d0 / d4096 / d8192 | c4 d0 / d4096 / d8192 | pp2048 |
  |---|---|---|---|---:|
  | parity (dense8+fp8) | 42.9 / 43.9 / 44.0 | 36.9 / 23.0 / 16.9 | 37.5 / 18.8 / 13.0 | 1,018 |
  | **dense4fp8** | **48.3 / 49.0 / 48.0** | **41.3 / 23.5 / 17.4** | **38.7 / 19.2 / 13.3** | **1,025** |

  dense4fp8 leads every cell: **+13% solo** (48.3 vs 42.9), +2–3% concurrent, prefill
  equal. Solo 48.3 vs the Oct-22 dense4 cell's 55.4 (at 40960/bf16 KV) prices the 512K
  window + fp8 at ~13% of solo decode. **The depth cliff reconfirms**: aggregate decode
  48.3 solo → 13.3 at c4@d8192 (3.3 t/s per stream), TTFT 2.0s → 28.4s — still THE open
  performance question. Prefill probe: **1,063 t/s on a 26,869-token prompt** — no
  deep-prefill collapse on the native line (the upstream-fairness note writes itself).
  **Climb: 6 and 8 streams refused at 512K with default knobs** — the kept-states store's
  worst-case sizing is the suspected blocker; the climb phase now retries with
  `PROMPT_CACHE_GIB=4` on refusal (rerun `PHASES=climb` to test).
- [x] **Climb ceiling found — it is the engine's own arithmetic** (`campaign-20261023-121811/`):
  the admission refusal prints its math — **"8 streams need 25.29 GiB … budget leaves
  15.83 GiB: serve --parallel 5, or lower --context"**. So 5 streams fit at 524288
  (15.81 of 15.83 GiB, knife-edge) and 6+ is blocked by the engine's **12.2 GiB
  reserve**, which no flag exposes (checked `serve --help`). `PROMPT_CACHE_GIB=4`
  freed +0.06 GiB (≈ the default) — not the lever. The climb phase now runs 5/6/8 so
  the ladder completes with the engine's own suggestion.
- [x] **The concurrency cliff was a bench artifact — steady-state decode matches production**
      (`campaign-20261023-145449/`, TF_GLM_PROF=1, 2026-10-23): with long generations
      (2048 max_tokens) the aggregate decode **rises** solo→c4 — solo 53.7/57.6 t/s
      (d0/d4096-class), c4 **99.2 / 102.8 t/s** — versus production's steady-state 103.7
      (derived from its bench rows). The engine's "tg" numbers collapse at depth because
      llama-benchy's tg128 rows divide by TOTAL time (TTFT included): at c4@d8192 the
      4×8192-token prefills are ~80% of the wall, so the 128-token generations score
      ~13 t/s. The matrix's "structural collapse at depth" was the same artifact.
      Section economics (head rank, ms/call over 64-call windows):
      - **wall ≈ gpu in every line** (40.08 vs 40.00) — zero scheduling bubbles; the GPU
        pipeline is saturated. There was never a scheduling cliff.
      - **`routed` dominates everything**: 18.5 ms of a 40 ms 2-row verify (46%), 54.7 of
        80 ms at 8 rows — **marginal cost per draft row ≈ 6.0 ms, nearly all in routed
        experts**. The RoCE gather (`gather_a`+`gather_b`) is 2.7–3.7 ms — 4–8%, innocent.
      - **prompt passes are 90% MoE**: a 2048-row chunk costs 1,892 gpu ms — router 910 +
        routed 793. Our prefill gap to production (1,025 vs 1,530) is the expert path in
        prompt chunks, not attention — a different target than #1080's kernels (which
        attack attention/projections; qwen-family).
      - attention (`kda`+`mla`) ≈ 13–16 ms/call, growing gently with depth (mla 4.4→6.7
        by ~2k context) — not the driver at our working depths.
      Consequences: (1) **more streams = more aggregate** for long-generation (agent)
      workloads — the KV-pages admission unlock buys real throughput; (2) the honest
      steady-state probe is long-generation, not tg128 — the prof phase doubles as the
      steady-state bench; (3) the upstream prefill conversation for GLM is about expert
      batching in prompt passes.
- [x] **The 3-point quality gap is located**: production PASSED TC-65/66/67/69
  (nested schema, enum, multi-tool) where BOTH our configs got partials — and both
  failed TC-68 like production. The gap is entirely in the structured-output
  category, is config-independent (parity cell = same pattern), and is therefore
  engine-level (native vs Python decoding path), not quant-level.
- [x] **Round 3 — the production head-to-head** (`results/campaign-20261023-110446/`,
      2026-10-23 11:04, prodqual read-only while production served): production 0.6.6
      (fp8 dense+KV, DFlash2, RoCE, 524288) measured with the SAME harness as our
      dense4fp8 receipts.

  | instrument | production 0.6.6 | dense4fp8 (native 1.1.2) |
  |---|---|---|
  | tool-eval (69 scenarios, same version/seed) | **95** (131 pts) ★★★★★ | 92 (125 pts) ★★★★★ |
  | deployability / responsiveness | 73 / 21 | 68 / 11 |
  | pp2048 | **1,530** | 1,025 |
  | solo tg d0 / d4096 / d8192 | **51.6 / 56.0 / 53.1** | 48.3 / 49.0 / 48.0 |
  | c2 agg d0 / d4096 / d8192 | **67.3 / 36.9 / 26.4** | 41.3 / 23.5 / 17.4 |
  | c4 agg d0 / d4096 / d8192 | **60.5 / 28.5 / 19.0** | 38.7 / 19.2 / 13.3 |
  | TTFT d0 | **1,339 ms** | 1,999 ms |
  | pg1 greedy determinism | `choices` byte-identical | `choices` byte-identical |
  | pg3/pg4/pg5 | PASS / PASS / PASS | PASS / PASS / PASS |

  Three takeaways. **(1) The determinism correction:** production's pg1 pair differs
  from our g1 pair in exactly the same way — envelope `id`/`created`/metadata only,
  `choices` byte-identical. Both engines are token-deterministic; the first campaign's
  "byte-level nondeterminism" was the instrument (whole-envelope sha), not the engine.
  **(2) The exactness guarantees hold identically on both engines** — pg2-class gates
  all PASS read-only on production. **(3) The real native-line gap is concurrency
  scheduling**: production's aggregate decode RISES with streams (51.6 solo → 67.3 at
  c2) while ours FALLS (48.3 → 41.3); at c4@d8192 it's 19.0 vs 13.3. Prefill +49% and
  TTFT −33% to production as well. This is the like-for-like evidence for the upstream
  concurrency note, and it sharpens the TF_GLM_PROF target. (Bench quirk to check:
  production's max points read 138 vs our 136 — same 69 scenarios, cause unknown.)

- [x] **Rounds 15–30 — the concurrency front's root-cause ladder** (2026-10-24/25):
  - **prof4** (`campaign-20261024-000708`): verify rounds DO batch across streams — 8-row
    shared windows at c4 (76.5 ms; solo 2-row 41 ms); `TF_GLM_SHARED=0` halves the c4
    aggregate (51.6). Both Front-1 levers answered and retired: batching exists, sharing
    is load-bearing.
  - **routerks2** (`campaign-20261024-115242`): the `router_ks 8→2` patch — the router
    section unmoved (1096→1090 ms of the chunk) + the content gate flipped a greedy token
    → RETIRED (#1079 protocol); the pipeline-depth hypothesis rejected.
  - **routerpart** (`campaign-20261024-125609`): the forced unstaged kernel costs the same
    (1140 vs 1096) with the gate IDENTICAL → the pathology is the hand-rolled mma design;
    the GEMM replacement (glm_mm's row-stable K-slices, the preserved 8-slice summation) is
    the scoped fix.
  - **the top4 ladder** (`campaign-20261024-132628` + `135458`): four attempts, one stage
    deeper each — the `weightShare` parse-failure billing (a 163.6-GiB budget illusion from
    the silent catch-return-total), the snapshot config's SYMLINK aliasing (fixed with a REAL
    file + byte-surgical edits), the parse's `want` pin and the EXL3 experts' module-global
    (both to topk=4 in `glm-topk4b.patch`). **Real top-4 routing measured**: routed
    18.45→11.0 ms; **solo 63.6 t/s = 108% of production — CROSSED**; c2 103%;
    **c4 120.4 = 78% (was 65%)**; slope 0.593 (−10%); acceptance 4.8 tok/chunk (was 3.37).
  - **top4qual battery** (`campaign-20261024-142659`): §2 gates ALL PASS on the modified
    checkpoint; tool-eval **93** (above stock's 92; production's anchor 95); hardmode
    **89** (production's anchor 91; the line's first hardmode datum).
  - **(user) the served default = top4cfg + PATCHES=glm-topk4b** (round 27); the anchors
    and the chart regenerated from the receipts (4 rows over the 100% line).
  - **the AZ route**: azampatti's retrained selector downloaded (the Xet CAS survived the
    site's reboot); the requant pipeline staged (az_dequant → az_conv_fix → convert
    `-b 4 -hb 16 -eb 16 -vb 16` → local model → the azqual battery); the convert in flight
    (layer 34/45 at 19:36; `done.` ~02:30 tonight). Discovered en route: **AxonZeta's
    AZ-Focus-EXL3** (the EXL3-native reasoning-efficiency checkpoint, topk=8) — test-ready
    on both hosts; a separate lever.
  - **the lessons**: the config fingerprint hashes the patch LIST, not contents (an amended
    patch needs a new name to force a rebuild); a json round-trip fails the engine's strict
    config parser (byte-surgical edits only); background watchers must exclude their own
    command text from liveness pgreps.
- [ ] **The AZ battery digest** (tonight) — the retrained selector vs the top4cfg references.
- [ ] **The isolation quality round**: TC-92 (tenant isolation) + TC-51 — bernisse's bake-off
      flags them as the instrument our §2 gates don't cover (the fork fails TC-92
      deterministically; technigma is the only clean stack).
- [ ] **The pages×fp8 marriage watch** (#1163 mergeable-clean; #1211 stacks): when
      paging×fp8 lands, 512K admits 8 streams on fp8 — re-qualify the ladder then.
- [ ] **The v1.1.5 candidate test**: #1218 (the slot-leak) is merged and slated upstream —
      the first correctness reason to test a bump (rebase glm-topk4b, gates + probes decide).
- [ ] **Publish the AZ quant to huggingface.co** (user, 2026-10-26): gated on the AZ
      battery passing — repo under gilesco, the ~154-GiB EXL3 upload, and a model card
      carrying the quant contract (`-b 4 -hb 16 -eb 16 -vb 16`), provenance (azampatti's
      retrained top-4 selector → `az_dequant.py` → `az_conv_fix.py` → exllamav3 convert),
      and the gates/tool-eval/hardmode receipts summary. The `az-az-fp16` deletion
      (599 GiB) lands with this choreography once the verdict is in.
- [x] **The v1.1.6 rebase test** (round 33, `campaign-20261027-085035`): the pin → `eea9dc9c`
      (Release 1.1.6). ADOPTED — the A/B on the identical Focus serve: c2 74.7→**94.3 (+26%)**,
      c4 119.8→**150.6 (+26%)**, hardmode 90→**91** (ties production's anchor), gates/solo/
      slope/tool-eval unchanged. CAUGHT: the first A/B (`053451`) silently ran the old engine —
      the auto-prepare keys on the config fingerprint (the patch names), NOT the pin; a pin
      change needs an explicit `prepare.sh` (the image ID betrayed it).
- [ ] **The bench v3.0.1 boundary**: tool-eval-bench updated to `3.0.1.dev4+gb672b53ff`
      (release v3.0.0 + two fixes) — the graphics results, the web UI (`serve`), and the
      Hard-Mode suite grows to TC-104 (seven new scenarios; the hardmode run is now 104
      scenarios, was 97). **Scores across the bench-version boundary are not comparable.**
      Production's side is DONE (round 35): idle-gated re-baseline `prod-score-rebaseline-
      20261027-114634` — tool-eval **96/100** (133/138: 66 pass, 1 partial, 2 fail TC-43/
      TC-51), hardmode run **94** (196/208 over all 104), hardmode subset **63/70 = 90%**
      (30 pass, 3 partial, 2 fail TC-92/TC-104 — production fails the newest scenario).
      The old anchors (95/91) are formally cross-version — not comparable, kept only as
      history. OUR SERVES STILL OWED: the Focus + the top4cfg re-bench on v3.0.1 in the
      next window; no cross-engine score comparison until then (the perf/duel side IS
      comparable — round 33 addendum).
- [ ] **Production's patch series changed** (user): Mia's patches 0109-0122 + 0128-0129 and the
      0088 NoRoom fix folded onto the python-0.6 tip (`879d7cfc`). DELTAS's production column
      carries it; the re-baseline refreshes the cross-engine anchors.
- [~] **The top4cfg re-qual on v1.1.6** — the PERF side is DONE (round 36,
      campaign-20261027-124115): §2 gates ALL PASS (g1–g4 content form), the duel receipt
      (solo 64.8 / c2 108.0 / c4 167.4 t/s, TTFT d0 148 ms, ITL 8.9 ms/tok — the
      scoreboard/chart moved to CROSSED on every concurrency front), prefill 1,175 t/s
      (was ~1,026), loading 256 s (was 344), depth 12/12, the throughput/climb/knobs
      receipts fresh. STILL OWED: its tool-eval + hardmode on bench v3.0.1 — one more
      window with the Focus's re-bench (the wrapper's PHASES export bug is fixed and
      verified: guard path + child-visibility probes pass).
