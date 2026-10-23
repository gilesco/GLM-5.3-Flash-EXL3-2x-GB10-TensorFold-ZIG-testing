# Test plan: performance first, then quality (phased)

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
- [ ] Bench the untested combination: `dense4fp8` (leader + fp8 caches) — the ladder
      candidate if its solo cell lands near dense4's.
- [ ] **Qualify dense4** (exactness gate + recall probes) — at 55.4 solo it approaches
      production Python's 62–63 and frees 6 GiB for the 512K ladder; 4-bit dense linears
      are a quality risk until the gate passes. Gate fp8 the same way before it serves.
- [ ] Concurrency cliff remains: aggregate decode is ~flat solo→c4 on baseline (36.4 →
      37.8) and every config drops at depth (best c4@d8192 ≈ 14 agg). Diagnose with
      `TF_GLM_PROF=1` at c4@d4096.
- [ ] Corrected probes 3 (prefill ~14k) and 4 (exactness) on v1.1.2 baseline, then the winner.
- [ ] Phase 2/3 on the qualified winner.
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
